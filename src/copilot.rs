use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tokio::io::AsyncWriteExt;

use crate::ipc::{write_frame, Response};

const CHAT_ENDPOINT: &str = "https://api.githubcopilot.com/chat/completions";

// ---------------------------------------------------------------------------
// Retry policy constants
// ---------------------------------------------------------------------------

/// How many times to retry transient failures (5xx, 429, network errors)
/// before surfacing the error to the caller.
const MAX_RETRIES: u32 = 3;

/// Base delay for exponential backoff: attempt 0 → 1 s, 1 → 2 s, 2 → 4 s.
const BACKOFF_BASE_MS: u64 = 1_000;

/// Hard ceiling on a Retry-After value.  Prevents hanging for unreasonably
/// long server-imposed back-off windows.
const MAX_RETRY_AFTER_SECS: u64 = 30;

// ---------------------------------------------------------------------------
// Typed HTTP error — lets callers inspect the status and body
// ---------------------------------------------------------------------------

/// An HTTP error response from the Copilot API.
///
/// Returned (via `anyhow::Error::new`) when the chat endpoint responds with a
/// non-2xx status.  Callers can use `anyhow::Error::downcast_ref::<CopilotHttpError>()`
/// to inspect the status code and decide how to recover (e.g. retry on 4xx).
#[derive(Debug)]
pub struct CopilotHttpError {
    pub status: reqwest::StatusCode,
    /// Raw response body — may contain a JSON error description.
    pub body: String,
}

impl std::fmt::Display for CopilotHttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Truncate the body to 200 chars to prevent flooding logs when this
        // error is formatted via tracing::warn!(error = %e, ...).
        let body = &self.body;
        match body.char_indices().nth(200) {
            None => write!(f, "Copilot API returned {}: {}", self.status, body),
            Some((byte_idx, _)) => write!(
                f,
                "Copilot API returned {}: {}…",
                self.status,
                &body[..byte_idx]
            ),
        }
    }
}

impl std::error::Error for CopilotHttpError {}

// ---------------------------------------------------------------------------
// Public API message types
// ---------------------------------------------------------------------------

/// A single message in the conversation history.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Message {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool calls requested by an assistant turn.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tool_calls: Vec<ApiToolCall>,
    /// Links a tool-result message back to the call that triggered it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ApiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ApiToolCallFunction,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ApiToolCallFunction {
    pub name: String,
    /// JSON-encoded arguments string, as required by the OpenAI wire format.
    pub arguments: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: Some(content.into()),
            tool_calls: vec![],
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: Some(content.into()),
            tool_calls: vec![],
            tool_call_id: None,
        }
    }

    /// Assistant turn that requested tool calls (content may be empty).
    pub fn assistant(text: Option<String>, calls: Vec<ApiToolCall>) -> Self {
        Self {
            role: "assistant".into(),
            content: text,
            tool_calls: calls,
            tool_call_id: None,
        }
    }

    /// Tool result message to be sent back after executing a tool.
    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: Some(content.into()),
            tool_calls: vec![],
            tool_call_id: Some(call_id.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// Public response types returned by stream_chat
// ---------------------------------------------------------------------------

/// A fully-parsed tool call extracted from the SSE stream.
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Raw JSON-encoded arguments string (used when replaying history).
    pub arguments: String,
}

impl ToolCall {
    /// Parse the JSON arguments into a `serde_json::Value` for execution.
    pub fn parse_args(&self) -> Result<serde_json::Value> {
        serde_json::from_str(&self.arguments)
            .with_context(|| format!("parsing arguments for tool '{}'", self.name))
    }
}

pub enum FinishReason {
    Stop,
    ToolCalls,
    Length,
    Other(String),
}

/// Everything the daemon needs after one round-trip with the Copilot API.
pub struct CopilotResponse {
    /// Accumulated text content (empty when the turn was tool-call-only).
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: FinishReason,
}

// ---------------------------------------------------------------------------
// Private SSE deserialization types
// ---------------------------------------------------------------------------

/// Delta fragment for a single tool call, as it arrives in one SSE chunk.
#[derive(Deserialize, Debug)]
struct ToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<FunctionDelta>,
}

#[derive(Deserialize, Debug)]
struct FunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    #[serde(default)]
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<String>,
}

/// One SSE data line.  The Copilot API sometimes puts text in choices\[0\] and
/// tool_calls in choices\[1\] — we always collect all choices.
#[derive(Deserialize, Debug)]
struct ChatChunk {
    choices: Vec<Choice>,
}

// ---------------------------------------------------------------------------
// Accumulator used while processing the SSE stream
// ---------------------------------------------------------------------------

/// In-progress tool call being assembled from multiple SSE delta chunks.
#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Default)]
struct SseAccumulator {
    text: String,
    /// Keyed by the delta `index` field.
    partial_calls: HashMap<usize, PartialToolCall>,
    finish_reason: Option<String>,
}

impl SseAccumulator {
    fn apply_chunk(&mut self, chunk: &ChatChunk) {
        for choice in &chunk.choices {
            // Collect text content.
            if let Some(ref text) = choice.delta.content {
                self.text.push_str(text);
            }
            // Accumulate tool call deltas.
            if let Some(ref deltas) = choice.delta.tool_calls {
                for delta in deltas {
                    let entry = self.partial_calls.entry(delta.index).or_default();
                    if let Some(ref id) = delta.id {
                        entry.id.clone_from(id);
                    }
                    if let Some(ref func) = delta.function {
                        if let Some(ref name) = func.name {
                            entry.name.clone_from(name);
                        }
                        if let Some(ref args) = func.arguments {
                            entry.arguments.push_str(args);
                        }
                    }
                }
            }
            // Capture the first non-None finish_reason across all choices.
            if self.finish_reason.is_none() {
                if let Some(ref reason) = choice.finish_reason {
                    self.finish_reason = Some(reason.clone());
                }
            }
        }
    }

    fn finalize(mut self) -> CopilotResponse {
        // Sort partial calls by index so they arrive in declaration order.
        let mut indices: Vec<usize> = self.partial_calls.keys().copied().collect();
        indices.sort_unstable();

        let tool_calls = indices
            .into_iter()
            .map(|i| {
                let p = self.partial_calls.remove(&i).unwrap_or_default();
                ToolCall {
                    id: p.id,
                    name: p.name,
                    arguments: p.arguments,
                }
            })
            .collect();

        let finish_reason = match self.finish_reason.as_deref() {
            Some("stop") => FinishReason::Stop,
            Some("tool_calls") => FinishReason::ToolCalls,
            Some("length") => FinishReason::Length,
            Some(other) => FinishReason::Other(other.to_owned()),
            // Stream closed without a finish_reason — treat as Stop.
            None => FinishReason::Stop,
        };

        CopilotResponse {
            text: self.text,
            tool_calls,
            finish_reason,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderValue, RETRY_AFTER};

    // ---- parse_retry_after_header ---------------------------------------

    #[test]
    fn parse_retry_after_header_missing() {
        let headers = HeaderMap::new();
        assert_eq!(parse_retry_after_header(&headers), None);
    }

    #[test]
    fn parse_retry_after_header_invalid_value() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("not-a-number"));
        assert_eq!(parse_retry_after_header(&headers), None);
    }

    #[test]
    fn parse_retry_after_header_valid_value() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("10"));
        let dur = parse_retry_after_header(&headers).expect("expected Some(Duration)");
        assert_eq!(dur, Duration::from_secs(10));
    }

    #[test]
    fn parse_retry_after_header_caps_large_values() {
        let mut headers = HeaderMap::new();
        let large = (MAX_RETRY_AFTER_SECS.saturating_add(10)).to_string();
        headers.insert(
            RETRY_AFTER,
            HeaderValue::from_str(&large).expect("valid header value"),
        );
        let dur = parse_retry_after_header(&headers).expect("expected Some(Duration)");
        assert_eq!(dur, Duration::from_secs(MAX_RETRY_AFTER_SECS));
    }

    // ---- Message constructors -------------------------------------------

    #[test]
    fn message_system_fields() {
        let m = Message::system("be helpful");
        assert_eq!(m.role, "system");
        assert_eq!(m.content.as_deref(), Some("be helpful"));
        assert!(m.tool_calls.is_empty());
        assert!(m.tool_call_id.is_none());
    }

    #[test]
    fn message_user_fields() {
        let m = Message::user("what is Rust?");
        assert_eq!(m.role, "user");
        assert_eq!(m.content.as_deref(), Some("what is Rust?"));
        assert!(m.tool_calls.is_empty());
        assert!(m.tool_call_id.is_none());
    }

    #[test]
    fn message_assistant_text_only() {
        let m = Message::assistant(Some("42".into()), vec![]);
        assert_eq!(m.role, "assistant");
        assert_eq!(m.content.as_deref(), Some("42"));
        assert!(m.tool_calls.is_empty());
        assert!(m.tool_call_id.is_none());
    }

    #[test]
    fn message_assistant_with_tool_calls() {
        let call = ApiToolCall {
            id: "call_001".into(),
            kind: "function".into(),
            function: ApiToolCallFunction {
                name: "shell_command".into(),
                arguments: r#"{"command":"ls"}"#.into(),
            },
        };
        let m = Message::assistant(None, vec![call]);
        assert_eq!(m.role, "assistant");
        assert!(m.content.is_none());
        assert_eq!(m.tool_calls.len(), 1);
        assert_eq!(m.tool_calls[0].id, "call_001");
        assert_eq!(m.tool_calls[0].function.name, "shell_command");
    }

    #[test]
    fn message_tool_result_fields() {
        let m = Message::tool_result("cid_42", "stdout output");
        assert_eq!(m.role, "tool");
        assert_eq!(m.content.as_deref(), Some("stdout output"));
        assert_eq!(m.tool_call_id.as_deref(), Some("cid_42"));
        assert!(m.tool_calls.is_empty());
    }

    // ---- Message serialization ------------------------------------------

    #[test]
    fn system_message_skips_empty_fields() {
        let m = Message::system("prompt");
        let v: serde_json::Value = serde_json::to_value(&m).unwrap();
        assert_eq!(v["role"], "system");
        assert_eq!(v["content"], "prompt");
        // tool_calls and tool_call_id are skipped when empty/absent
        assert!(
            v.get("tool_calls").is_none(),
            "tool_calls should be omitted"
        );
        assert!(
            v.get("tool_call_id").is_none(),
            "tool_call_id should be omitted"
        );
    }

    #[test]
    fn tool_result_message_serializes_tool_call_id() {
        let m = Message::tool_result("cid", "result");
        let v: serde_json::Value = serde_json::to_value(&m).unwrap();
        assert_eq!(v["role"], "tool");
        assert_eq!(v["content"], "result");
        assert_eq!(v["tool_call_id"], "cid");
    }

    // ---- Retry helpers --------------------------------------------------

    #[test]
    fn backoff_delay_increases_exponentially() {
        assert_eq!(backoff_delay(0), Duration::from_millis(1_000));
        assert_eq!(backoff_delay(1), Duration::from_millis(2_000));
        assert_eq!(backoff_delay(2), Duration::from_millis(4_000));
    }

    #[test]
    fn backoff_delay_saturates_at_high_attempt() {
        // Must not panic on large attempt numbers.
        let _ = backoff_delay(30);
        let _ = backoff_delay(u32::MAX);
    }

    // ---- ToolCall::parse_args -------------------------------------------

    #[test]
    fn parse_args_valid_object() {
        let tc = ToolCall {
            id: "t1".into(),
            name: "shell_command".into(),
            arguments: r#"{"command":"echo hi"}"#.into(),
        };
        let args = tc.parse_args().unwrap();
        assert_eq!(args["command"], "echo hi");
    }

    #[test]
    fn parse_args_empty_object() {
        let tc = ToolCall {
            id: "t2".into(),
            name: "tmux_capture_pane".into(),
            arguments: "{}".into(),
        };
        assert!(tc.parse_args().unwrap().is_object());
    }

    #[test]
    fn parse_args_invalid_json_returns_err_with_name() {
        let tc = ToolCall {
            id: "t3".into(),
            name: "bad_tool".into(),
            arguments: "{not: valid".into(),
        };
        let err = tc.parse_args().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("bad_tool"),
            "error should mention tool name: {msg}"
        );
    }
}

// ---------------------------------------------------------------------------
// Retry helpers
// ---------------------------------------------------------------------------

/// Exponential back-off delay for `attempt` (0-indexed).
///
/// Returns 1 s, 2 s, 4 s for attempts 0, 1, 2.  The exponent is capped at
/// 10, so the delay saturates at `BACKOFF_BASE_MS << 10` (about 17 minutes),
/// but in practice `MAX_RETRIES` is 3 so the maximum used delay is 4 s.
pub(crate) fn backoff_delay(attempt: u32) -> Duration {
    Duration::from_millis(BACKOFF_BASE_MS << attempt.min(10))
}

/// Parse the `Retry-After` response header into a `Duration`.
///
/// Accepts an integer number of seconds only (date-form is not handled).
/// Caps the returned delay at [`MAX_RETRY_AFTER_SECS`] to prevent the daemon
/// from sleeping for unreasonably long periods.
pub(crate) fn parse_retry_after(resp: &reqwest::Response) -> Option<Duration> {
    parse_retry_after_header(resp.headers())
}

/// Parse the `Retry-After` header from a [`HeaderMap`] into a [`Duration`].
///
/// Factored out of [`parse_retry_after`] so it can be tested without
/// constructing a full `reqwest::Response`.
pub(crate) fn parse_retry_after_header(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .map(|secs| Duration::from_secs(secs.min(MAX_RETRY_AFTER_SECS)))
}

/// Send a single chat-completions POST to Copilot with a transparent retry
/// policy for transient failures.
///
/// **Retry policy:**
/// - **Transport / network errors** (connection reset, timeout, DNS failure):
///   retry up to `MAX_RETRIES` times with exponential back-off.
/// - **429 Too Many Requests**: retry up to `MAX_RETRIES` times; delay is
///   taken from the `Retry-After` header when present, otherwise exponential
///   back-off, capped at `MAX_RETRY_AFTER_SECS`.
/// - **5xx Server Error**: retry up to `MAX_RETRIES` times with exponential
///   back-off.
/// - **4xx Client Error** (other than 429): returned immediately as
///   [`CopilotHttpError`] without retrying; the caller is responsible for
///   deciding whether to refresh the token and retry.
///
/// On success returns the raw `reqwest::Response` (status 2xx).
async fn send_with_retry(
    http: &reqwest::Client,
    token: &str,
    model: &str,
    messages: &[Message],
    tools: &[serde_json::Value],
) -> Result<reqwest::Response> {
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": true,
        "max_tokens": 4096,
    });

    let mut attempt = 0u32;
    loop {
        let result = http
            .post(CHAT_ENDPOINT)
            .header("Authorization", format!("Bearer {token}"))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .header("Copilot-Integration-Id", "vscode-chat")
            .header("Editor-Version", "vscode/1.90.0")
            .header("User-Agent", concat!("amaebi/", env!("CARGO_PKG_VERSION")))
            .json(&body)
            .send()
            .await;

        match result {
            // ── Success ────────────────────────────────────────────────────
            Ok(resp) if resp.status().is_success() => return Ok(resp),

            // ── 429 Too Many Requests ──────────────────────────────────────
            Ok(resp) if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS => {
                if attempt >= MAX_RETRIES {
                    let status = resp.status();
                    let body_text = resp
                        .text()
                        .await
                        .unwrap_or_else(|_| "<unreadable body>".into());
                    return Err(anyhow::Error::new(CopilotHttpError {
                        status,
                        body: body_text,
                    }));
                }
                let delay = parse_retry_after(&resp).unwrap_or_else(|| backoff_delay(attempt));
                tracing::warn!(
                    attempt,
                    delay_ms = delay.as_millis(),
                    "Copilot rate-limited (429); backing off before retry"
                );
                // Drain the body so the underlying connection can be returned
                // to the pool for reuse, then sleep before retrying.
                let _ = resp.bytes().await;
                tokio::time::sleep(delay).await;
                attempt += 1;
            }

            // ── 5xx Server Error ───────────────────────────────────────────
            Ok(resp) if resp.status().is_server_error() => {
                if attempt >= MAX_RETRIES {
                    let status = resp.status();
                    let body_text = resp
                        .text()
                        .await
                        .unwrap_or_else(|_| "<unreadable body>".into());
                    return Err(anyhow::Error::new(CopilotHttpError {
                        status,
                        body: body_text,
                    }));
                }
                let status = resp.status();
                let delay = backoff_delay(attempt);
                tracing::warn!(
                    attempt,
                    status = %status,
                    delay_ms = delay.as_millis(),
                    "Copilot server error; retrying with exponential backoff"
                );
                // Drain the body so the underlying connection can be returned
                // to the pool for reuse, then sleep before retrying.
                let _ = resp.bytes().await;
                tokio::time::sleep(delay).await;
                attempt += 1;
            }

            // ── Other 4xx Client Error ─────────────────────────────────────
            // Returned immediately — token refresh and higher-level retry are
            // the caller's responsibility (see run_agentic_loop in daemon.rs).
            Ok(resp) => {
                let status = resp.status();
                let body_text = resp
                    .text()
                    .await
                    .unwrap_or_else(|_| "<unreadable body>".into());
                return Err(anyhow::Error::new(CopilotHttpError {
                    status,
                    body: body_text,
                }));
            }

            // ── Transport / network error ──────────────────────────────────
            Err(e) => {
                if attempt >= MAX_RETRIES {
                    return Err(anyhow::Error::from(e).context("sending chat request to Copilot"));
                }
                let delay = backoff_delay(attempt);
                tracing::warn!(
                    attempt,
                    error = %e,
                    delay_ms = delay.as_millis(),
                    "Copilot request failed (transport error); retrying"
                );
                tokio::time::sleep(delay).await;
                attempt += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public streaming function
// ---------------------------------------------------------------------------

/// Send a streaming chat-completions request to Copilot with built-in retry.
///
/// Delegates to [`send_with_retry`] for the HTTP layer (handles 5xx, 429,
/// and transport errors transparently).  Auth errors (4xx other than 429) are
/// surfaced as [`CopilotHttpError`] so the caller can refresh the token.
///
/// Text chunks are forwarded to `writer` as `Response::Text` frames as they
/// arrive.  Returns a [`CopilotResponse`] describing the full turn result.
pub async fn stream_chat<W>(
    http: &reqwest::Client,
    token: &str,
    model: &str,
    messages: &[Message],
    tools: &[serde_json::Value],
    writer: &mut W,
) -> Result<CopilotResponse>
where
    W: AsyncWriteExt + Unpin,
{
    tracing::debug!(messages = messages.len(), "sending chat request to Copilot");
    let resp = send_with_retry(http, token, model, messages, tools).await?;
    parse_sse_stream(resp, writer).await
}

/// Protocol marker the model emits to signal it needs interactive input.
/// Stripped from streamed Text chunks so it never appears in the terminal.
const WAITING_MARKER: &str = "[WAITING_FOR_INPUT]";

async fn parse_sse_stream<W>(resp: reqwest::Response, writer: &mut W) -> Result<CopilotResponse>
where
    W: AsyncWriteExt + Unpin,
{
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut acc = SseAccumulator::default();

    // Buffer the opening chars of the response to check for WAITING_MARKER
    // before forwarding any Text frames to the client.
    let mut prefix_buf = String::with_capacity(WAITING_MARKER.len());
    let mut prefix_checked = false;

    while let Some(chunk) = stream.next().await {
        let bytes = chunk.context("reading SSE chunk")?;
        buf.push_str(&String::from_utf8_lossy(&bytes));

        while let Some(newline) = buf.find('\n') {
            let line = buf[..newline].trim_end_matches('\r').to_owned();
            buf = buf[newline + 1..].to_owned();

            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    // Flush any buffered prefix that never reached WAITING_MARKER.len()
                    // (response was shorter than the marker — cannot be a marker).
                    if !prefix_checked && !prefix_buf.is_empty() {
                        write_frame(
                            writer,
                            &Response::Text {
                                chunk: prefix_buf.clone(),
                            },
                        )
                        .await?;
                        writer.flush().await?;
                    }
                    tracing::debug!("SSE stream complete");
                    return Ok(acc.finalize());
                }
                match serde_json::from_str::<ChatChunk>(data) {
                    Ok(chunk) => {
                        // Forward any text content to the client immediately,
                        // suppressing the WAITING_MARKER prefix if present.
                        for choice in &chunk.choices {
                            if let Some(ref text) = choice.delta.content {
                                if !text.is_empty() {
                                    if prefix_checked {
                                        write_frame(
                                            writer,
                                            &Response::Text {
                                                chunk: text.clone(),
                                            },
                                        )
                                        .await?;
                                        writer.flush().await?;
                                    } else {
                                        prefix_buf.push_str(text);
                                        if prefix_buf.len() >= WAITING_MARKER.len() {
                                            prefix_checked = true;
                                            if prefix_buf.starts_with(WAITING_MARKER) {
                                                // Suppress the marker; forward only the remainder.
                                                let rest =
                                                    prefix_buf[WAITING_MARKER.len()..].to_owned();
                                                prefix_buf.clear();
                                                if !rest.is_empty() {
                                                    write_frame(
                                                        writer,
                                                        &Response::Text { chunk: rest },
                                                    )
                                                    .await?;
                                                    writer.flush().await?;
                                                }
                                            } else {
                                                // Not a marker — flush the buffer as-is.
                                                let to_forward = std::mem::take(&mut prefix_buf);
                                                write_frame(
                                                    writer,
                                                    &Response::Text { chunk: to_forward },
                                                )
                                                .await?;
                                                writer.flush().await?;
                                            }
                                        }
                                        // else: still buffering; wait for more chunks.
                                    }
                                }
                            }
                        }
                        acc.apply_chunk(&chunk);
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "skipping unparseable SSE chunk");
                    }
                }
            }
        }
    }

    // Stream closed without [DONE].
    tracing::debug!("SSE stream closed without [DONE]");
    Ok(acc.finalize())
}
