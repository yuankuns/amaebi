use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::io::AsyncWriteExt;

use crate::ipc::{write_frame, Response};

/// Chat-completions path suffix appended to the base URL.
const CHAT_PATH: &str = "/chat/completions";

/// Build the full chat-completions URL from a Copilot API base URL.
///
/// The base URL is derived from the `proxy-ep` field embedded in the Copilot
/// JWT (see `auth::base_url_from_token`), so all models — Claude, GPT, Gemini
/// — are routed through the user's own Copilot API endpoint rather than a
/// hardcoded host.
///
/// `AMAEBI_COPILOT_URL` is a test-only override that redirects all requests
/// to a local mock server; it is never set in production.
pub(crate) fn chat_endpoint(base_url: &str) -> String {
    if let Ok(url) = std::env::var("AMAEBI_COPILOT_URL") {
        if !url.trim().is_empty() {
            return url.trim().to_string();
        }
    }
    format!("{base_url}{CHAT_PATH}")
}

// ---------------------------------------------------------------------------
// Retry policy constants
// ---------------------------------------------------------------------------

/// How many times to retry transient failures (5xx, 429, network errors)
/// before surfacing the error to the caller.
const MAX_RETRIES: u32 = 3;

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
    /// The server completed the turn but flagged the output as malformed —
    /// currently only emitted by Bedrock ConverseStream with stopReason
    /// `malformed_model_output` or `malformed_tool_use` (added in the
    /// 2025-12-02 `aws-sdk-bedrockruntime` 1.119 release).
    ///
    /// These are treated as transient: the server returned HTTP 200 with an
    /// application-level "model botched this one turn" signal, and retrying
    /// the same request with the same messages usually succeeds.  The daemon
    /// retries once before falling back to the same termination path as
    /// [`FinishReason::Other`].
    Malformed,
    Other(String),
}

/// Everything the daemon needs after one round-trip with the Copilot API.
pub struct CopilotResponse {
    /// Accumulated text content (empty when the turn was tool-call-only).
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: FinishReason,
    /// Input tokens consumed by this request, as reported by the API.
    /// Zero when the server did not include usage data (e.g. older endpoints).
    pub prompt_tokens: usize,
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

/// Token usage reported in the final SSE chunk when `stream_options.include_usage` is set.
#[derive(Deserialize, Debug, Default)]
struct Usage {
    #[serde(default)]
    prompt_tokens: usize,
}

/// One SSE data line.  The Copilot API sometimes puts text in choices\[0\] and
/// tool_calls in choices\[1\] — we always collect all choices.
/// The final chunk (just before `[DONE]`) carries usage when requested.
#[derive(Deserialize, Debug)]
struct ChatChunk {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
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
    /// Captured from the final usage chunk; zero if server did not include it.
    prompt_tokens: usize,
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
        // Capture usage from the final chunk (non-zero when server includes it).
        if let Some(ref usage) = chunk.usage {
            if usage.prompt_tokens > 0 {
                self.prompt_tokens = usage.prompt_tokens;
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
            prompt_tokens: self.prompt_tokens,
        }
    }
}

// ---------------------------------------------------------------------------
// Retry helpers (shared utilities re-exported from retry module)
// ---------------------------------------------------------------------------

/// Exponential back-off: 1 s, 2 s, 4 s for attempts 0, 1, 2 (capped at 10).
pub(crate) use crate::retry::backoff_delay;

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
#[allow(clippy::too_many_arguments)]
async fn send_with_retry(
    http: &reqwest::Client,
    token: &str,
    model: &str,
    messages: &[Message],
    tools: &[serde_json::Value],
    max_completion_tokens: usize,
    endpoint_url: &str,
    use_copilot_headers: bool,
) -> Result<reqwest::Response> {
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": true,
        "max_completion_tokens": max_completion_tokens,
        "stream_options": { "include_usage": true },
    });

    let mut attempt = 0u32;
    loop {
        let mut req = http
            .post(endpoint_url)
            .header("Authorization", format!("Bearer {token}"))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .header("User-Agent", concat!("amaebi/", env!("CARGO_PKG_VERSION")));
        if use_copilot_headers {
            req = req
                .header("Copilot-Integration-Id", "vscode-chat")
                .header("Editor-Version", "vscode/1.90.0");
        }
        let result = req.json(&body).send().await;

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
                let delay = crate::retry::parse_retry_after_header(resp.headers())
                    .unwrap_or_else(|| backoff_delay(attempt));
                tracing::warn!(
                    attempt,
                    delay_ms = delay.as_millis(),
                    endpoint = %endpoint_url,
                    "API rate-limited (429); backing off before retry"
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
                    endpoint = %endpoint_url,
                    "API server error; retrying with exponential backoff"
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
                    return Err(anyhow::Error::from(e)
                        .context(format!("sending chat request to {endpoint_url}")));
                }
                let delay = backoff_delay(attempt);
                tracing::warn!(
                    attempt,
                    error = %e,
                    delay_ms = delay.as_millis(),
                    endpoint = %endpoint_url,
                    "API request failed (transport error); retrying"
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
/// Send a streaming chat request to the Copilot API.
///
/// `token` is the short-lived Copilot JWT; `base_url` is derived from the
/// `proxy-ep` field embedded in that JWT (see `auth::base_url_from_token`).
/// All models — Claude, GPT, Gemini — are routed through the same endpoint;
/// the Copilot API handles model dispatch internally.
#[allow(clippy::too_many_arguments)]
pub async fn stream_chat<W>(
    http: &reqwest::Client,
    token: &str,
    base_url: &str,
    model: &str,
    messages: &[Message],
    tools: &[serde_json::Value],
    max_completion_tokens: usize,
    writer: &mut W,
) -> Result<CopilotResponse>
where
    W: AsyncWriteExt + Unpin,
{
    let endpoint = chat_endpoint(base_url);
    tracing::debug!(messages = messages.len(), endpoint = %endpoint, model, "sending chat request");
    let resp = send_with_retry(
        http,
        token,
        model,
        messages,
        tools,
        max_completion_tokens,
        &endpoint,
        true,
    )
    .await?;
    parse_sse_stream(resp, writer).await
}

/// Protocol marker the model emits to signal it needs interactive input.
/// Detected in the fully-accumulated `resp.text` after streaming completes;
/// exported so `daemon.rs` can share the same constant.
pub const WAITING_MARKER: &str = "[WAITING_FOR_INPUT]";

async fn parse_sse_stream<W>(resp: reqwest::Response, writer: &mut W) -> Result<CopilotResponse>
where
    W: AsyncWriteExt + Unpin,
{
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut acc = SseAccumulator::default();

    while let Some(chunk) = stream.next().await {
        let bytes = chunk.context("reading SSE chunk")?;
        buf.push_str(&String::from_utf8_lossy(&bytes));

        while let Some(newline) = buf.find('\n') {
            let line = buf[..newline].trim_end_matches('\r').to_owned();
            buf = buf[newline + 1..].to_owned();

            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    tracing::debug!("SSE stream complete");
                    return Ok(acc.finalize());
                }
                match serde_json::from_str::<ChatChunk>(data) {
                    Ok(chunk) => {
                        // Forward any text content to the client immediately.
                        for choice in &chunk.choices {
                            if let Some(ref text) = choice.delta.content {
                                if !text.is_empty() {
                                    write_frame(
                                        writer,
                                        &Response::Text {
                                            chunk: text.clone(),
                                        },
                                    )
                                    .await?;
                                    writer.flush().await?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retry::{parse_retry_after_header, MAX_RETRY_AFTER_SECS};
    use reqwest::header::{HeaderMap, HeaderValue, RETRY_AFTER};
    use std::time::Duration;

    // ---- chat_endpoint -----------------------------------------------------

    #[test]
    #[serial_test::serial]
    fn chat_endpoint_uses_provided_base_url() {
        // The endpoint is constructed from the caller-supplied base URL
        // (derived from proxy-ep in the Copilot JWT).
        // Serialized so it cannot race with chat_endpoint_test_override_wins,
        // which sets AMAEBI_COPILOT_URL.
        std::env::remove_var("AMAEBI_COPILOT_URL");
        assert_eq!(
            chat_endpoint("https://api.individual.githubcopilot.com"),
            "https://api.individual.githubcopilot.com/chat/completions"
        );
    }

    #[test]
    #[serial_test::serial]
    fn chat_endpoint_test_override_wins() {
        std::env::set_var("AMAEBI_COPILOT_URL", "http://mock:1234/chat/completions");
        let result = chat_endpoint("https://api.individual.githubcopilot.com");
        std::env::remove_var("AMAEBI_COPILOT_URL");
        assert_eq!(result, "http://mock:1234/chat/completions");
    }

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
