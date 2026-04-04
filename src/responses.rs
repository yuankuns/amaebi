//! OpenAI Responses API backend (`POST /v1/responses`).
//!
//! Newer GitHub Copilot models (gpt-5.x, etc.) are only accessible via the
//! Responses API; they return 400 `unsupported_api_for_model` when called
//! through the older `/v1/chat/completions` endpoint.
//!
//! The Responses API uses a different wire format:
//! - Request body: `input` array (not `messages`), flat tool definitions.
//! - Response SSE: `response.output_text.delta` /
//!   `response.function_call_arguments.*` events instead of `choices[].delta` events.

use anyhow::{Context, Result};
use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;

use crate::copilot::{CopilotHttpError, CopilotResponse, FinishReason, Message, ToolCall};
use crate::ipc::{write_frame, Response};
use crate::retry::{backoff_delay, parse_retry_after_header};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Build the Responses API URL from the account-specific `base_url`.
///
/// `base_url` is derived from the `proxy-ep` field in the Copilot JWT and
/// already points to the correct per-account gateway (individual, enterprise,
/// etc.).  Appending `/v1/responses` gives the Responses API endpoint for
/// that account.
///
/// `AMAEBI_COPILOT_URL` is a test-only override; when set it is used as the
/// base URL so that tests pointing at a single mock server work unchanged.
fn responses_endpoint(base_url: &str) -> String {
    if let Ok(url) = std::env::var("AMAEBI_COPILOT_URL") {
        let trimmed = url.trim();
        if !trimmed.is_empty() {
            // AMAEBI_COPILOT_URL is typically set to the full /chat/completions
            // URL in tests.  Strip that path suffix so both Chat Completions and
            // the Responses API share the same host, then append /v1/responses.
            // This lets a single mock server handle both endpoints.
            let base = trimmed
                .trim_end_matches("/chat/completions")
                .trim_end_matches('/');
            return format!("{base}/v1/responses");
        }
    }
    format!("{}/v1/responses", base_url.trim_end_matches('/'))
}
const MAX_RETRIES: u32 = 3;

// ---------------------------------------------------------------------------
// Message format conversion: Chat Completions → Responses API
// ---------------------------------------------------------------------------

/// Convert the Chat Completions `messages` array to the Responses API `input`
/// array.
///
/// Conversion rules:
/// - `system` → `{"role":"system","content":"..."}`
/// - `user` → `{"role":"user","content":[{"type":"input_text","text":"..."}]}`
/// - `assistant` (text) → `{"type":"message","role":"assistant",...}`
/// - `assistant` (tool calls) → one `{"type":"function_call",...}` per call
/// - `tool` → `{"type":"function_call_output","call_id":"...","output":"..."}`
pub(crate) fn to_responses_input(messages: &[Message]) -> Vec<serde_json::Value> {
    let mut out: Vec<serde_json::Value> = Vec::new();
    let mut msg_idx = 0usize;

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                if let Some(ref text) = msg.content {
                    out.push(serde_json::json!({
                        "role": "system",
                        "content": text
                    }));
                }
            }

            "user" => {
                let text = msg.content.as_deref().unwrap_or("");
                // Tool-result messages piggy-backed on role=user (some callers do this).
                if let Some(ref call_id) = msg.tool_call_id {
                    out.push(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": text
                    }));
                } else {
                    out.push(serde_json::json!({
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}]
                    }));
                }
            }

            "tool" => {
                let text = msg.content.as_deref().unwrap_or("");
                if let Some(ref call_id) = msg.tool_call_id {
                    out.push(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": text
                    }));
                }
            }

            "assistant" => {
                // Emit assistant text first (if any), then function_call items.
                // Both can be present in the same turn (text + tool call).
                if let Some(ref text) = msg.content {
                    if !text.is_empty() {
                        let id = format!("msg_{msg_idx}");
                        msg_idx += 1;
                        out.push(serde_json::json!({
                            "type": "message",
                            "role": "assistant",
                            "id": id,
                            "status": "completed",
                            "content": [{"type": "output_text", "text": text, "annotations": []}]
                        }));
                    }
                }
                for tc in &msg.tool_calls {
                    out.push(serde_json::json!({
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }));
                }
            }

            _ => {}
        }
    }

    out
}

/// Convert the Chat Completions tool schema array to the Responses API format.
///
/// Chat Completions: `[{"type":"function","function":{"name":...,"description":...,"parameters":...}}]`
/// Responses API:    `[{"type":"function","name":...,"description":...,"parameters":...,"strict":false}]`
pub(crate) fn to_responses_tools(tools: &[serde_json::Value]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .filter_map(|t| {
            let func = t.get("function")?;
            let name = func.get("name")?.as_str()?;
            let description = func
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let parameters = func
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({"type":"object","properties":{}}));
            Some(serde_json::json!({
                "type": "function",
                "name": name,
                "description": description,
                "parameters": parameters,
                "strict": false
            }))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// HTTP send with retry
// ---------------------------------------------------------------------------

/// Infer the `X-Initiator` header value.
///
/// Copilot uses this to distinguish user-driven turns from agent follow-ups.
fn infer_initiator(messages: &[Message]) -> &'static str {
    match messages.last().map(|m| m.role.as_str()) {
        Some("user") => "user",
        _ => "agent",
    }
}

#[allow(clippy::too_many_arguments)]
async fn send_with_retry(
    http: &reqwest::Client,
    token: &str,
    base_url: &str,
    model: &str,
    messages: &[Message],
    tools: &[serde_json::Value],
    max_tokens: usize,
) -> Result<reqwest::Response> {
    let input = to_responses_input(messages);
    let responses_tools = to_responses_tools(tools);

    let mut body = serde_json::json!({
        "model": model,
        "input": input,
        "stream": true,
        "max_output_tokens": max_tokens,
        "store": false,
    });
    if !responses_tools.is_empty() {
        body["tools"] = serde_json::Value::Array(responses_tools);
    }

    let url = responses_endpoint(base_url);

    let initiator = infer_initiator(messages);

    let mut attempt = 0u32;
    loop {
        let result = http
            .post(&url)
            .header("Authorization", format!("Bearer {token}"))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .header("User-Agent", concat!("amaebi/", env!("CARGO_PKG_VERSION")))
            .header("Copilot-Integration-Id", "vscode-chat")
            .header("Editor-Version", "vscode/1.90.0")
            .header("X-Initiator", initiator)
            .header("Openai-Intent", "conversation-edits")
            .json(&body)
            .send()
            .await;

        match result {
            Ok(resp) if resp.status().is_success() => return Ok(resp),

            Ok(resp) if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS => {
                if attempt >= MAX_RETRIES {
                    let status = resp.status();
                    let body_text = resp.text().await.unwrap_or_default();
                    return Err(anyhow::Error::new(CopilotHttpError {
                        status,
                        body: body_text,
                    }));
                }
                let delay = parse_retry_after_header(resp.headers())
                    .unwrap_or_else(|| backoff_delay(attempt));
                tracing::warn!(
                    attempt,
                    delay_ms = delay.as_millis(),
                    endpoint = %url,
                    "Responses API rate-limited (429); backing off"
                );
                let _ = resp.bytes().await;
                tokio::time::sleep(delay).await;
                attempt += 1;
            }

            Ok(resp) if resp.status().is_server_error() => {
                if attempt >= MAX_RETRIES {
                    let status = resp.status();
                    let body_text = resp.text().await.unwrap_or_default();
                    return Err(anyhow::Error::new(CopilotHttpError {
                        status,
                        body: body_text,
                    }));
                }
                let delay = backoff_delay(attempt);
                tracing::warn!(
                    attempt,
                    delay_ms = delay.as_millis(),
                    endpoint = %url,
                    "Responses API server error; retrying"
                );
                let _ = resp.bytes().await;
                tokio::time::sleep(delay).await;
                attempt += 1;
            }

            Ok(resp) => {
                let status = resp.status();
                let body_text = resp.text().await.unwrap_or_default();
                return Err(anyhow::Error::new(CopilotHttpError {
                    status,
                    body: body_text,
                }));
            }

            Err(e) => {
                if attempt >= MAX_RETRIES {
                    return Err(anyhow::Error::from(e)
                        .context(format!("sending request to Responses API ({url})")));
                }
                let delay = backoff_delay(attempt);
                tracing::warn!(
                    attempt,
                    error = %e,
                    delay_ms = delay.as_millis(),
                    "Responses API transport error; retrying"
                );
                tokio::time::sleep(delay).await;
                attempt += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SSE stream parser for the Responses API
// ---------------------------------------------------------------------------

#[derive(Default)]
struct PartialToolCall {
    call_id: String,
    name: String,
    arguments: String,
}

/// Parse the Responses API SSE stream into a `CopilotResponse`.
///
/// Key events handled:
/// - `response.output_item.added` — function_call item start
/// - `response.output_text.delta` — text chunk
/// - `response.function_call_arguments.delta` — tool call args fragment
/// - `response.function_call_arguments.done` — tool call args finalised
/// - `response.output_item.done` — function_call item done
/// - `response.completed` — usage / stop reason
async fn parse_responses_stream<W>(
    resp: reqwest::Response,
    writer: &mut W,
) -> Result<CopilotResponse>
where
    W: AsyncWriteExt + Unpin,
{
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();

    let mut text = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    // At most one in-flight function_call item at a time.
    let mut pending: Option<PartialToolCall> = None;
    let mut finish_reason = FinishReason::Stop;
    let mut prompt_tokens = 0usize;

    while let Some(chunk) = stream.next().await {
        let bytes = chunk.context("reading Responses API SSE chunk")?;
        buf.push_str(&String::from_utf8_lossy(&bytes));

        while let Some(newline) = buf.find('\n') {
            let line = buf[..newline].trim_end_matches('\r').to_owned();
            buf = buf[newline + 1..].to_owned();

            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };

            if data == "[DONE]" {
                tracing::debug!("Responses API SSE stream complete");
                return Ok(build_response(
                    text,
                    tool_calls,
                    finish_reason,
                    prompt_tokens,
                ));
            }

            let event: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(error = %e, "skipping unparseable Responses API event");
                    continue;
                }
            };

            let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");

            match event_type {
                // A new output item is starting.  We only care about
                // function_call items here; message items produce text via
                // response.output_text.delta.
                "response.output_item.added" => {
                    if let Some(item) = event.get("item") {
                        if item.get("type").and_then(|v| v.as_str()) == Some("function_call") {
                            let call_id = item
                                .get("call_id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let name = item
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            pending = Some(PartialToolCall {
                                call_id,
                                name,
                                arguments: String::new(),
                            });
                        }
                    }
                }

                // Text delta — forward immediately.
                "response.output_text.delta" => {
                    if let Some(delta) = event.get("delta").and_then(|v| v.as_str()) {
                        text.push_str(delta);
                        if !delta.is_empty() {
                            write_frame(
                                writer,
                                &Response::Text {
                                    chunk: delta.to_string(),
                                },
                            )
                            .await?;
                            writer.flush().await?;
                        }
                    }
                }

                // Tool call argument fragment.
                "response.function_call_arguments.delta" => {
                    if let Some(delta) = event.get("delta").and_then(|v| v.as_str()) {
                        if let Some(ref mut p) = pending {
                            p.arguments.push_str(delta);
                        }
                    }
                }

                // Tool call arguments fully streamed.
                "response.function_call_arguments.done" => {
                    if let Some(args) = event.get("arguments").and_then(|v| v.as_str()) {
                        if let Some(ref mut p) = pending {
                            p.arguments = args.to_string();
                        }
                    }
                }

                // Output item (function_call) completed.
                "response.output_item.done" => {
                    if let Some(item) = event.get("item") {
                        if item.get("type").and_then(|v| v.as_str()) == Some("function_call") {
                            if let Some(p) = pending.take() {
                                if !p.name.is_empty() {
                                    tool_calls.push(ToolCall {
                                        id: p.call_id,
                                        name: p.name,
                                        arguments: p.arguments,
                                    });
                                }
                            }
                        }
                    }
                }

                // Stream complete — capture usage and stop reason.
                "response.completed" => {
                    if let Some(response) = event.get("response") {
                        let stop_reason = response
                            .get("stop_reason")
                            .or_else(|| response.get("status"))
                            .and_then(|v| v.as_str());
                        finish_reason = match stop_reason {
                            Some("max_tokens") => FinishReason::Length,
                            Some("tool_calls") => FinishReason::ToolCalls,
                            _ if !tool_calls.is_empty() || pending.is_some() => {
                                FinishReason::ToolCalls
                            }
                            _ => FinishReason::Stop,
                        };
                        if let Some(usage) = response.get("usage") {
                            if let Some(n) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
                                prompt_tokens = n as usize;
                            }
                        }
                    }
                }

                _ => {
                    tracing::debug!(event_type, "unhandled Responses API event");
                }
            }
        }
    }

    // Stream closed without [DONE] — flush any in-flight tool call so it is
    // not silently dropped when the server closes the connection early.
    if let Some(p) = pending.take() {
        if !p.name.is_empty() {
            tool_calls.push(ToolCall {
                id: p.call_id,
                name: p.name,
                arguments: p.arguments,
            });
        }
    }
    tracing::debug!("Responses API SSE stream closed without [DONE]");
    Ok(build_response(
        text,
        tool_calls,
        finish_reason,
        prompt_tokens,
    ))
}

fn build_response(
    text: String,
    tool_calls: Vec<ToolCall>,
    finish_reason: FinishReason,
    prompt_tokens: usize,
) -> CopilotResponse {
    CopilotResponse {
        text,
        tool_calls,
        finish_reason,
        prompt_tokens,
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Send a streaming chat request using the OpenAI Responses API.
///
/// Used when the Chat Completions endpoint returns 400
/// `unsupported_api_for_model` — newer Copilot models (gpt-5.x, etc.) are
/// only accessible via this newer endpoint.
#[allow(clippy::too_many_arguments)]
pub async fn stream_chat<W>(
    http: &reqwest::Client,
    token: &str,
    base_url: &str,
    model: &str,
    messages: &[Message],
    tools: &[serde_json::Value],
    max_tokens: usize,
    writer: &mut W,
) -> Result<CopilotResponse>
where
    W: AsyncWriteExt + Unpin,
{
    tracing::debug!(
        messages = messages.len(),
        model,
        "sending chat request via Responses API"
    );
    let resp = send_with_retry(http, token, base_url, model, messages, tools, max_tokens).await?;
    parse_responses_stream(resp, writer).await
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::copilot::{ApiToolCall, ApiToolCallFunction};

    // ---- to_responses_input ------------------------------------------------

    #[test]
    fn system_message_converted() {
        let msgs = vec![Message::system("Be helpful.")];
        let input = to_responses_input(&msgs);
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["role"], "system");
        assert_eq!(input[0]["content"], "Be helpful.");
    }

    #[test]
    fn user_message_wrapped_in_input_text() {
        let msgs = vec![Message::user("hello")];
        let input = to_responses_input(&msgs);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[0]["content"][0]["text"], "hello");
    }

    #[test]
    fn assistant_text_becomes_output_message() {
        let msgs = vec![
            Message::user("hi"),
            Message::assistant(Some("hello".into()), vec![]),
        ];
        let input = to_responses_input(&msgs);
        let asst = &input[1];
        assert_eq!(asst["type"], "message");
        assert_eq!(asst["role"], "assistant");
        assert_eq!(asst["status"], "completed");
        assert_eq!(asst["content"][0]["type"], "output_text");
        assert_eq!(asst["content"][0]["text"], "hello");
    }

    #[test]
    fn tool_call_becomes_function_call_item() {
        let tc = ApiToolCall {
            id: "call_001".into(),
            kind: "function".into(),
            function: ApiToolCallFunction {
                name: "shell_command".into(),
                arguments: r#"{"command":"ls"}"#.into(),
            },
        };
        let msgs = vec![Message::user("run ls"), Message::assistant(None, vec![tc])];
        let input = to_responses_input(&msgs);
        let fc = &input[1];
        assert_eq!(fc["type"], "function_call");
        assert_eq!(fc["call_id"], "call_001");
        assert_eq!(fc["name"], "shell_command");
        assert_eq!(fc["arguments"], r#"{"command":"ls"}"#);
    }

    #[test]
    fn assistant_text_and_tool_calls_both_emitted() {
        // Regression: when an assistant turn has both text and tool calls,
        // to_responses_input must emit the text message AND the function_call
        // items — not drop the text in favour of the tool calls only.
        let tc = ApiToolCall {
            id: "call_002".into(),
            kind: "function".into(),
            function: ApiToolCallFunction {
                name: "read_file".into(),
                arguments: r#"{"path":"x"}"#.into(),
            },
        };
        let msgs = vec![
            Message::user("do both"),
            Message::assistant(Some("I will read the file.".into()), vec![tc]),
        ];
        let input = to_responses_input(&msgs);
        // Expect: user + output_text message + function_call = 3 items.
        assert_eq!(input.len(), 3, "expected 3 input items, got: {input:?}");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[1]["type"], "message");
        assert_eq!(input[1]["content"][0]["text"], "I will read the file.");
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[2]["call_id"], "call_002");
    }

    #[test]
    fn tool_result_becomes_function_call_output() {
        let msgs = vec![Message::tool_result("call_001", "file.txt\n")];
        let input = to_responses_input(&msgs);
        assert_eq!(input[0]["type"], "function_call_output");
        assert_eq!(input[0]["call_id"], "call_001");
        assert_eq!(input[0]["output"], "file.txt\n");
    }

    // ---- to_responses_tools ------------------------------------------------

    #[test]
    fn tool_schema_flattened() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "shell_command",
                "description": "Run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]
                }
            }
        })];
        let out = to_responses_tools(&tools);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["type"], "function");
        assert_eq!(out[0]["name"], "shell_command");
        assert_eq!(out[0]["description"], "Run a shell command");
        assert!(out[0]["parameters"].is_object());
        // Must NOT have nested "function" key.
        assert!(out[0].get("function").is_none());
        assert_eq!(out[0]["strict"], false);
    }

    #[test]
    fn empty_tools_returns_empty() {
        assert!(to_responses_tools(&[]).is_empty());
    }

    // ---- infer_initiator ---------------------------------------------------

    #[test]
    fn user_last_message_gives_user_initiator() {
        let msgs = vec![Message::user("hello")];
        assert_eq!(infer_initiator(&msgs), "user");
    }

    #[test]
    fn tool_result_last_gives_agent_initiator() {
        let msgs = vec![Message::user("run"), Message::tool_result("c1", "result")];
        assert_eq!(infer_initiator(&msgs), "agent");
    }

    #[test]
    fn empty_messages_gives_agent_initiator() {
        assert_eq!(infer_initiator(&[]), "agent");
    }
}
