use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::io::AsyncWriteExt;

use crate::ipc::{write_frame, Response};

const CHAT_ENDPOINT: &str = "https://api.githubcopilot.com/chat/completions";

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

/// One SSE data line.  The Copilot API sometimes puts text in choices[0] and
/// tool_calls in choices[1] — we always collect all choices.
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
// Public streaming function
// ---------------------------------------------------------------------------

/// Send a streaming chat-completions request to Copilot.
///
/// Text chunks are forwarded to `writer` as `Response::Text` frames as they
/// arrive.  The caller is responsible for writing `Response::Done` (so the
/// agentic loop can decide whether to loop or finish).
///
/// Returns a `CopilotResponse` describing the full turn result.
pub async fn stream_chat<W>(
    http: &reqwest::Client,
    token: &str,
    messages: &[Message],
    tools: &[serde_json::Value],
    writer: &mut W,
) -> Result<CopilotResponse>
where
    W: AsyncWriteExt + Unpin,
{
    let body = serde_json::json!({
        "model": "gpt-4o",
        "messages": messages,
        "tools": tools,
        "stream": true,
        "max_tokens": 4096,
    });

    tracing::debug!(messages = messages.len(), "sending chat request to Copilot");

    let resp = http
        .post(CHAT_ENDPOINT)
        .header("Authorization", format!("Bearer {token}"))
        .header("Content-Type", "application/json")
        .header("Accept", "application/json")
        .header("Copilot-Integration-Id", "vscode-chat")
        .header("Editor-Version", "vscode/1.90.0")
        .header(
            "User-Agent",
            concat!("tmux-copilot/", env!("CARGO_PKG_VERSION")),
        )
        .json(&body)
        .send()
        .await
        .context("sending chat request to Copilot")?
        .error_for_status()
        .context("Copilot chat endpoint returned an error")?;

    parse_sse_stream(resp, writer).await
}

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
