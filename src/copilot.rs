use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

use crate::ipc::{write_frame, Response};

const CHAT_ENDPOINT: &str = "https://api.githubcopilot.com/chat/completions";

// ---------------------------------------------------------------------------
// API message types
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Message {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: Some(content.into()),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: Some(content.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// SSE response types (internal, for deserializing streaming chunks)
// ---------------------------------------------------------------------------

#[derive(Deserialize, Debug)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct ChatChunk {
    /// The Copilot API sometimes puts text in choices[0] and tool_calls in
    /// choices[1]; we collect all choices here.  See project notes.
    choices: Vec<Choice>,
}

// ---------------------------------------------------------------------------
// Streaming chat
// ---------------------------------------------------------------------------

/// Stream a chat-completions request to `writer`, forwarding each text chunk
/// as a `Response::Text` frame and finishing with `Response::Done`.
///
/// `writer` is the write half of a connected Unix socket.
pub async fn stream_chat<W>(
    http: &reqwest::Client,
    token: &str,
    messages: Vec<Message>,
    writer: &mut W,
) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let body = serde_json::json!({
        "model": "gpt-4o",
        "messages": messages,
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
        .context("sending chat request")?
        .error_for_status()
        .context("Copilot chat endpoint returned an error")?;

    parse_sse_stream(resp, writer).await
}

/// Read a Server-Sent Events response body and forward text deltas to `writer`.
async fn parse_sse_stream<W>(resp: reqwest::Response, writer: &mut W) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let mut stream = resp.bytes_stream();
    // Accumulate bytes until we have complete `\n`-terminated SSE lines.
    let mut buf = String::new();

    while let Some(chunk) = stream.next().await {
        let bytes = chunk.context("reading SSE stream chunk")?;
        // SSE is always UTF-8; replace any invalid bytes rather than failing.
        buf.push_str(&String::from_utf8_lossy(&bytes));

        // Process every complete line in the buffer.
        while let Some(newline) = buf.find('\n') {
            let line = buf[..newline].trim_end_matches('\r').to_owned();
            buf = buf[newline + 1..].to_owned();

            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    tracing::debug!("SSE stream finished");
                    write_frame(writer, &Response::Done).await?;
                    return Ok(());
                }
                if let Err(e) = forward_chunk(data, writer).await {
                    tracing::warn!(error = %e, "skipping unparseable SSE chunk");
                }
            }
        }
    }

    // Stream closed without a [DONE] — treat as end-of-stream.
    tracing::debug!("SSE stream closed without [DONE]");
    write_frame(writer, &Response::Done).await
}

/// Parse one SSE data line and forward any text content to `writer`.
async fn forward_chunk<W>(data: &str, writer: &mut W) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let chunk: ChatChunk = serde_json::from_str(data).context("parsing SSE JSON chunk")?;

    // Collect text from all choices — Copilot sometimes spreads content across
    // choices[0] and tool_calls across choices[1].
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
        if let Some(ref reason) = choice.finish_reason {
            tracing::debug!(finish_reason = %reason, "choice finished");
        }
    }

    Ok(())
}
