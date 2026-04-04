//! Amazon Bedrock ConverseStream backend.
//!
//! Calls the Bedrock Runtime `ConverseStream` API with bearer-token
//! authentication (`AWS_BEARER_TOKEN_BEDROCK` env var).
//!
//! The response uses the AWS event-stream binary protocol — each frame is
//! a length-prefixed message with typed headers and a JSON payload.  A
//! minimal parser is implemented inline (see [`eventstream`]) so we avoid
//! pulling in the full AWS SDK.

use anyhow::{Context, Result};
use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;

use crate::copilot::{CopilotResponse, FinishReason, Message, ToolCall};
use crate::ipc::{write_frame, Response};
use crate::retry::{backoff_delay, parse_retry_after_header};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum retries for transient errors (same policy as copilot.rs).
const MAX_RETRIES: u32 = 3;

/// Default AWS region when `AWS_REGION` is not set.
const DEFAULT_REGION: &str = "us-east-1";

// ---------------------------------------------------------------------------
// Auth helpers
// ---------------------------------------------------------------------------

/// Read the Bedrock bearer token from the environment.
fn read_bearer_token() -> Result<String> {
    std::env::var("AWS_BEARER_TOKEN_BEDROCK").map_err(|_| {
        anyhow::anyhow!(
            "AWS_BEARER_TOKEN_BEDROCK is not set. \
             Set this environment variable with your Bedrock bearer token."
        )
    })
}

/// Resolve the AWS region from the environment.
fn aws_region() -> String {
    std::env::var("AWS_REGION")
        .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
        .unwrap_or_else(|_| DEFAULT_REGION.to_owned())
}

/// Percent-encode a string for use as a single URL path segment.
///
/// Encodes everything except RFC 3986 unreserved characters:
/// `ALPHA / DIGIT / "-" / "." / "_" / "~"`
fn percent_encode_path_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &byte in s.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(byte as char);
            }
            _ => {
                out.push('%');
                out.push(char::from(b"0123456789ABCDEF"[(byte >> 4) as usize]));
                out.push(char::from(b"0123456789ABCDEF"[(byte & 0x0F) as usize]));
            }
        }
    }
    out
}

/// Build the ConverseStream endpoint URL.
///
/// `AMAEBI_BEDROCK_URL` is a test-only override (analogous to
/// `AMAEBI_COPILOT_URL` in copilot.rs).
fn converse_stream_endpoint(region: &str, model_id: &str) -> String {
    if let Ok(url) = std::env::var("AMAEBI_BEDROCK_URL") {
        let trimmed = url.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    // Percent-encode the model ID for use as a URL path segment.
    // Bedrock model IDs may contain `:` (e.g. `…-v1:0`) or other reserved
    // characters; encode everything except unreserved chars (RFC 3986).
    let encoded_model = percent_encode_path_segment(model_id);
    format!("https://bedrock-runtime.{region}.amazonaws.com/model/{encoded_model}/converse-stream")
}

// ---------------------------------------------------------------------------
// Message format conversion: internal → Bedrock Converse
// ---------------------------------------------------------------------------

/// Result of converting the internal message array into Bedrock's format.
pub(crate) struct BedrockRequestParts {
    /// System prompts extracted from messages with `role=system`.
    pub system: Vec<serde_json::Value>,
    /// Conversation messages (user / assistant turns).
    pub messages: Vec<serde_json::Value>,
}

/// Convert internal [`Message`] array into Bedrock Converse request parts.
///
/// Conversion rules:
/// - `system` → extracted into `BedrockRequestParts::system`
/// - `user` → `{"role":"user","content":[{"text":"..."}]}`
/// - `assistant` (text only) → `{"role":"assistant","content":[{"text":"..."}]}`
/// - `assistant` (tool calls) → content blocks with `toolUse`
/// - `tool` result → merged into the next/preceding user turn as `toolResult`
///
/// Bedrock requires strictly alternating user/assistant roles.  Consecutive
/// messages of the same role are merged into one message with a combined
/// content array.
pub(crate) fn to_bedrock_request(messages: &[Message]) -> BedrockRequestParts {
    let mut system_blocks: Vec<serde_json::Value> = Vec::new();
    // Intermediate (role, content_blocks) pairs before merging.
    let mut turns: Vec<(String, Vec<serde_json::Value>)> = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                if let Some(ref text) = msg.content {
                    if !text.is_empty() {
                        system_blocks.push(serde_json::json!({"text": text}));
                    }
                }
            }

            "user" => {
                let text = msg.content.as_deref().unwrap_or("");
                let block = serde_json::json!({"text": text});
                push_or_merge(&mut turns, "user", block);
            }

            "assistant" => {
                let mut blocks: Vec<serde_json::Value> = Vec::new();
                if let Some(ref text) = msg.content {
                    if !text.is_empty() {
                        blocks.push(serde_json::json!({"text": text}));
                    }
                }
                for tc in &msg.tool_calls {
                    // Parse the JSON-encoded arguments string into a Value so
                    // it is embedded as an object, not a string.
                    let input: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                    blocks.push(serde_json::json!({
                        "toolUse": {
                            "toolUseId": tc.id,
                            "name": tc.function.name,
                            "input": input,
                        }
                    }));
                }
                if !blocks.is_empty() {
                    push_or_merge_all(&mut turns, "assistant", blocks);
                }
            }

            "tool" => {
                // Tool results become toolResult blocks inside a user turn.
                let output = msg.content.as_deref().unwrap_or("");
                if let Some(ref call_id) = msg.tool_call_id {
                    let block = serde_json::json!({
                        "toolResult": {
                            "toolUseId": call_id,
                            "content": [{"text": output}],
                        }
                    });
                    push_or_merge(&mut turns, "user", block);
                }
            }

            _ => {}
        }
    }

    let bedrock_messages: Vec<serde_json::Value> = turns
        .into_iter()
        .map(|(role, content)| {
            serde_json::json!({
                "role": role,
                "content": content,
            })
        })
        .collect();

    BedrockRequestParts {
        system: system_blocks,
        messages: bedrock_messages,
    }
}

/// Push a single content block, merging with the last turn if the role matches.
fn push_or_merge(
    turns: &mut Vec<(String, Vec<serde_json::Value>)>,
    role: &str,
    block: serde_json::Value,
) {
    if let Some(last) = turns.last_mut() {
        if last.0 == role {
            last.1.push(block);
            return;
        }
    }
    turns.push((role.to_owned(), vec![block]));
}

/// Push multiple content blocks, merging with the last turn if the role matches.
fn push_or_merge_all(
    turns: &mut Vec<(String, Vec<serde_json::Value>)>,
    role: &str,
    blocks: Vec<serde_json::Value>,
) {
    if let Some(last) = turns.last_mut() {
        if last.0 == role {
            last.1.extend(blocks);
            return;
        }
    }
    turns.push((role.to_owned(), blocks));
}

// ---------------------------------------------------------------------------
// Tool schema conversion
// ---------------------------------------------------------------------------

/// Convert OpenAI-format tool schemas to Bedrock `toolSpec` format.
///
/// Input (OpenAI):
/// ```json
/// [{"type":"function","function":{"name":"...","description":"...","parameters":{...}}}]
/// ```
///
/// Output (Bedrock):
/// ```json
/// [{"toolSpec":{"name":"...","description":"...","inputSchema":{"json":{...}}}}]
/// ```
pub(crate) fn to_bedrock_tools(tools: &[serde_json::Value]) -> Vec<serde_json::Value> {
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
                "toolSpec": {
                    "name": name,
                    "description": description,
                    "inputSchema": { "json": parameters },
                }
            }))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// AWS event-stream binary protocol parser
// ---------------------------------------------------------------------------

/// Minimal AWS event-stream frame parser.
///
/// Reference: <https://docs.aws.amazon.com/transcribe/latest/dg/streaming-format.html>
///
/// Frame layout:
/// ```text
/// | total_len (4B BE) | header_len (4B BE) | prelude_crc (4B) |
/// | headers (header_len bytes) | payload (remaining) | message_crc (4B) |
/// ```
pub(crate) mod eventstream {
    use anyhow::{Context, Result};

    /// A decoded event-stream frame.
    #[derive(Debug)]
    pub struct Frame {
        /// Header values keyed by name.  Only string-typed headers are kept.
        pub headers: std::collections::HashMap<String, String>,
        /// Raw payload bytes (usually JSON).
        pub payload: Vec<u8>,
    }

    /// CRC-32 — used by AWS event-stream for frame checksums.
    ///
    /// The AWS event-stream specification calls for CRC-32C (Castagnoli),
    /// but the Bedrock ConverseStream endpoint empirically uses standard
    /// CRC-32 (ISO 3309 / ITU-T V.42, polynomial 0xEDB88320 reflected).
    /// Verified against real Bedrock responses on 2026-04-03.
    fn crc32(data: &[u8]) -> u32 {
        // Standard CRC-32 polynomial 0x04C11DB7 reflected → 0xEDB88320
        const POLY: u32 = 0xEDB8_8320;
        let mut crc: u32 = 0xFFFF_FFFF;
        for &byte in data {
            crc ^= byte as u32;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ POLY;
                } else {
                    crc >>= 1;
                }
            }
        }
        crc ^ 0xFFFF_FFFF
    }

    /// Try to parse one frame from the front of `buf`.
    ///
    /// Returns `Ok(Some((frame, consumed)))` on success, `Ok(None)` if `buf`
    /// does not yet contain a complete frame, or `Err` on corruption.
    pub fn try_parse_frame(buf: &[u8]) -> Result<Option<(Frame, usize)>> {
        // Need at least 12 bytes to read the prelude (total_len + header_len + CRC).
        if buf.len() < 12 {
            return Ok(None);
        }

        let total_len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        let header_len = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
        let prelude_crc_expected = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]);

        // Structural validation: minimum frame is 16 bytes (prelude 12 + msg CRC 4).
        // header_len must fit within the frame (total - prelude 12 - msg CRC 4).
        if total_len < 16 {
            anyhow::bail!("event-stream frame too small: total_len={total_len}");
        }
        let max_header_len = total_len - 16;
        if header_len > max_header_len {
            anyhow::bail!(
                "event-stream header_len ({header_len}) exceeds frame capacity ({max_header_len})"
            );
        }

        if buf.len() < total_len {
            return Ok(None); // incomplete frame
        }

        // Verify prelude CRC (first 8 bytes).
        let prelude_crc_actual = crc32(&buf[..8]);
        if prelude_crc_actual != prelude_crc_expected {
            anyhow::bail!(
                "event-stream prelude CRC mismatch: expected {prelude_crc_expected:#010X}, \
                 got {prelude_crc_actual:#010X}"
            );
        }

        // Verify message CRC (entire frame minus the last 4 bytes).
        let msg_crc_offset = total_len - 4;
        let msg_crc_expected = u32::from_be_bytes([
            buf[msg_crc_offset],
            buf[msg_crc_offset + 1],
            buf[msg_crc_offset + 2],
            buf[msg_crc_offset + 3],
        ]);
        let msg_crc_actual = crc32(&buf[..msg_crc_offset]);
        if msg_crc_actual != msg_crc_expected {
            anyhow::bail!(
                "event-stream message CRC mismatch: expected {msg_crc_expected:#010X}, \
                 got {msg_crc_actual:#010X}"
            );
        }

        // Parse headers.
        let headers_start = 12; // after prelude (8) + prelude CRC (4)
        let headers_end = headers_start + header_len;
        let headers = parse_headers(&buf[headers_start..headers_end])
            .context("parsing event-stream headers")?;

        // Extract payload.
        let payload_end = total_len - 4; // before message CRC
        let payload = buf[headers_end..payload_end].to_vec();

        Ok(Some((Frame { headers, payload }, total_len)))
    }

    /// Parse the header section of an event-stream frame.
    ///
    /// Each header:
    /// - 1 byte: name length
    /// - N bytes: name (UTF-8)
    /// - 1 byte: value type
    /// - For type 7 (string): 2 bytes value length (BE) + value bytes
    /// - For type 6 (bytes): 2 bytes value length (BE) + value bytes (stored as hex)
    /// - Other types: skipped with best-effort length consumption
    fn parse_headers(mut buf: &[u8]) -> Result<std::collections::HashMap<String, String>> {
        let mut map = std::collections::HashMap::new();

        while !buf.is_empty() {
            let name_len = buf[0] as usize;
            buf = &buf[1..];
            anyhow::ensure!(
                buf.len() >= name_len,
                "truncated header name (need {name_len}, have {})",
                buf.len()
            );
            let name = std::str::from_utf8(&buf[..name_len])
                .context("header name is not UTF-8")?
                .to_owned();
            buf = &buf[name_len..];

            anyhow::ensure!(!buf.is_empty(), "truncated header: missing value type");
            let value_type = buf[0];
            buf = &buf[1..];

            /// Consume exactly `n` bytes from `buf`, failing if not enough remain.
            macro_rules! consume {
                ($n:expr) => {{
                    let n = $n;
                    anyhow::ensure!(
                        buf.len() >= n,
                        "truncated header value for {name:?} (need {n}, have {})",
                        buf.len()
                    );
                    let (head, tail) = buf.split_at(n);
                    buf = tail;
                    head
                }};
            }

            match value_type {
                // Type 7 = string
                7 => {
                    let len_bytes = consume!(2);
                    let val_len = u16::from_be_bytes([len_bytes[0], len_bytes[1]]) as usize;
                    let val_bytes = consume!(val_len);
                    let value = std::str::from_utf8(val_bytes).unwrap_or("").to_owned();
                    map.insert(name, value);
                }
                // Type 6 = bytes (skip for now)
                6 => {
                    let len_bytes = consume!(2);
                    let val_len = u16::from_be_bytes([len_bytes[0], len_bytes[1]]) as usize;
                    consume!(val_len);
                }
                // Type 0 = bool_true, 1 = bool_false (no payload)
                0 | 1 => {}
                // Type 2 = byte (1 byte)
                2 => {
                    consume!(1);
                }
                // Type 3 = short (2 bytes)
                3 => {
                    consume!(2);
                }
                // Type 4 = int (4 bytes)
                4 => {
                    consume!(4);
                }
                // Type 5 = long (8 bytes)
                5 => {
                    consume!(8);
                }
                // Type 8 = timestamp (8 bytes)
                8 => {
                    consume!(8);
                }
                // Type 9 = uuid (16 bytes)
                9 => {
                    consume!(16);
                }
                _ => {
                    anyhow::bail!(
                        "unknown event-stream header type {value_type} for header {name:?}"
                    );
                }
            }
        }

        Ok(map)
    }

    // -- Test helpers -------------------------------------------------------

    /// Build a minimal event-stream frame for testing.
    ///
    /// Only supports string-typed headers (type 7).
    #[cfg(test)]
    pub fn build_test_frame(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
        // Encode headers.
        let mut hdr_buf: Vec<u8> = Vec::new();
        for &(name, value) in headers {
            hdr_buf.push(name.len() as u8);
            hdr_buf.extend_from_slice(name.as_bytes());
            hdr_buf.push(7); // type = string
            hdr_buf.extend_from_slice(&(value.len() as u16).to_be_bytes());
            hdr_buf.extend_from_slice(value.as_bytes());
        }

        let header_len = hdr_buf.len();
        let total_len = 4 + 4 + 4 + header_len + payload.len() + 4; // prelude + headers + payload + msg_crc

        let mut frame = Vec::with_capacity(total_len);
        frame.extend_from_slice(&(total_len as u32).to_be_bytes());
        frame.extend_from_slice(&(header_len as u32).to_be_bytes());

        // Prelude CRC (over first 8 bytes).
        let prelude_crc = crc32(&frame[..8]);
        frame.extend_from_slice(&prelude_crc.to_be_bytes());

        // Headers + payload.
        frame.extend_from_slice(&hdr_buf);
        frame.extend_from_slice(payload);

        // Message CRC (over everything so far).
        let msg_crc = crc32(&frame);
        frame.extend_from_slice(&msg_crc.to_be_bytes());

        assert_eq!(frame.len(), total_len);
        frame
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn round_trip_empty_payload() {
            let frame_bytes = build_test_frame(
                &[
                    (":event-type", "messageStart"),
                    (":content-type", "application/json"),
                ],
                b"",
            );
            let (frame, consumed) = try_parse_frame(&frame_bytes).unwrap().unwrap();
            assert_eq!(consumed, frame_bytes.len());
            assert_eq!(frame.headers.get(":event-type").unwrap(), "messageStart");
            assert_eq!(
                frame.headers.get(":content-type").unwrap(),
                "application/json"
            );
            assert!(frame.payload.is_empty());
        }

        #[test]
        fn round_trip_with_json_payload() {
            let payload = br#"{"contentBlockIndex":0,"delta":{"text":"Hello"}}"#;
            let frame_bytes = build_test_frame(
                &[
                    (":event-type", "contentBlockDelta"),
                    (":content-type", "application/json"),
                    (":message-type", "event"),
                ],
                payload,
            );
            let (frame, consumed) = try_parse_frame(&frame_bytes).unwrap().unwrap();
            assert_eq!(consumed, frame_bytes.len());
            assert_eq!(
                frame.headers.get(":event-type").unwrap(),
                "contentBlockDelta"
            );
            assert_eq!(frame.payload, payload);
        }

        #[test]
        fn incomplete_buffer_returns_none() {
            let frame_bytes = build_test_frame(&[(":event-type", "messageStart")], b"{}");
            // Pass only the first 10 bytes — not enough for a full frame.
            let result = try_parse_frame(&frame_bytes[..10]).unwrap();
            assert!(result.is_none());
        }

        #[test]
        fn corrupted_prelude_crc_returns_error() {
            let mut frame_bytes = build_test_frame(&[(":event-type", "messageStart")], b"{}");
            // Corrupt the prelude CRC (bytes 8..12).
            frame_bytes[8] ^= 0xFF;
            let result = try_parse_frame(&frame_bytes);
            assert!(result.is_err());
            assert!(
                format!("{:?}", result.unwrap_err()).contains("prelude CRC"),
                "error should mention prelude CRC"
            );
        }

        #[test]
        fn corrupted_message_crc_returns_error() {
            let mut frame_bytes = build_test_frame(&[(":event-type", "messageStart")], b"{}");
            // Corrupt the last byte (part of the message CRC).
            let last = frame_bytes.len() - 1;
            frame_bytes[last] ^= 0xFF;
            let result = try_parse_frame(&frame_bytes);
            assert!(result.is_err());
            assert!(
                format!("{:?}", result.unwrap_err()).contains("message CRC"),
                "error should mention message CRC"
            );
        }

        #[test]
        fn parse_multiple_frames_from_buffer() {
            let f1 = build_test_frame(&[(":event-type", "messageStart")], b"{}");
            let f2 = build_test_frame(
                &[(":event-type", "messageStop")],
                br#"{"stopReason":"end_turn"}"#,
            );
            let mut combined = Vec::new();
            combined.extend_from_slice(&f1);
            combined.extend_from_slice(&f2);

            let (frame1, consumed1) = try_parse_frame(&combined).unwrap().unwrap();
            assert_eq!(frame1.headers.get(":event-type").unwrap(), "messageStart");

            let (frame2, consumed2) = try_parse_frame(&combined[consumed1..]).unwrap().unwrap();
            assert_eq!(frame2.headers.get(":event-type").unwrap(), "messageStop");
            assert_eq!(consumed1 + consumed2, combined.len());
        }

        #[test]
        fn crc32_known_value() {
            // "123456789" → CRC-32 = 0xCBF43926 (ISO 3309)
            assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        }
    }
}

// ---------------------------------------------------------------------------
// Bedrock HTTP error type
// ---------------------------------------------------------------------------

/// An HTTP error response from the Bedrock API.
///
/// Mirrors `CopilotHttpError` so callers can inspect status codes uniformly.
#[derive(Debug)]
pub struct BedrockHttpError {
    pub status: reqwest::StatusCode,
    pub body: String,
}

impl std::fmt::Display for BedrockHttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let body = &self.body;
        match body.char_indices().nth(200) {
            None => write!(f, "Bedrock API returned {}: {}", self.status, body),
            Some((byte_idx, _)) => write!(
                f,
                "Bedrock API returned {}: {}…",
                self.status,
                &body[..byte_idx]
            ),
        }
    }
}

impl std::error::Error for BedrockHttpError {}

// ---------------------------------------------------------------------------
// HTTP send with retry
// ---------------------------------------------------------------------------

async fn send_with_retry(
    http: &reqwest::Client,
    token: &str,
    region: &str,
    model_id: &str,
    messages: &[Message],
    tools: &[serde_json::Value],
    max_tokens: usize,
) -> Result<reqwest::Response> {
    let parts = to_bedrock_request(messages);
    let bedrock_tools = to_bedrock_tools(tools);

    let mut body = serde_json::json!({
        "messages": parts.messages,
        "inferenceConfig": { "maxTokens": max_tokens },
    });

    if !parts.system.is_empty() {
        body["system"] = serde_json::Value::Array(parts.system);
    }
    if !bedrock_tools.is_empty() {
        body["toolConfig"] = serde_json::json!({ "tools": bedrock_tools });
    }

    let url = converse_stream_endpoint(region, model_id);

    let mut attempt = 0u32;
    loop {
        let result = http
            .post(&url)
            .header("Authorization", format!("Bearer {token}"))
            .header("Content-Type", "application/json")
            .header("Accept", "application/vnd.amazon.eventstream")
            .header("User-Agent", concat!("amaebi/", env!("CARGO_PKG_VERSION")))
            .json(&body)
            .send()
            .await;

        match result {
            Ok(resp) if resp.status().is_success() => return Ok(resp),

            Ok(resp) if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS => {
                if attempt >= MAX_RETRIES {
                    let status = resp.status();
                    let body_text = resp.text().await.unwrap_or_default();
                    return Err(anyhow::Error::new(BedrockHttpError {
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
                    "Bedrock rate-limited (429); backing off"
                );
                let _ = resp.bytes().await;
                tokio::time::sleep(delay).await;
                attempt += 1;
            }

            Ok(resp) if resp.status().is_server_error() => {
                if attempt >= MAX_RETRIES {
                    let status = resp.status();
                    let body_text = resp.text().await.unwrap_or_default();
                    return Err(anyhow::Error::new(BedrockHttpError {
                        status,
                        body: body_text,
                    }));
                }
                let delay = backoff_delay(attempt);
                tracing::warn!(
                    attempt,
                    delay_ms = delay.as_millis(),
                    endpoint = %url,
                    "Bedrock server error; retrying"
                );
                let _ = resp.bytes().await;
                tokio::time::sleep(delay).await;
                attempt += 1;
            }

            Ok(resp) => {
                let status = resp.status();
                let body_text = resp.text().await.unwrap_or_default();
                return Err(anyhow::Error::new(BedrockHttpError {
                    status,
                    body: body_text,
                }));
            }

            Err(e) => {
                if attempt >= MAX_RETRIES {
                    return Err(anyhow::Error::from(e)
                        .context(format!("sending request to Bedrock ({url})")));
                }
                let delay = backoff_delay(attempt);
                tracing::warn!(
                    attempt,
                    error = %e,
                    delay_ms = delay.as_millis(),
                    "Bedrock transport error; retrying"
                );
                tokio::time::sleep(delay).await;
                attempt += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ConverseStream response parser
// ---------------------------------------------------------------------------

/// In-progress tool-use block being assembled from streaming events.
#[derive(Default)]
struct PartialToolUse {
    tool_use_id: String,
    name: String,
    input_json: String,
}

/// Parse the Bedrock ConverseStream event-stream response into a `CopilotResponse`.
async fn parse_converse_stream<W>(
    resp: reqwest::Response,
    writer: &mut W,
) -> Result<CopilotResponse>
where
    W: AsyncWriteExt + Unpin,
{
    let mut stream = resp.bytes_stream();
    let mut raw_buf: Vec<u8> = Vec::new();

    let mut text = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    // Keyed by content block index.
    let mut pending_tools: std::collections::HashMap<u64, PartialToolUse> =
        std::collections::HashMap::new();
    let mut finish_reason = FinishReason::Stop;
    let mut prompt_tokens = 0usize;

    // Cursor into raw_buf: bytes before `start` have been consumed.
    let mut start = 0usize;

    while let Some(chunk) = stream.next().await {
        let bytes = chunk.context("reading Bedrock stream chunk")?;
        raw_buf.extend_from_slice(&bytes);

        // Parse as many complete frames as possible from the buffer.
        loop {
            match eventstream::try_parse_frame(&raw_buf[start..]) {
                Ok(Some((frame, consumed))) => {
                    start += consumed;
                    handle_frame(
                        &frame,
                        &mut text,
                        &mut tool_calls,
                        &mut pending_tools,
                        &mut finish_reason,
                        &mut prompt_tokens,
                        writer,
                    )
                    .await?;
                }
                Ok(None) => break, // need more data
                Err(e) => {
                    tracing::warn!(error = %e, "skipping corrupted event-stream frame");
                    // Advance past the bad byte and retry.
                    if start < raw_buf.len() {
                        start += 1;
                    }
                }
            }
        }

        // Compact: drop consumed bytes so the buffer doesn't grow unboundedly.
        if start > 0 {
            raw_buf.drain(..start);
            start = 0;
        }
    }

    // Flush any in-flight tool calls that were not explicitly finalised.
    for (_, p) in pending_tools {
        if !p.name.is_empty() {
            tool_calls.push(ToolCall {
                id: p.tool_use_id,
                name: p.name,
                arguments: p.input_json,
            });
        }
    }

    tracing::debug!(
        text_len = text.len(),
        tool_calls = tool_calls.len(),
        "Bedrock ConverseStream complete"
    );

    Ok(CopilotResponse {
        text,
        tool_calls,
        finish_reason,
        prompt_tokens,
    })
}

/// Process a single decoded event-stream frame.
async fn handle_frame<W>(
    frame: &eventstream::Frame,
    text: &mut String,
    tool_calls: &mut Vec<ToolCall>,
    pending_tools: &mut std::collections::HashMap<u64, PartialToolUse>,
    finish_reason: &mut FinishReason,
    prompt_tokens: &mut usize,
    writer: &mut W,
) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let msg_type = frame.headers.get(":message-type").map(|s| s.as_str());
    if msg_type == Some("exception") || msg_type == Some("error") {
        let body = String::from_utf8_lossy(&frame.payload);
        anyhow::bail!("Bedrock exception: {body}");
    }

    let event_type = match frame.headers.get(":event-type") {
        Some(t) => t.as_str(),
        None => return Ok(()),
    };

    // Parse the JSON payload (most events carry one).
    let payload: serde_json::Value = if frame.payload.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&frame.payload).unwrap_or(serde_json::Value::Null)
    };

    match event_type {
        "messageStart" => {
            tracing::debug!("Bedrock: messageStart");
        }

        "contentBlockStart" => {
            let idx = payload
                .get("contentBlockIndex")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            // Check if this block is a toolUse.
            if let Some(start) = payload.get("start") {
                if let Some(tool_use) = start.get("toolUse") {
                    let id = tool_use
                        .get("toolUseId")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let name = tool_use
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    pending_tools.insert(
                        idx,
                        PartialToolUse {
                            tool_use_id: id,
                            name,
                            input_json: String::new(),
                        },
                    );
                }
            }
        }

        "contentBlockDelta" => {
            let idx = payload
                .get("contentBlockIndex")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);

            if let Some(delta) = payload.get("delta") {
                // Text delta.
                if let Some(t) = delta.get("text").and_then(|v| v.as_str()) {
                    text.push_str(t);
                    if !t.is_empty() {
                        write_frame(
                            writer,
                            &Response::Text {
                                chunk: t.to_string(),
                            },
                        )
                        .await?;
                        writer.flush().await?;
                    }
                }
                // Tool-use input delta.
                if let Some(tool_delta) = delta.get("toolUse") {
                    if let Some(input_chunk) = tool_delta.get("input").and_then(|v| v.as_str()) {
                        if let Some(p) = pending_tools.get_mut(&idx) {
                            p.input_json.push_str(input_chunk);
                        }
                    }
                }
            }
        }

        "contentBlockStop" => {
            let idx = payload
                .get("contentBlockIndex")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            // Finalise any pending tool-use for this block.
            if let Some(p) = pending_tools.remove(&idx) {
                if !p.name.is_empty() {
                    tool_calls.push(ToolCall {
                        id: p.tool_use_id,
                        name: p.name,
                        arguments: p.input_json,
                    });
                }
            }
        }

        "messageStop" => {
            let reason = payload
                .get("stopReason")
                .and_then(|v| v.as_str())
                .unwrap_or("end_turn");
            *finish_reason = match reason {
                "end_turn" => FinishReason::Stop,
                "tool_use" => FinishReason::ToolCalls,
                "max_tokens" => FinishReason::Length,
                other => FinishReason::Other(other.to_owned()),
            };
            tracing::debug!(stop_reason = reason, "Bedrock: messageStop");
        }

        "metadata" => {
            if let Some(usage) = payload.get("usage") {
                if let Some(n) = usage.get("inputTokens").and_then(|v| v.as_u64()) {
                    *prompt_tokens = n as usize;
                }
            }
        }

        _ => {
            tracing::debug!(event_type, "unhandled Bedrock event");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Send a streaming chat request to Amazon Bedrock via the ConverseStream API.
///
/// Reads `AWS_BEARER_TOKEN_BEDROCK` and `AWS_REGION` from the environment.
/// Text chunks are forwarded to `writer` as `Response::Text` frames as they
/// arrive.  Returns a [`CopilotResponse`] describing the full turn result.
#[allow(clippy::too_many_arguments)]
pub async fn stream_chat<W>(
    http: &reqwest::Client,
    model_id: &str,
    messages: &[Message],
    tools: &[serde_json::Value],
    max_tokens: usize,
    writer: &mut W,
) -> Result<CopilotResponse>
where
    W: AsyncWriteExt + Unpin,
{
    let token = read_bearer_token()?;
    let region = aws_region();

    tracing::debug!(
        messages = messages.len(),
        model = model_id,
        region = %region,
        "sending Bedrock ConverseStream request"
    );

    let resp =
        send_with_retry(http, &token, &region, model_id, messages, tools, max_tokens).await?;
    parse_converse_stream(resp, writer).await
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::copilot::{ApiToolCall, ApiToolCallFunction};

    // ---- to_bedrock_request: message conversion ---------------------------

    #[test]
    fn system_messages_extracted_to_system_array() {
        let msgs = vec![Message::system("Be helpful."), Message::user("Hi")];
        let parts = to_bedrock_request(&msgs);
        assert_eq!(parts.system.len(), 1);
        assert_eq!(parts.system[0]["text"], "Be helpful.");
        assert_eq!(parts.messages.len(), 1);
        assert_eq!(parts.messages[0]["role"], "user");
    }

    #[test]
    fn user_message_wrapped_in_text_content() {
        let msgs = vec![Message::user("Hello")];
        let parts = to_bedrock_request(&msgs);
        assert_eq!(parts.messages.len(), 1);
        assert_eq!(parts.messages[0]["role"], "user");
        assert_eq!(parts.messages[0]["content"][0]["text"], "Hello");
    }

    #[test]
    fn assistant_text_wrapped_in_text_content() {
        let msgs = vec![
            Message::user("Hi"),
            Message::assistant(Some("Hello!".into()), vec![]),
        ];
        let parts = to_bedrock_request(&msgs);
        assert_eq!(parts.messages.len(), 2);
        assert_eq!(parts.messages[1]["role"], "assistant");
        assert_eq!(parts.messages[1]["content"][0]["text"], "Hello!");
    }

    #[test]
    fn assistant_tool_calls_become_tool_use() {
        let tc = ApiToolCall {
            id: "call_001".into(),
            kind: "function".into(),
            function: ApiToolCallFunction {
                name: "shell_command".into(),
                arguments: r#"{"command":"ls"}"#.into(),
            },
        };
        let msgs = vec![Message::user("run ls"), Message::assistant(None, vec![tc])];
        let parts = to_bedrock_request(&msgs);
        let asst = &parts.messages[1];
        assert_eq!(asst["role"], "assistant");
        let tool_use = &asst["content"][0]["toolUse"];
        assert_eq!(tool_use["toolUseId"], "call_001");
        assert_eq!(tool_use["name"], "shell_command");
        assert_eq!(tool_use["input"]["command"], "ls");
    }

    #[test]
    fn assistant_text_and_tool_calls_both_emitted() {
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
        let parts = to_bedrock_request(&msgs);
        let asst = &parts.messages[1];
        assert_eq!(asst["content"].as_array().unwrap().len(), 2);
        assert_eq!(asst["content"][0]["text"], "I will read the file.");
        assert!(asst["content"][1].get("toolUse").is_some());
    }

    #[test]
    fn tool_result_becomes_tool_result_in_user_role() {
        let tc = ApiToolCall {
            id: "call_001".into(),
            kind: "function".into(),
            function: ApiToolCallFunction {
                name: "shell_command".into(),
                arguments: r#"{"command":"ls"}"#.into(),
            },
        };
        let msgs = vec![
            Message::user("run ls"),
            Message::assistant(None, vec![tc]),
            Message::tool_result("call_001", "file.txt\n"),
        ];
        let parts = to_bedrock_request(&msgs);
        // tool_result should be in a user-role message.
        let tool_msg = &parts.messages[2];
        assert_eq!(tool_msg["role"], "user");
        let tr = &tool_msg["content"][0]["toolResult"];
        assert_eq!(tr["toolUseId"], "call_001");
        assert_eq!(tr["content"][0]["text"], "file.txt\n");
    }

    #[test]
    fn consecutive_same_role_messages_merged() {
        // Two consecutive user messages should be merged into one.
        let msgs = vec![Message::user("first"), Message::user("second")];
        let parts = to_bedrock_request(&msgs);
        assert_eq!(
            parts.messages.len(),
            1,
            "should merge into one user message"
        );
        let content = parts.messages[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["text"], "first");
        assert_eq!(content[1]["text"], "second");
    }

    #[test]
    fn empty_messages_produces_empty_output() {
        let parts = to_bedrock_request(&[]);
        assert!(parts.system.is_empty());
        assert!(parts.messages.is_empty());
    }

    // ---- to_bedrock_tools: tool schema conversion -------------------------

    #[test]
    fn tool_schema_converted_to_tool_spec() {
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
        let out = to_bedrock_tools(&tools);
        assert_eq!(out.len(), 1);
        let spec = &out[0]["toolSpec"];
        assert_eq!(spec["name"], "shell_command");
        assert_eq!(spec["description"], "Run a shell command");
        assert!(spec["inputSchema"]["json"].is_object());
        // Must NOT have nested "function" key.
        assert!(out[0].get("function").is_none());
    }

    #[test]
    fn empty_tools_returns_empty() {
        assert!(to_bedrock_tools(&[]).is_empty());
    }

    // ---- converse_stream_endpoint -----------------------------------------

    #[test]
    #[serial_test::serial]
    fn bedrock_endpoint_uses_region() {
        std::env::remove_var("AMAEBI_BEDROCK_URL");
        let url = converse_stream_endpoint("us-east-2", "us.anthropic.claude-sonnet-4-6");
        assert_eq!(
            url,
            "https://bedrock-runtime.us-east-2.amazonaws.com/model/us.anthropic.claude-sonnet-4-6/converse-stream"
        );
    }

    #[test]
    #[serial_test::serial]
    fn bedrock_endpoint_test_override() {
        std::env::set_var("AMAEBI_BEDROCK_URL", "http://mock:9999/converse-stream");
        let url = converse_stream_endpoint("us-east-1", "anything");
        std::env::remove_var("AMAEBI_BEDROCK_URL");
        assert_eq!(url, "http://mock:9999/converse-stream");
    }

    // ---- Event stream integration tests (using build_test_frame) ----------

    #[tokio::test]
    async fn parse_text_delta_event() {
        let payload = br#"{"contentBlockIndex":0,"delta":{"text":"Hello"}}"#;
        let frame = eventstream::build_test_frame(
            &[
                (":event-type", "contentBlockDelta"),
                (":content-type", "application/json"),
                (":message-type", "event"),
            ],
            payload,
        );

        let mut text = String::new();
        let mut tool_calls = Vec::new();
        let mut pending = std::collections::HashMap::new();
        let mut finish = FinishReason::Stop;
        let mut tokens = 0usize;
        let mut sink = tokio::io::sink();

        let parsed = eventstream::try_parse_frame(&frame).unwrap().unwrap().0;
        handle_frame(
            &parsed,
            &mut text,
            &mut tool_calls,
            &mut pending,
            &mut finish,
            &mut tokens,
            &mut sink,
        )
        .await
        .unwrap();

        assert_eq!(text, "Hello");
    }

    #[tokio::test]
    async fn parse_tool_use_flow() {
        // 1. contentBlockStart with toolUse
        let start_payload =
            br#"{"contentBlockIndex":1,"start":{"toolUse":{"toolUseId":"tc_1","name":"shell_command"}}}"#;
        let start_frame = eventstream::build_test_frame(
            &[
                (":event-type", "contentBlockStart"),
                (":message-type", "event"),
            ],
            start_payload,
        );

        // 2. contentBlockDelta with toolUse input
        let delta_payload =
            br#"{"contentBlockIndex":1,"delta":{"toolUse":{"input":"{\"command\":\"ls\"}"}}}"#;
        let delta_frame = eventstream::build_test_frame(
            &[
                (":event-type", "contentBlockDelta"),
                (":message-type", "event"),
            ],
            delta_payload,
        );

        // 3. contentBlockStop
        let stop_payload = br#"{"contentBlockIndex":1}"#;
        let stop_frame = eventstream::build_test_frame(
            &[
                (":event-type", "contentBlockStop"),
                (":message-type", "event"),
            ],
            stop_payload,
        );

        let mut text = String::new();
        let mut tool_calls = Vec::new();
        let mut pending = std::collections::HashMap::new();
        let mut finish = FinishReason::Stop;
        let mut tokens = 0usize;
        let mut sink = tokio::io::sink();

        for frame_bytes in [&start_frame, &delta_frame, &stop_frame] {
            let (parsed, _) = eventstream::try_parse_frame(frame_bytes).unwrap().unwrap();
            handle_frame(
                &parsed,
                &mut text,
                &mut tool_calls,
                &mut pending,
                &mut finish,
                &mut tokens,
                &mut sink,
            )
            .await
            .unwrap();
        }

        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "tc_1");
        assert_eq!(tool_calls[0].name, "shell_command");
        assert_eq!(tool_calls[0].arguments, r#"{"command":"ls"}"#);
    }

    #[tokio::test]
    async fn parse_message_stop_end_turn() {
        let payload = br#"{"stopReason":"end_turn"}"#;
        let frame_bytes = eventstream::build_test_frame(
            &[(":event-type", "messageStop"), (":message-type", "event")],
            payload,
        );

        let mut text = String::new();
        let mut tool_calls = Vec::new();
        let mut pending = std::collections::HashMap::new();
        let mut finish = FinishReason::Other("init".into());
        let mut tokens = 0usize;
        let mut sink = tokio::io::sink();

        let (parsed, _) = eventstream::try_parse_frame(&frame_bytes).unwrap().unwrap();
        handle_frame(
            &parsed,
            &mut text,
            &mut tool_calls,
            &mut pending,
            &mut finish,
            &mut tokens,
            &mut sink,
        )
        .await
        .unwrap();

        assert!(matches!(finish, FinishReason::Stop));
    }

    #[tokio::test]
    async fn parse_message_stop_tool_use() {
        let payload = br#"{"stopReason":"tool_use"}"#;
        let frame_bytes = eventstream::build_test_frame(
            &[(":event-type", "messageStop"), (":message-type", "event")],
            payload,
        );

        let mut finish = FinishReason::Stop;
        let mut sink = tokio::io::sink();
        let (parsed, _) = eventstream::try_parse_frame(&frame_bytes).unwrap().unwrap();
        handle_frame(
            &parsed,
            &mut String::new(),
            &mut Vec::new(),
            &mut std::collections::HashMap::new(),
            &mut finish,
            &mut 0,
            &mut sink,
        )
        .await
        .unwrap();
        assert!(matches!(finish, FinishReason::ToolCalls));
    }

    #[tokio::test]
    async fn parse_message_stop_max_tokens() {
        let payload = br#"{"stopReason":"max_tokens"}"#;
        let frame_bytes = eventstream::build_test_frame(
            &[(":event-type", "messageStop"), (":message-type", "event")],
            payload,
        );

        let mut finish = FinishReason::Stop;
        let mut sink = tokio::io::sink();
        let (parsed, _) = eventstream::try_parse_frame(&frame_bytes).unwrap().unwrap();
        handle_frame(
            &parsed,
            &mut String::new(),
            &mut Vec::new(),
            &mut std::collections::HashMap::new(),
            &mut finish,
            &mut 0,
            &mut sink,
        )
        .await
        .unwrap();
        assert!(matches!(finish, FinishReason::Length));
    }

    #[tokio::test]
    async fn parse_metadata_usage() {
        let payload = br#"{"usage":{"inputTokens":42,"outputTokens":10}}"#;
        let frame_bytes = eventstream::build_test_frame(
            &[(":event-type", "metadata"), (":message-type", "event")],
            payload,
        );

        let mut tokens = 0usize;
        let mut sink = tokio::io::sink();
        let (parsed, _) = eventstream::try_parse_frame(&frame_bytes).unwrap().unwrap();
        handle_frame(
            &parsed,
            &mut String::new(),
            &mut Vec::new(),
            &mut std::collections::HashMap::new(),
            &mut FinishReason::Stop,
            &mut tokens,
            &mut sink,
        )
        .await
        .unwrap();
        assert_eq!(tokens, 42);
    }

    #[tokio::test]
    async fn parse_exception_frame_returns_error() {
        let payload = br#"{"message":"model not found"}"#;
        let frame_bytes = eventstream::build_test_frame(
            &[(":event-type", "error"), (":message-type", "exception")],
            payload,
        );

        let mut sink = tokio::io::sink();
        let (parsed, _) = eventstream::try_parse_frame(&frame_bytes).unwrap().unwrap();
        let result = handle_frame(
            &parsed,
            &mut String::new(),
            &mut Vec::new(),
            &mut std::collections::HashMap::new(),
            &mut FinishReason::Stop,
            &mut 0,
            &mut sink,
        )
        .await;
        assert!(result.is_err());
        assert!(format!("{:?}", result.unwrap_err()).contains("exception"));
    }
}
