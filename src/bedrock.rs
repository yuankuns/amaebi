//! AWS Bedrock Converse Stream API backend.
//!
//! Routes requests to Bedrock when the model ID looks like a Bedrock model ID
//! (e.g. `anthropic.claude-3-5-sonnet-20241022-v2:0`, `amazon.nova-pro-v1:0`,
//! `us.anthropic.claude-3-haiku-20240307-v1:0`).
//!
//! Authentication uses AWS Signature Version 4 (SigV4) — no AWS SDK required.
//!
//! ## Configuration
//!
//! | Env var                          | Fallback                  | Description              |
//! |----------------------------------|---------------------------|--------------------------|
//! | `AMAEBI_BEDROCK_REGION`          | `AWS_DEFAULT_REGION`, `AWS_REGION` | AWS region (required) |
//! | `AMAEBI_BEDROCK_ACCESS_KEY_ID`   | `AWS_ACCESS_KEY_ID`       | IAM access key (required)|
//! | `AMAEBI_BEDROCK_SECRET_ACCESS_KEY` | `AWS_SECRET_ACCESS_KEY` | IAM secret key (required)|
//! | `AMAEBI_BEDROCK_SESSION_TOKEN`   | `AWS_SESSION_TOKEN`       | STS session token (opt.) |

use anyhow::{Context, Result};
use futures_util::StreamExt;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::copilot::{CopilotResponse, FinishReason, Message, ToolCall};
use crate::ipc::{write_frame, Response};

type HmacSha256 = Hmac<Sha256>;

// ---------------------------------------------------------------------------
// Model detection
// ---------------------------------------------------------------------------

/// Known provider prefixes for Bedrock model IDs.
///
/// Bedrock model IDs use the format `{provider}.{name}` or the
/// cross-region inference format `{geo}.{provider}.{name}`.
const BEDROCK_PREFIXES: &[&str] = &[
    "amazon.",
    "anthropic.",
    "cohere.",
    "meta.",
    "mistral.",
    "ai21.",
    "stability.",
    "writer.",
    // Cross-region inference profile prefixes
    "us.",
    "eu.",
    "ap.",
];

/// Returns `true` if `model` looks like a Bedrock model ID.
///
/// Bedrock model IDs typically look like
/// `anthropic.claude-3-5-sonnet-20241022-v2:0` or
/// `us.amazon.nova-pro-v1:0`.
pub fn is_bedrock_model(model: &str) -> bool {
    BEDROCK_PREFIXES.iter().any(|p| model.starts_with(p))
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct BedrockConfig {
    region: String,
    access_key_id: String,
    secret_access_key: String,
    session_token: Option<String>,
}

impl BedrockConfig {
    fn from_env() -> Result<Self> {
        let region = std::env::var("AMAEBI_BEDROCK_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .or_else(|_| std::env::var("AWS_REGION"))
            .context(
                "AMAEBI_BEDROCK_REGION (or AWS_DEFAULT_REGION / AWS_REGION) is required \
                 for Bedrock models",
            )?;

        let access_key_id = std::env::var("AMAEBI_BEDROCK_ACCESS_KEY_ID")
            .or_else(|_| std::env::var("AWS_ACCESS_KEY_ID"))
            .context(
                "AMAEBI_BEDROCK_ACCESS_KEY_ID (or AWS_ACCESS_KEY_ID) is required \
                 for Bedrock models",
            )?;

        let secret_access_key = std::env::var("AMAEBI_BEDROCK_SECRET_ACCESS_KEY")
            .or_else(|_| std::env::var("AWS_SECRET_ACCESS_KEY"))
            .context(
                "AMAEBI_BEDROCK_SECRET_ACCESS_KEY (or AWS_SECRET_ACCESS_KEY) is required \
                 for Bedrock models",
            )?;

        let session_token = std::env::var("AMAEBI_BEDROCK_SESSION_TOKEN")
            .or_else(|_| std::env::var("AWS_SESSION_TOKEN"))
            .ok();

        Ok(Self {
            region,
            access_key_id,
            secret_access_key,
            session_token,
        })
    }
}

// ---------------------------------------------------------------------------
// SigV4 signing helpers
// ---------------------------------------------------------------------------

fn sha256_hex(data: &[u8]) -> String {
    let hash = Sha256::digest(data);
    hex_encode(&hash)
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key size");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

/// Derive the SigV4 signing key from secret + date + region + service.
fn signing_key(secret: &str, date: &str, region: &str, service: &str) -> Vec<u8> {
    let k_secret = format!("AWS4{secret}");
    let k_date = hmac_sha256(k_secret.as_bytes(), date.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    hmac_sha256(&k_service, b"aws4_request")
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Build AWS SigV4 `Authorization` and related headers for a Bedrock POST.
///
/// Returns a `Vec<(name, value)>` of headers to add to the request.
fn sigv4_headers(
    config: &BedrockConfig,
    path: &str,
    body: &[u8],
    datetime: &str, // "YYYYMMDDTHHmmssZ"
) -> Vec<(String, String)> {
    let date = &datetime[..8]; // "YYYYMMDD"
    let service = "bedrock";
    let host = format!("bedrock-runtime.{}.amazonaws.com", config.region);
    let payload_hash = sha256_hex(body);

    // Canonical headers must be in sorted (lowercase name) order.
    // We include host, x-amz-content-sha256, x-amz-date, and optionally
    // x-amz-security-token.
    let mut canonical_headers =
        format!("host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{datetime}\n");
    let mut signed_headers = "host;x-amz-content-sha256;x-amz-date".to_string();

    if let Some(ref token) = config.session_token {
        // x-amz-security-token sorts after x-amz-date alphabetically.
        canonical_headers.push_str(&format!("x-amz-security-token:{token}\n"));
        signed_headers.push_str(";x-amz-security-token");
    }

    let canonical_request =
        format!("POST\n{path}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}");

    let credential_scope = format!("{date}/{}/{service}/aws4_request", config.region);

    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{datetime}\n{credential_scope}\n{}",
        sha256_hex(canonical_request.as_bytes())
    );

    let sig_key = signing_key(&config.secret_access_key, date, &config.region, service);
    let signature = hex_encode(&hmac_sha256(&sig_key, string_to_sign.as_bytes()));

    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{credential_scope}, \
         SignedHeaders={signed_headers}, Signature={signature}",
        config.access_key_id
    );

    let mut headers = vec![
        ("Authorization".to_string(), authorization),
        ("x-amz-date".to_string(), datetime.to_string()),
        ("x-amz-content-sha256".to_string(), payload_hash),
    ];

    if let Some(ref token) = config.session_token {
        headers.push(("x-amz-security-token".to_string(), token.clone()));
    }

    headers
}

// ---------------------------------------------------------------------------
// OpenAI-format → Bedrock Converse format conversion
// ---------------------------------------------------------------------------

/// Convert an OpenAI-format messages list into the Bedrock Converse format.
///
/// Returns `(system_text, converse_messages)` where `system_text` is the
/// content of the first system message (if any) and `converse_messages` is
/// the converted array.
pub(crate) fn to_converse_messages(
    messages: &[Message],
) -> (Option<String>, Vec<serde_json::Value>) {
    let mut system: Option<String> = None;
    let mut out: Vec<serde_json::Value> = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                if let Some(ref text) = msg.content {
                    system = Some(text.clone());
                }
            }
            "user" => {
                let content_text = msg.content.as_deref().unwrap_or("").to_string();

                // OpenAI tool result messages use role="tool"; however the
                // daemon may also emit them as role="user" with a tool_call_id.
                if let Some(ref call_id) = msg.tool_call_id {
                    // Merge consecutive tool results into a single user turn.
                    if merge_tool_result_into_last(&mut out, call_id, &content_text) {
                        continue;
                    }
                    out.push(serde_json::json!({
                        "role": "user",
                        "content": [{
                            "toolResult": {
                                "toolUseId": call_id,
                                "content": [{"text": content_text}],
                                "status": "success"
                            }
                        }]
                    }));
                } else {
                    out.push(serde_json::json!({
                        "role": "user",
                        "content": [{"text": content_text}]
                    }));
                }
            }
            "tool" => {
                let content_text = msg.content.as_deref().unwrap_or("").to_string();
                if let Some(ref call_id) = msg.tool_call_id {
                    if merge_tool_result_into_last(&mut out, call_id, &content_text) {
                        continue;
                    }
                    out.push(serde_json::json!({
                        "role": "user",
                        "content": [{
                            "toolResult": {
                                "toolUseId": call_id,
                                "content": [{"text": content_text}],
                                "status": "success"
                            }
                        }]
                    }));
                }
            }
            "assistant" => {
                let mut content: Vec<serde_json::Value> = Vec::new();

                if let Some(ref text) = msg.content {
                    if !text.is_empty() {
                        content.push(serde_json::json!({"text": text}));
                    }
                }

                for tc in &msg.tool_calls {
                    let input: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                    content.push(serde_json::json!({
                        "toolUse": {
                            "toolUseId": tc.id,
                            "name": tc.function.name,
                            "input": input
                        }
                    }));
                }

                if !content.is_empty() {
                    out.push(serde_json::json!({
                        "role": "assistant",
                        "content": content
                    }));
                }
            }
            _ => {}
        }
    }

    (system, out)
}

/// If the last message in `out` is a user turn consisting only of toolResult
/// blocks, append `call_id` / `content` as another toolResult and return true.
/// Otherwise returns false.
fn merge_tool_result_into_last(
    out: &mut [serde_json::Value],
    call_id: &str,
    content: &str,
) -> bool {
    if let Some(last) = out.last_mut() {
        if last["role"] == "user" {
            let all_tool_results = last["content"]
                .as_array()
                .is_some_and(|arr| arr.iter().all(|e| e.get("toolResult").is_some()));
            if all_tool_results {
                if let Some(arr) = last["content"].as_array_mut() {
                    arr.push(serde_json::json!({
                        "toolResult": {
                            "toolUseId": call_id,
                            "content": [{"text": content}],
                            "status": "success"
                        }
                    }));
                    return true;
                }
            }
        }
    }
    false
}

/// Convert OpenAI tool schemas to Bedrock Converse `toolSpec` format.
pub(crate) fn to_converse_tools(tools: &[serde_json::Value]) -> Vec<serde_json::Value> {
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
                .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));
            Some(serde_json::json!({
                "toolSpec": {
                    "name": name,
                    "description": description,
                    "inputSchema": {"json": parameters}
                }
            }))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// AWS EventStream binary protocol parser
// ---------------------------------------------------------------------------
//
// Frame layout (all multi-byte integers are big-endian):
//   [4]  total_length    – total frame size in bytes including this field
//   [4]  headers_length  – byte length of the headers section
//   [4]  prelude_crc32   – CRC32 of the first 8 bytes (not validated here)
//   [N]  headers         – binary-encoded key/value headers
//   [M]  payload         – raw JSON payload
//   [4]  message_crc32   – CRC32 of everything except this field (not validated)

#[derive(Debug)]
struct EventFrame {
    event_type: String,
    payload: Vec<u8>,
}

fn parse_event_frame(data: &[u8]) -> Option<(EventFrame, usize)> {
    if data.len() < 12 {
        return None;
    }
    let total_len = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
    if data.len() < total_len || total_len < 12 {
        return None;
    }
    let headers_len = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;

    let headers_start = 12;
    let headers_end = headers_start + headers_len;
    let payload_end = total_len.saturating_sub(4);

    if headers_end > payload_end {
        return None;
    }

    let event_type = parse_event_type_header(&data[headers_start..headers_end])?;
    let payload = data[headers_end..payload_end].to_vec();

    Some((
        EventFrame {
            event_type,
            payload,
        },
        total_len,
    ))
}

/// Walk the binary headers block looking for `:event-type` (value type 7 = string).
fn parse_event_type_header(headers: &[u8]) -> Option<String> {
    let mut pos = 0;
    while pos < headers.len() {
        // 1-byte name length
        let name_len = *headers.get(pos)? as usize;
        pos += 1;

        // name bytes
        if pos + name_len > headers.len() {
            break;
        }
        let name = std::str::from_utf8(&headers[pos..pos + name_len]).ok()?;
        pos += name_len;

        // 1-byte value type
        let value_type = *headers.get(pos)?;
        pos += 1;

        // 2-byte value length (big-endian)
        if pos + 2 > headers.len() {
            break;
        }
        let value_len = u16::from_be_bytes([headers[pos], headers[pos + 1]]) as usize;
        pos += 2;

        // value bytes
        if pos + value_len > headers.len() {
            break;
        }
        let value = std::str::from_utf8(&headers[pos..pos + value_len]).ok()?;
        pos += value_len;

        // value_type 7 = string
        if name == ":event-type" && value_type == 7 {
            return Some(value.to_string());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// In-progress tool-use accumulator
// ---------------------------------------------------------------------------

#[derive(Default)]
struct PartialToolUse {
    tool_use_id: String,
    name: String,
    input_json: String,
}

// ---------------------------------------------------------------------------
// Public streaming entry point
// ---------------------------------------------------------------------------

/// Send a streaming chat request to the Bedrock Converse Stream API.
///
/// Signature matches `copilot::stream_chat` so the daemon can dispatch to
/// either backend without duplicating call-site logic.
pub async fn stream_chat<W>(
    http: &reqwest::Client,
    model: &str,
    messages: &[Message],
    tools: &[serde_json::Value],
    max_tokens: usize,
    writer: &mut W,
) -> Result<CopilotResponse>
where
    W: AsyncWriteExt + Unpin,
{
    let config = BedrockConfig::from_env()?;

    let (system, converse_messages) = to_converse_messages(messages);
    let converse_tools = to_converse_tools(tools);

    let mut body = serde_json::json!({
        "messages": converse_messages,
        "inferenceConfig": {
            "maxTokens": max_tokens
        }
    });

    if let Some(ref sys) = system {
        body["system"] = serde_json::json!([{"text": sys}]);
    }

    if !converse_tools.is_empty() {
        body["toolConfig"] = serde_json::json!({ "tools": converse_tools });
    }

    let body_bytes = serde_json::to_vec(&body).context("serializing Bedrock request")?;

    let path = format!("/model/{model}/converse-stream");
    let url = format!(
        "https://bedrock-runtime.{}.amazonaws.com{path}",
        config.region
    );

    let datetime = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let auth_headers = sigv4_headers(&config, &path, &body_bytes, &datetime);

    let mut req = http
        .post(&url)
        .header("Content-Type", "application/json")
        .header("Accept", "application/vnd.amazon.eventstream")
        .header("User-Agent", concat!("amaebi/", env!("CARGO_PKG_VERSION")));

    for (k, v) in &auth_headers {
        req = req.header(k.as_str(), v.as_str());
    }

    let resp = req
        .body(body_bytes)
        .send()
        .await
        .context("sending request to Bedrock")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        anyhow::bail!("Bedrock API returned {status}: {body_text}");
    }

    parse_eventstream_response(resp, writer).await
}

// ---------------------------------------------------------------------------
// EventStream response parser
// ---------------------------------------------------------------------------

async fn parse_eventstream_response<W>(
    resp: reqwest::Response,
    writer: &mut W,
) -> Result<CopilotResponse>
where
    W: AsyncWriteExt + Unpin,
{
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();

    let mut text = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut partial_tools: std::collections::HashMap<usize, PartialToolUse> =
        std::collections::HashMap::new();
    let mut finish_reason = FinishReason::Stop;
    let mut prompt_tokens: usize = 0;

    while let Some(chunk) = stream.next().await {
        let bytes = chunk.context("reading Bedrock response chunk")?;
        buf.extend_from_slice(&bytes);

        // Drain all complete frames from `buf`.
        while let Some((frame, consumed)) = parse_event_frame(&buf) {
            buf.drain(..consumed);
            process_converse_event(
                &frame,
                &mut text,
                &mut partial_tools,
                &mut tool_calls,
                &mut finish_reason,
                &mut prompt_tokens,
                writer,
            )
            .await?;
        }
    }

    // Flush any tool-use blocks that arrived without a contentBlockStop
    // (defensive — should not happen with a well-behaved server).
    let mut indices: Vec<usize> = partial_tools.keys().copied().collect();
    indices.sort_unstable();
    for i in indices {
        if let Some(p) = partial_tools.remove(&i) {
            if !p.name.is_empty() {
                tool_calls.push(ToolCall {
                    id: p.tool_use_id,
                    name: p.name,
                    arguments: p.input_json,
                });
            }
        }
    }

    Ok(CopilotResponse {
        text,
        tool_calls,
        finish_reason,
        prompt_tokens,
    })
}

async fn process_converse_event<W>(
    frame: &EventFrame,
    text: &mut String,
    partial_tools: &mut std::collections::HashMap<usize, PartialToolUse>,
    tool_calls: &mut Vec<ToolCall>,
    finish_reason: &mut FinishReason,
    prompt_tokens: &mut usize,
    writer: &mut W,
) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    // Ignore empty payloads (some frames carry no JSON body).
    if frame.payload.is_empty() {
        return Ok(());
    }

    let payload: serde_json::Value = match serde_json::from_slice(&frame.payload) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                event_type = %frame.event_type,
                error = %e,
                "skipping unparseable Bedrock event"
            );
            return Ok(());
        }
    };

    match frame.event_type.as_str() {
        "contentBlockStart" => {
            let index = payload
                .get("contentBlockIndex")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            if let Some(tool_use) = payload.get("start").and_then(|s| s.get("toolUse")) {
                let tool_use_id = tool_use
                    .get("toolUseId")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = tool_use
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                partial_tools.insert(
                    index,
                    PartialToolUse {
                        tool_use_id,
                        name,
                        input_json: String::new(),
                    },
                );
            }
        }

        "contentBlockDelta" => {
            let index = payload
                .get("contentBlockIndex")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;

            let delta = match payload.get("delta") {
                Some(d) => d,
                None => return Ok(()),
            };

            // Text delta
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

            // Tool-use input fragment
            if let Some(input_str) = delta
                .get("toolUse")
                .and_then(|tu| tu.get("input"))
                .and_then(|v| v.as_str())
            {
                if let Some(partial) = partial_tools.get_mut(&index) {
                    partial.input_json.push_str(input_str);
                }
            }
        }

        "contentBlockStop" => {
            // Move a completed toolUse block into `tool_calls`.
            let index = payload
                .get("contentBlockIndex")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            if let Some(p) = partial_tools.remove(&index) {
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
            let stop_reason = payload
                .get("stopReason")
                .and_then(|v| v.as_str())
                .unwrap_or("end_turn");
            *finish_reason = match stop_reason {
                "end_turn" | "stop_sequence" => FinishReason::Stop,
                "tool_use" => FinishReason::ToolCalls,
                "max_tokens" => FinishReason::Length,
                other => FinishReason::Other(other.to_string()),
            };
        }

        "metadata" => {
            if let Some(usage) = payload.get("usage") {
                if let Some(n) = usage.get("inputTokens").and_then(|v| v.as_u64()) {
                    *prompt_tokens = n as usize;
                }
            }
        }

        // Server-side error events — surface them as Rust errors.
        "internalServerException"
        | "modelStreamErrorException"
        | "validationException"
        | "throttlingException"
        | "serviceUnavailableException"
        | "modelTimeoutException" => {
            let msg = payload
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("(no message)");
            anyhow::bail!("Bedrock error ({}): {msg}", frame.event_type);
        }

        _ => {
            tracing::debug!(event_type = %frame.event_type, "unhandled Bedrock event");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- is_bedrock_model --------------------------------------------------

    #[test]
    fn bedrock_model_detection_positive() {
        let bedrock_ids = [
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "amazon.nova-pro-v1:0",
            "amazon.titan-text-lite-v1",
            "cohere.command-r-plus-v1:0",
            "meta.llama3-8b-instruct-v1:0",
            "mistral.mistral-7b-instruct-v0:2",
            "ai21.jamba-1-5-large-v1:0",
            // Cross-region inference profile prefixes
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "eu.anthropic.claude-3-haiku-20240307-v1:0",
            "ap.amazon.nova-pro-v1:0",
        ];
        for id in &bedrock_ids {
            assert!(is_bedrock_model(id), "expected Bedrock: {id}");
        }
    }

    #[test]
    fn bedrock_model_detection_negative() {
        let non_bedrock = [
            "claude-sonnet-4.6",    // Copilot claude
            "claude-opus-4.6",      // Copilot claude
            "gpt-4o",               // AMAEBI_URL / Copilot
            "gpt-4.1",              // AMAEBI_URL / Copilot
            "gemini-flash",         // AMAEBI_URL
            "gemini-2.0-flash-001", // AMAEBI_URL
            "o3-mini",              // Copilot
        ];
        for id in &non_bedrock {
            assert!(!is_bedrock_model(id), "should not be Bedrock: {id}");
        }
    }

    // ---- SHA-256 / HMAC ----------------------------------------------------

    #[test]
    fn sha256_hex_empty_input() {
        // Well-known: SHA256("") = e3b0c44298fc1c149afbf4c8996fb924...
        let h = sha256_hex(b"");
        assert_eq!(h.len(), 64, "hex digest must be 64 chars");
        assert!(
            h.starts_with("e3b0c44298fc1c14"),
            "unexpected hash prefix: {h}"
        );
    }

    #[test]
    fn sha256_hex_known_value() {
        // SHA256("abc") = ba7816bf8f01cfea414140de5dae2ec7...
        let h = sha256_hex(b"abc");
        assert!(
            h.starts_with("ba7816bf8f01cfea"),
            "unexpected hash prefix: {h}"
        );
    }

    #[test]
    fn hmac_sha256_produces_32_bytes() {
        let mac = hmac_sha256(b"secret", b"message");
        assert_eq!(mac.len(), 32, "HMAC-SHA256 output must be 32 bytes");
    }

    #[test]
    fn hmac_sha256_is_deterministic() {
        let a = hmac_sha256(b"key", b"data");
        let b = hmac_sha256(b"key", b"data");
        assert_eq!(a, b);
    }

    #[test]
    fn hmac_sha256_changes_with_key() {
        let a = hmac_sha256(b"key1", b"data");
        let b = hmac_sha256(b"key2", b"data");
        assert_ne!(a, b);
    }

    // ---- sigv4_headers structure -------------------------------------------

    #[test]
    fn sigv4_headers_always_includes_required_fields() {
        let config = BedrockConfig {
            region: "us-east-1".to_string(),
            access_key_id: "AKIDEXAMPLE".to_string(),
            secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".to_string(),
            session_token: None,
        };
        let headers = sigv4_headers(
            &config,
            "/model/anthropic.claude/converse-stream",
            b"{}",
            "20240101T120000Z",
        );

        let header_names: Vec<&str> = headers.iter().map(|(k, _)| k.as_str()).collect();
        assert!(
            header_names.contains(&"Authorization"),
            "missing Authorization"
        );
        assert!(header_names.contains(&"x-amz-date"), "missing x-amz-date");
        assert!(
            header_names.contains(&"x-amz-content-sha256"),
            "missing x-amz-content-sha256"
        );
        // No session token — x-amz-security-token must NOT appear
        assert!(
            !header_names.contains(&"x-amz-security-token"),
            "unexpected x-amz-security-token without session token"
        );
    }

    #[test]
    fn sigv4_headers_includes_session_token_when_set() {
        let config = BedrockConfig {
            region: "eu-west-1".to_string(),
            access_key_id: "AKIDEXAMPLE".to_string(),
            secret_access_key: "secret".to_string(),
            session_token: Some("my-session-token".to_string()),
        };
        let headers = sigv4_headers(
            &config,
            "/model/amazon.titan/converse-stream",
            b"{}",
            "20240101T120000Z",
        );

        let has_token = headers
            .iter()
            .any(|(k, v)| k == "x-amz-security-token" && v == "my-session-token");
        assert!(
            has_token,
            "x-amz-security-token not found with session token set"
        );
    }

    #[test]
    fn sigv4_authorization_header_format() {
        let config = BedrockConfig {
            region: "us-west-2".to_string(),
            access_key_id: "AKIDTEST".to_string(),
            secret_access_key: "supersecret".to_string(),
            session_token: None,
        };
        let headers = sigv4_headers(
            &config,
            "/model/test/converse-stream",
            b"hello",
            "20240315T093000Z",
        );

        let auth = headers
            .iter()
            .find(|(k, _)| k == "Authorization")
            .map(|(_, v)| v.as_str())
            .unwrap_or("");

        assert!(
            auth.starts_with("AWS4-HMAC-SHA256 Credential=AKIDTEST/20240315/"),
            "Authorization header format wrong: {auth}"
        );
        assert!(auth.contains("SignedHeaders="), "missing SignedHeaders");
        assert!(auth.contains("Signature="), "missing Signature");
    }

    // ---- EventStream frame parser ------------------------------------------

    fn build_frame(event_type: &str, payload: &[u8]) -> Vec<u8> {
        // Build header: `:event-type` = event_type
        let name = b":event-type";
        let val = event_type.as_bytes();
        let mut headers: Vec<u8> = Vec::new();
        headers.push(name.len() as u8);
        headers.extend_from_slice(name);
        headers.push(7u8); // string value type
        headers.push((val.len() >> 8) as u8);
        headers.push(val.len() as u8);
        headers.extend_from_slice(val);

        let headers_len = headers.len() as u32;
        let total_len = 12u32 + headers_len + payload.len() as u32 + 4;

        let mut frame: Vec<u8> = Vec::new();
        frame.extend_from_slice(&total_len.to_be_bytes());
        frame.extend_from_slice(&headers_len.to_be_bytes());
        frame.extend_from_slice(&0u32.to_be_bytes()); // prelude CRC (not validated)
        frame.extend_from_slice(&headers);
        frame.extend_from_slice(payload);
        frame.extend_from_slice(&0u32.to_be_bytes()); // message CRC (not validated)
        frame
    }

    #[test]
    fn parse_event_frame_empty_returns_none() {
        assert!(parse_event_frame(&[]).is_none());
    }

    #[test]
    fn parse_event_frame_truncated_returns_none() {
        // Only 4 bytes — not enough for a header.
        let frame = build_frame("messageStop", b"{}");
        assert!(parse_event_frame(&frame[..4]).is_none());
    }

    #[test]
    fn parse_event_frame_message_stop() {
        let payload = b"{\"stopReason\":\"end_turn\"}";
        let frame = build_frame("messageStop", payload);

        let result = parse_event_frame(&frame);
        assert!(result.is_some(), "frame should parse");
        let (ef, consumed) = result.unwrap();
        assert_eq!(ef.event_type, "messageStop");
        assert_eq!(consumed, frame.len());
        let val: serde_json::Value = serde_json::from_slice(&ef.payload).unwrap();
        assert_eq!(val["stopReason"], "end_turn");
    }

    #[test]
    fn parse_event_frame_content_block_delta() {
        let payload = br#"{"contentBlockIndex":0,"delta":{"text":"hello"}}"#;
        let frame = build_frame("contentBlockDelta", payload);

        let (ef, _) = parse_event_frame(&frame).expect("should parse");
        assert_eq!(ef.event_type, "contentBlockDelta");
        let val: serde_json::Value = serde_json::from_slice(&ef.payload).unwrap();
        assert_eq!(val["delta"]["text"], "hello");
    }

    #[test]
    fn parse_event_frame_two_consecutive_frames() {
        let frame1 = build_frame("contentBlockDelta", b"{\"delta\":{\"text\":\"a\"}}");
        let frame2 = build_frame("messageStop", b"{\"stopReason\":\"end_turn\"}");
        let mut combined = frame1.clone();
        combined.extend_from_slice(&frame2);

        let (ef1, consumed1) = parse_event_frame(&combined).expect("first frame");
        assert_eq!(ef1.event_type, "contentBlockDelta");
        assert_eq!(consumed1, frame1.len());

        let (ef2, consumed2) = parse_event_frame(&combined[consumed1..]).expect("second frame");
        assert_eq!(ef2.event_type, "messageStop");
        assert_eq!(consumed2, frame2.len());
    }

    // ---- Message conversion ------------------------------------------------

    #[test]
    fn to_converse_messages_system_extracted() {
        let messages = vec![
            Message::system("You are a helpful assistant."),
            Message::user("Hello"),
        ];
        let (system, msgs) = to_converse_messages(&messages);
        assert_eq!(system.as_deref(), Some("You are a helpful assistant."));
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"][0]["text"], "Hello");
    }

    #[test]
    fn to_converse_messages_no_system() {
        let messages = vec![Message::user("ping")];
        let (system, msgs) = to_converse_messages(&messages);
        assert!(system.is_none());
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn to_converse_messages_assistant_with_text() {
        let messages = vec![
            Message::user("hi"),
            Message::assistant(Some("hello there".to_string()), vec![]),
        ];
        let (_, msgs) = to_converse_messages(&messages);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"][0]["text"], "hello there");
    }

    #[test]
    fn to_converse_messages_tool_result_merged() {
        use crate::copilot::ApiToolCall;
        use crate::copilot::ApiToolCallFunction;
        let tool_call = ApiToolCall {
            id: "call_1".to_string(),
            kind: "function".to_string(),
            function: ApiToolCallFunction {
                name: "shell_command".to_string(),
                arguments: r#"{"command":"echo hi"}"#.to_string(),
            },
        };
        let messages = vec![
            Message::user("run it"),
            Message::assistant(None, vec![tool_call]),
            Message::tool_result("call_1", "hi\n"),
        ];
        let (_, msgs) = to_converse_messages(&messages);
        // Should produce: user + assistant + user(toolResult)
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[2]["role"], "user");
        assert!(msgs[2]["content"][0].get("toolResult").is_some());
    }

    #[test]
    fn to_converse_tools_basic() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "shell_command",
                "description": "Run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                }
            }
        })];
        let converse = to_converse_tools(&tools);
        assert_eq!(converse.len(), 1);
        assert_eq!(converse[0]["toolSpec"]["name"], "shell_command");
        assert_eq!(
            converse[0]["toolSpec"]["description"],
            "Run a shell command"
        );
        assert!(converse[0]["toolSpec"]["inputSchema"]["json"].is_object());
    }

    #[test]
    fn to_converse_tools_empty() {
        let converse = to_converse_tools(&[]);
        assert!(converse.is_empty());
    }
}
