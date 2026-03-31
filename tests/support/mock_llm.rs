//! Mock LLM HTTP server that mimics the OpenAI/Copilot streaming chat API.
//!
//! # Usage
//!
//! ```rust,no_run
//! let server = MockLlmServer::start().await;
//! server.enqueue(ScriptedResponse::text_chunks(vec!["Hello", " world"]));
//! let url = server.url();         // e.g. "http://127.0.0.1:PORT/chat/completions"
//! // … point daemon at url …
//! let captured = server.take_requests();   // inspect what the daemon sent
//! ```

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::Response as AxumResponse,
    routing::post,
    Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::sync::oneshot;
use tokio_stream::wrappers::ReceiverStream;

// ---------------------------------------------------------------------------
// Scripted response types
// ---------------------------------------------------------------------------

/// A single chunk returned as part of an SSE stream.
#[derive(Clone, Debug)]
pub enum Chunk {
    /// A plain text delta.
    Text(String),
    /// A tool-call delta — first chunk should include function name, subsequent
    /// ones append to the arguments string.
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        function_name: Option<String>,
        arguments_fragment: String,
    },
    /// Mark the stream as finished (sends `finish_reason`).
    Finish(FinishReason),
}

#[derive(Clone, Debug)]
pub enum FinishReason {
    Stop,
    ToolCalls,
    #[allow(dead_code)]
    Length,
}

impl FinishReason {
    fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::Length => "length",
        }
    }
}

/// A complete scripted LLM response: a sequence of chunks with optional delays.
#[derive(Clone, Debug)]
pub struct ScriptedResponse {
    pub chunks: Vec<(Chunk, Option<Duration>)>,
}

impl ScriptedResponse {
    /// Create a response that streams `parts` as plain text chunks, then stops.
    pub fn text_chunks(parts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let mut chunks: Vec<(Chunk, Option<Duration>)> = parts
            .into_iter()
            .map(|p| (Chunk::Text(p.into()), None))
            .collect();
        chunks.push((Chunk::Finish(FinishReason::Stop), None));
        Self { chunks }
    }

    /// Create a response that returns a single tool call then stops.
    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        args: impl Into<String>,
    ) -> Self {
        Self {
            chunks: vec![
                (
                    Chunk::ToolCallDelta {
                        index: 0,
                        id: Some(id.into()),
                        function_name: Some(name.into()),
                        arguments_fragment: String::new(),
                    },
                    None,
                ),
                (
                    Chunk::ToolCallDelta {
                        index: 0,
                        id: None,
                        function_name: None,
                        arguments_fragment: args.into(),
                    },
                    None,
                ),
                (Chunk::Finish(FinishReason::ToolCalls), None),
            ],
        }
    }

    /// Create a response that returns multiple tool calls then stops.
    ///
    /// Each item in `calls` is `(id, name, arguments_json)`.
    pub fn multi_tool_calls<I, S1, S2, S3>(calls: I) -> Self
    where
        I: IntoIterator<Item = (S1, S2, S3)>,
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
    {
        let mut chunks: Vec<(Chunk, Option<Duration>)> = Vec::new();
        for (index, (id, name, args)) in calls.into_iter().enumerate() {
            chunks.push((
                Chunk::ToolCallDelta {
                    index,
                    id: Some(id.into()),
                    function_name: Some(name.into()),
                    arguments_fragment: String::new(),
                },
                None,
            ));
            chunks.push((
                Chunk::ToolCallDelta {
                    index,
                    id: None,
                    function_name: None,
                    arguments_fragment: args.into(),
                },
                None,
            ));
        }
        chunks.push((Chunk::Finish(FinishReason::ToolCalls), None));
        Self { chunks }
    }
}

// ---------------------------------------------------------------------------
// Captured request
// ---------------------------------------------------------------------------

/// A captured incoming request from the daemon.
#[derive(Clone, Debug)]
pub struct CapturedRequest {
    #[allow(dead_code)]
    pub headers: Vec<(String, String)>,
    pub body: Value,
}

impl CapturedRequest {
    /// Return the `model` field.
    pub fn model(&self) -> Option<&str> {
        self.body.get("model").and_then(|v| v.as_str())
    }

    /// Return the `max_tokens` field.
    pub fn max_tokens(&self) -> Option<u64> {
        self.body.get("max_tokens").and_then(|v| v.as_u64())
    }

    /// Return `true` if `stream` is `true`.
    pub fn is_streaming(&self) -> bool {
        self.body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    /// Return the messages array.
    pub fn messages(&self) -> Option<&Vec<Value>> {
        self.body.get("messages").and_then(|v| v.as_array())
    }
}

// ---------------------------------------------------------------------------
// Queued item (scripted SSE response or a pre-canned HTTP error)
// ---------------------------------------------------------------------------

/// An item in the mock server's response queue.
#[derive(Clone, Debug)]
pub(crate) enum QueuedItem {
    /// Stream a normal SSE scripted response.
    Response(ScriptedResponse),
    /// Return an HTTP error immediately (no SSE body).
    Error { status: u16, body: String },
}

// ---------------------------------------------------------------------------
// Server state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ServerState {
    /// Queue of scripted responses. Tests push; handler pops.
    responses: Arc<Mutex<VecDeque<QueuedItem>>>,
    /// Log of all captured requests.
    captured: Arc<Mutex<Vec<CapturedRequest>>>,
    /// Whether to validate incoming requests.
    validate: bool,
}

// ---------------------------------------------------------------------------
// Public handle
// ---------------------------------------------------------------------------

/// A running mock LLM server.
pub struct MockLlmServer {
    addr: SocketAddr,
    state: ServerState,
    _shutdown: oneshot::Sender<()>,
}

impl MockLlmServer {
    /// Start the server on a random free port and return a handle.
    pub async fn start() -> Self {
        let state = ServerState {
            responses: Arc::new(Mutex::new(VecDeque::<QueuedItem>::new())),
            captured: Arc::new(Mutex::new(Vec::new())),
            validate: true,
        };

        let app = Router::new()
            .route("/chat/completions", post(handle_completion))
            .with_state(state.clone());

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("binding mock server");
        let addr = listener.local_addr().expect("getting local addr");

        let (tx, rx) = oneshot::channel::<()>();

        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = rx.await;
                })
                .await
                .ok();
        });

        Self {
            addr,
            state,
            _shutdown: tx,
        }
    }

    /// The full URL to use as the `AMAEBI_COPILOT_URL` env var override.
    pub fn url(&self) -> String {
        format!("http://{}/chat/completions", self.addr)
    }

    /// Push a scripted response onto the queue.
    pub fn enqueue(&self, resp: ScriptedResponse) {
        self.state
            .responses
            .lock()
            .unwrap()
            .push_back(QueuedItem::Response(resp));
    }

    /// Push an HTTP error response (non-retryable 4xx) onto the queue.
    ///
    /// When this item is popped the handler returns the given status code and
    /// body without streaming any SSE.  Use a 4xx status to bypass the
    /// daemon's retry-on-5xx logic so tests complete without backoff delays.
    pub fn enqueue_error(&self, status: u16, body: impl Into<String>) {
        self.state
            .responses
            .lock()
            .unwrap()
            .push_back(QueuedItem::Error {
                status,
                body: body.into(),
            });
    }

    /// Take and return all captured requests since the last call.
    pub fn take_requests(&self) -> Vec<CapturedRequest> {
        std::mem::take(&mut *self.state.captured.lock().unwrap())
    }

    /// Return the number of captured requests without draining the queue.
    pub fn peek_request_count(&self) -> usize {
        self.state.captured.lock().unwrap().len()
    }
}

// ---------------------------------------------------------------------------
// Request validation
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Value>,
    #[allow(dead_code)]
    stream: Option<bool>,
    max_tokens: Option<u64>,
}

fn validate_request(req: &ChatRequest) -> Result<(), String> {
    if req.model.is_empty() {
        return Err("model field is empty".into());
    }
    if req.messages.is_empty() {
        return Err("messages array is empty".into());
    }
    for (i, msg) in req.messages.iter().enumerate() {
        if msg.get("role").is_none() {
            return Err(format!("message[{i}] missing role"));
        }
    }
    if req.max_tokens.is_none() {
        return Err("max_tokens is missing".into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// SSE helpers
// ---------------------------------------------------------------------------

fn sse_line(data: &Value) -> String {
    format!("data: {}\n\n", serde_json::to_string(data).unwrap())
}

fn text_delta_event(chunk: &str) -> String {
    let data = json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "model": "mock-model",
        "choices": [{
            "index": 0,
            "delta": { "role": "assistant", "content": chunk },
            "finish_reason": null
        }]
    });
    sse_line(&data)
}

fn tool_call_delta_event(
    index: usize,
    id: &Option<String>,
    function_name: &Option<String>,
    arguments_fragment: &str,
) -> String {
    let mut tc = json!({
        "index": index,
        "type": "function",
        "function": {
            "arguments": arguments_fragment
        }
    });
    if let Some(id_val) = id {
        tc["id"] = json!(id_val);
    }
    if let Some(name) = function_name {
        tc["function"]["name"] = json!(name);
    }
    let data = json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "model": "mock-model",
        "choices": [{
            "index": 0,
            "delta": { "tool_calls": [tc] },
            "finish_reason": null
        }]
    });
    sse_line(&data)
}

fn finish_event(reason: &FinishReason) -> String {
    let data = json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "model": "mock-model",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": reason.as_str()
        }]
    });
    sse_line(&data)
}

// ---------------------------------------------------------------------------
// Axum handler
// ---------------------------------------------------------------------------

async fn handle_completion(
    State(state): State<ServerState>,
    headers: HeaderMap,
    axum::Json(req_body): axum::Json<ChatRequest>,
) -> Result<AxumResponse<Body>, (StatusCode, String)> {
    // Validate if enabled.
    if state.validate {
        if let Err(msg) = validate_request(&req_body) {
            return Err((StatusCode::BAD_REQUEST, msg));
        }
    }

    // Capture request.
    let captured_headers: Vec<(String, String)> = headers
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
        .collect();
    let body_value = json!({
        "model": req_body.model,
        "messages": req_body.messages,
        "stream": req_body.stream,
        "max_tokens": req_body.max_tokens,
    });
    state.captured.lock().unwrap().push(CapturedRequest {
        headers: captured_headers,
        body: body_value,
    });

    // Pop a queued item — fail fast if nothing queued.
    let item = match state.responses.lock().unwrap().pop_front() {
        Some(i) => i,
        None => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "no scripted response queued".to_string(),
            ))
        }
    };

    // Pre-canned HTTP error: return immediately without streaming.
    let scripted = match item {
        QueuedItem::Response(r) => r,
        QueuedItem::Error { status, body } => {
            let code = StatusCode::from_u16(status)
                .expect("invalid HTTP status code for scripted error in mock LLM server");
            return Err((code, body));
        }
    };

    // Build a real async stream that yields SSE chunks with optional delays.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<bytes::Bytes, std::convert::Infallible>>(16);

    tokio::spawn(async move {
        for (chunk, delay) in scripted.chunks {
            if let Some(d) = delay {
                tokio::time::sleep(d).await;
            }
            let line = match &chunk {
                Chunk::Text(t) => text_delta_event(t),
                Chunk::ToolCallDelta {
                    index,
                    id,
                    function_name,
                    arguments_fragment,
                } => tool_call_delta_event(*index, id, function_name, arguments_fragment),
                Chunk::Finish(reason) => finish_event(reason),
            };
            if tx.send(Ok(bytes::Bytes::from(line))).await.is_err() {
                return;
            }
        }
        let _ = tx.send(Ok(bytes::Bytes::from("data: [DONE]\n\n"))).await;
    });

    let stream = ReceiverStream::new(rx);
    let response = AxumResponse::builder()
        .status(200)
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .body(Body::from_stream(stream))
        .expect("building SSE response");

    Ok(response)
}

// ---------------------------------------------------------------------------
// Tests for the mock server itself
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_server_returns_text() {
        let server = MockLlmServer::start().await;
        server.enqueue(ScriptedResponse::text_chunks(vec!["hello"]));

        // Use no_proxy() so the client connects directly to the local mock
        // server even when an HTTP proxy is configured in the environment.
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let resp = client
            .post(server.url())
            .json(&json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true,
                "max_tokens": 4096
            }))
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_success());
        let text = resp.text().await.unwrap();
        assert!(
            text.contains("hello"),
            "SSE body should contain 'hello': {text}"
        );
    }

    #[tokio::test]
    async fn mock_server_captures_request() {
        let server = MockLlmServer::start().await;
        server.enqueue(ScriptedResponse::text_chunks(vec!["ok"]));

        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        client
            .post(server.url())
            .json(&json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "test"}],
                "stream": true,
                "max_tokens": 1234
            }))
            .send()
            .await
            .unwrap();

        let reqs = server.take_requests();
        assert_eq!(reqs.len(), 1);
        assert_eq!(reqs[0].max_tokens(), Some(1234));
        assert_eq!(reqs[0].model(), Some("gpt-4o"));
        assert!(reqs[0].is_streaming());
    }
}
