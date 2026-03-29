//! Integration tests for the amaebi daemon using the mock LLM server.
//!
//! Each test:
//! 1. Starts the mock LLM server on a random port.
//! 2. Starts the `amaebi daemon` binary, pointed at the mock server.
//! 3. Connects a client to the daemon socket.
//! 4. Exercises the daemon and asserts on the responses / captured requests.

mod support;

use support::{
    helpers::{collect_text, connect_client, send_message, start_daemon},
    mock_llm::{MockLlmServer, ScriptedResponse},
};

// ---------------------------------------------------------------------------
// 1. Basic chat round-trip
// ---------------------------------------------------------------------------

/// Send a simple text message → mock returns text chunks → client receives output.
#[tokio::test]
async fn test_basic_chat_roundtrip() {
    let server = MockLlmServer::start().await;
    server.enqueue(ScriptedResponse::text_chunks(vec!["Hello", ", ", "world!"]));

    let daemon = start_daemon(&server.url()).await.expect("start_daemon");
    let client = connect_client(&daemon.socket);

    let responses = send_message(&client, "hi there")
        .await
        .expect("send_message");

    let text = collect_text(&responses);
    assert_eq!(text, "Hello, world!", "unexpected text: {text:?}");

    // There should be a Done frame at the end.
    assert!(
        responses
            .iter()
            .any(|r| matches!(r, support::helpers::Response::Done)),
        "no Done frame in {responses:?}"
    );
}

// ---------------------------------------------------------------------------
// 2. Tool-call round-trip
// ---------------------------------------------------------------------------

/// Mock returns a tool_call → daemon executes → sends result back → mock returns final text.
///
/// This test uses the `shell` tool with a harmless `echo` command so the
/// daemon's `LocalExecutor` can run it without Docker.
#[tokio::test]
async fn test_tool_call_roundtrip() {
    let server = MockLlmServer::start().await;

    // First response: request a shell tool call that echoes a string.
    server.enqueue(ScriptedResponse::tool_call(
        "call-001",
        "shell",
        r#"{"command":"echo integration-test-marker"}"#,
    ));
    // Second response (after daemon sends tool result): plain text reply.
    server.enqueue(ScriptedResponse::text_chunks(vec![
        "Tool done. ",
        "integration-test-marker",
    ]));

    let daemon = start_daemon(&server.url()).await.expect("start_daemon");
    let client = connect_client(&daemon.socket);

    let responses = send_message(&client, "run echo")
        .await
        .expect("send_message");

    // The daemon should have forwarded the tool result and streamed back the
    // second text response.
    let text = collect_text(&responses);
    assert!(
        text.contains("Tool done."),
        "expected final text in response, got: {text:?}"
    );

    // The mock server should have received exactly 2 requests:
    // the initial user chat and the follow-up with the tool result.
    let reqs = server.take_requests();
    assert_eq!(
        reqs.len(),
        2,
        "expected 2 requests to mock, got {}",
        reqs.len()
    );
}

// ---------------------------------------------------------------------------
// 3. max_tokens sent correctly
// ---------------------------------------------------------------------------

/// The daemon must include `max_tokens` in every request to the LLM.
#[tokio::test]
async fn test_max_tokens_sent_correctly() {
    let server = MockLlmServer::start().await;
    server.enqueue(ScriptedResponse::text_chunks(vec!["ok"]));

    let daemon = start_daemon(&server.url()).await.expect("start_daemon");
    let client = connect_client(&daemon.socket);
    send_message(&client, "test tokens")
        .await
        .expect("send_message");

    let reqs = server.take_requests();
    assert!(!reqs.is_empty(), "no requests captured");

    let max_tokens = reqs[0].max_tokens();
    assert!(
        max_tokens.is_some(),
        "max_tokens missing from request body: {:?}",
        reqs[0].body
    );
    let val = max_tokens.unwrap();
    assert!(val > 0, "max_tokens should be positive, got {val}");
}

// ---------------------------------------------------------------------------
// 4. Request format validation
// ---------------------------------------------------------------------------

/// The request sent to the LLM must include: messages array, model, stream:true.
#[tokio::test]
async fn test_request_format_valid() {
    let server = MockLlmServer::start().await;
    server.enqueue(ScriptedResponse::text_chunks(vec!["valid"]));

    let daemon = start_daemon(&server.url()).await.expect("start_daemon");
    let client = connect_client(&daemon.socket);
    send_message(&client, "format check")
        .await
        .expect("send_message");

    let reqs = server.take_requests();
    assert!(!reqs.is_empty(), "no requests captured");

    let req = &reqs[0];

    // model must be non-empty
    assert!(
        req.model().is_some_and(|m| !m.is_empty()),
        "model field missing or empty: {:?}",
        req.body
    );

    // stream must be true
    assert!(
        req.is_streaming(),
        "stream field is not true: {:?}",
        req.body
    );

    // messages must be a non-empty array where every entry has a role
    let messages = req.messages().expect("messages array missing");
    assert!(!messages.is_empty(), "messages array is empty");
    for (i, msg) in messages.iter().enumerate() {
        assert!(
            msg.get("role").is_some(),
            "message[{i}] missing role: {msg:?}"
        );
    }

    // max_tokens must be present
    assert!(
        req.max_tokens().is_some(),
        "max_tokens missing: {:?}",
        req.body
    );
}
