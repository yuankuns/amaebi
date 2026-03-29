//! Integration tests for the amaebi daemon using the mock LLM server.
//!
//! Each test:
//! 1. Starts the mock LLM server on a random port.
//! 2. Starts the `amaebi daemon` binary, pointed at the mock server.
//! 3. Connects a client to the daemon socket.
//! 4. Exercises the daemon and asserts on the responses / captured requests.

mod support;

use support::{
    helpers::{
        collect_text, connect_client, send_message, send_message_with_session, send_resume,
        start_daemon, start_daemon_at_home_with_env, start_daemon_with_env,
    },
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
        "shell_command",
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
    // The daemon uses a fixed max_tokens of 4096 (see copilot.rs send_with_retry).
    assert_eq!(val, 4096, "max_tokens should equal 4096, got {val}");
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

// ---------------------------------------------------------------------------
// 5. Compaction: triggered at threshold
// ---------------------------------------------------------------------------

/// When the conversation context exceeds the compaction threshold, the daemon
/// sends a `Response::Compacting` frame to the client and spawns a background
/// summary task.
///
/// Strategy:
/// 1. Seed 4 turns (8 history messages > HOT_TAIL_PAIRS*2=6) at a high threshold.
/// 2. Kill the seeding daemon; restart with the same home dir and threshold=50.
/// 3. Send one more turn → expect Compacting frame + background summary LLM call.
#[tokio::test]
async fn test_compaction_triggered_at_threshold() {
    let server = MockLlmServer::start().await;

    // --- Phase 1: seed 4 turns at high threshold ---
    let seed_daemon =
        start_daemon_with_env(&server.url(), &[("AMAEBI_COMPACTION_THRESHOLD", "100000")])
            .await
            .expect("seed daemon");
    let seed_client = connect_client(&seed_daemon.socket);
    let session_id = uuid::Uuid::new_v4().to_string();

    for i in 1..=4u32 {
        server.enqueue(ScriptedResponse::text_chunks(vec![&format!("Seed {i}.")]));
        let r = send_message_with_session(
            &seed_client,
            &format!("Seed message {i}."),
            &session_id,
            "gpt-4o",
        )
        .await
        .unwrap_or_else(|e| panic!("seed turn {i} failed: {e}"));
        assert!(
            !r.iter()
                .any(|f| matches!(f, support::helpers::Response::Compacting)),
            "unexpected Compacting during seeding at turn {i}"
        );
    }
    server.take_requests(); // drain seed requests

    // --- Phase 2: restart daemon with low threshold ---
    // Destructure seed_daemon to keep home_dir alive (preserving the SQLite DB)
    // while killing only the child process and cleaning up the socket.
    let (home_path, _home_dir) = seed_daemon.kill_and_keep_home();
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    server.enqueue(ScriptedResponse::text_chunks(vec!["Trigger reply."]));
    server.enqueue(ScriptedResponse::text_chunks(vec![
        "- Key fact: user triggered compaction.",
    ]));

    let (trigger_socket, mut trigger_child, _dir) = start_daemon_at_home_with_env(
        &home_path,
        &server.url(),
        &[("AMAEBI_COMPACTION_THRESHOLD", "50")],
    )
    .await
    .expect("trigger daemon");
    let trigger_client = connect_client(&trigger_socket);

    let responses = send_message_with_session(
        &trigger_client,
        "Trigger compaction now.",
        &session_id,
        "gpt-4o",
    )
    .await
    .expect("trigger turn");

    let text = collect_text(&responses);
    assert!(
        text.contains("Trigger reply."),
        "expected reply in response, got: {text:?}"
    );

    // Must receive a Compacting frame.
    assert!(
        responses
            .iter()
            .any(|r| matches!(r, support::helpers::Response::Compacting)),
        "expected Compacting frame in responses: {responses:?}"
    );

    // Wait for background summary call to arrive.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        if server.peek_request_count() >= 2 || std::time::Instant::now() > deadline {
            break;
        }
    }

    let reqs = server.take_requests();
    assert!(
        reqs.len() >= 2,
        "expected at least 2 LLM requests (chat + compaction summary), got {}",
        reqs.len()
    );

    let _ = trigger_child.kill();
}

// ---------------------------------------------------------------------------
// 6. Compaction: summary preserved in next turn
// ---------------------------------------------------------------------------

/// After compaction runs, the next message should include the summary text in
/// the messages sent to the LLM.
///
/// Uses the same two-phase pattern as test 5: seed 4 turns, restart at low
/// threshold to trigger compaction+summary, then send a follow-up turn and
/// assert the summary appears in the context.
#[tokio::test]
async fn test_compaction_preserves_summary() {
    const SUMMARY_TEXT: &str = "- Key fact: compaction was tested successfully.";

    let server = MockLlmServer::start().await;

    // Phase 1: seed 4 turns.
    let seed_daemon =
        start_daemon_with_env(&server.url(), &[("AMAEBI_COMPACTION_THRESHOLD", "100000")])
            .await
            .expect("seed daemon");
    let seed_client = connect_client(&seed_daemon.socket);
    let session_id = uuid::Uuid::new_v4().to_string();

    for i in 1..=4u32 {
        server.enqueue(ScriptedResponse::text_chunks(vec![&format!("Seed {i}.")]));
        send_message_with_session(
            &seed_client,
            &format!("Seed message {i}."),
            &session_id,
            "gpt-4o",
        )
        .await
        .unwrap_or_else(|e| panic!("seed turn {i}: {e}"));
    }
    server.take_requests();

    let (home_path, _home_dir2) = seed_daemon.kill_and_keep_home();
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // Phase 2: restart with low threshold → compaction fires on first turn.
    server.enqueue(ScriptedResponse::text_chunks(vec!["Turn A reply."]));
    server.enqueue(ScriptedResponse::text_chunks(vec![SUMMARY_TEXT]));

    let (socket2, mut child2, _dir2) = start_daemon_at_home_with_env(
        &home_path,
        &server.url(),
        &[("AMAEBI_COMPACTION_THRESHOLD", "50")],
    )
    .await
    .expect("compaction daemon");
    let client2 = connect_client(&socket2);

    let ra = send_message_with_session(&client2, "What do you know?", &session_id, "gpt-4o")
        .await
        .expect("turn A");
    assert!(
        ra.iter()
            .any(|r| matches!(r, support::helpers::Response::Compacting)),
        "expected Compacting in turn A: {ra:?}"
    );

    // Wait for background summary to complete (2 requests: chat + summary).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        if server.peek_request_count() >= 2 || std::time::Instant::now() > deadline {
            break;
        }
    }
    server.take_requests();
    // Extra wait for SQLite write.
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Phase 3: follow-up turn on same daemon — summary should appear in context.
    server.enqueue(ScriptedResponse::text_chunks(vec!["Turn B reply."]));
    let rb = send_message_with_session(&client2, "What was the summary?", &session_id, "gpt-4o")
        .await
        .expect("turn B");
    let text_b = collect_text(&rb);
    assert!(
        text_b.contains("Turn B reply."),
        "expected turn B reply, got: {text_b:?}"
    );

    let reqs_b = server.take_requests();
    assert!(!reqs_b.is_empty(), "no requests for turn B");

    let messages = reqs_b[0].messages().expect("messages array");
    let all_content: String = messages
        .iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        all_content.contains("Key fact"),
        "summary text not found in turn B messages. Content:\n{all_content}"
    );

    let _ = child2.kill();
}

// ---------------------------------------------------------------------------
// 7. Compaction: hot tail preserved after compaction
// ---------------------------------------------------------------------------

/// After compaction the messages sent to the LLM for the next turn should
/// contain at most HOT_TAIL_PAIRS*2 + 2 (summary pair) + 1 (current prompt)
/// user/assistant messages — not the full unbounded history.
///
/// HOT_TAIL_PAIRS = 3, so hot window = 6 history messages.
/// We seed 8 turns (16 messages >> 6 hot tail) then trigger compaction.
/// After compaction the next turn should have ≤ 6 + 2 + 1 = 9 u/a messages.
#[tokio::test]
async fn test_hot_tail_preserved_after_compaction() {
    const SUMMARY_TEXT: &str = "- Summary: 8 seed turns about hot tail testing.";
    const HOT_TAIL: usize = 6; // HOT_TAIL_PAIRS * 2

    let server = MockLlmServer::start().await;

    // Phase 1: seed 8 turns (well above hot tail).
    let seed_daemon =
        start_daemon_with_env(&server.url(), &[("AMAEBI_COMPACTION_THRESHOLD", "100000")])
            .await
            .expect("seed daemon");
    let seed_client = connect_client(&seed_daemon.socket);
    let session_id = uuid::Uuid::new_v4().to_string();

    for i in 1..=8u32 {
        server.enqueue(ScriptedResponse::text_chunks(vec![&format!("Seed {i}.")]));
        send_message_with_session(
            &seed_client,
            &format!("Seed message {i}."),
            &session_id,
            "gpt-4o",
        )
        .await
        .unwrap_or_else(|e| panic!("seed turn {i}: {e}"));
    }
    server.take_requests();

    let (home_path, _home_dir3) = seed_daemon.kill_and_keep_home();
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // Phase 2: trigger compaction.
    server.enqueue(ScriptedResponse::text_chunks(vec!["Trigger reply."]));
    server.enqueue(ScriptedResponse::text_chunks(vec![SUMMARY_TEXT]));

    let (socket2, mut child2, _dir2) = start_daemon_at_home_with_env(
        &home_path,
        &server.url(),
        &[("AMAEBI_COMPACTION_THRESHOLD", "50")],
    )
    .await
    .expect("compaction daemon");
    let client2 = connect_client(&socket2);

    let rt = send_message_with_session(&client2, "Trigger compaction.", &session_id, "gpt-4o")
        .await
        .expect("trigger turn");
    assert!(
        rt.iter()
            .any(|r| matches!(r, support::helpers::Response::Compacting)),
        "expected Compacting: {rt:?}"
    );

    // Wait for background summary.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        if server.peek_request_count() >= 2 || std::time::Instant::now() > deadline {
            break;
        }
    }
    server.take_requests();
    tokio::time::sleep(std::time::Duration::from_millis(600)).await;

    // Phase 3: follow-up turn — should only include hot tail + summary, not full history.
    server.enqueue(ScriptedResponse::text_chunks(vec!["Hot tail reply."]));
    let rh = send_message_with_session(&client2, "What is the hot tail?", &session_id, "gpt-4o")
        .await
        .expect("hot tail turn");
    assert!(
        rh.iter()
            .any(|r| matches!(r, support::helpers::Response::Done)),
        "no Done in hot tail turn: {rh:?}"
    );

    let reqs = server.take_requests();
    assert!(!reqs.is_empty(), "no request for hot tail turn");

    let messages = reqs[0].messages().expect("messages array");
    let history_count = messages
        .iter()
        .filter(|m| {
            m.get("role")
                .and_then(|r| r.as_str())
                .map(|r| r == "user" || r == "assistant")
                .unwrap_or(false)
        })
        .count();

    // Bound: summary pair (2) + hot tail (HOT_TAIL=6) + current prompt (1) = 9.
    // Being generous to +2 for any edge cases.
    assert!(
        history_count <= HOT_TAIL + 4,
        "history message count {history_count} exceeds hot tail bound ({} expected ≤ {})",
        history_count,
        HOT_TAIL + 4
    );

    let _ = child2.kill();
}

// ---------------------------------------------------------------------------
// 8. Pre-flight trim on resume
// ---------------------------------------------------------------------------

/// When a session is *resumed*, the daemon loads the session history and
/// includes it in the request to the LLM.  If the history exceeds the
/// compaction threshold a pre-flight trim fires and only the hot tail
/// (HOT_TAIL_PAIRS * 2 messages) is sent.
///
/// This test verifies the Resume code path loads session history at all.
/// We use a high threshold so no trim fires, then verify that the seeded
/// history appears in the resume request.
#[tokio::test]
async fn test_pre_flight_trim_on_resume() {
    let server = MockLlmServer::start().await;

    // Phase 1: seed 1 turn at high threshold.
    server.enqueue(ScriptedResponse::text_chunks(vec!["Seed reply."]));

    let daemon = start_daemon_with_env(&server.url(), &[("AMAEBI_COMPACTION_THRESHOLD", "100000")])
        .await
        .expect("seed daemon");
    let client = connect_client(&daemon.socket);

    let session_id = uuid::Uuid::new_v4().to_string();
    let r1 = send_message_with_session(
        &client,
        "Seed message for resume test.",
        &session_id,
        "gpt-4o",
    )
    .await
    .expect("seed turn");
    assert!(
        collect_text(&r1).contains("Seed reply."),
        "seed turn failed: {r1:?}"
    );
    assert!(
        !r1.iter()
            .any(|r| matches!(r, support::helpers::Response::Compacting)),
        "unexpected Compacting in seed turn: {r1:?}"
    );
    // Drain seed request so we get a clean slate for the resume turn.
    server.take_requests();

    // Phase 2: resume the same session.  Should include the seeded history.
    server.enqueue(ScriptedResponse::text_chunks(vec!["Resume reply."]));

    let r2 = send_resume(&client, "Continue from before.", &session_id, "gpt-4o")
        .await
        .expect("resume turn");
    assert!(
        collect_text(&r2).contains("Resume reply."),
        "resume turn failed: {r2:?}"
    );

    // The resume request to the mock must include the prior history.
    let reqs = server.take_requests();
    assert!(!reqs.is_empty(), "no requests captured for resume turn");

    let resume_req = &reqs[0];
    let messages = resume_req.messages().expect("messages in resume request");
    let all_content: String = messages
        .iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .collect::<Vec<_>>()
        .join(" ");

    assert!(
        all_content.contains("Seed message for resume test."),
        "resume request should include prior session history, got:\n{all_content}"
    );

    // Phase 3: verify pre-flight trim fires when threshold is low.
    // Restart the daemon with the same home dir but low threshold.
    let (home_path, _home_dir4) = daemon.kill_and_keep_home();
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // Seed 4 more turns so there's meaningful history (> HOT_TAIL_PAIRS*2 = 6 msgs).
    let (socket2, mut child2, _dir2) = start_daemon_at_home_with_env(
        &home_path,
        &server.url(),
        &[("AMAEBI_COMPACTION_THRESHOLD", "100000")],
    )
    .await
    .expect("seed2 daemon");
    let client2 = connect_client(&socket2);

    for i in 2..=5u32 {
        server.enqueue(ScriptedResponse::text_chunks(vec![&format!("Seed {i}.")]));
        send_message_with_session(&client2, &format!("Extra seed {i}."), &session_id, "gpt-4o")
            .await
            .unwrap_or_else(|e| panic!("extra seed {i}: {e}"));
    }
    server.take_requests();
    let _ = child2.kill();
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // Restart with low threshold; the resume/chat request should apply pre-flight trim.
    server.enqueue(ScriptedResponse::text_chunks(vec!["Trim resume reply."]));
    // Background summary if compaction also fires.
    server.enqueue(ScriptedResponse::text_chunks(vec!["- Trim summary."]));

    let (socket3, mut child3, _dir3) = start_daemon_at_home_with_env(
        &home_path,
        &server.url(),
        &[("AMAEBI_COMPACTION_THRESHOLD", "50")],
    )
    .await
    .expect("trim daemon");
    let client3 = connect_client(&socket3);

    let r3 = send_resume(&client3, "What is the trim?", &session_id, "gpt-4o")
        .await
        .expect("trim resume");

    // The resume reply must arrive (no crash).
    let text3 = collect_text(&r3);
    assert!(
        text3.contains("Trim resume reply."),
        "expected trim resume reply, got: {text3:?}"
    );

    // Wait for any background compaction.
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let reqs3 = server.take_requests();
    assert!(!reqs3.is_empty(), "no request for trim resume turn");

    // The trim request should have ≤ HOT_TAIL_PAIRS*2 + 2 user/assistant messages.
    let msg3 = reqs3[0].messages().expect("messages");
    let hist3 = msg3
        .iter()
        .filter(|m| {
            m.get("role")
                .and_then(|r| r.as_str())
                .map(|r| r == "user" || r == "assistant")
                .unwrap_or(false)
        })
        .count();
    // HOT_TAIL_PAIRS*2=6 + current prompt (1) + possible summary pair (2) = 9.
    assert!(
        hist3 <= 10,
        "trim resume message count {hist3} exceeds expected hot tail bound"
    );

    let _ = child3.kill();
}
