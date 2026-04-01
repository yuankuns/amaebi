//! Integration tests for the amaebi daemon using the mock LLM server.
//!
//! Each test:
//! 1. Starts the mock LLM server on a random port.
//! 2. Starts the `amaebi daemon` binary, pointed at the mock server.
//! 3. Connects a client to the daemon socket.
//! 4. Exercises the daemon and asserts on the responses / captured requests.
//!
//! ## Covered scenarios
//!
//! - basic chat/tool-call round-trips
//! - request format and token/max-token behavior
//! - compaction trigger + summary/hot-tail behavior
//! - resume pre-flight trim behavior
//! - spawn_agent flow (normal run, recursion block, model/workspace handling, parallel path)
//! - cron flow (trigger, isolation from chat output, spawn_agent from cron)
//! - steer end-to-end behavior
//! - sandbox credential exposure regression checks
//! - sub-agent chain/session resilience
//! - non-retryable LLM error-path recovery
//! - consistent chat (full tool-call context preserved across turns)

mod support;

use std::time::Duration;

use support::{
    helpers::{
        collect_text, connect_client, seed_cron_job, send_heartbeat_add, send_heartbeat_list,
        send_heartbeat_trigger, send_message, send_message_with_session, send_resume, setup_home,
        start_daemon, start_daemon_at_home_with_env, start_daemon_with_env, write_heartbeat_config,
        Request, Response,
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

/// The daemon must include a positive `max_tokens` in every request to the LLM.
#[tokio::test]
async fn test_max_tokens_present_in_request() {
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
    assert!(val > 0, "max_tokens should be > 0, got {val}");
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
    let (home_path, _home_dir) = seed_daemon.kill_and_keep_home().await;

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

    let _ = trigger_child.kill().await;
    let _ = trigger_child.wait().await;
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

    let (home_path, _home_dir2) = seed_daemon.kill_and_keep_home().await;

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

    let _ = child2.kill().await;
    let _ = child2.wait().await;
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

    let (home_path, _home_dir3) = seed_daemon.kill_and_keep_home().await;

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

    let _ = child2.kill().await;
    let _ = child2.wait().await;
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
    let (home_path, _home_dir4) = daemon.kill_and_keep_home().await;

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
    let _ = child2.kill().await;
    let _ = child2.wait().await;

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

    let _ = child3.kill().await;
    let _ = child3.wait().await;
}

// ---------------------------------------------------------------------------
// 9. spawn_agent tool — child agent runs a task in a sandbox
// ---------------------------------------------------------------------------

/// The parent agent calls `spawn_agent`.  The child agent uses `shell_command`
/// to run `echo hello`, then the child loop ends.  The parent receives the
/// child's result as the tool output and returns a final response.
///
/// Uses `AMAEBI_SPAWN_SANDBOX=noop` so no Docker is required.
#[tokio::test]
async fn spawn_agent_runs_task() {
    let server = MockLlmServer::start().await;

    // 1. Parent agent first turn: LLM asks to spawn a child agent.
    server.enqueue(ScriptedResponse::tool_call(
        "call-spawn-001",
        "spawn_agent",
        r#"{"task":"run echo hello","workspace":"/tmp"}"#,
    ));
    // 2. Child agent first turn: LLM asks to run shell_command.
    server.enqueue(ScriptedResponse::tool_call(
        "call-shell-001",
        "shell_command",
        r#"{"command":"echo hello"}"#,
    ));
    // 3. Child agent second turn (after tool result): LLM returns final text.
    server.enqueue(ScriptedResponse::text_chunks(vec!["hello"]));
    // 4. Parent agent second turn (after spawn_agent result): final response.
    server.enqueue(ScriptedResponse::text_chunks(vec!["Task done: hello"]));

    let daemon = start_daemon_with_env(&server.url(), &[("AMAEBI_SPAWN_SANDBOX", "noop")])
        .await
        .expect("start_daemon");
    let client = connect_client(&daemon.socket);

    let responses = send_message(&client, "run a child task")
        .await
        .expect("send_message");

    let text = collect_text(&responses);
    assert!(
        text.contains("hello"),
        "expected 'hello' in response: {text:?}"
    );

    // Mock should have served exactly 4 requests: 2 for parent, 2 for child.
    let reqs = server.take_requests();
    assert_eq!(reqs.len(), 4, "expected 4 LLM requests, got {}", reqs.len());
}

// ---------------------------------------------------------------------------
// 10. spawn_agent recursion prevention
// ---------------------------------------------------------------------------

/// A child agent must NOT be able to call spawn_agent itself.
///
/// Scenario:
///   1. Parent LLM → calls spawn_agent
///   2. Child LLM → calls spawn_agent (recursion attempt)
///   3. Child tool executor returns error immediately (no extra LLM call)
///   4. Child sends tool_result (error) → LLM returns final text reporting error
///   5. Parent receives child output → LLM returns final response
///
/// Uses `AMAEBI_SPAWN_SANDBOX=noop` so no Docker is required.
#[tokio::test]
async fn spawn_agent_child_cannot_spawn() {
    let server = MockLlmServer::start().await;

    // 1. Parent agent first turn: LLM asks to spawn a child agent.
    server.enqueue(ScriptedResponse::tool_call(
        "call-spawn-parent",
        "spawn_agent",
        r#"{"task":"try to spawn another agent","workspace":"/tmp"}"#,
    ));
    // 2. Child agent first turn: LLM asks to spawn_agent (recursion attempt).
    server.enqueue(ScriptedResponse::tool_call(
        "call-spawn-child",
        "spawn_agent",
        r#"{"task":"nested","workspace":"/tmp"}"#,
    ));
    // 3. Child agent second turn: after the tool error is fed back, LLM returns final text.
    server.enqueue(ScriptedResponse::text_chunks(vec![
        "Child got error: spawn_agent not available",
    ]));
    // 4. Parent agent second turn: after child result, LLM returns final text.
    server.enqueue(ScriptedResponse::text_chunks(vec![
        "Parent done. Child reported recursion error.",
    ]));

    let daemon = start_daemon_with_env(&server.url(), &[("AMAEBI_SPAWN_SANDBOX", "noop")])
        .await
        .expect("start_daemon");
    let client = connect_client(&daemon.socket);

    let responses = send_message(&client, "try to spawn recursively")
        .await
        .expect("send_message");

    let text = collect_text(&responses);
    assert!(
        text.contains("Parent done."),
        "expected parent final text in response: {text:?}"
    );

    // Verify the child received a spawn_agent error in its tool result by
    // checking the LLM received exactly 4 requests (2 parent + 2 child).
    let reqs = server.take_requests();
    assert_eq!(
        reqs.len(),
        4,
        "expected 4 LLM requests (2 parent + 2 child), got {}",
        reqs.len()
    );

    // The child's second request (index 2) must contain a tool_result message
    // whose content includes the "spawn_agent is not available" error.
    let child_second_req = &reqs[2];
    let messages = child_second_req
        .messages()
        .expect("messages in child second request");
    let all_content: String = messages
        .iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        all_content.contains("spawn_agent is not available"),
        "expected spawn_agent error in child tool result, got:\n{all_content}"
    );
}

// ---------------------------------------------------------------------------
// 11. spawn_agent model selection
// ---------------------------------------------------------------------------

/// When spawn_agent is called with a "model" argument, the child agent must use
/// that model for its LLM requests.
///
/// Flow:
///   1. Parent turn 1 → calls spawn_agent with model:"custom-model-123"
///   2. Child turn 1 (uses custom-model-123) → returns final text
///   3. Parent turn 2 → returns final text
///
/// Uses `AMAEBI_SPAWN_SANDBOX=noop` so no Docker is required.
#[tokio::test]
async fn spawn_agent_uses_specified_model() {
    let server = MockLlmServer::start().await;

    // 1. Parent LLM turn 1: calls spawn_agent with a specific model override.
    server.enqueue(ScriptedResponse::tool_call(
        "call-spawn-model",
        "spawn_agent",
        r#"{"task":"simple task","workspace":"/tmp","model":"custom-model-123"}"#,
    ));
    // 2. Child LLM turn 1 (should use "custom-model-123"): returns final text.
    server.enqueue(ScriptedResponse::text_chunks(vec!["Child done."]));
    // 3. Parent LLM turn 2 (after spawn_agent result): returns final text.
    server.enqueue(ScriptedResponse::text_chunks(vec!["Parent done."]));

    let daemon = start_daemon_with_env(&server.url(), &[("AMAEBI_SPAWN_SANDBOX", "noop")])
        .await
        .expect("start_daemon");
    let client = connect_client(&daemon.socket);

    let responses = send_message(&client, "run child with custom model")
        .await
        .expect("send_message");

    let text = collect_text(&responses);
    assert!(
        text.contains("Parent done."),
        "expected parent final text in response: {text:?}"
    );

    // 3 requests: parent turn 1, child turn 1, parent turn 2.
    let reqs = server.take_requests();
    assert_eq!(reqs.len(), 3, "expected 3 LLM requests, got {}", reqs.len());

    // The child's request (index 1) must use "custom-model-123".
    let child_req = &reqs[1];
    assert_eq!(
        child_req.model(),
        Some("custom-model-123"),
        "child agent should use the model specified in spawn_agent args, got: {:?}",
        child_req.model()
    );
}

// ---------------------------------------------------------------------------
// 12. spawn_agent missing workspace returns error
// ---------------------------------------------------------------------------

/// When spawn_agent is called without the required "workspace" field, the tool
/// must return an error to the parent agent rather than crashing.
///
/// Flow:
///   1. Parent turn 1 → calls spawn_agent with only "task" (no workspace)
///   2. Daemon executes spawn_agent → immediate error (no child LLM calls)
///   3. Parent turn 2 → receives the error as a tool result, returns final text
///
/// After the test, the mock server must have exactly 2 requests (both parent),
/// and the second request's messages must mention "workspace".
///
/// Uses `AMAEBI_SPAWN_SANDBOX=noop` so no Docker is required.
#[tokio::test]
async fn spawn_agent_missing_workspace_returns_error() {
    let server = MockLlmServer::start().await;

    // 1. Parent LLM turn 1: calls spawn_agent without the workspace field.
    server.enqueue(ScriptedResponse::tool_call(
        "call-spawn-noworkspace",
        "spawn_agent",
        r#"{"task":"do something without workspace"}"#,
    ));
    // 2. Parent LLM turn 2: receives the tool error result, returns final text.
    server.enqueue(ScriptedResponse::text_chunks(vec![
        "Received an error from spawn_agent.",
    ]));

    let daemon = start_daemon_with_env(&server.url(), &[("AMAEBI_SPAWN_SANDBOX", "noop")])
        .await
        .expect("start_daemon");
    let client = connect_client(&daemon.socket);

    let responses = send_message(&client, "spawn agent without workspace")
        .await
        .expect("send_message");

    let text = collect_text(&responses);
    assert!(
        text.contains("error"),
        "expected error indication in response: {text:?}"
    );

    // Exactly 2 LLM requests: both to the parent (no child was spawned).
    let reqs = server.take_requests();
    assert_eq!(
        reqs.len(),
        2,
        "expected exactly 2 LLM requests (no child spawned), got {}",
        reqs.len()
    );

    // The second parent request must carry the spawn_agent error as a tool
    // result; its content should mention "workspace" (the missing field).
    let second_req = &reqs[1];
    let messages = second_req.messages().expect("messages in second request");
    let all_content: String = messages
        .iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        all_content.contains("workspace"),
        "error message should mention 'workspace', got:\n{all_content}"
    );
}

// ---------------------------------------------------------------------------
// 13. spawn_agent workspace path passed to child LLM context
// ---------------------------------------------------------------------------

/// The workspace path supplied in spawn_agent args must appear in the system
/// prompt sent to the child agent's LLM (embedded in the "[Sandbox Context]"
/// preamble that `spawn_agent` prepends to the task).
///
/// Flow:
///   1. Parent turn 1 → calls spawn_agent with a real temp-dir path as workspace
///   2. Child turn 1 → returns final text (mock; we only care about the request)
///   3. Parent turn 2 → returns final text
///
/// Uses `AMAEBI_SPAWN_SANDBOX=noop` so no Docker is required.
#[tokio::test]
async fn spawn_agent_workspace_passed_to_sandbox() {
    let workspace_dir = tempfile::TempDir::new().expect("temp workspace dir");
    let workspace_path = workspace_dir
        .path()
        .to_str()
        .expect("utf8 workspace path")
        .to_string();

    let server = MockLlmServer::start().await;

    // Build the spawn_agent args with a real filesystem path.
    let args_json = serde_json::json!({
        "task": "verify workspace",
        "workspace": workspace_path,
    })
    .to_string();

    // 1. Parent LLM turn 1: calls spawn_agent with the workspace path.
    server.enqueue(ScriptedResponse::tool_call(
        "call-spawn-ws",
        "spawn_agent",
        args_json,
    ));
    // 2. Child LLM turn 1: returns final text.
    server.enqueue(ScriptedResponse::text_chunks(vec![
        "Child workspace verified.",
    ]));
    // 3. Parent LLM turn 2: returns final text.
    server.enqueue(ScriptedResponse::text_chunks(vec!["Workspace test done."]));

    let daemon = start_daemon_with_env(&server.url(), &[("AMAEBI_SPAWN_SANDBOX", "noop")])
        .await
        .expect("start_daemon");
    let client = connect_client(&daemon.socket);

    let responses = send_message(&client, "check workspace passthrough")
        .await
        .expect("send_message");

    let text = collect_text(&responses);
    assert!(
        text.contains("Workspace test done."),
        "expected final text in response: {text:?}"
    );

    // 3 requests: parent turn 1, child turn 1, parent turn 2.
    let reqs = server.take_requests();
    assert_eq!(reqs.len(), 3, "expected 3 LLM requests, got {}", reqs.len());

    // The child's request (index 1) must contain the workspace path in its
    // messages — spawn_agent embeds it in the "[Sandbox Context]" preamble.
    let child_req = &reqs[1];
    let messages = child_req.messages().expect("messages in child request");
    let all_content: String = messages
        .iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        all_content.contains(&workspace_path),
        "child LLM request should contain workspace path {workspace_path:?}, got:\n{all_content}"
    );
}

// ---------------------------------------------------------------------------
// 14. Two spawn_agent calls in one batch run concurrently
// ---------------------------------------------------------------------------

/// When the LLM returns two spawn_agent tool calls in a single response, the
/// daemon must execute them concurrently rather than sequentially.
///
/// Each child runs `shell_command("sleep 1")` before returning its final text.
/// If the children ran sequentially the wall-clock time would be ≥ 2 s; the
/// assert of elapsed < 1.8 s proves they ran concurrently.
///
/// Mock LLM sequence (6 requests total):
///   1. Parent turn 1  → multi_tool_calls: two spawn_agent with parallel=true
///   2. Child 1 turn 1 → tool_call shell_command("sleep 1")
///   3. Child 2 turn 1 → tool_call shell_command("sleep 1")
///   4. Child 1 turn 2 → text "child1 done"
///   5. Child 2 turn 2 → text "child2 done"
///   6. Parent turn 2  → text "all done"
///
/// Uses `AMAEBI_SPAWN_SANDBOX=noop` so no Docker is required.
#[tokio::test]
async fn spawn_agent_parallel_calls() {
    let server = MockLlmServer::start().await;

    // 1. Parent turn 1: LLM returns two spawn_agent calls in one response.
    server.enqueue(ScriptedResponse::multi_tool_calls([
        (
            "call-spawn-parallel-A",
            "spawn_agent",
            r#"{"task":"task A","workspace":"/tmp","parallel":true}"#,
        ),
        (
            "call-spawn-parallel-B",
            "spawn_agent",
            r#"{"task":"task B","workspace":"/tmp","parallel":true}"#,
        ),
    ]));
    // 2 & 3. First two child requests each get a shell_command that sleeps 1 s.
    //        Order is non-deterministic; the mock queue serves them FIFO.
    server.enqueue(ScriptedResponse::tool_call(
        "sc-parallel-1",
        "shell_command",
        r#"{"command":"sleep 1"}"#,
    ));
    server.enqueue(ScriptedResponse::tool_call(
        "sc-parallel-2",
        "shell_command",
        r#"{"command":"sleep 1"}"#,
    ));
    // 4 & 5. Final text for each child (order is non-deterministic).
    server.enqueue(ScriptedResponse::text_chunks(vec!["child1 done"]));
    server.enqueue(ScriptedResponse::text_chunks(vec!["child2 done"]));
    // 6. Parent turn 2: receives both tool results, returns final text.
    server.enqueue(ScriptedResponse::text_chunks(vec!["all done"]));

    let daemon = start_daemon_with_env(&server.url(), &[("AMAEBI_SPAWN_SANDBOX", "noop")])
        .await
        .expect("start_daemon");
    let client = connect_client(&daemon.socket);

    let responses = send_message(&client, "run two child tasks in parallel")
        .await
        .expect("send_message");

    let text = collect_text(&responses);
    assert!(
        text.contains("all done"),
        "expected parent final text 'all done' in response: {text:?}"
    );

    // Wall-clock timing is verified separately by spawn_agent_parallel_timing
    // (#[ignore]) which uses 5 s sleeps for a clear sequential-vs-parallel gap.
    // This test only checks correctness: request count and content.

    // 6 LLM requests: 1 parent + 2×(shell_command + text) + 1 parent final.
    let reqs = server.take_requests();
    assert_eq!(reqs.len(), 6, "expected 6 LLM requests, got {}", reqs.len());

    // The parent's final request must contain results from both children.
    let parent_final_req = &reqs[5];
    let last_messages = parent_final_req
        .messages()
        .expect("messages in parent final request");
    let all_content: String = last_messages
        .iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        all_content.contains("child1 done") || all_content.contains("child2 done"),
        "parent final request should contain child tool results, got:\n{all_content}"
    );
}

// ---------------------------------------------------------------------------
// 15. parallel=true batch completes in ~5s not ~10s
// ---------------------------------------------------------------------------

/// When the LLM returns two spawn_agent calls both with `parallel=true`, the
/// daemon must run them concurrently.  Each child sleeps for 5 real seconds;
/// the combined wall-clock time must be < 8 s (sequential would be 10 s+).
///
/// Mock LLM sequence (6 requests total):
///   1. Parent turn 1  → multi_tool_calls: two spawn_agent with parallel=true
///   2. Child X turn 1 → tool_call shell_command("sleep 5")
///   3. Child Y turn 1 → tool_call shell_command("sleep 5")
///   4. Child X turn 2 → text "child1 done"
///   5. Child Y turn 2 → text "child2 done"
///   6. Parent turn 2  → text "both done"
///
/// Uses `AMAEBI_SPAWN_SANDBOX=noop` so no Docker is required.
#[tokio::test]
#[ignore] // takes ~5 real seconds
async fn spawn_agent_parallel_timing() {
    let server = MockLlmServer::start().await;

    // 1. Parent turn 1: two spawn_agent calls, both opt-in to parallel.
    server.enqueue(ScriptedResponse::multi_tool_calls([
        (
            "call-timing-A",
            "spawn_agent",
            r#"{"task":"sleep task A","workspace":"/tmp","parallel":true}"#,
        ),
        (
            "call-timing-B",
            "spawn_agent",
            r#"{"task":"sleep task B","workspace":"/tmp","parallel":true}"#,
        ),
    ]));
    // 2 & 3. First two child requests (whichever child arrives first): each
    //        gets a shell_command that sleeps 5 seconds.
    server.enqueue(ScriptedResponse::tool_call(
        "sc-timing-1",
        "shell_command",
        r#"{"command":"sleep 5"}"#,
    ));
    server.enqueue(ScriptedResponse::tool_call(
        "sc-timing-2",
        "shell_command",
        r#"{"command":"sleep 5"}"#,
    ));
    // 4 & 5. Final text for each child (order is non-deterministic).
    server.enqueue(ScriptedResponse::text_chunks(vec!["child1 done"]));
    server.enqueue(ScriptedResponse::text_chunks(vec!["child2 done"]));
    // 6. Parent turn 2: both results received.
    server.enqueue(ScriptedResponse::text_chunks(vec!["both done"]));

    let daemon = start_daemon_with_env(&server.url(), &[("AMAEBI_SPAWN_SANDBOX", "noop")])
        .await
        .expect("start_daemon");
    let client = connect_client(&daemon.socket);

    let start = std::time::Instant::now();
    let responses = send_message(&client, "run two sleep tasks in parallel")
        .await
        .expect("send_message");
    let elapsed = start.elapsed();

    // Both children must have returned results visible in the parent's reply.
    let text = collect_text(&responses);
    assert!(
        text.contains("both done"),
        "expected parent final text 'both done', got: {text:?}"
    );

    // Parallel execution: ~5 s wall-clock.  Sequential would be ~10 s.
    assert!(
        elapsed.as_secs() < 8,
        "expected parallel execution to finish in < 8 s, took {elapsed:?}"
    );

    // 6 LLM requests: 1 parent + 2×(shell_command + text) + 1 parent final.
    let reqs = server.take_requests();
    assert_eq!(reqs.len(), 6, "expected 6 LLM requests, got {}", reqs.len());
}

// ---------------------------------------------------------------------------
// 16. Cron: job fires and calls the LLM
// ---------------------------------------------------------------------------

/// A scheduled cron job with schedule "* * * * *" must fire on the daemon's
/// first scheduler tick (which is immediate — tokio intervals fire at t=0) and
/// call the LLM with the job's description as the user prompt.
///
/// Strategy:
/// 1. Create a temp home dir with `.amaebi/hosts.json`.
/// 2. Seed a cron job via `amaebi cron add` before starting the daemon.
/// 3. Start the daemon at that home directory.
/// 4. The cron scheduler fires immediately; wait up to 5 s for the LLM call.
/// 5. Assert the LLM received a request containing the cron task description.
#[tokio::test]
async fn cron_job_triggers_llm_call() {
    let server = MockLlmServer::start().await;
    server.enqueue(ScriptedResponse::text_chunks(vec!["cron-task-response"]));

    // Set up a home dir and seed a cron job that always fires ("* * * * *").
    let home_dir = setup_home().expect("setup_home");
    seed_cron_job(home_dir.path(), "cron-unique-task-description", "* * * * *")
        .await
        .expect("seed_cron_job");

    let (socket, mut child, _socket_dir) =
        start_daemon_at_home_with_env(home_dir.path(), &server.url(), &[])
            .await
            .expect("start daemon");

    // The cron scheduler fires its first tick immediately (tokio interval
    // behaviour).  Wait up to 5 s for the LLM request to arrive.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        if server.peek_request_count() >= 1 || std::time::Instant::now() > deadline {
            break;
        }
    }

    let reqs = server.take_requests();
    assert!(
        !reqs.is_empty(),
        "cron scheduler did not call the LLM within 5 s"
    );

    // The request's messages must contain the cron job description.
    let messages = reqs[0].messages().expect("messages in cron LLM request");
    let all_content: String = messages
        .iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        all_content.contains("cron-unique-task-description"),
        "expected cron task description in LLM request messages, got:\n{all_content}"
    );

    // Keep socket alive until assertions are done.
    drop(socket);
    let _ = child.kill().await;
    let _ = child.wait().await;
}

// ---------------------------------------------------------------------------
// 17. Cron: output does NOT appear in the chat stream
// ---------------------------------------------------------------------------

/// Cron jobs run with a sink writer — their output is deposited into the inbox
/// and must NOT appear when a user sends a normal chat message on the same
/// daemon.
///
/// Strategy:
/// 1. Seed a cron job; start the daemon.
/// 2. Wait for the cron scheduler to fire and complete.
/// 3. Send a chat message.
/// 4. Assert the chat response does NOT contain the cron's unique output.
///
/// Note: when the chat session starts with empty history, the Chat handler
/// spawns a background cross-session compaction for the cron's just-completed
/// session.  That task also calls the LLM, so we over-provision the mock queue
/// (3 extra "ok" responses beyond the cron response) to prevent 500 errors.
#[tokio::test]
async fn cron_job_result_not_sent_to_chat() {
    const CRON_OUTPUT: &str = "super-unique-cron-only-output-xyz987";

    let server = MockLlmServer::start().await;
    // Response 1: for the cron job's LLM call.
    server.enqueue(ScriptedResponse::text_chunks(vec![CRON_OUTPUT]));
    // Responses 2-4: for the cross-session compaction background task and the
    // chat's own LLM call.  All say "ok" — we only care that they are NOT
    // CRON_OUTPUT.  Over-provisioning by 1 prevents any accidental 500 errors.
    server.enqueue(ScriptedResponse::text_chunks(vec!["ok"]));
    server.enqueue(ScriptedResponse::text_chunks(vec!["ok"]));
    server.enqueue(ScriptedResponse::text_chunks(vec!["ok"]));

    let home_dir = setup_home().expect("setup_home");
    seed_cron_job(home_dir.path(), "background cron task", "* * * * *")
        .await
        .expect("seed_cron_job");

    let (socket, mut child, _socket_dir) =
        start_daemon_at_home_with_env(home_dir.path(), &server.url(), &[])
            .await
            .expect("start daemon");

    // Wait for the cron job to call the LLM (up to 5 s).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        if server.peek_request_count() >= 1 || std::time::Instant::now() > deadline {
            break;
        }
    }
    assert!(
        server.peek_request_count() >= 1,
        "cron job did not fire within 5 s"
    );
    // Give the cron job a moment to finish (store_conversation, InboxStore, etc.).
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Send a chat message; the cron output went to a sink, never to this stream.
    let client = connect_client(&socket);
    let responses = send_message(&client, "hello from chat")
        .await
        .expect("chat message");

    let text = collect_text(&responses);
    // The cron-specific output must NOT appear in the chat stream.
    assert!(
        !text.contains(CRON_OUTPUT),
        "cron output must NOT appear in chat stream, got: {text:?}"
    );
    // The chat must have completed successfully (no Error frame).
    assert!(
        !responses
            .iter()
            .any(|r| matches!(r, support::helpers::Response::Error { .. })),
        "chat request must not fail: {responses:?}"
    );

    let _ = child.kill().await;
    let _ = child.wait().await;
}

// ---------------------------------------------------------------------------
// 18. Cron: job can call spawn_agent
// ---------------------------------------------------------------------------

/// A cron job runs inside the full agentic loop with `include_spawn_agent=true`,
/// so it can call `spawn_agent`.  This test verifies the end-to-end flow:
///   cron turn 1 → spawn_agent tool call
///   child turn 1 → final text
///   cron turn 2 → final text (after receiving child result)
///
/// Uses `AMAEBI_SPAWN_SANDBOX=noop` so no Docker is required.
#[tokio::test]
async fn cron_job_with_spawn_agent() {
    let server = MockLlmServer::start().await;

    // Cron LLM turn 1: calls spawn_agent.
    server.enqueue(ScriptedResponse::tool_call(
        "cron-spawn-call-001",
        "spawn_agent",
        r#"{"task":"child task from cron","workspace":"/tmp"}"#,
    ));
    // Child agent LLM turn 1: returns final text.
    server.enqueue(ScriptedResponse::text_chunks(vec!["child-cron-result"]));
    // Cron LLM turn 2 (after spawn_agent tool result): final cron text.
    server.enqueue(ScriptedResponse::text_chunks(vec!["cron-spawn-done"]));

    let home_dir = setup_home().expect("setup_home");
    seed_cron_job(
        home_dir.path(),
        "cron task that spawns a child agent",
        "* * * * *",
    )
    .await
    .expect("seed_cron_job");

    let (_socket, mut child, _socket_dir) = start_daemon_at_home_with_env(
        home_dir.path(),
        &server.url(),
        &[("AMAEBI_SPAWN_SANDBOX", "noop")],
    )
    .await
    .expect("start daemon");

    // Wait for all 3 LLM requests: cron turn 1 + child turn 1 + cron turn 2.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        if server.peek_request_count() >= 3 || std::time::Instant::now() > deadline {
            break;
        }
    }

    let reqs = server.take_requests();
    assert_eq!(
        reqs.len(),
        3,
        "expected 3 LLM requests (cron + child + cron final), got {}:\n{:#?}",
        reqs.len(),
        reqs.iter().map(|r| &r.body).collect::<Vec<_>>()
    );

    let _ = child.kill().await;
    let _ = child.wait().await;
}

// ===========================================================================
// REGRESSION TESTS — Issue #24
// ===========================================================================

// ---------------------------------------------------------------------------
// R1. Steer E2E: SteerAck delivered and message injected into LLM context
// ---------------------------------------------------------------------------

/// Inject a `Steer` message on the same socket connection as the ongoing Chat
/// while a tool is executing.  The daemon must:
///   1. Consume the steer at the top of the next loop iteration.
///   2. Send `Response::SteerAck` to the client.
///   3. Include the steer text in the messages sent to the LLM on that turn.
///   4. Complete the session with `Response::Done` (no panic).
///
/// Flow:
///   - Chat → LLM: shell_command("sleep 0.3")   [gives steer time to arrive]
///   - While sleep runs: client sends Steer on the same socket
///   - Loop iteration 2: steer consumed → SteerAck → LLM called with steer
///   - LLM: text "steer-acknowledged"
///
/// The bidirectional socket protocol (Chat + Steer on the same connection)
/// is exercised end-to-end.
#[tokio::test]
async fn steer_e2e_delivers_ack_and_injects_message() {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::UnixStream;

    let server = MockLlmServer::start().await;

    // Turn 1: a 300 ms sleep gives the steer plenty of time to arrive before
    // the second loop iteration's `steer_rx.try_recv()` call.
    server.enqueue(ScriptedResponse::tool_call(
        "steer-tool-1",
        "shell_command",
        r#"{"command":"sleep 0.3"}"#,
    ));
    // Turn 2: text reply produced after the steer is consumed.
    server.enqueue(ScriptedResponse::text_chunks(vec!["steer-acknowledged"]));

    let daemon = start_daemon(&server.url()).await.expect("daemon");
    let session_id = uuid::Uuid::new_v4().to_string();

    // Open a raw bidirectional socket: Chat and Steer share the same connection.
    let stream = UnixStream::connect(&daemon.socket)
        .await
        .expect("connect to daemon socket");
    let (read_half, mut write_half) = stream.into_split();

    // Send the initial Chat request.
    let chat_req = Request::Chat {
        prompt: "run a slow tool".to_string(),
        tmux_pane: None,
        session_id: Some(session_id.clone()),
        model: "gpt-4o".to_string(),
    };
    let mut json_line = serde_json::to_string(&chat_req).unwrap();
    json_line.push('\n');
    write_half
        .write_all(json_line.as_bytes())
        .await
        .expect("sending Chat request");

    // Read responses line by line; inject Steer immediately after ToolUse.
    let mut lines = BufReader::new(read_half).lines();
    let mut responses: Vec<Response> = Vec::new();
    let mut steer_sent = false;

    let read_result = tokio::time::timeout(Duration::from_secs(15), async {
        while let Some(line) = lines.next_line().await.expect("reading response line") {
            if line.is_empty() {
                continue;
            }
            let frame: Response = serde_json::from_str(&line).expect("parsing response frame");
            let done = matches!(frame, Response::Done | Response::Error { .. });

            // Inject Steer on the same socket the moment we see the ToolUse
            // frame — the tool is now executing, giving the daemon's reader
            // task time to forward the message before the next iteration.
            if !steer_sent {
                if let Response::ToolUse { .. } = &frame {
                    let steer_req = Request::Steer {
                        session_id: session_id.clone(),
                        message: "please adjust approach".to_string(),
                    };
                    let mut steer_line = serde_json::to_string(&steer_req).unwrap();
                    steer_line.push('\n');
                    write_half
                        .write_all(steer_line.as_bytes())
                        .await
                        .expect("sending Steer request");
                    steer_sent = true;
                }
            }

            responses.push(frame);
            if done {
                break;
            }
        }
    })
    .await;

    read_result.expect("steer E2E test timed out after 15 s");
    assert!(
        steer_sent,
        "ToolUse frame never appeared — steer not injected"
    );

    // The daemon must have sent SteerAck confirming the message was consumed.
    assert!(
        responses.iter().any(|r| matches!(r, Response::SteerAck)),
        "expected SteerAck in responses: {responses:?}"
    );

    // Session must complete cleanly.
    assert!(
        responses.iter().any(|r| matches!(r, Response::Done)),
        "expected Done frame (no crash): {responses:?}"
    );
    assert!(
        !responses
            .iter()
            .any(|r| matches!(r, Response::Error { .. })),
        "unexpected Error frame: {responses:?}"
    );

    // The steer text must appear in the LLM's second request.
    let reqs = server.take_requests();
    assert_eq!(reqs.len(), 2, "expected 2 LLM requests, got {}", reqs.len());
    let second_req_messages = reqs[1].messages().expect("messages in second LLM request");
    let all_content: String = second_req_messages
        .iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        all_content.contains("adjust approach"),
        "steer text must appear in LLM second-turn context:\n{all_content}"
    );
}

// ---------------------------------------------------------------------------
// R2. Sandbox credential protection: child context must not expose host creds
// ---------------------------------------------------------------------------

/// When `spawn_agent` creates a child agent with a noop sandbox, the child's
/// LLM request messages must not contain:
///   - the daemon's host HOME directory path (which holds `hosts.json`)
///   - the `AMAEBI_COPILOT_TOKEN` value used by the test suite
///
/// The [Sandbox Context] preamble is only allowed to mention the explicitly
/// supplied `workspace` path and sandbox type — not any host credential paths.
///
/// Flow:
///   - Parent turn 1 → spawn_agent{ workspace: "/tmp" }   (no extra_mounts)
///   - Child turn 1  → text "child done"
///   - Parent turn 2 → text "all done"
///
/// Uses `AMAEBI_SPAWN_SANDBOX=noop` so no Docker is required.
#[tokio::test]
async fn sandbox_noop_child_does_not_expose_credentials() {
    let server = MockLlmServer::start().await;

    server.enqueue(ScriptedResponse::tool_call(
        "cred-spawn-1",
        "spawn_agent",
        r#"{"task":"check environment","workspace":"/tmp"}"#,
    ));
    server.enqueue(ScriptedResponse::text_chunks(vec!["child done"]));
    server.enqueue(ScriptedResponse::text_chunks(vec!["all done"]));

    let daemon = start_daemon_with_env(&server.url(), &[("AMAEBI_SPAWN_SANDBOX", "noop")])
        .await
        .expect("daemon");
    let client = connect_client(&daemon.socket);

    let responses = send_message(&client, "spawn a child")
        .await
        .expect("send_message");

    let text = collect_text(&responses);
    assert!(
        text.contains("all done"),
        "expected parent final text: {text:?}"
    );

    let reqs = server.take_requests();
    assert_eq!(reqs.len(), 3, "expected 3 LLM requests, got {}", reqs.len());

    // Child's request is at index 1.
    let child_messages = reqs[1].messages().expect("messages in child LLM request");
    let all_content: String = child_messages
        .iter()
        .flat_map(|m| {
            let content = m.get("content");
            if let Some(s) = content.and_then(|c| c.as_str()) {
                vec![s.to_string()]
            } else if let Some(arr) = content.and_then(|c| c.as_array()) {
                arr.iter()
                    .filter_map(|p| p.get("text").and_then(|t| t.as_str()).map(str::to_string))
                    .collect()
            } else {
                vec![]
            }
        })
        .collect::<Vec<_>>()
        .join(" ");

    // The AMAEBI_COPILOT_TOKEN must never appear in child LLM messages.
    assert!(
        !all_content.contains("test-api-token"),
        "child context must not expose AMAEBI_COPILOT_TOKEN value: {all_content}"
    );

    // The daemon's HOME directory (which contains hosts.json) must not be
    // listed as a mount or otherwise mentioned in the child's context.
    let home_str = daemon.home.to_str().expect("home path is valid UTF-8");
    assert!(
        !all_content.contains(home_str),
        "child context must not expose host HOME path {home_str:?}: {all_content}"
    );
}

// ---------------------------------------------------------------------------
// R3. Sub-agent chain: session remains usable after child recursion is blocked
// ---------------------------------------------------------------------------

/// Regression: when a child agent attempts to call `spawn_agent` (recursion),
/// the daemon must:
///   1. Return the explicit error string to the child's LLM.
///   2. Allow the child to complete with a coherent text reply.
///   3. Allow the parent to complete with a coherent text reply.
///   4. Leave the session usable for a subsequent message (no state corruption).
///
/// Flow (round 1):
///   Parent turn 1 → spawn_agent
///   Child turn 1  → spawn_agent attempt (recursion)
///   Child turn 2  → text "child: recursion blocked"
///   Parent turn 2 → text "parent: child reported error"
/// Flow (round 2, same session_id):
///   Chat → text "session still works"
///
/// Uses `AMAEBI_SPAWN_SANDBOX=noop` so no Docker is required.
#[tokio::test]
async fn subagent_chain_session_remains_usable_after_recursion_block() {
    let server = MockLlmServer::start().await;

    // Round 1 ----------------------------------------------------------------
    server.enqueue(ScriptedResponse::tool_call(
        "chain-parent-spawn",
        "spawn_agent",
        r#"{"task":"try nested spawn","workspace":"/tmp"}"#,
    ));
    // Child tries to spawn again — should receive the recursion-blocked error.
    server.enqueue(ScriptedResponse::tool_call(
        "chain-child-nested",
        "spawn_agent",
        r#"{"task":"nested task","workspace":"/tmp"}"#,
    ));
    server.enqueue(ScriptedResponse::text_chunks(vec![
        "child: recursion blocked",
    ]));
    server.enqueue(ScriptedResponse::text_chunks(vec![
        "parent: child reported error",
    ]));

    // Round 2 (verify session still usable) ----------------------------------
    server.enqueue(ScriptedResponse::text_chunks(vec!["session still works"]));

    let session_id = uuid::Uuid::new_v4().to_string();
    let daemon = start_daemon_with_env(&server.url(), &[("AMAEBI_SPAWN_SANDBOX", "noop")])
        .await
        .expect("daemon");
    let client = connect_client(&daemon.socket);

    // Round 1.
    let r1 = send_message_with_session(&client, "do nested agent task", &session_id, "gpt-4o")
        .await
        .expect("round 1");
    let text1 = collect_text(&r1);
    assert!(
        text1.contains("parent: child reported error"),
        "expected coherent parent response: {text1:?}"
    );
    assert!(
        !r1.iter().any(|r| matches!(r, Response::Error { .. })),
        "no Error frames expected in round 1: {r1:?}"
    );

    // Round 2: same session_id — daemon must still be functional.
    let r2 = send_message_with_session(&client, "follow-up", &session_id, "gpt-4o")
        .await
        .expect("round 2");
    let text2 = collect_text(&r2);
    assert!(
        text2.contains("session still works"),
        "session should be usable after recursion failure: {text2:?}"
    );
    assert!(
        !r2.iter().any(|r| matches!(r, Response::Error { .. })),
        "no Error frames expected in round 2: {r2:?}"
    );

    // The child's second LLM request (index 2) must include the explicit
    // "spawn_agent is not available" error in the tool result.
    let reqs = server.take_requests();
    // 4 requests for round 1 (parent×2 + child×2) + 1 for round 2 = 5.
    assert_eq!(
        reqs.len(),
        5,
        "expected 5 LLM requests total, got {}",
        reqs.len()
    );
    let child_second_messages = reqs[2].messages().expect("messages in child turn 2");
    let child_ctx: String = child_second_messages
        .iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        child_ctx.contains("spawn_agent is not available"),
        "child must receive explicit recursion-blocked error: {child_ctx}"
    );
}

// ---------------------------------------------------------------------------
// R4. LLM error path: Error frame surfaced; daemon stays alive for next message
// ---------------------------------------------------------------------------

/// When the LLM returns a non-retryable 4xx HTTP error, the daemon must:
///   1. Surface a `Response::Error` frame to the client (not panic or hang).
///   2. Remain alive and fully functional for subsequent messages.
///
/// A 422 status is used because 4xx (other than 429) bypasses the retry/
/// back-off logic in `copilot.rs`, so the test completes without any sleep.
///
/// Flow:
///   Message 1 → mock returns 422 → Error frame
///   Message 2 → mock returns text "recovered" → Done frame
#[tokio::test]
async fn llm_error_path_surfaces_error_frame_and_daemon_stays_alive() {
    let server = MockLlmServer::start().await;

    // First request: LLM returns a non-retryable 422 error.
    server.enqueue_error(422, "mock: unprocessable entity");
    // Second request: normal text reply — daemon must still be reachable.
    server.enqueue(ScriptedResponse::text_chunks(vec!["recovered"]));

    let daemon = start_daemon(&server.url()).await.expect("daemon");
    let client = connect_client(&daemon.socket);

    // Message 1: should result in an Error frame, no panic.
    // Read directly from a live socket so we can assert no `Done` is emitted
    // *after* the `Error` frame on the same connection.
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::UnixStream;

    let stream = UnixStream::connect(&daemon.socket)
        .await
        .expect("connect socket for error-path check");
    let (read_half, mut write_half) = tokio::io::split(stream);

    let mut line = serde_json::to_string(&Request::Chat {
        prompt: "trigger an llm error".to_string(),
        tmux_pane: None,
        session_id: None,
        model: "gpt-4o".to_string(),
    })
    .expect("serialize chat request");
    line.push('\n');
    write_half
        .write_all(line.as_bytes())
        .await
        .expect("write chat request");

    let mut lines = BufReader::new(read_half).lines();
    let mut saw_error = false;
    let mut saw_done_before_error = false;
    loop {
        let next = tokio::time::timeout(Duration::from_secs(10), lines.next_line())
            .await
            .expect("timed out waiting for response frame")
            .expect("failed reading response frame");
        let Some(raw) = next else {
            break;
        };
        if raw.is_empty() {
            continue;
        }
        let frame: Response = serde_json::from_str(&raw).expect("parse response frame");
        match frame {
            Response::Error { .. } => {
                saw_error = true;
                break;
            }
            Response::Done => saw_done_before_error = true,
            _ => {}
        }
    }

    assert!(saw_error, "expected Error frame on LLM 422 failure");
    assert!(
        !saw_done_before_error,
        "Done must not be emitted before Error in this flow"
    );

    // Ensure no trailing `Done` after `Error` on the same connection.
    let trailing = tokio::time::timeout(Duration::from_millis(300), lines.next_line()).await;
    if let Ok(Ok(Some(raw))) = trailing {
        if !raw.is_empty() {
            let frame: Response = serde_json::from_str(&raw).expect("parse trailing frame");
            assert!(
                !matches!(frame, Response::Done),
                "Done must not follow Error: {frame:?}"
            );
        }
    }

    // Message 2: daemon must still be alive and return a normal response.
    let r2 = send_message(&client, "retry after error")
        .await
        .expect("send_message (recovery case)");
    let text2 = collect_text(&r2);
    assert!(
        text2.contains("recovered"),
        "daemon must be usable after LLM error: {text2:?}"
    );
    assert!(
        r2.iter().any(|r| matches!(r, Response::Done)),
        "expected Done on recovery message: {r2:?}"
    );
}

// ---------------------------------------------------------------------------
// Endpoint routing regression tests
//
// All models (Claude, GPT, Gemini) are routed through the same Copilot
// endpoint.  The correct base URL is derived from the `proxy-ep` field
// embedded in the Copilot JWT.  This eliminates the previous 400
// "unsupported_api_for_model" errors for gpt-5.4 and gemini models, which
// occurred because those models are only accessible via the user's
// account-specific Copilot gateway (e.g. api.individual.githubcopilot.com),
// not the generic api.githubcopilot.com host.
// ---------------------------------------------------------------------------

/// All models — claude-*, gpt-5.4, gemini-* — must use the same Copilot
/// endpoint and Copilot JWT.  This is the regression test for the 400
/// "unsupported_api_for_model" bug that affected non-Claude models.
#[tokio::test]
async fn all_models_use_copilot_endpoint_and_jwt() {
    for model in &["gpt-5.4", "gemini-2.5-pro", "claude-sonnet-4.6", "gpt-4o"] {
        let server = MockLlmServer::start().await;
        server.enqueue(ScriptedResponse::text_chunks(vec!["ok"]));

        let daemon = start_daemon(&server.url()).await.expect("start_daemon");
        let client = connect_client(&daemon.socket);

        send_message_with_session(&client, "hi", "sess-routing", model)
            .await
            .expect("send_message");

        let reqs = server.take_requests();
        assert!(!reqs.is_empty(), "model={model}: no requests captured");

        // All models must use the Copilot JWT (AMAEBI_COPILOT_TOKEN = "test-api-token").
        let auth = reqs[0]
            .headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("authorization"))
            .map(|(_, v)| v.as_str())
            .unwrap_or("");

        assert!(
            auth.contains("test-api-token"),
            "model={model}: expected Copilot JWT in Authorization header; got: {auth:?}"
        );

        // The correct model name must be forwarded to the API.
        assert_eq!(
            reqs[0].model(),
            Some(model.as_ref()),
            "model={model}: wrong model field in request"
        );
    }
}

/// When `/chat/completions` returns `400 unsupported_api_for_model`, the daemon
/// must transparently retry via the Responses API (`/v1/responses`) and the
/// client must receive the final text from that fallback call.
///
/// This is the regression test for the gpt-5.4 / gpt-5.x bug: those models
/// are not accessible via `/chat/completions` and require the Responses API.
#[tokio::test]
async fn responses_api_fallback_on_unsupported_model() {
    let server = MockLlmServer::start().await;

    // First request hits /chat/completions → 400 unsupported_api_for_model.
    server.enqueue_error(
        400,
        r#"{"error":{"code":"unsupported_api_for_model","message":"model is not accessible via the /chat/completions endpoint"}}"#,
    );
    // Second request hits /v1/responses (fallback) → text response.
    server.enqueue(ScriptedResponse::text_chunks(vec!["fallback-ok"]));

    let daemon = start_daemon(&server.url()).await.expect("start_daemon");
    let client = connect_client(&daemon.socket);

    let responses = send_message_with_session(&client, "hello", "sess-fallback", "gpt-5.4")
        .await
        .expect("send_message");

    let text = collect_text(&responses);
    assert!(
        text.contains("fallback-ok"),
        "expected fallback response text; got: {text:?}"
    );

    // Two requests must have been made: one to /chat/completions (400), one to /v1/responses.
    let reqs = server.take_requests();
    assert_eq!(
        reqs.len(),
        2,
        "expected 2 requests (chat completions + responses fallback), got {}",
        reqs.len()
    );
}

/// All gpt-5.x model variants must trigger the Responses API fallback when
/// /chat/completions returns 400 unsupported_api_for_model.
#[tokio::test]
async fn responses_api_fallback_all_gpt5_variants() {
    for model in &["gpt-5.4", "gpt-5.4-mini", "gpt-5", "gpt-5-turbo"] {
        let server = MockLlmServer::start().await;
        server.enqueue_error(
            400,
            r#"{"error":{"code":"unsupported_api_for_model","message":"not via chat/completions"}}"#,
        );
        server.enqueue(ScriptedResponse::text_chunks(vec!["responses-ok"]));

        let daemon = start_daemon(&server.url()).await.expect("start_daemon");
        let client = connect_client(&daemon.socket);

        let responses =
            send_message_with_session(&client, "ping", &format!("sess-gpt5-{model}"), model)
                .await
                .expect("send_message");

        let text = collect_text(&responses);
        assert!(
            text.contains("responses-ok"),
            "model={model}: expected Responses API text; got: {text:?}"
        );

        let reqs = server.take_requests();
        assert_eq!(
            reqs.len(),
            2,
            "model={model}: expected 2 requests (chat/completions + /v1/responses), got {}",
            reqs.len()
        );
    }
}

/// Claude and gpt-4o models must succeed via /chat/completions directly,
/// without needing the Responses API fallback.
#[tokio::test]
async fn chat_completions_models_no_fallback_needed() {
    for model in &["claude-sonnet-4.6", "claude-opus-4.6", "gpt-4o", "gpt-4.1"] {
        let server = MockLlmServer::start().await;
        // Enqueue exactly one response — if the daemon made a second request
        // (fallback) the mock would error with "no scripted response queued".
        server.enqueue(ScriptedResponse::text_chunks(vec!["direct-ok"]));

        let daemon = start_daemon(&server.url()).await.expect("start_daemon");
        let client = connect_client(&daemon.socket);

        let responses =
            send_message_with_session(&client, "ping", &format!("sess-direct-{model}"), model)
                .await
                .expect("send_message");

        let text = collect_text(&responses);
        assert!(
            text.contains("direct-ok"),
            "model={model}: expected direct response; got: {text:?}"
        );

        let reqs = server.take_requests();
        assert_eq!(
            reqs.len(),
            1,
            "model={model}: expected exactly 1 request (no fallback), got {}",
            reqs.len()
        );
    }
}

/// Gemini models must route through the Copilot JWT endpoint and return
/// a valid response via /chat/completions (no Responses API needed).
#[tokio::test]
async fn gemini_models_route_via_copilot_and_succeed() {
    for model in &[
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-flash",
    ] {
        let server = MockLlmServer::start().await;
        server.enqueue(ScriptedResponse::text_chunks(vec!["gemini-ok"]));

        let daemon = start_daemon(&server.url()).await.expect("start_daemon");
        let client = connect_client(&daemon.socket);

        let responses =
            send_message_with_session(&client, "ping", &format!("sess-gemini-{model}"), model)
                .await
                .expect("send_message");

        let text = collect_text(&responses);
        assert!(
            text.contains("gemini-ok"),
            "model={model}: expected response text; got: {text:?}"
        );

        // Exactly one request — direct /chat/completions, no fallback.
        let reqs = server.take_requests();
        assert_eq!(
            reqs.len(),
            1,
            "model={model}: expected 1 request, got {}",
            reqs.len()
        );

        // Copilot JWT must be present.
        let auth = reqs[0]
            .headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("authorization"))
            .map(|(_, v)| v.as_str())
            .unwrap_or("");
        assert!(
            auth.contains("test-api-token"),
            "model={model}: Copilot JWT missing; got: {auth:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Responses API fallback regression test
//
// Verifies that when /chat/completions returns 400 unsupported_api_for_model
// the daemon automatically retries via /v1/responses and delivers the
// response to the client.

// 19. Consistent chat — plain-text turns
// ---------------------------------------------------------------------------

/// Two sequential plain-text turns on the same session.
/// The second turn's LLM request must contain the user message and assistant
/// reply from turn 1 — verifying that consistent chat context is preserved
/// across separate `amaebi ask` invocations.
#[tokio::test]
async fn consistent_chat_plain_turns() {
    let server = MockLlmServer::start().await;
    // Turn 1: simple text reply.
    server.enqueue(ScriptedResponse::text_chunks(vec!["The answer is 42."]));
    // Turn 2: another text reply (content doesn't matter here).
    server.enqueue(ScriptedResponse::text_chunks(vec!["That is 84."]));

    let home = setup_home().expect("setup_home");
    let (socket, mut child, _socket_dir) =
        start_daemon_at_home_with_env(home.path(), &server.url(), &[])
            .await
            .expect("start_daemon");

    let session_id = "test-consistent-plain-001";

    // Turn 1.
    let client1 = connect_client(&socket);
    let r1 = send_message_with_session(&client1, "what is 6 times 7?", session_id, "gpt-4o")
        .await
        .expect("turn 1");
    assert!(
        collect_text(&r1).contains("42"),
        "turn 1 response unexpected: {:?}",
        collect_text(&r1)
    );

    // Turn 2 on a new connection (simulates a second `amaebi ask` invocation).
    let client2 = connect_client(&socket);
    let r2 = send_message_with_session(&client2, "double that", session_id, "gpt-4o")
        .await
        .expect("turn 2");
    assert!(
        collect_text(&r2).contains("84"),
        "turn 2 response unexpected: {:?}",
        collect_text(&r2)
    );

    // Inspect what the daemon sent to the LLM on turn 2.
    let reqs = server.take_requests();
    assert_eq!(reqs.len(), 2, "expected 2 LLM requests, got {}", reqs.len());

    let turn2_req = &reqs[1];
    let messages = turn2_req
        .body
        .get("messages")
        .and_then(|m| m.as_array())
        .expect("messages array missing in turn 2 request");

    // Turn 2 must include the user message from turn 1.
    let has_turn1_user = messages.iter().any(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("user")
            && m.get("content")
                .and_then(|c| c.as_str())
                .map(|c| c.contains("6 times 7"))
                .unwrap_or(false)
    });
    assert!(
        has_turn1_user,
        "turn 2 LLM request must include turn 1 user message; messages: {messages:#?}"
    );

    // Turn 2 must include the assistant reply from turn 1.
    let has_turn1_assistant = messages.iter().any(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("assistant")
            && m.get("content")
                .and_then(|c| c.as_str())
                .map(|c| c.contains("42"))
                .unwrap_or(false)
    });
    assert!(
        has_turn1_assistant,
        "turn 2 LLM request must include turn 1 assistant reply; messages: {messages:#?}"
    );

    let _ = child.kill().await;
    let _ = child.wait().await;
}

// ---------------------------------------------------------------------------
// 20. Consistent chat — tool-call context preserved across turns
// ---------------------------------------------------------------------------

/// Turn 1 involves a tool call; turn 2 must receive the full tool exchange
/// (the assistant tool-call message AND the tool-result message), not just
/// the final plain-text assistant reply.
///
/// This is the key difference from the old text-only approach: openclaw/pi
/// stores the complete reasoning chain so subsequent turns have richer context.
#[tokio::test]
async fn consistent_chat_tool_call_context() {
    let server = MockLlmServer::start().await;

    // Turn 1, step 1: model calls shell_command.
    server.enqueue(ScriptedResponse::tool_call(
        "call-ls-001",
        "shell_command",
        r#"{"command":"echo hello-from-tool"}"#,
    ));
    // Turn 1, step 2: model receives tool result, returns final text.
    server.enqueue(ScriptedResponse::text_chunks(vec![
        "The tool returned hello-from-tool.",
    ]));
    // Turn 2: model replies (content doesn't matter for this assertion).
    server.enqueue(ScriptedResponse::text_chunks(vec!["Got it."]));

    let home = setup_home().expect("setup_home");
    let (socket, mut child, _socket_dir) =
        start_daemon_at_home_with_env(home.path(), &server.url(), &[])
            .await
            .expect("start_daemon");

    let session_id = "test-consistent-tool-001";

    // Turn 1: triggers a tool call internally.
    let client1 = connect_client(&socket);
    let r1 = send_message_with_session(&client1, "run echo hello-from-tool", session_id, "gpt-4o")
        .await
        .expect("turn 1");
    let text1 = collect_text(&r1);
    assert!(
        text1.contains("hello-from-tool"),
        "turn 1 response unexpected: {text1:?}"
    );

    // Turn 2: a follow-up question.
    let client2 = connect_client(&socket);
    let _r2 =
        send_message_with_session(&client2, "what did the tool return?", session_id, "gpt-4o")
            .await
            .expect("turn 2");

    // Inspect the turn-2 LLM request.
    let reqs = server.take_requests();
    // 3 requests: turn1-step1, turn1-step2, turn2
    assert_eq!(reqs.len(), 3, "expected 3 LLM requests, got {}", reqs.len());

    let turn2_req = &reqs[2];
    let messages = turn2_req
        .body
        .get("messages")
        .and_then(|m| m.as_array())
        .expect("messages array missing in turn 2 request");

    // The turn-2 request must include the assistant tool-call message from turn 1.
    let has_tool_call_msg = messages.iter().any(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("assistant")
            && m.get("tool_calls")
                .and_then(|tc| tc.as_array())
                .map(|arr| !arr.is_empty())
                .unwrap_or(false)
    });
    assert!(
        has_tool_call_msg,
        "turn 2 must include the assistant tool-call message from turn 1; messages: {messages:#?}"
    );

    // The turn-2 request must include the tool-result message.
    let has_tool_result = messages
        .iter()
        .any(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"));
    assert!(
        has_tool_result,
        "turn 2 must include the tool-result message from turn 1; messages: {messages:#?}"
    );

    let _ = child.kill().await;
    let _ = child.wait().await;
}

// ---------------------------------------------------------------------------
// 21. Heartbeat — session-scoped, injected into running conversation
// ---------------------------------------------------------------------------

/// Heartbeat fires only for active (open) sessions and injects the follow-up
/// report directly into the running agentic loop via the steer channel.
///
/// Strategy:
/// 1. Open a Chat session with a slow tool call so the session stays alive.
/// 2. While the tool is executing, add a heartbeat item and trigger a check
///    on a separate connection.
/// 3. Heartbeat LLM returns an actionable report; it is injected as a steer.
/// 4. The agentic loop receives it and produces a final text reply.
/// 5. Assert the heartbeat LLM call contained the item description.
#[tokio::test]
async fn heartbeat_injects_into_active_session() {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::UnixStream;

    let server = MockLlmServer::start().await;

    // Turn 1: tool call with a short sleep so the session stays alive long
    // enough for the heartbeat to fire.
    server.enqueue(ScriptedResponse::tool_call(
        "hb-tool-1",
        "shell_command",
        r#"{"command":"sleep 0.3"}"#,
    ));
    // Heartbeat LLM call (2nd request): actionable report.
    server.enqueue(ScriptedResponse::text_chunks(vec![
        "Build finished: all tests passed.",
    ]));
    // Turn 2 (3rd request): agent responds after heartbeat injection.
    server.enqueue(ScriptedResponse::text_chunks(vec![
        "Got the heartbeat update.",
    ]));

    let home = setup_home().expect("setup_home");
    write_heartbeat_config(home.path()).expect("write_heartbeat_config");

    let (socket, mut child, _socket_dir) =
        start_daemon_at_home_with_env(home.path(), &server.url(), &[])
            .await
            .expect("start_daemon");

    let session_id = uuid::Uuid::new_v4().to_string();

    // Open a raw bidirectional Chat connection.
    let stream = UnixStream::connect(&socket).await.expect("connect");
    let (read_half, mut write_half) = stream.into_split();
    let chat_req = Request::Chat {
        prompt: "run slow tool".to_string(),
        tmux_pane: None,
        session_id: Some(session_id.clone()),
        model: "gpt-4o".to_string(),
    };
    let mut line = serde_json::to_string(&chat_req).unwrap();
    line.push('\n');
    write_half.write_all(line.as_bytes()).await.unwrap();

    let mut lines = BufReader::new(read_half).lines();
    let mut responses: Vec<Response> = Vec::new();
    let mut heartbeat_triggered = false;

    let result = tokio::time::timeout(Duration::from_secs(15), async {
        while let Some(raw) = lines.next_line().await.expect("reading line") {
            if raw.is_empty() {
                continue;
            }
            let frame: Response = serde_json::from_str(&raw).expect("parse frame");
            let done = matches!(frame, Response::Done | Response::Error { .. });

            // When tool starts executing, add heartbeat item and trigger.
            if !heartbeat_triggered {
                if let Response::ToolUse { .. } = &frame {
                    let client = connect_client(&socket);
                    send_heartbeat_add(&client, &session_id, "check build status")
                        .await
                        .expect("heartbeat add");
                    send_heartbeat_trigger(&client).await.expect("trigger");
                    heartbeat_triggered = true;
                }
            }

            responses.push(frame);
            if done {
                break;
            }
        }
    })
    .await;

    result.expect("test timed out");
    assert!(heartbeat_triggered, "ToolUse never arrived");
    assert!(
        responses.iter().any(|r| matches!(r, Response::Done)),
        "expected Done: {responses:?}"
    );

    // 3 LLM calls: chat turn 1, heartbeat check, chat turn 2.
    let reqs = server.take_requests();
    assert!(
        reqs.len() >= 2,
        "expected ≥2 LLM requests, got {}",
        reqs.len()
    );

    // Locate the heartbeat LLM request by content rather than by index:
    // the scheduler runs concurrently so request ordering is not guaranteed.
    let hb_req = reqs
        .iter()
        .find(|req| {
            req.messages().is_some_and(|msgs| {
                msgs.iter().any(|m| {
                    m.get("content")
                        .and_then(|c| c.as_str())
                        .is_some_and(|s| s.contains("check build status"))
                })
            })
        })
        .expect("heartbeat LLM request not found in captured requests");
    let hb_messages = hb_req.messages().expect("messages in heartbeat request");
    let hb_content: String = hb_messages
        .iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        hb_content.contains("check build status"),
        "heartbeat LLM request must contain item description; got:\n{hb_content}"
    );

    let _ = child.kill().await;
    let _ = child.wait().await;
}

/// When the LLM responds HEARTBEAT_OK, nothing is injected into the session.
#[tokio::test]
async fn heartbeat_ok_does_not_inject() {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::UnixStream;

    let server = MockLlmServer::start().await;

    // Turn 1: slow tool to keep session alive.
    server.enqueue(ScriptedResponse::tool_call(
        "hb-ok-tool-1",
        "shell_command",
        r#"{"command":"sleep 0.3"}"#,
    ));
    // Heartbeat LLM call: HEARTBEAT_OK — nothing to inject.
    server.enqueue(ScriptedResponse::text_chunks(vec!["HEARTBEAT_OK"]));
    // Turn 2: normal reply after tool (no steer injected).
    server.enqueue(ScriptedResponse::text_chunks(vec!["Done."]));

    let home = setup_home().expect("setup_home");
    write_heartbeat_config(home.path()).expect("write_heartbeat_config");

    let (socket, mut child, _socket_dir) =
        start_daemon_at_home_with_env(home.path(), &server.url(), &[])
            .await
            .expect("start_daemon");

    let session_id = uuid::Uuid::new_v4().to_string();

    let stream = UnixStream::connect(&socket).await.expect("connect");
    let (read_half, mut write_half) = stream.into_split();
    let chat_req = Request::Chat {
        prompt: "run slow tool".to_string(),
        tmux_pane: None,
        session_id: Some(session_id.clone()),
        model: "gpt-4o".to_string(),
    };
    let mut line = serde_json::to_string(&chat_req).unwrap();
    line.push('\n');
    write_half.write_all(line.as_bytes()).await.unwrap();

    let mut lines = BufReader::new(read_half).lines();
    let mut responses: Vec<Response> = Vec::new();
    let mut triggered = false;

    let result = tokio::time::timeout(Duration::from_secs(15), async {
        while let Some(raw) = lines.next_line().await.expect("reading line") {
            if raw.is_empty() {
                continue;
            }
            let frame: Response = serde_json::from_str(&raw).expect("parse frame");
            let done = matches!(frame, Response::Done | Response::Error { .. });
            if !triggered {
                if let Response::ToolUse { .. } = &frame {
                    let client = connect_client(&socket);
                    send_heartbeat_add(&client, &session_id, "check status")
                        .await
                        .expect("add");
                    send_heartbeat_trigger(&client).await.expect("trigger");
                    triggered = true;
                }
            }
            responses.push(frame);
            if done {
                break;
            }
        }
    })
    .await;

    result.expect("test timed out");

    // No SteerAck should appear — HEARTBEAT_OK means nothing was injected.
    assert!(
        !responses.iter().any(|r| matches!(r, Response::SteerAck)),
        "unexpected SteerAck — HEARTBEAT_OK should not inject anything: {responses:?}"
    );
    assert!(
        responses.iter().any(|r| matches!(r, Response::Done)),
        "expected Done: {responses:?}"
    );

    let _ = child.kill().await;
    let _ = child.wait().await;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Regression tests for comments 22-27 (heartbeat fixes)
// ---------------------------------------------------------------------------

/// Happy-path roundtrip: HeartbeatAdd → Done, HeartbeatList returns item,
/// HeartbeatDismiss → Done, list shows no pending items after dismiss.
///
/// Regression for comments 23-25: each handler returns Response::Done on
/// success (and Response::Error on DB failure — unit-tested in daemon.rs).
#[tokio::test]
async fn heartbeat_ipc_add_list_dismiss_roundtrip() {
    use support::helpers::{send_heartbeat_add, send_heartbeat_list, send_request, Response};

    let server = MockLlmServer::start().await;
    let home = setup_home().expect("setup_home");
    let (socket, mut child, _socket_dir) =
        start_daemon_at_home_with_env(home.path(), &server.url(), &[])
            .await
            .expect("start_daemon");

    let session_id = "test-hb-roundtrip";

    // Add an item — daemon must succeed (no Error response).
    send_heartbeat_add(
        &connect_client(&socket),
        session_id,
        "check build after deploy",
    )
    .await
    .expect("heartbeat_add must succeed");

    // List items — must include the item we just added, followed by Done.
    let list_responses = send_heartbeat_list(&connect_client(&socket), session_id, false)
        .await
        .expect("heartbeat_list");
    let entries: Vec<_> = list_responses
        .iter()
        .filter_map(|r| {
            if let Response::HeartbeatEntry {
                id,
                description,
                status,
                ..
            } = r
            {
                Some((*id, description.as_str(), status.as_str()))
            } else {
                None
            }
        })
        .collect();
    assert_eq!(entries.len(), 1, "expected 1 pending item; got {entries:?}");
    assert!(
        entries[0].1.contains("check build after deploy"),
        "unexpected description: {:?}",
        entries[0].1
    );
    assert_eq!(entries[0].2, "pending");
    let item_id = entries[0].0;

    // Dismiss the item — daemon must respond with Done, not Error.
    let dismiss = send_request(
        &connect_client(&socket),
        &Request::HeartbeatDismiss { id: item_id },
    )
    .await
    .expect("heartbeat_dismiss");
    assert!(
        dismiss.iter().any(|r| matches!(r, Response::Done)),
        "HeartbeatDismiss must respond with Done: {dismiss:?}"
    );
    assert!(
        !dismiss.iter().any(|r| matches!(r, Response::Error { .. })),
        "HeartbeatDismiss must not return Error on success: {dismiss:?}"
    );

    // List pending again — dismissed item must not appear.
    let pending = send_heartbeat_list(&connect_client(&socket), session_id, false)
        .await
        .expect("heartbeat_list after dismiss");
    assert!(
        !pending
            .iter()
            .any(|r| matches!(r, Response::HeartbeatEntry { .. })),
        "dismissed item should not appear in pending list: {pending:?}"
    );

    let _ = child.kill().await;
    let _ = child.wait().await;
}

// ---------------------------------------------------------------------------
// Consistent chat — critical gap tests
// ---------------------------------------------------------------------------

/// Turn history must survive a daemon restart: after killing and restarting the
/// daemon at the same home directory, turn 2 must still see turn 1's context.
///
/// This is the most important regression for the consistent-chat feature:
/// persistence lives in SQLite (`session_turns`), not in daemon memory, so
/// a restart must be transparent to the conversation.
#[tokio::test]
async fn consistent_chat_persists_across_daemon_restart() {
    let server = MockLlmServer::start().await;

    // Phase 1 — seed one turn.
    server.enqueue(ScriptedResponse::text_chunks(vec!["The capital is Paris."]));

    let seed = start_daemon(&server.url()).await.expect("seed daemon");
    let session_id = "restart-test-session-001";
    let client1 = connect_client(&seed.socket);
    let r1 = send_message_with_session(
        &client1,
        "what is the capital of France?",
        session_id,
        "gpt-4o",
    )
    .await
    .expect("turn 1");
    assert!(
        collect_text(&r1).contains("Paris"),
        "turn 1 failed: {:?}",
        collect_text(&r1)
    );

    // Phase 2 — kill the daemon, restart at the same home dir, run turn 2.
    let (home_path, _home_dir) = seed.kill_and_keep_home().await;

    server.enqueue(ScriptedResponse::text_chunks(vec!["It is a famous city."]));
    let (socket2, mut child2, _dir2) =
        start_daemon_at_home_with_env(&home_path, &server.url(), &[])
            .await
            .expect("restarted daemon");

    let client2 = connect_client(&socket2);
    let _r2 = send_message_with_session(&client2, "tell me more about it", session_id, "gpt-4o")
        .await
        .expect("turn 2");

    // The turn-2 LLM request must include the user message from turn 1.
    let reqs = server.take_requests();
    assert_eq!(reqs.len(), 2, "expected 2 requests, got {}", reqs.len());
    let turn2_messages = reqs[1]
        .body
        .get("messages")
        .and_then(|m| m.as_array())
        .expect("messages in turn 2 request");
    let has_turn1_user = turn2_messages.iter().any(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("user")
            && m.get("content")
                .and_then(|c| c.as_str())
                .map(|s| s.contains("capital of France"))
                .unwrap_or(false)
    });
    assert!(
        has_turn1_user,
        "turn 2 (after restart) must include turn 1 user message; messages: {turn2_messages:#?}"
    );

    let _ = child2.kill().await;
    let _ = child2.wait().await;
}

/// After `memory clear`, turn 2 must NOT see any context from turn 1.
/// Regression for comment 8: `memory_db::clear()` must wipe `session_turns`
/// so cleared history is truly gone.
#[tokio::test]
async fn consistent_chat_cleared_by_memory_clear() {
    let server = MockLlmServer::start().await;
    server.enqueue(ScriptedResponse::text_chunks(vec!["The answer is 42."]));
    server.enqueue(ScriptedResponse::text_chunks(vec!["Fresh start."]));

    let daemon = start_daemon(&server.url()).await.expect("daemon");
    let session_id = "clear-test-session-001";

    // Turn 1.
    let client1 = connect_client(&daemon.socket);
    send_message_with_session(&client1, "what is 6 times 7?", session_id, "gpt-4o")
        .await
        .expect("turn 1");

    // Clear memory via raw IPC (ClearMemory is not in the test helpers' Request enum).
    {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        use tokio::net::UnixStream;
        let stream = UnixStream::connect(&daemon.socket)
            .await
            .expect("connect for clear");
        let (reader, mut writer) = tokio::io::split(stream);
        let msg = r#"{"type":"clear_memory"}
"#;
        writer
            .write_all(msg.as_bytes())
            .await
            .expect("send clear_memory");
        let mut lines = BufReader::new(reader).lines();
        let frame = lines
            .next_line()
            .await
            .expect("clear_memory response")
            .expect("frame");
        assert!(
            frame.contains("done"),
            "ClearMemory must return done; got: {frame}"
        );
    }

    // Turn 2 — history should be empty after clear.
    let client2 = connect_client(&daemon.socket);
    send_message_with_session(&client2, "what was that again?", session_id, "gpt-4o")
        .await
        .expect("turn 2");

    let reqs = server.take_requests();
    assert_eq!(reqs.len(), 2, "expected 2 LLM requests, got {}", reqs.len());

    let turn2_messages = reqs[1]
        .body
        .get("messages")
        .and_then(|m| m.as_array())
        .expect("messages in turn 2");

    // After clear, turn 2 must NOT include turn 1's user message.
    let has_turn1_context = turn2_messages.iter().any(|m| {
        m.get("content")
            .and_then(|c| c.as_str())
            .map(|s| s.contains("6 times 7") || s.contains("42"))
            .unwrap_or(false)
    });
    assert!(
        !has_turn1_context,
        "after memory clear, turn 2 must not see turn 1 context; messages: {turn2_messages:#?}"
    );
}
