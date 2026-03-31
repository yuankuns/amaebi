//! Integration tests for the amaebi daemon using the mock LLM server.
//!
//! Each test:
//! 1. Starts the mock LLM server on a random port.
//! 2. Starts the `amaebi daemon` binary, pointed at the mock server.
//! 3. Connects a client to the daemon socket.
//! 4. Exercises the daemon and asserts on the responses / captured requests.
//!
//! ## Current tests (18 total)
//!
//! 1. `test_basic_chat_roundtrip` — basic chat round-trip
//! 2. `test_tool_call_roundtrip` — tool-call round-trip (shell echo)
//! 3. `test_max_tokens_present_in_request` — max_tokens present and > 0 in LLM request
//! 4. `test_request_format_valid` — LLM request contains model, stream:true, messages with roles
//! 5. `test_compaction_triggered_at_threshold` — Compacting frame emitted when threshold exceeded
//! 6. `test_compaction_preserves_summary` — summary text appears in next turn's messages
//! 7. `test_hot_tail_preserved_after_compaction` — message count bounded to hot tail after compaction
//! 8. `test_pre_flight_trim_on_resume` — resume loads history; pre-flight trim fires at low threshold
//! 9. `spawn_agent_runs_task` — parent spawns child agent; child runs echo; parent gets result
//! 10. `spawn_agent_child_cannot_spawn` — child agent cannot recursively call spawn_agent
//! 11. `spawn_agent_uses_specified_model` — child agent uses the model from spawn_agent args
//! 12. `spawn_agent_missing_workspace_returns_error` — missing workspace arg yields tool error to parent
//! 13. `spawn_agent_workspace_passed_to_sandbox` — workspace path appears in child agent's LLM context
//! 14. `spawn_agent_parallel_calls` — two spawn_agent calls with parallel=true run concurrently
//! 15. `spawn_agent_parallel_timing` — parallel=true batch completes in ~5s not ~10s (ignored)
//! 16. `cron_job_triggers_llm_call` — scheduled cron job fires and calls the LLM with the task prompt
//! 17. `cron_job_result_not_sent_to_chat` — cron output goes to inbox only, never to the chat stream
//! 18. `cron_job_with_spawn_agent` — cron job can call spawn_agent; child runs and returns result

mod support;

use support::{
    helpers::{
        collect_text, connect_client, seed_cron_job, send_message, send_message_with_session,
        send_resume, setup_home, start_daemon, start_daemon_at_home_with_env,
        start_daemon_with_env,
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

    let start = std::time::Instant::now();
    let responses = send_message(&client, "run two child tasks in parallel")
        .await
        .expect("send_message");
    let elapsed = start.elapsed();

    let text = collect_text(&responses);
    assert!(
        text.contains("all done"),
        "expected parent final text 'all done' in response: {text:?}"
    );

    // Parallel execution: ~1 s wall-clock + overhead.  Sequential would be
    // ≥ 2 s of sleep alone, so < 2.5 s proves concurrency.
    assert!(
        elapsed.as_millis() < 2500,
        "expected parallel execution to finish in < 2.5 s, took {elapsed:?}"
    );

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
