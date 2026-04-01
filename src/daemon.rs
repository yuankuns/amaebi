use anyhow::{Context, Result};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

use crate::auth::{amaebi_home, TokenCache};
use crate::copilot::{self, ApiToolCall, ApiToolCallFunction, FinishReason, Message};
use crate::cron;
use crate::inbox::InboxStore;
use crate::ipc::{write_frame, Request, Response};
use crate::memory_db;
use crate::tools::{self, ToolExecutor};

/// Compute `max_tokens` for a request to `model`: capped at half the model's context window
/// so it never exceeds what the model supports (e.g. gpt-4's 8,192-token limit).
fn max_output_tokens_for_model(model: &str) -> usize {
    // Ordered longest-prefix-first so that e.g. "gpt-4-turbo" beats "gpt-4".
    const TABLE: &[(&str, usize)] = &[
        ("gpt-5.4", 32_768), // gpt-5.4 via Responses API
        ("gpt-5.3", 32_768),
        ("gpt-5.2", 32_768),
        ("gpt-5.1", 32_768),
        ("gpt-5-mini", 16_384),
        ("gpt-5", 32_768), // catch-all for gpt-5 family
        ("gpt-4.1", 32_768),
        ("gpt-4o", 16_384),
        ("gpt-4-turbo", 4_096),
        ("gpt-4", 8_192),
        ("gpt-3.5-turbo", 4_096),
        ("o1", 100_000),
        ("o3", 100_000),
        ("claude", 16_384),
        // Gemini models: all variants cap at 8k output.
        ("gemini", 8_192),
    ];
    TABLE
        .iter()
        .find(|(prefix, _)| model.starts_with(prefix))
        .map(|(_, limit)| *limit)
        .unwrap_or(16_384) // conservative default for unknown models
}

fn response_max_tokens(model: &str) -> usize {
    let model_max = max_output_tokens_for_model(model);
    let context_budget = context_limit_for_model(model) / 2;
    let result = model_max.min(context_budget);
    tracing::debug!(
        model,
        model_max,
        context_budget,
        max_tokens = result,
        "resolved max output tokens"
    );
    result
}
/// Compact session history when prompt tokens exceed this fraction of available input.
const COMPACTION_THRESHOLD: f64 = 0.85;
/// Minimum recent user/assistant *pairs* to keep in the hot tail after a token-budget trim.
const HOT_TAIL_PAIRS: usize = 3;
/// How many past-session summaries to prepend to the system message.
const MAX_SUMMARIES: usize = 5;
/// Maximum chars per injected past-session summary.
const MAX_SUMMARY_CHARS: usize = 500;

/// State shared across all concurrent client connections via `Arc`.
pub struct DaemonState {
    pub http: reqwest::Client,
    pub tokens: Arc<TokenCache>,
    /// Tool executor — `LocalExecutor` now; swappable with `DockerExecutor` in Phase 4.
    pub executor: Box<dyn ToolExecutor>,
    /// Persistent SQLite connection opened once at startup.
    ///
    /// Wrapped in `Mutex` so that concurrent `spawn_blocking` tasks serialise
    /// all reads and writes through a single connection without re-running the
    /// schema setup (`PRAGMA`s, `CREATE TABLE`, triggers) on every request.
    pub db: Arc<Mutex<rusqlite::Connection>>,
    /// Sessions that currently have a compaction task in flight.
    ///
    /// Before spawning a new `compact_session` for a within-session compaction,
    /// insert the session ID.  The task removes itself on completion.  If the
    /// ID is already present, skip the spawn to prevent overlapping compactions.
    pub compacting_sessions: Arc<Mutex<HashSet<String>>>,
}

impl DaemonState {
    /// Create a new `DaemonState`, opening the SQLite DB inside
    /// `spawn_blocking` so the file I/O never blocks the async reactor.
    pub async fn new() -> Result<Self> {
        let http = reqwest::Client::builder()
            .build()
            .context("building HTTP client")?;
        let db_path = memory_db::db_path().context("resolving memory DB path")?;
        let conn = tokio::task::spawn_blocking(move || memory_db::init_db(&db_path))
            .await
            .unwrap_or_else(|e| Err(anyhow::anyhow!("DB init panicked: {e}")))
            .context("opening memory DB")?;

        let db = Arc::new(Mutex::new(conn));
        let compacting_sessions: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));

        let tokens = Arc::new(TokenCache::new());

        let spawn_ctx = Arc::new(tools::SpawnContext {
            http: http.clone(),
            db: Arc::clone(&db),
            compacting_sessions: Arc::clone(&compacting_sessions),
            tokens: Arc::clone(&tokens),
        });

        let mut executor = tools::LocalExecutor::new();
        executor.spawn_ctx = Some(spawn_ctx);

        Ok(Self {
            http,
            tokens,
            executor: Box::new(executor),
            db,
            compacting_sessions,
        })
    }
}

// ---------------------------------------------------------------------------
// Token budget helpers
// ---------------------------------------------------------------------------

/// Approximate token count for a message list.
///
/// Uses tiktoken o200k_base (GPT-4o's encoding) with the OpenAI overhead formula:
/// 4 tokens per message + role + content + 3 priming tokens for the reply.
/// The tokenizer is initialised lazily on the first call (OnceLock).
///
/// Falls back to a conservative char-based estimate (~4 chars/token) if the
/// tokenizer cannot be loaded, so a failed initialisation never panics the daemon.
fn count_message_tokens(messages: &[Message]) -> usize {
    use std::sync::OnceLock;
    static BPE: OnceLock<Result<tiktoken_rs::CoreBPE, String>> = OnceLock::new();
    let bpe_result = BPE.get_or_init(|| {
        tiktoken_rs::o200k_base().map_err(|e| format!("failed to load o200k_base tokenizer: {e}"))
    });

    match bpe_result {
        Ok(bpe) => {
            let mut total = 3usize; // priming tokens for the assistant reply
            for msg in messages {
                total += 4; // per-message overhead: <|start|>, role, separator, <|end|>
                total += bpe.encode_with_special_tokens(&msg.role).len();
                if let Some(ref content) = msg.content {
                    total += bpe.encode_with_special_tokens(content).len();
                }
                if !msg.tool_calls.is_empty() {
                    if let Ok(s) = serde_json::to_string(&msg.tool_calls) {
                        total += bpe.encode_with_special_tokens(&s).len();
                    }
                }
                if let Some(ref id) = msg.tool_call_id {
                    total += bpe.encode_with_special_tokens(id).len();
                }
            }
            total
        }
        Err(e) => {
            use std::sync::atomic::{AtomicBool, Ordering};
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!(error = %e, "tokenizer unavailable; using char-based estimate");
            }
            let mut char_count = 0usize;
            for msg in messages {
                char_count += msg.role.len();
                if let Some(ref content) = msg.content {
                    char_count += content.len();
                }
                if !msg.tool_calls.is_empty() {
                    if let Ok(s) = serde_json::to_string(&msg.tool_calls) {
                        char_count += s.len();
                    }
                }
                if let Some(ref id) = msg.tool_call_id {
                    char_count += id.len();
                }
            }
            3 + messages.len() * 4 + char_count.div_ceil(4)
        }
    }
}

/// Context window size for `model`, matched by prefix (longest wins).
///
/// Falls back to a conservative 32 k for unknown models so we never send
/// more tokens than the server can handle.
fn context_limit_for_model(model: &str) -> usize {
    // Ordered longest-prefix-first so that e.g. "gpt-4-turbo" beats "gpt-4".
    const TABLE: &[(&str, usize)] = &[
        // gpt-5.x via Responses API: 128k context (conservative)
        ("gpt-5", 128_000),
        ("gpt-4o", 128_000),
        ("gpt-4.1", 1_047_576),
        ("gpt-4-turbo", 128_000),
        ("gpt-4", 8_192),
        ("gpt-3.5-turbo", 16_385),
        ("o1", 200_000),
        ("o3", 200_000),
        // Claude models: 200k context window.
        ("claude", 200_000),
        // Gemini models — more specific prefixes must precede less specific ones.
        ("gemini-2.0-flash", 1_048_576), // Gemini 2.0 Flash: 1 M context
        ("gemini-1.5-pro", 2_097_152),   // Gemini 1.5 Pro: 2 M context
        ("gemini-1.5-flash", 1_048_576), // Gemini 1.5 Flash: 1 M context
        ("gemini-1.0", 32_768),          // Gemini 1.0: 32 k context
        ("gemini", 128_000),             // conservative catch-all for other Gemini variants
    ];
    TABLE
        .iter()
        .find(|(prefix, _)| model.starts_with(prefix))
        .map(|(_, limit)| *limit)
        .unwrap_or(32_768) // conservative default for unknown models
}

/// Token threshold above which compaction should be triggered for `model`.
///
/// Returns `usize::MAX` for models whose context window is too small to
/// produce a non-zero compaction threshold, disabling compaction/trimming for
/// those models rather than triggering it on every turn.
fn compaction_threshold_tokens(model: &str) -> usize {
    // Allow manual override for debugging: AMAEBI_COMPACTION_THRESHOLD=<tokens>
    if let Ok(val) = std::env::var("AMAEBI_COMPACTION_THRESHOLD") {
        if let Ok(n) = val.parse::<usize>() {
            return n;
        }
    }
    let available = context_limit_for_model(model).saturating_sub(response_max_tokens(model));
    let threshold = (available as f64 * COMPACTION_THRESHOLD) as usize;
    if threshold == 0 {
        return usize::MAX;
    }
    threshold
}

// ---------------------------------------------------------------------------
// Listener loop
// ---------------------------------------------------------------------------

pub async fn run(socket: PathBuf) -> Result<()> {
    if socket.exists() {
        std::fs::remove_file(&socket)
            .with_context(|| format!("removing stale socket {}", socket.display()))?;
    }

    let listener = UnixListener::bind(&socket)
        .with_context(|| format!("binding Unix socket {}", socket.display()))?;

    tracing::info!(path = %socket.display(), "daemon listening");

    let state = Arc::new(DaemonState::new().await?);

    // Spawn the 1-minute cron scheduler in the background.
    {
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            run_cron_scheduler(state).await;
        });
    }

    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                let state = Arc::clone(&state);
                tokio::spawn(async move {
                    if let Err(e) = handle_connection(stream, state).await {
                        tracing::error!(error = %e, "connection error");
                    }
                });
            }
            Err(e) => {
                tracing::error!(error = %e, "accept error");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-connection handler
// ---------------------------------------------------------------------------

async fn handle_connection(stream: UnixStream, state: Arc<DaemonState>) -> Result<()> {
    let (owned_reader, owned_writer) = stream.into_split();

    // Shared writer — the agentic-loop task holds the lock during generation;
    // the main task acquires it for Done/Error and for single-turn requests.
    let writer: Arc<tokio::sync::Mutex<tokio::net::unix::OwnedWriteHalf>> =
        Arc::new(tokio::sync::Mutex::new(owned_writer));

    // Forwarding task: reads every line from the socket and sends it to a
    // channel so the main task can use select! to route steer frames while
    // the agentic loop runs.
    let (frame_tx, mut frame_rx) = tokio::sync::mpsc::channel::<String>(64);
    tokio::spawn(async move {
        let mut lines = BufReader::new(owned_reader).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if frame_tx.send(line).await.is_err() {
                break;
            }
        }
    });

    // Carries the messages array across Chat turns on this connection.
    // None → first turn (load from DB). Some → subsequent turns (extend).
    // Tracks the session_id so context is reset when the client switches sessions.
    let mut carried_messages: Option<Vec<Message>> = None;
    let mut carried_session_id: Option<String> = None;

    'connection: loop {
        let line = match frame_rx.recv().await {
            Some(l) => l,
            None => break 'connection,
        };
        if line.trim().is_empty() {
            continue;
        }
        let req: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, "unparseable request frame; sending error to client");
                let mut w = writer.lock().await;
                let _ = write_frame(
                    &mut *w,
                    &Response::Error {
                        message: format!("invalid request: {e}"),
                    },
                )
                .await;
                continue;
            }
        };

        match req {
            Request::ClearMemory => {
                tracing::info!("received memory clear request");
                let db = Arc::clone(&state.db);
                let result = tokio::task::spawn_blocking(move || {
                    let conn = db.lock().unwrap_or_else(|p| p.into_inner());
                    memory_db::clear(&conn)
                })
                .await
                .unwrap_or_else(|e| Err(anyhow::anyhow!("DB clear panicked: {e}")));
                if let Err(e) = result {
                    tracing::warn!(error = %e, "failed to clear memory DB");
                }
                let mut w = writer.lock().await;
                write_frame(&mut *w, &Response::Done).await?;
            }

            Request::StoreMemory { user, assistant } => {
                store_conversation(&state, "global", &user, &assistant).await;
                let mut w = writer.lock().await;
                write_frame(&mut *w, &Response::Done).await?;
            }

            Request::RetrieveContext { prompt } => {
                let db = Arc::clone(&state.db);
                let entries = tokio::task::spawn_blocking(move || {
                    let conn = db.lock().unwrap_or_else(|p| p.into_inner());
                    memory_db::retrieve_context(&conn, &prompt, 4, 10)
                })
                .await
                .unwrap_or_else(|e| Err(anyhow::anyhow!("memory read panicked: {e}")))
                .unwrap_or_else(|e| {
                    tracing::warn!(error = %e, "failed to retrieve memory context via IPC");
                    vec![]
                });
                let mut w = writer.lock().await;
                for entry in entries {
                    write_frame(
                        &mut *w,
                        &Response::MemoryEntry {
                            role: entry.role,
                            content: truncate_chars(entry.content, MAX_HISTORY_CHARS),
                        },
                    )
                    .await?;
                }
                write_frame(&mut *w, &Response::Done).await?;
            }

            Request::Steer { .. } => {
                let mut w = writer.lock().await;
                write_frame(
                    &mut *w,
                    &Response::Error {
                        message: "no active agentic loop to steer on this connection".into(),
                    },
                )
                .await?;
            }

            Request::Interrupt { .. } => {
                tracing::debug!("ignoring Interrupt on fresh connection (no active loop)");
            }

            Request::SubmitDetach {
                prompt,
                tmux_pane,
                model,
                session_id,
            } => {
                tracing::info!(model = %model, prompt_len = prompt.len(), "received detach request");
                if let Err(e) = state.tokens.get(&state.http).await {
                    let mut w = writer.lock().await;
                    write_frame(
                        &mut *w,
                        &Response::Error {
                            message: format!("authentication error: {e:#}"),
                        },
                    )
                    .await?;
                    break 'connection;
                }
                let sid = session_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                {
                    let mut w = writer.lock().await;
                    write_frame(
                        &mut *w,
                        &Response::DetachAccepted {
                            session_id: sid.clone(),
                        },
                    )
                    .await?;
                }
                let state = Arc::clone(&state);
                tokio::spawn(async move {
                    let db = Arc::clone(&state.db);
                    let sid_c = sid.clone();
                    let history = tokio::task::spawn_blocking(move || {
                        let conn = db.lock().unwrap_or_else(|p| p.into_inner());
                        memory_db::get_session_history(&conn, &sid_c)
                    })
                    .await
                    .unwrap_or_else(|_| Ok(vec![]))
                    .unwrap_or_default();
                    let mut messages =
                        build_messages(&prompt, tmux_pane.as_deref(), &history, &[], None);
                    inject_skill_files(&mut messages).await;
                    let threshold = compaction_threshold_tokens(&model);
                    if count_message_tokens(&messages) > threshold {
                        let hot = HOT_TAIL_PAIRS * 2;
                        let trimmed = if history.len() > hot {
                            &history[history.len() - hot..]
                        } else {
                            &history[..]
                        };
                        messages =
                            build_messages(&prompt, tmux_pane.as_deref(), trimmed, &[], None);
                        inject_skill_files(&mut messages).await;
                    }
                    let mut sink = tokio::io::sink();
                    let (_, mut steer_rx) = tokio::sync::mpsc::channel::<Option<String>>(1);
                    match run_agentic_loop(&state, &model, messages, &mut sink, &mut steer_rx, true)
                        .await
                    {
                        Ok((final_text, _, _)) => {
                            store_conversation(
                                &state,
                                &sid,
                                &truncate_chars(prompt.clone(), MAX_PROMPT_CHARS),
                                &truncate_chars(final_text.clone(), MAX_RESPONSE_CHARS),
                            )
                            .await;
                            let task_desc = truncate_chars(prompt, 200);
                            if let Ok(inbox) = InboxStore::open() {
                                let _ = inbox.save_report(&sid, &task_desc, &final_text);
                            }
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "detach agentic loop error");
                            let task_desc = truncate_chars(prompt, 200);
                            if let Ok(inbox) = InboxStore::open() {
                                let _ =
                                    inbox.save_report(&sid, &task_desc, &format!("[error] {e:#}"));
                            }
                        }
                    }
                });
            }

            Request::Resume {
                prompt,
                tmux_pane,
                model,
                session_id,
            } => {
                tracing::info!(model = %model, session_id = %session_id, prompt_len = prompt.len(), "received resume request");
                if let Err(e) = state.tokens.get(&state.http).await {
                    let mut w = writer.lock().await;
                    write_frame(
                        &mut *w,
                        &Response::Error {
                            message: format!("authentication error: {e:#}"),
                        },
                    )
                    .await?;
                    break 'connection;
                }
                let (steer_tx, mut steer_rx) = tokio::sync::mpsc::channel::<Option<String>>(16);
                let expected_sid = session_id.clone();
                let db = Arc::clone(&state.db);
                let sid_c = session_id.clone();
                let (history, summaries, own_summary) =
                    tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
                        let conn = db.lock().unwrap_or_else(|p| p.into_inner());
                        Ok((
                            memory_db::get_session_history(&conn, &sid_c)?,
                            memory_db::get_recent_summaries(&conn, &sid_c, MAX_SUMMARIES)?,
                            memory_db::get_session_own_summary(&conn, &sid_c)?,
                        ))
                    })
                    .await
                    .unwrap_or_else(|e| {
                        tracing::warn!(error=%e,"resume load panicked");
                        Ok((vec![], vec![], None))
                    })
                    .unwrap_or_else(|e| {
                        tracing::warn!(error=%e,"resume load failed");
                        (vec![], vec![], None)
                    });
                let mut messages = build_messages(
                    &prompt,
                    tmux_pane.as_deref(),
                    &history,
                    &summaries,
                    own_summary.as_deref(),
                );
                inject_skill_files(&mut messages).await;
                let threshold = compaction_threshold_tokens(&model);
                if count_message_tokens(&messages) > threshold {
                    let hot = HOT_TAIL_PAIRS * 2;
                    let trimmed = if history.len() > hot {
                        &history[history.len() - hot..]
                    } else {
                        &history[..]
                    };
                    messages = build_messages(
                        &prompt,
                        tmux_pane.as_deref(),
                        trimmed,
                        &summaries,
                        own_summary.as_deref(),
                    );
                    inject_skill_files(&mut messages).await;
                }
                let writer_loop = Arc::clone(&writer);
                let state_loop = Arc::clone(&state);
                let model_loop = model.clone();
                let mut loop_handle = tokio::spawn(async move {
                    let mut w = writer_loop.lock().await;
                    run_agentic_loop(
                        &state_loop,
                        &model_loop,
                        messages,
                        &mut *w,
                        &mut steer_rx,
                        true,
                    )
                    .await
                });
                let result = loop {
                    tokio::select! { biased;
                        r = &mut loop_handle => { break r.unwrap_or_else(|e| Err(anyhow::anyhow!("loop panicked: {e}"))); }
                        f = frame_rx.recv() => { match f {
                            None => { loop_handle.abort(); break 'connection; }
                            Some(line) => { if let Ok(req) = serde_json::from_str::<Request>(&line) { match req {
                                Request::Steer { session_id: sid, message } if sid == expected_sid => { if !message.is_empty() { let _ = steer_tx.send(Some(message)).await; } }
                                Request::Interrupt { session_id: sid } if sid == expected_sid => { let _ = steer_tx.send(None).await; }
                                // Steer/Interrupt for a different session: silently ignore per IPC contract.
                                Request::Steer { .. } | Request::Interrupt { .. } => {
                                    tracing::debug!("ignoring steer/interrupt for non-active session");
                                }
                                _ => {
                                    tracing::warn!("dropping unsolicited frame during active agentic loop");
                                    // Use try_lock to avoid blocking Steer/Interrupt routing:
                                    // the agentic loop holds the writer lock for most of the
                                    // turn; awaiting it here would stall the select loop and
                                    // delay steering frames.  If we can't acquire it immediately
                                    // we skip the error reply rather than blocking.
                                    if let Ok(mut w) = writer.try_lock() {
                                        let _ = write_frame(&mut *w, &Response::Error {
                                            message: "busy: another request is already in progress on this connection".into(),
                                        }).await;
                                    }
                                }
                            }}}
                        }}
                    }
                };
                match result {
                    Ok((response_text, _, _)) => {
                        store_conversation(
                            &state,
                            &session_id,
                            &truncate_chars(prompt.clone(), MAX_PROMPT_CHARS),
                            &truncate_chars(response_text, MAX_RESPONSE_CHARS),
                        )
                        .await;
                        let mut w = writer.lock().await;
                        write_frame(&mut *w, &Response::Done).await?;
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "resume agentic loop error");
                        let mut w = writer.lock().await;
                        let _ = write_frame(
                            &mut *w,
                            &Response::Error {
                                message: format!("agent error: {e:#}"),
                            },
                        )
                        .await;
                    }
                }
            }

            Request::Chat {
                prompt,
                tmux_pane,
                model,
                session_id,
            } => {
                tracing::info!(pane = ?tmux_pane, model = %model, prompt_len = prompt.len(), "received chat request");
                if let Err(e) = state.tokens.get(&state.http).await {
                    let mut w = writer.lock().await;
                    write_frame(
                        &mut *w,
                        &Response::Error {
                            message: format!("authentication error: {e:#}"),
                        },
                    )
                    .await?;
                    break 'connection;
                }
                let sid = session_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

                // If the session_id changed, discard carried context (comment 5).
                if carried_session_id.as_deref() != Some(&sid) {
                    carried_messages = None;
                }

                // First turn: load history from DB.  Subsequent turns: extend carried messages.
                // Apply token-budget trim either way so long-lived connections stay bounded (comment 6).
                let (messages, pre_flight_trimmed) = if let Some(mut prev) = carried_messages.take()
                {
                    prev.push(Message::user(prompt.clone()));
                    // Do NOT re-inject skill files — they were injected on the
                    // first turn and are already in `prev`.  Re-injecting every
                    // turn duplicates system messages and grows context unboundedly.
                    let threshold_inner = compaction_threshold_tokens(&model);
                    let trimmed = if count_message_tokens(&prev) > threshold_inner {
                        // Rebuild from persisted history when over budget
                        let db2 = Arc::clone(&state.db);
                        let sid2 = sid.clone();
                        let (hist2, sum2, own2) =
                            tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
                                let conn = db2.lock().unwrap_or_else(|p| p.into_inner());
                                Ok((
                                    memory_db::get_session_history(&conn, &sid2)?,
                                    memory_db::get_recent_summaries(&conn, &sid2, MAX_SUMMARIES)?,
                                    memory_db::get_session_own_summary(&conn, &sid2)?,
                                ))
                            })
                            .await
                            .unwrap_or_else(|_| Ok((vec![], vec![], None)))
                            .unwrap_or((vec![], vec![], None));
                        let hot = HOT_TAIL_PAIRS * 2;
                        let trimmed_hist = if hist2.len() > hot {
                            &hist2[hist2.len() - hot..]
                        } else {
                            &hist2[..]
                        };
                        let mut rebuilt = build_messages(
                            &prompt,
                            tmux_pane.as_deref(),
                            trimmed_hist,
                            &sum2,
                            own2.as_deref(),
                        );
                        inject_skill_files(&mut rebuilt).await;
                        (rebuilt, true)
                    } else {
                        (prev, false)
                    };
                    trimmed
                } else {
                    let db = Arc::clone(&state.db);
                    let sid_c = sid.clone();
                    let (history, summaries, own_summary) =
                        tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
                            let conn = db.lock().unwrap_or_else(|p| p.into_inner());
                            Ok((
                                memory_db::get_session_history(&conn, &sid_c)?,
                                memory_db::get_recent_summaries(&conn, &sid_c, MAX_SUMMARIES)?,
                                memory_db::get_session_own_summary(&conn, &sid_c)?,
                            ))
                        })
                        .await
                        .unwrap_or_else(|e| {
                            tracing::warn!(error=%e,"chat history load panicked");
                            Ok((vec![], vec![], None))
                        })
                        .unwrap_or_else(|e| {
                            tracing::warn!(error=%e,"failed to load chat history");
                            (vec![], vec![], None)
                        });

                    if history.is_empty() {
                        let db = Arc::clone(&state.db);
                        let sid_c = sid.clone();
                        let old = tokio::task::spawn_blocking(move || {
                            let conn = db.lock().unwrap_or_else(|p| p.into_inner());
                            memory_db::get_sessions_without_summary(&conn, &sid_c, MAX_SUMMARIES)
                        })
                        .await
                        .unwrap_or_else(|_| Ok(vec![]))
                        .unwrap_or_default();
                        for old_sid in old {
                            tokio::spawn(compact_session(
                                Arc::clone(&state),
                                old_sid,
                                model.clone(),
                                0,
                            ));
                        }
                    }

                    let mut msgs = build_messages(
                        &prompt,
                        tmux_pane.as_deref(),
                        &history,
                        &summaries,
                        own_summary.as_deref(),
                    );
                    inject_skill_files(&mut msgs).await;
                    let threshold_inner = compaction_threshold_tokens(&model);
                    let pft = if count_message_tokens(&msgs) > threshold_inner {
                        let hot = HOT_TAIL_PAIRS * 2;
                        let trimmed = if history.len() > hot {
                            &history[history.len() - hot..]
                        } else {
                            &history[..]
                        };
                        msgs = build_messages(
                            &prompt,
                            tmux_pane.as_deref(),
                            trimmed,
                            &summaries,
                            own_summary.as_deref(),
                        );
                        inject_skill_files(&mut msgs).await;
                        true
                    } else {
                        false
                    };
                    (msgs, pft)
                };

                let pre_send_tokens = count_message_tokens(&messages);
                let threshold = compaction_threshold_tokens(&model);
                // pre_flight_trimmed restores the original compaction trigger (comment 7):
                // if we had to trim, compact even if prompt_tokens stays below threshold.
                let (steer_tx, mut steer_rx) = tokio::sync::mpsc::channel::<Option<String>>(16);
                let expected_chat_sid = sid.clone();

                let writer_loop = Arc::clone(&writer);
                let state_loop = Arc::clone(&state);
                let model_loop = model.clone();
                let mut loop_handle = tokio::spawn(async move {
                    let mut w = writer_loop.lock().await;
                    run_agentic_loop(
                        &state_loop,
                        &model_loop,
                        messages,
                        &mut *w,
                        &mut steer_rx,
                        true,
                    )
                    .await
                });

                let loop_result = loop {
                    tokio::select! { biased;
                        r = &mut loop_handle => { break r.unwrap_or_else(|e| Err(anyhow::anyhow!("loop panicked: {e}"))); }
                        f = frame_rx.recv() => { match f {
                            None => { loop_handle.abort(); break 'connection; }
                            Some(line) => { if let Ok(req) = serde_json::from_str::<Request>(&line) { match req {
                                Request::Steer { session_id: sid, message } if sid == expected_chat_sid => { if !message.is_empty() { let _ = steer_tx.send(Some(message)).await; } }
                                Request::Interrupt { session_id: sid } if sid == expected_chat_sid => { let _ = steer_tx.send(None).await; }
                                // Steer/Interrupt for a different session: silently ignore per IPC contract.
                                Request::Steer { .. } | Request::Interrupt { .. } => {
                                    tracing::debug!("ignoring steer/interrupt for non-active session");
                                }
                                _ => {
                                    tracing::warn!("dropping unsolicited frame during active chat loop");
                                    // Use try_lock to avoid blocking Steer/Interrupt routing:
                                    // the agentic loop holds the writer lock for most of the
                                    // turn; awaiting it here would stall the select loop and
                                    // delay steering frames.  If we can't acquire it immediately
                                    // we skip the error reply rather than blocking.
                                    if let Ok(mut w) = writer.try_lock() {
                                        let _ = write_frame(&mut *w, &Response::Error {
                                            message: "busy: another request is already in progress on this connection".into(),
                                        }).await;
                                    }
                                }
                            }}}
                        }}
                    }
                };

                match loop_result {
                    Ok((response_text, prompt_tokens, final_messages)) => {
                        store_conversation(
                            &state,
                            &sid,
                            &truncate_chars(prompt.clone(), MAX_PROMPT_CHARS),
                            &truncate_chars(response_text.clone(), MAX_RESPONSE_CHARS),
                        )
                        .await;
                        let effective_tokens = if prompt_tokens > 0 {
                            prompt_tokens
                        } else {
                            pre_send_tokens
                        };
                        let mut w = writer.lock().await;
                        if pre_flight_trimmed || effective_tokens > threshold {
                            tracing::info!(session=%sid, effective_tokens, threshold, "compacting conversation history");
                            let _ = write_frame(&mut *w, &Response::Compacting).await;
                            let already = {
                                let mut g = state
                                    .compacting_sessions
                                    .lock()
                                    .unwrap_or_else(|p| p.into_inner());
                                !g.insert(sid.clone())
                            };
                            if !already {
                                tokio::spawn(compact_session(
                                    Arc::clone(&state),
                                    sid.clone(),
                                    model.clone(),
                                    HOT_TAIL_PAIRS * 2,
                                ));
                            }
                        }
                        write_frame(&mut *w, &Response::Done).await?;
                        drop(w);
                        // Carry messages and session_id for the next turn on this connection.
                        carried_messages = Some(final_messages);
                        carried_session_id = Some(sid.clone());
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "agentic loop error");
                        let mut w = writer.lock().await;
                        let _ = write_frame(
                            &mut *w,
                            &Response::Error {
                                message: format!("agent error: {e:#}"),
                            },
                        )
                        .await;
                        carried_messages = None;
                        carried_session_id = None;
                    }
                }
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Character-level truncation
// ---------------------------------------------------------------------------

/// Maximum chars stored for a user prompt in session history.
const MAX_PROMPT_CHARS: usize = 4_000;
/// Maximum chars stored for an assistant response in session history.
const MAX_RESPONSE_CHARS: usize = 8_000;

/// Truncate `s` to at most `max` Unicode scalar values (including the marker).
///
/// If truncation occurs, appends `"…[truncated]"` so the model knows the
/// content was cut.  The returned string always contains at most `max` chars.
/// Operates on char boundaries, never slicing multi-byte sequences.
///
/// Edge case: if `max` is smaller than or equal to the marker length, the
/// marker itself is truncated to `max` chars.
fn truncate_chars(s: String, max: usize) -> String {
    // Fast path: if there is no (max+1)-th character the string is within limit.
    // char_indices().nth(max) is O(max), unlike chars().count() which is O(n).
    if s.char_indices().nth(max).is_none() {
        return s; // already within limit — no additional allocation
    }
    const MARKER: &str = "…[truncated]";
    let marker_len = MARKER.chars().count(); // derived, never drifts from MARKER
    if max <= marker_len {
        return MARKER.chars().take(max).collect();
    }
    let content_len = max - marker_len;
    let mut out: String = s.chars().take(content_len).collect();
    out.push_str(MARKER);
    out
}

// ---------------------------------------------------------------------------
// Memory helpers — canonical DB access for daemon and ACP agent
// ---------------------------------------------------------------------------

/// Persist a user/assistant exchange to SQLite.
///
/// Runs inside `spawn_blocking`; locks `state.db` to serialise concurrent
/// writes within this process.  Best-effort: errors are logged but not
/// propagated.
pub(crate) async fn store_conversation(
    state: &DaemonState,
    session_id: &str,
    user: &str,
    assistant: &str,
) {
    let db = Arc::clone(&state.db);
    let sid = session_id.to_owned();
    let user_owned = user.to_owned();
    let assistant_owned = assistant.to_owned();
    let result = tokio::task::spawn_blocking(move || {
        let timestamp = chrono::Utc::now().to_rfc3339();
        let mut conn = db.lock().unwrap_or_else(|p| p.into_inner());
        // Write the user/assistant pair atomically so they are never split.
        let tx = conn.transaction().context("beginning memory transaction")?;
        memory_db::store_memory(&tx, &timestamp, &sid, "user", &user_owned, "")?;
        memory_db::store_memory(&tx, &timestamp, &sid, "assistant", &assistant_owned, "")?;
        tx.commit().context("committing memory transaction")
    })
    .await
    .unwrap_or_else(|e| Err(anyhow::anyhow!("memory write panicked: {e}")));
    if let Err(e) = result {
        tracing::warn!(error = %e, "failed to save memory");
    }
}

// ---------------------------------------------------------------------------
// Session compaction
// ---------------------------------------------------------------------------

/// Summarise a portion of a session's history and persist it to `session_summaries`.
///
/// Triggered lazily via `tokio::spawn` in two cases:
/// 1. **Cross-session** (`keep_recent = 0`): a closed session is summarised in full
///    so future sessions can learn from it.
/// 2. **Within-session** (`keep_recent = HOT_TAIL_PAIRS * 2`): only the turns that the
///    token-budget trim is *dropping* are summarised — the recent `keep_recent` rows are
///    excluded so the summary and the raw hot tail never overlap.
///
/// Uses the same agentic loop as normal requests (handles token refresh and retries),
/// but with a sink writer and a dropped steer sender so it is fully non-interactive.
/// Best-effort: errors are only logged.
/// RAII guard that removes `session_id` from `compacting_sessions` when dropped,
/// ensuring the slot is freed on every exit path (early return, error, or normal).
struct CompactingGuard {
    compacting_sessions: Arc<Mutex<HashSet<String>>>,
    session_id: String,
}

impl Drop for CompactingGuard {
    fn drop(&mut self) {
        self.compacting_sessions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&self.session_id);
    }
}

async fn compact_session(
    state: Arc<DaemonState>,
    session_id: String,
    model: String,
    keep_recent: usize,
) {
    // Hold a guard so `compacting_sessions` is cleaned up on every exit path.
    let _guard = CompactingGuard {
        compacting_sessions: Arc::clone(&state.compacting_sessions),
        session_id: session_id.clone(),
    };

    let db = Arc::clone(&state.db);
    let sid = session_id.clone();
    // Load the non-archived turns to summarise, plus any existing summary so the
    // new summary can incorporate it (cumulative re-compaction).
    let (history, existing_summary) = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let conn = db.lock().unwrap_or_else(|p| p.into_inner());
        let total = memory_db::count_session_turns(&conn, &sid)?;
        // Only summarise non-archived turns; keep_recent holds the hot tail out.
        let to_summarise = total.saturating_sub(keep_recent);
        tracing::debug!(
            session_id = %sid,
            total,
            keep_recent,
            to_summarise,
            "compact_session: loaded history counts"
        );
        let history = memory_db::get_session_oldest(&conn, &sid, to_summarise)?;
        let existing_summary = memory_db::get_session_own_summary(&conn, &sid)?;
        Ok((history, existing_summary))
    })
    .await
    .unwrap_or_else(|e| {
        tracing::warn!(error = %e, "compact_session: history load panicked");
        Ok((vec![], None))
    })
    .unwrap_or_else(|e| {
        tracing::warn!(error = %e, "compact_session: failed to load history");
        (vec![], None)
    });

    if history.is_empty() {
        return;
    }

    // Build the compaction prompt.  If a previous summary exists, inject it before
    // the new turns so the new summary is cumulative (covers all archived history).
    let mut messages = vec![Message::system(
        "You are a memory compactor. Output 3-5 bullet points capturing the key outcomes, \
         decisions, and facts. Be concise and factual. Output only the bullet points, no preamble.",
    )];

    if let Some(ref prev) = existing_summary {
        // Re-compaction: show the previous summary as an already-produced assistant
        // turn so the model can extend it with the new turns that follow.
        messages.push(Message::user(
            "Summarise the conversation so far:".to_owned(),
        ));
        messages.push(Message::assistant(Some(prev.clone()), vec![]));
    }

    for entry in &history {
        let content = truncate_chars(entry.content.clone(), 1_500);
        match entry.role.as_str() {
            "user" => messages.push(Message::user(content)),
            "assistant" => messages.push(Message::assistant(Some(content), vec![])),
            _ => {}
        }
    }
    // The API requires the last message to be from the user role.
    // For re-compaction always append the update request so the model knows to
    // produce a combined summary rather than just continue the conversation.
    if existing_summary.is_some() {
        messages.push(Message::user(
            "Produce an updated combined summary covering everything above.".to_owned(),
        ));
    } else if messages.last().is_some_and(|m| m.role == "assistant") {
        messages.push(Message::user(
            "Summarise the conversation above into 3-5 bullet points.".to_owned(),
        ));
    }

    // Collect the IDs to archive on success.
    let ids_to_archive: Vec<i64> = history.iter().map(|e| e.id).collect();

    // Use a direct API call with no tools so the model cannot execute shell
    // commands during background summarisation.
    let compact_result: Result<String> = async {
        let mut sink = tokio::io::sink();
        let resp = invoke_model(
            &state,
            &model,
            &messages,
            &[],
            response_max_tokens(&model),
            &mut sink,
        )
        .await
        .context("compact_session: model call failed")?;
        Ok(resp.text)
    }
    .await;

    match compact_result {
        Ok(ref text) if !text.trim().is_empty() => {
            let summary = text.trim().to_owned();
            let db = Arc::clone(&state.db);
            let sid = session_id.clone();
            let ts = chrono::Utc::now().to_rfc3339();
            let result = tokio::task::spawn_blocking(move || {
                let conn = db.lock().unwrap_or_else(|p| p.into_inner());
                let tx = conn
                    .unchecked_transaction()
                    .context("compact_session: begin transaction")?;
                memory_db::store_session_summary(&conn, &sid, &summary, &ts)?;
                memory_db::archive_session_turns(&conn, &ids_to_archive)?;
                tx.commit().context("compact_session: commit transaction")
            })
            .await
            .unwrap_or_else(|e| Err(anyhow::anyhow!("compact_session write panicked: {e}")));
            if let Err(e) = result {
                tracing::warn!(error = %e, "compact_session: failed to store summary");
            } else {
                tracing::debug!(session_id = %session_id, "session compacted");
            }
        }
        Ok(_) => {
            tracing::debug!(session_id = %session_id, "compact_session: empty summary (ignored)")
        }
        Err(ref e) => tracing::warn!(error = %e, "compact_session: API error"),
    }
    // _guard is dropped here (and on any earlier return), releasing the slot.
}

// ---------------------------------------------------------------------------
// Model dispatch
// ---------------------------------------------------------------------------

/// Returns `true` when `e` is the specific 400 indicating the model needs the
/// Responses API (`/v1/responses`) instead of Chat Completions.
fn is_unsupported_via_chat_completions(e: &anyhow::Error) -> bool {
    e.downcast_ref::<copilot::CopilotHttpError>()
        .is_some_and(|he| {
            he.status.as_u16() == 400 && he.body.contains("unsupported_api_for_model")
        })
}

/// Send one model turn, with automatic endpoint selection and fallback.
///
/// 1. The base URL is derived from the `proxy-ep` field in the Copilot JWT
///    so each user reaches their account-specific gateway automatically.
/// 2. All models try `/v1/chat/completions` first.
/// 3. If the model returns `400 unsupported_api_for_model` (e.g. gpt-5.x),
///    the request is transparently retried via `/v1/responses` (OpenAI
///    Responses API), which uses a different request/response format handled
///    by `crate::responses`.
/// 4. Auth errors (401/403) evict the token cache and retry once.
async fn invoke_model<W>(
    state: &DaemonState,
    model: &str,
    messages: &[Message],
    tools: &[serde_json::Value],
    max_tokens: usize,
    writer: &mut W,
) -> Result<copilot::CopilotResponse>
where
    W: AsyncWriteExt + Unpin,
{
    let tok = state
        .tokens
        .get(&state.http)
        .await
        .context("refreshing Copilot API token")?;

    let result = copilot::stream_chat(
        &state.http,
        &tok.value,
        &tok.base_url,
        model,
        messages,
        tools,
        max_tokens,
        writer,
    )
    .await;

    match result {
        Ok(r) => Ok(r),

        // Model requires Responses API — retry transparently using the same
        // per-account base URL so enterprise gateways are reached correctly.
        Err(ref e) if is_unsupported_via_chat_completions(e) => {
            tracing::debug!(
                model,
                "model not accessible via /chat/completions; retrying via Responses API"
            );
            let r = crate::responses::stream_chat(
                &state.http,
                &tok.value,
                &tok.base_url,
                model,
                messages,
                tools,
                max_tokens,
                writer,
            )
            .await;
            // The Responses API can also return 401/403 on token expiry.
            // Evict the cache and retry once with a fresh token.
            match r {
                Ok(r) => Ok(r),
                Err(e2) => {
                    let is_auth = e2
                        .downcast_ref::<copilot::CopilotHttpError>()
                        .is_some_and(|he| matches!(he.status.as_u16(), 401 | 403));
                    if is_auth {
                        tracing::warn!(
                            error = %e2,
                            "Responses API auth error; evicting token and retrying"
                        );
                        state.tokens.invalidate().await;
                        let fresh = state
                            .tokens
                            .get(&state.http)
                            .await
                            .context("fetching fresh token after Responses API auth error")?;
                        crate::responses::stream_chat(
                            &state.http,
                            &fresh.value,
                            &fresh.base_url,
                            model,
                            messages,
                            tools,
                            max_tokens,
                            writer,
                        )
                        .await
                    } else {
                        Err(e2)
                    }
                }
            }
        }

        // Auth error — evict the token cache and retry once with a fresh token.
        Err(e) => {
            let is_auth_err = e
                .downcast_ref::<copilot::CopilotHttpError>()
                .is_some_and(|he| matches!(he.status.as_u16(), 401 | 403));
            if is_auth_err {
                tracing::warn!(error = %e, "Copilot auth error; evicting token and retrying");
                state.tokens.invalidate().await;
                let fresh = state
                    .tokens
                    .get(&state.http)
                    .await
                    .context("fetching fresh Copilot token after auth error")?;
                // Use Chat Completions with fresh token; Responses API fallback
                // applies here too if needed.
                let r2 = copilot::stream_chat(
                    &state.http,
                    &fresh.value,
                    &fresh.base_url,
                    model,
                    messages,
                    tools,
                    max_tokens,
                    writer,
                )
                .await;
                match r2 {
                    Ok(r) => Ok(r),
                    Err(ref e2) if is_unsupported_via_chat_completions(e2) => {
                        crate::responses::stream_chat(
                            &state.http,
                            &fresh.value,
                            &fresh.base_url,
                            model,
                            messages,
                            tools,
                            max_tokens,
                            writer,
                        )
                        .await
                    }
                    Err(e2) => Err(e2),
                }
            } else {
                Err(e)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Agentic loop
// ---------------------------------------------------------------------------

/// Maximum number of Unicode scalar values kept from a single historical
/// memory message injected into a request.  Prevents accumulated long tool
/// outputs from blowing the model's context window after extended use.
const MAX_HISTORY_CHARS: usize = 4_000;

/// Maximum number of Unicode scalar values kept from a single tool-call
/// output within the current agentic loop iteration.  Large file reads and
/// shell command outputs are truncated before being fed back to the model.
const MAX_TOOL_OUTPUT_CHARS: usize = 8_000;

/// Drive the conversation until Copilot responds with `finish_reason: stop`
/// (or an error).  Executes tool calls and feeds results back in a loop.
///
/// `steer_rx` receives user corrections injected mid-flight from the CLI's
/// stdin.  Pending steers are drained **at the start of every loop iteration**
/// (before the model call), so corrections are visible regardless of whether
/// the previous turn used tools or returned Stop/Length.  A
/// [`Response::SteerAck`] frame is sent for each message consumed.
///
/// `stream_chat` retries 5xx, 429, and transport errors internally up to its
/// `MAX_RETRIES`, but those errors can still surface here if retries are
/// exhausted or if a non-retryable error occurs.
pub(crate) async fn run_agentic_loop<W>(
    state: &DaemonState,
    model: &str,
    mut messages: Vec<Message>,
    writer: &mut W,
    steer_rx: &mut tokio::sync::mpsc::Receiver<Option<String>>,
    include_spawn_agent: bool,
) -> Result<(String, usize, Vec<Message>)>
where
    W: AsyncWriteExt + Unpin,
{
    let schemas = tools::tool_schemas(include_spawn_agent);
    let final_text;
    let mut tools_were_used = false;
    let mut conclusion_nudge_sent = false;
    let mut last_prompt_tokens: usize;

    loop {
        // Drain any steering corrections that arrived since the last model
        // call (covers non-tool turns and the time between tool completion
        // and the next iteration).
        while let Ok(steer_msg) = steer_rx.try_recv() {
            if let Some(msg) = steer_msg {
                messages.push(Message::user(msg));
                write_frame(writer, &Response::SteerAck).await?;
            }
            // None = interrupt-only (no user message to inject; loop already
            // skipped the tool chain at execution time).  No SteerAck is sent.
        }

        // All models route through the Copilot JWT endpoint; invoke_model
        // falls back to the Responses API automatically when needed.
        // Token management and auth-error retry are handled inside invoke_model.
        let resp = invoke_model(
            state,
            model,
            &messages,
            &schemas,
            response_max_tokens(model),
            writer,
        )
        .await?;

        last_prompt_tokens = resp.prompt_tokens;

        let finish_label = match &resp.finish_reason {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::Other(_) => "other",
        };
        tracing::debug!(
            finish_reason = finish_label,
            text_len = resp.text.len(),
            tools_were_used,
            "model turn complete"
        );

        match resp.finish_reason {
            FinishReason::Stop | FinishReason::Length => {
                // Check if the model is asking for clarification via the
                // [WAITING_FOR_INPUT] protocol marker.  Use trim_start() so
                // detection is robust to any leading whitespace the model emits
                // before the marker (parse_sse_stream suppresses the marker
                // only when it sits at byte 0, but detection must not miss it
                // when it doesn't).
                if resp.text.trim_start().starts_with(copilot::WAITING_MARKER) {
                    let clarification_prompt = resp
                        .text
                        .trim_start()
                        .strip_prefix(copilot::WAITING_MARKER)
                        .unwrap_or(&resp.text)
                        .trim()
                        .to_owned();

                    // Tell the client we need input.  The marker and clarification
                    // text were already suppressed/streamed by parse_sse_stream,
                    // so pass an empty prompt here to avoid duplicating on screen.
                    write_frame(
                        writer,
                        &Response::WaitingForInput {
                            prompt: String::new(),
                        },
                    )
                    .await?;

                    // Record the assistant's question in history using the
                    // stripped text (marker removed).  Use None when the
                    // stripped text is empty to match the existing pattern
                    // used in the tool-call branch (empty assistant text → None).
                    let cp_text = if clarification_prompt.is_empty() {
                        None
                    } else {
                        Some(clarification_prompt.clone())
                    };
                    messages.push(Message::assistant(cp_text, vec![]));

                    // Block until the user replies via the steering channel.
                    // Timeout after 5 minutes to avoid infinite hangs.
                    // An interrupt (None) now breaks out immediately so the session
                    // is not left stuck when the user cancels the reply prompt.
                    let steer_result = match tokio::time::timeout(
                        std::time::Duration::from_secs(300),
                        steer_rx.recv(),
                    )
                    .await
                    {
                        Ok(Some(Some(reply))) => Ok(Some(reply)),
                        Ok(Some(None)) => {
                            // Interrupt: user cancelled the reply prompt.
                            tracing::debug!(
                                context = "waiting_for_input_reply",
                                "interrupt received; aborting waiting-for-input"
                            );
                            Ok(None)
                        }
                        Ok(None) => Ok(None),
                        Err(e) => Err(e),
                    };
                    match steer_result {
                        Ok(Some(user_reply)) => {
                            messages.push(Message::user(user_reply));
                            write_frame(writer, &Response::SteerAck).await?;
                            continue;
                        }
                        Ok(None) => {
                            // Channel closed — client disconnected.
                            tracing::debug!(
                                reason = "client_disconnected",
                                context = "waiting_for_input_reply",
                                tools_were_used,
                                "session end"
                            );
                            final_text = clarification_prompt;
                            break;
                        }
                        Err(_timeout) => {
                            tracing::debug!(
                                reason = "input_timeout_300s",
                                context = "waiting_for_input_reply",
                                tools_were_used,
                                "session end"
                            );
                            write_frame(
                                writer,
                                &Response::Text {
                                    chunk: "\n[timed out waiting for input]\n".into(),
                                },
                            )
                            .await?;
                            final_text = clarification_prompt;
                            break;
                        }
                    }
                }

                // Multi-turn heuristic: if the model's response ends with a
                // question mark, treat it as an implicit request for
                // clarification and keep the session alive for follow-up.
                if resp.text.trim_end().ends_with('?') {
                    let question = resp.text.trim().to_owned();

                    // The full LLM text was already streamed to the client as
                    // Response::Text chunks by stream_chat.  Send an empty
                    // prompt here so the client only shows the '>' cursor
                    // without duplicating the already-displayed question text.
                    write_frame(
                        writer,
                        &Response::WaitingForInput {
                            prompt: String::new(),
                        },
                    )
                    .await?;

                    // Use the trimmed question (no marker contamination) in history.
                    messages.push(Message::assistant(Some(question.clone()), vec![]));

                    // A bare interrupt (Ctrl-C / None) is ignored while the
                    // session is already waiting for a question reply — keep waiting.
                    let steer_result = 'wait_question: loop {
                        match tokio::time::timeout(
                            std::time::Duration::from_secs(300),
                            steer_rx.recv(),
                        )
                        .await
                        {
                            Ok(Some(Some(reply))) => break 'wait_question Ok(Some(reply)),
                            Ok(Some(None)) => {
                                tracing::debug!(
                                    context = "waiting_for_question_reply",
                                    "interrupt ignored while waiting for question reply"
                                );
                                continue 'wait_question;
                            }
                            Ok(None) => break 'wait_question Ok(None),
                            Err(e) => break 'wait_question Err(e),
                        }
                    };
                    match steer_result {
                        Ok(Some(user_reply)) => {
                            messages.push(Message::user(user_reply));
                            write_frame(writer, &Response::SteerAck).await?;
                            continue;
                        }
                        Ok(None) => {
                            tracing::debug!(
                                reason = "client_disconnected",
                                context = "waiting_for_question_reply",
                                tools_were_used,
                                "session end"
                            );
                            final_text = question;
                            break;
                        }
                        Err(_timeout) => {
                            tracing::debug!(
                                reason = "input_timeout_300s",
                                context = "waiting_for_question_reply",
                                tools_were_used,
                                "session end"
                            );
                            write_frame(
                                writer,
                                &Response::Text {
                                    chunk: "\n[timed out waiting for input]\n".into(),
                                },
                            )
                            .await?;
                            final_text = question;
                            break;
                        }
                    }
                }

                // Safety net: if the model ends silently after having used tools,
                // inject one follow-up asking for a conclusion rather than letting
                // the session end without visible output.
                if resp.text.is_empty() && tools_were_used && !conclusion_nudge_sent {
                    tracing::debug!("session: empty Stop after tool use — nudging for conclusion");
                    conclusion_nudge_sent = true;
                    messages.push(Message::assistant(None, vec![]));
                    messages.push(Message::user(
                        "Please summarise what you just did and the outcome.".to_owned(),
                    ));
                    continue;
                }

                let reason = if resp.text.is_empty() {
                    "stop_empty_text"
                } else if finish_label == "length" {
                    "length_limit"
                } else {
                    "stop"
                };
                tracing::debug!(
                    reason,
                    finish_reason = finish_label,
                    text_len = resp.text.len(),
                    tools_were_used,
                    conclusion_nudge_sent,
                    text_preview = &resp.text.chars().take(80).collect::<String>(),
                    "session end"
                );
                // Notify the user when the model hit the output token limit.
                if finish_label == "length" {
                    write_frame(
                        writer,
                        &Response::Text {
                            chunk: "\n[response truncated — output token limit reached]\n".into(),
                        },
                    )
                    .await?;
                }

                final_text = resp.text;
                if !final_text.is_empty() {
                    messages.push(Message::assistant(Some(final_text.clone()), vec![]));
                }
                break;
            }

            FinishReason::ToolCalls => {
                tools_were_used = true;
                // Destructure resp up front to avoid a partial move: consuming
                // resp.text while still needing to borrow resp.tool_calls.
                let resp_text = resp.text;
                let tool_calls = resp.tool_calls;
                let api_calls: Vec<ApiToolCall> = tool_calls
                    .iter()
                    .map(|tc| ApiToolCall {
                        id: tc.id.clone(),
                        kind: "function".into(),
                        function: ApiToolCallFunction {
                            name: tc.name.clone(),
                            arguments: tc.arguments.clone(),
                        },
                    })
                    .collect();

                let assistant_text = if resp_text.is_empty() {
                    None
                } else {
                    Some(resp_text)
                };
                messages.push(Message::assistant(assistant_text, api_calls));

                let tool_calls_snapshot = &tool_calls;

                // When every call in this batch is spawn_agent AND all of them
                // opt in with `parallel=true`, run them concurrently.
                // Default is sequential — the caller must explicitly request
                // parallel execution for each call in the batch.
                let all_spawn_agent = tool_calls_snapshot.len() > 1
                    && tool_calls_snapshot
                        .iter()
                        .all(|tc| tc.name == "spawn_agent")
                    && tool_calls_snapshot.iter().all(|tc| {
                        tc.parse_args()
                            .ok()
                            .and_then(|v| v["parallel"].as_bool())
                            .unwrap_or(false)
                    });

                if all_spawn_agent {
                    // Emit all ToolUse frames sequentially first: the writer
                    // is not Sync so we cannot interleave writes.
                    for tc in tool_calls_snapshot.iter() {
                        write_frame(
                            writer,
                            &Response::ToolUse {
                                name: tc.name.clone(),
                                detail: String::new(),
                            },
                        )
                        .await?;
                    }

                    // Execute all spawn_agent calls concurrently, collecting
                    // results in original order.
                    // Steer interrupts are not checked here — they only apply
                    // to the sequential path where we can abort cleanly before
                    // the next tool runs.
                    let results = futures_util::future::join_all(tool_calls_snapshot.iter().map(
                        |tc| async move {
                            let args = match tc.parse_args() {
                                Ok(v) => v,
                                Err(e) => {
                                    tracing::warn!(
                                        tool = %tc.name,
                                        error = %e,
                                        "bad tool arguments"
                                    );
                                    return format!("argument error: {e:#}");
                                }
                            };
                            match state.executor.execute(&tc.name, args).await {
                                Ok(output) => {
                                    tracing::debug!(
                                        tool = %tc.name,
                                        output_len = output.len(),
                                        "tool succeeded"
                                    );
                                    truncate_chars(output, MAX_TOOL_OUTPUT_CHARS)
                                }
                                Err(e) => {
                                    tracing::warn!(tool = %tc.name, error = %e, "tool failed");
                                    format!("error: {e:#}")
                                }
                            }
                        },
                    ))
                    .await;

                    for (tc, result) in tool_calls_snapshot.iter().zip(results) {
                        messages.push(Message::tool_result(&tc.id, result));
                    }
                } else {
                    // Sequential path: honour steer interrupts between tools.
                    let mut interrupted_at: Option<usize> = None;
                    let mut steer_text: Option<String> = None;

                    for (i, tc) in tool_calls_snapshot.iter().enumerate() {
                        // Check for a mid-execution steer before running this tool.
                        // If the user pressed Ctrl-C and typed a correction, honour
                        // it immediately: skip this and all remaining tools.
                        if let Ok(msg) = steer_rx.try_recv() {
                            tracing::debug!(
                                "mid-execution steer received; interrupting tool chain at index {i}"
                            );
                            // Stash the steer text — we must emit placeholder tool_result
                            // entries for ALL skipped calls before appending the user message,
                            // because the API requires assistant(tool_calls) →
                            // tool(tool_result…) → user/assistant ordering.
                            // msg is None for interrupt-only (no text to inject).
                            steer_text = msg;
                            if steer_text.is_some() {
                                write_frame(writer, &Response::SteerAck).await?;
                            }
                            interrupted_at = Some(i);
                            break;
                        }

                        tracing::debug!(tool = %tc.name, "executing tool");

                        let tool_detail = {
                            let args: serde_json::Value = serde_json::from_str(&tc.arguments)
                                .unwrap_or(serde_json::Value::Null);
                            match tc.name.as_str() {
                                "shell_command" => args
                                    .get("command")
                                    .and_then(|v| v.as_str())
                                    .map(|s| {
                                        if s.len() > 80 {
                                            format!("{}…", &s[..80])
                                        } else {
                                            s.to_string()
                                        }
                                    })
                                    .unwrap_or_default(),
                                "read_file" => args
                                    .get("path")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or_default()
                                    .to_string(),
                                "edit_file" => args
                                    .get("path")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or_default()
                                    .to_string(),
                                "tmux_send_keys" => args
                                    .get("keys")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or_default()
                                    .to_string(),
                                "tmux_capture_pane" => args
                                    .get("target")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or_default()
                                    .to_string(),
                                _ => String::new(),
                            }
                        };
                        write_frame(
                            writer,
                            &Response::ToolUse {
                                name: tc.name.clone(),
                                detail: tool_detail,
                            },
                        )
                        .await?;

                        let args = match tc.parse_args() {
                            Ok(v) => v,
                            Err(e) => {
                                tracing::warn!(tool = %tc.name, error = %e, "bad tool arguments");
                                messages.push(Message::tool_result(
                                    &tc.id,
                                    format!("argument error: {e:#}"),
                                ));
                                continue;
                            }
                        };

                        let result = match state.executor.execute(&tc.name, args).await {
                            Ok(output) => {
                                tracing::debug!(
                                    tool = %tc.name,
                                    output_len = output.len(),
                                    "tool succeeded"
                                );
                                truncate_chars(output, MAX_TOOL_OUTPUT_CHARS)
                            }
                            Err(e) => {
                                tracing::warn!(tool = %tc.name, error = %e, "tool failed");
                                format!("error: {e:#}")
                            }
                        };

                        messages.push(Message::tool_result(&tc.id, result));
                    }

                    // If the user injected a steer mid-chain, push placeholder
                    // tool results for the tools that were never executed.  The
                    // OpenAI API requires a tool_result for every tool_call in the
                    // assistant message, even if we didn't actually run them.
                    if let Some(skip_from) = interrupted_at {
                        for tc in &tool_calls_snapshot[skip_from..] {
                            messages.push(Message::tool_result(
                                &tc.id,
                                "[interrupted by user before execution]",
                            ));
                        }
                        // When the interrupt carried steer text, use it directly.
                        // When it was interrupt-only (None), block until the user
                        // sends a correction before starting the next model turn —
                        // proceeding immediately would race against incoming steer
                        // text and re-enter the model without user guidance.
                        let effective_steer = if let Some(text) = steer_text {
                            Some(text)
                        } else {
                            loop {
                                match steer_rx.recv().await {
                                    Some(Some(t)) => {
                                        write_frame(writer, &Response::SteerAck).await?;
                                        break Some(t);
                                    }
                                    Some(None) => continue, // another bare interrupt; keep waiting
                                    None => break None,     // channel closed; proceed without steer
                                }
                            }
                        };
                        if let Some(text) = effective_steer {
                            messages.push(Message::user(text));
                        }
                    }
                }

                // Steers that arrived during tool execution are drained at
                // the top of the next loop iteration, before the model call.
            }

            FinishReason::Other(ref reason) => {
                tracing::warn!(finish_reason = %reason, "session end: unexpected finish reason");
                let warning = format!("\n[stopped: unexpected finish reason '{reason}']");
                write_frame(writer, &Response::Text { chunk: warning }).await?;
                final_text = resp.text;
                if !final_text.is_empty() {
                    messages.push(Message::assistant(Some(final_text.clone()), vec![]));
                }
                break;
            }
        }
    }

    Ok((final_text, last_prompt_tokens, messages))
}

// ---------------------------------------------------------------------------
// Skill-file injection
// ---------------------------------------------------------------------------

/// Read global config files from `~/.amaebi/` and inject them as system messages.
///
/// Loads `AGENTS.md` and `SOUL.md` from the user's amaebi home directory
/// (`~/.amaebi/`).  Files that do not exist or are whitespace-only are
/// silently skipped.  No per-project or CWD-relative files are read.
pub(crate) async fn inject_skill_files(messages: &mut Vec<Message>) {
    let home = match amaebi_home() {
        Ok(p) => p,
        Err(e) => {
            tracing::debug!(error = %e, "could not resolve amaebi home for skill injection");
            return;
        }
    };
    inject_skill_files_from(messages, &home).await;
}

/// Internal helper used by [`inject_skill_files`] and tests.
async fn inject_skill_files_from(messages: &mut Vec<Message>, amaebi_home: &std::path::Path) {
    // Always-injected files: loaded unconditionally at the start of every turn.
    const FIXED_FILES: &[(&str, &str)] =
        &[("AGENTS.md", "## Agent Guidelines"), ("SOUL.md", "## Soul")];
    for (filename, header) in FIXED_FILES {
        let path = amaebi_home.join(filename);
        match tokio::fs::read_to_string(&path).await {
            Ok(content) => {
                let trimmed = content.trim();
                if !trimmed.is_empty() {
                    messages.push(Message::system(format!("{header}\n\n{trimmed}")));
                    tracing::info!(
                        file = %path.display(),
                        header,
                        bytes = trimmed.len(),
                        "injected skill file"
                    );
                } else {
                    tracing::debug!(
                        file = %path.display(),
                        "config file empty after trimming; skipping injection"
                    );
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                tracing::debug!(file = %path.display(), error = %e, "could not read config file");
            }
        }
    }

    // On-demand operations docs: not preloaded, but the agent needs absolute paths
    // to use the read_file tool.  Inject a single pointer message listing whichever
    // files are present so the agent can load them when the task requires it.
    const ONDEMAND_FILES: &[&str] = &[
        "OPERATIONS_INDEX.md",
        "DEPLOYMENT.md",
        "CONFIG_REFERENCE.md",
        "RUNBOOK.md",
    ];
    let mut available: Vec<String> = Vec::new();
    for filename in ONDEMAND_FILES {
        let path = amaebi_home.join(filename);
        match tokio::fs::metadata(&path).await {
            Ok(meta) if meta.is_file() => available.push(path.display().to_string()),
            Ok(_) => {
                tracing::debug!(file = %path.display(), "on-demand path exists but is not a regular file; skipping");
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                tracing::debug!(file = %path.display(), error = %e, "could not stat on-demand file");
            }
        }
    }
    if !available.is_empty() {
        let list = available
            .iter()
            .map(|p| format!("- {p}"))
            .collect::<Vec<_>>()
            .join("\n");
        messages.push(Message::system(format!(
            "## On-demand Operations Docs\n\n\
             The following files can be loaded with read_file when the task involves \
             deployment, configuration, or troubleshooting:\n\n{list}"
        )));
        tracing::info!(
            files = ?available,
            count = available.len(),
            "injected on-demand operations docs pointer"
        );
    }
}
// ---------------------------------------------------------------------------
// Message construction
// ---------------------------------------------------------------------------

/// Build a message list for a Chat or Resume turn.
///
/// `past_summaries` — compacted summaries from other sessions (cross-session context).
/// `own_summary`    — this session's running summary, injected before the history
///                    rows so the model sees: summary → history → prompt.
///
/// All `history` rows are included without a sliding-window cap.  Callers are
/// responsible for trimming the history slice to a token budget before calling
/// (see the pre-flight check in `Request::Chat` / `Request::Resume` handlers).
pub(crate) fn build_messages(
    prompt: &str,
    tmux_pane: Option<&str>,
    history: &[memory_db::DbMemoryEntry],
    past_summaries: &[String],
    own_summary: Option<&str>,
) -> Vec<Message> {
    let mut system = "You are a helpful, concise AI assistant embedded in a tmux terminal. \
                      Answer in plain text; avoid markdown unless the user asks for it. \
                      You have tools available to inspect the terminal, run commands, \
                      and read or edit files — use them when they help you answer accurately. \
                      After using any tool, you MUST always follow up with a text response \
                      summarising what you did and the outcome — never end silently after a tool call."
        .to_owned();

    if let Some(pane) = tmux_pane {
        system.push_str(&format!(" The user's active tmux pane is {pane}."));
    }

    if !past_summaries.is_empty() {
        system.push_str("\n\nSummaries from past sessions (oldest first):\n");
        for s in past_summaries {
            system.push_str(&truncate_chars(s.clone(), MAX_SUMMARY_CHARS));
            system.push('\n');
        }
    }

    let mut messages = vec![Message::system(system)];

    // If this session has been partially compacted, prepend the running summary
    // so the model knows what happened before the history window.
    if let Some(summary) = own_summary {
        let summary = truncate_chars(summary.to_owned(), MAX_SUMMARY_CHARS * 2);
        messages.push(Message::user(
            "[Summary of earlier in this session]".to_owned(),
        ));
        messages.push(Message::assistant(Some(summary), vec![]));
    }

    for entry in history {
        let content = truncate_chars(entry.content.clone(), MAX_HISTORY_CHARS);
        match entry.role.as_str() {
            "user" => messages.push(Message::user(content)),
            "assistant" => messages.push(Message::assistant(Some(content), vec![])),
            _ => {} // skip unknown roles
        }
    }

    messages.push(Message::user(prompt.to_owned()));
    messages
}

// ---------------------------------------------------------------------------
// Cron scheduler
// ---------------------------------------------------------------------------

/// Background task: ticks every 60 seconds, fires any due cron jobs.
///
/// Never returns (runs for the lifetime of the daemon).  Each due job is
/// spawned as an independent `tokio::task` so that slow LLM calls do not
/// block subsequent ticks.
///
/// A `running_jobs` guard prevents a second spawn of the same job if the
/// previous invocation is still running (e.g. job takes >60 s with a 1-min
/// schedule).  The guard is cleared when `run_cron_job` returns.
async fn run_cron_scheduler(state: Arc<DaemonState>) {
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
    // Skip missed ticks — don't fire a backlog of jobs if the daemon was paused.
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // Set of job IDs currently executing; prevents duplicate concurrent runs.
    let running_jobs: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));

    loop {
        interval.tick().await;

        let now = chrono::Utc::now();
        let jobs = match cron::load_jobs() {
            Ok(j) => j,
            Err(e) => {
                tracing::warn!(error = %e, "cron: failed to load jobs from cron.db");
                continue;
            }
        };

        for job in cron::due_jobs(&jobs, &now) {
            // Skip if a previous run of this job is still in flight.
            {
                let mut guard = running_jobs.lock().unwrap_or_else(|p| p.into_inner());
                if guard.contains(&job.id) {
                    tracing::debug!(id = %job.id, "cron: job still running, skipping tick");
                    continue;
                }
                guard.insert(job.id.clone());
            }

            // Stamp last_run BEFORE spawning so that if the daemon restarts
            // while the job is in flight, due_jobs won't immediately re-fire
            // it on the next tick (the in-memory running_jobs guard is lost
            // across restarts, but the DB timestamp persists).
            let now_str = now.to_rfc3339();
            if let Err(e) = cron::update_last_run(&job.id, &now_str) {
                tracing::warn!(error = %e, id = %job.id, "cron: failed to pre-stamp last_run");
            }

            let state = Arc::clone(&state);
            let running = Arc::clone(&running_jobs);
            tokio::spawn(async move {
                run_cron_job(Arc::clone(&state), &job).await;
                // Clear the guard so the next tick can re-fire the job.
                let mut guard = running.lock().unwrap_or_else(|p| p.into_inner());
                guard.remove(&job.id);
            });
        }
    }
}

/// Execute a single cron job: runs the agentic loop, saves output to inbox.
async fn run_cron_job(state: Arc<DaemonState>, job: &cron::CronJob) {
    tracing::info!(id = %job.id, description = %job.description, "cron: firing job");

    // Generate a fresh session UUID for this run.
    let session_id = uuid::Uuid::new_v4().to_string();

    // Resolve model from env var (same default as CLI client).
    let model = std::env::var("AMAEBI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());

    let mut messages = build_messages(&job.description, None, &[], &[], None);
    inject_skill_files(&mut messages).await;
    // Cron jobs are non-interactive: drop the sender immediately so steer_rx.recv()
    // returns None at once if the model ends with '?', rather than timing out.
    let mut sink = tokio::io::sink();
    let (_, mut steer_rx) = tokio::sync::mpsc::channel::<Option<String>>(1);

    let result = run_agentic_loop(&state, &model, messages, &mut sink, &mut steer_rx, true).await;

    let (output, run_ok) = match result {
        Ok((final_text, _, _)) => {
            store_conversation(&state, &session_id, &job.description, &final_text).await;
            (final_text, true)
        }
        Err(e) => {
            tracing::error!(error = %e, id = %job.id, "cron: job failed");
            (format!("[error] {e:#}"), false)
        }
    };

    let task_desc = truncate_chars(job.description.clone(), 200);

    // Persist to inbox.
    match InboxStore::open() {
        Ok(inbox) => {
            if let Err(e) = inbox.save_report(&session_id, &task_desc, &output) {
                tracing::error!(error = %e, "cron: failed to save inbox report");
            }
        }
        Err(e) => tracing::error!(error = %e, "cron: failed to open inbox"),
    }

    // Update last_run timestamp so due_jobs won't fire this job again this minute.
    if run_ok {
        let now_str = chrono::Utc::now().to_rfc3339();
        if let Err(e) = cron::update_last_run(&job.id, &now_str) {
            tracing::warn!(error = %e, id = %job.id, "cron: failed to update last_run");
        }
    }

    tracing::info!(id = %job.id, "cron: job complete");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_db_entry(id: i64, role: &str, content: &str) -> memory_db::DbMemoryEntry {
        memory_db::DbMemoryEntry {
            id,
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            session_id: "test".to_string(),
            role: role.to_string(),
            content: content.to_string(),
            summary: String::new(),
        }
    }

    /// Create a history of `n` user/assistant pairs as DbMemoryEntry rows.
    fn make_history(n: usize) -> Vec<memory_db::DbMemoryEntry> {
        let mut entries = Vec::with_capacity(n * 2);
        for i in 0..n {
            entries.push(make_db_entry((i * 2) as i64, "user", &format!("u{i}")));
            entries.push(make_db_entry(
                (i * 2 + 1) as i64,
                "assistant",
                &format!("a{i}"),
            ));
        }
        entries
    }

    // ------------------------------------------------------------------
    // truncate_chars tests
    // ------------------------------------------------------------------

    #[test]
    fn truncate_chars_short_string_unchanged() {
        let s = "hello".to_owned();
        assert_eq!(truncate_chars(s, 10), "hello");
    }

    #[test]
    fn truncate_chars_at_limit_unchanged() {
        let s = "hello".to_owned();
        assert_eq!(truncate_chars(s, 5), "hello");
    }

    #[test]
    fn truncate_chars_over_limit_appends_marker() {
        let s = "hello world extra text here".to_owned();
        let result = truncate_chars(s, 20);
        assert!(result.ends_with("…[truncated]"), "should end with marker");
        assert_eq!(result.chars().count(), 20);
    }

    #[test]
    fn truncate_chars_total_length_never_exceeds_max() {
        let s = "a".repeat(100);
        for max in [14, 20, 50, 99] {
            let result = truncate_chars(s.clone(), max);
            assert!(
                result.chars().count() <= max,
                "max={max}: got {} chars",
                result.chars().count()
            );
        }
    }

    #[test]
    fn truncate_chars_max_smaller_than_marker_returns_partial_marker() {
        let result = truncate_chars("hello world".to_owned(), 3);
        assert_eq!(result.chars().count(), 3);
        assert!(result.starts_with('…'));
    }

    #[test]
    fn truncate_chars_respects_unicode_boundaries() {
        let s = "日本語テスト".repeat(5);
        let result = truncate_chars(s, 20);
        assert!(result.chars().count() <= 20);
        assert!(result.ends_with("…[truncated]"));
    }

    #[test]
    fn truncate_chars_empty_string_unchanged() {
        assert_eq!(truncate_chars(String::new(), 10), "");
    }

    #[test]
    fn truncate_chars_multibyte_safe() {
        let s = "日".repeat(25);
        let result = truncate_chars(s, 20);
        assert!(result.chars().count() <= 20);
        assert!(result.ends_with("…[truncated]"));
    }

    // ---- build_messages tests ----------------------------------------------

    #[test]
    fn build_messages_empty_history() {
        let msgs = build_messages("hello", None, &[], &[], None);
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn build_messages_injects_history_rows() {
        let history = make_history(2); // 4 rows: u0, a0, u1, a1
        let msgs = build_messages("q3", None, &history, &[], None);
        // system + 4 history rows + user
        assert_eq!(msgs.len(), 6);
    }

    #[test]
    fn build_messages_all_history_included() {
        // build_messages no longer caps — all rows are included.
        let history = make_history(10);
        let msgs = build_messages("new", None, &history, &[], None);
        // system + 20 history rows + user
        assert_eq!(msgs.len(), 22);
    }

    #[test]
    fn build_messages_tmux_pane_in_system() {
        let msgs = build_messages("prompt", Some("%3"), &[], &[], None);
        let content = msgs[0].content.as_deref().unwrap_or("");
        assert!(
            content.contains("%3"),
            "system prompt should mention the pane"
        );
    }

    #[test]
    fn build_messages_injects_past_summaries_into_system() {
        let summaries = vec![
            "- Fixed the auth bug.".to_owned(),
            "- Added cron.".to_owned(),
        ];
        let msgs = build_messages("hi", None, &[], &summaries, None);
        let system = msgs[0].content.as_deref().unwrap_or("");
        assert!(
            system.contains("Fixed the auth bug"),
            "summaries must appear in system message"
        );
        assert!(
            system.contains("Added cron"),
            "all summaries must be injected"
        );
    }

    #[test]
    fn build_messages_own_summary_inserted_before_history() {
        let history = make_history(1); // 2 rows: u0, a0
        let msgs = build_messages("q", None, &history, &[], Some("- Did X earlier."));
        // system + [user summary label + assistant summary] + 2 history rows + user
        assert_eq!(msgs.len(), 6);
        // The summary pair comes before the history rows.
        let summary_placeholder = msgs[1].content.as_deref().unwrap_or("");
        assert!(
            summary_placeholder.contains("Summary of earlier"),
            "summary label must appear before history"
        );
    }

    // ---- token budget tests -----

    #[test]
    fn count_message_tokens_increases_with_content() {
        let short = vec![Message::system("hi"), Message::user("hello")];
        let long = vec![Message::system("hi"), Message::user("hello ".repeat(200))];
        assert!(
            count_message_tokens(&long) > count_message_tokens(&short),
            "longer content must produce higher token count"
        );
    }

    #[test]
    fn count_message_tokens_empty_list() {
        // Empty list: only the 3 priming tokens.
        assert_eq!(count_message_tokens(&[]), 3);
    }

    #[test]
    fn context_limit_for_known_models() {
        assert_eq!(context_limit_for_model("gpt-4o"), 128_000);
        assert_eq!(context_limit_for_model("gpt-4o-mini"), 128_000);
        assert_eq!(context_limit_for_model("gpt-4-turbo"), 128_000);
        assert_eq!(context_limit_for_model("gpt-4"), 8_192);
        assert_eq!(context_limit_for_model("gpt-3.5-turbo"), 16_385);
        assert_eq!(context_limit_for_model("o1-preview"), 200_000);
        assert_eq!(context_limit_for_model("o3-mini"), 200_000);
        assert_eq!(context_limit_for_model("claude-3-5-sonnet"), 200_000);
    }

    #[test]
    fn context_limit_gemini_models() {
        assert_eq!(context_limit_for_model("gemini-2.0-flash"), 1_048_576);
        assert_eq!(context_limit_for_model("gemini-2.0-flash-001"), 1_048_576);
        assert_eq!(context_limit_for_model("gemini-1.5-pro"), 2_097_152);
        assert_eq!(context_limit_for_model("gemini-1.5-flash"), 1_048_576);
        assert_eq!(context_limit_for_model("gemini-1.5-flash-8b"), 1_048_576);
        assert_eq!(context_limit_for_model("gemini-1.0-pro"), 32_768);
        // Catch-all for unrecognised Gemini variants.
        assert_eq!(context_limit_for_model("gemini-flash"), 128_000);
        assert_eq!(context_limit_for_model("gemini-pro"), 128_000);
    }

    #[test]
    fn context_limit_unknown_model_is_conservative() {
        // Unknown models fall back to 32k — never 0, never larger than any known model.
        let limit = context_limit_for_model("some-future-model-xyz");
        assert_eq!(limit, 32_768);
    }

    #[test]
    #[serial_test::serial]
    fn compaction_threshold_is_below_context_limit() {
        // Unset the override so this test always exercises the default formula.
        std::env::remove_var("AMAEBI_COMPACTION_THRESHOLD");
        for model in &["gpt-4o", "gpt-4", "gpt-3.5-turbo", "o1", "unknown-model"] {
            let t = compaction_threshold_tokens(model);
            let available =
                context_limit_for_model(model).saturating_sub(response_max_tokens(model));
            // When context_limit minus response_max_tokens(model) leaves ≤1
            // token, (available as f64 * COMPACTION_THRESHOLD) truncates to 0,
            // so compaction_threshold_tokens returns usize::MAX to disable
            // compaction rather than triggering it on every turn.
            // Skip the < available / > 0 assertions for those.
            if available <= 1 {
                assert_eq!(
                    t,
                    usize::MAX,
                    "model={model}: tiny-context model should return usize::MAX to disable compaction"
                );
                continue;
            }
            assert!(
                t < available,
                "model={model}: threshold must be below available input budget"
            );
            assert!(t > 0, "model={model}: threshold must be positive");
        }
    }

    // ---- inject_skill_files tests -----

    #[tokio::test]
    async fn skill_files_agents_and_soul_injected_from_home() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "agent guidelines").unwrap();
        std::fs::write(dir.path().join("SOUL.md"), "soul content").unwrap();
        let mut messages: Vec<Message> = vec![];
        inject_skill_files_from(&mut messages, dir.path()).await;
        assert_eq!(messages.len(), 2);
        let body = |m: &Message| m.content.as_deref().unwrap_or("").to_owned();
        assert!(body(&messages[0]).contains("## Agent Guidelines"));
        assert!(body(&messages[0]).contains("agent guidelines"));
        assert!(body(&messages[1]).contains("## Soul"));
        assert!(body(&messages[1]).contains("soul content"));
    }

    #[tokio::test]
    async fn skill_files_dev_workflow_not_a_fixed_file() {
        // DEV_WORKFLOW.md is intentionally NOT a fixed file — only AGENTS.md
        // and SOUL.md are auto-injected.  A lone DEV_WORKFLOW.md must not
        // produce any messages.
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("DEV_WORKFLOW.md"), "workflow rules").unwrap();
        let mut messages: Vec<Message> = vec![];
        inject_skill_files_from(&mut messages, dir.path()).await;
        assert!(
            messages.is_empty(),
            "DEV_WORKFLOW.md must not be auto-injected as a fixed file"
        );
    }

    #[tokio::test]
    async fn skill_files_absent_produces_no_messages() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut messages: Vec<Message> = vec![];
        inject_skill_files_from(&mut messages, dir.path()).await;
        assert!(messages.is_empty());
    }

    #[tokio::test]
    async fn skill_files_empty_file_skipped() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "   \n  ").unwrap();
        let mut messages: Vec<Message> = vec![];
        inject_skill_files_from(&mut messages, dir.path()).await;
        assert!(
            messages.is_empty(),
            "whitespace-only file must not inject a message"
        );
    }

    #[tokio::test]
    async fn skill_files_partial_presence() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("SOUL.md"), "soul only").unwrap();
        let mut messages: Vec<Message> = vec![];
        inject_skill_files_from(&mut messages, dir.path()).await;
        assert_eq!(messages.len(), 1);
        assert!(messages[0]
            .content
            .as_deref()
            .unwrap_or("")
            .contains("soul only"));
    }

    #[tokio::test]
    async fn skill_files_ondemand_paths_injected_when_present() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("OPERATIONS_INDEX.md"), "ops index").unwrap();
        std::fs::write(dir.path().join("DEPLOYMENT.md"), "deploy steps").unwrap();
        let mut messages: Vec<Message> = vec![];
        inject_skill_files_from(&mut messages, dir.path()).await;
        assert_eq!(messages.len(), 1, "one on-demand pointer message expected");
        let body = messages[0].content.as_deref().unwrap_or("");
        assert!(body.contains("## On-demand Operations Docs"));
        assert!(body.contains("OPERATIONS_INDEX.md"));
        assert!(body.contains("DEPLOYMENT.md"));
        assert!(body.contains("read_file"));
    }

    #[tokio::test]
    async fn skill_files_ondemand_absent_produces_no_pointer_message() {
        let dir = tempfile::TempDir::new().unwrap();
        // Only a non-on-demand file present — no pointer message expected.
        std::fs::write(dir.path().join("AGENTS.md"), "guidelines").unwrap();
        let mut messages: Vec<Message> = vec![];
        inject_skill_files_from(&mut messages, dir.path()).await;
        let has_ondemand = messages.iter().any(|m| {
            m.content
                .as_deref()
                .unwrap_or("")
                .contains("On-demand Operations Docs")
        });
        assert!(
            !has_ondemand,
            "no on-demand pointer message when files absent"
        );
    }

    #[test]
    fn max_output_tokens_for_known_models() {
        assert_eq!(max_output_tokens_for_model("gpt-4.1"), 32_768);
        assert_eq!(max_output_tokens_for_model("gpt-4.1-mini"), 32_768);
        assert_eq!(max_output_tokens_for_model("gpt-4o"), 16_384);
        assert_eq!(max_output_tokens_for_model("gpt-4o-mini"), 16_384);
        assert_eq!(max_output_tokens_for_model("gpt-4-turbo"), 4_096);
        assert_eq!(max_output_tokens_for_model("gpt-4"), 8_192);
        assert_eq!(max_output_tokens_for_model("gpt-3.5-turbo"), 4_096);
        assert_eq!(max_output_tokens_for_model("o1"), 100_000);
        assert_eq!(max_output_tokens_for_model("o1-preview"), 100_000);
        assert_eq!(max_output_tokens_for_model("o3"), 100_000);
        assert_eq!(max_output_tokens_for_model("o3-mini"), 100_000);
        assert_eq!(max_output_tokens_for_model("claude-3-5-sonnet"), 16_384);
        assert_eq!(max_output_tokens_for_model("claude-opus-4-6"), 16_384);
    }

    #[test]
    fn max_output_tokens_gemini_models() {
        assert_eq!(max_output_tokens_for_model("gemini-2.0-flash"), 8_192);
        assert_eq!(max_output_tokens_for_model("gemini-1.5-pro"), 8_192);
        assert_eq!(max_output_tokens_for_model("gemini-1.5-flash"), 8_192);
        assert_eq!(max_output_tokens_for_model("gemini-flash"), 8_192);
    }

    #[test]
    fn max_output_tokens_unknown_model_is_conservative() {
        assert_eq!(max_output_tokens_for_model("some-future-model-xyz"), 16_384);
    }

    #[test]
    fn response_max_tokens_is_min_of_model_max_and_half_context() {
        // For each model, response_max_tokens must equal min(model_max, context_limit/2).
        for model in &[
            "gpt-4.1",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "o1",
            "o3",
            "claude-3-5-sonnet",
            "unknown-model",
        ] {
            let model_max = max_output_tokens_for_model(model);
            let context_half = context_limit_for_model(model) / 2;
            let expected = model_max.min(context_half);
            assert_eq!(
                response_max_tokens(model),
                expected,
                "model={model}: expected min({model_max}, {context_half}) = {expected}"
            );
        }
    }

    #[test]
    fn response_max_tokens_context_half_wins_for_small_context_model() {
        // gpt-4 has context=8192, so context/2=4096 < model_max=8192 → capped at 4096.
        assert_eq!(response_max_tokens("gpt-4"), 4_096);
    }

    #[test]
    fn response_max_tokens_model_max_wins_when_context_is_large() {
        // gpt-4o has model_max=16384 < context/2=64000 → capped at model_max.
        assert_eq!(response_max_tokens("gpt-4o"), 16_384);
    }

    // ---- mid-execution steer / interrupt tests ----------------------------

    /// Build an assistant `Message` carrying `n` synthetic tool calls with IDs
    /// "id0", "id1", … and a parallel `Vec<copilot::ToolCall>` for execution.
    fn make_tool_calls(
        n: usize,
    ) -> (
        Vec<crate::copilot::ToolCall>,
        Vec<crate::copilot::ApiToolCall>,
    ) {
        let tool_calls: Vec<crate::copilot::ToolCall> = (0..n)
            .map(|i| crate::copilot::ToolCall {
                id: format!("id{i}"),
                name: format!("tool_{i}"),
                arguments: "{}".into(),
            })
            .collect();
        let api_calls: Vec<crate::copilot::ApiToolCall> = tool_calls
            .iter()
            .map(|tc| crate::copilot::ApiToolCall {
                id: tc.id.clone(),
                kind: "function".into(),
                function: crate::copilot::ApiToolCallFunction {
                    name: tc.name.clone(),
                    arguments: tc.arguments.clone(),
                },
            })
            .collect();
        (tool_calls, api_calls)
    }

    /// Simulate the steer-mid-chain path and return the resulting messages.
    ///
    /// `n_calls`: total number of tool calls in the assistant turn.
    /// `interrupt_at`: index at which the steer is received (no tool at this
    ///   index or beyond is "executed").
    /// `steer`: optional user-injected text appended after the placeholders.
    fn simulate_mid_steer(
        n_calls: usize,
        interrupt_at: usize,
        steer: Option<&str>,
    ) -> Vec<Message> {
        let (tool_calls, api_calls) = make_tool_calls(n_calls);
        let mut messages: Vec<Message> = Vec::new();

        // Push the assistant turn with all tool_calls, just like daemon.rs does.
        messages.push(Message::assistant(None, api_calls));

        // Simulate executing tools up to the interrupt point.
        for tc in &tool_calls[..interrupt_at] {
            messages.push(Message::tool_result(&tc.id, "ok"));
        }

        // Push placeholder tool_results for skipped tools — mirrors daemon.rs:1569.
        for tc in &tool_calls[interrupt_at..] {
            messages.push(Message::tool_result(
                &tc.id,
                "[interrupted by user before execution]",
            ));
        }

        // Append the steer as a user turn — mirrors daemon.rs:1578.
        if let Some(text) = steer {
            messages.push(Message::user(text));
        }

        messages
    }

    #[test]
    fn steer_mid_chain_placeholder_count_correct() {
        // 3 tools, interrupt after executing 1 → 2 placeholders.
        let msgs = simulate_mid_steer(3, 1, None);
        let tool_results: Vec<_> = msgs.iter().filter(|m| m.role == "tool").collect();
        assert_eq!(
            tool_results.len(),
            3,
            "must have one tool_result per tool_call"
        );
    }

    #[test]
    fn steer_mid_chain_every_tool_call_has_matching_result() {
        let msgs = simulate_mid_steer(4, 2, Some("go left instead"));
        let assistant = &msgs[0];
        assert_eq!(assistant.role, "assistant");

        for tc in &assistant.tool_calls {
            let found = msgs
                .iter()
                .any(|m| m.role == "tool" && m.tool_call_id.as_deref() == Some(tc.id.as_str()));
            assert!(found, "tool_call id={} has no matching tool_result", tc.id);
        }
    }

    #[test]
    fn steer_mid_chain_message_ordering() {
        // Ordering: assistant → tool… → user (steer)
        let msgs = simulate_mid_steer(3, 1, Some("new direction"));

        assert_eq!(msgs[0].role, "assistant", "first message must be assistant");
        for (i, msg) in msgs.iter().enumerate().take(4).skip(1) {
            assert_eq!(msg.role, "tool", "messages[{i}] must be role=tool");
        }
        assert_eq!(
            msgs.last().unwrap().role,
            "user",
            "last message must be user (steer)"
        );
    }

    #[test]
    fn steer_at_first_tool_all_placeholders() {
        // Interrupt before any tool runs → all 3 results are placeholders.
        let msgs = simulate_mid_steer(3, 0, None);
        let placeholders: Vec<_> = msgs
            .iter()
            .filter(|m| {
                m.role == "tool"
                    && m.content
                        .as_deref()
                        .unwrap_or("")
                        .contains("[interrupted by user before execution]")
            })
            .collect();
        assert_eq!(
            placeholders.len(),
            3,
            "all tool_results must be placeholders"
        );
    }

    #[test]
    fn interrupt_no_steer_text_no_trailing_user_message() {
        // Interrupt with no steer text → no user message appended.
        let msgs = simulate_mid_steer(2, 1, None);
        assert_ne!(
            msgs.last().unwrap().role,
            "user",
            "without steer text there must be no trailing user message"
        );
    }
}
