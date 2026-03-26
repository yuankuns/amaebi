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

/// Tokens reserved for model output — matches `max_tokens` in the request body.
const RESPONSE_RESERVE_TOKENS: usize = 8_192;
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
    pub tokens: TokenCache,
    /// Tool executor — `LocalExecutor` now; swappable with `DockerExecutor` in Phase 4.
    pub executor: Box<dyn ToolExecutor>,
    /// Persistent SQLite connection opened once at startup.
    ///
    /// Wrapped in `Mutex` so that concurrent `spawn_blocking` tasks serialise
    /// all reads and writes through a single connection without re-running the
    /// schema setup (`PRAGMA`s, `CREATE TABLE`, triggers) on every request.
    pub db: Arc<Mutex<rusqlite::Connection>>,
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
        Ok(Self {
            http,
            tokens: TokenCache::new(),
            executor: Box::new(tools::LocalExecutor),
            db: Arc::new(Mutex::new(conn)),
        })
    }
}

// ---------------------------------------------------------------------------
// Token budget helpers
// ---------------------------------------------------------------------------

/// Approximate GPT-4o token count for a message list.
///
/// Uses tiktoken o200k_base (GPT-4o's encoding) with the OpenAI overhead formula:
/// 4 tokens per message + role + content + 3 priming tokens for the reply.
/// The tokenizer is initialised lazily on the first call (OnceLock).
fn count_message_tokens(messages: &[Message]) -> usize {
    use std::sync::OnceLock;
    static BPE: OnceLock<tiktoken_rs::CoreBPE> = OnceLock::new();
    let bpe =
        BPE.get_or_init(|| tiktoken_rs::o200k_base().expect("failed to load o200k_base tokenizer"));

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
    }
    total
}

/// Context window size for `model`, matched by prefix (longest wins).
///
/// Falls back to a conservative 32 k for unknown models so we never send
/// more tokens than the server can handle.
fn context_limit_for_model(model: &str) -> usize {
    // Ordered longest-prefix-first so that e.g. "gpt-4-turbo" beats "gpt-4".
    const TABLE: &[(&str, usize)] = &[
        ("gpt-4o", 128_000),
        ("gpt-4-turbo", 128_000),
        ("gpt-4", 8_192),
        ("gpt-3.5-turbo", 16_385),
        ("o1", 200_000),
        ("o3", 200_000),
        ("claude", 200_000),
    ];
    TABLE
        .iter()
        .find(|(prefix, _)| model.starts_with(prefix))
        .map(|(_, limit)| *limit)
        .unwrap_or(32_768) // conservative default for unknown models
}

/// Token threshold above which compaction should be triggered for `model`.
fn compaction_threshold_tokens(model: &str) -> usize {
    let available = context_limit_for_model(model).saturating_sub(RESPONSE_RESERVE_TOKENS);
    (available as f64 * COMPACTION_THRESHOLD) as usize
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
    let (reader, mut writer) = tokio::io::split(stream);
    let mut lines = BufReader::new(reader).lines();

    let line = lines
        .next_line()
        .await
        .context("reading request")?
        .context("client disconnected before sending a request")?;

    let req: Request = serde_json::from_str(&line).context("parsing request JSON")?;

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
            write_frame(&mut writer, &Response::Done).await?;
        }

        Request::StoreMemory { user, assistant } => {
            // Use a stable session id for ACP-sourced memory so it lands in
            // the same logical bucket as other global (non-directory) writes.
            store_conversation(&state, "global", &user, &assistant).await;
            write_frame(&mut writer, &Response::Done).await?;
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
            for entry in entries {
                write_frame(
                    &mut writer,
                    &Response::MemoryEntry {
                        role: entry.role,
                        content: truncate_chars(entry.content, MAX_HISTORY_CHARS),
                    },
                )
                .await?;
            }
            write_frame(&mut writer, &Response::Done).await?;
        }

        Request::Steer { .. } => {
            // A Steer frame arriving on a fresh connection (not mid-Chat) is
            // an error — there is no running agentic loop to steer.
            write_frame(
                &mut writer,
                &Response::Error {
                    message: "no active agentic loop to steer on this connection".into(),
                },
            )
            .await?;
        }

        Request::SubmitDetach {
            prompt,
            tmux_pane,
            model,
            session_id,
        } => {
            tracing::info!(
                model = %model,
                prompt_len = prompt.len(),
                "received detach request"
            );

            // Verify auth eagerly so we can return an error before detaching.
            if let Err(e) = state.tokens.get(&state.http).await {
                write_frame(
                    &mut writer,
                    &Response::Error {
                        message: format!("authentication error: {e:#}"),
                    },
                )
                .await?;
                return Ok(());
            }

            // Generate a fresh UUID for detached tasks when the client does
            // not provide one, so every background task has a stable identifier
            // that appears in inbox reports and can be resumed later.
            let sid = session_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

            // Acknowledge immediately — client can exit after this.
            write_frame(
                &mut writer,
                &Response::DetachAccepted {
                    session_id: sid.clone(),
                },
            )
            .await?;

            // Spawn the agentic loop as a fully detached background task.
            let state = Arc::clone(&state);
            tokio::spawn(async move {
                // Load full session history from SQLite; token budget trims it below.
                let db = Arc::clone(&state.db);
                let sid_clone = sid.clone();
                let history = tokio::task::spawn_blocking(move || {
                    let conn = db.lock().unwrap_or_else(|p| p.into_inner());
                    memory_db::get_session_history(&conn, &sid_clone)
                })
                .await
                .unwrap_or_else(|e| {
                    tracing::warn!(error = %e, "detach history load panicked");
                    Ok(vec![])
                })
                .unwrap_or_else(|e| {
                    tracing::warn!(error = %e, "failed to load detach history");
                    vec![]
                });

                let mut messages =
                    build_messages(&prompt, tmux_pane.as_deref(), &history, &[], None);
                inject_skill_files(&mut messages).await;

                // Pre-flight token check: trim to hot tail if over budget.
                let threshold = compaction_threshold_tokens(&model);
                if count_message_tokens(&messages) > threshold {
                    let hot = HOT_TAIL_PAIRS * 2;
                    let trimmed = if history.len() > hot {
                        &history[history.len() - hot..]
                    } else {
                        &history[..]
                    };
                    messages = build_messages(&prompt, tmux_pane.as_deref(), trimmed, &[], None);
                    inject_skill_files(&mut messages).await;
                }

                // Use a sink writer — output frames are discarded; we only
                // need the return value (final_text) for the inbox.
                // Drop the sender immediately (`_`) so steer_rx.recv() in the
                // agentic loop returns None at once instead of timing out if
                // the model ends with '?'.
                let mut sink = tokio::io::sink();
                let (_, mut steer_rx) = tokio::sync::mpsc::channel::<String>(1);

                match run_agentic_loop(&state, &model, messages, &mut sink, &mut steer_rx).await {
                    Ok((final_text, _)) => {
                        let user_text = truncate_chars(prompt.clone(), MAX_PROMPT_CHARS);
                        let assistant_text = truncate_chars(final_text.clone(), MAX_RESPONSE_CHARS);

                        // Persist to SQLite memory store.
                        store_conversation(&state, &sid, &user_text, &assistant_text).await;

                        // Save result to inbox so the user gets a notification.
                        let task_desc = truncate_chars(prompt, 200);
                        match InboxStore::open() {
                            Ok(inbox) => {
                                if let Err(e) = inbox.save_report(&sid, &task_desc, &final_text) {
                                    tracing::warn!(
                                        error = %e,
                                        "detach: failed to save inbox report"
                                    );
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    error = %e,
                                    "detach: failed to open inbox"
                                );
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "detach agentic loop error");
                        // Save the error itself to inbox so user is informed.
                        let task_desc = truncate_chars(prompt, 200);
                        if let Ok(inbox) = InboxStore::open() {
                            let _ = inbox.save_report(&sid, &task_desc, &format!("[error] {e:#}"));
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
            tracing::info!(
                model = %model,
                session_id = %session_id,
                prompt_len = prompt.len(),
                "received resume request"
            );

            if let Err(e) = state.tokens.get(&state.http).await {
                write_frame(
                    &mut writer,
                    &Response::Error {
                        message: format!("authentication error: {e:#}"),
                    },
                )
                .await?;
                return Ok(());
            }

            let (steer_tx, mut steer_rx) = tokio::sync::mpsc::channel::<String>(16);
            let expected_resume_sid = session_id.clone();
            tokio::spawn(async move {
                while let Ok(Some(frame)) = lines.next_line().await {
                    match serde_json::from_str::<Request>(&frame) {
                        Ok(Request::Steer {
                            session_id: steer_sid,
                            message,
                        }) => {
                            if steer_sid != expected_resume_sid {
                                tracing::debug!(
                                    expected = %expected_resume_sid,
                                    got = %steer_sid,
                                    "ignoring steer frame with mismatched session_id on resume"
                                );
                                continue;
                            }
                            if steer_tx.send(message).await.is_err() {
                                break;
                            }
                        }
                        Ok(_) | Err(_) => {
                            tracing::debug!("unexpected frame on established resume connection");
                        }
                    }
                }
            });

            // Resume: load the FULL history from SQLite without a sliding-window cap,
            // plus summaries from other sessions for cross-session context.
            let db = Arc::clone(&state.db);
            let sid_clone = session_id.clone();
            let (history, summaries) = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
                let conn = db.lock().unwrap_or_else(|p| p.into_inner());
                let history = memory_db::get_session_history(&conn, &sid_clone)?;
                let summaries = memory_db::get_recent_summaries(&conn, &sid_clone, MAX_SUMMARIES)?;
                Ok((history, summaries))
            })
            .await
            .unwrap_or_else(|e| {
                tracing::warn!(error = %e, "resume history load panicked");
                Ok((vec![], vec![]))
            })
            .unwrap_or_else(|e| {
                tracing::warn!(error = %e, "failed to load resume history");
                (vec![], vec![])
            });

            // Resume: full history for re-hydration; token budget trims if needed.
            let mut messages =
                build_messages(&prompt, tmux_pane.as_deref(), &history, &summaries, None);
            inject_skill_files(&mut messages).await;

            // Pre-flight token check.
            let threshold = compaction_threshold_tokens(&model);
            if count_message_tokens(&messages) > threshold {
                let hot = HOT_TAIL_PAIRS * 2;
                let trimmed = if history.len() > hot {
                    &history[history.len() - hot..]
                } else {
                    &history[..]
                };
                messages = build_messages(&prompt, tmux_pane.as_deref(), trimmed, &summaries, None);
                inject_skill_files(&mut messages).await;
            }

            match run_agentic_loop(&state, &model, messages, &mut writer, &mut steer_rx).await {
                Ok((response_text, _)) => {
                    let user_text = truncate_chars(prompt.clone(), MAX_PROMPT_CHARS);
                    let assistant_text = truncate_chars(response_text.clone(), MAX_RESPONSE_CHARS);

                    store_conversation(&state, &session_id, &user_text, &assistant_text).await;
                }
                Err(e) => {
                    tracing::error!(error = %e, "resume agentic loop error");
                    let _ = write_frame(
                        &mut writer,
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
            tracing::info!(
                pane = ?tmux_pane,
                model = %model,
                prompt_len = prompt.len(),
                "received chat request"
            );

            // Verify authentication before entering the loop so we can return
            // a clear error to the user instead of failing mid-conversation.
            if let Err(e) = state.tokens.get(&state.http).await {
                tracing::error!(error = %e, "failed to get Copilot API token");
                write_frame(
                    &mut writer,
                    &Response::Error {
                        message: format!("authentication error: {e:#}"),
                    },
                )
                .await?;
                return Ok(());
            }

            // Resolve session first so the steer reader can validate the
            // session_id on incoming Steer frames.
            // Older clients that omit session_id get a fresh UUID so their
            // history and steer lookups are isolated (not all lumped under "").
            let sid = session_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

            // Steering channel: the spawned reader task sends user corrections
            // here; the agentic loop drains them between model turns.
            let (steer_tx, mut steer_rx) = tokio::sync::mpsc::channel::<String>(16);

            // Spawn a task that reads subsequent frames from the client on
            // this connection.  Any Steer frames are forwarded to steer_tx so
            // the running agentic loop can inject them between tool turns.
            // The task exits when the client closes the connection (EOF) or
            // when steer_tx is dropped (agentic loop finished).
            let expected_chat_sid = sid.clone();
            tokio::spawn(async move {
                while let Ok(Some(frame)) = lines.next_line().await {
                    match serde_json::from_str::<Request>(&frame) {
                        Ok(Request::Steer {
                            session_id: steer_sid,
                            message,
                        }) => {
                            if steer_sid != expected_chat_sid {
                                tracing::debug!(
                                    expected = %expected_chat_sid,
                                    got = %steer_sid,
                                    "ignoring steer frame with mismatched session_id on chat"
                                );
                                continue;
                            }
                            if steer_tx.send(message).await.is_err() {
                                break; // agentic loop has finished; channel closed
                            }
                        }
                        Ok(_) | Err(_) => {
                            // Unexpected frame type mid-stream; ignore.
                            tracing::debug!("unexpected frame type on established chat connection");
                        }
                    }
                }
            });

            // Load full session history, past-session summaries, and the session's own
            // running summary (if any) for token-budget-driven compaction decisions.
            let db = Arc::clone(&state.db);
            let sid_clone = sid.clone();
            let (history, past_summaries, own_summary) =
                tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
                    let conn = db.lock().unwrap_or_else(|p| p.into_inner());
                    let history = memory_db::get_session_history(&conn, &sid_clone)?;
                    let past_summaries =
                        memory_db::get_recent_summaries(&conn, &sid_clone, MAX_SUMMARIES)?;
                    let own_summary = memory_db::get_session_own_summary(&conn, &sid_clone)?;
                    Ok((history, past_summaries, own_summary))
                })
                .await
                .unwrap_or_else(|e| {
                    tracing::warn!(error = %e, "session history load panicked");
                    Ok((vec![], vec![], None))
                })
                .unwrap_or_else(|e| {
                    tracing::warn!(error = %e, "failed to load session history");
                    (vec![], vec![], None)
                });

            // Cross-session: if this is the first turn of a new session, compact any
            // old sessions that have never been summarised.
            if history.is_empty() {
                let db = Arc::clone(&state.db);
                let sid_clone = sid.clone();
                let old_sessions = tokio::task::spawn_blocking(move || {
                    let conn = db.lock().unwrap_or_else(|p| p.into_inner());
                    memory_db::get_sessions_without_summary(&conn, &sid_clone, MAX_SUMMARIES)
                })
                .await
                .unwrap_or_else(|_| Ok(vec![]))
                .unwrap_or_default();

                for old_sid in old_sessions {
                    // Cross-session: summarise the full closed session (keep_recent = 0).
                    tokio::spawn(compact_session(
                        Arc::clone(&state),
                        old_sid,
                        model.clone(),
                        0,
                    ));
                }
            }

            let mut messages = build_messages(
                &prompt,
                tmux_pane.as_deref(),
                &history,
                &past_summaries,
                own_summary.as_deref(),
            );
            inject_skill_files(&mut messages).await;

            // Pre-flight token check: if over the compaction threshold, trim to hot tail.
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
                    &past_summaries,
                    own_summary.as_deref(),
                );
                inject_skill_files(&mut messages).await;
                tracing::debug!(
                    hot_tail = trimmed.len(),
                    "pre-flight token trim: history reduced to hot tail"
                );
            }

            match run_agentic_loop(&state, &model, messages, &mut writer, &mut steer_rx).await {
                Ok((response_text, prompt_tokens)) => {
                    let user_text = truncate_chars(prompt.clone(), MAX_PROMPT_CHARS);
                    let assistant_text = truncate_chars(response_text.clone(), MAX_RESPONSE_CHARS);

                    // Persist to SQLite memory store.
                    store_conversation(&state, &sid, &user_text, &assistant_text).await;

                    // Within-session compaction: if prompt tokens exceeded the threshold
                    // and no summary exists yet, compact the portion before the hot tail.
                    if prompt_tokens > threshold && own_summary.is_none() {
                        tokio::spawn(compact_session(
                            Arc::clone(&state),
                            sid.clone(),
                            model.clone(),
                            HOT_TAIL_PAIRS * 2, // keep hot tail out of the summary
                        ));
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, "agentic loop error");
                    let _ = write_frame(
                        &mut writer,
                        &Response::Error {
                            message: format!("agent error: {e:#}"),
                        },
                    )
                    .await;
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
async fn compact_session(
    state: Arc<DaemonState>,
    session_id: String,
    model: String,
    keep_recent: usize,
) {
    let db = Arc::clone(&state.db);
    let sid = session_id.clone();
    let history = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let conn = db.lock().unwrap_or_else(|p| p.into_inner());
        let total = memory_db::count_session_turns(&conn, &sid)?;
        // How many of the oldest turns to summarise: all of them for cross-session
        // (keep_recent == 0), or only the portion beyond the sliding window for
        // within-session so summary and raw history window do not overlap.
        let to_summarise = total.saturating_sub(keep_recent);
        memory_db::get_session_oldest(&conn, &sid, to_summarise)
    })
    .await
    .unwrap_or_else(|e| {
        tracing::warn!(error = %e, "compact_session: history load panicked");
        Ok(vec![])
    })
    .unwrap_or_else(|e| {
        tracing::warn!(error = %e, "compact_session: failed to load history");
        vec![]
    });

    if history.is_empty() {
        return;
    }

    let mut messages = vec![Message::system(
        "You are a memory compactor. Given a conversation, output 3-5 bullet points \
         capturing the key outcomes, decisions, and facts learned. \
         Be concise and factual. Output only the bullet points, no preamble.",
    )];
    for entry in &history {
        let content = truncate_chars(entry.content.clone(), 1_500);
        match entry.role.as_str() {
            "user" => messages.push(Message::user(content)),
            "assistant" => messages.push(Message::assistant(Some(content), vec![])),
            _ => {}
        }
    }

    // Sink writer + dropped sender = non-interactive, no output.
    let mut sink = tokio::io::sink();
    let (_, mut steer_rx) = tokio::sync::mpsc::channel::<String>(1);

    match run_agentic_loop(&state, &model, messages, &mut sink, &mut steer_rx).await {
        Ok((summary, _)) if !summary.trim().is_empty() => {
            let db = Arc::clone(&state.db);
            let sid = session_id.clone();
            let ts = chrono::Utc::now().to_rfc3339();
            let summary = summary.trim().to_owned();
            let result = tokio::task::spawn_blocking(move || {
                let conn = db.lock().unwrap_or_else(|p| p.into_inner());
                memory_db::store_session_summary(&conn, &sid, &summary, &ts)
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
        Err(e) => tracing::warn!(error = %e, "compact_session: API error"),
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
    steer_rx: &mut tokio::sync::mpsc::Receiver<String>,
) -> Result<(String, usize)>
where
    W: AsyncWriteExt + Unpin,
{
    let schemas = tools::tool_schemas();
    let final_text;
    let mut tools_were_used = false;
    let mut conclusion_nudge_sent = false;
    let mut last_prompt_tokens = 0usize;

    loop {
        // Drain any steering corrections that arrived since the last model
        // call (covers non-tool turns and the time between tool completion
        // and the next iteration).
        while let Ok(steer_msg) = steer_rx.try_recv() {
            messages.push(Message::user(steer_msg));
            write_frame(writer, &Response::SteerAck).await?;
        }

        // Re-fetch the token on every iteration so long-running agentic loops
        // survive token expiration.
        let token = state
            .tokens
            .get(&state.http)
            .await
            .context("refreshing Copilot API token inside agentic loop")?;

        // stream_chat retries 5xx, 429, and transport errors internally up to
        // its MAX_RETRIES, but those errors can still surface here if retries
        // are exhausted, or if parsing/IO errors occur while streaming.
        // 4xx responses (except 429) are surfaced immediately as CopilotHttpError;
        // for auth-adjacent ones (400/401/403) we evict the cache and retry once
        // with a fresh token. Any other error (exhausted retries, context overflow,
        // etc.) propagates.
        let resp =
            match copilot::stream_chat(&state.http, &token, model, &messages, &schemas, writer)
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    let is_auth_err = e
                        .downcast_ref::<copilot::CopilotHttpError>()
                        .is_some_and(|he| matches!(he.status.as_u16(), 400 | 401 | 403));
                    if is_auth_err {
                        tracing::warn!(
                            error = %e,
                            "Copilot auth error; evicting token cache and retrying once"
                        );
                        state.tokens.invalidate().await;
                        let fresh_token = state
                            .tokens
                            .get(&state.http)
                            .await
                            .context("fetching fresh token after auth error")?;
                        copilot::stream_chat(
                            &state.http,
                            &fresh_token,
                            model,
                            &messages,
                            &schemas,
                            writer,
                        )
                        .await?
                    } else {
                        return Err(e);
                    }
                }
            };

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
                    match tokio::time::timeout(std::time::Duration::from_secs(300), steer_rx.recv())
                        .await
                    {
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

                    match tokio::time::timeout(std::time::Duration::from_secs(300), steer_rx.recv())
                        .await
                    {
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
                break;
            }

            FinishReason::ToolCalls => {
                tools_were_used = true;
                let api_calls: Vec<ApiToolCall> = resp
                    .tool_calls
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

                let assistant_text = if resp.text.is_empty() {
                    None
                } else {
                    Some(resp.text)
                };
                messages.push(Message::assistant(assistant_text, api_calls));

                for tc in &resp.tool_calls {
                    tracing::debug!(tool = %tc.name, "executing tool");

                    let tool_detail = {
                        let args: serde_json::Value =
                            serde_json::from_str(&tc.arguments).unwrap_or(serde_json::Value::Null);
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

                // Steers that arrived during tool execution are drained at
                // the top of the next loop iteration, before the model call.
            }

            FinishReason::Other(ref reason) => {
                tracing::warn!(finish_reason = %reason, "session end: unexpected finish reason");
                let warning = format!("\n[stopped: unexpected finish reason '{reason}']");
                write_frame(writer, &Response::Text { chunk: warning }).await?;
                final_text = resp.text;
                break;
            }
        }
    }

    write_frame(writer, &Response::Done).await?;
    Ok((final_text, last_prompt_tokens))
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
    const FIXED_FILES: &[(&str, &str)] =
        &[("AGENTS.md", "## Agent Guidelines"), ("SOUL.md", "## Soul")];
    for (filename, header) in FIXED_FILES {
        let path = amaebi_home.join(filename);
        match tokio::fs::read_to_string(&path).await {
            Ok(content) => {
                let trimmed = content.trim();
                if !trimmed.is_empty() {
                    messages.push(Message::system(format!("{header}\n\n{trimmed}")));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                tracing::debug!(file = %path.display(), error = %e, "could not read config file");
            }
        }
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
    let (_, mut steer_rx) = tokio::sync::mpsc::channel::<String>(1);

    let result = run_agentic_loop(&state, &model, messages, &mut sink, &mut steer_rx).await;

    let (output, run_ok) = match result {
        Ok((final_text, _)) => {
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
        assert!(system.contains("Added cron"), "all summaries must be injected");
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
    fn context_limit_unknown_model_is_conservative() {
        // Unknown models fall back to 32k — never 0, never larger than any known model.
        let limit = context_limit_for_model("some-future-model-xyz");
        assert_eq!(limit, 32_768);
    }

    #[test]
    fn compaction_threshold_is_below_context_limit() {
        for model in &["gpt-4o", "gpt-4", "gpt-3.5-turbo", "o1", "unknown-model"] {
            let t = compaction_threshold_tokens(model);
            let available = context_limit_for_model(model).saturating_sub(RESPONSE_RESERVE_TOKENS);
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
}
