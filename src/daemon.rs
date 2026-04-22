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
use crate::pane_lease;
use crate::session;
use crate::tools::{self, ToolExecutor};

/// Resolve the raw model string to the API-level model ID for token lookups.
///
/// The raw model may be `bedrock/claude-sonnet-4.6` or plain `gpt-4o`.
/// We resolve via [`crate::provider::resolve`] so the token tables can
/// match against Bedrock IDs (e.g. `us.anthropic.claude-*`) or Copilot
/// IDs (e.g. `gpt-4o`).
fn resolved_model_id(model: &str) -> String {
    crate::provider::resolve(model).model_id
}

/// Compute `max_tokens` for a request to `model`: capped at half the model's context window
/// so it never exceeds what the model supports (e.g. gpt-4's 8,192-token limit).
fn max_output_tokens_for_model(model: &str) -> usize {
    let model = resolved_model_id(model);
    // Ordered longest-prefix-first so that e.g. "gpt-4-turbo" beats "gpt-4".
    const TABLE: &[(&str, usize)] = &[
        // Bedrock model IDs (cross-region us.anthropic.* format)
        ("us.anthropic.claude-3-5-haiku", 8_192),
        ("us.anthropic.claude-3-5-sonnet", 8_192),
        ("us.anthropic.claude", 16_384), // catch-all for Bedrock Claude models
        // Copilot model IDs
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
    let model_id = &resolved_model_id(model);
    let model_max = max_output_tokens_for_model(model_id);
    // Pass the raw model string (preserving [1m]) so context_limit_for_model
    // can return 1_000_000 when the suffix is present.
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
/// Resolve the model to use for background session compaction.
///
/// Defaults to [`crate::provider::DEFAULT_MODEL`] so compaction never
/// inherits an expensive main-agent model (e.g. opus) unless the caller
/// explicitly opts in.  The provider prefix of `main_model` is preserved so
/// compaction stays on the same API backend (e.g. `copilot/` stays Copilot).
///
/// Resolution order:
///   1. `AMAEBI_COMPACT_MODEL` env var (used verbatim)
///   2. Same provider prefix as `main_model` + `DEFAULT_MODEL` (sonnet)
fn compact_model(main_model: &str) -> String {
    if let Ok(override_model) = std::env::var("AMAEBI_COMPACT_MODEL") {
        return override_model;
    }
    // Preserve the provider prefix so compaction uses the same API backend.
    let prefix = main_model
        .split_once('/')
        .map(|(p, _)| p)
        .filter(|p| matches!(*p, "copilot" | "bedrock"));
    match prefix {
        Some(p) => format!("{}/{}", p, crate::provider::DEFAULT_MODEL),
        None => crate::provider::DEFAULT_MODEL.to_string(),
    }
}

/// Compact session history when prompt tokens exceed this fraction of available input.
const COMPACTION_THRESHOLD: f64 = 0.75;
/// Minimum recent user/assistant *pairs* to keep in the hot tail after a token-budget trim.
const HOT_TAIL_PAIRS: usize = 3;
/// How many past-session summaries to prepend to the system message.
const MAX_SUMMARIES: usize = 5;
/// Maximum chars per injected past-session summary.
const MAX_SUMMARY_CHARS: usize = 500;
/// Stop attempting in-loop compaction after this many consecutive failures.
/// Mirrors Claude Code's `MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES` circuit breaker
/// so an irrecoverably-oversized context does not trigger a retry storm.
const MAX_CONSECUTIVE_COMPACT_FAILURES: u32 = 3;
/// Cap on the summariser's `max_tokens`.  The summary prompt asks for 3-5
/// bullet points; hard-capping the output budget prevents a verbose summary
/// from failing to reduce context (which would re-trigger compaction on the
/// next iteration).  Matches the same order of magnitude as the DB-persisted
/// summary truncation in [`MAX_SUMMARY_CHARS`].
const COMPACT_SUMMARY_MAX_TOKENS: usize = 2_048;

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
    /// Sessions currently held by an active connection (Chat, Ask, or Resume).
    ///
    /// A session may only be used by one connection at a time.  Before processing
    /// any Chat/Ask/Resume request, the daemon inserts the session ID here.  If
    /// it is already present the request is rejected with a Response::Error so
    /// concurrent clients cannot interleave writes to the same session history.
    pub active_sessions: Arc<Mutex<HashSet<String>>>,
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
        let active_sessions: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));

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
            active_sessions,
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
    // Resolve once to get the provider, model ID, and use_1m flag together.
    // Gating on provider == Bedrock prevents Copilot model IDs that contain
    // "claude-sonnet-4" / "claude-opus-4-6" (e.g. copilot/claude-sonnet-4[1m])
    // from incorrectly receiving a 1M token budget.
    let spec = crate::provider::resolve(model);
    if spec.provider == crate::provider::ProviderKind::Bedrock
        && spec.use_1m
        && crate::bedrock::supports_1m_context(&spec.model_id)
    {
        return 1_000_000;
    }

    let model = spec.model_id;
    // Ordered longest-prefix-first so that e.g. "gpt-4-turbo" beats "gpt-4".
    const TABLE: &[(&str, usize)] = &[
        // Bedrock model IDs (cross-region us.anthropic.* format): 200k context.
        ("us.anthropic.claude", 200_000),
        // Copilot model IDs
        // gpt-5.x via Responses API: 128k context (conservative)
        ("gpt-5", 128_000),
        ("gpt-4o", 128_000),
        ("gpt-4.1", 1_047_576),
        ("gpt-4-turbo", 128_000),
        ("gpt-4", 8_192),
        ("gpt-3.5-turbo", 16_385),
        ("o1", 200_000),
        ("o3", 200_000),
        // Claude models (via Copilot): 200k context window.
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
// Provider auth helpers
// ---------------------------------------------------------------------------

/// Returns true when the resolved provider for `model` is Copilot, meaning a
/// token pre-flight check is required before dispatching the request.
fn needs_copilot_auth(model: &str) -> bool {
    crate::provider::resolve(model).provider == crate::provider::ProviderKind::Copilot
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
// Database helper
// ---------------------------------------------------------------------------

/// Run a blocking database operation on `db` inside `spawn_blocking`.
///
/// Locks the connection, calls `f`, and maps any `JoinError` (panic or
/// cancellation) into an `anyhow` error so callers never need to deal with
/// `JoinError` directly.
async fn with_db<F, T>(db: Arc<Mutex<rusqlite::Connection>>, f: F) -> anyhow::Result<T>
where
    F: FnOnce(&rusqlite::Connection) -> anyhow::Result<T> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        let conn = db.lock().unwrap_or_else(|p| p.into_inner());
        f(&conn)
    })
    .await
    .map_err(|e| {
        if e.is_cancelled() {
            anyhow::anyhow!("database task was cancelled")
        } else {
            anyhow::anyhow!("database task panicked: {e}")
        }
    })?
}

/// Load the three pieces of session state needed to build a message list.
///
/// Returns `(history, summaries, own_summary)`.  On error, logs a warning and
/// returns empty defaults so the caller can still proceed with a fresh context.
async fn load_session_state(
    state: &Arc<DaemonState>,
    session_id: &str,
) -> (Vec<memory_db::DbMemoryEntry>, Vec<String>, Option<String>) {
    with_db(Arc::clone(&state.db), {
        let sid = session_id.to_owned();
        move |conn| {
            Ok((
                memory_db::get_session_history(conn, &sid)?,
                memory_db::get_recent_summaries(conn, &sid, MAX_SUMMARIES)?,
                memory_db::get_session_own_summary(conn, &sid)?,
            ))
        }
    })
    .await
    .unwrap_or_else(|e| {
        tracing::warn!(error = %e, session_id, "failed to load session state");
        (vec![], vec![], None)
    })
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
    // Carries the effective model across Chat turns: updated when switch_model
    // is called inside run_agentic_loop so the new model persists for the
    // remainder of the session, not just the current agentic loop invocation.
    let mut carried_model: Option<String> = None;
    // Count of unsolicited frames received while an agentic loop was running.
    // Flushed as error responses after the loop releases the writer lock.
    let mut pending_unsolicited: u32 = 0;
    // Held for the lifetime of a Chat connection once a session is claimed.
    // Ensures a second connection cannot use the same session concurrently.
    let mut chat_session_guard: Option<ActiveSessionGuard> = None;

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
                handle_clear_memory(&state, &writer).await?;
            }

            Request::StoreMemory { user, assistant } => {
                handle_store_memory(&state, &writer, &user, &assistant).await?;
            }

            Request::RetrieveContext { prompt } => {
                handle_retrieve_context(&state, &writer, prompt).await?;
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
                if needs_copilot_auth(&model) {
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
                    let history = with_db(Arc::clone(&state.db), {
                        let sid = sid.clone();
                        move |conn| memory_db::get_session_history(conn, &sid)
                    })
                    .await
                    .unwrap_or_default();
                    let mut messages =
                        build_messages(&prompt, tmux_pane.as_deref(), &history, &[], None, &model);
                    inject_skill_files(&mut messages).await;
                    let mut sink = tokio::io::sink();
                    let (_, mut steer_rx) = tokio::sync::mpsc::channel::<Option<String>>(1);
                    let task_desc = truncate_chars(&prompt, 200);
                    match run_agentic_loop(
                        &state,
                        &model,
                        messages,
                        &mut sink,
                        &mut steer_rx,
                        true,
                        Some(&sid),
                    )
                    .await
                    {
                        Ok((final_text, _, _, _)) => {
                            store_conversation(
                                &state,
                                &sid,
                                &truncate_chars(&prompt, MAX_PROMPT_CHARS),
                                &truncate_chars(&final_text, MAX_RESPONSE_CHARS),
                            )
                            .await;
                            if let Ok(inbox) = InboxStore::open() {
                                let _ = inbox.save_report(&sid, &task_desc, &final_text);
                            }
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "detach agentic loop error");
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
                let mut conn_state = ConnState {
                    frame_rx: &mut frame_rx,
                    pending_unsolicited: &mut pending_unsolicited,
                };
                match handle_resume_request(
                    &state,
                    &writer,
                    &mut conn_state,
                    prompt,
                    tmux_pane,
                    model,
                    session_id,
                )
                .await?
                {
                    ConnAction::Continue => {}
                    ConnAction::Break => break 'connection,
                }
            }

            Request::Chat {
                prompt,
                tmux_pane,
                model,
                session_id,
            } => {
                let mut conn_state = ConnState {
                    frame_rx: &mut frame_rx,
                    pending_unsolicited: &mut pending_unsolicited,
                };
                match handle_chat_request(
                    &state,
                    &writer,
                    &mut conn_state,
                    &mut carried_messages,
                    &mut carried_session_id,
                    &mut carried_model,
                    &mut chat_session_guard,
                    prompt,
                    tmux_pane,
                    model,
                    session_id,
                )
                .await?
                {
                    ConnAction::Continue => {}
                    ConnAction::Break => break 'connection,
                }
            }

            Request::ClaudeLaunch { tasks } => {
                handle_claude_launch(&writer, tasks).await?;
            }

            Request::SupervisePanes {
                panes,
                model,
                session_id,
            } => {
                handle_supervision(&writer, &mut frame_rx, panes, model, &state, session_id)
                    .await?;
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Connection-local state passed into sub-handlers
// ---------------------------------------------------------------------------

/// Outcome returned by sub-handlers that may need to terminate the connection loop.
enum ConnAction {
    /// Continue processing frames on this connection.
    Continue,
    /// Break out of the connection loop (client disconnected or fatal auth error).
    Break,
}

/// Mutable per-connection state shared between `handle_connection` and sub-handlers.
struct ConnState<'a> {
    /// Channel receiving frames from the reader task.
    frame_rx: &'a mut tokio::sync::mpsc::Receiver<String>,
    /// Count of unsolicited frames buffered while an agentic loop held the writer lock.
    pending_unsolicited: &'a mut u32,
}

// ---------------------------------------------------------------------------
// Shared agentic-loop driver
// ---------------------------------------------------------------------------

/// Spawn the agentic loop in a separate task and route steer/interrupt frames
/// from the connection until the loop finishes or the client disconnects.
///
/// Returns `Some(Ok(...))` when the loop finishes successfully, or
/// `Some(Err(...))` when it finishes with an error. In either case the
/// caller must flush pending unsolicited errors and send `Done`/`Error`
/// to the client based on that inner result.
/// Returns `None` when the client disconnects mid-loop; the caller should
/// then return `ConnAction::Break`.
async fn drive_agentic_loop(
    state: &Arc<DaemonState>,
    writer: &Arc<tokio::sync::Mutex<tokio::net::unix::OwnedWriteHalf>>,
    conn_state: &mut ConnState<'_>,
    expected_sid: &str,
    messages: Vec<Message>,
    model: &str,
) -> Option<anyhow::Result<(String, usize, Vec<Message>, String)>> {
    let (steer_tx, mut steer_rx) = tokio::sync::mpsc::channel::<Option<String>>(16);
    let writer_loop = Arc::clone(writer);
    let state_loop = Arc::clone(state);
    let model_loop = model.to_owned();
    let sid_loop = expected_sid.to_owned();
    let mut loop_handle = tokio::spawn(async move {
        let mut w = writer_loop.lock().await;
        run_agentic_loop(
            &state_loop,
            &model_loop,
            messages,
            &mut *w,
            &mut steer_rx,
            true,
            Some(&sid_loop),
        )
        .await
    });
    let result = loop {
        tokio::select! { biased;
            r = &mut loop_handle => {
                break match r {
                    Ok(result) => result,
                    Err(e) if e.is_cancelled() => Err(anyhow::anyhow!("loop cancelled: {e}")),
                    Err(e) => Err(anyhow::anyhow!("loop panicked: {e}")),
                };
            }
            f = conn_state.frame_rx.recv() => match f {
                None => {
                    loop_handle.abort();
                    return None;
                }
                Some(line) => {
                    if let Ok(req) = serde_json::from_str::<Request>(&line) {
                        match req {
                            Request::Steer { session_id: sid, message } if sid == expected_sid => {
                                if !message.is_empty() {
                                    let _ = steer_tx.send(Some(message)).await;
                                }
                            }
                            Request::Interrupt { session_id: sid } if sid == expected_sid => {
                                let _ = steer_tx.send(None).await;
                            }
                            Request::Steer { .. } | Request::Interrupt { .. } => {
                                tracing::debug!("ignoring steer/interrupt for non-active session");
                            }
                            _ => {
                                tracing::warn!("dropping unsolicited frame during active agentic loop");
                                *conn_state.pending_unsolicited += 1;
                            }
                        }
                    }
                }
            },
        }
    };
    Some(result)
}

// ---------------------------------------------------------------------------
// Sub-handlers — one per Request variant
// ---------------------------------------------------------------------------

/// Handle `Request::ClaudeLaunch`: assign tmux panes and launch `claude`
/// (Claude Code CLI) sessions for each task.
///
/// Steps for each task:
/// 1. `ensure_and_acquire_idle` — atomically expand the pane pool if needed
///    and acquire an idle pane.
/// 2. Rename the pane title to `"cc-{N}"` (tmux pane numeric index).
/// 3. If the pane already has `claude` running, inject `description` as a new
///    prompt.  Otherwise launch `claude` (with optional `cd <worktree>`) via
///    `tmux send-keys`, then inject `description` as a second keystroke.
/// 4. Stream `Response::PaneAssigned` for each task, then `Response::Done`.
async fn handle_claude_launch(
    writer: &Arc<tokio::sync::Mutex<tokio::net::unix::OwnedWriteHalf>>,
    tasks: Vec<crate::ipc::TaskSpec>,
) -> Result<()> {
    if tasks.is_empty() {
        let mut w = writer.lock().await;
        write_frame(&mut *w, &Response::Done).await?;
        return Ok(());
    }

    // For each task: acquire a pane (auto-expanding the pool if needed), then
    // launch `claude` (or inject a prompt into an already-running session) via
    // tmux send-keys.
    // `ensure_and_acquire_idle` holds a single LOCK_EX for both expansion and
    // acquisition, eliminating the TOCTOU race.
    let total_tasks = tasks.len();
    for (task_idx, task) in tasks.into_iter().enumerate() {
        let task_id = task.task_id.clone();
        let auto_enter = task.auto_enter;

        // Gather git context from the client's working directory: current
        // branch, remote URL, recent commits, and PR-specific information if
        // the task description mentions a PR number.  The context is prepended
        // to the description so Claude knows where to start, what branch it is
        // on, and how to push when done.
        let (description, ctx_start_branch) = {
            let raw_desc = task.description.clone();
            let cwd = task.client_cwd.clone();
            tokio::task::spawn_blocking(move || {
                let ctx = gather_task_context(cwd.as_deref(), &raw_desc);
                let enriched = format!("{}\n{}", ctx.preamble, raw_desc);
                (enriched, ctx.start_branch)
            })
            .await
            .unwrap_or_else(|_| (task.description.clone(), None))
        };

        // Each parallel Claude session must work in its own git worktree so
        // concurrent tasks cannot trample each other's in-progress file edits.
        // Auto-create a worktree when the caller did not supply one explicitly.
        // Track whether the worktree was auto-created so we can clean it up
        // if pane acquisition subsequently fails (avoiding orphaned branches).
        let was_explicit_worktree = task.worktree.is_some();
        let worktree: Option<String> = match task.worktree {
            Some(wt) => Some(wt),
            None => {
                let tid = task_id.clone();
                let cwd = task.client_cwd.clone();
                let base = ctx_start_branch.clone();
                match tokio::task::spawn_blocking(move || {
                    create_task_worktree(&tid, cwd.as_deref(), base.as_deref())
                })
                .await
                .context("create_task_worktree panicked")?
                {
                    Ok(path) => Some(path.to_string_lossy().into_owned()),
                    Err(e) => {
                        tracing::warn!(
                            task_id = %task_id,
                            error = %e,
                            "auto-worktree creation failed; launching claude without worktree isolation"
                        );
                        None
                    }
                }
            }
        };

        let tid_for_lease = task_id.clone();
        let wt_for_lease = worktree.clone();

        // Acquire a pane lease *before* creating session state so that a
        // capacity failure does not leave orphan session entries on disk.
        // A placeholder session_id is stored now; it is corrected to the real
        // UUID via `update_session_id` after `session::get_or_create` returns.
        let sid_placeholder = uuid::Uuid::new_v4().to_string();
        let sid_for_lease = sid_placeholder.clone();
        let pane_result = tokio::task::spawn_blocking(move || {
            pane_lease::ensure_and_acquire_idle(
                &tid_for_lease,
                &sid_for_lease,
                wt_for_lease.as_deref(),
            )
        })
        .await
        .context("ensure_and_acquire_idle task panicked")?;

        let (pane_id, had_claude) = match pane_result {
            Ok(p) => p,
            Err(e) => {
                // If the worktree was auto-created, remove it and its branch
                // to avoid orphaned state after a capacity error.
                // Use client_cwd with -C so git targets the right repo
                // regardless of where the daemon was started.
                // The branch name equals the worktree directory's basename
                // (both set to unique_name = "<task_id>-<uuid8>").
                if !was_explicit_worktree {
                    if let Some(ref wt) = worktree {
                        let wt_path = wt.clone();
                        let cleanup_cwd = task.client_cwd.clone();
                        tokio::task::spawn_blocking(move || {
                            let branch = std::path::Path::new(&wt_path)
                                .file_name()
                                .and_then(|n| n.to_str())
                                .map(str::to_string);
                            let mut rm_cmd = std::process::Command::new("git");
                            if let Some(ref cwd) = cleanup_cwd {
                                rm_cmd.args(["-C", cwd.as_str()]);
                            }
                            let removed = rm_cmd
                                .args(["worktree", "remove", "--force", &wt_path])
                                .output()
                                .map(|o| o.status.success())
                                .unwrap_or(false);
                            if removed {
                                if let (Some(ref cwd), Some(ref br)) = (&cleanup_cwd, &branch) {
                                    let _ = std::process::Command::new("git")
                                        .args(["-C", cwd.as_str(), "branch", "-D", br.as_str()])
                                        .output();
                                }
                            }
                        })
                        .await
                        .ok();
                    }
                }
                // Surface CapacityError as a typed terminal response.
                // Report tasks still unassigned (including this one) so the
                // caller sees the real demand that exceeded capacity.
                let remaining = total_tasks - task_idx;
                let mut w = writer.lock().await;
                if let Some(cap) = e.downcast_ref::<pane_lease::CapacityError>() {
                    write_frame(
                        &mut *w,
                        &Response::CapacityError {
                            requested: remaining,
                            max_panes: cap.max_panes,
                            current_busy: cap.current_busy,
                        },
                    )
                    .await?;
                } else {
                    write_frame(
                        &mut *w,
                        &Response::Error {
                            message: format!("[error] {e:#}"),
                        },
                    )
                    .await?;
                }
                // Both Error and CapacityError are terminal: the client
                // breaks its read loop on either, so return immediately.
                return Ok(());
            }
        };

        // Resolve the real session UUID now that the pane is secured, then
        // correct the placeholder stored in the lease.
        // Prefer the worktree path for session identity; fall back to the
        // client's cwd (not the daemon's cwd, which may be unrelated to the
        // repo the client was invoked from).
        let session_dir = worktree
            .as_deref()
            .map(std::path::Path::new)
            .map(std::path::Path::to_path_buf)
            .or_else(|| task.client_cwd.as_deref().map(std::path::PathBuf::from))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
        // If session resolution fails, release the pane so it doesn't get
        // stuck Busy until TTL expiry.
        let session_id_result =
            tokio::task::spawn_blocking(move || session::get_or_create(&session_dir))
                .await
                .map_err(|e| anyhow::anyhow!("session::get_or_create panicked: {e}"))
                .and_then(|r| r.context("resolving session ID"));
        let session_id = match session_id_result {
            Ok(id) => id,
            Err(e) => {
                let failed_pane = pane_id.clone();
                tokio::task::spawn_blocking(move || {
                    let _ = pane_lease::release_lease(&failed_pane);
                })
                .await
                .ok();
                let mut w = writer.lock().await;
                write_frame(
                    &mut *w,
                    &Response::Error {
                        message: format!("[error] {e:#}"),
                    },
                )
                .await?;
                return Ok(());
            }
        };
        let update_pane = pane_id.clone();
        let update_sid = session_id.clone();
        if let Err(e) = tokio::task::spawn_blocking(move || {
            pane_lease::update_session_id(&update_pane, &update_sid)
        })
        .await
        .map_err(|e| anyhow::anyhow!("update_session_id panicked: {e}"))
        .and_then(|r| r)
        {
            tracing::warn!(pane_id = %pane_id, error = %e, "failed to persist session ID in pane lease; tmux-state.json may be inconsistent");
        }

        // Rename the pane for visibility.  Use the tmux pane numeric index
        // (strip the leading '%' from e.g. "%5") to keep the title short.
        let rename_pane = pane_id.clone();
        let pane_num = pane_id.trim_start_matches('%');
        let rename_title = format!("cc-{}", pane_num);
        tokio::task::spawn_blocking(move || {
            // Best-effort — ignore errors (non-tmux environments).
            let _ = pane_lease::rename_pane(&rename_pane, &rename_title);
        })
        .await
        .ok();

        // Build the key sequences to inject into the pane.
        //
        // Priority:
        //  - `had_claude = true`: pane already has `claude` running at its
        //    prompt → send just the description as a new user message.
        //  - `had_claude = false`: pane is blank (freshly created or at a
        //    shell prompt) → launch `claude` first, then send the description
        //    as a second keystroke so it lands at the Claude Code prompt.
        //
        // Each element is (keys, press_enter).
        let key_sequence: Vec<(String, bool)> = if had_claude {
            // Reusing an existing claude session in the same worktree: compact
            // the prior conversation first so stale context does not pollute
            // the new task, then inject the description.
            vec![
                ("/compact".to_string(), true),
                (description.clone(), auto_enter),
            ]
        } else {
            // Fresh pane: launch claude with --dangerously-skip-permissions so
            // the autonomous session never blocks on an interactive approval
            // prompt, then inject the description as the opening message.
            let launch_cmd = if let Some(ref wt) = worktree {
                format!(
                    "cd {} && claude --dangerously-skip-permissions",
                    shell_escape(wt)
                )
            } else {
                "claude --dangerously-skip-permissions".to_string()
            };
            vec![(launch_cmd, true), (description.clone(), auto_enter)]
        };

        let send_pane = pane_id.clone();

        // Send key sequences to the pane.  Each step: send text literally
        // with `send-keys -l --`, then send Enter as a separate key press.
        //
        // Timing: 1 s pause before the first send (bash init), 5 s before
        // subsequent sends (e.g. claude startup), then 1 s after each text
        // injection before pressing Enter.  No prompt-polling — simple
        // fixed delays are robust against prompt character variations
        // (❯, >, $, etc.).
        let send_result = tokio::task::spawn_blocking(move || {
            for (idx, (keys, press_enter)) in key_sequence.iter().enumerate() {
                // Before the very first send: let the new pane's shell
                // initialise (.bashrc, prompt rendering, etc.).
                // Before subsequent sends (e.g. description after claude
                // launch): let the target process start up.
                let wait = if idx == 0 { 1 } else { 5 };
                std::thread::sleep(std::time::Duration::from_secs(wait));

                // For the description injection (idx > 0 on a fresh pane),
                // dismiss any Claude Code splash/welcome overlay first.
                // Escape is a safe no-op if no overlay is active.
                if idx > 0 && !had_claude {
                    let _ = std::process::Command::new("tmux")
                        .args(["send-keys", "-t", &send_pane, "Escape"])
                        .output();
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }

                // Send text literally.
                let out = std::process::Command::new("tmux")
                    .args(["send-keys", "-t", &send_pane, "-l", "--", keys.as_str()])
                    .output()?;
                if !out.status.success() {
                    return Ok::<Option<String>, std::io::Error>(Some(
                        String::from_utf8_lossy(&out.stderr).trim().to_string(),
                    ));
                }
                if *press_enter {
                    // Wait for the TUI to render and accept the pasted text,
                    // then press Enter.
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    let out = std::process::Command::new("tmux")
                        .args(["send-keys", "-t", &send_pane, "Enter"])
                        .output()?;
                    if !out.status.success() {
                        return Ok(Some(
                            String::from_utf8_lossy(&out.stderr).trim().to_string(),
                        ));
                    }
                }
            }
            Ok(None)
        })
        .await;

        let tmux_err = match send_result {
            Ok(Ok(None)) => None,
            Ok(Ok(Some(stderr))) => Some(stderr),
            Ok(Err(e)) => Some(e.to_string()),
            Err(e) => Some(format!("send-keys task panicked: {e}")),
        };
        if let Some(err_msg) = tmux_err {
            // Injection failed.  If tmux reports the pane no longer exists,
            // remove it from the lease map entirely so the scheduler stops
            // selecting it.  Otherwise just release it back to Idle.
            let failed_pane = pane_id.clone();
            let is_stale = err_msg.contains("unknown pane")
                || err_msg.contains("can't find pane")
                || err_msg.contains("no server running");
            tokio::task::spawn_blocking(move || {
                if is_stale {
                    let _ = pane_lease::remove_pane(&failed_pane);
                } else {
                    let _ = pane_lease::release_lease(&failed_pane);
                }
            })
            .await
            .ok();
            let mut w = writer.lock().await;
            write_frame(
                &mut *w,
                &Response::Error {
                    message: format!(
                        "[error] failed to inject command into pane {pane_id}: {err_msg}"
                    ),
                },
            )
            .await?;
            return Ok(());
        }

        // If we just launched `claude` in a blank pane, record that so
        // future task assignments can inject prompts directly.
        if !had_claude {
            let started_pane = pane_id.clone();
            tokio::task::spawn_blocking(move || {
                let _ = pane_lease::mark_claude_started(&started_pane);
            })
            .await
            .ok();
        }

        let mut w = writer.lock().await;
        write_frame(
            &mut *w,
            &Response::PaneAssigned {
                task_id: task.task_id,
                pane_id,
                session_id,
            },
        )
        .await?;
    }

    let mut w = writer.lock().await;
    write_frame(&mut *w, &Response::Done).await?;
    Ok(())
}

/// Capture the last 200 lines of a tmux pane as plain text.
/// Returns an empty string on failure so supervision can continue.
///
/// 200-line window matches Claude Code's own default capture depth — enough
/// that a busy pane (tool outputs, build logs) still shows what the Claude
/// agent most recently did, not just the idle prompt that followed.
fn capture_pane_text(pane_id: &str) -> String {
    match std::process::Command::new("tmux")
        .args(["capture-pane", "-t", pane_id, "-p", "-S", "-200"])
        .output()
    {
        Ok(output) => {
            if !output.status.success() {
                tracing::warn!(
                    pane_id,
                    status = %output.status,
                    stderr = %String::from_utf8_lossy(&output.stderr),
                    "tmux capture-pane failed"
                );
                return String::new();
            }
            String::from_utf8_lossy(&output.stdout).into_owned()
        }
        Err(e) => {
            tracing::warn!(pane_id, error = %e, "failed to spawn tmux capture-pane");
            String::new()
        }
    }
}

/// Wait until the target pane's output has been stable for `idle_secs`, or
/// until `timeout` elapses, then return the final snapshot.
///
/// This is the supervision-side equivalent of the `tmux_wait` LLM tool: the
/// supervisor gets a *stable* pane snapshot (not mid-render garbage) but
/// also gives up after `timeout` so a long-running Claude session cannot
/// block supervision indefinitely.
///
/// Returns `(snapshot, idle)` — `idle=true` when the pane stabilised before
/// timeout, `idle=false` when we hit the timeout with the pane still active.
/// Either way the snapshot is the latest captured content, suitable for
/// feeding to the supervision LLM.
async fn wait_for_pane_idle(
    pane_id: &str,
    idle: std::time::Duration,
    timeout: std::time::Duration,
    poll: std::time::Duration,
) -> (String, bool) {
    let deadline = tokio::time::Instant::now() + timeout;
    let pid = pane_id.to_owned();
    let mut last_content = {
        let pid = pid.clone();
        tokio::task::spawn_blocking(move || capture_pane_text(&pid))
            .await
            .unwrap_or_default()
    };
    let mut stable_since = tokio::time::Instant::now();

    loop {
        if tokio::time::Instant::now() >= deadline {
            return (last_content, false);
        }
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let sleep_dur = poll.min(remaining);
        tokio::time::sleep(sleep_dur).await;

        let pid = pid.clone();
        let content = tokio::task::spawn_blocking(move || capture_pane_text(&pid))
            .await
            .unwrap_or_default();
        if content != last_content {
            last_content = content;
            stable_since = tokio::time::Instant::now();
        } else if stable_since.elapsed() >= idle {
            return (last_content, true);
        }
    }
}

/// Send literal text + Enter to a tmux pane (best-effort).
///
/// Sends the text with `send-keys -l --` (literal, no keyname interpretation),
/// pauses for 1 s to let the receiving TUI's paste buffer drain, then sends
/// Enter as a separate key press.  The pause matches the `handle_claude_launch`
/// injection path (daemon.rs:1081) and exists because Claude Code's TUI can
/// swallow or defer the trailing Enter when it arrives before the pasted text
/// has been rendered into the input field — which manifests as a STEER
/// message appearing in the pane input but never submitting.
fn send_pane_keys(pane_id: &str, text: &str) {
    match std::process::Command::new("tmux")
        .args(["send-keys", "-t", pane_id, "-l", "--", text])
        .status()
    {
        Ok(s) if !s.success() => {
            tracing::warn!(pane_id, status = %s, "tmux send-keys (text) failed");
        }
        Err(e) => {
            tracing::warn!(pane_id, error = %e, "failed to spawn tmux send-keys (text)");
        }
        _ => {}
    }
    // Let the TUI process the pasted text before pressing Enter.
    std::thread::sleep(std::time::Duration::from_secs(1));
    match std::process::Command::new("tmux")
        .args(["send-keys", "-t", pane_id, "Enter"])
        .status()
    {
        Ok(s) if !s.success() => {
            tracing::warn!(pane_id, status = %s, "tmux send-keys (Enter) failed");
        }
        Err(e) => {
            tracing::warn!(pane_id, error = %e, "failed to spawn tmux send-keys (Enter)");
        }
        _ => {}
    }
}

/// Handle `Request::SupervisePanes`: run a Rust polling loop that captures pane
/// content, calls the LLM for analysis (no tools), and acts on the response.
///
/// The loop iterates with a 60-second sleep between turns (override with
/// `AMAEBI_SUPERVISION_INTERVAL_SECS`). Each turn the LLM
/// returns exactly one of:
/// - `WAIT` — still working, check again
/// - `STEER: <pane_id>: <message>` — send a correction to the pane
/// - `DONE: <summary>` — task is complete; stream the summary and exit
///
/// The loop can also be interrupted by an `Interrupt` frame arriving on
/// `frame_rx`.  A maximum of 240 completion tokens is requested (sufficient for
/// the short WAIT/STEER/DONE responses).
async fn handle_supervision(
    writer: &Arc<tokio::sync::Mutex<tokio::net::unix::OwnedWriteHalf>>,
    frame_rx: &mut tokio::sync::mpsc::Receiver<String>,
    panes: Vec<crate::ipc::SupervisionTarget>,
    model: String,
    state: &Arc<DaemonState>,
    session_id: Option<String>,
) -> Result<()> {
    // Max time to wait between LLM checks.  Default 5 min; override with
    // AMAEBI_SUPERVISION_INTERVAL_SECS.  This is the *ceiling*: each iteration
    // actually waits for the pane to go idle (see `IDLE_SECS` below) so that
    // the snapshot fed to the LLM is stable, and only falls back to the
    // ceiling when Claude is continuously producing output.  The higher
    // ceiling reduces supervision LLM cost when Claude is in a long
    // compile / test / think phase where the pane barely changes for
    // minutes at a time.
    let poll_interval = tokio::time::Duration::from_secs(
        std::env::var("AMAEBI_SUPERVISION_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(300u64) // 5 min
            .max(1), // clamp to >= 1 s to prevent a tight loop
    );
    // How long a pane must be unchanged to be considered idle before
    // triggering a supervision LLM call.  10 s is a deliberately loose
    // threshold: short inter-tool-call pauses (2-5 s) during active work
    // no longer trigger a supervision check, so the LLM only runs when
    // Claude has genuinely paused (waiting for user input, finished,
    // stuck on an error).  Reduces noise checks and LLM cost by ~70-80 %
    // in typical sessions.
    const IDLE_SECS: u64 = 10;
    const IDLE_POLL_SECS: u64 = 2;

    // Hard wall-clock limit before supervision gives up. Default 10 hours;
    // override with AMAEBI_SUPERVISION_TIMEOUT_SECS.
    let max_duration = std::time::Duration::from_secs(
        std::env::var("AMAEBI_SUPERVISION_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10 * 3600u64), // 10 hours default — enough for a night shift
    );

    const MAX_SUPERVISION_TOKENS: usize = 1024;

    let supervision_start = std::time::Instant::now();
    let deadline = supervision_start + max_duration;
    let mut turn: u64 = 0;

    // Remember the previous turn's verdict so the LLM has continuity across
    // iterations (e.g. "last turn you STEERed X; did Claude pick it up?").
    // Stored as a short one-line summary that we feed back into `user_content`
    // on the next iteration.
    let mut last_verdict: Option<String> = None;

    // Load skill files (SOUL.md, AGENTS.md, GPU_KERNEL.md) once and reuse
    // across all supervision turns so the LLM has project context for
    // higher-quality STEER decisions.
    let skill_msgs = load_skill_messages().await;

    let system_prompt =
        "You are supervising a Claude Code session executing a task in a tmux pane.\n\
        Compare the pane content to the task description and respond with EXACTLY ONE of:\n\
        \n\
        WAIT: <one sentence — what is Claude currently doing?>\n\
          Claude is actively working AND on track. No intervention needed.\n\
          Use WAIT as the default when you're not certain between WAIT and DONE/STEER.\n\
        \n\
        STEER: <pane_id>: <message to send>\n\
          Claude needs input or correction. The message will be typed into the pane.\n\
          Use STEER when:\n\
          - Claude is at an idle prompt and is ASKING A QUESTION or presenting OPTIONS\n\
            (e.g. 'Should I also do X?', 'Which approach would you prefer?') — answer it.\n\
          - Claude reports partial completion and the task description is NOT fully done yet\n\
            (e.g. claims 'step 1 done' but step 2 was also requested).\n\
          - Claude is stuck on an error and a concrete hint will unblock it.\n\
          - Claude is going in the wrong direction vs. the task description.\n\
        \n\
        DONE: <paragraph summary of what was accomplished>\n\
          The task is FULLY complete. Require ALL of:\n\
          1. Pane shows an explicit completion signal — a finished report, passing tests,\n\
             a merged PR URL, 'done', '✓', 'all tests passed', or similar.\n\
          2. That completion directly covers the task description given at session start\n\
             (not just a sub-step or unrelated output).\n\
          3. Claude is no longer working (idle prompt or returned to shell).\n\
          An idle prompt alone is NOT sufficient — Claude may be waiting for user input\n\
          (see STEER) or may have been interrupted. If unsure, prefer WAIT and re-check\n\
          next turn.\n\
        \n\
        Your response must start with WAIT:, STEER:, or DONE: — nothing else before it."
            .to_string();

    loop {
        // --- Check wall-clock deadline ---
        if std::time::Instant::now() >= deadline {
            let mut w = writer.lock().await;
            write_frame(
                &mut *w,
                &Response::Text {
                    chunk: format!(
                        "[supervision] timeout after {:.1} hours; stopping\n",
                        max_duration.as_secs_f64() / 3600.0
                    ),
                },
            )
            .await?;
            write_frame(&mut *w, &Response::Done).await?;
            return Ok(());
        }

        // --- Interruptible wait-for-pane-idle (skip on first iteration) ---
        // Instead of a blind fixed-interval sleep, wait until the first pane
        // in the set has been stable for IDLE_SECS seconds, falling back to
        // `poll_interval` as the hard ceiling if the pane stays active.  This
        // guarantees the snapshot we feed to the LLM is not mid-render and
        // that busy Claude sessions do not waste LLM calls on near-identical
        // frames.  Client `Interrupt` frames still preempt the wait.
        if turn > 0 {
            let primary_pane = panes.first().map(|t| t.pane_id.clone()).unwrap_or_default();
            let wait_fut = wait_for_pane_idle(
                &primary_pane,
                std::time::Duration::from_secs(IDLE_SECS),
                poll_interval,
                std::time::Duration::from_secs(IDLE_POLL_SECS),
            );
            tokio::pin!(wait_fut);
            let interrupted = loop {
                tokio::select! {
                    biased;
                    frame = frame_rx.recv() => {
                        match frame {
                            None => {
                                // Client disconnected.
                                return Ok(());
                            }
                            Some(line) => {
                                if let Ok(Request::Interrupt { session_id: sid }) =
                                    serde_json::from_str::<Request>(&line)
                                {
                                    if session_id
                                        .as_deref()
                                        .is_none_or(|expected| sid == expected)
                                    {
                                        break true;
                                    }
                                }
                            }
                        }
                    }
                    _ = &mut wait_fut => break false,
                }
            };
            if interrupted {
                let mut w = writer.lock().await;
                write_frame(
                    &mut *w,
                    &Response::Text {
                        chunk: "[supervision] interrupted by user\n".into(),
                    },
                )
                .await?;
                write_frame(&mut *w, &Response::Done).await?;
                return Ok(());
            }
        }

        turn += 1;
        let elapsed_mins = supervision_start.elapsed().as_secs() / 60;

        // --- Capture pane snapshots (full for LLM, tail for display) ---
        struct PaneSnapshot {
            pane_id: String,
            task_description: String,
            full_content: String, // sent to LLM
            tail: String,         // last 8 lines shown to user
        }

        let mut snapshots: Vec<PaneSnapshot> = Vec::new();
        for target in &panes {
            let pid = target.pane_id.clone();
            let full = tokio::task::spawn_blocking(move || capture_pane_text(&pid))
                .await
                .unwrap_or_default();
            // Last 8 non-empty lines for the user-visible tail.
            let tail = full
                .lines()
                .filter(|l| !l.trim().is_empty())
                .rev()
                .take(8)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n");
            snapshots.push(PaneSnapshot {
                pane_id: target.pane_id.clone(),
                task_description: target.task_description.clone(),
                full_content: full,
                tail,
            });
        }

        // Print the check header + pane tails to the user before calling LLM.
        {
            let mut header = format!("\n[supervision +{elapsed_mins}m check #{turn}]\n");
            for snap in &snapshots {
                header.push_str(&format!(
                    "  ┌─ pane {} — {}\n",
                    snap.pane_id, snap.task_description
                ));
                for line in snap.tail.lines() {
                    header.push_str(&format!("  │ {line}\n"));
                }
                header.push_str("  └─\n");
            }
            let mut w = writer.lock().await;
            write_frame(&mut *w, &Response::Text { chunk: header }).await?;
        }

        // Security note: `snap.full_content` is the output of `capture_pane_text`,
        // which captures only the last 60 visible lines of the pane.  This bounds
        // the amount of data sent to the LLM, but those lines could still contain
        // secrets (e.g. env vars printed by a build script).  A future improvement
        // could add redaction of common secret patterns.
        let mut pane_snapshots = String::new();
        for snap in &snapshots {
            pane_snapshots.push_str(&format!(
                "=== Pane {} — task: {} ===\n{}\n",
                snap.pane_id, snap.task_description, snap.full_content
            ));
        }

        // Thread the previous verdict into the prompt so the LLM can judge
        // whether its last STEER landed, whether Claude actually finished
        // the step it last reported, etc.  First iteration sends "(none)".
        let prior = last_verdict
            .as_deref()
            .unwrap_or("(none yet — this is the first check)");
        let user_content = format!(
            "Current pane snapshots (check #{turn}, elapsed {elapsed_mins}m):\n\n\
             {pane_snapshots}\n\
             Your previous verdict this session: {prior}"
        );

        let mut messages = vec![Message::system(system_prompt.clone())];
        // Inject SOUL.md / AGENTS.md etc. right after the system message
        // so the supervision LLM has full project context.
        splice_skill_messages(&mut messages, skill_msgs.clone());
        messages.push(Message::user(user_content));

        // --- Drain any pending interrupts before invoking the model ---
        // The model call is short (240 tokens max) so mid-call interrupts
        // are not critical; they will be handled at the next sleep interval.
        while let Ok(line) = frame_rx.try_recv() {
            if let Ok(Request::Interrupt { session_id: isid }) =
                serde_json::from_str::<Request>(&line)
            {
                if session_id
                    .as_deref()
                    .is_none_or(|expected| isid == expected)
                {
                    let mut w = writer.lock().await;
                    write_frame(
                        &mut *w,
                        &Response::Text {
                            chunk: "[supervision] interrupted\n".into(),
                        },
                    )
                    .await?;
                    write_frame(&mut *w, &Response::Done).await?;
                    return Ok(());
                }
            }
        }

        // --- Invoke model (no tools) ---
        // Use sink() so the raw LLM tokens are NOT streamed to the client.
        // We write our own formatted one-line status after parsing the response.
        let response_text = {
            let mut sink = tokio::io::sink();
            match invoke_model(
                state,
                &model,
                &messages,
                &[],
                MAX_SUPERVISION_TOKENS,
                &mut sink,
            )
            .await
            {
                Ok(r) => r.text,
                Err(e) => {
                    let mut w = writer.lock().await;
                    write_frame(
                        &mut *w,
                        &Response::Error {
                            message: format!("[supervision] model error: {e:#}"),
                        },
                    )
                    .await?;
                    return Ok(());
                }
            }
        };

        let trimmed = response_text.trim();

        // --- Parse response and act ---
        let verdict_line = if trimmed.starts_with("DONE:") || trimmed == "DONE" {
            let summary = trimmed.strip_prefix("DONE:").unwrap_or("").trim();
            let mut w = writer.lock().await;
            write_frame(
                &mut *w,
                &Response::Text {
                    chunk: format!("  → DONE\n\n{summary}\n"),
                },
            )
            .await?;
            write_frame(&mut *w, &Response::Done).await?;
            return Ok(());
        } else if let Some(rest) = trimmed.strip_prefix("STEER:") {
            if let Some((pane_id_raw, message)) = rest.trim().split_once(':') {
                let pane_id = pane_id_raw.trim().to_owned();
                let message = message.trim().to_owned();
                let is_valid_pane = panes.iter().any(|t| t.pane_id == pane_id);
                if !pane_id.is_empty() && !message.is_empty() && is_valid_pane {
                    let pid = pane_id.clone();
                    let msg = message.clone();
                    tokio::task::spawn_blocking(move || send_pane_keys(&pid, &msg))
                        .await
                        .ok();
                    format!("  → STEER {pane_id}: {message}\n")
                } else if !pane_id.is_empty() && !message.is_empty() && !is_valid_pane {
                    format!("  → STEER {pane_id} (unknown pane, ignored)\n")
                } else {
                    "  → STEER (malformed response)\n".to_string()
                }
            } else {
                "  → STEER (malformed response)\n".to_string()
            }
        } else {
            // WAIT: <note> or bare WAIT
            let note = trimmed.strip_prefix("WAIT:").unwrap_or("").trim();
            if note.is_empty() {
                "  → WAIT\n".to_string()
            } else {
                format!("  → WAIT: {note}\n")
            }
        };
        // Remember this verdict (minus the display prefix) so the next
        // iteration can pass it back into the user prompt.  `verdict_line`
        // starts with "  → " and ends with "\n"; strip both for a compact
        // one-line carry-over.
        last_verdict = Some(
            verdict_line
                .trim_start_matches("  → ")
                .trim_end_matches('\n')
                .to_string(),
        );

        let mut w = writer.lock().await;
        write_frame(
            &mut *w,
            &Response::Text {
                chunk: verdict_line,
            },
        )
        .await?;
    }
}

/// Create a git worktree at `~/.amaebi/worktrees/<repo-name>/<task_id>-<uuid8>`
/// on a new branch named `<task_id>-<uuid8>`.
///
/// Every parallel `/claude` task needs its own worktree so that concurrent
/// Claude sessions editing the same repository do not trample each other's
/// in-progress changes.  Worktrees are stored under `~/.amaebi/worktrees/`
/// (alongside other amaebi state) rather than inside the repository directory,
/// which avoids polluting the project tree and requires no `.gitignore` entry.
/// A per-repo subdirectory (the basename of the git root) prevents collisions
/// across different repositories; the `-<uuid8>` suffix makes each
/// worktree/branch unique for a given `task_id` across runs.
///
/// `client_cwd` is the working directory of the invoking client.  Git is run
/// with `-C <client_cwd>` so the correct repository is targeted even when the
/// daemon was started from a different directory.
///
/// Returns the absolute path of the newly created worktree, or an error if:
/// - `task_id` contains unsafe characters (path separators, `..`), or
/// - the client's directory is not inside a git repository, or
/// - `git worktree add` fails (e.g. branch name already exists).
///
/// All git commands are synchronous; call this from `spawn_blocking`.
fn create_task_worktree(
    task_id: &str,
    client_cwd: Option<&str>,
    start_branch: Option<&str>,
) -> anyhow::Result<std::path::PathBuf> {
    use std::path::PathBuf;

    // Sanitize task_id: allow only characters that are safe as both a
    // filesystem path component and a git branch name.
    if task_id.is_empty()
        || task_id == ".."
        || task_id.contains('/')
        || task_id.contains('\\')
        || task_id.contains("..")
        || !task_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        anyhow::bail!(
            "task_id {:?} contains unsafe characters; \
             only ASCII alphanumerics, '-', '_', and '.' are allowed",
            task_id
        );
    }

    // Locate the repository root using the client's cwd so the daemon (which
    // may have been started from a different directory) targets the right repo.
    let mut git_cmd = std::process::Command::new("git");
    if let Some(cwd) = client_cwd {
        git_cmd.args(["-C", cwd]);
    }
    let out = git_cmd
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .context("spawning git rev-parse")?;
    if !out.status.success() {
        anyhow::bail!(
            "not in a git repository (git rev-parse failed: {})",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let git_root = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());

    // Build a repo namespace that combines the basename with a short hash of
    // the full path.  The basename alone is not unique: `~/src/api` and
    // `~/work/api` share the same name but are different repos.
    let repo_basename = git_root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("repo");
    let path_hash = {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        git_root.hash(&mut h);
        format!("{:016x}", h.finish())
            .chars()
            .take(8)
            .collect::<String>()
    };
    let repo_namespace = format!("{repo_basename}-{path_hash}");

    // Place worktrees under ~/.amaebi/worktrees/<repo-hash>/<task_id>-<uuid8>.
    // A short UUID suffix guarantees uniqueness across runs so repeated
    // invocations with the same task description never collide on the branch
    // name or directory path — the same approach Claude Code uses for its own
    // agent worktrees (agent-{uuid8}).
    let short_id = uuid::Uuid::new_v4()
        .to_string()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(8)
        .collect::<String>();
    let unique_name = format!("{task_id}-{short_id}");

    let wt_path = amaebi_home()?
        .join("worktrees")
        .join(&repo_namespace)
        .join(&unique_name);

    // Ensure the parent directory exists before calling git.
    let wt_parent = wt_path
        .parent()
        .context("worktree path has no parent directory")?;
    std::fs::create_dir_all(wt_parent)
        .with_context(|| format!("creating worktree parent directory {}", wt_parent.display()))?;

    // Create the worktree on a new branch named <task_id>-<uuid8>.
    // If a start_branch is provided (e.g. the head branch of a PR), the new
    // branch is forked from that branch rather than from HEAD.  This ensures
    // Claude starts with the right commits already in place.
    let mut git_cmd = std::process::Command::new("git");
    if let Some(cwd) = client_cwd {
        git_cmd.args(["-C", cwd]);
    }
    let wt_str = wt_path
        .to_str()
        .context("worktree path is not valid UTF-8")?;
    // If a base branch is provided, fetch it from origin first so the ref
    // exists locally.  Use `origin/<base>` as the start-point so we don't
    // require a local tracking branch.  If the fetch fails (e.g. the ref
    // doesn't exist on the remote), fall back to HEAD (omit the start-point).
    let fetched_base: Option<String> = start_branch.and_then(|base| {
        let mut fetch_cmd = std::process::Command::new("git");
        if let Some(cwd) = client_cwd {
            fetch_cmd.args(["-C", cwd]);
        }
        let fetch_ok = fetch_cmd
            .args(["fetch", "origin", base])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if fetch_ok {
            Some(format!("origin/{base}"))
        } else {
            tracing::warn!(
                branch = base,
                "git fetch origin failed; falling back to HEAD as worktree start-point"
            );
            None
        }
    });
    let mut args = vec!["worktree", "add", wt_str, "-b", &unique_name];
    if let Some(ref start_point) = fetched_base {
        args.push(start_point);
    }
    let out = git_cmd
        .args(&args)
        .output()
        .context("spawning git worktree add")?;
    if !out.status.success() {
        anyhow::bail!(
            "git worktree add failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(wt_path)
}

/// Escape a string for safe use as a single shell argument (single-quote wrapping).
fn shell_escape(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

// ---------------------------------------------------------------------------
// Task context enrichment
// ---------------------------------------------------------------------------

/// Context gathered from the client's git repository before launching a task.
struct TaskContext {
    /// One or two paragraph preamble to prepend to the task description sent
    /// to Claude so it knows which branch it is on, where to push, and any
    /// PR-specific information.
    preamble: String,
    /// If a PR number was detected in the task description and `gh` resolved
    /// the PR's head branch, this is the branch name to use as the base for
    /// the new worktree (instead of HEAD).
    start_branch: Option<String>,
}

/// Run a single git command with an optional `-C <cwd>` prefix.
/// Returns trimmed stdout on success, empty string on failure (best-effort).
fn git_output(cwd: Option<&str>, args: &[&str]) -> String {
    let mut cmd = std::process::Command::new("git");
    if let Some(c) = cwd {
        cmd.args(["-C", c]);
    }
    cmd.args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default()
        .trim()
        .to_string()
}

/// Extract the first PR number mentioned in a task description.
/// Matches patterns like "PR99", "PR #99", "pr 99", "#99".
fn extract_pr_number(description: &str) -> Option<u32> {
    // Walk through the string looking for a PR-like pattern.
    let lower = description.to_lowercase();
    // "pr" followed by optional space/# and digits.
    for (i, c) in lower.char_indices() {
        if c == 'p' {
            if lower[i..].starts_with("pr") {
                let after_pr = lower[i + 2..].trim_start_matches([' ', '#']);
                let digits: String = after_pr
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                if !digits.is_empty() {
                    if let Ok(n) = digits.parse::<u32>() {
                        return Some(n);
                    }
                }
            }
        } else if c == '#' {
            let after_hash = &lower[i + 1..];
            let digits: String = after_hash
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if !digits.is_empty() {
                if let Ok(n) = digits.parse::<u32>() {
                    return Some(n);
                }
            }
        }
    }
    None
}

/// Strip userinfo (credentials/tokens) from a remote URL to avoid leaking secrets
/// into LLM context.
///
/// Handles both standard URLs (`https://token@github.com/...` → `https://***@github.com/...`)
/// and SCP-style URLs (`token@github.com:org/repo.git` → `***@github.com:org/repo.git`).
fn sanitize_remote_url(url: &str) -> String {
    // https://token@github.com/... → https://***@github.com/...
    if let Some(scheme_end) = url.find("://") {
        let authority_start = scheme_end + 3;
        let authority_end = url[authority_start..]
            .find(['/', '?', '#'])
            .map(|i| authority_start + i)
            .unwrap_or(url.len());
        if let Some(at_rel) = url[authority_start..authority_end].find('@') {
            let at_pos = authority_start + at_rel;
            return format!("{}***@{}", &url[..authority_start], &url[at_pos + 1..]);
        }
        return url.to_string();
    }
    // SCP-style: token@github.com:org/repo.git → ***@github.com:org/repo.git
    if let Some(at_pos) = url.find('@') {
        if let Some(colon_rel) = url[at_pos + 1..].find(':') {
            let colon_pos = at_pos + 1 + colon_rel;
            let userinfo = &url[..at_pos];
            let host = &url[at_pos + 1..colon_pos];
            if !userinfo.is_empty()
                && !host.is_empty()
                && !userinfo.contains('/')
                && !host.contains('/')
            {
                return format!("***@{}", &url[at_pos + 1..]);
            }
        }
    }
    url.to_string()
}

/// Gather git context from `client_cwd` and return a preamble to prepend to
/// the task description plus an optional base branch for the worktree.
///
/// All subprocess calls are best-effort — failures produce empty strings so
/// the task is never blocked by unavailable tooling (e.g. no `gh` CLI).
fn gather_task_context(client_cwd: Option<&str>, description: &str) -> TaskContext {
    let branch = git_output(client_cwd, &["rev-parse", "--abbrev-ref", "HEAD"]);
    let remote = git_output(client_cwd, &["remote", "get-url", "origin"]);
    let log = git_output(client_cwd, &["log", "--oneline", "-5"]);

    let mut lines: Vec<String> = Vec::new();
    lines.push("=== Repository context (injected by amaebi) ===".into());
    if !branch.is_empty() {
        lines.push(format!("Current branch : {branch}"));
    }
    if !remote.is_empty() {
        lines.push(format!("Remote origin  : {}", sanitize_remote_url(&remote)));
    }
    if !log.is_empty() {
        lines.push("Recent commits :".into());
        for commit_line in log.lines() {
            lines.push(format!("  {commit_line}"));
        }
    }

    // --- PR-specific context ---
    let mut start_branch: Option<String> = None;
    if let Some(pr_num) = extract_pr_number(description) {
        // Try gh pr view to get the head branch and title.
        let gh_json = std::process::Command::new("gh")
            .args([
                "pr",
                "view",
                &pr_num.to_string(),
                "--json",
                "headRefName,title,baseRefName",
            ])
            .current_dir(client_cwd.unwrap_or("."))
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .unwrap_or_default();

        if !gh_json.is_empty() {
            // Parse the JSON minimally (avoid a heavy dep; fields are simple strings).
            let head = json_str_field(&gh_json, "headRefName");
            let base = json_str_field(&gh_json, "baseRefName");
            let title = json_str_field(&gh_json, "title");

            lines.push(format!("PR #{pr_num}"));
            if !title.is_empty() {
                lines.push(format!("  Title      : {title}"));
            }
            if !head.is_empty() {
                lines.push(format!("  Head branch: {head}"));
                start_branch = Some(head.clone());
            }
            if !base.is_empty() {
                lines.push(format!("  Base branch: {base}"));
            }
        }
    }

    if !branch.is_empty() || start_branch.is_some() {
        lines.push("When your changes are ready, push with: git push -u origin HEAD".to_string());
    }

    lines.push("=== End of injected context ===".into());
    lines.push(String::new()); // blank line before the actual task

    TaskContext {
        preamble: lines.join("\n"),
        start_branch,
    }
}

/// Extract a string field value from a simple flat JSON object.
/// Only handles the case where the value is a JSON string (double-quoted).
/// Returns an empty string on failure.
fn json_str_field(json: &str, field: &str) -> String {
    serde_json::from_str::<serde_json::Value>(json)
        .ok()
        .and_then(|v| v.get(field)?.as_str().map(str::to_owned))
        .unwrap_or_default()
}

/// Handle `Request::ClearMemory`: clear the memory DB and send `Done`.
async fn handle_clear_memory(
    state: &DaemonState,
    writer: &Arc<tokio::sync::Mutex<tokio::net::unix::OwnedWriteHalf>>,
) -> Result<()> {
    tracing::info!("received memory clear request");
    if let Err(e) = with_db(Arc::clone(&state.db), memory_db::clear).await {
        tracing::warn!(error = %e, "failed to clear memory DB");
    }
    let mut w = writer.lock().await;
    write_frame(&mut *w, &Response::Done).await
}

/// Handle `Request::StoreMemory`: persist a user/assistant pair and send `Done`.
async fn handle_store_memory(
    state: &DaemonState,
    writer: &Arc<tokio::sync::Mutex<tokio::net::unix::OwnedWriteHalf>>,
    user: &str,
    assistant: &str,
) -> Result<()> {
    store_conversation(state, "global", user, assistant).await;
    let mut w = writer.lock().await;
    write_frame(&mut *w, &Response::Done).await
}

/// Handle `Request::RetrieveContext`: look up memory entries and stream them to the client.
async fn handle_retrieve_context(
    state: &DaemonState,
    writer: &Arc<tokio::sync::Mutex<tokio::net::unix::OwnedWriteHalf>>,
    prompt: String,
) -> Result<()> {
    let entries = with_db(Arc::clone(&state.db), move |conn| {
        memory_db::retrieve_context(conn, &prompt, 4, 10)
    })
    .await
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
                content: truncate_chars(&entry.content, MAX_HISTORY_CHARS),
            },
        )
        .await?;
    }
    write_frame(&mut *w, &Response::Done).await
}

/// Handle `Request::Resume`: load history, run the agentic loop, store the result.
///
/// Returns `ConnAction::Break` when the client disconnects during the loop or
/// when authentication fails.
async fn handle_resume_request(
    state: &Arc<DaemonState>,
    writer: &Arc<tokio::sync::Mutex<tokio::net::unix::OwnedWriteHalf>>,
    conn_state: &mut ConnState<'_>,
    prompt: String,
    tmux_pane: Option<String>,
    model: String,
    session_id: String,
) -> Result<ConnAction> {
    tracing::info!(model = %model, session_id = %session_id, prompt_len = prompt.len(), "received resume request");
    let _resume_session_guard = match claim_session(&state.active_sessions, &session_id) {
        Ok(g) => g,
        Err(()) => {
            let mut w = writer.lock().await;
            write_frame(
                &mut *w,
                &Response::Error {
                    message: format!(
                        "session {session_id} is already in use by another connection"
                    ),
                },
            )
            .await?;
            return Ok(ConnAction::Continue);
        }
    };
    if needs_copilot_auth(&model) {
        if let Err(e) = state.tokens.get(&state.http).await {
            let mut w = writer.lock().await;
            write_frame(
                &mut *w,
                &Response::Error {
                    message: format!("authentication error: {e:#}"),
                },
            )
            .await?;
            return Ok(ConnAction::Break);
        }
    }
    let (history, summaries, own_summary) = load_session_state(state, &session_id).await;
    let mut messages = build_messages(
        &prompt,
        tmux_pane.as_deref(),
        &history,
        &summaries,
        own_summary.as_deref(),
        &model,
    );
    inject_skill_files(&mut messages).await;
    let Some(result) =
        drive_agentic_loop(state, writer, conn_state, &session_id, messages, &model).await
    else {
        return Ok(ConnAction::Break);
    };
    flush_pending_unsolicited(writer, conn_state.pending_unsolicited).await;
    match result {
        Ok((response_text, _, _, _)) => {
            store_conversation(
                state,
                &session_id,
                &truncate_chars(&prompt, MAX_PROMPT_CHARS),
                &truncate_chars(&response_text, MAX_RESPONSE_CHARS),
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
    Ok(ConnAction::Continue)
}

/// Handle `Request::Chat`: manage multi-turn session context, run the agentic loop.
///
/// Updates `carried_messages`, `carried_session_id`, and `chat_session_guard` in-place
/// so the next `Chat` frame on the same connection inherits the in-memory context.
///
/// Returns `ConnAction::Break` when the client disconnects or auth fails.
#[allow(clippy::too_many_arguments)]
async fn handle_chat_request(
    state: &Arc<DaemonState>,
    writer: &Arc<tokio::sync::Mutex<tokio::net::unix::OwnedWriteHalf>>,
    conn_state: &mut ConnState<'_>,
    carried_messages: &mut Option<Vec<Message>>,
    carried_session_id: &mut Option<String>,
    carried_model: &mut Option<String>,
    chat_session_guard: &mut Option<ActiveSessionGuard>,
    prompt: String,
    tmux_pane: Option<String>,
    model: String,
    session_id: Option<String>,
) -> Result<ConnAction> {
    tracing::info!(pane = ?tmux_pane, model = %model, prompt_len = prompt.len(), "received chat request");
    let sid = session_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    // If the session_id changed, release any prior session claim and discard carried context.
    if carried_session_id.as_deref() != Some(&sid) {
        *chat_session_guard = None;
        *carried_messages = None;
    }

    // Claim the session on the first turn (or after a session change).
    if chat_session_guard.is_none() {
        match claim_session(&state.active_sessions, &sid) {
            Ok(g) => *chat_session_guard = Some(g),
            Err(()) => {
                let mut w = writer.lock().await;
                write_frame(
                    &mut *w,
                    &Response::Error {
                        message: format!("session {sid} is already in use by another connection"),
                    },
                )
                .await?;
                return Ok(ConnAction::Continue);
            }
        }
    }

    if needs_copilot_auth(&model) {
        if let Err(e) = state.tokens.get(&state.http).await {
            let mut w = writer.lock().await;
            write_frame(
                &mut *w,
                &Response::Error {
                    message: format!("authentication error: {e:#}"),
                },
            )
            .await?;
            return Ok(ConnAction::Break);
        }
    }

    // First turn: load history from DB.  Subsequent turns: extend carried messages.
    //
    // No pre-flight trim here: `run_agentic_loop` runs a synchronous
    // compaction check before every `invoke_model` call and archives exactly
    // the DB turns it summarises.  A pre-flight `hot_tail` trim would drop
    // older DB rows from memory without the summariser ever seeing them,
    // and the subsequent DB archive would then silently lose their content
    // on future resumes.  Feeding the full history through is also what
    // keeps the in-memory middle and the persisted archive aligned.
    let messages = if let Some(mut prev) = carried_messages.take() {
        // Update the model name in the system message so the LLM always knows
        // what model it's currently running as, even after a /model switch.
        if let Some(sys) = prev.first_mut() {
            if sys.role == "system" {
                if let Some(content) = sys.content.as_mut() {
                    update_embedded_model_name(content, &model);
                }
            }
        }
        prev.push(Message::user(prompt.clone()));
        // Do NOT re-inject skill files — they were injected on the first turn and are
        // already in `prev`.  Re-injecting every turn grows context unboundedly.
        prev
    } else {
        let (history, summaries, own_summary) = load_session_state(state, &sid).await;

        if history.is_empty() {
            let old = with_db(Arc::clone(&state.db), {
                let sid = sid.clone();
                move |conn| memory_db::get_sessions_without_summary(conn, &sid, MAX_SUMMARIES)
            })
            .await
            .unwrap_or_default();
            for old_sid in old {
                tokio::spawn(compact_session(
                    Arc::clone(state),
                    old_sid,
                    compact_model(&model),
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
            &model,
        );
        inject_skill_files(&mut msgs).await;
        msgs
    };

    // If the client explicitly changed the model (e.g. via /model), it wins
    // over carried_model from a previous switch_model call.  This ensures
    // suffixes like [1m] are not lost.
    let effective_model = match carried_model.take() {
        Some(carried) if carried == model => carried,
        Some(_carried) => {
            // Client sent a different model than what was carried — client wins.
            tracing::debug!(
                client_model = %model,
                "client model overrides carried_model"
            );
            model
        }
        None => model,
    };
    let Some(loop_result) =
        drive_agentic_loop(state, writer, conn_state, &sid, messages, &effective_model).await
    else {
        return Ok(ConnAction::Break);
    };
    flush_pending_unsolicited(writer, conn_state.pending_unsolicited).await;

    match loop_result {
        Ok((response_text, _prompt_tokens, final_messages, final_model)) => {
            store_conversation(
                state,
                &sid,
                &truncate_chars(&prompt, MAX_PROMPT_CHARS),
                &truncate_chars(&response_text, MAX_RESPONSE_CHARS),
            )
            .await;
            // Compaction is now driven synchronously from inside the agentic
            // loop (see `compact_in_loop`), so no post-loop spawn is needed:
            // by the time we reach this branch, `final_messages` is already
            // bounded by the compaction threshold.
            let mut w = writer.lock().await;
            write_frame(&mut *w, &Response::Done).await?;
            drop(w);
            *carried_messages = Some(final_messages);
            *carried_session_id = Some(sid.clone());
            // Persist the final model so the next turn starts with it.
            *carried_model = Some(final_model);
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
            *carried_messages = None;
            *carried_session_id = None;
            *carried_model = None;
        }
    }
    Ok(ConnAction::Continue)
}

/// Flush pending unsolicited-frame errors to the client.
///
/// Called after an agentic loop releases the writer lock so the deferred
/// errors can be sent without blocking steer routing.
async fn flush_pending_unsolicited(
    writer: &Arc<tokio::sync::Mutex<tokio::net::unix::OwnedWriteHalf>>,
    pending: &mut u32,
) {
    if *pending > 0 {
        let mut w = writer.lock().await;
        for _ in 0..*pending {
            let _ = write_frame(
                &mut *w,
                &Response::Error {
                    message: "busy: another request is already in progress on this connection"
                        .into(),
                },
            )
            .await;
        }
        *pending = 0;
    }
}

// ---------------------------------------------------------------------------
// Character-level truncation
// ---------------------------------------------------------------------------

/// Maximum chars stored for a user prompt in session history.
const MAX_PROMPT_CHARS: usize = 4_000;
/// Maximum chars stored for an assistant response in session history.
const MAX_RESPONSE_CHARS: usize = 8_000;
/// Maximum Unicode scalars kept in the `shell_command` preview returned by
/// [`summarise_tool_detail`] before appending `…`.  Kept at module scope so
/// regression tests can reference it without duplicating the `80` literal.
const SHELL_DETAIL_MAX_CHARS: usize = 80;

/// Truncate `s` to at most `max` Unicode scalar values (including the marker).
///
/// If truncation occurs, appends `"…[truncated]"` so the model knows the
/// content was cut.  The returned string always contains at most `max` chars.
/// Operates on char boundaries, never slicing multi-byte sequences.
///
/// Edge case: if `max` is smaller than or equal to the marker length, the
/// marker itself is truncated to `max` chars.
fn truncate_chars(s: &str, max: usize) -> String {
    // Fast path: if there is no (max+1)-th character the string is within limit.
    // char_indices().nth(max) is O(max), unlike chars().count() which is O(n).
    if s.char_indices().nth(max).is_none() {
        return s.to_owned();
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

/// Returns `true` if large outputs from `tool_name` should be persisted to /tmp
/// rather than injected inline into the model context.
///
/// `read_file` is always injected fully: it is the model's only path to access
/// complete file content.  Persisting it would restrict the model to a 2 KB
/// preview, severing its ability to read files larger than the preview window.
fn should_persist_tool_output(tool_name: &str) -> bool {
    tool_name != "read_file"
}

/// Build a short human-readable detail string for a tool call, used in
/// `Response::ToolUse` frames so the client can render something like
/// `shell_command: ls -la`.
///
/// Must operate on Unicode scalar boundaries: shell commands routinely contain
/// non-ASCII text (Chinese comments, em-dashes) and a raw byte slice like
/// `&s[..80]` panics when the truncation index falls inside a multi-byte UTF-8
/// sequence.  The `shell_command` branch truncates at 80 chars, appending `…`
/// when the original was longer; other tools return their primary argument
/// verbatim.
fn summarise_tool_detail(tool_name: &str, args: &serde_json::Value) -> String {
    match tool_name {
        "shell_command" => args
            .get("command")
            .and_then(|v| v.as_str())
            .map(|s| {
                // `char_indices().nth(N)` walks the string once and returns
                // the byte offset of the (N+1)-th Unicode scalar, which is by
                // construction a char boundary.  Slicing at that byte offset
                // is safe and avoids the double-iteration / allocation of the
                // prior `chars().nth` + `chars().take(N).collect()` approach.
                // `None` means the string has ≤ SHELL_DETAIL_MAX_CHARS chars
                // and fits verbatim.
                match s.char_indices().nth(SHELL_DETAIL_MAX_CHARS) {
                    Some((byte_idx, _)) => format!("{}…", &s[..byte_idx]),
                    None => s.to_string(),
                }
            })
            .unwrap_or_default(),
        "read_file" | "edit_file" => args
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
        "switch_model" => args
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    }
}

/// Validate and normalise the `model` argument from a `switch_model` tool call.
///
/// Returns `Ok(trimmed_model)` on success, or `Err(error_message)` if the
/// argument is missing or blank.  Trimming ensures leading/trailing whitespace
/// does not silently change provider routing behaviour.
/// In-place update of the model name embedded in a system-prompt string.
///
/// `build_messages` injects the model as `"...as model: [<name>]."` with the
/// unambiguous `"]."` suffix — `.` cannot appear inside `[1m]` or inside any
/// `provider/model` alias, so it reliably terminates the model field even for
/// names like `claude-sonnet-4.6[1m]` that contain a closing bracket.
///
/// Returns `true` if the replacement succeeded.  Control characters in `model`
/// are stripped to prevent prompt injection (mirrors `build_messages`).
fn update_embedded_model_name(content: &mut String, model: &str) -> bool {
    const PREFIX: &str = "You are currently running as model: [";
    const SUFFIX: &str = "].";

    let Some(start) = content.find(PREFIX) else {
        return false;
    };
    let model_start = start + PREFIX.len();
    let Some(close_offset) = content[model_start..].find(SUFFIX) else {
        return false;
    };
    let model_end = model_start + close_offset;

    let safe_model: String = model.chars().filter(|c| !c.is_control()).collect();
    content.replace_range(model_start..model_end, &safe_model);
    true
}

fn parse_switch_model_arg(raw: Option<&str>) -> Result<String, String> {
    match raw {
        None => Err("error: switch_model: missing 'model' argument".to_string()),
        Some(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return Err("error: switch_model: 'model' must not be empty".to_string());
            }
            // Reject control characters (including newlines) to prevent prompt injection
            // when the model name is embedded into the system message.
            if trimmed.chars().any(|c| c.is_control()) {
                return Err(
                    "error: switch_model: 'model' must not contain control characters".to_string(),
                );
            }
            Ok(trimmed.to_string())
        }
    }
}

/// Write `content` to `scratch_dir/{uuid}.txt` and return a stub message
/// containing a preview (up to 2 000 Unicode scalar values) and the file path.
///
/// **Unix only**: the file is created with mode 0600 so that tool outputs,
/// which may include secrets, are not world-readable.  On non-unix platforms
/// this function always returns an inline fallback stub (no disk write) because
/// equivalent permission hardening is not available.
///
/// The filename is a fresh UUID v4, guaranteeing uniqueness across concurrent
/// tool calls.  The blocking open+write runs in `spawn_blocking` to avoid
/// stalling the Tokio runtime.
///
/// If the write fails the stub is returned inline with an explicit note so the
/// model knows the full output is unavailable.
///
/// Note: persistence is triggered when an output *exceeds* `LARGE_TOOL_RESULT_CHARS`;
/// files contain the full output and may therefore be larger than that threshold.
/// Files are intentionally **not** deleted when the loop exits: stubs in
/// `carried_messages` reference these paths across Chat turns, and deleting them
/// would cause `read_file` calls on those paths to fail with NotFound.  /tmp is
/// cleaned by the OS on reboot.
async fn persist_large_result(content: String, scratch_dir: &std::path::Path) -> String {
    let total = content.len(); // byte count — O(1), good enough for display
    let preview: String = content.chars().take(2_000).collect();

    // Unix: write with mode 0600 and return a path stub.
    #[cfg(unix)]
    let result = {
        let path = scratch_dir.join(format!("{}.txt", uuid::Uuid::new_v4()));
        match write_tool_output_file(&path, content).await {
            Ok(()) => format!(
                "[output truncated: {total} bytes → {path}\n\nPreview:\n{preview}\n...]",
                path = path.display()
            ),
            Err(e) => format!(
                "[output truncated: {total} bytes — could not persist to /tmp ({e}). \
                 Preview:\n{preview}\n...]"
            ),
        }
    };

    // Non-unix: no 0600-equivalent permission guarantee, return inline.
    #[cfg(not(unix))]
    let result = {
        let _ = scratch_dir;
        format!(
            "[output truncated: {total} bytes — persistence unavailable on this platform. \
             Preview:\n{preview}\n...]"
        )
    };

    result
}

/// Write `content` to `path` with mode 0600 using `spawn_blocking`.
/// Only compiled on unix; callers on other platforms skip the write entirely.
#[cfg(unix)]
async fn write_tool_output_file(path: &std::path::Path, content: String) -> std::io::Result<()> {
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || {
        use std::io::Write as _;
        use std::os::unix::fs::OpenOptionsExt as _;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)?;
        file.write_all(content.as_bytes())
    })
    .await
    .map_err(std::io::Error::other)?
}

/// Format the "file unchanged" stub returned by the read_file dedup check.
///
/// Centralised so that tests can assert on the same format without duplicating
/// the string literal — changes to the wording only need to be made once.
fn file_unchanged_stub(cached_tool_use_id: &str) -> String {
    format!(
        "[File metadata (mtime, size) unchanged since last read \
         (see tool_result for call {cached_tool_use_id}) \
         — re-reading is skipped]"
    )
}

/// Return `(mtime_nanos, size_bytes)` for `path`, or `None` if metadata is
/// unavailable.  Using both fields reduces false cache-hits on filesystems with
/// coarse mtime granularity (e.g. FAT32 at 2-second resolution) where content
/// can change without the mtime advancing.
async fn file_cache_key(path: &std::path::Path) -> Option<(u128, u64)> {
    let meta = tokio::fs::metadata(path).await.ok()?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())?;
    Some((mtime, meta.len()))
}

// Note: there is intentionally no ScratchDirGuard / cleanup for the per-run
// scratch directory.  Stubs written into tool_result messages reference paths
// under this directory, and those messages are carried across Chat turns in
// carried_messages.  Deleting the directory when run_agentic_loop returns would
// cause read_file calls on those paths to fail with NotFound on the next turn.
// The directory is left in /tmp for the OS to clean up on reboot.

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

/// RAII guard that releases a session from `active_sessions` when dropped.
///
/// Held for the lifetime of a Chat connection or an Ask/Resume agentic loop.
/// Ensures concurrent connections to the same session are rejected.
struct ActiveSessionGuard {
    active_sessions: Arc<Mutex<HashSet<String>>>,
    session_id: String,
}

impl Drop for ActiveSessionGuard {
    fn drop(&mut self) {
        self.active_sessions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&self.session_id);
    }
}

/// Try to claim `session_id` in `active_sessions`.
///
/// Returns `Ok(guard)` on success.  Returns `Err` when the session is already
/// held by another connection — the caller should send `Response::Error` and
/// skip processing.
fn claim_session(
    active_sessions: &Arc<Mutex<HashSet<String>>>,
    session_id: &str,
) -> Result<ActiveSessionGuard, ()> {
    let mut guard = active_sessions.lock().unwrap_or_else(|p| p.into_inner());
    if guard.contains(session_id) {
        return Err(());
    }
    guard.insert(session_id.to_owned());
    Ok(ActiveSessionGuard {
        active_sessions: Arc::clone(active_sessions),
        session_id: session_id.to_owned(),
    })
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
        let content = truncate_chars(&entry.content, 1_500);
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

/// Find the index at which a safe "hot tail" begins in a message list.
///
/// The hot tail starts at a `user`-role message so tool_call/tool_result
/// pairings are never split (an orphan `tool` result with no preceding
/// `assistant` tool_call would be rejected by every provider).
///
/// `desired_pairs` is the number of **user/assistant pairs** (i.e. user
/// turns) to keep in the tail.
///
/// Always leaves at least one user turn in the middle so the compactor has
/// something to summarise.  When there is only one user message in the
/// list, returns `messages.len()` — the caller must treat that as
/// "nothing to summarise" and bail out.
fn find_hot_tail_start(messages: &[Message], desired_pairs: usize) -> usize {
    if messages.is_empty() {
        return 0;
    }
    let user_indexes: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter_map(|(i, m)| (m.role == "user").then_some(i))
        .collect();
    let n_users = user_indexes.len();
    if n_users <= 1 {
        // Zero or one user turn: nothing historical to split off.
        return messages.len();
    }
    // Leave at least one user turn in the middle for the summariser, so cap
    // the tail at (n_users - 1) user turns.
    let tail_user_count = desired_pairs.min(n_users - 1);
    user_indexes[n_users - tail_user_count]
}

/// Partition `messages` into `[head | middle | tail]` for compaction.
///
/// - `head` (0..head_end) is the leading `system` block plus any injected
///   own_summary user/assistant pair — always preserved.
/// - `middle` (head_end..tail_start) is the only region the summariser
///   touches.  Empty when there is no compactable history.
/// - `tail` (tail_start..) is the hot tail up through the active user
///   prompt — always preserved.
///
/// Returns `(head_end, tail_start)`.  Callers should treat
/// `tail_start <= head_end` as "nothing to compact".
fn compaction_boundaries(messages: &[Message]) -> (usize, usize) {
    let mut head_end = messages
        .iter()
        .position(|m| m.role != "system")
        .unwrap_or(messages.len());
    if head_end + 1 < messages.len()
        && messages[head_end].role == "user"
        && messages[head_end]
            .content
            .as_deref()
            .is_some_and(|s| s.starts_with("[Summary of earlier in this session]"))
        && messages[head_end + 1].role == "assistant"
    {
        head_end += 2;
    }
    // `find_hot_tail_start` returns `messages.len()` as a sentinel meaning
    // "≤1 user turn, nothing historical to split off".  If we let that
    // through, `tail_start` would be `messages.len()` and the caller's
    // middle slice would cover everything from `head_end` to the end —
    // including the active user prompt and any trailing assistant/tool
    // messages.  Splicing that away would either overwrite the current
    // prompt or leave messages ending on an assistant turn (which the
    // Bedrock converse endpoint rejects with 400 "must end with a user
    // message").  Collapse to `head_end` so the caller sees "nothing to
    // compact" and bails cleanly.
    let raw_tail_start = find_hot_tail_start(messages, HOT_TAIL_PAIRS);
    let tail_start = if raw_tail_start >= messages.len() {
        head_end
    } else {
        raw_tail_start.max(head_end)
    };
    (head_end, tail_start)
}

/// Token cost of the compactable middle section only.
///
/// Used by the pre-send threshold check in `run_agentic_loop` so that
/// un-compactable overhead (system prompt, skill files, the preserved
/// own_summary prefix, the hot tail, and the active user prompt) cannot
/// single-handedly trigger a compaction attempt that would immediately
/// bail with "no middle section to summarise".  Only tokens that
/// `compact_in_loop` can actually remove count toward the threshold.
fn compactable_middle_tokens(messages: &[Message]) -> usize {
    let (head_end, tail_start) = compaction_boundaries(messages);
    if tail_start <= head_end {
        return 0;
    }
    count_message_tokens(&messages[head_end..tail_start])
}

/// Synchronously compact `messages` in-place when it exceeds the token threshold.
///
/// Called from [`run_agentic_loop`] before each `invoke_model` call.  On success:
/// 1. Generates a summary via a dedicated LLM call (non-streaming, no tools,
///    cheap default model preserving provider prefix).
/// 2. Replaces the middle of `messages` with a
///    `[user "[Compacted summary]"] [assistant <summary>]` pair, keeping the
///    leading system/skill block and a trailing hot tail.
/// 3. If `session_id` is `Some`, archives the corresponding DB turns and
///    upserts the session summary so future resumes start from the compacted
///    state rather than the raw history.
///
/// Returns `Ok(())` on success, `Err(_)` on failure — the caller increments a
/// consecutive-failure counter and trips a circuit breaker after repeated
/// failures so an irrecoverably-oversized context cannot loop forever.
async fn compact_in_loop(
    state: &DaemonState,
    messages: &mut Vec<Message>,
    main_model: &str,
    session_id: Option<&str>,
) -> Result<()> {
    // Partition: [system/skill head] + [preserved own_summary prefix, if any]
    // + [middle to summarise] + [hot tail].  See `compaction_boundaries`.
    let (head_end, tail_start) = compaction_boundaries(messages);
    if tail_start <= head_end {
        anyhow::bail!("compact_in_loop: no middle section to summarise");
    }

    // Build a self-contained summariser prompt that does not inherit the
    // agent's tool schemas — the summariser must never execute tools.
    // Only user text and assistant text turns are kept: tool_call-only
    // assistant turns (content=None, tool_calls!=[]) and `tool` results
    // carry no durable semantics once the surrounding assistant text has
    // described the outcome, and sending empty assistant messages would
    // add noise to the summariser input.
    let middle = &messages[head_end..tail_start];
    let mut summary_msgs = vec![Message::system(
        "You are a memory compactor. Output 3-5 bullet points capturing the key outcomes, \
         decisions, and facts. Be concise and factual. Output only the bullet points, no preamble.",
    )];
    for m in middle {
        match m.role.as_str() {
            "user" => {
                if let Some(text) = m.content.as_deref() {
                    if !text.is_empty() {
                        summary_msgs.push(Message::user(text.to_owned()));
                    }
                }
            }
            "assistant" => {
                // Skip tool-call-only turns (content=None or empty) — they
                // add no information once the following assistant text has
                // reported the outcome.
                if let Some(text) = m.content.as_deref() {
                    if !text.is_empty() {
                        summary_msgs.push(Message::assistant(Some(text.to_owned()), vec![]));
                    }
                }
            }
            // Skip `tool` results and any other role — no durable semantics.
            _ => {}
        }
    }
    if summary_msgs
        .last()
        .is_some_and(|m| m.role == "assistant" || m.role == "system")
    {
        summary_msgs.push(Message::user(
            "Summarise the conversation above into 3-5 bullet points.".to_owned(),
        ));
    }

    // Use the cheap default model (sonnet) for the summary; preserve the
    // provider prefix so e.g. copilot sessions stay on copilot.
    let summary_model = compact_model(main_model);

    // Cap the summariser's output so a verbose reply cannot defeat the whole
    // point of compaction.  A bullet-point summary fits comfortably in 2k
    // tokens; letting the model run up to `response_max_tokens` (half the
    // context window) could return tens of thousands of tokens and keep the
    // prompt over threshold, re-triggering compaction next iteration.
    let summary_budget = COMPACT_SUMMARY_MAX_TOKENS.min(response_max_tokens(&summary_model));

    let mut sink = tokio::io::sink();
    let resp = invoke_model(
        state,
        &summary_model,
        &summary_msgs,
        &[],
        summary_budget,
        &mut sink,
    )
    .await
    .context("compact_in_loop: summary model call failed")?;
    let summary_text = resp.text.trim().to_owned();
    if summary_text.is_empty() {
        anyhow::bail!("compact_in_loop: empty summary");
    }
    // Truncate to the same cap `build_messages()` applies to `own_summary`
    // on resume-injection (`MAX_SUMMARY_CHARS * 2`).  Storing a longer
    // string is pointless — it would be silently re-truncated the next
    // time the session resumes — and aligning the caps keeps in-memory
    // and persisted forms identical.
    let summary_text = truncate_chars(&summary_text, MAX_SUMMARY_CHARS * 2);

    // Count the user turns in the summarised middle slice and derive the
    // DB-row archive count from that.  `store_conversation` writes exactly
    // two `memories` rows per turn (one `user`, one `assistant` text), so
    // `user_turns * 2` is the authoritative DB row count for those turns.
    //
    // Counting *all* in-memory `user`/`assistant` messages here would be
    // wrong: a single agentic turn can produce many in-memory assistant
    // messages (tool-call-only turns, intermediate text) that are never
    // persisted.  Including them would inflate `summarised_row_count`
    // and cause `archive_session_turns` to overshoot into real DB rows
    // the summariser never saw — silently losing their content on resume.
    let middle_user_count = messages[head_end..tail_start]
        .iter()
        .filter(|m| m.role == "user")
        .count();
    let summarised_row_count = middle_user_count * 2;

    let pre_tokens = count_message_tokens(messages);

    // Replace [head_end..tail_start] with a user/assistant summary pair.
    // Use the SAME marker string that `build_messages()` uses for
    // own_summary injection so that on a subsequent in-loop compaction
    // within the same connection the `head_end` guard (which looks for
    // this exact marker) preserves this pair rather than feeding it
    // back to the summariser — and so `summarised_row_count` cannot
    // include synthetic in-memory rows that have no DB-row counterpart
    // (which would make the archive step overshoot).
    let replacement = vec![
        Message::user("[Summary of earlier in this session]".to_owned()),
        Message::assistant(Some(summary_text.clone()), vec![]),
    ];
    messages.splice(head_end..tail_start, replacement);
    let post_tokens = count_message_tokens(messages);
    tracing::info!(
        session = ?session_id,
        pre_tokens,
        post_tokens,
        "compact_in_loop: summary applied"
    );

    // If the summary did not reduce the total token count at all, treat it
    // as a compaction failure so the caller's circuit breaker can engage
    // rather than hammering the API in a retry storm.  This catches the
    // pathological case where a verbose summary is larger than the turns
    // it replaced — which would otherwise re-trigger compaction on the
    // very next iteration and loop forever.  We do NOT gate on the
    // threshold itself: even a small reduction is progress, and a system
    // prompt that alone exceeds the threshold is a configuration bug
    // rather than something compaction can fix.
    if post_tokens >= pre_tokens {
        anyhow::bail!(
            "compact_in_loop: summary did not reduce token count \
             (pre={pre_tokens}, post={post_tokens})"
        );
    }

    // Best-effort DB archive: persist the summary and mark exactly the
    // turns that the summariser saw as archived so resuming this session
    // later starts from the summary rather than replaying the raw history.
    // Crucially we archive `summarised_row_count` (what was actually fed
    // into the summariser) rather than `total - keep_recent`: pre-flight
    // trimming in `handle_chat_request` can drop older DB rows from
    // memory before `run_agentic_loop` starts, and archiving rows the
    // summariser never saw would silently lose their content on resume
    // (the summary would not cover them either).  Log and continue on
    // failure.
    if let Some(sid) = session_id {
        let db = Arc::clone(&state.db);
        let sid_owned = sid.to_owned();
        let ts = chrono::Utc::now().to_rfc3339();
        let archive_result = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = db.lock().unwrap_or_else(|p| p.into_inner());
            // Always persist the summary (even when nothing is archived) so a
            // later resume rebuilds from the compacted state rather than the
            // raw history — the in-memory splice has already happened, so
            // skipping the DB write here would desync memory vs. persistence.
            let tx = conn
                .unchecked_transaction()
                .context("compact_in_loop: begin transaction")?;
            memory_db::store_session_summary(&conn, &sid_owned, &summary_text, &ts)?;
            if summarised_row_count > 0 {
                let rows = memory_db::get_session_oldest(&conn, &sid_owned, summarised_row_count)?;
                let ids: Vec<i64> = rows.iter().map(|r| r.id).collect();
                memory_db::archive_session_turns(&conn, &ids)?;
            }
            tx.commit().context("compact_in_loop: commit transaction")?;
            Ok(())
        })
        .await
        .unwrap_or_else(|e| Err(anyhow::anyhow!("compact_in_loop archive panicked: {e}")));
        if let Err(e) = archive_result {
            tracing::warn!(error = %e, session = %sid, "compact_in_loop: DB archive failed");
        }
    }

    Ok(())
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

/// Returns `true` for models that must be routed directly to the Responses API
/// and do not support Chat Completions at all (e.g. the gpt-5 family).
fn requires_responses_api(model: &str) -> bool {
    model == "gpt-5" || model.starts_with("gpt-5.") || model.starts_with("gpt-5-")
}

/// Send one model turn, dispatching to the correct provider backend.
///
/// The raw `model` string is resolved via [`crate::provider::resolve`] to
/// determine the provider (Copilot or Bedrock) and the actual model ID.
///
/// **Copilot** path:
/// 1. The base URL is derived from the `proxy-ep` field in the Copilot JWT.
/// 2. gpt-5.x models go directly to `/v1/responses`.
/// 3. All other models try `/v1/chat/completions` first, falling back to
///    `/v1/responses` on 400 `unsupported_api_for_model`.
/// 4. Auth errors (401/403) evict the token cache and retry once.
///
/// **Bedrock** path:
/// 1. Reads `AWS_BEARER_TOKEN_BEDROCK` and `AWS_REGION` from the environment.
/// 2. Calls the ConverseStream API directly.
async fn invoke_model<W>(
    state: &DaemonState,
    model: &str,
    messages: &[Message],
    tools: &[serde_json::Value],
    max_completion_tokens: usize,
    writer: &mut W,
) -> Result<copilot::CopilotResponse>
where
    W: AsyncWriteExt + Unpin,
{
    let spec = crate::provider::resolve(model);
    tracing::debug!(
        provider = %spec.provider,
        model_id = %spec.model_id,
        display = %spec.display_name,
        use_1m = spec.use_1m,
        "invoke_model: resolved provider"
    );

    match spec.provider {
        crate::provider::ProviderKind::Bedrock => {
            crate::bedrock::stream_chat(
                &state.http,
                &spec,
                messages,
                tools,
                max_completion_tokens,
                writer,
            )
            .await
        }

        crate::provider::ProviderKind::Copilot => {
            if spec.use_1m {
                tracing::warn!(
                    display = %spec.display_name,
                    "1M context is not supported via Copilot; proceeding with standard context window"
                );
                write_frame(
                    writer,
                    &crate::ipc::Response::Text {
                        chunk: "[warning] 1M context ([1m]) is not supported via Copilot; \
                                proceeding with standard context window.\n"
                            .into(),
                    },
                )
                .await?;
            }
            invoke_copilot(
                state,
                &spec.model_id,
                messages,
                tools,
                max_completion_tokens,
                writer,
            )
            .await
        }
    }
}

/// Call the Responses API with `tok`, evicting the token cache and retrying
/// once on a 401/403 auth error.  Centralises retry/telemetry logic so both
/// the gpt-5.x direct path and the chat-completions fallback path stay in sync.
async fn stream_via_responses_with_auth_retry<W>(
    state: &DaemonState,
    tok: &crate::auth::CopilotToken,
    model: &str,
    messages: &[Message],
    tools: &[serde_json::Value],
    max_completion_tokens: usize,
    writer: &mut W,
) -> Result<copilot::CopilotResponse>
where
    W: AsyncWriteExt + Unpin,
{
    let r = crate::responses::stream_chat(
        &state.http,
        &tok.value,
        &tok.base_url,
        model,
        messages,
        tools,
        max_completion_tokens,
        writer,
    )
    .await;
    match r {
        Ok(r) => Ok(r),
        Err(e) => {
            let is_auth = e
                .downcast_ref::<copilot::CopilotHttpError>()
                .is_some_and(|he| matches!(he.status.as_u16(), 401 | 403));
            if is_auth {
                tracing::warn!(
                    error = %e,
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
                    max_completion_tokens,
                    writer,
                )
                .await
            } else {
                Err(e)
            }
        }
    }
}

/// Copilot-specific model invocation.  gpt-5.x models are routed directly to
/// the Responses API; all other models try Chat Completions first, falling
/// back to Responses API on `unsupported_api_for_model`.  Auth errors evict
/// the token cache and retry once.
async fn invoke_copilot<W>(
    state: &DaemonState,
    model: &str,
    messages: &[Message],
    tools: &[serde_json::Value],
    max_completion_tokens: usize,
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

    // Models in the gpt-5 family do not support Chat Completions at all.
    // Skip the first-hop attempt and go directly to the Responses API.
    if requires_responses_api(model) {
        tracing::debug!(model, "routing directly to Responses API (gpt-5 family)");
        return stream_via_responses_with_auth_retry(
            state,
            &tok,
            model,
            messages,
            tools,
            max_completion_tokens,
            writer,
        )
        .await;
    }

    let result = copilot::stream_chat(
        &state.http,
        &tok.value,
        &tok.base_url,
        model,
        messages,
        tools,
        max_completion_tokens,
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
            stream_via_responses_with_auth_retry(
                state,
                &tok,
                model,
                messages,
                tools,
                max_completion_tokens,
                writer,
            )
            .await
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
                    max_completion_tokens,
                    writer,
                )
                .await;
                match r2 {
                    Ok(r) => Ok(r),
                    Err(ref e2) if is_unsupported_via_chat_completions(e2) => {
                        stream_via_responses_with_auth_retry(
                            state,
                            &fresh,
                            model,
                            messages,
                            tools,
                            max_completion_tokens,
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

/// Tool outputs larger than this (chars) are written to /tmp and replaced with a stub.
/// Matches Claude Code's DEFAULT_MAX_RESULT_SIZE_CHARS. read_file is exempt (always full).
const LARGE_TOOL_RESULT_CHARS: usize = 50_000;

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
    session_id: Option<&str>,
) -> Result<(String, usize, Vec<Message>, String)>
where
    W: AsyncWriteExt + Unpin,
{
    let schemas = tools::tool_schemas(include_spawn_agent);
    let final_text;
    let mut tools_were_used = false;
    let mut conclusion_nudge_sent = false;
    let mut last_prompt_tokens: usize;
    // Mutable so switch_model tool calls can change the model mid-session.
    let mut current_model = model.to_string();
    // Circuit breaker: once we exhaust MAX_CONSECUTIVE_COMPACT_FAILURES in a row
    // within this loop invocation, stop attempting in-loop compaction so the
    // loop cannot hammer the API in a retry storm when the context is
    // irrecoverably over the limit.  Scope is intentionally per-invocation —
    // each new user message starts a fresh `run_agentic_loop` with the
    // counter reset to 0, so a transient failure (rate limit, flake) on one
    // turn does not permanently disable compaction for the rest of the
    // session.
    let mut consecutive_compact_failures: u32 = 0;
    // AMAEBI_TOOL_MODEL: if set, use this cheaper model on turns where the
    // previous turn finished with tool calls (FinishReason::ToolCalls),
    // unless the LLM explicitly switched the model via switch_model.
    let tool_model: Option<String> = std::env::var("AMAEBI_TOOL_MODEL").ok();
    let mut model_explicitly_switched = false;
    let mut last_turn_used_tools = false;

    // Per-run scratch directory for large tool outputs (unix only).
    // Intentionally not cleaned up on exit — see comment near ScratchDirGuard
    // for rationale.  /tmp is cleaned by the OS on reboot.
    let scratch_dir = {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let run_id = uuid::Uuid::new_v4().to_string();
            let dir = std::path::PathBuf::from(format!("/tmp/amaebi-tool-results/{run_id}"));
            if let Err(e) = tokio::fs::create_dir_all(&dir).await {
                tracing::warn!(
                    path = %dir.display(),
                    error = %e,
                    "failed to create tool-output scratch dir; large outputs will be truncated inline"
                );
            } else if let Err(e) =
                tokio::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).await
            {
                tracing::warn!(
                    path = %dir.display(),
                    error = %e,
                    "failed to set scratch dir permissions to 0700"
                );
            }
            dir
        }
        #[cfg(not(unix))]
        std::path::PathBuf::new() // unused placeholder; persist_large_result is no-op on non-unix
    };
    // read_file dedup cache: path → ((mtime_nanos, size_bytes), tool_use_id of last read).
    // Both mtime and size are stored to reduce false hits on filesystems with
    // coarse mtime granularity where content can change without mtime advancing.
    let mut read_cache: std::collections::HashMap<std::path::PathBuf, ((u128, u64), String)> =
        Default::default();

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

        // Pre-send compaction: if the total token count exceeds the
        // model's compaction threshold AND there is actually compactable
        // content in the middle slice, synchronously summarise it before
        // dispatching.  Claude Code does the same check at the start of
        // every query iteration (query.ts:453).  The summariser uses
        // the cheap default model (sonnet), not `current_model`.
        //
        // The `middle_tokens > 0` guard matters: on the very first turn
        // of a session, system + skills + current prompt can already
        // exceed the threshold (especially with very low
        // AMAEBI_COMPACTION_THRESHOLD values), but there is nothing
        // compactable — no history between the preserved head and the
        // hot tail.  Triggering compaction here would immediately bail
        // with "no middle section to summarise", tripping the circuit
        // breaker on every iteration for no benefit.
        //
        // A circuit breaker caps consecutive failures so an irrecoverably
        // oversized context does not trigger an infinite retry storm.
        if consecutive_compact_failures < MAX_CONSECUTIVE_COMPACT_FAILURES {
            let threshold = compaction_threshold_tokens(&current_model);
            let current_tokens = count_message_tokens(&messages);
            let middle_tokens = compactable_middle_tokens(&messages);
            if current_tokens > threshold && middle_tokens > 0 {
                let _ = write_frame(writer, &Response::Compacting).await;
                match compact_in_loop(state, &mut messages, &current_model, session_id).await {
                    Ok(()) => {
                        consecutive_compact_failures = 0;
                    }
                    Err(e) => {
                        consecutive_compact_failures += 1;
                        tracing::warn!(
                            error = %e,
                            attempt = consecutive_compact_failures,
                            "compact_in_loop failed; continuing without compaction"
                        );
                    }
                }
            }
        }

        // All models route through the Copilot JWT endpoint; invoke_model
        // falls back to the Responses API automatically when needed.
        // Token management and auth-error retry are handled inside invoke_model.
        //
        // If AMAEBI_TOOL_MODEL is configured and the LLM has not explicitly
        // switched the model via switch_model, use the tool model for every
        // turn after the first to reduce cost on mechanical tool-execution turns.
        let invoke_with: &str =
            if last_turn_used_tools && !model_explicitly_switched && tool_model.is_some() {
                tool_model.as_deref().unwrap()
            } else {
                &current_model
            };
        if invoke_with != current_model {
            tracing::debug!(
                tool_model = invoke_with,
                main_model = %current_model,
                "auto-switching to AMAEBI_TOOL_MODEL for tool-execution turn"
            );
        }
        let resp = invoke_model(
            state,
            invoke_with,
            &messages,
            &schemas,
            response_max_tokens(invoke_with),
            writer,
        )
        .await?;
        last_turn_used_tools = matches!(resp.finish_reason, FinishReason::ToolCalls);

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
                    enum WaitInputOutcome {
                        Reply(String),
                        UserCancelled,
                        Disconnected,
                    }
                    let outcome = match tokio::time::timeout(
                        std::time::Duration::from_secs(300),
                        steer_rx.recv(),
                    )
                    .await
                    {
                        Ok(Some(Some(reply))) => Ok(WaitInputOutcome::Reply(reply)),
                        Ok(Some(None)) => {
                            // Interrupt: user cancelled the reply prompt.
                            tracing::debug!(
                                context = "waiting_for_input_reply",
                                "interrupt received; aborting waiting-for-input"
                            );
                            Ok(WaitInputOutcome::UserCancelled)
                        }
                        Ok(None) => Ok(WaitInputOutcome::Disconnected),
                        Err(e) => Err(e),
                    };
                    match outcome {
                        Ok(WaitInputOutcome::Reply(user_reply)) => {
                            messages.push(Message::user(user_reply));
                            write_frame(writer, &Response::SteerAck).await?;
                            continue;
                        }
                        Ok(WaitInputOutcome::UserCancelled) => {
                            // User cancelled — resume the loop; session stays open.
                            tracing::debug!(
                                context = "waiting_for_input_reply",
                                "user cancelled waiting-for-input; resuming session"
                            );
                            continue;
                        }
                        Ok(WaitInputOutcome::Disconnected) => {
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
                    let results =
                        futures_util::future::join_all(tool_calls_snapshot.iter().map(|tc| {
                            let scratch_dir = scratch_dir.clone();
                            let current_model = current_model.clone();
                            async move {
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
                                tracing::debug!(
                                    tool = %tc.name,
                                    model = %current_model,
                                    "executing tool"
                                );
                                match state.executor.execute(&tc.name, args).await {
                                    Ok(output) => {
                                        tracing::debug!(
                                            tool = %tc.name,
                                            model = %current_model,
                                            output_len = output.len(),
                                            "tool succeeded"
                                        );
                                        if !should_persist_tool_output(&tc.name) {
                                            output
                                        } else if output
                                            .char_indices()
                                            .nth(LARGE_TOOL_RESULT_CHARS)
                                            .is_some()
                                        {
                                            persist_large_result(output, &scratch_dir).await
                                        } else {
                                            output
                                        }
                                    }
                                    Err(e) => {
                                        tracing::warn!(tool = %tc.name, error = %e, "tool failed");
                                        format!("error: {e:#}")
                                    }
                                }
                            }
                        }))
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

                        tracing::debug!(tool = %tc.name, model = %current_model, "executing tool");

                        // Parse args once; reused for dedup, tool_detail, and execution.
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

                        // read_file dedup: skip re-read if file is unchanged.
                        // Both mtime and size are compared to reduce false hits on
                        // filesystems with coarse mtime resolution (e.g. FAT32).
                        if tc.name == "read_file" {
                            if let Some(path_str) = args["path"].as_str() {
                                let path = std::path::PathBuf::from(path_str);
                                if let Some(key) = file_cache_key(&path).await {
                                    if let Some((cached_key, cached_id)) = read_cache.get(&path) {
                                        if *cached_key == key {
                                            write_frame(
                                                writer,
                                                &Response::ToolUse {
                                                    name: tc.name.clone(),
                                                    detail: format!("{path_str} (unchanged)"),
                                                },
                                            )
                                            .await?;
                                            messages.push(Message::tool_result(
                                                &tc.id,
                                                file_unchanged_stub(cached_id),
                                            ));
                                            continue;
                                        }
                                    }
                                }
                            }
                        }

                        let tool_detail = summarise_tool_detail(&tc.name, &args);
                        write_frame(
                            writer,
                            &Response::ToolUse {
                                name: tc.name.clone(),
                                detail: tool_detail,
                            },
                        )
                        .await?;

                        // switch_model is handled here, not by the executor.
                        if tc.name == "switch_model" {
                            let result = match parse_switch_model_arg(args["model"].as_str()) {
                                Err(e) => e,
                                Ok(new_model) => {
                                    let old =
                                        std::mem::replace(&mut current_model, new_model.clone());
                                    // LLM made an explicit choice — stop auto-switching
                                    // to AMAEBI_TOOL_MODEL so we respect the LLM's intent.
                                    model_explicitly_switched = true;
                                    tracing::info!(
                                        old = %old,
                                        new = %current_model,
                                        "model switched by LLM"
                                    );
                                    write_frame(
                                        writer,
                                        &Response::Text {
                                            chunk: format!(
                                                "[model switched: {old} → {current_model}]\n"
                                            ),
                                        },
                                    )
                                    .await?;
                                    // Notify the client so it can update its local model
                                    // variable, keeping carried_model in sync on the next turn.
                                    write_frame(
                                        writer,
                                        &Response::ModelSwitched {
                                            model: current_model.clone(),
                                        },
                                    )
                                    .await?;
                                    format!("Model switched to {current_model}")
                                }
                            };
                            messages.push(Message::tool_result(&tc.id, result));
                            continue;
                        }

                        // Extract the read_file path before args is consumed by execute.
                        let read_path: Option<std::path::PathBuf> = if tc.name == "read_file" {
                            args["path"].as_str().map(std::path::PathBuf::from)
                        } else {
                            None
                        };

                        let result = match state.executor.execute(&tc.name, args).await {
                            Ok(output) => {
                                tracing::debug!(
                                    tool = %tc.name,
                                    model = %current_model,
                                    output_len = output.len(),
                                    "tool succeeded"
                                );
                                if !should_persist_tool_output(&tc.name) {
                                    // Update dedup cache after a fresh read_file.
                                    if let Some(path) = read_path {
                                        if let Some(key) = file_cache_key(&path).await {
                                            read_cache.insert(path, (key, tc.id.clone()));
                                        }
                                    }
                                    // read_file is injected in full: it is the model's
                                    // only path to complete file content.  Capping it
                                    // would leave the model with a truncated view and
                                    // no way to retrieve the rest.  The dedup cache
                                    // above avoids re-injecting unchanged files.
                                    output
                                } else if output
                                    .char_indices()
                                    .nth(LARGE_TOOL_RESULT_CHARS)
                                    .is_some()
                                {
                                    persist_large_result(output, &scratch_dir).await
                                } else {
                                    output
                                }
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

    Ok((final_text, last_prompt_tokens, messages, current_model))
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
    let skill_msgs = load_skill_messages().await;
    splice_skill_messages(messages, skill_msgs);
}

/// Load skill messages from `~/.amaebi/` without modifying any message list.
///
/// Returns the messages that would be injected by [`inject_skill_files`].
/// Callers that need to inject into multiple lists (e.g. after a rebuild) can
/// call this once and reuse the result with [`splice_skill_messages`].
async fn load_skill_messages() -> Vec<Message> {
    let home = match amaebi_home() {
        Ok(p) => p,
        Err(e) => {
            tracing::debug!(error = %e, "could not resolve amaebi home for skill injection");
            return vec![];
        }
    };
    load_skill_messages_from(&home).await
}

/// Splice a pre-loaded set of skill messages into `messages` at the correct position.
fn splice_skill_messages(messages: &mut Vec<Message>, skill_msgs: Vec<Message>) {
    if skill_msgs.is_empty() {
        return;
    }
    let insert_at = messages
        .iter()
        .position(|m| m.role == "system")
        .map(|i| i + 1)
        .unwrap_or(0);
    messages.splice(insert_at..insert_at, skill_msgs);
}

/// Load skill messages from `amaebi_home` without modifying any message list.
/// Used by [`load_skill_messages`] and tests.
async fn load_skill_messages_from(amaebi_home: &std::path::Path) -> Vec<Message> {
    // Final prompt order once spliced in:
    //   [system] / [SOUL.md] / [AGENTS.md] / [on-demand docs] /
    //   [own_summary (user+assistant, if any)] / [history...] / [current prompt]
    // Skills take highest priority and must never be displaced by history trimming.
    const FIXED_FILES: &[(&str, &str)] =
        &[("SOUL.md", "## Soul"), ("AGENTS.md", "## Agent Guidelines")];

    let mut msgs: Vec<Message> = Vec::new();
    for (filename, header) in FIXED_FILES {
        let path = amaebi_home.join(filename);
        match tokio::fs::read_to_string(&path).await {
            Ok(content) => {
                let trimmed = content.trim();
                if !trimmed.is_empty() {
                    msgs.push(Message::system(format!("{header}\n\n{trimmed}")));
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
        msgs.push(Message::system(format!(
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

    msgs
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
    model: &str,
) -> Vec<Message> {
    let mut system = "You are a helpful, concise AI assistant embedded in a tmux terminal. \
                      Answer in plain text; avoid markdown unless the user asks for it. \
                      You have tools available to inspect the terminal, run commands, \
                      and read or edit files — use them when they help you answer accurately. \
                      After using any tool, you MUST always follow up with a text response \
                      summarising what you did and the outcome — never end silently after a tool call."
        .to_owned();

    // Sanitize model name before embedding to prevent prompt injection.
    let safe_model: String = model.chars().filter(|c| !c.is_control()).collect();
    system.push_str(&format!(
        " You are currently running as model: [{safe_model}]."
    ));

    if let Some(pane) = tmux_pane {
        system.push_str(&format!(" The user's active tmux pane is {pane}."));
    }

    if !past_summaries.is_empty() {
        system.push_str("\n\nSummaries from past sessions (oldest first):\n");
        for s in past_summaries {
            system.push_str(&truncate_chars(s, MAX_SUMMARY_CHARS));
            system.push('\n');
        }
    }

    let mut messages = vec![Message::system(system)];

    // If this session has been partially compacted, prepend the running summary
    // so the model knows what happened before the history window.
    if let Some(summary) = own_summary {
        let summary = truncate_chars(summary, MAX_SUMMARY_CHARS * 2);
        messages.push(Message::user(
            "[Summary of earlier in this session]".to_owned(),
        ));
        messages.push(Message::assistant(Some(summary), vec![]));
    }

    for entry in history {
        let content = truncate_chars(&entry.content, MAX_HISTORY_CHARS);
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
    let model = std::env::var("AMAEBI_MODEL")
        .unwrap_or_else(|_| crate::provider::DEFAULT_MODEL.to_string());

    let mut messages = build_messages(&job.description, None, &[], &[], None, &model);
    inject_skill_files(&mut messages).await;
    // Cron jobs are non-interactive: drop the sender immediately so steer_rx.recv()
    // returns None at once if the model ends with '?', rather than timing out.
    let mut sink = tokio::io::sink();
    let (_, mut steer_rx) = tokio::sync::mpsc::channel::<Option<String>>(1);

    let result = run_agentic_loop(
        &state,
        &model,
        messages,
        &mut sink,
        &mut steer_rx,
        true,
        Some(&session_id),
    )
    .await;

    let (output, run_ok) = match result {
        Ok((final_text, _, _, _)) => {
            store_conversation(&state, &session_id, &job.description, &final_text).await;
            (final_text, true)
        }
        Err(e) => {
            tracing::error!(error = %e, id = %job.id, "cron: job failed");
            (format!("[error] {e:#}"), false)
        }
    };

    let task_desc = truncate_chars(&job.description, 200);

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
        assert_eq!(truncate_chars("hello", 10), "hello");
    }

    #[test]
    fn truncate_chars_at_limit_unchanged() {
        assert_eq!(truncate_chars("hello", 5), "hello");
    }

    #[test]
    fn truncate_chars_over_limit_appends_marker() {
        let result = truncate_chars("hello world extra text here", 20);
        assert!(result.ends_with("…[truncated]"), "should end with marker");
        assert_eq!(result.chars().count(), 20);
    }

    #[test]
    fn truncate_chars_total_length_never_exceeds_max() {
        let s = "a".repeat(100);
        for max in [14, 20, 50, 99] {
            let result = truncate_chars(&s, max);
            assert!(
                result.chars().count() <= max,
                "max={max}: got {} chars",
                result.chars().count()
            );
        }
    }

    #[test]
    fn truncate_chars_max_smaller_than_marker_returns_partial_marker() {
        let result = truncate_chars("hello world", 3);
        assert_eq!(result.chars().count(), 3);
        assert!(result.starts_with('…'));
    }

    #[test]
    fn truncate_chars_respects_unicode_boundaries() {
        let s = "日本語テスト".repeat(5);
        let result = truncate_chars(&s, 20);
        assert!(result.chars().count() <= 20);
        assert!(result.ends_with("…[truncated]"));
    }

    #[test]
    fn truncate_chars_empty_string_unchanged() {
        assert_eq!(truncate_chars("", 10), "");
    }

    #[test]
    fn truncate_chars_multibyte_safe() {
        let s = "日".repeat(25);
        let result = truncate_chars(&s, 20);
        assert!(result.chars().count() <= 20);
        assert!(result.ends_with("…[truncated]"));
    }

    // ---- build_messages tests ----------------------------------------------

    #[test]
    fn build_messages_empty_history() {
        let msgs = build_messages("hello", None, &[], &[], None, "claude-sonnet-4.6");
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn build_messages_injects_history_rows() {
        let history = make_history(2); // 4 rows: u0, a0, u1, a1
        let msgs = build_messages("q3", None, &history, &[], None, "claude-sonnet-4.6");
        // system + 4 history rows + user
        assert_eq!(msgs.len(), 6);
    }

    #[test]
    fn build_messages_injects_model_into_system() {
        let msgs = build_messages("hi", None, &[], &[], None, "claude-opus-4.7");
        let system = msgs[0].content.as_deref().unwrap_or("");
        // Model is bracketed: "...as model: [claude-opus-4.7]."
        assert!(
            system.contains("[claude-opus-4.7]"),
            "system prompt must mention the current model in brackets"
        );
    }

    #[test]
    fn build_messages_all_history_included() {
        // build_messages no longer caps — all rows are included.
        let history = make_history(10);
        let msgs = build_messages("new", None, &history, &[], None, "claude-sonnet-4.6");
        // system + 20 history rows + user
        assert_eq!(msgs.len(), 22);
    }

    #[test]
    fn build_messages_tmux_pane_in_system() {
        let msgs = build_messages("prompt", Some("%3"), &[], &[], None, "claude-sonnet-4.6");
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
        let msgs = build_messages("hi", None, &[], &summaries, None, "claude-sonnet-4.6");
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
        let msgs = build_messages(
            "q",
            None,
            &history,
            &[],
            Some("- Did X earlier."),
            "claude-sonnet-4.6",
        );
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

    // ------------------------------------------------------------------
    // find_hot_tail_start tests — boundary selection for in-loop compaction
    // ------------------------------------------------------------------

    #[test]
    fn hot_tail_start_empty_returns_zero() {
        assert_eq!(find_hot_tail_start(&[], 3), 0);
    }

    #[test]
    fn hot_tail_start_single_user_returns_len() {
        // Only one user turn means there is nothing historical to split off.
        let msgs = vec![Message::system("sys"), Message::user("u0")];
        assert_eq!(find_hot_tail_start(&msgs, 3), msgs.len());
    }

    #[test]
    fn hot_tail_start_keeps_one_middle_user() {
        // Four user turns, desired = 3 user turns.  Cap at n_users - 1 = 3,
        // leaving exactly one user turn in the middle for the summariser.
        let msgs = vec![
            Message::system("sys"),
            Message::user("u0"),
            Message::assistant(Some("a0".into()), vec![]),
            Message::user("u1"),
            Message::assistant(Some("a1".into()), vec![]),
            Message::user("u2"),
            Message::assistant(Some("a2".into()), vec![]),
            Message::user("u3"),
        ];
        // Tail starts at the 2nd user (index 3), leaving u0/a0 as the middle.
        assert_eq!(find_hot_tail_start(&msgs, 3), 3);
    }

    #[test]
    fn hot_tail_start_small_desired_keeps_most_as_middle() {
        // Five user turns with desired = 1 user turn → tail starts at the
        // last user turn (index 9), leaving u0..u3 + their assistants as
        // the middle.
        let msgs = vec![
            Message::system("sys"),
            Message::user("u0"),
            Message::assistant(Some("a0".into()), vec![]),
            Message::user("u1"),
            Message::assistant(Some("a1".into()), vec![]),
            Message::user("u2"),
            Message::assistant(Some("a2".into()), vec![]),
            Message::user("u3"),
            Message::assistant(Some("a3".into()), vec![]),
            Message::user("u4"),
        ];
        assert_eq!(find_hot_tail_start(&msgs, 1), 9);
    }

    #[test]
    fn hot_tail_start_preserves_exactly_hot_tail_pairs_user_turns() {
        // Invariant: the number of user turns preserved in-memory equals
        // HOT_TAIL_PAIRS.  With 5 historical user turns and
        // HOT_TAIL_PAIRS = 3 we keep the 3 most recent user turns, each
        // with its assistant reply.
        let msgs = vec![
            Message::system("sys"),
            Message::user("u0"),
            Message::assistant(Some("a0".into()), vec![]),
            Message::user("u1"),
            Message::assistant(Some("a1".into()), vec![]),
            Message::user("u2"),
            Message::assistant(Some("a2".into()), vec![]),
            Message::user("u3"),
            Message::assistant(Some("a3".into()), vec![]),
            Message::user("u4"),
        ];
        let start = find_hot_tail_start(&msgs, HOT_TAIL_PAIRS);
        // Count user turns from `start` to end.
        let preserved_users = msgs[start..].iter().filter(|m| m.role == "user").count();
        assert_eq!(preserved_users, HOT_TAIL_PAIRS);
    }

    #[test]
    fn build_messages_own_summary_marker_matches_compact_head_end_guard() {
        // Invariant: the exact marker string that `build_messages()` injects
        // as the own_summary user turn must match the prefix that
        // `compact_in_loop`'s head_end guard looks for AND the marker that
        // `compact_in_loop` itself writes when splicing a new summary pair.
        //
        // Unifying these three places means: (1) on resume, the injected
        // own_summary pair is preserved by the head_end guard rather than
        // re-summarised on every compaction; (2) within a single connection,
        // a previously-spliced in-memory summary pair is also preserved by
        // the same guard and therefore does NOT leak into `summarised_row_count`
        // (which would cause the DB archive step to overshoot, archiving
        // real rows that the summariser never saw).
        let history = make_history(1);
        let msgs = build_messages(
            "q",
            None,
            &history,
            &[],
            Some("- Did X earlier."),
            "claude-sonnet-4.6",
        );
        // msgs[1] should be the own_summary user label.
        let label = msgs[1].content.as_deref().unwrap_or("");
        assert_eq!(
            label, "[Summary of earlier in this session]",
            "own_summary user marker drifted: {label:?}"
        );
        // And msgs[2] should be the assistant-role summary body.
        assert_eq!(msgs[2].role, "assistant");
    }

    #[test]
    fn compactable_middle_tokens_zero_when_only_current_prompt() {
        // Regression: when messages = [system*, user_prompt] (only the
        // active user prompt, no history), `find_hot_tail_start` returns
        // the `messages.len()` sentinel.  Previously `compaction_boundaries`
        // would let that sentinel survive as `tail_start`, making the
        // caller's middle slice cover everything including the active
        // prompt — so `compact_in_loop` would splice it away and leave
        // messages ending on an assistant summary, which Bedrock rejects
        // with a 400 "must end with a user message".
        //
        // `compactable_middle_tokens` must return 0 in this case so the
        // pre-send threshold check in `run_agentic_loop` does NOT invoke
        // `compact_in_loop` at all.
        let msgs = vec![
            Message::system("base"),
            Message::system("soul"),
            Message::system("agents"),
            Message::user("first prompt in fresh session"),
        ];
        assert_eq!(compactable_middle_tokens(&msgs), 0);
        let (head_end, tail_start) = compaction_boundaries(&msgs);
        assert_eq!(
            tail_start, head_end,
            "boundaries must collapse so middle is empty (got head_end={head_end}, tail_start={tail_start})"
        );
    }

    #[test]
    fn compact_in_loop_bails_when_only_current_prompt_present() {
        // Regression: when the message list has ≤1 user turn (only the
        // active prompt), find_hot_tail_start returns messages.len() as a
        // sentinel.  compact_in_loop must detect that and bail rather than
        // splicing over the current prompt and producing an invalid request
        // ending on the assistant summary.
        let msgs = vec![Message::system("sys"), Message::user("only prompt")];
        let start = find_hot_tail_start(&msgs, 3);
        assert_eq!(start, msgs.len(), "sentinel must equal messages.len()");
    }

    #[test]
    fn hot_tail_start_boundary_is_always_a_user_role() {
        // Invariant: whatever index is returned (if < len) must point at a
        // user-role message so tool_call/tool_result pairings are not split.
        let msgs = vec![
            Message::system("sys"),
            Message::user("u0"),
            Message::assistant(Some("a0".into()), vec![]),
            Message::user("u1"),
            Message::assistant(Some("a1".into()), vec![]),
            Message::user("u2"),
        ];
        let idx = find_hot_tail_start(&msgs, 3);
        if idx < msgs.len() {
            assert_eq!(msgs[idx].role, "user");
        }
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
    fn context_limit_1m_suffix_supported_models() {
        // Supported models with [1m] suffix return 1_000_000.
        assert_eq!(context_limit_for_model("claude-sonnet-4.6[1m]"), 1_000_000);
        assert_eq!(context_limit_for_model("claude-opus-4.6[1m]"), 1_000_000);
        assert_eq!(
            context_limit_for_model("us.anthropic.claude-sonnet-4-6[1m]"),
            1_000_000
        );
        assert_eq!(
            context_limit_for_model("bedrock/us.anthropic.claude-sonnet-4-6[1m]"),
            1_000_000
        );
    }

    #[test]
    fn context_limit_1m_suffix_unsupported_model_falls_back() {
        // Haiku does not support 1M — suffix is ignored, returns standard limit.
        assert_eq!(context_limit_for_model("claude-haiku-3.5[1m]"), 200_000);
    }

    #[test]
    fn context_limit_1m_suffix_copilot_does_not_get_1m_budget() {
        // Copilot provider does not support 1M — even a Claude model ID with
        // [1m] must not receive a 1M token budget.
        assert_eq!(
            context_limit_for_model("copilot/claude-sonnet-4-5[1m]"),
            200_000
        );
        assert_eq!(
            context_limit_for_model("copilot/claude-opus-4.6[1m]"),
            200_000
        );
    }

    #[test]
    fn context_limit_no_1m_suffix_unchanged() {
        // Without suffix, supported models still return standard 200k.
        assert_eq!(context_limit_for_model("claude-sonnet-4.6"), 200_000);
        assert_eq!(
            context_limit_for_model("us.anthropic.claude-sonnet-4-6"),
            200_000
        );
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
        {
            let s = load_skill_messages_from(dir.path()).await;
            splice_skill_messages(&mut messages, s);
        };
        assert_eq!(messages.len(), 2);
        let body = |m: &Message| m.content.as_deref().unwrap_or("").to_owned();
        // Order: SOUL.md first, then AGENTS.md (skills before guidelines).
        assert!(body(&messages[0]).contains("## Soul"));
        assert!(body(&messages[0]).contains("soul content"));
        assert!(body(&messages[1]).contains("## Agent Guidelines"));
        assert!(body(&messages[1]).contains("agent guidelines"));
    }

    #[tokio::test]
    async fn skill_files_inserted_after_system_message() {
        // When messages already contains a system message (and a user message),
        // skill files must be inserted at index 1, preserving [system] /
        // [SOUL.md] / [AGENTS.md] / [user...] order.
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("SOUL.md"), "soul content").unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "agent guidelines").unwrap();
        let mut messages = vec![
            Message::system("base system".to_owned()),
            Message::user("user turn".to_owned()),
        ];
        {
            let s = load_skill_messages_from(dir.path()).await;
            splice_skill_messages(&mut messages, s);
        };
        assert_eq!(messages.len(), 4);
        let body = |m: &Message| m.content.as_deref().unwrap_or("").to_owned();
        assert!(
            body(&messages[0]).contains("base system"),
            "system must stay at index 0"
        );
        assert!(
            body(&messages[1]).contains("## Soul"),
            "SOUL.md must be at index 1"
        );
        assert!(
            body(&messages[2]).contains("## Agent Guidelines"),
            "AGENTS.md must be at index 2"
        );
        assert!(
            body(&messages[3]).contains("user turn"),
            "user message must stay last"
        );
    }

    #[tokio::test]
    async fn skill_files_dev_workflow_not_a_fixed_file() {
        // DEV_WORKFLOW.md is intentionally NOT a fixed file — only AGENTS.md
        // and SOUL.md are auto-injected.  A lone DEV_WORKFLOW.md must not
        // produce any messages.
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("DEV_WORKFLOW.md"), "workflow rules").unwrap();
        let mut messages: Vec<Message> = vec![];
        {
            let s = load_skill_messages_from(dir.path()).await;
            splice_skill_messages(&mut messages, s);
        };
        assert!(
            messages.is_empty(),
            "DEV_WORKFLOW.md must not be auto-injected as a fixed file"
        );
    }

    #[tokio::test]
    async fn skill_files_absent_produces_no_messages() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut messages: Vec<Message> = vec![];
        {
            let s = load_skill_messages_from(dir.path()).await;
            splice_skill_messages(&mut messages, s);
        };
        assert!(messages.is_empty());
    }

    #[tokio::test]
    async fn skill_files_empty_file_skipped() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "   \n  ").unwrap();
        let mut messages: Vec<Message> = vec![];
        {
            let s = load_skill_messages_from(dir.path()).await;
            splice_skill_messages(&mut messages, s);
        };
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
        {
            let s = load_skill_messages_from(dir.path()).await;
            splice_skill_messages(&mut messages, s);
        };
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
        {
            let s = load_skill_messages_from(dir.path()).await;
            splice_skill_messages(&mut messages, s);
        };
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
        {
            let s = load_skill_messages_from(dir.path()).await;
            splice_skill_messages(&mut messages, s);
        };
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

    // ------------------------------------------------------------------
    // needs_copilot_auth tests
    // ------------------------------------------------------------------

    #[test]
    fn bedrock_models_do_not_require_copilot_auth() {
        // Bedrock inference profiles must NOT trigger Copilot token pre-flight.
        assert!(!needs_copilot_auth(
            "us.anthropic.claude-opus-4-6-20251101-v1:0"
        ));
        assert!(!needs_copilot_auth(
            "us.anthropic.claude-sonnet-4-6-20251101-v1:0"
        ));
        assert!(!needs_copilot_auth(
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        ));
    }

    #[test]
    fn copilot_models_require_copilot_auth() {
        // Models explicitly routed through the Copilot provider must trigger auth.
        assert!(needs_copilot_auth("copilot/claude-opus-4-6-20251101"));
        assert!(needs_copilot_auth("copilot/claude-sonnet-4-5"));
        assert!(needs_copilot_auth("copilot/gpt-4o"));
    }

    // ------------------------------------------------------------------
    // requires_responses_api tests
    // ------------------------------------------------------------------

    #[test]
    fn gpt_5_4_requires_responses_api() {
        assert!(requires_responses_api("gpt-5.4"));
    }

    #[test]
    fn gpt_4o_does_not_require_responses_api() {
        assert!(!requires_responses_api("gpt-4o"));
    }

    #[test]
    fn gpt_5_mini_requires_responses_api() {
        assert!(requires_responses_api("gpt-5-mini"));
    }

    #[test]
    #[serial_test::serial]
    fn compact_model_override_used_verbatim() {
        std::env::set_var("AMAEBI_COMPACT_MODEL", "custom/provider-model");
        let result = compact_model("copilot/claude-opus-4-6");
        std::env::remove_var("AMAEBI_COMPACT_MODEL");
        assert_eq!(result, "custom/provider-model");
    }

    #[test]
    #[serial_test::serial]
    fn compact_model_defaults_to_default_model_no_prefix() {
        std::env::remove_var("AMAEBI_COMPACT_MODEL");
        assert_eq!(
            compact_model("claude-opus-4-6"),
            crate::provider::DEFAULT_MODEL
        );
    }

    #[test]
    #[serial_test::serial]
    fn compact_model_preserves_copilot_prefix() {
        std::env::remove_var("AMAEBI_COMPACT_MODEL");
        assert_eq!(
            compact_model("copilot/claude-opus-4-6"),
            format!("copilot/{}", crate::provider::DEFAULT_MODEL),
        );
    }

    #[test]
    #[serial_test::serial]
    fn compact_model_preserves_bedrock_prefix() {
        std::env::remove_var("AMAEBI_COMPACT_MODEL");
        assert_eq!(
            compact_model("bedrock/us.anthropic.claude-opus-4-6-v1:0"),
            format!("bedrock/{}", crate::provider::DEFAULT_MODEL),
        );
    }

    // ------------------------------------------------------------------
    // persist_large_result tests
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn large_tool_output_persisted_to_tmp() {
        let dir = tempfile::TempDir::new().unwrap();
        let big_output = "x".repeat(51_000);
        let result = persist_large_result(big_output, dir.path()).await;
        // Stub must contain a path inside the scratch dir and a Preview section.
        assert!(
            result.contains(dir.path().to_str().unwrap()),
            "result must contain the scratch dir path: {result}"
        );
        assert!(
            result.contains(".txt"),
            "result must reference a .txt file: {result}"
        );
        assert!(
            result.contains("Preview:"),
            "result must contain a preview header: {result}"
        );
        // Stub must be much shorter than the original 51K content.
        assert!(
            result.len() < 10_000,
            "stub must be much shorter than 51K chars, got {} chars",
            result.len()
        );
        // The file must exist in the scratch dir and contain the full content.
        let files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(files.len(), 1, "exactly one file must be written");
        let written = std::fs::read_to_string(files[0].path()).unwrap();
        assert_eq!(written.len(), 51_000);
    }

    #[tokio::test]
    async fn read_file_not_truncated_by_persist() {
        // Validate that persist_large_result produces a short stub — confirming
        // that the loop MUST skip it for read_file to preserve full content.
        // The exemption itself is enforced by should_persist_tool_output (tested below).
        let dir = tempfile::TempDir::new().unwrap();
        let big_output = "x".repeat(51_000);
        let stub = persist_large_result(big_output, dir.path()).await;
        assert!(
            stub.len() < 10_000,
            "persist_large_result must produce a short stub: len={}",
            stub.len()
        );
        assert!(
            stub.contains("Preview:"),
            "stub must contain a preview section"
        );
    }

    #[test]
    fn read_file_exempt_from_persistence() {
        // should_persist_tool_output drives the read_file exemption in the loop.
        // If this returns true for read_file, the loop would call persist_large_result
        // and replace full file content with a stub, breaking the model's only path
        // to read complete files.
        assert!(!should_persist_tool_output("read_file"));
        assert!(should_persist_tool_output("shell_command"));
        assert!(should_persist_tool_output("tmux_capture_pane"));
        assert!(should_persist_tool_output("tmux_wait"));
        assert!(should_persist_tool_output("spawn_agent"));
    }

    #[tokio::test]
    async fn read_file_dedup_returns_stub_when_unchanged() {
        let dir = tempfile::TempDir::new().unwrap();
        let file_path = dir.path().join("test_file.txt");
        std::fs::write(&file_path, "hello world content").unwrap();

        let mut read_cache: std::collections::HashMap<std::path::PathBuf, ((u128, u64), String)> =
            Default::default();

        // First read: capture cache key and populate cache.
        let key = file_cache_key(&file_path)
            .await
            .expect("metadata must be available");
        read_cache.insert(file_path.clone(), (key, "first-call-id".to_string()));

        // Second read with same key → dedup should produce a stub.
        let key2 = file_cache_key(&file_path)
            .await
            .expect("metadata must be available");
        let stub = if let Some((cached_key, cached_id)) = read_cache.get(&file_path) {
            if *cached_key == key2 {
                file_unchanged_stub(cached_id)
            } else {
                "NOT_STUB".to_string()
            }
        } else {
            "NOT_STUB".to_string()
        };
        let expected = file_unchanged_stub("first-call-id");
        assert_eq!(stub, expected, "second read must return unchanged stub");
    }

    #[tokio::test]
    async fn read_file_dedup_re_reads_when_changed() {
        let dir = tempfile::TempDir::new().unwrap();
        let file_path = dir.path().join("test_file2.txt");
        std::fs::write(&file_path, "original content").unwrap();

        let key1 = file_cache_key(&file_path)
            .await
            .expect("metadata must be available");

        let mut read_cache: std::collections::HashMap<std::path::PathBuf, ((u128, u64), String)> =
            Default::default();
        read_cache.insert(file_path.clone(), (key1, "first-call-id".to_string()));

        // Modify the file to change its mtime and/or size.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        std::fs::write(&file_path, "modified content — longer now").unwrap();

        let key2 = file_cache_key(&file_path)
            .await
            .expect("metadata must be available");

        // If both mtime and size are identical (coarse-resolution filesystem),
        // the cache would not correctly detect the change — skip rather than fail.
        if key1 == key2 {
            return;
        }

        let is_stub = if let Some((cached_key, _cached_id)) = read_cache.get(&file_path) {
            *cached_key == key2
        } else {
            false
        };
        assert!(
            !is_stub,
            "after file change, dedup must NOT produce a stub (cache key differs)"
        );
    }

    // ---- update_embedded_model_name ----------------------------------------

    #[test]
    fn update_embedded_model_name_handles_1m_suffix_across_turns() {
        let mut content = String::from(
            "System preface. You are currently running as model: [gpt-4o]. Continue helping.",
        );

        assert!(update_embedded_model_name(
            &mut content,
            "claude-sonnet-4.6[1m]"
        ));
        assert_eq!(
            content,
            "System preface. You are currently running as model: [claude-sonnet-4.6[1m]]. Continue helping."
        );

        // Second turn with the same 1m model must remain stable.
        assert!(update_embedded_model_name(
            &mut content,
            "claude-sonnet-4.6[1m]"
        ));
        assert_eq!(
            content,
            "System preface. You are currently running as model: [claude-sonnet-4.6[1m]]. Continue helping."
        );

        // Switching back to a plain model must also work.
        assert!(update_embedded_model_name(&mut content, "o3-mini"));
        assert_eq!(
            content,
            "System preface. You are currently running as model: [o3-mini]. Continue helping."
        );
    }

    #[test]
    fn update_embedded_model_name_strips_control_chars() {
        let mut content = String::from("You are currently running as model: [old]. rest of prompt");
        assert!(update_embedded_model_name(&mut content, "evil\ninjected"));
        assert_eq!(
            content,
            "You are currently running as model: [evilinjected]. rest of prompt"
        );
    }

    // ---- switch_model validation (via parse_switch_model_arg helper) -------

    #[test]
    fn switch_model_rejects_missing_arg() {
        assert!(parse_switch_model_arg(None).is_err());
    }

    #[test]
    fn switch_model_rejects_empty_string() {
        assert!(parse_switch_model_arg(Some("")).is_err());
    }

    #[test]
    fn switch_model_rejects_whitespace_only() {
        assert!(parse_switch_model_arg(Some("   ")).is_err());
    }

    #[test]
    fn switch_model_accepts_valid_model_name() {
        for raw in [
            "claude-sonnet-4.6",
            "bedrock/claude-opus-4.6",
            "copilot/gpt-4o",
        ] {
            let result = parse_switch_model_arg(Some(raw));
            assert!(result.is_ok(), "valid model must be accepted: {raw}");
            assert_eq!(result.unwrap(), raw);
        }
    }

    #[test]
    fn switch_model_rejects_control_characters() {
        assert!(parse_switch_model_arg(Some("model\nignore above")).is_err());
        assert!(parse_switch_model_arg(Some("model\x00null")).is_err());
        assert!(parse_switch_model_arg(Some("model\ttab")).is_err());
    }

    #[test]
    fn switch_model_trims_whitespace() {
        let result = parse_switch_model_arg(Some("  claude-opus-4.6  "));
        assert_eq!(result.unwrap(), "claude-opus-4.6");
    }

    #[test]
    fn switch_model_preserves_1m_suffix() {
        let result = parse_switch_model_arg(Some("claude-sonnet-4.6[1m]"));
        assert_eq!(result.unwrap(), "claude-sonnet-4.6[1m]");
    }

    // ---- /model client-side intercept: parse_model_command -----------------

    /// Simulate the `/model` parsing logic used in run_chat_loop.
    fn parse_model_command(input: &str) -> Option<Option<String>> {
        let rest = input
            .strip_prefix("/model")
            .and_then(|r| r.strip_prefix(|c: char| c.is_whitespace()).or(Some(r)))?;
        let trimmed = rest.trim();
        if trimmed.is_empty() {
            Some(None) // query current model
        } else {
            Some(Some(trimmed.to_string())) // set model
        }
    }

    #[test]
    fn model_command_set_preserves_1m() {
        assert_eq!(
            parse_model_command("/model claude-sonnet-4.6[1m]"),
            Some(Some("claude-sonnet-4.6[1m]".into()))
        );
    }

    #[test]
    fn model_command_set_opus() {
        assert_eq!(
            parse_model_command("/model claude-opus-4.6[1m]"),
            Some(Some("claude-opus-4.6[1m]".into()))
        );
    }

    #[test]
    fn model_command_query() {
        assert_eq!(parse_model_command("/model"), Some(None));
    }

    #[test]
    fn model_command_not_model() {
        assert_eq!(parse_model_command("hello"), None);
    }

    #[test]
    fn model_command_unicode_space() {
        assert_eq!(
            parse_model_command("/model\u{3000}claude-sonnet-4.6[1m]"),
            Some(Some("claude-sonnet-4.6[1m]".into()))
        );
    }

    // ---- carried_model vs client model priority ----------------------------

    #[test]
    fn carried_model_yields_to_different_client_model() {
        // Simulates the logic at handle_chat_request line ~2174:
        // if client sends a different model, client wins.
        let carried: Option<String> = Some("claude-sonnet-4.6".into());
        let client_model = "claude-sonnet-4.6[1m]".to_string();
        let effective = match carried {
            Some(c) if c == client_model => c,
            Some(_) => client_model.clone(),
            None => client_model.clone(),
        };
        assert_eq!(effective, "claude-sonnet-4.6[1m]");
    }

    #[test]
    fn carried_model_used_when_same_as_client() {
        let carried: Option<String> = Some("claude-sonnet-4.6".into());
        let client_model = "claude-sonnet-4.6".to_string();
        let effective = match carried {
            Some(c) if c == client_model => c,
            Some(_) => client_model.clone(),
            None => client_model.clone(),
        };
        assert_eq!(effective, "claude-sonnet-4.6");
    }

    #[test]
    fn no_carried_model_uses_client() {
        let carried: Option<String> = None;
        let client_model = "claude-opus-4.6[1m]".to_string();
        let effective = match carried {
            Some(c) if c == client_model => c,
            Some(_) => client_model.clone(),
            None => client_model.clone(),
        };
        assert_eq!(effective, "claude-opus-4.6[1m]");
    }

    // -----------------------------------------------------------------------
    // extract_pr_number
    // -----------------------------------------------------------------------

    #[test]
    fn extract_pr_number_pr99() {
        assert_eq!(extract_pr_number("fix PR99 copilot review"), Some(99));
    }

    #[test]
    fn extract_pr_number_pr_hash_99() {
        assert_eq!(extract_pr_number("fix PR #99"), Some(99));
    }

    #[test]
    fn extract_pr_number_hash_only() {
        assert_eq!(extract_pr_number("fix #123 issue"), Some(123));
    }

    #[test]
    fn extract_pr_number_case_insensitive() {
        assert_eq!(extract_pr_number("fix pr42"), Some(42));
    }

    #[test]
    fn extract_pr_number_chinese_context() {
        assert_eq!(
            extract_pr_number("修复sglang的PR99里的copilot review"),
            Some(99)
        );
    }

    #[test]
    fn extract_pr_number_none_when_absent() {
        assert_eq!(extract_pr_number("fix the build"), None);
    }

    #[test]
    fn extract_pr_number_ignores_non_numeric() {
        assert_eq!(extract_pr_number("improve PRoductivity"), None);
    }

    // -----------------------------------------------------------------------
    // json_str_field
    // -----------------------------------------------------------------------

    #[test]
    fn json_str_field_extracts_value() {
        let json = r#"{"headRefName":"feat/paged","title":"Add FMHA"}"#;
        assert_eq!(json_str_field(json, "headRefName"), "feat/paged");
        assert_eq!(json_str_field(json, "title"), "Add FMHA");
    }

    #[test]
    fn json_str_field_missing_key() {
        let json = r#"{"headRefName":"feat/paged"}"#;
        assert_eq!(json_str_field(json, "baseRefName"), "");
    }

    #[test]
    fn json_str_field_empty_json() {
        assert_eq!(json_str_field("", "key"), "");
    }

    // -----------------------------------------------------------------------
    // sanitize_remote_url
    // -----------------------------------------------------------------------

    #[test]
    fn sanitize_remote_url_https_with_token() {
        assert_eq!(
            sanitize_remote_url("https://ghp_abc123@github.com/org/repo.git"),
            "https://***@github.com/org/repo.git"
        );
    }

    #[test]
    fn sanitize_remote_url_https_no_creds() {
        assert_eq!(
            sanitize_remote_url("https://github.com/org/repo.git"),
            "https://github.com/org/repo.git"
        );
    }

    #[test]
    fn sanitize_remote_url_scp_style_with_token() {
        assert_eq!(
            sanitize_remote_url("ghp_abc123@github.com:org/repo.git"),
            "***@github.com:org/repo.git"
        );
    }

    #[test]
    fn sanitize_remote_url_scp_style_git_user() {
        assert_eq!(
            sanitize_remote_url("git@github.com:org/repo.git"),
            "***@github.com:org/repo.git"
        );
    }

    #[test]
    fn sanitize_remote_url_plain_url() {
        assert_eq!(
            sanitize_remote_url("https://github.com/org/repo"),
            "https://github.com/org/repo"
        );
    }

    #[test]
    fn sanitize_remote_url_ssh_scheme() {
        assert_eq!(
            sanitize_remote_url("ssh://user@host/path"),
            "ssh://***@host/path"
        );
    }

    // -----------------------------------------------------------------------
    // summarise_tool_detail
    // -----------------------------------------------------------------------

    #[test]
    fn summarise_tool_detail_shell_short_ascii() {
        let args = serde_json::json!({ "command": "ls -la" });
        assert_eq!(summarise_tool_detail("shell_command", &args), "ls -la");
    }

    #[test]
    fn summarise_tool_detail_shell_long_ascii_truncates_with_ellipsis() {
        let cmd: String = "x".repeat(SHELL_DETAIL_MAX_CHARS * 2);
        let args = serde_json::json!({ "command": cmd });
        let detail = summarise_tool_detail("shell_command", &args);
        // SHELL_DETAIL_MAX_CHARS chars + `…`.
        assert_eq!(detail.chars().count(), SHELL_DETAIL_MAX_CHARS + 1);
        assert!(detail.ends_with('…'));
        assert!(detail.starts_with(&"x".repeat(SHELL_DETAIL_MAX_CHARS)));
    }

    #[test]
    fn summarise_tool_detail_shell_multibyte_utf8_does_not_panic() {
        // Regression: previously the branch used `&s[..80]` (byte slice) which
        // panicked when byte 80 landed inside a multi-byte UTF-8 sequence.
        // This exact input — mixed ASCII + Chinese + em-dash + newline + path —
        // was seen in a real Bedrock supervision turn and crashed the daemon.
        let cmd = "# 看 flash-attention 的 tile_scheduler 怎么 dispatch m_block — \
                   它是怎么避免 OOB 的\nsed -n '440,480p' /home/yuankuns/path.hpp";
        let args = serde_json::json!({ "command": cmd });
        // Should return Ok without panicking.
        let detail = summarise_tool_detail("shell_command", &args);
        // If the input has more than SHELL_DETAIL_MAX_CHARS chars, output
        // should end with '…'; otherwise it's returned verbatim.
        if cmd.chars().count() > SHELL_DETAIL_MAX_CHARS {
            assert!(detail.ends_with('…'));
            assert_eq!(detail.chars().count(), SHELL_DETAIL_MAX_CHARS + 1);
        } else {
            assert_eq!(detail, cmd);
        }
    }

    #[test]
    fn summarise_tool_detail_shell_exactly_max_chars_not_truncated() {
        // Boundary: exactly SHELL_DETAIL_MAX_CHARS scalars → return as-is, no ellipsis.
        let cmd: String = "a".repeat(SHELL_DETAIL_MAX_CHARS);
        let args = serde_json::json!({ "command": cmd.clone() });
        assert_eq!(summarise_tool_detail("shell_command", &args), cmd);
    }

    #[test]
    fn summarise_tool_detail_read_file_returns_path() {
        let args = serde_json::json!({ "path": "/tmp/foo.rs" });
        assert_eq!(summarise_tool_detail("read_file", &args), "/tmp/foo.rs");
        assert_eq!(summarise_tool_detail("edit_file", &args), "/tmp/foo.rs");
    }

    #[test]
    fn summarise_tool_detail_missing_arg_returns_empty() {
        let args = serde_json::json!({});
        assert_eq!(summarise_tool_detail("shell_command", &args), "");
        assert_eq!(summarise_tool_detail("read_file", &args), "");
    }

    #[test]
    fn summarise_tool_detail_unknown_tool_returns_empty() {
        let args = serde_json::json!({ "command": "ignored" });
        assert_eq!(summarise_tool_detail("bogus_tool", &args), "");
    }
}
