use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

use crate::auth::{amaebi_home, TokenCache};
use crate::copilot::{self, ApiToolCall, ApiToolCallFunction, FinishReason, Message};
use crate::inbox::InboxStore;
use crate::ipc::{write_frame, Request, Response};
use crate::memory_db;
use crate::tools::{self, ToolExecutor};

// ---------------------------------------------------------------------------
// Session TTL tiers
// ---------------------------------------------------------------------------

/// Default TTL tiers for in-memory session eviction.
///
/// These can be overridden via `AMAEBI_SESSION_TTL_<TIER>` env vars (seconds).
pub struct TtlConfig {
    pub tiers: HashMap<String, Duration>,
}

impl TtlConfig {
    pub fn from_env() -> Self {
        let mut tiers = HashMap::new();
        tiers.insert("default".to_string(), Duration::from_secs(30 * 60));
        tiers.insert("ephemeral".to_string(), Duration::from_secs(5 * 60));
        tiers.insert("persistent".to_string(), Duration::from_secs(24 * 60 * 60));

        // Load from ~/.amaebi/config.json if present.
        let cfg = crate::config::Config::load();
        if let Some(&minutes) = cfg.ttl_minutes.get("default") {
            tiers.insert("default".to_string(), Duration::from_secs(minutes * 60));
        }
        // Non-path keys in ttl_minutes that aren't "default" are tier names.
        for (key, &minutes) in &cfg.ttl_minutes {
            if key != "default" && !key.starts_with('/') {
                tiers.insert(key.clone(), Duration::from_secs(minutes * 60));
            }
        }

        // Allow env overrides: AMAEBI_SESSION_TTL_DEFAULT=3600, etc.
        for (key, val) in std::env::vars() {
            if let Some(tier) = key
                .strip_prefix("AMAEBI_SESSION_TTL_")
                .map(|s| s.to_lowercase())
            {
                if let Ok(secs) = val.parse::<u64>() {
                    tiers.insert(tier, Duration::from_secs(secs));
                }
            }
        }

        Self { tiers }
    }

    pub fn ttl_for(&self, tier: &str) -> Duration {
        self.tiers.get(tier).copied().unwrap_or_else(|| {
            self.tiers
                .get("default")
                .copied()
                .unwrap_or(Duration::from_secs(30 * 60))
        })
    }
}

/// How many history entries to inject as context when building messages.
const MAX_HISTORY: usize = 20;

/// Background eviction check interval.
const EVICTION_INTERVAL: Duration = Duration::from_secs(5 * 60);

// ---------------------------------------------------------------------------
// Per-session memory entry
// ---------------------------------------------------------------------------

/// A single user/assistant exchange stored in per-session in-memory history.
#[derive(Clone, Debug)]
pub struct MemoryEntry {
    /// The user's prompt (possibly truncated).
    pub user: String,
    /// The assistant's response (possibly truncated).
    pub assistant: String,
}

// ---------------------------------------------------------------------------
// Per-session conversation state
// ---------------------------------------------------------------------------

/// In-memory state for a single session (one directory's conversation thread).
///
/// Protected by its own `Mutex` so that concurrent requests for the *same*
/// session block until the previous agentic loop completes, ensuring causal
/// ordering of history writes.  Requests for *different* sessions run freely
/// in parallel.
pub struct Session {
    /// Chronological exchange history, newest last.
    pub history: Vec<MemoryEntry>,
    /// Updated at the start of each request; used for TTL eviction.
    pub last_active: Instant,
    /// TTL tier label — determines eviction timing.
    pub ttl_tier: String,
}

impl Session {
    fn new() -> Self {
        Self {
            history: Vec::new(),
            last_active: Instant::now(),
            ttl_tier: "default".to_string(),
        }
    }

    #[allow(dead_code)]
    fn with_tier(tier: &str) -> Self {
        Self {
            history: Vec::new(),
            last_active: Instant::now(),
            ttl_tier: tier.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Session store
// ---------------------------------------------------------------------------

/// Thread-safe map of session UUID → `Arc<Mutex<Session>>`.
///
/// The top-level `Mutex` is held only while doing the HashMap lookup/insert;
/// callers then hold the per-session `Mutex` for the duration of the agentic
/// loop.  This design allows different sessions to run concurrently while
/// serialising same-session requests.
pub struct SessionStore {
    inner: tokio::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<Session>>>>,
    pub ttl_config: TtlConfig,
}

impl SessionStore {
    pub fn new() -> Self {
        Self {
            inner: tokio::sync::Mutex::new(HashMap::new()),
            ttl_config: TtlConfig::from_env(),
        }
    }

    /// Retrieve (or create) the `Arc<Mutex<Session>>` for `session_id`.
    ///
    /// Uses a two-phase approach: first check if the key exists (read-only),
    /// then insert only if missing.  The top-level lock is released before
    /// returning so callers can hold the per-session lock without blocking
    /// other sessions.
    pub async fn get_or_create(&self, session_id: &str) -> Arc<tokio::sync::Mutex<Session>> {
        // Single lock acquisition — the map lock is held only briefly.
        let mut map = self.inner.lock().await;
        if let Some(arc) = map.get(session_id) {
            Arc::clone(arc)
        } else {
            let arc = Arc::new(tokio::sync::Mutex::new(Session::new()));
            map.insert(session_id.to_string(), Arc::clone(&arc));
            arc
        }
    }

    /// Retrieve (or create) with a specific TTL tier.
    #[allow(dead_code)]
    pub async fn get_or_create_with_tier(
        &self,
        session_id: &str,
        tier: &str,
    ) -> Arc<tokio::sync::Mutex<Session>> {
        let mut map = self.inner.lock().await;
        if let Some(arc) = map.get(session_id) {
            Arc::clone(arc)
        } else {
            let arc = Arc::new(tokio::sync::Mutex::new(Session::with_tier(tier)));
            map.insert(session_id.to_string(), Arc::clone(&arc));
            arc
        }
    }

    /// Return the number of tracked sessions.
    #[allow(dead_code)]
    pub async fn len(&self) -> usize {
        self.inner.lock().await.len()
    }

    /// Remove all sessions from the store.
    #[cfg(test)]
    pub async fn clear(&self) {
        self.inner.lock().await.clear();
    }

    /// Evict sessions that have been inactive longer than their tier's TTL.
    ///
    /// Sessions that are currently locked (actively running an agentic loop)
    /// are never evicted — `try_lock` returns `Err` in that case.
    pub async fn evict_expired(&self) {
        let mut map = self.inner.lock().await;
        let before = map.len();
        let config = &self.ttl_config;
        map.retain(|_, arc| {
            if let Ok(session) = arc.try_lock() {
                session.last_active.elapsed() < config.ttl_for(&session.ttl_tier)
            } else {
                true // actively in use — keep it
            }
        });
        let evicted = before - map.len();
        if evicted > 0 {
            tracing::info!(evicted, "evicted expired sessions");
        }
    }
}

// ---------------------------------------------------------------------------
// Shared daemon state
// ---------------------------------------------------------------------------

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
    /// Per-session conversation history, keyed by session UUID.
    pub sessions: SessionStore,
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
            sessions: SessionStore::new(),
        })
    }
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

    // Background task: periodically evict expired sessions.
    let state_evict = Arc::clone(&state);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(EVICTION_INTERVAL);
        interval.tick().await; // skip the immediate first tick
        loop {
            interval.tick().await;
            state_evict.sessions.evict_expired().await;
        }
    });

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
            store_conversation(&state, &user, &assistant).await;
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

            let sid = session_id.unwrap_or_else(|| "global".to_string());

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
                let session_arc = state.sessions.get_or_create(&sid).await;
                let mut session = session_arc.lock().await;
                session.last_active = Instant::now();
                let mut messages = build_messages(&prompt, tmux_pane.as_deref(), &session.history);
                inject_skill_files(&mut messages).await;

                // Use a sink writer — output frames are discarded; we only
                // need the return value (final_text) for the inbox.
                let mut sink = tokio::io::sink();
                let (_steer_tx, mut steer_rx) = tokio::sync::mpsc::channel::<String>(1);

                match run_agentic_loop(&state, &model, messages, &mut sink, &mut steer_rx).await {
                    Ok(final_text) => {
                        let entry = MemoryEntry {
                            user: truncate_chars(prompt.clone(), MAX_PROMPT_CHARS),
                            assistant: truncate_chars(final_text.clone(), MAX_RESPONSE_CHARS),
                        };
                        session.history.push(entry.clone());
                        session.last_active = Instant::now();
                        drop(session);

                        // Persist to SQLite memory store.
                        store_conversation(&state, &entry.user, &entry.assistant).await;

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
                        drop(session);
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
            tokio::spawn(async move {
                while let Ok(Some(frame)) = lines.next_line().await {
                    match serde_json::from_str::<Request>(&frame) {
                        Ok(Request::Steer { message, .. }) => {
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

            let session_arc = state.sessions.get_or_create(&session_id).await;
            let mut session = session_arc.lock().await;
            session.last_active = Instant::now();

            // Resume: load the FULL history without the MAX_HISTORY cap.
            let mut messages =
                build_messages_resume(&prompt, tmux_pane.as_deref(), &session.history);
            inject_skill_files(&mut messages).await;

            match run_agentic_loop(&state, &model, messages, &mut writer, &mut steer_rx).await {
                Ok(response_text) => {
                    let entry = MemoryEntry {
                        user: truncate_chars(prompt.clone(), MAX_PROMPT_CHARS),
                        assistant: truncate_chars(response_text.clone(), MAX_RESPONSE_CHARS),
                    };
                    session.history.push(entry.clone());
                    session.last_active = Instant::now();
                    drop(session);

                    store_conversation(&state, &entry.user, &entry.assistant).await;
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

            // Steering channel: the spawned reader task sends user corrections
            // here; the agentic loop drains them between model turns.
            let (steer_tx, mut steer_rx) = tokio::sync::mpsc::channel::<String>(16);

            // Spawn a task that reads subsequent frames from the client on
            // this connection.  Any Steer frames are forwarded to steer_tx so
            // the running agentic loop can inject them between tool turns.
            // The task exits when the client closes the connection (EOF) or
            // when steer_tx is dropped (agentic loop finished).
            tokio::spawn(async move {
                while let Ok(Some(frame)) = lines.next_line().await {
                    match serde_json::from_str::<Request>(&frame) {
                        Ok(Request::Steer { message, .. }) => {
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

            // Resolve session — fall back to "global" when the client does not
            // provide one (e.g., old clients or non-directory contexts).
            let sid = session_id.unwrap_or_else(|| "global".to_string());
            let session_arc = state.sessions.get_or_create(&sid).await;

            // Acquire the per-session lock for the full duration of the
            // agentic loop.  This ensures causal ordering: a second request
            // for the same session blocks here until the first completes and
            // writes its exchange into `session.history`.
            let mut session = session_arc.lock().await;
            session.last_active = Instant::now();

            let mut messages = build_messages(&prompt, tmux_pane.as_deref(), &session.history);
            inject_skill_files(&mut messages).await;

            match run_agentic_loop(&state, &model, messages, &mut writer, &mut steer_rx).await {
                Ok(response_text) => {
                    let entry = MemoryEntry {
                        user: truncate_chars(prompt.clone(), MAX_PROMPT_CHARS),
                        assistant: truncate_chars(response_text.clone(), MAX_RESPONSE_CHARS),
                    };

                    // Append to per-session in-memory history.
                    session.history.push(entry.clone());
                    session.last_active = Instant::now();

                    // Release the per-session lock before doing any I/O.
                    drop(session);

                    // Persist to SQLite memory store.
                    store_conversation(&state, &entry.user, &entry.assistant).await;
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
pub(crate) async fn store_conversation(state: &DaemonState, user: &str, assistant: &str) {
    let db = Arc::clone(&state.db);
    let user_owned = user.to_owned();
    let assistant_owned = assistant.to_owned();
    let result = tokio::task::spawn_blocking(move || {
        let timestamp = chrono::Utc::now().to_rfc3339();
        let mut conn = db.lock().unwrap_or_else(|p| p.into_inner());
        // Write the user/assistant pair atomically so they are never split.
        let tx = conn.transaction().context("beginning memory transaction")?;
        memory_db::store_memory(&tx, &timestamp, "", "user", &user_owned, "")?;
        memory_db::store_memory(&tx, &timestamp, "", "assistant", &assistant_owned, "")?;
        tx.commit().context("committing memory transaction")
    })
    .await
    .unwrap_or_else(|e| Err(anyhow::anyhow!("memory write panicked: {e}")));
    if let Err(e) = result {
        tracing::warn!(error = %e, "failed to save memory");
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
/// stdin.  Any messages in the channel are drained **between tool turns**
/// (after all tool results are appended but before the next model call) and
/// pushed as `user` messages.  A [`Response::SteerAck`] frame is sent for
/// each message consumed.
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
) -> Result<String>
where
    W: AsyncWriteExt + Unpin,
{
    let schemas = tools::tool_schemas();
    let final_text;

    loop {
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

        match resp.finish_reason {
            FinishReason::Stop | FinishReason::Length => {
                final_text = resp.text;
                break;
            }

            FinishReason::ToolCalls => {
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

                // Drain any steering corrections that arrived while tools were
                // running.  Each is injected as a fresh `user` turn so the
                // model sees it before its next response.
                while let Ok(steer_msg) = steer_rx.try_recv() {
                    messages.push(Message::user(steer_msg));
                    write_frame(writer, &Response::SteerAck).await?;
                }
            }

            FinishReason::Other(ref reason) => {
                tracing::warn!(finish_reason = %reason, "unexpected finish reason, stopping");
                let warning = format!("\n[stopped: unexpected finish reason '{reason}']");
                write_frame(writer, &Response::Text { chunk: warning }).await?;
                final_text = resp.text;
                break;
            }
        }
    }

    write_frame(writer, &Response::Done).await?;
    Ok(final_text)
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

pub(crate) fn build_messages(
    prompt: &str,
    tmux_pane: Option<&str>,
    history: &[MemoryEntry],
) -> Vec<Message> {
    build_messages_inner(prompt, tmux_pane, history, false)
}

/// Like [`build_messages`] but loads the **full** chronological history without
/// the `MAX_HISTORY` sliding-window cap.  Used by the `--resume` path to
/// re-hydrate a complete prior session.
pub(crate) fn build_messages_resume(
    prompt: &str,
    tmux_pane: Option<&str>,
    history: &[MemoryEntry],
) -> Vec<Message> {
    build_messages_inner(prompt, tmux_pane, history, true)
}

fn build_messages_inner(
    prompt: &str,
    tmux_pane: Option<&str>,
    history: &[MemoryEntry],
    full_history: bool,
) -> Vec<Message> {
    let mut system = "You are a helpful, concise AI assistant embedded in a tmux terminal. \
                      Answer in plain text; avoid markdown unless the user asks for it. \
                      You have tools available to inspect the terminal, run commands, \
                      and read or edit files — use them when they help you answer accurately."
        .to_owned();

    if let Some(pane) = tmux_pane {
        system.push_str(&format!(" The user's active tmux pane is {pane}."));
    }

    let mut messages = vec![Message::system(system)];

    // In normal mode apply a sliding-window cap; in resume mode load everything.
    let window = if full_history || history.len() <= MAX_HISTORY {
        history
    } else {
        &history[history.len() - MAX_HISTORY..]
    };
    for entry in window {
        messages.push(Message::user(entry.user.clone()));
        messages.push(Message::assistant(Some(entry.assistant.clone()), vec![]));
    }

    messages.push(Message::user(prompt.to_owned()));
    messages
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(user: &str, assistant: &str) -> MemoryEntry {
        MemoryEntry {
            user: user.to_string(),
            assistant: assistant.to_string(),
        }
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
        // max=20: 12 chars for marker + 8 chars of content = 20 total
        let s = "hello world extra text here".to_owned(); // 27 chars > 20
        let result = truncate_chars(s, 20);
        assert!(result.ends_with("…[truncated]"), "should end with marker");
        assert_eq!(result.chars().count(), 20);
    }

    #[test]
    fn truncate_chars_total_length_never_exceeds_max() {
        // Verify the hard cap for various max values.
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
        // "日本語テスト" is 6 chars, each 3 bytes; slicing bytes would panic.
        // max=20 gives room for content + marker (total ≤ 20).
        let s = "日本語テスト".repeat(5); // 30 chars
        let result = truncate_chars(s, 20);
        assert!(
            result.chars().count() <= 20,
            "total length must not exceed max"
        );
        assert!(result.ends_with("…[truncated]"));
    }

    #[test]
    fn truncate_chars_empty_string_unchanged() {
        assert_eq!(truncate_chars(String::new(), 10), "");
    }

    #[test]
    fn truncate_chars_multibyte_safe() {
        // Each '日' is 3 bytes but 1 char; make sure we don't split bytes.
        let s = "日".repeat(25);
        let result = truncate_chars(s, 20);
        assert!(result.chars().count() <= 20);
        assert!(result.ends_with("…[truncated]"));
    }

    // ---- SessionStore tests ------------------------------------------------

    #[tokio::test]
    async fn session_store_get_or_create_returns_same_arc() {
        let store = SessionStore::new();
        let a1 = store.get_or_create("abc").await;
        let a2 = store.get_or_create("abc").await;
        assert!(Arc::ptr_eq(&a1, &a2));
    }

    #[tokio::test]
    async fn session_store_different_ids_get_different_arcs() {
        let store = SessionStore::new();
        let a = store.get_or_create("session-a").await;
        let b = store.get_or_create("session-b").await;
        assert!(!Arc::ptr_eq(&a, &b));
    }

    #[tokio::test]
    async fn session_store_clear_removes_all() {
        let store = SessionStore::new();
        store.get_or_create("x").await;
        store.get_or_create("y").await;
        store.clear().await;
        // After clear, getting "x" should return a *new* Arc (different pointer).
        let pre = {
            let mut map = store.inner.lock().await;
            let arc = Arc::new(tokio::sync::Mutex::new(Session::new()));
            map.insert("x".to_string(), Arc::clone(&arc));
            arc
        };
        store.clear().await;
        let post = store.get_or_create("x").await;
        assert!(!Arc::ptr_eq(&pre, &post));
    }

    #[tokio::test]
    async fn session_store_evict_expired_removes_stale() {
        let store = SessionStore::new();
        let default_ttl = store.ttl_config.ttl_for("default");
        {
            let arc = store.get_or_create("stale").await;
            let mut s = arc.lock().await;
            s.last_active = Instant::now()
                .checked_sub(default_ttl + Duration::from_secs(1))
                .expect("test clock arithmetic");
        }
        store.evict_expired().await;
        let map = store.inner.lock().await;
        assert!(map.is_empty());
    }

    #[tokio::test]
    async fn session_store_evict_keeps_active() {
        let store = SessionStore::new();
        store.get_or_create("active").await;
        store.evict_expired().await;
        let map = store.inner.lock().await;
        assert!(map.contains_key("active"));
    }

    #[tokio::test]
    async fn session_store_multi_tier_eviction() {
        let store = SessionStore::new();
        let ephemeral_ttl = store.ttl_config.ttl_for("ephemeral");

        // Create an ephemeral session backdated past ephemeral TTL but within persistent TTL.
        {
            let arc = store.get_or_create_with_tier("eph", "ephemeral").await;
            let mut s = arc.lock().await;
            s.last_active = Instant::now()
                .checked_sub(ephemeral_ttl + Duration::from_secs(1))
                .expect("clock");
        }
        // Create a persistent session backdated the same amount.
        {
            let arc = store.get_or_create_with_tier("persist", "persistent").await;
            let mut s = arc.lock().await;
            s.last_active = Instant::now()
                .checked_sub(ephemeral_ttl + Duration::from_secs(1))
                .expect("clock");
        }

        store.evict_expired().await;
        let map = store.inner.lock().await;
        // Ephemeral should be evicted, persistent should survive.
        assert!(
            !map.contains_key("eph"),
            "ephemeral session should be evicted"
        );
        assert!(
            map.contains_key("persist"),
            "persistent session should survive"
        );
    }

    #[tokio::test]
    async fn session_store_len() {
        let store = SessionStore::new();
        assert_eq!(store.len().await, 0);
        store.get_or_create("a").await;
        store.get_or_create("b").await;
        assert_eq!(store.len().await, 2);
        store.clear().await;
        assert_eq!(store.len().await, 0);
    }

    // ---- build_messages tests ----------------------------------------------

    #[test]
    fn build_messages_empty_history() {
        let msgs = build_messages("hello", None, &[]);
        // system + user
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn build_messages_injects_history_as_pairs() {
        let history = vec![make_entry("q1", "a1"), make_entry("q2", "a2")];
        let msgs = build_messages("q3", None, &history);
        // system + 2*(user+assistant) + user
        assert_eq!(msgs.len(), 6);
    }

    #[test]
    fn build_messages_caps_history_at_max() {
        let history: Vec<MemoryEntry> = (0..=MAX_HISTORY)
            .map(|i| make_entry(&format!("u{i}"), &format!("a{i}")))
            .collect();
        let msgs = build_messages("new", None, &history);
        // system + MAX_HISTORY*(user+assistant) + user
        let expected = 1 + MAX_HISTORY * 2 + 1;
        assert_eq!(msgs.len(), expected);
    }

    #[test]
    fn build_messages_tmux_pane_in_system() {
        let msgs = build_messages("prompt", Some("%3"), &[]);
        // The first message is the system message; its content contains the pane id.
        let content = msgs[0].content.as_deref().unwrap_or("");
        assert!(
            content.contains("%3"),
            "system prompt should mention the pane"
        );
    }

    // ---- build_messages_resume tests (full history, no sliding window) -----

    #[test]
    fn build_messages_resume_loads_all_history_beyond_max() {
        // Create history larger than MAX_HISTORY so we can verify it's all loaded.
        let history: Vec<MemoryEntry> = (0..=MAX_HISTORY)
            .map(|i| make_entry(&format!("u{i}"), &format!("a{i}")))
            .collect();
        let msgs = build_messages_resume("new", None, &history);
        // system + ALL history entries as pairs + new user prompt
        let expected = 1 + (MAX_HISTORY + 1) * 2 + 1;
        assert_eq!(
            msgs.len(),
            expected,
            "resume should load full history, not just last {MAX_HISTORY}"
        );
    }

    #[test]
    fn build_messages_resume_small_history_unchanged() {
        let history = vec![make_entry("q1", "a1"), make_entry("q2", "a2")];
        // When history fits within MAX_HISTORY, both variants produce identical output.
        let normal = build_messages("q3", None, &history);
        let resume = build_messages_resume("q3", None, &history);
        assert_eq!(
            normal.len(),
            resume.len(),
            "small history should produce same message count"
        );
    }

    #[test]
    fn build_messages_caps_at_max_history_normal() {
        // Verify normal path still applies the cap after refactor.
        let history: Vec<MemoryEntry> = (0..=MAX_HISTORY)
            .map(|i| make_entry(&format!("u{i}"), &format!("a{i}")))
            .collect();
        let msgs = build_messages("new", None, &history);
        // system + MAX_HISTORY*(user+assistant) + user
        let expected = 1 + MAX_HISTORY * 2 + 1;
        assert_eq!(msgs.len(), expected);
    }

    // ------------------------------------------------------------------
    // inject_skill_files tests
    // ------------------------------------------------------------------

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
        // Only SOUL.md present.
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
