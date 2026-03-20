use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

use crate::auth::TokenCache;
use crate::copilot::{self, ApiToolCall, ApiToolCallFunction, FinishReason, Message};
use crate::inbox::InboxStore;
use crate::ipc::{write_frame, Request, Response};
use crate::memory::{self, MemoryEntry};
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

    /// Remove all sessions from the store.
    pub async fn clear(&self) {
        self.inner.lock().await.clear();
    }

    /// Return the number of tracked sessions.
    #[allow(dead_code)]
    pub async fn len(&self) -> usize {
        self.inner.lock().await.len()
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
    /// Serialises concurrent `memory::append` calls within this process so that
    /// parallel client connections cannot interleave their writes to the memory file.
    pub memory_lock: tokio::sync::Mutex<()>,
    /// Per-session conversation history, keyed by session UUID.
    pub sessions: SessionStore,
}

impl DaemonState {
    pub fn new() -> Result<Self> {
        let http = reqwest::Client::builder()
            .build()
            .context("building HTTP client")?;
        Ok(Self {
            http,
            tokens: TokenCache::new(),
            executor: Box::new(tools::LocalExecutor),
            memory_lock: tokio::sync::Mutex::new(()),
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

    let state = Arc::new(DaemonState::new()?);

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
        Request::ClearCache => {
            tracing::info!("received cache clear request");
            state.sessions.clear().await;
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
                let messages = build_messages(&prompt, tmux_pane.as_deref(), &session.history);

                // Use a sink writer — output frames are discarded; we only
                // need the return value (final_text) for the inbox.
                let mut sink = tokio::io::sink();
                let (_steer_tx, mut steer_rx) = tokio::sync::mpsc::channel::<String>(1);

                match run_agentic_loop(&state, &model, messages, &mut sink, &mut steer_rx).await {
                    Ok(final_text) => {
                        let entry = MemoryEntry {
                            timestamp: chrono::Utc::now().to_rfc3339(),
                            user: truncate_chars(&prompt, MAX_PROMPT_CHARS),
                            assistant: truncate_chars(&final_text, MAX_RESPONSE_CHARS),
                        };
                        session.history.push(entry.clone());
                        session.last_active = Instant::now();
                        drop(session);

                        // Persist to global memory store.
                        let mem_guard = state.memory_lock.lock().await;
                        let p = entry.user.clone();
                        let r = entry.assistant.clone();
                        let mem_res = tokio::task::spawn_blocking(move || memory::append(&p, &r))
                            .await
                            .unwrap_or_else(|e| {
                                Err(anyhow::anyhow!("memory::append panicked: {e}"))
                            });
                        drop(mem_guard);
                        if let Err(e) = mem_res {
                            tracing::warn!(error = %e, "detach: failed to persist memory");
                        }

                        // Save result to inbox so the user gets a notification.
                        let task_desc = truncate_chars(&prompt, 200);
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
                        let task_desc = truncate_chars(&prompt, 200);
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
            let messages = build_messages_resume(&prompt, tmux_pane.as_deref(), &session.history);

            match run_agentic_loop(&state, &model, messages, &mut writer, &mut steer_rx).await {
                Ok(response_text) => {
                    let entry = MemoryEntry {
                        timestamp: chrono::Utc::now().to_rfc3339(),
                        user: truncate_chars(&prompt, MAX_PROMPT_CHARS),
                        assistant: truncate_chars(&response_text, MAX_RESPONSE_CHARS),
                    };
                    session.history.push(entry.clone());
                    session.last_active = Instant::now();
                    drop(session);

                    let mem_guard = state.memory_lock.lock().await;
                    let p = entry.user.clone();
                    let r = entry.assistant.clone();
                    let mem_result = tokio::task::spawn_blocking(move || memory::append(&p, &r))
                        .await
                        .unwrap_or_else(|e| Err(anyhow::anyhow!("memory::append panicked: {e}")));
                    drop(mem_guard);
                    if let Err(e) = mem_result {
                        tracing::warn!(error = %e, "resume: failed to persist memory");
                    }
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

            let messages = build_messages(&prompt, tmux_pane.as_deref(), &session.history);

            match run_agentic_loop(&state, &model, messages, &mut writer, &mut steer_rx).await {
                Ok(response_text) => {
                    let entry = MemoryEntry {
                        timestamp: chrono::Utc::now().to_rfc3339(),
                        user: truncate_chars(&prompt, MAX_PROMPT_CHARS),
                        assistant: truncate_chars(&response_text, MAX_RESPONSE_CHARS),
                    };

                    // Append to per-session in-memory history.
                    session.history.push(entry.clone());
                    session.last_active = Instant::now();

                    // Release the per-session lock before doing any I/O.
                    drop(session);

                    // Persist to the global JSONL backing store (for
                    // `amaebi memory list/search`).  The memory_lock
                    // serialises concurrent writes within this process;
                    // memory::append holds an flock for cross-process safety.
                    let mem_guard = state.memory_lock.lock().await;
                    let prompt_copy = entry.user.clone();
                    let response_copy = entry.assistant.clone();
                    let mem_result = tokio::task::spawn_blocking(move || {
                        memory::append(&prompt_copy, &response_copy)
                    })
                    .await
                    .unwrap_or_else(|e| Err(anyhow::anyhow!("memory::append panicked: {e}")));
                    drop(mem_guard);

                    if let Err(e) = mem_result {
                        tracing::warn!(error = %e, "failed to persist memory to disk");
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

/// Truncate `s` to at most `max` Unicode scalar values.
///
/// If truncation occurs, appends `MARKER` so the recipient can see the text
/// was cut.  The total output is guaranteed to be ≤ `max` chars.
pub(crate) fn truncate_chars(s: &str, max: usize) -> String {
    const MARKER: &str = "…[truncated]";
    let marker_len = MARKER.chars().count();

    if s.chars().count() <= max {
        return s.to_string();
    }

    // We want `body_max` chars of content + marker ≤ max chars total.
    let body_max = max.saturating_sub(marker_len);
    let end = s
        .char_indices()
        .nth(body_max)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    format!("{}{}", &s[..end], MARKER)
}

// ---------------------------------------------------------------------------
// Agentic loop
// ---------------------------------------------------------------------------

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
        let resp =
            copilot::stream_chat(&state.http, &token, model, &messages, &schemas, writer).await?;

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
                            output
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
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            user: user.to_string(),
            assistant: assistant.to_string(),
        }
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
        let persistent_ttl = store.ttl_config.ttl_for("persistent");

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

    // ---- truncate_chars tests ----------------------------------------------

    #[test]
    fn truncate_chars_short_string_unchanged() {
        assert_eq!(truncate_chars("hello", 20), "hello");
    }

    #[test]
    fn truncate_chars_exact_limit_unchanged() {
        let s = "a".repeat(20);
        assert_eq!(truncate_chars(&s, 20), s);
    }

    #[test]
    fn truncate_chars_over_limit_appends_marker() {
        let s = "a".repeat(27);
        let result = truncate_chars(&s, 20);
        assert!(result.ends_with("…[truncated]"));
        assert!(result.chars().count() <= 20);
    }

    #[test]
    fn truncate_chars_multibyte_safe() {
        // Each '日' is 3 bytes but 1 char; make sure we don't split bytes.
        let s = "日".repeat(25);
        let result = truncate_chars(&s, 20);
        assert!(result.chars().count() <= 20);
        assert!(result.ends_with("…[truncated]"));
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
}
