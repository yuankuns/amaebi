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
use crate::resource_lease;
use crate::session;
use crate::tasks;
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
///   2. Same provider prefix as `main_model` + per-provider default
///      (`default_model_for_provider`): Bedrock gets `[1m]`, Copilot gets bare
fn compact_model(main_model: &str) -> String {
    if let Ok(override_model) = std::env::var("AMAEBI_COMPACT_MODEL") {
        return override_model;
    }
    // Preserve the provider prefix so compaction uses the same API backend.
    // Pick the per-provider default so copilot never gets `[1m]`, which it
    // does not support.
    let prefix = main_model
        .split_once('/')
        .map(|(p, _)| p)
        .filter(|p| matches!(*p, "copilot" | "bedrock"));
    match prefix {
        Some(p) => format!("{}/{}", p, crate::provider::default_model_for_provider(p)),
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
    /// User-defined model aliases loaded once from `~/.amaebi/config.json` at
    /// daemon startup.  Expanded by `expand_user_alias` at each request entry
    /// point (and inside the `switch_model` tool handler) so the rest of the
    /// pipeline can keep using bare `provider::resolve()`.  Empty when the
    /// config file is missing or has no aliases.
    ///
    /// **Not hot-reloaded**: the daemon snapshots the alias table once at
    /// startup.  The user must restart the daemon after editing
    /// `~/.amaebi/config.json` for changes to take effect.
    pub user_aliases: Arc<std::collections::HashMap<String, String>>,
    /// Persistent SQLite connection for the task notebook
    /// (`~/.amaebi/tasks.db`).  Shared `Mutex` for the same reason as
    /// `db`: all reads and writes serialise through a single connection.
    /// Lazy-initialised — the first `/claude --tag <tag>` request opens
    /// it; invocations without `--tag` never touch the file.
    pub tasks_db: Arc<Mutex<Option<rusqlite::Connection>>>,
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

        // Load user model aliases from the config file.  Never fatal — falls
        // back to an empty map if the file is missing or malformed.
        let user_aliases = Arc::new(crate::config::Config::load().model_aliases);

        let spawn_ctx = Arc::new(tools::SpawnContext {
            http: http.clone(),
            db: Arc::clone(&db),
            compacting_sessions: Arc::clone(&compacting_sessions),
            tokens: Arc::clone(&tokens),
            user_aliases: Arc::clone(&user_aliases),
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
            user_aliases,
            tasks_db: Arc::new(Mutex::new(None)),
        })
    }
}

/// Open (or reuse) the task notebook database handle on `state`.
/// On first invocation opens `~/.amaebi/tasks.db` and stashes the
/// connection in `state.tasks_db`; subsequent calls are no-ops.
/// Callers must be inside `spawn_blocking` — this helper runs file I/O
/// on first invocation.
pub(crate) fn ensure_tasks_db(state: &Arc<DaemonState>) -> Result<()> {
    let mut guard = state
        .tasks_db
        .lock()
        .map_err(|e| anyhow::anyhow!("tasks_db mutex poisoned: {e}"))?;
    if guard.is_some() {
        return Ok(());
    }
    let path = tasks::db_path().context("resolving tasks DB path")?;
    let conn = tasks::init_db(&path).context("opening tasks DB")?;
    *guard = Some(conn);
    Ok(())
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

/// Expand a user-defined alias in `model` using the daemon's loaded alias
/// table.  Returns the expanded target string, or the original input if no
/// alias matches.  The `[1m]` suffix is preserved across expansion so
/// `amaebi chat --model opus[1m]` works when `opus` maps to a Bedrock model.
///
/// This is called once on every request-entry so the rest of the daemon can
/// keep using bare `provider::resolve()` without knowing about user aliases.
pub(crate) fn expand_user_alias(
    model: &str,
    user_aliases: &std::collections::HashMap<String, String>,
) -> String {
    // If the caller typed `name[1m]`, strip the suffix, expand the bare name,
    // then reattach the suffix.  Alias targets in config.json may themselves
    // carry `[1m]` (e.g. `"sonnet": "bedrock/claude-sonnet-4.6[1m]"`) — the
    // downstream `provider::resolve()` normalizes the suffix either way.
    let (bare, suffix) = if let Some(stripped) = model.strip_suffix("[1m]") {
        (stripped, "[1m]")
    } else {
        (model, "")
    };

    // Built-in aliases win on conflict (same rule as provider::resolve_with_aliases).
    if bare.contains('/') || crate::provider::is_builtin_bedrock_alias(bare) {
        return model.to_owned();
    }
    match user_aliases.get(bare) {
        Some(target) => format!("{target}{suffix}"),
        None => model.to_owned(),
    }
}

// ---------------------------------------------------------------------------
// Listener loop
// ---------------------------------------------------------------------------

/// Probe whether a Unix socket at `path` is backed by a live listener.
///
/// Returns `Ok(true)` if a peer accepts our connect (another daemon is alive);
/// `Ok(false)` if the path is stale — missing, a regular file, or a socket
/// with no listener.  Any other failure (permissions, hung peer, etc.) is
/// bubbled up as an error so we never silently unlink something we can't
/// reason about.
async fn socket_in_use(path: &std::path::Path) -> Result<bool> {
    let connect = tokio::net::UnixStream::connect(path);
    match tokio::time::timeout(std::time::Duration::from_millis(500), connect).await {
        Ok(Ok(_stream)) => Ok(true),
        Ok(Err(e)) => {
            // ENOTSOCK (88 on Linux) surfaces as ErrorKind::Other in current
            // stable Rust; match on the raw errno so a leftover regular file
            // at the socket path is still classified as stale.
            const ENOTSOCK: i32 = 88;
            match e.kind() {
                std::io::ErrorKind::ConnectionRefused | std::io::ErrorKind::NotFound => Ok(false),
                _ if e.raw_os_error() == Some(ENOTSOCK) => Ok(false),
                _ => Err(anyhow::Error::new(e)
                    .context(format!("probing daemon socket {}", path.display()))),
            }
        }
        Err(_) => anyhow::bail!(
            "timed out probing daemon socket {}; assuming another daemon is alive",
            path.display()
        ),
    }
}

/// Try to bind the daemon's Unix socket, safely handling a stale leftover
/// path.
///
/// TOCTOU-safe: if a bare `bind` fails with `AddrInUse`, we probe with
/// `socket_in_use`.  A live peer means refuse to start; a dead peer means
/// unlink and retry bind once.  If bind still fails we surface the error.
///
/// Racing daemon startups remain safe:
/// * If peer A bound first and is live, peer B's retry probe reports live
///   → peer B errors out (desired).
/// * If peer A crashed and left a stale socket, the probe reports stale →
///   peer B unlinks and binds.  If peer A's path was already unlinked (or
///   another process is in the middle of the same retry), the unlink /
///   bind may fail; we bubble the error up rather than looping forever.
async fn acquire_socket_listener(socket: &std::path::Path) -> Result<UnixListener> {
    match UnixListener::bind(socket) {
        Ok(listener) => return Ok(listener),
        Err(e) if e.kind() != std::io::ErrorKind::AddrInUse => {
            return Err(
                anyhow::Error::new(e).context(format!("binding Unix socket {}", socket.display()))
            );
        }
        Err(_) => { /* AddrInUse — fall through to probe + unlink + retry */ }
    }

    if socket_in_use(socket).await? {
        anyhow::bail!(
            "daemon already running on {}; refusing to start a second instance",
            socket.display()
        );
    }

    // Probe said stale.  Unlink and retry bind.  Tolerate NotFound in case
    // another process beat us to the cleanup.
    match std::fs::remove_file(socket) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            return Err(anyhow::Error::new(e)
                .context(format!("removing stale socket {}", socket.display())));
        }
    }

    UnixListener::bind(socket).with_context(|| format!("binding Unix socket {}", socket.display()))
}

pub async fn run(socket: PathBuf) -> Result<()> {
    let listener = acquire_socket_listener(&socket).await?;

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
///
/// Summaries are scoped to the session's working directory: we look up the
/// dir for `session_id` in `sessions.json` and pass it to
/// `get_recent_summaries` so cross-project sessions cannot see each other's
/// compacted context.  When `resolve_session_dir` falls back to
/// [`UNKNOWN_DIR_SENTINEL`] (unknown UUID, I/O error, lock failure, etc.)
/// we short-circuit and return no summaries at all — this avoids reading
/// back rows that were themselves written with the same sentinel after a
/// previous failed resolution, and it never reads legacy `dir = ''` rows
/// either.
async fn load_session_state(
    state: &Arc<DaemonState>,
    session_id: &str,
) -> (Vec<memory_db::DbMemoryEntry>, Vec<String>, Option<String>) {
    let dir = resolve_session_dir(session_id).await;
    let dir_for_read = dir.clone();
    with_db(Arc::clone(&state.db), {
        let sid = session_id.to_owned();
        move |conn| {
            let summaries = if dir_for_read == UNKNOWN_DIR_SENTINEL {
                Vec::new()
            } else {
                memory_db::get_recent_summaries(conn, &sid, &dir_for_read, MAX_SUMMARIES)?
            };
            Ok((
                memory_db::get_session_history(conn, &sid)?,
                summaries,
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

/// Sentinel stored as `session_summaries.dir` when the real directory for
/// a session cannot be recovered.
///
/// Collision reasoning: `session::canonical_key` prefers the result of
/// `std::fs::canonicalize`, which on Unix always produces an absolute path
/// (`/…`).  It only falls back to the raw input path when `canonicalize`
/// fails (dir deleted/inaccessible), so a relative path key is possible in
/// degenerate cases.  The sentinel is therefore chosen to not match any
/// typical path component: it starts with `<` and ends with `>`, a
/// combination no sane working directory would produce, and it is compared
/// as an exact whole string — not a prefix — so `/tmp/<unknown>/…` would
/// not collide.  If a real dir ever did happen to equal the sentinel
/// verbatim, the only consequence is that its sessions' summaries get
/// lumped into the quarantine set, which is strictly safer than leaking
/// across unrelated projects.
///
/// Defense-in-depth: `load_session_state` also explicitly short-circuits
/// when the resolved dir equals this sentinel, so a row written after one
/// failed resolve is not read back by the same session on a subsequent
/// failed resolve.  Sentinel rows are therefore quarantined on write AND
/// ignored on read.
const UNKNOWN_DIR_SENTINEL: &str = "<unknown>";

/// Resolve the working directory for `session_id` from `sessions.json`.
///
/// Scans the full per-directory history so resumed sessions (UUIDs that
/// have been rotated past) still resolve correctly.  On any failure —
/// I/O error, parse error, lock acquisition failure, unknown UUID, or a
/// panic in the spawned blocking thread — the sentinel
/// [`UNKNOWN_DIR_SENTINEL`] is returned so the caller's summary gets
/// quarantined instead of being written/read as an empty-dir row that
/// could match legacy pre-migration entries.
///
/// Blocking behavior: the underlying `session::dir_for_uuid` takes a
/// `lock_shared()` on the sessions lock file, which is a genuinely
/// blocking call — under contention this waits rather than failing
/// fast.  It runs on `spawn_blocking`, so the daemon's async runtime is
/// not stalled; only the blocking pool thread is held.
///
/// Real I/O / lock failures are logged at debug level so silent masking
/// of persistent problems can still be traced in logs.
async fn resolve_session_dir(session_id: &str) -> String {
    let sid = session_id.to_owned();
    tokio::task::spawn_blocking(move || match session::dir_for_uuid(&sid) {
        Ok(Some(dir)) => dir,
        Ok(None) => {
            // Unknown UUID — expected for brand-new or purged sessions;
            // quarantine silently without a log.
            UNKNOWN_DIR_SENTINEL.to_string()
        }
        Err(e) => {
            tracing::debug!(
                session_id = %sid,
                error = %e,
                "resolve_session_dir: dir_for_uuid failed; falling back to sentinel"
            );
            UNKNOWN_DIR_SENTINEL.to_string()
        }
    })
    .await
    .unwrap_or_else(|e| {
        tracing::debug!(
            session_id,
            error = %e,
            "resolve_session_dir: blocking task panicked; falling back to sentinel"
        );
        UNKNOWN_DIR_SENTINEL.to_string()
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
                let model = expand_user_alias(&model, &state.user_aliases);
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
                let model = expand_user_alias(&model, &state.user_aliases);
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
                let model = expand_user_alias(&model, &state.user_aliases);
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

            Request::ClaudeLaunch {
                tasks,
                session_id,
                repo_dir,
            } => {
                handle_claude_launch(&writer, tasks, session_id, repo_dir, &state).await?;
            }

            Request::GenerateTag {
                description,
                repo_dir,
            } => {
                let tag =
                    crate::task_tagger::generate_tag(&state, None, &description, &repo_dir).await;
                let mut w = writer.lock().await;
                write_frame(&mut *w, &Response::TagGenerated { tag }).await?;
            }

            Request::SupervisePanes {
                panes,
                model,
                session_id,
            } => {
                let model = expand_user_alias(&model, &state.user_aliases);
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
    session_id: Option<String>,
    repo_dir: Option<String>,
    state: &Arc<DaemonState>,
) -> Result<()> {
    if tasks.is_empty() {
        let mut w = writer.lock().await;
        write_frame(&mut *w, &Response::Done).await?;
        return Ok(());
    }

    // Acquire notebook leases BEFORE any pane allocation or `claude`
    // startup.  Without this, a tag-conflict rejection would fire from
    // `handle_supervision_inner` long after the racing launch has
    // already created worktrees and kicked off claude in real panes —
    // leaving an uncontrolled session behind.  All-or-nothing: on
    // conflict, roll back whatever we've acquired and return without
    // touching tmux.  Skipped when the caller didn't opt into the
    // notebook (no session_id or repo_dir).  Holder id matches the one
    // `handle_supervision` will later derive from the same session_id,
    // so `release_all_by_holder` in the supervision cleanup path
    // releases the leases we're taking here.
    //
    // `lease_holder` is `Some` iff we actually inserted rows above; on
    // any early-return path below we release by holder so a partial
    // per-task failure doesn't leak leases until the 24 h TTL expires.
    // The happy path disarms it just before returning, since the
    // follow-up `SupervisePanes` wrapper owns the cleanup from that
    // point on.
    let mut lease_holder: Option<String> = None;
    if let (Some(sid), Some(repo)) = (session_id.as_deref(), repo_dir.as_deref()) {
        let holder = format!("supervision:{sid}");
        // Dedup tags within this launch — multiple panes may share a
        // tag when the caller passed `--tag foo` once for several
        // tasks.  Without dedup the second acquire would see the row
        // the first one just inserted and reject the whole request as
        // a self-conflict.
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let lease_tags: Vec<String> = tasks
            .iter()
            .filter(|t| t.resume_pane.is_none())
            .filter_map(|t| {
                if seen.insert(t.tag.clone()) {
                    Some(t.tag.clone())
                } else {
                    None
                }
            })
            .collect();
        if !lease_tags.is_empty() {
            let state_cl = Arc::clone(state);
            let repo_cl = repo.to_string();
            let holder_cl = holder.clone();
            let acquire_result =
                tokio::task::spawn_blocking(move || -> Result<Result<(), String>> {
                    ensure_tasks_db(&state_cl)?;
                    let mut guard = state_cl
                        .tasks_db
                        .lock()
                        .map_err(|e| anyhow::anyhow!("tasks_db mutex poisoned: {e}"))?;
                    let conn = guard.as_mut().expect("tasks_db just ensured");
                    let mut acquired: Vec<String> = Vec::new();
                    for tag in &lease_tags {
                        match tasks::acquire_lease(conn, &repo_cl, tag, &holder_cl)? {
                            tasks::AcquireLeaseResult::Acquired => {
                                acquired.push(tag.clone());
                            }
                            tasks::AcquireLeaseResult::Held {
                                holder: incumbent,
                                age_secs,
                            } => {
                                for t in &acquired {
                                    let _ = tasks::release_lease(conn, &repo_cl, t, &holder_cl);
                                }
                                let hours = age_secs / 3600;
                                let mins = (age_secs % 3600) / 60;
                                let msg = format!(
                                    "task '{tag}' is already held by {incumbent} \
                                     (age {hours}h{mins}m, TTL 24h); wait for it to \
                                     finish, pick another tag, or use \
                                     `amaebi tag release {tag}` to force-release"
                                );
                                return Ok(Err(msg));
                            }
                        }
                    }
                    Ok(Ok(()))
                })
                .await
                .map_err(|e| anyhow::anyhow!("task-lease acquire panicked: {e}"))??;
            if let Err(msg) = acquire_result {
                let mut w = writer.lock().await;
                write_frame(
                    &mut *w,
                    &Response::Error {
                        message: format!("[supervision] {msg}"),
                    },
                )
                .await?;
                write_frame(&mut *w, &Response::Done).await?;
                return Ok(());
            }
            lease_holder = Some(holder);
        }
    }

    // For each task: acquire a pane (auto-expanding the pool if needed), then
    // launch `claude` (or inject a prompt into an already-running session) via
    // tmux send-keys.
    // `ensure_and_acquire_idle` holds a single LOCK_EX for both expansion and
    // acquisition, eliminating the TOCTOU race.
    //
    // On any error-short-circuit inside the per-task loop (pane acquisition
    // fail, session resolution fail, resource fail, tmux injection fail,
    // etc.) we release notebook leases before returning via
    // `release_launch_leases!` — otherwise the leases would sit until the
    // 24 h TTL since the client, having seen `Response::Error`, never
    // sends the follow-up `SupervisePanes` that would have taken over
    // cleanup.  Placed immediately before every mid-launch
    // `return Ok(());` that follows a `Response::Error` write.
    macro_rules! release_launch_leases {
        () => {{
            if let Some(holder) = lease_holder.as_deref() {
                release_task_leases_for_holder(state, holder).await;
            }
        }};
    }

    let total_tasks = tasks.len();
    for (tagx, task) in tasks.into_iter().enumerate() {
        let tag = task.tag.clone();
        let auto_enter = task.auto_enter;

        // `--resume-pane` + `--resource` is allowed: we still need the
        // lock in resource-state.json even on the reuse path so two
        // tasks can't race for the same simulator.  The env/prompt_hint
        // parts are skipped further down, guarded by `!had_claude`.

        // Gather git context from the client's working directory: current
        // branch, remote URL, recent commits, and PR-specific information if
        // the task description mentions a PR number.  The context is prepended
        // to the description so Claude knows where to start, what branch it is
        // on, and how to push when done.
        // For the resume-pane path we may need to pull the description from
        // the lease first (when the user typed `/claude --resume-pane %N`
        // with no description).  So resolve `raw_description` BEFORE running
        // the git-context prefix step.  In the normal path `task.description`
        // is always non-empty (parser guards it).
        let resume_prefetched_desc: Option<String> = if let Some(ref rp) = task.resume_pane {
            if task.description.trim().is_empty() {
                // Distinguish four failure modes so the error message matches
                // what actually went wrong: state-read I/O failure, unknown
                // pane id, pane exists but has no saved description, or the
                // blocking task itself panicked/was cancelled.
                enum LeaseDescLookup {
                    Found(String),
                    PaneMissing,
                    NoDescription,
                    ReadFailed(String),
                }
                let rp_owned = rp.clone();
                let join_res =
                    tokio::task::spawn_blocking(move || match pane_lease::read_state() {
                        Err(e) => LeaseDescLookup::ReadFailed(e.to_string()),
                        Ok(state) => match state.get(&rp_owned) {
                            None => LeaseDescLookup::PaneMissing,
                            Some(lease) => match lease
                                .task_description
                                .as_deref()
                                .map(str::trim)
                                .filter(|s| !s.is_empty())
                            {
                                Some(d) => LeaseDescLookup::Found(d.to_string()),
                                None => LeaseDescLookup::NoDescription,
                            },
                        },
                    })
                    .await;

                let lookup = match join_res {
                    Ok(l) => l,
                    Err(join_err) => {
                        // Blocking task panicked or was cancelled — surface
                        // that distinctly instead of claiming a disk-read
                        // failure we did not actually observe.
                        let mut w = writer.lock().await;
                        write_frame(
                            &mut *w,
                            &Response::Error {
                                message: format!(
                                    "[error] resume-pane lookup task panicked \
                                     while resolving --resume-pane {rp}: {join_err}"
                                ),
                            },
                        )
                        .await?;
                        drop(w);
                        release_launch_leases!();
                        return Ok(());
                    }
                };

                match lookup {
                    LeaseDescLookup::Found(d) => Some(d),
                    other => {
                        let message = match other {
                            LeaseDescLookup::PaneMissing => format!(
                                "[error] pane {rp} not found in lease state; \
                                 run `amaebi dashboard` to list active panes"
                            ),
                            LeaseDescLookup::NoDescription => format!(
                                "[error] pane {rp} has no saved task description; \
                                 pass a description explicitly after --resume-pane"
                            ),
                            LeaseDescLookup::ReadFailed(e) => format!(
                                "[error] failed to read pane lease state while \
                                 resolving --resume-pane {rp}: {e}"
                            ),
                            LeaseDescLookup::Found(_) => unreachable!(),
                        };
                        let mut w = writer.lock().await;
                        write_frame(&mut *w, &Response::Error { message }).await?;
                        drop(w);
                        release_launch_leases!();
                        return Ok(());
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        let raw_desc = resume_prefetched_desc
            .clone()
            .unwrap_or_else(|| task.description.clone());

        let (description, ctx_start_branch) = {
            let raw = raw_desc.clone();
            let cwd = task.client_cwd.clone();
            tokio::task::spawn_blocking(move || {
                let ctx = gather_task_context(cwd.as_deref(), &raw);
                let enriched = format!("{}\n{}", ctx.preamble, raw);
                (enriched, ctx.start_branch)
            })
            .await
            .unwrap_or_else(|_| (raw_desc.clone(), None))
        };

        // Determine worktree + acquire pane.  Two paths:
        //
        // 1. `--resume-pane <pid>` path: reuse a specific pane whose lease
        //    already records a worktree and `has_claude = true`.  The CLI
        //    parser rejects `--resume-pane` combined with `--worktree`, so
        //    `task.worktree` is always None here.  We read the lease, inherit
        //    its worktree, and acquire THAT pane specifically via
        //    `pane_lease::acquire_lease`.  `had_claude` is forced true so
        //    `handle_claude_launch` runs the tier-1 reuse path (inject task
        //    into existing claude) instead of launching a fresh `claude`
        //    process.
        //
        // 2. Normal path: auto-create a worktree if the caller didn't pass
        //    `--worktree`, then let `ensure_and_acquire_idle` pick a pane.
        //
        // `was_explicit_worktree` gates cleanup of auto-created worktrees on
        // pane-acquisition failure.  The resume-pane path never auto-creates
        // anything, so no cleanup is needed there.
        let was_explicit_worktree = task.worktree.is_some() || task.resume_pane.is_some();
        let sid_placeholder = uuid::Uuid::new_v4().to_string();

        let (pane_id, had_claude, worktree): (String, bool, Option<String>) = if let Some(ref rp) =
            task.resume_pane
        {
            // --- resume-pane path ---
            let rp_owned = rp.clone();
            let tid_for_lease = tag.clone();
            let sid_for_lease = sid_placeholder.clone();
            let probe = tokio::task::spawn_blocking(move || {
                    let state = pane_lease::read_state()?;
                    let lease = state.get(&rp_owned).ok_or_else(|| {
                        anyhow::anyhow!(
                            "pane {rp_owned} not found in lease state; run `amaebi dashboard` to list active panes"
                        )
                    })?;
                    if !lease.has_claude {
                        anyhow::bail!(
                            "pane {rp_owned} is not marked in lease state as having `claude` started; drop --resume-pane and let the scheduler pick or start a new pane"
                        );
                    }
                    let wt = lease.worktree.clone().ok_or_else(|| {
                        anyhow::anyhow!(
                            "pane {rp_owned} has no associated worktree; cannot resume"
                        )
                    })?;
                    // The lease's `has_claude` flag is persisted state and can
                    // go stale (e.g. user `Ctrl-C`'d claude without the daemon
                    // noticing).  Cross-check at the tmux layer so we don't
                    // inject a task prompt into a bare shell.
                    let tmux_probe = std::process::Command::new("tmux")
                        .args([
                            "display-message",
                            "-p",
                            "-t",
                            &rp_owned,
                            "#{pane_current_command}",
                        ])
                        .output()
                        .with_context(|| {
                            format!(
                                "failed to inspect tmux pane {rp_owned}; cannot verify that `claude` is running"
                            )
                        })?;
                    if !tmux_probe.status.success() {
                        let stderr =
                            String::from_utf8_lossy(&tmux_probe.stderr).trim().to_string();
                        if stderr.is_empty() {
                            anyhow::bail!(
                                "failed to inspect tmux pane {rp_owned}; cannot verify that `claude` is running"
                            );
                        } else {
                            anyhow::bail!(
                                "failed to inspect tmux pane {rp_owned}; cannot verify that `claude` is running: {stderr}"
                            );
                        }
                    }
                    let pane_current_command =
                        String::from_utf8_lossy(&tmux_probe.stdout).trim().to_string();
                    if pane_current_command != "claude" {
                        anyhow::bail!(
                            "pane {rp_owned} is not currently running `claude` (tmux reports `{pane_current_command}`); drop --resume-pane and let the scheduler pick or start a new pane"
                        );
                    }
                    pane_lease::acquire_lease(
                        &rp_owned,
                        &tid_for_lease,
                        &sid_for_lease,
                        Some(&wt),
                    )?;
                    Ok::<(String, bool, Option<String>), anyhow::Error>((rp_owned, true, Some(wt)))
                })
                .await
                .context("resume-pane probe task panicked")?;

            match probe {
                Ok(triple) => triple,
                Err(e) => {
                    let mut w = writer.lock().await;
                    write_frame(
                        &mut *w,
                        &Response::Error {
                            message: format!("[error] {e:#}"),
                        },
                    )
                    .await?;
                    drop(w);
                    release_launch_leases!();
                    return Ok(());
                }
            }
        } else {
            // --- normal path: auto-worktree + scheduler-picked pane ---
            let wt_val: Option<String> = match task.worktree.clone() {
                Some(wt) => Some(wt),
                None => {
                    let tid = tag.clone();
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
                                tag = %tag,
                                error = %e,
                                "auto-worktree creation failed; launching claude without worktree isolation"
                            );
                            None
                        }
                    }
                }
            };

            let tid_for_lease = tag.clone();
            let wt_for_lease = wt_val.clone();
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

            match pane_result {
                Ok((pid, hc)) => (pid, hc, wt_val),
                Err(e) => {
                    cleanup_auto_worktree(was_explicit_worktree, &wt_val, &task.client_cwd).await;
                    let remaining = total_tasks - tagx;
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
                    drop(w);
                    release_launch_leases!();
                    return Ok(());
                }
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
                let resource_pane = failed_pane.clone();
                tokio::task::spawn_blocking(move || {
                    let _ = pane_lease::release_lease(&failed_pane);
                })
                .await
                .ok();
                // No resources have been acquired yet at this point (session
                // resolution runs before resource acquisition), but release
                // defensively so a future reordering can't strand resources.
                let _ = resource_lease::release_all_for_pane(&resource_pane).await;
                let mut w = writer.lock().await;
                write_frame(
                    &mut *w,
                    &Response::Error {
                        message: format!("[error] {e:#}"),
                    },
                )
                .await?;
                drop(w);
                release_launch_leases!();
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

        // Resource acquisition.  Held for the pane's supervision lifetime and
        // released from `release_supervised_panes`.  Failures here roll back
        // the pane lease so it doesn't stay Busy until TTL: a task that can't
        // get its hardware should free the pane for someone else.
        let (resource_leases, resource_pool): (
            Vec<resource_lease::ResourceLease>,
            Vec<resource_lease::ResourceDef>,
        ) = if task.resources.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            let requests: Vec<resource_lease::ResourceRequest> = task
                .resources
                .iter()
                .map(|s| resource_lease::ResourceRequest::parse(s))
                .collect();
            let holder = resource_lease::Holder {
                pane_id: pane_id.clone(),
                tag: tag.clone(),
                session_id: session_id.clone(),
            };
            let wait = match task.resource_timeout_secs {
                Some(secs) if secs > 0 => resource_lease::WaitPolicy::Wait {
                    timeout: std::time::Duration::from_secs(secs),
                },
                _ => resource_lease::WaitPolicy::Nowait,
            };
            match resource_lease::acquire_all(&requests, holder, wait).await {
                Ok(leases) => {
                    // Reload the pool for env/prompt-hint rendering.  If the
                    // TOML was edited into an invalid state between
                    // acquisition and this call, fall back to an empty pool
                    // — which means `render_env` / `render_prompt_hint`
                    // return nothing for leases whose pool entries are
                    // unrecoverable, logged inside `render_env`.  Logged
                    // loudly here so an operator can correlate a pane
                    // launching without expected env vars with a failed
                    // pool reload, rather than wondering why the LLM was
                    // never told about its resource.
                    let pool = match tokio::task::spawn_blocking(resource_lease::load_pool).await {
                        Ok(Ok(p)) => p,
                        Ok(Err(e)) => {
                            tracing::error!(
                                pane_id = %pane_id,
                                error = %e,
                                "failed to reload ~/.amaebi/resources.toml after acquisition; \
                                 env vars and prompt_hint will NOT be injected into this pane"
                            );
                            Vec::new()
                        }
                        Err(e) => {
                            tracing::error!(
                                pane_id = %pane_id,
                                error = %e,
                                "load_pool task panicked after acquisition; \
                                 env vars and prompt_hint will NOT be injected into this pane"
                            );
                            Vec::new()
                        }
                    };
                    (leases, pool)
                }
                Err(e) => {
                    let failed_pane = pane_id.clone();
                    tokio::task::spawn_blocking(move || {
                        let _ = pane_lease::release_lease(&failed_pane);
                    })
                    .await
                    .ok();
                    // Same cleanup as the pane-CapacityError path: if the
                    // worktree was auto-created for this task, remove it
                    // so a failed resource acquisition doesn't leak
                    // branches and worktree dirs on disk.
                    cleanup_auto_worktree(was_explicit_worktree, &worktree, &task.client_cwd).await;
                    let mut w = writer.lock().await;
                    write_frame(
                        &mut *w,
                        &Response::Error {
                            message: format!(
                                "[error] resource acquisition failed: {e}; \
                                 check `amaebi resource list` and \
                                 ~/.amaebi/resources.toml"
                            ),
                        },
                    )
                    .await?;
                    drop(w);
                    release_launch_leases!();
                    return Ok(());
                }
            }
        };

        // Render resource env vars ONLY on the fresh-launch path.  On the
        // reuse path (`had_claude == true`) claude is already running and
        // its shell is gone, so `export SIM_PORT=...` can no longer be
        // applied — the leases are still held (for scheduling) but env
        // injection is a no-op.  The prompt_hint used to be prepended to
        // the task description here; that channel was replaced by the
        // per-worktree AGENTS.md (see `ensure_worktree_agents_md` below),
        // which claude loads at session start and survives `/compact`.
        let resource_env: Vec<(String, String)> = if had_claude {
            Vec::new()
        } else {
            resource_lease::render_env(&resource_leases, &resource_pool)
        };

        // Write AGENTS.md once per worktree so the resource constraint
        // survives `/compact` and supervision restarts.  Best-effort:
        // I/O failure is logged and launch continues.
        if !resource_leases.is_empty() {
            if let Err(e) = ensure_worktree_agents_md(&worktree, &resource_leases, &resource_pool) {
                tracing::warn!(
                    pane_id = %pane_id,
                    error = %e,
                    "failed to write AGENTS.md; LLM will not see the resource hint on restart"
                );
            }
        }

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
            // Reusing an existing claude session in the same worktree: inject
            // the new task description directly into the existing conversation.
            // No automatic `/compact` — resume is meant to *continue* where the
            // pane left off.  The user can run `/compact` manually if they
            // decide stale context needs pruning.
            vec![(description.clone(), auto_enter)]
        } else {
            // Fresh pane: launch claude with --dangerously-skip-permissions so
            // the autonomous session never blocks on an interactive approval
            // prompt, then inject the description as the opening message.
            // Resource env vars go on an `export` line ahead of the cd/claude
            // command — keeping them shell-level so both any `cd` into the
            // worktree and Claude itself inherit them.  (Injecting them after
            // claude starts is impossible: claude intercepts all keystrokes
            // and would treat `export FOO=bar` as a chat message.)
            let env_prefix = if resource_env.is_empty() {
                String::new()
            } else {
                let exports: Vec<String> = resource_env
                    .iter()
                    .map(|(k, v)| format!("export {}={}", k, shell_escape(v)))
                    .collect();
                format!("{} && ", exports.join(" && "))
            };
            let launch_cmd = if let Some(ref wt) = worktree {
                format!(
                    "{env_prefix}cd {} && claude --dangerously-skip-permissions",
                    shell_escape(wt)
                )
            } else {
                format!("{env_prefix}claude --dangerously-skip-permissions")
            };
            vec![(launch_cmd, true), (description.clone(), auto_enter)]
        };

        let send_pane = pane_id.clone();

        // Send key sequences to the pane.  Each step: send text literally
        // with `send-keys -l --`, then send Enter as a separate key press.
        //
        // Timing: 1 s pause before the first send (bash init), 5 s before
        // subsequent sends (e.g. claude startup), then
        // `tools::TEXT_RENDER_SLEEP_SECS` after each text injection before
        // pressing Enter (shared with `send_pane_keys` / `tmux_send_text`
        // so every paste path uses the same render-delay — 1 s was not
        // enough for multi-KB markdown and dropped Enters).  No prompt-
        // polling — simple fixed delays are robust against prompt
        // character variations (❯, >, $, etc.).
        let send_result = tokio::task::spawn_blocking(move || {
            for (idx, (keys, press_enter)) in key_sequence.iter().enumerate() {
                // Before the very first send: let the new pane's shell
                // initialise (.bashrc, prompt rendering, etc.).
                // Before subsequent sends (e.g. description after claude
                // launch): let the target process start up.
                // idx=0: give the new pane's shell a beat to init (bashrc, prompt).
                // idx>0: give the launched process (claude TUI first render) time
                //        to come up before we interact with it.
                let wait = if idx == 0 { 1 } else { 5 };
                std::thread::sleep(std::time::Duration::from_secs(wait));

                // For the description injection (idx > 0 on a fresh pane):
                // accept the Claude Code trust dialog that fires on
                // first-ever-seen worktrees ("Quick safety check: Is this a
                // project you created or one you trust?" — default answer is
                // "Yes, I trust this folder", selected by Enter).  Without
                // this, our description is typed INTO the menu, chosen as
                // the wrong option, or the prior `Escape` here selected
                // "No, exit" and killed claude outright — which is what
                // happened before this fix.
                //
                // Timing (on top of the 5 s claude-startup wait above):
                //   2 s — let the splash / trust dialog fully render
                //   Enter — accept "Yes" (or no-op if dialog isn't up:
                //           the Claude Code TUI ignores Enter on an empty
                //           prompt)
                //   2 s — let the dialog dismiss and the TUI settle
                //
                // Shorter values (0.5 s / 1 s) were tried and occasionally
                // caused the subsequent paste to land in a mid-render TUI
                // state where it was discarded — the 2 s pads are the
                // minimum that proved stable in manual testing.
                //
                // Only runs on fresh-pane launches (`!had_claude`); reuse
                // path (inject task into existing claude) skips because the
                // trust dialog was already accepted when the pane was first
                // launched.
                if idx > 0 && !had_claude {
                    std::thread::sleep(std::time::Duration::from_secs(2));
                    let _ = std::process::Command::new("tmux")
                        .args(["send-keys", "-t", &send_pane, "Enter"])
                        .output();
                    std::thread::sleep(std::time::Duration::from_secs(2));
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
                    // then press Enter.  Shared with `send_pane_keys` /
                    // `tmux_send_text` via `TEXT_RENDER_SLEEP_SECS` so every
                    // paste path uses the same render-delay; 1 s was not
                    // enough for multi-KB markdown and dropped Enters.
                    std::thread::sleep(std::time::Duration::from_secs(
                        crate::tools::TEXT_RENDER_SLEEP_SECS,
                    ));
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
            let pane_for_resources = failed_pane.clone();
            tokio::task::spawn_blocking(move || {
                if is_stale {
                    let _ = pane_lease::remove_pane(&failed_pane);
                } else {
                    let _ = pane_lease::release_lease(&failed_pane);
                }
            })
            .await
            .ok();
            // Always release resources, regardless of whether the pane was
            // removed or just freed — the holder pane is gone either way.
            let _ = resource_lease::release_all_for_pane(&pane_for_resources).await;
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
            drop(w);
            release_launch_leases!();
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

        // Persist the raw (user-typed) task description on the lease so that
        // `/claude --resume-pane <pid>` can reuse it without the user
        // retyping on subsequent rounds.  Uses `raw_desc` (not `description`)
        // to avoid storing the git-context preamble; that gets re-derived on
        // each launch.  Skipped when resume-pane reused the lease's existing
        // description (no new info to write).  Awaited (not fire-and-forget)
        // so an immediate follow-up `/claude --resume-pane <pid>` is
        // guaranteed to observe the persisted description instead of racing
        // a background write.  Failures are logged but do not block pane
        // assignment — the user still gets a working pane; resume-pane just
        // won't auto-recover the description.
        if resume_prefetched_desc.is_none() && !raw_desc.trim().is_empty() {
            let desc_pane = pane_id.clone();
            let desc_text = raw_desc.clone();
            let persist_pane = pane_id.clone();
            match tokio::task::spawn_blocking(move || {
                pane_lease::set_task_description(&desc_pane, &desc_text)
            })
            .await
            {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    tracing::warn!(
                        pane_id = %persist_pane,
                        error = %e,
                        "failed to persist task description on lease"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        pane_id = %persist_pane,
                        error = %e,
                        "task-description persistence task panicked"
                    );
                }
            }
        }

        // Resolve notebook tag.  Explicit `--tag <name>` wins verbatim;
        // Tag was resolved by the client via Request::GenerateTag
        // before this ClaudeLaunch arrived (or supplied by `--tag`).
        // It's used as pane lease holder / worktree dir / tmux title /
        // notebook key — all three are already wired up upstream.
        let mut w = writer.lock().await;
        write_frame(
            &mut *w,
            &Response::PaneAssigned {
                tag: task.tag,
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

/// Return `true` when `line` is a visual TUI chrome fragment from the
/// Claude Code terminal UI with no informational content — a horizontal
/// separator, the always-present bottom status bar, an empty prompt
/// cursor, or a pure box-drawing border segment.
///
/// Match criteria deliberately target **whole-line** chrome only so that
/// real output containing these glyphs (e.g. a log line with `│` as a
/// field separator) survives untouched.  Missing a chrome variant is
/// fine (minor noise), false positives would silently drop real
/// information the supervisor LLM needs — so every rule here is
/// conservative and structure-anchored.
fn is_tui_chrome_line(line: &str) -> bool {
    let t = line.trim();
    if t.is_empty() {
        return false; // real blank lines are meaningful in tmux captures
    }

    // Rules 1 and 4 both need to scan every char in the line; fold them
    // into a single pass so a long-but-unmatched line (common case: real
    // Claude Code output) pays for exactly one iteration rather than
    // four.  Correctness is unchanged: we still check the same
    // "all-dashes and >=10" (rule 1) and "all-border and
    // has-vertical" (rule 4) predicates, just with the char scan
    // amortised.
    //
    // Rule 1 (all-horizontal-rule divider): Claude Code renders these as
    // long runs of U+2500 / U+2501; a 10-char minimum avoids hitting a
    // real line that happens to start with a few dashes (e.g. markdown).
    //
    // Rule 4 (pure box-drawing border): every char is a box-drawing
    // glyph or whitespace, AND at least one vertical/corner glyph is
    // present.  The "must contain a vertical" requirement stops rule 4
    // from swallowing short horizontal runs (e.g. a 3- or 8-dash
    // markdown separator) that rule 1's 10-char minimum intentionally
    // let through — rule 1 is the sole authority for "all-dashes"
    // lines.  Vertical/corner glyphs are the real signature of a
    // rendered box frame.
    let mut char_count = 0usize;
    let mut all_horizontal = true;
    let mut all_border = true;
    let mut has_vertical_glyph = false;
    for c in t.chars() {
        char_count += 1;
        let is_horizontal = c == '─' || c == '━';
        let is_vertical = matches!(c, '│' | '╭' | '╮' | '╯' | '╰' | '├' | '┤' | '┬' | '┴');
        let is_border_glyph = is_horizontal || is_vertical || c == ' ' || c == '\t';
        if !is_horizontal {
            all_horizontal = false;
        }
        if !is_border_glyph {
            all_border = false;
        }
        if is_vertical {
            has_vertical_glyph = true;
        }
        // Early exit once neither composite predicate can still succeed.
        if !all_horizontal && !all_border {
            break;
        }
    }
    if all_horizontal && char_count >= 10 {
        return true; // rule 1
    }
    if all_border && has_vertical_glyph {
        return true; // rule 4
    }

    // 2. Bottom status bar.  Anchored on two substrings Claude Code
    //    always emits in its hint line, plus the "bypass permissions"
    //    banner visible when `--dangerously-skip-permissions` is active.
    if t.starts_with("⏵⏵ bypass permissions")
        || t.contains("esc to interrupt")
        || t.contains("ctrl+t to hide tasks")
    {
        return true;
    }

    // 3. Empty input prompt `❯` on its own.
    if t == "❯" {
        return true;
    }

    false
}

/// Remove whole-line TUI chrome from `raw`, preserving line order and
/// all blank lines.  Line-based filter: never rewrites content, only
/// drops fully-chrome lines.
fn strip_tui_chrome(raw: &str) -> String {
    raw.lines()
        .filter(|l| !is_tui_chrome_line(l))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Capture the last 200 lines of a tmux pane as plain text.
/// Returns an empty string on failure so supervision can continue.
///
/// 200-line window matches Claude Code's own default capture depth — enough
/// that a busy pane (tool outputs, build logs) still shows what the Claude
/// agent most recently did, not just the idle prompt that followed.
///
/// Post-capture, Claude Code's visual TUI chrome (divider rules, status
/// bar, empty prompt cursor, border characters) is stripped via
/// [`strip_tui_chrome`] so downstream supervision / idle-detection /
/// LLM-prompt consumers see a high signal-to-noise view of the pane.
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
            let raw = String::from_utf8_lossy(&output.stdout);
            strip_tui_chrome(&raw)
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
/// pauses to let the receiving TUI's paste buffer drain, then sends
/// Enter as a separate key press.  The pause exists because Claude Code's
/// TUI can swallow or defer the trailing Enter when it arrives before the
/// pasted text has been rendered into the input field — which manifests
/// as a STEER message appearing in the pane input but never submitting.
///
/// This helper and the `handle_claude_launch` literal-paste path share the
/// same render delay via [`crate::tools::TEXT_RENDER_SLEEP_SECS`], kept in
/// sync with the LLM-facing `tmux_send_text` tool; the previous 1 s value
/// was not enough for multi-KB markdown pastes and caused dropped Enters.
///
/// Returns `true` when BOTH the literal text injection and the trailing
/// Enter press reported success to tmux.  Any tmux failure (exit-code non-
/// zero, or spawn error) yields `false` so callers can accurately report
/// whether the keystrokes actually reached the pane — important for
/// supervision's `steer_dispatched` carry-over: claiming a STEER landed
/// when tmux silently rejected it would poison the next turn's prompt.
fn send_pane_keys(pane_id: &str, text: &str) -> bool {
    let text_ok = match std::process::Command::new("tmux")
        .args(["send-keys", "-t", pane_id, "-l", "--", text])
        .status()
    {
        Ok(s) if s.success() => true,
        Ok(s) => {
            tracing::warn!(pane_id, status = %s, "tmux send-keys (text) failed");
            false
        }
        Err(e) => {
            tracing::warn!(pane_id, error = %e, "failed to spawn tmux send-keys (text)");
            false
        }
    };
    // Shared with `tmux_send_text` so both paste paths use the same render-delay.
    std::thread::sleep(std::time::Duration::from_secs(
        crate::tools::TEXT_RENDER_SLEEP_SECS,
    ));
    let enter_ok = match std::process::Command::new("tmux")
        .args(["send-keys", "-t", pane_id, "Enter"])
        .status()
    {
        Ok(s) if s.success() => true,
        Ok(s) => {
            tracing::warn!(pane_id, status = %s, "tmux send-keys (Enter) failed");
            false
        }
        Err(e) => {
            tracing::warn!(pane_id, error = %e, "failed to spawn tmux send-keys (Enter)");
            false
        }
    };
    text_ok && enter_ok
}

/// Release the pane lease for each `SupervisionTarget` in this supervision
/// request, best-effort.  Iterates `panes` because a single `SupervisePanes`
/// request can cover multiple panes in parallel; this releases every one.
///
/// Called unconditionally at every [`handle_supervision`] exit point — including
/// timeout, DONE, interrupted, client-disconnect, and model-error paths — so a
/// pane is never stranded in `Busy` past the end of its supervision loop.
///
/// Without this, a pane stays `Busy` until `LEASE_TTL_SECS` (24 h) downgrades
/// it via [`pane_lease::PaneLease::effective_status`], during which a second
/// `/claude` invocation cannot reuse it and the scheduler has to spin up a
/// fresh pane instead of reusing the existing `claude` session via the
/// tier-1 reuse path (inject task into existing claude).
///
/// Release failures are logged and swallowed: they must never mask the
/// supervision loop's `Result`.  Each release runs on a blocking thread since
/// [`pane_lease::release_lease`] takes an `flock` lock.
async fn release_supervised_panes(panes: &[crate::ipc::SupervisionTarget]) {
    for target in panes {
        let pane_id = target.pane_id.clone();
        let outcome = tokio::task::spawn_blocking(move || pane_lease::release_lease(&pane_id))
            .await
            .map_err(anyhow::Error::from)
            .and_then(|r| r);
        if let Err(e) = outcome {
            tracing::warn!(
                pane_id = %target.pane_id,
                error = %e,
                "failed to release supervision pane lease"
            );
        }
        // Release any external resources (GPUs, simulators) that were
        // acquired alongside this pane.  Matched by pane_id on the resource
        // lease, so this is safe whether or not the caller actually
        // requested resources.
        match resource_lease::release_all_for_pane(&target.pane_id).await {
            Ok(names) if !names.is_empty() => {
                tracing::debug!(
                    pane_id = %target.pane_id,
                    released = ?names,
                    "released resource leases on supervision exit"
                );
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(
                    pane_id = %target.pane_id,
                    error = %e,
                    "failed to release resource leases on supervision exit"
                );
            }
        }
    }
}

/// Handle `Request::SupervisePanes`: run a Rust polling loop that captures pane
/// content, calls the LLM for analysis (no tools), and acts on the response.
///
/// The loop iterates with a 5-minute ceiling between turns (override with
/// `AMAEBI_SUPERVISION_INTERVAL_SECS`), but each iteration additionally waits
/// for the pane to go idle before snapshotting.  Each turn the LLM returns
/// exactly one of:
/// - `WAIT` — still working, check again
/// - `STEER: <pane_id>: <message>` — send a correction to the pane
/// - `DONE: <summary>` — task is complete; stream the summary and exit
///
/// The loop can also be interrupted by an `Interrupt` frame arriving on
/// `frame_rx`, or hit the hard wall-clock ceiling.  Default matches the
/// pane-, resource-, and task-notebook lease TTLs (all 24 h) so
/// supervision never outlives the leases it holds on; override with
/// `AMAEBI_SUPERVISION_TIMEOUT_SECS`.  A maximum of
/// `MAX_SUPERVISION_TOKENS` completion tokens is requested per turn
/// (see the constant in `handle_supervision_inner`) — sufficient for
/// the short WAIT/STEER/DONE responses.
///
/// Regardless of which exit point the inner loop takes (timeout, DONE,
/// interrupted, client disconnect, model error), [`release_supervised_panes`] runs
/// unconditionally afterward so a pane is never stranded `Busy` — unblocking
/// the tier-1 reuse path (inject task into existing claude) for the next
/// `/claude` task in the same worktree.
async fn handle_supervision(
    writer: &Arc<tokio::sync::Mutex<tokio::net::unix::OwnedWriteHalf>>,
    frame_rx: &mut tokio::sync::mpsc::Receiver<String>,
    panes: Vec<crate::ipc::SupervisionTarget>,
    model: String,
    state: &Arc<DaemonState>,
    session_id: Option<String>,
) -> Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};

    // Holder id used for task notebook leases: prefer the session UUID
    // (stable per supervision request), fall back to the first pane id.
    let holder = supervision_holder_id(&panes, session_id.as_deref());

    // Shared turn counter: the inner loop bumps this on every iteration;
    // the heartbeat task reads it so stalled supervision is diagnosable.
    let turn_counter = Arc::new(AtomicU64::new(0));

    // Spawn the heartbeat emitter.  Kept entirely in the wrapper so every
    // inner exit path (DONE, STEER, timeout, interrupt, model error,
    // client disconnect, inner Err) cancels it symmetrically — same
    // pattern as `release_supervised_panes`.
    let heartbeat_writer = Arc::clone(writer);
    let heartbeat_start = std::time::Instant::now();
    let heartbeat_turn = Arc::clone(&turn_counter);
    let heartbeat_task = tokio::spawn(async move {
        let mut ticker =
            tokio::time::interval(std::time::Duration::from_secs(HEARTBEAT_INTERVAL_SECS));
        // After a suspend (SIGSTOP, laptop sleep, heavy scheduler lag) the
        // interval would otherwise fire a backlog burst of heartbeats as it
        // "catches up" — which is both pointless and would swamp the client.
        // Skip behavior drops the backlog and resumes at the next boundary.
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // Skip the immediate fire `tokio::time::interval` emits on first tick
        // so the first heartbeat lands one interval into the session.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            let elapsed_secs = heartbeat_start.elapsed().as_secs();
            let turn = heartbeat_turn.load(Ordering::Relaxed);
            let mut w = heartbeat_writer.lock().await;
            if write_frame(&mut *w, &Response::Heartbeat { elapsed_secs, turn })
                .await
                .is_err()
            {
                // Writer closed — client disconnected; stop heartbeating.
                break;
            }
        }
    });

    let result = handle_supervision_inner(
        writer,
        frame_rx,
        &panes,
        model,
        state,
        session_id,
        &turn_counter,
    )
    .await;

    // Stop the heartbeat before we write the resume hint + Done so those
    // terminal frames are not interleaved with a ticking heartbeat.
    // `abort()` only requests cancellation; awaiting the JoinHandle ensures
    // the task has actually unwound (including releasing the writer mutex)
    // before we go on to emit the final Done frame.  The JoinError we
    // expect here is cancellation, not a panic — swallow it.
    heartbeat_task.abort();
    let _ = heartbeat_task.await;

    // Unified resume hint + Done.  Inner no longer writes
    // Response::Done itself; this wrapper writes the hint first (so
    // clients still reading frames see it) and then the terminal Done
    // frame.  Runs on every exit path (DONE, timeout, interrupt,
    // model error, client disconnect, inner Err).  Best-effort:
    // errors writing to an already-dying stream are swallowed.
    {
        let mut w = writer.lock().await;
        if panes.iter().any(|t| t.tag.is_some()) {
            let mut hint = String::from("\n[supervision] to resume any of these panes:\n");
            for t in panes.iter() {
                if let Some(tag) = t.tag.as_deref() {
                    hint.push_str(&format!(
                        "  pane {pid} (tag={tag})\n    continue task:  /claude --tag {tag}\n    reuse pane:     /claude --resume-pane {pid}\n",
                        pid = t.pane_id,
                    ));
                }
            }
            let _ = write_frame(&mut *w, &Response::Text { chunk: hint }).await;
        }
        let _ = write_frame(&mut *w, &Response::Done).await;
    }

    // Best-effort cleanup: must run regardless of inner result so panes
    // are not stranded Busy for up to LEASE_TTL_SECS (24 h).  Task
    // leases released by `holder` identity — same outer-wrapper pattern
    // as pane_lease / resource_lease, so every exit path (DONE, STEER,
    // timeout, interrupt, model error, client disconnect, inner Err) is
    // covered.
    release_supervised_panes(&panes).await;
    release_task_leases_for_holder(state, &holder).await;
    result
}

/// Stable identifier stored as `task_notes.content` on the `kind='lease'`
/// row so cleanup can find exactly the leases this supervision request
/// owns.
fn supervision_holder_id(
    panes: &[crate::ipc::SupervisionTarget],
    session_id: Option<&str>,
) -> String {
    if let Some(sid) = session_id {
        return format!("supervision:{sid}");
    }
    // Fallback: first pane id.  Uniqueness is guaranteed inside a single
    // live amaebi daemon because pane ids never repeat.
    let first = panes
        .first()
        .map(|p| p.pane_id.as_str())
        .unwrap_or("unknown");
    format!("supervision:{first}")
}

/// Release every task lease held by `holder`, ignoring errors.  Runs in
/// `spawn_blocking` because tasks.db is synchronous SQLite.
async fn release_task_leases_for_holder(state: &Arc<DaemonState>, holder: &str) {
    let state = Arc::clone(state);
    let holder = holder.to_string();
    let _ = tokio::task::spawn_blocking(move || {
        let guard = match state.tasks_db.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let Some(conn) = guard.as_ref() else {
            return; // tasks.db never opened (no --tag in this daemon's lifetime)
        };
        if let Err(e) = tasks::release_all_by_holder(conn, &holder) {
            tracing::warn!(holder, error = %e, "failed to release task leases");
        }
    })
    .await;
}

/// Render the task-notebook preamble for one supervision iteration.
///
/// Returns the empty string when no pane opted into the notebook, so
/// supervision prompts for non-`--tag` invocations are bit-identical
/// to before this PR.
///
/// On the first iteration (`is_first_turn = true`) this also writes a
/// `desc` row for every notebook pane whose `task_description` is
/// non-empty: this is how the CLI-supplied `<desc>` gets persisted for
/// later resume (`/claude --tag foo` with no desc reads back the most
/// recent row).
async fn build_notebook_context(
    state: &Arc<DaemonState>,
    panes: &[crate::ipc::SupervisionTarget],
    is_first_turn: bool,
) -> String {
    // Fast path: no notebook participation → empty preamble.
    let any_notebook = panes
        .iter()
        .any(|p| p.tag.is_some() && p.repo_dir.is_some());
    if !any_notebook {
        return String::new();
    }

    let state_cl = Arc::clone(state);
    let panes_cl: Vec<crate::ipc::SupervisionTarget> = panes.to_vec();
    let rendered = tokio::task::spawn_blocking(move || -> Result<String> {
        // Lazy-open the DB if no prior code path did.  Resume-pane launches
        // skip the tag-acquisition block in `handle_claude_launch`, so on
        // the first supervision turn for a resumed tagged pane the handle
        // may still be `None` — which would silently drop this pane's own
        // prior verdict/desc history from the notebook preamble.
        // `ensure_tasks_db` is idempotent.
        ensure_tasks_db(&state_cl)?;
        let guard = state_cl
            .tasks_db
            .lock()
            .map_err(|e| anyhow::anyhow!("tasks_db mutex poisoned: {e}"))?;
        let Some(conn) = guard.as_ref() else {
            return Ok(String::new());
        };

        // Group panes by (repo_dir, tag) so each tag contributes at most
        // one notebook section (and at most one first-turn desc write).
        // Multiple panes sharing a tag would otherwise inflate the
        // prompt with duplicate sections and insert duplicate desc rows.
        // Within a group we keep the first non-empty task_description
        // as the canonical desc to record; the remaining panes are for
        // lookup only.
        let mut grouped: std::collections::BTreeMap<(String, String), String> =
            std::collections::BTreeMap::new();
        for p in &panes_cl {
            let (Some(repo_dir), Some(tag)) = (p.repo_dir.as_deref(), p.tag.as_deref()) else {
                continue;
            };
            let key = (repo_dir.to_string(), tag.to_string());
            grouped
                .entry(key)
                .or_insert_with(|| p.task_description.clone());
        }

        let mut out = String::new();
        for ((repo_dir, tag), desc) in &grouped {
            // First-turn desc persistence: record the CLI-supplied desc
            // (if any) so later resumes can recover it.  Re-running with
            // the same desc will create a duplicate row; harmless, the
            // reader picks the most recent by timestamp anyway.
            if is_first_turn && !desc.trim().is_empty() {
                if let Err(e) = tasks::append_desc(conn, repo_dir, tag, desc) {
                    tracing::warn!(tag, error = %e, "failed to persist task desc");
                }
            }

            let latest = tasks::latest_desc(conn, repo_dir, tag)?
                .unwrap_or_else(|| "(none recorded)".to_string());
            let verdicts = tasks::recent_verdicts(conn, repo_dir, tag)?;

            out.push_str(&format!("=== Task notebook for tag '{tag}' ===\n"));
            out.push_str(&format!("Desc (most recent on record): {latest}\n"));
            if verdicts.is_empty() {
                out.push_str("No prior supervision verdicts for this tag.\n");
            } else {
                out.push_str(&format!(
                    "Recent verdicts from prior supervision sessions ({}, oldest first):\n",
                    verdicts.len()
                ));
                for v in &verdicts {
                    out.push_str(&format!("  - {v}\n"));
                }
            }
            out.push_str("=== End task notebook ===\n\n");
        }
        Ok(out)
    })
    .await
    .unwrap_or_else(|e| Err(anyhow::anyhow!("notebook context task panicked: {e}")))
    .unwrap_or_else(|e| {
        tracing::warn!(error = %e, "failed to build notebook context");
        String::new()
    });
    rendered
}

/// System prompt for the supervision LLM.  Extracted as a module-level
/// constant so tests can assert its contents without rebuilding the whole
/// supervision loop.
///
/// The wording is structured so the LLM's reading order matches the
/// priority we want: drift is named as the primary failure mode; STEER
/// is listed before WAIT/DONE; and the tie-break is explicitly "prefer
/// STEER" rather than "default to WAIT" (a missed STEER costs hours, a
/// stray STEER costs one keystroke).
const SUPERVISION_SYSTEM_PROMPT: &str = concat!(
    "You are supervising a Claude Code session executing a specific task in a ",
    "tmux pane.  Your PRIMARY duty is to keep Claude on the task as stated.  ",
    "Drift — switching approach, using a different resource, skipping a requirement, ",
    "subtly redefining the goal — is the failure mode you must catch.  When drift ",
    "appears, STEER immediately; do not wait for another turn to \"see if it self-corrects\".\n",
    "\n",
    "Each turn you see: the original task description (pinned at the top), any hard ",
    "constraints, the last few verdicts you issued, and the current pane contents.  ",
    "You respond with EXACTLY ONE of:\n",
    "\n",
    "STEER: <pane_id>: <message to send>\n",
    "  Use STEER when ANY of the following holds:\n",
    "  - Claude is at an idle prompt and asked a question or offered options — answer.\n",
    "  - Claude's current action contradicts a hard constraint (wrong resource, wrong ",
    "    container, wrong branch, forbidden command).  Name the specific constraint ",
    "    in the STEER message.\n",
    "  - Claude drifted from the task: switched approach without checking in, skipped ",
    "    a requirement, reinterpreted the goal, or is working on a tangent.\n",
    "  - Claude reports partial completion and the task description is not fully done.\n",
    "  - Claude is stuck on an error and a concrete hint will unblock it.\n",
    "  When in doubt between STEER and WAIT, prefer STEER — a wrong STEER costs one ",
    "  keystroke, a missed STEER costs hours of wrong work.\n",
    "\n",
    "WAIT: <one sentence — what is Claude currently doing?>\n",
    "  Use WAIT only when Claude is visibly, measurably making progress on the ORIGINAL ",
    "  task, respecting hard constraints, and has not asked a question.  Long builds, ",
    "  sim runs, and test executions are normal — WAIT is correct there.\n",
    "\n",
    "DONE: <paragraph summary of what was accomplished>\n",
    "  The task is FULLY complete.  Require ALL of:\n",
    "  1. Pane shows an explicit completion signal — a finished report, passing tests,\n",
    "     a merged PR URL, 'done', '✓', 'all tests passed', or similar.\n",
    "  2. That completion directly covers the task description given at session start\n",
    "     (not a sub-step, not unrelated output).\n",
    "  3. Claude is no longer working (idle prompt or returned to shell).\n",
    "  An idle prompt alone is NOT sufficient — Claude may be waiting for user input ",
    "  (prefer STEER) or may have been interrupted (prefer WAIT and re-check).\n",
    "\n",
    "Your response must start with STEER:, WAIT:, or DONE: — nothing else before it.",
);

/// Pure helper that formats the user message fed to the supervision LLM
/// every turn.  Extracted from [`handle_supervision_inner`] so tests can
/// exercise the prompt layout without standing up a full daemon loop.
///
/// Layout (top-to-bottom):
///   1. `TASK — keep Claude focused on this:` with one line per pane.
///      Pinned at byte 0 because LLM attention otherwise drifts to the
///      long pane dumps.  The task is deliberately repeated inside each
///      pane header further down.
///   2. `HARD CONSTRAINTS` block (may be empty) from the pane's resource
///      leases.
///   3. Notebook context carried over from prior supervision sessions
///      (may be empty).
///   4. Recent verdict history (newest last) so the LLM can see whether
///      its own prior STEERs landed.
///   5. The last STEER in full, if one has been issued.
///   6. The current pane snapshots.
#[allow(clippy::too_many_arguments)] // the pieces are simple values from the loop;
                                     // packaging them into a struct would just shift the verbosity one level without
                                     // improving call-site readability.
fn build_supervision_user_content(
    task_lines: &[(String, String)],
    hard_constraints: &str,
    notebook_context: &str,
    verdict_history: &std::collections::VecDeque<String>,
    last_steer_full: Option<&str>,
    pane_snapshots: &str,
    turn: u64,
    elapsed_mins: u64,
) -> String {
    let mut task_section = String::from("TASK — keep Claude focused on this:\n");
    for (pane_id, desc) in task_lines {
        // Task descriptions may be pasted multi-line input.  Flatten to a
        // single line (escape newlines + CRs) so one pane stays on one
        // line and the downstream section headers keep their column-0
        // alignment.  See the verdict-history escape for the same reason.
        let desc = desc
            .replace("\r\n", "\\n")
            .replace('\n', "\\n")
            .replace('\r', "\\r");
        task_section.push_str(&format!("  pane {pane_id}: {desc}\n"));
    }

    let verdict_history_block = if verdict_history.is_empty() {
        "(none yet — this is the first check)".to_string()
    } else {
        let mut s = String::new();
        for (i, v) in verdict_history.iter().enumerate() {
            let age = verdict_history.len() - i;
            let label = if age == 1 {
                "last".to_string()
            } else {
                format!("{age} turns ago")
            };
            s.push_str(&format!("  [{label}] {}\n", v.trim()));
        }
        s
    };

    let last_steer_block = match last_steer_full {
        // Newlines in the verdict were already escaped to `\n` upstream
        // (see `verdict_single_line` in `handle_supervision_inner`) so the
        // block stays one visible line — hence "escaped newlines" rather
        // than "full text" in the header.
        Some(s) => format!(
            "Most recent STEER message (escaped newlines):\n  {}\n\n",
            s.trim()
        ),
        None => String::new(),
    };

    // Assemble without source-indentation so every section header (TASK,
    // HARD CONSTRAINTS, notebook context, verdict list, STEER carry-over,
    // snapshot header) lands at column 0 in the LLM-facing output.
    let mut out = task_section;
    out.push('\n');
    out.push_str(hard_constraints);
    out.push_str(notebook_context);
    out.push_str("Your recent verdicts this session (newest last):\n");
    out.push_str(&verdict_history_block);
    out.push('\n');
    out.push_str(&last_steer_block);
    out.push_str(&format!(
        "Current pane snapshots (check #{turn}, elapsed {elapsed_mins}m):\n\n"
    ));
    out.push_str(pane_snapshots);
    out
}

/// Render the hard constraints section for the supervision LLM.  For
/// every supervised pane, if the pane currently holds one or more
/// resource leases, render their `prompt_hint` under a section marked
/// "HARD CONSTRAINTS".  Empty string when no pane has leases — the
/// caller string-concats unconditionally.
///
/// Constraints are scoped per-pane so a supervised batch of two panes
/// with different resources still sees each pane's own bindings.
async fn render_hard_constraints(panes: &[crate::ipc::SupervisionTarget]) -> String {
    let load_result = tokio::task::spawn_blocking(|| -> Result<_> {
        let state = resource_lease::read_state()?;
        let pool = resource_lease::load_pool()?;
        Ok((state, pool))
    })
    .await;
    // A load failure on either side (missing / malformed TOML, corrupted
    // state JSON, mutex poisoning) is a **misconfiguration** we do not want
    // supervision to silently ignore — the whole point of this section is
    // to anchor hard constraints.  Surface a visible banner so the LLM
    // knows the constraint set is unavailable, and log the error for ops.
    let (state_map, pool) = match load_result {
        Ok(Ok(pair)) => pair,
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "supervision: failed to load hard constraints");
            return "HARD CONSTRAINTS — failed to load; see daemon logs.\n\n".to_string();
        }
        Err(e) => {
            tracing::warn!(error = %e, "supervision: hard-constraints loader task panicked");
            return "HARD CONSTRAINTS — failed to load; see daemon logs.\n\n".to_string();
        }
    };
    let mut out = String::new();
    for target in panes {
        // Collect leases whose `pane_id` matches this supervised pane AND
        // are still effectively Busy.  `effective_status()` maps a Busy
        // lease whose heartbeat is older than `LEASE_TTL_SECS` back to
        // Idle — rendering such stale leases as active "HARD CONSTRAINTS"
        // would anchor the LLM to a resource the task no longer actually
        // holds, driving spurious STEERs.
        let mut leases: Vec<_> = state_map
            .values()
            .filter(|l| l.pane_id.as_deref() == Some(&target.pane_id))
            .filter(|l| l.effective_status() == resource_lease::ResourceStatus::Busy)
            .cloned()
            .collect();
        if leases.is_empty() {
            continue;
        }
        // HashMap iteration order is nondeterministic.  Sort by (name, class)
        // so a pane with multiple resources renders the same block every
        // turn; otherwise the LLM sees prompt churn for what is logically
        // the same constraint set (see `dashboard.rs` for the same pattern).
        leases.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.class.cmp(&b.class)));
        let hint = resource_lease::render_prompt_hint(&leases, &pool);
        if hint.trim().is_empty() {
            continue;
        }
        if out.is_empty() {
            out.push_str("HARD CONSTRAINTS — violations require an immediate STEER:\n");
        }
        out.push_str(&format!(
            "  pane {}:\n{}\n",
            target.pane_id,
            hint.lines()
                .map(|l| format!("    {l}"))
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }
    if !out.is_empty() {
        out.push('\n');
    }
    out
}

/// How often `handle_supervision` emits a [`Response::Heartbeat`] frame on
/// the supervision socket.  Chosen so a stuck daemon trips the client's
/// 30-minute watchdog within ~30 minutes even on a pane that would
/// otherwise produce no other frames (WAIT/STEER/DONE headers only land
/// every `AMAEBI_SUPERVISION_INTERVAL_SECS`, defaulting to 5 minutes).
const HEARTBEAT_INTERVAL_SECS: u64 = 10 * 60;

/// Inner body of [`handle_supervision`]; see that function's doc for context.
///
/// Takes `panes` by reference so the outer wrapper retains ownership and can
/// pass it to [`release_supervised_panes`] after this returns.  `turn_counter`
/// is incremented on every iteration so the wrapper's heartbeat task can
/// report the current turn number.
async fn handle_supervision_inner(
    writer: &Arc<tokio::sync::Mutex<tokio::net::unix::OwnedWriteHalf>>,
    frame_rx: &mut tokio::sync::mpsc::Receiver<String>,
    panes: &[crate::ipc::SupervisionTarget],
    model: String,
    state: &Arc<DaemonState>,
    session_id: Option<String>,
    turn_counter: &Arc<std::sync::atomic::AtomicU64>,
) -> Result<()> {
    // Notebook leases are acquired in `handle_claude_launch` BEFORE any
    // pane/worktree/claude work so a tag-conflict rejection never
    // leaves a real running session behind.  By the time supervision
    // starts, the lease rows we'll use are already ours under the
    // same holder id derived in the wrapper; cleanup still flows
    // through `release_all_by_holder` there.

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

    // Hard wall-clock limit before supervision gives up. Default matches the
    // pane/resource lease TTL so supervision never outlives the leases it
    // holds on; override with `AMAEBI_SUPERVISION_TIMEOUT_SECS`.  Pulling
    // from the constant rather than a literal prevents the three TTL values
    // from drifting out of sync on future bumps.
    let max_duration = std::time::Duration::from_secs(
        std::env::var("AMAEBI_SUPERVISION_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(crate::pane_lease::LEASE_TTL_SECS),
    );

    const MAX_SUPERVISION_TOKENS: usize = 1024;

    let supervision_start = std::time::Instant::now();
    let deadline = supervision_start + max_duration;
    let mut turn: u64 = 0;

    // Verdicts from this supervision session, newest-last.  Bounded so
    // we never leak memory on 10-h runs; the LLM sees up to N most
    // recent entries each turn.
    const VERDICT_HISTORY_LEN: usize = 5;
    let mut verdict_history: std::collections::VecDeque<String> =
        std::collections::VecDeque::with_capacity(VERDICT_HISTORY_LEN);
    // Keep the most recent STEER in full so the LLM can judge whether
    // Claude acted on it.  None until the first STEER is emitted.
    let mut last_steer_full: Option<String> = None;

    // Load skill files (SOUL.md, AGENTS.md, GPU_KERNEL.md) once and reuse
    // across all supervision turns so the LLM has project context for
    // higher-quality STEER decisions.
    let skill_msgs = load_skill_messages().await;

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
            // wrapper writes Response::Done after the resume hint
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
                // wrapper writes Response::Done after the resume hint
                return Ok(());
            }
        }

        turn += 1;
        turn_counter.store(turn, std::sync::atomic::Ordering::Relaxed);
        let elapsed_mins = supervision_start.elapsed().as_secs() / 60;

        // --- Capture pane snapshots (full for LLM, tail for display) ---
        struct PaneSnapshot {
            pane_id: String,
            task_description: String,
            full_content: String, // sent to LLM
            tail: String,         // last 8 lines shown to user
        }

        let mut snapshots: Vec<PaneSnapshot> = Vec::new();
        for target in panes {
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
        // which captures the last 200 lines of the pane and then strips TUI
        // chrome (dividers, status bar, empty `❯` prompt, bordered-panel rows).
        // This bounds the amount of data sent to the LLM, but those lines
        // could still contain secrets (e.g. env vars printed by a build
        // script).  A future improvement could add redaction of common
        // secret patterns.
        let mut pane_snapshots = String::new();
        for snap in &snapshots {
            pane_snapshots.push_str(&format!(
                "=== Pane {} — task: {} ===\n{}\n",
                snap.pane_id, snap.task_description, snap.full_content
            ));
        }

        // Render hard constraints from the pane's resource leases.  Each
        // pane's own leases are pulled from resource-state.json; rendering
        // reuses `resource_lease::render_prompt_hint` so the wording is
        // identical to what lands in AGENTS.md.
        let hard_constraints = render_hard_constraints(panes).await;

        // Task notebook context (opt-in: only panes with tag + repo_dir).
        // Read-only: per-turn fetch of recent verdicts and the tag's latest
        // stored desc.  Rendered into a dedicated prompt section that is
        // clearly labelled "from prior supervision sessions" so the LLM does
        // not confuse it with in-session continuity (`verdict_history`).
        // First-turn side effect: write `task_description` as a `desc` row
        // so resumes without a CLI desc can find it.
        let notebook_context = build_notebook_context(state, panes, turn == 1).await;

        let task_lines: Vec<(String, String)> = snapshots
            .iter()
            .map(|s| (s.pane_id.clone(), s.task_description.clone()))
            .collect();

        let user_content = build_supervision_user_content(
            &task_lines,
            &hard_constraints,
            &notebook_context,
            &verdict_history,
            last_steer_full.as_deref(),
            &pane_snapshots,
            turn,
            elapsed_mins,
        );

        let mut messages = vec![Message::system(SUPERVISION_SYSTEM_PROMPT)];
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
                    // wrapper writes Response::Done after the resume hint
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
        // `steer_dispatched` is true only when the parsed STEER was valid
        // and actually sent into a pane — so the next turn's carry-over
        // (`last_steer_full`) reflects a message claude really saw.
        // Malformed / unknown-pane STEERs leave `last_steer_full` alone.
        let mut steer_dispatched = false;
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
            // wrapper writes Response::Done after the resume hint
            return Ok(());
        } else if let Some(rest) = trimmed.strip_prefix("STEER:") {
            if let Some((pane_id_raw, message)) = rest.trim().split_once(':') {
                let pane_id = pane_id_raw.trim().to_owned();
                let message = message.trim().to_owned();
                let is_valid_pane = panes.iter().any(|t| t.pane_id == pane_id);
                if !pane_id.is_empty() && !message.is_empty() && is_valid_pane {
                    let pid = pane_id.clone();
                    let msg = message.clone();
                    // `send_pane_keys` returns `true` only when both the
                    // text injection and the trailing Enter succeeded at
                    // tmux.  We also guard against JoinError (panic /
                    // cancel) by pattern-matching the JoinHandle result.
                    // Either failure leaves `steer_dispatched = false` so
                    // next turn does not claim claude received a message
                    // that never actually arrived.
                    match tokio::task::spawn_blocking(move || send_pane_keys(&pid, &msg)).await {
                        Ok(true) => {
                            steer_dispatched = true;
                        }
                        Ok(false) => {
                            tracing::warn!(
                                pane_id = %pane_id,
                                "tmux send-keys reported failure; treating STEER as undelivered"
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                pane_id = %pane_id,
                                error = %e,
                                "send_pane_keys task failed; treating STEER as undelivered"
                            );
                        }
                    }
                    // Keep the `STEER: <pane>: <msg>` shape exactly as the
                    // parser (and system prompt) expect, so when this line
                    // is echoed back via `verdict_history` / `last_steer_full`
                    // next turn the LLM is primed with the same grammar it
                    // must emit.  A `STEER %X: ...` (no colon after STEER)
                    // would be mis-parsed as WAIT on the next round-trip.
                    format!("  → STEER: {pane_id}: {message}\n")
                } else if !pane_id.is_empty() && !message.is_empty() && !is_valid_pane {
                    format!("  → STEER: {pane_id} (unknown pane, ignored)\n")
                } else {
                    "  → STEER: (malformed response)\n".to_string()
                }
            } else {
                "  → STEER: (malformed response)\n".to_string()
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
        let verdict_compact = verdict_line
            .trim_start_matches("  → ")
            .trim_end_matches('\n')
            .to_string();
        // The verdict_history rendering uses one line per entry; an
        // embedded `\n` (possible when a STEER message spans multiple
        // lines — parser doesn't reject them) would break the `[last] …`
        // / `[N turns ago] …` table shape and push downstream section
        // headers off column 0.  Escape newlines to a visible `\n` so
        // the LLM still sees the content but the layout is preserved.
        let verdict_single_line = verdict_compact.replace('\n', "\\n");
        if verdict_history.len() >= VERDICT_HISTORY_LEN {
            verdict_history.pop_front();
        }
        verdict_history.push_back(verdict_single_line.clone());
        // Only stash the full text when the STEER was actually dispatched.
        // `steer_dispatched` stays false for "unknown pane, ignored" and
        // "malformed response" variants so the next turn's prompt does not
        // claim claude received something it never did.  Use the escaped
        // single-line form for the same layout-protection reason.
        if steer_dispatched {
            last_steer_full = Some(verdict_single_line);
        }

        // Persist verdict to task notebook (best-effort, once per unique
        // `(repo_dir, tag)`).  Deduped because multiple panes sharing a
        // tag (user `--tag foo` for a multi-task launch) would otherwise
        // append duplicate rows for a single supervision turn.  Failures
        // (panic OR SQLite Err) are logged but must not break the loop —
        // notebook is auxiliary to the live verdict decision.
        let mut verdict_targets: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::new();
        for target in panes.iter() {
            if let (Some(repo_dir), Some(tag)) = (target.repo_dir.as_deref(), target.tag.as_deref())
            {
                verdict_targets.insert((repo_dir.to_string(), tag.to_string()));
            }
        }
        for (repo_dir, tag) in verdict_targets {
            let state_cl = Arc::clone(state);
            let verdict = verdict_compact.clone();
            let repo_dir_log = repo_dir.clone();
            let tag_log = tag.clone();
            match tokio::task::spawn_blocking(move || -> Result<()> {
                // Lazy-open the DB if no prior code path did.  Resume-pane
                // launches skip the tag-acquisition block in
                // `handle_claude_launch` that normally opens it, so when a
                // resumed pane produces the first verdict the handle may
                // still be `None`.  `ensure_tasks_db` is idempotent.
                ensure_tasks_db(&state_cl)?;
                let guard = state_cl
                    .tasks_db
                    .lock()
                    .map_err(|e| anyhow::anyhow!("tasks_db mutex poisoned: {e}"))?;
                let conn = guard
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("tasks_db not initialised"))?;
                tasks::append_verdict(conn, &repo_dir, &tag, &verdict)
            })
            .await
            {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::warn!(
                    error = %e,
                    repo_dir = %repo_dir_log,
                    tag = %tag_log,
                    "failed to persist verdict"
                ),
                Err(e) => tracing::warn!(
                    error = %e,
                    repo_dir = %repo_dir_log,
                    tag = %tag_log,
                    "verdict persist task panicked"
                ),
            }
        }

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

/// Create a git worktree at `~/.amaebi/worktrees/<repo-name>/<tag>-<uuid8>`
/// on a new branch named `<tag>-<uuid8>`.
///
/// Every parallel `/claude` task needs its own worktree so that concurrent
/// Claude sessions editing the same repository do not trample each other's
/// in-progress changes.  Worktrees are stored under `~/.amaebi/worktrees/`
/// (alongside other amaebi state) rather than inside the repository directory,
/// which avoids polluting the project tree and requires no `.gitignore` entry.
/// A per-repo subdirectory (the basename of the git root) prevents collisions
/// across different repositories; the `-<uuid8>` suffix makes each
/// worktree/branch unique for a given `tag` across runs.
///
/// `client_cwd` is the working directory of the invoking client.  Git is run
/// with `-C <client_cwd>` so the correct repository is targeted even when the
/// daemon was started from a different directory.
///
/// Returns the absolute path of the newly created worktree, or an error if:
/// - `tag` contains unsafe characters (path separators, `..`), or
/// - the client's directory is not inside a git repository, or
/// - `git worktree add` fails (e.g. branch name already exists).
///
/// All git commands are synchronous; call this from `spawn_blocking`.
fn create_task_worktree(
    tag: &str,
    client_cwd: Option<&str>,
    start_branch: Option<&str>,
) -> anyhow::Result<std::path::PathBuf> {
    use std::path::PathBuf;

    // Sanitize tag: allow only characters that are safe as both a
    // filesystem path component and a git branch name.
    if tag.is_empty()
        || tag == ".."
        || tag.contains('/')
        || tag.contains('\\')
        || tag.contains("..")
        || !tag
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        anyhow::bail!(
            "tag {:?} contains unsafe characters; \
             only ASCII alphanumerics, '-', '_', and '.' are allowed",
            tag
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

    // Place worktrees under ~/.amaebi/worktrees/<repo-hash>/<tag>-<uuid8>.
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
    let unique_name = format!("{tag}-{short_id}");

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

    // Create the worktree on a new branch named <tag>-<uuid8>.
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

const AGENTS_MD_BEGIN: &str = "<!-- amaebi-managed: begin -->";
const AGENTS_MD_END: &str = "<!-- amaebi-managed: end -->";

/// Ensure `<worktree>/AGENTS.md` contains an amaebi-managed block that
/// tells claude which resources this worktree is pinned to.  Called on
/// every launch (fresh and resume); the "block already present" branch
/// makes resume a no-op so we never clobber a user's edits.
///
/// - File missing: create it with a short header and the managed block.
/// - File exists without the begin marker: append the managed block.
/// - File exists with the begin marker: leave it alone (stale content
///   is intentionally kept so resume doesn't rewrite).
///
/// No-op when `worktree` is `None` or `leases` is empty.  Best-effort
/// for the caller: I/O failures surface as `Err` and should be logged
/// at warn-level, not treated as a hard launch failure.
fn ensure_worktree_agents_md(
    worktree: &Option<String>,
    leases: &[resource_lease::ResourceLease],
    pool: &[resource_lease::ResourceDef],
) -> std::io::Result<()> {
    let Some(wt) = worktree.as_ref() else {
        return Ok(());
    };
    if leases.is_empty() {
        return Ok(());
    }
    let path = std::path::Path::new(wt).join("AGENTS.md");
    let hint = resource_lease::render_prompt_hint(leases, pool);
    // An empty hint means every lease either has no `prompt_hint` set in
    // `resources.toml`, or its pool entry disappeared (stale state vs. a
    // re-edited TOML).  Either way, writing a block with empty body would
    // suggest "no constraints assigned" to claude, which is misleading —
    // skip instead.  A later launch with a non-empty hint will still
    // create the file.
    if hint.trim().is_empty() {
        return Ok(());
    }
    let block = format!("{AGENTS_MD_BEGIN}\n## Assigned resources\n\n{hint}\n{AGENTS_MD_END}\n");

    match std::fs::read_to_string(&path) {
        Ok(existing) => {
            if existing.contains(AGENTS_MD_BEGIN) {
                return Ok(());
            }
            // Append (preserving user content).  Ensure exactly one blank
            // line separator between existing content and our block.
            let mut out = existing;
            if !out.ends_with('\n') {
                out.push('\n');
            }
            out.push('\n');
            out.push_str(&block);
            std::fs::write(&path, out)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let header = "# Project rules\n\n\
                This file is loaded by Claude Code at session start and acts as a\n\
                persistent project rule set. Edits by you are preserved; the block\n\
                between the `amaebi-managed` markers is auto-generated by `amaebi`\n\
                when a task first launches with `--resource`.\n\n";
            std::fs::write(&path, format!("{header}{block}"))
        }
        Err(e) => Err(e),
    }
}

/// Remove an auto-created git worktree and its branch on task-setup failure.
///
/// Centralises the rollback logic used by every error path that fires
/// after `create_task_worktree` has succeeded but before the pane is
/// actually populated — pane-acquisition failure, resource-acquisition
/// failure, session-ID failure, etc.  Skipped when the worktree was
/// user-supplied (`was_explicit_worktree = true`) so we never touch a
/// worktree the caller owns.
///
/// Best-effort: git failures are swallowed; the caller has already
/// surfaced the primary error and any leftover worktree can be removed
/// manually with `git worktree remove --force`.
async fn cleanup_auto_worktree(
    was_explicit_worktree: bool,
    worktree: &Option<String>,
    client_cwd: &Option<String>,
) {
    if was_explicit_worktree {
        return;
    }
    let Some(wt) = worktree.clone() else {
        return;
    };
    let cleanup_cwd = client_cwd.clone();
    tokio::task::spawn_blocking(move || {
        let branch = std::path::Path::new(&wt)
            .file_name()
            .and_then(|n| n.to_str())
            .map(str::to_string);
        let mut rm_cmd = std::process::Command::new("git");
        if let Some(ref cwd) = cleanup_cwd {
            rm_cmd.args(["-C", cwd.as_str()]);
        }
        let removed = rm_cmd
            .args(["worktree", "remove", "--force", &wt])
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
pub(crate) fn git_output(cwd: Option<&str>, args: &[&str]) -> String {
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
/// Maximum Unicode scalars kept in the `shell_command` / `tmux_send_text`
/// preview returned by [`summarise_tool_detail`] before appending `…`.
/// Kept at module scope so regression tests can reference it without
/// duplicating the `80` literal.
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

/// Bound a preview string to [`SHELL_DETAIL_MAX_CHARS`] Unicode scalars,
/// appending `…` when truncation occurred.
///
/// Walks `char_indices` once to find the byte offset of the `(MAX+1)`-th
/// scalar (guaranteed to be a char boundary), so it is safe against
/// multi-byte UTF-8 — a raw `&s[..80]` byte slice would panic when byte 80
/// lands inside a multi-byte sequence.
fn truncate_detail_preview(s: &str) -> String {
    match s.char_indices().nth(SHELL_DETAIL_MAX_CHARS) {
        Some((byte_idx, _)) => format!("{}…", &s[..byte_idx]),
        None => s.to_string(),
    }
}

/// Build a short human-readable detail string for a tool call, used in
/// `Response::ToolUse` frames so the client can render something like
/// `shell_command: ls -la`.
///
/// Must operate on Unicode scalar boundaries: shell commands routinely contain
/// non-ASCII text (Chinese comments, em-dashes) and a raw byte slice like
/// `&s[..80]` panics when the truncation index falls inside a multi-byte UTF-8
/// sequence.  The `shell_command` and `tmux_send_text` branches truncate at
/// 80 chars, appending `…` when the original was longer (and `tmux_send_text`
/// additionally flattens newlines so multi-line pastes don't break the CLI's
/// single-line renderer); other tools return their primary argument verbatim.
fn summarise_tool_detail(tool_name: &str, args: &serde_json::Value) -> String {
    match tool_name {
        "shell_command" => args
            .get("command")
            .and_then(|v| v.as_str())
            .map(truncate_detail_preview)
            .unwrap_or_default(),
        "read_file" | "edit_file" => args
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        "tmux_send_text" => args
            .get("text")
            .and_then(|v| v.as_str())
            .map(|s| {
                // Multi-KB markdown pastes are the common case for
                // tmux_send_text; injecting the whole body into the tool
                // detail spams logs/client output and the raw `\n` bytes
                // break the CLI's single-line tool-use renderer.  Flatten
                // whitespace first, then bound the preview to the same
                // char limit as shell_command.
                let flat: String = s
                    .chars()
                    .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
                    .collect();
                truncate_detail_preview(&flat)
            })
            .unwrap_or_default(),
        "tmux_send_key" => args
            .get("key")
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
            let dir = resolve_session_dir(&sid).await;
            let result = tokio::task::spawn_blocking(move || {
                let conn = db.lock().unwrap_or_else(|p| p.into_inner());
                let tx = conn
                    .unchecked_transaction()
                    .context("compact_session: begin transaction")?;
                memory_db::store_session_summary(&conn, &sid, &summary, &ts, &dir)?;
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
        let dir = resolve_session_dir(sid).await;
        let archive_result = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = db.lock().unwrap_or_else(|p| p.into_inner());
            // Always persist the summary (even when nothing is archived) so a
            // later resume rebuilds from the compacted state rather than the
            // raw history — the in-memory splice has already happened, so
            // skipping the DB write here would desync memory vs. persistence.
            let tx = conn
                .unchecked_transaction()
                .context("compact_in_loop: begin transaction")?;
            memory_db::store_session_summary(&conn, &sid_owned, &summary_text, &ts, &dir)?;
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

/// Public-to-crate wrapper so `task_tagger` can call the model without
/// being a friend of this entire module.  Same semantics as
/// `invoke_model` — no tools, caller-supplied `max_tokens`.  Intended
/// only for short housekeeping calls (tag generation); production
/// chat/supervision should stay inside `invoke_model` directly.
pub(crate) async fn invoke_model_for_tagger<W>(
    state: &DaemonState,
    model: &str,
    messages: &[Message],
    max_completion_tokens: usize,
    writer: &mut W,
) -> Result<copilot::CopilotResponse>
where
    W: AsyncWriteExt + Unpin,
{
    invoke_model(state, model, messages, &[], max_completion_tokens, writer).await
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

    // Bedrock `malformed_model_output` / `malformed_tool_use` stopReasons
    // (since bedrockruntime 1.119, 2025-12-02) signal a transient per-turn
    // failure inside the model, not a protocol or quota error.  We retry
    // each such turn once with identical `messages`; on the second strike
    // we fall through to the `Other`-style termination path.  Bounded to
    // one to prevent loop storms if the model is in a stuck pattern.
    const MAX_MALFORMED_RETRIES: usize = 1;
    let mut malformed_retries: usize = 0;

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
            FinishReason::Malformed => "malformed",
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

                // Classify the batch in a single pass so we don't re-parse
                // JSON twice (once for the fast-path decision, once for the
                // diagnostic probe).
                let mut spawn_count: usize = 0;
                let mut all_parallel_true = true;
                let mut any_bad_json = false;
                for tc in tool_calls_snapshot.iter() {
                    if tc.name == "spawn_agent" {
                        spawn_count += 1;
                        match tc.parse_args() {
                            Ok(v) => {
                                if v["parallel"].as_bool() != Some(true) {
                                    all_parallel_true = false;
                                }
                            }
                            Err(_) => {
                                any_bad_json = true;
                                all_parallel_true = false;
                            }
                        }
                    }
                }

                // Fast-path: every call is spawn_agent AND each one opted in
                // with `parallel: true`.  Default is sequential.
                let all_spawn_agent = tool_calls_snapshot.len() > 1
                    && spawn_count == tool_calls_snapshot.len()
                    && all_parallel_true;

                // Diagnostic probe: WARN only when the fan-out *could* have
                // been concurrent but wasn't (>= 2 spawn_agent calls in the
                // batch without the fast-path).  Mixed batches with a single
                // spawn_agent never qualify for concurrency, so they are not
                // a "missed parallelism" case and do not trigger WARN.
                if tool_calls_snapshot.len() > 1 && !all_spawn_agent {
                    if spawn_count > 1 {
                        let names: Vec<&str> = tool_calls_snapshot
                            .iter()
                            .map(|tc| tc.name.as_str())
                            .collect();
                        let msg = if any_bad_json {
                            "spawn_agent batch contains calls with unparseable arguments; \
                             fix the JSON so each call can be dispatched (parallel fan-out \
                             requires parallel: true as well)"
                        } else if spawn_count == tool_calls_snapshot.len() {
                            "sequential spawn_agent batch — set parallel: true on every call \
                             to run them concurrently"
                        } else {
                            "mixed-tool batch including spawn_agent — \
                             concurrency only applies to pure spawn_agent batches; \
                             isolate spawn_agent calls into their own batch and set parallel: true"
                        };
                        tracing::warn!(
                            session_id = ?session_id,
                            tools = ?names,
                            spawn_agent_count = spawn_count,
                            batch_size = tool_calls_snapshot.len(),
                            "{msg}"
                        );
                    } else if tracing::enabled!(tracing::Level::DEBUG) {
                        // spawn_count <= 1: no concurrency possible, log at
                        // debug for audit without adding default-level noise.
                        let names: Vec<&str> = tool_calls_snapshot
                            .iter()
                            .map(|tc| tc.name.as_str())
                            .collect();
                        tracing::debug!(
                            session_id = ?session_id,
                            tools = ?names,
                            batch_size = tool_calls_snapshot.len(),
                            spawn_agent_count = spawn_count,
                            "multi-tool batch with <= 1 spawn_agent (no concurrent fan-out possible)"
                        );
                    }
                }

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
                                    // Expand user-defined aliases so the LLM can
                                    // switch_model with a short alias like "opus"
                                    // and land on the configured target.
                                    let new_model =
                                        expand_user_alias(&new_model, &state.user_aliases);
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

            FinishReason::Malformed if malformed_retries < MAX_MALFORMED_RETRIES => {
                // Transient per-turn model error (Bedrock stopReason
                // `malformed_model_output` / `malformed_tool_use`).  Retry
                // the exact same request once — do not push any partial
                // assistant text or tool calls onto `messages`, so the
                // next iteration re-sends an identical payload.
                malformed_retries += 1;
                tracing::warn!(
                    attempt = malformed_retries,
                    "malformed model output; retrying turn with unchanged messages"
                );
                continue;
            }

            FinishReason::Malformed | FinishReason::Other(_) => {
                // Second-strike Malformed (or any other unhandled
                // finish reason): terminate the session cleanly and
                // surface the reason string to the client.
                let reason: String = match &resp.finish_reason {
                    FinishReason::Other(s) => s.clone(),
                    FinishReason::Malformed => "malformed_output".to_string(),
                    _ => unreachable!(),
                };
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

    // Resolve model from env var (same default as CLI client), then expand
    // any user-defined alias (e.g. AMAEBI_MODEL=opus) using the daemon's
    // snapshotted alias table so cron jobs hit the same backend as
    // interactive requests.
    let model = std::env::var("AMAEBI_MODEL")
        .unwrap_or_else(|_| crate::provider::DEFAULT_MODEL.to_string());
    let model = expand_user_alias(&model, &state.user_aliases);

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

    /// Minimal in-memory DaemonState for tests that exercise
    /// handle_claude_launch error paths (resume-pane probes, etc.).
    /// No real HTTP, no real memory DB file — just enough plumbing
    /// for the code under test to run without panicking.  Tasks DB
    /// starts empty; tests that trigger the LLM tagger must wire a
    /// mock themselves.
    fn test_minimal_daemon_state() -> Arc<DaemonState> {
        Arc::new(DaemonState {
            http: reqwest::Client::new(),
            tokens: Arc::new(TokenCache::new()),
            executor: Box::new(tools::LocalExecutor::new()),
            db: Arc::new(Mutex::new(
                rusqlite::Connection::open_in_memory().expect("open in-memory memory DB"),
            )),
            compacting_sessions: Arc::new(Mutex::new(HashSet::new())),
            active_sessions: Arc::new(Mutex::new(HashSet::new())),
            user_aliases: Arc::new(std::collections::HashMap::new()),
            tasks_db: Arc::new(Mutex::new(None)),
        })
    }

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
            format!("copilot/{}", crate::provider::DEFAULT_MODEL_BARE),
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
    fn summarise_tool_detail_tmux_send_text_short_returned_verbatim() {
        let args = serde_json::json!({ "text": "hello world" });
        assert_eq!(
            summarise_tool_detail("tmux_send_text", &args),
            "hello world"
        );
    }

    #[test]
    fn summarise_tool_detail_tmux_send_text_long_truncates() {
        // Multi-KB paste is the common real-world case — confirm the
        // preview is bounded to SHELL_DETAIL_MAX_CHARS + ellipsis instead
        // of spamming the whole body into logs / tool-use frames.
        let text: String = "a".repeat(2_000);
        let args = serde_json::json!({ "text": text });
        let detail = summarise_tool_detail("tmux_send_text", &args);
        assert_eq!(detail.chars().count(), SHELL_DETAIL_MAX_CHARS + 1);
        assert!(detail.ends_with('…'));
    }

    #[test]
    fn summarise_tool_detail_tmux_send_text_flattens_newlines() {
        // Raw newlines would break the CLI's single-line tool-use renderer;
        // flatten to spaces so the preview stays on one visual line.
        let args = serde_json::json!({ "text": "line one\nline two\rline three" });
        let detail = summarise_tool_detail("tmux_send_text", &args);
        assert!(!detail.contains('\n'));
        assert!(!detail.contains('\r'));
        assert_eq!(detail, "line one line two line three");
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

    // ---- expand_user_alias -------------------------------------------------

    fn aliases(pairs: &[(&str, &str)]) -> std::collections::HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn expand_user_alias_expands_bare_name() {
        let map = aliases(&[("opus", "bedrock/claude-opus-4.7")]);
        assert_eq!(expand_user_alias("opus", &map), "bedrock/claude-opus-4.7");
    }

    #[test]
    fn expand_user_alias_preserves_1m_suffix() {
        // `opus[1m]` must expand the bare name and reattach the suffix.
        let map = aliases(&[("opus", "bedrock/claude-opus-4.7")]);
        assert_eq!(
            expand_user_alias("opus[1m]", &map),
            "bedrock/claude-opus-4.7[1m]"
        );
    }

    #[test]
    fn expand_user_alias_builtin_shadows_user() {
        // `claude-opus-4.6` is a built-in; user alias must be ignored.
        let map = aliases(&[("claude-opus-4.6", "bedrock/claude-sonnet-4.6")]);
        assert_eq!(
            expand_user_alias("claude-opus-4.6", &map),
            "claude-opus-4.6"
        );
    }

    #[test]
    fn expand_user_alias_passes_through_provider_prefix() {
        // `bedrock/...` already carries a provider prefix and must not be
        // treated as a bare-name candidate for expansion.
        let map = aliases(&[("bedrock/foo", "bedrock/bar")]);
        assert_eq!(expand_user_alias("bedrock/foo", &map), "bedrock/foo");
    }

    #[test]
    fn expand_user_alias_unknown_passes_through() {
        let map = aliases(&[("opus", "bedrock/claude-opus-4.7")]);
        assert_eq!(expand_user_alias("unknown", &map), "unknown");
    }

    #[test]
    fn expand_user_alias_no_chain_resolution() {
        // `a -> b`, `b -> bedrock/...`.  Expanding "a" must stop at "b".
        let map = aliases(&[("a", "b"), ("b", "bedrock/claude-opus-4.7")]);
        assert_eq!(expand_user_alias("a", &map), "b");
    }

    // ------------------------------------------------------------------
    // socket_in_use tests
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn socket_in_use_detects_live_listener_then_stale() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("amaebi.sock");

        // Bind a listener: the probe must report the socket as in-use.
        let listener = tokio::net::UnixListener::bind(&sock).expect("bind");
        assert!(
            socket_in_use(&sock).await.expect("probe live"),
            "live listener should be reported as in-use"
        );

        // Drop the listener and unlink the file: probe must report stale.
        drop(listener);
        let _ = std::fs::remove_file(&sock);
        assert!(
            !socket_in_use(&sock).await.expect("probe missing"),
            "missing socket should be reported as stale"
        );

        // Re-bind and drop *without unlinking* — leaves a real stale socket
        // path on disk.  Connect should yield ConnectionRefused, which the
        // probe classifies as stale.
        let listener = tokio::net::UnixListener::bind(&sock).expect("rebind");
        drop(listener);
        assert!(
            !socket_in_use(&sock).await.expect("probe stale socket path"),
            "leftover socket path with no listener should be reported as stale"
        );

        // A leftover *non-socket* regular file at the path must also be
        // treated as stale (ENOTSOCK → Ok(false)) so the daemon can unlink
        // random junk left behind by other tools.
        let _ = std::fs::remove_file(&sock);
        std::fs::write(&sock, b"").expect("touch non-socket file");
        assert!(
            !socket_in_use(&sock).await.expect("probe non-socket file"),
            "non-socket file at the socket path should be reported as stale"
        );
    }

    // ------------------------------------------------------------------
    // release_supervised_panes tests
    // ------------------------------------------------------------------

    /// `release_supervised_panes` must flip a Busy pane to Idle and clear its
    /// task/session fields while preserving `has_claude` and `worktree` so the
    /// next `/claude` invocation can reuse the same Claude Code session via
    /// the tier-1 reuse path (inject task into existing claude).  Without this
    /// call, the pane would stay Busy until `LEASE_TTL_SECS` (24 h) and a
    /// second `/claude` would appear stuck.
    #[tokio::test]
    async fn release_supervised_panes_unlocks_pane_and_preserves_reuse_fields() {
        let _guard = crate::test_utils::with_temp_home();

        let worktree = "/tmp/fake-worktree/task1";
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before UNIX epoch")
            .as_secs();
        let seed = pane_lease::PaneLease {
            pane_id: "%7".to_string(),
            window_id: "@3".to_string(),
            status: pane_lease::PaneStatus::Busy,
            tag: Some("task-abc".to_string()),
            session_id: Some("sess-xyz".to_string()),
            worktree: Some(worktree.to_string()),
            heartbeat_at: now,
            has_claude: true,
            task_description: None,
        };
        pane_lease::seed_state_for_test(seed).expect("seed pane state");

        let panes = vec![crate::ipc::SupervisionTarget {
            pane_id: "%7".to_string(),
            task_description: "do the thing".to_string(),
            tag: None,
            repo_dir: None,
        }];
        release_supervised_panes(&panes).await;

        let state = pane_lease::read_state().expect("read pane state after release");
        let lease = state.get("%7").expect("pane %7 still tracked");
        assert_eq!(
            lease.status,
            pane_lease::PaneStatus::Idle,
            "raw status must be flipped to Idle by release (not just TTL-expired)"
        );
        assert_eq!(
            lease.effective_status(),
            pane_lease::PaneStatus::Idle,
            "pane must be Idle after release"
        );
        assert!(
            lease.has_claude,
            "has_claude must be preserved so tier-1 reuse can match"
        );
        assert_eq!(
            lease.worktree.as_deref(),
            Some(worktree),
            "worktree must be preserved so tier-1 reuse can match"
        );
        assert!(lease.tag.is_none(), "tag must be cleared");
        assert!(lease.session_id.is_none(), "session_id must be cleared");
    }

    /// Releasing a target that does not exist in the state must not panic or
    /// return an error — release is best-effort and must never mask an inner
    /// supervision result.
    #[tokio::test]
    async fn release_supervised_panes_is_noop_for_missing_pane() {
        let _guard = crate::test_utils::with_temp_home();
        let panes = vec![crate::ipc::SupervisionTarget {
            pane_id: "%999".to_string(),
            task_description: "nonexistent".to_string(),
            tag: None,
            repo_dir: None,
        }];
        release_supervised_panes(&panes).await;
        let state = pane_lease::read_state().expect("read pane state");
        assert!(state.get("%999").is_none(), "no pane should be created");
    }

    // ------------------------------------------------------------------
    // handle_claude_launch --resume-pane error-path tests
    //
    // These tests cover the "description omitted" resume-pane lookup:
    // the daemon must read the lease, surface the specific failure mode
    // to the user, and return WITHOUT touching tmux.  The happy path
    // spawns `tmux send-keys` so is not exercised here (kept for
    // integration tests).
    // ------------------------------------------------------------------

    /// Collect every frame the daemon writes on a connected pair until EOF,
    /// parsing each newline-delimited JSON blob back into `Response`.
    async fn collect_responses(
        mut reader: tokio::net::unix::OwnedReadHalf,
    ) -> Vec<crate::ipc::Response> {
        use tokio::io::AsyncReadExt;
        let mut buf = Vec::new();
        reader
            .read_to_end(&mut buf)
            .await
            .expect("read daemon responses");
        let text = String::from_utf8(buf).expect("utf-8 daemon responses");
        text.lines()
            .filter(|l| !l.is_empty())
            .map(|l| {
                serde_json::from_str::<crate::ipc::Response>(l)
                    .unwrap_or_else(|e| panic!("parse response `{l}`: {e}"))
            })
            .collect()
    }

    /// When `--resume-pane` points at a pane that isn't in the lease state and
    /// the task description is empty, the daemon must reply with a "not found"
    /// error (distinct from the missing-description and read-failure cases).
    #[tokio::test]
    async fn resume_pane_missing_pane_returns_not_found_error() {
        let _guard = crate::test_utils::with_temp_home();
        let (client, server) = tokio::net::UnixStream::pair().expect("unix pair");
        // `handle_claude_launch` writes to the daemon-side `writer`; the test
        // reads those frames from the client-side reader.  `pair()` gives us
        // a duplex socket, so server.writer -> client.reader is the right
        // direction.  Drop the unused halves so EOF propagates cleanly.
        let (server_reader_unused, server_writer) = server.into_split();
        let (client_reader, client_writer_unused) = client.into_split();
        drop(server_reader_unused);
        drop(client_writer_unused);
        let writer = Arc::new(tokio::sync::Mutex::new(server_writer));

        let task = crate::ipc::TaskSpec {
            tag: "task-missing".to_string(),
            description: String::new(),
            worktree: None,
            client_cwd: None,
            auto_enter: true,
            resume_pane: Some("%999".to_string()),
            resources: Vec::new(),
            resource_timeout_secs: None,
        };
        let state = test_minimal_daemon_state();
        handle_claude_launch(&writer, vec![task], None, None, &state)
            .await
            .expect("launch returns ok even on per-task error");
        drop(writer);

        let responses = collect_responses(client_reader).await;
        assert!(
            responses.iter().any(|r| matches!(
                r,
                crate::ipc::Response::Error { message }
                    if message.contains("%999")
                        && message.contains("not found in lease state")
            )),
            "expected not-found error for %999, got {responses:?}"
        );
    }

    /// When `--resume-pane` points at a real lease that has no persisted
    /// `task_description` and the user omitted one, the daemon must reply with
    /// the "no saved task description" error rather than the not-found error.
    #[tokio::test]
    async fn resume_pane_existing_pane_without_description_errors() {
        let _guard = crate::test_utils::with_temp_home();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before UNIX epoch")
            .as_secs();
        let seed = pane_lease::PaneLease {
            pane_id: "%42".to_string(),
            window_id: "@1".to_string(),
            status: pane_lease::PaneStatus::Idle,
            tag: None,
            session_id: None,
            worktree: Some("/tmp/fake-worktree/resume".to_string()),
            heartbeat_at: now,
            has_claude: true,
            task_description: None,
        };
        pane_lease::seed_state_for_test(seed).expect("seed pane state");

        let (client, server) = tokio::net::UnixStream::pair().expect("unix pair");
        // `handle_claude_launch` writes to the daemon-side `writer`; the test
        // reads those frames from the client-side reader.  `pair()` gives us
        // a duplex socket, so server.writer -> client.reader is the right
        // direction.  Drop the unused halves so EOF propagates cleanly.
        let (server_reader_unused, server_writer) = server.into_split();
        let (client_reader, client_writer_unused) = client.into_split();
        drop(server_reader_unused);
        drop(client_writer_unused);
        let writer = Arc::new(tokio::sync::Mutex::new(server_writer));

        let task = crate::ipc::TaskSpec {
            tag: "task-nodesc".to_string(),
            description: String::new(),
            worktree: None,
            client_cwd: None,
            auto_enter: true,
            resume_pane: Some("%42".to_string()),
            resources: Vec::new(),
            resource_timeout_secs: None,
        };
        let state = test_minimal_daemon_state();
        handle_claude_launch(&writer, vec![task], None, None, &state)
            .await
            .expect("launch returns ok even on per-task error");
        drop(writer);

        let responses = collect_responses(client_reader).await;
        assert!(
            responses.iter().any(|r| matches!(
                r,
                crate::ipc::Response::Error { message }
                    if message.contains("%42")
                        && message.contains("has no saved task description")
            )),
            "expected missing-description error for %42, got {responses:?}"
        );
    }

    /// When `--resume-pane` points at a lease that claims `has_claude=true`
    /// but the target tmux pane does not actually exist (or isn't running
    /// `claude`), the daemon must reject the request at the tmux probe step
    /// rather than injecting a task prompt into whatever is there.  The task
    /// description is non-empty here so the lease-description prefetch is
    /// skipped and the probe runs.
    #[tokio::test]
    async fn resume_pane_tmux_probe_rejects_nonexistent_pane() {
        let _guard = crate::test_utils::with_temp_home();
        // Use a tmux pane id that does not exist in any tmux session.  The
        // `tmux display-message -t %9999999` call will fail with a non-zero
        // exit and an "unknown pane" / "can't find pane" stderr, which our
        // probe surfaces as a tmux-inspection failure.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before UNIX epoch")
            .as_secs();
        let seed = pane_lease::PaneLease {
            pane_id: "%9999999".to_string(),
            window_id: "@1".to_string(),
            status: pane_lease::PaneStatus::Idle,
            tag: None,
            session_id: None,
            worktree: Some("/tmp/fake-worktree/resume-probe".to_string()),
            heartbeat_at: now,
            has_claude: true,
            task_description: Some("old task".to_string()),
        };
        pane_lease::seed_state_for_test(seed).expect("seed pane state");

        let (client, server) = tokio::net::UnixStream::pair().expect("unix pair");
        let (server_reader_unused, server_writer) = server.into_split();
        let (client_reader, client_writer_unused) = client.into_split();
        drop(server_reader_unused);
        drop(client_writer_unused);
        let writer = Arc::new(tokio::sync::Mutex::new(server_writer));

        let task = crate::ipc::TaskSpec {
            tag: "task-probe".to_string(),
            // Non-empty description: skips the lease-description prefetch
            // early-return and forces execution into the tmux probe branch.
            description: "run this task".to_string(),
            worktree: None,
            client_cwd: None,
            auto_enter: true,
            resume_pane: Some("%9999999".to_string()),
            resources: Vec::new(),
            resource_timeout_secs: None,
        };
        let state = test_minimal_daemon_state();
        handle_claude_launch(&writer, vec![task], None, None, &state)
            .await
            .expect("launch returns ok even on per-task error");
        drop(writer);

        let responses = collect_responses(client_reader).await;
        let err_texts: Vec<&str> = responses
            .iter()
            .filter_map(|r| match r {
                crate::ipc::Response::Error { message } => Some(message.as_str()),
                _ => None,
            })
            .collect();
        // When running under a live tmux server (CI and local dev both have
        // one), `tmux display-message -t %9999999` exits 0 with empty stdout
        // because tmux silently falls back to a nearby pane's formatting
        // context — so our probe sees `pane_current_command == ""`, which
        // trips the "not currently running `claude`" branch.  When there is
        // no tmux at all, the probe fails on `.output()` and surfaces the
        // "failed to inspect tmux pane" branch.  Either is acceptable; both
        // are preferable to silently injecting a task prompt into the wrong
        // thing.
        assert!(
            err_texts.iter().any(|m| {
                m.contains("%9999999")
                    && (m.contains("failed to inspect tmux pane")
                        || m.contains("is not currently running `claude`"))
            }),
            "expected tmux-probe failure for %9999999, got errors: {err_texts:?}"
        );
    }

    // ------------------------------------------------------------------
    // resolve_session_dir + load_session_state dir-scoping regression
    //
    // These guard the core behaviour of PR #121: summaries must be scoped
    // by working directory, and the sentinel fallback must never read
    // back rows tagged with it (otherwise cross-project leakage returns).
    // ------------------------------------------------------------------

    /// A resumable (rotated) session UUID must resolve back to its owning
    /// directory, even after a newer UUID has been issued for the same dir.
    /// This is the regression test for the `list_all()` → `dir_for_uuid`
    /// switch (Copilot review comment on `resolve_session_dir`): `list_all()`
    /// only returns the most-recent record per dir and would miss the old UUID.
    #[tokio::test]
    async fn resolve_session_dir_finds_rotated_uuid() {
        let _guard = crate::test_utils::with_temp_home();
        let dir = tempfile::tempdir().expect("tempdir");
        let old_uuid = crate::session::get_or_create(dir.path()).expect("create first");
        let new_uuid = crate::session::create_fresh(dir.path()).expect("rotate");
        assert_ne!(old_uuid, new_uuid);

        let canonical = std::fs::canonicalize(dir.path())
            .expect("canonicalize")
            .to_string_lossy()
            .into_owned();

        assert_eq!(
            resolve_session_dir(&new_uuid).await,
            canonical,
            "current UUID must resolve to its canonical directory"
        );
        assert_eq!(
            resolve_session_dir(&old_uuid).await,
            canonical,
            "rotated/historical UUID must still resolve (list_all would miss this)"
        );
    }

    /// An unknown UUID must fall back to the sentinel, not to `""`.  Falling
    /// back to `""` would match legacy pre-migration rows (`dir = ''`) and
    /// re-introduce the cross-project leak this PR fixes.
    #[tokio::test]
    async fn resolve_session_dir_unknown_uuid_returns_sentinel() {
        let _guard = crate::test_utils::with_temp_home();
        // No sessions.json entries at all.
        assert_eq!(
            resolve_session_dir("not-in-sessions").await,
            UNKNOWN_DIR_SENTINEL,
            "unknown UUID must quarantine via sentinel, not leak via empty-string"
        );
    }

    /// `load_session_state` must short-circuit the summaries query when the
    /// resolved dir is the sentinel, so rows previously written with the
    /// sentinel (after a failed resolve) cannot be read back.  Exercises
    /// the integration between `resolve_session_dir` (returns sentinel for
    /// unknown UUID) and `load_session_state` (short-circuits on sentinel).
    #[tokio::test]
    async fn load_session_state_sentinel_skips_summaries_query() {
        let _guard = crate::test_utils::with_temp_home();

        // File-backed DB under the temp HOME so `init_db` runs the full
        // schema + migration path exactly as the daemon would.
        let db_dir = tempfile::tempdir().expect("tempdir for db");
        let db_path = db_dir.path().join("memory.db");
        let conn = memory_db::init_db(&db_path).expect("init memory db");

        // Seed a summary row tagged with the sentinel — simulates what
        // `compact_session` would have written for an unresolvable session.
        memory_db::store_session_summary(
            &conn,
            "sid-quarantined",
            "quarantined summary text",
            "2026-01-01T00:00:00Z",
            UNKNOWN_DIR_SENTINEL,
        )
        .expect("seed sentinel row");

        // Sanity: the raw query at the DB layer CAN read the row back if
        // asked directly with the sentinel — proves the short-circuit at
        // the daemon layer is load-bearing, not a trivial no-op.
        let raw = memory_db::get_recent_summaries(&conn, "other-sid", UNKNOWN_DIR_SENTINEL, 10)
            .expect("raw summaries read");
        assert_eq!(
            raw,
            vec!["quarantined summary text"],
            "sentinel rows are reachable at the memory_db layer — daemon must short-circuit"
        );

        let state = Arc::new(DaemonState {
            http: reqwest::Client::new(),
            tokens: Arc::new(TokenCache::new()),
            executor: Box::new(tools::LocalExecutor::new()),
            db: Arc::new(Mutex::new(conn)),
            compacting_sessions: Arc::new(Mutex::new(HashSet::new())),
            active_sessions: Arc::new(Mutex::new(HashSet::new())),
            user_aliases: Arc::new(std::collections::HashMap::new()),
            tasks_db: Arc::new(Mutex::new(None)),
        });

        // "sid-quarantined" has no entry in the empty sessions.json under
        // the temp $HOME, so resolve_session_dir returns the sentinel and
        // load_session_state MUST skip the query.
        let (_hist, summaries, _own) = load_session_state(&state, "sid-quarantined").await;
        assert!(
            summaries.is_empty(),
            "load_session_state must not read sentinel rows back; got {summaries:?}"
        );
    }

    // ------------------------------------------------------------------
    // TUI chrome stripping
    //
    // Fixtures modelled on real `tmux capture-pane` output of Claude Code
    // TUIs.  The guiding invariant is asymmetric: missing a chrome line
    // is merely noisy; eating a real-output line silently drops signal
    // from the supervisor LLM's view, which is strictly worse.
    // ------------------------------------------------------------------

    #[test]
    fn chrome_divider_line_is_stripped() {
        assert!(is_tui_chrome_line(
            "────────────────────────────────────────"
        ));
        // Heavy-weight divider variant.
        assert!(is_tui_chrome_line("━━━━━━━━━━━━━━"));
        // Indentation inside the divider line is tolerated.
        assert!(is_tui_chrome_line("    ──────────────    "));
    }

    #[test]
    fn chrome_short_dash_run_is_not_stripped() {
        // Three dashes could be a real `---` separator or a command-line
        // flag echo (`--help`).  The 10-char threshold prevents collateral.
        assert!(!is_tui_chrome_line("---"));
        assert!(!is_tui_chrome_line("────────")); // 8 chars, below threshold
    }

    #[test]
    fn chrome_status_bar_is_stripped() {
        assert!(is_tui_chrome_line(
            "  ⏵⏵ bypass permissions on · 1 shell · esc to interrupt · ctrl+t to hide tasks · ↓ to manage"
        ));
        assert!(is_tui_chrome_line("  ⏵⏵ bypass permissions on"));
        // Matched via the `esc to interrupt` substring (covers Claude Code
        // hint rows that don't start with the bypass banner).
        assert!(is_tui_chrome_line("  ↑/↓ navigate · esc to interrupt"));
    }

    #[test]
    fn chrome_empty_prompt_cursor_is_stripped() {
        assert!(is_tui_chrome_line("❯"));
        assert!(is_tui_chrome_line("   ❯   "));
    }

    #[test]
    fn chrome_empty_prompt_with_content_is_preserved() {
        // A populated prompt is real user/Claude input — must not be eaten.
        assert!(!is_tui_chrome_line("❯ what's the benchmark result?"));
        assert!(!is_tui_chrome_line("❯ ls -la"));
    }

    #[test]
    fn chrome_pure_border_line_is_stripped() {
        // Empty middle of a bordered box, e.g. rendered by Claude Code's
        // input/output panels.
        assert!(is_tui_chrome_line("│                                   │"));
        assert!(is_tui_chrome_line("╭───────────────╮"));
        assert!(is_tui_chrome_line("╰───────────────╯"));
    }

    #[test]
    fn chrome_log_with_pipe_separator_is_preserved() {
        // Real log lines often use `│` as a visual separator; they contain
        // non-border text so the all-border check rejects them.
        assert!(!is_tui_chrome_line("2026-04-25 14:22 │ INFO │ build done"));
        assert!(!is_tui_chrome_line("│ Build │ OK"));
    }

    #[test]
    fn chrome_checkmark_progress_rows_are_preserved() {
        // The ✔ rows in a Claude Code task list are *content*, not chrome.
        // Regression for the supervision screenshot the user reported.
        assert!(!is_tui_chrome_line("     ✔ A2: causal + paged combined"));
        assert!(!is_tui_chrome_line(
            "     ✔ Baseline perf (hd=128 fp16 GQA) before changes"
        ));
        assert!(!is_tui_chrome_line("      … +1 completed"));
    }

    #[test]
    fn chrome_shell_prompt_and_errors_are_preserved() {
        assert!(!is_tui_chrome_line("syk@host:/repo$ ls"));
        assert!(!is_tui_chrome_line("error: lint failed at line 10"));
        assert!(!is_tui_chrome_line("A1: 1.3× speedup"));
    }

    #[test]
    fn strip_tui_chrome_on_real_supervision_screenshot() {
        // Reproduces the pane snapshot the user pasted: a task header, a
        // list of ✔ completions, a divider, an empty ❯ prompt, another
        // divider, and the status bar.  Only the task + ✔ lines + the
        // `… +N completed` summary should survive; dividers, the empty
        // prompt, and the status bar must be filtered out.
        let raw = "\
继续fmha4_paged的功能迁移，根据ROADMAP.md一个一个功能往fmha4_paged加。每个功能加完要确保性能不掉，精度准确。
     ✔ A2: causal + paged combined
     ✔ Baseline perf (hd=128 fp16 GQA) before changes
     ✔ A1: hd=64 vrow fallback port
      … +1 completed

────────────────────────────────────────────────────────────────────────────────────────────────────────
❯

────────────────────────────────────────────────────────────────────────────────────────────────────────
  ⏵⏵ bypass permissions on · 1 shell · esc to interrupt · ctrl+t to hide tasks · ↓ to manage
";
        let cleaned = strip_tui_chrome(raw);
        // Content lines all survive.
        assert!(cleaned.contains("继续fmha4_paged的功能迁移"));
        assert!(cleaned.contains("✔ A2: causal + paged combined"));
        assert!(cleaned.contains("✔ Baseline perf"));
        assert!(cleaned.contains("✔ A1: hd=64 vrow fallback port"));
        assert!(cleaned.contains("… +1 completed"));
        // Chrome lines removed.
        assert!(
            !cleaned.contains("────────────────────────────────────────────────────────────────"),
            "divider must be stripped"
        );
        assert!(
            !cleaned.contains("bypass permissions"),
            "status bar must be stripped"
        );
        // Empty prompt `❯` line is gone but blank lines around it stay.
        let has_lone_caret = cleaned.lines().any(|l| l.trim() == "❯");
        assert!(!has_lone_caret, "empty ❯ prompt must be stripped");
    }

    #[test]
    fn strip_tui_chrome_preserves_blank_lines() {
        // Blank lines carry structure (paragraph breaks, shell idle); we
        // keep them so downstream trimming / tail extraction sees the
        // same line-count boundaries as the original capture.
        let raw = "line one\n\nline two\n\n\nline three\n";
        let cleaned = strip_tui_chrome(raw);
        assert_eq!(cleaned, "line one\n\nline two\n\n\nline three");
    }

    #[test]
    fn strip_tui_chrome_empty_input() {
        assert_eq!(strip_tui_chrome(""), "");
    }

    // ------------------------------------------------------------------
    // ensure_worktree_agents_md tests
    // ------------------------------------------------------------------

    fn agents_md_lease(name: &str, class: &str) -> resource_lease::ResourceLease {
        resource_lease::ResourceLease {
            name: name.to_string(),
            class: class.to_string(),
            status: resource_lease::ResourceStatus::Busy,
            pane_id: Some("%7".to_string()),
            tag: Some("t".to_string()),
            session_id: Some("s".to_string()),
            heartbeat_at: 0,
        }
    }

    fn agents_md_def(name: &str, class: &str, hint: &str) -> resource_lease::ResourceDef {
        resource_lease::ResourceDef {
            name: name.to_string(),
            class: class.to_string(),
            metadata: std::collections::HashMap::new(),
            env: std::collections::HashMap::new(),
            prompt_hint: Some(hint.to_string()),
        }
    }

    // ------------------------------------------------------------------
    // Supervision prompt shape tests
    //
    // These tests cover the structural contract of the user content fed to
    // the supervision LLM every turn: task pinned at byte 0, hard
    // constraints block derived from the pane's resource leases, verdict
    // history rendered in newest-last order, and the last STEER message
    // carried forward verbatim.
    // ------------------------------------------------------------------

    fn supervision_test_target(pane_id: &str, desc: &str) -> crate::ipc::SupervisionTarget {
        crate::ipc::SupervisionTarget {
            pane_id: pane_id.to_string(),
            task_description: desc.to_string(),
            tag: None,
            repo_dir: None,
        }
    }

    #[test]
    fn ensure_worktree_agents_md_creates_file_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        let wt = Some(dir.path().to_string_lossy().to_string());
        let leases = vec![agents_md_lease("sim-9902", "simulator")];
        let pool = vec![agents_md_def("sim-9902", "simulator", "use sim-9902 ONLY")];
        ensure_worktree_agents_md(&wt, &leases, &pool).expect("write ok");

        let contents = std::fs::read_to_string(dir.path().join("AGENTS.md")).unwrap();
        assert!(contents.contains("# Project rules"), "header missing");
        assert!(contents.contains(AGENTS_MD_BEGIN));
        assert!(contents.contains(AGENTS_MD_END));
        assert!(contents.contains("## Assigned resources"));
        assert!(contents.contains("use sim-9902 ONLY"));
    }

    #[test]
    fn ensure_worktree_agents_md_appends_to_existing_user_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("AGENTS.md");
        std::fs::write(&path, "# My rules\nFoo\n").unwrap();

        let wt = Some(dir.path().to_string_lossy().to_string());
        let leases = vec![agents_md_lease("sim-9902", "simulator")];
        let pool = vec![agents_md_def("sim-9902", "simulator", "use sim-9902 ONLY")];
        ensure_worktree_agents_md(&wt, &leases, &pool).expect("append ok");

        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(
            contents.starts_with("# My rules\nFoo\n"),
            "user content must stay at the top: {contents:?}"
        );
        assert!(contents.contains(AGENTS_MD_BEGIN));
        assert!(contents.contains("use sim-9902 ONLY"));
        let user_idx = contents.find("# My rules").unwrap();
        let marker_idx = contents.find(AGENTS_MD_BEGIN).unwrap();
        assert!(user_idx < marker_idx, "managed block must be appended");
    }

    #[test]
    fn ensure_worktree_agents_md_skips_when_block_present() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("AGENTS.md");
        let pre = format!(
            "# existing\n\n{AGENTS_MD_BEGIN}\n## Assigned resources\n\nold hint\n{AGENTS_MD_END}\n"
        );
        std::fs::write(&path, &pre).unwrap();

        let wt = Some(dir.path().to_string_lossy().to_string());
        let leases = vec![agents_md_lease("sim-9902", "simulator")];
        let pool = vec![agents_md_def("sim-9902", "simulator", "NEW HINT")];
        ensure_worktree_agents_md(&wt, &leases, &pool).expect("noop ok");
        ensure_worktree_agents_md(&wt, &leases, &pool).expect("idempotent");

        let after = std::fs::read_to_string(&path).unwrap();
        assert_eq!(after, pre, "existing managed block must not be rewritten");
        assert!(!after.contains("NEW HINT"));
    }

    #[test]
    fn ensure_worktree_agents_md_noop_when_no_resources() {
        let dir = tempfile::tempdir().unwrap();
        let wt = Some(dir.path().to_string_lossy().to_string());
        ensure_worktree_agents_md(&wt, &[], &[]).expect("no-op");
        assert!(!dir.path().join("AGENTS.md").exists());
    }

    #[test]
    fn ensure_worktree_agents_md_noop_when_no_worktree() {
        let leases = vec![agents_md_lease("sim-9902", "simulator")];
        let pool = vec![agents_md_def("sim-9902", "simulator", "hint")];
        ensure_worktree_agents_md(&None, &leases, &pool).expect("no-op");
    }

    #[test]
    fn ensure_worktree_agents_md_noop_when_hint_is_empty() {
        // Lease exists but its pool def has no `prompt_hint` — rendering
        // returns an empty string.  Writing a block with empty body would
        // mislead claude into thinking it has an empty constraint set; skip.
        let dir = tempfile::tempdir().unwrap();
        let wt = Some(dir.path().to_string_lossy().to_string());
        let leases = vec![agents_md_lease("sim-9902", "simulator")];
        let pool = vec![resource_lease::ResourceDef {
            name: "sim-9902".to_string(),
            class: "simulator".to_string(),
            metadata: std::collections::HashMap::new(),
            env: std::collections::HashMap::new(),
            prompt_hint: None,
        }];
        ensure_worktree_agents_md(&wt, &leases, &pool).expect("no-op");
        assert!(!dir.path().join("AGENTS.md").exists());
    }

    #[test]
    fn ensure_worktree_agents_md_noop_when_pool_missing_entry() {
        // Lease for `sim-9902` but the pool no longer has that entry
        // (e.g. resources.toml was re-edited).  render_prompt_hint returns
        // "" because the lookup fails — same no-op as the missing-hint case.
        let dir = tempfile::tempdir().unwrap();
        let wt = Some(dir.path().to_string_lossy().to_string());
        let leases = vec![agents_md_lease("sim-9902", "simulator")];
        let pool: Vec<resource_lease::ResourceDef> = Vec::new();
        ensure_worktree_agents_md(&wt, &leases, &pool).expect("no-op");
        assert!(!dir.path().join("AGENTS.md").exists());
    }

    #[test]
    fn supervision_prompt_contains_pinned_task_section() {
        let task_lines = vec![(
            "%5".to_string(),
            "implement feature X on sim-9900".to_string(),
        )];
        let history: std::collections::VecDeque<String> = std::collections::VecDeque::new();
        let out = build_supervision_user_content(
            &task_lines,
            "",
            "",
            &history,
            None,
            "=== Pane %5 ===\nidle\n",
            1,
            0,
        );
        assert!(
            out.starts_with("TASK — keep Claude focused on this:\n"),
            "user content must start with the pinned task header, got: {out:?}"
        );
        assert!(
            out.contains("implement feature X on sim-9900"),
            "task description must appear in the pinned section, got: {out:?}"
        );
        assert!(
            out.contains("pane %5:"),
            "pinned section must list the pane id, got: {out:?}"
        );
    }

    #[test]
    fn supervision_prompt_verdict_history_renders_multiple_lines() {
        let task_lines = vec![("%5".to_string(), "task".to_string())];
        let mut history = std::collections::VecDeque::new();
        history.push_back("WAIT: building".to_string());
        // Use the real on-wire shape the parser emits (`STEER: <pane>: …`
        // with the leading colon) so the test catches future drift if the
        // verdict format or parser prefix diverge.
        history.push_back("STEER: %5: use sim-9900".to_string());
        history.push_back("WAIT: now on sim-9900".to_string());
        let out = build_supervision_user_content(&task_lines, "", "", &history, None, "snap", 4, 3);
        assert!(
            out.contains("[last] WAIT: now on sim-9900"),
            "newest verdict must be labelled [last], got: {out:?}"
        );
        assert!(
            out.contains("[2 turns ago] STEER: %5: use sim-9900"),
            "second-newest verdict must be labelled [2 turns ago], got: {out:?}"
        );
        assert!(
            out.contains("[3 turns ago] WAIT: building"),
            "oldest verdict must be labelled [3 turns ago], got: {out:?}"
        );
    }

    #[test]
    fn supervision_prompt_last_steer_block_shows_full_text() {
        let task_lines = vec![("%5".to_string(), "task".to_string())];
        let history: std::collections::VecDeque<String> = std::collections::VecDeque::new();
        // Matches the real on-wire `STEER: <pane>: …` shape emitted by
        // `handle_supervision_inner`.
        let steer = "STEER: %5: stop — you are using sim-9903 but the \
                     task says sim-9900; rerun with the correct port";
        let out = build_supervision_user_content(
            &task_lines,
            "",
            "",
            &history,
            Some(steer),
            "snap",
            2,
            1,
        );
        assert!(
            out.contains("Most recent STEER message (escaped newlines):"),
            "missing STEER carry-over header, got: {out:?}"
        );
        assert!(
            out.contains("rerun with the correct port"),
            "full STEER text must appear verbatim, got: {out:?}"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn supervision_prompt_no_constraints_when_no_leases() {
        let _guard = crate::test_utils::with_temp_home();
        let panes = vec![supervision_test_target("%5", "do the thing")];
        let out = render_hard_constraints(&panes).await;
        assert!(
            out.is_empty(),
            "no leases for this pane → empty constraints block, got: {out:?}"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn supervision_prompt_hard_constraints_includes_resource_hint() {
        let _guard = crate::test_utils::with_temp_home();

        // Seed a pool file with a named resource that has a prompt_hint.
        let pool_path = crate::auth::amaebi_home()
            .expect("home")
            .join("resources.toml");
        std::fs::create_dir_all(pool_path.parent().unwrap()).expect("mkdir");
        std::fs::write(
            &pool_path,
            r#"
[[resource]]
name = "sim-9900"
class = "simulator"
metadata = { port = "9900" }
prompt_hint = "use sim-9900 (port {port}) only"
"#,
        )
        .expect("write pool");

        // Acquire the resource on %5 so the state file records a lease
        // whose pane_id matches our supervised pane.  This uses the real
        // resource_lease path, which also respects the flock.
        let _leases = resource_lease::acquire_all(
            &[resource_lease::ResourceRequest::Named("sim-9900".into())],
            resource_lease::Holder {
                pane_id: "%5".into(),
                tag: "t".into(),
                session_id: "s".into(),
            },
            resource_lease::WaitPolicy::Nowait,
        )
        .await
        .expect("acquire");

        let panes = vec![supervision_test_target("%5", "do the thing")];
        let out = render_hard_constraints(&panes).await;

        assert!(
            out.contains("HARD CONSTRAINTS"),
            "constraints section missing, got: {out:?}"
        );
        assert!(
            out.contains("pane %5:"),
            "constraints must be scoped per pane, got: {out:?}"
        );
        assert!(
            out.contains("use sim-9900 (port 9900) only"),
            "rendered prompt_hint must be included verbatim, got: {out:?}"
        );
    }

    // ------------------------------------------------------------------
    // supervision default timeout regression
    // ------------------------------------------------------------------

    #[test]
    #[serial_test::serial]
    fn supervision_default_timeout_matches_lease_ttl() {
        // Regression: the default supervision timeout must match every lease
        // TTL supervision holds on (pane, resource, task notebook) so the
        // loop never outlives its own bookkeeping.  The test derives the
        // expected value from the TTL constant rather than a literal, so
        // bumping lease TTL on a future PR does not require touching this
        // test — only the three constants need to stay equal.
        std::env::remove_var("AMAEBI_SUPERVISION_TIMEOUT_SECS");
        let lease_ttl_secs = crate::pane_lease::LEASE_TTL_SECS;
        let default_secs: u64 = std::env::var("AMAEBI_SUPERVISION_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(lease_ttl_secs);
        assert_eq!(default_secs, lease_ttl_secs);
        assert_eq!(crate::resource_lease::LEASE_TTL_SECS, lease_ttl_secs);
        assert_eq!(crate::tasks::LEASE_TTL_SECS as u64, lease_ttl_secs);
    }

    /// Regression: the supervision verdict-persist path must lazy-open
    /// `tasks.db` when the daemon only ever served resume-pane launches
    /// (which skip the tag-acquisition block that normally opens it).
    /// Verifies `ensure_tasks_db` opens the handle on first call, then
    /// is a cheap no-op on every subsequent call — so sprinkling it at
    /// the top of every tasks_db consumer is safe.
    #[test]
    fn ensure_tasks_db_is_idempotent_and_lazy() {
        // A daemon that has only served resume-pane launches reaches the
        // verdict-persist / notebook-context sites with `tasks_db == None`,
        // because the tag-acquisition block in `handle_claude_launch`
        // (the sole prior opener) is gated on `resume_pane.is_none()`.
        // Both sites must therefore call `ensure_tasks_db` unconditionally,
        // which only works if the second call is a safe no-op.
        let _guard = crate::test_utils::with_temp_home();
        let state = test_minimal_daemon_state();
        assert!(state.tasks_db.lock().unwrap().is_none());

        ensure_tasks_db(&state).expect("first ensure_tasks_db opens DB");
        assert!(state.tasks_db.lock().unwrap().is_some());

        ensure_tasks_db(&state).expect("second ensure_tasks_db is a no-op");
        assert!(state.tasks_db.lock().unwrap().is_some());
    }
}
