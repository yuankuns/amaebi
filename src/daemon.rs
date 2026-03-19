use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

use crate::auth::TokenCache;
use crate::copilot::{self, ApiToolCall, ApiToolCallFunction, FinishReason, Message};
use crate::ipc::{write_frame, Request, Response};
use crate::memory::{self, MemoryEntry};
use crate::memory_db;
use crate::tools::{self, ToolExecutor};

// ---------------------------------------------------------------------------
// In-memory cache for recent conversation history
// ---------------------------------------------------------------------------

/// Maximum number of entries kept in the in-memory cache (matches `load_recent` default).
const CACHE_SIZE: usize = 20;

/// Thread-safe in-memory ring buffer of the most recent memory entries.
///
/// The cache is warmed once at daemon startup and updated on every successful
/// `memory::append`, so `build_messages` never performs disk I/O on the hot path.
///
/// # Limitations
///
/// The cache is only updated through the daemon's own `memory::append` calls.
/// Entries written by external processes (e.g. future `amaebi` subcommands
/// that call `memory::append` directly) are **not** reflected in a running
/// daemon's cache.  A daemon restart is required to pick up such externally
/// written entries.  This is an acceptable trade-off for the current
/// single-user, single-daemon deployment model.
///
/// `load_from_disk` acquires a shared file lock (`lock_shared`) on
/// `memory.jsonl`.  If another process holds an exclusive lock at startup, the
/// call blocks until the lock is released — this is expected advisory-lock
/// behaviour and is harmless in practice.
pub struct MemoryCache {
    inner: tokio::sync::RwLock<VecDeque<MemoryEntry>>,
}

impl MemoryCache {
    pub fn new() -> Self {
        Self {
            inner: tokio::sync::RwLock::new(VecDeque::with_capacity(CACHE_SIZE)),
        }
    }

    /// Populate the cache from disk.  Called once at startup.
    pub async fn load_from_disk(&self) {
        match tokio::task::spawn_blocking(|| memory::load_recent(CACHE_SIZE)).await {
            Ok(Ok(entries)) => {
                let mut guard = self.inner.write().await;
                guard.clear();
                guard.extend(entries);
            }
            Ok(Err(e)) => tracing::warn!(error = %e, "failed to warm memory cache from disk"),
            Err(e) => tracing::warn!(error = %e, "memory::load_recent panicked during cache warm"),
        }
    }

    /// Push a new entry into the cache, evicting the oldest if necessary.
    pub async fn push(&self, entry: MemoryEntry) {
        let mut guard = self.inner.write().await;
        if guard.len() >= CACHE_SIZE {
            guard.pop_front();
        }
        guard.push_back(entry);
    }

    /// Return a snapshot of all cached entries in chronological order.
    ///
    /// Retained for potential future callers; context injection now uses SQLite.
    #[allow(dead_code)]
    pub async fn snapshot(&self) -> Vec<MemoryEntry> {
        self.inner.read().await.iter().cloned().collect()
    }

    /// Empty the cache.  Called when the persistent store is wiped by an
    /// external command (`amaebi memory clear`) so that subsequent requests
    /// do not replay stale history.
    pub async fn clear(&self) {
        self.inner.write().await.clear();
    }
}

// ---------------------------------------------------------------------------
// Shared daemon state
// ---------------------------------------------------------------------------

/// State shared across all concurrent client connections via `Arc`.
///
/// Phase 4 will extend this with a `SessionMap` for subagent tracking.
pub struct DaemonState {
    pub http: reqwest::Client,
    pub tokens: TokenCache,
    /// Tool executor — `LocalExecutor` now; swappable with `DockerExecutor` in Phase 4.
    pub executor: Box<dyn ToolExecutor>,
    /// Serialises concurrent `memory::append` calls within this process so that
    /// parallel client connections cannot interleave their writes to the memory file.
    pub memory_lock: tokio::sync::Mutex<()>,
    /// In-memory cache of recent conversation history; avoids disk I/O on the hot path.
    pub memory_cache: MemoryCache,
    /// Path to the SQLite memory database (`~/.amaebi/memory.db`).
    pub db_path: PathBuf,
}

impl DaemonState {
    pub fn new() -> Result<Self> {
        let http = reqwest::Client::builder()
            .build()
            .context("building HTTP client")?;
        let db_path = memory_db::db_path().context("resolving memory DB path")?;
        Ok(Self {
            http,
            tokens: TokenCache::new(),
            executor: Box::new(tools::LocalExecutor),
            memory_lock: tokio::sync::Mutex::new(()),
            memory_cache: MemoryCache::new(),
            db_path,
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
    state.memory_cache.load_from_disk().await;

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
            state.memory_cache.clear().await;
            write_frame(&mut writer, &Response::Done).await?;
        }

        Request::Chat {
            prompt,
            tmux_pane,
            model,
            session_id: _,
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

            let messages = build_messages(&prompt, tmux_pane.as_deref(), &state).await;

            match run_agentic_loop(&state, &model, messages, &mut writer).await {
                Ok(response_text) => {
                    // Serialise memory writes within this process; file-level flock in
                    // memory::append handles cross-process protection.
                    let mem_guard = state.memory_lock.lock().await;
                    let prompt_clone = prompt.clone();
                    let response_clone = response_text.clone();
                    let db_path = state.db_path.clone();
                    let mem_result = tokio::task::spawn_blocking(move || {
                        let timestamp = chrono::Utc::now().to_rfc3339();
                        // Primary: write to SQLite (full content, no truncation).
                        let conn = memory_db::init_db(&db_path)?;
                        memory_db::store_memory(&conn, &timestamp, "", "user", &prompt_clone, "")?;
                        memory_db::store_memory(
                            &conn,
                            &timestamp,
                            "",
                            "assistant",
                            &response_clone,
                            "",
                        )?;
                        // Secondary: keep JSONL for backward compatibility.
                        memory::append(&prompt_clone, &response_clone)
                    })
                    .await
                    .unwrap_or_else(|e| Err(anyhow::anyhow!("memory write panicked: {e}")));
                    // Release the lock immediately after the write — before logging —
                    // so other connections are not blocked while we format a warning.
                    drop(mem_guard);
                    match mem_result {
                        Ok(entry) => state.memory_cache.push(entry).await,
                        Err(e) => tracing::warn!(error = %e, "failed to save memory"),
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, "agentic loop error");
                    // Best-effort: the stream may be partially written already.
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
// Agentic loop
// ---------------------------------------------------------------------------

/// Drive the conversation until Copilot responds with `finish_reason: stop`
/// (or an error).  Executes tool calls and feeds results back in a loop.
pub(crate) async fn run_agentic_loop<W>(
    state: &DaemonState,
    model: &str,
    mut messages: Vec<Message>,
    writer: &mut W,
) -> Result<String>
where
    W: AsyncWriteExt + Unpin,
{
    let schemas = tools::tool_schemas();
    let final_text;

    loop {
        // Re-fetch the token on every iteration so long-running agentic loops
        // survive token expiration.  `TokenCache::get` returns the cached value
        // when it is still valid, so there is no extra network request on cache hit.
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
                // Append the assistant's turn (with tool_calls) to history.
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

                // Execute each requested tool and append results.
                for tc in &resp.tool_calls {
                    tracing::debug!(tool = %tc.name, "executing tool");

                    // Notify the client so it can show progress.
                    let tool_detail = {
                        let args: serde_json::Value =
                            serde_json::from_str(&tc.arguments).unwrap_or(serde_json::Value::Null);
                        let s = match tc.name.as_str() {
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
                        };
                        s
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
                // Continue the loop with the updated message history.
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

/// Files checked in the current working directory, in injection order.
/// Each tuple is `(filename, section_header)`.
const SKILL_FILES: &[(&str, &str)] = &[
    ("SKILL.md", "## Skill Context"),
    ("AGENTS.md", "## Agent Guidelines"),
    ("SOUL.md", "## Soul"),
];

/// Read `SKILL.md`, `AGENTS.md`, and `SOUL.md` from `base_dir` and append each
/// non-empty file as a `system` message with an appropriate section header.
///
/// Injected after the main system prompt and before memory/history so that
/// per-project instructions take precedence over generic guidance but remain
/// below the model's core identity.
///
/// `base_dir` is passed explicitly (rather than reading `current_dir` inside)
/// so callers can be tested without modifying the process working directory.
pub(crate) async fn inject_skill_files(messages: &mut Vec<Message>, base_dir: &std::path::Path) {
    for (filename, header) in SKILL_FILES {
        let path = base_dir.join(filename);
        match tokio::fs::read_to_string(&path).await {
            Ok(content) => {
                let trimmed = content.trim();
                if !trimmed.is_empty() {
                    messages.push(Message::system(format!("{header}\n\n{trimmed}")));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // File absent — skip silently.
            }
            Err(e) => {
                tracing::debug!(file = %path.display(), error = %e, "could not read skill file");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Message construction
// ---------------------------------------------------------------------------

pub(crate) async fn build_messages(
    prompt: &str,
    tmux_pane: Option<&str>,
    state: &DaemonState,
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

    // Inject any per-project skill/agent/soul files from the CWD.
    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    inject_skill_files(&mut messages, &cwd).await;

    // Retrieve context from SQLite: last 4 turns for continuity, plus up to
    // 10 FTS-relevant entries for historical context.  Deduplication and
    // chronological ordering are handled by `retrieve_context`.
    let db_path = state.db_path.clone();
    let prompt_owned = prompt.to_owned();
    match tokio::task::spawn_blocking(move || {
        let conn = memory_db::init_db(&db_path)?;
        memory_db::retrieve_context(&conn, &prompt_owned, 4, 10)
    })
    .await
    {
        Ok(Ok(entries)) => {
            for entry in entries {
                if entry.role == "user" {
                    messages.push(Message::user(entry.content));
                } else {
                    messages.push(Message::assistant(Some(entry.content), vec![]));
                }
            }
        }
        Ok(Err(e)) => tracing::warn!(error = %e, "failed to load memory context from SQLite"),
        Err(e) => tracing::warn!(error = %e, "memory context load task panicked"),
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

    #[tokio::test]
    async fn cache_starts_empty() {
        let cache = MemoryCache::new();
        assert!(cache.snapshot().await.is_empty());
    }

    #[tokio::test]
    async fn cache_push_and_snapshot() {
        let cache = MemoryCache::new();
        cache.push(make_entry("hello", "world")).await;
        let snap = cache.snapshot().await;
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].user, "hello");
        assert_eq!(snap[0].assistant, "world");
    }

    #[tokio::test]
    async fn cache_evicts_oldest_at_capacity() {
        let cache = MemoryCache::new();
        for i in 0..=CACHE_SIZE {
            cache
                .push(make_entry(&format!("u{i}"), &format!("a{i}")))
                .await;
        }
        let snap = cache.snapshot().await;
        assert_eq!(snap.len(), CACHE_SIZE);
        // Oldest entry (u0) should have been evicted.
        assert_eq!(snap[0].user, "u1");
        assert_eq!(snap[CACHE_SIZE - 1].user, format!("u{CACHE_SIZE}"));
    }

    #[tokio::test]
    async fn cache_preserves_order() {
        let cache = MemoryCache::new();
        cache.push(make_entry("first", "a")).await;
        cache.push(make_entry("second", "b")).await;
        cache.push(make_entry("third", "c")).await;
        let snap = cache.snapshot().await;
        assert_eq!(snap[0].user, "first");
        assert_eq!(snap[1].user, "second");
        assert_eq!(snap[2].user, "third");
    }

    #[tokio::test]
    async fn cache_clear_empties_all_entries() {
        let cache = MemoryCache::new();
        cache.push(make_entry("a", "b")).await;
        cache.push(make_entry("c", "d")).await;
        assert_eq!(cache.snapshot().await.len(), 2);
        cache.clear().await;
        assert!(cache.snapshot().await.is_empty());
    }

    #[tokio::test]
    async fn cache_clear_then_push_works() {
        let cache = MemoryCache::new();
        cache.push(make_entry("old", "data")).await;
        cache.clear().await;
        cache.push(make_entry("new", "data")).await;
        let snap = cache.snapshot().await;
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].user, "new");
    }

    // ------------------------------------------------------------------
    // inject_skill_files tests
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn skill_files_injected_when_present() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("SKILL.md"), "skill instructions").unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "agent guidelines").unwrap();
        std::fs::write(dir.path().join("SOUL.md"), "soul content").unwrap();

        let mut messages: Vec<Message> = vec![];
        inject_skill_files(&mut messages, dir.path()).await;

        assert_eq!(messages.len(), 3);
        // Order matches SKILL_FILES: SKILL.md, AGENTS.md, SOUL.md.
        let sys = |m: &Message| m.content.as_deref().unwrap_or("").to_owned();
        assert!(sys(&messages[0]).contains("## Skill Context"));
        assert!(sys(&messages[0]).contains("skill instructions"));
        assert!(sys(&messages[1]).contains("## Agent Guidelines"));
        assert!(sys(&messages[1]).contains("agent guidelines"));
        assert!(sys(&messages[2]).contains("## Soul"));
        assert!(sys(&messages[2]).contains("soul content"));
    }

    #[tokio::test]
    async fn skill_files_absent_produces_no_messages() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut messages: Vec<Message> = vec![];
        inject_skill_files(&mut messages, dir.path()).await;
        assert!(messages.is_empty());
    }

    #[tokio::test]
    async fn skill_files_empty_file_skipped() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("SKILL.md"), "   \n  ").unwrap();
        let mut messages: Vec<Message> = vec![];
        inject_skill_files(&mut messages, dir.path()).await;
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
        inject_skill_files(&mut messages, dir.path()).await;
        assert_eq!(messages.len(), 1);
        assert!(messages[0]
            .content
            .as_deref()
            .unwrap_or("")
            .contains("soul only"));
    }
}
