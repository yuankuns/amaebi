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
    pub async fn snapshot(&self) -> Vec<MemoryEntry> {
        self.inner.read().await.iter().cloned().collect()
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
            memory_cache: MemoryCache::new(),
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

    tracing::info!(
        pane = ?req.tmux_pane,
        model = %req.model,
        prompt_len = req.prompt.len(),
        "received request"
    );

    let token = match state.tokens.get(&state.http).await {
        Ok(t) => t,
        Err(e) => {
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
    };

    let messages = build_messages(&req, &state).await;

    match run_agentic_loop(&state, &token, &req.model, messages, &mut writer).await {
        Ok(response_text) => {
            let prompt = req.prompt.clone();
            // Serialise memory writes within this process; file-level flock in
            // memory::append handles cross-process protection.
            let mem_guard = state.memory_lock.lock().await;
            let mem_result =
                tokio::task::spawn_blocking(move || memory::append(&prompt, &response_text))
                    .await
                    .unwrap_or_else(|e| Err(anyhow::anyhow!("memory::append panicked: {e}")));
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

    Ok(())
}

// ---------------------------------------------------------------------------
// Agentic loop
// ---------------------------------------------------------------------------

/// Drive the conversation until Copilot responds with `finish_reason: stop`
/// (or an error).  Executes tool calls and feeds results back in a loop.
async fn run_agentic_loop<W>(
    state: &DaemonState,
    token: &str,
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
        let resp =
            copilot::stream_chat(&state.http, token, model, &messages, &schemas, writer).await?;

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

async fn build_messages(req: &Request, state: &DaemonState) -> Vec<Message> {
    let mut system = "You are a helpful, concise AI assistant embedded in a tmux terminal. \
                      Answer in plain text; avoid markdown unless the user asks for it. \
                      You have tools available to inspect the terminal, run commands, \
                      and read or edit files — use them when they help you answer accurately."
        .to_owned();

    if let Some(pane) = &req.tmux_pane {
        system.push_str(&format!(" The user's active tmux pane is {pane}."));
    }

    let mut messages = vec![Message::system(system)];

    // Inject recent conversation history as proper user/assistant turn pairs
    // rather than embedding it in the system prompt.  This preserves role
    // separation and prevents a malicious assistant response stored in memory
    // from being able to override system-level instructions.
    //
    // Reads from the in-memory cache — no disk I/O on the hot path.
    let entries = state.memory_cache.snapshot().await;
    for entry in entries {
        messages.push(Message::user(entry.user));
        messages.push(Message::assistant(Some(entry.assistant), vec![]));
    }

    messages.push(Message::user(req.prompt.clone()));
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
}
