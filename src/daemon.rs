use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

use crate::auth::{amaebi_home, TokenCache};
use crate::copilot::{self, ApiToolCall, ApiToolCallFunction, FinishReason, Message};
use crate::ipc::{write_frame, Request, Response};
use crate::memory_db;
use crate::tools::{self, ToolExecutor};

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
                    store_conversation(&state, &prompt, &response_text).await;
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
// Memory helpers — canonical DB access for daemon and ACP agent
// ---------------------------------------------------------------------------

/// Retrieve conversation context for `prompt` from SQLite.
///
/// Returns the last 4 turns (recency) plus up to 10 FTS-relevant entries,
/// deduplicated and sorted chronologically, as `Message` values ready for
/// injection into a Copilot API request.
pub(crate) async fn retrieve_memory_context(state: &DaemonState, prompt: &str) -> Vec<Message> {
    let db = Arc::clone(&state.db);
    let prompt_owned = prompt.to_owned();
    match tokio::task::spawn_blocking(move || {
        let conn = db.lock().unwrap_or_else(|p| p.into_inner());
        memory_db::retrieve_context(&conn, &prompt_owned, 4, 10)
    })
    .await
    {
        Ok(Ok(entries)) => entries
            .into_iter()
            .map(|e| {
                let content = truncate_chars(e.content, MAX_HISTORY_CHARS);
                if e.role == "user" {
                    Message::user(content)
                } else {
                    Message::assistant(Some(content), vec![])
                }
            })
            .collect(),
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "failed to load memory context from SQLite");
            vec![]
        }
        Err(e) => {
            tracing::warn!(error = %e, "memory context load task panicked");
            vec![]
        }
    }
}

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
                            truncate_chars(output, MAX_TOOL_OUTPUT_CHARS)
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

    inject_skill_files(&mut messages).await;

    for msg in retrieve_memory_context(state, prompt).await {
        messages.push(msg);
    }

    messages.push(Message::user(prompt.to_owned()));
    messages
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
    const MARKER_LEN: usize = 12; // "…[truncated]" is 12 chars (… = 1 char)
    if max <= MARKER_LEN {
        return MARKER.chars().take(max).collect();
    }
    let content_len = max - MARKER_LEN;
    let mut out: String = s.chars().take(content_len).collect();
    out.push_str(MARKER);
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
