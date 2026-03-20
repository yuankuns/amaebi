use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
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
    /// Serialises concurrent SQLite writes within this process so that
    /// parallel client connections cannot race on memory storage.
    pub memory_lock: tokio::sync::Mutex<()>,
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
            let db_path = state.db_path.clone();
            let result = tokio::task::spawn_blocking(move || {
                let conn = memory_db::init_db(&db_path)?;
                memory_db::clear(&conn)
            })
            .await
            .unwrap_or_else(|e| Err(anyhow::anyhow!("DB clear panicked: {e}")));
            if let Err(e) = result {
                tracing::warn!(error = %e, "failed to clear memory DB");
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
    let db_path = state.db_path.clone();
    let prompt_owned = prompt.to_owned();
    match tokio::task::spawn_blocking(move || {
        let conn = memory_db::init_db(&db_path)?;
        memory_db::retrieve_context(&conn, &prompt_owned, 4, 10)
    })
    .await
    {
        Ok(Ok(entries)) => entries
            .into_iter()
            .map(|e| {
                if e.role == "user" {
                    Message::user(e.content)
                } else {
                    Message::assistant(Some(e.content), vec![])
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
/// Acquires `memory_lock` to serialise concurrent writes within this process.
/// Best-effort: errors are logged but not propagated.
pub(crate) async fn store_conversation(state: &DaemonState, user: &str, assistant: &str) {
    let _guard = state.memory_lock.lock().await;
    let db_path = state.db_path.clone();
    let user_owned = user.to_owned();
    let assistant_owned = assistant.to_owned();
    let result = tokio::task::spawn_blocking(move || {
        let timestamp = chrono::Utc::now().to_rfc3339();
        let conn = memory_db::init_db(&db_path)?;
        memory_db::store_memory(&conn, &timestamp, "", "user", &user_owned, "")?;
        memory_db::store_memory(&conn, &timestamp, "", "assistant", &assistant_owned, "")
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

/// Read skill/config files from `~/.amaebi/` and inject them as system messages.
///
/// - `AGENTS.md` and `SOUL.md` are loaded directly from `~/.amaebi/`.
/// - Skills are loaded from `~/.amaebi/skills/<name>/SKILL.md` (one per subdirectory).
///   Each skill is injected with the header `## Skill: <name>`.
///
/// Files/directories that do not exist are silently skipped.
/// Empty or whitespace-only files are skipped.
/// Skills are injected in sorted directory-name order for determinism.
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
    // Fixed config files loaded from amaebi_home.
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

    // Skills: each subdirectory of ~/.amaebi/skills/ may contain a SKILL.md.
    let skills_dir = amaebi_home.join("skills");
    let mut read_dir = match tokio::fs::read_dir(&skills_dir).await {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
        Err(e) => {
            tracing::debug!(dir = %skills_dir.display(), error = %e, "could not read skills directory");
            return;
        }
    };

    // Collect (skill_name, content) pairs for sorting.
    let mut skills: Vec<(String, String)> = vec![];
    while let Ok(Some(entry)) = read_dir.next_entry().await {
        let ft = match entry.file_type().await {
            Ok(ft) => ft,
            Err(_) => continue,
        };
        if !ft.is_dir() {
            continue;
        }
        let skill_name = match entry.file_name().into_string() {
            Ok(n) => n,
            Err(_) => continue,
        };
        let skill_md = entry.path().join("SKILL.md");
        match tokio::fs::read_to_string(&skill_md).await {
            Ok(content) => {
                let trimmed = content.trim();
                if !trimmed.is_empty() {
                    skills.push((skill_name, trimmed.to_owned()));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                tracing::debug!(file = %skill_md.display(), error = %e, "could not read SKILL.md");
            }
        }
    }

    // Deterministic injection order.
    skills.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, content) in skills {
        messages.push(Message::system(format!("## Skill: {name}\n\n{content}")));
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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

    #[tokio::test]
    async fn skill_dirs_injected_with_header() {
        let dir = tempfile::TempDir::new().unwrap();
        let skills = dir.path().join("skills");
        std::fs::create_dir_all(skills.join("my-skill")).unwrap();
        std::fs::write(skills.join("my-skill/SKILL.md"), "do stuff").unwrap();

        let mut messages: Vec<Message> = vec![];
        inject_skill_files_from(&mut messages, dir.path()).await;

        assert_eq!(messages.len(), 1);
        let body = messages[0].content.as_deref().unwrap_or("");
        assert!(
            body.contains("## Skill: my-skill"),
            "header missing: {body}"
        );
        assert!(body.contains("do stuff"), "content missing: {body}");
    }

    #[tokio::test]
    async fn skill_dirs_injected_in_sorted_order() {
        let dir = tempfile::TempDir::new().unwrap();
        let skills = dir.path().join("skills");
        for name in &["zzz", "aaa", "mmm"] {
            std::fs::create_dir_all(skills.join(name)).unwrap();
            std::fs::write(
                skills.join(name).join("SKILL.md"),
                format!("{name} content"),
            )
            .unwrap();
        }

        let mut messages: Vec<Message> = vec![];
        inject_skill_files_from(&mut messages, dir.path()).await;

        assert_eq!(messages.len(), 3);
        let names: Vec<&str> = messages
            .iter()
            .map(|m| {
                let body = m.content.as_deref().unwrap_or("");
                if body.contains("aaa") {
                    "aaa"
                } else if body.contains("mmm") {
                    "mmm"
                } else {
                    "zzz"
                }
            })
            .collect();
        assert_eq!(names, vec!["aaa", "mmm", "zzz"]);
    }

    #[tokio::test]
    async fn skill_dir_without_skill_md_is_skipped() {
        let dir = tempfile::TempDir::new().unwrap();
        let skills = dir.path().join("skills");
        // Subdirectory exists but has no SKILL.md.
        std::fs::create_dir_all(skills.join("empty-skill")).unwrap();

        let mut messages: Vec<Message> = vec![];
        inject_skill_files_from(&mut messages, dir.path()).await;
        assert!(messages.is_empty());
    }

    #[tokio::test]
    async fn skill_dir_non_directory_files_are_skipped() {
        let dir = tempfile::TempDir::new().unwrap();
        let skills = dir.path().join("skills");
        std::fs::create_dir_all(&skills).unwrap();
        // A plain file at the skill level — must be ignored.
        std::fs::write(skills.join("not-a-dir.md"), "content").unwrap();
        // A real skill directory.
        std::fs::create_dir_all(skills.join("real-skill")).unwrap();
        std::fs::write(skills.join("real-skill/SKILL.md"), "real content").unwrap();

        let mut messages: Vec<Message> = vec![];
        inject_skill_files_from(&mut messages, dir.path()).await;

        assert_eq!(messages.len(), 1);
        assert!(messages[0]
            .content
            .as_deref()
            .unwrap_or("")
            .contains("real content"));
    }

    #[tokio::test]
    async fn fixed_files_and_skills_combined() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "agent rules").unwrap();
        let skills = dir.path().join("skills");
        std::fs::create_dir_all(skills.join("alpha")).unwrap();
        std::fs::write(skills.join("alpha/SKILL.md"), "alpha instructions").unwrap();

        let mut messages: Vec<Message> = vec![];
        inject_skill_files_from(&mut messages, dir.path()).await;

        // AGENTS.md first, then skill alpha.
        assert_eq!(messages.len(), 2);
        let body0 = messages[0].content.as_deref().unwrap_or("");
        let body1 = messages[1].content.as_deref().unwrap_or("");
        assert!(body0.contains("## Agent Guidelines"));
        assert!(body1.contains("## Skill: alpha"));
    }
}
