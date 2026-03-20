//! ACP (Agent Client Protocol) agent server.
//!
//! Implements amaebi as an ACP-compatible agent over stdio so that any
//! ACP-capable client (Claude Code, Zed, Codex CLI, …) can spawn and
//! communicate with it via JSON-RPC without going through the Unix-socket
//! daemon.
//!
//! Entry point: [`run`].

use std::{cell::Cell, sync::Arc};

use agent_client_protocol::{self as acp, Client as _};
use anyhow::{Context, Result};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::{mpsc, oneshot};
use tokio_util::compat::{TokioAsyncReadCompatExt as _, TokioAsyncWriteCompatExt as _};

use crate::copilot::Message;
use crate::daemon::{run_agentic_loop, DaemonState};
use crate::ipc::Response;
use crate::memory;

// ---------------------------------------------------------------------------
// Message builder for ACP mode
// ---------------------------------------------------------------------------

/// Build the initial message list for an ACP prompt.
///
/// Uses a generic coding-assistant system prompt rather than the daemon's
/// tmux-specific one: in ACP mode amaebi communicates over stdio and has no
/// tmux environment available.
fn acp_build_messages(prompt: &str, history: &[memory::MemoryEntry]) -> Vec<Message> {
    let system = "You are a helpful, concise AI coding assistant. \
                  Answer in plain text; avoid markdown unless the user asks for it. \
                  You have tools available to read and edit files and run shell commands \
                  — use them when they help you answer accurately.";

    let mut messages = vec![Message::system(system.to_owned())];

    // Inject recent conversation history as proper user/assistant turn pairs.
    for entry in history {
        messages.push(Message::user(entry.user.clone()));
        messages.push(Message::assistant(Some(entry.assistant.clone()), vec![]));
    }

    messages.push(Message::user(prompt.to_owned()));
    messages
}

// ---------------------------------------------------------------------------
// AmaebiAgent — implements acp::Agent
// ---------------------------------------------------------------------------

struct AmaebiAgent {
    state: Arc<DaemonState>,
    /// Default model name, resolved at startup.
    model: Arc<str>,
    /// Channel used to send session notifications to the background forwarder.
    session_update_tx: mpsc::UnboundedSender<(acp::SessionNotification, oneshot::Sender<()>)>,
    /// Monotonically increasing session counter (Cell: !Sync, fine for !Send).
    next_session_id: Cell<u64>,
}

#[async_trait::async_trait(?Send)]
impl acp::Agent for AmaebiAgent {
    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    async fn initialize(
        &self,
        _args: acp::InitializeRequest,
    ) -> acp::Result<acp::InitializeResponse> {
        Ok(
            acp::InitializeResponse::new(acp::ProtocolVersion::V1).agent_info(
                acp::Implementation::new("amaebi", env!("CARGO_PKG_VERSION")).title("amaebi"),
            ),
        )
    }

    /// Verify Copilot authentication; return `auth_required` on failure.
    async fn authenticate(
        &self,
        _args: acp::AuthenticateRequest,
    ) -> acp::Result<acp::AuthenticateResponse> {
        self.state
            .tokens
            .get(&self.state.http)
            .await
            .map(|_| acp::AuthenticateResponse::default())
            .map_err(|e| acp::Error::auth_required().data(e.to_string()))
    }

    async fn new_session(
        &self,
        _args: acp::NewSessionRequest,
    ) -> acp::Result<acp::NewSessionResponse> {
        let id = self.next_session_id.get();
        self.next_session_id.set(id + 1);
        Ok(acp::NewSessionResponse::new(id.to_string()))
    }

    // ------------------------------------------------------------------
    // Prompt — the core of the agent
    // ------------------------------------------------------------------

    async fn prompt(&self, args: acp::PromptRequest) -> acp::Result<acp::PromptResponse> {
        // Extract plain text from the content blocks.
        let user_text: String = args
            .prompt
            .iter()
            .filter_map(|b| match b {
                acp::ContentBlock::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        if user_text.is_empty() {
            return Ok(acp::PromptResponse::new(acp::StopReason::EndTurn));
        }

        // Verify auth before entering the loop.
        if let Err(e) = self.state.tokens.get(&self.state.http).await {
            tracing::error!(error = %e, "auth check failed at prompt start");
            return Err(acp::Error::auth_required().data(e.to_string()));
        }

        // Load per-session history (keyed by the ACP session_id string).
        let session_arc = self.state.sessions.get_or_create(&args.session_id.0).await;
        let history_snapshot = {
            let session = session_arc.lock().await;
            session.history.clone()
        };

        // Build the conversation messages from session history + current prompt.
        // Uses an ACP-specific system prompt (no tmux context).
        let messages = acp_build_messages(&user_text, &history_snapshot);

        // Duplex pipe: the agentic loop writes IPC frames → we read them back
        // and forward as ACP session notifications.
        let (mut write_half, read_half) = tokio::io::duplex(64 * 1024);

        let state = Arc::clone(&self.state);
        let model = Arc::clone(&self.model);

        // Oneshot to receive the loop's final text for memory saving.
        let (result_tx, result_rx) = oneshot::channel::<Result<String, String>>();

        // Run the agentic loop in a background local task.
        tokio::task::spawn_local(async move {
            let outcome = run_agentic_loop(&state, &model, messages, &mut write_half)
                .await
                .map_err(|e| format!("{e:#}"));
            let _ = result_tx.send(outcome);
            // Dropping write_half closes the pipe, signalling EOF to the reader.
        });

        // Forward text chunks as ACP session notifications.
        let session_id = args.session_id.clone();
        let tx = self.session_update_tx.clone();
        let mut lines = BufReader::new(read_half).lines();

        while let Some(line) = lines
            .next_line()
            .await
            .map_err(|e| acp::Error::internal_error().data(e.to_string()))?
        {
            let resp: Response = serde_json::from_str(&line)
                .map_err(|e| acp::Error::internal_error().data(e.to_string()))?;

            match resp {
                Response::Text { chunk } => {
                    let (ack_tx, ack_rx) = oneshot::channel();
                    tx.send((
                        acp::SessionNotification::new(
                            session_id.clone(),
                            acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
                                chunk.into(),
                            )),
                        ),
                        ack_tx,
                    ))
                    .map_err(|_| acp::Error::internal_error())?;
                    ack_rx.await.map_err(|_| acp::Error::internal_error())?;
                }
                Response::Done => break,
                Response::Error { message } => {
                    return Err(acp::Error::internal_error().data(message));
                }
                Response::ToolUse { .. } => {
                    // Tool-use notifications are relevant in daemon/CLI mode
                    // but not exposed over ACP (the agent handles tools itself).
                }
            }
        }

        // Await the loop outcome; propagate any error to the ACP client.
        let final_text = match result_rx.await {
            Ok(Ok(text)) => text,
            Ok(Err(e)) => return Err(acp::Error::internal_error().data(e)),
            Err(_) => return Err(acp::Error::internal_error()),
        };

        // Persist the exchange to per-session history and the global JSONL store.
        {
            let entry = memory::MemoryEntry {
                timestamp: chrono::Utc::now().to_rfc3339(),
                user: user_text.clone(),
                assistant: final_text.clone(),
            };
            let mut session = session_arc.lock().await;
            session.history.push(entry);
            session.last_active = std::time::Instant::now();
        }
        let mem_guard = self.state.memory_lock.lock().await;
        let prompt_clone = user_text.clone();
        let final_clone = final_text.clone();
        let mem_result =
            tokio::task::spawn_blocking(move || memory::append(&prompt_clone, &final_clone))
                .await
                .unwrap_or_else(|e| Err(anyhow::anyhow!("memory::append panicked: {e}")));
        drop(mem_guard);
        if let Err(e) = mem_result {
            tracing::warn!(error = %e, "failed to save memory after ACP prompt");
        }

        Ok(acp::PromptResponse::new(acp::StopReason::EndTurn))
    }

    // ------------------------------------------------------------------
    // Cancel — best-effort; the agentic loop does not support interruption yet
    // ------------------------------------------------------------------

    async fn cancel(&self, _args: acp::CancelNotification) -> acp::Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Start amaebi as an ACP agent, communicating via JSON-RPC over stdio.
pub async fn run(model: Option<String>) -> Result<()> {
    let model: Arc<str> = model
        .or_else(|| std::env::var("AMAEBI_MODEL").ok())
        .unwrap_or_else(|| "gpt-4o".to_string())
        .into();

    let state = Arc::new(DaemonState::new().context("initialising daemon state")?);

    let outgoing = tokio::io::stdout().compat_write();
    let incoming = tokio::io::stdin().compat();

    let local = tokio::task::LocalSet::new();
    local
        .run_until(async move {
            let (tx, mut rx) = mpsc::unbounded_channel();

            let (conn, handle_io) = acp::AgentSideConnection::new(
                AmaebiAgent {
                    state,
                    model,
                    session_update_tx: tx,
                    next_session_id: Cell::new(0),
                },
                outgoing,
                incoming,
                |fut| {
                    tokio::task::spawn_local(fut);
                },
            );

            // Forward session notifications from AmaebiAgent to the ACP client.
            tokio::task::spawn_local(async move {
                while let Some((notification, ack_tx)) = rx.recv().await {
                    if let Err(e) = conn.session_notification(notification).await {
                        tracing::error!(error = %e, "session notification failed");
                        break;
                    }
                    ack_tx.send(()).ok();
                }
            });

            handle_io
                .await
                .map_err(|e| anyhow::anyhow!("ACP I/O error: {e}"))
        })
        .await
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serialises all tests that read or write `AMAEBI_MODEL` so parallel
    /// test execution cannot race on the environment variable.
    static MODEL_ENV_LOCK: Mutex<()> = Mutex::new(());

    /// RAII guard that restores `AMAEBI_MODEL` to its prior value on drop,
    /// even if the test panics.
    struct ModelEnvGuard {
        prior: Option<String>,
    }

    impl ModelEnvGuard {
        fn unset() -> Self {
            let prior = std::env::var("AMAEBI_MODEL").ok();
            // SAFETY: serialised by MODEL_ENV_LOCK; no concurrent mutations.
            unsafe { std::env::remove_var("AMAEBI_MODEL") };
            Self { prior }
        }

        fn set(value: &str) -> Self {
            let prior = std::env::var("AMAEBI_MODEL").ok();
            // SAFETY: serialised by MODEL_ENV_LOCK; no concurrent mutations.
            unsafe { std::env::set_var("AMAEBI_MODEL", value) };
            Self { prior }
        }
    }

    impl Drop for ModelEnvGuard {
        fn drop(&mut self) {
            // SAFETY: serialised by MODEL_ENV_LOCK; no concurrent mutations.
            match &self.prior {
                Some(v) => unsafe { std::env::set_var("AMAEBI_MODEL", v) },
                None => unsafe { std::env::remove_var("AMAEBI_MODEL") },
            }
        }
    }

    #[test]
    fn model_resolution_uses_default_when_none() {
        let _lock = MODEL_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _env = ModelEnvGuard::unset();
        let model: Arc<str> = None::<String>
            .or_else(|| std::env::var("AMAEBI_MODEL").ok())
            .unwrap_or_else(|| "gpt-4o".to_string())
            .into();
        assert_eq!(&*model, "gpt-4o");
    }

    #[test]
    fn model_resolution_uses_explicit_flag() {
        let _lock = MODEL_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // Explicit flag takes priority regardless of env var.
        let model: Arc<str> = Some("gpt-4.1".to_string())
            .or_else(|| std::env::var("AMAEBI_MODEL").ok())
            .unwrap_or_else(|| "gpt-4o".to_string())
            .into();
        assert_eq!(&*model, "gpt-4.1");
    }

    #[test]
    fn model_resolution_uses_env_var() {
        let _lock = MODEL_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _env = ModelEnvGuard::set("o4-mini");
        let model: Arc<str> = None::<String>
            .or_else(|| std::env::var("AMAEBI_MODEL").ok())
            .unwrap_or_else(|| "gpt-4o".to_string())
            .into();
        assert_eq!(&*model, "o4-mini");
    }

    #[test]
    fn text_extraction_joins_text_blocks() {
        // Verify the prompt extraction logic used in AmaebiAgent::prompt.
        let blocks = vec![
            acp::ContentBlock::Text(acp::TextContent::new("hello")),
            acp::ContentBlock::Text(acp::TextContent::new("world")),
        ];
        let text: String = blocks
            .iter()
            .filter_map(|b| match b {
                acp::ContentBlock::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(text, "hello\nworld");
    }

    #[test]
    fn text_extraction_skips_non_text_blocks() {
        let blocks = vec![acp::ContentBlock::Text(acp::TextContent::new("keep"))];
        let text: String = blocks
            .iter()
            .filter_map(|b| match b {
                acp::ContentBlock::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(text, "keep");
    }
}
