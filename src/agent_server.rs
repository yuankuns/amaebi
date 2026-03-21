//! ACP (Agent Client Protocol) agent server.
//!
//! Implements amaebi as an ACP-compatible agent over stdio so that any
//! ACP-capable client (Claude Code, Zed, Codex CLI, …) can spawn and
//! communicate with it via JSON-RPC without going through the Unix-socket
//! daemon.
//!
//! **Memory architecture**: all SQLite reads and writes for this ACP agent
//! are routed through a running daemon process via the Unix socket (see
//! [`Request::StoreMemory`] and [`Request::RetrieveContext`]).  The ACP agent
//! never opens its own SQLite connection.  If no daemon is reachable, memory
//! operations are skipped (writes logged at warn level, reads at debug) so
//! the ACP session continues uninterrupted.  Note that other CLI subcommands
//! (e.g. `amaebi memory clear`) write to SQLite directly; the daemon is the
//! primary — not exclusive — writer.
//!
//! Entry point: [`run`].

use std::{cell::Cell, path::PathBuf, sync::Arc};

use agent_client_protocol::{self as acp, Client as _};
use anyhow::{Context, Result};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::sync::{mpsc, oneshot};
use tokio_util::compat::{TokioAsyncReadCompatExt as _, TokioAsyncWriteCompatExt as _};

use crate::copilot::Message;
use crate::daemon::{inject_skill_files, run_agentic_loop, DaemonState};
use crate::ipc::{Request, Response};

// ---------------------------------------------------------------------------
// Daemon IPC helpers
// ---------------------------------------------------------------------------

/// Send a [`Request::StoreMemory`] to the daemon and wait for `Done`.
///
/// Best-effort: if the daemon is not reachable the warning is logged and the
/// call returns without error so the ACP session is not disrupted.
async fn ipc_store_memory(socket: &std::path::Path, user: &str, assistant: &str) {
    let stream = match UnixStream::connect(socket).await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                socket = %socket.display(),
                error = %e,
                "could not connect to daemon to store memory; write skipped"
            );
            return;
        }
    };
    let (reader, mut writer) = tokio::io::split(stream);

    let req = Request::StoreMemory {
        user: user.to_owned(),
        assistant: assistant.to_owned(),
    };
    let mut line = match serde_json::to_string(&req) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "failed to serialise StoreMemory request");
            return;
        }
    };
    line.push('\n');

    if writer.write_all(line.as_bytes()).await.is_err() {
        return;
    }
    // Drain the Done response.
    let mut lines = BufReader::new(reader).lines();
    let _ = lines.next_line().await;
}

/// Send a [`Request::RetrieveContext`] to the daemon and collect the returned
/// [`Response::MemoryEntry`] frames as `Message` values.
///
/// Returns an empty list if the daemon is not reachable, so the ACP agent
/// continues without historical context rather than failing.
async fn ipc_retrieve_context(socket: &std::path::Path, prompt: &str) -> Vec<Message> {
    let stream = match UnixStream::connect(socket).await {
        Ok(s) => s,
        Err(e) => {
            tracing::debug!(
                socket = %socket.display(),
                error = %e,
                "daemon not reachable; proceeding without memory context"
            );
            return vec![];
        }
    };
    let (reader, mut writer) = tokio::io::split(stream);

    let req = Request::RetrieveContext {
        prompt: prompt.to_owned(),
    };
    let mut line = match serde_json::to_string(&req) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "failed to serialise RetrieveContext request");
            return vec![];
        }
    };
    line.push('\n');

    if writer.write_all(line.as_bytes()).await.is_err() {
        return vec![];
    }

    let mut messages = Vec::new();
    let mut lines = BufReader::new(reader).lines();
    loop {
        match lines.next_line().await {
            Ok(Some(raw)) => match serde_json::from_str::<Response>(&raw) {
                Ok(Response::MemoryEntry { role, content }) => {
                    if role == "user" {
                        messages.push(Message::user(content));
                    } else if role == "assistant" {
                        messages.push(Message::assistant(Some(content), vec![]));
                    } else {
                        tracing::warn!(role = %role, "unexpected role in MemoryEntry; skipping");
                    }
                }
                Ok(Response::Done) => break,
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "unexpected frame from daemon during RetrieveContext");
                    break;
                }
            },
            Ok(None) => break, // daemon closed connection
            Err(e) => {
                tracing::warn!(error = %e, "I/O error reading daemon response");
                break;
            }
        }
    }
    messages
}

// ---------------------------------------------------------------------------
// Message builder for ACP mode
// ---------------------------------------------------------------------------

/// Build the initial message list for an ACP prompt.
///
/// Uses a generic coding-assistant system prompt rather than the daemon's
/// tmux-specific one.  Memory context is fetched from the daemon via IPC.
async fn acp_build_messages(prompt: &str, socket: &std::path::Path) -> Vec<Message> {
    let system = "You are a helpful, concise AI coding assistant. \
                  Answer in plain text; avoid markdown unless the user asks for it. \
                  You have tools available to read and edit files and run shell commands \
                  — use them when they help you answer accurately.";

    let mut messages = vec![Message::system(system.to_owned())];

    // Inject global config files (AGENTS.md, SOUL.md) from ~/.amaebi/.
    inject_skill_files(&mut messages).await;

    // Retrieve context from the daemon via IPC (no direct SQLite access).
    for msg in ipc_retrieve_context(socket, prompt).await {
        messages.push(msg);
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
    /// Path to the daemon's Unix socket for memory IPC.
    daemon_socket: PathBuf,
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

        // Build conversation messages; memory context fetched from daemon via IPC.
        let messages = acp_build_messages(&user_text, &self.daemon_socket).await;

        // Duplex pipe: the agentic loop writes IPC frames → we read them back
        // and forward as ACP session notifications.
        let (mut write_half, read_half) = tokio::io::duplex(64 * 1024);

        let state = Arc::clone(&self.state);
        let model = Arc::clone(&self.model);

        // Oneshot to receive the loop's final text.
        let (result_tx, result_rx) = oneshot::channel::<Result<String, String>>();

        // Run the agentic loop in a background local task.
        // ACP mode has no steering channel — create a channel and immediately
        // drop the sender so the receiver observes a closed channel.
        let (steer_tx, mut steer_rx) = tokio::sync::mpsc::channel::<String>(1);
        drop(steer_tx);
        tokio::task::spawn_local(async move {
            let outcome =
                run_agentic_loop(&state, &model, messages, &mut write_half, &mut steer_rx)
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
                Response::ToolUse { .. } | Response::MemoryEntry { .. } | Response::WaitingForInput { .. } => {
                    // Not relevant on the ACP forwarding path.
                }
                Response::SteerAck => {
                    // ACP mode never sends Steer requests, so SteerAck should
                    // never arrive here — log and continue.
                    tracing::debug!("unexpected SteerAck in ACP mode");
                }
                Response::DetachAccepted { .. } => {
                    // ACP mode never submits detach requests.
                    tracing::debug!("unexpected DetachAccepted in ACP mode");
                }
            }
        }

        // Await the loop outcome; propagate any error to the ACP client.
        let final_text = match result_rx.await {
            Ok(Ok(text)) => text,
            Ok(Err(e)) => return Err(acp::Error::internal_error().data(e)),
            Err(_) => return Err(acp::Error::internal_error()),
        };

        // Store the exchange via daemon IPC — never writes SQLite directly.
        ipc_store_memory(&self.daemon_socket, &user_text, &final_text).await;

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
///
/// `socket` is the path to a running daemon's Unix socket.  Memory operations
/// are routed through it so the daemon remains the sole SQLite writer.
pub async fn run(model: Option<String>, socket: PathBuf) -> Result<()> {
    let model: Arc<str> = model
        .or_else(|| std::env::var("AMAEBI_MODEL").ok())
        .unwrap_or_else(|| "gpt-4o".to_string())
        .into();

    let state = Arc::new(
        DaemonState::new()
            .await
            .context("initialising daemon state")?,
    );

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
                    daemon_socket: socket,
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

    /// Verify that ipc_store_memory degrades gracefully when no daemon is running.
    #[tokio::test]
    async fn ipc_store_memory_no_daemon_is_silent() {
        let dir = tempfile::TempDir::new().unwrap();
        let socket = dir.path().join("nonexistent.sock");
        // Must not panic or return an error — just logs a warning.
        ipc_store_memory(&socket, "hello", "world").await;
    }

    /// Verify that ipc_retrieve_context returns empty when no daemon is running.
    #[tokio::test]
    async fn ipc_retrieve_context_no_daemon_returns_empty() {
        let dir = tempfile::TempDir::new().unwrap();
        let socket = dir.path().join("nonexistent.sock");
        let msgs = ipc_retrieve_context(&socket, "rust async").await;
        assert!(msgs.is_empty());
    }
}
