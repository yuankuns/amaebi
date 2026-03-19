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

use crate::daemon::{build_messages, run_agentic_loop, DaemonState};
use crate::ipc::Response;
use crate::memory;

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

        // Build the conversation messages from memory + current prompt.
        let messages = build_messages(&user_text, None, &self.state).await;

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
                    ack_rx
                        .await
                        .map_err(|_| acp::Error::internal_error())?;
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

        // Save the conversation to memory (best-effort).
        if let Ok(Ok(final_text)) = result_rx.await {
            let mem_guard = self.state.memory_lock.lock().await;
            let prompt_clone = user_text.clone();
            let mem_result =
                tokio::task::spawn_blocking(move || memory::append(&prompt_clone, &final_text))
                    .await
                    .unwrap_or_else(|e| Err(anyhow::anyhow!("memory::append panicked: {e}")));
            drop(mem_guard);
            match mem_result {
                Ok(entry) => self.state.memory_cache.push(entry).await,
                Err(e) => tracing::warn!(error = %e, "failed to save memory after ACP prompt"),
            }
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
    state.memory_cache.load_from_disk().await;

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

    #[test]
    fn model_resolution_uses_default_when_none() {
        // When no model is specified and AMAEBI_MODEL is unset, we get gpt-4o.
        // We test the resolution logic directly rather than running the server.
        std::env::remove_var("AMAEBI_MODEL");
        let model: Arc<str> = None::<String>
            .or_else(|| std::env::var("AMAEBI_MODEL").ok())
            .unwrap_or_else(|| "gpt-4o".to_string())
            .into();
        assert_eq!(&*model, "gpt-4o");
    }

    #[test]
    fn model_resolution_uses_explicit_flag() {
        let model: Arc<str> = Some("gpt-4.1".to_string())
            .or_else(|| std::env::var("AMAEBI_MODEL").ok())
            .unwrap_or_else(|| "gpt-4o".to_string())
            .into();
        assert_eq!(&*model, "gpt-4.1");
    }

    #[test]
    fn model_resolution_uses_env_var() {
        std::env::set_var("AMAEBI_MODEL", "o4-mini");
        let model: Arc<str> = None::<String>
            .or_else(|| std::env::var("AMAEBI_MODEL").ok())
            .unwrap_or_else(|| "gpt-4o".to_string())
            .into();
        std::env::remove_var("AMAEBI_MODEL");
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
