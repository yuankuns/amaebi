use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

use crate::auth::TokenCache;
use crate::copilot::{self, ApiToolCall, ApiToolCallFunction, FinishReason, Message};
use crate::ipc::{write_frame, Request, Response};
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

    tracing::info!(
        pane = ?req.tmux_pane,
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

    let messages = build_messages(&req);

    if let Err(e) = run_agentic_loop(&state, &token, messages, &mut writer).await {
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
    mut messages: Vec<Message>,
    writer: &mut W,
) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let schemas = tools::tool_schemas();
    // Guard against runaway tool loops (e.g. a broken tool that always returns
    // an error the model retries indefinitely).
    const MAX_TOOL_ROUNDS: usize = 10;
    let mut tool_rounds = 0;

    loop {
        let resp = copilot::stream_chat(&state.http, token, &messages, &schemas, writer).await?;

        match resp.finish_reason {
            FinishReason::Stop | FinishReason::Length => {
                break;
            }

            FinishReason::ToolCalls => {
                tool_rounds += 1;
                if tool_rounds > MAX_TOOL_ROUNDS {
                    tracing::warn!("exceeded maximum tool rounds ({MAX_TOOL_ROUNDS}), stopping");
                    break;
                }

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
                    tracing::info!(tool = %tc.name, "executing tool");

                    // Notify the client so it can show progress.
                    write_frame(
                        writer,
                        &Response::ToolUse {
                            name: tc.name.clone(),
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
                break;
            }
        }
    }

    write_frame(writer, &Response::Done).await
}

// ---------------------------------------------------------------------------
// Message construction
// ---------------------------------------------------------------------------

fn build_messages(req: &Request) -> Vec<Message> {
    let mut system = "You are a helpful, concise AI assistant embedded in a tmux terminal. \
                      Answer in plain text; avoid markdown unless the user asks for it. \
                      You have tools available to inspect the terminal, run commands, \
                      and read or edit files — use them when they help you answer accurately."
        .to_owned();

    if let Some(pane) = &req.tmux_pane {
        system.push_str(&format!(" The user's active tmux pane is {pane}."));
    }

    vec![Message::system(system), Message::user(req.prompt.clone())]
}
