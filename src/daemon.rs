use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

use crate::auth::TokenCache;
use crate::copilot::{self, Message};
use crate::ipc::{write_frame, Request, Response};

// ---------------------------------------------------------------------------
// Shared daemon state
// ---------------------------------------------------------------------------

/// State shared across all concurrent client connections.
///
/// Wrapped in `Arc` so each `tokio::spawn`-ed handler gets a cheap clone.
/// Phase 4 will extend this with a `SessionMap` for subagent tracking.
pub struct DaemonState {
    pub http: reqwest::Client,
    pub tokens: TokenCache,
}

impl DaemonState {
    pub fn new() -> Result<Self> {
        let http = reqwest::Client::builder()
            .build()
            .context("building HTTP client")?;
        Ok(Self {
            http,
            tokens: TokenCache::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// Listener loop
// ---------------------------------------------------------------------------

pub async fn run(socket: PathBuf) -> Result<()> {
    // Remove a stale socket file from a previous run.
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

/// Handle one client connection.
///
/// Protocol (newline-delimited JSON):
///   client → daemon : one `Request` JSON line
///   daemon → client : zero or more `Response` frames, ending with `Done`
async fn handle_connection(stream: UnixStream, state: Arc<DaemonState>) -> Result<()> {
    let (reader, mut writer) = tokio::io::split(stream);
    let mut lines = BufReader::new(reader).lines();

    // Read the single request line.
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

    // Fetch (or return cached) Copilot API token.
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

    // Build the initial message list.
    let messages = build_messages(&req);

    // Stream the Copilot response back to the client.
    // Any error mid-stream is forwarded as a Response::Error frame.
    if let Err(e) = copilot::stream_chat(&state.http, &token, messages, &mut writer).await {
        tracing::error!(error = %e, "Copilot stream error");
        write_frame(
            &mut writer,
            &Response::Error {
                message: format!("Copilot error: {e:#}"),
            },
        )
        .await?;
    }

    Ok(())
}

/// Construct the messages list from a client `Request`.
fn build_messages(req: &Request) -> Vec<Message> {
    let mut messages = Vec::new();

    let mut system = "You are a helpful, concise AI assistant embedded in a tmux terminal. \
                      Answer in plain text; avoid markdown unless the user asks for it."
        .to_owned();

    if let Some(pane) = &req.tmux_pane {
        system.push_str(&format!(" The user's active tmux pane is {pane}."));
    }

    messages.push(Message::system(system));
    messages.push(Message::user(req.prompt.clone()));
    messages
}
