use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

const SOCKET_PATH: &str = "/tmp/tmux-copilot.sock";

// ---------------------------------------------------------------------------
// IPC wire types
// ---------------------------------------------------------------------------

/// A message sent from the client to the daemon.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct Request {
    /// The user's prompt text.
    prompt: String,
    /// Value of $TMUX_PANE at the time the client was invoked, if set.
    tmux_pane: Option<String>,
}

/// A single frame streamed from the daemon back to the client.
/// Using a tagged enum lets us add more variants later (e.g. ToolCall, Done).
#[derive(serde::Serialize, serde::Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Response {
    /// A chunk of text to print immediately.
    Text { chunk: String },
    /// The stream is finished — the client should exit.
    Done,
    /// A hard error the client should display then exit.
    Error { message: String },
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "tmux-copilot",
    version,
    about = "Headless AI assistant for tmux backed by GitHub Copilot"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Start the background daemon that owns the Copilot connection.
    Daemon {
        /// Path to the Unix socket (default: /tmp/tmux-copilot.sock).
        #[arg(long, default_value = SOCKET_PATH)]
        socket: PathBuf,
    },
    /// Send a prompt to the daemon and stream the reply to stdout.
    Ask {
        /// The prompt to send.
        prompt: String,
        /// Path to the Unix socket (default: /tmp/tmux-copilot.sock).
        #[arg(long, default_value = SOCKET_PATH)]
        socket: PathBuf,
    },
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Daemon { socket } => run_daemon(socket).await,
        Command::Ask { prompt, socket } => run_client(socket, prompt).await,
    }
}

// ---------------------------------------------------------------------------
// Daemon
// ---------------------------------------------------------------------------

async fn run_daemon(socket: PathBuf) -> Result<()> {
    // Remove a stale socket file from a previous run.
    if socket.exists() {
        std::fs::remove_file(&socket)
            .with_context(|| format!("removing stale socket {}", socket.display()))?;
    }

    let listener = UnixListener::bind(&socket)
        .with_context(|| format!("binding Unix socket {}", socket.display()))?;

    eprintln!("[daemon] listening on {}", socket.display());

    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                tokio::spawn(async move {
                    if let Err(e) = handle_connection(stream).await {
                        eprintln!("[daemon] connection error: {e:#}");
                    }
                });
            }
            Err(e) => {
                eprintln!("[daemon] accept error: {e}");
            }
        }
    }
}

/// Handle one client connection.
///
/// Protocol (newline-delimited JSON):
///   client -> daemon : one `Request` JSON line
///   daemon -> client : zero or more `Response::Text` lines, then `Response::Done`
async fn handle_connection(stream: UnixStream) -> Result<()> {
    let (reader, mut writer) = tokio::io::split(stream);
    let mut lines = BufReader::new(reader).lines();

    // Read the single request line.
    let line = lines
        .next_line()
        .await
        .context("reading request")?
        .context("client disconnected before sending a request")?;

    let req: Request = serde_json::from_str(&line).context("parsing request JSON")?;

    eprintln!(
        "[daemon] received prompt from pane {:?}: {:?}",
        req.tmux_pane, req.prompt
    );

    // --- Phase 1: echo the prompt back word-by-word to demonstrate streaming ---
    // Phase 2 will replace this with a real Copilot API call.
    let echo_prefix = format!("[echo] {}", req.prompt);
    for word in echo_prefix.split_whitespace() {
        let frame = Response::Text {
            chunk: format!("{word} "),
        };
        let mut line = serde_json::to_string(&frame)?;
        line.push('\n');
        writer.write_all(line.as_bytes()).await?;
    }

    // Signal end-of-stream.
    let done = serde_json::to_string(&Response::Done)?;
    writer.write_all(done.as_bytes()).await?;
    writer.write_all(b"\n").await?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

async fn run_client(socket: PathBuf, prompt: String) -> Result<()> {
    let stream = UnixStream::connect(&socket)
        .await
        .with_context(|| format!("connecting to daemon at {} — is it running?", socket.display()))?;

    let (reader, mut writer) = tokio::io::split(stream);

    // Build and send the request.
    let req = Request {
        prompt,
        tmux_pane: std::env::var("TMUX_PANE").ok(),
    };
    let mut req_line = serde_json::to_string(&req)?;
    req_line.push('\n');
    writer.write_all(req_line.as_bytes()).await?;

    // Stream responses to stdout until Done.
    let mut lines = BufReader::new(reader).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = lines.next_line().await.context("reading response")? {
        let resp: Response = serde_json::from_str(&line).context("parsing response JSON")?;
        match resp {
            Response::Text { chunk } => {
                stdout.write_all(chunk.as_bytes()).await?;
                stdout.flush().await?;
            }
            Response::Done => break,
            Response::Error { message } => {
                anyhow::bail!("daemon error: {message}");
            }
        }
    }

    // Ensure the final newline is visible.
    stdout.write_all(b"\n").await?;

    Ok(())
}
