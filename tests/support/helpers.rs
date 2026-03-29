//! Integration-test helpers: start a daemon, connect, send messages.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::process::{Child, Command};

// ---------------------------------------------------------------------------
// IPC types (mirrors src/ipc.rs — keep in sync if the wire format changes)
// ---------------------------------------------------------------------------

/// A message sent from the client to the daemon.
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    Chat {
        prompt: String,
        tmux_pane: Option<String>,
        session_id: Option<String>,
        model: String,
    },
    Steer {
        session_id: String,
        message: String,
    },
}

/// A single frame streamed from the daemon back to the client.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    Text { chunk: String },
    Done,
    Error { message: String },
    ToolUse { name: String, detail: String },
    SteerAck,
    DetachAccepted { session_id: String },
    MemoryEntry { role: String, content: String },
    Compacting,
    WaitingForInput { prompt: String },
}

// ---------------------------------------------------------------------------
// Daemon lifecycle
// ---------------------------------------------------------------------------

/// A running daemon process and the temp home dir it uses.
pub struct DaemonHandle {
    pub socket: PathBuf,
    pub child: Child,
    _home: TempDir,
    _socket_dir: TempDir,
}

impl Drop for DaemonHandle {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

/// Start the `amaebi daemon` binary pointing at `mock_url` for the Copilot API.
pub async fn start_daemon(mock_url: &str) -> Result<DaemonHandle> {
    let home_dir = TempDir::new().context("creating temp HOME")?;
    let amaebi_dir = home_dir.path().join(".amaebi");
    std::fs::create_dir_all(&amaebi_dir).context("creating .amaebi dir")?;
    std::fs::write(
        amaebi_dir.join("hosts.json"),
        r#"{"github.com": {"oauth_token": "test-oauth-token", "user": "test-user"}}"#,
    )
    .context("writing dummy hosts.json")?;

    let socket_dir = TempDir::new().context("creating temp socket dir")?;
    let socket = socket_dir.path().join("amaebi.sock");

    // Locate the compiled binary next to the test runner.
    let exe = find_amaebi_binary()?;

    let child = Command::new(&exe)
        .arg("daemon")
        .arg("--socket")
        .arg(&socket)
        .env("HOME", home_dir.path())
        .env("AMAEBI_COPILOT_URL", mock_url)
        .env("AMAEBI_COPILOT_TOKEN", "test-api-token")
        .env("RUST_LOG", "error")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .with_context(|| format!("spawning daemon: {}", exe.display()))?;

    // Wait for the socket to appear (up to 5 s).
    for _ in 0..50 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        if socket.exists() {
            break;
        }
    }
    anyhow::ensure!(
        socket.exists(),
        "daemon socket did not appear within 5 s: {}",
        socket.display()
    );

    Ok(DaemonHandle {
        socket,
        child,
        _home: home_dir,
        _socket_dir: socket_dir,
    })
}

fn find_amaebi_binary() -> Result<PathBuf> {
    // The test binary is at <target>/<profile>/deps/integration_tests-<hash>.
    // The daemon binary is at <target>/<profile>/amaebi.
    let test_exe = std::env::current_exe().context("current_exe")?;
    let profile_dir = test_exe
        .parent()
        .and_then(|p| p.parent()) // deps -> profile dir
        .context("unexpected test exe path")?;

    let candidate = profile_dir.join("amaebi");
    if candidate.exists() {
        return Ok(candidate);
    }
    // Fallback: try profile_dir itself (in case test binary is directly in profile dir)
    let candidate2 = test_exe.parent().context("no parent")?.join("amaebi");
    if candidate2.exists() {
        return Ok(candidate2);
    }

    anyhow::bail!(
        "could not find amaebi binary; searched {} and {}",
        candidate.display(),
        candidate2.display()
    )
}

// ---------------------------------------------------------------------------
// Client helpers
// ---------------------------------------------------------------------------

pub struct ClientHandle {
    pub socket: PathBuf,
}

pub fn connect_client(socket: &Path) -> ClientHandle {
    ClientHandle {
        socket: socket.to_path_buf(),
    }
}

/// Send a chat message and collect all response frames until `Done` or `Error`.
pub async fn send_message(client: &ClientHandle, prompt: &str) -> Result<Vec<Response>> {
    send_request(
        client,
        &Request::Chat {
            prompt: prompt.to_string(),
            tmux_pane: None,
            session_id: None,
            model: "gpt-4o".to_string(),
        },
    )
    .await
}

/// Send a steer message to the daemon (fire-and-forget).
#[allow(dead_code)]
pub async fn send_steer(client: &ClientHandle, session_id: &str, message: &str) -> Result<()> {
    let stream = UnixStream::connect(&client.socket)
        .await
        .context("connecting to daemon")?;
    let (_, mut writer) = tokio::io::split(stream);
    let req = Request::Steer {
        session_id: session_id.to_string(),
        message: message.to_string(),
    };
    let mut line = serde_json::to_string(&req)?;
    line.push('\n');
    writer.write_all(line.as_bytes()).await?;
    Ok(())
}

/// Send an arbitrary request and collect all response frames.
pub async fn send_request(client: &ClientHandle, req: &Request) -> Result<Vec<Response>> {
    let stream = UnixStream::connect(&client.socket)
        .await
        .context("connecting to daemon socket")?;
    let (reader, mut writer) = tokio::io::split(stream);

    let mut line = serde_json::to_string(req).context("serialising request")?;
    line.push('\n');
    writer
        .write_all(line.as_bytes())
        .await
        .context("writing request")?;

    let mut responses = Vec::new();
    let mut lines = BufReader::new(reader).lines();
    while let Some(l) = lines.next_line().await.context("reading response line")? {
        if l.is_empty() {
            continue;
        }
        let frame: Response =
            serde_json::from_str(&l).with_context(|| format!("parsing response: {l:?}"))?;
        let done = matches!(frame, Response::Done | Response::Error { .. });
        responses.push(frame);
        if done {
            break;
        }
    }
    Ok(responses)
}

/// Collect all text chunks from a list of responses into a single string.
pub fn collect_text(responses: &[Response]) -> String {
    responses
        .iter()
        .filter_map(|r| {
            if let Response::Text { chunk } = r {
                Some(chunk.as_str())
            } else {
                None
            }
        })
        .collect()
}
