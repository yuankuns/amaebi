//! Integration-test helpers: start a daemon, connect, send messages.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
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
    Resume {
        prompt: String,
        tmux_pane: Option<String>,
        session_id: String,
        model: String,
    },
    Workflow {
        name: String,
        args: serde_json::Map<String, serde_json::Value>,
        model: String,
        session_id: Option<String>,
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
    pub home: PathBuf,
    pub child: Child,
    pub home_dir: TempDir,
    _socket_dir: TempDir,
}

impl Drop for DaemonHandle {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

impl DaemonHandle {
    /// Kill the daemon and return `(home_path, home_dir)` so the caller can
    /// keep the SQLite database alive while spinning up a new daemon at the
    /// same home directory.
    ///
    /// The socket temp-dir is cleaned up as a side effect.
    pub async fn kill_and_keep_home(mut self) -> (PathBuf, TempDir) {
        let _ = self.child.start_kill();
        let _ = self.child.wait().await;
        // We can't partially move out of a Drop type, so take the TempDir via
        // a manual replacement and the PathBuf is already Clone.
        let home_path = self.home.clone();
        // Replace home_dir with a fresh TempDir so `self` can be dropped
        // without removing the actual home directory.
        let real_home_dir = std::mem::replace(&mut self.home_dir, TempDir::new().expect("tmp"));
        // Drop `self` normally — child already waited above; socket dir cleaned.
        drop(self);
        (home_path, real_home_dir)
    }
}

/// Start the `amaebi daemon` binary with additional environment variables.
pub async fn start_daemon_with_env(
    mock_url: &str,
    extra_env: &[(&str, &str)],
) -> Result<DaemonHandle> {
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

    let exe = find_amaebi_binary()?;

    let mut cmd = Command::new(&exe);
    cmd.arg("daemon")
        .arg("--socket")
        .arg(&socket)
        .env("HOME", home_dir.path())
        .env("AMAEBI_COPILOT_URL", mock_url)
        .env("AMAEBI_COPILOT_TOKEN", "test-api-token")
        .env("RUST_LOG", "error")
        // Bypass any corporate/system HTTP proxy for localhost so that the
        // daemon's reqwest client connects directly to the mock server.
        .env("no_proxy", "127.0.0.1,localhost")
        .env("NO_PROXY", "127.0.0.1,localhost")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());

    for (k, v) in extra_env {
        cmd.env(k, v);
    }

    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawning daemon: {}", exe.display()))?;

    // Take stderr *before* moving `child` into `DaemonHandle` so we can
    // drain it in the startup-timeout error path.
    let mut stderr_capture = child.stderr.take();

    for _ in 0..50 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        if socket.exists() {
            break;
        }
    }

    if !socket.exists() {
        let mut stderr_text = String::new();
        if let Some(ref mut stderr) = stderr_capture {
            let mut buf = Vec::new();
            let _ = tokio::time::timeout(Duration::from_millis(500), stderr.read_to_end(&mut buf))
                .await;
            stderr_text = String::from_utf8_lossy(&buf).into_owned();
        }
        let stderr_display = if stderr_text.is_empty() {
            "(empty)".to_string()
        } else {
            stderr_text
        };
        anyhow::bail!(
            "daemon socket did not appear within 5 s: {}\ndaemon stderr:\n{}",
            socket.display(),
            stderr_display
        );
    }

    Ok(DaemonHandle {
        socket,
        home: home_dir.path().to_path_buf(),
        child,
        home_dir,
        _socket_dir: socket_dir,
    })
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
        // Bypass any corporate/system HTTP proxy for localhost.
        .env("no_proxy", "127.0.0.1,localhost")
        .env("NO_PROXY", "127.0.0.1,localhost")
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
        home: home_dir.path().to_path_buf(),
        child,
        home_dir,
        _socket_dir: socket_dir,
    })
}

/// Start a second daemon using an *existing* home directory (to share the SQLite DB)
/// with different environment variables.  The caller must ensure the first daemon has
/// been stopped before calling this, so that the socket lock and SQLite are free.
pub async fn start_daemon_at_home_with_env(
    home_path: &Path,
    mock_url: &str,
    extra_env: &[(&str, &str)],
) -> Result<(PathBuf, Child, TempDir)> {
    let socket_dir = TempDir::new().context("creating temp socket dir")?;
    let socket = socket_dir.path().join("amaebi.sock");

    let exe = find_amaebi_binary()?;

    let mut cmd = Command::new(&exe);
    cmd.arg("daemon")
        .arg("--socket")
        .arg(&socket)
        .env("HOME", home_path)
        .env("AMAEBI_COPILOT_URL", mock_url)
        .env("AMAEBI_COPILOT_TOKEN", "test-api-token")
        .env("RUST_LOG", "error")
        // Bypass any corporate/system HTTP proxy for localhost.
        .env("no_proxy", "127.0.0.1,localhost")
        .env("NO_PROXY", "127.0.0.1,localhost")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    for (k, v) in extra_env {
        cmd.env(k, v);
    }

    let child = cmd
        .spawn()
        .with_context(|| format!("spawning daemon at home: {}", exe.display()))?;
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

    Ok((socket, child, socket_dir))
}

fn find_amaebi_binary() -> Result<PathBuf> {
    // CARGO_BIN_EXE_amaebi is set by Cargo for integration tests whose package
    // declares a [[bin]] target named "amaebi".  It resolves to the absolute
    // path of the compiled binary at compile time, so no runtime heuristics
    // are needed.
    Ok(PathBuf::from(env!("CARGO_BIN_EXE_amaebi")))
}

// ---------------------------------------------------------------------------
// Cron helpers
// ---------------------------------------------------------------------------

/// Create a fresh home directory with `.amaebi/hosts.json` pre-populated.
///
/// Use this when you need to pre-seed a cron job (or other amaebi state)
/// before starting the daemon.
pub fn setup_home() -> Result<TempDir> {
    let home_dir = TempDir::new().context("creating temp HOME")?;
    let amaebi_dir = home_dir.path().join(".amaebi");
    std::fs::create_dir_all(&amaebi_dir).context("creating .amaebi dir")?;
    std::fs::write(
        amaebi_dir.join("hosts.json"),
        r#"{"github.com": {"oauth_token": "test-oauth-token", "user": "test-user"}}"#,
    )
    .context("writing dummy hosts.json")?;
    Ok(home_dir)
}

/// Seed a cron job into `<home_path>/.amaebi/cron.db` using the amaebi CLI.
///
/// `HOME` is set to `home_path` so the cron.db lands in the right place.
/// No auth tokens are needed — `cron add` is a pure SQLite write.
pub async fn seed_cron_job(home_path: &Path, description: &str, schedule: &str) -> Result<()> {
    let exe = find_amaebi_binary()?;
    let output = Command::new(&exe)
        .arg("cron")
        .arg("add")
        .arg(description)
        .arg("--cron")
        .arg(schedule)
        .env("HOME", home_path)
        .output()
        .await
        .context("running amaebi cron add")?;
    anyhow::ensure!(
        output.status.success(),
        "amaebi cron add failed (exit {}): {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
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

/// Send a chat message with a specific session_id and model.
pub async fn send_message_with_session(
    client: &ClientHandle,
    prompt: &str,
    session_id: &str,
    model: &str,
) -> Result<Vec<Response>> {
    send_request(
        client,
        &Request::Chat {
            prompt: prompt.to_string(),
            tmux_pane: None,
            session_id: Some(session_id.to_string()),
            model: model.to_string(),
        },
    )
    .await
}

/// Send a resume request with a specific session_id and model.
pub async fn send_resume(
    client: &ClientHandle,
    prompt: &str,
    session_id: &str,
    model: &str,
) -> Result<Vec<Response>> {
    send_request(
        client,
        &Request::Resume {
            prompt: prompt.to_string(),
            tmux_pane: None,
            session_id: session_id.to_string(),
            model: model.to_string(),
        },
    )
    .await
}

/// Send a workflow request and collect all response frames until `Done` or `Error`.
pub async fn send_workflow(
    client: &ClientHandle,
    name: &str,
    args: serde_json::Map<String, serde_json::Value>,
    model: &str,
    session_id: Option<&str>,
) -> Result<Vec<Response>> {
    send_request(
        client,
        &Request::Workflow {
            name: name.to_string(),
            args,
            model: model.to_string(),
            session_id: session_id.map(|s| s.to_string()),
        },
    )
    .await
}

/// Send a chat message and collect all response frames until `Done` or `Error`.
pub async fn send_message(client: &ClientHandle, prompt: &str) -> Result<Vec<Response>> {
    send_request(
        client,
        &Request::Chat {
            prompt: prompt.to_string(),
            tmux_pane: None,
            session_id: None,
            model: "copilot/gpt-4o".to_string(),
        },
    )
    .await
}

/// A live connection to the daemon that can send steer messages on the same
/// underlying Unix socket as the initiating Chat request.
#[allow(dead_code)]
pub struct ChatSession {
    writer: tokio::io::WriteHalf<UnixStream>,
}

impl ChatSession {
    /// Send a `Steer` message on this session's connection.
    #[allow(dead_code)]
    pub async fn steer(&mut self, session_id: &str, message: &str) -> Result<()> {
        let req = Request::Steer {
            session_id: session_id.to_string(),
            message: message.to_string(),
        };
        let mut line = serde_json::to_string(&req)?;
        line.push('\n');
        self.writer.write_all(line.as_bytes()).await?;
        Ok(())
    }
}

/// Send a steer message to the daemon on a *new* connection (fire-and-forget).
///
/// Note: for tests that need Steer delivered on the same connection as Chat,
/// construct a `ChatSession` directly and call `ChatSession::steer`.
#[allow(dead_code)]
pub async fn send_steer(client: &ClientHandle, session_id: &str, message: &str) -> Result<()> {
    // Re-use a ChatSession if you need Steer on the same connection.
    // This helper opens a fresh connection for callers that only need fire-and-forget.
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
    tokio::time::timeout(Duration::from_secs(30), async {
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
        Ok::<(), anyhow::Error>(())
    })
    .await
    .context("send_request timed out after 30 s waiting for Done/Error frame")??;
    Ok(responses)
}

// Long-connection helper for amaebi chat tests
// ---------------------------------------------------------------------------

/// A persistent daemon connection that can send multiple Chat requests on the
/// same socket, simulating what `amaebi chat` does (long connection).
pub struct LongChatConnection {
    writer: tokio::io::WriteHalf<UnixStream>,
    reader: BufReader<tokio::io::ReadHalf<UnixStream>>,
}

impl LongChatConnection {
    /// Open a persistent connection to the daemon.
    pub async fn connect(socket: &Path) -> Result<Self> {
        let stream = UnixStream::connect(socket)
            .await
            .context("connecting to daemon")?;
        let (reader, writer) = tokio::io::split(stream);
        Ok(Self {
            writer,
            reader: BufReader::new(reader),
        })
    }

    /// Read one newline-delimited response frame from the connection.
    async fn read_frame(&mut self) -> Result<Option<Response>> {
        loop {
            let mut line = String::new();
            let n = self.reader.read_line(&mut line).await?;
            if n == 0 {
                return Ok(None); // real EOF
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue; // blank line — keep reading
            }
            let frame: Response = serde_json::from_str(trimmed)
                .with_context(|| format!("parsing response: {trimmed:?}"))?;
            return Ok(Some(frame));
        }
    }

    /// Send a Chat request and collect all response frames until Done/Error.
    pub async fn chat(
        &mut self,
        prompt: &str,
        session_id: &str,
        model: &str,
    ) -> Result<Vec<Response>> {
        let req = Request::Chat {
            prompt: prompt.to_string(),
            tmux_pane: None,
            session_id: Some(session_id.to_string()),
            model: model.to_string(),
        };
        let mut line = serde_json::to_string(&req)?;
        line.push('\n');
        self.writer.write_all(line.as_bytes()).await?;

        let mut responses = Vec::new();
        tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                match self.read_frame().await? {
                    None => {
                        // EOF without Done/Error — daemon crashed or protocol error.
                        anyhow::bail!(
                            "connection closed before Done/Error frame (daemon crash or protocol violation?)"
                        );
                    }
                    Some(frame) => {
                        let done = matches!(frame, Response::Done | Response::Error { .. });
                        responses.push(frame);
                        if done {
                            break;
                        }
                    }
                }
            }
            Ok::<(), anyhow::Error>(())
        })
        .await
        .context("chat turn timed out after 30 s")??;
        Ok(responses)
    }

    /// Send a Steer request on this connection (same socket as the Chat).
    pub async fn steer(&mut self, session_id: &str, message: &str) -> Result<()> {
        let req = Request::Steer {
            session_id: session_id.to_string(),
            message: message.to_string(),
        };
        let mut line = serde_json::to_string(&req)?;
        line.push('\n');
        self.writer.write_all(line.as_bytes()).await?;
        Ok(())
    }
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
