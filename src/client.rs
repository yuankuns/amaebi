use anyhow::{Context, Result};
use std::io::IsTerminal as _;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::signal::unix::{signal, SignalKind};

use crate::ipc::{Request, Response};
use crate::session;

/// How long the user has to press Ctrl-C a second time to exit.
const DOUBLE_CTRLC_WINDOW: Duration = Duration::from_secs(2);

/// Returned when the user presses Ctrl-C twice within [`DOUBLE_CTRLC_WINDOW`].
///
/// `main` catches this and exits with code 130 (the SIGINT convention).
#[derive(Debug)]
pub struct Interrupted;

impl std::fmt::Display for Interrupted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "interrupted by user")
    }
}

impl std::error::Error for Interrupted {}

pub async fn run(socket: PathBuf, prompt: String, model: Option<String>) -> Result<()> {
    // Register the SIGINT handler before connecting so double-Ctrl-C applies
    // for the entire command lifecycle, including the connection attempt.
    let mut sigint = signal(SignalKind::interrupt()).context("setting up SIGINT handler")?;

    let stream = connect_or_start_daemon(&socket).await?;

    let (reader, mut writer) = tokio::io::split(stream);

    // Resolve model: CLI flag > AMAEBI_MODEL env var > default.
    let model = model
        .or_else(|| std::env::var("AMAEBI_MODEL").ok())
        .unwrap_or_else(|| "gpt-4o".to_string());

    // Resolve the session UUID for the current working directory.
    // Wrapped in spawn_blocking because session::get_or_create does file I/O.
    let cwd = std::env::current_dir().context("getting current directory")?;
    let session_id = tokio::task::spawn_blocking(move || session::get_or_create(&cwd))
        .await
        .context("session::get_or_create panicked")?
        .unwrap_or_else(|e| {
            tracing::warn!(error = %e, "failed to resolve session id; using \"global\"");
            "global".to_string()
        });

    // Keep a copy for the steering requests and the exit footer.
    let session_id_copy = session_id.clone();

    // Build and send the request as a single JSON line.
    let req = Request::Chat {
        prompt,
        tmux_pane: std::env::var("TMUX_PANE").ok(),
        session_id: Some(session_id),
        model,
    };
    let mut req_line = serde_json::to_string(&req).context("serializing request")?;
    req_line.push('\n');
    writer
        .write_all(req_line.as_bytes())
        .await
        .context("sending request to daemon")?;

    // Stream responses to stdout until Done or Error.
    // Interleave with SIGINT so we can implement double-Ctrl-C-to-exit.
    // When stdout is a terminal, also read stdin so the user can steer the
    // agent mid-flight by typing a line and pressing Enter.
    let mut lines = BufReader::new(reader).lines();
    let mut stdout = tokio::io::stdout();
    // Timestamp of the first Ctrl-C press; None means no pending first press.
    let mut last_ctrl_c: Option<Instant> = None;

    // Stdin reader — only created when stdin is a TTY (interactive terminal).
    // Piped invocations (`echo "fix" | amaebi ask "..."`) have stdin as a
    // pipe, not a TTY, so we skip the reader to avoid blocking on EOF.
    let use_stdin = std::io::stdin().is_terminal();
    let mut stdin_lines: Option<BufReader<tokio::io::Stdin>> = if use_stdin {
        Some(BufReader::new(tokio::io::stdin()))
    } else {
        None
    };

    loop {
        tokio::select! {
            biased;

            // Handle Ctrl-C (SIGINT).
            result = sigint.recv() => {
                // recv() returns None when the signal stream is closed; treat
                // that as a cue to stop rather than spinning the loop forever.
                let Some(_) = result else { break; };

                // Capture the time once and reuse for both the window check
                // and recording the press, so both operations see the same clock.
                let now = Instant::now();

                if is_within_window(last_ctrl_c, now, DOUBLE_CTRLC_WINDOW) {
                    // Second Ctrl-C within the window — flush buffers then signal exit.
                    eprintln!();
                    let _ = stdout.flush().await;
                    let _ = tokio::io::stderr().flush().await;
                    return Err(anyhow::Error::new(Interrupted));
                }
                // First press (or expired window): remind the user and record time.
                eprintln!(
                    "\nPress Ctrl-C again within {}s to exit",
                    DOUBLE_CTRLC_WINDOW.as_secs()
                );
                last_ctrl_c = Some(now);
            }

            // Handle the next response frame from the daemon.
            result = lines.next_line() => {
                let line = result.context("reading response from daemon")?;
                let line = match line {
                    Some(l) => l,
                    None => break,
                };
                let resp: Response =
                    serde_json::from_str(&line).context("parsing response frame")?;
                match resp {
                    Response::Text { chunk } => {
                        stdout
                            .write_all(chunk.as_bytes())
                            .await
                            .context("writing to stdout")?;
                        stdout.flush().await.context("flushing stdout")?;
                    }
                    Response::Done => break,
                    Response::Error { message } => {
                        anyhow::bail!("{message}");
                    }
                    Response::ToolUse { name, detail } => {
                        // Tool notifications go to stderr so stdout stays clean for the AI response.
                        eprintln!();
                        match name.as_str() {
                            "shell_command" => eprintln!("```bash\n$ {detail}\n```"),
                            "read_file" => eprintln!("📄 {detail}"),
                            "edit_file" => eprintln!("✏️  {detail}"),
                            "tmux_send_keys" => eprintln!("⌨️  send-keys: {detail}"),
                            "tmux_capture_pane" => eprintln!("🖥️  capture: {detail}"),
                            _ => eprintln!("🔧 {name}: {detail}"),
                        }
                    }
                    Response::SteerAck => {
                        // The daemon has acknowledged our steering message.
                        // No visible output — the next model turn will incorporate it.
                        tracing::debug!("steer acknowledged by daemon");
                    }
                    Response::DetachAccepted { .. } => {
                        // Should never arrive in a normal foreground Chat loop.
                        tracing::debug!("unexpected DetachAccepted in foreground loop");
                    }
                    Response::MemoryEntry { .. } => {
                        // Not sent to the CLI client — daemon-internal only.
                    }
                    Response::WaitingForInput { prompt } => {
                        // Daemon signals it needs a reply.  When prompt is
                        // empty the question was already on screen via Text
                        // chunks; just show the cursor.  When non-empty, print
                        // the extra context first, then the cursor line.
                        if prompt.is_empty() {
                            eprint!("\n>");
                        } else {
                            eprintln!("\n{prompt}");
                            eprint!(">");
                        }
                        let _ = tokio::io::stderr().flush().await;
                    }
                }
            }

            // Read a line from stdin and send it as a steering correction.
            // This arm is disabled (pending forever) when stdout is not a TTY.
            steer_line = next_stdin_line(&mut stdin_lines) => {
                match steer_line {
                    Some(text) => {
                        let steer_req = Request::Steer {
                            session_id: session_id_copy.clone(),
                            message: text,
                        };
                        let mut frame =
                            serde_json::to_string(&steer_req).context("serializing Steer")?;
                        frame.push('\n');
                        // If the daemon has already finished and closed the
                        // connection, the write will fail — swallow the error
                        // so the response loop can drain normally.
                        let _ = writer.write_all(frame.as_bytes()).await;
                        let _ = writer.flush().await;
                    }
                    None => {
                        // Ctrl+D on stdin — detach gracefully (task continues
                        // on the daemon side; we stop reading stdin).
                        stdin_lines = None;
                    }
                }
            }
        }
    }

    // Ensure the cursor ends up on a fresh line.
    stdout.write_all(b"\n").await.context("writing newline")?;

    // Print a dim session footer so the UUID is visible in scrollback for
    // use with `amaebi ask --resume <uuid>`.  Suppressed when stderr is not
    // a terminal (e.g., piped invocations).
    if std::io::stderr().is_terminal() {
        eprintln!("\nSession completed. ID: {}", session_id_copy);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Detach mode
// ---------------------------------------------------------------------------

/// Submit a task to the daemon in detached (background) mode.
///
/// Sends [`Request::SubmitDetach`], waits for [`Response::DetachAccepted`],
/// prints the session ID to stderr, and exits `0`.  The daemon continues
/// running the agentic loop in the background; results appear in the inbox.
pub async fn run_detach(socket: PathBuf, prompt: String, model: Option<String>) -> Result<()> {
    let stream = connect_or_start_daemon(&socket).await?;
    let (reader, mut writer) = tokio::io::split(stream);

    let model = model
        .or_else(|| std::env::var("AMAEBI_MODEL").ok())
        .unwrap_or_else(|| "gpt-4o".to_string());

    let cwd = std::env::current_dir().context("getting current directory")?;
    let session_id = tokio::task::spawn_blocking(move || session::get_or_create(&cwd))
        .await
        .context("session::get_or_create panicked")?
        .unwrap_or_else(|e| {
            tracing::warn!(error = %e, "failed to resolve session id; using \"global\"");
            "global".to_string()
        });

    let req = Request::SubmitDetach {
        prompt,
        tmux_pane: std::env::var("TMUX_PANE").ok(),
        session_id: Some(session_id),
        model,
    };
    let mut req_line = serde_json::to_string(&req).context("serializing detach request")?;
    req_line.push('\n');
    writer
        .write_all(req_line.as_bytes())
        .await
        .context("sending detach request to daemon")?;

    // Wait for the single DetachAccepted (or Error) frame.
    let mut lines = BufReader::new(reader).lines();
    let line = lines
        .next_line()
        .await
        .context("reading detach response")?
        .context("daemon closed connection without responding")?;
    let resp: Response = serde_json::from_str(&line).context("parsing detach response")?;
    match resp {
        Response::DetachAccepted { session_id } => {
            eprintln!("Task accepted. Running in background under session {session_id}.");
            eprintln!("Check `amaebi inbox list` for the result when done.");
            Ok(())
        }
        Response::Error { message } => anyhow::bail!("{message}"),
        other => anyhow::bail!("unexpected response from daemon: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Resume mode
// ---------------------------------------------------------------------------

/// Resume a prior session, loading its full history, then stream the reply.
///
/// Sends [`Request::Resume`] to the daemon, which bypasses the normal
/// `MAX_HISTORY` sliding-window and re-hydrates the complete chronological
/// conversation.  The streaming display loop is identical to [`run`].
pub async fn run_resume(
    socket: PathBuf,
    prompt: String,
    model: Option<String>,
    session_uuid: String,
) -> Result<()> {
    let mut sigint = signal(SignalKind::interrupt()).context("setting up SIGINT handler")?;

    let stream = connect_or_start_daemon(&socket).await?;
    let (reader, mut writer) = tokio::io::split(stream);

    let model = model
        .or_else(|| std::env::var("AMAEBI_MODEL").ok())
        .unwrap_or_else(|| "gpt-4o".to_string());

    let req = Request::Resume {
        prompt,
        tmux_pane: std::env::var("TMUX_PANE").ok(),
        session_id: session_uuid.clone(),
        model,
    };
    let mut req_line = serde_json::to_string(&req).context("serializing resume request")?;
    req_line.push('\n');
    writer
        .write_all(req_line.as_bytes())
        .await
        .context("sending resume request to daemon")?;

    // Streaming display loop — identical to run().
    let mut lines = BufReader::new(reader).lines();
    let mut stdout = tokio::io::stdout();
    let mut last_ctrl_c: Option<Instant> = None;

    let use_stdin = std::io::stdin().is_terminal();
    let mut stdin_lines: Option<BufReader<tokio::io::Stdin>> = if use_stdin {
        Some(BufReader::new(tokio::io::stdin()))
    } else {
        None
    };

    loop {
        tokio::select! {
            biased;

            result = sigint.recv() => {
                let Some(_) = result else { break; };
                let now = Instant::now();
                if is_within_window(last_ctrl_c, now, DOUBLE_CTRLC_WINDOW) {
                    eprintln!();
                    let _ = stdout.flush().await;
                    let _ = tokio::io::stderr().flush().await;
                    return Err(anyhow::Error::new(Interrupted));
                }
                eprintln!(
                    "\nPress Ctrl-C again within {}s to exit",
                    DOUBLE_CTRLC_WINDOW.as_secs()
                );
                last_ctrl_c = Some(now);
            }

            result = lines.next_line() => {
                let line = result.context("reading response from daemon")?;
                let Some(line) = line else { break; };
                let resp: Response = serde_json::from_str(&line).context("parsing response")?;
                match resp {
                    Response::Text { chunk } => {
                        stdout.write_all(chunk.as_bytes()).await.context("writing to stdout")?;
                        stdout.flush().await.context("flushing stdout")?;
                    }
                    Response::Done => break,
                    Response::Error { message } => anyhow::bail!("{message}"),
                    Response::ToolUse { name, detail } => {
                        eprintln!();
                        match name.as_str() {
                            "shell_command" => eprintln!("```bash\n$ {detail}\n```"),
                            "read_file" => eprintln!("📄 {detail}"),
                            "edit_file" => eprintln!("✏️  {detail}"),
                            "tmux_send_keys" => eprintln!("⌨️  send-keys: {detail}"),
                            "tmux_capture_pane" => eprintln!("🖥️  capture: {detail}"),
                            _ => eprintln!("🔧 {name}: {detail}"),
                        }
                    }
                    Response::SteerAck => {
                        tracing::debug!("steer acknowledged by daemon");
                    }
                    Response::DetachAccepted { .. } => {
                        tracing::debug!("unexpected DetachAccepted in resume loop");
                    }
                    Response::MemoryEntry { .. } => {
                        // Not sent to the CLI client — daemon-internal only.
                    }
                    Response::WaitingForInput { prompt } => {
                        if prompt.is_empty() {
                            eprint!("\n>");
                        } else {
                            eprintln!("\n{prompt}");
                            eprint!(">");
                        }
                        let _ = tokio::io::stderr().flush().await;
                    }
                }
            }

            steer_line = next_stdin_line(&mut stdin_lines) => {
                match steer_line {
                    Some(text) => {
                        let steer_req = Request::Steer {
                            session_id: session_uuid.clone(),
                            message: text,
                        };
                        let mut frame = serde_json::to_string(&steer_req)
                            .context("serializing Steer")?;
                        frame.push('\n');
                        let _ = writer.write_all(frame.as_bytes()).await;
                        let _ = writer.flush().await;
                    }
                    None => {
                        stdin_lines = None;
                    }
                }
            }
        }
    }

    stdout.write_all(b"\n").await.context("writing newline")?;

    if std::io::stderr().is_terminal() {
        eprintln!("\nSession completed. ID: {}", session_uuid);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Stdin helper
// ---------------------------------------------------------------------------

/// Read the next line from an optional stdin `Lines` reader.
///
/// Returns `Some(line)` when the user types a line, `None` on EOF (Ctrl+D),
/// and **never resolves** (`std::future::pending`) when `lines` is `None`.
/// This allows the caller to include this as an always-present arm in
/// `tokio::select!` without special-casing the TTY check in the macro body.
async fn next_stdin_line(lines: &mut Option<BufReader<tokio::io::Stdin>>) -> Option<String> {
    match lines {
        Some(buf) => {
            let mut line = String::new();
            match tokio::io::AsyncBufReadExt::read_line(buf, &mut line).await {
                Ok(0) => None, // EOF
                Ok(_) => {
                    // Strip trailing newline.
                    if line.ends_with('\n') {
                        line.pop();
                        if line.ends_with('\r') {
                            line.pop();
                        }
                    }
                    Some(line)
                }
                Err(_) => None,
            }
        }
        None => std::future::pending().await,
    }
}

// ---------------------------------------------------------------------------
// Daemon auto-start
// ---------------------------------------------------------------------------

/// Connect to the daemon socket, starting the daemon in the background if it
/// is not already running.
///
/// On the first failed connection attempt the daemon binary is spawned with
/// `stdin`/`stdout`/`stderr` all redirected to `/dev/null`.  Connection is
/// then retried with exponential back-off up to ~5 seconds before giving up.
async fn connect_or_start_daemon(socket: &std::path::Path) -> Result<UnixStream> {
    // Happy path: daemon is already running.
    if let Ok(stream) = UnixStream::connect(socket).await {
        return Ok(stream);
    }

    // Spawn the daemon in the background.
    tracing::info!(path = %socket.display(), "daemon not reachable; auto-starting");
    start_daemon(socket).await?;

    // Retry with exponential back-off (100 ms, 200 ms, 400 ms … capped at 1 s).
    for attempt in 0u32..10 {
        let wait_ms = 100u64 << attempt.min(3); // 100, 200, 400, 800, 800, …
        tokio::time::sleep(Duration::from_millis(wait_ms)).await;
        if let Ok(stream) = UnixStream::connect(socket).await {
            return Ok(stream);
        }
    }

    anyhow::bail!(
        "daemon did not become ready in time — socket: {}",
        socket.display()
    )
}

/// Spawn the amaebi daemon as a detached background process.
async fn start_daemon(socket: &std::path::Path) -> Result<()> {
    let exe = std::env::current_exe().context("finding current executable path")?;
    tokio::process::Command::new(&exe)
        .arg("daemon")
        .arg("--socket")
        .arg(socket)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .with_context(|| format!("spawning daemon from {}", exe.display()))?;
    Ok(())
}

/// Return `true` if `last_press` is `Some` and the duration from `last_press`
/// to `now` is at most `window` (inclusive boundary).
///
/// Uses [`Instant::checked_duration_since`] so a clock anomaly where
/// `now < last_press` returns `false` instead of panicking.
///
/// Accepts `now` as a parameter so callers pass `Instant::now()` and tests can
/// supply controlled values — the function itself has no side effects.
fn is_within_window(last_press: Option<Instant>, now: Instant, window: Duration) -> bool {
    match last_press {
        None => false,
        Some(t) => now.checked_duration_since(t).is_some_and(|d| d <= window),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_within_window_none_returns_false() {
        assert!(!is_within_window(None, Instant::now(), DOUBLE_CTRLC_WINDOW));
    }

    #[test]
    fn is_within_window_press_inside_window_returns_true() {
        let now = Instant::now();
        let press = now - Duration::from_secs(1);
        assert!(is_within_window(Some(press), now, Duration::from_secs(2)));
    }

    #[test]
    fn is_within_window_press_outside_window_returns_false() {
        let now = Instant::now();
        let press = now - Duration::from_secs(3);
        assert!(!is_within_window(Some(press), now, Duration::from_secs(2)));
    }

    #[test]
    fn is_within_window_press_at_boundary_is_inside() {
        // Exactly at the boundary counts as within the window (inclusive).
        let now = Instant::now();
        let press = now - Duration::from_secs(2);
        assert!(is_within_window(Some(press), now, Duration::from_secs(2)));
    }

    #[test]
    fn is_within_window_clock_skew_returns_false() {
        // If now < press (impossible in normal operation but guards against
        // platform clock anomalies), checked_duration_since returns None → false.
        let now = Instant::now();
        let future_press = now + Duration::from_secs(1);
        assert!(!is_within_window(
            Some(future_press),
            now,
            Duration::from_secs(2)
        ));
    }
}
