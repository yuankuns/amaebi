use anyhow::{Context, Result};
use std::collections::VecDeque;
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

/// Maximum number of frames buffered during a steer interaction.
/// If exceeded, the oldest frames are dropped and a truncation notice is shown
/// when the buffer is flushed.
const STEER_BUFFER_MAX_FRAMES: usize = 1000;

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
    // Set to true on first Ctrl-C to immediately buffer daemon output while
    // the user types a steering correction (prevents output/input overlap).
    let mut steer_pending = false;
    // Frames received while steer_pending — flushed to the terminal on SteerAck.
    let mut steer_buffer: VecDeque<Response> = VecDeque::new();
    // Set to true when eviction occurs so flush_steer_buffer can print a truncation notice.
    let mut buffer_truncated = false;

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
                // First press (or expired window): interrupt the agent and
                // immediately start buffering output to prevent overlap with
                // the user's correction text.  Double Ctrl-C (handled above) exits.
                steer_pending = true;
                if use_stdin {
                    eprintln!("\n^C interrupted. Enter correction (empty line to cancel): ");
                    eprint!(">");
                } else {
                    eprintln!("\n[interrupted — press Ctrl-C again quickly to exit]");
                }
                let _ = tokio::io::stderr().flush().await;
                // Notify the daemon to abort/skip remaining tool calls immediately.
                let interrupt_req = Request::Interrupt {
                    session_id: session_id_copy.clone(),
                };
                if let Ok(mut frame) = serde_json::to_string(&interrupt_req) {
                    frame.push('\n');
                    let _ = writer.write_all(frame.as_bytes()).await;
                    let _ = writer.flush().await;
                }
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
                        // Buffer output while the user is typing a steering correction
                        // to prevent mixing daemon output with the user's input.
                        if steer_pending {
                            push_steer_buffer(&mut steer_buffer, &mut buffer_truncated, Response::Text { chunk });
                        } else {
                            stdout
                                .write_all(chunk.as_bytes())
                                .await
                                .context("writing to stdout")?;
                            stdout.flush().await.context("flushing stdout")?;
                        }
                    }
                    Response::Done => {
                        // Flush any buffered frames before exiting.
                        flush_steer_buffer(&mut steer_buffer, &mut buffer_truncated, &mut stdout).await?;
                        break;
                    }
                    Response::Error { message } => {
                        // Flush any buffered frames before surfacing the error
                        // so the user can see what happened before it failed.
                        flush_steer_buffer(&mut steer_buffer, &mut buffer_truncated, &mut stdout).await?;
                        anyhow::bail!("{message}");
                    }
                    Response::ToolUse { name, detail } => {
                        // Buffer tool notifications while steering is pending.
                        if steer_pending {
                            push_steer_buffer(&mut steer_buffer, &mut buffer_truncated, Response::ToolUse { name, detail });
                        } else {
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
                    }
                    Response::Compacting => {
                        if steer_pending {
                            push_steer_buffer(&mut steer_buffer, &mut buffer_truncated, Response::Compacting);
                        } else if std::io::stderr().is_terminal() {
                            eprintln!("\n\x1b[1;5;34m[compacting conversation history…]\x1b[0m");
                        } else {
                            eprintln!("\n[compacting conversation history…]");
                        }
                    }
                    Response::SteerAck => {
                        // The daemon has acknowledged our steering message.
                        // Flush buffered frames so the user can see what happened
                        // before their correction took effect, then resume output.
                        steer_pending = false;
                        last_ctrl_c = None;
                        flush_steer_buffer(&mut steer_buffer, &mut buffer_truncated, &mut stdout).await?;
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
                        // Buffer while steer is pending (user is typing).
                        if steer_pending {
                            push_steer_buffer(&mut steer_buffer, &mut buffer_truncated, Response::WaitingForInput { prompt });
                        } else if prompt.is_empty() {
                            eprint!("\n>");
                            let _ = tokio::io::stderr().flush().await;
                        } else {
                            eprintln!("\n{prompt}");
                            eprint!(">");
                            let _ = tokio::io::stderr().flush().await;
                        }
                    }
                }
            }

            // Read a line from stdin and send it as a steering correction.
            // This arm is disabled (pending forever) when stdout is not a TTY.
            steer_line = next_stdin_line(&mut stdin_lines) => {
                match steer_line {
                    Some(text) if !text.trim().is_empty() => {
                        // steer_pending is already true (set on Ctrl-C); just send the request.
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
                    Some(_) => {
                        // Empty or whitespace-only line (e.g. bare Enter) — if a steer
                        // is pending, cancel it and resume normal output.
                        if steer_pending {
                            steer_pending = false;
                            last_ctrl_c = None;
                            flush_steer_buffer(&mut steer_buffer, &mut buffer_truncated, &mut stdout).await?;
                            buffer_truncated = false;
                        }
                        // Otherwise discard silently.
                    }
                    None => {
                        // Ctrl+D on stdin — if a steer is pending, cancel it
                        // and resume normal output; then stop reading stdin.
                        if steer_pending {
                            steer_pending = false;
                            last_ctrl_c = None;
                            flush_steer_buffer(&mut steer_buffer, &mut buffer_truncated, &mut stdout).await?;
                            buffer_truncated = false;
                        }
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
// Interactive chat REPL (long connection)
// ---------------------------------------------------------------------------

/// Run a persistent multi-turn chat session on a single socket connection.
///
/// Ctrl-C mid-generation → interrupt + steer prompt (same as `amaebi ask`).
/// Double Ctrl-C within 2 s → exit. Empty line or Ctrl-D → exit.
#[allow(unused_assignments)] // steer_pending is read inside select! async blocks
pub async fn run_chat_loop(
    socket: PathBuf,
    initial_prompt: Option<String>,
    model: Option<String>,
) -> Result<()> {
    let mut sigint = signal(SignalKind::interrupt()).context("setting up SIGINT handler")?;

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

    if std::io::stderr().is_terminal() {
        eprintln!("Chat session started (ID: {session_id}). Empty line or Ctrl-D to exit.\n");
    }

    let stream = connect_or_start_daemon(&socket).await?;
    let (read_half, mut write_half) = stream.into_split();
    let mut lines = BufReader::new(read_half).lines();

    let mut stdout = tokio::io::stdout();
    // Single BufReader for stdin — reused across all prompt/steer reads so
    // buffered bytes are never dropped between reads.
    let mut stdin = BufReader::new(tokio::io::stdin());
    let mut next_prompt = initial_prompt;
    let mut last_ctrl_c: Option<Instant> = None;
    let mut steer_pending = false;
    // Text chunks buffered while a steer correction is being typed so that
    // streaming output does not interleave with user input.
    let mut steer_text_buf: Vec<String> = Vec::new();

    'session: loop {
        let prompt = match next_prompt.take() {
            Some(p) => p,
            None => {
                if std::io::stderr().is_terminal() {
                    eprint!("> ");
                    let _ = tokio::io::stderr().flush().await;
                }
                // Use the raw-mode reader so wide (CJK) characters are erased
                // correctly on backspace: the terminal line-discipline only emits
                // one `\b \b` per backspace, leaving a ghost column for 2-wide
                // chars.  Our reader tracks display width and emits the right
                // number of erase columns.
                let raw_result = tokio::task::spawn_blocking(prompt_input::read_line_raw)
                    .await
                    .context("prompt input task panicked")?;
                match raw_result {
                    Ok(None) => break 'session,
                    Ok(Some(line)) if line.is_empty() => break 'session,
                    Ok(Some(line)) => line,
                    // Ctrl-C during prompt input — exit the session cleanly.
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => break 'session,
                    Err(e) => {
                        return Err(anyhow::Error::new(e).context("reading prompt from stdin"))
                    }
                }
            }
        };

        let req = Request::Chat {
            prompt: prompt.clone(),
            tmux_pane: std::env::var("TMUX_PANE").ok(),
            session_id: Some(session_id.clone()),
            model: model.clone(),
        };
        let mut req_line = serde_json::to_string(&req)?;
        req_line.push('\n');
        write_half.write_all(req_line.as_bytes()).await?;
        steer_pending = false;

        loop {
            tokio::select! {
                biased;

                result = sigint.recv() => {
                    let Some(_) = result else { break 'session; };
                    let now = Instant::now();
                    if is_within_window(last_ctrl_c, now, DOUBLE_CTRLC_WINDOW) {
                        eprintln!();
                        let _ = stdout.flush().await;
                        return Err(anyhow::Error::new(Interrupted));
                    }
                    steer_pending = true;
                    if std::io::stderr().is_terminal() {
                        eprintln!("\n^C interrupted. Enter correction (empty line to cancel): ");
                        eprint!(">");
                        let _ = tokio::io::stderr().flush().await;
                    }
                    let interrupt_req = Request::Interrupt { session_id: session_id.clone() };
                    if let Ok(mut frame) = serde_json::to_string(&interrupt_req) {
                        frame.push('\n');
                        let _ = write_half.write_all(frame.as_bytes()).await;
                    }
                    last_ctrl_c = Some(now);
                }

                result = lines.next_line() => {
                    let line = result.context("reading response")?;
                    let Some(line) = line else { break 'session; };
                    let resp: Response = serde_json::from_str(&line)?;
                    match resp {
                        Response::Text { chunk } => {
                            if steer_pending {
                                // Buffer text while the user is typing a steer
                                // correction so streaming output does not
                                // interleave with the correction prompt.
                                steer_text_buf.push(chunk);
                            } else {
                                stdout.write_all(chunk.as_bytes()).await?;
                                stdout.flush().await?;
                            }
                        }
                        Response::Done => {
                            // Flush any text that arrived while steer was pending.
                            for chunk in steer_text_buf.drain(..) {
                                stdout.write_all(chunk.as_bytes()).await?;
                            }
                            stdout.write_all(b"\n").await?;
                            stdout.flush().await?;
                            break;
                        }
                        Response::Error { message } => anyhow::bail!("{message}"),
                        Response::ToolUse { name, detail } => {
                            if std::io::stderr().is_terminal() {
                                match name.as_str() {
                                    "shell_command" => eprintln!("```bash\n$ {detail}\n```"),
                                    "read_file"     => eprintln!("📄 {detail}"),
                                    "edit_file"     => eprintln!("✏️  {detail}"),
                                    _ => eprintln!("🔧 {name}: {detail}"),
                                }
                            }
                        }
                        Response::WaitingForInput { prompt: extra } => {
                            // Show the prompt and set steer_pending so the next
                            // iteration of the select! loop reads stdin via the
                            // existing steer arm — keeping SIGINT responsive.
                            if !extra.is_empty() { eprintln!("\n{extra}"); }
                            eprint!(">");
                            let _ = tokio::io::stderr().flush().await;
                            steer_pending = true;
                        }
                        Response::SteerAck => {
                            steer_pending = false;
                            // Flush text buffered while the steer was pending.
                            for chunk in steer_text_buf.drain(..) {
                                let _ = stdout.write_all(chunk.as_bytes()).await;
                            }
                            let _ = stdout.flush().await;
                        }
                        Response::Compacting => {
                            if std::io::stderr().is_terminal() { eprintln!("\n[compacting…]"); }
                        }
                        _ => {}
                    }
                }

                line = async {
                    if steer_pending {
                        let mut buf = String::new();
                        let n = tokio::io::AsyncBufReadExt::read_line(
                            &mut stdin, &mut buf
                        ).await.unwrap_or(0); // EOF/error = treat as no input
                        if n > 0 { Some(buf) } else { None }
                    } else {
                        std::future::pending::<Option<String>>().await
                    }
                } => {
                    if let Some(text) = line {
                        let trimmed = text.trim_end_matches('\n').trim_end_matches('\r');
                        if trimmed.is_empty() {
                            steer_pending = false;
                            last_ctrl_c = None; // cancelling steer resets the double-Ctrl-C window
                            // No Steer frame sent on cancel: the daemon's WAITING_FOR_INPUT
                            // path now treats the already-sent Interrupt (None) as a cancel
                            // signal, so no follow-up message is needed.
                            // Flush buffered text now that steer is cancelled.
                            for chunk in steer_text_buf.drain(..) {
                                let _ = stdout.write_all(chunk.as_bytes()).await;
                            }
                            let _ = stdout.flush().await;
                        } else {
                            let steer_req = Request::Steer {
                                session_id: session_id.clone(),
                                message: trimmed.to_owned(),
                            };
                            if let Ok(mut frame) = serde_json::to_string(&steer_req) {
                                frame.push('\n');
                                let _ = write_half.write_all(frame.as_bytes()).await;
                            }
                            steer_pending = false;
                            last_ctrl_c = None;
                        }
                    } else {
                        break 'session;
                    }
                }
            }
        }
    }

    if std::io::stderr().is_terminal() {
        eprintln!("\nSession ended. ID: {session_id}");
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
    // Set to true when the user submits steer text, to buffer daemon output
    // until SteerAck arrives (prevents mixing with the next model turn).
    let mut steer_pending = false;
    // Frames received while steer_pending — flushed to the terminal on SteerAck.
    let mut steer_buffer: VecDeque<Response> = VecDeque::new();
    // Set to true when eviction occurs so flush_steer_buffer can print a truncation notice.
    let mut buffer_truncated = false;

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
                // Start buffering immediately so output doesn't overlap with user input.
                steer_pending = true;
                if use_stdin {
                    eprintln!("\n^C interrupted. Enter correction (empty line to cancel): ");
                } else {
                    eprintln!("\n[interrupted — press Ctrl-C again soon to exit]");
                }
                let _ = tokio::io::stderr().flush().await;
                // Notify the daemon to abort/skip remaining tool calls immediately.
                let interrupt_req = Request::Interrupt {
                    session_id: session_uuid.clone(),
                };
                if let Ok(mut frame) = serde_json::to_string(&interrupt_req) {
                    frame.push('\n');
                    let _ = writer.write_all(frame.as_bytes()).await;
                    let _ = writer.flush().await;
                }
                last_ctrl_c = Some(now);
            }

            result = lines.next_line() => {
                let line = result.context("reading response from daemon")?;
                let Some(line) = line else { break; };
                let resp: Response = serde_json::from_str(&line).context("parsing response")?;
                match resp {
                    Response::Text { chunk } => {
                        // Buffer output while the user is typing a steering correction.
                        if steer_pending {
                            push_steer_buffer(&mut steer_buffer, &mut buffer_truncated, Response::Text { chunk });
                        } else {
                            stdout.write_all(chunk.as_bytes()).await.context("writing to stdout")?;
                            stdout.flush().await.context("flushing stdout")?;
                        }
                    }
                    Response::Done => {
                        // Flush any buffered frames before exiting.
                        flush_steer_buffer(&mut steer_buffer, &mut buffer_truncated, &mut stdout).await?;
                        break;
                    }
                    Response::Error { message } => {
                        flush_steer_buffer(&mut steer_buffer, &mut buffer_truncated, &mut stdout).await?;
                        anyhow::bail!("{message}")
                    }
                    Response::ToolUse { name, detail } => {
                        // Buffer tool notifications while steering is pending.
                        if steer_pending {
                            push_steer_buffer(&mut steer_buffer, &mut buffer_truncated, Response::ToolUse { name, detail });
                        } else {
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
                    }
                    Response::Compacting => {
                        if steer_pending {
                            push_steer_buffer(&mut steer_buffer, &mut buffer_truncated, Response::Compacting);
                        } else if std::io::stderr().is_terminal() {
                            eprintln!("\n\x1b[1;5;34m[compacting conversation history…]\x1b[0m");
                        } else {
                            eprintln!("\n[compacting conversation history…]");
                        }
                    }
                    Response::SteerAck => {
                        // Resume output rendering for the next model turn.
                        // Flush buffered frames so the user can see what happened
                        // before their correction took effect.
                        steer_pending = false;
                        last_ctrl_c = None;
                        flush_steer_buffer(&mut steer_buffer, &mut buffer_truncated, &mut stdout).await?;
                        tracing::debug!("steer acknowledged by daemon");
                    }
                    Response::DetachAccepted { .. } => {
                        tracing::debug!("unexpected DetachAccepted in resume loop");
                    }
                    Response::MemoryEntry { .. } => {
                        // Not sent to the CLI client — daemon-internal only.
                    }
                    Response::WaitingForInput { prompt } => {
                        // Buffer while steer is pending (user is typing).
                        if steer_pending {
                            push_steer_buffer(&mut steer_buffer, &mut buffer_truncated, Response::WaitingForInput { prompt });
                        } else if prompt.is_empty() {
                            eprint!("\n>");
                            let _ = tokio::io::stderr().flush().await;
                        } else {
                            eprintln!("\n{prompt}");
                            eprint!(">");
                            let _ = tokio::io::stderr().flush().await;
                        }
                    }
                }
            }

            steer_line = next_stdin_line(&mut stdin_lines) => {
                match steer_line {
                    Some(text) if !text.trim().is_empty() => {
                        // steer_pending already set on Ctrl-C; just send the request.
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
                    Some(_) => {
                        // Empty or whitespace-only line — if a steer is pending,
                        // cancel it and resume normal output.
                        if steer_pending {
                            steer_pending = false;
                            last_ctrl_c = None;
                            flush_steer_buffer(&mut steer_buffer, &mut buffer_truncated, &mut stdout).await?;
                            buffer_truncated = false;
                        }
                    }
                    None => {
                        // EOF (Ctrl+D) — if a steer is pending, cancel it and
                        // resume normal output; then stop reading stdin.
                        if steer_pending {
                            steer_pending = false;
                            last_ctrl_c = None;
                            flush_steer_buffer(&mut steer_buffer, &mut buffer_truncated, &mut stdout).await?;
                            buffer_truncated = false;
                        }
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
// Steer buffer helper
// ---------------------------------------------------------------------------

/// Push a response frame onto the steer buffer, enforcing the cap.
///
/// If the buffer has reached [`STEER_BUFFER_MAX_FRAMES`], the oldest entry is
/// evicted and `truncated` is set to `true`.  `flush_steer_buffer` checks
/// that flag to print a truncation notice without comparing frame content.
/// This keeps memory bounded regardless of how long the user takes to type
/// their correction.
fn push_steer_buffer(buffer: &mut VecDeque<Response>, truncated: &mut bool, frame: Response) {
    if buffer.len() >= STEER_BUFFER_MAX_FRAMES {
        // Evict the oldest frame and record that truncation has occurred.
        buffer.pop_front();
        *truncated = true;
        // buffer.len() == STEER_BUFFER_MAX_FRAMES - 1 here.
    }
    buffer.push_back(frame);
}

/// Flush buffered response frames that were held while a steer was pending.
///
/// Each frame is rendered to the terminal exactly as it would have been on
/// first receipt.  A dim header is printed before the buffered frames so the
/// user knows they're seeing suppressed output.  If `truncated` is `true`, a
/// notice is printed first to indicate that some frames were dropped.
async fn flush_steer_buffer(
    buffer: &mut VecDeque<Response>,
    truncated: &mut bool,
    stdout: &mut tokio::io::Stdout,
) -> Result<()> {
    if buffer.is_empty() && !*truncated {
        return Ok(());
    }
    if std::io::stderr().is_terminal() {
        eprintln!("\n\x1b[2m[output before steer took effect:]\x1b[0m");
    } else {
        eprintln!("\n[output before steer took effect:]");
    }
    if *truncated {
        eprintln!("\n[some output was truncated]");
        *truncated = false;
    }
    for resp in buffer.drain(..) {
        match resp {
            Response::Text { chunk } => {
                stdout
                    .write_all(chunk.as_bytes())
                    .await
                    .context("writing buffered text to stdout")?;
                stdout.flush().await.context("flushing stdout")?;
            }
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
            Response::Compacting => {
                if std::io::stderr().is_terminal() {
                    eprintln!("\n\x1b[1;5;34m[compacting conversation history…]\x1b[0m");
                } else {
                    eprintln!("\n[compacting conversation history…]");
                }
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
            // Other variants are not buffered; ignore defensively.
            _ => {}
        }
    }
    Ok(())
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

// ---------------------------------------------------------------------------
// Raw-mode prompt input — wide-character-aware line editor
// ---------------------------------------------------------------------------

/// Prompt input routines that handle CJK/wide characters correctly.
///
/// In the default cooked (canonical) terminal mode the kernel's line discipline
/// emits `\b \b` on backspace, which only moves the cursor one column.  For
/// two-column-wide CJK characters this leaves a ghost column on screen.
///
/// This module puts the terminal in raw mode for the duration of a single line
/// read so it can track each character's display width and emit the correct
/// number of erase columns on backspace.
mod prompt_input {
    use std::io::{Read, Write};
    use unicode_width::UnicodeWidthChar as _;

    /// RAII guard that restores the terminal to its saved settings on drop.
    #[cfg(unix)]
    struct RawModeGuard {
        fd: std::os::unix::io::RawFd,
        orig: libc::termios,
    }

    #[cfg(unix)]
    impl RawModeGuard {
        fn enter(fd: std::os::unix::io::RawFd) -> std::io::Result<Self> {
            let mut orig = unsafe { std::mem::zeroed::<libc::termios>() };
            if unsafe { libc::tcgetattr(fd, &mut orig) } != 0 {
                return Err(std::io::Error::last_os_error());
            }
            let mut raw = orig;
            unsafe { libc::cfmakeraw(&mut raw) };
            // Keep output post-processing (OPOST) so "\r\n" renders correctly.
            raw.c_oflag |= libc::OPOST;
            if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &raw) } != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(Self { fd, orig })
        }
    }

    #[cfg(unix)]
    impl Drop for RawModeGuard {
        fn drop(&mut self) {
            // Best-effort restore; ignore errors during drop.
            unsafe { libc::tcsetattr(self.fd, libc::TCSANOW, &self.orig) };
        }
    }

    /// Read one line from stdin with correct wide-character backspace handling.
    ///
    /// * `Ok(Some(s))` — user pressed Enter; `s` has no trailing newline.
    /// * `Ok(None)` — Ctrl-D (EOF).
    /// * `Err(e)` with `e.kind() == Interrupted` — Ctrl-C.
    ///
    /// Falls back to plain `read_line` when stdin is not a TTY.
    pub fn read_line_raw() -> std::io::Result<Option<String>> {
        use std::io::IsTerminal as _;

        if !std::io::stdin().is_terminal() {
            return read_line_cooked();
        }

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd as _;
            let fd = std::io::stdin().as_raw_fd();
            // On failure (e.g. stdin is not a real tty), fall through to cooked.
            if let Ok(guard) = RawModeGuard::enter(fd) {
                return read_line_raw_inner(guard);
            }
        }

        read_line_cooked()
    }

    /// Cooked-mode fallback used when stdin is not a TTY or raw mode fails.
    fn read_line_cooked() -> std::io::Result<Option<String>> {
        use std::io::BufRead as _;
        let mut s = String::new();
        match std::io::stdin().lock().read_line(&mut s) {
            Ok(0) => Ok(None),
            Ok(_) => {
                if s.ends_with('\n') {
                    s.pop();
                }
                if s.ends_with('\r') {
                    s.pop();
                }
                Ok(Some(s))
            }
            Err(e) => Err(e),
        }
    }

    #[cfg(unix)]
    fn read_line_raw_inner(_guard: RawModeGuard) -> std::io::Result<Option<String>> {
        let mut chars: Vec<char> = Vec::new();
        // Display width (columns) of each char, matched by index to `chars`.
        let mut widths: Vec<usize> = Vec::new();
        let mut stdin = std::io::stdin();
        let mut out = std::io::stderr();

        loop {
            let mut byte = [0u8; 1];
            match stdin.read_exact(&mut byte) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
                Err(e) => return Err(e),
            }

            match byte[0] {
                // Enter (CR in raw mode; some environments send LF)
                b'\r' | b'\n' => {
                    out.write_all(b"\r\n")?;
                    out.flush()?;
                    return Ok(Some(chars.iter().collect()));
                }
                // Ctrl-D
                4 => return Ok(None),
                // Ctrl-C
                3 => {
                    out.write_all(b"^C\r\n")?;
                    out.flush()?;
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Interrupted,
                        "ctrl-c",
                    ));
                }
                // Backspace (DEL = 0x7f on most terminals; BS = 0x08 on some)
                0x7f | 0x08 => {
                    if let (Some(_), Some(w)) = (chars.pop(), widths.pop()) {
                        // Erase exactly `w` display columns:
                        // move cursor back w, write w spaces, move back w again.
                        let back = vec![b'\x08'; w];
                        let spaces = vec![b' '; w];
                        out.write_all(&back)?;
                        out.write_all(&spaces)?;
                        out.write_all(&back)?;
                        out.flush()?;
                    }
                }
                // Escape sequences (arrows, function keys) — consume and discard.
                0x1b => {
                    let mut eb = [0u8; 1];
                    if stdin.read_exact(&mut eb).is_ok() && eb[0] == b'[' {
                        // CSI sequence: read until final byte in 0x40–0x7E.
                        loop {
                            let mut fb = [0u8; 1];
                            if stdin.read_exact(&mut fb).is_err() {
                                break;
                            }
                            if (0x40..=0x7e).contains(&fb[0]) {
                                break;
                            }
                        }
                    }
                    // Other 2-byte ESC sequences: second byte already consumed above.
                }
                // ASCII printable
                b @ 0x20..=0x7e => {
                    out.write_all(&[b])?;
                    out.flush()?;
                    chars.push(b as char);
                    widths.push(1);
                }
                // UTF-8 multi-byte lead byte
                b @ 0xC0..=0xFF => {
                    let n_extra: usize = if b >= 0xF0 {
                        3
                    } else if b >= 0xE0 {
                        2
                    } else {
                        1
                    };
                    let mut utf8_buf: Vec<u8> = vec![b];
                    for _ in 0..n_extra {
                        let mut cb = [0u8; 1];
                        if stdin.read_exact(&mut cb).is_err() {
                            break;
                        }
                        utf8_buf.push(cb[0]);
                    }
                    if let Ok(s) = std::str::from_utf8(&utf8_buf) {
                        if let Some(ch) = s.chars().next() {
                            let w = ch.width().unwrap_or(1);
                            out.write_all(s.as_bytes())?;
                            out.flush()?;
                            chars.push(ch);
                            widths.push(w);
                        }
                    }
                    // Silently discard invalid UTF-8 sequences.
                }
                // Other control bytes — ignore silently.
                _ => {}
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use std::io::Write as _;

        /// Verify cooked fallback strips exactly one trailing newline.
        #[test]
        fn cooked_strips_newline() {
            // We can't easily invoke read_line_cooked() against a real stdin in a
            // unit test, so we test the trimming logic directly.
            let mut s = "hello\n".to_string();
            if s.ends_with('\n') {
                s.pop();
            }
            if s.ends_with('\r') {
                s.pop();
            }
            assert_eq!(s, "hello");
        }

        /// Verify CRLF is trimmed by the cooked fallback logic.
        #[test]
        fn cooked_strips_crlf() {
            let mut s = "hello\r\n".to_string();
            if s.ends_with('\n') {
                s.pop();
            }
            if s.ends_with('\r') {
                s.pop();
            }
            assert_eq!(s, "hello");
        }

        /// CJK characters should have display width 2.
        #[test]
        fn cjk_char_has_width_2() {
            use unicode_width::UnicodeWidthChar as _;
            assert_eq!('中'.width(), Some(2));
            assert_eq!('あ'.width(), Some(2));
            assert_eq!('한'.width(), Some(2));
        }

        /// ASCII characters have display width 1.
        #[test]
        fn ascii_char_has_width_1() {
            use unicode_width::UnicodeWidthChar as _;
            assert_eq!('a'.width(), Some(1));
            assert_eq!('Z'.width(), Some(1));
        }

        // Ensure the module compiles cleanly on non-unix targets (write to
        // stderr so the import is exercised).
        #[test]
        fn prompt_input_smoke() {
            let _ = std::io::stderr().write_all(b"");
        }
    }
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

    // -----------------------------------------------------------------------
    // push_steer_buffer / flush_steer_buffer
    // -----------------------------------------------------------------------

    fn text(s: &str) -> Response {
        Response::Text {
            chunk: s.to_string(),
        }
    }

    #[test]
    fn push_steer_buffer_evicts_at_max_frames() {
        let mut buf: VecDeque<Response> = VecDeque::new();
        let mut truncated = false;
        // Fill up to the cap.
        for i in 0..STEER_BUFFER_MAX_FRAMES {
            push_steer_buffer(&mut buf, &mut truncated, text(&format!("frame-{i}")));
        }
        assert_eq!(buf.len(), STEER_BUFFER_MAX_FRAMES);
        assert!(!truncated, "no eviction yet");
        // One more push should evict the oldest.
        push_steer_buffer(&mut buf, &mut truncated, text("extra"));
        assert_eq!(buf.len(), STEER_BUFFER_MAX_FRAMES, "len stays at cap");
        assert!(truncated, "eviction sets truncated flag");
        // The oldest frame ("frame-0") should be gone; "extra" should be last.
        assert!(
            !matches!(buf.front(), Some(Response::Text { chunk }) if chunk == "frame-0"),
            "oldest frame was evicted"
        );
        assert!(
            matches!(buf.back(), Some(Response::Text { chunk }) if chunk == "extra"),
            "new frame is at the back"
        );
    }

    #[test]
    fn push_steer_buffer_sentinel_set_only_once() {
        let mut buf: VecDeque<Response> = VecDeque::new();
        let mut truncated = false;
        // Fill to the cap.
        for i in 0..STEER_BUFFER_MAX_FRAMES {
            push_steer_buffer(&mut buf, &mut truncated, text(&format!("{i}")));
        }
        // Cause eviction multiple times.
        for _ in 0..5 {
            push_steer_buffer(&mut buf, &mut truncated, text("x"));
        }
        // truncated is a bool flag — just one bit, not a counter.  Verify it
        // stayed true after the first eviction and didn't flip back.
        assert!(truncated);
        assert_eq!(buf.len(), STEER_BUFFER_MAX_FRAMES);
    }

    #[test]
    fn push_steer_buffer_frame_ordering_after_eviction() {
        let mut buf: VecDeque<Response> = VecDeque::new();
        let mut truncated = false;
        // Fill to cap with labelled frames.
        for i in 0..STEER_BUFFER_MAX_FRAMES {
            push_steer_buffer(&mut buf, &mut truncated, text(&format!("old-{i}")));
        }
        // Evict 3 oldest, add 3 new.
        for i in 0..3 {
            push_steer_buffer(&mut buf, &mut truncated, text(&format!("new-{i}")));
        }
        // The first 3 "old-*" frames should be gone; "new-0..2" at the back.
        let frames: Vec<&str> = buf
            .iter()
            .filter_map(|r| {
                if let Response::Text { chunk } = r {
                    Some(chunk.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert!(!frames.contains(&"old-0"));
        assert!(!frames.contains(&"old-1"));
        assert!(!frames.contains(&"old-2"));
        assert_eq!(frames[frames.len() - 3], "new-0");
        assert_eq!(frames[frames.len() - 2], "new-1");
        assert_eq!(frames[frames.len() - 1], "new-2");
    }
}
