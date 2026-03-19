use anyhow::{Context, Result};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::signal::unix::{signal, SignalKind};

use crate::ipc::{Request, Response};

/// How long the user has to press Ctrl-C a second time to exit.
const DOUBLE_CTRLC_WINDOW: Duration = Duration::from_secs(2);

pub async fn run(socket: PathBuf, prompt: String, model: Option<String>) -> Result<()> {
    let stream = UnixStream::connect(&socket).await.with_context(|| {
        format!(
            "connecting to daemon at {} — is it running? (`amaebi daemon`)",
            socket.display()
        )
    })?;

    let (reader, mut writer) = tokio::io::split(stream);

    // Resolve model: CLI flag > AMAEBI_MODEL env var > default.
    let model = model
        .or_else(|| std::env::var("AMAEBI_MODEL").ok())
        .unwrap_or_else(|| "gpt-4o".to_string());

    // Build and send the request as a single JSON line.
    let req = Request::Chat {
        prompt,
        tmux_pane: std::env::var("TMUX_PANE").ok(),
        session_id: None,
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
    let mut lines = BufReader::new(reader).lines();
    let mut stdout = tokio::io::stdout();
    let mut sigint = signal(SignalKind::interrupt()).context("setting up SIGINT handler")?;
    // Timestamp of the first Ctrl-C press; None means no pending first press.
    let mut last_ctrl_c: Option<Instant> = None;

    loop {
        tokio::select! {
            biased;

            // Handle Ctrl-C (SIGINT).
            result = sigint.recv() => {
                // recv() returns None when the signal stream is closed; treat
                // that as a cue to stop rather than spinning the loop forever.
                let Some(_) = result else { break; };

                if is_within_window(last_ctrl_c, Instant::now(), DOUBLE_CTRLC_WINDOW) {
                    // Second Ctrl-C within the window — flush buffers then exit.
                    eprintln!();
                    let _ = stdout.flush().await;
                    let _ = std::io::Write::flush(&mut std::io::stderr());
                    std::process::exit(130);
                }
                // First press (or expired window): remind the user and record time.
                eprintln!(
                    "\nPress Ctrl-C again within {}s to exit",
                    DOUBLE_CTRLC_WINDOW.as_secs()
                );
                last_ctrl_c = Some(Instant::now());
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
                }
            }
        }
    }

    // Ensure the cursor ends up on a fresh line.
    stdout.write_all(b"\n").await.context("writing newline")?;

    Ok(())
}

/// Return `true` if `last_press` is `Some` and the duration from `last_press`
/// to `now` is less than `window`.
///
/// Accepts `now` as a parameter so callers pass `Instant::now()` and tests can
/// supply controlled values — the function itself has no side effects.
fn is_within_window(last_press: Option<Instant>, now: Instant, window: Duration) -> bool {
    match last_press {
        None => false,
        Some(t) => now.duration_since(t) < window,
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
    fn is_within_window_press_at_boundary_is_outside() {
        // Exactly at the boundary is not strictly less than the window.
        let now = Instant::now();
        let press = now - Duration::from_secs(2);
        assert!(!is_within_window(Some(press), now, Duration::from_secs(2)));
    }
}
