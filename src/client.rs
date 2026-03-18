use anyhow::{Context, Result};
use std::path::PathBuf;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::signal::unix::{signal, SignalKind};

use crate::ipc::{Request, Response};

/// How long (seconds) the user has to press Ctrl-C a second time to exit.
const DOUBLE_CTRLC_WINDOW_SECS: f64 = 2.0;

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
            _ = sigint.recv() => {
                if let Some(t) = last_ctrl_c {
                    if t.elapsed().as_secs_f64() < DOUBLE_CTRLC_WINDOW_SECS {
                        // Second Ctrl-C within the window — exit.
                        eprintln!();
                        std::process::exit(130);
                    }
                }
                // First press (or expired window): remind the user and record time.
                eprintln!("\nPress Ctrl-C again to exit");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_ctrlc_window_is_positive() {
        assert!(DOUBLE_CTRLC_WINDOW_SECS > 0.0);
    }

    #[test]
    fn instant_within_window() {
        let t = Instant::now();
        // A freshly captured instant should be within the 2-second window.
        assert!(t.elapsed().as_secs_f64() < DOUBLE_CTRLC_WINDOW_SECS);
    }

    #[test]
    fn instant_outside_window() {
        // An instant 3 seconds in the past should be outside the window.
        let t = Instant::now()
            .checked_sub(std::time::Duration::from_secs(3))
            .expect("invariant: 3-second subtraction fits in Instant range");
        assert!(t.elapsed().as_secs_f64() >= DOUBLE_CTRLC_WINDOW_SECS);
    }
}
