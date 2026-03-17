use anyhow::{Context, Result};
use std::path::PathBuf;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

use crate::ipc::{Request, Response};

pub async fn run(socket: PathBuf, prompt: String) -> Result<()> {
    let stream = UnixStream::connect(&socket).await.with_context(|| {
        format!(
            "connecting to daemon at {} — is it running? (`tmux-copilot daemon`)",
            socket.display()
        )
    })?;

    let (reader, mut writer) = tokio::io::split(stream);

    // Build and send the request as a single JSON line.
    let req = Request {
        prompt,
        tmux_pane: std::env::var("TMUX_PANE").ok(),
        session_id: None,
    };
    let mut req_line = serde_json::to_string(&req).context("serializing request")?;
    req_line.push('\n');
    writer
        .write_all(req_line.as_bytes())
        .await
        .context("sending request to daemon")?;

    // Stream responses to stdout until Done or Error.
    let mut lines = BufReader::new(reader).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = lines
        .next_line()
        .await
        .context("reading response from daemon")?
    {
        let resp: Response = serde_json::from_str(&line).context("parsing response frame")?;
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
            Response::ToolUse { name } => {
                // Tool notifications go to stderr so stdout stays clean for the AI response.
                eprintln!("[tool] {name}");
            }
        }
    }

    // Ensure the cursor ends up on a fresh line.
    stdout.write_all(b"\n").await.context("writing newline")?;

    Ok(())
}
