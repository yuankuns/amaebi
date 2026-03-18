/// A message sent from the client to the daemon over the Unix socket.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct Request {
    /// The user's prompt text.
    pub prompt: String,
    /// Value of $TMUX_PANE at the time the client was invoked, if set.
    pub tmux_pane: Option<String>,
    /// Attach to an existing subagent session (Phase 4).
    pub session_id: Option<String>,
    /// Chat model to use (e.g. "gpt-4o").
    pub model: String,
}

/// A single frame streamed from the daemon back to the client.
///
/// Newline-delimited JSON: the daemon writes one frame per line;
/// the client reads lines until `Done` or `Error`.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    /// A chunk of text to print immediately.
    Text { chunk: String },
    /// The stream is finished — the client should exit cleanly.
    Done,
    /// A hard error the client should display, then exit non-zero.
    Error { message: String },
    /// The agent is about to invoke a tool — the client may display this.
    ToolUse { name: String, detail: String },
}

/// Write one `Response` frame as a JSON line to `writer`.
pub async fn write_frame<W>(writer: &mut W, frame: &Response) -> anyhow::Result<()>
where
    W: tokio::io::AsyncWriteExt + Unpin,
{
    let mut line = serde_json::to_string(frame)?;
    line.push('\n');
    writer
        .write_all(line.as_bytes())
        .await
        .map_err(anyhow::Error::from)
}
