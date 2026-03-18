/// A message sent from the client to the daemon over the Unix socket.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    /// Send a prompt and stream a reply.
    Chat {
        /// The user's prompt text.
        prompt: String,
        /// Value of $TMUX_PANE at the time the client was invoked, if set.
        tmux_pane: Option<String>,
        /// Attach to an existing subagent session (Phase 4).
        session_id: Option<String>,
        /// Chat model to use (e.g. "gpt-4o").
        model: String,
    },
    /// Ask the daemon to clear its in-memory conversation cache.
    ///
    /// Sent after `amaebi memory clear` so the running daemon does not serve
    /// stale entries.  The daemon responds with a single [`Response::Done`] frame.
    ClearCache,
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Request serialization -------------------------------------------

    #[test]
    fn request_chat_round_trip() {
        let req = Request::Chat {
            prompt: "hello world".into(),
            tmux_pane: Some("%3".into()),
            session_id: None,
            model: "gpt-4o".into(),
        };
        let json = serde_json::to_string(&req).unwrap();
        // Wire format must carry the discriminant tag.
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "chat");

        let back: Request = serde_json::from_str(&json).unwrap();
        let Request::Chat {
            prompt,
            tmux_pane,
            session_id,
            model,
        } = back
        else {
            panic!("expected Chat variant");
        };
        assert_eq!(prompt, "hello world");
        assert_eq!(tmux_pane.as_deref(), Some("%3"));
        assert!(session_id.is_none());
        assert_eq!(model, "gpt-4o");
    }

    #[test]
    fn request_chat_optional_fields_can_be_null() {
        let json = r#"{"type":"chat","prompt":"p","tmux_pane":null,"session_id":null,"model":"m"}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        let Request::Chat {
            tmux_pane,
            session_id,
            ..
        } = req
        else {
            panic!("expected Chat variant");
        };
        assert!(tmux_pane.is_none());
        assert!(session_id.is_none());
    }

    #[test]
    fn request_clear_cache_round_trip() {
        let req = Request::ClearCache;
        let json = serde_json::to_string(&req).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "clear_cache");
        let back: Request = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, Request::ClearCache));
    }

    // ---- Response tag encoding ------------------------------------------

    #[test]
    fn response_text_has_snake_case_type() {
        let r = Response::Text { chunk: "hi".into() };
        let v: serde_json::Value = serde_json::to_value(&r).unwrap();
        assert_eq!(v["type"], "text");
        assert_eq!(v["chunk"], "hi");
    }

    #[test]
    fn response_done_has_snake_case_type() {
        let v: serde_json::Value = serde_json::to_value(Response::Done).unwrap();
        assert_eq!(v["type"], "done");
    }

    #[test]
    fn response_error_has_snake_case_type() {
        let r = Response::Error {
            message: "boom".into(),
        };
        let v: serde_json::Value = serde_json::to_value(&r).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["message"], "boom");
    }

    #[test]
    fn response_tool_use_has_snake_case_type() {
        let r = Response::ToolUse {
            name: "shell_command".into(),
            detail: "ls -la".into(),
        };
        let v: serde_json::Value = serde_json::to_value(&r).unwrap();
        assert_eq!(v["type"], "tool_use");
        assert_eq!(v["name"], "shell_command");
        assert_eq!(v["detail"], "ls -la");
    }

    #[test]
    fn response_all_variants_round_trip() {
        let frames = [
            r#"{"type":"text","chunk":"hello"}"#,
            r#"{"type":"done"}"#,
            r#"{"type":"error","message":"fail"}"#,
            r#"{"type":"tool_use","name":"read_file","detail":"/tmp/x"}"#,
        ];
        for frame in frames {
            let r: Response = serde_json::from_str(frame).unwrap();
            assert!(matches!(
                r,
                Response::Text { .. }
                    | Response::Done
                    | Response::Error { .. }
                    | Response::ToolUse { .. }
            ));
        }
    }

    // ---- write_frame -----------------------------------------------------

    #[tokio::test]
    async fn write_frame_emits_newline_terminated_json() {
        let mut buf: Vec<u8> = Vec::new();
        write_frame(&mut buf, &Response::Done).await.unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.ends_with('\n'), "frame must end with newline");
        let v: serde_json::Value = serde_json::from_str(s.trim_end()).unwrap();
        assert_eq!(v["type"], "done");
    }

    #[tokio::test]
    async fn write_frame_text_content_survives() {
        let mut buf: Vec<u8> = Vec::new();
        write_frame(
            &mut buf,
            &Response::Text {
                chunk: "streamed".into(),
            },
        )
        .await
        .unwrap();
        let s = String::from_utf8(buf).unwrap();
        let v: serde_json::Value = serde_json::from_str(s.trim_end()).unwrap();
        assert_eq!(v["chunk"], "streamed");
    }
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
