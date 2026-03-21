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
    /// Inject a user correction into a running agentic loop on the same connection.
    ///
    /// The daemon drains these between model turns and injects each message as a
    /// `user` turn, then responds with [`Response::SteerAck`].  If the `session_id`
    /// does not match the active loop the frame is silently ignored (logged at
    /// debug level); no error response is sent.
    Steer {
        /// Session UUID to steer (matches the `session_id` of the active `Chat`).
        session_id: String,
        /// The correction text to inject as a user message.
        message: String,
    },
    /// Submit a task to run entirely in the background.
    ///
    /// The daemon spawns the agentic loop in a background `tokio::spawn`, responds
    /// immediately with [`Response::DetachAccepted`], and saves the final output to
    /// `~/.amaebi/inbox.db` when the loop completes.
    SubmitDetach {
        /// The user's prompt text.
        prompt: String,
        /// Value of `$TMUX_PANE` at the time the client was invoked, if set.
        tmux_pane: Option<String>,
        /// Session UUID to associate with this task.
        session_id: Option<String>,
        /// Chat model to use (e.g. "gpt-4o").
        model: String,
    },
    /// Resume a prior session by UUID, loading its **full** chronological history
    /// from the SQLite memory store.
    ///
    /// Unlike [`Request::Chat`], which applies a sliding-window cap over recent
    /// turns, this variant bypasses the `MAX_HISTORY` limit so the LLM receives the
    /// entire persisted conversation for `session_id`.  History is read from the
    /// SQLite DB on every call, so it survives daemon restarts.
    /// Useful for multi-day projects where full context matters.
    Resume {
        /// The prompt to send in this turn.
        prompt: String,
        /// Value of `$TMUX_PANE` at the time the client was invoked, if set.
        tmux_pane: Option<String>,
        /// Session UUID to re-hydrate.  Required — this variant always targets a
        /// specific historical session.
        session_id: String,
        /// Chat model to use (e.g. "gpt-4o").
        model: String,
    },
    /// Ask the daemon to clear its persisted SQLite memory database.
    ///
    /// Sent after `amaebi memory clear` so the running daemon also clears its
    /// copy of the SQLite database.  The daemon responds with a single
    /// [`Response::Done`] frame.
    ClearMemory,
    /// Ask the daemon to persist a user/assistant exchange.
    ///
    /// Used by the ACP agent so all SQLite writes go through the single daemon
    /// process.  The daemon responds with a single [`Response::Done`] frame.
    StoreMemory {
        /// The user's prompt text.
        user: String,
        /// The assistant's response text.
        assistant: String,
    },
    /// Ask the daemon for conversation context relevant to `prompt`.
    ///
    /// Used by the ACP agent so all SQLite reads go through the daemon.
    /// The daemon responds with zero or more [`Response::MemoryEntry`] frames
    /// followed by a single [`Response::Done`] frame.
    RetrieveContext { prompt: String },
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
    /// Acknowledgement that a [`Request::Steer`] correction was injected.
    SteerAck,
    /// Confirms that a [`Request::SubmitDetach`] was accepted.
    ///
    /// Sent immediately before the daemon starts the background task.  The client
    /// should print the session ID and exit cleanly.
    DetachAccepted {
        /// The session UUID under which the task will run.
        session_id: String,
    },
    /// A single memory entry returned in response to [`Request::RetrieveContext`].
    MemoryEntry {
        /// `"user"` or `"assistant"`.
        role: String,
        /// The message content.
        content: String,
    },
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncReadExt;

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
    fn request_steer_round_trip() {
        let req = Request::Steer {
            session_id: "abc-123".into(),
            message: "no, keep the old signature".into(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "steer");
        assert_eq!(v["session_id"], "abc-123");
        assert_eq!(v["message"], "no, keep the old signature");
        let back: Request = serde_json::from_str(&json).unwrap();
        let Request::Steer {
            session_id,
            message,
        } = back
        else {
            panic!("expected Steer variant");
        };
        assert_eq!(session_id, "abc-123");
        assert_eq!(message, "no, keep the old signature");
    }

    #[test]
    fn request_submit_detach_round_trip() {
        let req = Request::SubmitDetach {
            prompt: "run tests".into(),
            tmux_pane: None,
            session_id: Some("uuid-abc".into()),
            model: "gpt-4o".into(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "submit_detach");
        assert_eq!(v["prompt"], "run tests");
        let back: Request = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, Request::SubmitDetach { .. }));
    }

    #[test]
    fn request_resume_round_trip() {
        let req = Request::Resume {
            prompt: "continue".into(),
            tmux_pane: Some("%3".into()),
            session_id: "old-uuid".into(),
            model: "gpt-4o".into(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "resume");
        assert_eq!(v["session_id"], "old-uuid");
        let back: Request = serde_json::from_str(&json).unwrap();
        let Request::Resume {
            session_id, prompt, ..
        } = back
        else {
            panic!("expected Resume variant");
        };
        assert_eq!(session_id, "old-uuid");
        assert_eq!(prompt, "continue");
    }

    #[test]
    fn response_detach_accepted_round_trip() {
        let r = Response::DetachAccepted {
            session_id: "task-uuid".into(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "detach_accepted");
        assert_eq!(v["session_id"], "task-uuid");
        let back: Response = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, Response::DetachAccepted { .. }));
    }

    #[test]
    fn request_clear_memory_round_trip() {
        let req = Request::ClearMemory;
        let json = serde_json::to_string(&req).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "clear_memory");
        let back: Request = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, Request::ClearMemory));
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
    fn response_steer_ack_has_snake_case_type() {
        let v: serde_json::Value = serde_json::to_value(Response::SteerAck).unwrap();
        assert_eq!(v["type"], "steer_ack");
    }

    #[test]
    fn response_steer_ack_round_trip() {
        let json = r#"{"type":"steer_ack"}"#;
        let r: Response = serde_json::from_str(json).unwrap();
        assert!(matches!(r, Response::SteerAck));
    }

    #[test]
    fn request_store_memory_round_trip() {
        let req = Request::StoreMemory {
            user: "hello".into(),
            assistant: "world".into(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "store_memory");
        assert_eq!(v["user"], "hello");
        assert_eq!(v["assistant"], "world");
        let back: Request = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, Request::StoreMemory { .. }));
    }

    #[test]
    fn request_retrieve_context_round_trip() {
        let req = Request::RetrieveContext {
            prompt: "rust async".into(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "retrieve_context");
        assert_eq!(v["prompt"], "rust async");
        let back: Request = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, Request::RetrieveContext { .. }));
    }

    #[test]
    fn response_memory_entry_has_snake_case_type() {
        let r = Response::MemoryEntry {
            role: "user".into(),
            content: "hello".into(),
        };
        let v: serde_json::Value = serde_json::to_value(&r).unwrap();
        assert_eq!(v["type"], "memory_entry");
        assert_eq!(v["role"], "user");
        assert_eq!(v["content"], "hello");
    }

    #[test]
    fn response_all_variants_round_trip() {
        let frames = [
            r#"{"type":"text","chunk":"hello"}"#,
            r#"{"type":"done"}"#,
            r#"{"type":"error","message":"fail"}"#,
            r#"{"type":"tool_use","name":"read_file","detail":"/tmp/x"}"#,
            r#"{"type":"steer_ack"}"#,
            r#"{"type":"detach_accepted","session_id":"uuid-1"}"#,
            r#"{"type":"memory_entry","role":"user","content":"hi"}"#,
        ];
        for frame in frames {
            let r: Response = serde_json::from_str(frame).unwrap();
            assert!(matches!(
                r,
                Response::Text { .. }
                    | Response::Done
                    | Response::Error { .. }
                    | Response::ToolUse { .. }
                    | Response::SteerAck
                    | Response::DetachAccepted { .. }
                    | Response::MemoryEntry { .. }
            ));
        }
    }

    // ---- write_frame -----------------------------------------------------

    #[tokio::test]
    async fn write_frame_emits_newline_terminated_json() {
        let (mut writer, mut reader) = tokio::io::duplex(1024);
        write_frame(&mut writer, &Response::Done).await.unwrap();
        drop(writer);
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).await.unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.ends_with('\n'), "frame must end with newline");
        let v: serde_json::Value = serde_json::from_str(s.trim_end()).unwrap();
        assert_eq!(v["type"], "done");
    }

    #[tokio::test]
    async fn write_frame_text_content_survives() {
        let (mut writer, mut reader) = tokio::io::duplex(1024);
        write_frame(
            &mut writer,
            &Response::Text {
                chunk: "streamed".into(),
            },
        )
        .await
        .unwrap();
        drop(writer);
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).await.unwrap();
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
