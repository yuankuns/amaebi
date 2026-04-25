/// A single task specification for [`Request::ClaudeLaunch`].
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct TaskSpec {
    /// User-supplied label, e.g. `"pr-123"` or `"issue-76"`.
    pub task_id: String,
    /// Description sent to the Claude session as the opening prompt.
    pub description: String,
    /// Optional absolute path to a git worktree for this task.
    /// Enforced as unique across all currently Busy panes.
    pub worktree: Option<String>,
    /// Absolute path to the client's working directory at the time `/claude`
    /// was invoked.  Used by the daemon to locate the correct git repository
    /// for auto-worktree creation, since the daemon may have been started from
    /// a different directory.
    pub client_cwd: Option<String>,
    /// If `false`, the command is injected into the pane without a trailing
    /// Enter key (useful for commands the user wants to review first).
    pub auto_enter: bool,
    /// Optional tmux pane id (e.g. `"%41"`) to reuse instead of allocating a
    /// new one.  When `Some`, the daemon validates the pane exists and has
    /// `has_claude=true` in the lease map, acquires THAT pane specifically
    /// (not a scheduler-picked one), inherits its existing worktree, and
    /// injects `/compact + description` (tier-1 reuse path).  Mutually
    /// exclusive with `worktree` at the CLI parser; the daemon treats a
    /// `Some(resume_pane)` as authoritative and ignores any stray `worktree`.
    #[serde(default)]
    pub resume_pane: Option<String>,
    /// Resource specs to acquire before the pane starts running `claude`.
    ///
    /// Each spec is either a resource name (looked up in
    /// `~/.amaebi/resources.toml`) or `class:<name>` / `any:<name>` / a
    /// bare class name when the entry matches a declared class with no
    /// literal resource of that name.  Held for the supervision lifetime of
    /// the pane and released when supervision exits.
    #[serde(default)]
    pub resources: Vec<String>,
    /// Seconds to wait for resources to free up before failing.  `None` or
    /// `0` means don't wait (fail immediately if any resource is busy).
    #[serde(default)]
    pub resource_timeout_secs: Option<u64>,
}

/// A single pane+task pair for supervision.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct SupervisionTarget {
    pub pane_id: String,
    pub task_description: String,
}

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
        /// Chat model to use (e.g. "bedrock/claude-sonnet-4.6" or "copilot/gpt-4o").
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
        /// Chat model to use (e.g. "bedrock/claude-sonnet-4.6" or "copilot/gpt-4o").
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
        /// Chat model to use (e.g. "bedrock/claude-sonnet-4.6" or "copilot/gpt-4o").
        model: String,
    },
    /// Ask the daemon to abort the current tool execution and skip remaining
    /// tool calls in the active agentic loop for the given session.
    ///
    /// Sent by the client immediately on the first Ctrl-C so the daemon can
    /// stop mid-chain without waiting for the user to type a correction.
    /// A [`Response::SteerAck`] is NOT sent in response; the loop simply
    /// drains any pending steer messages at the start of the next iteration.
    Interrupt {
        /// Session UUID of the loop to interrupt.
        session_id: String,
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
    /// Launch one or more independent `chat ↔ Claude` pairs in separate tmux
    /// panes.
    ///
    /// The daemon acquires pane leases (auto-expanding tmux panes if needed),
    /// starts each session, and responds with one [`Response::PaneAssigned`]
    /// per task followed by [`Response::Done`].  If the pane capacity limit
    /// would be exceeded it responds with [`Response::CapacityError`] instead.
    ClaudeLaunch {
        /// The tasks to launch in parallel.
        tasks: Vec<TaskSpec>,
    },
    /// Supervise tmux panes where Claude is executing tasks.
    /// The daemon runs a Rust polling loop: capture pane → LLM analysis → act.
    SupervisePanes {
        panes: Vec<SupervisionTarget>,
        model: String,
        session_id: Option<String>,
    },
}

/// A single frame streamed from the daemon back to the client.
///
/// Newline-delimited JSON: the daemon writes one frame per line;
/// the client reads lines until `Done` or `Error`.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
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
    /// The daemon has started a background compaction of conversation history.
    Compacting,
    /// The daemon is waiting for interactive user input before proceeding.
    ///
    /// Sent when the model asks a clarifying question or ends its response
    /// with a question mark.  The client should display a `>` cursor and
    /// forward the user's reply as a [`Request::Steer`] frame.
    WaitingForInput {
        /// Optional clarification text to display above the `>` cursor.
        ///
        /// **Empty string** (the common case): the question was already
        /// streamed to the client as [`Response::Text`] chunks, so the
        /// client should only show a bare `>` cursor.
        ///
        /// **Non-empty**: the daemon has additional context to display that
        /// was not part of the streamed text (e.g. a synthesised prompt);
        /// the client should print this text before the `>` cursor.
        prompt: String,
    },
    /// One tmux pane has been successfully assigned to a task launched via
    /// [`Request::ClaudeLaunch`].
    ///
    /// The client receives one frame per task, in submission order, before the
    /// final [`Response::Done`].
    PaneAssigned {
        /// The task label supplied in [`TaskSpec::task_id`].
        task_id: String,
        /// tmux pane ID, e.g. `"%3"`.
        pane_id: String,
        /// amaebi session UUID for the new chat session running in the pane.
        session_id: String,
    },
    /// The LLM called the `switch_model` tool and the active model changed.
    ///
    /// The client should update its local model variable so the next
    /// [`Request::Chat`] carries the new model, keeping `carried_model` in
    /// sync on the daemon side.
    ModelSwitched {
        /// The exact active model string forwarded by the daemon verbatim.
        /// May include provider prefixes such as `bedrock/...` or
        /// `copilot/...`; clients should treat it as opaque and send it back
        /// unchanged in the next [`Request::Chat`].
        model: String,
    },
    /// The [`Request::ClaudeLaunch`] was rejected because adding the requested
    /// panes would exceed the configured maximum.
    CapacityError {
        /// Number of tasks still unassigned when capacity was reached.
        ///
        /// This is the remaining portion of the launch request at the point of
        /// failure, not necessarily the original total number of tasks
        /// submitted by the client.
        requested: usize,
        /// Configured maximum total pane count.
        max_panes: usize,
        /// Number of panes currently in Busy state.
        current_busy: usize,
    },
}

// ---------------------------------------------------------------------------
// Frame helpers
// ---------------------------------------------------------------------------

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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncReadExt;

    // ---- Request serialization -------------------------------------------

    #[test]
    fn request_chat_round_trip() -> anyhow::Result<()> {
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
            anyhow::bail!("expected Chat variant");
        };
        assert_eq!(prompt, "hello world");
        assert_eq!(tmux_pane.as_deref(), Some("%3"));
        assert!(session_id.is_none());
        assert_eq!(model, "gpt-4o");
        Ok(())
    }

    #[test]
    fn request_chat_optional_fields_can_be_null() -> anyhow::Result<()> {
        let json = r#"{"type":"chat","prompt":"p","tmux_pane":null,"session_id":null,"model":"m"}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        let Request::Chat {
            tmux_pane,
            session_id,
            ..
        } = req
        else {
            anyhow::bail!("expected Chat variant");
        };
        assert!(tmux_pane.is_none());
        assert!(session_id.is_none());
        Ok(())
    }

    #[test]
    fn request_steer_round_trip() -> anyhow::Result<()> {
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
            anyhow::bail!("expected Steer variant");
        };
        assert_eq!(session_id, "abc-123");
        assert_eq!(message, "no, keep the old signature");
        Ok(())
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
    fn request_resume_round_trip() -> anyhow::Result<()> {
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
            anyhow::bail!("expected Resume variant");
        };
        assert_eq!(session_id, "old-uuid");
        assert_eq!(prompt, "continue");
        Ok(())
    }

    #[test]
    fn request_interrupt_round_trip() -> anyhow::Result<()> {
        let req = Request::Interrupt {
            session_id: "sess-xyz".into(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "interrupt");
        assert_eq!(v["session_id"], "sess-xyz");

        let back: Request = serde_json::from_str(&json).unwrap();
        let Request::Interrupt { session_id } = back else {
            anyhow::bail!("expected Interrupt variant");
        };
        assert_eq!(session_id, "sess-xyz");
        Ok(())
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
    fn response_waiting_for_input_round_trip() {
        let r = Response::WaitingForInput {
            prompt: "Which language?".into(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "waiting_for_input");
        assert_eq!(v["prompt"], "Which language?");
        let back: Response = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, Response::WaitingForInput { .. }));
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
            r#"{"type":"waiting_for_input","prompt":"Which language?"}"#,
            r#"{"type":"compacting"}"#,
            r#"{"type":"pane_assigned","task_id":"pr-1","pane_id":"%3","session_id":"uuid-abc"}"#,
            r#"{"type":"capacity_error","requested":3,"max_panes":16,"current_busy":14}"#,
            r#"{"type":"model_switched","model":"bedrock/claude-opus-4.7"}"#,
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
                    | Response::WaitingForInput { .. }
                    | Response::Compacting
                    | Response::PaneAssigned { .. }
                    | Response::CapacityError { .. }
                    | Response::ModelSwitched { .. }
            ));
        }
    }

    // ---- ClaudeLaunch / TaskSpec ------------------------------------------

    #[test]
    fn task_spec_round_trip() {
        let spec = TaskSpec {
            task_id: "pr-123".into(),
            description: "implement feature X".into(),
            worktree: Some("/home/user/repo-wt/feat-x".into()),
            client_cwd: Some("/home/user/repo".into()),
            auto_enter: true,
            resume_pane: None,
            resources: Vec::new(),
            resource_timeout_secs: None,
        };
        let json = serde_json::to_string(&spec).unwrap();
        let back: TaskSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.task_id, "pr-123");
        assert_eq!(back.description, "implement feature X");
        assert_eq!(back.worktree.as_deref(), Some("/home/user/repo-wt/feat-x"));
        assert!(back.auto_enter);
        assert!(back.resume_pane.is_none());
    }

    #[test]
    fn task_spec_resume_pane_round_trip() {
        // With resume_pane set.
        let spec = TaskSpec {
            task_id: "t1".into(),
            description: "continue".into(),
            worktree: None,
            client_cwd: None,
            auto_enter: true,
            resume_pane: Some("%41".into()),
            resources: Vec::new(),
            resource_timeout_secs: None,
        };
        let json = serde_json::to_string(&spec).unwrap();
        let back: TaskSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.resume_pane.as_deref(), Some("%41"));
    }

    #[test]
    fn task_spec_resume_pane_absent_field_deserializes_as_none() {
        // Older clients on master still encode without `resume_pane`; the
        // daemon must accept that payload and default the field to None.
        let legacy = r#"{"task_id":"x","description":"d","worktree":null,"client_cwd":null,"auto_enter":true}"#;
        let back: TaskSpec = serde_json::from_str(legacy).unwrap();
        assert!(back.resume_pane.is_none());
    }

    #[test]
    fn request_claude_launch_round_trip() {
        let req = Request::ClaudeLaunch {
            tasks: vec![
                TaskSpec {
                    task_id: "t1".into(),
                    description: "do A".into(),
                    worktree: None,
                    client_cwd: Some("/home/user/repo".into()),
                    auto_enter: true,
                    resume_pane: None,
                    resources: Vec::new(),
                    resource_timeout_secs: None,
                },
                TaskSpec {
                    task_id: "t2".into(),
                    description: "do B".into(),
                    worktree: Some("/wt/b".into()),
                    client_cwd: None,
                    auto_enter: false,
                    resume_pane: None,
                    resources: Vec::new(),
                    resource_timeout_secs: None,
                },
            ],
        };
        let json = serde_json::to_string(&req).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "claude_launch");
        assert_eq!(v["tasks"][0]["task_id"], "t1");
        assert_eq!(v["tasks"][1]["worktree"], "/wt/b");

        let back: Request = serde_json::from_str(&json).unwrap();
        let Request::ClaudeLaunch { tasks } = back else {
            panic!("expected ClaudeLaunch");
        };
        assert_eq!(tasks.len(), 2);
    }

    #[test]
    fn request_supervise_panes_round_trip() {
        let req = Request::SupervisePanes {
            panes: vec![SupervisionTarget {
                pane_id: "%3".into(),
                task_description: "implement feature X".into(),
            }],
            model: "gpt-4o".into(),
            session_id: Some("uuid-abc".into()),
        };
        let json = serde_json::to_string(&req).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "supervise_panes");
        assert_eq!(v["panes"][0]["pane_id"], "%3");
        assert_eq!(v["model"], "gpt-4o");
        assert_eq!(v["session_id"], "uuid-abc");

        let back: Request = serde_json::from_str(&json).unwrap();
        let Request::SupervisePanes {
            panes,
            model,
            session_id,
        } = back
        else {
            panic!("expected SupervisePanes");
        };
        assert_eq!(panes.len(), 1);
        assert_eq!(panes[0].pane_id, "%3");
        assert_eq!(model, "gpt-4o");
        assert_eq!(session_id.as_deref(), Some("uuid-abc"));
    }

    #[test]
    fn response_pane_assigned_round_trip() {
        let r = Response::PaneAssigned {
            task_id: "pr-123".into(),
            pane_id: "%3".into(),
            session_id: "uuid-xyz".into(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "pane_assigned");
        assert_eq!(v["task_id"], "pr-123");
        assert_eq!(v["pane_id"], "%3");
        assert_eq!(v["session_id"], "uuid-xyz");

        let back: Response = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, Response::PaneAssigned { .. }));
    }

    #[test]
    fn response_capacity_error_round_trip() {
        let r = Response::CapacityError {
            requested: 3,
            max_panes: 16,
            current_busy: 14,
        };
        let json = serde_json::to_string(&r).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "capacity_error");
        assert_eq!(v["requested"], 3);
        assert_eq!(v["max_panes"], 16);
        assert_eq!(v["current_busy"], 14);

        let back: Response = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, Response::CapacityError { .. }));
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
