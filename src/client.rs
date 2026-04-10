use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::io::IsTerminal as _;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::signal::unix::{signal, SignalKind};

use crate::ipc::{Request, Response, TaskSpec};
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

// ---------------------------------------------------------------------------
// Markdown rendering
// ---------------------------------------------------------------------------

/// Streaming-friendly state machine that buffers raw Markdown text chunks and
/// emits complete render units (paragraphs, code blocks, tables) that can be
/// pretty-printed with [`termimad`].
#[derive(Default)]
struct MarkdownBuffer {
    buf: String,
    state: MdState,
    /// Byte offset into `buf` up to which lines have already been scanned.
    scanned: usize,
    /// First safe flush boundary within `[0, scanned)`.
    flush_at: Option<usize>,
}

#[derive(Default, PartialEq, Debug)]
enum MdState {
    #[default]
    Normal,
    InCodeBlock,
    InTable,
}

impl MarkdownBuffer {
    fn push(&mut self, chunk: &str) {
        self.buf.push_str(chunk);
        self.scan_new_lines();
    }

    /// Scan only newly completed lines, updating `self.state` and `self.flush_at`.
    ///
    /// Only processes lines that are newline-terminated so that a partial
    /// fence (`` ``` `` without its `\n`) never toggles state prematurely.
    /// The next call picks up the line once its terminating newline arrives.
    fn scan_new_lines(&mut self) {
        // Only scan up to the last `\n`; the trailing incomplete line (if any)
        // will be processed when its newline arrives.
        let complete_end = self.buf.rfind('\n').map_or(0, |p| p + 1);
        let mut offset = self.scanned;
        if offset >= complete_end {
            return;
        }
        while offset < complete_end {
            let rel_nl = self.buf[offset..complete_end].find('\n').unwrap();
            let line_end = offset + rel_nl + 1;
            let trimmed = self.buf[offset..line_end]
                .trim_end_matches('\n')
                .trim_end_matches('\r')
                .trim();
            match self.state {
                MdState::Normal => {
                    if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
                        self.state = MdState::InCodeBlock;
                    } else if trimmed.starts_with('|') {
                        self.state = MdState::InTable;
                    } else if trimmed.is_empty() && self.flush_at.is_none() {
                        self.flush_at = Some(line_end);
                    }
                }
                MdState::InCodeBlock => {
                    if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
                        self.state = MdState::Normal;
                        if self.flush_at.is_none() {
                            self.flush_at = Some(line_end);
                        }
                    }
                }
                MdState::InTable => {
                    if !trimmed.starts_with('|') && !trimmed.is_empty() {
                        self.state = MdState::Normal;
                        if self.flush_at.is_none() {
                            self.flush_at = Some(offset); // flush BEFORE the non-table line
                        }
                    }
                }
            }
            offset = line_end;
        }
        self.scanned = complete_end;
    }

    /// Scan `text` (complete lines, starting from `Normal` state) and return the
    /// byte offset of the first flush boundary.  Used after a drain to locate the
    /// next boundary in the already-scanned suffix of the buffer.
    fn find_first_boundary_from_normal(text: &str) -> Option<usize> {
        let mut state = MdState::Normal;
        let mut offset = 0usize;
        while offset < text.len() {
            let rel_nl = text[offset..].find('\n')?;
            let line_end = offset + rel_nl + 1;
            let trimmed = text[offset..line_end]
                .trim_end_matches('\n')
                .trim_end_matches('\r')
                .trim();
            match state {
                MdState::Normal => {
                    if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
                        state = MdState::InCodeBlock;
                    } else if trimmed.starts_with('|') {
                        state = MdState::InTable;
                    } else if trimmed.is_empty() {
                        return Some(line_end);
                    }
                }
                MdState::InCodeBlock => {
                    if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
                        return Some(line_end);
                    }
                }
                MdState::InTable => {
                    if !trimmed.starts_with('|') && !trimmed.is_empty() {
                        return Some(offset);
                    }
                }
            }
            offset = line_end;
        }
        None
    }

    fn flush_if_ready(&mut self) -> Option<String> {
        let end = self.flush_at.take()?;
        let para: String = self.buf.drain(..end).collect();
        self.scanned = self.scanned.saturating_sub(end);
        // After a flush the state machine is always back in Normal.
        // Re-scan the already-processed suffix to find the next boundary, if any.
        if self.scanned > 0 {
            self.flush_at = Self::find_first_boundary_from_normal(&self.buf[..self.scanned]);
        }
        if para.trim().is_empty() {
            None
        } else {
            Some(para)
        }
    }

    fn flush_all(&mut self) -> Option<String> {
        if self.buf.trim().is_empty() {
            self.buf.clear();
            self.scanned = 0;
            self.state = MdState::Normal;
            self.flush_at = None;
            return None;
        }
        let out = std::mem::take(&mut self.buf);
        self.scanned = 0;
        self.state = MdState::Normal;
        self.flush_at = None;
        Some(out)
    }
}

fn render_markdown(text: &str) -> String {
    static SKIN: std::sync::OnceLock<termimad::MadSkin> = std::sync::OnceLock::new();
    if std::io::stdout().is_terminal() {
        SKIN.get_or_init(termimad::MadSkin::default)
            .term_text(text)
            .to_string()
    } else {
        text.to_string()
    }
}

// ---------------------------------------------------------------------------
// /claude command parsing
// ---------------------------------------------------------------------------

/// A task parsed from the `/claude` command.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ClaudeTask {
    /// Short task label derived from the description.
    task_id: String,
    /// Task description / opening prompt.
    description: String,
    /// Optional absolute worktree path.
    worktree: Option<String>,
    /// Whether to auto-send Enter after injecting the command into the pane.
    auto_enter: bool,
}

/// Parse a `/claude` command into a list of [`ClaudeTask`]s.
///
/// Syntax:
/// ```text
/// /claude [--worktree <path>] [--no-enter] "task description" ["task2" ...]
/// ```
///
/// Returns:
/// - `None` — input does not start with `/claude ` (not a `/claude` command)
/// - `Some(Err(msg))` — input is a `/claude` command but has a parse error
/// - `Some(Ok(tasks))` — successfully parsed task list
fn parse_claude_command(input: &str) -> Option<Result<Vec<ClaudeTask>, String>> {
    let rest = input.strip_prefix("/claude ")?.trim();
    if rest.is_empty() {
        return Some(Err(
            "usage: /claude [--worktree <path>] [--no-enter] \"task description\" [...]"
                .to_string(),
        ));
    }

    // Tokenise (shell-style quoting); each entry is (token, was_quoted).
    let tokens = parse_quoted_args(rest);
    if tokens.is_empty() {
        return Some(Err(
            "usage: /claude [--worktree <path>] [--no-enter] \"task description\" [...]"
                .to_string(),
        ));
    }

    let mut worktree: Option<String> = None;
    let mut auto_enter = true;
    // (description, was_quoted) pairs for non-flag tokens.
    let mut desc_tokens: Vec<(String, bool)> = Vec::new();

    let mut i = 0;
    while i < tokens.len() {
        match tokens[i].0.as_str() {
            "--worktree" => {
                i += 1;
                // Require a non-flag value to follow --worktree.
                if i >= tokens.len() || tokens[i].0.starts_with("--") {
                    return Some(Err("--worktree requires a path argument".to_string()));
                }
                // Canonicalize to an absolute path so worktree uniqueness
                // checks in the daemon are reliable regardless of how the
                // path was spelled (relative vs. symlink vs. absolute).
                let raw = &tokens[i].0;
                let abs =
                    std::fs::canonicalize(raw).unwrap_or_else(|_| std::path::PathBuf::from(raw));
                worktree = Some(abs.to_string_lossy().into_owned());
            }
            "--no-enter" => {
                auto_enter = false;
            }
            tok => {
                desc_tokens.push((tok.to_string(), tokens[i].1));
            }
        }
        i += 1;
    }

    if desc_tokens.is_empty() {
        return Some(Err(
            "usage: /claude [--worktree <path>] [--no-enter] \"task description\" [...]"
                .to_string(),
        ));
    }

    // Build task list.  Quoted tokens each become a separate task.  Unquoted
    // tokens are joined as a single task (e.g. `/claude write some code` →
    // one task "write some code", not three separate tasks).
    let all_quoted = desc_tokens.iter().all(|(_, q)| *q);
    let descriptions: Vec<String> = if all_quoted || desc_tokens.len() == 1 {
        desc_tokens.into_iter().map(|(s, _)| s).collect()
    } else {
        // Mix of quoted and unquoted, or all unquoted: join as one description.
        vec![desc_tokens
            .into_iter()
            .map(|(s, _)| s)
            .collect::<Vec<_>>()
            .join(" ")]
    };

    let tasks: Vec<ClaudeTask> = descriptions
        .into_iter()
        .enumerate()
        .map(|(idx, desc)| {
            let task_id = make_task_id(&desc, idx);
            ClaudeTask {
                task_id,
                description: desc,
                worktree: worktree.clone(),
                auto_enter,
            }
        })
        .collect();

    Some(Ok(tasks))
}

/// Derive a short task label from a description + index.
fn make_task_id(description: &str, idx: usize) -> String {
    let lower = description.to_lowercase();
    let slug: String = lower
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();

    // Collapse runs of dashes.
    let mut result = String::new();
    let mut prev_dash = false;
    for c in slug.chars() {
        if c == '-' {
            if !prev_dash {
                result.push('-');
            }
            prev_dash = true;
        } else {
            result.push(c);
            prev_dash = false;
        }
    }
    let trimmed = result.trim_matches('-');

    let base = if trimmed.is_empty() {
        format!("task-{idx}")
    } else if trimmed.len() > 32 {
        trimmed[..32].trim_end_matches('-').to_string()
    } else {
        trimmed.to_string()
    };

    base
}

/// Parse shell-style arguments, supporting both quoted and unquoted tokens.
///
/// Returns `(token, was_quoted)` pairs so callers can distinguish
/// `"foo" "bar"` (two separately-quoted tasks) from `foo bar` (one task whose
/// words were split by whitespace).
///
/// - `"implement cron" "fix context limit"` → two quoted tokens
/// - `foo "bar"` → one unquoted + one quoted token
/// - Escaped quotes inside quoted strings: `"say \"hi\""` → `[r#"say "hi""#]`
fn parse_quoted_args(input: &str) -> Vec<(String, bool)> {
    let mut results = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        if ch == '"' {
            chars.next();
            let mut arg = String::new();
            loop {
                match chars.next() {
                    Some('\\') => {
                        if let Some(escaped) = chars.next() {
                            arg.push(escaped);
                        }
                    }
                    Some('"') => break,
                    Some(c) => arg.push(c),
                    None => break,
                }
            }
            let trimmed = arg.trim().to_string();
            if !trimmed.is_empty() {
                results.push((trimmed, true));
            }
        } else if ch.is_whitespace() {
            chars.next();
        } else {
            let mut arg = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() || c == '"' {
                    break;
                }
                arg.push(c);
                chars.next();
            }
            if !arg.is_empty() {
                results.push((arg, false));
            }
        }
    }

    results
}

pub async fn run(socket: PathBuf, prompt: String, model: Option<String>) -> Result<()> {
    // Register the SIGINT handler before connecting so double-Ctrl-C applies
    // for the entire command lifecycle, including the connection attempt.
    let mut sigint = signal(SignalKind::interrupt()).context("setting up SIGINT handler")?;

    let stream = connect_or_start_daemon(&socket).await?;

    let (reader, mut writer) = tokio::io::split(stream);

    // Resolve model: CLI flag > AMAEBI_MODEL env var > default.
    let model = model
        .or_else(|| std::env::var("AMAEBI_MODEL").ok())
        .unwrap_or_else(|| crate::provider::DEFAULT_MODEL.to_string());

    // Resolve the session UUID for the current working directory.
    // Wrapped in spawn_blocking because session::get_or_create does file I/O.
    let cwd = std::env::current_dir().context("getting current directory")?;
    let cwd_for_session = cwd.clone();
    let session_id = tokio::task::spawn_blocking(move || session::get_or_create(&cwd_for_session))
        .await
        .context("session::get_or_create panicked")?
        .unwrap_or_else(|e| {
            tracing::warn!(error = %e, "failed to resolve session id; using \"global\"");
            "global".to_string()
        });

    // Keep a copy for the steering requests and the exit footer.
    let session_id_copy = session_id.clone();

    // Intercept `/claude` commands: send a ClaudeLaunch request instead of Chat.
    if let Some(parse_result) = parse_claude_command(&prompt) {
        let tasks = match parse_result {
            Ok(t) => t,
            Err(msg) => {
                let mut stdout = tokio::io::stdout();
                stdout.write_all(msg.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
                return Ok(());
            }
        };
        let req = Request::ClaudeLaunch {
            tasks: tasks
                .into_iter()
                .map(|t| TaskSpec {
                    task_id: t.task_id,
                    description: t.description,
                    worktree: t.worktree,
                    auto_enter: t.auto_enter,
                })
                .collect(),
        };
        let mut req_line = serde_json::to_string(&req).context("serializing ClaudeLaunch")?;
        req_line.push('\n');
        writer
            .write_all(req_line.as_bytes())
            .await
            .context("sending ClaudeLaunch to daemon")?;

        let mut lines = BufReader::new(reader).lines();
        let mut stdout = tokio::io::stdout();
        loop {
            let line = lines.next_line().await.context("reading daemon response")?;
            let Some(line) = line else { break };
            let frame: Response = serde_json::from_str(&line).context("parsing daemon response")?;
            match frame {
                Response::Done => break,
                Response::Error { message } => {
                    stdout.write_all(message.as_bytes()).await?;
                    stdout.write_all(b"\n").await?;
                    break;
                }
                Response::PaneAssigned {
                    task_id,
                    pane_id,
                    session_id: sid,
                } => {
                    let msg = format!("[pane {pane_id}] {task_id} → session {sid}\n");
                    stdout.write_all(msg.as_bytes()).await?;
                }
                Response::CapacityError {
                    requested,
                    max_panes,
                    current_busy,
                } => {
                    let msg = format!(
                        "[error] capacity limit reached: max_panes={max_panes}, \
                         busy={current_busy}, requested={requested}; \
                         free existing panes to continue\n"
                    );
                    stdout.write_all(msg.as_bytes()).await?;
                    break;
                }
                _ => {}
            }
        }
        stdout.flush().await?;
        return Ok(());
    }

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
    let mut md_buf = MarkdownBuffer::default();

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
                            md_buf.push(&chunk);
                            while let Some(ready) = md_buf.flush_if_ready() {
                                let out = render_markdown(&ready);
                                stdout
                                    .write_all(out.as_bytes())
                                    .await
                                    .context("writing to stdout")?;
                                stdout.flush().await.context("flushing stdout")?;
                            }
                        }
                    }
                    Response::Done => {
                        if let Some(remaining) = md_buf.flush_all() {
                            let out = render_markdown(&remaining);
                            stdout.write_all(out.as_bytes()).await.context("writing to stdout")?;
                            stdout.flush().await.context("flushing stdout")?;
                        }
                        // Flush any buffered frames before exiting.
                        flush_steer_buffer(&mut steer_buffer, &mut buffer_truncated, &mut stdout).await?;
                        break;
                    }
                    Response::Error { message } => {
                        // Flush any pending markdown before surfacing the error.
                        if let Some(remaining) = md_buf.flush_all() {
                            let out = render_markdown(&remaining);
                            let _ = stdout.write_all(out.as_bytes()).await;
                            let _ = stdout.flush().await;
                        }
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
                            // Flush pending markdown before the tool notice.
                            if let Some(remaining) = md_buf.flush_all() {
                                let out = render_markdown(&remaining);
                                let _ = stdout.write_all(out.as_bytes()).await;
                                let _ = stdout.flush().await;
                            }
                            // Tool notifications go to stderr so stdout stays clean for the AI response.
                            eprintln!();
                            match name.as_str() {
                                "shell_command" => eprint!("{}", render_markdown(&format!("```bash\n$ {detail}\n```\n"))),
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
                        } else {
                            // Flush pending markdown before the compacting notice.
                            if let Some(remaining) = md_buf.flush_all() {
                                let out = render_markdown(&remaining);
                                let _ = stdout.write_all(out.as_bytes()).await;
                                let _ = stdout.flush().await;
                            }
                            if std::io::stderr().is_terminal() {
                                eprintln!("\n\x1b[1;5;34m[compacting conversation history…]\x1b[0m");
                            } else {
                                eprintln!("\n[compacting conversation history…]");
                            }
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
                    Response::PaneAssigned { .. } | Response::CapacityError { .. } => {
                        // Not expected in a normal Chat response stream.
                        tracing::debug!("unexpected pane scheduler response in chat loop");
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
        .unwrap_or_else(|| crate::provider::DEFAULT_MODEL.to_string());

    let cwd = std::env::current_dir().context("getting current directory")?;
    let cwd_for_session = cwd.clone();
    let session_id = tokio::task::spawn_blocking(move || session::create_fresh(&cwd_for_session))
        .await
        .context("session::create_fresh panicked")?
        .unwrap_or_else(|e| {
            tracing::warn!(error = %e, "failed to create fresh session id; using \"global\"");
            "global".to_string()
        });

    if std::io::stderr().is_terminal() {
        eprintln!("Chat session started (ID: {session_id}). Empty line or Ctrl-D to exit.\n");
    }

    let stream = connect_or_start_daemon(&socket).await?;
    let (read_half, mut write_half) = stream.into_split();
    let mut lines = BufReader::new(read_half).lines();

    let mut stdout = tokio::io::stdout();
    // In-flight spawn_blocking task for reading a steer correction in raw mode.
    // Spawned when steer_pending becomes true; awaited in the inner select! arm.
    // Using raw mode (same as the main prompt) gives the user history navigation
    // and CJK-width-correct backspace inside steer corrections too.
    let mut steer_task: Option<tokio::task::JoinHandle<std::io::Result<Option<String>>>> = None;
    let mut next_prompt = initial_prompt;
    let mut last_ctrl_c: Option<Instant> = None;
    let mut steer_pending = false;
    // Text chunks buffered while a steer correction is being typed so that
    // streaming output does not interleave with user input.
    let mut steer_text_buf: Vec<String> = Vec::new();
    let mut md_buf = MarkdownBuffer::default();

    'session: loop {
        let prompt = match next_prompt.take() {
            Some(p) => p,
            None => {
                if std::io::stderr().is_terminal() {
                    eprint!("{}", prompt_input::PROMPT);
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

        // Intercept `/claude` commands: send ClaudeLaunch and handle the response
        // before resuming the normal chat loop.
        if let Some(parse_result) = parse_claude_command(&prompt) {
            let tasks = match parse_result {
                Ok(t) => t,
                Err(msg) => {
                    stdout.write_all(msg.as_bytes()).await?;
                    stdout.write_all(b"\n").await?;
                    stdout.flush().await?;
                    continue 'session;
                }
            };
            let req = Request::ClaudeLaunch {
                tasks: tasks
                    .into_iter()
                    .map(|t| TaskSpec {
                        task_id: t.task_id,
                        description: t.description,
                        worktree: t.worktree,
                        auto_enter: t.auto_enter,
                    })
                    .collect(),
            };
            let mut req_line = serde_json::to_string(&req)?;
            req_line.push('\n');
            write_half.write_all(req_line.as_bytes()).await?;

            // Read ClaudeLaunch responses until Done/Error.
            loop {
                let line = lines.next_line().await.context("reading daemon response")?;
                let Some(line) = line else { break 'session };
                let frame: Response =
                    serde_json::from_str(&line).context("parsing daemon response")?;
                match frame {
                    Response::Done => break,
                    Response::Error { message } => {
                        stdout.write_all(message.as_bytes()).await?;
                        stdout.write_all(b"\n").await?;
                        break;
                    }
                    Response::PaneAssigned {
                        task_id,
                        pane_id,
                        session_id: sid,
                    } => {
                        let msg = format!("[pane {pane_id}] {task_id} → session {sid}\n");
                        stdout.write_all(msg.as_bytes()).await?;
                    }
                    Response::CapacityError {
                        requested,
                        max_panes,
                        current_busy,
                    } => {
                        let msg = format!(
                            "[error] capacity limit reached: max_panes={max_panes}, \
                             busy={current_busy}, requested={requested}; \
                             free existing panes to continue\n"
                        );
                        stdout.write_all(msg.as_bytes()).await?;
                        break;
                    }
                    _ => {}
                }
            }
            stdout.flush().await?;
            // Continue the chat loop for the next user input.
            continue 'session;
        }

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
                        eprint!("{}", prompt_input::PROMPT);
                        let _ = tokio::io::stderr().flush().await;
                    }
                    if steer_task.is_none() {
                        steer_task = Some(tokio::task::spawn_blocking(
                            prompt_input::read_line_raw,
                        ));
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
                                md_buf.push(&chunk);
                                while let Some(ready) = md_buf.flush_if_ready() {
                                    let out = render_markdown(&ready);
                                    stdout.write_all(out.as_bytes()).await?;
                                    stdout.flush().await?;
                                }
                            }
                        }
                        Response::Done => {
                            // Drain any steer-buffered text through md_buf before
                            // flush_all so the final output is markdown-rendered.
                            for chunk in steer_text_buf.drain(..) {
                                md_buf.push(&chunk);
                            }
                            if let Some(remaining) = md_buf.flush_all() {
                                let out = render_markdown(&remaining);
                                stdout.write_all(out.as_bytes()).await.context("writing to stdout")?;
                                stdout.flush().await.context("flushing stdout")?;
                            }
                            stdout.write_all(b"\n").await?;
                            stdout.flush().await?;
                            break;
                        }
                        Response::Error { message } => {
                            // Flush any pending markdown before surfacing the error.
                            if let Some(remaining) = md_buf.flush_all() {
                                let out = render_markdown(&remaining);
                                let _ = stdout.write_all(out.as_bytes()).await;
                                let _ = stdout.flush().await;
                            }
                            anyhow::bail!("{message}");
                        }
                        Response::ToolUse { name, detail } => {
                            // Flush pending markdown before the tool notice.
                            if let Some(remaining) = md_buf.flush_all() {
                                let out = render_markdown(&remaining);
                                let _ = stdout.write_all(out.as_bytes()).await;
                                let _ = stdout.flush().await;
                            }
                            if std::io::stderr().is_terminal() {
                                match name.as_str() {
                                    "shell_command" => eprint!("{}", render_markdown(&format!("```bash\n$ {detail}\n```\n"))),
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
                            if std::io::stderr().is_terminal() {
                                if !extra.is_empty() { eprintln!("\n{extra}"); }
                                eprint!("{}", prompt_input::PROMPT);
                                let _ = tokio::io::stderr().flush().await;
                            }
                            steer_pending = true;
                            if steer_task.is_none() {
                                steer_task = Some(tokio::task::spawn_blocking(
                                    prompt_input::read_line_raw,
                                ));
                            }
                        }
                        Response::SteerAck => {
                            steer_pending = false;
                            // Flush text buffered while steer was pending through md_buf.
                            for chunk in steer_text_buf.drain(..) {
                                md_buf.push(&chunk);
                                if let Some(ready) = md_buf.flush_if_ready() {
                                    let out = render_markdown(&ready);
                                    let _ = stdout.write_all(out.as_bytes()).await;
                                }
                            }
                            let _ = stdout.flush().await;
                        }
                        Response::Compacting => {
                            // Flush pending markdown before the compacting notice.
                            if let Some(remaining) = md_buf.flush_all() {
                                let out = render_markdown(&remaining);
                                let _ = stdout.write_all(out.as_bytes()).await;
                                let _ = stdout.flush().await;
                            }
                            if std::io::stderr().is_terminal() { eprintln!("\n[compacting…]"); }
                        }
                        _ => {}
                    }
                }

                steer_result = async {
                    match steer_task.as_mut() {
                        Some(h) => h.await.map_err(std::io::Error::other),
                        None => std::future::pending().await,
                    }
                } => {
                    steer_task = None;
                    match steer_result {
                        // User typed a non-empty line — send as steer correction.
                        Ok(Ok(Some(text))) if !text.trim().is_empty() => {
                            let steer_req = Request::Steer {
                                session_id: session_id.clone(),
                                message: text.trim().to_owned(),
                            };
                            if let Ok(mut frame) = serde_json::to_string(&steer_req) {
                                frame.push('\n');
                                let _ = write_half.write_all(frame.as_bytes()).await;
                            }
                            steer_pending = false;
                            last_ctrl_c = None;
                        }
                        // Empty line — cancel steer.
                        Ok(Ok(Some(_))) => {
                            steer_pending = false;
                            last_ctrl_c = None;
                            // Flush buffered text through md_buf now that steer is cancelled.
                            for chunk in steer_text_buf.drain(..) {
                                md_buf.push(&chunk);
                                while let Some(ready) = md_buf.flush_if_ready() {
                                    let out = render_markdown(&ready);
                                    let _ = stdout.write_all(out.as_bytes()).await;
                                }
                            }
                            let _ = stdout.flush().await;
                        }
                        // I/O error from the steer read.
                        Ok(Err(e)) => {
                            if e.kind() == std::io::ErrorKind::Interrupted {
                                // Ctrl-C during steer input — cancel.
                                steer_pending = false;
                                last_ctrl_c = None;
                                for chunk in steer_text_buf.drain(..) {
                                    md_buf.push(&chunk);
                                    while let Some(ready) = md_buf.flush_if_ready() {
                                        let out = render_markdown(&ready);
                                        let _ = stdout.write_all(out.as_bytes()).await;
                                    }
                                }
                                let _ = stdout.flush().await;
                            } else {
                                // Real I/O error — surface and exit.
                                return Err(
                                    anyhow::Error::new(e).context("steer input error")
                                );
                            }
                        }
                        // EOF (Ctrl-D) or task panic — exit session.
                        Ok(Ok(None)) | Err(_) => break 'session,
                    }
                }
            }
        }
    }

    // If a steer task was in-flight when the session ended its JoinHandle is
    // dropped here, detaching the blocking thread.  That thread is stuck in
    // read_exact and will never drop its RawModeGuard.  Restore the terminal
    // immediately so the shell gets a sane state.
    #[cfg(unix)]
    if steer_task.is_some() {
        prompt_input::restore_terminal_now();
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
        .unwrap_or_else(|| crate::provider::DEFAULT_MODEL.to_string());

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
        .unwrap_or_else(|| crate::provider::DEFAULT_MODEL.to_string());

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
    let mut md_buf = MarkdownBuffer::default();

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
                            md_buf.push(&chunk);
                            if let Some(ready) = md_buf.flush_if_ready() {
                                let out = render_markdown(&ready);
                                stdout.write_all(out.as_bytes()).await.context("writing to stdout")?;
                                stdout.flush().await.context("flushing stdout")?;
                            }
                        }
                    }
                    Response::Done => {
                        if let Some(remaining) = md_buf.flush_all() {
                            let out = render_markdown(&remaining);
                            let _ = stdout.write_all(out.as_bytes()).await;
                            let _ = stdout.flush().await;
                        }
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
                                "shell_command" => eprint!("{}", render_markdown(&format!("```bash\n$ {detail}\n```\n"))),
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
                    Response::PaneAssigned { .. } | Response::CapacityError { .. } => {
                        tracing::debug!("unexpected pane scheduler response in resume loop");
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
    // Collect all buffered text through a local MarkdownBuffer so the
    // suppressed output is rendered consistently with the normal streaming
    // path.  Frames are replayed in original order; pending markdown is flushed
    // when a non-text frame is encountered so preceding content is fully
    // emitted before tool/compacting notices appear.
    let mut md_buf = MarkdownBuffer::default();
    for resp in buffer.drain(..) {
        match resp {
            Response::Text { chunk } => {
                md_buf.push(&chunk);
                while let Some(ready) = md_buf.flush_if_ready() {
                    let out = render_markdown(&ready);
                    stdout
                        .write_all(out.as_bytes())
                        .await
                        .context("writing buffered text to stdout")?;
                }
            }
            Response::ToolUse { name, detail } => {
                // Flush any pending markdown before printing the tool notice.
                if let Some(remaining) = md_buf.flush_all() {
                    let out = render_markdown(&remaining);
                    stdout
                        .write_all(out.as_bytes())
                        .await
                        .context("writing buffered text to stdout")?;
                    stdout.flush().await.context("flushing stdout")?;
                }
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
                if let Some(remaining) = md_buf.flush_all() {
                    let out = render_markdown(&remaining);
                    stdout
                        .write_all(out.as_bytes())
                        .await
                        .context("writing buffered text to stdout")?;
                    stdout.flush().await.context("flushing stdout")?;
                }
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
    // Flush any remaining markdown (e.g. last paragraph without trailing \n).
    if let Some(remaining) = md_buf.flush_all() {
        let out = render_markdown(&remaining);
        stdout
            .write_all(out.as_bytes())
            .await
            .context("writing buffered text to stdout")?;
        stdout.flush().await.context("flushing stdout")?;
    }
    stdout
        .flush()
        .await
        .context("flushing stdout after steer buffer")?;
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
    use std::collections::VecDeque;
    use std::io::{Read, Write};
    use std::sync::{Arc, Mutex, OnceLock};
    use unicode_width::UnicodeWidthChar as _;

    /// The prompt prefix displayed before each input line.  Defined here as a
    /// single source of truth so the outer chat loop and the internal `redraw()`
    /// always print exactly the same string.
    pub(super) const PROMPT: &str = "> ";

    /// Maximum number of entries kept in the prompt history.
    const MAX_HISTORY: usize = 1000;

    /// Process-wide history buffer.  Using a `static` instead of `thread_local!`
    /// ensures history persists across `tokio::task::spawn_blocking` calls, which
    /// are not guaranteed to reuse the same OS thread.
    ///
    /// `VecDeque` gives O(1) front-eviction when the buffer is full.
    /// `Arc<str>` entries mean history snapshots only clone the pointer, not the
    /// string bytes, keeping each snapshot O(n arcs) rather than O(total bytes).
    static HISTORY: OnceLock<Mutex<VecDeque<Arc<str>>>> = OnceLock::new();

    fn history_store() -> &'static Mutex<VecDeque<Arc<str>>> {
        HISTORY.get_or_init(|| Mutex::new(VecDeque::new()))
    }

    /// Saved original terminal state set when raw mode is entered, cleared on exit.
    /// Allows `restore_terminal_now()` to recover the terminal if the session exits
    /// while a `spawn_blocking(read_line_raw)` task is still in flight (the detached
    /// thread holds `RawModeGuard` but is blocked in `read_exact` indefinitely).
    #[cfg(unix)]
    static SAVED_ORIG_TERM: OnceLock<Mutex<Option<(std::os::unix::io::RawFd, libc::termios)>>> =
        OnceLock::new();

    #[cfg(unix)]
    fn saved_orig_term() -> &'static Mutex<Option<(std::os::unix::io::RawFd, libc::termios)>> {
        SAVED_ORIG_TERM.get_or_init(|| Mutex::new(None))
    }

    /// Restore the terminal to the state saved when raw mode was last entered.
    /// No-op if raw mode is not currently active.  Safe to call from async context.
    #[cfg(unix)]
    pub fn restore_terminal_now() {
        if let Some((fd, orig)) = *saved_orig_term().lock().unwrap_or_else(|p| p.into_inner()) {
            let rc = unsafe { libc::tcsetattr(fd, libc::TCSANOW, &orig) };
            if rc != 0 {
                eprintln!(
                    "warning: failed to restore terminal settings: {}",
                    std::io::Error::last_os_error()
                );
            }
        }
    }

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
            *saved_orig_term().lock().unwrap_or_else(|p| p.into_inner()) = Some((fd, orig));
            Ok(Self { fd, orig })
        }
    }

    #[cfg(unix)]
    impl Drop for RawModeGuard {
        fn drop(&mut self) {
            *saved_orig_term().lock().unwrap_or_else(|p| p.into_inner()) = None;
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

    /// Strip one trailing `\n` (and an optional preceding `\r`) from `s` in place.
    fn strip_trailing_newline(s: &mut String) {
        if s.ends_with('\n') {
            s.pop();
        }
        if s.ends_with('\r') {
            s.pop();
        }
    }

    /// Cooked-mode fallback used when stdin is not a TTY or raw mode fails.
    fn read_line_cooked() -> std::io::Result<Option<String>> {
        use std::io::BufRead as _;
        let mut s = String::new();
        match std::io::stdin().lock().read_line(&mut s) {
            Ok(0) => Ok(None),
            Ok(_) => {
                strip_trailing_newline(&mut s);
                Ok(Some(s))
            }
            Err(e) => Err(e),
        }
    }

    #[cfg(unix)]
    fn read_line_raw_inner(_guard: RawModeGuard) -> std::io::Result<Option<String>> {
        // Snapshot history while holding the lock for the minimum time, then
        // release it before blocking on I/O so other threads are not stalled.
        // Cloning Arc<str> entries is O(n arcs), not O(total string bytes).
        let history_snapshot: Vec<Arc<str>> = history_store()
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .cloned()
            .collect();
        let result = process_input(
            &mut std::io::stdin(),
            &mut std::io::stderr(),
            &history_snapshot,
            PROMPT,
            terminal_cols(),
        );
        if let Ok(Some(ref line)) = result {
            if !line.is_empty() {
                let mut hist = history_store().lock().unwrap_or_else(|p| p.into_inner());
                if hist.len() >= MAX_HISTORY {
                    hist.pop_front();
                }
                hist.push_back(Arc::from(line.as_str()));
            }
        }
        result
    }

    /// Returns the terminal width in columns by querying stderr with `TIOCGWINSZ`.
    /// Falls back to 80 if the ioctl fails (e.g. in tests or when stderr is redirected).
    #[cfg(unix)]
    fn terminal_cols() -> usize {
        unsafe {
            let mut ws = std::mem::zeroed::<libc::winsize>();
            if libc::ioctl(libc::STDERR_FILENO, libc::TIOCGWINSZ, &mut ws) == 0 && ws.ws_col > 0 {
                ws.ws_col as usize
            } else {
                80
            }
        }
    }

    #[cfg(not(unix))]
    fn terminal_cols() -> usize {
        80
    }

    /// Redraw the current line after any edit or cursor movement.
    ///
    /// Moves up to the first visual line of the prompt, erases to end of
    /// display (multi-line input) or end of line (single-line input), reprints
    /// `prompt` and all `chars`, then repositions the terminal cursor to
    /// `cursor`.  `term_cols` is the terminal width used for line-wrap math.
    #[allow(clippy::too_many_arguments)]
    fn redraw<W: Write>(
        output: &mut W,
        prompt: &str,
        chars: &[char],
        widths: &[usize],
        cursor: usize,
        term_cols: usize,
        from_line: usize,
        old_end_line: usize,
    ) -> std::io::Result<()> {
        let prompt_cols: usize = prompt.chars().map(|c| c.width().unwrap_or(1)).sum();
        let cols_before_cursor: usize = widths[..cursor].iter().sum();
        let total_char_width: usize = widths.iter().sum();

        let cursor_col_abs = prompt_cols + cols_before_cursor;
        let end_col_abs = prompt_cols + total_char_width;

        // ANSI terminals use a "pending wrap" flag: a character printed in the
        // last column stays on that line until the next printable char.  So a
        // cursor or end position that is an exact multiple of term_cols is still
        // on the *previous* visual line, not the next one.
        let visual_line_of = |col: usize| {
            if col == 0 {
                0
            } else {
                (col - 1) / term_cols
            }
        };
        let cursor_visual_line = visual_line_of(cursor_col_abs);
        let end_visual_line = visual_line_of(end_col_abs);

        // Move up to the first visual line of this prompt before clearing.
        // Use `from_line` (the actual terminal cursor line) rather than the
        // new cursor's visual line: when the user presses Home the new cursor
        // is on line 0 but the terminal cursor is still on line 1, so we must
        // move up by `from_line` lines to reach the top of the prompt.
        if from_line > 0 {
            write!(output, "\x1b[{from_line}A")?;
        }

        // Erase: use ED (erase to end of display) when the old or new content
        // spans multiple visual lines.  We need the larger of:
        //   - end_visual_line: how many lines the new content needs
        //   - old_end_line:    how many lines the old content occupied
        //                      (independent of where the cursor was; e.g. the
        //                      user pressed Home before navigating history, so
        //                      from_line==0 but old wrapped lines still exist)
        // EL (\r\x1b[K) suffices only when both old and new fit on one line.
        if end_visual_line > 0 || old_end_line > 0 {
            output.write_all(b"\r\x1b[J")?;
        } else {
            output.write_all(b"\r\x1b[K")?;
        }

        // Reprint prompt and all characters.
        output.write_all(prompt.as_bytes())?;
        for ch in chars {
            let mut buf = [0u8; 4];
            output.write_all(ch.encode_utf8(&mut buf).as_bytes())?;
        }

        // Reposition the terminal cursor to `cursor`.
        // After printing all chars the terminal cursor sits at end_visual_line.
        let cols_from_cursor: usize = widths[cursor..].iter().sum();
        if cols_from_cursor > 0 {
            let lines_above_end = end_visual_line - cursor_visual_line;
            if lines_above_end == 0 {
                // Same visual line — simple backward move.
                write!(output, "\x1b[{cols_from_cursor}D")?;
            } else {
                // Move up from end_visual_line to cursor_visual_line, then
                // jump to the exact column with CHA (1-based).
                write!(output, "\x1b[{lines_above_end}A")?;
                // CHA is 1-based.  Use the same pending-wrap adjustment as
                // visual_line_of: a cursor at an exact column multiple sits at
                // the last column of the previous line, not column 1 of the
                // next, so saturating_sub(1) before the modulo is required.
                let target_col = cursor_col_abs.saturating_sub(1) % term_cols + 1;
                write!(output, "\x1b[{target_col}G")?;
            }
        }

        Ok(())
    }

    /// Core line-editor loop.  Separated from the raw-mode setup so it can be
    /// driven by in-memory byte slices in unit tests without needing a real TTY.
    ///
    /// Reads bytes from `input`, echoes/erases via `output`, and returns:
    /// * `Ok(Some(s))` on Enter — `s` contains no trailing newline.
    /// * `Ok(None)` on EOF (Ctrl-D or exhausted reader).
    /// * `Err` with `ErrorKind::Interrupted` on Ctrl-C.
    ///
    /// `history` is a slice of prior input lines; up/down arrows navigate it.
    /// `prompt` is the prefix string already displayed on the current line; it
    /// is reprinted on every redraw so cursor movements do not erase it.
    fn process_input<R: Read, W: Write>(
        input: &mut R,
        output: &mut W,
        history: &[Arc<str>],
        prompt: &str,
        term_cols: usize,
    ) -> std::io::Result<Option<String>> {
        let prompt_cols: usize = prompt.chars().map(|c| c.width().unwrap_or(1)).sum();
        // See redraw() for why (col-1)/term_cols is used instead of col/term_cols.
        let visual_line_of = |col: usize| if col == 0 { 0 } else { (col - 1) / term_cols };
        let mut chars: Vec<char> = Vec::new();
        // Display width (columns) of each char, matched by index to `chars`.
        let mut widths: Vec<usize> = Vec::new();
        // Cursor position: index into `chars` (0 = start, chars.len() = end).
        let mut cursor: usize = 0;
        // History navigation state.
        // `hist_idx` == history.len() means "current draft" (not navigating history).
        let mut hist_idx: usize = history.len();
        // Draft saved the first time the user presses Up from the live buffer,
        // including the cursor position so returning from history restores it.
        let mut draft_chars: Vec<char> = Vec::new();
        let mut draft_cursor: usize = 0;

        loop {
            // Visual line (0-based) where the terminal cursor currently sits.
            // Every key handler in this iteration leaves the terminal cursor at
            // `cursor`'s position, so at the top of each iteration `cursor`
            // reflects the terminal position from the previous iteration.
            let terminal_line =
                visual_line_of(prompt_cols + widths[..cursor].iter().sum::<usize>());
            // Visual line of the last character in the buffer (the full extent
            // of the old content on screen).  Needed by redraw() so it can
            // choose ED over EL even when the cursor is at position 0 (e.g.
            // after Home) but wrapped lines still occupy lines 1+.
            let terminal_end_line = visual_line_of(prompt_cols + widths.iter().sum::<usize>());

            let mut byte = [0u8; 1];
            match input.read_exact(&mut byte) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
                // EINTR (Interrupted) is raised by signals such as SIGWINCH
                // (terminal resize).  It is not a fatal error — just retry.
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }

            match byte[0] {
                // Enter (CR in raw mode; some environments send LF)
                b'\r' | b'\n' => {
                    output.write_all(b"\r\n")?;
                    output.flush()?;
                    return Ok(Some(chars.iter().collect()));
                }
                // Ctrl-D
                4 => return Ok(None),
                // Ctrl-C
                3 => {
                    output.write_all(b"^C\r\n")?;
                    output.flush()?;
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Interrupted,
                        "ctrl-c",
                    ));
                }
                // Backspace (DEL = 0x7f on most terminals; BS = 0x08 on some)
                0x7f | 0x08 => {
                    if cursor > 0 {
                        let w = widths[cursor - 1];
                        chars.remove(cursor - 1);
                        widths.remove(cursor - 1);
                        cursor -= 1;
                        if w > 0 {
                            // Normal (non-zero-width) char: reposition and erase.
                            //
                            // ESC[wD (CUB) is used instead of w×\x08 (BS) because some
                            // terminals snap the cursor to the left boundary of a wide
                            // glyph on the first \x08, causing two \x08 to overshoot by
                            // one column at odd column positions and leave the right-half
                            // cell visible.  CUB always moves exactly w columns.
                            //
                            // Known limitation: CUB does not wrap to the previous visual
                            // line on soft-wrapped input.  This editor targets single-line
                            // prompts; multi-line soft-wrap support is out of scope.
                            if cursor == chars.len() {
                                // Cursor is at end — simple inline erase.
                                write!(output, "\x1b[{w}D\x1b[K")?;
                            } else {
                                // Mid-line deletion: full redraw to shift remaining chars.
                                redraw(
                                    output,
                                    prompt,
                                    &chars,
                                    &widths,
                                    cursor,
                                    term_cols,
                                    terminal_line,
                                    terminal_end_line,
                                )?;
                            }
                        } else {
                            // Zero-width combining mark: it was rendered on top of the
                            // preceding base character without advancing the cursor.
                            // Always do a full redraw: if there was a preceding base char
                            // we need to repaint it cleanly; if the mark was the very
                            // first character it may have combined visually with the
                            // trailing character of the prompt, so reprinting the prompt
                            // via redraw() is the only way to restore it.
                            redraw(
                                output,
                                prompt,
                                &chars,
                                &widths,
                                cursor,
                                term_cols,
                                terminal_line,
                                terminal_end_line,
                            )?;
                        }
                    }
                }
                // Escape sequences (arrows, function keys).
                //
                // Limitation: a bare ESC press (no following byte) will block
                // here until the next character arrives because we read the
                // second byte unconditionally.  Resolving this correctly
                // requires either a VTIME-based read timeout (termios VMIN=0
                // VTIME=1) or non-blocking I/O with a short poll, which adds
                // significant complexity for a rarely-used key.  Users who
                // need bare ESC can press Ctrl-C instead.
                0x1b => {
                    let mut eb = [0u8; 1];
                    if input.read_exact(&mut eb).is_ok() {
                        match eb[0] {
                            // CSI sequence (ESC [): read parameter bytes (0x30–0x3F) then
                            // the final byte (0x40–0x7E).  Accumulate params
                            // so we can distinguish bare sequences (arrows) from
                            // parametrised ones (ESC[1~ / ESC[4~, etc.).
                            b'[' => {
                                let mut params: Vec<u8> = Vec::new();
                                let final_byte = loop {
                                    let mut fb = [0u8; 1];
                                    if input.read_exact(&mut fb).is_err() {
                                        break None;
                                    }
                                    if (0x40..=0x7e).contains(&fb[0]) {
                                        break Some(fb[0]);
                                    }
                                    params.push(fb[0]);
                                };
                                match (final_byte, params.as_slice()) {
                                    // Left arrow (ESC [ D)
                                    (Some(b'D'), []) => {
                                        if cursor > 0 {
                                            cursor -= 1;
                                            redraw(
                                                output,
                                                prompt,
                                                &chars,
                                                &widths,
                                                cursor,
                                                term_cols,
                                                terminal_line,
                                                terminal_end_line,
                                            )?;
                                        }
                                    }
                                    // Right arrow (ESC [ C)
                                    (Some(b'C'), []) => {
                                        if cursor < chars.len() {
                                            cursor += 1;
                                            redraw(
                                                output,
                                                prompt,
                                                &chars,
                                                &widths,
                                                cursor,
                                                term_cols,
                                                terminal_line,
                                                terminal_end_line,
                                            )?;
                                        }
                                    }
                                    // Up arrow (ESC [ A) — navigate backwards in history
                                    (Some(b'A'), []) => {
                                        if hist_idx == history.len() {
                                            // Save current draft (chars and cursor) before
                                            // entering history so it can be restored intact.
                                            draft_chars = chars.clone();
                                            draft_cursor = cursor;
                                        }
                                        if hist_idx > 0 {
                                            hist_idx -= 1;
                                            chars = history[hist_idx].chars().collect();
                                            widths = chars
                                                .iter()
                                                .map(|c| c.width().unwrap_or(1))
                                                .collect();
                                            cursor = chars.len();
                                            redraw(
                                                output,
                                                prompt,
                                                &chars,
                                                &widths,
                                                cursor,
                                                term_cols,
                                                terminal_line,
                                                terminal_end_line,
                                            )?;
                                        }
                                    }
                                    // Down arrow (ESC [ B) — navigate forwards in history
                                    (Some(b'B'), []) => {
                                        if hist_idx < history.len() {
                                            hist_idx += 1;
                                            if hist_idx == history.len() {
                                                // Restore the saved draft, including the
                                                // cursor position the user had when they
                                                // pressed Up.
                                                chars = draft_chars.clone();
                                                cursor = draft_cursor;
                                            } else {
                                                chars = history[hist_idx].chars().collect();
                                                cursor = chars.len();
                                            }
                                            widths = chars
                                                .iter()
                                                .map(|c| c.width().unwrap_or(1))
                                                .collect();
                                            redraw(
                                                output,
                                                prompt,
                                                &chars,
                                                &widths,
                                                cursor,
                                                term_cols,
                                                terminal_line,
                                                terminal_end_line,
                                            )?;
                                        }
                                    }
                                    // Home: ESC [ H  (VT220) or ESC [ 1 ~ (xterm)
                                    (Some(b'H'), []) | (Some(b'~'), b"1") => {
                                        if cursor > 0 {
                                            cursor = 0;
                                            redraw(
                                                output,
                                                prompt,
                                                &chars,
                                                &widths,
                                                cursor,
                                                term_cols,
                                                terminal_line,
                                                terminal_end_line,
                                            )?;
                                        }
                                    }
                                    // End: ESC [ F  (VT220) or ESC [ 4 ~ (xterm)
                                    (Some(b'F'), []) | (Some(b'~'), b"4") => {
                                        if cursor < chars.len() {
                                            cursor = chars.len();
                                            redraw(
                                                output,
                                                prompt,
                                                &chars,
                                                &widths,
                                                cursor,
                                                term_cols,
                                                terminal_line,
                                                terminal_end_line,
                                            )?;
                                        }
                                    }
                                    // Delete (forward): ESC [ 3 ~
                                    (Some(b'~'), b"3") => {
                                        if cursor < chars.len() {
                                            let w = widths[cursor];
                                            chars.remove(cursor);
                                            widths.remove(cursor);
                                            // Use DCH (ESC[P) only for a simple 1-wide char at
                                            // end-of-line.  Zero-width combining marks (w == 0)
                                            // and wide CJK chars (w > 1) need a full redraw to
                                            // avoid deleting the wrong terminal cell or leaving
                                            // visual artifacts.  Mid-line deletions always redraw.
                                            if cursor == chars.len() && w == 1 {
                                                write!(output, "\x1b[P")?;
                                            } else {
                                                redraw(
                                                    output,
                                                    prompt,
                                                    &chars,
                                                    &widths,
                                                    cursor,
                                                    term_cols,
                                                    terminal_line,
                                                    terminal_end_line,
                                                )?;
                                            }
                                        }
                                    }
                                    // All other CSI sequences: discard.
                                    _ => {}
                                }
                            }
                            // SS3 sequence (ESC O): used by some terminals for
                            // function keys (F1–F4) and keypad.  Always exactly
                            // 3 bytes total (ESC O <final>), so consume one more.
                            b'O' => {
                                let mut fb = [0u8; 1];
                                let _ = input.read_exact(&mut fb);
                            }
                            // Other 2-byte ESC sequences (e.g. ESC M): the
                            // second byte was already consumed by read_exact
                            // above, so nothing more to discard.
                            _ => {}
                        }
                    }
                }
                // ASCII printable
                b @ 0x20..=0x7e => {
                    let ch = b as char;
                    chars.insert(cursor, ch);
                    widths.insert(cursor, 1);
                    cursor += 1;
                    // Cursor at end: echo directly, avoiding an unnecessary full redraw.
                    if cursor == chars.len() {
                        output.write_all(&[b])?;
                    } else {
                        redraw(
                            output,
                            prompt,
                            &chars,
                            &widths,
                            cursor,
                            term_cols,
                            terminal_line,
                            terminal_end_line,
                        )?;
                    }
                }
                // Valid UTF-8 multi-byte lead bytes only.
                //
                // 0xC0/0xC1 produce "overlong" encodings of ASCII and are
                // never valid in UTF-8; 0xF5..=0xFF exceed U+10FFFF and are
                // also invalid.  Those bytes fall through to `_` (silently
                // ignored) WITHOUT consuming any continuation bytes.
                b @ 0xC2..=0xF4 => {
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
                        match input.read_exact(&mut cb) {
                            // Valid continuation byte: collect it.
                            Ok(()) if (0x80..=0xBF).contains(&cb[0]) => {
                                utf8_buf.push(cb[0]);
                            }
                            // Invalid continuation or read error: stop.  The
                            // byte has been consumed from the stream; discard
                            // the whole sequence below.
                            _ => break,
                        }
                    }
                    if let Ok(s) = std::str::from_utf8(&utf8_buf) {
                        if let Some(ch) = s.chars().next() {
                            // width() returns Some(0) for zero-width combining
                            // marks; we preserve that so backspace knows not to
                            // move the cursor for them.  Control/format characters
                            // return None — default those to 1.
                            let w = ch.width().unwrap_or(1);
                            chars.insert(cursor, ch);
                            widths.insert(cursor, w);
                            cursor += 1;
                            // Cursor at end: echo directly, avoiding an unnecessary full redraw.
                            if cursor == chars.len() {
                                output.write_all(s.as_bytes())?;
                            } else {
                                redraw(
                                    output,
                                    prompt,
                                    &chars,
                                    &widths,
                                    cursor,
                                    term_cols,
                                    terminal_line,
                                    terminal_end_line,
                                )?;
                            }
                        }
                    }
                    // Silently discard invalid or truncated UTF-8 sequences.
                }
                // Other control bytes — ignore silently.
                _ => {}
            }

            // Flush once per iteration rather than after every individual
            // write.  This batches output during fast input (e.g. pastes)
            // while keeping the display responsive for single keystrokes.
            output.flush()?;
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{process_input, Arc};
        use std::io::Cursor;
        use unicode_width::UnicodeWidthChar as _;

        // ------------------------------------------------------------------ //
        // Helpers
        // ------------------------------------------------------------------ //

        /// Run `process_input` on a raw byte sequence with empty history.
        fn run(input: &[u8]) -> (std::io::Result<Option<String>>, Vec<u8>) {
            run_with_history(input, &[])
        }

        /// Run `process_input` with a given history slice.
        fn run_with_history(
            input: &[u8],
            history: &[&str],
        ) -> (std::io::Result<Option<String>>, Vec<u8>) {
            run_with_history_prompt(input, history, "", 80)
        }

        fn run_with_history_prompt(
            input: &[u8],
            history: &[&str],
            prompt: &str,
            term_cols: usize,
        ) -> (std::io::Result<Option<String>>, Vec<u8>) {
            let history_owned: Vec<Arc<str>> = history.iter().map(|s| Arc::from(*s)).collect();
            let mut reader = Cursor::new(input.to_vec());
            let mut output: Vec<u8> = Vec::new();
            let result = process_input(&mut reader, &mut output, &history_owned, prompt, term_cols);
            (result, output)
        }

        // ------------------------------------------------------------------ //
        // Enter / EOF / Ctrl-C
        // ------------------------------------------------------------------ //

        #[test]
        fn enter_cr_returns_accumulated_text() {
            let (res, echo) = run(b"hello\r");
            assert_eq!(res.unwrap(), Some("hello".to_string()));
            // Echo ends with CRLF
            assert!(echo.ends_with(b"\r\n"));
        }

        #[test]
        fn enter_lf_also_accepted() {
            let (res, _) = run(b"hi\n");
            assert_eq!(res.unwrap(), Some("hi".to_string()));
        }

        #[test]
        fn empty_enter_returns_empty_string() {
            let (res, _) = run(b"\r");
            assert_eq!(res.unwrap(), Some(String::new()));
        }

        #[test]
        fn ctrl_d_returns_none() {
            let (res, echo) = run(&[4]);
            assert_eq!(res.unwrap(), None);
            // Ctrl-D produces no echo output.
            assert!(echo.is_empty());
        }

        #[test]
        fn eof_on_empty_reader_returns_none() {
            let (res, _) = run(b"");
            assert_eq!(res.unwrap(), None);
        }

        #[test]
        fn ctrl_c_returns_interrupted_error() {
            let (res, echo) = run(&[3]);
            let err = res.unwrap_err();
            assert_eq!(err.kind(), std::io::ErrorKind::Interrupted);
            // Echo should contain "^C\r\n"
            assert_eq!(echo, b"^C\r\n");
        }

        // ------------------------------------------------------------------ //
        // ASCII backspace
        // ------------------------------------------------------------------ //

        #[test]
        fn backspace_del_removes_last_ascii_char() {
            // "ab" + DEL + Enter → "a"
            let (res, _) = run(b"ab\x7f\r");
            assert_eq!(res.unwrap(), Some("a".to_string()));
        }

        #[test]
        fn backspace_bs_also_removes_last_ascii_char() {
            let (res, _) = run(b"ab\x08\r");
            assert_eq!(res.unwrap(), Some("a".to_string()));
        }

        #[test]
        fn backspace_ascii_emits_one_column_erase() {
            // After "a", DEL should emit ESC[1D ESC[K (CSI cursor-back 1 + erase EOL).
            let (_, echo) = run(b"a\x7f\r");
            // "a" is echoed first (1 byte), then the CSI erase sequence, then \r\n.
            let after_a = &echo[1..];
            assert!(
                after_a.starts_with(b"\x1b[1D\x1b[K"),
                "expected ESC[1D ESC[K, got: {:?}",
                after_a
            );
        }

        #[test]
        fn backspace_on_empty_buffer_does_nothing() {
            // DEL on empty buffer — no erase, just Enter result
            let (res, echo) = run(b"\x7f\r");
            assert_eq!(res.unwrap(), Some(String::new()));
            // Only echo should be the CRLF from Enter; no erase sequences emitted.
            let before_crlf = &echo[..echo.len().saturating_sub(2)];
            assert!(
                !before_crlf.contains(&b'\x1b'),
                "no erase sequence expected on empty backspace, got: {:?}",
                before_crlf
            );
        }

        #[test]
        fn multiple_backspaces_empty_buffer() {
            // Three chars then three DELs → empty
            let (res, _) = run(b"abc\x7f\x7f\x7f\r");
            assert_eq!(res.unwrap(), Some(String::new()));
        }

        // ------------------------------------------------------------------ //
        // CJK / wide-character backspace — the bug this PR fixes
        // ------------------------------------------------------------------ //

        #[test]
        fn cjk_char_is_accepted() {
            let input = "中\r".as_bytes();
            let (res, _) = run(input);
            assert_eq!(res.unwrap(), Some("中".to_string()));
        }

        #[test]
        fn backspace_after_cjk_removes_char() {
            // "中" + DEL + Enter → empty
            let mut input = "中".as_bytes().to_vec();
            input.push(0x7f); // DEL
            input.push(b'\r');
            let (res, _) = run(&input);
            assert_eq!(res.unwrap(), Some(String::new()));
        }

        #[test]
        fn backspace_after_cjk_emits_csi_two_column_erase() {
            // "中" is 2 columns wide → erase sequence must be ESC[2D ESC[K
            // (CSI cursor-back 2 + erase-to-EOL), NOT raw \x08 pairs.
            let mut input = "中".as_bytes().to_vec();
            input.push(0x7f);
            input.push(b'\r');
            let (_, echo) = run(&input);

            let cjk_bytes = "中".len(); // 3 UTF-8 bytes
            let after_cjk = &echo[cjk_bytes..];
            assert!(
                after_cjk.starts_with(b"\x1b[2D\x1b[K"),
                "expected CSI 2-column erase (ESC[2D ESC[K), got: {:?}",
                after_cjk
            );
        }

        #[test]
        fn backspace_after_ascii_emits_csi_one_column_erase() {
            // "a" is 1 column wide → erase sequence must be ESC[1D ESC[K
            let (_, echo) = run(b"a\x7f\r");
            let after_a = &echo[1..]; // skip echoed 'a'
            assert!(
                after_a.starts_with(b"\x1b[1D\x1b[K"),
                "expected CSI 1-column erase (ESC[1D ESC[K), got: {:?}",
                after_a
            );
            // Must NOT use the 2-column variant
            assert!(
                !after_a.starts_with(b"\x1b[2D"),
                "ASCII backspace must not use 2-column erase"
            );
        }

        #[test]
        fn mixed_ascii_and_cjk_backspace_sequence() {
            // "a中b" + 2×DEL → "a"
            let mut input = "a中b".as_bytes().to_vec();
            input.extend_from_slice(b"\x7f\x7f\r");
            let (res, _) = run(&input);
            assert_eq!(res.unwrap(), Some("a".to_string()));
        }

        // ------------------------------------------------------------------ //
        // Regression: odd-ASCII-count + CJK backspace (the ghost-cell bug)
        //
        // When an odd number of ASCII chars precede a CJK character, the CJK
        // glyph starts at an odd terminal column.  Some terminals "snap" the
        // cursor to the left boundary of a wide char on the first raw \x08,
        // making two \x08s overshoot by one column and leaving the right-half
        // cell of the glyph visible.  Using ESC[wD instead avoids the snap.
        // ------------------------------------------------------------------ //

        /// Deleting CJK after 1 ASCII char must emit ESC[2D (not raw \x08\x08).
        #[test]
        fn odd_ascii_then_cjk_backspace_uses_csi_not_raw_bs() {
            // "a中" + DEL → "a"
            let mut input = "a中".as_bytes().to_vec();
            input.push(0x7f);
            input.push(b'\r');
            let (res, echo) = run(&input);
            assert_eq!(res.unwrap(), Some("a".to_string()));

            // Locate the erase sequence (after "a" echo + "中" echo bytes).
            let prefix_len = 1 + "中".len(); // 'a' + '中'
            let after_cjk = &echo[prefix_len..];

            // Must use CSI cursor-back 2, not two raw \x08 bytes.
            assert!(
                after_cjk.starts_with(b"\x1b[2D\x1b[K"),
                "expected ESC[2D ESC[K after odd-column CJK, got: {:?}",
                after_cjk
            );
            assert!(
                !after_cjk.starts_with(b"\x08\x08"),
                "must not use raw \\x08 pairs — they overshoot at odd column positions"
            );
        }

        /// Deleting CJK after 3 ASCII chars (another odd count) works correctly.
        #[test]
        fn three_ascii_then_cjk_backspace_uses_csi() {
            let mut input = "abc中".as_bytes().to_vec();
            input.push(0x7f);
            input.push(b'\r');
            let (res, echo) = run(&input);
            assert_eq!(res.unwrap(), Some("abc".to_string()));

            let prefix_len = 3 + "中".len();
            let after_cjk = &echo[prefix_len..];
            assert!(
                after_cjk.starts_with(b"\x1b[2D\x1b[K"),
                "expected ESC[2D ESC[K, got: {:?}",
                after_cjk
            );
        }

        /// Deleting CJK after 2 ASCII chars (even count) also uses CSI.
        #[test]
        fn even_ascii_then_cjk_backspace_uses_csi() {
            let mut input = "ab中".as_bytes().to_vec();
            input.push(0x7f);
            input.push(b'\r');
            let (res, echo) = run(&input);
            assert_eq!(res.unwrap(), Some("ab".to_string()));

            let prefix_len = 2 + "中".len();
            let after_cjk = &echo[prefix_len..];
            assert!(
                after_cjk.starts_with(b"\x1b[2D\x1b[K"),
                "expected ESC[2D ESC[K, got: {:?}",
                after_cjk
            );
        }

        /// Mixed delete: "a中b" + 2×DEL checks buffer is "a" AND second erase
        /// (for 'b', width=1) uses ESC[1D.
        #[test]
        fn odd_ascii_cjk_ascii_two_backspaces_erase_sequence() {
            let mut input = "a中b".as_bytes().to_vec();
            input.extend_from_slice(b"\x7f\x7f\r");
            let (res, echo) = run(&input);
            assert_eq!(res.unwrap(), Some("a".to_string()));

            // First erase: 'b' (width 1) → ESC[1D ESC[K
            let prefix = 1 + "中".len() + 1; // "a" + "中" + "b"
            let after_b = &echo[prefix..];
            assert!(
                after_b.starts_with(b"\x1b[1D\x1b[K"),
                "erase of 'b' should be ESC[1D ESC[K, got: {:?}",
                after_b
            );

            // Second erase: '中' (width 2) → ESC[2D ESC[K
            let after_b_erase = &after_b[b"\x1b[1D\x1b[K".len()..];
            assert!(
                after_b_erase.starts_with(b"\x1b[2D\x1b[K"),
                "erase of '中' should be ESC[2D ESC[K, got: {:?}",
                after_b_erase
            );
        }

        #[test]
        fn cjk_hiragana_has_width_2() {
            assert_eq!('あ'.width(), Some(2));
        }

        #[test]
        fn cjk_hangul_has_width_2() {
            assert_eq!('한'.width(), Some(2));
        }

        // ------------------------------------------------------------------ //
        // Zero-width combining marks — backspace must not over-erase
        // ------------------------------------------------------------------ //

        #[test]
        fn combining_mark_stored_with_zero_width() {
            // U+0301 COMBINING ACUTE ACCENT — zero-width
            // "e" + combining accent + Enter → "e\u{0301}"
            // Build the input manually for clarity (U+0301 = 0xCC 0x81).
            let input = {
                let mut v = Vec::new();
                v.push(b'e');
                // U+0301 is 0xCC 0x81 in UTF-8
                v.push(0xCC);
                v.push(0x81);
                v.push(b'\r');
                v
            };
            let (res, _) = run(&input);
            assert_eq!(res.unwrap(), Some("e\u{0301}".to_string()));
        }

        #[test]
        fn backspace_combining_mark_repaints_base() {
            // "e" + U+0301 + DEL + Enter → "e"
            // Backspacing the combining mark must repaint the base char so the
            // accent glyph is cleared from the terminal display.
            let input = {
                let mut v = Vec::new();
                v.push(b'e');
                v.push(0xCC); // U+0301 lead
                v.push(0x81); // U+0301 cont
                v.push(0x7f); // DEL
                v.push(b'\r');
                v
            };
            let (res, echo) = run(&input);
            assert_eq!(res.unwrap(), Some("e".to_string()));

            // Echo: 'e' (1) + U+0301 (2) = 3 bytes, then the repaint sequence.
            let after_char = &echo[3..]; // skip 'e' + 2-byte combining mark
                                         // Repaint via redraw(): \r ESC[K (go to col 0, clear EOL) …
            assert!(
                after_char.starts_with(b"\r\x1b[K"),
                "zero-width backspace must redraw with \\r ESC[K; got: {:?}",
                &after_char[..after_char.len().min(15)]
            );
            // … then re-echo 'e' to restore the base char without the accent.
            let after_erase = &after_char[b"\r\x1b[K".len()..];
            assert!(
                after_erase.starts_with(b"e"),
                "base char 'e' must be re-echoed after combining mark erase; got: {:?}",
                &after_erase[..after_erase.len().min(10)]
            );
        }

        #[test]
        fn backspace_second_combining_mark_repaints_base_with_first() {
            // "e" + U+0301 + U+0302 + DEL + Enter → "e\u{0301}"
            // Backspacing U+0302 repaints 'e' + U+0301 so only U+0302 is cleared.
            let input = {
                let mut v = Vec::new();
                v.push(b'e');
                v.extend_from_slice(&[0xCC, 0x81]); // U+0301
                v.extend_from_slice(&[0xCC, 0x82]); // U+0302 combining circumflex
                v.push(0x7f); // DEL
                v.push(b'\r');
                v
            };
            let (res, echo) = run(&input);
            assert_eq!(res.unwrap(), Some("e\u{0301}".to_string()));

            // Echo: 'e'(1) + U+0301(2) + U+0302(2) = 5 bytes, then repaint.
            let after_chars = &echo[5..];
            // Repaint via redraw(): \r ESC[K (go to col 0, clear EOL) …
            assert!(
                after_chars.starts_with(b"\r\x1b[K"),
                "backspace of second mark should emit \\r ESC[K; got: {:?}",
                &after_chars[..after_chars.len().min(15)]
            );
            // Re-echo must include 'e' + U+0301 but not U+0302.
            let after_erase = &after_chars[b"\r\x1b[K".len()..];
            let expected: &[u8] = "e\u{0301}".as_bytes(); // 'e' + 0xCC 0x81
            assert!(
                after_erase.starts_with(expected),
                "repaint must include 'e' + U+0301 only; got: {:?}",
                &after_erase[..after_erase.len().min(10)]
            );
        }

        #[test]
        fn latin_ascii_has_width_1() {
            assert_eq!('a'.width(), Some(1));
            assert_eq!('Z'.width(), Some(1));
        }

        // ------------------------------------------------------------------ //
        // Escape / CSI sequences
        // ------------------------------------------------------------------ //

        #[test]
        fn arrow_up_with_no_history_does_nothing() {
            // ESC [ A = cursor up — with empty history, leaves subsequent text unchanged
            let (res, _) = run(b"\x1b[Ahello\r");
            assert_eq!(res.unwrap(), Some("hello".to_string()));
        }

        #[test]
        fn arrow_right_at_end_does_nothing_to_content() {
            // Right arrow at end of "hi" should not change the text.
            let (res, _) = run(b"hi\x1b[Cthere\r");
            assert_eq!(res.unwrap(), Some("hithere".to_string()));
        }

        #[test]
        fn multi_param_csi_sequence_discarded() {
            // ESC [ 1 ; 2 H — cursor position with two params
            let (res, _) = run(b"\x1b[1;2Hok\r");
            assert_eq!(res.unwrap(), Some("ok".to_string()));
        }

        #[test]
        fn non_csi_esc_sequence_discarded() {
            // ESC M (reverse index) — 2-byte, not CSI
            let (res, _) = run(b"\x1bMok\r");
            assert_eq!(res.unwrap(), Some("ok".to_string()));
        }

        #[test]
        fn ss3_sequence_fully_consumed() {
            // ESC O P = F1 key (SS3 sequence) — all 3 bytes must be consumed
            // so the 'P' (0x50) does not leak into the input buffer.
            let (res, _) = run(b"\x1bOPok\r");
            assert_eq!(res.unwrap(), Some("ok".to_string()));
        }

        #[test]
        fn ss3_f4_sequence_fully_consumed() {
            // ESC O S = F4 key — another common SS3 sequence.
            let (res, _) = run(b"hi\x1bOSthere\r");
            assert_eq!(res.unwrap(), Some("hithere".to_string()));
        }

        // ------------------------------------------------------------------ //
        // Control characters (other than the named ones) — silently ignored
        // ------------------------------------------------------------------ //

        #[test]
        fn control_chars_other_than_special_are_ignored() {
            // Ctrl-A (0x01), Ctrl-E (0x05), Ctrl-K (0x0b) — not inserted into buffer
            let (res, _) = run(b"\x01hello\x05\x0bworld\r");
            assert_eq!(res.unwrap(), Some("helloworld".to_string()));
        }

        // ------------------------------------------------------------------ //
        // strip_trailing_newline helper
        // ------------------------------------------------------------------ //

        #[test]
        fn strip_newline_lf() {
            let mut s = "hello\n".to_string();
            super::strip_trailing_newline(&mut s);
            assert_eq!(s, "hello");
        }

        #[test]
        fn strip_newline_crlf() {
            let mut s = "hello\r\n".to_string();
            super::strip_trailing_newline(&mut s);
            assert_eq!(s, "hello");
        }

        #[test]
        fn strip_newline_no_newline_unchanged() {
            let mut s = "hello".to_string();
            super::strip_trailing_newline(&mut s);
            assert_eq!(s, "hello");
        }

        #[test]
        fn strip_newline_empty_unchanged() {
            let mut s = String::new();
            super::strip_trailing_newline(&mut s);
            assert_eq!(s, "");
        }

        // ------------------------------------------------------------------ //
        // Cursor movement — left/right arrows
        // ------------------------------------------------------------------ //

        #[test]
        fn left_arrow_moves_cursor_left_then_insert() {
            // "ab" + left + "X" + Enter → "aXb"
            let (res, _) = run(b"ab\x1b[DX\r");
            assert_eq!(res.unwrap(), Some("aXb".to_string()));
        }

        #[test]
        fn left_arrow_at_start_does_nothing() {
            // Three left arrows on empty buffer, then type "hi" → "hi"
            let (res, _) = run(b"\x1b[D\x1b[D\x1b[Dhi\r");
            assert_eq!(res.unwrap(), Some("hi".to_string()));
        }

        #[test]
        fn right_arrow_at_end_does_nothing() {
            // "hi" + right arrow (cursor already at end) + "!" → "hi!"
            let (res, _) = run(b"hi\x1b[C!\r");
            assert_eq!(res.unwrap(), Some("hi!".to_string()));
        }

        #[test]
        fn left_then_right_returns_to_end() {
            // "ab" + left + right + "c" → "abc"
            let (res, _) = run(b"ab\x1b[D\x1b[Cc\r");
            assert_eq!(res.unwrap(), Some("abc".to_string()));
        }

        #[test]
        fn backspace_at_cursor_position_deletes_char_before_cursor() {
            // "abc" + 2 lefts (cursor=1) + backspace (deletes chars[0]='a') → "bc"
            let (res, _) = run(b"abc\x1b[D\x1b[D\x7f\r");
            assert_eq!(res.unwrap(), Some("bc".to_string()));
        }

        #[test]
        fn backspace_mid_line_deletes_char_before_cursor() {
            // "abc" + 1 left (cursor=2) + backspace (deletes chars[1]='b') → "ac"
            let (res, _) = run(b"abc\x1b[D\x7f\r");
            assert_eq!(res.unwrap(), Some("ac".to_string()));
        }

        // ------------------------------------------------------------------ //
        // Home / End
        // ------------------------------------------------------------------ //

        #[test]
        fn home_vt_moves_cursor_to_start() {
            // "abc" + Home (ESC[H) + "X" → "Xabc"
            let (res, _) = run(b"abc\x1b[HX\r");
            assert_eq!(res.unwrap(), Some("Xabc".to_string()));
        }

        #[test]
        fn home_xterm_tilde_moves_cursor_to_start() {
            // "abc" + Home (ESC[1~) + "X" → "Xabc"
            let (res, _) = run(b"abc\x1b[1~X\r");
            assert_eq!(res.unwrap(), Some("Xabc".to_string()));
        }

        #[test]
        fn end_vt_moves_cursor_to_end() {
            // "abc" + 2 lefts + End (ESC[F) + "Z" → "abcZ"
            let (res, _) = run(b"abc\x1b[D\x1b[D\x1b[FZ\r");
            assert_eq!(res.unwrap(), Some("abcZ".to_string()));
        }

        #[test]
        fn end_xterm_tilde_moves_cursor_to_end() {
            // "abc" + 2 lefts + End (ESC[4~) + "Z" → "abcZ"
            let (res, _) = run(b"abc\x1b[D\x1b[D\x1b[4~Z\r");
            assert_eq!(res.unwrap(), Some("abcZ".to_string()));
        }

        #[test]
        fn home_at_start_does_nothing() {
            let (res, _) = run(b"\x1b[Habc\r");
            assert_eq!(res.unwrap(), Some("abc".to_string()));
        }

        #[test]
        fn end_at_end_does_nothing() {
            let (res, _) = run(b"abc\x1b[Fz\r");
            assert_eq!(res.unwrap(), Some("abcz".to_string()));
        }

        // ------------------------------------------------------------------ //
        // Multi-line redraw (content wider than term_cols)
        // ------------------------------------------------------------------ //

        #[test]
        fn multi_line_redraw_uses_erase_display() {
            // 81 'a' chars wraps past col 80 because the test helper passes
            // term_cols = 80 into process_input.  A left arrow triggers
            // redraw with end_visual_line = 1, so output must contain
            // ESC[J (erase to end of display).
            let mut input: Vec<u8> = vec![b'a'; 81];
            input.extend_from_slice(b"\x1b[D\r"); // left arrow + Enter
            let (res, output) = run(&input);
            assert_eq!(res.unwrap(), Some("a".repeat(81)));
            assert!(
                output.windows(3).any(|w| w == b"\x1b[J"),
                "expected ESC[J for multi-line erase, got: {:?}",
                output
            );
        }

        #[test]
        fn multi_line_redraw_cross_line_cursor_reposition() {
            // 82 'a' chars + Home moves cursor to col 0 (term_cols = 80 passed
            // by test helper).  end_visual_line = 1, cursor on line 0 → CUU needed.
            let mut input: Vec<u8> = vec![b'a'; 82];
            input.extend_from_slice(b"\x1b[H\r"); // Home + Enter
            let (res, output) = run(&input);
            assert_eq!(res.unwrap(), Some("a".repeat(82)));
            // Must contain CUU (ESC [ n A) to move cursor up across visual lines.
            assert!(
                output.windows(4).any(|w| w == b"\x1b[1A"),
                "expected ESC[1A (CUU) for cross-line cursor reposition, got: {:?}",
                output
            );
        }

        #[test]
        fn multi_line_redraw_cursor_at_exact_wrap_boundary() {
            // 81 'a' chars (term_cols = 80 passed by test helper), cursor
            // moves left once: lands at col 80 (pending-wrap boundary).
            // CHA should emit ESC[80G, not ESC[1G.
            let mut input: Vec<u8> = vec![b'a'; 81];
            input.extend_from_slice(b"\x1b[D\r"); // left arrow + Enter
            let (res, output) = run(&input);
            assert_eq!(res.unwrap(), Some("a".repeat(81)));
            // ESC[80G — cursor at last column of line 0, not column 1.
            assert!(
                output.windows(5).any(|w| w == b"\x1b[80G"),
                "expected ESC[80G for wrap-boundary CHA, got: {:?}",
                output
            );
        }

        #[test]
        fn history_multiline_to_short_clears_lower_lines() {
            // Reproduce: first history entry is short ("hello"), current buffer
            // is a long multi-line text (81 'a' chars, wraps to line 1 with
            // term_cols=80).  Pressing Up should replace the multi-line display
            // with "hello" and clear the wrapped lines below — verified by the
            // presence of ESC[J (erase to end of display) in the redraw output.
            let long_text = "a".repeat(81);
            // Start with multi-line text already typed; press Up to go to "hello".
            let mut input: Vec<u8> = long_text.bytes().collect();
            input.extend_from_slice(b"\x1b[A\r"); // Up arrow + Enter
            let (res, output) = run_with_history(&input, &["hello"]);
            assert_eq!(res.unwrap(), Some("hello".to_string()));
            // The redraw replacing the 81-char text with "hello" must use ED,
            // not EL, so that the second visual line is erased.
            assert!(
                output.windows(3).any(|w| w == b"\x1b[J"),
                "expected ESC[J when replacing multi-line buffer with short history entry, got: {:?}",
                output
            );
        }

        #[test]
        fn history_long_then_short_exact_pane_scenario() {
            // Exact reproduction of the reported bug:
            // term_cols=104, prompt="> "(2 cols), history[0]="hello",
            // history[1]="x"*110.  Up Up Enter must return "hello" and
            // must emit ESC[J twice (once per Up redraw).
            let long_text = "x".repeat(110); // 2+110=112 > 104, wraps to line 1
            let input = b"\x1b[A\x1b[A\r";
            let (res, output) = run_with_history_prompt(input, &["hello", &long_text], "> ", 104);
            assert_eq!(res.unwrap(), Some("hello".to_string()));
            let esc_j_count = output.windows(3).filter(|w| *w == b"\x1b[J").count();
            // The second Up (long→hello) must emit ESC[J to clear the wrapped line.
            // The first Up (empty→long) also emits ESC[J because end_visual_line>0.
            assert!(
                esc_j_count >= 2,
                "expected >=2 ESC[J, got {esc_j_count}; output: {:?}",
                output
            );
        }

        #[test]
        fn history_long_home_then_up_clears_lower_lines() {
            // Regression: type long text (wraps), press Home (cursor→0,
            // terminal_line=0), then Up (navigate to short history entry).
            // With cursor at position 0, old_end_line must still trigger ED
            // to erase the wrapped lines that remain on screen.
            let long_text = "x".repeat(110); // wraps to line 1 with term_cols=104
            let mut input: Vec<u8> = long_text.bytes().collect();
            input.extend_from_slice(b"\x1b[H\x1b[A\r"); // Home + Up + Enter
            let (res, output) = run_with_history_prompt(&input, &["hello"], "> ", 104);
            assert_eq!(res.unwrap(), Some("hello".to_string()));
            assert!(
                output.windows(3).any(|w| w == b"\x1b[J"),
                "expected ESC[J after Home+Up to clear wrapped lines, got: {:?}",
                output
            );
        }

        #[test]
        fn history_long_then_short_clears_lower_lines() {
            // Scenario: history[0]="hello", history[1]=81-char long text (already
            // submitted).  Current buffer is empty.  Press Up once (→ long text),
            // then Up again (→ "hello").  The second Up must erase the wrapped
            // lines left by the long text, i.e. ESC[J must appear after the
            // second Up's CUU.
            let long_text = "a".repeat(81);
            // Empty buffer: just two Up arrows then Enter.
            let input = b"\x1b[A\x1b[A\r";
            let (res, output) = run_with_history(input, &["hello", &long_text]);
            assert_eq!(res.unwrap(), Some("hello".to_string()));
            // Find the second ESC[J (first is from loading long_text, second
            // must appear when replacing long_text with "hello").
            let esc_j_count = output.windows(3).filter(|w| *w == b"\x1b[J").count();
            assert!(
                esc_j_count >= 2,
                "expected at least 2 ESC[J (one per Up), got {esc_j_count}; output: {:?}",
                output
            );
        }

        // ------------------------------------------------------------------ //
        // Delete (forward)
        // ------------------------------------------------------------------ //

        #[test]
        fn delete_at_end_does_nothing() {
            let (res, _) = run(b"abc\x1b[3~\r");
            assert_eq!(res.unwrap(), Some("abc".to_string()));
        }

        #[test]
        fn delete_removes_char_under_cursor() {
            // "abc" + 2 lefts (cursor=1) + Delete → "ac"
            let (res, _) = run(b"abc\x1b[D\x1b[D\x1b[3~\r");
            assert_eq!(res.unwrap(), Some("ac".to_string()));
        }

        #[test]
        fn delete_at_start_removes_first_char() {
            // Home + Delete on "abc" → "bc"
            let (res, _) = run(b"abc\x1b[H\x1b[3~\r");
            assert_eq!(res.unwrap(), Some("bc".to_string()));
        }

        #[test]
        fn insert_at_start_of_line() {
            // Three lefts to go to start, insert 'X' → "Xabc"
            let (res, _) = run(b"abc\x1b[D\x1b[D\x1b[DX\r");
            assert_eq!(res.unwrap(), Some("Xabc".to_string()));
        }

        // ------------------------------------------------------------------ //
        // History navigation — up/down arrows
        // ------------------------------------------------------------------ //

        #[test]
        fn up_arrow_with_empty_history_does_nothing() {
            // With no history, up arrow changes nothing; typed text stays.
            let (res, _) = run_with_history(b"\x1b[Aworld\r", &[]);
            assert_eq!(res.unwrap(), Some("world".to_string()));
        }

        #[test]
        fn up_arrow_loads_most_recent_history_entry() {
            // history = ["prev"], up → "prev", Enter
            let (res, _) = run_with_history(b"\x1b[A\r", &["prev"]);
            assert_eq!(res.unwrap(), Some("prev".to_string()));
        }

        #[test]
        fn up_arrow_multiple_times_navigates_history() {
            // history = ["first", "second"], two ups → "first"
            let (res, _) = run_with_history(b"\x1b[A\x1b[A\r", &["first", "second"]);
            assert_eq!(res.unwrap(), Some("first".to_string()));
        }

        #[test]
        fn up_arrow_at_oldest_entry_does_nothing_further() {
            // history = ["only"], two ups → still "only"
            let (res, _) = run_with_history(b"\x1b[A\x1b[A\r", &["only"]);
            assert_eq!(res.unwrap(), Some("only".to_string()));
        }

        #[test]
        fn down_arrow_with_no_history_navigation_does_nothing() {
            // Down arrow without prior up should leave current text unchanged.
            let (res, _) = run_with_history(b"hi\x1b[B!\r", &["prev"]);
            assert_eq!(res.unwrap(), Some("hi!".to_string()));
        }

        #[test]
        fn up_then_down_restores_draft() {
            // Type "draft", up (loads "prev"), down (restores "draft"), Enter
            let (res, _) = run_with_history(b"draft\x1b[A\x1b[B\r", &["prev"]);
            assert_eq!(res.unwrap(), Some("draft".to_string()));
        }

        #[test]
        fn up_twice_down_once_shows_second_history_entry() {
            // history = ["first", "second"]
            // up → "second", up → "first", down → "second", Enter
            let (res, _) = run_with_history(b"\x1b[A\x1b[A\x1b[B\r", &["first", "second"]);
            assert_eq!(res.unwrap(), Some("second".to_string()));
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

    // -----------------------------------------------------------------------
    // /claude command parsing
    // -----------------------------------------------------------------------

    #[test]
    fn parse_claude_two_quoted_tasks() {
        let result = parse_claude_command(r#"/claude "task one" "task two""#);
        let tasks = result.expect("should be Some").expect("should be Ok");
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].description, "task one");
        assert_eq!(tasks[1].description, "task two");
        assert!(tasks[0].auto_enter);
    }

    #[test]
    fn parse_claude_single_quoted_task() {
        let result = parse_claude_command(r#"/claude "implement something""#);
        let tasks = result.expect("should be Some").expect("should be Ok");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].description, "implement something");
    }

    #[test]
    fn parse_claude_unquoted_tokens_join_as_single_task() {
        let result = parse_claude_command("/claude implement something");
        let tasks = result.expect("should be Some").expect("should be Ok");
        // Unquoted words are joined into one task (avoids surprising N-task launch).
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].description, "implement something");
    }

    #[test]
    fn parse_claude_no_args_returns_none() {
        // No trailing space — not a /claude command at all.
        assert!(parse_claude_command("/claude").is_none());
    }

    #[test]
    fn parse_claude_only_space_returns_err() {
        // Has trailing spaces → recognized as /claude command but missing description.
        let result = parse_claude_command("/claude   ");
        assert!(result.expect("should be Some").is_err());
    }

    #[test]
    fn parse_not_claude_command_returns_none() {
        assert!(parse_claude_command("not a claude command").is_none());
    }

    #[test]
    fn parse_claude_no_enter_flag() {
        let result = parse_claude_command(r#"/claude --no-enter "do something""#);
        let tasks = result.expect("should be Some").expect("should be Ok");
        assert_eq!(tasks.len(), 1);
        assert!(!tasks[0].auto_enter);
        assert_eq!(tasks[0].description, "do something");
    }

    #[test]
    fn parse_claude_worktree_flag() {
        // Non-existent path: canonicalize falls back to the raw string.
        let result = parse_claude_command(r#"/claude --worktree /repo/wt/t1 "implement X""#);
        let tasks = result.expect("should be Some").expect("should be Ok");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].worktree.as_deref(), Some("/repo/wt/t1"));
        assert_eq!(tasks[0].description, "implement X");
    }

    #[test]
    fn parse_claude_escaped_quotes_in_description() {
        let result = parse_claude_command(r#"/claude "task with \"quotes\"""#);
        let tasks = result.expect("should be Some").expect("should be Ok");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].description, r#"task with "quotes""#);
    }

    #[test]
    fn parse_claude_task_id_derived_from_description() {
        let result = parse_claude_command(r#"/claude "Implement Cron Scheduling""#);
        let tasks = result.expect("should be Some").expect("should be Ok");
        assert_eq!(tasks[0].task_id, "implement-cron-scheduling");
    }

    #[test]
    fn parse_claude_task_id_truncated() {
        let long = format!("/claude \"{}\"", "a".repeat(100));
        let result = parse_claude_command(&long);
        let tasks = result.expect("should be Some").expect("should be Ok");
        assert!(tasks[0].task_id.len() <= 32);
    }

    #[test]
    fn make_task_id_simple() {
        assert_eq!(
            make_task_id("implement something", 0),
            "implement-something"
        );
    }

    #[test]
    fn make_task_id_collapses_dashes() {
        assert_eq!(make_task_id("hello   world", 0), "hello-world");
    }

    #[test]
    fn make_task_id_fallback_for_pure_non_ascii() {
        let id = make_task_id("こんにちは", 3);
        assert_eq!(id, "task-3");
    }

    // -----------------------------------------------------------------------
    // MarkdownBuffer state machine
    // -----------------------------------------------------------------------

    #[test]
    fn paragraph_flush_on_double_newline() {
        let mut buf = MarkdownBuffer::default();
        buf.push("Hello world");
        assert_eq!(buf.flush_if_ready(), None);
        buf.push("\n\nNext paragraph");
        let flushed = buf.flush_if_ready().expect("should flush paragraph");
        assert!(flushed.contains("Hello world"));
        assert_eq!(buf.buf, "Next paragraph");
    }

    #[test]
    fn code_block_not_flushed_mid_block() {
        let mut buf = MarkdownBuffer::default();
        buf.push("```rust\nfn main() {}\n");
        assert_eq!(buf.flush_if_ready(), None);
        buf.push("let x = 1;\n");
        assert_eq!(buf.flush_if_ready(), None);
        assert_eq!(buf.state, MdState::InCodeBlock);
    }

    #[test]
    fn table_buffered_until_non_pipe_line() {
        let mut buf = MarkdownBuffer::default();
        buf.push("| col1 | col2 |\n| --- | --- |\n| a | b |\n");
        assert_eq!(buf.flush_if_ready(), None);
        assert_eq!(buf.state, MdState::InTable);
        buf.push("after table\n");
        assert_eq!(buf.state, MdState::Normal);
        let flushed = buf
            .flush_if_ready()
            .expect("table should flush when first non-pipe line arrives");
        assert!(
            flushed.contains("| col1 | col2 |"),
            "table header missing; got: {flushed:?}"
        );
        assert!(
            flushed.contains("| a | b |"),
            "table row missing; got: {flushed:?}"
        );
        assert!(
            !flushed.contains("after table"),
            "non-table line must not be in flushed table; got: {flushed:?}"
        );
        assert_eq!(buf.buf, "after table\n");
    }

    #[test]
    fn flush_all_returns_remaining() {
        let mut buf = MarkdownBuffer::default();
        buf.push("Some text without double newline");
        let out = buf.flush_all().expect("flush_all should return content");
        assert!(out.contains("Some text without double newline"));
        assert_eq!(buf.flush_all(), None);
    }

    /// A code block that contains an internal blank line must not be flushed
    /// mid-block, even when the entire block arrives in a single chunk and the
    /// final state after `scan_new_lines` is `Normal`.
    #[test]
    fn code_block_with_internal_blank_line_not_flushed_mid_block() {
        let mut buf = MarkdownBuffer::default();
        // One chunk: complete code block with an internal blank line.
        buf.push("```rust\nfn a() {}\n\nfn b() {}\n```\n\n");
        let flushed = buf
            .flush_if_ready()
            .expect("should flush the complete code block");
        assert!(
            flushed.contains("fn a()"),
            "expected full code block; got: {flushed:?}"
        );
        assert!(
            flushed.contains("fn b()"),
            "expected full code block; got: {flushed:?}"
        );
        // The internal \n\n must NOT have split the block.
        assert!(
            flushed.contains("```rust"),
            "opening fence must be in the flushed unit"
        );
    }

    /// A fence line (`` ``` ``) that arrives without its terminating `\n` must
    /// not toggle state.  When the `\n` arrives in the next chunk the state
    /// machine should transition exactly once.
    #[test]
    fn fence_split_across_chunks_does_not_toggle_state() {
        let mut buf = MarkdownBuffer::default();

        // Opening fence arrives without its newline — state must stay Normal.
        buf.push("```rust");
        assert_eq!(
            buf.state,
            MdState::Normal,
            "incomplete opening fence must not change state"
        );

        // Newline arrives — now the fence is complete.
        buf.push("\n");
        assert_eq!(
            buf.state,
            MdState::InCodeBlock,
            "complete opening fence must enter InCodeBlock"
        );

        // Some code, then the closing fence without its newline.
        buf.push("let x = 1;\n```");
        assert_eq!(
            buf.state,
            MdState::InCodeBlock,
            "incomplete closing fence must not exit InCodeBlock"
        );

        // Newline completes the closing fence.
        buf.push("\n");
        assert_eq!(
            buf.state,
            MdState::Normal,
            "complete closing fence must exit InCodeBlock"
        );
    }
}
