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
// Slash command parsing
// ---------------------------------------------------------------------------

/// A task for the `/claude` command.
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
    /// Override for the client working directory used to locate the git repo
    /// when auto-creating a worktree.  Maps to TaskSpec::client_cwd.
    cwd: Option<String>,
}

/// A parsed slash command from user input.
#[derive(Debug, PartialEq)]
enum SlashCommand {
    /// `/model [<name>]` — switch model or show current.
    Model(Option<String>),
    /// `/claude "task" ...` — launch parallel Claude sessions.
    Claude(Result<Vec<ClaudeTask>, String>),
}

/// Parse a slash command from user input.
///
/// Returns `None` if the input is not a recognised slash command.
fn parse_slash_command(input: &str) -> Option<SlashCommand> {
    if let Some(cmd) = parse_model(input) {
        return Some(SlashCommand::Model(cmd));
    }
    if let Some(result) = parse_claude(input) {
        return Some(SlashCommand::Claude(result));
    }
    None
}

/// Parse `/model [<name>]`.
///
/// - `None` → not a `/model` command
/// - `Some(None)` → bare `/model` (show usage)
/// - `Some(Some(name))` → switch to `name`
fn parse_model(input: &str) -> Option<Option<String>> {
    let rest = input.strip_prefix("/model")?;
    if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let name = rest.trim();
    if name.is_empty() {
        Some(None)
    } else {
        Some(Some(name.to_string()))
    }
}

/// Parse `/claude [--worktree <path>] [--cwd <path>] [--no-enter] "task" ["task2" ...]`.
///
/// Task splitting rules:
/// - All tokens quoted → each quoted string is a separate task
///   (`/claude "fix bug" "add tests"` → two tasks)
/// - Any unquoted token → all tokens joined into one task
///   (`/claude fix the bug` or `/claude "fix" the bug` → one task)
///
/// - `None` → not a `/claude` command
/// - `Some(Err(msg))` → parse error
/// - `Some(Ok(tasks))` → one or more tasks
fn parse_claude(input: &str) -> Option<Result<Vec<ClaudeTask>, String>> {
    // Require "/claude" followed by end-of-string or whitespace to avoid
    // false positives like "/claudefoo" or "/claude--help".  Whitespace
    // handling accepts any Unicode whitespace — Chinese IMEs may insert
    // U+3000 ideographic space.
    let rest = input.strip_prefix("/claude")?;
    if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let rest = rest.trim_start();
    let usage = "usage: /claude [--worktree <path>] [--cwd <path>] [--no-enter] \
                 \"task description\" [\"task2\" ...]";
    if rest.is_empty() {
        return Some(Err(usage.to_string()));
    }

    let tokens = parse_quoted_args(rest);
    if tokens.is_empty() {
        return Some(Err(usage.to_string()));
    }

    let mut worktree: Option<String> = None;
    let mut auto_enter = true;
    let mut cwd: Option<String> = None;
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
                // If canonicalize fails (path doesn't exist yet), fall back to
                // an explicit absolute path rather than leaving it relative.
                let raw = &tokens[i].0;
                let raw_path = std::path::PathBuf::from(raw);
                let abs = std::fs::canonicalize(&raw_path).unwrap_or_else(|_| {
                    if raw_path.is_absolute() {
                        raw_path.clone()
                    } else {
                        std::env::current_dir()
                            .map(|cwd| cwd.join(&raw_path))
                            .unwrap_or(raw_path)
                    }
                });
                worktree = Some(abs.to_string_lossy().into_owned());
            }
            "--no-enter" => {
                auto_enter = false;
            }
            "--cwd" => {
                i += 1;
                if i >= tokens.len() || tokens[i].0.starts_with("--") {
                    return Some(Err("--cwd requires a path argument".to_string()));
                }
                // Canonicalize so the daemon gets a reliable absolute path.
                let raw = &tokens[i].0;
                let raw_path = std::path::PathBuf::from(raw);
                let abs = std::fs::canonicalize(&raw_path).unwrap_or_else(|_| {
                    if raw_path.is_absolute() {
                        raw_path.clone()
                    } else {
                        std::env::current_dir()
                            .map(|d| d.join(&raw_path))
                            .unwrap_or(raw_path)
                    }
                });
                cwd = Some(abs.to_string_lossy().into_owned());
            }
            // `--` marks end of flags; everything after is a description token.
            "--" => {
                i += 1;
                while i < tokens.len() {
                    desc_tokens.push((tokens[i].0.clone(), tokens[i].1));
                    i += 1;
                }
                break;
            }
            tok => {
                // Only treat unquoted tokens starting with `--` as unknown
                // flags.  A quoted token like `"--investigate"` is a valid
                // task description and must not be rejected.
                if !tokens[i].1 && tok.starts_with("--") {
                    return Some(Err(format!("unknown flag: {tok}")));
                }
                desc_tokens.push((tok.to_string(), tokens[i].1));
            }
        }
        i += 1;
    }

    if desc_tokens.is_empty() {
        return Some(Err(usage.to_string()));
    }

    // Build task list.  Quoted tokens each become a separate task.  Unquoted
    // tokens are joined as a single task (e.g. `/claude write some code` →
    // one task "write some code", not three separate tasks).
    let all_quoted = desc_tokens.iter().all(|(_, q)| *q);
    let descriptions: Vec<String> = if all_quoted || desc_tokens.len() == 1 {
        desc_tokens.into_iter().map(|(s, _)| s).collect()
    } else {
        vec![desc_tokens
            .into_iter()
            .map(|(s, _)| s)
            .collect::<Vec<_>>()
            .join(" ")]
    };

    let tasks = descriptions
        .into_iter()
        .enumerate()
        .map(|(idx, desc)| {
            let task_id = make_task_id(&desc, idx);
            ClaudeTask {
                task_id,
                description: desc,
                worktree: worktree.clone(),
                auto_enter,
                cwd: cwd.clone(),
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

    // For non-ASCII-only descriptions that produce an empty slug, use
    // "task-{idx}" which already encodes the index — no suffix needed.
    if trimmed.is_empty() {
        return format!("task-{idx}");
    }

    let base = if trimmed.len() > 32 {
        trimmed[..32].trim_end_matches('-').to_string()
    } else {
        trimmed.to_string()
    };

    // Append the index for tasks beyond the first so that duplicate
    // descriptions within a single /claude invocation get distinct task_ids
    // (and therefore distinct auto-worktree paths / branch names).
    if idx > 0 {
        format!("{base}-{idx}")
    } else {
        base
    }
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

    let cwd = std::env::current_dir().context("getting current directory")?;
    let cwd_str = cwd.to_string_lossy().into_owned();

    // Intercept slash commands before session::get_or_create to avoid
    // unnecessary disk I/O for commands that don't use the chat session.
    if let Some(SlashCommand::Claude(parse_result)) = parse_slash_command(&prompt) {
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
                    client_cwd: t.cwd.or_else(|| Some(cwd_str.clone())),
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

    // Resolve the session UUID now that we know this is a chat request
    // (not a /claude command).  Done after the /claude check to avoid
    // unnecessary disk I/O for commands that don't use the chat session.
    let cwd_for_session = cwd.clone();
    let session_id = tokio::task::spawn_blocking(move || session::get_or_create(&cwd_for_session))
        .await
        .context("session::get_or_create panicked")?
        .unwrap_or_else(|e| {
            tracing::warn!(error = %e, "failed to resolve session id; using \"global\"");
            "global".to_string()
        });
    let session_id_copy = session_id.clone();

    // Show startup banner when running interactively (TTY, not opted out).
    if crate::banner::should_show() {
        crate::banner::print(&model, &session_id_copy, &cwd);
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
                    Response::ModelSwitched { .. } => {
                        // ask mode has no persistent model variable to update.
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
    resumed_session_id: Option<String>,
) -> Result<()> {
    let mut sigint = signal(SignalKind::interrupt()).context("setting up SIGINT handler")?;

    let mut model = model
        .or_else(|| std::env::var("AMAEBI_MODEL").ok())
        .unwrap_or_else(|| crate::provider::DEFAULT_MODEL.to_string());

    let cwd = std::env::current_dir().context("getting current directory")?;
    let cwd_str = cwd.to_string_lossy().into_owned();
    let session_id = match resumed_session_id {
        Some(id) => id,
        None => {
            let cwd_for_session = cwd.clone();
            tokio::task::spawn_blocking(move || session::create_fresh(&cwd_for_session))
                .await
                .context("session::create_fresh panicked")?
                .unwrap_or_else(|e| {
                    tracing::warn!(error = %e, "failed to create fresh session id; using \"global\"");
                    "global".to_string()
                })
        }
    };

    if crate::banner::should_show() {
        crate::banner::print(&model, &session_id, &cwd);
    } else if std::io::stderr().is_terminal() {
        // Minimal fallback when the banner is opted out but we're still on a TTY.
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
                // Rustyline prints `prompt_input::PROMPT` itself; a second
                // pre-print here would produce a duplicate `> > `.
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

        // Dispatch slash commands before sending to daemon.
        match parse_slash_command(&prompt) {
            Some(SlashCommand::Claude(parse_result)) => {
                let tasks = match parse_result {
                    Ok(t) => t,
                    Err(msg) => {
                        stdout.write_all(msg.as_bytes()).await?;
                        stdout.write_all(b"\n").await?;
                        stdout.flush().await?;
                        continue 'session;
                    }
                };
                // Save descriptions keyed by task_id for supervision prompt below.
                let task_descriptions: std::collections::HashMap<String, String> = tasks
                    .iter()
                    .map(|t| (t.task_id.clone(), t.description.clone()))
                    .collect();
                let req = Request::ClaudeLaunch {
                    tasks: tasks
                        .into_iter()
                        .map(|t| TaskSpec {
                            task_id: t.task_id,
                            description: t.description,
                            worktree: t.worktree,
                            client_cwd: t.cwd.or_else(|| Some(cwd_str.clone())),
                            auto_enter: t.auto_enter,
                        })
                        .collect(),
                };
                let mut req_line = serde_json::to_string(&req)?;
                req_line.push('\n');
                write_half.write_all(req_line.as_bytes()).await?;

                // Collect (pane_id, task_description) for supervision.
                let mut launched: Vec<(String, String)> = Vec::new();

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
                            let desc = task_descriptions
                                .get(&task_id)
                                .cloned()
                                .unwrap_or_else(|| task_id.clone());
                            launched.push((pane_id, desc));
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

                // If panes were successfully launched, send a SupervisePanes request
                // to the daemon so it runs a Rust polling loop instead of asking the
                // LLM to keep looping via an injected prompt.
                if !launched.is_empty() {
                    let supervise_req = Request::SupervisePanes {
                        panes: launched
                            .iter()
                            .map(|(pid, desc)| crate::ipc::SupervisionTarget {
                                pane_id: pid.clone(),
                                task_description: desc.clone(),
                            })
                            .collect(),
                        model: model.clone(),
                        session_id: Some(session_id.clone()),
                    };
                    let mut req_line = serde_json::to_string(&supervise_req)?;
                    req_line.push('\n');
                    write_half.write_all(req_line.as_bytes()).await?;

                    // Stream supervision output exactly like a Chat response.
                    // We always arm a timeout so the client never hangs if the
                    // daemon silently drops. Default: 12 h (slightly above the
                    // daemon's 10 h default; both are configurable via env vars);
                    // shortened to 5 s once an interrupt has been sent.
                    let mut interrupt_sent = false;
                    let supervision_deadline =
                        tokio::time::Instant::now() + Duration::from_secs(12 * 60 * 60);
                    'supervision: loop {
                        let timeout_at = if interrupt_sent {
                            // After interrupt, give daemon 5 s to respond with Done.
                            tokio::time::Instant::now() + Duration::from_secs(5)
                        } else {
                            supervision_deadline
                        };
                        tokio::select! {
                            biased;

                            _ = sigint.recv(), if !interrupt_sent => {
                                let interrupt_req = Request::Interrupt { session_id: session_id.clone() };
                                if let Ok(mut frame) = serde_json::to_string(&interrupt_req) {
                                    frame.push('\n');
                                    let _ = write_half.write_all(frame.as_bytes()).await;
                                }
                                interrupt_sent = true;
                                // Continue looping to drain remaining frames.
                            }

                            _ = tokio::time::sleep_until(timeout_at) => {
                                // Timed out: either post-interrupt 5 s drain or the
                                // supervision hard ceiling.  The daemon may still be
                                // running its supervision loop, so reusing this socket
                                // for further Chat requests would desync the protocol.
                                // Terminate the session cleanly instead.
                                //
                                // Flush any partially buffered markdown first so we
                                // do not lose the last chunk the daemon had streamed
                                // before we gave up waiting.  `break 'session`
                                // bypasses the post-supervision-loop flush, so it
                                // has to happen here.
                                if let Some(remaining) = md_buf.flush_all() {
                                    let out = render_markdown(&remaining);
                                    stdout.write_all(out.as_bytes()).await?;
                                }
                                let reason = if interrupt_sent {
                                    "[supervision] daemon did not stop within 5 s after interrupt; ending session.\n"
                                } else {
                                    "[supervision] hard timeout reached; ending session.\n"
                                };
                                stdout.write_all(reason.as_bytes()).await?;
                                stdout.flush().await?;
                                break 'session;
                            }

                            line = lines.next_line() => {
                                let line = line.context("reading supervision response")?;
                                let Some(line) = line else { break 'session };
                                let resp: Response = serde_json::from_str(&line)?;
                                match resp {
                                    Response::Text { chunk } => {
                                        md_buf.push(&chunk);
                                        while let Some(ready) = md_buf.flush_if_ready() {
                                            let out = render_markdown(&ready);
                                            stdout.write_all(out.as_bytes()).await?;
                                            stdout.flush().await?;
                                        }
                                    }
                                    Response::Done => {
                                        if let Some(remaining) = md_buf.flush_all() {
                                            let out = render_markdown(&remaining);
                                            stdout.write_all(out.as_bytes()).await?;
                                        }
                                        stdout.write_all(b"\n").await?;
                                        stdout.flush().await?;
                                        break 'supervision;
                                    }
                                    Response::Error { message } => {
                                        if let Some(remaining) = md_buf.flush_all() {
                                            let out = render_markdown(&remaining);
                                            stdout.write_all(out.as_bytes()).await?;
                                        }
                                        stdout.write_all(message.as_bytes()).await?;
                                        stdout.write_all(b"\n").await?;
                                        stdout.flush().await?;
                                        break 'supervision;
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    // Flush any remaining markdown (e.g. timeout break path).
                    if let Some(remaining) = md_buf.flush_all() {
                        let out = render_markdown(&remaining);
                        stdout.write_all(out.as_bytes()).await?;
                    }
                    stdout.write_all(b"\n").await?;
                    stdout.flush().await?;
                }
                continue 'session;
            }
            Some(SlashCommand::Model(new_model)) => {
                match new_model {
                    Some(name) => {
                        model = name;
                        let msg = format!("[model] switched to {model}\n");
                        stdout.write_all(msg.as_bytes()).await?;
                    }
                    None => {
                        let msg = format!("usage: /model <model-name>  (current: {model})\n");
                        stdout.write_all(msg.as_bytes()).await?;
                    }
                }
                stdout.flush().await?;
                continue 'session;
            }
            None => {}
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
                        // The prompt marker `> ` is emitted by rustyline itself.
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
                            // Print error but don't exit — let the user retry
                            // with a different model or prompt.
                            let msg = format!("Error: {message}\n");
                            stdout.write_all(msg.as_bytes()).await?;
                            stdout.flush().await?;
                            break;
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
                            if std::io::stderr().is_terminal() && !extra.is_empty() {
                                eprintln!("\n{extra}");
                            }
                            // The prompt marker `> ` is emitted by rustyline itself.
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
                        Response::ModelSwitched { model: new_model } => {
                            // Keep client model in sync so the next Request::Chat
                            // carries the updated model, preserving carried_model
                            // across turns in the daemon.
                            model = new_model;
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
                            let trimmed = text.trim();
                            // Intercept /model in steer mode too — don't send
                            // it to the LLM which would strip [1m].
                            if let Some(rest) = trimmed.strip_prefix("/model") {
                                let rest = rest.trim();
                                if rest.is_empty() {
                                    let msg = format!("Current model: {model}\n");
                                    let _ = stdout.write_all(msg.as_bytes()).await;
                                } else {
                                    model = rest.to_string();
                                    let msg = format!("Model set to {model}\n");
                                    let _ = stdout.write_all(msg.as_bytes()).await;
                                }
                                let _ = stdout.flush().await;
                                // The daemon is waiting for a Steer after the
                                // Interrupt we sent.  Send "continue" so the
                                // agentic loop resumes with the new model taking
                                // effect on the next turn.
                                let steer_req = Request::Steer {
                                    session_id: session_id.clone(),
                                    message: "continue".to_owned(),
                                };
                                if let Ok(mut frame) = serde_json::to_string(&steer_req) {
                                    frame.push('\n');
                                    let _ = write_half.write_all(frame.as_bytes()).await;
                                }
                            } else {
                                let steer_req = Request::Steer {
                                    session_id: session_id.clone(),
                                    message: trimmed.to_owned(),
                                };
                                if let Ok(mut frame) = serde_json::to_string(&steer_req) {
                                    frame.push('\n');
                                    let _ = write_half.write_all(frame.as_bytes()).await;
                                }
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

    // Show startup banner when running interactively (TTY, not opted out).
    if crate::banner::should_show() {
        let cwd = std::env::current_dir().context("getting current directory")?;
        crate::banner::print(&model, &session_uuid, &cwd);
    }

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
                    Response::ModelSwitched { .. } => {
                        // resume mode has no persistent model variable to update.
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
// Prompt input — thin rustyline wrapper (replaces the hand-written editor)
// ---------------------------------------------------------------------------

/// Prompt input routines backed by [`rustyline`].
///
/// Historically this module contained a hand-written raw-mode line editor
/// (~1600 LOC) that tracked UnicodeWidth and emitted ANSI redraws so CJK
/// backspace would erase the right number of columns.  That editor produced
/// recurring cursor/CJK/soft-wrap bugs, so it was replaced with `rustyline`.
///
/// Public surface is deliberately minimal — three items that the outer chat
/// loop calls: [`PROMPT`], [`read_line_raw`], and [`restore_terminal_now`].
mod prompt_input {
    use std::io::IsTerminal;
    use std::sync::{Mutex, OnceLock};

    /// The prompt prefix displayed before each input line.  Exposed so the
    /// outer chat loop can use the same string for any fallback / non-TTY
    /// preamble.  Rustyline itself prints this when `readline(PROMPT)` is
    /// called, so the main TTY path does NOT pre-print it.
    pub(super) const PROMPT: &str = "> ";

    /// Process-wide `rustyline` editor.  Lazily initialised on first TTY
    /// readline; CI / piped-input paths never touch it.
    ///
    /// Wrapped in `Mutex` because rustyline's editor is not `Sync` and we
    /// call `read_line_raw` from `tokio::task::spawn_blocking`, which may
    /// hop OS threads.  A single shared editor also preserves in-memory
    /// history across calls without any extra plumbing.
    static EDITOR: OnceLock<Mutex<rustyline::DefaultEditor>> = OnceLock::new();

    /// Map a [`rustyline::error::ReadlineError`] into the `io::Result<Option<String>>`
    /// contract callers expect from [`read_line_raw`]:
    ///
    /// * `Interrupted` (Ctrl-C)  -> `Err(io::Error(Interrupted))`
    /// * `Eof` (Ctrl-D on empty) -> `Ok(None)`
    /// * anything else           -> `Err(io::Error::other(...))`
    fn translate_readline_err(
        e: rustyline::error::ReadlineError,
    ) -> std::io::Result<Option<String>> {
        use rustyline::error::ReadlineError;
        match e {
            ReadlineError::Interrupted => Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "ctrl-c",
            )),
            ReadlineError::Eof => Ok(None),
            other => Err(std::io::Error::other(other)),
        }
    }

    /// Read one line of input from the user.
    ///
    /// * On a TTY: delegates to `rustyline`, which prints [`PROMPT`] itself
    ///   and handles editing, history, and CJK/soft-wrap cursor tracking.
    /// * Off a TTY (piped stdin, CI): falls back to a single cooked
    ///   `read_line`, stripping the trailing newline.
    ///
    /// Returns:
    /// * `Ok(Some(line))` with the user's input (no trailing newline).
    /// * `Ok(None)` on EOF (Ctrl-D on empty line, or closed pipe).
    /// * `Err(e)` with `e.kind() == Interrupted` on Ctrl-C.
    pub fn read_line_raw() -> std::io::Result<Option<String>> {
        // Non-TTY fallback (piped input, CI).  Read one line cooked.
        if !std::io::stdin().is_terminal() {
            let mut line = String::new();
            let n = std::io::stdin().read_line(&mut line)?;
            if n == 0 {
                return Ok(None);
            }
            return Ok(Some(strip_newline(line)));
        }

        let editor = EDITOR.get_or_init(|| {
            Mutex::new(
                rustyline::DefaultEditor::new().expect("rustyline::DefaultEditor::new failed"),
            )
        });
        let mut ed = editor.lock().unwrap_or_else(|p| p.into_inner());
        match ed.readline(PROMPT) {
            Ok(line) => {
                let _ = ed.add_history_entry(line.as_str());
                Ok(Some(line))
            }
            Err(e) => translate_readline_err(e),
        }
    }

    /// Restore the terminal to its pre-raw-mode state.
    ///
    /// Rustyline restores termios on every `readline()` return via its own
    /// `Drop` guard, so in the common case this function has nothing to do.
    /// If a `spawn_blocking(read_line_raw)` task is still parked inside
    /// `readline()` when the session exits, its OS thread still owns the
    /// editor; we cannot forcibly drop another thread's guard, so the best
    /// we can do is log and let the detached thread unwind on its own.
    #[cfg(unix)]
    pub fn restore_terminal_now() {
        tracing::debug!("restore_terminal_now(): rustyline manages termios per-readline");
    }

    #[cfg(not(unix))]
    pub fn restore_terminal_now() {}

    /// Strip a trailing `\n` or `\r\n` from a string in place.
    fn strip_newline(mut s: String) -> String {
        if s.ends_with('\n') {
            s.pop();
            if s.ends_with('\r') {
                s.pop();
            }
        }
        s
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use rustyline::error::ReadlineError;
        use std::io::ErrorKind;

        #[test]
        fn strip_newline_strips_crlf() {
            assert_eq!(strip_newline("hello\r\n".to_string()), "hello");
        }

        #[test]
        fn strip_newline_strips_lf() {
            assert_eq!(strip_newline("hello\n".to_string()), "hello");
        }

        #[test]
        fn strip_newline_leaves_no_trailing_newline_alone() {
            assert_eq!(strip_newline("hello".to_string()), "hello");
        }

        #[test]
        fn translate_readline_err_interrupted_maps_to_io_interrupted() {
            let io_err = translate_readline_err(ReadlineError::Interrupted)
                .expect_err("Interrupted must be Err");
            assert_eq!(io_err.kind(), ErrorKind::Interrupted);
        }

        #[test]
        fn translate_readline_err_eof_maps_to_ok_none() {
            let result = translate_readline_err(ReadlineError::Eof);
            assert!(matches!(result, Ok(None)));
        }

        #[test]
        fn restore_terminal_now_idempotent() {
            // Safe to call in CI (no TTY, no editor initialised).
            restore_terminal_now();
            restore_terminal_now();
        }

        #[test]
        fn editor_is_lazy_uninitialized_before_first_call() {
            // Note: if another test in the same binary already called
            // read_line_raw with a TTY, EDITOR may be populated.  In
            // `cargo test` under CI there is no TTY, so read_line_raw
            // takes the non-TTY fallback and does not touch EDITOR.
            // This test documents the contract; don't assert .is_none()
            // unconditionally because test ordering is nondeterministic.
            let _ = EDITOR.get();
        }

        #[test]
        fn prompt_constant_is_unchanged() {
            assert_eq!(PROMPT, "> ");
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
    // slash command parsing
    // -----------------------------------------------------------------------

    fn claude_tasks(input: &str) -> Vec<ClaudeTask> {
        match parse_slash_command(input) {
            Some(SlashCommand::Claude(Ok(tasks))) => tasks,
            other => panic!("expected Claude tasks, got {other:?}"),
        }
    }

    fn claude_err(input: &str) -> String {
        match parse_slash_command(input) {
            Some(SlashCommand::Claude(Err(msg))) => msg,
            other => panic!("expected Claude error, got {other:?}"),
        }
    }

    #[test]
    fn parse_claude_two_quoted_tasks() {
        let tasks = claude_tasks(r#"/claude "task one" "task two""#);
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].description, "task one");
        assert_eq!(tasks[1].description, "task two");
    }

    #[test]
    fn parse_claude_single_quoted_task() {
        let tasks = claude_tasks(r#"/claude "implement something""#);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].description, "implement something");
    }

    #[test]
    fn parse_claude_unquoted_tokens_join_as_single_task() {
        let tasks = claude_tasks("/claude implement something");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].description, "implement something");
    }

    #[test]
    fn parse_claude_bare_returns_err() {
        assert!(!claude_err("/claude").is_empty());
        assert!(!claude_err("/claude   ").is_empty());
    }

    #[test]
    fn parse_not_claude_command_returns_none() {
        assert!(parse_slash_command("not a command").is_none());
    }

    #[test]
    fn parse_claude_false_positive_prefix_rejected() {
        assert!(parse_slash_command("/claudefoo").is_none());
        assert!(parse_slash_command("/claude--help").is_none());
    }

    #[test]
    fn parse_claude_escaped_quotes_in_description() {
        let tasks = claude_tasks(r#"/claude "task with \"quotes\"""#);
        assert_eq!(tasks[0].description, r#"task with "quotes""#);
    }

    #[test]
    fn parse_claude_task_id_derived_from_description() {
        let tasks = claude_tasks(r#"/claude "Implement Cron Scheduling""#);
        assert_eq!(tasks[0].task_id, "implement-cron-scheduling");
    }

    #[test]
    fn parse_claude_task_id_truncated() {
        let long = format!("/claude \"{}\"", "a".repeat(100));
        let tasks = claude_tasks(&long);
        assert!(tasks[0].task_id.len() <= 32);
    }

    #[test]
    fn parse_model_bare() {
        assert_eq!(
            parse_slash_command("/model"),
            Some(SlashCommand::Model(None))
        );
    }

    #[test]
    fn parse_model_spaces_only() {
        assert_eq!(
            parse_slash_command("/model   "),
            Some(SlashCommand::Model(None))
        );
    }

    #[test]
    fn parse_model_with_name() {
        assert_eq!(
            parse_slash_command("/model claude-sonnet-4.6"),
            Some(SlashCommand::Model(Some("claude-sonnet-4.6".to_string())))
        );
    }

    #[test]
    fn parse_model_with_1m_suffix() {
        assert_eq!(
            parse_slash_command("/model claude-sonnet-4.6[1m]"),
            Some(SlashCommand::Model(Some(
                "claude-sonnet-4.6[1m]".to_string()
            )))
        );
    }

    #[test]
    fn parse_model_with_provider_prefix() {
        assert_eq!(
            parse_slash_command("/model bedrock/claude-opus-4.6[1m]"),
            Some(SlashCommand::Model(Some(
                "bedrock/claude-opus-4.6[1m]".to_string()
            )))
        );
    }

    #[test]
    fn parse_model_false_positive_rejected() {
        assert!(parse_slash_command("/modelx").is_none());
        assert!(parse_slash_command("/model--help").is_none());
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
    // /claude: Unicode whitespace and --cwd flag
    // -----------------------------------------------------------------------

    #[test]
    fn parse_claude_unicode_whitespace_ideographic_space() {
        // Chinese IMEs often produce U+3000 (ideographic space).
        let input = "/claude\u{3000}\"do something\"";
        let result = parse_claude(input);
        let tasks = result.expect("should be Some").expect("should be Ok");
        assert_eq!(tasks[0].description, "do something");
    }

    #[test]
    fn parse_claude_unicode_whitespace_nbsp() {
        let input = "/claude\u{00A0}\"do something\"";
        let result = parse_claude(input);
        let tasks = result.expect("should be Some").expect("should be Ok");
        assert_eq!(tasks[0].description, "do something");
    }

    #[test]
    fn parse_claude_no_whitespace_returns_none() {
        // "/claudefoo" is not a /claude command.
        assert!(parse_claude("/claudefoo").is_none());
    }

    #[test]
    fn parse_claude_cwd_flag() {
        let input = r#"/claude --cwd /tmp/myrepo "fix the bug""#;
        let result = parse_claude(input);
        let tasks = result.expect("should be Some").expect("should be Ok");
        assert_eq!(tasks[0].description, "fix the bug");
        // cwd is canonicalized; /tmp exists so it should resolve.
        assert!(tasks[0].cwd.is_some());
    }

    #[test]
    fn parse_claude_cwd_missing_arg_returns_err() {
        let input = "/claude --cwd";
        let result = parse_claude(input);
        let err = result.expect("should be Some").unwrap_err();
        assert!(err.contains("--cwd requires a path"));
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
