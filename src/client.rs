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
    tag: String,
    /// Task description / opening prompt.
    description: String,
    /// Optional absolute worktree path.
    worktree: Option<String>,
    /// Whether to auto-send Enter after injecting the command into the pane.
    auto_enter: bool,
    /// Override for the client working directory used to locate the git repo
    /// when auto-creating a worktree.  Maps to TaskSpec::client_cwd.
    cwd: Option<String>,
    /// Optional tmux pane id (e.g. `"%41"`) to reuse via `--resume-pane`.
    /// Mutually exclusive with `worktree`; checked at parse time.
    resume_pane: Option<String>,
    /// One or more resource specs passed via `--resource`.  Each string is
    /// either a resource name (e.g. `sim-9900`) or `class:<name>` /
    /// `any:<name>` for any-idle-of-class selection.  Parsed by the daemon.
    resources: Vec<String>,
    /// Seconds to wait for busy resources.  `None` / `0` → fail fast.
    resource_timeout_secs: Option<u64>,
}

/// A parsed `/release` command.
#[derive(Debug, PartialEq, Clone)]
enum ReleaseCmd {
    Pane {
        pane_id: String,
        clean: bool,
        summary: Option<String>,
    },
    All {
        clean: bool,
    },
}

/// A parsed slash command from user input.
#[derive(Debug, PartialEq)]
enum SlashCommand {
    /// `/model [<name>]` — switch model or show current.
    Model(Option<String>),
    /// `/claude "task" ...` — launch parallel Claude sessions.
    Claude(Result<Vec<ClaudeTask>, String>),
    /// `/release %pane [--clean] [--summary "..."]` or `/release all [--clean]`.
    Release(Result<ReleaseCmd, String>),
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
    if let Some(result) = parse_release(input) {
        return Some(SlashCommand::Release(result));
    }
    None
}

/// Format a `Response::TaskReleased` frame for terminal output.  Shared by the
/// Chat, Resume, Ask, and `/release` stream handlers so rendering is
/// consistent no matter how the release was triggered (task_done, /release,
/// socket break in a different conn, etc.).
fn format_task_released(
    pane_id: &str,
    resources_freed: &[String],
    tag: Option<&str>,
    summary: Option<&str>,
    worktree_path: Option<&str>,
    worktree_dirty: bool,
    pane_tail: &str,
) -> String {
    let mut out = format!("[released {pane_id}]");
    if let Some(t) = tag {
        out.push_str(&format!(" tag={t}"));
    }
    if !resources_freed.is_empty() {
        out.push_str(&format!(" resources=[{}]", resources_freed.join(",")));
    }
    if let Some(wt) = worktree_path {
        out.push_str(&format!(
            " worktree={wt}{}",
            if worktree_dirty { " (dirty)" } else { "" }
        ));
    }
    out.push('\n');
    if let Some(s) = summary {
        out.push_str(&format!("  summary: {s}\n"));
    }
    if !pane_tail.trim().is_empty() {
        out.push_str("  --- pane tail ---\n");
        let tail_lines: Vec<&str> = pane_tail.lines().collect();
        for line in tail_lines.iter().rev().take(20).rev() {
            out.push_str(&format!("  | {line}\n"));
        }
    }
    out
}

/// Parse `/release %54 [--clean] [--summary "..."]` or `/release all [--clean]`.
///
/// - `None` → not a `/release` command
/// - `Some(Err(msg))` → parse error
/// - `Some(Ok(cmd))` → a valid release target
fn parse_release(input: &str) -> Option<Result<ReleaseCmd, String>> {
    let rest = input.strip_prefix("/release")?;
    if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let rest = rest.trim_start();
    let usage = "usage: /release %<pane_id> [--clean] [--summary \"…\"] \
                 or /release all [--clean]";
    if rest.is_empty() {
        return Some(Err(usage.into()));
    }

    let tokens: Vec<(String, bool)> = parse_quoted_args(rest);
    let mut iter = tokens.into_iter();
    let target_tok = match iter.next() {
        Some((t, _)) => t,
        None => return Some(Err(usage.into())),
    };

    let mut clean = false;
    let mut summary: Option<String> = None;
    while let Some((tok, _)) = iter.next() {
        match tok.as_str() {
            "--clean" => clean = true,
            "--summary" => match iter.next() {
                Some((s, _)) => summary = Some(s),
                None => return Some(Err("--summary requires a value".into())),
            },
            other => return Some(Err(format!("unknown flag: {other}"))),
        }
    }

    if target_tok == "all" {
        if summary.is_some() {
            return Some(Err(
                "--summary is only valid with /release %<pane_id>".into()
            ));
        }
        return Some(Ok(ReleaseCmd::All { clean }));
    }
    if let Some(stripped) = target_tok.strip_prefix('%') {
        if stripped.is_empty() {
            return Some(Err("pane id must follow % (e.g. %54)".into()));
        }
        return Some(Ok(ReleaseCmd::Pane {
            pane_id: target_tok.clone(),
            clean,
            summary,
        }));
    }
    Some(Err(format!(
        "expected 'all' or a pane id like %54, got {target_tok}"
    )))
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
    let usage = "usage: /claude [--worktree <path> | --resume-pane <pane_id>] \
                 [--cwd <path>] [--no-enter] \
                 [--resource <name|class:name>]... [--resource-timeout <secs>] \
                 [\"task description\" [\"task2\" ...]] \
                 (--resume-pane supports at most one optional task description; \
                 omitting the task description is only valid with --resume-pane)";
    if rest.is_empty() {
        return Some(Err(usage.to_string()));
    }

    let tokens = parse_quoted_args(rest);
    if tokens.is_empty() {
        return Some(Err(usage.to_string()));
    }

    let mut worktree: Option<String> = None;
    let mut resume_pane: Option<String> = None;
    let mut auto_enter = true;
    let mut cwd: Option<String> = None;
    let mut tag: Option<String> = None;
    let mut resources: Vec<String> = Vec::new();
    let mut resource_timeout_secs: Option<u64> = None;
    // (description, was_quoted) pairs for non-flag tokens.
    let mut desc_tokens: Vec<(String, bool)> = Vec::new();

    let mut i = 0;
    while i < tokens.len() {
        match tokens[i].0.as_str() {
            "--resume-pane" => {
                i += 1;
                if i >= tokens.len() || tokens[i].0.starts_with("--") {
                    return Some(Err(
                        "--resume-pane requires a pane id argument (e.g. `%41`)".to_string(),
                    ));
                }
                resume_pane = Some(tokens[i].0.clone());
            }
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
            "--tag" => {
                i += 1;
                if i >= tokens.len() || tokens[i].0.starts_with("--") {
                    return Some(Err(
                        "--tag requires a tag (short identifier for the notebook)".to_string(),
                    ));
                }
                tag = Some(tokens[i].0.clone());
            }
            "--resource" => {
                i += 1;
                if i >= tokens.len() || tokens[i].0.starts_with("--") {
                    return Some(Err(
                        "--resource requires a spec (resource name or `class:<name>`)".to_string(),
                    ));
                }
                resources.push(tokens[i].0.clone());
            }
            "--resource-timeout" => {
                i += 1;
                if i >= tokens.len() || tokens[i].0.starts_with("--") {
                    return Some(Err(
                        "--resource-timeout requires a number of seconds".to_string()
                    ));
                }
                match tokens[i].0.parse::<u64>() {
                    Ok(n) => resource_timeout_secs = Some(n),
                    Err(_) => {
                        return Some(Err(format!(
                            "--resource-timeout expects a non-negative integer, got {:?}",
                            tokens[i].0
                        )));
                    }
                }
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

    // --resume-pane and --worktree are mutually exclusive: resume-pane
    // inherits the target pane's existing worktree and would conflict with
    // an explicit --worktree.
    if resume_pane.is_some() && worktree.is_some() {
        return Some(Err("--resume-pane and --worktree are mutually exclusive; \
             --resume-pane inherits the target pane's worktree"
            .to_string()));
    }

    // `--resume-pane` + `--resource` IS allowed: on the reuse path we
    // still acquire the lock in `resource-state.json` (that constraint
    // is process-independent), we just skip env injection and rely on
    // the worktree's AGENTS.md (auto-generated on first launch) to keep
    // the LLM aware of its resource assignment.  See the daemon's
    // `handle_claude_launch` for the had_claude-branch that skips
    // env/prompt_hint rendering.

    // Description is required in the normal path, but optional with
    // --resume-pane: if omitted, the daemon reuses the description
    // previously persisted on the pane's lease.
    if desc_tokens.is_empty() && resume_pane.is_none() {
        return Some(Err(usage.to_string()));
    }

    // Build task list.  Quoted tokens each become a separate task.  Unquoted
    // tokens are joined as a single task (e.g. `/claude write some code` →
    // one task "write some code", not three separate tasks).
    let all_quoted = desc_tokens.iter().all(|(_, q)| *q);
    let descriptions: Vec<String> = if desc_tokens.is_empty() {
        // resume-pane with no description: use an empty-string sentinel so
        // the daemon will look up the description from the lease.
        vec![String::new()]
    } else if all_quoted || desc_tokens.len() == 1 {
        desc_tokens.into_iter().map(|(s, _)| s).collect()
    } else {
        vec![desc_tokens
            .into_iter()
            .map(|(s, _)| s)
            .collect::<Vec<_>>()
            .join(" ")]
    };

    // One pane can only run one task at a time.
    if resume_pane.is_some() && descriptions.len() > 1 {
        return Some(Err(
            "--resume-pane can only be used with a single task".to_string()
        ));
    }

    // Tag is filled later: the client does a GenerateTag round-trip
    // per task (or uses `--tag <override>` verbatim) before sending
    // ClaudeLaunch.  Parser emits ClaudeTask with a placeholder tag
    // the later code is responsible for replacing.
    let tasks = descriptions
        .into_iter()
        .map(|desc| ClaudeTask {
            tag: tag.clone().unwrap_or_default(),
            description: desc,
            worktree: worktree.clone(),
            auto_enter,
            cwd: cwd.clone(),
            resume_pane: resume_pane.clone(),
            resources: resources.clone(),
            resource_timeout_secs,
        })
        .collect();

    Some(Ok(tasks))
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
        let mut tasks = match parse_result {
            Ok(t) => t,
            Err(msg) => {
                let mut stdout = tokio::io::stdout();
                stdout.write_all(msg.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
                return Ok(());
            }
        };
        // For each task with no explicit `--tag`, ask the daemon for
        // one via GenerateTag (Haiku under the hood).  Must happen
        // before ClaudeLaunch so pane/worktree/notebook all use the
        // resolved tag from the start.
        resolve_missing_tags(&socket, &mut tasks, &cwd_str).await?;
        // One-shot `amaebi ask "/claude ..."` never sends SupervisePanes,
        // so there is no holder lifecycle to tie a notebook lease to.
        // Skip the lease by leaving session_id/repo_dir as None — matches
        // the pre-existing behaviour for this path.  The supervised
        // interactive chat loop (below, in run_chat_loop) supplies both
        // and is where the race-safe acquire actually runs.
        let req = Request::ClaudeLaunch {
            tasks: tasks
                .into_iter()
                .map(|t| TaskSpec {
                    tag: t.tag,
                    description: t.description,
                    worktree: t.worktree,
                    client_cwd: t.cwd.or_else(|| Some(cwd_str.clone())),
                    auto_enter: t.auto_enter,
                    resume_pane: t.resume_pane,
                    resources: t.resources,
                    resource_timeout_secs: t.resource_timeout_secs,
                })
                .collect(),
            session_id: None,
            repo_dir: None,
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
                    tag,
                    pane_id,
                    session_id: sid,
                    ..
                } => {
                    let msg = format!("[pane {pane_id}] tag={tag} → session {sid}\n");
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
                                "tmux_send_text" => eprintln!("⌨️  send-text: {detail}"),
                                "tmux_send_key" => eprintln!("⌨️  send-key: {detail}"),
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
                    Response::PaneAssigned { .. }
                    | Response::CapacityError { .. }
                    | Response::TagGenerated { .. } => {
                        // Not expected in a normal Chat response stream.
                        tracing::debug!("unexpected pane/tag scheduler response in chat loop");
                    }
                    Response::ModelSwitched { .. } => {
                        // ask mode has no persistent model variable to update.
                    }
                    Response::Heartbeat { .. } => {
                        // Supervision-only frame; never emitted on the Chat
                        // path but keep an explicit arm so future enum
                        // additions still force a compile error here.
                        tracing::debug!("ignoring Heartbeat outside supervision loop");
                    }
                    Response::TaskReleased {
                        pane_id,
                        resources_freed,
                        tag,
                        summary,
                        worktree_path,
                        worktree_dirty,
                        pane_tail,
                    } => {
                        let out = format_task_released(
                            &pane_id,
                            &resources_freed,
                            tag.as_deref(),
                            summary.as_deref(),
                            worktree_path.as_deref(),
                            worktree_dirty,
                            &pane_tail,
                        );
                        stdout.write_all(out.as_bytes()).await?;
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
/// Double Ctrl-C within 2 s → exit. Ctrl-D → exit. Empty / whitespace-only
/// Enter is a silent re-prompt (never forwarded to the daemon).
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
        eprintln!("Chat session started (ID: {session_id}). Ctrl-D or double Ctrl-C to exit.\n");
    }

    // Snapshot session id for the history writer so each persisted
    // prompt carries the chat's UUID in its `session_id` field, keeping
    // `history.jsonl` correlatable with `~/.amaebi/sessions.json` and
    // the daemon's per-session memory.
    prompt_input::set_session_id(&session_id);

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
                // Translate the rustyline outcome + current Ctrl-C state
                // into a `PromptInputDecision` (exit / send / continue /
                // first-tap).  The contract:
                //   * Ctrl-D (Ok(None))      → Exit (one-press)
                //   * Empty / whitespace-only → Continue (silent re-prompt,
                //                                never forwarded to the LLM)
                //   * Single Ctrl-C           → FirstCtrlC (warn + re-prompt)
                //   * Second Ctrl-C within
                //     DOUBLE_CTRLC_WINDOW     → Exit
                //   * Non-empty text          → Send
                let (line_opt, is_interrupt) = match raw_result {
                    Ok(line) => (line, false),
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => (None, true),
                    Err(e) => {
                        return Err(anyhow::Error::new(e).context("reading prompt from stdin"))
                    }
                };
                // Capture `now` once and reuse it for both the
                // double-tap-window check inside `classify_prompt_input`
                // and the `last_ctrl_c` write below.  Two separate
                // `Instant::now()` calls would let a scheduler delay
                // between them flip a just-inside-window press into a
                // just-outside one, and vice versa — the existing
                // streaming-path double-Ctrl-C handling already captures
                // `now` once for the same reason.
                let now = Instant::now();
                match classify_prompt_input(line_opt, is_interrupt, last_ctrl_c, now) {
                    PromptInputDecision::Exit => break 'session,
                    PromptInputDecision::Continue => {
                        // Any intervening input — including an empty /
                        // whitespace-only Enter — resets the double-tap
                        // window.  The contract is that **consecutive**
                        // Ctrl-C presses exit; a stray Enter between two
                        // Ctrl-Cs should count as "not consecutive" and
                        // make the second press behave like a fresh
                        // first tap.  Without this clear, Ctrl-C → Enter
                        // → Ctrl-C inside `DOUBLE_CTRLC_WINDOW` would
                        // still exit.
                        last_ctrl_c = None;
                        continue 'session;
                    }
                    PromptInputDecision::FirstCtrlC => {
                        if std::io::stderr().is_terminal() {
                            eprintln!(
                                "[Ctrl-C again within {}s to exit, Ctrl-D to exit immediately]",
                                DOUBLE_CTRLC_WINDOW.as_secs()
                            );
                        }
                        last_ctrl_c = Some(now);
                        continue 'session;
                    }
                    PromptInputDecision::Send(line) => {
                        // Non-empty input consumed; clear any pending
                        // single-Ctrl-C so the next press starts a fresh
                        // double-tap window.
                        last_ctrl_c = None;
                        line
                    }
                }
            }
        };

        // Dispatch slash commands before sending to daemon.
        match parse_slash_command(&prompt) {
            Some(SlashCommand::Claude(parse_result)) => {
                let mut tasks = match parse_result {
                    Ok(t) => t,
                    Err(msg) => {
                        stdout.write_all(msg.as_bytes()).await?;
                        stdout.write_all(b"\n").await?;
                        stdout.flush().await?;
                        continue 'session;
                    }
                };
                // Resolve any empty tags via the daemon's tagger before
                // building TaskSpec.  Skipped for tasks that arrived
                // with `--tag <override>` already set.
                if let Err(e) = resolve_missing_tags(&socket, &mut tasks, &cwd_str).await {
                    let msg = format!("[error] tag generation failed: {e:#}\n");
                    stdout.write_all(msg.as_bytes()).await?;
                    stdout.flush().await?;
                    continue 'session;
                }
                // Keyed by tag — used to look up the original description
                // when daemon replies with PaneAssigned.  Tag is resolved
                // (Haiku or `--tag` override) before tasks land here, so
                // it's a stable id for the supervision handoff.
                let task_descriptions: std::collections::HashMap<String, String> = tasks
                    .iter()
                    .map(|t| (t.tag.clone(), t.description.clone()))
                    .collect();
                // Canonicalise the effective client cwd — honour `--cwd` when
                // set so `repo_dir` matches the path sent as `client_cwd` on
                // each TaskSpec.  Without this, `/claude --cwd /other --tag foo`
                // would key the notebook against the chat process's cwd instead
                // of the requested directory, breaking resume and lease.
                let invocation_repo_dir: Option<String> = Some({
                    let effective_cwd = tasks
                        .iter()
                        .find_map(|t| t.cwd.clone())
                        .unwrap_or_else(|| cwd_str.clone());
                    crate::session::canonical_key(std::path::Path::new(&effective_cwd))
                });
                let req = Request::ClaudeLaunch {
                    tasks: tasks
                        .into_iter()
                        .map(|t| TaskSpec {
                            tag: t.tag,
                            description: t.description,
                            worktree: t.worktree,
                            client_cwd: t.cwd.or_else(|| Some(cwd_str.clone())),
                            auto_enter: t.auto_enter,
                            resume_pane: t.resume_pane,
                            resources: t.resources,
                            resource_timeout_secs: t.resource_timeout_secs,
                        })
                        .collect(),
                    session_id: Some(session_id.clone()),
                    repo_dir: invocation_repo_dir.clone(),
                };
                let mut req_line = serde_json::to_string(&req)?;
                req_line.push('\n');
                write_half.write_all(req_line.as_bytes()).await?;

                // Collect (pane_id, task_description) for supervision.
                // (pane_id, description, tag_tag) per launched pane.
                // (pane_id, description, tag) per launched pane.
                let mut launched: Vec<(String, String, String)> = Vec::new();

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
                            tag,
                            pane_id,
                            session_id: sid,
                            ..
                        } => {
                            let msg = format!("[pane {pane_id}] tag={tag} → session {sid}\n");
                            stdout.write_all(msg.as_bytes()).await?;
                            let desc = task_descriptions
                                .get(&tag)
                                .cloned()
                                .unwrap_or_else(|| tag.clone());
                            launched.push((pane_id, desc, tag));
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
                            .map(|(pid, desc, tag)| crate::ipc::SupervisionTarget {
                                pane_id: pid.clone(),
                                task_description: desc.clone(),
                                tag: Some(tag.clone()),
                                repo_dir: invocation_repo_dir.clone(),
                            })
                            .collect(),
                        model: model.clone(),
                        session_id: Some(session_id.clone()),
                    };
                    let mut req_line = serde_json::to_string(&supervise_req)?;
                    req_line.push('\n');
                    write_half.write_all(req_line.as_bytes()).await?;

                    // Stream supervision output exactly like a Chat response.
                    // There is no client-side business timeout — supervision
                    // length is a daemon concept (`AMAEBI_SUPERVISION_TIMEOUT_SECS`,
                    // default 24 h) and mirroring it here via an env var would
                    // silently desync whenever client + daemon run in different
                    // shells.  Instead the client runs a 30-min watchdog: if
                    // the daemon stops sending ANY frame (text, heartbeat,
                    // done), assume it's stuck and end the session so the TUI
                    // does not hang forever.  The daemon emits `Heartbeat`
                    // every 10 min (see `HEARTBEAT_INTERVAL_SECS` in daemon.rs)
                    // so a normal supervision — which sleeps ~5 min between
                    // LLM calls — always has at most a 10-min frame gap.
                    const WATCHDOG_INTERVAL_SECS: u64 = 30 * 60;
                    // Pinned (not per-iteration) so a heartbeat arriving
                    // during the 5 s drain does not push the deadline back —
                    // we promise the user a bounded wait after Ctrl-C.
                    let mut interrupt_drain_deadline: Option<tokio::time::Instant> = None;
                    'supervision: loop {
                        // Recomputed every iteration: any arriving frame (or
                        // sigint) re-enters the loop and resets the watchdog.
                        let watchdog_at = tokio::time::Instant::now()
                            + Duration::from_secs(WATCHDOG_INTERVAL_SECS);
                        let interrupt_sent = interrupt_drain_deadline.is_some();
                        // Fallback to `watchdog_at` when the arm is disabled so
                        // `sleep_until` still receives a valid Instant.
                        let interrupt_drain_at = interrupt_drain_deadline.unwrap_or(watchdog_at);
                        tokio::select! {
                            biased;

                            _ = sigint.recv(), if !interrupt_sent => {
                                let interrupt_req = Request::Interrupt { session_id: session_id.clone() };
                                if let Ok(mut frame) = serde_json::to_string(&interrupt_req) {
                                    frame.push('\n');
                                    let _ = write_half.write_all(frame.as_bytes()).await;
                                }
                                interrupt_drain_deadline = Some(
                                    tokio::time::Instant::now() + Duration::from_secs(5),
                                );
                                // Continue looping to drain remaining frames.
                            }

                            _ = tokio::time::sleep_until(interrupt_drain_at), if interrupt_sent => {
                                // Post-interrupt 5 s drain: daemon should have
                                // responded with Done by now; if not, give up
                                // cleanly so the socket does not stay half-
                                // alive for another Chat request (which would
                                // desync the frame protocol).
                                if let Some(remaining) = md_buf.flush_all() {
                                    let out = render_markdown(&remaining);
                                    stdout.write_all(out.as_bytes()).await?;
                                }
                                stdout
                                    .write_all(b"[supervision] daemon did not stop within 5 s after interrupt; ending session.\n")
                                    .await?;
                                stdout.flush().await?;
                                break 'session;
                            }

                            _ = tokio::time::sleep_until(watchdog_at), if !interrupt_sent => {
                                // 30 min without any frame — treat daemon as
                                // stuck (process alive but supervision task
                                // deadlocked, etc.) and end the session.
                                if let Some(remaining) = md_buf.flush_all() {
                                    let out = render_markdown(&remaining);
                                    stdout.write_all(out.as_bytes()).await?;
                                }
                                stdout
                                    .write_all(b"[supervision] no frames from daemon for 30 min; assuming stuck daemon and ending session.\n")
                                    .await?;
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
                                    Response::Heartbeat { elapsed_secs, turn } => {
                                        // Status line on stderr so it never
                                        // interleaves with streamed stdout
                                        // markdown.  Overwrites itself with `\r`
                                        // so scrollback stays clean — the
                                        // 5-min WAIT/STEER/DONE headers are
                                        // already there for history.
                                        let mins = elapsed_secs / 60;
                                        let hours = mins / 60;
                                        let rem_mins = mins % 60;
                                        let msg = if hours > 0 {
                                            format!(
                                                "\r[supervision alive — turn #{turn}, {hours}h{rem_mins}m elapsed]"
                                            )
                                        } else {
                                            format!(
                                                "\r[supervision alive — turn #{turn}, {mins}m elapsed]"
                                            )
                                        };
                                        let _ = tokio::io::stderr().write_all(msg.as_bytes()).await;
                                        let _ = tokio::io::stderr().flush().await;
                                        // Fall through — next select! iteration
                                        // recomputes `watchdog_at`, so receiving
                                        // this heartbeat resets the watchdog.
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
            Some(SlashCommand::Release(parse_result)) => {
                match parse_result {
                    Err(msg) => {
                        stdout.write_all(format!("{msg}\n").as_bytes()).await?;
                        stdout.flush().await?;
                    }
                    Ok(cmd) => {
                        let (target, clean, summary) = match cmd {
                            ReleaseCmd::Pane {
                                pane_id,
                                clean,
                                summary,
                            } => (
                                crate::ipc::ClaudeReleaseTarget::Pane { pane_id },
                                clean,
                                summary,
                            ),
                            ReleaseCmd::All { clean } => {
                                (crate::ipc::ClaudeReleaseTarget::All, clean, None)
                            }
                        };
                        let req = Request::ClaudeRelease {
                            target,
                            clean_worktree: clean,
                            summary,
                        };
                        let mut req_line = serde_json::to_string(&req)?;
                        req_line.push('\n');
                        write_half.write_all(req_line.as_bytes()).await?;
                        // Drain frames until Done — render each TaskReleased
                        // inline so the user sees summary + pane tail + dirty
                        // worktree flag for every pane they released.
                        loop {
                            let line = match lines.next_line().await {
                                Ok(Some(l)) => l,
                                _ => break 'session,
                            };
                            let resp: Response = match serde_json::from_str(&line) {
                                Ok(r) => r,
                                Err(_) => continue,
                            };
                            match resp {
                                Response::Done => break,
                                Response::Error { message } => {
                                    stdout.write_all(format!("{message}\n").as_bytes()).await?;
                                }
                                Response::TaskReleased {
                                    pane_id,
                                    resources_freed,
                                    tag,
                                    summary,
                                    worktree_path,
                                    worktree_dirty,
                                    pane_tail,
                                } => {
                                    let out = format_task_released(
                                        &pane_id,
                                        &resources_freed,
                                        tag.as_deref(),
                                        summary.as_deref(),
                                        worktree_path.as_deref(),
                                        worktree_dirty,
                                        &pane_tail,
                                    );
                                    stdout.write_all(out.as_bytes()).await?;
                                }
                                _ => {}
                            }
                        }
                        stdout.flush().await?;
                    }
                }
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
                                "tmux_send_text" => eprintln!("⌨️  send-text: {detail}"),
                                "tmux_send_key" => eprintln!("⌨️  send-key: {detail}"),
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
                    Response::PaneAssigned { .. }
                    | Response::CapacityError { .. }
                    | Response::TagGenerated { .. } => {
                        tracing::debug!("unexpected pane/tag scheduler response in resume loop");
                    }
                    Response::ModelSwitched { .. } => {
                        // resume mode has no persistent model variable to update.
                    }
                    Response::Heartbeat { .. } => {
                        // Supervision-only frame; never emitted on the
                        // Resume path.
                        tracing::debug!("ignoring Heartbeat outside supervision loop");
                    }
                    Response::TaskReleased {
                        pane_id,
                        resources_freed,
                        tag,
                        summary,
                        worktree_path,
                        worktree_dirty,
                        pane_tail,
                    } => {
                        let out = format_task_released(
                            &pane_id,
                            &resources_freed,
                            tag.as_deref(),
                            summary.as_deref(),
                            worktree_path.as_deref(),
                            worktree_dirty,
                            &pane_tail,
                        );
                        stdout.write_all(out.as_bytes()).await?;
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
                    "tmux_send_text" => eprintln!("⌨️  send-text: {detail}"),
                    "tmux_send_key" => eprintln!("⌨️  send-key: {detail}"),
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

/// Fill in `tasks[*].tag` when empty, asking the daemon to generate a
/// tag via its Haiku tagger.  One dedicated short-lived connection per
/// task — keeps the dispatcher on the main connection cleanly focused
/// on ClaudeLaunch / SupervisePanes frames.  Any IPC failure propagates
/// as `Err` to the caller; the caller is expected to surface the error
/// to the user and skip the launch rather than ship an empty tag.
async fn resolve_missing_tags(
    socket: &std::path::Path,
    tasks: &mut [ClaudeTask],
    client_cwd: &str,
) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    for t in tasks.iter_mut() {
        if !t.tag.is_empty() {
            continue;
        }
        let repo_dir = crate::session::canonical_key(std::path::Path::new(
            t.cwd.as_deref().unwrap_or(client_cwd),
        ));
        let stream = UnixStream::connect(socket)
            .await
            .context("connecting to daemon for GenerateTag")?;
        let (reader, mut writer) = tokio::io::split(stream);
        let req = Request::GenerateTag {
            description: t.description.clone(),
            repo_dir,
        };
        let mut line = serde_json::to_string(&req).context("serialising GenerateTag")?;
        line.push('\n');
        writer
            .write_all(line.as_bytes())
            .await
            .context("sending GenerateTag")?;
        let mut lines = BufReader::new(reader).lines();
        let reply = lines
            .next_line()
            .await
            .context("reading GenerateTag response")?
            .context("daemon closed before GenerateTag reply")?;
        let frame: Response = serde_json::from_str(&reply).context("parsing GenerateTag reply")?;
        match frame {
            Response::TagGenerated { tag } => {
                t.tag = tag;
            }
            Response::Error { message } => {
                anyhow::bail!("daemon rejected GenerateTag: {message}");
            }
            other => {
                anyhow::bail!("unexpected daemon frame for GenerateTag: {other:?}");
            }
        }
    }
    Ok(())
}

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
    use serde::{Deserialize, Serialize};
    use std::io::IsTerminal;
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, OnceLock};

    /// The prompt prefix displayed before each input line.  Exposed so the
    /// outer chat loop can use the same string for any fallback / non-TTY
    /// preamble.  Rustyline itself prints this when `readline(PROMPT)` is
    /// called, so the main TTY path does NOT pre-print it.
    pub(super) const PROMPT: &str = "> ";

    /// Max in-memory history entries.  Matches the old hand-written editor's
    /// 1000-entry VecDeque so Up-arrow depth is preserved across the swap.
    /// Rustyline's own default is 100, which would be a regression.
    const HISTORY_CAP: usize = 1000;

    /// Basename of the persistent history file under `~/.amaebi/`.
    const HISTORY_FILENAME: &str = "history.jsonl";

    /// Upper bound on bytes read from the tail of the history file when
    /// seeding rustyline's in-memory ring.  The file is append-only
    /// and grows without compaction; without this cap a multi-month
    /// heavy user's first `amaebi chat` readline would scan tens of
    /// MB every launch.  4 MiB comfortably holds `HISTORY_CAP = 1000`
    /// rows at the `MAX_LINE_BYTES` (~3.8 KiB) ceiling, so a full
    /// project's worth of matching history still loads — we only
    /// drop the tail of a globally oversized file.
    const LOAD_TAIL_BYTES: u64 = 4 * 1024 * 1024;

    /// Maximum byte length of a serialised history row *including* the
    /// trailing `\n`.  Kept comfortably below Linux's `PIPE_BUF` (4096) as
    /// a best-effort sizing budget for the "no explicit lock" design —
    /// note that POSIX's `PIPE_BUF` atomicity guarantee is defined for
    /// pipes/FIFOs, NOT for regular files.  In practice, Linux + common
    /// local filesystems (ext4, xfs, btrfs, tmpfs) deliver single
    /// `write(2)` of this size atomically between concurrent `O_APPEND`
    /// writers; we rely on that empirical behaviour rather than on any
    /// POSIX guarantee.  Strict inter-process record atomicity on every
    /// filesystem would require an explicit `flock`, which we have
    /// judged not worth the complexity for a user-convenience history.
    const MAX_LINE_BYTES: usize = 3800;

    /// Maximum byte length retained for `pasted_contents` before we
    /// truncate with a marker.  Not strictly necessary today — we do not
    /// currently populate `pasted_contents` — but the schema reserves the
    /// field and the truncation keeps future writers inside
    /// [`MAX_LINE_BYTES`].
    const MAX_PASTED_BYTES: usize = 200;

    /// One line of the on-disk prompt history.  Mirrors Claude Code's
    /// `~/.claude/history.jsonl` schema so the file is human-greppable
    /// and the semantics are familiar.  `cwd` stores a *canonicalized*
    /// path (via [`session::canonical_key`]) derived from the capturing
    /// process's current directory; the loader filters by that canonical
    /// key so ↑ inside a project only surfaces that project's prompts
    /// even across symlink / `..` / bind-mount path variants.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    struct HistoryRow {
        display: String,
        #[serde(default)]
        pasted_contents: String,
        timestamp_ms: u64,
        cwd: String,
        #[serde(default)]
        session_id: String,
    }

    /// Process-wide `rustyline` editor.  Lazily initialised on first TTY
    /// readline; CI / piped-input paths never touch it.
    ///
    /// Wrapped in `Mutex` because rustyline's editor is not `Sync` and we
    /// call `read_line_raw` from `tokio::task::spawn_blocking`, which may
    /// hop OS threads.  A single shared editor also preserves in-memory
    /// history across calls without any extra plumbing.
    static EDITOR: OnceLock<Mutex<rustyline::DefaultEditor>> = OnceLock::new();

    /// Current chat session UUID, set once by [`set_session_id`] at the
    /// start of `run_chat_loop` and read by [`append_history_line`] when
    /// persisting each prompt row.  Kept as module-level state rather
    /// than a `read_line_raw` parameter because the three call sites
    /// pass the bare function pointer to `spawn_blocking`, and adding
    /// an argument would require capturing via a closure at every site.
    ///
    /// `OnceLock` is intentional: a chat session only ever has one id,
    /// and later resumes start a new `run_chat_loop` in a fresh
    /// process, so a single set-per-process slot is sufficient.
    static SESSION_ID: OnceLock<String> = OnceLock::new();

    /// Record the current chat session id so subsequent history rows
    /// carry it in their `session_id` field.  Idempotent and safe to
    /// call before any readline; called once from `run_chat_loop`
    /// after the session id has been resolved.
    pub(super) fn set_session_id(id: &str) {
        let _ = SESSION_ID.set(id.to_owned());
    }

    /// Read the session id snapshot for history rows.  Empty string
    /// when `set_session_id` was not called (e.g. tests or non-chat
    /// paths — `amaebi ask` does not persist history today).
    fn current_session_id() -> String {
        SESSION_ID.get().cloned().unwrap_or_default()
    }

    /// Saved termios from the first TTY entry, used by [`restore_terminal_now`]
    /// as a belt-and-suspenders path when a detached readline task may have
    /// left the terminal in raw mode.  Snapshotted before rustyline ever
    /// touches termios, so restoring it always lands us back in cooked mode.
    #[cfg(unix)]
    static SAVED_TERMIOS: OnceLock<(std::os::unix::io::RawFd, libc::termios)> = OnceLock::new();

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

    /// Snapshot stdin's termios once so [`restore_terminal_now`] has something
    /// to restore to even if the rustyline editor is stuck on another thread.
    /// Only populates `SAVED_TERMIOS` on a successful `tcgetattr`; a failure
    /// leaves the `OnceLock` unset and future calls will retry.  No-op on
    /// non-Unix.
    #[cfg(unix)]
    fn snapshot_termios_once() {
        use std::os::unix::io::AsRawFd;
        if SAVED_TERMIOS.get().is_some() {
            return;
        }
        let fd = std::io::stdin().as_raw_fd();
        let mut t: libc::termios = unsafe { std::mem::zeroed() };
        // SAFETY: `t` is a valid termios out-parameter, `fd` is stdin.
        let rc = unsafe { libc::tcgetattr(fd, &mut t) };
        if rc == 0 {
            // `set` only fails if another thread won the race — harmless.
            let _ = SAVED_TERMIOS.set((fd, t));
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

        #[cfg(unix)]
        snapshot_termios_once();

        let editor = match EDITOR.get() {
            Some(ed) => ed,
            None => {
                // `Behavior::PreferTerm` makes rustyline prefer `/dev/tty`
                // for interactive input/output instead of stdin/stdout when
                // a controlling terminal is available.  This preserves the
                // old hand-written editor's usability when stdin or stdout
                // are piped/redirected: prompting and line editing still
                // work via the terminal.
                //
                // `max_history_size(HISTORY_CAP)` matches the old editor's
                // 1000-entry in-memory cap (rustyline defaults to 100).
                let config = rustyline::Config::builder()
                    .behavior(rustyline::config::Behavior::PreferTerm)
                    .max_history_size(HISTORY_CAP)
                    .expect("max_history_size rejects 0; HISTORY_CAP is > 0")
                    .build();
                // `set` fails only if another thread won the init race; in
                // that case we discard our fresh editor and use theirs.
                let mut fresh = rustyline::DefaultEditor::with_config(config).map_err(|e| {
                    std::io::Error::other(format!(
                        "rustyline::DefaultEditor::with_config(Behavior::PreferTerm): {e}"
                    ))
                })?;
                // Seed the in-memory ring from `~/.amaebi/history.jsonl`
                // filtered by cwd, so ↑ on this first readline already
                // surfaces this project's prior prompts.  Best-effort: a
                // missing or partially-written file is not worth
                // surfacing to the user.
                if let Err(e) = load_history_for_current_cwd(&mut fresh) {
                    tracing::debug!(error = %e, "failed to load history.jsonl; starting fresh");
                }
                let _ = EDITOR.set(Mutex::new(fresh));
                EDITOR.get().expect("EDITOR set above")
            }
        };
        let mut ed = editor.lock().unwrap_or_else(|p| p.into_inner());
        let read = ed.readline(PROMPT);
        match read {
            Ok(line) => {
                // Match the old editor: only non-empty lines join history,
                // so Up/Down navigation skips accidental bare-Enter presses.
                // Use `trim().is_empty()` so whitespace-only input — which
                // the chat REPL classifies as `Continue` (silent re-prompt,
                // never forwarded to the daemon) — also stays out of
                // history, keeping what ↑ surfaces aligned with what was
                // actually sent.
                let persist = !line.trim().is_empty();
                if persist {
                    let _ = ed.add_history_entry(line.as_str());
                }
                // Release the global editor mutex BEFORE the filesystem
                // append so a slow disk (NFS stall, fsync backlog on
                // another tenant) does not block a concurrent
                // `spawn_blocking(read_line_raw)` from picking up its
                // lock.  Today the outer chat REPL serialises readlines
                // so this is defensive rather than load-bearing, but
                // keeping the hot path lock-free matches the "EDITOR
                // guards in-memory state only" invariant.
                drop(ed);
                if persist {
                    // Best-effort: an I/O failure here is not worth
                    // surfacing to the user — log at debug and
                    // continue.  See `append_history_line` for
                    // atomicity notes.
                    if let Err(e) = append_history_line(&line) {
                        tracing::debug!(error = %e, "failed to append to history.jsonl");
                    }
                }
                Ok(Some(line))
            }
            Err(e) => {
                drop(ed);
                translate_readline_err(e)
            }
        }
    }

    /// Return the default path to the persistent history file.  Separated
    /// from the read/write helpers so tests can point them at a tempdir
    /// without needing to set `$HOME`.
    fn default_history_path() -> std::io::Result<PathBuf> {
        let home = crate::auth::amaebi_home()
            .map_err(|e| std::io::Error::other(format!("resolve ~/.amaebi: {e}")))?;
        Ok(home.join(HISTORY_FILENAME))
    }

    /// Milliseconds since the Unix epoch, saturating at 0 on clock error.
    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Byte-truncate `s` so the returned string is at most `max` bytes
    /// (preserving UTF-8 boundaries), appending a visible truncation
    /// marker when there is room.  Returns the original string when it
    /// already fits.  This function is **monotonic**: calling it with a
    /// smaller `max` can only produce a shorter-or-equal output, which
    /// is what makes the shrink loop in [`build_history_line`]
    /// converge.
    ///
    /// Degenerate case: when `max` is smaller than `MARKER.len()` (e.g.
    /// a caller requests an absurdly tight cap), the function returns
    /// the empty string.  The "output ≤ max" invariant and the
    /// monotonicity guarantee still hold; the caller is responsible
    /// for not passing a `max` below `MARKER.len()` if they need a
    /// non-empty result.  Real callers here (`MAX_PASTED_BYTES = 200`,
    /// `capped_display.len() / 2` with `.max(1)`) only hit this path
    /// when the input is already degenerate, in which case dropping
    /// the row entirely is acceptable.
    fn truncate_utf8_with_marker(s: &str, max: usize) -> String {
        const MARKER: &str = "…(truncated)";

        if s.len() <= max {
            return s.to_owned();
        }
        if max < MARKER.len() {
            return String::new();
        }
        let content_budget = max - MARKER.len();
        let mut cut = content_budget;
        while cut > 0 && !s.is_char_boundary(cut) {
            cut -= 1;
        }
        let mut out = String::with_capacity(cut + MARKER.len());
        out.push_str(&s[..cut]);
        out.push_str(MARKER);
        out
    }

    /// Build a [`HistoryRow`] and its trailing-`\n` JSON encoding,
    /// applying our size caps.  Pure — used directly by tests and by
    /// the writer helpers below.
    ///
    /// Truncation strategy: cap `pasted_contents` and `display` at the
    /// field level **before** serialization so the emitted JSON is
    /// always well-formed (no risk of producing invalid syntax by
    /// slicing bytes inside a string escape).  If the first shrink
    /// isn't enough — rare but possible when both fields are huge and
    /// the cwd / session_id strings are also long — iteratively shrink
    /// `display` until the serialised line fits under [`MAX_LINE_BYTES`]
    /// (including the trailing `\n`).
    fn build_history_line(
        display: &str,
        pasted_contents: &str,
        cwd: &str,
        session_id: &str,
        timestamp_ms: u64,
    ) -> String {
        let capped_pasted = truncate_utf8_with_marker(pasted_contents, MAX_PASTED_BYTES);
        let mut capped_display = display.to_owned();

        // `serde_json::to_string` on a struct of `String`s / `u64` cannot
        // fail; the `unwrap_or_else` is defensive so a bug in the writer
        // path never kills the whole history append.
        let serialize = |d: &str, p: &str| -> String {
            let row = HistoryRow {
                display: d.to_owned(),
                pasted_contents: p.to_owned(),
                timestamp_ms,
                cwd: cwd.to_owned(),
                session_id: session_id.to_owned(),
            };
            serde_json::to_string(&row).unwrap_or_else(|_| "{}".to_string())
        };

        let mut json = serialize(&capped_display, &capped_pasted);

        // Budget includes the trailing '\n' we will append at the end.
        // Shrink `display` by halves until the serialized row fits —
        // cheap in practice (at most ~log2(MAX_LINE_BYTES) iterations)
        // and guaranteed to terminate because `truncate_utf8_with_marker`
        // is monotonic (the returned length is ≤ the requested max), so
        // halving the budget strictly reduces output length.
        let mut last_display_len = capped_display.len() + 1; // force first iter
        while json.len() + 1 > MAX_LINE_BYTES
            && !capped_display.is_empty()
            && capped_display.len() < last_display_len
        {
            last_display_len = capped_display.len();
            let target = capped_display.len() / 2;
            capped_display = truncate_utf8_with_marker(&capped_display, target.max(1));
            json = serialize(&capped_display, &capped_pasted);
        }

        // Final size check: if non-shrinkable fields (cwd, session_id,
        // pasted_contents) alone exceed the budget, we cannot honour
        // the single-write size contract on which append atomicity
        // relies.  Drop the row entirely by returning an empty string;
        // the caller writes nothing when it receives "", and the load
        // path tolerates a missing row.  This is preferable to emitting
        // an oversize line that could split under concurrent appends.
        if json.len() + 1 > MAX_LINE_BYTES {
            return String::new();
        }

        // Drop rows whose `display` was non-empty originally but whose
        // truncated form is empty — a blank history entry is useless to
        // the user (nothing recognisable to navigate to via ↑) and
        // misleading when inspecting the file.  Non-empty inputs can
        // only reach this state when the shrink loop drove the display
        // budget below `MARKER.len()`, which means cwd / session_id
        // already dominate the budget on their own.
        if capped_display.is_empty() && !display.is_empty() {
            return String::new();
        }

        json.push('\n');
        json
    }

    /// Append one line of prompt history to `~/.amaebi/history.jsonl`,
    /// creating the file on first use.  Relies on typical Linux +
    /// local-filesystem behaviour: a single `write(2)` below
    /// [`MAX_LINE_BYTES`] to an `O_APPEND` file is delivered as one
    /// unit relative to other `O_APPEND` writers.  This is NOT a POSIX
    /// guarantee for regular files (POSIX `PIPE_BUF` atomicity is
    /// defined for pipes/FIFOs) — just the common, observed behaviour
    /// we accept in lieu of an explicit lock.  [`build_history_line`]
    /// enforces the line-size ceiling and
    /// [`append_history_line_to_path`] issues exactly one `write(2)`
    /// syscall (no retry loop) so a short write surfaces as an error
    /// rather than silently producing a split/interleaved record.
    fn append_history_line(display: &str) -> std::io::Result<()> {
        let path = default_history_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let cwd = cwd_for_history();
        let session_id = current_session_id();
        let json = build_history_line(display, "", &cwd, &session_id, now_ms());
        append_history_line_to_path(&path, &json)
    }

    /// Canonicalised path identity used for the `cwd` field in the
    /// history file — keeps the writer and reader consistent even when
    /// the user enters the same project through a symlink, `..`, or a
    /// bind-mounted path.  Falls back to the raw cwd string only when
    /// `current_dir()` itself fails.
    fn cwd_for_history() -> String {
        match std::env::current_dir() {
            Ok(p) => crate::session::canonical_key(&p),
            Err(_) => String::new(),
        }
    }

    /// Raw append to `path`.  Split out so tests can target a tempfile.
    ///
    /// Two guarantees we rely on here:
    ///
    /// 1. **0600 permissions.** History rows contain user prompts,
    ///    which can be sensitive.  Create with mode 0o600 on Unix
    ///    (mirror `~/.amaebi/sessions.json` and similar state files)
    ///    and defensively re-apply 0o600 on an existing file whose
    ///    permissions are looser — covers the case where umask on a
    ///    prior run was wrong or someone changed the mode manually.
    /// 2. **Single `write(2)` per record.** On typical Linux + local
    ///    filesystems, `O_APPEND` plus one `write(2)` ≤ `MAX_LINE_BYTES`
    ///    is delivered to the file as a single, non-interleaved unit
    ///    relative to other `O_APPEND` writers.  This is NOT a POSIX
    ///    guarantee for regular files — POSIX defines `PIPE_BUF`
    ///    atomicity only for pipes/FIFOs — so treat it as best-effort.
    ///    `std::io::Write::write_all` will loop on short writes which
    ///    can cause concurrent appends to interleave; we call the
    ///    low-level `write` once and treat a short write as a hard
    ///    error so the caller doesn't retry and split the record.
    fn append_history_line_to_path(path: &Path, json_with_newline: &str) -> std::io::Result<()> {
        // Empty input means `build_history_line` could not fit the row
        // under MAX_LINE_BYTES (e.g. pathologically long cwd /
        // session_id).  Drop silently rather than create/touch the
        // file; the history is best-effort and a dropped row is
        // strictly better than emitting something that could corrupt
        // concurrent writers.
        if json_with_newline.is_empty() {
            return Ok(());
        }

        #[cfg(unix)]
        use std::os::unix::fs::{OpenOptionsExt as _, PermissionsExt as _};

        let mut opts = std::fs::OpenOptions::new();
        opts.create(true).append(true);
        #[cfg(unix)]
        opts.mode(0o600);
        let mut f = opts.open(path)?;

        #[cfg(unix)]
        {
            // Single `metadata()` syscall per append — reuse for both
            // the 0o600 check and the `Permissions` handle if we need
            // to call `set_permissions`.
            let mut perms = f.metadata()?.permissions();
            if perms.mode() & 0o777 != 0o600 {
                perms.set_mode(0o600);
                f.set_permissions(perms)?;
            }
        }

        use std::io::Write as _;
        let buf = json_with_newline.as_bytes();
        let n = f.write(buf)?;
        if n != buf.len() {
            // A short write on a local append-only file basically never
            // happens outside disk-full / EINTR scenarios.  Refuse to
            // retry: another concurrent writer might have appended a
            // record between our halves, interleaving the two lines.
            return Err(std::io::Error::other(format!(
                "short write to history.jsonl: wrote {n}/{} bytes",
                buf.len()
            )));
        }
        // No fsync: the cost outweighs the value for a "user convenience"
        // history; the kernel flushes normally within seconds.
        Ok(())
    }

    /// Load `~/.amaebi/history.jsonl`, filter rows by the current process
    /// cwd, and feed the matches into `editor`'s in-memory history in
    /// file order (oldest first).  Malformed / mid-write lines are
    /// skipped rather than aborting the load.
    fn load_history_for_current_cwd(editor: &mut rustyline::DefaultEditor) -> std::io::Result<()> {
        let path = default_history_path()?;
        let cwd = cwd_for_history();
        load_history_from_path(&path, &cwd, editor)
    }

    /// Testable core of [`load_history_for_current_cwd`].  Empty `cwd` is
    /// treated as "no filter possible" and returns without loading
    /// anything — matching the behaviour when `std::env::current_dir()`
    /// fails on a real run.
    ///
    /// Bounded start-up cost: for files larger than [`LOAD_TAIL_BYTES`]
    /// we seek to `file_len - LOAD_TAIL_BYTES` and skip the partial
    /// first line, so the startup scan is capped regardless of how
    /// much history has accumulated across months of use.  Oldest-
    /// first ordering within the loaded window is preserved.
    ///
    /// Truly-bounded line read: we scan `BufRead::fill_buf()` for `\n`
    /// and consume in chunks, so a pathologically huge line without a
    /// newline can never force the in-memory line buffer above
    /// [`READ_CAP`].  Lines whose length hits [`READ_CAP`] are skipped
    /// entirely — we keep consuming bytes (discarded) until we find the
    /// next `\n` and then resume parsing, so one bad row doesn't lose
    /// the rest of the file.
    fn load_history_from_path(
        path: &Path,
        cwd: &str,
        editor: &mut rustyline::DefaultEditor,
    ) -> std::io::Result<()> {
        if cwd.is_empty() {
            return Ok(());
        }
        let mut file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e),
        };
        use std::io::{BufRead as _, Seek as _};
        // For files larger than the tail budget, seek near the end and
        // discard the (probably-partial) first line we land in.  This
        // bounds startup I/O to O(LOAD_TAIL_BYTES) regardless of how
        // much history has accumulated, while still producing an
        // oldest-first sequence within the returned window.
        let len = file.metadata()?.len();
        // Seeking mid-file almost always lands inside a record, so
        // enter the loop in `discarding` mode: the existing skip-
        // until-newline logic will drop the partial head of the first
        // surviving line and resume parsing at the next line boundary.
        let discard_head = if len > LOAD_TAIL_BYTES {
            let offset = len - LOAD_TAIL_BYTES;
            file.seek(std::io::SeekFrom::Start(offset))?;
            true
        } else {
            false
        };
        let mut reader = std::io::BufReader::new(file);
        // Per-line memory cap — slightly above MAX_LINE_BYTES so a
        // legitimate written-at-the-cap line still parses while anything
        // bigger is treated as garbage and discarded without OOMing.
        const READ_CAP: usize = MAX_LINE_BYTES + 64;
        let mut buf: Vec<u8> = Vec::with_capacity(512);
        let mut discarding = discard_head;
        loop {
            let chunk = reader.fill_buf()?;
            if chunk.is_empty() {
                // EOF.  Flush whatever's in `buf` as a final line iff
                // we weren't discarding and it's non-empty.
                if !discarding && !buf.is_empty() {
                    try_load_row(&buf, cwd, editor);
                }
                break;
            }
            let consumed = match memchr_newline(chunk) {
                Some(idx) => {
                    let take = idx + 1; // include the '\n'
                    if !discarding {
                        // Copy up to the newline; stop early if it
                        // would overflow the cap.
                        let remaining = READ_CAP.saturating_sub(buf.len());
                        let copy_bytes = remaining.min(idx);
                        buf.extend_from_slice(&chunk[..copy_bytes]);
                        if copy_bytes == idx {
                            try_load_row(&buf, cwd, editor);
                        }
                        // Else: this line exceeded READ_CAP; we filled
                        // `buf` up to the cap and now discard the rest
                        // of the oversize line up to the newline we
                        // just found, then fall through to reset and
                        // resume parsing at the next line boundary.
                    }
                    // Newline found: line boundary.  Reset and keep
                    // parsing the next line even if we were discarding.
                    buf.clear();
                    discarding = false;
                    take
                }
                None => {
                    // No newline in this chunk.  Either grow buf up to
                    // the cap or flip into discard mode.
                    let remaining = READ_CAP.saturating_sub(buf.len());
                    if !discarding && chunk.len() <= remaining {
                        buf.extend_from_slice(chunk);
                    } else if !discarding {
                        // Hitting the cap — fill what we can then flip
                        // to discard for the rest of this line.
                        buf.extend_from_slice(&chunk[..remaining]);
                        discarding = true;
                    }
                    // Consume whatever we looked at; we'll re-enter
                    // fill_buf on the next iteration.
                    chunk.len()
                }
            };
            reader.consume(consumed);
        }
        Ok(())
    }

    /// Byte scan for `b'\n'`.  Inlined so the loader has no `memchr`
    /// crate dependency — the file is small enough that the overhead
    /// is negligible.
    fn memchr_newline(haystack: &[u8]) -> Option<usize> {
        haystack.iter().position(|b| *b == b'\n')
    }

    /// Parse one line (already trimmed of its trailing `\n`) and feed
    /// the resulting row to `editor` if its `cwd` matches.  All parse
    /// errors are swallowed — malformed lines are tolerated rather
    /// than fatal.
    fn try_load_row(line_bytes: &[u8], cwd: &str, editor: &mut rustyline::DefaultEditor) {
        // Drop a trailing '\r' if the file has CRLF endings.
        let end = if line_bytes.last() == Some(&b'\r') {
            line_bytes.len() - 1
        } else {
            line_bytes.len()
        };
        let Ok(line) = std::str::from_utf8(&line_bytes[..end]) else {
            return;
        };
        let Ok(row) = serde_json::from_str::<HistoryRow>(line) else {
            return;
        };
        if row.cwd != cwd {
            return;
        }
        // `add_history_entry` dedups the most-recent entry for us.
        let _ = editor.add_history_entry(row.display.as_str());
    }

    /// Best-effort restore of the terminal to its pre-raw-mode state.
    ///
    /// Rustyline restores termios on every `readline()` return via its own
    /// `Drop` guard, so in the common case this function has nothing to do.
    /// If a `spawn_blocking(read_line_raw)` task is still parked inside
    /// `readline()` when the session exits, its OS thread still owns the
    /// rustyline editor and we cannot forcibly drop its guard.  But we CAN
    /// call `tcsetattr` with the termios we snapshotted before rustyline
    /// first touched the terminal — that's enough to unstick the user's
    /// shell from raw mode.  The still-parked readline thread will end up
    /// reading garbage on its next byte, but the process is exiting anyway.
    #[cfg(unix)]
    pub fn restore_terminal_now() {
        if let Some((fd, termios)) = SAVED_TERMIOS.get() {
            // SAFETY: `fd` is stdin and `termios` is the snapshot we took.
            // Best-effort cleanup during shutdown: do not propagate errors,
            // but log the outcome accurately so a broken terminal after
            // crash can be traced to tcsetattr rather than a no-op path.
            let rc = unsafe { libc::tcsetattr(*fd, libc::TCSANOW, termios) };
            if rc == 0 {
                tracing::debug!("restore_terminal_now(): applied snapshot termios");
            } else {
                tracing::debug!(
                    "restore_terminal_now(): tcsetattr failed: {}",
                    std::io::Error::last_os_error()
                );
            }
        }
    }

    #[cfg(not(unix))]
    pub fn restore_terminal_now() {}

    /// Return `s` with any trailing `\n` or `\r\n` removed.
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
            // Safe to call in CI (no TTY, no snapshot taken).  Call twice
            // to verify idempotence; with no SAVED_TERMIOS populated this
            // becomes a pure no-op.
            restore_terminal_now();
            restore_terminal_now();
        }

        #[test]
        fn prompt_constant_is_unchanged() {
            assert_eq!(PROMPT, "> ");
        }

        // ----- persistent history -----

        /// Build a fresh editor with the same config as `read_line_raw`
        /// uses on first-init, but without `Behavior::PreferTerm` (CI has
        /// no TTY and the config field is irrelevant for history-only
        /// tests anyway).
        fn test_editor() -> rustyline::DefaultEditor {
            let config = rustyline::Config::builder()
                .max_history_size(HISTORY_CAP)
                .expect("max_history_size > 0")
                .build();
            rustyline::DefaultEditor::with_config(config).expect("editor")
        }

        /// Collect the editor's history entries oldest-first.  Uses the
        /// `History` trait's `get(index, Forward)` since rustyline v14
        /// does not expose an iterator on its public history trait.
        fn collect_history(editor: &rustyline::DefaultEditor) -> Vec<String> {
            use rustyline::history::{History as _, SearchDirection};
            let h = editor.history();
            let len = h.len();
            let mut out = Vec::with_capacity(len);
            for i in 0..len {
                if let Ok(Some(res)) = h.get(i, SearchDirection::Forward) {
                    out.push(res.entry.into_owned());
                }
            }
            out
        }

        #[test]
        fn history_row_json_round_trip() {
            let row = HistoryRow {
                display: "/claude --resource sim-9902 \"do X\"".into(),
                pasted_contents: String::new(),
                timestamp_ms: 1_775_151_914_679,
                cwd: "/home/yuankuns/libraries.ai.cutlass.internal".into(),
                session_id: "381c777e-67d8-4add-b282-26df8d903a7a".into(),
            };
            let json = serde_json::to_string(&row).expect("serialize");
            let parsed: HistoryRow = serde_json::from_str(&json).expect("parse");
            assert_eq!(row, parsed);
        }

        #[test]
        fn append_and_filter_by_cwd() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("history.jsonl");

            // Two rows matching the "project" cwd and one foreign.
            let rows = [
                build_history_line("hello", "", "/project", "", 1),
                build_history_line("/model", "", "/project", "", 2),
                build_history_line("elsewhere", "", "/other", "", 3),
            ];
            for row in &rows {
                append_history_line_to_path(&path, row).expect("append");
            }

            let mut editor = test_editor();
            load_history_from_path(&path, "/project", &mut editor).expect("load");
            let history = collect_history(&editor);
            assert_eq!(history, vec!["hello".to_string(), "/model".to_string()]);
        }

        #[test]
        fn load_history_tolerates_corrupted_line() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("history.jsonl");
            // One valid row, one garbled mid-write line, then another
            // valid row — loader should keep the two valid ones.
            let valid_a = build_history_line("good-1", "", "/p", "", 10);
            let valid_b = build_history_line("good-2", "", "/p", "", 20);
            append_history_line_to_path(&path, &valid_a).unwrap();
            append_history_line_to_path(&path, "{not valid json\n").unwrap();
            append_history_line_to_path(&path, &valid_b).unwrap();

            let mut editor = test_editor();
            load_history_from_path(&path, "/p", &mut editor).expect("load");
            assert_eq!(
                collect_history(&editor),
                vec!["good-1".to_string(), "good-2".to_string()]
            );
        }

        #[test]
        fn append_line_truncates_oversize() {
            // A 10 KiB display deliberately exceeds PIPE_BUF once
            // serialised.  `build_history_line` must cap it.
            let big = "x".repeat(10 * 1024);
            let line = build_history_line(&big, "", "/cwd", "", 42);
            assert!(
                line.len() <= MAX_LINE_BYTES,
                "line was {} bytes, MAX_LINE_BYTES={}",
                line.len(),
                MAX_LINE_BYTES
            );
            assert!(line.ends_with('\n'), "line must end with newline");
        }

        #[test]
        fn load_missing_file_is_ok() {
            // First run of `amaebi chat` on a new machine: no
            // history.jsonl yet.  Loader must succeed silently.
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("history.jsonl");
            let mut editor = test_editor();
            load_history_from_path(&path, "/cwd", &mut editor).expect("missing file is ok");
            assert!(collect_history(&editor).is_empty());
        }

        #[test]
        fn load_history_bounds_tail_for_oversized_file() {
            // Write far more than LOAD_TAIL_BYTES of valid rows.  The
            // loader must seek to the tail, drop the partial first
            // line, and preserve oldest-first ordering of what survives.
            // Using "/p" as cwd keeps fixed fields cheap so we can pad
            // `display` to ~4 KiB per row and exceed 4 MiB quickly.
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("history.jsonl");
            let row_ts = |i: u64| i; // stable ordering keys
            let filler = "x".repeat(3000);
            let total = (LOAD_TAIL_BYTES as usize) / 3000 + 256;
            for i in 0..total {
                let display = format!("{filler}-{i}");
                let line = build_history_line(&display, "", "/p", "", row_ts(i as u64));
                append_history_line_to_path(&path, &line).unwrap();
            }
            // Final marker row we definitely want to see — shorter and
            // unique so we can assert it lands in the editor.
            let tail = build_history_line("tail-marker", "", "/p", "", u64::MAX);
            append_history_line_to_path(&path, &tail).unwrap();

            let mut editor = test_editor();
            load_history_from_path(&path, "/p", &mut editor).expect("load");
            let loaded = collect_history(&editor);
            assert!(
                loaded.last().map(|s| s.as_str()) == Some("tail-marker"),
                "tail must be the last entry, got {:?}",
                loaded.last()
            );
            assert!(
                loaded.len() < total,
                "tail window must drop older rows, loaded {} of {}",
                loaded.len(),
                total
            );
        }

        #[test]
        fn build_history_line_drops_row_when_display_truncates_to_empty() {
            // Pick `cwd` large enough that the initial serialisation
            // overflows `MAX_LINE_BYTES`, forcing the shrink loop to
            // halve the 17-byte display to 8 bytes.  `MARKER.len()` is
            // 14, so `truncate_utf8_with_marker(_, 8)` returns "" and
            // `capped_display` becomes empty.  Re-serialised length is
            // now small enough to fit, so the "total oversize" bail-out
            // does NOT fire — only the explicit empty-display drop
            // guards against writing a blank-display row.
            //
            // Target math for the chosen cwd_size = 3715:
            //   fixed JSON overhead = 77 bytes (empty fields + ts=1)
            //   initial: 77 + 17 + 3715 = 3809  (> MAX_LINE_BYTES)
            //   shrunk: 77 +  0 + 3715 = 3792  (<= MAX_LINE_BYTES)
            let cwd_size = 3715;
            let cwd = "/".to_string() + &"x".repeat(cwd_size - 1);
            let line = build_history_line("important prompt!", "", &cwd, "", 1);
            assert!(
                line.is_empty(),
                "expected empty result (row dropped), got {} bytes: {line:?}",
                line.len()
            );
        }

        #[test]
        fn build_history_line_includes_session_id_in_output() {
            let line = build_history_line("hello", "", "/project", "sess-abc-123", 7);
            assert!(line.ends_with('\n'));
            let row: HistoryRow = serde_json::from_str(line.trim_end()).expect("parse");
            assert_eq!(row.session_id, "sess-abc-123");
            assert_eq!(row.display, "hello");
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

/// What to do with a readline result at the idle chat prompt.
///
/// Kept as a pure enum + classifier (`classify_prompt_input`) so the
/// chat-exit contract is unit-testable without spinning up an async
/// client.  The live loop in [`run_chat_loop`] mirrors these outcomes
/// one-to-one.
#[derive(Debug, PartialEq, Eq)]
enum PromptInputDecision {
    /// Use the line as a user message to the daemon.
    Send(String),
    /// Silently re-prompt.  Used only for empty / whitespace-only
    /// Enter.  (Single Ctrl-C presses go through `FirstCtrlC`
    /// instead, so the caller can warn the user and record the
    /// timestamp for the double-tap check.)
    Continue,
    /// Exit the chat session.  Triggered by Ctrl-D / EOF and by a
    /// second Ctrl-C inside [`DOUBLE_CTRLC_WINDOW`].
    Exit,
    /// Record `now` as the first press of a potential Ctrl-C double
    /// tap and re-prompt.  The caller stores this in `last_ctrl_c`.
    FirstCtrlC,
}

/// Classify a raw readline outcome plus current Ctrl-C state into a
/// [`PromptInputDecision`].  Does not mutate state; the caller is
/// responsible for recording `FirstCtrlC` into `last_ctrl_c`.
fn classify_prompt_input(
    line: Option<String>,
    is_interrupt: bool,
    last_ctrl_c: Option<Instant>,
    now: Instant,
) -> PromptInputDecision {
    if is_interrupt {
        return if is_within_window(last_ctrl_c, now, DOUBLE_CTRLC_WINDOW) {
            PromptInputDecision::Exit
        } else {
            PromptInputDecision::FirstCtrlC
        };
    }
    match line {
        None => PromptInputDecision::Exit,
        Some(s) if s.trim().is_empty() => PromptInputDecision::Continue,
        Some(s) => PromptInputDecision::Send(s),
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
    // classify_prompt_input — chat idle-prompt exit contract
    //
    // The chat session's exit contract at the idle prompt is:
    //   * Ctrl-D (EOF)                              → immediate exit
    //   * Empty Enter / whitespace-only Enter       → silent re-prompt
    //   * Single Ctrl-C                             → warn + re-prompt
    //   * Second Ctrl-C within DOUBLE_CTRLC_WINDOW  → exit
    //   * Non-empty text                            → send to daemon
    //
    // These tests pin that contract — regressing any of them turns a
    // stray keystroke back into a session killer.
    // -----------------------------------------------------------------------

    #[test]
    fn classify_eof_is_exit() {
        let d = classify_prompt_input(None, false, None, Instant::now());
        assert_eq!(d, PromptInputDecision::Exit);
    }

    #[test]
    fn classify_empty_line_is_continue() {
        // An empty Enter on the idle prompt must not exit the chat and
        // must not be forwarded to the daemon.  Regression guard against
        // the pre-fix behaviour that broke sessions after long streams.
        let d = classify_prompt_input(Some(String::new()), false, None, Instant::now());
        assert_eq!(d, PromptInputDecision::Continue);
    }

    #[test]
    fn classify_whitespace_only_is_continue() {
        let d = classify_prompt_input(Some("   \t  ".to_string()), false, None, Instant::now());
        assert_eq!(d, PromptInputDecision::Continue);
    }

    #[test]
    fn classify_non_empty_line_is_send() {
        let d = classify_prompt_input(Some("hello".into()), false, None, Instant::now());
        assert_eq!(d, PromptInputDecision::Send("hello".into()));
    }

    #[test]
    fn classify_first_ctrl_c_is_first_tap() {
        // No prior Ctrl-C recorded → single Ctrl-C is a first tap, not an exit.
        let d = classify_prompt_input(None, true, None, Instant::now());
        assert_eq!(d, PromptInputDecision::FirstCtrlC);
    }

    #[test]
    fn classify_second_ctrl_c_inside_window_is_exit() {
        let now = Instant::now();
        let first = now - Duration::from_secs(1);
        let d = classify_prompt_input(None, true, Some(first), now);
        assert_eq!(d, PromptInputDecision::Exit);
    }

    #[test]
    fn classify_second_ctrl_c_after_window_is_first_tap() {
        // Two Ctrl-Cs spaced beyond the window are independent presses
        // — the second one starts its own double-tap window and does
        // NOT exit.
        let now = Instant::now();
        let first = now - DOUBLE_CTRLC_WINDOW - Duration::from_secs(1);
        let d = classify_prompt_input(None, true, Some(first), now);
        assert_eq!(d, PromptInputDecision::FirstCtrlC);
    }

    #[test]
    fn ctrl_c_then_empty_enter_then_ctrl_c_does_not_exit() {
        // The "two Ctrl-C to exit" contract requires the two presses to
        // be **consecutive**.  Any intervening input — including an
        // empty / whitespace-only Enter — must reset the double-tap
        // window, so a subsequent Ctrl-C behaves like a fresh first
        // tap.  This threads the same `last_ctrl_c` state that
        // `run_chat_loop` threads, to make sure the clear-on-Continue
        // call-site wiring stays in place.
        let t0 = Instant::now();
        let mut last_ctrl_c: Option<Instant> = None;

        // Step 1: single Ctrl-C at the idle prompt → FirstCtrlC.  The
        // caller records the timestamp.
        let d1 = classify_prompt_input(None, true, last_ctrl_c, t0);
        assert_eq!(d1, PromptInputDecision::FirstCtrlC);
        last_ctrl_c = Some(t0);

        // Step 2: empty Enter a moment later — well inside the
        // double-tap window — is a `Continue`.  `run_chat_loop`
        // clears `last_ctrl_c` on this path.
        let t1 = t0 + Duration::from_millis(200);
        let d2 = classify_prompt_input(Some(String::new()), false, last_ctrl_c, t1);
        assert_eq!(d2, PromptInputDecision::Continue);
        last_ctrl_c = None;

        // Step 3: Ctrl-C again, still inside what would have been the
        // original double-tap window.  Because Step 2 cleared the
        // state, this is a fresh first tap — NOT Exit.
        let t2 = t0 + Duration::from_millis(400);
        assert!(t2 - t0 < DOUBLE_CTRLC_WINDOW);
        let d3 = classify_prompt_input(None, true, last_ctrl_c, t2);
        assert_eq!(d3, PromptInputDecision::FirstCtrlC);
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
    fn parse_release_pane_basic() {
        let got = parse_release("/release %54").unwrap().unwrap();
        assert_eq!(
            got,
            ReleaseCmd::Pane {
                pane_id: "%54".into(),
                clean: false,
                summary: None,
            }
        );
    }

    #[test]
    fn parse_release_pane_with_flags_and_summary() {
        let got = parse_release("/release %54 --clean --summary \"all green\"")
            .unwrap()
            .unwrap();
        assert_eq!(
            got,
            ReleaseCmd::Pane {
                pane_id: "%54".into(),
                clean: true,
                summary: Some("all green".into()),
            }
        );
    }

    #[test]
    fn parse_release_all_with_clean() {
        let got = parse_release("/release all --clean").unwrap().unwrap();
        assert_eq!(got, ReleaseCmd::All { clean: true });
    }

    #[test]
    fn parse_release_all_rejects_summary() {
        let err = parse_release("/release all --summary \"whatever\"")
            .unwrap()
            .unwrap_err();
        assert!(err.contains("--summary"));
    }

    #[test]
    fn parse_release_bare_returns_usage_err() {
        let err = parse_release("/release").unwrap().unwrap_err();
        assert!(err.contains("usage"));
    }

    #[test]
    fn parse_release_false_positive_prefix_rejected() {
        assert!(parse_release("/releases").is_none());
        assert!(parse_release("/release-all").is_none());
    }

    #[test]
    fn parse_claude_escaped_quotes_in_description() {
        let tasks = claude_tasks(r#"/claude "task with \"quotes\"""#);
        assert_eq!(tasks[0].description, r#"task with "quotes""#);
    }

    #[test]
    fn parse_claude_resource_flag_collects_specs() {
        let tasks = claude_tasks("/claude --resource sim-9900 --resource class:gpu \"run kernel\"");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].resources, vec!["sim-9900", "class:gpu"]);
        assert!(
            tasks[0].resource_timeout_secs.is_none(),
            "timeout defaults to None (Nowait)"
        );
    }

    #[test]
    fn parse_claude_resource_timeout_parses_seconds() {
        let tasks = claude_tasks("/claude --resource class:gpu --resource-timeout 300 \"task\"");
        assert_eq!(tasks[0].resource_timeout_secs, Some(300));
    }

    #[test]
    fn parse_claude_resource_requires_value() {
        let result = parse_claude("/claude --resource");
        assert!(matches!(result, Some(Err(_))));
    }

    #[test]
    fn parse_claude_resume_pane_with_resource_is_allowed() {
        // Regression: the combo used to be rejected at parse time because
        // env-var injection can't run against an already-launched claude.
        // We now accept it — the daemon acquires the resource lock (which
        // is process-independent) and skips env/prompt_hint rendering on
        // the had_claude branch.  AGENTS.md carries the constraint forward
        // across /compact.
        let tasks = claude_tasks("/claude --resume-pane %41 --resource sim-9900 \"work\"");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].resume_pane.as_deref(), Some("%41"));
        assert_eq!(tasks[0].resources, vec!["sim-9900".to_string()]);
        assert_eq!(tasks[0].description, "work");
    }

    #[test]
    fn parse_claude_resume_pane_with_resource_and_timeout_ok() {
        // Timeout is just a number; the lock acquisition path on resume
        // still honours it the same way as a fresh launch.
        let tasks = claude_tasks(
            "/claude --resume-pane %41 --resource sim-9900 --resource-timeout 120 \"work\"",
        );
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].resume_pane.as_deref(), Some("%41"));
        assert_eq!(tasks[0].resources, vec!["sim-9900".to_string()]);
        assert_eq!(tasks[0].resource_timeout_secs, Some(120));
    }

    #[test]
    fn parse_claude_resource_specs_map_to_correct_request_variants() {
        // Regression: guards the handoff between the CLI parser (strings) and
        // `ResourceRequest::parse` (typed variant).  A future rename on either
        // side (`class:` prefix or the enum shape) would fail here before it
        // reaches the daemon and silently routes every request as Named.
        use crate::resource_lease::ResourceRequest;
        let tasks = claude_tasks(
            "/claude --resource sim-9900 --resource class:simulator --resource any:gpu \"run\"",
        );
        let requests: Vec<ResourceRequest> = tasks[0]
            .resources
            .iter()
            .map(|s| ResourceRequest::parse(s))
            .collect();
        assert_eq!(
            requests,
            vec![
                ResourceRequest::Named("sim-9900".into()),
                ResourceRequest::Class("simulator".into()),
                ResourceRequest::Class("gpu".into()),
            ]
        );
    }

    #[test]
    fn parse_claude_resource_timeout_rejects_non_integer() {
        let result = parse_claude("/claude --resource-timeout abc \"t\"");
        let err = match result {
            Some(Err(s)) => s,
            other => panic!("expected Err, got {other:?}"),
        };
        assert!(
            err.contains("non-negative integer"),
            "msg should explain the format: {err}"
        );
    }

    #[test]
    fn parse_claude_tag_override_is_collected() {
        let tasks = claude_tasks("/claude --tag kernel-opt \"run this\"");
        assert_eq!(tasks[0].tag, "kernel-opt");
        assert_eq!(tasks[0].description, "run this");
    }

    #[test]
    fn parse_claude_tag_without_value_errors() {
        let result = parse_claude("/claude --tag");
        assert!(matches!(result, Some(Err(_))));
    }

    #[test]
    fn parse_claude_without_tag_flag_leaves_empty() {
        // Parser leaves tag empty when `--tag` isn't passed.  The
        // client will run an IPC GenerateTag round-trip against the
        // daemon (Haiku or fallback slug) before sending ClaudeLaunch.
        let tasks = claude_tasks("/claude \"just run\"");
        assert_eq!(tasks[0].tag, "");
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
    // /claude --resume-pane
    // -----------------------------------------------------------------------

    #[test]
    fn parse_claude_resume_pane_basic() {
        let tasks = claude_tasks(r#"/claude --resume-pane %41 "continue fmha4""#);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].description, "continue fmha4");
        assert_eq!(tasks[0].resume_pane.as_deref(), Some("%41"));
        assert!(tasks[0].worktree.is_none());
    }

    #[test]
    fn parse_claude_resume_pane_unquoted_description() {
        let tasks = claude_tasks("/claude --resume-pane %41 继续完成上次的开发");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].description, "继续完成上次的开发");
        assert_eq!(tasks[0].resume_pane.as_deref(), Some("%41"));
    }

    #[test]
    fn parse_claude_resume_pane_no_value_errors() {
        let err = claude_err("/claude --resume-pane");
        assert!(err.contains("--resume-pane requires a pane id"));
    }

    #[test]
    fn parse_claude_resume_pane_with_worktree_is_rejected() {
        let err = claude_err(r#"/claude --resume-pane %41 --worktree /tmp "x""#);
        assert!(err.contains("mutually exclusive"));
    }

    #[test]
    fn parse_claude_resume_pane_multiple_tasks_rejected() {
        let err = claude_err(r#"/claude --resume-pane %41 "task a" "task b""#);
        assert!(err.contains("single task"));
    }

    #[test]
    fn parse_claude_resume_pane_empty_description_is_ok() {
        // Valid: daemon will pull the description from the lease.
        let tasks = claude_tasks("/claude --resume-pane %41");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].description, "");
        assert_eq!(tasks[0].resume_pane.as_deref(), Some("%41"));
    }

    #[test]
    fn parse_claude_bare_without_resume_pane_still_errors() {
        // Without --resume-pane, a missing description is still an error
        // (guards the normal-path invariant).
        let err = claude_err("/claude");
        assert!(err.contains("usage"));
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
