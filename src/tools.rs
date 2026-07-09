use std::collections::{HashMap, HashSet};
use std::io::SeekFrom;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tokio::process::Command;

use crate::sandbox::{docker::DockerSandboxConfig, DockerSandbox, NoopSandbox, Sandbox};

pub(crate) const DISTILLED_PROMPT_TARGET_LINES: usize = 10;
pub(crate) const MAX_DISTILLED_PROMPT_LINES: usize = 20;

// ---------------------------------------------------------------------------
// SpawnContext — shared state injected by the daemon for spawn_agent support
// ---------------------------------------------------------------------------

/// Context passed to `LocalExecutor` so the `spawn_agent` tool can launch a
/// child agentic loop without holding a reference back to `DaemonState`
/// (which would create a circular type dependency).
pub struct SpawnContext {
    /// HTTP client inherited from the parent daemon.
    pub http: reqwest::Client,
    /// Shared SQLite memory-DB connection.
    pub db: Arc<Mutex<rusqlite::Connection>>,
    /// Tracks sessions that currently have a compaction task in flight.
    pub compacting_sessions: Arc<Mutex<HashSet<String>>>,
    /// Shared Copilot token cache — reused by child agents to avoid redundant
    /// token fetches.
    pub tokens: Arc<crate::auth::TokenCache>,
    /// User-defined model aliases from `~/.amaebi/config.json`.  Consulted by
    /// `spawn_agent` (parent scope) when expanding the `model` argument
    /// before launching a child agent.  Also propagated into the child's
    /// `DaemonState.user_aliases` so the child's `switch_model` tool
    /// handler can resolve user aliases the same way the parent's does —
    /// children cannot `spawn_agent` themselves, but `switch_model` is
    /// always available.
    pub user_aliases: Arc<HashMap<String, String>>,
}

// ---------------------------------------------------------------------------
// ToolExecutor trait
// ---------------------------------------------------------------------------

/// Executes agent tools by name.  The trait exists so Phase 4 can swap in a
/// `DockerExecutor` without touching the agentic loop.
#[async_trait::async_trait]
pub trait ToolExecutor: Send + Sync {
    async fn execute(&self, name: &str, args: serde_json::Value) -> Result<String>;
}

// ---------------------------------------------------------------------------
// Local (host) executor
// ---------------------------------------------------------------------------

/// A local tool executor that optionally routes `shell_command` calls through
/// a sandbox backend.
///
/// # Environment variables
///
/// - `AMAEBI_SANDBOX=docker` — enable the Docker sandbox backend.
/// - `AMAEBI_SANDBOX_IMAGE` — override the Docker image used by the sandbox
///   (default: `"amaebi-sandbox:bookworm-slim"`).
///
/// When `AMAEBI_SANDBOX` is unset or set to any value other than `"docker"`,
/// commands run directly on the host via `sh -c`.
///
/// Set `AMAEBI_SANDBOX_WORKSPACE` to mount a specific directory (e.g. a git
/// worktree) as the workspace. Defaults to the current working directory.
#[derive(Default)]
pub struct LocalExecutor {
    /// Optional sandbox backend. When `Some`, `shell_command` runs inside the
    /// sandbox instead of directly on the host.
    pub sandbox: Option<Box<dyn Sandbox>>,
    /// Optional context for the `spawn_agent` tool.  Injected by the daemon
    /// at startup; `None` in child agents to prevent unbounded recursion.
    pub spawn_ctx: Option<Arc<SpawnContext>>,
    /// Default working directory for `shell_command` when a sandbox is active.
    /// Set to the agent's workspace in child executors so sandbox cwd matches
    /// the mounted workspace rather than the daemon process cwd.
    pub default_cwd: Option<PathBuf>,
}

impl LocalExecutor {
    pub fn new() -> Self {
        let mut default_cwd: Option<PathBuf> = None;
        let sandbox: Option<Box<dyn Sandbox>> = match std::env::var("AMAEBI_SANDBOX").as_deref() {
            Ok("docker") => {
                let image = std::env::var("AMAEBI_SANDBOX_IMAGE")
                    .unwrap_or_else(|_| "amaebi-sandbox:bookworm-slim".to_string());
                let workspace = std::env::var("AMAEBI_SANDBOX_WORKSPACE")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| std::env::current_dir().unwrap_or_default());
                default_cwd = Some(workspace.clone());
                Some(Box::new(DockerSandbox::new(DockerSandboxConfig {
                    image,
                    workspace,
                    ro_paths: vec![],
                    rw_paths: vec![],
                    env: HashMap::new(),
                })))
            }
            _ => None,
        };
        Self {
            sandbox,
            spawn_ctx: None,
            default_cwd,
        }
    }
}

#[async_trait::async_trait]
impl ToolExecutor for LocalExecutor {
    async fn execute(&self, name: &str, args: serde_json::Value) -> Result<String> {
        tracing::debug!(tool = %name, "executing tool");
        match name {
            "shell_command" => {
                shell_command(args, self.sandbox.as_deref(), self.default_cwd.as_deref()).await
            }
            "tmux_capture_pane" => tmux_capture_pane(args).await,
            "tmux_send_text" => tmux_send_text(args).await,
            "tmux_send_key" => tmux_send_key(args).await,
            "tmux_wait" => tmux_wait(args).await,
            "wait_for_file" => wait_for_file(args).await,
            "wait_for_task_event" => wait_for_task_event(args).await,
            "tmux_rename_pane" => tmux_rename_pane(args).await,
            "read_file" => read_file(args).await,
            "edit_file" => edit_file(args).await,
            "spawn_agent" => match &self.spawn_ctx {
                Some(ctx) => spawn_agent(args, ctx).await,
                None => anyhow::bail!(
                    "spawn_agent is not available in this context \
                     (child agents cannot spawn further agents)"
                ),
            },
            "task_done" => task_done(args),
            "emit_distilled_prompt" => emit_distilled_prompt(args),
            other => anyhow::bail!("unknown tool: {other}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

/// Run an arbitrary shell command, capturing stdout+stderr.
/// If a sandbox is provided the command runs inside it; otherwise it runs
/// directly on the host via `sh -c`.
async fn shell_command(
    args: serde_json::Value,
    sandbox: Option<&dyn Sandbox>,
    default_cwd: Option<&Path>,
) -> Result<String> {
    let command = args["command"]
        .as_str()
        .context("shell_command: missing string argument 'command'")?;

    tracing::debug!(command = %command, "running shell command");

    let (stdout, stderr, exit_code, success) = if let Some(sb) = sandbox {
        let cwd = if let Some(dcwd) = default_cwd {
            dcwd.to_path_buf()
        } else {
            std::env::current_dir().context("shell_command: getting current directory")?
        };
        let out = sb.spawn(command, &cwd).await?;
        let success = out.exit_code == 0;
        (out.stdout, out.stderr, out.exit_code, success)
    } else {
        let output = Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .await
            .with_context(|| format!("spawning shell command: {command}"))?;
        let exit_code = output.status.code().unwrap_or(-1);
        let success = output.status.success();
        (
            String::from_utf8_lossy(&output.stdout).into_owned(),
            String::from_utf8_lossy(&output.stderr).into_owned(),
            exit_code,
            success,
        )
    };

    let mut result = String::new();

    if !stdout.is_empty() {
        result.push_str(stdout.trim_end());
    }
    if !stderr.is_empty() {
        if !result.is_empty() {
            result.push_str("\n[stderr]\n");
        }
        result.push_str(stderr.trim_end());
    }
    if result.is_empty() {
        result = format!("[exit {exit_code}]");
    } else if !success {
        result.push_str(&format!("\n[exit {exit_code}]"));
    }

    Ok(result)
}

/// Capture the visible text of a tmux pane.
async fn tmux_capture_pane(args: serde_json::Value) -> Result<String> {
    // Default to the first pane if no target provided.
    let target = args["target"].as_str().unwrap_or("%0");

    let output = Command::new("tmux")
        .args(["capture-pane", "-t", target, "-p"])
        .output()
        .await
        .context("spawning tmux capture-pane")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("tmux capture-pane failed: {stderr}");
    }

    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Seconds to wait between sending a literal text paste and the
/// trailing `Enter`, giving the receiving TUI time to render the paste.
/// 5s covers Claude Code TUI markdown layout at typical paste sizes;
/// the old 1s-or-none behavior raced the renderer and dropped Enters.
pub const TEXT_RENDER_SLEEP_SECS: u64 = 5;

/// Send a block of text into a tmux pane and submit it.
///
/// Uses `tmux send-keys -l --` (literal mode) so the bytes are sent
/// verbatim — `Enter`, `C-c`, `\n`, and any other escape-like substring
/// inside `text` are NOT interpreted as tmux key names.  After the text
/// is injected we sleep `TEXT_RENDER_SLEEP_SECS` so the receiving TUI
/// (Claude Code, shells, etc.) can finish rendering before the trailing
/// `Enter` arrives — otherwise the Enter can be dropped or fire on an
/// empty input field.
///
/// For single control keys (`C-c`, `Escape`, `q`, arrow keys), use
/// `tmux_send_key` instead: no literal flag, no sleep, no Enter.
async fn tmux_send_text(args: serde_json::Value) -> Result<String> {
    let text = args["text"]
        .as_str()
        .context("tmux_send_text: missing string argument 'text'")?;
    let target = args["target"].as_str().unwrap_or("%0");

    // `-l --` forces literal mode and defends against payloads starting with `-`.
    let send = Command::new("tmux")
        .args(["send-keys", "-t", target, "-l", "--", text])
        .output()
        .await
        .context("spawning tmux send-keys (text)")?;
    if !send.status.success() {
        let stderr = String::from_utf8_lossy(&send.stderr);
        anyhow::bail!("tmux send-keys (text) failed: {stderr}");
    }

    tokio::time::sleep(std::time::Duration::from_secs(TEXT_RENDER_SLEEP_SECS)).await;

    let enter = Command::new("tmux")
        .args(["send-keys", "-t", target, "Enter"])
        .output()
        .await
        .context("spawning tmux send-keys (Enter)")?;
    if !enter.status.success() {
        let stderr = String::from_utf8_lossy(&enter.stderr);
        anyhow::bail!("tmux send-keys (Enter) failed: {stderr}");
    }

    Ok(format!("sent {} bytes to pane {target}", text.len()))
}

/// Send a single tmux key (or key combo) to a pane.
///
/// `key` is passed straight to `tmux send-keys` WITHOUT `-l`, so tmux's
/// key-name parser interprets it: `C-c`, `Escape`, `Up`, `q`, `Enter`,
/// etc. all work.  `--` is still passed before `key` so a payload that
/// starts with `-` can't be misread as a `send-keys` option flag.  No
/// Enter is appended and no sleep runs — use `tmux_send_text` for
/// prompt-like text + submit flows.
async fn tmux_send_key(args: serde_json::Value) -> Result<String> {
    let key = args["key"]
        .as_str()
        .context("tmux_send_key: missing string argument 'key'")?;
    let target = args["target"].as_str().unwrap_or("%0");

    let out = Command::new("tmux")
        .args(["send-keys", "-t", target, "--", key])
        .output()
        .await
        .context("spawning tmux send-keys (key)")?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        anyhow::bail!("tmux send-keys (key) failed: {stderr}");
    }

    Ok(format!("sent key {key:?} to pane {target}"))
}

/// Normalize a pane capture so live-UI animations don't masquerade as
/// activity.  Claude Code's TUI cycles a spinner glyph every poll and
/// re-renders elapsed-time counters every second (`(4m 35s)`, `↓ 5.8k
/// tokens`, `Running… (1m 6s)`); a byte-exact comparison therefore
/// never converges and `tmux_wait` blocks until `timeout_secs`.
///
/// Strategy:
/// 1. Per-line: detect Claude Code's "thinking" status line — a line
///    starting with a spinner glyph followed by `<Verb> for <duration>`
///    where the verb cycles every few seconds (Baked, Cogitated,
///    Worked, Brewed, Crunched, Mapping, Calculating, Comparing,
///    Pondering, …).  Collapse the whole line to a fixed token so
///    verb rotation does not register as activity.
/// 2. Char-level: collapse every run of ASCII digits to a single `0`
///    and every known spinner glyph to `*`.  Real content changes
///    (new lines, new tool calls, text generation) still differ
///    after normalization because the surrounding non-digit,
///    non-spinner characters change.
fn normalize_for_idle_check(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for (i, line) in s.split('\n').enumerate() {
        if i > 0 {
            out.push('\n');
        }
        if is_claude_thinking_line(line) {
            out.push_str("__claude_thinking__");
        } else {
            normalize_chars_into(line, &mut out);
        }
    }
    out
}

/// Detect a Claude Code "thinking" status line so the verb-cycling
/// animation (Baked / Cogitated / Worked / …) doesn't masquerade as
/// activity.  The line shape is roughly:
///   `<spinner> <Verb> for <duration>[ · <token info>][ · with <effort>]`
/// All text after the leading whitespace + spinner glyph is volatile;
/// the only stable signal is "this line exists at all" so we collapse
/// it to a fixed token regardless of the surrounding content.
fn is_claude_thinking_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    let mut chars = trimmed.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    let is_spinner = matches!(
        first,
        '✶' | '✻'
            | '✷'
            | '✸'
            | '✹'
            | '✺'
            | '✦'
            | '✧'
            | '⠋'
            | '⠙'
            | '⠹'
            | '⠸'
            | '⠼'
            | '⠴'
            | '⠦'
            | '⠧'
            | '⠇'
            | '⠏'
    );
    if !is_spinner {
        return false;
    }
    // Skip whitespace after spinner.
    let after_spinner = chars.as_str().trim_start();
    // The thinking line always contains " for " followed by a duration
    // such as "1h 28m 15s", "5s", "2m 30s".  Looking for ` for ` lets
    // us match the verb-cycling line without enumerating every verb,
    // and excludes the simpler "✻ Plan in progress" / similar lines
    // that don't follow the timer pattern.
    after_spinner.contains(" for ")
}

/// Char-level normalization shared by the thinking-line short-circuit
/// path and every other line: collapse digit runs to `0` and known
/// spinner glyphs to `*`.
fn normalize_chars_into(line: &str, out: &mut String) {
    let mut prev_digit = false;
    for c in line.chars() {
        match c {
            '✶' | '✻' | '✷' | '✸' | '✹' | '✺' | '✦' | '✧' | '⠋' | '⠙' | '⠹' | '⠸' | '⠼' | '⠴'
            | '⠦' | '⠧' | '⠇' | '⠏' => {
                out.push('*');
                prev_digit = false;
            }
            d if d.is_ascii_digit() => {
                if !prev_digit {
                    out.push('0');
                }
                prev_digit = true;
            }
            other => {
                out.push(other);
                prev_digit = false;
            }
        }
    }
}

/// Poll a tmux pane until its output has been stable for `idle_secs`, then
/// return the final pane content.
///
/// Instead of the LLM calling `tmux_capture_pane` in a loop (burning one LLM
/// turn per poll), a single `tmux_wait` call blocks until the command running
/// in the pane appears to have finished.
async fn tmux_wait(args: serde_json::Value) -> Result<String> {
    let target = args["target"].as_str().unwrap_or("%0");
    let idle_secs = args["idle_secs"].as_u64().unwrap_or(3);
    let timeout_secs = args["timeout_secs"].as_u64().unwrap_or(600).min(86_400);
    let poll_secs = args["poll_interval_secs"].as_u64().unwrap_or(2).max(1);

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    let mut last_normalized = String::new();
    let mut stable_since = tokio::time::Instant::now();

    loop {
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!(
                "tmux_wait: timed out after {timeout_secs}s waiting for pane '{target}' to become idle"
            );
        }

        let capture_fut = Command::new("tmux")
            .args(["capture-pane", "-t", target, "-p"])
            .output();
        let output = tokio::time::timeout_at(deadline, capture_fut)
            .await
            .map_err(|_| {
                anyhow::anyhow!("tmux_wait: capture-pane timed out waiting for pane '{target}'")
            })?
            .context("tmux_wait: spawning tmux capture-pane")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "tmux_wait: capture-pane failed (exit {}): {}",
                output.status,
                stderr.trim()
            );
        }
        let content = String::from_utf8_lossy(&output.stdout).into_owned();
        let normalized = normalize_for_idle_check(&content);

        if normalized != last_normalized {
            last_normalized = normalized;
            stable_since = tokio::time::Instant::now();
        } else if stable_since.elapsed().as_secs() >= idle_secs {
            return Ok(content);
        }

        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let sleep_dur = std::time::Duration::from_secs(poll_secs).min(remaining);
        tokio::time::sleep(sleep_dur).await;
    }
}

/// Block until `path` exists on the filesystem, then return `"found"`.
async fn wait_for_file(args: serde_json::Value) -> Result<String> {
    let path = args["path"]
        .as_str()
        .context("wait_for_file: missing string argument 'path'")?;
    let timeout_secs = args["timeout_secs"].as_u64().unwrap_or(300).min(86_400);
    let poll_ms = args["poll_interval_ms"].as_u64().unwrap_or(500);
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    loop {
        match tokio::fs::metadata(path).await {
            Ok(m) if m.is_file() => return Ok("found".to_owned()),
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "wait_for_file: error checking '{path}': {e}"
                ))
            }
        }
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Ok(format!(
                "timeout: '{path}' did not appear within {timeout_secs}s"
            ));
        }
        tokio::time::sleep(std::time::Duration::from_millis(poll_ms).min(remaining)).await;
    }
}

const DEFAULT_TASK_EVENT_TAIL_LINES: usize = 80;
const MAX_TASK_EVENT_TAIL_LINES: usize = 1_000;
const MAX_TASK_EVENT_PAYLOAD_BYTES: u64 = 16 * 1024;
const MAX_TASK_EVENT_LOG_TAIL_BYTES: u64 = 256 * 1024;

#[derive(Debug, Clone)]
struct TaskEventSpec {
    name: String,
    path: String,
}

/// Block until one of several task event sentinel files appears.
///
/// This is intentionally a mechanical wake-up primitive: scripts or downstream
/// panes decide when to create event files, but the LLM still decides what the
/// event means and what to do next. Full logs remain on disk; this tool returns
/// only the event metadata plus an optional tail for fast triage.
async fn wait_for_task_event(args: serde_json::Value) -> Result<String> {
    let events = parse_task_event_specs(&args)?;
    let timeout_secs = args["timeout_secs"].as_u64().unwrap_or(300).min(86_400);
    let poll_ms = args["poll_interval_ms"].as_u64().unwrap_or(500).max(1);
    let log_path = args["log_path"].as_str().map(str::to_owned);
    let tail_lines = args["tail_lines"]
        .as_u64()
        .map(|v| (v as usize).min(MAX_TASK_EVENT_TAIL_LINES))
        .unwrap_or(DEFAULT_TASK_EVENT_TAIL_LINES);
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);

    loop {
        for event in &events {
            match tokio::fs::metadata(&event.path).await {
                Ok(m) if m.is_file() => {
                    return format_task_event_result(event, log_path.as_deref(), tail_lines).await;
                }
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "wait_for_task_event: error checking '{}': {e}",
                        event.path
                    ));
                }
            }
        }

        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return format_task_event_timeout(
                &events,
                timeout_secs,
                log_path.as_deref(),
                tail_lines,
            )
            .await;
        }
        tokio::time::sleep(std::time::Duration::from_millis(poll_ms).min(remaining)).await;
    }
}

fn parse_task_event_specs(args: &serde_json::Value) -> Result<Vec<TaskEventSpec>> {
    let raw = args["events"]
        .as_array()
        .context("wait_for_task_event: missing array argument 'events'")?;
    if raw.is_empty() {
        anyhow::bail!("wait_for_task_event: 'events' must not be empty");
    }

    let mut events = Vec::with_capacity(raw.len());
    for (idx, event) in raw.iter().enumerate() {
        let name = event["name"]
            .as_str()
            .with_context(|| format!("wait_for_task_event: events[{idx}].name must be a string"))?;
        let path = event["path"]
            .as_str()
            .with_context(|| format!("wait_for_task_event: events[{idx}].path must be a string"))?;
        if name.trim().is_empty() {
            anyhow::bail!("wait_for_task_event: events[{idx}].name must be non-empty");
        }
        if path.trim().is_empty() {
            anyhow::bail!("wait_for_task_event: events[{idx}].path must be non-empty");
        }
        events.push(TaskEventSpec {
            name: name.to_owned(),
            path: path.to_owned(),
        });
    }

    Ok(events)
}

async fn format_task_event_result(
    event: &TaskEventSpec,
    log_path: Option<&str>,
    tail_lines: usize,
) -> Result<String> {
    let mut out = format!(
        "event: {}\nevent_path: {}\nevent_payload:\n{}",
        event.name,
        event.path,
        read_task_event_payload(&event.path).await?
    );
    append_optional_log_tail(&mut out, log_path, tail_lines).await?;
    Ok(out)
}

async fn format_task_event_timeout(
    events: &[TaskEventSpec],
    timeout_secs: u64,
    log_path: Option<&str>,
    tail_lines: usize,
) -> Result<String> {
    let mut out =
        format!("timeout: no task event appeared within {timeout_secs}s\nwatched_events:");
    for event in events {
        out.push_str(&format!("\n- {}: {}", event.name, event.path));
    }
    append_optional_log_tail(&mut out, log_path, tail_lines).await?;
    Ok(out)
}

async fn read_task_event_payload(path: &str) -> Result<String> {
    read_file_prefix(path, MAX_TASK_EVENT_PAYLOAD_BYTES)
        .await
        .with_context(|| format!("wait_for_task_event: reading event payload '{path}'"))
}

async fn append_optional_log_tail(
    out: &mut String,
    log_path: Option<&str>,
    tail_lines: usize,
) -> Result<()> {
    let Some(path) = log_path else {
        return Ok(());
    };
    out.push_str(&format!("\nlog_path: {path}\nlog_tail:\n"));
    match read_file_tail(path, tail_lines, MAX_TASK_EVENT_LOG_TAIL_BYTES).await {
        Ok(tail) => out.push_str(&tail),
        Err(e) => out.push_str(&format!("[unable to read log tail: {e}]")),
    }
    Ok(())
}

async fn read_file_prefix(path: &str, max_bytes: u64) -> Result<String> {
    let mut file = tokio::fs::File::open(path).await?;
    let mut limited = (&mut file).take(max_bytes + 1);
    let mut buf = Vec::new();
    limited.read_to_end(&mut buf).await?;
    let truncated = buf.len() as u64 > max_bytes;
    if truncated {
        buf.truncate(max_bytes as usize);
    }
    let mut text = String::from_utf8_lossy(&buf).into_owned();
    if truncated {
        text.push_str("\n[truncated]");
    }
    Ok(text)
}

async fn read_file_tail(path: &str, lines: usize, max_bytes: u64) -> Result<String> {
    if lines == 0 {
        return Ok(String::new());
    }
    let mut file = tokio::fs::File::open(path).await?;
    let len = file.metadata().await?.len();
    let start = len.saturating_sub(max_bytes);
    file.seek(SeekFrom::Start(start)).await?;

    let mut buf = Vec::new();
    file.read_to_end(&mut buf).await?;
    let mut text = String::from_utf8_lossy(&buf).into_owned();
    if start > 0 {
        text = match text.find('\n') {
            Some(pos) => text[pos + 1..].to_owned(),
            None => String::new(),
        };
    }

    let collected: Vec<&str> = text.lines().rev().take(lines).collect();
    let mut tail = collected.into_iter().rev().collect::<Vec<_>>().join("\n");
    if text.ends_with('\n') && !tail.is_empty() {
        tail.push('\n');
    }
    Ok(tail)
}

/// Rename a tmux pane by setting its title.
async fn tmux_rename_pane(args: serde_json::Value) -> Result<String> {
    let target = args["target"]
        .as_str()
        .context("tmux_rename_pane: missing string argument 'target'")?;
    let title = args["title"]
        .as_str()
        .context("tmux_rename_pane: missing string argument 'title'")?;

    let output = Command::new("tmux")
        .args(["select-pane", "-t", target, "-T", title])
        .output()
        .await
        .context("spawning tmux select-pane")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("tmux select-pane -T failed: {stderr}");
    }

    Ok(format!("renamed pane {target} to \"{title}\""))
}

/// Read the full contents of a file.
async fn read_file(args: serde_json::Value) -> Result<String> {
    let path = args["path"]
        .as_str()
        .context("read_file: missing string argument 'path'")?;

    tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("read_file: reading '{path}'"))
}

/// Spawn a child agent session to complete a task in an isolated sandbox.
///
/// # Sandbox selection
///
/// - When `AMAEBI_SPAWN_SANDBOX=noop` is set, a [`NoopSandbox`] is used
///   (runs commands directly on the host). Intended for tests.
/// - Otherwise a [`DockerSandbox`] is created with `--network none`.
///   If Docker is not available, an error is returned.
///
/// # Recursion prevention
///
/// The child executor is created with `spawn_ctx: None` so it cannot
/// call `spawn_agent` itself.
/// TODO: enforce a depth limit for nested agents if needed.
/// Resolve the default model for a spawned sub-agent.
///
/// Mirrors the provider-prefix preservation logic in `compact_model` in
/// `daemon.rs`: if the parent session uses `copilot/` or `bedrock/` (as
/// indicated by `AMAEBI_MODEL`), the sub-agent defaults to the same backend
/// rather than falling back to bare `DEFAULT_MODEL` (Bedrock).
///
/// Resolution order:
///   1. `AMAEBI_SUBAGENT_MODEL` env var (used verbatim)
///   2. Provider prefix from `AMAEBI_MODEL` + per-provider default
///      (`default_model_for_provider`): Bedrock gets `[1m]`, Copilot gets bare
///   3. Bare `DEFAULT_MODEL` (Bedrock, with `[1m]`)
fn subagent_default_model() -> String {
    if let Ok(m) = std::env::var("AMAEBI_SUBAGENT_MODEL") {
        return m;
    }
    let parent = std::env::var("AMAEBI_MODEL").unwrap_or_default();
    let prefix = parent
        .split_once('/')
        .map(|(p, _)| p)
        .filter(|p| matches!(*p, "copilot" | "bedrock"));
    match prefix {
        Some(p) => format!("{}/{}", p, crate::provider::default_model_for_provider(p)),
        None => crate::provider::DEFAULT_MODEL.to_string(),
    }
}

/// `task_done` is intentionally a soft signal: the tool body only validates
/// arguments and echoes them back.  The daemon's agentic-loop dispatch
/// recognises `name == "task_done"` after the tool completes and calls
/// `release_held_entry` + streams the `TaskReleased` frame.  This keeps the
/// tool schema simple (no access to daemon state required at tool-exec
/// time) and mirrors how `switch_model` / `spawn_agent` have side effects
/// the loop observes via tool-name inspection.
fn task_done(args: serde_json::Value) -> Result<String> {
    let pane_id = args["pane_id"]
        .as_str()
        .context("task_done: missing string argument 'pane_id'")?
        .trim();
    if pane_id.is_empty() {
        anyhow::bail!("task_done: 'pane_id' must be non-empty");
    }
    let summary = args["summary"]
        .as_str()
        .context("task_done: missing string argument 'summary'")?
        .trim();
    if summary.is_empty() {
        anyhow::bail!("task_done: 'summary' must be non-empty");
    }
    let validation_evidence = args["validation_evidence"]
        .as_str()
        .context("task_done: missing string argument 'validation_evidence'")?
        .trim();
    if validation_evidence.is_empty() {
        anyhow::bail!("task_done: 'validation_evidence' must be non-empty");
    }
    validate_task_done_evidence(validation_evidence)?;
    Ok(format!(
        "[task_done signalled pane={pane_id}]\n{summary}\n\nvalidation evidence:\n{validation_evidence}"
    ))
}

fn validate_task_done_evidence(validation_evidence: &str) -> Result<()> {
    let evidence = validation_evidence.to_ascii_lowercase();
    let explicit_missing_validation = [
        "did not run",
        "didn't run",
        "has not run",
        "have not run",
        "not run",
        "not yet run",
        "not tested",
        "tests were not run",
        "no functional test",
        "no functionality test",
        "no simulator test",
        "without running test",
        "without running the test",
    ];
    if explicit_missing_validation
        .iter()
        .any(|needle| evidence.contains(needle))
    {
        anyhow::bail!(
            "task_done: validation_evidence says required validation is missing; \
             steer the pane to run the required validation before task_done"
        );
    }

    let has_validation_command = evidence.lines().any(line_has_downstream_validation_command);
    let has_validation_result = evidence.lines().any(line_has_validation_result);
    if has_validation_command && has_validation_result {
        return Ok(());
    }
    if has_validation_command {
        anyhow::bail!(
            "task_done: validation_evidence must include both a downstream validation \
             command and its passing result before task_done"
        );
    }

    let has_insufficient_only_marker = [
        "cmake --build",
        "cargo build",
        "bash -n",
        "git diff --check",
        "git status",
        "git push",
        "clean diff",
        "build-only",
        "build only",
        "added test",
        "committed test",
        "test script",
        "test list",
        "test coverage",
        "script syntax",
        "syntax validation",
        "whitespace check",
        "push verification",
    ]
    .iter()
    .any(|needle| evidence.contains(needle));
    if has_insufficient_only_marker {
        anyhow::bail!(
            "task_done: validation_evidence only shows build/syntax/diff/push checks \
             or test script/list/change-only evidence; \
             include downstream test, simulator, benchmark, or accuracy/performance \
             command output before task_done"
        );
    }

    anyhow::bail!(
        "task_done: validation_evidence must include both a downstream validation \
         command and its passing result before task_done"
    );
}

fn line_has_downstream_validation_command(line: &str) -> bool {
    let raw_line = line.trim();
    let commandish = is_commandish_line(raw_line);
    let line = line_without_syntax_check_segments(raw_line);
    let line = line.trim();
    if line.is_empty() {
        return false;
    }

    let has_explicit_command = [
        "scripts/test.sh",
        "cargo test",
        "cargo nextest",
        "cargo bench",
        "pytest",
        "go test",
        "npm test",
        "pnpm test",
        "yarn test",
        "ctest",
        "run_tests.sh --",
        "run_tests.sh --filter",
        "run_matrix.sh --",
    ]
    .iter()
    .any(|needle| line.contains(needle));
    if has_explicit_command {
        return true;
    }

    if commandish
        && ["run_tests.sh", "run_matrix.sh"]
            .iter()
            .any(|needle| line.contains(needle))
    {
        return true;
    }

    commandish
        && [
            "benchmark",
            "bench ",
            "accuracy",
            "performance",
            "simulator",
            "coralsim",
            "xesim",
        ]
        .iter()
        .any(|needle| line.contains(needle))
}

fn line_without_syntax_check_segments(line: &str) -> String {
    line.split("&&")
        .flat_map(|part| part.split(';'))
        .map(str::trim)
        .filter(|segment| !segment.contains("bash -n"))
        .collect::<Vec<_>>()
        .join(" && ")
}

fn is_commandish_line(line: &str) -> bool {
    let line = line.trim_start();
    let line = line
        .strip_prefix("- ")
        .or_else(|| line.strip_prefix("* "))
        .unwrap_or(line);
    line.starts_with("command:")
        || line.starts_with("cmd:")
        || line.starts_with("run:")
        || line.starts_with("running:")
        || line.starts_with("ran:")
        || line.starts_with("executed:")
        || line.starts_with("$ ")
        || line.starts_with("> ")
        || line.starts_with("./")
}

fn line_has_validation_result(line: &str) -> bool {
    let line = line.trim();
    if line == "pass" || line.starts_with("ok ") {
        return true;
    }
    if line.contains("0 failed") {
        return true;
    }
    if [
        "not passed",
        "not pass",
        "failed",
        "failure",
        "exit 1",
        "exit code 1",
        "error:",
    ]
    .iter()
    .any(|needle| line.contains(needle))
    {
        return false;
    }
    if line.starts_with("exit ") || line.starts_with("exit code ") {
        return line.contains("exit 0") || line.contains("exit code 0");
    }

    let structured_result = [
        "result:",
        "- result:",
        "status:",
        "- status:",
        "test result:",
        "disposition:",
        "numeric check:",
    ]
    .iter()
    .any(|prefix| line.starts_with(prefix));
    let structured_success = [
        "passed",
        "pass:",
        "ok",
        "exit 0",
        "exit code 0",
        "mismatch=0",
        "mismatch 0/",
        "cosine=1",
        "no regression",
        "within tolerance",
    ]
    .iter()
    .any(|needle| line.contains(needle));
    if structured_result && structured_success {
        return true;
    }

    ["mismatch=0", "mismatch 0/", "cosine=1", "no regression"]
        .iter()
        .any(|needle| line.contains(needle))
}

/// `emit_distilled_prompt` is the distillation analogue of `task_done`:
/// the tool body only validates and echoes; the daemon's agentic-loop
/// dispatch recognises `name == "emit_distilled_prompt"`, extracts the
/// `prompt` field, ships it back to the client as
/// `Response::DistilledPromptReady`, and exits the loop.
///
/// Only available in the distillation agentic loop launched by
/// `Request::DistillClaudePrompt` — the schema is gated on
/// `ToolMode::Distill` in `tool_schemas`.
fn emit_distilled_prompt(args: serde_json::Value) -> Result<String> {
    let prompt = args["prompt"]
        .as_str()
        .context("emit_distilled_prompt: missing string argument 'prompt'")?;
    if prompt.trim().is_empty() {
        anyhow::bail!("emit_distilled_prompt: 'prompt' must be non-empty");
    }
    let line_count = prompt.lines().count();
    if line_count > MAX_DISTILLED_PROMPT_LINES {
        anyhow::bail!(
            "emit_distilled_prompt: 'prompt' has {line_count} lines; target about \
             {DISTILLED_PROMPT_TARGET_LINES} lines and hard maximum is \
             {MAX_DISTILLED_PROMPT_LINES}"
        );
    }
    Ok(format!("[distilled prompt emitted, len={}]", prompt.len()))
}

async fn spawn_agent(args: serde_json::Value, ctx: &SpawnContext) -> Result<String> {
    let task = args["task"]
        .as_str()
        .context("spawn_agent: missing string argument 'task'")?;
    let workspace = PathBuf::from(
        args["workspace"]
            .as_str()
            .context("spawn_agent: missing string argument 'workspace'")?,
    );

    // Fix 1: validate workspace path early.
    if !workspace.is_absolute() {
        anyhow::bail!(
            "spawn_agent: workspace must be an absolute path, got: {}",
            workspace.display()
        );
    }
    if !workspace.exists() {
        anyhow::bail!(
            "spawn_agent: workspace does not exist: {}",
            workspace.display()
        );
    }
    if !workspace.is_dir() {
        anyhow::bail!(
            "spawn_agent: workspace is not a directory: {}",
            workspace.display()
        );
    }
    let workspace = workspace.canonicalize().with_context(|| {
        format!(
            "spawn_agent: canonicalizing workspace: {}",
            workspace.display()
        )
    })?;

    let model = args["model"]
        .as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(subagent_default_model);
    // Expand user-defined aliases here: the downstream run_agentic_loop
    // resolves the model via provider::resolve() which does not consult
    // user aliases, so `{"model": "opus"}` from the LLM must be expanded
    // before the child loop sees it.
    let model = crate::daemon::expand_user_alias(&model, &ctx.user_aliases);

    let extra_mounts = args["extra_mounts"].as_array().cloned().unwrap_or_default();
    let mut ro_paths: Vec<PathBuf> = vec![];
    let mut rw_paths: Vec<PathBuf> = vec![];
    for mount in &extra_mounts {
        let path = PathBuf::from(
            mount["path"]
                .as_str()
                .context("extra_mounts[].path must be a string")?,
        );
        if !path.is_absolute() {
            anyhow::bail!(
                "spawn_agent: extra_mounts path must be absolute, got: {}",
                path.display()
            );
        }
        if !path.exists() {
            anyhow::bail!(
                "spawn_agent: extra_mounts path does not exist: {}",
                path.display()
            );
        }
        let canonical_path = path
            .canonicalize()
            .with_context(|| format!("extra_mounts: canonicalizing path: {}", path.display()))?;
        let readonly = mount["readonly"].as_bool().unwrap_or(false);
        if readonly {
            ro_paths.push(canonical_path);
        } else {
            rw_paths.push(canonical_path);
        }
    }
    let env: HashMap<String, String> = if let Some(obj) = args["env"].as_object() {
        let mut map = HashMap::new();
        for (k, v) in obj {
            let val = v.as_str().ok_or_else(|| {
                anyhow::anyhow!(
                    "spawn_agent: env value for key {} must be a string, got: {}",
                    k,
                    v
                )
            })?;
            map.insert(k.clone(), val.to_string());
        }
        map
    } else {
        HashMap::new()
    };

    // Determine sandbox mode: explicit `sandbox` arg takes priority, then env var,
    // then default to docker.
    let sandbox_override = match args.get("sandbox") {
        Some(value) => {
            let s = value
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("spawn_agent: sandbox must be a string"))?;
            match s {
                "docker" | "noop" => Some(s),
                other => anyhow::bail!(
                    "spawn_agent: unsupported sandbox {other:?}; expected \"docker\" or \"noop\""
                ),
            }
        }
        None => None,
    };
    let using_noop = sandbox_override == Some("noop")
        || (sandbox_override.is_none()
            && std::env::var("AMAEBI_SPAWN_SANDBOX").as_deref() == Ok("noop"));
    let mut context_lines = vec![
        "[Sandbox Context]".to_string(),
        if using_noop {
            "You are running with a noop sandbox (no isolation); commands execute directly on the host.".to_string()
        } else {
            "You are running inside an isolated Docker sandbox.".to_string()
        },
        format!("- Working directory (read-write): {}", workspace.display()),
    ];
    for mount in &extra_mounts {
        if let Some(path) = mount["path"].as_str() {
            let readonly = mount["readonly"].as_bool().unwrap_or(false);
            let mode = if readonly { "read-only" } else { "read-write" };
            context_lines.push(format!("- Mount ({mode}): {path}"));
        }
    }
    if !using_noop {
        context_lines.push(
            "- /tmp is isolated from the host; files written here do not persist across sessions"
                .to_string(),
        );
        context_lines.push("- No outbound network access".to_string());
    }
    context_lines
        .push("- Do not attempt to access paths outside the listed mounts above".to_string());
    context_lines.push(String::new());
    context_lines.push("Task:".to_string());
    context_lines.push(task.to_string());
    let full_task = context_lines.join("\n");

    let model_source = if args["model"].as_str().is_some() {
        "explicit"
    } else if std::env::var("AMAEBI_SUBAGENT_MODEL").is_ok() {
        "AMAEBI_SUBAGENT_MODEL"
    } else {
        "default"
    };
    tracing::info!(
        task = %task,
        workspace = %workspace.display(),
        model = %model,
        model_source = %model_source,
        "spawn_agent: starting child agent"
    );

    // Build the child sandbox using the pre-computed `using_noop` flag.
    let child_sandbox: Box<dyn Sandbox> = if using_noop {
        Box::new(NoopSandbox)
    } else {
        let image = std::env::var("AMAEBI_SANDBOX_IMAGE")
            .unwrap_or_else(|_| "amaebi-sandbox:bookworm-slim".to_string());
        let docker = DockerSandbox::new(DockerSandboxConfig {
            image,
            workspace: workspace.clone(),
            ro_paths,
            rw_paths,
            env,
        });
        if !docker.available() {
            anyhow::bail!("Docker is not available; cannot spawn agent");
        }
        Box::new(docker)
    };

    // Child executor: no spawn_ctx (prevents unbounded recursion), and cwd
    // defaults to the workspace so sandbox commands start in the right place.
    let child_executor = LocalExecutor {
        sandbox: Some(child_sandbox),
        spawn_ctx: None,
        default_cwd: Some(workspace.clone()),
    };

    // Build a minimal DaemonState for the child: reuse the parent's HTTP
    // client, DB, compacting-sessions set, and shared token cache.  The child
    // has no spawn_ctx so it cannot recursively spawn further agents.
    let child_state = crate::daemon::DaemonState {
        http: ctx.http.clone(),
        tokens: Arc::clone(&ctx.tokens),
        executor: Box::new(child_executor),
        db: Arc::clone(&ctx.db),
        compacting_sessions: Arc::clone(&ctx.compacting_sessions),
        // Child agents get their own active_sessions set; they are ephemeral
        // and don't share the parent's session-lock namespace.
        active_sessions: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        // Children cannot themselves spawn further agents, but the
        // switch_model tool schema is still available to them.  Propagating
        // the alias table lets a child's switch_model call resolve user
        // aliases the same way the parent's does.
        user_aliases: Arc::clone(&ctx.user_aliases),
        // Child agents never touch the task notebook; supervision-loop
        // persistence is the parent supervision's concern, and children
        // don't run supervision themselves.
        tasks_db: Arc::new(std::sync::Mutex::new(None)),
        // Children don't hold /claude panes — /claude goes through the
        // parent daemon.  Fresh empty structures are cheap.
        conn_id_counter: Arc::new(std::sync::atomic::AtomicU64::new(1)),
        held: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
    };

    let messages = vec![
        crate::copilot::Message::system(
            "You are a child agent completing a specific task in an isolated sandbox. \
             Use available tools to complete the task, then provide a concise summary \
             of what you did and the outcome.",
        ),
        crate::copilot::Message::user(full_task),
    ];

    let mut sink = tokio::io::sink();
    // Drop the sender immediately so the child loop treats the session as
    // non-interactive (no steering, no question-asking).
    let (_, mut steer_rx) = tokio::sync::mpsc::channel::<Option<String>>(1);

    // Fix 5: child agents do not get spawn_agent in their tool schema to
    // prevent unbounded recursion at the schema level.
    let (final_text, _, _, _) = crate::daemon::run_agentic_loop(
        &child_state,
        &model,
        messages,
        &mut sink,
        &mut steer_rx,
        false,
        None,
        None, // child agents never hold /claude panes
    )
    .await?;

    tracing::info!(result_len = %final_text.len(), "spawn_agent: child agent completed");
    Ok(final_text)
}

/// Overwrite a file with new content.
async fn edit_file(args: serde_json::Value) -> Result<String> {
    let path = args["path"]
        .as_str()
        .context("edit_file: missing string argument 'path'")?;
    let content = args["content"]
        .as_str()
        .context("edit_file: missing string argument 'content'")?;

    tokio::fs::write(path, content)
        .await
        .with_context(|| format!("edit_file: writing '{path}'"))?;

    Ok(format!("wrote {} bytes to {path}", content.len()))
}

// ---------------------------------------------------------------------------
// Tool schemas (OpenAI function-calling format)
// ---------------------------------------------------------------------------

/// Which set of tools to expose to the LLM.
///
/// `Chat` is the everyday agentic loop: full tool set, optional
/// `spawn_agent` for the parent connection, includes `task_done` so a
/// `/claude` supervisor can release a pane.
///
/// `Distill` is the bounded loop launched by
/// `Request::DistillClaudePrompt`: read-only investigation tools plus
/// `emit_distilled_prompt`.  Excludes `edit_file`, `task_done`,
/// `spawn_agent`, and the tmux pane-mutation tools so the distiller
/// cannot accidentally start work it's only supposed to plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolMode {
    Chat { include_spawn_agent: bool },
    Distill,
}

/// Return the JSON schema array to include in a chat request.
pub fn tool_schemas(mode: ToolMode) -> Vec<serde_json::Value> {
    match mode {
        ToolMode::Chat {
            include_spawn_agent,
        } => chat_tool_schemas(include_spawn_agent),
        ToolMode::Distill => distill_tool_schemas(),
    }
}

fn distill_tool_schemas() -> Vec<serde_json::Value> {
    let all = chat_tool_schemas(false);
    // Whitelist by name: investigation-only + the terminator.  The
    // distiller is supposed to read code and produce a plan, not start
    // editing or spawning.
    let allowed = ["shell_command", "read_file", "emit_distilled_prompt"];
    let mut filtered: Vec<serde_json::Value> = all
        .into_iter()
        .filter(|s| {
            s["function"]["name"]
                .as_str()
                .map(|n| allowed.contains(&n))
                .unwrap_or(false)
        })
        .collect();
    // Append the distill-only emit tool (not present in the chat set).
    filtered.push(serde_json::json!({
        "type": "function",
        "function": {
            "name": "emit_distilled_prompt",
            "description": format!(
                "Emit the FINAL distilled prompt that will be injected into the \
                 downstream Claude Code pane. Call this exactly ONCE, after you \
                 have finished investigating the codebase and decided what \
                 Claude should actually do. Keep the prompt concise: target \
                 about {DISTILLED_PROMPT_TARGET_LINES} lines, hard maximum \
                 {MAX_DISTILLED_PROMPT_LINES} newline-separated lines. The \
                 string you pass becomes the opening user message for that \
                 Claude session. Do NOT call this speculatively before reading \
                 code; do NOT call it more than once per session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": format!(
                            "The concise prompt to inject into Claude Code. Target about \
                             {DISTILLED_PROMPT_TARGET_LINES} lines; hard maximum \
                             {MAX_DISTILLED_PROMPT_LINES} lines. No preamble, no \
                             meta-commentary; Claude will read this verbatim as its \
                             first user turn."
                        )
                    }
                },
                "required": ["prompt"]
            }
        }
    }));
    filtered
}

fn chat_tool_schemas(include_spawn_agent: bool) -> Vec<serde_json::Value> {
    let mut schemas = vec![
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "shell_command",
                "description": "Run a shell command (via sh -c) in the background and \
                                return its stdout and stderr. Use this for grep, find, git, \
                                cargo, systemctl, etc. Does NOT interact with the user's \
                                tmux pane.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute."
                        }
                    },
                    "required": ["command"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "tmux_capture_pane",
                "description": "Capture and return the current visible text of a tmux pane.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "tmux target pane (e.g. '%0', '0:1.0'). \
                                            Defaults to %0."
                        }
                    },
                    "required": []
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "tmux_send_text",
                "description": "Send a block of text to a tmux pane and submit it.  Use this \
                                when you want to paste a prompt / command / message into an \
                                interactive TUI (like Claude Code, a REPL, a shell).  The \
                                text is sent literally (escape sequences and key names inside \
                                the text are NOT interpreted), followed by a 5-second wait so \
                                the receiving TUI can finish rendering, followed by Enter to \
                                submit.  For single control keys like C-c, Escape, arrow \
                                keys, or q, use tmux_send_key instead.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "tmux pane target (e.g. '%3' or 'session:0.1'). \
                                            Defaults to '%0'."
                        },
                        "text": {
                            "type": "string",
                            "description": "The text to paste into the pane.  Will be \
                                            followed by an automatic Enter after a \
                                            render-delay."
                        }
                    },
                    "required": ["text"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "tmux_send_key",
                "description": "Send a single tmux key (or key combo) to a pane.  Examples: \
                                'C-c' (Ctrl-C), 'Escape', 'Up', 'q', 'Enter', 'C-m'.  Does \
                                NOT append an Enter or sleep — use tmux_send_text for \
                                prompt-style paste + submit flows.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "tmux pane target (e.g. '%3' or 'session:0.1'). \
                                            Defaults to '%0'."
                        },
                        "key": {
                            "type": "string",
                            "description": "A tmux key name (e.g. 'C-c', 'Escape', 'Up', \
                                            'Enter') or printable character."
                        }
                    },
                    "required": ["key"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "tmux_wait",
                "description": "Poll a tmux pane until its output has been stable for idle_secs, \
                                then return the final pane content. Use this instead of calling \
                                tmux_capture_pane in a loop while waiting for a long-running command \
                                (e.g. a build) to finish. Stability is measured against a normalized \
                                view of the capture: spinner glyphs and runs of ASCII digits are \
                                collapsed before comparison, so live TUI animations (spinners, \
                                elapsed-time counters, token counts, percentage tickers) do not \
                                count as activity. Real text changes — new lines, new tool-call \
                                names, generated prose — still register and reset the idle timer. \
                                The returned string is the raw, un-normalized capture.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Tmux pane target (e.g. \"%0\", \"session:window.pane\"). Default: \"%0\"."
                        },
                        "idle_secs": {
                            "type": "integer",
                            "description": "Seconds of unchanged output before returning. Default: 3."
                        },
                        "timeout_secs": {
                            "type": "integer",
                            "description": "Hard timeout in seconds before giving up. Default: 600. Maximum: 86400.",
                            "minimum": 1,
                            "maximum": 86400
                        },
                        "poll_interval_secs": {
                            "type": "integer",
                            "description": "How often to sample the pane, in seconds. Minimum: 1. Default: 2.",
                            "minimum": 1
                        }
                    },
                    "required": []
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "wait_for_file",
                "description": "Block until a file appears at the given path, then return \"found\". \
                                Returns a timeout message if the file does not appear within timeout_secs. \
                                Use this instead of polling tmux_capture_pane when a script can write \
                                a sentinel file on completion.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative path of the file to wait for."
                        },
                        "timeout_secs": {
                            "type": "integer",
                            "description": "Maximum seconds to wait before returning timeout message. Default: 300."
                        },
                        "poll_interval_ms": {
                            "type": "integer",
                            "description": "How often to check for the file, in milliseconds. Default: 500."
                        }
                    },
                    "required": ["path"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "wait_for_task_event",
                "description": "Block until one of several task-event sentinel files appears, \
                                then return the event name, event file payload, and optional \
                                tail of a full log file. Use this for an LLM wake-up protocol: \
                                first steer the downstream pane to run a long build/test/simulator \
                                command with stdout/stderr tee'd to a log file and to write \
                                sentinel files for decision points such as passed, failed, \
                                anomaly, or no_progress; then call this tool once instead of \
                                repeatedly polling tmux_capture_pane. This tool does not judge \
                                the event or summarize the task; the supervisor still decides \
                                the next action and can read the full log_path when needed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "events": {
                            "type": "array",
                            "description": "Sentinel files to watch. The first file that appears wakes the supervisor.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Event label to return, e.g. 'passed', 'failed', 'anomaly', 'no_progress'."
                                    },
                                    "path": {
                                        "type": "string",
                                        "description": "Absolute or relative path to the sentinel file."
                                    }
                                },
                                "required": ["name", "path"]
                            }
                        },
                        "timeout_secs": {
                            "type": "integer",
                            "description": "Hard timeout in seconds before returning a timeout status. Default: 300. Maximum: 86400.",
                            "minimum": 0,
                            "maximum": 86400
                        },
                        "poll_interval_ms": {
                            "type": "integer",
                            "description": "How often to check sentinel files, in milliseconds. Minimum: 1. Default: 500.",
                            "minimum": 1
                        },
                        "log_path": {
                            "type": "string",
                            "description": "Optional path to the full log file for the running task. The full log remains on disk."
                        },
                        "tail_lines": {
                            "type": "integer",
                            "description": "Optional number of log tail lines to return for triage. Default: 80. Maximum: 1000.",
                            "minimum": 0,
                            "maximum": 1000
                        }
                    },
                    "required": ["events"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "tmux_rename_pane",
                "description": "Set the title of a tmux pane using 'tmux select-pane -T'. \
                                Useful for labelling panes with the current task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "tmux target pane (e.g. '%3'). Required."
                        },
                        "title": {
                            "type": "string",
                            "description": "New pane title to display."
                        }
                    },
                    "required": ["target", "title"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the full contents of a file on disk.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative path to the file."
                        }
                    },
                    "required": ["path"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Overwrite a file with new content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path of the file to write."
                        },
                        "content": {
                            "type": "string",
                            "description": "New full content for the file."
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "task_done",
                "description": "Declare a /claude-launched pane's task complete.  Call this only \
                                after you have observed the downstream Claude/Codex pane run the \
                                required validation (tests, benchmarks, accuracy/performance \
                                checks) and pass the task's acceptance criteria.  A clean diff, \
                                committed test script, build-only check, or self-report without \
                                validation output is not enough.  If validation is missing, steer \
                                the pane with tmux_send_text instead of calling task_done.  The \
                                required validation_evidence argument must cite the downstream \
                                pane's validation commands and results.  The daemon then releases amaebi's \
                                ownership of the pane (pane lease, resource lease, task-notebook \
                                lease) and streams a TaskReleased frame to the user with the pane \
                                tail + worktree status + your summary.  The tmux pane and the \
                                `claude` process inside it are NOT killed; the worktree is kept.  \
                                After this call, the LLM no longer needs to monitor that pane.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pane_id": {
                            "type": "string",
                            "description": "The tmux pane id (e.g. '%54') reported by the \
                                            preceding [launched] block.  No default — explicit \
                                            even in single-pane chats."
                        },
                        "summary": {
                            "type": "string",
                            "description": "Your final summary of what was accomplished.  \
                                            Stream-rendered to the user AND archived to the \
                                            inbox for later review."
                        },
                        "validation_evidence": {
                            "type": "string",
                            "description": "Evidence copied or summarized from the downstream \
                                            Claude/Codex pane output before release: exact \
                                            validation commands it ran, pass/fail output, relevant \
                                            accuracy/performance results, and baseline/no-regression \
                                            comparison when applicable.  A build-only check, clean \
                                            diff, committed test script, or self-report that tests \
                                            were not run is not valid evidence.  If this evidence is \
                                            missing, do not call task_done; use tmux_send_text to ask \
                                            the pane to run the required validation."
                        }
                    },
                    "required": ["pane_id", "summary", "validation_evidence"]
                }
            }
        }),
    ];
    if include_spawn_agent {
        schemas.push(serde_json::json!({
            "type": "function",
            "function": {
                "name": "spawn_agent",
                "description": "Run an independent sub-task in a child agent. \
                                \n\n\
                                WHEN TO USE — mandate:\n\
                                If the user request contains >= 2 independent sub-tasks, \
                                you MUST emit one spawn_agent call per sub-task in the SAME \
                                tool-call batch, each with `parallel: true`. Do NOT run them \
                                sequentially via shell_command / edit_file — that is strictly \
                                slower and wastes context.\n\n\
                                BATCH PURITY — required for concurrent execution:\n\
                                The daemon only runs a batch concurrently when EVERY call in \
                                the batch is `spawn_agent` with `parallel: true`. Mixing any \
                                other tool (read_file, shell_command, edit_file, …) into the \
                                same batch disables the concurrent fast path and the whole \
                                batch runs sequentially. The daemon emits a WARN when a batch \
                                with >=2 `spawn_agent` calls misses the fast path (mixed or \
                                missing `parallel: true`). If you need to read a file or run a \
                                shell command, do it in a SEPARATE turn before or after the \
                                spawn_agent fan-out, not in the same batch.\n\n\
                                \"Independent\" means: no data dependency, separate files or \
                                directories, either execution order is valid.\n\n\
                                MUST fan out via parallel spawn_agent:\n\
                                - Reviewing multiple files → one spawn per file\n\
                                - Running multiple benchmarks / test suites → one spawn each\n\
                                - Fixing multiple unrelated bugs → one spawn per bug\n\
                                - Summarizing multiple docs → one spawn per doc\n\n\
                                MUST NOT spawn_agent:\n\
                                - Single-file edit or single shell command\n\
                                - Strict ordering (build → test → deploy)\n\
                                - Any step whose input is the previous step's output\n\n\
                                Sandbox: default Docker with --network none. Set sandbox='noop' \
                                for host-direct execution when the task needs network or the \
                                host toolchain (cargo, git push, etc.).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task for the child agent to complete."
                        },
                        "workspace": {
                            "type": "string",
                            "description": "Absolute path to the workspace directory \
                                            (e.g. a git worktree). Will be bind-mounted rw."
                        },
                        "model": {
                            "type": "string",
                            "description": (format!(
                                "LLM model to use (optional; defaults to AMAEBI_SUBAGENT_MODEL \
                                 env var, or {} if unset). Supports provider/model format \
                                 (e.g. bedrock/claude-haiku-4.5).",
                                crate::provider::DEFAULT_MODEL
                            ))
                        },
                        "extra_mounts": {
                            "type": "array",
                            "description": "Additional directories to mount into the sandbox (optional). \
                                            Each path must be absolute and exist on the host.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {
                                        "type": "string",
                                        "description": "Absolute path on the host."
                                    },
                                    "readonly": {
                                        "type": "boolean",
                                        "description": "Mount as read-only (default: false)."
                                    }
                                },
                                "required": ["path"]
                            }
                        },
                        "env": {
                            "type": "object",
                            "description": "Environment variables to set inside the sandbox container (e.g. HTTP_PROXY).",
                            "additionalProperties": { "type": "string" }
                        },
                        "parallel": {
                            "type": "boolean",
                            "description": "Run concurrently with sibling spawn_agent calls in the same batch. \
                                            DEFAULT INTENT IS TRUE — set `parallel: true` unless the sub-tasks \
                                            have a strict ordering dependency. Only set `parallel: false` when \
                                            this sub-task must wait for another's output. Wire default remains \
                                            false for backward compat; you should almost always pass true explicitly."
                        },
                        "sandbox": {
                            "type": "string",
                            "description": "Sandbox backend: 'docker' (default, network-isolated) or 'noop' \
                                            (host-direct, for tasks needing cargo/git).",
                            "enum": ["docker", "noop"]
                        }
                    },
                    "required": ["task", "workspace"]
                }
            }
        }));
    }

    // switch_model is always available (not gated on include_spawn_agent).
    // The tool has no executor implementation — it is intercepted and handled
    // directly inside run_agentic_loop before the executor is called.
    schemas.push(serde_json::json!({
        "type": "function",
        "function": {
            "name": "switch_model",
            "description": "Switch the AI model used for the remainder of this session. \
                            Use a more capable model (e.g. claude-opus-4.6) for tasks \
                            requiring deep reasoning or planning; switch back to a faster \
                            model (e.g. claude-sonnet-4.6) for routine work like reading \
                            files or running commands.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": (format!(
                            "Model to switch to. Supports provider/model format \
                             (e.g. bedrock/claude-opus-4.6, copilot/gpt-4o). \
                             Project default: {}.",
                            crate::provider::DEFAULT_MODEL
                        ))
                    }
                },
                "required": ["model"]
            }
        }
    }));

    schemas
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use tempfile::TempDir;

    // ---- tool_schemas ----------------------------------------------------

    #[test]
    fn tool_schemas_have_expected_names() {
        let schemas = tool_schemas(ToolMode::Chat {
            include_spawn_agent: true,
        });
        let names: Vec<&str> = schemas
            .iter()
            .map(|s| s["function"]["name"].as_str().unwrap())
            .collect();
        for name in [
            "shell_command",
            "tmux_capture_pane",
            "tmux_send_text",
            "tmux_send_key",
            "tmux_wait",
            "wait_for_file",
            "wait_for_task_event",
            "read_file",
            "edit_file",
            "spawn_agent",
            "switch_model",
        ] {
            assert!(names.contains(&name), "missing tool: {name}");
        }
    }

    #[test]
    fn tool_schemas_all_have_type_function() {
        for schema in tool_schemas(ToolMode::Chat {
            include_spawn_agent: true,
        }) {
            assert_eq!(
                schema["type"].as_str().unwrap(),
                "function",
                "unexpected type for: {}",
                schema["function"]["name"]
            );
        }
    }

    #[test]
    fn tool_schemas_all_have_parameters_with_required_array() {
        for schema in tool_schemas(ToolMode::Chat {
            include_spawn_agent: true,
        }) {
            let name = schema["function"]["name"].as_str().unwrap();
            assert!(
                schema["function"]["parameters"]["required"].is_array(),
                "missing required array for {name}"
            );
        }
    }

    // ---- spawn_agent schema -------------------------------------------------

    #[test]
    fn spawn_agent_schema_has_extra_mounts() {
        let schemas = tool_schemas(ToolMode::Chat {
            include_spawn_agent: true,
        });
        let spawn = schemas
            .iter()
            .find(|s| s["function"]["name"].as_str() == Some("spawn_agent"))
            .expect("spawn_agent schema missing");
        let props = &spawn["function"]["parameters"]["properties"];
        assert!(
            props["extra_mounts"].is_object(),
            "extra_mounts property missing from spawn_agent schema"
        );
        assert_eq!(
            props["extra_mounts"]["type"].as_str(),
            Some("array"),
            "extra_mounts should be type array"
        );
        // items.required must include "path"
        let required = &props["extra_mounts"]["items"]["required"];
        assert!(
            required
                .as_array()
                .is_some_and(|r| r.iter().any(|v| v.as_str() == Some("path"))),
            "extra_mounts items must require 'path'"
        );
    }

    #[test]
    fn spawn_agent_schema_has_env() {
        let schemas = tool_schemas(ToolMode::Chat {
            include_spawn_agent: true,
        });
        let spawn = schemas
            .iter()
            .find(|s| s["function"]["name"].as_str() == Some("spawn_agent"))
            .expect("spawn_agent schema missing");
        let props = &spawn["function"]["parameters"]["properties"];
        assert!(
            props["env"].is_object(),
            "env property missing from spawn_agent schema"
        );
        assert_eq!(
            props["env"]["type"].as_str(),
            Some("object"),
            "env should be type object"
        );
        assert!(
            props["env"]["additionalProperties"].is_object(),
            "env should have additionalProperties"
        );
    }

    #[test]
    fn spawn_agent_schema_has_parallel() {
        let schemas = tool_schemas(ToolMode::Chat {
            include_spawn_agent: true,
        });
        let spawn = schemas
            .iter()
            .find(|s| s["function"]["name"].as_str() == Some("spawn_agent"))
            .expect("spawn_agent schema missing");
        let props = &spawn["function"]["parameters"]["properties"];
        assert_eq!(
            props["parallel"]["type"].as_str(),
            Some("boolean"),
            "parallel property should be type boolean in spawn_agent schema"
        );
    }

    #[test]
    fn spawn_agent_schema_has_sandbox() {
        let schemas = tool_schemas(ToolMode::Chat {
            include_spawn_agent: true,
        });
        let spawn = schemas
            .iter()
            .find(|s| s["function"]["name"].as_str() == Some("spawn_agent"))
            .expect("spawn_agent schema missing");
        let props = &spawn["function"]["parameters"]["properties"];
        assert_eq!(
            props["sandbox"]["type"].as_str(),
            Some("string"),
            "sandbox property should be type string in spawn_agent schema"
        );
        let enum_values = props["sandbox"]["enum"]
            .as_array()
            .expect("sandbox should have an enum array");
        let values: Vec<&str> = enum_values.iter().filter_map(|v| v.as_str()).collect();
        assert_eq!(
            values,
            vec!["docker", "noop"],
            "sandbox enum should exactly match the supported values"
        );
    }

    // ---- shell_command ---------------------------------------------------

    #[tokio::test]
    async fn shell_command_captures_stdout() {
        let exec = LocalExecutor::new();
        let out = exec
            .execute(
                "shell_command",
                serde_json::json!({"command": "echo hello"}),
            )
            .await
            .unwrap();
        assert_eq!(out.trim(), "hello");
    }

    #[tokio::test]
    async fn shell_command_appends_exit_code_on_failure() {
        let exec = LocalExecutor::new();
        let out = exec
            .execute(
                "shell_command",
                serde_json::json!({"command": "echo bad && exit 2"}),
            )
            .await
            .unwrap();
        assert!(out.contains("[exit 2]"), "got: {out}");
        assert!(out.contains("bad"), "stdout should be present: {out}");
    }

    #[tokio::test]
    async fn shell_command_empty_output_shows_exit_zero() {
        let exec = LocalExecutor::new();
        let out = exec
            .execute("shell_command", serde_json::json!({"command": "true"}))
            .await
            .unwrap();
        assert_eq!(out, "[exit 0]");
    }

    #[tokio::test]
    async fn shell_command_missing_arg_returns_err() {
        let exec = LocalExecutor::new();
        let result = exec.execute("shell_command", serde_json::json!({})).await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("command"),
            "error should mention 'command': {msg}"
        );
    }

    // ---- read_file -------------------------------------------------------

    #[tokio::test]
    async fn read_file_returns_content() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.txt");
        std::fs::write(&path, "file contents").unwrap();

        let exec = LocalExecutor::new();
        let out = exec
            .execute(
                "read_file",
                serde_json::json!({"path": path.to_str().unwrap()}),
            )
            .await
            .unwrap();
        assert_eq!(out, "file contents");
    }

    #[tokio::test]
    async fn read_file_nonexistent_returns_err() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("does_not_exist.txt");
        let exec = LocalExecutor::new();
        let result = exec
            .execute(
                "read_file",
                serde_json::json!({"path": path.to_str().unwrap()}),
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn read_file_missing_path_arg_returns_err() {
        let exec = LocalExecutor::new();
        let result = exec.execute("read_file", serde_json::json!({})).await;
        assert!(result.is_err());
    }

    // ---- edit_file -------------------------------------------------------

    #[tokio::test]
    async fn edit_file_writes_new_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("new.txt");

        let exec = LocalExecutor::new();
        let out = exec
            .execute(
                "edit_file",
                serde_json::json!({"path": path.to_str().unwrap(), "content": "written"}),
            )
            .await
            .unwrap();
        assert!(
            out.contains("wrote"),
            "return message should mention 'wrote': {out}"
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "written");
    }

    #[tokio::test]
    async fn edit_file_overwrites_existing() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("existing.txt");
        std::fs::write(&path, "old").unwrap();

        let exec = LocalExecutor::new();
        exec.execute(
            "edit_file",
            serde_json::json!({"path": path.to_str().unwrap(), "content": "new"}),
        )
        .await
        .unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new");
    }

    // ---- LocalExecutor::new env-var wiring ----------------------------------

    #[test]
    #[serial]
    fn new_without_env_var_has_no_sandbox() {
        std::env::remove_var("AMAEBI_SANDBOX");
        let exec = LocalExecutor::new();
        assert!(exec.sandbox.is_none());
    }

    #[test]
    #[serial]
    fn new_with_docker_env_var_creates_docker_sandbox() {
        std::env::set_var("AMAEBI_SANDBOX", "docker");
        let exec = LocalExecutor::new();
        std::env::remove_var("AMAEBI_SANDBOX");
        assert!(exec.sandbox.is_some());
        assert_eq!(exec.sandbox.as_deref().map(|s| s.name()), Some("docker"));
    }

    #[test]
    #[serial]
    fn new_with_sandbox_workspace_env_var_uses_that_path() {
        std::env::set_var("AMAEBI_SANDBOX", "docker");
        std::env::set_var("AMAEBI_SANDBOX_WORKSPACE", "/tmp/my-worktree");
        let exec = LocalExecutor::new();
        std::env::remove_var("AMAEBI_SANDBOX");
        std::env::remove_var("AMAEBI_SANDBOX_WORKSPACE");
        assert!(exec.sandbox.is_some());
        assert_eq!(
            exec.default_cwd.as_deref(),
            Some(std::path::Path::new("/tmp/my-worktree")),
            "default_cwd should be set to AMAEBI_SANDBOX_WORKSPACE"
        );
    }

    #[test]
    #[serial]
    fn new_with_unknown_env_var_value_has_no_sandbox() {
        std::env::set_var("AMAEBI_SANDBOX", "unknown");
        let exec = LocalExecutor::new();
        std::env::remove_var("AMAEBI_SANDBOX");
        assert!(exec.sandbox.is_none());
    }

    // ---- shell_command with NoopSandbox ----------------------------------

    #[tokio::test]
    async fn shell_command_with_noop_sandbox() {
        let exec = LocalExecutor {
            sandbox: Some(Box::new(crate::sandbox::NoopSandbox)),
            spawn_ctx: None,
            default_cwd: None,
        };
        let args = serde_json::json!({"command": "echo hello"});
        let result = exec.execute("shell_command", args).await.unwrap();
        assert!(result.contains("hello"));
    }

    #[tokio::test]
    async fn shell_command_noop_sandbox_uses_default_cwd() {
        let tmp = TempDir::new().unwrap();
        let exec = LocalExecutor {
            sandbox: Some(Box::new(crate::sandbox::NoopSandbox)),
            spawn_ctx: None,
            default_cwd: Some(tmp.path().to_path_buf()),
        };
        // pwd should print the default_cwd, not the daemon process cwd.
        let result = exec
            .execute("shell_command", serde_json::json!({"command": "pwd"}))
            .await
            .unwrap();
        assert!(
            result
                .trim()
                .ends_with(tmp.path().file_name().unwrap().to_str().unwrap()),
            "expected cwd under tmp, got: {result}"
        );
    }

    // ---- tool_schemas include/exclude spawn_agent -----------------------

    #[test]
    fn tool_schemas_false_excludes_spawn_agent() {
        let schemas = tool_schemas(ToolMode::Chat {
            include_spawn_agent: false,
        });
        let names: Vec<&str> = schemas
            .iter()
            .map(|s| s["function"]["name"].as_str().unwrap())
            .collect();
        assert!(
            !names.contains(&"spawn_agent"),
            "spawn_agent should be excluded when include_spawn_agent=false"
        );
        // Core tools must still be present.
        for name in ["shell_command", "read_file", "edit_file"] {
            assert!(names.contains(&name), "missing core tool: {name}");
        }
    }

    #[test]
    fn tool_schemas_true_includes_spawn_agent() {
        let schemas = tool_schemas(ToolMode::Chat {
            include_spawn_agent: true,
        });
        let names: Vec<&str> = schemas
            .iter()
            .map(|s| s["function"]["name"].as_str().unwrap())
            .collect();
        assert!(
            names.contains(&"spawn_agent"),
            "spawn_agent should be present when include_spawn_agent=true"
        );
    }

    // ---- spawn_agent workspace validation --------------------------------

    #[tokio::test]
    #[serial]
    async fn spawn_agent_rejects_relative_workspace() {
        let ctx = make_spawn_ctx();
        let result = spawn_agent(
            serde_json::json!({"task": "t", "workspace": "relative/path"}),
            &ctx,
        )
        .await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("absolute"), "got: {msg}");
    }

    #[tokio::test]
    #[serial]
    async fn spawn_agent_rejects_nonexistent_workspace() {
        let ctx = make_spawn_ctx();
        let result = spawn_agent(
            serde_json::json!({"task": "t", "workspace": "/tmp/amaebi_test_nonexistent_xyz"}),
            &ctx,
        )
        .await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("does not exist"), "got: {msg}");
    }

    #[tokio::test]
    #[serial]
    async fn spawn_agent_rejects_workspace_that_is_a_file() {
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("notadir.txt");
        std::fs::write(&file, "x").unwrap();
        let ctx = make_spawn_ctx();
        let result = spawn_agent(
            serde_json::json!({"task": "t", "workspace": file.to_str().unwrap()}),
            &ctx,
        )
        .await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("not a directory"), "got: {msg}");
    }

    /// Helper: build a minimal SpawnContext suitable for unit tests.
    fn make_spawn_ctx() -> SpawnContext {
        SpawnContext {
            http: reqwest::Client::new(),
            db: Arc::new(Mutex::new(rusqlite::Connection::open_in_memory().unwrap())),
            compacting_sessions: Arc::new(Mutex::new(HashSet::new())),
            tokens: Arc::new(crate::auth::TokenCache::new()),
            user_aliases: Arc::new(HashMap::new()),
        }
    }

    // ---- subagent_default_model -----------------------------------------

    #[test]
    #[serial_test::serial]
    fn subagent_default_model_uses_subagent_env_verbatim() {
        std::env::set_var("AMAEBI_MODEL", "copilot/claude-opus-4-6");
        std::env::set_var("AMAEBI_SUBAGENT_MODEL", "bedrock/claude-haiku-4.5");
        let result = subagent_default_model();
        std::env::remove_var("AMAEBI_MODEL");
        std::env::remove_var("AMAEBI_SUBAGENT_MODEL");
        // AMAEBI_SUBAGENT_MODEL wins over AMAEBI_MODEL.
        assert_eq!(result, "bedrock/claude-haiku-4.5");
    }

    #[test]
    #[serial_test::serial]
    fn subagent_default_model_does_not_inherit_amaebi_model() {
        std::env::set_var("AMAEBI_MODEL", "copilot/claude-opus-4-6");
        std::env::remove_var("AMAEBI_SUBAGENT_MODEL");
        let result = subagent_default_model();
        std::env::remove_var("AMAEBI_MODEL");
        // Must NOT be the parent model — just the copilot prefix + the
        // Copilot-safe default (no `[1m]` suffix).
        assert_ne!(result, "copilot/claude-opus-4-6");
        assert_eq!(
            result,
            format!("copilot/{}", crate::provider::DEFAULT_MODEL_BARE)
        );
    }

    #[test]
    #[serial_test::serial]
    fn subagent_default_model_preserves_copilot_prefix() {
        std::env::set_var("AMAEBI_MODEL", "copilot/gpt-4o");
        std::env::remove_var("AMAEBI_SUBAGENT_MODEL");
        let result = subagent_default_model();
        std::env::remove_var("AMAEBI_MODEL");
        assert_eq!(
            result,
            format!("copilot/{}", crate::provider::DEFAULT_MODEL_BARE)
        );
    }

    #[test]
    #[serial_test::serial]
    fn subagent_default_model_no_prefix_falls_back_to_default() {
        std::env::remove_var("AMAEBI_MODEL");
        std::env::remove_var("AMAEBI_SUBAGENT_MODEL");
        let result = subagent_default_model();
        assert_eq!(result, crate::provider::DEFAULT_MODEL);
    }

    // ---- unknown tool ---------------------------------------------------

    #[tokio::test]
    async fn unknown_tool_returns_descriptive_error() {
        let exec = LocalExecutor::new();
        let result = exec
            .execute("nonexistent_tool", serde_json::json!({}))
            .await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("unknown tool"), "got: {msg}");
    }

    // ---- wait_for_file --------------------------------------------------

    #[tokio::test]
    async fn wait_for_file_returns_found_immediately() {
        let tmp = TempDir::new().unwrap();
        let sentinel = tmp.path().join("done.txt");
        std::fs::write(&sentinel, "").unwrap();
        let result = wait_for_file(serde_json::json!({
            "path": sentinel.to_str().unwrap(),
            "timeout_secs": 5
        }))
        .await
        .unwrap();
        assert_eq!(result, "found");
    }

    #[tokio::test]
    async fn wait_for_file_times_out_when_file_absent() {
        let tmp = TempDir::new().unwrap();
        let sentinel = tmp.path().join("never.txt");
        let result = wait_for_file(serde_json::json!({
            "path": sentinel.to_str().unwrap(),
            "timeout_secs": 0,
            "poll_interval_ms": 10
        }))
        .await
        .unwrap();
        assert!(result.starts_with("timeout:"), "got: {result}");
    }

    // ---- wait_for_task_event -------------------------------------------

    #[tokio::test]
    async fn wait_for_task_event_returns_event_payload_and_log_tail() {
        let tmp = TempDir::new().unwrap();
        let passed = tmp.path().join("passed.event");
        let failed = tmp.path().join("failed.event");
        let log = tmp.path().join("task.log");
        std::fs::write(&passed, "exit_code=0\n").unwrap();
        std::fs::write(&log, "line 1\nline 2\nline 3\n").unwrap();

        let result = wait_for_task_event(serde_json::json!({
            "events": [
                {"name": "failed", "path": failed.to_str().unwrap()},
                {"name": "passed", "path": passed.to_str().unwrap()}
            ],
            "log_path": log.to_str().unwrap(),
            "tail_lines": 2,
            "timeout_secs": 5
        }))
        .await
        .unwrap();

        assert!(result.contains("event: passed"), "got: {result}");
        assert!(result.contains("exit_code=0"), "got: {result}");
        assert!(result.contains("log_path:"), "got: {result}");
        assert!(!result.contains("line 1"), "got: {result}");
        assert!(result.contains("line 2\nline 3"), "got: {result}");
    }

    #[tokio::test]
    async fn wait_for_task_event_times_out_with_watched_events() {
        let tmp = TempDir::new().unwrap();
        let passed = tmp.path().join("passed.event");
        let result = wait_for_task_event(serde_json::json!({
            "events": [
                {"name": "passed", "path": passed.to_str().unwrap()}
            ],
            "timeout_secs": 0,
            "poll_interval_ms": 10
        }))
        .await
        .unwrap();

        assert!(
            result.starts_with("timeout: no task event appeared within 0s"),
            "got: {result}"
        );
        assert!(result.contains("watched_events:"), "got: {result}");
        assert!(result.contains("passed:"), "got: {result}");
    }

    #[tokio::test]
    async fn wait_for_task_event_rejects_empty_event_list() {
        let err = wait_for_task_event(serde_json::json!({
            "events": []
        }))
        .await
        .expect_err("empty event list should be rejected")
        .to_string();

        assert!(
            err.contains("'events' must not be empty"),
            "unexpected error: {err}"
        );
    }

    // ---- tmux_wait normalize_for_idle_check --------------------------------

    #[test]
    fn normalize_collapses_elapsed_timer_runs() {
        let a = "✻ Comparing… (4m 35s · ↓ 5.8k tokens)";
        let b = "✻ Comparing… (4m 36s · ↓ 5.8k tokens)";
        assert_eq!(
            normalize_for_idle_check(a),
            normalize_for_idle_check(b),
            "ticking elapsed-time counter should not register as activity"
        );
    }

    #[test]
    fn normalize_collapses_spinner_glyph() {
        let a = "✶ thinking…";
        let b = "✻ thinking…";
        let c = "✷ thinking…";
        assert_eq!(normalize_for_idle_check(a), normalize_for_idle_check(b));
        assert_eq!(normalize_for_idle_check(b), normalize_for_idle_check(c));
    }

    #[test]
    fn normalize_collapses_running_timer() {
        let a = "  ⎿  Running… (1m 1s)";
        let b = "  ⎿  Running… (1m 6s)";
        assert_eq!(normalize_for_idle_check(a), normalize_for_idle_check(b));
    }

    #[test]
    fn normalize_preserves_real_text_changes() {
        let a = "● Bash(echo hello)";
        let b = "● Bash(echo world)";
        assert_ne!(
            normalize_for_idle_check(a),
            normalize_for_idle_check(b),
            "non-numeric, non-spinner text changes must still differ"
        );
    }

    #[test]
    fn normalize_collapses_thinking_verb_rotation() {
        // Claude Code rotates the activity verb every few seconds while
        // the model is thinking — this is the hot bug that left
        // tmux_wait blocked for 90+ minutes against an idle pane.
        let a = "✻ Baked for 1h 28m 15s · 19 shells still running";
        let b = "✻ Cogitated for 1h 28m 17s · 19 shells still running";
        let c = "✷ Worked for 1h 28m 21s · 19 shells still running";
        assert_eq!(normalize_for_idle_check(a), normalize_for_idle_check(b));
        assert_eq!(normalize_for_idle_check(b), normalize_for_idle_check(c));
    }

    #[test]
    fn normalize_collapses_thinking_with_token_counter() {
        // The thinking line sometimes includes a token counter and an
        // effort tag — both volatile.  Same collapse should apply.
        let a = "✻ Mapping for 2m 4s · 1.2k tokens · with xhigh effort";
        let b = "✻ Calculating for 2m 11s · 1.5k tokens · with xhigh effort";
        assert_eq!(normalize_for_idle_check(a), normalize_for_idle_check(b));
    }

    #[test]
    fn normalize_thinking_line_distinct_from_other_content() {
        // Collapsing the thinking line must NOT make every line
        // collapse to the same token — real new content (spinner-led
        // tool-call lines, plain output) should still differ.
        let thinking = "✻ Baked for 1h 28m 15s · 19 shells still running";
        let other = "● Bash(echo hello)";
        assert_ne!(
            normalize_for_idle_check(thinking),
            normalize_for_idle_check(other)
        );
    }

    #[test]
    fn normalize_treats_new_lines_as_activity() {
        let a = "● step 1";
        let b = "● step 1\n● step 2";
        assert_ne!(normalize_for_idle_check(a), normalize_for_idle_check(b));
    }

    // ---- tmux_send_text / tmux_send_key split ------------------------------

    #[test]
    fn tmux_send_text_schema_exists_and_takes_text() {
        let schemas = tool_schemas(ToolMode::Chat {
            include_spawn_agent: true,
        });
        let schema = schemas
            .iter()
            .find(|s| s["function"]["name"] == "tmux_send_text")
            .expect("tmux_send_text schema must exist");
        let props = &schema["function"]["parameters"]["properties"];
        assert!(
            props.get("text").is_some(),
            "tmux_send_text must expose 'text'"
        );
        assert!(
            props.get("enter").is_none(),
            "tmux_send_text must NOT expose 'enter' (split tool replaces mode flag)"
        );
    }

    #[test]
    fn tmux_send_key_schema_exists_and_takes_key() {
        let schemas = tool_schemas(ToolMode::Chat {
            include_spawn_agent: true,
        });
        let schema = schemas
            .iter()
            .find(|s| s["function"]["name"] == "tmux_send_key")
            .expect("tmux_send_key schema must exist");
        let props = &schema["function"]["parameters"]["properties"];
        assert!(
            props.get("key").is_some(),
            "tmux_send_key must expose 'key'"
        );
    }

    #[test]
    fn tmux_send_keys_schema_removed() {
        let schemas = tool_schemas(ToolMode::Chat {
            include_spawn_agent: true,
        });
        assert!(
            schemas
                .iter()
                .all(|s| s["function"]["name"] != "tmux_send_keys"),
            "legacy tmux_send_keys schema must be removed"
        );
    }

    #[test]
    fn task_done_schema_requires_downstream_validation_evidence() {
        let schemas = tool_schemas(ToolMode::Chat {
            include_spawn_agent: true,
        });
        let schema = schemas
            .iter()
            .find(|s| s["function"]["name"] == "task_done")
            .expect("task_done schema must exist");
        let props = schema["function"]["parameters"]["properties"]
            .as_object()
            .expect("task_done properties must be an object");
        assert!(
            props.contains_key("validation_evidence"),
            "task_done must expose validation_evidence property: {schema}"
        );
        assert_eq!(
            props["validation_evidence"]["type"], "string",
            "validation_evidence must be a string property: {schema}"
        );
        let required = schema["function"]["parameters"]["required"]
            .as_array()
            .expect("task_done required fields must be an array");
        assert!(
            required.iter().any(|v| v == "validation_evidence"),
            "task_done must require validation_evidence: {schema}"
        );
    }

    #[test]
    fn task_done_rejects_empty_pane_id() {
        let err = task_done(serde_json::json!({
            "pane_id": "   ",
            "summary": "implemented",
            "validation_evidence": "cargo test passed"
        }))
        .expect_err("task_done should reject empty pane_id")
        .to_string();
        assert!(
            err.contains("pane_id") && err.contains("non-empty"),
            "error should mention non-empty pane_id: {err}"
        );
    }

    #[test]
    fn task_done_rejects_empty_summary() {
        let err = task_done(serde_json::json!({
            "pane_id": "%1",
            "summary": "   ",
            "validation_evidence": "cargo test passed"
        }))
        .expect_err("task_done should reject empty summary")
        .to_string();
        assert!(
            err.contains("summary") && err.contains("non-empty"),
            "error should mention non-empty summary: {err}"
        );
    }

    #[test]
    fn task_done_rejects_missing_validation_evidence() {
        let err = task_done(serde_json::json!({
            "pane_id": "%1",
            "summary": "implemented"
        }))
        .expect_err("task_done should require validation evidence")
        .to_string();
        assert!(
            err.contains("validation_evidence"),
            "error should mention validation_evidence: {err}"
        );

        let err = task_done(serde_json::json!({
            "pane_id": "%1",
            "summary": "implemented",
            "validation_evidence": "   "
        }))
        .expect_err("task_done should reject empty validation evidence")
        .to_string();
        assert!(
            err.contains("non-empty"),
            "error should mention non-empty evidence: {err}"
        );
    }

    #[test]
    fn task_done_rejects_build_only_validation_evidence() {
        let err = task_done(serde_json::json!({
            "pane_id": "%11",
            "summary": "implemented and pushed",
            "validation_evidence": "\
Build validation on sim-9900 container:
- Command: `cmake --build build --target xe4_fmha4_paged_fwd -j$(nproc)`
- Result: Build completed successfully

Script syntax validation:
- Command: `bash -n examples/xe4/fmha4_paged/run_tests.sh`
- Result: No output

Whitespace check:
- Command: `git diff --check`
- Result: No output

Push verification:
- Command: `git push -u origin HEAD`
- Result: Successfully pushed"
        }))
        .expect_err("task_done should reject build/syntax/diff/push-only evidence")
        .to_string();
        assert!(
            err.contains("build/syntax/diff/push"),
            "error should explain insufficient evidence: {err}"
        );
    }

    #[test]
    fn task_done_rejects_evidence_without_validation_command_and_result() {
        let err = task_done(serde_json::json!({
            "pane_id": "%11",
            "summary": "implemented",
            "validation_evidence": "Reviewed the diff and the implementation looks correct."
        }))
        .expect_err("task_done should reject generic evidence")
        .to_string();
        assert!(
            err.contains("validation command") && err.contains("passing result"),
            "error should require command and result: {err}"
        );
    }

    #[test]
    fn task_done_rejects_narrative_performance_result_without_command() {
        let err = task_done(serde_json::json!({
            "pane_id": "%11",
            "summary": "implemented",
            "validation_evidence": "Performance within tolerance.\nResult: passed"
        }))
        .expect_err("task_done should reject narrative result without command evidence")
        .to_string();
        assert!(
            err.contains("validation command") && err.contains("passing result"),
            "error should require command and result: {err}"
        );
    }

    #[test]
    fn task_done_rejects_dash_dash_narrative_performance_result_without_command() {
        let err = task_done(serde_json::json!({
            "pane_id": "%11",
            "summary": "implemented",
            "validation_evidence": "Performance within tolerance -- passed"
        }))
        .expect_err("task_done should reject narrative dash-dash result without command evidence")
        .to_string();
        assert!(
            err.contains("validation command") && err.contains("passing result"),
            "error should require command and result: {err}"
        );
    }

    #[test]
    fn task_done_rejects_test_command_without_passing_result() {
        let err = task_done(serde_json::json!({
            "pane_id": "%11",
            "summary": "implemented",
            "validation_evidence": "Command: cargo test task_done"
        }))
        .expect_err("task_done should reject command-only evidence")
        .to_string();
        assert!(
            err.contains("validation command") && err.contains("passing result"),
            "error should require command and result: {err}"
        );
    }

    #[test]
    fn task_done_rejects_combined_build_and_test_without_passing_result() {
        let err = task_done(serde_json::json!({
            "pane_id": "%11",
            "summary": "implemented",
            "validation_evidence": "Command: cargo build && cargo test"
        }))
        .expect_err("task_done should reject combined command without result")
        .to_string();
        assert!(
            err.contains("validation command") && err.contains("passing result"),
            "error should require command and result: {err}"
        );
        assert!(
            !err.contains("only shows build/syntax/diff/push"),
            "error should not misclassify a validation command as build-only evidence: {err}"
        );
    }

    #[test]
    fn task_done_rejects_bash_syntax_check_of_test_script_as_insufficient() {
        let err = task_done(serde_json::json!({
            "pane_id": "%11",
            "summary": "implemented",
            "validation_evidence": "Command: bash -n scripts/test.sh\nResult: exit code 0"
        }))
        .expect_err("task_done should reject syntax-only script checks")
        .to_string();
        assert!(
            err.contains("build/syntax/diff/push"),
            "error should explain syntax-only evidence is insufficient: {err}"
        );
    }

    #[test]
    fn task_done_accepts_chained_syntax_check_and_test_command() {
        task_done(serde_json::json!({
            "pane_id": "%11",
            "summary": "implemented",
            "validation_evidence": "Command: bash -n scripts/test.sh && cargo test\nResult: passed"
        }))
        .expect("syntax check chained with test command and passing result should be valid");
    }

    #[test]
    fn task_done_rejects_negative_passed_result() {
        let err = task_done(serde_json::json!({
            "pane_id": "%11",
            "summary": "implemented",
            "validation_evidence": "Command: cargo test\nResult: not passed"
        }))
        .expect_err("task_done should reject negative passing result")
        .to_string();
        assert!(
            err.contains("validation command") && err.contains("passing result"),
            "error should require a passing result: {err}"
        );
    }

    #[test]
    fn task_done_rejects_unstructured_build_passed_result() {
        let err = task_done(serde_json::json!({
            "pane_id": "%11",
            "summary": "implemented",
            "validation_evidence": "Command: cargo test\nbuild passed"
        }))
        .expect_err("task_done should reject unstructured build-passed summaries")
        .to_string();
        assert!(
            err.contains("validation command") && err.contains("passing result"),
            "error should require a passing result: {err}"
        );
    }

    #[test]
    fn task_done_rejects_self_report_that_tests_were_not_run() {
        let err = task_done(serde_json::json!({
            "pane_id": "%11",
            "summary": "implemented",
            "validation_evidence": "I did not run the functional tests; build passed."
        }))
        .expect_err("task_done should reject self-reported missing validation")
        .to_string();
        assert!(
            err.contains("required validation is missing"),
            "error should explain missing validation: {err}"
        );
    }

    #[test]
    fn task_done_rejects_test_change_only_evidence_with_accurate_message() {
        let err = task_done(serde_json::json!({
            "pane_id": "%11",
            "summary": "implemented",
            "validation_evidence": "Added test coverage for task_done validation."
        }))
        .expect_err("task_done should reject test-change-only evidence")
        .to_string();
        assert!(
            err.contains("test script/list/change-only"),
            "error should mention test-change-only evidence: {err}"
        );
    }

    #[test]
    fn task_done_accepts_project_test_command_with_result() {
        task_done(serde_json::json!({
            "pane_id": "%11",
            "summary": "implemented",
            "validation_evidence": "Command: ./scripts/test.sh\nResult: passed, exit code 0"
        }))
        .expect("project test command and passing result should be valid evidence");
    }

    #[test]
    fn task_done_accepts_combined_build_and_test_command_with_result() {
        task_done(serde_json::json!({
            "pane_id": "%11",
            "summary": "implemented",
            "validation_evidence": "Command: cargo build && cargo test\nResult: passed, 0 failed"
        }))
        .expect("combined build plus test command should count as validation evidence");
    }

    #[test]
    fn task_done_accepts_go_test_ok_output() {
        task_done(serde_json::json!({
            "pane_id": "%11",
            "summary": "implemented",
            "validation_evidence": "Command: go test ./...\nok example.com/project/pkg 0.012s"
        }))
        .expect("go test command and ok output should be valid evidence");
    }

    #[test]
    fn task_done_accepts_go_test_pass_output() {
        task_done(serde_json::json!({
            "pane_id": "%11",
            "summary": "implemented",
            "validation_evidence": "Command: go test ./pkg\nPASS"
        }))
        .expect("go test command and PASS output should be valid evidence");
    }

    #[test]
    fn task_done_accepts_commandish_benchmark_evidence_with_result() {
        task_done(serde_json::json!({
            "pane_id": "%11",
            "summary": "implemented",
            "validation_evidence": "Command: benchmark regression suite\nResult: passed, no regression"
        }))
        .expect("commandish benchmark evidence and passing result should be valid evidence");
    }

    #[test]
    fn task_done_accepts_commandish_run_tests_without_args() {
        task_done(serde_json::json!({
            "pane_id": "%11",
            "summary": "implemented",
            "validation_evidence": "Command: run_tests.sh\nResult: passed"
        }))
        .expect("commandish run_tests.sh evidence and passing result should be valid evidence");
    }

    #[test]
    fn task_done_rejects_run_tests_script_change_without_command() {
        let err = task_done(serde_json::json!({
            "pane_id": "%11",
            "summary": "implemented",
            "validation_evidence": "Updated run_tests.sh coverage.\nResult: passed"
        }))
        .expect_err("task_done should reject run_tests.sh mentions without command evidence")
        .to_string();
        assert!(
            err.contains("validation command") && err.contains("passing result"),
            "error should require command and result: {err}"
        );
    }

    #[test]
    fn task_done_accepts_fmha4_simulator_functional_evidence() {
        task_done(serde_json::json!({
            "pane_id": "%11",
            "summary": "implemented",
            "validation_evidence": "\
Command: examples/xe4/fmha4_paged/run_tests.sh --filter=LEFTPAD_K --port=9900
Result: Disposition: Passed
Numeric check: Cosine=1, Mismatch=0/4194304"
        }))
        .expect("simulator command and numeric pass result should be valid evidence");
    }

    #[tokio::test]
    async fn wait_for_file_does_not_match_directory() {
        // If a directory exists at the sentinel path it should NOT be treated as
        // "found" — wait_for_file expects a regular file.
        let tmp = TempDir::new().unwrap();
        let dir_path = tmp.path().join("subdir");
        std::fs::create_dir(&dir_path).unwrap();
        // With a zero timeout the call must time out rather than return "found".
        let result = wait_for_file(serde_json::json!({
            "path": dir_path.to_str().unwrap(),
            "timeout_secs": 0,
            "poll_interval_ms": 10
        }))
        .await
        .unwrap();
        assert!(
            result.starts_with("timeout:"),
            "directory must not match: {result}"
        );
    }

    // ---- switch_model schema -----------------------------------------------

    #[test]
    fn switch_model_schema_present_in_all_modes() {
        // switch_model must always be available, regardless of include_spawn_agent.
        for include in [true, false] {
            let schemas = tool_schemas(ToolMode::Chat {
                include_spawn_agent: include,
            });
            let names: Vec<&str> = schemas
                .iter()
                .map(|s| s["function"]["name"].as_str().unwrap())
                .collect();
            assert!(
                names.contains(&"switch_model"),
                "switch_model must be present when include_spawn_agent={include}"
            );
        }
    }

    #[test]
    fn switch_model_schema_has_required_model_param() {
        let schemas = tool_schemas(ToolMode::Chat {
            include_spawn_agent: true,
        });
        let schema = schemas
            .iter()
            .find(|s| s["function"]["name"] == "switch_model")
            .expect("switch_model schema must exist");
        let required = schema["function"]["parameters"]["required"]
            .as_array()
            .expect("required must be an array");
        let required_names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(
            required_names.contains(&"model"),
            "switch_model must require 'model': {required_names:?}"
        );
    }

    // ---- emit_distilled_prompt + ToolMode::Distill schema gating -------------

    #[test]
    fn distill_mode_includes_emit_distilled_prompt_and_excludes_writes() {
        let schemas = tool_schemas(ToolMode::Distill);
        let names: Vec<&str> = schemas
            .iter()
            .map(|s| s["function"]["name"].as_str().unwrap())
            .collect();
        assert!(
            names.contains(&"emit_distilled_prompt"),
            "Distill mode must expose emit_distilled_prompt: {names:?}"
        );
        // Investigation tools allowed.
        assert!(names.contains(&"shell_command"));
        assert!(names.contains(&"read_file"));
        // Mutation tools must NOT be present.
        for forbidden in ["edit_file", "task_done", "spawn_agent", "tmux_send_text"] {
            assert!(
                !names.contains(&forbidden),
                "Distill mode must NOT expose {forbidden}: {names:?}"
            );
        }
    }

    #[test]
    fn chat_mode_excludes_emit_distilled_prompt() {
        // The terminator should ONLY appear in Distill mode — exposing it
        // in Chat mode would let a regular agentic loop short-circuit
        // itself by calling a tool meant for distillation.
        for include in [true, false] {
            let schemas = tool_schemas(ToolMode::Chat {
                include_spawn_agent: include,
            });
            let names: Vec<&str> = schemas
                .iter()
                .map(|s| s["function"]["name"].as_str().unwrap())
                .collect();
            assert!(
                !names.contains(&"emit_distilled_prompt"),
                "Chat mode must NOT expose emit_distilled_prompt (include_spawn_agent={include}): {names:?}"
            );
        }
    }

    #[test]
    fn emit_distilled_prompt_validates_prompt_field() {
        // Missing prompt → error
        let r = emit_distilled_prompt(serde_json::json!({}));
        assert!(r.is_err(), "missing prompt should error");
        // Empty prompt → error (we don't want a no-op result)
        let r = emit_distilled_prompt(serde_json::json!({"prompt": "   "}));
        assert!(r.is_err(), "whitespace-only prompt should error");
        // Valid prompt → echo with length
        let r = emit_distilled_prompt(serde_json::json!({"prompt": "hello world"}));
        assert!(r.is_ok());
        let s = r.unwrap();
        assert!(s.contains("len=11"), "result should report length: {s}");

        // Exactly the hard line limit is allowed.
        let max_line_prompt = (1..=MAX_DISTILLED_PROMPT_LINES)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let r = emit_distilled_prompt(serde_json::json!({"prompt": max_line_prompt}));
        assert!(r.is_ok(), "prompt at line limit should be accepted");

        // One line over the hard limit is rejected so the downstream paste stays bounded.
        let too_long_prompt = (1..=(MAX_DISTILLED_PROMPT_LINES + 1))
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let r = emit_distilled_prompt(serde_json::json!({"prompt": too_long_prompt}));
        assert!(r.is_err(), "prompt over line limit should error");
    }
}
