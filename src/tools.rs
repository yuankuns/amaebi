use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use tokio::process::Command;

use crate::sandbox::{docker::DockerSandboxConfig, DockerSandbox, NoopSandbox, Sandbox};

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
            "tmux_send_keys" => tmux_send_keys(args).await,
            "tmux_wait" => tmux_wait(args).await,
            "wait_for_file" => wait_for_file(args).await,
            "read_file" => read_file(args).await,
            "edit_file" => edit_file(args).await,
            "spawn_agent" => match &self.spawn_ctx {
                Some(ctx) => spawn_agent(args, ctx).await,
                None => anyhow::bail!(
                    "spawn_agent is not available in this context \
                     (child agents cannot spawn further agents)"
                ),
            },
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

/// Send keystrokes to a tmux pane (for interactive programs).
async fn tmux_send_keys(args: serde_json::Value) -> Result<String> {
    let keys = args["keys"]
        .as_str()
        .context("tmux_send_keys: missing string argument 'keys'")?;
    let target = args["target"].as_str().unwrap_or("%0");

    let output = Command::new("tmux")
        .args(["send-keys", "-t", target, keys])
        .output()
        .await
        .context("spawning tmux send-keys")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("tmux send-keys failed: {stderr}");
    }

    Ok(format!("sent keys to pane {target}"))
}

/// Poll a tmux pane until its output has been stable for `idle_secs`, then
/// return the final pane content.
///
/// Instead of the LLM calling `tmux_capture_pane` in a loop (burning one LLM
/// turn per poll), a single `tmux_wait` call blocks until the command running
/// in the pane appears to have finished.
///
/// The function observes `idle_secs` of unchanged output before returning, so
/// it always waits at least `idle_secs` regardless of the initial pane state.
/// On timeout an error is returned so callers can distinguish it from a real
/// pane output.
async fn tmux_wait(args: serde_json::Value) -> Result<String> {
    let target = args["target"].as_str().unwrap_or("%0");
    let idle_secs = args["idle_secs"].as_u64().unwrap_or(3);
    let timeout_secs = args["timeout_secs"].as_u64().unwrap_or(600).min(86_400);
    // Clamp poll interval to at least 1 s to avoid busy-polling tmux.
    let poll_secs = args["poll_interval_secs"].as_u64().unwrap_or(2).max(1);

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    let mut last_content = String::new();
    let mut stable_since = tokio::time::Instant::now();

    loop {
        // Check the hard deadline at the top of every iteration so we never
        // start a new capture call after time has already expired.
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!(
                "tmux_wait: timed out after {timeout_secs}s waiting for pane '{target}' to become idle"
            );
        }

        // Wrap the capture call with timeout_at so a hung tmux process cannot
        // block past the deadline.
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

        if content != last_content {
            last_content = content;
            stable_since = tokio::time::Instant::now();
        } else if stable_since.elapsed().as_secs() >= idle_secs {
            return Ok(last_content);
        }

        // Sleep at most until the deadline to keep the timeout accurate.
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let sleep_dur = std::time::Duration::from_secs(poll_secs).min(remaining);
        tokio::time::sleep(sleep_dur).await;
    }
}

/// Block until `path` exists on the filesystem, then return `"found"`.
///
/// Useful for scripts that drop a sentinel file on completion, avoiding the
/// need for the LLM to call `tmux_capture_pane` in a polling loop.
async fn wait_for_file(args: serde_json::Value) -> Result<String> {
    let path = args["path"]
        .as_str()
        .context("wait_for_file: missing string argument 'path'")?;
    // Cap timeout at 24 h to avoid Instant overflow with very large values.
    let timeout_secs = args["timeout_secs"].as_u64().unwrap_or(300).min(86_400);
    let poll_ms = args["poll_interval_ms"].as_u64().unwrap_or(500);
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    loop {
        match tokio::fs::metadata(path).await {
            Ok(m) if m.is_file() => return Ok("found".to_owned()),
            // A directory at the sentinel path is not the expected file — keep polling.
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
///   2. Provider prefix from `AMAEBI_MODEL` + `DEFAULT_MODEL`
///   3. Bare `DEFAULT_MODEL`
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
        Some(p) => format!("{}/{}", p, crate::provider::DEFAULT_MODEL),
        None => crate::provider::DEFAULT_MODEL.to_string(),
    }
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

/// Return the JSON schema array to include in a chat request.
///
/// Pass `include_spawn_agent = true` for the parent (daemon) context.
/// Pass `false` for child agent loops to prevent recursive spawning.
pub fn tool_schemas(include_spawn_agent: bool) -> Vec<serde_json::Value> {
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
                "name": "tmux_send_keys",
                "description": "Send keystrokes to a tmux pane. Use for interactive \
                                programs (e.g. pressing Enter, Ctrl-C). For background \
                                tasks prefer shell_command.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keys": {
                            "type": "string",
                            "description": "Keys to send, e.g. 'q', 'Enter', 'C-c'."
                        },
                        "target": {
                            "type": "string",
                            "description": "tmux target pane. Defaults to %0."
                        }
                    },
                    "required": ["keys"]
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
                                (e.g. a build) to finish.",
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
    ];
    if include_spawn_agent {
        schemas.push(serde_json::json!({
            "type": "function",
            "function": {
                "name": "spawn_agent",
                "description": "Spawn a child agent session to complete a task. \
                                By default the child runs in an isolated Docker sandbox \
                                (--network none). Set sandbox to 'noop' for host-direct \
                                execution (needed for tasks requiring network or host toolchain).",
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
                            "description": "If true, this call may run concurrently with other spawn_agent calls \
                                            in the same batch (default: false)."
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
                             Append [1m] to request 1M-context Bedrock inference \
                             (e.g. claude-sonnet-4.6[1m], claude-opus-4.6[1m]). \
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
        let schemas = tool_schemas(true);
        let names: Vec<&str> = schemas
            .iter()
            .map(|s| s["function"]["name"].as_str().unwrap())
            .collect();
        for name in [
            "shell_command",
            "tmux_capture_pane",
            "tmux_send_keys",
            "tmux_wait",
            "wait_for_file",
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
        for schema in tool_schemas(true) {
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
        for schema in tool_schemas(true) {
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
        let schemas = tool_schemas(true);
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
        let schemas = tool_schemas(true);
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
        let schemas = tool_schemas(true);
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
        let schemas = tool_schemas(true);
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
        let schemas = tool_schemas(false);
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
        let schemas = tool_schemas(true);
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
        // Must NOT be the parent model — just the prefix + DEFAULT_MODEL.
        assert_ne!(result, "copilot/claude-opus-4-6");
        assert_eq!(
            result,
            format!("copilot/{}", crate::provider::DEFAULT_MODEL)
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
            format!("copilot/{}", crate::provider::DEFAULT_MODEL)
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
            let schemas = tool_schemas(include);
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
        let schemas = tool_schemas(true);
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
}
