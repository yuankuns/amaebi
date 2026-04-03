pub mod builtins;
/// Workflow engine: static skill execution guaranteeing every step runs.
///
/// Design principles:
/// - Flow control is 100% code (no LLM decides what happens next)
/// - LLM is invoked only for content: writing code, analysis, summaries
/// - Shell commands are executed directly (exit codes, not LLM judgement)
/// - Resources (GPU, etc.) are semaphore-guarded — declared per stage
pub mod executor;

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Semaphore;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A complete workflow: a named sequence of stages.
#[derive(Clone, Debug)]
pub struct Workflow {
    pub name: String,
    pub stages: Vec<Stage>,
}

/// One step in the workflow.
#[derive(Clone, Debug)]
pub struct Stage {
    pub name: String,
    pub action: Action,
    pub check: Option<Check>,
    pub on_fail: FailStrategy,
    /// Named resource this stage must hold while running (semaphore-guarded).
    /// None = no resource constraint.
    pub requires: Option<String>,
}

impl Stage {
    pub fn new(name: impl Into<String>, action: Action) -> Self {
        Self {
            name: name.into(),
            action,
            check: None,
            on_fail: FailStrategy::Abort,
            requires: None,
        }
    }

    pub fn with_check(mut self, check: Check) -> Self {
        self.check = Some(check);
        self
    }

    pub fn with_on_fail(mut self, on_fail: FailStrategy) -> Self {
        self.on_fail = on_fail;
        self
    }

    pub fn with_requires(mut self, resource: impl Into<String>) -> Self {
        self.requires = Some(resource.into());
        self
    }
}

/// What a stage does.
#[derive(Clone, Debug)]
pub enum Action {
    /// Ask the LLM to do something. The prompt may contain `{var}` placeholders
    /// resolved from the current `Context`.
    Llm { prompt: String },

    /// Run a shell command. May contain `{var}` placeholders.
    Shell { command: String },

    /// Delegate a task to an external Claude Code REPL running in a tmux pane.
    /// The workflow sends the prompt via tmux, waits for Claude to finish
    /// (detected by asking the internal LLM if the pane is idle), then captures
    /// the pane output.  The delegate pane is auto-detected at workflow start.
    Delegate { prompt: String },

    /// Parse the previous LLM output into a list, then run sub-stages on each
    /// item (sequentially or in parallel).
    Map {
        /// Regex with one capture group to extract items from the prior LLM text.
        /// e.g. `r"- OPT: (.+)"` or `r"(?m)^\d+\. (.+)$"`
        parse: String,
        /// Stages to execute for each item. Use `{item}` in prompts/commands.
        stages: Vec<Stage>,
        /// false = sequential (items depend on each other, e.g. perf_sweep).
        /// true  = parallel (items are independent, e.g. bug_fix, tune_sweep).
        parallel: bool,
        /// Metadata hint: documents which named resource the sub-stages
        /// reference for concurrency control.  **Not enforced at the Map
        /// level** — actual semaphore acquisition happens inside each
        /// sub-stage via `Stage::requires` / `with_requires()`.
        /// e.g. Some("gpu") indicates sub-stages hold the "gpu" resource.
        resource_hint: Option<String>,
    },
}

/// How to decide whether a stage succeeded.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum Check {
    /// Shell command must exit 0.
    ExitCode { command: String },
    /// Shell command stdout must contain `pattern`.
    Contains { command: String, pattern: String },
    /// Shell command must output JSON; compare numeric fields against a saved
    /// baseline and fail if any field regresses by more than `threshold` (0–1).
    BenchmarkNoRegression { command: String, threshold: f64 },
}

/// What to do when a stage's Check fails (or the action itself errors).
#[derive(Clone, Debug)]
pub enum FailStrategy {
    /// Terminate the whole workflow.
    Abort,

    /// Re-run the most recent `Llm` stage with an injected error message.
    /// `{stderr}`, `{stdout}`, `{error}` are substituted from the failed run.
    Retry { max: usize, inject_prompt: String },

    /// Skip this item (only valid inside a `Map`).
    Skip,

    /// `git checkout` back to the pre-stage commit, then skip (for Map items
    /// that modify the repo and regress metrics).
    RevertAndSkip,
}

// ---------------------------------------------------------------------------
// Resource pool
// ---------------------------------------------------------------------------

/// Shared semaphore pool for resource-constrained stages.
/// Each named resource maps to a semaphore with a fixed number of permits.
///
/// Example: `ResourcePool::new([("gpu", 2)])` allows at most 2 GPU stages
/// to run concurrently.
#[derive(Clone)]
pub struct ResourcePool {
    pools: Arc<HashMap<String, Arc<Semaphore>>>,
}

impl ResourcePool {
    pub fn new(resources: impl IntoIterator<Item = (impl Into<String>, usize)>) -> Self {
        let pools = resources
            .into_iter()
            .map(|(name, permits)| (name.into(), Arc::new(Semaphore::new(permits))))
            .collect();
        Self {
            pools: Arc::new(pools),
        }
    }

    pub fn empty() -> Self {
        Self {
            pools: Arc::new(HashMap::new()),
        }
    }

    pub async fn acquire(&self, name: &str) -> Option<tokio::sync::SemaphorePermit<'_>> {
        self.pools.get(name)?.acquire().await.ok()
    }
}

// ---------------------------------------------------------------------------
// Execution context
// ---------------------------------------------------------------------------

/// Variables available for `{var}` substitution in prompts and commands.
/// Populated and updated as the workflow progresses.
#[derive(Clone, Default)]
pub struct Context {
    vars: HashMap<String, String>,
    /// Paths to temporary files created during workflow execution.
    /// Cleaned up by [`Context::cleanup_temp_files`] at the end of `execute()`.
    pub temp_files: Vec<String>,
}

impl Context {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.vars.insert(key.into(), value.into());
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.vars.get(key).map(|s| s.as_str())
    }

    /// Substitute all `{key}` occurrences in `template` with their values.
    pub fn render(&self, template: &str) -> String {
        let mut out = template.to_owned();
        for (k, v) in &self.vars {
            out = out.replace(&format!("{{{k}}}"), v);
        }
        out
    }

    /// Record a temporary file path for cleanup at the end of the workflow.
    pub fn track_temp_file(&mut self, path: impl Into<String>) {
        self.temp_files.push(path.into());
    }

    /// Delete all tracked temporary files.  Errors are logged but not propagated.
    pub fn cleanup_temp_files(&self) {
        for path in &self.temp_files {
            if let Err(e) = std::fs::remove_file(path) {
                tracing::debug!(path = %path, error = %e, "failed to clean up temp file");
            }
        }
    }

    /// Like [`render`](Self::render), but single-quotes each variable value
    /// before substitution to prevent shell injection.
    ///
    /// Any embedded single quotes in the value are escaped as `'\''`
    /// (end-quote, escaped quote, re-open quote).
    ///
    /// Not used by built-in workflows (which reference LLM output via safe
    /// file paths), but available for custom workflows with untrusted vars.
    #[allow(dead_code)]
    pub fn render_shell(&self, template: &str) -> String {
        let mut out = template.to_owned();
        for (k, v) in &self.vars {
            let escaped = format!("'{}'", v.replace('\'', "'\\''"));
            out = out.replace(&format!("{{{k}}}"), &escaped);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Shell result helper
// ---------------------------------------------------------------------------

pub struct ShellResult {
    pub success: bool,
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

pub async fn sh(command: &str) -> anyhow::Result<ShellResult> {
    let output = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .output()
        .await?;
    Ok(ShellResult {
        success: output.status.success(),
        code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

/// Print a workflow progress marker to stderr.
pub fn step(name: &str) {
    eprintln!("\n\x1b[1;36m==> {name}\x1b[0m");
}

// ---------------------------------------------------------------------------
// Shared workflow builder — single source of truth for all entry points
// ---------------------------------------------------------------------------

/// Convert a `u64` argument to `usize` safely, capping at a reasonable
/// maximum (50) to prevent absurd retry counts.  Returns an error if the
/// value cannot be represented as `usize`.
fn safe_usize(value: u64, name: &str) -> anyhow::Result<usize> {
    const MAX_REASONABLE: u64 = 50;
    if value > MAX_REASONABLE {
        anyhow::bail!("invalid {name}: {value} exceeds maximum ({MAX_REASONABLE})");
    }
    usize::try_from(value)
        .map_err(|_| anyhow::anyhow!("invalid {name}: value {value} too large for usize"))
}

/// Build a `(Workflow, ResourcePool)` from a workflow name and a flat JSON
/// argument map.  Used by the daemon's `Request::Workflow` handler and
/// `run_workflow` LLM tool so all entry points share the same construction
/// logic.
pub fn build_workflow(
    name: &str,
    args: &serde_json::Map<String, serde_json::Value>,
) -> anyhow::Result<(Workflow, ResourcePool)> {
    let str_arg = |key: &str| -> Option<&str> { args.get(key).and_then(|v| v.as_str()) };
    let u64_arg = |key: &str| -> Option<u64> { args.get(key).and_then(|v| v.as_u64()) };
    let f64_arg = |key: &str| -> Option<f64> { args.get(key).and_then(|v| v.as_f64()) };

    // Auto-detect test script: prefer scripts/test.sh when it exists.
    let default_test_cmd = if std::path::Path::new("scripts/test.sh").exists() {
        "scripts/test.sh"
    } else {
        "cargo test"
    };

    match name {
        "dev-loop" => {
            let task = str_arg("task").unwrap_or("complete the task");
            let test_cmd = str_arg("test_cmd").unwrap_or(default_test_cmd);
            let max_retries = safe_usize(u64_arg("max_retries").unwrap_or(5), "max_retries")?;
            Ok((
                builtins::dev_loop(task, test_cmd, max_retries, max_retries),
                ResourcePool::empty(),
            ))
        }
        "bug-fix" => {
            let repo = str_arg("repo").unwrap_or(".");
            let test_cmd = str_arg("test_cmd").unwrap_or(default_test_cmd);
            let max_retries = safe_usize(u64_arg("max_retries").unwrap_or(3), "max_retries")?;
            Ok((
                builtins::bug_fix(repo, test_cmd, max_retries)?,
                ResourcePool::empty(),
            ))
        }
        "perf-sweep" => {
            let target = str_arg("target").unwrap_or("the target");
            let bench_cmd = str_arg("bench_cmd").unwrap_or("make bench");
            let threshold = f64_arg("regression_threshold").unwrap_or(0.05);
            if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
                anyhow::bail!(
                    "invalid regression_threshold: must be a finite number in 0.0..=1.0, got {threshold}"
                );
            }
            Ok((
                builtins::perf_sweep(target, "", bench_cmd, threshold),
                ResourcePool::empty(),
            ))
        }
        "tune-sweep" => {
            let target = str_arg("target").unwrap_or("the target");
            let run_cmd = str_arg("run_cmd").unwrap_or("echo {item_index}");
            let result_cmd = str_arg("result_cmd").unwrap_or("echo done");
            let resource = str_arg("resource").unwrap_or("gpu");
            let count = safe_usize(u64_arg("resource_count").unwrap_or(1), "resource_count")?;
            if count == 0 {
                anyhow::bail!("invalid resource_count: must be greater than 0");
            }
            let wf = builtins::tune_sweep(target, "", run_cmd, result_cmd, resource);
            let pool = ResourcePool::new([(resource, count)]);
            Ok((wf, pool))
        }
        // Minimal workflow for integration tests: one LLM stage + one Shell stage.
        // Not exposed in tool schemas or CLI docs.
        "test-echo" => {
            let task = str_arg("task").unwrap_or("do the task");
            let test_cmd = str_arg("test_cmd").unwrap_or("true");
            Ok((
                Workflow {
                    name: "test-echo".into(),
                    stages: vec![
                        Stage::new(
                            "develop",
                            Action::Llm {
                                prompt: format!("Complete this task:\n\n{task}"),
                            },
                        )
                        .with_on_fail(FailStrategy::Abort),
                        Stage::new(
                            "test",
                            Action::Shell {
                                command: test_cmd.to_owned(),
                            },
                        )
                        .with_on_fail(FailStrategy::Abort),
                    ],
                },
                ResourcePool::empty(),
            ))
        }
        other => anyhow::bail!(
            "unknown workflow: '{other}'. Valid: dev-loop, bug-fix, perf-sweep, tune-sweep"
        ),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflows::executor;

    // ---- parse_items -------------------------------------------------------

    #[test]
    fn parse_items_extracts_opt_lines() {
        let text = "Some intro text.\n- OPT: reduce memory allocations\n- OPT: vectorise the inner loop\nTrailing text.";
        let items = executor::parse_items_pub(r"- OPT: (.+)", text).unwrap();
        assert_eq!(
            items,
            vec!["reduce memory allocations", "vectorise the inner loop"]
        );
    }

    #[test]
    fn parse_items_empty_when_no_match() {
        let items = executor::parse_items_pub(r"- OPT: (.+)", "no matches here").unwrap();
        assert!(items.is_empty());
    }

    #[test]
    fn parse_items_trims_whitespace() {
        let items = executor::parse_items_pub(r"- BUG: (.+)", "- BUG:   fix the crash  ").unwrap();
        assert_eq!(items, vec!["fix the crash"]);
    }

    // ---- Context::render ---------------------------------------------------

    #[test]
    fn context_render_substitutes_vars() {
        let mut ctx = Context::new();
        ctx.set("item", "fix null dereference");
        ctx.set("stderr", "error: null ptr");
        let result = ctx.render("Fix this: {item}\nDetails: {stderr}");
        assert_eq!(
            result,
            "Fix this: fix null dereference\nDetails: error: null ptr"
        );
    }

    #[test]
    fn context_render_leaves_unknown_vars_intact() {
        let ctx = Context::new();
        let result = ctx.render("Hello {unknown_var}");
        assert_eq!(result, "Hello {unknown_var}");
    }

    #[test]
    fn context_render_multiple_occurrences() {
        let mut ctx = Context::new();
        ctx.set("item", "X");
        let result = ctx.render("{item} and {item} again");
        assert_eq!(result, "X and X again");
    }

    // ---- detect_regression -------------------------------------------------

    #[test]
    fn no_regression_when_metrics_improve() {
        // Only throughput-style metrics (higher = better).
        let baseline = r#"{"fps": 100.0, "throughput": 200.0}"#;
        let current = r#"{"fps": 110.0, "throughput": 210.0}"#; // both improve
        let result = executor::detect_regression_pub(baseline, current, 0.05).unwrap();
        assert!(
            result.is_none(),
            "metric improvement should not trigger regression: {result:?}"
        );
    }

    #[test]
    fn regression_detected_when_metric_drops() {
        let baseline = r#"{"throughput": 100.0}"#;
        let current = r#"{"throughput": 80.0}"#; // -20% > 5% threshold
        let result = executor::detect_regression_pub(baseline, current, 0.05).unwrap();
        assert!(result.is_some(), "should detect 20% throughput drop");
        assert!(result.unwrap().contains("throughput"));
    }

    #[test]
    fn no_regression_within_threshold() {
        let baseline = r#"{"latency_ms": 100.0}"#;
        let current = r#"{"latency_ms": 104.0}"#; // +4%, within 5% threshold
        let result = executor::detect_regression_pub(baseline, current, 0.05).unwrap();
        assert!(
            result.is_none(),
            "4% change within 5% threshold should not trigger"
        );
    }

    #[test]
    fn empty_baseline_never_triggers_regression() {
        let result =
            executor::detect_regression_pub("{}", r#"{"latency_ms": 200.0}"#, 0.05).unwrap();
        assert!(result.is_none());
    }
}
