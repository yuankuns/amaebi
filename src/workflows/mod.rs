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
#[derive(Clone)]
pub struct Workflow {
    pub name: String,
    pub stages: Vec<Stage>,
}

/// One step in the workflow.
#[derive(Clone)]
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
#[derive(Clone)]
pub enum Action {
    /// Ask the LLM to do something. The prompt may contain `{var}` placeholders
    /// resolved from the current `Context`.
    Llm { prompt: String },

    /// Run a shell command. May contain `{var}` placeholders.
    Shell { command: String },

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
        // NOTE: per-item resource constraints are declared with
        // `Stage::with_requires(name)` on individual sub-stages, which lets
        // ResourcePool semaphores limit how many items hold a resource at once.
    },
}

/// How to decide whether a stage succeeded.
#[derive(Clone)]
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
#[derive(Clone)]
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
    /// Use for Llm prompts where values are embedded as plain text.
    pub fn render(&self, template: &str) -> String {
        let mut out = template.to_owned();
        for (k, v) in &self.vars {
            out = out.replace(&format!("{{{k}}}"), v);
        }
        out
    }

    /// Like `render`, but each substituted value is shell-quoted (POSIX
    /// single-quote style) so that special characters in context variables
    /// — semicolons, backticks, `$()`, etc. — cannot be interpreted by the
    /// shell.  Use for `Action::Shell` commands.
    pub fn render_shell(&self, template: &str) -> String {
        let mut out = template.to_owned();
        for (k, v) in &self.vars {
            out = out.replace(&format!("{{{k}}}"), &shell_quote(v));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Shell quoting helper
// ---------------------------------------------------------------------------

/// POSIX-style shell quoting: wraps `s` in single quotes and escapes any
/// single quotes inside using the `'\''` pattern.  The result can be used as
/// a single token in any sh-compatible command without injection risk.
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
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

// ---------------------------------------------------------------------------
// Progress channel — lets daemon forward step() output to the IPC stream
// ---------------------------------------------------------------------------

static PROGRESS_TX: std::sync::Mutex<Option<tokio::sync::mpsc::UnboundedSender<String>>> =
    std::sync::Mutex::new(None);

pub fn set_progress(tx: tokio::sync::mpsc::UnboundedSender<String>) {
    if let Ok(mut g) = PROGRESS_TX.lock() {
        *g = Some(tx);
    }
}

pub fn clear_progress() {
    if let Ok(mut g) = PROGRESS_TX.lock() {
        *g = None;
    }
}

/// Print a workflow progress marker to stderr.
pub fn step(name: &str) {
    eprintln!("\n\x1b[1;36m==> {name}\x1b[0m");
    if let Ok(g) = PROGRESS_TX.lock() {
        if let Some(tx) = g.as_ref() {
            let _ = tx.send(name.to_owned());
        }
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
        // detect_regression treats all metrics as "higher is better".
        // A small drop within the threshold must not fire.
        let baseline = r#"{"throughput": 100.0}"#;
        let current = r#"{"throughput": 96.0}"#; // -4%, within 5% threshold
        let result = executor::detect_regression_pub(baseline, current, 0.05).unwrap();
        assert!(
            result.is_none(),
            "4% drop within 5% threshold should not trigger"
        );
    }

    #[test]
    fn empty_baseline_never_triggers_regression() {
        let result =
            executor::detect_regression_pub("{}", r#"{"throughput": 200.0}"#, 0.05).unwrap();
        assert!(result.is_none());
    }
}
