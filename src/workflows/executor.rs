use anyhow::{anyhow, Context as _, Result};
use async_recursion::async_recursion;
use regex::Regex;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};
use tokio::io::{self as tio, AsyncWrite};
use tokio::sync::Mutex;

/// Monotonic counter for unique LLM output temp-file names.
/// Combined with the process ID it prevents parallel workflow stages from
/// overwriting each other's output files.
static LLM_OUTPUT_SEQ: AtomicU64 = AtomicU64::new(0);

use crate::copilot::Message;
use crate::daemon::DaemonState;
use crate::daemon::{build_messages, inject_skill_files, run_agentic_loop};

use super::{sh, step, Action, Check, Context, FailStrategy, ResourcePool, Stage, Workflow};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Execute a workflow from start to finish.
///
/// Returns a summary string (from the final LLM stage, or a generated one).
pub async fn execute(
    workflow: &Workflow,
    state: &Arc<DaemonState>,
    model: &str,
    initial_ctx: Context,
    resources: &ResourcePool,
) -> Result<String> {
    tracing::info!(
        name = %workflow.name,
        stages = workflow.stages.len(),
        model = %model,
        "workflow: starting"
    );
    step(&format!("Workflow: {}", workflow.name));

    let mut messages: Vec<Message> = {
        let mut msgs = build_messages(
            &format!(
                "You are executing the workflow: {}. \
                 Work autonomously — use available tools (shell_command, edit_file, \
                 read_file) to complete each stage. \
                 Never ask for clarification or user input; make your best judgment \
                 and proceed.",
                workflow.name
            ),
            None,
            &[],
            &[],
            None,
        );
        inject_skill_files(&mut msgs).await;
        msgs
    };

    let mut ctx = initial_ctx;
    let result = run_stages(
        &workflow.stages,
        state,
        model,
        &mut messages,
        &mut ctx,
        resources,
    )
    .await;

    match &result {
        Ok(_) => tracing::info!(name = %workflow.name, "workflow: completed successfully"),
        Err(e) => tracing::info!(name = %workflow.name, error = %e, "workflow: failed"),
    }

    Ok(result?.unwrap_or_else(|| "Workflow completed.".to_owned()))
}

// ---------------------------------------------------------------------------
// Stage runner — returns the last LLM text produced (for summaries)
// ---------------------------------------------------------------------------

#[async_recursion]
async fn run_stages(
    stages: &[Stage],
    state: &Arc<DaemonState>,
    model: &str,
    messages: &mut Vec<Message>,
    ctx: &mut Context,
    resources: &ResourcePool,
) -> Result<Option<String>> {
    let mut last_text: Option<String> = None;

    for stage in stages {
        step(&format!("  Stage: {}", stage.name));

        let text = run_stage_with_retry(stage, state, model, messages, ctx, resources).await?;
        if let Some(ref t) = text {
            last_text = Some(t.clone());
        }
    }

    Ok(last_text)
}

#[async_recursion]
async fn run_stage_with_retry(
    stage: &Stage,
    state: &Arc<DaemonState>,
    model: &str,
    messages: &mut Vec<Message>,
    ctx: &mut Context,
    resources: &ResourcePool,
) -> Result<Option<String>> {
    let max_attempts = match &stage.on_fail {
        FailStrategy::Retry { max, .. } => *max + 1,
        _ => 1,
    };

    for attempt in 0..max_attempts {
        if attempt > 0 {
            step(&format!("    Retry {attempt}/{}", max_attempts - 1));
        }

        match run_single_stage(stage, state, model, messages, ctx, resources).await {
            Ok(text) => return Ok(text),
            Err(e) => {
                match &stage.on_fail {
                    FailStrategy::Abort => return Err(e),
                    FailStrategy::Skip => {
                        step(&format!("    Skipping stage '{}': {e}", stage.name));
                        return Ok(None);
                    }
                    FailStrategy::RevertAndSkip => {
                        step(&format!("    Reverting and skipping '{}': {e}", stage.name));
                        // Reset tracked files to HEAD and remove untracked files
                        // introduced by this stage only (previous committed stages
                        // are already in HEAD so they are unaffected).
                        let _ = sh("git reset --hard HEAD 2>/dev/null || true").await;
                        let _ = sh("git clean -fd 2>/dev/null || true").await;
                        return Ok(None);
                    }
                    FailStrategy::Retry { inject_prompt, .. } => {
                        if attempt + 1 >= max_attempts {
                            return Err(e.context(format!(
                                "Stage '{}' failed after {max_attempts} attempts",
                                stage.name
                            )));
                        }
                        // Inject error context back to LLM for the next attempt.
                        let error_msg = e.to_string();
                        let injection = ctx.render(inject_prompt).replace("{error}", &error_msg);
                        step("    Injecting error context to LLM");
                        let _ = llm_turn(state, model, messages, &injection).await?;
                    }
                }
            }
        }
    }

    unreachable!()
}

#[async_recursion]
async fn run_single_stage(
    stage: &Stage,
    state: &Arc<DaemonState>,
    model: &str,
    messages: &mut Vec<Message>,
    ctx: &mut Context,
    resources: &ResourcePool,
) -> Result<Option<String>> {
    // Acquire resource permit if required.
    let _permit = if let Some(ref res_name) = stage.requires {
        let permit = resources
            .acquire(res_name)
            .await
            .ok_or_else(|| anyhow!("unknown resource '{res_name}'"))?;
        Some(permit)
    } else {
        None
    };

    match &stage.action {
        Action::Llm { prompt } => {
            let rendered = ctx.render(prompt);
            let text = llm_turn(state, model, messages, &rendered).await?;
            ctx.set("last_llm_output", &text);
            // Write to a unique temp file so parallel stages don't clobber each
            // other.  The path is stored in ctx so Shell stages can use it via
            // {last_llm_output_file} without any shell injection risk.
            let seq = LLM_OUTPUT_SEQ.fetch_add(1, Ordering::Relaxed);
            let tmp = format!("/tmp/amaebi_llm_{}_{}.txt", std::process::id(), seq);
            let _ = tokio::fs::write(&tmp, &text).await;
            ctx.set("last_llm_output_file", &tmp);
            run_check(&stage.check, ctx).await?;
            Ok(Some(text))
        }

        Action::Shell { command } => {
            let rendered = ctx.render_shell(command);
            let result = sh(&rendered).await?;
            ctx.set("stdout", &result.stdout);
            ctx.set("stderr", &result.stderr);
            ctx.set("exit_code", result.code.to_string());
            if !result.success {
                return Err(anyhow!(
                    "Shell command failed (exit {})\nstderr: {}",
                    result.code,
                    &result.stderr[..result.stderr.len().min(2000)]
                ));
            }
            run_check(&stage.check, ctx).await?;
            Ok(None)
        }

        Action::Map {
            parse,
            stages: sub_stages,
            parallel,
        } => {
            let source = ctx.get("last_llm_output").unwrap_or("").to_owned();
            let items = parse_items(parse, &source)?;
            step(&format!(
                "    Map over {} items (parallel={})",
                items.len(),
                parallel
            ));

            if items.is_empty() {
                return Err(anyhow!("Map stage found 0 items — check parse regex"));
            }

            if *parallel {
                run_map_parallel(&items, sub_stages, state, model, ctx, resources).await?;
            } else {
                run_map_serial(&items, sub_stages, state, model, messages, ctx, resources).await?;
            }

            Ok(None)
        }
    }
}

// ---------------------------------------------------------------------------
// Map: serial
// ---------------------------------------------------------------------------

#[async_recursion]
async fn run_map_serial(
    items: &[String],
    sub_stages: &[Stage],
    state: &Arc<DaemonState>,
    model: &str,
    messages: &mut Vec<Message>,
    ctx: &mut Context,
    resources: &ResourcePool,
) -> Result<()> {
    for (i, item) in items.iter().enumerate() {
        step(&format!("    [{}/{}] {}", i + 1, items.len(), item));
        ctx.set("item", item);
        ctx.set("item_index", i.to_string());

        match run_stages(sub_stages, state, model, messages, ctx, resources).await {
            Ok(_) => {}
            Err(e) => {
                step(&format!("    Item failed, skipping: {e}"));
                // Serial map: skip failed items and continue.
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Map: parallel
// ---------------------------------------------------------------------------

async fn run_map_parallel(
    items: &[String],
    sub_stages: &[Stage],
    state: &Arc<DaemonState>,
    model: &str,
    ctx: &Context,
    resources: &ResourcePool,
) -> Result<()> {
    // Each parallel item gets its own messages Vec (independent context).
    #[allow(clippy::type_complexity)]
    let results: Arc<Mutex<Vec<(usize, Result<()>)>>> = Arc::new(Mutex::new(Vec::new()));

    let mut handles = Vec::new();

    for (i, item) in items.iter().enumerate() {
        let item = item.clone();
        let state = Arc::clone(state);
        let sub_stages: Vec<Stage> = sub_stages.to_vec();
        let mut item_ctx = ctx.clone();
        item_ctx.set("item", &item);
        item_ctx.set("item_index", i.to_string());
        let resources = resources.clone();
        let results = Arc::clone(&results);
        let model = model.to_owned();
        let total = items.len();

        // Concurrency semaphore — only for stages that declare `requires`.
        // The overall Map concurrency is unlimited; individual stages within
        // the sub-pipeline are constrained by their own `requires` field.
        let handle = tokio::spawn(async move {
            step(&format!(
                "    [parallel {i_plus_one}/{total}] {item}",
                i_plus_one = i + 1
            ));

            // Each parallel worker gets a fresh messages vec.
            let mut messages = build_messages(&format!("Working on: {item}"), None, &[], &[], None);
            inject_skill_files(&mut messages).await;

            // Collect sub-stages by cloning the action/check data.
            // We can't share &Stage across tasks, so we build owned copies
            // of the flat list here via a local runner.
            let result = run_parallel_item_stages(
                &sub_stages,
                &state,
                &model,
                &mut messages,
                &mut item_ctx,
                &resources,
            )
            .await;

            let mut r = results.lock().await;
            r.push((i, result.map(|_| ())));
        });

        handles.push(handle);
    }

    for handle in handles {
        handle
            .await
            .map_err(|e| anyhow!("parallel task panicked: {e}"))?;
    }

    // Report any failures (they were already skipped individually).
    let results = results.lock().await;
    let failed: Vec<_> = results.iter().filter(|(_, r)| r.is_err()).collect();
    if !failed.is_empty() {
        step(&format!(
            "    {}/{} items had errors (skipped)",
            failed.len(),
            items.len()
        ));
    }

    Ok(())
}

/// Run stages for one parallel Map item.  Mirrors `run_stages` on an owned
/// `Vec<Stage>` that was cloned into the spawned task.
#[async_recursion]
async fn run_parallel_item_stages(
    stages: &[Stage],
    state: &Arc<DaemonState>,
    model: &str,
    messages: &mut Vec<Message>,
    ctx: &mut Context,
    resources: &ResourcePool,
) -> Result<Option<String>> {
    let mut last_text = None;
    for stage in stages {
        let text = run_stage_with_retry(stage, state, model, messages, ctx, resources).await?;
        if let Some(t) = text {
            last_text = Some(t);
        }
    }
    Ok(last_text)
}

// ---------------------------------------------------------------------------
// Check evaluation
// ---------------------------------------------------------------------------

async fn run_check(check: &Option<Check>, ctx: &mut Context) -> Result<()> {
    let check = match check {
        Some(c) => c,
        None => return Ok(()),
    };

    match check {
        Check::ExitCode { command } => {
            let rendered = ctx.render_shell(command);
            let r = sh(&rendered).await?;
            if !r.success {
                ctx.set("stderr", &r.stderr);
                ctx.set("stdout", &r.stdout);
                return Err(anyhow!(
                    "Check failed (exit {}): {}\nstderr: {}",
                    r.code,
                    rendered,
                    &r.stderr[..r.stderr.len().min(2000)]
                ));
            }
            ctx.set("stdout", &r.stdout);
            ctx.set("stderr", &r.stderr);
        }

        Check::Contains { command, pattern } => {
            let rendered = ctx.render_shell(command);
            let r = sh(&rendered).await?;
            ctx.set("stdout", &r.stdout);
            ctx.set("stderr", &r.stderr);
            if !r.success {
                return Err(anyhow!(
                    "Check command failed (exit {}): {}\nstderr: {}",
                    r.code,
                    rendered,
                    &r.stderr[..r.stderr.len().min(2000)]
                ));
            }
            if !r.stdout.contains(pattern.as_str()) {
                return Err(anyhow!(
                    "Check failed: output does not contain {:?}\nstdout: {}",
                    pattern,
                    &r.stdout[..r.stdout.len().min(2000)]
                ));
            }
        }

        Check::BenchmarkNoRegression { command, threshold } => {
            let rendered = ctx.render_shell(command);
            let r = sh(&rendered).await?;
            ctx.set("stdout", &r.stdout);
            if !r.success {
                return Err(anyhow!(
                    "Benchmark command failed (exit {}): {}\nstderr: {}",
                    r.code,
                    rendered,
                    &r.stderr[..r.stderr.len().min(2000)]
                ));
            }

            let baseline_json = ctx.get("benchmark_baseline").unwrap_or("{}").to_owned();
            let regression = detect_regression(&baseline_json, &r.stdout, *threshold)?;
            if let Some(msg) = regression {
                ctx.set("regression_summary", &msg);
                return Err(anyhow!("Benchmark regression detected: {msg}"));
            }
            // Update baseline on success.
            ctx.set("benchmark_baseline", &r.stdout);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// LLM turn
// ---------------------------------------------------------------------------

/// A writer that discards all output but intercepts `WaitingForInput` frames
/// and automatically steers the agentic loop to continue working autonomously.
///
/// Without this, dropping `steer_tx` causes `steer_rx.recv()` to return `None`
/// (Disconnected) immediately when Claude asks a question, making the agentic
/// loop exit early — before Claude has done any actual work.
struct AutoContinueWriter {
    pending: Vec<u8>,
    steer_tx: tokio::sync::mpsc::Sender<Option<String>>,
}

impl AutoContinueWriter {
    fn new(steer_tx: tokio::sync::mpsc::Sender<Option<String>>) -> Self {
        Self {
            pending: Vec::new(),
            steer_tx,
        }
    }
}

impl Unpin for AutoContinueWriter {}

impl AsyncWrite for AutoContinueWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut TaskContext<'_>,
        buf: &[u8],
    ) -> Poll<tio::Result<usize>> {
        self.pending.extend_from_slice(buf);
        // Process complete newline-delimited JSON frames.
        while let Some(pos) = self.pending.iter().position(|&b| b == b'\n') {
            let line = self.pending[..pos].to_vec();
            self.pending.drain(..=pos);
            if std::str::from_utf8(&line).is_ok_and(|s| s.contains("\"waiting_for_input\"")) {
                // Claude is asking a question. Reply immediately so the agentic
                // loop continues rather than blocking for 5 minutes or exiting.
                let _ = self.steer_tx.try_send(Some(
                    "Continue working autonomously. \
                     Do not ask for clarification — use your best judgment and proceed."
                        .into(),
                ));
            }
        }
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _: &mut TaskContext<'_>) -> Poll<tio::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _: &mut TaskContext<'_>) -> Poll<tio::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

pub async fn llm_turn(
    state: &Arc<DaemonState>,
    model: &str,
    messages: &mut Vec<Message>,
    prompt: &str,
) -> Result<String> {
    messages.push(Message::user(prompt.to_owned()));

    // Keep steer_tx alive for the duration of the call so the Disconnected
    // branch in run_agentic_loop is never triggered prematurely.
    // AutoContinueWriter intercepts WaitingForInput frames and auto-replies.
    let (steer_tx, mut steer_rx) = tokio::sync::mpsc::channel::<Option<String>>(4);
    let mut writer = AutoContinueWriter::new(steer_tx.clone());

    let (text, _, returned_messages) = run_agentic_loop(
        state,
        model,
        messages.clone(),
        &mut writer,
        &mut steer_rx,
        false, // workflow LLM turns must not expose spawn_agent / run_workflow
        false, // workflow LLM turns must not schedule follow-ups
        "",    // no session id needed for workflow turns
    )
    .await
    .context("LLM turn failed")?;

    drop(steer_tx); // safe to drop now that the loop has finished

    // Replace messages with the full history returned by run_agentic_loop,
    // which includes all tool calls and results from the stage.  Discarding
    // this (as the previous `_` did) caused Claude to lose context about what
    // it had already tried, making retries much less effective.
    *messages = returned_messages;

    Ok(text)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_items(pattern: &str, text: &str) -> Result<Vec<String>> {
    let re = Regex::new(pattern).with_context(|| format!("invalid parse regex: {pattern:?}"))?;
    let items: Vec<String> = re
        .captures_iter(text)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect();
    Ok(items)
}

/// Naive benchmark regression: compare JSON numbers between two snapshots.
/// Returns Some(description) if any metric regressed beyond `threshold`.
fn detect_regression(
    baseline_json: &str,
    current_json: &str,
    threshold: f64,
) -> Result<Option<String>> {
    // Parse as flat JSON objects {metric: number}.
    let baseline: serde_json::Value = serde_json::from_str(baseline_json).unwrap_or_default();
    let current: serde_json::Value = serde_json::from_str(current_json).unwrap_or_default();

    let baseline_obj = match baseline.as_object() {
        Some(o) => o,
        None => return Ok(None),
    };
    let current_obj = match current.as_object() {
        Some(o) => o,
        None => return Ok(None),
    };

    let mut regressions = Vec::new();
    for (key, base_val) in baseline_obj {
        if let (Some(b), Some(c)) = (
            base_val.as_f64(),
            current_obj.get(key).and_then(|v| v.as_f64()),
        ) {
            if b > 0.0 {
                let change = (c - b) / b; // assumes higher is better; a drop flags regression
                if change < -threshold {
                    regressions.push(format!("{key}: {b:.4} → {c:.4} ({:+.1}%)", change * 100.0));
                }
            }
        }
    }

    if regressions.is_empty() {
        Ok(None)
    } else {
        Ok(Some(regressions.join(", ")))
    }
}

// ---------------------------------------------------------------------------
// Test-only exports for unit testing pure functions
// ---------------------------------------------------------------------------

#[cfg(test)]
pub fn parse_items_pub(pattern: &str, text: &str) -> anyhow::Result<Vec<String>> {
    parse_items(pattern, text)
}

#[cfg(test)]
pub fn detect_regression_pub(
    baseline: &str,
    current: &str,
    threshold: f64,
) -> anyhow::Result<Option<String>> {
    detect_regression(baseline, current, threshold)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Instant;

    use super::*;
    use crate::test_utils::with_temp_home;
    use crate::workflows::{Action, Check, Context, FailStrategy, ResourcePool, Stage, Workflow};

    /// Build a minimal DaemonState suitable for Shell-only workflow tests.
    /// Caller must already hold a with_temp_home() guard.
    async fn test_state() -> Arc<DaemonState> {
        Arc::new(DaemonState::new().await.expect("DaemonState::new"))
    }

    // ---- serial Map preserves order ----------------------------------------

    #[tokio::test]
    async fn serial_map_runs_stages_in_order() {
        let _guard = with_temp_home();
        let state = test_state().await;
        let outfile = tempfile::NamedTempFile::new().unwrap();
        let path = outfile.path().to_str().unwrap().to_owned();

        let wf = Workflow {
            name: "order-test".into(),
            stages: vec![
                // Seed last_llm_output with an item list (Map parses this).
                Stage::new(
                    "seed",
                    Action::Shell {
                        command: "echo '- ITEM: alpha\n- ITEM: beta\n- ITEM: gamma'".into(),
                    },
                )
                .with_on_fail(FailStrategy::Abort),
                // Append each item to file in order.
                Stage::new(
                    "append",
                    Action::Map {
                        parse: r"- ITEM: (.+)".into(),
                        parallel: false,
                        stages: vec![Stage::new(
                            "write",
                            Action::Shell {
                                command: format!("echo {{item}} >> {path}"),
                            },
                        )
                        .with_on_fail(FailStrategy::Abort)],
                    },
                )
                .with_on_fail(FailStrategy::Abort),
            ],
        };

        // Seed last_llm_output manually since we can't call LLM in tests.
        let mut ctx = Context::new();
        ctx.set(
            "last_llm_output",
            "- ITEM: alpha\n- ITEM: beta\n- ITEM: gamma",
        );

        execute_with_ctx(&wf, &state, "gpt-4o", ctx, &ResourcePool::empty())
            .await
            .unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(
            lines,
            vec!["alpha", "beta", "gamma"],
            "items must appear in order"
        );
    }

    // ---- parallel Map runs concurrently ------------------------------------

    #[tokio::test]
    async fn parallel_map_runs_faster_than_serial() {
        let _guard = with_temp_home();
        let state = test_state().await;

        let wf = Workflow {
            name: "parallel-test".into(),
            stages: vec![Stage::new(
                "sleep-all",
                Action::Map {
                    parse: r"- SLEEP: (.+)".into(),
                    parallel: true,
                    stages: vec![Stage::new(
                        "sleep",
                        Action::Shell {
                            command: "sleep 0.3".into(),
                        },
                    )
                    .with_on_fail(FailStrategy::Skip)],
                },
            )
            .with_on_fail(FailStrategy::Abort)],
        };

        // 3 items × 0.3 s = 0.9 s serial; parallel should complete in ~0.3 s.
        let mut ctx = Context::new();
        ctx.set("last_llm_output", "- SLEEP: a\n- SLEEP: b\n- SLEEP: c");

        let t = Instant::now();
        execute_with_ctx(&wf, &state, "gpt-4o", ctx, &ResourcePool::empty())
            .await
            .unwrap();
        let elapsed = t.elapsed();

        assert!(
            elapsed.as_millis() < 800,
            "parallel map should complete in < 0.8 s, took {elapsed:?}"
        );
    }

    // ---- ResourcePool limits concurrency -----------------------------------

    #[tokio::test]
    async fn resource_pool_serialises_parallel_map() {
        let _guard = with_temp_home();
        let state = test_state().await;
        let outfile = tempfile::NamedTempFile::new().unwrap();
        let path = outfile.path().to_str().unwrap().to_owned();

        // Each item sleeps then appends a timestamp.  With concurrency=1 the
        // sleeps cannot overlap, so total time ≥ 3 × sleep.
        let wf = Workflow {
            name: "resource-test".into(),
            stages: vec![Stage::new(
                "limited",
                Action::Map {
                    parse: r"- ITEM: (.+)".into(),
                    parallel: true,
                    stages: vec![Stage::new(
                        "work",
                        Action::Shell {
                            command: format!("sleep 0.15 && date +%s%3N >> {path}"),
                        },
                    )
                    .with_requires("slot")
                    .with_on_fail(FailStrategy::Skip)],
                },
            )
            .with_on_fail(FailStrategy::Abort)],
        };

        let mut ctx = Context::new();
        ctx.set("last_llm_output", "- ITEM: a\n- ITEM: b\n- ITEM: c");

        let pool = ResourcePool::new([("slot", 1usize)]);
        let t = Instant::now();
        execute_with_ctx(&wf, &state, "gpt-4o", ctx, &pool)
            .await
            .unwrap();
        let elapsed = t.elapsed();

        // With concurrency=1: 3 × 150 ms = ≥ 450 ms.
        assert!(
            elapsed.as_millis() >= 400,
            "pool=1 should serialise; expected ≥ 400 ms, got {elapsed:?}"
        );
    }

    // ---- Retry injects error context into prompt ---------------------------

    #[tokio::test]
    async fn retry_skip_continues_after_shell_failure() {
        let _guard = with_temp_home();
        let state = test_state().await;

        // A Shell stage that always fails, with Skip strategy.
        let wf = Workflow {
            name: "retry-test".into(),
            stages: vec![
                Stage::new(
                    "always-fail",
                    Action::Shell {
                        command: "exit 1".into(),
                    },
                )
                .with_on_fail(FailStrategy::Skip),
                Stage::new(
                    "always-pass",
                    Action::Shell {
                        command: "exit 0".into(),
                    },
                )
                .with_on_fail(FailStrategy::Abort),
            ],
        };

        // Should not error: the failing stage is skipped, the passing one runs.
        let result = execute_with_ctx(
            &wf,
            &state,
            "gpt-4o",
            Context::new(),
            &ResourcePool::empty(),
        )
        .await;
        assert!(
            result.is_ok(),
            "Skip strategy should allow workflow to continue: {result:?}"
        );
    }

    // ---- Check::ExitCode gates the stage -----------------------------------

    #[tokio::test]
    async fn check_exit_code_fails_stage() {
        let _guard = with_temp_home();
        let state = test_state().await;

        let wf = Workflow {
            name: "check-test".into(),
            stages: vec![Stage::new(
                "pass-then-check-fail",
                Action::Shell {
                    command: "true".into(),
                },
            )
            .with_check(Check::ExitCode {
                command: "false".into(),
            })
            .with_on_fail(FailStrategy::Abort)],
        };

        let result = execute_with_ctx(
            &wf,
            &state,
            "gpt-4o",
            Context::new(),
            &ResourcePool::empty(),
        )
        .await;
        assert!(result.is_err(), "check failure should propagate as error");
    }

    // ---- Context variable substitution in Shell command --------------------

    #[tokio::test]
    async fn shell_command_substitutes_context_vars() {
        let _guard = with_temp_home();
        let state = test_state().await;
        let outfile = tempfile::NamedTempFile::new().unwrap();
        let path = outfile.path().to_str().unwrap().to_owned();

        let wf = Workflow {
            name: "subst-test".into(),
            stages: vec![Stage::new(
                "write",
                Action::Shell {
                    command: format!("echo {{greeting}} > {path}"),
                },
            )
            .with_on_fail(FailStrategy::Abort)],
        };

        let mut ctx = Context::new();
        ctx.set("greeting", "hello-workflow");

        execute_with_ctx(&wf, &state, "gpt-4o", ctx, &ResourcePool::empty())
            .await
            .unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(
            content.contains("hello-workflow"),
            "context var must be substituted"
        );
    }

    // ---- Benchmark regression detection ------------------------------------

    #[test]
    fn benchmark_regression_fires_on_drop() {
        let baseline = r#"{"fps": 100.0, "throughput": 10.0}"#;
        let regressed = r#"{"fps": 70.0, "throughput": 10.0}"#; // -30% fps

        let result = detect_regression(baseline, regressed, 0.05).unwrap();
        assert!(result.is_some(), "30% fps drop should trigger regression");
        let msg = result.unwrap();
        assert!(
            msg.contains("fps"),
            "regression message should name the metric"
        );
    }

    #[test]
    fn benchmark_regression_silent_on_improvement() {
        let baseline = r#"{"fps": 100.0}"#;
        let better = r#"{"fps": 120.0}"#;
        let result = detect_regression(baseline, better, 0.05).unwrap();
        assert!(result.is_none());
    }
}

// ---------------------------------------------------------------------------
// Shell-only regression tests for the perf_sweep baseline/benchmark fixes.
// These do not call the LLM — they verify the executor's Check evaluation and
// context-variable propagation are correct.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod shell_regression_tests {
    use super::*;
    use crate::test_utils::with_temp_home;
    use crate::workflows::{Action, Check, Context, FailStrategy, ResourcePool, Stage, Workflow};
    use std::sync::Arc;

    async fn test_state() -> Arc<DaemonState> {
        Arc::new(DaemonState::new().await.expect("DaemonState::new"))
    }

    /// The perf_sweep baseline stage uses Action::Shell{"true"} +
    /// Check::BenchmarkNoRegression{threshold=∞}.  After it runs,
    /// ctx["benchmark_baseline"] must contain the JSON from the check command.
    #[tokio::test]
    async fn baseline_check_seeds_benchmark_baseline() {
        let _guard = with_temp_home();
        let state = test_state().await;

        let wf = Workflow {
            name: "baseline-test".into(),
            stages: vec![Stage::new(
                "baseline",
                Action::Shell {
                    command: "true".into(),
                },
            )
            .with_check(Check::BenchmarkNoRegression {
                command: r#"echo '{"fps": 100.0, "throughput": 10.0}'"#.into(),
                threshold: f64::INFINITY,
            })
            .with_on_fail(FailStrategy::Abort)],
        };

        let mut ctx = Context::new();
        // No baseline yet.
        assert!(ctx.get("benchmark_baseline").is_none());

        execute_with_ctx(&wf, &state, "gpt-4o", ctx.clone(), &ResourcePool::empty())
            .await
            .unwrap();

        // The BenchmarkNoRegression check must have seeded ctx["benchmark_baseline"].
        // We verify by running the check again on a fresh context seeded from the
        // workflow: re-run in isolation to inspect the context.
        // Simpler: run a two-stage workflow and inspect via a Shell stage.
        let outfile = tempfile::NamedTempFile::new().unwrap();
        let path = outfile.path().to_str().unwrap().to_owned();

        let wf2 = Workflow {
            name: "verify-baseline".into(),
            stages: vec![
                // Stage 1: seed baseline (threshold=∞ → always passes, saves benchmark_baseline)
                Stage::new(
                    "baseline",
                    Action::Shell {
                        command: "true".into(),
                    },
                )
                .with_check(Check::BenchmarkNoRegression {
                    command: r#"echo '{"fps": 100.0}'"#.into(),
                    threshold: f64::INFINITY,
                })
                .with_on_fail(FailStrategy::Abort),
                // Stage 2: write ctx["benchmark_baseline"] to a file for inspection
                Stage::new(
                    "dump",
                    Action::Shell {
                        // No manual quoting — render_shell will single-quote {benchmark_baseline}.
                        command: format!("printf '%s' {{benchmark_baseline}} > {path}"),
                    },
                )
                .with_on_fail(FailStrategy::Abort),
            ],
        };

        ctx = Context::new();
        execute_with_ctx(&wf2, &state, "gpt-4o", ctx, &ResourcePool::empty())
            .await
            .unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(
            content.contains("fps"),
            "benchmark_baseline should contain the bench output; got: {content:?}"
        );
    }

    /// After seeding the baseline, a subsequent BenchmarkNoRegression check
    /// must detect a real regression and propagate it as an error.
    #[tokio::test]
    async fn benchmark_regression_detected_after_baseline_seeded() {
        let _guard = with_temp_home();
        let state = test_state().await;

        let wf = Workflow {
            name: "regression-detect".into(),
            stages: vec![
                // Seed baseline: fps=100
                Stage::new(
                    "baseline",
                    Action::Shell {
                        command: "true".into(),
                    },
                )
                .with_check(Check::BenchmarkNoRegression {
                    command: r#"echo '{"fps": 100.0}'"#.into(),
                    threshold: f64::INFINITY,
                })
                .with_on_fail(FailStrategy::Abort),
                // Simulate a regressed run: fps=50 (−50%, well above 5% threshold)
                Stage::new(
                    "bench-regressed",
                    Action::Shell {
                        command: "true".into(),
                    },
                )
                .with_check(Check::BenchmarkNoRegression {
                    command: r#"echo '{"fps": 50.0}'"#.into(),
                    threshold: 0.05,
                })
                .with_on_fail(FailStrategy::Abort),
            ],
        };

        let result = execute_with_ctx(
            &wf,
            &state,
            "gpt-4o",
            Context::new(),
            &ResourcePool::empty(),
        )
        .await;
        assert!(
            result.is_err(),
            "50% fps regression should be detected: {result:?}"
        );
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("fps") || msg.contains("regression"),
            "error should name the regressing metric; got: {msg}"
        );
    }

    /// Bench should NOT regress when the metric improves after baseline is seeded.
    #[tokio::test]
    async fn no_false_regression_when_metric_improves_after_baseline() {
        let _guard = with_temp_home();
        let state = test_state().await;

        let wf = Workflow {
            name: "no-regression".into(),
            stages: vec![
                Stage::new(
                    "baseline",
                    Action::Shell {
                        command: "true".into(),
                    },
                )
                .with_check(Check::BenchmarkNoRegression {
                    command: r#"echo '{"fps": 100.0}'"#.into(),
                    threshold: f64::INFINITY,
                })
                .with_on_fail(FailStrategy::Abort),
                // fps improved: no regression
                Stage::new(
                    "bench-improved",
                    Action::Shell {
                        command: "true".into(),
                    },
                )
                .with_check(Check::BenchmarkNoRegression {
                    command: r#"echo '{"fps": 120.0}'"#.into(),
                    threshold: 0.05,
                })
                .with_on_fail(FailStrategy::Abort),
            ],
        };

        execute_with_ctx(
            &wf,
            &state,
            "gpt-4o",
            Context::new(),
            &ResourcePool::empty(),
        )
        .await
        .expect("improvement should not trigger regression");
    }
}

// ---------------------------------------------------------------------------
// LLM-mocked integration tests.
// Use a minimal inline SSE server (axum dev-dep) to exercise the full
// executor code path including Action::Llm and the Retry strategy.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod llm_tests {
    use super::*;
    use crate::test_utils::with_temp_home;
    use crate::workflows::{Action, Check, Context, FailStrategy, ResourcePool, Stage, Workflow};
    use axum::{
        extract::State as AxState, response::Response as AxResponse, routing::post, Router,
    };
    use serial_test::serial;
    use std::{
        collections::VecDeque,
        sync::{Arc as SArc, Mutex as SMutex},
    };
    use tokio::sync::oneshot;

    // ---- Minimal mock SSE server ----------------------------------------

    /// Queue of text responses + a request counter.
    #[derive(Clone, Default)]
    struct MockState {
        queue: SArc<SMutex<VecDeque<String>>>,
        request_count: SArc<SMutex<usize>>,
    }

    struct MiniMock {
        url: String,
        state: MockState,
        _shutdown: oneshot::Sender<()>,
    }

    impl MiniMock {
        async fn start() -> Self {
            let st = MockState::default();
            let app = Router::new()
                .route("/chat/completions", post(Self::handle))
                .with_state(st.clone());
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let (tx, rx) = oneshot::channel::<()>();
            tokio::spawn(async move {
                axum::serve(listener, app)
                    .with_graceful_shutdown(async {
                        let _ = rx.await;
                    })
                    .await
                    .ok();
            });
            Self {
                url: format!("http://{addr}/chat/completions"),
                state: st,
                _shutdown: tx,
            }
        }

        fn enqueue(&self, text: impl Into<String>) {
            self.state.queue.lock().unwrap().push_back(text.into());
        }

        fn request_count(&self) -> usize {
            *self.state.request_count.lock().unwrap()
        }

        async fn handle(
            AxState(st): AxState<MockState>,
            _body: axum::body::Bytes,
        ) -> AxResponse<axum::body::Body> {
            *st.request_count.lock().unwrap() += 1;
            let text = st.queue.lock().unwrap().pop_front().unwrap_or_default();
            let escaped = serde_json::to_string(&text).unwrap();
            let sse = format!(
                "data: {{\"id\":\"m\",\"object\":\"chat.completion.chunk\",\"model\":\"mock\",\
                 \"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":{escaped}}},\
                 \"finish_reason\":null}}]}}\n\n\
                 data: {{\"id\":\"m\",\"object\":\"chat.completion.chunk\",\"model\":\"mock\",\
                 \"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\n\
                 data: [DONE]\n\n"
            );
            AxResponse::builder()
                .status(200)
                .header("content-type", "text/event-stream")
                .body(axum::body::Body::from(sse))
                .unwrap()
        }
    }

    // ---- Env guard for AMAEBI_COPILOT_URL / TOKEN -----------------------

    struct LlmEnvGuard {
        old_url: Option<String>,
        old_token: Option<String>,
    }

    impl LlmEnvGuard {
        fn set(url: &str) -> Self {
            let old_url = std::env::var("AMAEBI_COPILOT_URL").ok();
            let old_token = std::env::var("AMAEBI_COPILOT_TOKEN").ok();
            // SAFETY: HOME_LOCK (held by with_temp_home caller) serialises
            // all env-var mutations in this test process.
            unsafe {
                std::env::set_var("AMAEBI_COPILOT_URL", url);
                std::env::set_var("AMAEBI_COPILOT_TOKEN", "test-mock-token");
            }
            Self { old_url, old_token }
        }
    }

    impl Drop for LlmEnvGuard {
        fn drop(&mut self) {
            unsafe {
                match &self.old_url {
                    Some(v) => std::env::set_var("AMAEBI_COPILOT_URL", v),
                    None => std::env::remove_var("AMAEBI_COPILOT_URL"),
                }
                match &self.old_token {
                    Some(v) => std::env::set_var("AMAEBI_COPILOT_TOKEN", v),
                    None => std::env::remove_var("AMAEBI_COPILOT_TOKEN"),
                }
            }
        }
    }

    // ---- DaemonState for mock tests -------------------------------------
    //
    // Build a DaemonState whose reqwest Client has `.no_proxy()` so that
    // requests to the localhost mock server bypass any corporate HTTP proxy
    // configured in the test environment.  Using DaemonState::new() would
    // build a client that honours the system proxy, causing requests to
    // http://127.0.0.1:PORT to be forwarded through the proxy and fail.

    async fn mock_daemon_state(mock_url: &str) -> (Arc<DaemonState>, LlmEnvGuard) {
        let guard = LlmEnvGuard::set(mock_url);

        let http = reqwest::Client::builder()
            .no_proxy() // bypass corporate proxy; mock runs on localhost
            .build()
            .expect("mock reqwest client");

        let db_path = crate::memory_db::db_path().expect("db_path");
        let conn = tokio::task::spawn_blocking(move || crate::memory_db::init_db(&db_path))
            .await
            .unwrap()
            .expect("init_db");

        let db = Arc::new(std::sync::Mutex::new(conn));
        let compacting = Arc::new(std::sync::Mutex::new(
            std::collections::HashSet::<String>::new(),
        ));
        let active = Arc::new(std::sync::Mutex::new(
            std::collections::HashSet::<String>::new(),
        ));
        let tokens = Arc::new(crate::auth::TokenCache::new());

        let spawn_ctx = Arc::new(crate::tools::SpawnContext {
            http: http.clone(),
            db: Arc::clone(&db),
            compacting_sessions: Arc::clone(&compacting),
            tokens: Arc::clone(&tokens),
        });
        let mut executor = crate::tools::LocalExecutor::new();
        executor.spawn_ctx = Some(spawn_ctx);

        let state = Arc::new(DaemonState {
            http,
            tokens,
            executor: Box::new(executor),
            db,
            compacting_sessions: compacting,
            active_sessions: active,
        });

        (state, guard)
    }

    // ---- Tests ----------------------------------------------------------

    /// Action::Llm must write its output to a unique temp file of the form
    /// `/tmp/amaebi_llm_{pid}_{seq}.txt` and set ctx["last_llm_output_file"]
    /// so that downstream Shell stages can read it safely.
    #[tokio::test]
    #[serial]
    async fn llm_stage_writes_output_file() {
        let _home = with_temp_home();
        let mock = MiniMock::start().await;
        mock.enqueue("fix: add caching layer");
        let (state, _env) = mock_daemon_state(&mock.url).await;
        let wf = Workflow {
            name: "file-test".into(),
            stages: vec![Stage::new(
                "gen",
                Action::Llm {
                    prompt: "write a commit message".into(),
                },
            )],
        };

        let summary = execute(
            &wf,
            &state,
            "gpt-4o",
            Context::new(),
            &ResourcePool::empty(),
        )
        .await
        .unwrap();

        assert!(
            summary.contains("fix: add caching layer"),
            "summary: {summary}"
        );
        // The executor writes to /tmp/amaebi_llm_{pid}_{seq}.txt (unique per call).
        // Find the file by scanning /tmp for the PID prefix.
        let pid = std::process::id();
        let prefix = format!("amaebi_llm_{pid}_");
        let found = std::fs::read_dir("/tmp")
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| {
                let name = e.file_name();
                let name = name.to_string_lossy();
                if name.starts_with(&prefix) && name.ends_with(".txt") {
                    std::fs::read_to_string(e.path())
                        .unwrap_or_default()
                        .contains("fix: add caching layer")
                } else {
                    false
                }
            });
        assert!(
            found,
            "LLM output file with expected content not found in /tmp"
        );
        assert_eq!(mock.request_count(), 1);
    }

    /// A Shell stage following an Llm stage must be able to reference
    /// {last_llm_output_file} and read the LLM's output from it.
    /// This is the core regression test for the dev-loop commit-message bug.
    #[tokio::test]
    #[serial]
    async fn shell_stage_reads_last_llm_output_file() {
        let _home = with_temp_home();
        let mock = MiniMock::start().await;
        mock.enqueue("fix: implement new cache");
        let (state, _env) = mock_daemon_state(&mock.url).await;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_owned();

        // Mirrors the dev-loop push-pr pattern: Llm generates text, Shell
        // reads it via {last_llm_output_file}.
        let wf = Workflow {
            name: "commit-msg-test".into(),
            stages: vec![
                Stage::new(
                    "generate-msg",
                    Action::Llm {
                        prompt: "write a commit message".into(),
                    },
                ),
                Stage::new(
                    "use-msg",
                    Action::Shell {
                        command: format!("cp {{last_llm_output_file}} {path}"),
                    },
                )
                .with_on_fail(FailStrategy::Abort),
            ],
        };

        execute(
            &wf,
            &state,
            "gpt-4o",
            Context::new(),
            &ResourcePool::empty(),
        )
        .await
        .unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(
            content.contains("fix: implement new cache"),
            "Shell stage did not read LLM output correctly; got: {content:?}"
        );
    }

    /// When a Shell stage fails and the on_fail strategy is Retry, the
    /// executor must call the LLM with the inject_prompt (containing the error
    /// context) and then re-run the Shell stage.
    #[tokio::test]
    #[serial]
    async fn retry_calls_llm_with_inject_prompt_then_reruns_stage() {
        let _home = with_temp_home();
        let mock = MiniMock::start().await;
        // The workflow has 1 Llm stage + 1 Shell stage with Retry.
        // The Shell stage fails on the first attempt; the Retry calls the LLM
        // (mock response 1) and then re-runs the shell.  The shell succeeds on
        // the second attempt because it detects a sentinel file written by the
        // first failure.
        mock.enqueue("retry fix"); // consumed by inject_prompt LLM call
        let _env = LlmEnvGuard::set(&mock.url);

        let (state, _env) = mock_daemon_state(&mock.url).await;
        let sentinel = tempfile::NamedTempFile::new().unwrap();
        let sentinel_path = sentinel.path().to_str().unwrap().to_owned();
        // Pre-delete so the shell can recreate it.
        std::fs::remove_file(&sentinel_path).ok();

        // Shell logic:
        // - 1st run: sentinel absent → create it → exit 1 (simulate test failure)
        // - 2nd run: sentinel present → exit 0 (simulate fixed code)
        let shell_cmd = format!(
            "if [ -f {sentinel_path} ]; then exit 0; else touch {sentinel_path}; exit 1; fi"
        );

        let wf = Workflow {
            name: "retry-llm-test".into(),
            stages: vec![
                Stage::new("flaky-stage", Action::Shell { command: shell_cmd }).with_on_fail(
                    FailStrategy::Retry {
                        max: 1,
                        inject_prompt: "Tests failed with: {stderr}. Please fix.".into(),
                    },
                ),
            ],
        };

        execute(
            &wf,
            &state,
            "gpt-4o",
            Context::new(),
            &ResourcePool::empty(),
        )
        .await
        .expect("workflow should succeed after retry");

        // The mock must have received exactly 1 request (the inject_prompt LLM call).
        assert_eq!(
            mock.request_count(),
            1,
            "expected 1 LLM call for inject_prompt; got {}",
            mock.request_count()
        );
        // The sentinel file must exist (was created on the first failed run).
        assert!(
            std::path::Path::new(&sentinel_path).exists(),
            "sentinel file missing — first stage run did not execute"
        );
    }

    /// A BenchmarkNoRegression check with threshold=∞ must always pass and
    /// seed ctx["benchmark_baseline"].  A subsequent check with threshold=0.05
    /// must detect a real regression.  This exercises the end-to-end perf_sweep
    /// baseline initialisation fix in a workflow that calls the LLM for context.
    #[tokio::test]
    #[serial]
    async fn perf_sweep_baseline_then_regression_detected() {
        let _home = with_temp_home();
        let mock = MiniMock::start().await;
        // The Llm "analyze" stage consumes one mock response.
        mock.enqueue("- OPT: vectorise inner loop\n- OPT: use SIMD");
        let _env = LlmEnvGuard::set(&mock.url);

        let (state, _env) = mock_daemon_state(&mock.url).await;

        let wf = Workflow {
            name: "perf-regression-test".into(),
            stages: vec![
                // 1. LLM analyzes (produces last_llm_output — not used here)
                Stage::new(
                    "analyze",
                    Action::Llm {
                        prompt: "list optimizations".into(),
                    },
                )
                .with_on_fail(FailStrategy::Abort),
                // 2. Baseline: noop action + threshold=∞ check seeds benchmark_baseline
                Stage::new(
                    "baseline",
                    Action::Shell {
                        command: "true".into(),
                    },
                )
                .with_check(Check::BenchmarkNoRegression {
                    command: r#"echo '{"fps": 100.0}'"#.into(),
                    threshold: f64::INFINITY,
                })
                .with_on_fail(FailStrategy::Abort),
                // 3. Regression check: fps dropped to 60 — should fail
                Stage::new(
                    "bench-after-opt",
                    Action::Shell {
                        command: "true".into(),
                    },
                )
                .with_check(Check::BenchmarkNoRegression {
                    command: r#"echo '{"fps": 60.0}'"#.into(),
                    threshold: 0.05,
                })
                .with_on_fail(FailStrategy::Abort),
            ],
        };

        let result = execute(
            &wf,
            &state,
            "gpt-4o",
            Context::new(),
            &ResourcePool::empty(),
        )
        .await;
        assert!(
            result.is_err(),
            "40% fps regression must be detected; result: {result:?}"
        );
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("fps") || msg.contains("regression"),
            "error must name the regressed metric; got: {msg}"
        );
        // LLM was called exactly once (for the analyze stage).
        assert_eq!(mock.request_count(), 1);
    }
}

// Helper: execute a workflow with a pre-seeded context (bypasses LLM stages
// that would require a real token/daemon in tests).
#[cfg(test)]
async fn execute_with_ctx(
    workflow: &Workflow,
    state: &Arc<DaemonState>,
    model: &str,
    ctx: Context,
    resources: &ResourcePool,
) -> anyhow::Result<String> {
    use crate::copilot::Message;
    use crate::daemon::{build_messages, inject_skill_files};

    let mut messages: Vec<Message> = {
        let mut msgs = build_messages(
            &format!("workflow: {}", workflow.name),
            None,
            &[],
            &[],
            None,
        );
        inject_skill_files(&mut msgs).await;
        msgs
    };

    let mut ctx = ctx;
    let result = run_stages(
        &workflow.stages,
        state,
        model,
        &mut messages,
        &mut ctx,
        resources,
    )
    .await?;
    Ok(result.unwrap_or_default())
}
