use anyhow::{anyhow, Context as _, Result};
use async_recursion::async_recursion;
use regex::Regex;
use std::sync::Arc;
use tokio::sync::Mutex;

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
    step(&format!("Workflow: {}", workflow.name));

    let mut messages: Vec<Message> = {
        let mut msgs = build_messages(
            &format!("You are executing the workflow: {}.", workflow.name),
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
    .await?;

    Ok(result.unwrap_or_else(|| "Workflow completed.".to_owned()))
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
                        let _ = sh("git stash --include-untracked").await;
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
            // Write to a temp file so Shell stages can reference LLM output safely
            // without shell-injection risk from inlining it into a command string.
            let tmp = "/tmp/amaebi_llm_output.txt";
            let _ = tokio::fs::write(tmp, &text).await;
            ctx.set("last_llm_output_file", tmp);
            run_check(&stage.check, ctx).await?;
            Ok(Some(text))
        }

        Action::Shell { command } => {
            let rendered = ctx.render(command);
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
            concurrency,
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
                run_map_parallel(
                    &items,
                    sub_stages,
                    state,
                    model,
                    ctx,
                    resources,
                    concurrency.as_deref(),
                )
                .await?;
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
    _concurrency_resource: Option<&str>,
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

/// Run stages for one parallel Map item.  Mirrors `run_stages` but takes
/// `&[&Stage]` because the parallel closure can't borrow the owned Vec.
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
            let rendered = ctx.render(command);
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
            let rendered = ctx.render(command);
            let r = sh(&rendered).await?;
            ctx.set("stdout", &r.stdout);
            ctx.set("stderr", &r.stderr);
            if !r.stdout.contains(pattern.as_str()) {
                return Err(anyhow!(
                    "Check failed: output does not contain {:?}\nstdout: {}",
                    pattern,
                    &r.stdout[..r.stdout.len().min(2000)]
                ));
            }
        }

        Check::BenchmarkNoRegression { command, threshold } => {
            let rendered = ctx.render(command);
            let r = sh(&rendered).await?;
            ctx.set("stdout", &r.stdout);

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

pub async fn llm_turn(
    state: &Arc<DaemonState>,
    model: &str,
    messages: &mut Vec<Message>,
    prompt: &str,
) -> Result<String> {
    messages.push(Message::user(prompt.to_owned()));
    let (_, mut steer_rx) = tokio::sync::mpsc::channel::<Option<String>>(1);
    let mut sink = tokio::io::sink();
    let (text, _, _) = run_agentic_loop(
        state,
        model,
        messages.clone(),
        &mut sink,
        &mut steer_rx,
        true,
    )
    .await
    .context("LLM turn failed")?;
    // Note: messages are not returned by run_agentic_loop on this branch;
    // push the user message manually so context accumulates.
    messages.push(crate::copilot::Message::assistant(
        Some(text.clone()),
        vec![],
    ));
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
                let change = (c - b) / b; // negative = regression (lower is better for latency)
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
                        concurrency: None,
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
                    concurrency: None,
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
                    concurrency: Some("slot".into()),
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
        let baseline = r#"{"fps": 100.0, "latency_ms": 10.0}"#;
        let regressed = r#"{"fps": 70.0, "latency_ms": 10.0}"#; // -30% fps

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
