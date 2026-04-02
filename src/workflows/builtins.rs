/// Built-in workflow definitions.
///
/// Each function constructs a `Workflow` from the static primitives.
/// No LLM is involved in flow decisions — only in content (coding, analysis).
use super::{Action, Check, FailStrategy, Stage, Workflow};

// ---------------------------------------------------------------------------
// 1. dev_loop — develop → test → PR → review → loop
// ---------------------------------------------------------------------------

/// Supervise Claude through a full development cycle:
/// develop → test → (fix on fail) → create PR → @copilot review →
/// (fix on comment) → done.
///
/// `test_cmd` is the **single quality gate** before the PR is created.
/// It should run everything the project requires: unit tests, formatting,
/// linting, type-checking, etc.  Use a script rather than a bare `cargo test`
/// so that formatting and lint failures are caught in the same retry loop.
///
/// Example `scripts/test.sh` for a Rust project:
/// ```bash
/// #!/usr/bin/env bash
/// set -e
/// cargo test
/// cargo fmt --check
/// cargo clippy -- -D warnings
/// ```
pub fn dev_loop(
    task: &str,
    test_cmd: &str, // comprehensive quality-gate script, e.g. "scripts/test.sh"
    max_test_retries: usize,
    max_review_retries: usize,
) -> Workflow {
    Workflow {
        name: "dev-loop".into(),
        stages: vec![
            // Phase 1: Claude develops
            Stage::new(
                "develop",
                Action::Llm {
                    prompt: format!(
                        "Complete the following development task. \
                         Make sure the code compiles before finishing.\n\n{task}"
                    ),
                },
            )
            .with_on_fail(FailStrategy::Abort),
            // Phase 2: full quality gate — tests, formatting, linting, etc.
            // test_cmd should be a script that runs all checks in one shot.
            Stage::new(
                "test",
                Action::Shell {
                    command: test_cmd.to_owned(),
                },
            )
            .with_on_fail(FailStrategy::Retry {
                max: max_test_retries,
                inject_prompt: "The test suite failed (exit code {exit_code}).\n\nstderr:\n```\n{stderr}\n```\n\nPlease fix the code so all tests pass.".into(),
            }),
            // Phase 3: commit + push + create PR (code-guaranteed)
            Stage::new(
                "commit-and-pr",
                Action::Llm {
                    prompt: "Generate a concise commit message for the changes just made. \
                             Output only the commit message text, nothing else."
                        .into(),
                },
            )
            .with_on_fail(FailStrategy::Abort),
            Stage::new(
                "push-pr",
                Action::Shell {
                    // -F reads the commit message from the file written by the preceding
                    // Llm stage (last_llm_output_file), avoiding shell injection.
                    // `gh pr create --fill` is idempotent: if a PR already exists for this
                    // branch the command exits non-zero, so we fall back to printing the
                    // existing PR URL instead.
                    // GIT_TERMINAL_PROMPT=0 and GH_PROMPT_DISABLED=1 prevent
                    // git/gh from blocking on credential prompts in non-interactive
                    // environments; auth failures surface immediately instead.
                    // Copilot is added as a reviewer only when the PR is first
                    // created; subsequent fix-and-push cycles skip the add since
                    // Copilot is already assigned on the existing PR.
                    command: "GIT_TERMINAL_PROMPT=0 GH_PROMPT_DISABLED=1 \
                              git add -A && \
                              git commit -F {last_llm_output_file} && \
                              git push && \
                              if gh pr create --fill 2>/dev/null; then \
                                gh pr edit --add-reviewer Copilot 2>/dev/null || true; \
                              else \
                                gh pr view --json url -q '.url'; \
                              fi"
                        .into(),
                },
            )
            .with_on_fail(FailStrategy::Abort),
            // Phase 6: rename the current tmux window to "pr<N>" so the user
            // can identify which PR this session is tracking at a glance.
            // Uses FailStrategy::Skip so the workflow continues even when tmux
            // is not available or gh cannot resolve the PR number.
            Stage::new(
                "rename-tmux",
                Action::Shell {
                    command: "PR=$(gh pr view --json number -q '.number' 2>/dev/null) && \
                              [ -n \"$PR\" ] && \
                              tmux rename-window \"pr${PR}\" 2>/dev/null"
                        .into(),
                },
            )
            .with_on_fail(FailStrategy::Skip),
            // Phase 7: wait for @copilot review then act on the result.
            // The action polls every 30 s (up to 60 retries = 30 min) until a
            // review appears.  This prevents the stage from exiting early when
            // Copilot has not yet finished reviewing.
            Stage::new(
                "review",
                Action::Shell {
                    // Wait 60 s after the push before the first poll so Copilot
                    // has time to pick up the review request.  Then poll every
                    // 30 s (up to 60 more times) for a review submitted after
                    // the most recent commit, ignoring any stale earlier reviews.
                    command: "GH_PROMPT_DISABLED=1; \
                              echo 'Waiting 60s for Copilot to start reviewing...'; \
                              sleep 60; \
                              COMMIT_DATE=$(gh pr view --json commits \
                                --jq '.commits[-1].committedDate' 2>/dev/null \
                                || echo '1970-01-01T00:00:00Z'); \
                              for i in $(seq 60); do \
                                state=$(gh pr view --json reviews \
                                  --jq --arg cd \"$COMMIT_DATE\" \
                                  '[.reviews[] | select(.submittedAt > $cd)] | last | .state // \"\"' \
                                  2>/dev/null); \
                                [ -n \"$state\" ] && echo \"$state\" && break; \
                                echo \"Waiting for review ($i/60)...\"; \
                                sleep 30; \
                              done"
                        .into(),
                },
            )
            .with_check(Check::Contains {
                // Only look at reviews submitted after the most recent commit so
                // that a stale CHANGES_REQUESTED does not count as a new review.
                command: "gh pr view --json reviews,commits --jq \
                          '(.commits[-1].committedDate) as $cd | \
                           [.reviews[] | select(.submittedAt > $cd)] | last | \
                           .state + \": \" + (.body // \"\")'"
                    .into(),
                pattern: "APPROVED".into(),
            })
            .with_on_fail(FailStrategy::Retry {
                max: max_review_retries,
                inject_prompt:
                    "The code review requires changes. Review feedback:\n\n{stdout}\n\n\
                     Please address the comments and push an updated commit."
                        .into(),
            }),
        ],
    }
}

// ---------------------------------------------------------------------------
// 2. perf_sweep — list optimizations → try each serially → summarize
// ---------------------------------------------------------------------------

/// Supervise Claude through a systematic performance optimization sweep:
/// read docs → list optimization points → try each (serial, bench-gated) →
/// revert regressions → summarize.
///
/// Serial because later optimizations build on earlier ones.
pub fn perf_sweep(
    target: &str,
    docs_content: &str,        // pre-loaded file contents
    bench_cmd: &str,           // must output JSON with numeric metrics
    regression_threshold: f64, // e.g. 0.05 = 5% regression triggers revert
) -> Workflow {
    Workflow {
        name: "perf-sweep".into(),
        stages: vec![
            // Phase 1: read docs, list optimization points
            Stage::new(
                "analyze",
                Action::Llm {
                    prompt: format!(
                        "You are a performance engineer analyzing {target}.\n\n\
                         Here is the relevant code and documentation:\n\n{docs_content}\n\n\
                         List all promising performance optimization opportunities.\n\
                         Format: one per line, starting with `- OPT: `\n\
                         Be specific. Only list; do not implement yet."
                    ),
                },
            )
            .with_on_fail(FailStrategy::Abort),
            // Phase 2: capture baseline benchmark (code-guaranteed).
            // The Check runs bench_cmd once and saves the result to ctx["benchmark_baseline"]
            // so that later BenchmarkNoRegression checks have a real baseline to compare against.
            // threshold=f64::INFINITY means this check never fails — it only initialises the baseline.
            Stage::new(
                "baseline",
                Action::Shell {
                    command: "true".into(), // noop; benchmark is run exactly once by the check below
                },
            )
            .with_check(Check::BenchmarkNoRegression {
                command: bench_cmd.to_owned(),
                threshold: f64::INFINITY,
            })
            .with_on_fail(FailStrategy::Abort),
            // Phase 3: try each optimization point serially
            Stage::new(
                "optimize-each",
                Action::Map {
                    parse: r"- OPT: (.+)".into(),
                    parallel: false, // serial: each builds on previous
                    stages: vec![
                        // 3a: Claude implements the optimization
                        Stage::new(
                            "implement",
                            Action::Llm {
                                prompt: "Implement the following optimization. \
                                         Make only this change, nothing else:\n\n{item}"
                                    .into(),
                            },
                        )
                        .with_on_fail(FailStrategy::RevertAndSkip),
                        // 3b: compile check (code-guaranteed)
                        Stage::new(
                            "compile",
                            Action::Shell {
                                // No 2>&1: build errors go to stderr, surfaced via {stderr}.
                                command: "cargo build".into(),
                            },
                        )
                        .with_on_fail(FailStrategy::RevertAndSkip),
                        // 3c: benchmark with regression check (code-guaranteed).
                        // The action is a noop; bench_cmd runs exactly once inside the
                        // BenchmarkNoRegression check.  Previously the action also ran
                        // bench_cmd, which caused two runs per optimization and compared
                        // the check's run against itself rather than against the baseline.
                        Stage::new(
                            "benchmark",
                            Action::Shell {
                                command: "true".into(), // noop; benchmark runs via the check below
                            },
                        )
                        .with_check(Check::BenchmarkNoRegression {
                            command: bench_cmd.to_owned(),
                            threshold: regression_threshold,
                        })
                        .with_on_fail(FailStrategy::Retry {
                            max: 1,
                            inject_prompt:
                                "The optimization caused a regression: {regression_summary}\n\n\
                                 Please analyze why and fix it. \
                                 If there is no viable fix, reply with SKIP."
                                    .into(),
                        }),
                        // 3d: commit successful optimization.
                        // printf safely builds the commit message, keeping {item} out of
                        // the shell command string and preventing injection.
                        Stage::new(
                            "commit",
                            Action::Shell {
                                command: r#"git add -A && git commit -m "$(printf 'perf: %s' {item})" || true"#
                                    .into(),
                            },
                        )
                        .with_on_fail(FailStrategy::Skip),
                    ],
                },
            )
            .with_on_fail(FailStrategy::Abort),
            // Phase 4: summarize results
            Stage::new(
                "summarize",
                Action::Llm {
                    prompt: "All optimization points have been attempted. \
                         Please write a concise performance optimization report covering:\n\
                         1. What was done and what improved\n\
                         2. What was reverted and why\n\
                         3. Recommended next steps"
                        .into(),
                },
            )
            .with_on_fail(FailStrategy::Abort),
        ],
    }
}

// ---------------------------------------------------------------------------
// 3. bug_fix — list bugs → fix each in parallel → summarize
// ---------------------------------------------------------------------------

/// Supervise Claude fixing a list of bugs in parallel.
/// Each bug is independent: fix → test → PR → review.
///
/// `list_cmd` is the shell command used to fetch open bugs.  It defaults to
/// `gh issue list -R {repo} --label bug …` for real use but can be overridden
/// in tests (e.g. `echo '- BUG #1: test bug'`) to avoid requiring GitHub auth.
pub fn bug_fix(
    repo: &str, // e.g. "yuankun/amaebi"
    test_cmd: &str,
    max_retries: usize,
    list_cmd: Option<&str>,
) -> Workflow {
    // Shell-quote the repo argument so that owner/repo values with special
    // characters cannot escape the single-quoted shell context.
    let repo_q = format!("'{}'", repo.replace('\'', r"'\''"));
    let list_command = list_cmd.map(|s| s.to_owned()).unwrap_or_else(|| {
        format!(
            "gh issue list -R {repo_q} --label bug --json number,title,body \
             --jq '.[] | \"- BUG #\" + (.number|tostring) + \": \" + .title'"
        )
    });
    Workflow {
        name: "bug-fix".into(),
        stages: vec![
            // Phase 1: fetch open bug issues (code-guaranteed)
            Stage::new(
                "list-bugs",
                Action::Shell {
                    command: list_command,
                },
            )
            .with_on_fail(FailStrategy::Abort),
            // Phase 2: LLM parses the list into individual items
            Stage::new(
                "parse-bugs",
                Action::Llm {
                    prompt: "Here is a list of open bugs:\n\n{stdout}\n\n\
                             For each bug, output one line: `- BUG: #N title`\n\
                             This will be parsed automatically."
                        .into(),
                },
            )
            .with_on_fail(FailStrategy::Abort),
            // Phase 3: fix each bug in parallel
            Stage::new(
                "fix-each",
                Action::Map {
                    parse: r"- BUG: (.+)".into(),
                    parallel: true, // bugs are independent
                    stages: vec![
                        // 3a: checkout a new branch for this bug.
                        // Detects the default branch dynamically instead of
                        // hardcoding "master" (repos may use "main" or other names).
                        // NOTE: parallel workers share the same working tree; if
                        // true isolation is needed, use per-bug git worktrees instead.
                        Stage::new(
                            "branch",
                            Action::Shell {
                                command: "DEFAULT=$(git remote show origin 2>/dev/null \
                                            | awk '/HEAD branch/{print $NF}'); \
                                          DEFAULT=${DEFAULT:-main}; \
                                          git checkout \"$DEFAULT\" && \
                                          git checkout -b fix/bug-{item_index}"
                                    .into(),
                            },
                        )
                        .with_on_fail(FailStrategy::Skip),
                        // 3b: Claude fixes the bug
                        Stage::new(
                            "fix",
                            Action::Llm {
                                prompt:
                                    "Fix the following bug. Make targeted changes only.\n\n{item}"
                                        .into(),
                            },
                        )
                        .with_on_fail(FailStrategy::Skip),
                        // 3c: run tests (code-guaranteed)
                        Stage::new(
                            "test",
                            Action::Shell {
                                command: test_cmd.to_owned(),
                            },
                        )
                        .with_on_fail(FailStrategy::Retry {
                            max: max_retries,
                            inject_prompt:
                                "Tests failed while fixing {item}.\n\nstderr:\n```\n{stderr}\n```\n\
                                 Please fix the code."
                                    .into(),
                        }),
                        // 3d: push + PR (code-guaranteed).
                        // printf builds the commit message safely so {item} is
                        // never interpreted by the shell.
                        Stage::new(
                            "pr",
                            Action::Shell {
                                command: r#"GIT_TERMINAL_PROMPT=0 GH_PROMPT_DISABLED=1 \
                                          git add -A && \
                                          git commit -m "$(printf 'fix: %s' {item})" && \
                                          git push -u origin HEAD && \
                                          (gh pr create --fill 2>/dev/null || gh pr view --json url -q '.url')"#
                                    .into(),
                            },
                        )
                        .with_on_fail(FailStrategy::Skip),
                    ],
                },
            )
            .with_on_fail(FailStrategy::Abort),
            // Phase 4: summary
            Stage::new(
                "summarize",
                Action::Llm {
                    prompt: "All bugs have been processed. \
                             Please summarize: which were fixed, which failed, and why."
                        .into(),
                },
            )
            .with_on_fail(FailStrategy::Abort),
        ],
    }
}

// ---------------------------------------------------------------------------
// 4. tune_sweep — list tuning directions → run experiments in parallel →
//    respect resource constraints → summarize
// ---------------------------------------------------------------------------

/// Supervise a hyperparameter / configuration tuning sweep.
/// LLM lists directions; shell runs experiments in parallel, constrained by
/// available resources (e.g. 2 GPUs).
pub fn tune_sweep(
    target: &str,
    context: &str,    // code/docs context
    run_cmd: &str,    // e.g. "python train.py --config {item_config}"
    result_cmd: &str, // e.g. "cat results/{item_index}/metrics.json"
    resource: &str,   // e.g. "gpu"
) -> Workflow {
    Workflow {
        name: "tune-sweep".into(),
        stages: vec![
            // Phase 1: LLM lists tuning directions
            Stage::new(
                "plan",
                Action::Llm {
                    prompt: format!(
                        "You are planning a tuning sweep for {target}.\n\n\
                         Context:\n{context}\n\n\
                         List all promising tuning directions (hyperparameters, \
                         architectural choices, config options).\n\
                         Format: one per line, starting with `- TUNE: `\n\
                         Be specific enough that each can be run as an independent experiment."
                    ),
                },
            )
            .with_on_fail(FailStrategy::Abort),
            // Phase 2: LLM generates a config file for each direction
            Stage::new(
                "generate-configs",
                Action::Map {
                    parse: r"- TUNE: (.+)".into(),
                    parallel: true,
                    stages: vec![Stage::new(
                        "write-config",
                        Action::Llm {
                            prompt:
                                "Generate a configuration for this tuning experiment:\n\n{item}\n\n\
                                     Write the config to `configs/tune_{item_index}.json`."
                                    .into(),
                        },
                    )
                    .with_on_fail(FailStrategy::Skip)],
                },
            )
            .with_on_fail(FailStrategy::Abort),
            // Phase 3: run experiments in parallel, resource-constrained
            Stage::new(
                "run-experiments",
                Action::Map {
                    parse: r"- TUNE: (.+)".into(),
                    parallel: true,
                    stages: vec![
                        Stage::new(
                            "run",
                            Action::Shell {
                                command: format!("mkdir -p results/{{item_index}} && {run_cmd}"),
                            },
                        )
                        .with_requires(resource) // hold resource during experiment
                        .with_on_fail(FailStrategy::Skip),
                        Stage::new(
                            "collect",
                            Action::Shell {
                                command: result_cmd.to_owned(),
                            },
                        )
                        .with_on_fail(FailStrategy::Skip),
                    ],
                },
            )
            .with_on_fail(FailStrategy::Abort),
            // Phase 4: LLM summarizes all results
            Stage::new(
                "summarize",
                Action::Llm {
                    prompt: format!(
                        "All tuning experiments for {target} have completed. \
                         The results are in the `results/` directory.\n\n\
                         Please:\n\
                         1. Read and compare the results across all experiments\n\
                         2. Identify the best configuration and why\n\
                         3. Recommend next steps"
                    ),
                },
            )
            .with_on_fail(FailStrategy::Abort),
        ],
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dev_loop_push_pr_uses_llm_output_file() {
        let wf = dev_loop("test task", "cargo test", 3, 3);
        let stage = wf
            .stages
            .iter()
            .find(|s| s.name == "push-pr")
            .expect("push-pr stage missing");
        if let Action::Shell { command } = &stage.action {
            assert!(
                command.contains("{last_llm_output_file}"),
                "push-pr must use {{last_llm_output_file}} to avoid shell injection; got: {command}"
            );
            assert!(
                !command.contains("/tmp/amaebi_commit_msg.txt"),
                "push-pr must not reference the old hard-coded temp path"
            );
        } else {
            panic!("push-pr must be a Shell action");
        }
    }

    #[test]
    fn dev_loop_review_check_includes_body() {
        let wf = dev_loop("test task", "cargo test", 3, 3);
        let stage = wf
            .stages
            .iter()
            .find(|s| s.name == "review")
            .expect("review stage missing");
        if let Some(Check::Contains { command, pattern }) = &stage.check {
            assert_eq!(pattern, "APPROVED");
            // The check command must output the review body so that inject_prompt
            // has useful content when the check fails.
            assert!(
                command.contains("body"),
                "review check command must include .body to surface review comments; got: {command}"
            );
        } else {
            panic!("review stage must have a Contains check");
        }
    }

    #[test]
    fn perf_sweep_baseline_initialises_benchmark_baseline() {
        let wf = perf_sweep("target", "", "bench --output json", 0.05);
        let stage = wf
            .stages
            .iter()
            .find(|s| s.name == "baseline")
            .expect("baseline stage missing");
        // Action must be a noop so bench_cmd only runs once (inside the check).
        if let Action::Shell { command } = &stage.action {
            assert_eq!(
                command, "true",
                "baseline action must be noop; benchmark runs via the check"
            );
        } else {
            panic!("baseline must be a Shell action");
        }
        // Must have a BenchmarkNoRegression check to initialise ctx["benchmark_baseline"].
        assert!(
            matches!(stage.check, Some(Check::BenchmarkNoRegression { .. })),
            "baseline stage must have BenchmarkNoRegression check to initialise the baseline"
        );
    }

    #[test]
    fn perf_sweep_benchmark_runs_only_via_check() {
        let wf = perf_sweep("target", "", "bench --output json", 0.05);
        let optimize_each = wf
            .stages
            .iter()
            .find(|s| s.name == "optimize-each")
            .expect("optimize-each stage missing");
        if let Action::Map { stages, .. } = &optimize_each.action {
            let benchmark = stages
                .iter()
                .find(|s| s.name == "benchmark")
                .expect("benchmark sub-stage missing");
            if let Action::Shell { command } = &benchmark.action {
                assert_eq!(
                    command, "true",
                    "benchmark stage action must be noop to avoid running bench_cmd twice"
                );
            } else {
                panic!("benchmark must be a Shell action");
            }
            assert!(
                matches!(benchmark.check, Some(Check::BenchmarkNoRegression { .. })),
                "benchmark stage must have BenchmarkNoRegression check"
            );
        } else {
            panic!("optimize-each must be a Map action");
        }
    }
}
