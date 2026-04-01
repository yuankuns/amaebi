/// Built-in workflow definitions.
///
/// Each function constructs a `Workflow` from the static primitives.
/// No LLM is involved in flow decisions — only in content (coding, analysis).
use super::{Action, Check, FailStrategy, Stage, Workflow};

// ---------------------------------------------------------------------------
// 1. dev_loop — develop → test → PR → review → loop
// ---------------------------------------------------------------------------

/// Supervise Claude through a full development cycle:
/// develop → run tests → (fix on fail) → create PR → @copilot review →
/// (fix on comment) → done.
pub fn dev_loop(
    task: &str,
    test_cmd: &str, // e.g. "scripts/test.sh" or "cargo test"
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
            // Phase 2: run tests (code-guaranteed)
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
            // Phase 3: cargo fmt (code-guaranteed)
            Stage::new(
                "fmt",
                Action::Shell {
                    command: "cargo fmt".into(),
                },
            )
            .with_on_fail(FailStrategy::Abort),
            // Phase 4: clippy (code-guaranteed)
            Stage::new(
                "clippy",
                Action::Shell {
                    command: "cargo clippy -- -D warnings 2>&1".into(),
                },
            )
            .with_on_fail(FailStrategy::Retry {
                max: 2,
                inject_prompt:
                    "cargo clippy reported warnings:\n```\n{stderr}\n```\nPlease fix them.".into(),
            }),
            // Phase 5: commit + push + create PR (code-guaranteed)
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
                    command: "git add -A && \
                              git commit -m \"$(cat /tmp/amaebi_commit_msg.txt)\" && \
                              git push && \
                              gh pr create --fill"
                        .into(),
                },
            )
            .with_on_fail(FailStrategy::Abort),
            // Phase 6: wait for @copilot review (code polls GitHub API)
            Stage::new(
                "review",
                Action::Shell {
                    command: "sleep 30 && gh pr view --json reviews,state -q \
                              '.reviews[-1].state // \"PENDING\"'"
                        .into(),
                },
            )
            .with_check(Check::Contains {
                command: "gh pr view --json reviews -q '.reviews[-1].state'".into(),
                pattern: "APPROVED".into(),
            })
            .with_on_fail(FailStrategy::Retry {
                max: max_review_retries,
                inject_prompt:
                    "The Copilot reviewer left the following comments. Please address them:\n\n\
                     {stdout}"
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
            // Phase 2: capture baseline benchmark (code-guaranteed)
            Stage::new(
                "baseline",
                Action::Shell {
                    command: bench_cmd.to_owned(),
                },
            )
            .with_on_fail(FailStrategy::Abort),
            // Phase 3: try each optimization point serially
            Stage::new(
                "optimize-each",
                Action::Map {
                    parse: r"- OPT: (.+)".into(),
                    parallel: false, // serial: each builds on previous
                    concurrency: None,
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
                                command: "cargo build 2>&1".into(),
                            },
                        )
                        .with_on_fail(FailStrategy::RevertAndSkip),
                        // 3c: benchmark with regression check (code-guaranteed)
                        Stage::new(
                            "benchmark",
                            Action::Shell {
                                command: bench_cmd.to_owned(),
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
                        // 3d: commit successful optimization
                        Stage::new(
                            "commit",
                            Action::Shell {
                                command: "git add -A && git commit -m 'perf: {item}' || true"
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
pub fn bug_fix(
    repo: &str, // e.g. "yuankuns/amaebi"
    test_cmd: &str,
    max_retries: usize,
) -> Workflow {
    Workflow {
        name: "bug-fix".into(),
        stages: vec![
            // Phase 1: fetch open bug issues (code-guaranteed)
            Stage::new(
                "list-bugs",
                Action::Shell {
                    command: format!(
                        "gh issue list -R {repo} --label bug --json number,title,body \
                         --jq '.[] | \"- BUG #\" + (.number|tostring) + \": \" + .title'"
                    ),
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
                    concurrency: None,
                    stages: vec![
                        // 3a: checkout a new branch for this bug
                        Stage::new(
                            "branch",
                            Action::Shell {
                                command: "git checkout master && \
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
                        // 3d: push + PR (code-guaranteed)
                        Stage::new(
                            "pr",
                            Action::Shell {
                                command: "git add -A && \
                                          git commit -m 'fix: {item}' && \
                                          git push -u origin HEAD && \
                                          gh pr create --fill"
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
                    concurrency: None, // config generation is just LLM, no resource needed
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
                    concurrency: Some(resource.to_owned()), // e.g. "gpu" limits concurrency
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
