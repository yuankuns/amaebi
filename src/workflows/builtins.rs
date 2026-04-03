/// Built-in workflow definitions.
///
/// Each function constructs a `Workflow` from the static primitives.
/// No LLM is involved in flow decisions — only in content (coding, analysis).
use anyhow::{bail, Result};

use super::{Action, Check, FailStrategy, Stage, Workflow};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Shell command that resolves all unresolved PR review conversations and
/// re-requests a Copilot review.  Idempotent: safe to run when there are
/// no unresolved threads or no PR yet.
fn resolve_and_request_review_cmd() -> String {
    // Step 1: resolve every unresolved review thread via GraphQL.
    // Step 2: re-request @copilot review so it evaluates the new commit.
    r#"
GH_VER=$(gh --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
GH_MAJOR=$(echo "$GH_VER" | cut -d. -f1)
GH_MINOR=$(echo "$GH_VER" | cut -d. -f2)
if [ -z "$GH_VER" ] || [ "$GH_MAJOR" -lt 2 ] || { [ "$GH_MAJOR" -eq 2 ] && [ "$GH_MINOR" -lt 87 ]; }; then
  echo "ERROR: gh >= 2.87.0 required for @copilot review (found ${GH_VER:-none})" >&2
  exit 1
fi
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null) || true
PR_NUM=$(gh pr view --json number -q .number 2>/dev/null) || true
if [ -n "$REPO" ] && [ -n "$PR_NUM" ]; then
  OWNER=$(echo "$REPO" | cut -d/ -f1)
  NAME=$(echo "$REPO" | cut -d/ -f2)
  THREADS=$(gh api graphql -f query="
    { repository(owner: \"$OWNER\", name: \"$NAME\") {
        pullRequest(number: $PR_NUM) {
          reviewThreads(first: 100) {
            nodes { id isResolved }
    } } } }" --jq '.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false) | .id' 2>/dev/null)
  for tid in $THREADS; do
    gh api graphql -f query="mutation { resolveReviewThread(input: {threadId: \"$tid\"}) { thread { isResolved } } }" >/dev/null 2>&1 || true
  done
  gh pr edit "$PR_NUM" --add-reviewer "@copilot" 2>/dev/null || true
fi
echo "resolve-and-request-review done"
"#
    .to_owned()
}

// ---------------------------------------------------------------------------
// 1. dev_loop — develop → test → PR → review → loop
// ---------------------------------------------------------------------------

/// Supervise Claude through a full development cycle:
/// develop → test script → (fix on fail) → create PR → @copilot review →
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
            // Phase 1: Delegate coding to external Claude pane.
            Stage::new(
                "develop",
                Action::Delegate {
                    prompt: format!(
                        "Complete the following development task. \
                         Make sure the code compiles and all tests pass before finishing. \
                         Commit and push your changes when done.\n\n{task}"
                    ),
                },
            )
            .with_on_fail(FailStrategy::Abort),
            // Phase 2: run test script (fmt + clippy + tests in one shot)
            Stage::new(
                "test",
                Action::Shell {
                    command: test_cmd.to_owned(),
                },
            )
            .with_on_fail(FailStrategy::Retry {
                max: max_test_retries,
                inject_prompt: "The test script failed (exit code {exit_code}).\n\nstderr:\n```\n{stderr}\n```\n\nPlease fix the code so all checks pass, then commit and push.".into(),
            }),
            // Phase 3: push + create PR.
            // Claude may have already committed during develop, so tolerate
            // clean working tree.
            Stage::new(
                "push-pr",
                Action::Shell {
                    command: "git add -A && \
                              if ! git diff --cached --quiet; then \
                                git commit -m 'feat: dev-loop automated commit'; \
                              fi && \
                              git push && \
                              (gh pr create --fill 2>&1 || true)"
                        .into(),
                },
            )
            .with_on_fail(FailStrategy::Abort),
            // Phase 4: resolve conversations + re-request review + poll.
            //
            // On the first run, resolve/re-request are no-ops.  On retries
            // (after the LLM fixed code and pushed), this resolves the old
            // CHANGES_REQUESTED conversations and re-requests review so
            // Copilot evaluates the new commit before we poll.
            Stage::new(
                "review",
                Action::Shell {
                    command: format!(
                        "{} && sleep 60 && gh pr view --json reviews,state -q \
                         '.reviews[-1].state // \"PENDING\"'",
                        resolve_and_request_review_cmd().trim()
                    ),
                },
            )
            .with_check(Check::Contains {
                command: "gh pr view --json reviews --jq \
                          '.reviews[-1] | .state + \": \" + (.body // \"\")'"
                    .into(),
                pattern: "APPROVED".into(),
            })
            .with_on_fail(FailStrategy::Retry {
                max: max_review_retries,
                inject_prompt:
                    "The code review requires changes. Review feedback:\n\n{stdout}\n\n\
                     Please address ALL the review comments, then:\n\
                     1. Run the test suite to make sure everything passes\n\
                     2. Commit and push the fixes\n\
                     After you push, I will resolve the conversations and re-request review."
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
                    resource_hint: None,
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
                        // 3d: generate a commit message without embedding
                        // LLM-controlled text directly into shell.
                        Stage::new(
                            "compose-commit-message",
                            Action::Llm {
                                prompt: "Write a git commit message for this optimization.\n\
                                         Use exactly one subject line prefixed with `perf: `.\n\
                                         Do not use code fences or surrounding quotes.\n\n\
                                         Optimization:\n{item}"
                                    .into(),
                            },
                        )
                        .with_on_fail(FailStrategy::Skip),
                        // 3e: commit via file input to avoid shell injection.
                        Stage::new(
                            "commit",
                            Action::Shell {
                                command:
                                    "git add -A && git commit -F {last_llm_output_file} || true"
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
/// `repo` must match the `owner/name` pattern (e.g. `"yuankuns/amaebi"`).
/// Returns an error on invalid values to prevent shell injection.
pub fn bug_fix(
    repo: &str, // e.g. "yuankuns/amaebi"
    test_cmd: &str,
    max_retries: usize,
) -> Result<Workflow> {
    // Validate repo matches owner/name to prevent shell injection.
    if repo != "."
        && !repo
            .chars()
            .all(|c| c.is_alphanumeric() || c == '/' || c == '-' || c == '_' || c == '.')
    {
        bail!("invalid repo: must be 'owner/name' or '.', got: {repo:?}");
    }

    // When repo is ".", omit `-R` so gh infers from the current git remote.
    let gh_issue_cmd = if repo == "." {
        "gh issue list --label bug --json number,title,body \
         --jq '.[] | \"- BUG #\" + (.number|tostring) + \": \" + .title'"
            .to_owned()
    } else {
        format!(
            "gh issue list -R {repo} --label bug --json number,title,body \
             --jq '.[] | \"- BUG #\" + (.number|tostring) + \": \" + .title'"
        )
    };

    // Detect default branch dynamically instead of hardcoding `master`.
    let checkout_default_branch =
        "git checkout $(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null \
         | sed 's|refs/remotes/origin/||' || echo master)";

    Ok(Workflow {
        name: "bug-fix".into(),
        stages: vec![
            // Phase 1: fetch open bug issues (code-guaranteed)
            Stage::new(
                "list-bugs",
                Action::Shell {
                    command: gh_issue_cmd,
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
                    // Serial: sub-stages do git operations (checkout, commit,
                    // push) in the same working tree so they cannot run in
                    // parallel without racing on the index and HEAD.
                    parallel: false,
                    resource_hint: None,
                    stages: vec![
                        // 3a: checkout a new branch for this bug
                        Stage::new(
                            "branch",
                            Action::Shell {
                                command: format!(
                                    "{checkout_default_branch} && \
                                     git checkout -b fix/bug-{{item_index}}"
                                ),
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
                        // Uses {item_index} instead of {item} to avoid
                        // shell injection from LLM-controlled text.
                        Stage::new(
                            "pr",
                            Action::Shell {
                                command: "git add -A && \
                                          git commit -m 'fix: bug {item_index}' && \
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
    })
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
                    resource_hint: None, // config generation is just LLM, no resource needed
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
                    resource_hint: Some(resource.to_owned()), // e.g. "gpu" — metadata hint only
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
    fn dev_loop_develop_uses_delegate() {
        let wf = dev_loop("test task", "cargo test", 3, 3);
        let stage = wf
            .stages
            .iter()
            .find(|s| s.name == "develop")
            .expect("develop stage missing");
        assert!(
            matches!(&stage.action, Action::Delegate { .. }),
            "develop stage must be Action::Delegate"
        );
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

    #[test]
    fn perf_sweep_commit_uses_file_not_item() {
        let wf = perf_sweep("target", "", "bench --output json", 0.05);
        let optimize_each = wf
            .stages
            .iter()
            .find(|s| s.name == "optimize-each")
            .expect("optimize-each stage missing");
        if let Action::Map { stages, .. } = &optimize_each.action {
            let commit = stages
                .iter()
                .find(|s| s.name == "commit")
                .expect("commit sub-stage missing");
            if let Action::Shell { command } = &commit.action {
                assert!(
                    command.contains("{last_llm_output_file}"),
                    "commit must use {{last_llm_output_file}} to avoid shell injection; got: {command}"
                );
                assert!(
                    !command.contains("{item}"),
                    "commit must not embed {{item}} in shell; got: {command}"
                );
            } else {
                panic!("commit must be a Shell action");
            }
        } else {
            panic!("optimize-each must be a Map action");
        }
    }

    #[test]
    fn bug_fix_pr_uses_item_index_not_item() {
        let wf = bug_fix("owner/repo", "cargo test", 3).unwrap();
        let fix_each = wf
            .stages
            .iter()
            .find(|s| s.name == "fix-each")
            .expect("fix-each stage missing");
        if let Action::Map { stages, .. } = &fix_each.action {
            let pr = stages
                .iter()
                .find(|s| s.name == "pr")
                .expect("pr sub-stage missing");
            if let Action::Shell { command } = &pr.action {
                assert!(
                    command.contains("{item_index}"),
                    "pr commit must use {{item_index}}; got: {command}"
                );
                assert!(
                    !command.contains("{item}"),
                    "pr commit must not embed {{item}} in shell; got: {command}"
                );
            } else {
                panic!("pr must be a Shell action");
            }
        } else {
            panic!("fix-each must be a Map action");
        }
    }

    #[test]
    fn bug_fix_repo_validation() {
        // Valid repos should not error.
        bug_fix("owner/repo", "true", 1).unwrap();
        bug_fix(".", "true", 1).unwrap();
        bug_fix("my-org/my_repo.rs", "true", 1).unwrap();
    }

    #[test]
    fn bug_fix_rejects_shell_metacharacters_in_repo() {
        let result = bug_fix("owner/repo; rm -rf /", "true", 1);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("invalid repo"),
            "error should mention 'invalid repo', got: {msg}"
        );
    }

    #[test]
    fn bug_fix_dot_repo_omits_dash_r() {
        let wf = bug_fix(".", "true", 1).unwrap();
        let list_bugs = wf
            .stages
            .iter()
            .find(|s| s.name == "list-bugs")
            .expect("list-bugs stage missing");
        if let Action::Shell { command } = &list_bugs.action {
            assert!(
                !command.contains("-R"),
                "repo='.' should omit -R flag; got: {command}"
            );
        } else {
            panic!("list-bugs must be a Shell action");
        }
    }

    #[test]
    fn bug_fix_serial_not_parallel() {
        let wf = bug_fix("owner/repo", "true", 1).unwrap();
        let fix_each = wf
            .stages
            .iter()
            .find(|s| s.name == "fix-each")
            .expect("fix-each stage missing");
        if let Action::Map { parallel, .. } = &fix_each.action {
            assert!(
                !parallel,
                "fix-each Map must be serial (parallel=false) because sub-stages do git operations"
            );
        } else {
            panic!("fix-each must be a Map action");
        }
    }

    #[test]
    fn bug_fix_branch_detects_default_branch() {
        let wf = bug_fix("owner/repo", "true", 1).unwrap();
        let fix_each = wf
            .stages
            .iter()
            .find(|s| s.name == "fix-each")
            .expect("fix-each stage missing");
        if let Action::Map { stages, .. } = &fix_each.action {
            let branch = stages
                .iter()
                .find(|s| s.name == "branch")
                .expect("branch sub-stage missing");
            if let Action::Shell { command } = &branch.action {
                assert!(
                    command.contains("symbolic-ref"),
                    "branch stage must detect default branch dynamically; got: {command}"
                );
                assert!(
                    !command.contains("git checkout master"),
                    "branch stage must not hardcode 'git checkout master'; got: {command}"
                );
            } else {
                panic!("branch must be a Shell action");
            }
        } else {
            panic!("fix-each must be a Map action");
        }
    }
}
