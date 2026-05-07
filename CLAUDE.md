# Claude Code Project Rules for Amaebi

## Git Branching Discipline
- ALWAYS fetch and rebase/checkout from the ABSOLUTE LATEST `master` before starting any new feature or bugfix.
- Never work on a stale branch to avoid massive merge conflicts in core files like `daemon.rs`.
- NEVER push directly to `master`. All changes must go through a pull request. Create a feature branch, push it, and open a PR via `gh pr create`.
- After opening a PR, check Copilot/human review comments with `gh pr view {N} --comments` and `gh api repos/$(gh repo view --json nameWithOwner -q .nameWithOwner)/pulls/{N}/comments`. For each comment: if it has merit, fix it and push; if it doesn't, reply explaining why the current design is correct.

## CI and Pre-commit Checks
- Before EVERY commit that modifies Rust code, you MUST run the following "素质三连" (Triple Check) locally to ensure CI passes:
  1. `cargo test` - Ensure all tests pass.
  2. `cargo fmt --check` - If this fails, run `cargo fmt` to fix formatting.
  3. `cargo clippy -- -D warnings` - Ensure there are zero clippy warnings.
- Do not push code if `cargo fmt --check` or `cargo clippy` fails. GitHub Actions CI will reject it.

## Versioning
- `Cargo.toml` `version` is calendar-versioned `YYYY.M.N` (no leading zeros on `M`/`N`).
- The rule depends on whether HEAD is on master or on a feature branch; see `scripts/next-version.sh` header for the full rationale.
  - **On master (HEAD is an ancestor of `origin/master`).**  Expected = parent's `Cargo.toml` + delta(subject): `+1` for qualifying commits (`feat(…)` / `feat:` / `fix(…)` / `fix:` / `docs(…)` / `docs:`), `+0` otherwise (`refactor`, `chore`, `test`, `revert`, `spike`, …).  Squash-merge commits on master are 1-parent, so the PR title's prefix determines the bump.  Month rollover resets `N` to `1` (qualifying) or `0` (non-qualifying).
  - **On a feature branch (non-ancestor of `origin/master`).**  Expected = master's `Cargo.toml` + `1` if **any** non-merge commit in `origin/master..HEAD` has a qualifying subject, else master's `Cargo.toml` unchanged.  Every commit on a single PR shares ONE expected value, so follow-up review fixes don't need a second bump.  Month rollover (HEAD's committer month differs from master's) resets `N` to `1` (qualifying PR) or `0` (non-qualifying PR).
  - **Merge commits**: master MUST be linear — squash or rebase merges only.  A "Create a merge commit" PR subject is `Merge pull request ...` (non-qualifying → expected delta 0), but the merge tree usually contains the PR's bumped `Cargo.toml`, so `--check` would report a false mismatch.  Master-mode therefore fails fast on any 2+-parent commit with a message telling the operator to switch merge style.  Inside a PR branch it's fine: branch mode skips merges via `git log --no-merges`, so merging master back into a branch is transparent.  GitHub's synthetic `refs/pull/<N>/merge` is non-ancestor of master and falls into branch mode where the guard doesn't apply.
- Each PR bumps `Cargo.toml` exactly once (on the first qualifying commit of the branch).  Subsequent review commits on the same PR — whether `fix`, `refactor`, `chore`, or another `feat` — must NOT bump the version; the branch-mode check enforces a single expected value for the whole PR.  Run `scripts/next-version.sh` (no flag) to print the expected value for HEAD; run with `--check` to pass/fail.
- CI runs `scripts/next-version.sh --check` on every PR and red-fails if `Cargo.toml` disagrees with the expected value.  `scripts/next-version.test.sh` is a companion shell-level regression suite for the versioning logic itself.

## Architecture
- SQLite is the source of truth for `memory_db`, `inbox.db`, and `cron.db`. Do not use `.jsonl` or `.json` files for state storage. Avoid `tempfile` atomic writes for data that belongs in SQLite.
  - Exception: `~/.amaebi/sessions.json` is a lightweight non-authoritative directory→UUID mapping cache. It is intentionally JSON (not SQLite) because it is written by every CLI invocation and must tolerate concurrent readers without WAL overhead.
- Respect the Dual-Channel UX: the CLI is meant to stream output while asynchronously reading `stdin` for `Request::Steer` events.