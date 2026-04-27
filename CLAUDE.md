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
- `Cargo.toml` `version` is calendar-versioned `YYYY.M.N` (no leading zeros on `M`/`N`).  The version is computed from the git history of the checked-out branch (HEAD) — on a PR branch that includes the PR's own commits, on `master` it is the master history.  Rule:
  - `YYYY` / `M` = year and month of the latest commit on HEAD
  - `N` = count of `feat(...)` / `fix(...)` / `docs(...)` commits on HEAD within the current `(YYYY, M)` month
  - Other prefixes (`refactor`, `chore`, `test`, `revert`, `spike`, merges…) do not bump `N`
  - A new month automatically resets `N` to 0 (and increments to 1 on the first qualifying commit of the month)
- Each PR author bumps `Cargo.toml` in the PR itself so it matches what master will look like after the PR lands.  Run `scripts/next-version.sh` to see what the version should be, then edit `Cargo.toml`.
- CI runs `scripts/next-version.sh --check` on every PR and red-fails if `Cargo.toml` disagrees with what the commit history implies.

## Architecture
- SQLite is the source of truth for `memory_db`, `inbox.db`, and `cron.db`. Do not use `.jsonl` or `.json` files for state storage. Avoid `tempfile` atomic writes for data that belongs in SQLite.
  - Exception: `~/.amaebi/sessions.json` is a lightweight non-authoritative directory→UUID mapping cache. It is intentionally JSON (not SQLite) because it is written by every CLI invocation and must tolerate concurrent readers without WAL overhead.
- Respect the Dual-Channel UX: the CLI is meant to stream output while asynchronously reading `stdin` for `Request::Steer` events.