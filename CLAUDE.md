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
- The version is validated **per-commit** by comparing HEAD's `Cargo.toml` to HEAD's first-non-merge parent's `Cargo.toml`.  Rule:
  - `YYYY` / `M` = year and month of HEAD's committer date.
  - `N` = parent's `N` + 1 when HEAD is a qualifying commit (`feat(…)` / `feat:` / `fix(…)` / `fix:` / `docs(…)` / `docs:`) AND HEAD lands in the same `(YYYY, M)` as the parent.
  - `N` = parent's `N` (no bump) for non-qualifying prefixes (`refactor`, `chore`, `test`, `revert`, `spike`, …) in the same month.
  - Month rollover: when HEAD's `(YYYY, M)` differs from the parent's, `N` resets to `1` (qualifying) or `0` (non-qualifying), regardless of where the parent stood.  Squash-merge "跳号" gaps (e.g. master jumps from `.2` → `.13` in a single merge) are therefore tolerated — only neighbor-to-neighbor deltas are validated, not the absolute count.
  - Merge commits (2+ parents) are transparent: `--check` walks down via `^2` (GitHub PR convention: `^2` is the incoming branch) to the first non-merge ancestor and validates THAT commit's `Cargo.toml` against its own parent's `Cargo.toml` + delta.  Handles GitHub's synthetic `refs/pull/<N>/merge` commits.
- Each PR author bumps `Cargo.toml` in the commit that introduces qualifying changes.  Run `scripts/next-version.sh` (no flag) to print the expected value for HEAD; run with `--check` to pass/fail.
- CI runs `scripts/next-version.sh --check` on every PR and red-fails if HEAD's `Cargo.toml` disagrees with parent + delta.

## Architecture
- SQLite is the source of truth for `memory_db`, `inbox.db`, and `cron.db`. Do not use `.jsonl` or `.json` files for state storage. Avoid `tempfile` atomic writes for data that belongs in SQLite.
  - Exception: `~/.amaebi/sessions.json` is a lightweight non-authoritative directory→UUID mapping cache. It is intentionally JSON (not SQLite) because it is written by every CLI invocation and must tolerate concurrent readers without WAL overhead.
- Respect the Dual-Channel UX: the CLI is meant to stream output while asynchronously reading `stdin` for `Request::Steer` events.