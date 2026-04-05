# Claude Code Project Rules for Amaebi

## Git Branching Discipline
- ALWAYS fetch and rebase/checkout from the ABSOLUTE LATEST `master` before starting any new feature or bugfix.
- Never work on a stale branch to avoid massive merge conflicts in core files like `daemon.rs`.
- NEVER push directly to `master`. All changes must go through a pull request. Create a feature branch, push it, and open a PR via `gh pr create`.
- After opening a PR, check Copilot/human review comments with `gh api repos/yuankuns/amaebi/pulls/{N}/comments`. For each comment: if it has merit, fix it and push; if it doesn't, reply explaining why the current design is correct.

## CI and Pre-commit Checks
- Before EVERY commit that modifies Rust code, you MUST run the following "素质三连" (Triple Check) locally to ensure CI passes:
  1. `cargo test` - Ensure all tests pass.
  2. `cargo fmt --check` - If this fails, run `cargo fmt` to fix formatting.
  3. `cargo clippy -- -D warnings` - Ensure there are zero clippy warnings.
- Do not push code if `cargo fmt --check` or `cargo clippy` fails. GitHub Actions CI will reject it.

## Architecture
- SQLite is the source of truth for `memory_db`, `inbox.db`, and `cron.db`. Do not use `.jsonl` or `.json` files for state storage. Avoid `tempfile` atomic writes for data that belongs in SQLite.
  - Exception: `~/.amaebi/sessions.json` is a lightweight non-authoritative directory→UUID mapping cache. It is intentionally JSON (not SQLite) because it is written by every CLI invocation and must tolerate concurrent readers without WAL overhead.
- Respect the Dual-Channel UX: the CLI is meant to stream output while asynchronously reading `stdin` for `Request::Steer` events.