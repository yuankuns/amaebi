# AGENTS.md

## Startup behavior
- Read this file first.
- For coding workflow rules, read `DEV_WORKFLOW.md`.

## On-demand operations docs (do not preload all)
When task is about deployment/config/troubleshooting, read `OPERATIONS_INDEX.md` first, then only the referenced file(s):
- `DEPLOYMENT.md`
- `CONFIG_REFERENCE.md`
- `RUNBOOK.md`

## Execution policy
- Use `scripts/test.sh` for standard verification.
- Use `scripts/test.sh --docker` for ignored/docker integration tests.
- Keep work on task-specific worktrees; do not modify `master` directly.
