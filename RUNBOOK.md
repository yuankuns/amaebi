# RUNBOOK.md

Day-2 operations for amaebi.

## Quick commands

### Full verification
```bash
./scripts/test.sh
./scripts/test.sh --docker
```

### Rebuild images
```bash
./scripts/build-dev-image.sh
./scripts/build-sandbox-image.sh
```

### Inspect latest test log
```bash
ls -t logs/test-*.log | head -1 | xargs -r tail -n 120
```

## Common failures

### 1) `Dev image ... not found`
Fix:
```bash
./scripts/build-dev-image.sh
```

### 2) `amaebi-sandbox:bookworm-slim not found`
Fix:
```bash
./scripts/build-sandbox-image.sh
```

### 3) Docker tests fail on host cargo path
Fix:
```bash
which cargo
CARGO=$(which cargo) ./scripts/test.sh --docker
```

### 4) Model 400 / unsupported model
- Check provider+endpoint compatibility.
- Try known-good default model first.
- Avoid assuming model availability across different gateways.

## Incident response (minimal)
1. Stop risky automation/actions.
2. Capture logs (`logs/test-*.log`, daemon logs, docker logs).
3. Reproduce in clean worktree.
4. Roll back to last green commit if needed.

## Rollback
```bash
git checkout <last-known-good-commit>
./scripts/test.sh
./scripts/test.sh --docker
```
