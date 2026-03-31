# DEPLOYMENT.md

Portable deployment guide for running amaebi on a new Linux server.

## 1) Prerequisites
- Linux x86_64
- Docker installed and running
- Git installed
- Rust toolchain available on host (for `--docker` test path)
- GitHub auth (`gh auth status`) if creating PRs from server

## 2) Clone
```bash
git clone git@github.com:yuankuns/amaebi.git
cd amaebi
```

## 3) Build dev/test images
```bash
./scripts/build-dev-image.sh
./scripts/build-sandbox-image.sh
```

Expected images:
- `amaebi-dev:bookworm-slim` (for `scripts/test.sh`)
- `amaebi-sandbox:bookworm-slim` (for Docker integration tests)

## 4) Baseline verification
```bash
./scripts/test.sh
./scripts/test.sh --docker
```

## 5) Agent-supervised development bootstrap
```bash
git fetch origin
git worktree add ../amaebi-wt/<task-name> -b feat/<task-name> origin/master
cd ../amaebi-wt/<task-name>
```

Run coding + test workflow:
```bash
# coding step (agent/coder)
scripts/dev.sh <task-name> "<prompt>"

# verification step
scripts/test.sh
scripts/test.sh --docker
```

## 6) Smoke test checklist
- [ ] `scripts/build-dev-image.sh` succeeds
- [ ] `scripts/test.sh` passes
- [ ] `scripts/test.sh --docker` passes
- [ ] `docker ps` healthy, no daemon errors

## Notes
- `scripts/test.sh` logs are written to `logs/test-YYYY-MM-DD-HHMMSS.log`.
- Use `AMAEBI_DEV_IMAGE` to override the dev image if needed.
