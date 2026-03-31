# CONFIG_REFERENCE.md

Operational config reference (practical subset).

## Test / build related

### `AMAEBI_DEV_IMAGE`
- Purpose: image used by `scripts/test.sh` step 1 (check/test/clippy)
- Default: `amaebi-dev:bookworm-slim`
- Example:
```bash
AMAEBI_DEV_IMAGE=my-custom-dev:latest ./scripts/test.sh
```

### `CARGO`
- Purpose: host cargo path for `scripts/test.sh --docker`
- Default: `$HOME/.cargo/bin/cargo`
- Example:
```bash
CARGO=/usr/local/bin/cargo ./scripts/test.sh --docker
```

## Runtime behavior (spawn/sandbox)

### `AMAEBI_SPAWN_SANDBOX`
- Purpose: child-agent sandbox mode for `spawn_agent`
- Typical values:
  - `docker` (sandboxed)
  - `noop` (host/no isolation, testing only)

### `AMAEBI_SANDBOX_WORKSPACE`
- Purpose: workspace root used by DockerSandbox path checks
- Must be absolute path

### `AMAEBI_MODEL`
- Purpose: default model when request does not specify one
- Note: model/endpoint compatibility depends on provider and route.

## Security notes
- Prefer `docker` sandbox for untrusted or broad tasks.
- Treat `noop` as debug-only.
- Keep credentials outside repo and use least-privilege mounts.
