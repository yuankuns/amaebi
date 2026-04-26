# Resource pool

> **Note:** Sections marked *(requires PR α merged)* describe behaviour from
> the in-flight PR α. Until α lands, `--resume-pane` combined with
> `--resource` returns a parse error at `src/client.rs:446`, and there is no
> AGENTS.md injection from `resources.toml`.

Parallel `/claude` sessions often need to share externally-arbitrated
resources — GPU slots, simulator ports, serial devices — where naive
concurrency would break things. The resource pool provides leased, all-or-
nothing acquisition with env-var injection.

## Two files

| File | Role | Format |
|------|------|--------|
| `~/.amaebi/resources.toml` | Pool definition (static) | TOML, hand-edited |
| `~/.amaebi/resource-state.json` | Lease table (runtime) | JSON, flocked |

Lock file: `~/.amaebi/resource-state.lock` (exclusive `flock` via `fs2`).

## `resources.toml` format

One entry per resource:

```toml
[[resource]]
name = "sim-9902"
class = "simulator"

[resource.metadata]
host = "sim-host-02"
port = "9902"

[resource.env]
SIM_HOST = "{host}"
SIM_PORT = "{port}"
SIM_NAME = "{name}"

prompt_hint = """
You have been assigned simulator {name} at {host}:{port}.
Export SIM_HOST and SIM_PORT before running the testbench, and release the
port when you finish.
"""

[[resource]]
name = "sim-9903"
class = "simulator"

[resource.metadata]
host = "sim-host-02"
port = "9903"

[resource.env]
SIM_HOST = "{host}"
SIM_PORT = "{port}"
```

Fields (`src/resource_lease.rs:66`):

- `name` — unique identifier. Duplicates fail validation at load time.
- `class` — grouping key for class-based acquisition.
- `metadata` — machine-readable key/value; feeds placeholder substitution in
  `env` and `prompt_hint`.
- `env` — environment variables to export into the pane shell. Values may
  reference `{metadata_key}` and `{name}`.
- `prompt_hint` — text prepended to the task description. *(requires PR α
  merged)* The same text is also written to `<worktree>/AGENTS.md` on first
  launch, in a marker-bracketed block so user content is preserved.

Missing file is equivalent to an empty pool (no error; see
`src/resource_lease.rs:94`).

## `--resource` spec forms

Three forms, parsed by `ResourceRequest::parse` (`src/resource_lease.rs:196`):

| Spec | Request kind | Behaviour |
|------|--------------|-----------|
| `sim-9902` | `Named` | Acquire this specific resource; fail/wait if busy |
| `class:simulator` | `Class` | Acquire any idle resource of class `simulator` |
| `any:simulator` | `Class` (alias) | Same as `class:` |

The explicit `class:` / `any:` prefix is required: a pool can legitimately
contain both a resource named `gpu` and a class named `gpu`, and the parser
refuses to guess.

## All-or-nothing acquisition

`acquire_all` (`src/resource_lease.rs`) either returns every requested lease
or rolls back partial holds. Requests are processed in canonical `(class,
name)` order so two concurrent callers cannot interleave into a cycle.

```text
/claude --resource sim-9902 --resource class:gpu "run the benchmark"
```

If `sim-9902` is free but every GPU is busy, neither is acquired and the
caller either fails (no timeout) or waits (`--resource-timeout`). Waiters are
woken via `tokio::sync::Notify` when leases are released.

## Env var injection

On successful acquisition the daemon:

1. Renders the `env` map for each lease, substituting `{name}` and
   `{metadata_key}` placeholders.
2. Exports the resulting variables into the pane shell **before** `claude`
   launches. (`src/daemon.rs:1658` renders the prompt hint; the env pass
   happens just upstream.)

Example: with the TOML above, `/claude --resource sim-9902` gives the pane
shell:

```bash
export SIM_HOST=sim-host-02
export SIM_PORT=9902
export SIM_NAME=sim-9902
```

These persist for the pane's lifetime. They cannot be applied to an
already-running `claude` — which is why `--resume-pane` is rejected in
combination with `--resource` (see `src/client.rs:446`).

## 24-hour TTL and orphan detection

Every `Busy` lease carries a `heartbeat_at` timestamp. If it is not refreshed
within `LEASE_TTL_SECS` (86,400 s = 24 h, `src/resource_lease.rs:59`) the
lease is treated as effective-`Idle` by the next acquirer — `effective_status`
at `src/resource_lease.rs:159` does the check. This is how a crashed daemon's
leases are eventually reclaimed.

`amaebi resource list` shows every resource with its class, effective status,
and current holder (pane id + tag + session). Entries present in
`resource-state.json` but missing from `resources.toml` are printed as
`orphaned: not in resources.toml`.

## AGENTS.md injection *(requires PR α merged)*

When PR α lands, the first `/claude --resource <spec>` in a worktree will
write the resource's `prompt_hint` into `<worktree>/AGENTS.md` inside a
marker-bracketed block. Subsequent `/compact` cycles inside Claude Code
preserve `AGENTS.md` verbatim, so the assigned resource information survives
context compression — something the prompt-hint prepend alone does not
guarantee.

The marker block is delimited so that user content in `AGENTS.md` above or
below the block is left untouched. Re-launching in the same worktree with a
different resource rewrites only the marked section.

## `--resume-pane` + `--resource` *(requires PR α merged)*

Currently a parse error (see `src/client.rs:446`). PR α introduces a
re-acquire-on-resume path that re-claims the pane's prior resource set
without attempting env-var injection (the prior `export`s are still in the
shell).
