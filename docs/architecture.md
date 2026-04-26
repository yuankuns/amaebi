# Architecture

A deeper look at how the pieces fit together. See the top-level README for
the one-paragraph summary.

## Two-process model

```
┌──────────┐    Unix socket      ┌───────────────┐    HTTPS/SSE    ┌──────────────────┐
│  Client  │ ◄─────────────────► │    Daemon     │ ◄─────────────► │  Copilot API  or │
│ amaebi   │  /tmp/amaebi.sock   │  (persistent) │                 │  Amazon Bedrock  │
│ ask/chat │                     └───────────────┘                 └──────────────────┘
└──────────┘                             │
                                         ▼
                                 ┌───────────────┐
                                 │ Tool Executor │
                                 │ shell · tmux  │
                                 │ read · edit   │
                                 │ spawn_agent   │
                                 └───────────────┘
```

The **client** is one of:

- `amaebi ask` — short-lived; one request, stream the reply, exit.
- `amaebi chat` — interactive REPL on a long-lived socket connection.
- `amaebi acp` — ACP stdio agent for Zed / Claude Code integration. Routes
  memory reads and writes through the daemon socket so only the daemon ever
  writes to SQLite.

The **daemon** (`amaebi daemon`) is a single persistent process. It owns:

- The HTTPS connection to Bedrock or the Copilot API, including token
  caching.
- The tool executor (shell, read/edit file, tmux control, spawn_agent).
- SQLite connections for `memory.db`, `inbox.db`, `cron.db`, `tasks.db`.
- The cron scheduler (1-minute tick).
- The supervision loop for `/claude` runs (see
  [supervision.md](supervision.md)).
- Pane and resource leases.

The socket is `/tmp/amaebi.sock` by default (`DEFAULT_SOCKET` in
`src/cli.rs:3`). One daemon per user.

## `~/.amaebi/` file inventory

| Path | Format | Purpose |
|------|--------|---------|
| `memory.db` | SQLite WAL | Per-session conversation history, FTS5-indexed |
| `inbox.db` | SQLite WAL | Results from detached and cron tasks |
| `cron.db` | SQLite WAL | Scheduled cron job definitions + `last_run` |
| `tasks.db` | SQLite WAL | Task notebook leases and verdict history (`--tag`) |
| `sessions.json` | JSON, 0600 | Directory → session UUID mapping |
| `tmux-state.json` | JSON, flocked | Pane leases (status, tag, worktree, heartbeat) |
| `resource-state.json` | JSON, flocked | Resource lease table |
| `resources.toml` | TOML | Resource pool definition (hand-edited) |
| `config.json` | JSON | TTL overrides + model aliases |
| `AGENTS.md` | Markdown | Standing agent guidelines (injected each turn) |
| `SOUL.md` | Markdown | Personality directives (injected each turn) |
| `worktrees/` | dir | Per-task git worktrees for `/claude` |

All SQLite files use WAL mode and 0600 permissions.

### SQLite vs. JSON

Source of truth is SQLite for `memory_db`, `inbox.db`, `cron.db`, and
`tasks.db`. JSON is reserved for files that must tolerate frequent concurrent
readers without WAL overhead:

- `sessions.json` — written by every CLI invocation; a lightweight
  non-authoritative directory→UUID cache.
- `tmux-state.json` — protected by `flock` on `tmux-state.lock`; readable by
  the dashboard and by every `/claude` acquisition check.
- `resource-state.json` — protected by `flock` on `resource-state.lock`.

Atomic writes use `rename(2)` after staging to a temp file in the same
directory; no `tempfile` crate usage. The project convention (see
`/home/yuankuns/amaebi/CLAUDE.md`) is: state goes into SQLite unless the
concurrent-reader pattern demands JSON.

## Worktree layout

```
~/.amaebi/worktrees/<repo>/<tag>-<uuid8>/
```

- `<repo>` — basename of the originating git repo root.
- `<tag>` — the `/claude --tag` value, or an auto-generated short label.
- `<uuid8>` — 8 random alphanumeric chars (`src/daemon.rs:2919`) so parallel
  tasks with the same tag never collide.

The worktree is created on a new branch named `<tag>-<uuid8>`. If the user
passes `--worktree <existing-path>` the auto-create is skipped.

## `/claude` request lifecycle

```
┌───────────────────────────────────────────────────────────────────┐
│  Client:  parse  /claude flags                                    │
│             ↓                                                     │
│           GenerateTag (Haiku, if --tag not given)                 │
│             ↓                                                     │
│           send Request::ClaudeLaunch                              │
└──────────────────────────┬────────────────────────────────────────┘
                           │ Unix socket frame
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│  Daemon:                                                          │
│                                                                   │
│    validate flags  (--resume-pane vs --worktree / --resource)     │
│      ↓                                                            │
│    acquire task notebook lease   (tasks.db, if --tag)             │
│      ↓                                                            │
│    acquire pane lease            (tmux-state.json)                │
│      ↓                                                            │
│    create worktree               (unless --worktree given)        │
│      ↓                                                            │
│    acquire resource leases       (resource-state.json, all-or-    │
│                                   nothing, canonical order)      │
│      ↓                                                            │
│    render env vars + prompt_hint; export into pane shell          │
│      ↓                                                            │
│    spawn `claude` in pane, inject task description                │
│      ↓                                                            │
│    enter supervision loop        (WAIT / STEER / DONE)            │
│      ↓                                                            │
│    on any exit:                                                   │
│      release_supervised_panes()                                   │
│      release_task_leases_for_holder()                             │
└───────────────────────────────────────────────────────────────────┘
```

Validation at the client (parse-time) rejects impossible combinations early:
`--resume-pane` + `--worktree`, and `--resume-pane` + `--resource` (see
`src/client.rs:433` and `src/client.rs:446`). The daemon re-validates and
rolls back partial acquisitions if any later step fails.

Cleanup at supervision exit is unconditional. See [supervision.md](
supervision.md) for the release guarantees.

## Model routing

The daemon accepts `provider/model` strings and dispatches to one of two
provider backends (`src/provider.rs`):

- `bedrock/<model>` or a known Bedrock alias → `src/bedrock.rs`
- `copilot/<model>` or a bare Copilot model → `src/copilot.rs`

Unprefixed names first consult `~/.amaebi/config.json` user aliases, then
the built-in Bedrock alias table (`src/provider.rs:64`), falling back to
Copilot.

The default (`DEFAULT_MODEL` at `src/provider.rs:105`) is
`claude-sonnet-4.6[1m]`. The `[1m]` suffix opts into Bedrock's 1M-context
beta. Copilot ignores the suffix.

## Concurrency and leases

- **Pane leases** (`src/pane_lease.rs`) — 24 h TTL, flocked JSON, one writer
  at a time. Supervision heartbeats refresh the lease; daemon crashes let the
  TTL reclaim path take over.
- **Resource leases** (`src/resource_lease.rs`) — same TTL
  (`LEASE_TTL_SECS = 86_400` at `src/resource_lease.rs:59`). All-or-nothing
  acquisition in canonical `(class, name)` order so two callers cannot
  deadlock.
- **Task notebook leases** (`src/tasks.rs`) — one live lease per
  `(repo_dir, tag)` pair, bound to the supervision holder id.

All three follow the same pattern: acquire at entry, release on every exit
path, TTL reclaim if the holder crashes.

## Client protocol (summary)

The socket speaks newline-delimited JSON. Requests and responses are
typed enums defined in `src/ipc.rs`. The main request types:

- `Ask { prompt, session_id, model, ... }` — one-shot or chat turn.
- `ClaudeLaunch { tasks, session_id, repo_dir }` — launch one or more
  supervised Claude Code panes.
- `SupervisePanes { panes, model }` — re-attach supervision to existing
  panes (used by the `chat` REPL).
- `Memory { op, ... }` — memory reads/writes (routed via the daemon so only
  one process ever writes to `memory.db`).

Responses are streamed as a sequence of frames ending in `Response::Done`.
