# /claude — supervised Claude Code subprocesses

`/claude` is a slash command (not a subcommand: you type it inside `amaebi
chat` or pass it as the prompt to `amaebi ask`). It does four things:

1. Creates a git worktree under `~/.amaebi/worktrees/<repo>/<tag>-<uuid8>/`.
2. Allocates a tmux pane and starts `claude` (the Claude Code TUI) inside it.
3. Emits `Response::PaneAssigned { tag, pane_id, session_id, worktree, resources }`.
4. Client synthesises a single `[launched]` user turn that hands the
   description + pane context to the chat agentic loop, which then drives
   supervision via the standard tmux/shell/file tools plus the lifecycle
   tool `task_done`.  See
   [design/claude-chat-takeover.md](design/claude-chat-takeover.md) for
   the frozen contract.

Parsing lives in `src/client.rs:280` (`parse_claude`); launch handling is in
`src/daemon.rs` (`handle_claude_launch`).

## Quick examples

```text
/claude "fix the flaky test in tests/auth_test.rs"
/claude "investigate the memory leak" "add a regression test"     # two parallel tasks
/claude --tag kernel-opt "optimise the attention kernel"
/claude --resume-pane %41                                          # reuse existing pane
/claude --resource sim-9902 "run the new testbench"
```

## Flags

| Flag | Purpose |
|------|---------|
| `--worktree <path>` | Use an existing worktree instead of auto-creating one |
| `--cwd <path>` | Client working directory (used to locate the git repo for auto-worktree) |
| `--no-enter` | Inject the task description but do not press Enter |
| `--tag <name>` | Persistent task notebook identifier |
| `--resume-pane <pane_id>` | Reuse an existing pane (e.g. `%41`) — inject task into existing claude |
| `--resource <spec>` | Acquire a resource lease and inject its env vars |
| `--resource-timeout <secs>` | How long to wait for a busy resource before failing |

Flags may appear in any order. Everything after `--` is treated as a task
description.

### `--worktree <path>`

Explicit worktree path. Skips the auto-create step. The path is canonicalised
at parse time (`src/client.rs:340`) so relative and symlinked paths work.

Mutually exclusive with `--resume-pane`: reusing a pane inherits its existing
worktree, so an explicit `--worktree` would conflict.

### `--cwd <path>`

The client's working directory, used by the daemon to locate the enclosing git
repo when auto-creating a worktree. Defaults to wherever you ran `amaebi`.

### `--no-enter`

Inject the task description into the pane but don't press Enter. Useful when
you want to edit the prompt interactively before Claude starts.

### `--tag <name>` (PR #128)

Opt the pane into the **task notebook**, a persistent `(repo_dir, tag)` keyed
record in `~/.amaebi/tasks.db`. Every supervision verdict (`WAIT`, `STEER`,
`DONE`, summary text) is appended; the task description (`<desc>`) is
persisted on first launch.

Later runs with the same tag recover the description and recent verdicts:

```text
# First launch: record the desc, start fresh
/claude --tag refactor-auth "refactor the auth module, keep signatures stable"

# Days later, from a crashed pane or new terminal:
/claude --tag refactor-auth
# Daemon re-injects the stored desc and prior verdicts into the supervision
# prompt, so the new Claude pane knows what was already tried.
```

Exactly one live session per `(repo_dir, tag)` at a time — the notebook holds
a 24 h lease. Inspect or force-release via:

```bash
amaebi tag list
amaebi tag release <tag>
```

### `--resume-pane <pane_id>` (PR #124)

Reuse the `claude` already running in a tmux pane instead of launching a new
one. The daemon injects the new task description directly into the existing
conversation — resume means *continue*, so prior hard constraints, todo
lists, and partial state are preserved. If stale context needs pruning, type
`/compact` into the pane manually. This is the tier-1 reuse path: no new
worktree, no 30-second `claude` startup.

```text
/claude --resume-pane %41 "now run the benchmarks on the optimised kernel"
```

Pane ids come from `amaebi dashboard` (Panes panel) or `tmux list-panes -F
'#{pane_id}'`.

Mutually exclusive at parse time with both `--worktree` and `--resource` (see
`src/client.rs:433` and `src/client.rs:446`):

- `--worktree` would conflict with the pane's existing worktree.
- `--resource` cannot inject env vars into a running shell (the Claude TUI
  intercepts every keystroke as chat input). Until PR α lands, the combination
  returns a parse error.

Task description is optional with `--resume-pane`: if omitted, the daemon
re-uses the description from the pane's previous lease.

Only one task per `--resume-pane` invocation — one pane, one task.

### `--resource <spec>` (PR #126)

Acquire a lease from the resource pool (see
[resource-pool.md](resource-pool.md)). Three spec forms:

| Spec | Meaning |
|------|---------|
| `sim-9902` | Named — acquire this specific resource |
| `class:simulator` | Class — any idle resource of class `simulator` |
| `any:simulator` | Class alias (`any:` and `class:` are equivalent) |

Parsing lives in `src/resource_lease.rs:196` (`ResourceRequest::parse`). Repeat
`--resource` for multiple resources; acquisition is **all-or-nothing** with
canonical ordering (see `src/resource_lease.rs:27`) so concurrent callers
cannot deadlock on partial holds.

On success the daemon:

1. Marks each lease `Busy` in `~/.amaebi/resource-state.json`.
2. Renders `env` entries from `resources.toml` (with `{name}` and
   `{metadata_key}` placeholders) and exports them into the pane shell
   before `claude` launches.
3. Prepends the resource's `prompt_hint` to the task description.

```text
/claude --resource sim-9902 --resource class:gpu "run the full benchmark"
```

### `--resource-timeout <secs>`

Wait up to N seconds for a busy resource to free up, then fail. Default (no
flag / 0) is fail-fast. The daemon uses `tokio::sync::Notify` to wake waiters
when leases are released.

## The tier-1 reuse path

When `--resume-pane` is given, the daemon takes a fast path:

1. Look up the pane in `~/.amaebi/tmux-state.json` — must exist and be live.
2. Inject the new task description into the existing claude conversation.
3. Resume supervision under the same pane lease.

No worktree creation, no new `claude` process spawn, no API key dance, and
no automatic `/compact` — the prior conversation is preserved so hard
constraints, todo lists, and partial state carry over.  Use this for
iterative refinement: "run the benchmarks now", "try the other branch".  If
stale context ever needs pruning, type `/compact` into the pane manually.

## Where worktrees live

```
~/.amaebi/worktrees/<repo>/<tag>-<uuid8>/
```

- `<repo>` is derived from the repo root basename.
- `<tag>` is the `--tag` value (or a short label auto-generated by the
  `GenerateTag` daemon round-trip, which asks a small model to distil a tag
  from the description).
- `<uuid8>` is a random 8-char suffix so repeated tasks with the same tag do
  not collide (`src/daemon.rs:2919`).

The worktree is created on a new branch named `<tag>-<uuid8>`. If you pass
`--worktree <existing-path>` the daemon skips the create step.

## Combining flags

```text
# Named resource + tag, two independent tasks
/claude --tag perf --resource sim-9902 "profile the hot path" "write the report"

# Class-based acquisition with a wait
/claude --resource class:gpu --resource-timeout 600 "train the small model"

# Resume a pane, no new task desc (reuses the one from the prior launch)
/claude --resume-pane %41
```

The usage string printed on a parse error lives at `src/client.rs:290` — read
it for the current canonical grammar.

## Exit and cleanup

Five release paths funnel into one idempotent Rust function
(`release_held_entry`):

- LLM calls `task_done(pane_id, summary)` — the normal "task verified
  complete" exit.  Streams a `Response::TaskReleased` frame to the chat
  with the summary + pane tail + worktree status.
- User types `/release %54 [--clean] [--summary "..."]` or
  `/release all [--clean]` — same release frame, but the summary is
  whatever the user typed (or empty for `all`).
- Chat exits cleanly (Ctrl-D / Ctrl-C×2) — drains every held pane.
- Socket drops abnormally (chat crash, terminal kill) — same drain,
  archived to `~/.amaebi/inbox.db` as an `[abandoned]` report.
- 24 h lease TTL — disaster backstop for daemon crash etc.

Every path releases pane + resource + task-notebook leases.  The tmux
pane and its `claude` process are never killed; the worktree is kept
unless `--clean` is passed.  See
[design/claude-chat-takeover.md](design/claude-chat-takeover.md) for the
full contract.
