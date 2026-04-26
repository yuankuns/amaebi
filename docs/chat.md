# amaebi ask & amaebi chat

Two ways to talk to the daemon.

- `amaebi ask "<prompt>"` — one-shot. Streams the reply, exits when the task
  finishes.
- `amaebi chat [<prompt>]` — interactive REPL. Stays open until you hit Ctrl-D
  or type an empty line.

Both share the same session UUID per working directory, so a `chat` started in
`~/projectX` and a later `ask` in the same directory see the same history
(subject to the session TTL, see `amaebi session --help`).

## Basic usage

```bash
amaebi ask "explain the retry logic in bedrock.rs"
amaebi chat
# > explain the retry logic in bedrock.rs
# [...response...]
# > how does it compare to the copilot implementation?
```

## Model override

```bash
amaebi ask --model bedrock/claude-sonnet-4.6 "hello"
amaebi chat --model copilot/gpt-4o
```

The default is `claude-sonnet-4.6[1m]` (see `src/provider.rs:105`). The `[1m]`
suffix opts into Bedrock's 1M-context beta. Copilot ignores the suffix.

Resolution order: `--model` flag → `AMAEBI_MODEL` env → default.

### Model aliases

`~/.amaebi/config.json` can define short aliases:

```json
{
  "model_aliases": {
    "opus": "bedrock/claude-opus-4.7",
    "fast": "copilot/gpt-4o-mini"
  }
}
```

Aliases are resolved in one hop (no chaining). Built-in aliases in
`src/provider.rs` (e.g. `claude-sonnet-4.6`) take precedence on name conflict.

## Detached runs (`ask --detach`)

```bash
amaebi ask "run the full test suite and summarise failures" --detach
# → [queued: a1b2c3d4]   exits in < 100 ms
```

The daemon continues the task autonomously. When it finishes, the result lands
in `amaebi inbox`. `--detach` and `--resume` are mutually exclusive.

## Session resume

Resume a prior session for this directory with full chronological history
(bypasses the normal sliding-window cap):

```bash
amaebi ask --resume "continue the migration from yesterday"
amaebi chat -r
```

Bare `-r` / `--resume` with no value opens an interactive picker listing this
directory's recent sessions, newest first. Pick a number or press Enter to
cancel.

Pass a specific UUID (or a prefix ≥ 4 chars) with `=` syntax so it is not
confused with the prompt positional:

```bash
amaebi ask -r=abc123de "what was the conclusion on that auth bug?"
amaebi chat --resume=abc123de
```

The TTY check in `src/session.rs:414` rejects bare `--resume` when stdin/stdout
is not a terminal (no picker possible).

## Mid-response steering

Both `ask` and `chat` read stdin while the model streams. Type a line + Enter
and the correction is injected before the model's next turn:

```
$ amaebi ask "refactor the auth module and add tests"
[tool] read_file src/auth.rs
[tool] shell_command cargo test
> wait, keep the old function signature      ← you type this mid-stream
[steer] correction injected
[tool] edit_file src/auth.rs
Done.
```

In `chat`, Ctrl-C during a response does the same thing: it interrupts
generation and drops you back at the `>` prompt where you can type a
correction.

## Exit

- `ask`: Ctrl-D detaches the stream (task continues in the background, result
  goes to inbox). Ctrl-C twice exits immediately.
- `chat`: empty line or Ctrl-D exits. First Ctrl-C interrupts the current
  response; second Ctrl-C exits.

## History editor

The prompt input uses `rustyline` for line editing — arrow keys, Emacs
bindings, CJK-aware cursor movement, and persistent history across
sessions. This replaced a hand-rolled editor; see commit `0bb236a`.

## Session identity

Every working directory gets a stable UUID stored in `~/.amaebi/sessions.json`
(JSON, 0600). The daemon keys `memory.db` by that UUID so per-project history
never leaks across projects.

```bash
amaebi session show         # UUID for cwd
amaebi session new          # forget the current context; start fresh
amaebi session status       # list every dir → UUID mapping
amaebi session set-tier persistent  # 24h TTL for this directory
```

Sessions expire after 30 minutes of inactivity by default. Override per tier
or per directory in `~/.amaebi/config.json` (see the `ttl_minutes` field at
`src/config.rs:45`).
