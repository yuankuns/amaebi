# amaebi

A tiny, memory-efficient AI assistant for the terminal, powered by GitHub Copilot.

**amaebi** (甘エビ, sweet shrimp) runs as a lightweight daemon that connects to the GitHub Copilot API. It can run shell commands, read and edit files, interact with tmux panes, schedule autonomous cron jobs, and steer live AI responses mid-flight — all from a single binary under 7 MB.

## Quick Start

```bash
# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build
git clone https://github.com/yuankuns/amaebi.git
cd amaebi
cargo build --release

# Authenticate with GitHub Copilot
./target/release/amaebi auth

# Start the daemon (once; keep it running)
./target/release/amaebi daemon &

# Ask away
./target/release/amaebi ask "what's using the most disk space?"
```

Add the binary to your PATH:

```bash
ln -sf $(pwd)/target/release/amaebi ~/.local/bin/amaebi
```

---

## How It Works

amaebi splits into two processes connected by a Unix socket:

```
┌──────────┐    Unix socket      ┌───────────────┐    HTTPS/SSE    ┌─────────────┐
│  Client  │ ◄─────────────────► │    Daemon     │ ◄─────────────► │ Copilot API │
│ amaebi   │  /tmp/amaebi.sock   │  (persistent) │                 │             │
│   ask    │                     └───────────────┘                 └─────────────┘
└──────────┘                             │
                                         ▼
                                 ┌───────────────┐
                                 │ Tool Executor │
                                 │ shell · tmux  │
                                 │ read · edit   │
                                 └───────────────┘
                                         │
                              ┌──────────┴──────────┐
                              ▼                     ▼
                      ~/.amaebi/memory.db    ~/.amaebi/inbox.db
                      (conversation memory)  (async task results)
```

- **Client** (`amaebi ask`): Short-lived process. Sends the prompt, streams the response to stdout, and reads stdin for mid-flight steering corrections.
- **Daemon** (`amaebi daemon`): Persistent process. Manages the Copilot API connection, token caching, tool execution, SQLite-backed session history, and the cron scheduler.

---

## Dual-Channel UX

### Foreground mode (default)

The CLI stays open and streams output live. You can type corrections mid-flight:

```
$ amaebi ask "refactor the auth module and add tests"
[tool] read_file src/auth.rs
[tool] shell_command cargo test
[tool] edit_file src/auth.rs
> wait, keep the old function signature      ← you type this mid-stream
[steer] correction injected
[tool] edit_file src/auth.rs
Done. Refactored auth module. 3 new tests added, all passing.
[Session: a1b2c3d4-e5f6-7890-abcd-ef1234567890]
```

- Type a line + Enter to inject a correction (steering). The model sees it before its next turn.
- Press **Ctrl+D** to detach — the task continues in the background; result goes to `amaebi inbox`.
- Press **Ctrl+C** twice to exit immediately.

### Detached mode (`--detach`)

```bash
amaebi ask "run the full test suite and summarise failures" --detach
# → [queued: a1b2c3d4]   exits in < 100 ms
```

The daemon runs the task autonomously. When it completes, the result is deposited into the inbox.

---

## Session Management

Every working directory gets a stable UUID stored in `~/.amaebi/sessions.json`. amaebi uses this UUID to isolate per-project conversation history — no manual setup required.

### TTL-based implicit reset

Sessions expire automatically after 30 minutes of inactivity. Come back after lunch and get a fresh context automatically, with no commands needed.

TTL tiers are configurable in `~/.amaebi/config.json`:

```json
{
  "ttl_minutes": {
    "default": 30,
    "/home/user/long-project": 120,
    "ephemeral": 5,
    "persistent": 1440
  }
}
```

### `--resume`: explicit full-history recall

```bash
amaebi ask --resume "continue the migration from yesterday"
# Loads full chronological history for this directory's session UUID
# (bypasses the normal sliding-window cap)

amaebi ask --resume abc123de "what was the conclusion on that auth bug?"
# Loads history for a specific UUID (cross-directory or archived session)
```

`--resume` and `--detach` are mutually exclusive.

### Session subcommands

```bash
amaebi session show       # print the UUID for the current directory
amaebi session new        # generate a fresh UUID (discard old context)
amaebi session status     # list all directory → UUID mappings
amaebi session set-tier <tier>   # set TTL tier (default|ephemeral|persistent)
amaebi session clear      # evict expired sessions from sessions.json
```

---

## Commands

### Core

| Command | Description |
|---------|-------------|
| `amaebi auth` | Authenticate via GitHub Copilot device flow |
| `amaebi daemon` | Start the background daemon |
| `amaebi ask "<prompt>"` | Send a prompt and stream the reply |
| `amaebi ask "<prompt>" --detach` | Submit as a background task |
| `amaebi ask "<prompt>" --resume [uuid]` | Resume with full session history |
| `amaebi models` | List available Copilot models |

### Inbox

Background tasks (detached runs and cron jobs) deposit their results here.

```bash
amaebi inbox list           # list unread reports
amaebi inbox list --all     # include already-read reports
amaebi inbox read <id>      # display a report and mark it read
amaebi inbox mark-read      # mark all unread as read
amaebi inbox clear          # delete all reports
```

A bell notification is printed to stderr at the start of every `amaebi ask` when unread reports are waiting:

```
[🔔 You have 2 unread cron reports. Run `amaebi inbox list` to read.]
```

### Cron

Schedule recurring autonomous tasks. The daemon fires them on a 1-minute tick; results go to the inbox.

```bash
amaebi cron add "check disk usage and warn if >90%" --cron "0 9 * * *"
amaebi cron list
amaebi cron delete <uuid>
```

Cron expressions are 5-field UTC (`min hour dom mon dow`). Supports `*`, ranges (`1-5`), steps (`*/2`), and comma lists.

### Memory

```bash
amaebi memory list          # show the last 40 remembered messages
amaebi memory search <query>  # full-text search (FTS5)
amaebi memory count         # total number of stored memories
amaebi memory clear         # delete all memories
```

### Cache

```bash
amaebi cache stats          # session count, memory entry count, disk usage
amaebi cache prune          # evict expired sessions (respects TTL tiers)
amaebi cache prune --aggressive   # remove all sessions regardless of TTL
amaebi cache prune --dry-run      # preview without deleting
```

### ACP (Zed / Claude Code integration)

```bash
amaebi acp [--model <model>]
```

Runs amaebi as an [Agent Client Protocol](https://github.com/zed-industries/acp) agent over stdio, compatible with Zed, Claude Code, and any other ACP client. Memory reads/writes are routed through the daemon's Unix socket so only one process ever touches the SQLite databases.

---

## Model Selection

```bash
# CLI flag (highest priority)
amaebi ask --model claude-sonnet-4.6 "explain this error"

# Environment variable
export AMAEBI_MODEL=gpt-4o-mini
amaebi ask "summarise this file"

# Default: gpt-4o
```

---

## Tools

The agent autonomously selects from these tools during a task:

| Tool | Description |
|------|-------------|
| `shell_command` | Run any shell command |
| `read_file` | Read file contents |
| `edit_file` | Write/overwrite a file |
| `tmux_capture_pane` | Read the visible text of a tmux pane |
| `tmux_send_keys` | Send keystrokes to a tmux pane |

---

## Data Files

| Path | Purpose |
|------|---------|
| `~/.amaebi/sessions.json` | Directory → session UUID mapping (JSON, 0600) |
| `~/.amaebi/memory.db` | Conversation history per session UUID (SQLite, WAL) |
| `~/.amaebi/inbox.db` | Async task results from detached/cron runs (SQLite, WAL) |
| `~/.amaebi/cron.db` | Scheduled cron job definitions (SQLite, WAL) |
| `~/.amaebi/config.json` | Optional: TTL overrides per directory or tier |
| `/tmp/amaebi.sock` | Unix domain socket for client–daemon IPC |

All SQLite files use WAL mode and 0600 permissions. `sessions.json` is intentionally JSON (not SQLite) because every CLI invocation writes to it and it must tolerate concurrent readers without WAL overhead.

---

## Debugging

```bash
# Enable verbose daemon logs
AMAEBI_LOG=debug amaebi daemon

# Daemon logs go to stderr; redirect to a file if needed
AMAEBI_LOG=debug amaebi daemon 2>daemon.log
```

---

## Requirements

- Rust 1.85+ (install via [rustup](https://rustup.rs/) — Ubuntu's packaged Rust is too old)
- A GitHub account with an active [Copilot](https://github.com/features/copilot) subscription
- tmux (optional — only needed for `tmux_capture_pane` / `tmux_send_keys` tools)

---

## License

GPL-3.0
