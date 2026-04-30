# amaebi

A small, memory-efficient AI assistant for the terminal, backed by Amazon Bedrock and GitHub Copilot.

**amaebi** (甘エビ, sweet shrimp) runs as a lightweight daemon. It can run shell commands, read and edit files, drive tmux panes, spawn parallel sub-agents, schedule cron jobs, and steer live AI responses mid-flight — all from a single binary under 7 MB.

## Quick Start

```bash
# Build
git clone https://github.com/yuankuns/amaebi.git
cd amaebi
cargo build --release
ln -sf $(pwd)/target/release/amaebi ~/.local/bin/amaebi

# Authenticate (skip if using Bedrock; see docs/architecture.md)
amaebi auth

# Start the daemon (keep it running)
amaebi daemon &

# One-shot question
amaebi ask "what's using the most disk space?"

# Or an interactive session
amaebi chat
```

Requires a recent stable Rust toolchain (install via
[rustup](https://rustup.rs/)), and either a GitHub Copilot subscription or an
Amazon Bedrock bearer token. tmux and Docker are optional.

## How It Works

amaebi splits into two processes connected by a Unix socket:

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

The **client** (`ask` / `chat`) is short-lived: it streams output and reads
stdin for mid-flight steering. The **daemon** (`amaebi daemon`) owns the API
connection, session history, tool execution, cron scheduler, and pane /
resource leases.

See [docs/architecture.md](docs/architecture.md) for file layouts, database
schemas, and the `/claude` request lifecycle.

## Common Commands

| Command | Purpose |
|---------|---------|
| `amaebi ask "<prompt>"` | Send a prompt and stream the reply |
| `amaebi chat` | Interactive multi-turn session ([docs](docs/chat.md)) |
| `/claude "<task>"` | Launch a supervised Claude Code subprocess in tmux ([docs](docs/claude.md)) |
| `amaebi dashboard` | Live TUI view of panes, sessions, inbox, cron ([docs](docs/dashboard.md)) |
| `amaebi memory search <q>` | Full-text search of conversation memory |
| `amaebi inbox list` | Read results from detached and cron tasks |
| `amaebi cron add "<desc>" --cron "<expr>"` | Schedule a recurring autonomous task |

Run `amaebi --help` for the full subcommand list.

## Feature Index

- **[chat.md](docs/chat.md)** — `amaebi ask` and `amaebi chat`, session resume, steering, detached runs
- **[claude.md](docs/claude.md)** — the `/claude` slash command, worktrees, flags (`--tag`, `--resume-pane`, `--resource`, etc.)
- **[supervision.md](docs/supervision.md)** — WAIT / STEER / DONE model, timing knobs, release guarantees
- **[resource-pool.md](docs/resource-pool.md)** — `~/.amaebi/resources.toml`, `--resource` semantics, env injection
- **[dashboard.md](docs/dashboard.md)** — `amaebi dashboard` TUI
- **[architecture.md](docs/architecture.md)** — two-process model, `~/.amaebi/` file inventory, request lifecycle
- **[bedrock-upstream-check.md](docs/bedrock-upstream-check.md)** — how to fetch recent Bedrock API changes from `aws-sdk-rust` releases and hand them to Opus for analysis

### Not yet covered by dedicated docs

- `amaebi cron` — 5-field UTC cron expressions; jobs run in the daemon, results land in the inbox
- `amaebi memory` — SQLite-backed conversation history (FTS5)
- `amaebi inbox` — mailbox for detached and cron task results
- `amaebi session` — per-directory UUIDs and TTL tiers
- `amaebi cache prune` — evict expired sessions
- `amaebi acp` — ACP (Agent Client Protocol) agent over stdio for Zed / Claude Code integration
- `amaebi models` — list available Bedrock/Copilot models
- `amaebi tag list` / `release` — inspect and force-release task notebook leases
- `amaebi resource list` — inspect the resource pool and lease state

See the `amaebi <subcommand> --help` output for flags.

## Debugging

```bash
AMAEBI_LOG=debug amaebi daemon 2>daemon.log
```

## License

GPL-3.0
