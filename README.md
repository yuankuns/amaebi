# 🦐 amaebi

A tiny, memory-efficient AI assistant for the terminal, powered by GitHub Copilot.

**amaebi** (甘エビ/小甜虾, sweet shrimp) runs as a lightweight daemon that connects to the GitHub Copilot API. It can run shell commands, read and edit files, and interact with tmux panes — all from a single binary under 7MB.

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

# Start the daemon
./target/release/amaebi daemon &

# Ask away
./target/release/amaebi ask "what's using the most disk space?"
```

## Installation

After building, symlink the binary to your PATH:

```bash
ln -sf $(pwd)/target/release/amaebi ~/.local/bin/amaebi
```

## Commands

| Command | Description |
|---------|-------------|
| `amaebi auth` | Authenticate via GitHub device flow |
| `amaebi daemon` | Start the background daemon |
| `amaebi ask "<prompt>"` | Send a prompt and stream the response |
| `amaebi models` | List available Copilot models |

## Model Selection

```bash
# CLI flag (highest priority)
amaebi ask --model claude-sonnet-4.6 "explain this error"

# Environment variable
export AMAEBI_MODEL=gpt-5.4
amaebi ask "refactor this function"

# Default: gpt-4o
```

## Tools

The agent has access to these tools, which the LLM calls autonomously:

| Tool | Description |
|------|-------------|
| `shell_command` | Run any shell command in the background |
| `tmux_capture_pane` | Read the visible text of a tmux pane |
| `tmux_send_keys` | Send keystrokes to a tmux pane |
| `read_file` | Read file contents |
| `edit_file` | Write/overwrite a file |

## Architecture

```
┌──────────┐     Unix Socket     ┌──────────────┐     HTTPS/SSE     ┌─────────┐
│  Client   │ ◄──────────────► │    Daemon     │ ◄──────────────► │ Copilot │
│ (< 1MB)   │   /tmp/amaebi.sock │  (< 15MB)     │                  │   API   │
└──────────┘                    └──────────────┘                  └─────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  Tool Executor   │
                              │  (shell, tmux,   │
                              │   file I/O)      │
                              └─────────────────┘
```

- **Client** (`amaebi ask`): Ultra-lightweight, short-lived process. Connects to the Unix socket, sends the prompt, streams the response to stdout.
- **Daemon** (`amaebi daemon`): Persistent background process. Manages the Copilot API connection, token caching, and the agentic tool-call loop.

## Debugging

Logs are silent by default. Enable verbose output:

```bash
AMAEBI_LOG=debug amaebi daemon
```

## Requirements

- Rust 1.85+ (install via [rustup](https://rustup.rs/), not apt — Ubuntu's packaged Rust is too old)
- A GitHub account with an active [Copilot](https://github.com/features/copilot) subscription
- (Optional) tmux, for pane capture/send-keys tools

## License

MIT
