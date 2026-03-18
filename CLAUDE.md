# CLAUDE.md — Amaebi

Read `~/.claude/skills/rust/SKILL.md` for Rust coding guidelines. Follow all rules there.

## Project-Specific Notes

- **Name**: amaebi (小甜虾) — tiny, memory-efficient AI assistant for tmux
- **License**: GPL-3.0
- **Config directory**: `~/.amaebi/`
- **Branch**: `master` (not `main`)
- **IPC**: Unix domain socket at `/tmp/amaebi.sock`, newline-delimited JSON frames
- **Daemon**: single tokio runtime, concurrent connections via `tokio::spawn`
- **Memory**: JSONL at `~/.amaebi/memory.jsonl`, append-only, VecDeque ring buffer
- **Auth**: GitHub Device Flow → OAuth token in `~/.amaebi/hosts.json`, fallback `~/.config/github-copilot/`
- **Tools**: 5 tools (shell_command, tmux_capture_pane, tmux_send_keys, read_file, edit_file) via `ToolExecutor` trait
- **Dependencies**: tokio, serde, serde_json, anyhow, tracing, reqwest, clap, fs2, chrono
