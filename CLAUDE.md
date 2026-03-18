# CLAUDE.md — Amaebi Coding Guidelines

All AI coding agents MUST follow these rules when writing code for this project.

## Project Overview

- **Name**: amaebi (小甜虾) — a tiny, memory-efficient AI assistant for tmux
- **Language**: Rust (edition 2021, MSRV 1.85+)
- **Async Runtime**: tokio (multi-threaded)
- **Error Handling**: `anyhow` for application errors, `?` propagation
- **Serialization**: `serde` + `serde_json`
- **Logging**: `tracing` crate (never `println!` for errors/warnings)
- **License**: GPL-3.0
- **Config directory**: `~/.amaebi/`

## Build & Test

```bash
cargo fmt --check    # formatting (MUST pass before commit)
cargo clippy -- -D warnings  # lint (MUST pass before commit)
cargo test           # all tests must pass
cargo build --release
cargo doc --no-deps  # doc comments must compile
```

**Always run all 5 checks before committing.** Do not commit code that fails any of these.

## Critical Rules

### Async / Tokio
- **NEVER** do synchronous filesystem I/O (`std::fs`) on the tokio runtime thread
- Use `tokio::fs` or wrap sync I/O in `tokio::task::spawn_blocking`
- CPU-bound work MUST use `spawn_blocking` or `spawn` on a dedicated runtime
- When using `spawn_blocking`, handle both the `JoinError` (panic) and the inner `Result`

### File I/O & Concurrency
- **ALWAYS** use file locking when multiple tasks/processes may access the same file
  - Readers: `fs2::FileExt::lock_shared()`
  - Writers: `fs2::FileExt::lock_exclusive()`
- For append-mode files (like JSONL): build the complete line as a `String` first, then `write_all` in one call (avoid partial writes from `writeln!`)
- Use in-process synchronization (e.g., `tokio::sync::Mutex`) to serialize concurrent access within the same process
- Explicitly `drop()` lock guards as early as possible — don't rely on implicit scope drops

### Security & Privacy
- **File permissions**: sensitive files (tokens, memory) MUST be created with mode `0o600`
  - Also call `std::fs::set_permissions()` after opening, in case the file pre-exists with broader permissions
- **NEVER** log sensitive data (user prompts, assistant responses, tokens) in tracing output
  - When logging malformed data, log only metadata: byte length, line number, or a truncated/sanitized prefix (max 20 chars)
- **NEVER** inject user-generated content into the system prompt — it enables prompt injection
  - Conversation history → inject as proper `user`/`assistant` message pairs
  - System prompt → only hardcoded instructions
- **Terminal output safety**: when printing user-controlled strings to the terminal, sanitize ANSI escape sequences and control characters to prevent terminal manipulation
- **NEVER** use `.unwrap()` in non-test code — use `.context()` or `.expect("invariant: ...")` with a descriptive message

### Error Handling
- Use `anyhow::Result` and `anyhow::Context` for descriptive errors
- Propagate errors with `?` — don't swallow them silently
- When skipping malformed data (e.g., bad JSONL lines), log a `tracing::warn!` with non-sensitive metadata — never silently drop errors via `.ok()`
- Handle all `Result` paths explicitly — no silent fallthrough

### Code Style
- Run `cargo fmt` before every commit — zero tolerance for format drift
- Follow Rust API Guidelines: `snake_case` functions, `PascalCase` types, `SCREAMING_SNAKE_CASE` constants
- Prefer iterators and combinators over explicit loops where clearer
- Keep functions focused (single responsibility)
- Return early to reduce nesting
- Derive `Debug`, `Clone`, `PartialEq` where appropriate

### Testing
- All new functions/modules MUST have unit tests
- Use `#[cfg(test)]` modules with `use super::*`
- Use `tempfile::tempdir()` for file-based tests — never write to real paths
- Follow Arrange-Act-Assert pattern
- Async tests use `#[tokio::test]`
- Mock external dependencies (network, filesystem)

### Documentation
- Doc comments (`///`) on all public items
- Keep PR descriptions accurate and up-to-date with the actual implementation
- If implementation diverges from PR description, update the description

### Git & CI
- Branch: `master` (not `main`)
- CI runs: fmt → clippy → test → build → doc
- All checks must pass before merge
- Commit messages: conventional commits (`feat:`, `fix:`, `test:`, `ci:`, `docs:`, `refactor:`)

## Architecture Notes

- **IPC**: Unix domain socket at `/tmp/amaebi.sock`, newline-delimited JSON frames
- **Daemon**: single tokio runtime, concurrent connections via `tokio::spawn`
- **Memory**: JSONL file at `~/.amaebi/memory.jsonl`, append-only, read with streaming (VecDeque ring buffer for last N)
- **Auth**: GitHub Device Flow → OAuth token stored in `~/.amaebi/hosts.json`, fallback reads `~/.config/github-copilot/`
- **Tools**: 5 tools (shell_command, tmux_capture_pane, tmux_send_keys, read_file, edit_file) via `ToolExecutor` trait
