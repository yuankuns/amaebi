# Tmux Copilot Agent (Rust)

## Objective
Build a headless, ultra-lightweight, memory-efficient AI assistant for `tmux` using Rust. 
It connects to the GitHub Copilot API as its "brain" and uses `tmux` as its "eyes and hands".

## Architecture (C/S over Unix Socket)
To keep memory usage extremely low per terminal window, the application uses a Daemon-Client architecture compiled into a single binary (`tmux-copilot`).

1. **Daemon (`tmux-copilot daemon`)**:
   - Runs in the background (memory footprint < 15MB).
   - Listens on a Unix Domain Socket (e.g., `/tmp/tmux-copilot.sock`).
   - Handles the GitHub Copilot API connection (OAuth token reading, HTTP/SSE streaming).
   - Manages the Agentic Loop (Parse `tool_calls` -> Execute tool -> Send tool result back).

2. **Client (`tmux-copilot ask "<prompt>"`)**:
   - Extremely lightweight, short-lived process.
   - Connects to the Unix socket, sends the user's prompt (along with the current `$TMUX_PANE` environment variable for context).
   - Streams the daemon's text responses to `stdout` in real-time.

## Technology Stack
- **Language**: Rust (Edition 2021)
- **Async Runtime**: `tokio` (for Unix Sockets and HTTP requests)
- **HTTP Client**: `reqwest`
- **JSON Serialization**: `serde` and `serde_json`
- **CLI Parsing**: `clap`
- **Error Handling**: `anyhow`

## Tools (Agent Capabilities)
The agent must expose these tools to the GitHub Copilot LLM:
1. `tmux_capture_pane`: Runs `tmux capture-pane -t <target> -p` to read screen contents.
2. `tmux_send_keys`: Runs `tmux send-keys -t <target> "<keys>"` to interact with the shell.
3. `shell_command`: Runs an arbitrary background shell command.
4. `read_file` / `edit_file`: Basic file I/O for codebase manipulation.

## Implementation Phases

### Phase 1: Skeleton, CLI, and IPC (Unix Sockets)
1. Initialize the cargo project.
2. Set up `clap` with subcommands: `daemon` and `ask <prompt>`.
3. Implement `tokio::net::UnixListener` for the daemon.
4. Implement the client to connect to the socket and pass the string to the daemon, then print the echo response.

### Phase 2: Copilot Auth & HTTP Client
1. Read the GitHub Copilot token from `~/.config/github-copilot/hosts.json` (or `apps.json`).
2. Implement the API request struct (system prompt, messages list, tool schemas).
3. Connect the socket message to a dummy API call and stream the text response back to the client.

### Phase 3: Tools & Agentic Loop
1. Implement the `tmux` wrapper functions in Rust (using `std::process::Command`).
2. Add the agentic loop: when Copilot responds with `tool_calls` (using the `choices[1]` pattern we discovered in Emacs), pause the text stream, execute the tool, and send the result back to Copilot.
3. Stream the final answer back to the client.

## Special Note on Copilot API
Remember that the GitHub Copilot Chat API sometimes splits the text response and `tool_calls` into different `choices` array elements (e.g., `choices[0]` has `content`, `choices[1]` has `tool_calls`). The parser must check all choices.

## Phase 4: Subagent Orchestration & Sandboxing (Advanced)
To support complex, multi-step tasks without blocking the main event loop or risking destructive commands on the host:

1. **Subagent Spawning**:
   - The daemon can spawn child subagents (e.g., as background threads or isolated processes) to handle long-running coding tasks or deep research.
   - Introduce a tool: `spawn_subagent(task_description)` which returns a subagent ID.

2. **Sandboxing (Docker/Containers)**:
   - When a subagent needs to run shell commands or compile untrusted code, it should optionally be routed to an isolated sandbox.
   - E.g., Use lightweight container APIs (or run via an external runner like Docker) to execute `shell_command` within a safe, isolated directory.

## Execution & Native Commands
To act as a capable assistant without relying entirely on tmux side-effects, the Agent needs a robust native command execution tool:
1. `shell_command`: A primary tool to run bash/sh commands natively on the host system or sandbox.
   - This allows the agent to run `grep`, `find`, `systemctl restart`, `git`, or any other background operation without taking over the user's active tmux pane.
   - Standard output and standard error should be captured and returned to the LLM.
   - For commands requiring elevated privileges (like restarting system services), the tool should politely ask the user via the client before proceeding, or handle `sudo` via proper escalation paths.
