use std::path::PathBuf;

pub const DEFAULT_SOCKET: &str = "/tmp/amaebi.sock";

#[derive(clap::Parser, Debug)]
#[command(
    name = "amaebi",
    version,
    about = "Tiny, memory-efficient AI assistant for tmux backed by GitHub Copilot"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(clap::Subcommand, Debug)]
pub enum Command {
    /// Start the background daemon that owns the Copilot connection.
    Daemon {
        /// Path to the Unix socket.
        #[arg(long, default_value = DEFAULT_SOCKET)]
        socket: PathBuf,
    },
    /// Send a prompt to the daemon and stream the reply to stdout.
    Ask {
        /// The prompt to send.
        prompt: String,
        /// Path to the Unix socket.
        #[arg(long, default_value = DEFAULT_SOCKET)]
        socket: PathBuf,
        /// Model to use (overrides AMAEBI_MODEL env var; default: gpt-4o).
        #[arg(long)]
        model: Option<String>,
    },
    /// Authenticate with GitHub Copilot via the device flow.
    Auth {
        /// GitHub OAuth App client ID (defaults to the public neovim copilot.vim ID).
        #[arg(long, default_value = crate::auth_flow::DEFAULT_CLIENT_ID)]
        client_id: String,
        /// Skip Copilot subscription validation after login.
        #[arg(long)]
        skip_validate: bool,
    },
    /// List available Copilot models.
    Models,
    /// Start amaebi as an ACP (Agent Client Protocol) agent over stdio.
    ///
    /// Implements the Zed ACP standard so amaebi can be used as a coding agent
    /// from any ACP-compatible client (Claude Code, Zed, etc.).
    /// Communicates via JSON-RPC over stdin/stdout.
    ///
    /// Memory reads and writes are routed through a running daemon process via
    /// the Unix socket so that only the daemon ever writes to SQLite.
    /// If no daemon is reachable, memory operations are skipped; connection
    /// failures are logged (writes at warning level, reads at debug level).
    ///
    /// Example: amaebi acp
    Acp {
        /// Model to use (overrides AMAEBI_MODEL env var; default: gpt-4o).
        #[arg(long)]
        model: Option<String>,
        /// Path to the daemon's Unix socket (used for memory IPC).
        #[arg(long, default_value = DEFAULT_SOCKET)]
        socket: PathBuf,
    },
    /// Manage conversation memory.
    Memory {
        #[command(subcommand)]
        action: MemoryAction,
        /// Path to the Unix socket (used to notify a running daemon after clear).
        #[arg(long, default_value = DEFAULT_SOCKET, global = true)]
        socket: PathBuf,
    },
}

#[derive(clap::Subcommand, Debug)]
pub enum MemoryAction {
    /// Show the last 40 remembered messages.
    List,
    /// Search memories using full-text search (FTS5).
    Search {
        /// Query string (phrase search; special FTS5 operators are escaped).
        query: String,
    },
    /// Delete all stored memories.
    Clear,
    /// Show total number of stored memories.
    Count,
}
