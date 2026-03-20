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
    /// Example: amaebi acp
    Acp {
        /// Model to use (overrides AMAEBI_MODEL env var; default: gpt-4o).
        #[arg(long)]
        model: Option<String>,
    },
    /// Manage conversation memory.
    Memory {
        #[command(subcommand)]
        action: MemoryAction,
        /// Path to the Unix socket (used to notify a running daemon after clear).
        #[arg(long, default_value = DEFAULT_SOCKET, global = true)]
        socket: PathBuf,
    },
    /// Manage the session identity for the current directory.
    ///
    /// Each directory has a stable UUID stored in `~/.amaebi/sessions.json`.
    /// The daemon uses this UUID to isolate per-project conversation history.
    Session {
        #[command(subcommand)]
        action: SessionAction,
    },
    /// Manage cache state (sessions, memory, disk usage).
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },
}

#[derive(clap::Subcommand, Debug)]
pub enum MemoryAction {
    /// Show the last 20 remembered conversations.
    List,
    /// Search memories by substring.
    Search {
        /// Query string to search for.
        query: String,
    },
    /// Delete all stored memories.
    Clear,
    /// Show total number of stored memories.
    Count,
}

#[derive(clap::Subcommand, Debug)]
pub enum SessionAction {
    /// Show the session UUID for the current directory.
    ///
    /// Prints the UUID if one has been assigned, or "(none)" if the current
    /// directory has not started a session yet.
    Show,
    /// Reset the session for the current directory by generating a new UUID.
    ///
    /// The old conversation context is abandoned; the next `amaebi ask` will
    /// start with a blank slate.
    New,
    /// List all session mappings with their last-access timestamps.
    Status,
    /// Clear expired session entries from `~/.amaebi/sessions.json`.
    Clear {
        /// Show which entries would be evicted without actually removing them.
        #[arg(long)]
        dry_run: bool,
        /// TTL in seconds for the "default" tier (default: 1800).
        #[arg(long, default_value = "1800")]
        default_ttl: u64,
        /// TTL in seconds for the "ephemeral" tier (default: 300).
        #[arg(long, default_value = "300")]
        ephemeral_ttl: u64,
        /// TTL in seconds for the "persistent" tier (default: 86400).
        #[arg(long, default_value = "86400")]
        persistent_ttl: u64,
    },
    /// Set the TTL tier for the current directory's session.
    SetTier {
        /// Tier name (e.g. "default", "ephemeral", "persistent").
        tier: String,
    },
}

#[derive(clap::Subcommand, Debug)]
pub enum CacheAction {
    /// Prune stale history and session allocation state.
    ///
    /// Removes expired sessions from `sessions.json` and optionally trims
    /// the memory JSONL file to a maximum number of entries.
    Prune {
        /// Maximum memory entries to keep (oldest are removed first).
        /// Defaults to keeping all entries.
        #[arg(long)]
        max_memory: Option<usize>,
        /// Show what would be pruned without actually doing it.
        #[arg(long)]
        dry_run: bool,
        /// Force prune ALL sessions (ignores TTL; keeps only active ones).
        #[arg(long)]
        aggressive: bool,
    },
    /// Show cache statistics (session count, memory entry count, disk usage).
    Stats,
}
