use std::path::PathBuf;

pub const DEFAULT_SOCKET: &str = "/tmp/amaebi.sock";

#[derive(clap::Parser, Debug)]
#[command(
    name = "amaebi",
    version,
    about = "Tiny, memory-efficient AI assistant for tmux backed by Amazon Bedrock and GitHub Copilot"
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
        /// Model to use (overrides AMAEBI_MODEL env var; default: claude-sonnet-4.6[1m]).
        /// Format: [provider/]model — e.g. bedrock/claude-sonnet-4.6, copilot/gpt-4o.
        #[arg(long)]
        model: Option<String>,
        /// Submit the task in the background; print a task ID to stderr and exit
        /// immediately.  The result is deposited into `amaebi inbox` when done.
        /// Cannot be combined with --resume.
        #[arg(long, conflicts_with = "resume")]
        detach: bool,
        /// Resume a prior session, loading its full chronological history
        /// instead of the normal sliding-window context (last N turns).
        ///
        /// To pass a UUID you must use `=` syntax (`-r=<uuid>` or
        /// `--resume=<uuid>`) so the value cannot be confused with the
        /// required `<PROMPT>` positional.  Bare `-r` / `--resume` opens an
        /// interactive picker of this directory's session history.
        ///
        /// Accepts a session UUID (or unique prefix ≥ 4 chars).
        /// Cannot be combined with --detach.
        #[arg(
            short = 'r',
            long,
            conflicts_with = "detach",
            num_args = 0..=1,
            require_equals = true,
            default_missing_value = "",
        )]
        resume: Option<String>,
    },
    /// Start an interactive multi-turn chat session (long connection).
    ///
    /// Stays open after each response: type the next message at the `>` prompt.
    /// Ctrl-C mid-generation → interrupt/steer. Empty line or Ctrl-D → exit.
    Chat {
        /// Optional opening message.
        prompt: Option<String>,
        #[arg(long, default_value = DEFAULT_SOCKET)]
        socket: PathBuf,
        /// Model to use (overrides AMAEBI_MODEL env var; default: claude-sonnet-4.6[1m]).
        /// Format: [provider/]model — e.g. bedrock/claude-sonnet-4.6, copilot/gpt-4o.
        #[arg(long)]
        model: Option<String>,
        /// Resume a prior chat session, loading its full chronological history.
        ///
        /// To pass a UUID you must use `=` syntax (`-r=<uuid>` or
        /// `--resume=<uuid>`) so the value cannot be confused with the
        /// optional `[PROMPT]` positional.  Bare `-r` / `--resume` opens an
        /// interactive picker of this directory's session history.
        ///
        /// Accepts a session UUID (or unique prefix ≥ 4 chars).
        #[arg(
            short = 'r',
            long,
            num_args = 0..=1,
            require_equals = true,
            default_missing_value = "",
        )]
        resume: Option<String>,
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
        /// Model to use (overrides AMAEBI_MODEL env var; default: claude-sonnet-4.6[1m]).
        /// Format: [provider/]model — e.g. bedrock/claude-sonnet-4.6, copilot/gpt-4o.
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
    /// Manage scheduled cron jobs.
    ///
    /// Jobs are stored in `~/.amaebi/cron.db` and executed autonomously by
    /// a running daemon process.  Results are deposited into `amaebi inbox`.
    ///
    /// Example: amaebi cron add "check disk usage" --cron "0 9 * * *"
    Cron {
        #[command(subcommand)]
        action: CronAction,
    },
    /// Read reports from completed cron tasks.
    ///
    /// Cron tasks run autonomously in the background; their output is stored
    /// in `~/.amaebi/inbox.db`.  A bell notification is printed at the start
    /// of every `amaebi ask` invocation when unread reports are present.
    Inbox {
        #[command(subcommand)]
        action: InboxAction,
    },
    /// Inspect the resource pool and current lease state.
    ///
    /// Reads `~/.amaebi/resources.toml` (the pool definition) and
    /// `~/.amaebi/resource-state.json` (the runtime lease table) and prints
    /// one line per resource showing its class, status, and current holder.
    ///
    /// Only `list` is available for now; `register` / `unregister` will be
    /// added when dynamic registration is needed.
    Resource {
        #[command(subcommand)]
        action: ResourceAction,
    },
    /// Live TUI dashboard aggregating session, pane, inbox, and cron state.
    ///
    /// Full-screen view that auto-refreshes every 2 s.  Shows environment
    /// context, a task-summary card, and a unified activity stream merged
    /// from panes (`~/.amaebi/tmux-state.json`), sessions (`memory.db`,
    /// newest turn per session), inbox arrivals, and cron `last_run` events.
    ///
    /// Keys: `q` / Esc / Ctrl-C exits; `r` forces a re-read.
    Dashboard,
}

#[derive(clap::Subcommand, Debug)]
pub enum ResourceAction {
    /// List all resources in the pool with their status and current holder.
    List,
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

#[derive(clap::Subcommand, Debug)]
pub enum SessionAction {
    /// Show the session history for the current directory.
    ///
    /// Prints all past session UUIDs (newest first) with their creation and
    /// last-access timestamps.  Prints "(none)" if no session has been started
    /// in this directory yet.  Use a listed UUID with `--resume` to continue
    /// a previous session.
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
pub enum InboxAction {
    /// List all unread cron reports (default view).
    ///
    /// Pass `--all` to include already-read reports.
    List {
        /// Include reports that have already been read.
        #[arg(long)]
        all: bool,
    },
    /// Display a specific report and mark it as read.
    Read {
        /// The numeric ID of the report (shown in `inbox list`).
        id: i64,
    },
    /// Mark all unread reports as read without displaying them.
    MarkRead,
    /// Delete all inbox reports.
    Clear,
}

#[derive(clap::Subcommand, Debug)]
pub enum CronAction {
    /// Add a new cron job.
    Add {
        /// Task description used as the LLM prompt when the job fires.
        description: String,
        /// 5-field cron expression (min hour dom mon dow).
        ///
        /// Supports: `*`, `n`, `n-m`, `*/n`, and comma-separated lists.
        /// Example: "0 9 * * *" runs every day at 09:00 UTC.
        #[arg(long = "cron")]
        schedule: String,
    },
    /// List all scheduled cron jobs.
    List,
    /// Delete a cron job by its UUID.
    Delete {
        /// UUID of the job to remove (shown in `cron list`).
        id: String,
    },
}

#[derive(clap::Subcommand, Debug)]
pub enum CacheAction {
    /// Prune stale history and session allocation state.
    ///
    /// Removes expired sessions from `sessions.json`.  The SQLite memory
    /// backend does not support entry-level trimming, so `--max-memory` is
    /// accepted for backwards compatibility but has no effect.
    Prune {
        /// Maximum memory entries to keep (oldest are removed first).
        /// NOTE: ignored when using the SQLite memory backend (the default).
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
