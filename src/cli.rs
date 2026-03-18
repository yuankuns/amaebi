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
}
