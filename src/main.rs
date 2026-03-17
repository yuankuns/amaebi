use anyhow::Result;
use clap::Parser;

mod auth;
mod cli;
mod client;
mod copilot;
mod daemon;
mod ipc;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let cli = cli::Cli::parse();
    match cli.command {
        cli::Command::Daemon { socket } => daemon::run(socket).await,
        cli::Command::Ask { prompt, socket } => client::run(socket, prompt).await,
    }
}
