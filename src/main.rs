use anyhow::Result;
use clap::Parser;

mod auth;
mod auth_flow;
mod cli;
mod client;
mod copilot;
mod daemon;
mod ipc;
mod models;
mod tools;

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
        cli::Command::Ask { prompt, socket, model } => client::run(socket, prompt, model).await,
        cli::Command::Auth {
            client_id,
            skip_validate,
        } => {
            let http = reqwest::Client::new();
            auth_flow::ensure_authenticated(&http, &client_id, skip_validate).await
        }
        cli::Command::Models => models::run().await,
    }
}
