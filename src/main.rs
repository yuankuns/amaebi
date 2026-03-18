use anyhow::Result;
use clap::Parser;

mod auth;
mod auth_flow;
mod cli;
mod client;
mod copilot;
mod daemon;
mod ipc;
mod memory;
mod models;
mod tools;

#[tokio::main]
async fn main() -> Result<()> {
    let filter = std::env::var("AMAEBI_LOG")
        .or_else(|_| std::env::var("RUST_LOG"))
        .map(tracing_subscriber::EnvFilter::new)
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    let cli = cli::Cli::parse();
    match cli.command {
        cli::Command::Daemon { socket } => daemon::run(socket).await,
        cli::Command::Ask {
            prompt,
            socket,
            model,
        } => client::run(socket, prompt, model).await,
        cli::Command::Auth {
            client_id,
            skip_validate,
        } => {
            let http = reqwest::Client::new();
            auth_flow::ensure_authenticated(&http, &client_id, skip_validate).await
        }
        cli::Command::Models => models::run().await,
        cli::Command::Memory { action } => run_memory(action),
    }
}

/// Strip ANSI/VT escape sequences and ASCII control characters from `s`.
///
/// Keeps newline (0x0A) and tab (0x09); removes all other bytes below 0x20,
/// DEL (0x7F), and CSI sequences (`ESC [` … final-byte in 0x40–0x7E).
/// This prevents stored user/assistant text from manipulating the terminal.
fn sanitize(s: &str) -> String {
    enum State {
        Normal,
        Esc,
        Csi,
    }
    let mut out = String::with_capacity(s.len());
    let mut state = State::Normal;
    for ch in s.chars() {
        state = match state {
            State::Normal => {
                if ch == '\x1b' {
                    State::Esc
                } else if ch == '\n' || ch == '\t' {
                    out.push(ch);
                    State::Normal
                } else if (ch as u32) < 0x20 || ch == '\x7f' {
                    State::Normal // drop control character
                } else {
                    out.push(ch);
                    State::Normal
                }
            }
            State::Esc => {
                // '[' begins a CSI sequence; anything else is a 2-char escape.
                if ch == '[' {
                    State::Csi
                } else {
                    State::Normal
                }
            }
            State::Csi => {
                // CSI final byte is in range 0x40–0x7E ('@' to '~').
                if ('@'..='~').contains(&ch) {
                    State::Normal
                } else {
                    State::Csi
                }
            }
        };
    }
    out
}

fn run_memory(action: cli::MemoryAction) -> Result<()> {
    match action {
        cli::MemoryAction::List => {
            let entries = memory::load_recent(20)?;
            if entries.is_empty() {
                println!("No memories stored.");
            } else {
                for e in &entries {
                    println!(
                        "[{}]\n  Q: {}\n  A: {}\n",
                        sanitize(&e.timestamp),
                        sanitize(&e.user),
                        sanitize(&e.assistant)
                    );
                }
            }
            Ok(())
        }
        cli::MemoryAction::Search { query } => {
            let entries = memory::search(&query)?;
            if entries.is_empty() {
                println!("No matches for {:?}.", query);
            } else {
                for e in &entries {
                    println!(
                        "[{}]\n  Q: {}\n  A: {}\n",
                        sanitize(&e.timestamp),
                        sanitize(&e.user),
                        sanitize(&e.assistant)
                    );
                }
            }
            Ok(())
        }
        cli::MemoryAction::Clear => {
            memory::clear()?;
            println!("Memory cleared.");
            Ok(())
        }
        cli::MemoryAction::Count => {
            let n = memory::count()?;
            println!("{n}");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::sanitize;

    #[test]
    fn sanitize_plain_text_unchanged() {
        assert_eq!(sanitize("hello world"), "hello world");
    }

    #[test]
    fn sanitize_strips_ansi_color_sequence() {
        assert_eq!(sanitize("\x1b[31mred\x1b[0m"), "red");
    }

    #[test]
    fn sanitize_strips_ansi_bold_and_reset() {
        assert_eq!(sanitize("\x1b[1mbold\x1b[m"), "bold");
    }

    #[test]
    fn sanitize_strips_non_printable_control_chars() {
        assert_eq!(sanitize("a\x01\x02\x03b"), "ab");
        assert_eq!(sanitize("a\x7fb"), "ab");
        assert_eq!(sanitize("a\x08b"), "ab"); // backspace
    }

    #[test]
    fn sanitize_keeps_newline_and_tab() {
        assert_eq!(sanitize("line1\nline2\ttabbed"), "line1\nline2\ttabbed");
    }

    #[test]
    fn sanitize_strips_two_char_esc_sequence() {
        // ESC c (terminal reset) — 2-char sequence
        assert_eq!(sanitize("\x1bctext"), "text");
    }

    #[test]
    fn sanitize_empty_string() {
        assert_eq!(sanitize(""), "");
    }
}
