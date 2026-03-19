use anyhow::Result;
use clap::Parser;

mod agent_server;
mod auth;
mod auth_flow;
mod cli;
mod client;
mod copilot;
mod daemon;
mod ipc;
mod memory;
mod models;
#[cfg(test)]
mod test_utils;
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
        } => match client::run(socket, prompt, model).await {
            Ok(()) => Ok(()),
            Err(e) if e.is::<client::Interrupted>() => std::process::exit(130),
            Err(e) => Err(e),
        },
        cli::Command::Auth {
            client_id,
            skip_validate,
        } => {
            let http = reqwest::Client::new();
            auth_flow::ensure_authenticated(&http, &client_id, skip_validate).await
        }
        cli::Command::Acp { model } => agent_server::run(model).await,
        cli::Command::Models => models::run().await,
        cli::Command::Memory { action, socket } => run_memory(action, socket).await,
    }
}

/// Strip ANSI/VT escape sequences and ASCII control characters from `s`.
///
/// Keeps newline (0x0A) and tab (0x09); removes all other bytes below 0x20,
/// DEL (0x7F), and the following escape sequence families:
///
/// - **CSI** (`ESC [` … final-byte 0x40–0x7E) — e.g. colour/cursor codes
/// - **OSC** (`ESC ]` … BEL or ST) — e.g. title-set, clipboard write
/// - **DCS** (`ESC P` … ST) — device control strings
/// - **APC** (`ESC _` … ST) — application program commands
/// - **PM**  (`ESC ^` … ST) — privacy messages
/// - 2-char ESC sequences (everything else after ESC)
///
/// String sequences (OSC/DCS/APC/PM) are terminated by BEL (0x07) or
/// ST (`ESC \`).  This prevents stored user/assistant text from
/// manipulating the terminal.
fn sanitize(s: &str) -> String {
    enum State {
        Normal,
        Esc,
        Csi,
        /// Inside an OSC / DCS / APC / PM string — consume until BEL or ST.
        StringSeq,
        /// Saw ESC inside a string sequence — next `\` completes the ST terminator.
        StringSeqEsc,
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
            State::Esc => match ch {
                // CSI sequence.
                '[' => State::Csi,
                // String-terminated sequences: OSC, DCS, APC, PM.
                ']' | 'P' | '_' | '^' => State::StringSeq,
                // Anything else is a 2-char escape — consumed, back to Normal.
                _ => State::Normal,
            },
            State::Csi => {
                // CSI final byte is in range 0x40–0x7E ('@' to '~').
                if ('@'..='~').contains(&ch) {
                    State::Normal
                } else {
                    State::Csi
                }
            }
            State::StringSeq => {
                if ch == '\x07' {
                    // BEL terminates the string sequence.
                    State::Normal
                } else if ch == '\x1b' {
                    // Possible ST (`ESC \`) — check next char.
                    State::StringSeqEsc
                } else {
                    State::StringSeq
                }
            }
            State::StringSeqEsc => {
                if ch == '\\' {
                    // ST (`ESC \`) — string sequence complete.
                    State::Normal
                } else {
                    // Not an ST; continue consuming the string sequence.
                    State::StringSeq
                }
            }
        };
    }
    out
}

async fn run_memory(action: cli::MemoryAction, socket: std::path::PathBuf) -> Result<()> {
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
            // Best-effort: notify a running daemon to clear its in-memory cache.
            // Silently ignores connection failures (daemon may not be running).
            notify_daemon_cache_clear(&socket).await;
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

/// Tell a running daemon to flush its in-memory conversation cache.
///
/// This is best-effort: if the daemon is not running, or the connection fails
/// for any reason, the error is silently discarded.
async fn notify_daemon_cache_clear(socket: &std::path::Path) {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::UnixStream;

    let Ok(stream) = UnixStream::connect(socket).await else {
        return;
    };
    let (reader, mut writer) = tokio::io::split(stream);

    let Ok(mut line) = serde_json::to_string(&ipc::Request::ClearCache) else {
        return;
    };
    line.push('\n');

    if writer.write_all(line.as_bytes()).await.is_err() {
        return;
    }
    // Drain the Done response to confirm the daemon processed the clear.
    let mut lines = BufReader::new(reader).lines();
    let _ = lines.next_line().await;
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

    // --- OSC / DCS / APC / PM string-sequence tests ---

    #[test]
    fn sanitize_strips_osc_title_set_bel_terminated() {
        // OSC 0 ; title BEL  — common title-set sequence
        assert_eq!(sanitize("\x1b]0;My Title\x07after"), "after");
    }

    #[test]
    fn sanitize_strips_osc_title_set_st_terminated() {
        // OSC 0 ; title ST (ESC \)
        assert_eq!(sanitize("\x1b]0;My Title\x1b\\after"), "after");
    }

    #[test]
    fn sanitize_strips_osc_52_clipboard() {
        // OSC 52 ; c ; <base64> BEL  — clipboard write (CVE-class attack)
        assert_eq!(sanitize("\x1b]52;c;aGVsbG8=\x07visible"), "visible");
    }

    #[test]
    fn sanitize_strips_osc_52_clipboard_st_terminated() {
        assert_eq!(sanitize("\x1b]52;c;aGVsbG8=\x1b\\visible"), "visible");
    }

    #[test]
    fn sanitize_strips_dcs_sequence() {
        // DCS (ESC P) … ST
        assert_eq!(sanitize("\x1bPsomething\x1b\\after"), "after");
    }

    #[test]
    fn sanitize_strips_apc_sequence() {
        // APC (ESC _) … ST
        assert_eq!(sanitize("\x1b_payload\x1b\\after"), "after");
    }

    #[test]
    fn sanitize_strips_pm_sequence() {
        // PM (ESC ^) … BEL
        assert_eq!(sanitize("\x1b^payload\x07after"), "after");
    }

    #[test]
    fn sanitize_osc_embedded_in_text() {
        // Text before and after an OSC sequence is preserved.
        assert_eq!(sanitize("before\x1b]0;title\x07after"), "beforeafter");
    }

    #[test]
    fn sanitize_string_seq_false_esc_continues() {
        // ESC not followed by '\' inside a string seq does NOT terminate it.
        // ESC 'x' is not ST — sequence continues; only terminated by BEL here.
        assert_eq!(sanitize("\x1b]data\x1bxmore\x07end"), "end");
    }
}
