use anyhow::{Context, Result};
use clap::Parser;

mod agent_server;
mod auth;
mod auth_flow;
mod cli;
mod client;
mod config;
mod copilot;
mod cron;
mod daemon;
mod inbox;
mod ipc;
mod memory_db;
mod models;
mod session;
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

    // Print a bell notification if there are unread cron reports.
    // Shown only for user-facing commands; silently skipped if the inbox db
    // does not exist yet (no cron tasks have ever run).
    if matches!(
        &cli.command,
        cli::Command::Ask { .. }
            | cli::Command::Session { .. }
            | cli::Command::Memory { .. }
            | cli::Command::Cache { .. }
    ) {
        print_inbox_notification();
    }

    match cli.command {
        cli::Command::Daemon { socket } => daemon::run(socket).await,
        cli::Command::Ask {
            prompt,
            socket,
            model,
            detach,
            resume,
        } => {
            if detach {
                client::run_detach(socket, prompt, model).await
            } else if let Some(session_uuid) = resume {
                match client::run_resume(socket, prompt, model, session_uuid).await {
                    Ok(()) => Ok(()),
                    Err(e) if e.is::<client::Interrupted>() => std::process::exit(130),
                    Err(e) => Err(e),
                }
            } else {
                match client::run(socket, prompt, model).await {
                    Ok(()) => Ok(()),
                    Err(e) if e.is::<client::Interrupted>() => std::process::exit(130),
                    Err(e) => Err(e),
                }
            }
        }
        cli::Command::Auth {
            client_id,
            skip_validate,
        } => {
            let http = reqwest::Client::new();
            auth_flow::ensure_authenticated(&http, &client_id, skip_validate).await
        }
        cli::Command::Acp { model, socket } => agent_server::run(model, socket).await,
        cli::Command::Models => models::run().await,
        cli::Command::Memory { action, socket } => run_memory(action, socket).await,
        cli::Command::Session { action } => run_session(action),
        cli::Command::Cache { action } => run_cache(action),
        cli::Command::Inbox { action } => run_inbox(action),
        cli::Command::Cron { action } => run_cron(action),
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
            let db_path = memory_db::db_path()?;
            let conn = memory_db::init_db(&db_path)?;
            let entries = memory_db::get_recent(&conn, 40)?;
            if entries.is_empty() {
                println!("No memories stored.");
            } else {
                for e in &entries {
                    println!(
                        "[{}] ({})\n  {}\n",
                        sanitize(&e.timestamp),
                        sanitize(&e.role),
                        sanitize(&e.content)
                    );
                }
            }
            Ok(())
        }
        cli::MemoryAction::Search { query } => {
            let db_path = memory_db::db_path()?;
            let conn = memory_db::init_db(&db_path)?;
            let entries = memory_db::search_relevant(&conn, &query, 20)?;
            if entries.is_empty() {
                println!("No matches for {:?}.", query);
            } else {
                for e in &entries {
                    println!(
                        "[{}] ({})\n  {}\n",
                        sanitize(&e.timestamp),
                        sanitize(&e.role),
                        sanitize(&e.content)
                    );
                }
            }
            Ok(())
        }
        cli::MemoryAction::Clear => {
            // Clear SQLite.
            let db_path = memory_db::db_path()?;
            if db_path.exists() {
                let conn = memory_db::init_db(&db_path)?;
                memory_db::clear(&conn)?;
            }
            // Best-effort: notify a running daemon to also clear its SQLite DB.
            // Silently ignores connection failures (daemon may not be running).
            notify_daemon_cache_clear(&socket).await;
            println!("Memory cleared.");
            Ok(())
        }
        cli::MemoryAction::Count => {
            let db_path = memory_db::db_path()?;
            let conn = memory_db::init_db(&db_path)?;
            let n = memory_db::count(&conn)?;
            println!("{n}");
            Ok(())
        }
    }
}

/// Print a bell notification to stderr if there are unread inbox reports.
///
/// Silently no-ops if the inbox database does not yet exist or cannot be read,
/// so a cold-start installation never produces a confusing error.
fn print_inbox_notification() {
    match inbox::InboxStore::open() {
        Ok(store) => match store.unread_count() {
            Ok(0) => {}
            Ok(n) => {
                let noun = if n == 1 { "report" } else { "reports" };
                eprintln!("[🔔 You have {n} unread cron {noun}. Run `amaebi inbox list` to read.]");
            }
            Err(e) => tracing::debug!(error = %e, "could not check inbox unread count"),
        },
        Err(e) => tracing::debug!(error = %e, "could not open inbox store for notification"),
    }
}

fn run_inbox(action: cli::InboxAction) -> Result<()> {
    let store = inbox::InboxStore::open().context("opening inbox database")?;
    match action {
        cli::InboxAction::List { all } => {
            let reports = if all {
                store.get_all()?
            } else {
                store.get_unread()?
            };
            if reports.is_empty() {
                if all {
                    println!("Inbox is empty.");
                } else {
                    println!("No unread cron reports.");
                }
            } else {
                for r in &reports {
                    let status = if r.read { "read" } else { "UNREAD" };
                    println!(
                        "[{}] #{} — {} ({})\n  Task: {}\n",
                        status,
                        r.id,
                        r.created_at,
                        r.session_id,
                        sanitize(&r.task_description),
                    );
                }
                let unread = reports.iter().filter(|r| !r.read).count();
                if unread > 0 {
                    println!("{unread} unread. Use `amaebi inbox read <id>` to view a report.");
                }
            }
            Ok(())
        }
        cli::InboxAction::Read { id } => match store.get_by_id(id)? {
            None => anyhow::bail!("no report with id {id}"),
            Some(report) => {
                println!("Task: {}", sanitize(&report.task_description));
                println!("Session: {}", report.session_id);
                println!("Created: {}", report.created_at);
                println!();
                println!("{}", sanitize(&report.output));
                store.mark_read(id)?;
                Ok(())
            }
        },
        cli::InboxAction::MarkRead => {
            store.mark_all_read()?;
            println!("All reports marked as read.");
            Ok(())
        }
        cli::InboxAction::Clear => {
            store.clear()?;
            println!("Inbox cleared.");
            Ok(())
        }
    }
}

fn run_session(action: cli::SessionAction) -> Result<()> {
    let cwd = std::env::current_dir().context("getting current directory")?;
    match action {
        cli::SessionAction::Show => match session::current(&cwd)? {
            Some(id) => println!("{id}"),
            None => println!("(none)"),
        },
        cli::SessionAction::New => {
            let id = session::reset(&cwd)?;
            println!("{id}");
        }
        cli::SessionAction::Status => {
            let entries = session::list_all()?;
            if entries.is_empty() {
                println!("No sessions.");
            } else {
                let mut entries: Vec<_> = entries.into_iter().collect();
                entries.sort_by(|a, b| b.1.last_accessed.cmp(&a.1.last_accessed));
                for (dir, rec) in &entries {
                    println!(
                        "{}\n  uuid:     {}\n  created:  {}\n  accessed: {}\n  tier:     {}\n",
                        dir, rec.uuid, rec.created_at, rec.last_accessed, rec.ttl_tier
                    );
                }
                println!("{} session(s) total.", entries.len());
            }
        }
        cli::SessionAction::Clear {
            dry_run,
            default_ttl,
            ephemeral_ttl,
            persistent_ttl,
        } => {
            let mut ttls = std::collections::HashMap::new();
            ttls.insert("default".to_string(), default_ttl);
            ttls.insert("ephemeral".to_string(), ephemeral_ttl);
            ttls.insert("persistent".to_string(), persistent_ttl);

            let removed = session::clear_expired(&ttls, dry_run)?;
            if removed.is_empty() {
                println!("No expired sessions.");
            } else {
                let verb = if dry_run { "Would remove" } else { "Removed" };
                for (dir, rec) in &removed {
                    println!(
                        "{verb}: {} (uuid: {}, last: {}, tier: {})",
                        dir, rec.uuid, rec.last_accessed, rec.ttl_tier
                    );
                }
                println!("\n{} {} session(s).", verb, removed.len());
            }
        }
        cli::SessionAction::SetTier { tier } => {
            let uuid = session::get_or_create_with_tier(&cwd, &tier)?;
            println!("Session {uuid} set to tier \"{tier}\"");
        }
    }
    Ok(())
}

fn run_cache(action: cli::CacheAction) -> Result<()> {
    match action {
        cli::CacheAction::Prune {
            max_memory,
            dry_run,
            aggressive,
        } => {
            let verb = if dry_run { "Would prune" } else { "Pruned" };

            // 1. Prune expired sessions.
            let cfg = config::Config::load();
            let session_count = if aggressive {
                // Aggressive mode: remove ALL sessions from sessions.json.
                let all = session::list_all()?;
                let count = all.len();
                if !dry_run && count > 0 {
                    // Use clear_expired with TTL of 0 to expire everything.
                    let mut ttls = std::collections::HashMap::new();
                    ttls.insert("default".to_string(), 0u64);
                    ttls.insert("ephemeral".to_string(), 0u64);
                    ttls.insert("persistent".to_string(), 0u64);
                    let _ = session::clear_expired(&ttls, false)?;
                }
                count
            } else {
                // Use config-based TTLs for tier-aware expiry.
                let mut ttls = std::collections::HashMap::new();
                let default_secs = cfg.default_ttl().as_secs();
                ttls.insert("default".to_string(), default_secs);
                ttls.insert("ephemeral".to_string(), 300u64);
                ttls.insert("persistent".to_string(), 86400u64);
                // Merge any per-tier overrides from config ttl_minutes.
                for (key, &minutes) in &cfg.ttl_minutes {
                    if key != "default" && !key.starts_with('/') {
                        ttls.insert(key.clone(), minutes * 60);
                    }
                }
                let removed = session::clear_expired(&ttls, dry_run)?;
                removed.len()
            };
            println!("{verb} {session_count} session(s).");

            // 2. Prune memory entries (max_memory not supported with SQLite backend).
            if let Some(_max) = max_memory {
                tracing::warn!(
                    "--max-memory is not supported in the SQLite memory backend; ignoring"
                );
            }

            Ok(())
        }
        cli::CacheAction::Stats => {
            let sessions = session::list_all()?;
            let db_path = memory_db::db_path()?;
            let conn = memory_db::init_db(&db_path)?;
            let mem_count = memory_db::count(&conn)?;
            let mem_bytes = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);

            println!("Sessions: {}", sessions.len());
            for (dir, rec) in &sessions {
                println!("  {} ({}, tier: {})", dir, rec.uuid, rec.ttl_tier);
            }
            println!("\nMemory entries: {mem_count}");
            println!("Memory disk usage: {}", format_bytes(mem_bytes));

            Ok(())
        }
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn run_cron(action: cli::CronAction) -> Result<()> {
    match action {
        cli::CronAction::Add { description, schedule } => {
            // Validate the expression before writing to disk.
            cron::parse_schedule(&schedule)
                .with_context(|| format!("invalid cron expression: {schedule:?}"))?;
            let id = cron::add_job(&description, &schedule)?;
            println!("Cron job added: {id}");
            println!("  Description: {description}");
            println!("  Schedule:    {schedule}");
        }
        cli::CronAction::List => {
            let jobs = cron::load_jobs()?;
            if jobs.is_empty() {
                println!("No cron jobs scheduled.");
            } else {
                for job in &jobs {
                    let last = job
                        .last_run
                        .as_deref()
                        .unwrap_or("never");
                    println!(
                        "[{}] {}\n  schedule: {}\n  last_run: {}\n",
                        job.id, job.description, job.schedule, last
                    );
                }
                println!("{} job(s) total.", jobs.len());
            }
        }
        cli::CronAction::Delete { id } => {
            if cron::delete_job(&id)? {
                println!("Cron job {id} deleted.");
            } else {
                anyhow::bail!("no cron job with id {id:?}");
            }
        }
    }
    Ok(())
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

    let Ok(mut line) = serde_json::to_string(&ipc::Request::ClearMemory) else {
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
