//! `amaebi dashboard` — live TUI aggregating pane / session / inbox / cron state.
//!
//! Read-only observer: does not touch the daemon via IPC and does not modify
//! any on-disk state. Every tick re-reads the existing files (sessions.json,
//! tmux-state.json, inbox.db, cron.db, memory.db) and rebuilds a fresh
//! `Snapshot`. Sources that are missing or fail to open are treated as empty —
//! the dashboard must render even on a freshly installed machine.
//!
//! Layout is three vertical panels, top to bottom:
//! * Environment (5 lines) — cwd, git branch, sandbox
//! * Task summary (5 lines) — pane / cron / inbox counts
//! * Activity (remaining) — unified event stream, tail mode, newest first
//!
//! Keys: `q` / Esc / Ctrl-C exit; `r` forces an immediate re-read. Auto-refresh
//! ticks every [`REFRESH`].

use std::io::{stdout, Stdout};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Terminal;

use crate::cron::CronJob;
use crate::inbox::InboxReport;
use crate::memory_db;
use crate::pane_lease::{PaneLease, PaneStatus};

const REFRESH: Duration = Duration::from_secs(2);
const ACTIVITY_CAP: usize = 20;
const SESSION_EVENT_CAP: usize = 20;
const PROMPT_SNIPPET_CHARS: usize = 80;

// ---------------------------------------------------------------------------
// Snapshot aggregation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Source {
    Pane(String),
    Session(String),
    Cron,
    Inbox(i64),
}

#[derive(Clone, Debug)]
struct ActivityEvent {
    when: DateTime<Utc>,
    source: Source,
    summary: String,
}

#[derive(Default, Clone, Debug)]
struct PaneCounts {
    busy: usize,
    idle: usize,
    /// Busy panes that have not yet started `claude` — still spinning up.
    starting: usize,
}

#[derive(Clone, Debug)]
struct CronSummary {
    scheduled: usize,
    last_run: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug)]
struct Environment {
    cwd: String,
    git_branch: String,
    sandbox: String,
}

#[derive(Clone, Debug)]
struct Snapshot {
    env: Environment,
    panes: PaneCounts,
    cron: CronSummary,
    inbox_unread: usize,
    activity: Vec<ActivityEvent>,
}

impl Snapshot {
    fn collect() -> Self {
        let env = collect_env();
        let panes_vec = crate::pane_lease::read_state().unwrap_or_default();
        let panes = count_panes(&panes_vec.values().cloned().collect::<Vec<_>>());

        // All three SQLite sources are opened as "best-effort read".  If the
        // DB file does not exist yet (user hasn't used that feature), skip
        // the open entirely so the dashboard never creates empty .db/WAL/SHM
        // files as a side effect of running `amaebi dashboard`.
        let inbox_reports = if amaebi_file_exists("inbox.db") {
            crate::inbox::InboxStore::open()
                .and_then(|s| s.get_all())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let inbox_unread = inbox_reports.iter().filter(|r| !r.read).count();

        let cron_jobs = if amaebi_file_exists("cron.db") {
            crate::cron::load_jobs().unwrap_or_default()
        } else {
            Vec::new()
        };
        let cron = summarize_cron(&cron_jobs);

        let session_events = if memory_db_exists() {
            collect_session_events().unwrap_or_default()
        } else {
            Vec::new()
        };

        let mut activity = Vec::new();
        activity.extend(collect_pane_events(panes_vec.values()));
        activity.extend(collect_cron_events(&cron_jobs));
        activity.extend(collect_inbox_events(&inbox_reports));
        activity.extend(session_events);
        finalize_activity(&mut activity);

        Self {
            env,
            panes,
            cron,
            inbox_unread,
            activity,
        }
    }
}

/// Sort `events` newest-first and truncate to [`ACTIVITY_CAP`].
///
/// Extracted from `Snapshot::collect` so tests can exercise the real
/// merge/sort/truncate pipeline instead of re-implementing it.
fn finalize_activity(events: &mut Vec<ActivityEvent>) {
    events.sort_by_key(|e| std::cmp::Reverse(e.when));
    events.truncate(ACTIVITY_CAP);
}

/// True when `~/.amaebi/<name>` exists.  Used to skip opening SQLite
/// connections for sources the user hasn't initialised yet, so the
/// dashboard never creates empty DB/WAL/SHM files as a side effect.
fn amaebi_file_exists(name: &str) -> bool {
    crate::auth::amaebi_home()
        .map(|p| p.join(name).exists())
        .unwrap_or(false)
}

fn memory_db_exists() -> bool {
    crate::memory_db::db_path()
        .map(|p| p.exists())
        .unwrap_or(false)
}

fn collect_env() -> Environment {
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "(unknown)".to_string());

    let cwd_for_git = cwd.clone();
    let git_branch = git_output(Some(&cwd_for_git), &["rev-parse", "--abbrev-ref", "HEAD"]);

    let sandbox = match std::env::var("AMAEBI_SANDBOX").as_deref() {
        Ok("docker") => {
            let image = std::env::var("AMAEBI_SANDBOX_IMAGE")
                .unwrap_or_else(|_| "amaebi-sandbox:bookworm-slim".to_string());
            format!("docker ({image})")
        }
        _ => "off".to_string(),
    };

    Environment {
        cwd,
        git_branch: if git_branch.is_empty() {
            "(not a git repo)".to_string()
        } else {
            git_branch
        },
        sandbox,
    }
}

// Re-use the daemon's best-effort git helper so the two stay in sync.
use crate::daemon::git_output;

fn count_panes(panes: &[PaneLease]) -> PaneCounts {
    let mut out = PaneCounts::default();
    for p in panes {
        match p.effective_status() {
            PaneStatus::Busy => {
                if p.has_claude {
                    out.busy += 1;
                } else {
                    // Busy but claude not yet running = still starting up.
                    out.starting += 1;
                }
            }
            PaneStatus::Idle => out.idle += 1,
        }
    }
    out
}

fn summarize_cron(jobs: &[CronJob]) -> CronSummary {
    let last_run = jobs
        .iter()
        .filter_map(|j| j.last_run.as_deref().and_then(parse_rfc3339))
        .max();
    CronSummary {
        scheduled: jobs.len(),
        last_run,
    }
}

fn parse_rfc3339(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|d| d.with_timezone(&Utc))
}

fn from_unix_secs(ts: u64) -> Option<DateTime<Utc>> {
    Utc.timestamp_opt(ts as i64, 0).single()
}

fn short_uuid(uuid: &str) -> String {
    uuid.chars().take(8).collect()
}

fn truncate_snippet(s: &str, max: usize) -> String {
    // Drop C0 control chars (ESC, BEL, bare CR, etc.) so ANSI escape
    // sequences stored in memory/inbox content cannot escape the ratatui
    // buffer and mess up the dashboard's rendering.  Keep \n/\t and collapse
    // whitespace.
    let compact: String = s
        .chars()
        .map(|c| if c == '\n' { ' ' } else { c })
        .filter(|c| !c.is_control() || *c == '\t')
        .collect();
    let compact = compact.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() <= max {
        compact
    } else {
        let head: String = compact.chars().take(max).collect();
        format!("{head}…")
    }
}

// ---------------------------------------------------------------------------
// Event collectors (pure, individually testable)
// ---------------------------------------------------------------------------

fn collect_pane_events<'a, I>(panes: I) -> Vec<ActivityEvent>
where
    I: IntoIterator<Item = &'a PaneLease>,
{
    let mut events = Vec::new();
    for p in panes {
        let Some(when) = from_unix_secs(p.heartbeat_at) else {
            continue;
        };
        let status = match p.effective_status() {
            PaneStatus::Busy if p.has_claude => "busy",
            PaneStatus::Busy => "starting",
            PaneStatus::Idle => "idle",
        };
        let task = p.tag.as_deref().unwrap_or("(idle)");
        let summary = format!("{status} — {task}");
        events.push(ActivityEvent {
            when,
            source: Source::Pane(p.pane_id.clone()),
            summary,
        });
    }
    events
}

fn collect_cron_events(jobs: &[CronJob]) -> Vec<ActivityEvent> {
    jobs.iter()
        .filter_map(|j| {
            let last = j.last_run.as_deref().and_then(parse_rfc3339)?;
            Some(ActivityEvent {
                when: last,
                source: Source::Cron,
                summary: format!("\"{}\" ran", j.description),
            })
        })
        .collect()
}

fn collect_inbox_events(reports: &[InboxReport]) -> Vec<ActivityEvent> {
    reports
        .iter()
        .filter_map(|r| {
            let when = parse_rfc3339(&r.created_at)?;
            let read_marker = if r.read { "" } else { " (unread)" };
            Some(ActivityEvent {
                when,
                source: Source::Inbox(r.id),
                summary: format!("{}{}", r.task_description, read_marker),
            })
        })
        .collect()
}

fn collect_session_events() -> Result<Vec<ActivityEvent>> {
    // Reading memory.db requires opening a fresh connection each tick; keep
    // it best-effort so a missing file or schema mismatch does not kill the
    // dashboard.
    let path = memory_db::db_path()?;
    if !path.exists() {
        return Ok(Vec::new());
    }
    let conn = memory_db::init_db(&path)?;
    let entries = memory_db::get_latest_per_session(&conn, SESSION_EVENT_CAP)?;
    let mut events = Vec::new();
    for e in entries {
        let Some(when) = parse_rfc3339(&e.timestamp) else {
            continue;
        };
        let snippet = truncate_snippet(&e.content, PROMPT_SNIPPET_CHARS);
        let summary = format!("{}: \"{}\"", e.role, snippet);
        events.push(ActivityEvent {
            when,
            source: Source::Session(short_uuid(&e.session_id)),
            summary,
        });
    }
    Ok(events)
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

fn relative_time(when: DateTime<Utc>, now: DateTime<Utc>) -> String {
    let delta = now.signed_duration_since(when);
    let secs = delta.num_seconds();
    if secs < 0 {
        return "soon".to_string();
    }
    if secs < 60 {
        return format!("{secs}s ago");
    }
    let mins = secs / 60;
    if mins < 60 {
        return format!("{mins}m ago");
    }
    let hours = mins / 60;
    if hours < 24 {
        return format!("{hours}h ago");
    }
    let days = hours / 24;
    format!("{days}d ago")
}

fn format_source(src: &Source) -> String {
    match src {
        Source::Pane(id) => format!("[pane {id}]"),
        Source::Session(id) => format!("[session {id}]"),
        Source::Cron => "[cron]".to_string(),
        Source::Inbox(id) => format!("[inbox #{id}]"),
    }
}

fn env_lines(env: &Environment) -> Vec<Line<'_>> {
    vec![
        Line::from(vec![
            Span::styled("cwd: ", Style::default().fg(Color::DarkGray)),
            Span::raw(env.cwd.clone()),
            Span::raw("   "),
            Span::styled("branch: ", Style::default().fg(Color::DarkGray)),
            Span::raw(env.git_branch.clone()),
        ]),
        Line::from(vec![
            Span::styled("sandbox: ", Style::default().fg(Color::DarkGray)),
            Span::raw(env.sandbox.clone()),
        ]),
    ]
}

fn summary_lines(snap: &Snapshot, now: DateTime<Utc>) -> Vec<Line<'_>> {
    let panes_line = format!(
        "Panes    {busy} busy · {idle} idle · {starting} starting",
        busy = snap.panes.busy,
        idle = snap.panes.idle,
        starting = snap.panes.starting,
    );
    let last_run = snap
        .cron
        .last_run
        .map(|t| relative_time(t, now))
        .unwrap_or_else(|| "never".to_string());
    let cron_line = format!(
        "Cron     {n} scheduled · last {last}",
        n = snap.cron.scheduled,
        last = last_run,
    );
    let inbox_line = format!("Inbox    {} unread", snap.inbox_unread);
    vec![
        Line::from(panes_line),
        Line::from(cron_line),
        Line::from(inbox_line),
    ]
}

fn activity_lines(events: &[ActivityEvent], now: DateTime<Utc>) -> Vec<Line<'_>> {
    if events.is_empty() {
        return vec![Line::from(Span::styled(
            "(no activity yet — start a chat, spawn a pane, or wait for cron)",
            Style::default().fg(Color::DarkGray),
        ))];
    }
    events
        .iter()
        .map(|e| {
            let time = relative_time(e.when, now);
            let src = format_source(&e.source);
            Line::from(vec![
                Span::styled(format!("{time:>8}  "), Style::default().fg(Color::DarkGray)),
                Span::styled(format!("{src:<18} "), Style::default().fg(Color::Cyan)),
                Span::raw(e.summary.clone()),
            ])
        })
        .collect()
}

fn draw(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    snap: &Snapshot,
    now: DateTime<Utc>,
) -> Result<()> {
    terminal
        .draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(5),
                    Constraint::Length(5),
                    Constraint::Min(5),
                ])
                .split(f.area());

            let env_block = Paragraph::new(env_lines(&snap.env)).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Environment "),
            );
            f.render_widget(env_block, chunks[0]);

            let summary_block = Paragraph::new(summary_lines(snap, now)).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Task summary "),
            );
            f.render_widget(summary_block, chunks[1]);

            let activity_title = format!(
                " Activity (last {}, newest first — q quit, r refresh) ",
                snap.activity.len()
            );
            let activity_block = Paragraph::new(activity_lines(&snap.activity, now)).block(
                Block::default().borders(Borders::ALL).title(Span::styled(
                    activity_title,
                    Style::default().add_modifier(Modifier::BOLD),
                )),
            );
            f.render_widget(activity_block, chunks[2]);
        })
        .context("drawing dashboard frame")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Entry point + terminal lifecycle
// ---------------------------------------------------------------------------

struct TerminalGuard;

impl TerminalGuard {
    fn enter() -> Result<Self> {
        enable_raw_mode().context("enabling raw mode")?;
        // If entering the alt screen fails we need to roll raw mode back;
        // otherwise the caller's terminal is left in a broken state.
        if let Err(e) = execute!(stdout(), EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(anyhow::Error::new(e).context("entering alternate screen"));
        }
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(stdout(), LeaveAlternateScreen);
    }
}

/// Run the dashboard until the user exits.
pub async fn run() -> Result<()> {
    // Spin up terminal in a `spawn_blocking` so we don't hold the runtime
    // busy in the raw-mode polling loop.
    tokio::task::spawn_blocking(run_blocking)
        .await
        .context("dashboard task panicked")?
}

fn run_blocking() -> Result<()> {
    let _guard = TerminalGuard::enter()?;
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend).context("creating terminal")?;

    let mut snap = Snapshot::collect();
    let mut last_refresh = Instant::now();
    draw(&mut terminal, &snap, Utc::now())?;

    loop {
        // Poll with a short timeout so we can both react to keys and refresh.
        let remaining = REFRESH.saturating_sub(last_refresh.elapsed());
        let poll_timeout = remaining.min(Duration::from_millis(250));
        let had_event = event::poll(poll_timeout).context("polling terminal events")?;
        if had_event {
            match event::read().context("reading terminal event")? {
                Event::Key(KeyEvent {
                    code, modifiers, ..
                }) => {
                    let ctrl_c = modifiers.contains(KeyModifiers::CONTROL)
                        && matches!(code, KeyCode::Char('c') | KeyCode::Char('C'));
                    let quit = ctrl_c
                        || matches!(code, KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc);
                    if quit {
                        break;
                    }
                    if matches!(code, KeyCode::Char('r') | KeyCode::Char('R')) {
                        snap = Snapshot::collect();
                        last_refresh = Instant::now();
                        draw(&mut terminal, &snap, Utc::now())?;
                    }
                }
                Event::Resize(_, _) => {
                    draw(&mut terminal, &snap, Utc::now())?;
                }
                _ => {}
            }
        }
        if last_refresh.elapsed() >= REFRESH {
            snap = Snapshot::collect();
            last_refresh = Instant::now();
            draw(&mut terminal, &snap, Utc::now())?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cron::CronJob;
    use crate::inbox::{InboxReport, InboxStore};
    use crate::pane_lease::{PaneLease, PaneStatus};
    use crate::test_utils::with_temp_home;

    fn pane(
        id: &str,
        status: PaneStatus,
        has_claude: bool,
        heartbeat: u64,
        task: Option<&str>,
    ) -> PaneLease {
        PaneLease {
            pane_id: id.to_string(),
            window_id: "@0".to_string(),
            status,
            tag: task.map(String::from),
            session_id: None,
            worktree: None,
            heartbeat_at: heartbeat,
            has_claude,
            task_description: None,
        }
    }

    fn now_secs() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    #[test]
    fn count_panes_separates_busy_idle_starting() {
        // Use fresh heartbeats so effective_status() doesn't downgrade Busy
        // past LEASE_TTL_SECS.
        let now = now_secs();
        let panes = vec![
            pane("%1", PaneStatus::Busy, true, now, Some("pr-1")),
            pane("%2", PaneStatus::Busy, false, now, Some("pr-2")), // starting
            pane("%3", PaneStatus::Idle, true, now, None),
            pane("%4", PaneStatus::Idle, false, now, None),
        ];
        let c = count_panes(&panes);
        assert_eq!(c.busy, 1);
        assert_eq!(c.starting, 1);
        assert_eq!(c.idle, 2);
    }

    #[test]
    fn collect_pane_events_one_per_pane() {
        let now = now_secs();
        let panes = vec![
            pane("%1", PaneStatus::Busy, true, now, Some("pr-1")),
            pane("%2", PaneStatus::Idle, false, now, None),
        ];
        let events: Vec<_> = collect_pane_events(panes.iter()).into_iter().collect();
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0].source, Source::Pane(ref id) if id == "%1"));
        assert!(events[0].summary.contains("busy"));
        assert!(events[1].summary.contains("idle"));
    }

    #[test]
    fn collect_cron_events_skips_jobs_with_no_last_run() {
        let jobs = vec![
            CronJob {
                id: "a".into(),
                description: "ran".into(),
                schedule: "* * * * *".into(),
                created_at: "2026-04-22T00:00:00Z".into(),
                last_run: Some("2026-04-22T10:00:00Z".into()),
            },
            CronJob {
                id: "b".into(),
                description: "never".into(),
                schedule: "* * * * *".into(),
                created_at: "2026-04-22T00:00:00Z".into(),
                last_run: None,
            },
        ];
        let events = collect_cron_events(&jobs);
        assert_eq!(events.len(), 1);
        assert!(events[0].summary.contains("\"ran\""));
    }

    #[test]
    fn collect_inbox_events_tags_unread() {
        let reports = vec![
            InboxReport {
                id: 1,
                session_id: "s1".into(),
                task_description: "unread task".into(),
                output: String::new(),
                created_at: "2026-04-22T09:00:00Z".into(),
                read: false,
            },
            InboxReport {
                id: 2,
                session_id: "s2".into(),
                task_description: "read task".into(),
                output: String::new(),
                created_at: "2026-04-22T08:00:00Z".into(),
                read: true,
            },
        ];
        let events = collect_inbox_events(&reports);
        assert_eq!(events.len(), 2);
        assert!(events[0].summary.contains("(unread)"));
        assert!(!events[1].summary.contains("(unread)"));
    }

    #[test]
    fn snapshot_sort_is_descending_and_capped() {
        // Build unsorted events and run them through the same helper the
        // production Snapshot::collect uses, so this test fails if the
        // merge/sort/truncate logic regresses.
        let now = Utc::now();
        let mut events: Vec<ActivityEvent> = (0..50)
            .map(|i| ActivityEvent {
                // Interleave so the input is not already sorted.
                when: now - chrono::Duration::minutes((i * 37) % 50),
                source: Source::Cron,
                summary: format!("e{i}"),
            })
            .collect();
        finalize_activity(&mut events);
        assert_eq!(events.len(), ACTIVITY_CAP);
        // Newest first.
        for w in events.windows(2) {
            assert!(w[0].when >= w[1].when);
        }
    }

    #[test]
    fn snapshot_collect_runs_with_empty_home() {
        let _guard = with_temp_home();
        // Should not panic or hang with zero data sources in a fresh HOME.
        let snap = Snapshot::collect();
        assert_eq!(snap.panes.busy, 0);
        assert_eq!(snap.panes.idle, 0);
        assert_eq!(snap.cron.scheduled, 0);
        assert_eq!(snap.inbox_unread, 0);
    }

    #[test]
    fn snapshot_collect_sees_inbox_unread() {
        let _guard = with_temp_home();
        let store = InboxStore::open().unwrap();
        store.save_report("sid", "task 1", "out").unwrap();
        store.save_report("sid", "task 2", "out").unwrap();

        let snap = Snapshot::collect();
        assert_eq!(snap.inbox_unread, 2);
        // Inbox events should be present.
        assert!(snap
            .activity
            .iter()
            .any(|e| matches!(e.source, Source::Inbox(_))));
    }

    #[test]
    fn relative_time_formats() {
        let now = Utc::now();
        assert!(relative_time(now, now).ends_with("s ago"));
        assert_eq!(
            relative_time(now - chrono::Duration::minutes(5), now),
            "5m ago"
        );
        assert_eq!(
            relative_time(now - chrono::Duration::hours(2), now),
            "2h ago"
        );
        assert_eq!(
            relative_time(now - chrono::Duration::days(3), now),
            "3d ago"
        );
    }

    #[test]
    fn truncate_snippet_respects_char_boundary() {
        let s = "abcdefghij";
        assert_eq!(truncate_snippet(s, 5), "abcde…");
        assert_eq!(truncate_snippet(s, 20), "abcdefghij");
        // Multi-line collapses.
        assert_eq!(truncate_snippet("a\nb\nc", 20), "a b c");
    }

    #[test]
    fn truncate_snippet_strips_control_chars() {
        // Stripping the leading ESC is enough to neutralise the sequence —
        // the terminal can no longer interpret `[31m...` as a color code
        // without the preceding ESC, so the TUI buffer stays intact.  The
        // residual `[31m` characters are harmless ASCII.
        let out = truncate_snippet("\x1b[31mred\x1b[0m done", 50);
        assert!(!out.contains('\x1b'), "ESC must be stripped: {out:?}");
        assert_eq!(truncate_snippet("before\x07after", 50), "beforeafter");
        // Tab is preserved (gets collapsed by split_whitespace).
        assert_eq!(truncate_snippet("a\tb", 50), "a b");
    }

    #[test]
    fn short_uuid_is_first_eight() {
        assert_eq!(
            short_uuid("7fd69d7f-9195-4d98-bfe5-6b61d887ec97"),
            "7fd69d7f"
        );
    }

    #[test]
    fn format_source_shapes() {
        assert_eq!(format_source(&Source::Cron), "[cron]");
        assert_eq!(format_source(&Source::Inbox(17)), "[inbox #17]");
        assert_eq!(format_source(&Source::Pane("%3".into())), "[pane %3]");
        assert_eq!(
            format_source(&Source::Session("abc".into())),
            "[session abc]"
        );
    }
}
