//! `amaebi chat --tui` — experimental split-screen interactive TUI.
//!
//! The long-term goal is to replace the classic `run_chat_loop` line-based
//! UI with a split-screen layout (transcript on top, persistent input box
//! on the bottom) so that streaming assistant output and user typing never
//! compete for the same terminal cursor.  See discussion in the 2026-05-14
//! chat thread and feedback memory `feedback_prefer_libraries` for
//! rationale.
//!
//! **Currently in development on `feat/tui`.**  The structural pieces
//! land first; behaviour gets ported step by step.  Already wired:
//!
//! - Split layout (alt screen + raw mode + transcript / input split).
//! - Streaming `Response::Text` chunks render into the transcript
//!   while the user types in the bottom box without any cursor war.
//! - `/model` (show / switch) — daemon-side `Response::ModelSwitched`
//!   also keeps `state.model` in sync.
//! - CJK / fullwidth / emoji input (cursor positioned by display
//!   width, not by Unicode scalar count).
//! - Cursor-aware line editing: Left / Right / Home / End / Delete /
//!   Backspace / Ctrl-A / Ctrl-E / Ctrl-B / Ctrl-F / Ctrl-K / Ctrl-U.
//!   All step by full Unicode scalars so CJK input doesn't desync.
//! - Startup banner (logo + version + model + sandbox + session +
//!   cwd) rendered into the transcript so it survives the alt-screen
//!   switch instead of being cleared by the first ratatui draw.
//! - History navigation: ↑/↓ walk through this cwd's prior prompts
//!   (loaded from `~/.amaebi/history.jsonl` at startup, appended to
//!   on each submit).  ↑ from a draft snapshots it; ↓ past the
//!   newest entry restores it.
//! - `/claude` and `/replyreview` (full launch flow): tag generation,
//!   `Request::ClaudeLaunch`, `PaneAssigned` accumulation, and the
//!   synthesised `[launched]` supervision prompt that hands control
//!   over to the chat-takeover loop.
//! - Char-grid wrap (no word-boundary surprises) for both transcript
//!   and input box, sized correctly for CJK / fullwidth glyphs.
//! - Mid-turn Ctrl-C steer: first Ctrl-C while streaming sends
//!   `Request::Interrupt`, buffers subsequent stream output, and
//!   prompts for a correction; Enter submits the correction as a
//!   `Request::Steer`; empty Enter cancels and flushes the buffer
//!   back into the transcript; second Ctrl-C exits.
//! - `--resume <UUID>`: passes through to the daemon's session
//!   rehydrate (same behaviour as `amaebi chat -r=<UUID>` classic).
//! - Plan progress: shares the parser from PR #157, surfaces the
//!   live `[plan N/M done]` count in the input box title and pins
//!   the final state into the transcript on Done.
//! - Per-kind transcript colour + glyph: ToolUse magenta with a
//!   per-tool glyph (📄 read / ✏️ edit / ⌨️ tmux / 🔧 generic),
//!   Compacting yellow with ⏳, Steer yellow with ↳, Launch
//!   (PaneAssigned) green with 🚀, errors red with `!`.
//! - PgUp / PgDn scrollback inside the transcript.  Title bar
//!   shows "↑ N rows from tail" while pinned; PgDn down to 0 (or
//!   keep pressing) returns to follow-tail.
//! - `/release` (full release flow, same as classic chat).
//! - Inline markdown rendering: backticks → `Code` style, `**bold**`,
//!   `*italic*` / `_italic_`, plus heading-level prefixes for `# `,
//!   `## `, `### `.

use std::collections::VecDeque;
use std::io::stdout;
use std::path::PathBuf;

use anyhow::{Context, Result};
use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use futures_util::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Terminal;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::client::{parse_slash_command, SlashCommand};
use crate::ipc::{Request, Response};
use crate::provider;
use crate::session;

/// How long the user has between two Ctrl-C presses to trigger an
/// exit.  Mirrors classic chat's `DOUBLE_CTRLC_WINDOW` so the muscle
/// memory is identical between the two UIs.
const DOUBLE_CTRLC_WINDOW: std::time::Duration = std::time::Duration::from_secs(2);

/// One-line hint shown the first time the user presses Ctrl-C with
/// an empty input box.  Single string so the de-dup check in
/// `handle_key` can compare cheaply (no need to spam the transcript
/// when the user mashes Ctrl-C).
const CTRLC_EXIT_HINT: &str =
    "press Ctrl-C again within 2s to exit (or type a message and Enter to continue)";

/// Number of visual rows PgUp / PgDn step through the transcript.
/// We don't know the live viewport height from inside `handle_key`,
/// so the page step is a reasonable approximation that works on
/// any normal-sized terminal — small enough that even a 20-row
/// window scrolls usefully, big enough to feel like a real page.
const PAGE_STEP_ROWS: u16 = 10;

/// Cap on how many `Response` frames we buffer while `steer_pending`
/// is true.  Mirrors classic chat's `STEER_BUFFER_MAX_FRAMES`.  A
/// long-running tool-heavy turn could blast hundreds of frames into
/// the buffer if the user takes their time typing the correction;
/// past this cap we evict oldest-first and emit a single truncation
/// notice on flush so the user knows some frames were dropped.
const STEER_BUFFER_MAX_FRAMES: usize = 1000;

/// Public entry — called from `main.rs` when `--tui` is set.
///
/// Mirrors the `run_chat_loop` signature so the two paths are
/// interchangeable at the call site.  When `resumed_session_id` is
/// `Some`, the daemon reuses its in-memory + SQLite history for that
/// session UUID on every `Request::Chat` (same behaviour as classic
/// `amaebi chat -r=<UUID>` — the dedicated full-rehydrate path is
/// `amaebi resume`, which is a separate one-shot subcommand and
/// not exercised here).
pub async fn run_chat_tui(
    socket: PathBuf,
    initial_prompt: Option<String>,
    model: Option<String>,
    resumed_session_id: Option<String>,
) -> Result<()> {
    let model = model
        .or_else(|| std::env::var("AMAEBI_MODEL").ok())
        .unwrap_or_else(|| provider::DEFAULT_MODEL.to_string());

    let cwd = std::env::current_dir().context("getting current directory")?;
    // True iff the user passed `-r=<UUID>` and we'll resume that
    // session.  Used purely for a startup breadcrumb; the daemon
    // does the actual rehydrate based on session_id alone.
    let is_resumed = resumed_session_id.is_some();
    let session_id = match resumed_session_id {
        Some(id) => id,
        None => {
            let cwd_for_session = cwd.clone();
            tokio::task::spawn_blocking(move || session::create_fresh(&cwd_for_session))
                .await
                .context("session::create_fresh panicked")?
                .unwrap_or_else(|e| {
                    tracing::warn!(error = %e, "failed to create fresh session id; using \"global\"");
                    "global".to_string()
                })
        }
    };

    let stream = crate::client::connect_or_start_daemon(&socket).await?;
    let (read_half, mut write_half) = stream.into_split();
    let mut daemon_lines = BufReader::new(read_half).lines();

    // Enter raw + alt screen now.  The TerminalGuard's Drop restores
    // cooked mode + the normal screen even on panic / early return, so
    // a crash inside this function cannot leave the user's shell
    // broken.
    let _guard = TerminalGuard::enter()?;
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend).context("creating ratatui terminal")?;

    let cwd_str = cwd.to_string_lossy().into_owned();
    let mut state = AppState::new(session_id.clone(), model.clone(), socket.clone(), cwd_str);
    // Load this cwd's prior prompts so ↑/↓ can recall them.  Done in
    // a spawn_blocking because the loader does sync file I/O (seek +
    // read up to LOAD_TAIL_BYTES) and we don't want to stall the
    // tokio runtime waiting on disk.  Set the session id first so
    // subsequent record_history_line calls tag fresh rows correctly.
    crate::client::set_history_session_id(&session_id);
    state.history = tokio::task::spawn_blocking(crate::client::load_cwd_history)
        .await
        .unwrap_or_default();

    push_banner(&mut state, &cwd);
    if is_resumed {
        state.push_system_line(format!(
            "[resumed] daemon will rehydrate prior turns for session {}.",
            &session_id[..8.min(session_id.len())]
        ));
    }
    state.push_system_line(
        "Type a message and press Enter to send.  ↑/↓ for history.  Ctrl-C / Ctrl-D exits.".into(),
    );
    state.push_system_line(String::new());

    // Pump an optional opening prompt synthetically so the user doesn't
    // have to re-type it after `--tui` is set.
    if let Some(opening) = initial_prompt {
        if !opening.trim().is_empty() {
            send_prompt(&mut write_half, &mut state, opening).await?;
        }
    }

    // Render once before entering the event loop so the user sees the
    // chrome before any input/output happens.
    draw(&mut terminal, &state)?;

    let mut key_events = EventStream::new();

    loop {
        tokio::select! {
            // Keyboard / resize events from the terminal.  EventStream
            // uses a dedicated mio thread internally (set up by the
            // `event-stream` crossterm feature), so this future
            // composes cleanly with tokio's select!.
            maybe_event = key_events.next() => {
                match maybe_event {
                    Some(Ok(Event::Key(key))) => {
                        match handle_key(key, &mut state) {
                            KeyOutcome::Continue => {}
                            KeyOutcome::SubmitInput(text) => {
                                if !text.trim().is_empty() && !state.streaming {
                                    dispatch_input(&mut write_half, &mut state, text).await?;
                                }
                            }
                            KeyOutcome::InterruptForSteer => {
                                send_interrupt_and_arm_steer(&mut write_half, &mut state).await?;
                            }
                            KeyOutcome::SubmitSteer(text) => {
                                send_steer(&mut write_half, &mut state, text).await?;
                            }
                            KeyOutcome::CancelSteer => {
                                cancel_steer(&mut write_half, &mut state).await?;
                            }
                            KeyOutcome::Exit => break,
                        }
                        draw(&mut terminal, &state)?;
                    }
                    Some(Ok(Event::Resize(_, _))) => {
                        draw(&mut terminal, &state)?;
                    }
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        tracing::warn!(error = %e, "crossterm event stream error; exiting TUI");
                        break;
                    }
                    None => break,
                }
            }

            // Daemon response frames.  Arrives as newline-delimited JSON
            // on the same Unix socket as classic chat.
            frame = daemon_lines.next_line() => {
                match frame {
                    Ok(Some(line)) => {
                        match serde_json::from_str::<Response>(&line) {
                            Ok(resp) => {
                                let outcome = handle_response(resp, &mut state);
                                // Some outcomes need an async follow-up
                                // that handle_response can't perform —
                                // /claude's "send synthesised
                                // supervision prompt" is the main one.
                                if let ResponseOutcome::TurnEndedSendSynth(synth) = outcome {
                                    send_prompt(&mut write_half, &mut state, synth).await?;
                                }
                                draw(&mut terminal, &state)?;
                            }
                            Err(e) => {
                                state.push_error_line(format!(
                                    "decode frame failed: {e}  raw={line}"
                                ));
                                draw(&mut terminal, &state)?;
                            }
                        }
                    }
                    Ok(None) => {
                        state.push_error_line("daemon closed the connection".to_string());
                        draw(&mut terminal, &state)?;
                        break;
                    }
                    Err(e) => {
                        state.push_error_line(format!("daemon read error: {e}"));
                        draw(&mut terminal, &state)?;
                        break;
                    }
                }
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

/// What armed steer mode.  See `AppState::steer_source` for why it
/// matters on the cancel path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SteerSource {
    /// No steer in flight.  Default value; not all `steer_pending=false`
    /// states explicitly reset this, but the field is only read when
    /// `steer_pending=true` so the lingering value is harmless.
    Idle,
    /// User pressed Ctrl-C mid-stream — `Request::Interrupt` was
    /// already shipped, so cancel is a local-only operation.
    UserCtrlC,
    /// Daemon emitted `Response::WaitingForInput` — the daemon's
    /// reply consumer is parked waiting for the user's typed
    /// reply.  A bare `Request::Interrupt` does NOT unblock it
    /// (daemon explicitly ignores interrupts in that loop, see
    /// `daemon.rs` "interrupt ignored while waiting for question
    /// reply"); the only way out is a real Steer message, a
    /// disconnect, or the daemon's 300s timeout.  `cancel_steer`
    /// therefore leaves `steer_pending` set and just shows a hint.
    DaemonWaitingForInput,
}

/// Where a transcript line originated from — controls colour / prefix.
///
/// We deliberately keep the variant set compact: each one maps to a
/// distinct (style, prefix-glyph) pair in `push_wrapped_transcript_line`,
/// and a richer set is more visual noise than information.  Anything
/// that doesn't fit one of the named buckets goes through `System`.
#[derive(Clone, Copy)]
enum LineKind {
    /// Default amaebi metadata — greeting, banner, model switches,
    /// session announcements, anything not specifically classified.
    System,
    /// A user prompt we just submitted.  Rendered in a distinct colour
    /// so the transcript is readable when scrolling.
    User,
    /// Streaming assistant reply.  `is_open` means the last chunk did
    /// not end in a newline, so the next text delta should continue
    /// this line rather than start a new one.
    Assistant { is_open: bool },
    /// Hard errors from the daemon or protocol.  Red `! ` prefix.
    Error,
    /// `Response::ToolUse` notices.  Per-tool glyph + magenta tint to
    /// distinguish tool activity from model text and from amaebi's
    /// own system breadcrumbs.
    Tool,
    /// `Response::Compacting` background work.  Yellow.
    Compacting,
    /// Steer breadcrumbs ("[steer] type a correction…", etc.).
    /// Yellow + `↳` glyph so the user can spot the steer mode change
    /// at a glance.
    Steer,
    /// `Response::PaneAssigned` and the synthesised `[launched]`
    /// announcement — green so successful launches stand out.
    Launch,
}

struct TranscriptLine {
    kind: LineKind,
    text: String,
}

struct AppState {
    session_id: String,
    /// The model used for the next outgoing `Request::Chat`.  Mutated
    /// by `/model <name>` (see `dispatch_input`) and by daemon-side
    /// `Response::ModelSwitched` events on the server's request.
    model: String,
    transcript: Vec<TranscriptLine>,
    input: String,
    input_cursor: usize,
    streaming: bool,
    /// Persisted history (oldest-first) of past prompts the user
    /// submitted in this cwd.  Populated once at startup from
    /// `~/.amaebi/history.jsonl`; appended to in-place every time
    /// the user submits a new line so ↑/↓ within a single session
    /// can recall what was just sent.
    history: Vec<String>,
    /// Where the user is in the history when scrolling with ↑/↓.
    /// `None` means "not in history mode — the input box shows a
    /// fresh draft".  `Some(i)` means we're showing `history[i]`.
    /// Indices count from the END of `history` (i.e. `Some(0)` is
    /// the most recent submitted prompt) so newly-appended history
    /// rows don't shift the user's position out from under them.
    history_pos: Option<usize>,
    /// The user's in-progress draft, captured the first time they
    /// press ↑ from a non-history-mode input.  Restored when they
    /// press ↓ past the most-recent history entry, so arrowing up
    /// then down doesn't lose what they were typing.
    history_draft: String,
    /// True between the first mid-turn Ctrl-C and either a successful
    /// steer (sent + `Response::SteerAck` received), an empty-Enter
    /// cancel, or a second Ctrl-C exit.  While set, `Response::Text`
    /// / `ToolUse` / `Compacting` frames are buffered into
    /// `steer_buffer` instead of going to the transcript so they
    /// don't fight the steer prompt for screen real estate.
    steer_pending: bool,
    /// What armed the steer mode — affects how we handle the
    /// "empty Enter cancels" path.  Ctrl-C-armed steer can simply
    /// drop the local flag (the daemon already received an
    /// Interrupt on the first Ctrl-C).  WaitingForInput-armed
    /// steer cannot be cancelled by an Interrupt at all: the
    /// daemon's question-reply consumer (see `daemon.rs` —
    /// "interrupt ignored while waiting for question reply")
    /// drops bare interrupts and keeps blocking.  The TUI handles
    /// that asymmetry in `cancel_steer_local` by leaving
    /// `steer_pending = true` and pushing a hint instead of
    /// shipping a useless Interrupt.
    steer_source: SteerSource,
    /// Frames received while `steer_pending`, replayed through
    /// `handle_response` once steering ends.  Capped at
    /// `STEER_BUFFER_MAX_FRAMES` (oldest-first eviction); when an
    /// eviction has happened the next flush prepends a truncation
    /// notice so the user knows some output was dropped.
    /// `VecDeque` so cap-eviction is O(1) `pop_front` instead of
    /// O(n) `Vec::remove(0)` — matters under heavy tool/stream
    /// output that could otherwise touch the cap repeatedly.
    steer_buffer: VecDeque<Response>,
    steer_buffer_truncated: bool,
    /// Timestamp of the last Ctrl-C press, used to detect the
    /// double-Ctrl-C-within-window exit gesture.  Cleared every time
    /// we leave the steer/exit-pending state.
    last_ctrl_c: Option<std::time::Instant>,
    /// User's scroll position in the transcript, expressed as
    /// "rows above the tail-following position".  `0` (or any value
    /// that would put us at or past the tail anyway) means "follow
    /// tail" — every new frame auto-scrolls so the latest content
    /// stays visible.  Any other value means the user has scrolled
    /// up and wants to stay there even as new content lands.
    /// PgDn down to 0 (or End) restores follow-tail.
    transcript_scroll_back: u16,
    /// Per-turn parser for the LLM's `- [ ] / - [x]` checklist.
    /// Inherits the parser logic from classic chat (PR #157) but
    /// surfaces the result as a transient status line drawn in the
    /// input title, not as a stderr `\r\x1b[K` overwrite.  Reset
    /// (`PlanProgressTracker::new(false)`) on every `Response::Done`
    /// so progress from one turn doesn't leak into the next.
    plan_tracker: crate::client::PlanProgressTracker,
    /// Set while a `/claude` (or `/replyreview`) launch is in flight.
    /// Holds the tag→description map needed to reconstitute the
    /// `[launched]` block once the daemon emits `Response::Done`,
    /// plus a running list of `PaneAssigned` frames received so far.
    /// Cleared the moment the synthesised supervision prompt is sent.
    pending_claude: Option<PendingClaudeLaunch>,
    /// Absolute path to the daemon socket — needed by /claude to
    /// open a side-channel for `Request::GenerateTag`.  Set once at
    /// startup; never mutated.
    socket_path: PathBuf,
    /// Canonical cwd at startup, snapshotted so /claude can pin
    /// `client_cwd` in TaskSpec the same way classic chat does.
    cwd_str: String,
}

/// One pane that the daemon has assigned to us during a /claude
/// launch.  Used to synthesise the post-launch user turn.
#[derive(Debug, Clone)]
struct LaunchedPane {
    pane_id: String,
    description: String,
    tag: String,
    worktree: Option<String>,
    resources: Vec<String>,
}

/// Per-launch in-flight state.  Built when `/claude` ships
/// `Request::ClaudeLaunch`, drained when the daemon emits
/// `Response::Done` (success path) or `Response::Error` (failure).
#[derive(Debug, Clone)]
struct PendingClaudeLaunch {
    /// tag → original task description, used to look up the
    /// description when daemon replies with `PaneAssigned { tag }`.
    descriptions: std::collections::HashMap<String, String>,
    /// Accumulator: one entry per `PaneAssigned` frame received,
    /// flushed into the synthesised `[launched]` block on Done.
    launched: Vec<LaunchedPane>,
}

impl AppState {
    fn new(session_id: String, model: String, socket_path: PathBuf, cwd_str: String) -> Self {
        Self {
            session_id,
            model,
            transcript: Vec::new(),
            input: String::new(),
            input_cursor: 0,
            streaming: false,
            history: Vec::new(),
            history_pos: None,
            history_draft: String::new(),
            steer_pending: false,
            steer_source: SteerSource::Idle,
            steer_buffer: VecDeque::new(),
            steer_buffer_truncated: false,
            last_ctrl_c: None,
            transcript_scroll_back: 0,
            // `false` for render_enabled — the TUI doesn't use the
            // tracker's stderr-rendering half (we read
            // `latest_progress` from `draw` and stitch the result
            // into the input box title).  The flag gates the async
            // render/finish methods we never call, so its value is
            // moot today; passing false documents intent.
            plan_tracker: crate::client::PlanProgressTracker::new(false),
            pending_claude: None,
            socket_path,
            cwd_str,
        }
    }

    /// Replace the input buffer with `text`, reset the cursor to the
    /// end, and (typically) clear history-mode.  Used by both the
    /// up/down history walkers and by the dispatch path's "restore
    /// the user's draft" branch.
    fn set_input(&mut self, text: String) {
        self.input = text;
        self.input_cursor = self.input.len();
    }

    /// Walk back one step in history (↑).  First press from a
    /// non-history-mode input also captures `self.input` as the
    /// draft to restore on ↓.  No-op when there's no history or
    /// we're already at the oldest entry.
    fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        // history_pos counts from the END of self.history, so 0 = most
        // recent, len-1 = oldest.  Stepping ↑ increases the index.
        let new_pos = match self.history_pos {
            None => {
                // Entering history mode — snapshot the draft so we can
                // come back to it via ↓.
                self.history_draft = self.input.clone();
                0
            }
            Some(p) if p + 1 < self.history.len() => p + 1,
            Some(p) => p, // already at oldest
        };
        self.history_pos = Some(new_pos);
        let i = self.history.len() - 1 - new_pos;
        let text = self.history[i].clone();
        self.set_input(text);
    }

    /// Walk forward one step in history (↓).  Stepping past the most
    /// recent entry leaves history mode and restores the draft we
    /// captured on the first ↑ press.
    fn history_next(&mut self) {
        let Some(pos) = self.history_pos else { return };
        if pos == 0 {
            // Stepping past most recent — restore the draft.
            let draft = std::mem::take(&mut self.history_draft);
            self.history_pos = None;
            self.set_input(draft);
            return;
        }
        let new_pos = pos - 1;
        self.history_pos = Some(new_pos);
        let i = self.history.len() - 1 - new_pos;
        let text = self.history[i].clone();
        self.set_input(text);
    }

    /// Append `display` to in-memory history (so ↑ in this session
    /// can recall it without re-reading the file) and reset history
    /// scroll state.  Called from `send_prompt` after a successful
    /// dispatch.  Dedupes the most-recent entry the same way
    /// `load_cwd_history` does.
    fn record_submitted_prompt(&mut self, display: &str) {
        if self.history.last().map(String::as_str) != Some(display) {
            self.history.push(display.to_string());
        }
        self.history_pos = None;
        self.history_draft.clear();
    }

    /// Push `text` as one or more transcript entries, all of `kind`.
    /// Splits on `\n` so a multi-line block (the synthesised
    /// `[launched]` user turn, a multi-line `format_task_released`
    /// release block, etc.) renders as separate visual rows instead
    /// of one mashed paragraph that the wrap logic re-flows
    /// arbitrarily.  Single-line text takes the fast path.
    fn push_kind_line(&mut self, kind: LineKind, text: String) {
        // Strip ANSI/VT escapes and other control characters before
        // the text reaches the transcript.  ratatui passes our cell
        // contents through to crossterm verbatim, so an unsanitised
        // ESC `]52;c;…` from model output, tool detail, or pasted
        // input would manipulate the host terminal (clipboard,
        // window title, cursor state, etc.).  Same `sanitize` helper
        // the classic CLI uses for stderr/stdout.
        let text = crate::sanitize(&text);
        if text.contains('\n') {
            for line in text.split('\n') {
                self.transcript.push(TranscriptLine {
                    kind,
                    text: line.to_string(),
                });
            }
        } else {
            self.transcript.push(TranscriptLine { kind, text });
        }
    }

    fn push_system_line(&mut self, text: String) {
        self.push_kind_line(LineKind::System, text);
    }
    fn push_user_line(&mut self, text: String) {
        self.push_kind_line(LineKind::User, text);
    }
    fn push_error_line(&mut self, text: String) {
        self.push_kind_line(LineKind::Error, text);
    }
    fn push_tool_line(&mut self, text: String) {
        self.push_kind_line(LineKind::Tool, text);
    }
    fn push_compacting_line(&mut self, text: String) {
        self.push_kind_line(LineKind::Compacting, text);
    }
    fn push_steer_line(&mut self, text: String) {
        self.push_kind_line(LineKind::Steer, text);
    }
    fn push_launch_line(&mut self, text: String) {
        self.push_kind_line(LineKind::Launch, text);
    }

    /// Append an assistant text chunk, continuing the previous
    /// assistant line if it was left "open" (no trailing newline).
    fn push_assistant_chunk(&mut self, chunk: &str) {
        // Strip ANSI/VT escapes from each chunk before it lands in
        // the transcript — see `push_kind_line` for the rationale.
        // Sanitising per-chunk can in theory split an escape across
        // chunks; in practice model SSE chunks are line-oriented so
        // this is fine, and the alternative (buffering until newline)
        // would defeat streaming.
        let sanitized = crate::sanitize(chunk);
        let mut remaining = sanitized.as_str();
        loop {
            match remaining.find('\n') {
                Some(idx) => {
                    let (first, rest) = remaining.split_at(idx);
                    self.append_or_open(first);
                    // The current assistant line now ends; close it so
                    // the next piece starts fresh.
                    if let Some(last) = self.transcript.last_mut() {
                        if let LineKind::Assistant { ref mut is_open } = last.kind {
                            *is_open = false;
                        }
                    }
                    // Skip the newline itself.
                    remaining = &rest[1..];
                }
                None => {
                    if !remaining.is_empty() {
                        self.append_or_open(remaining);
                    }
                    break;
                }
            }
        }
    }

    /// Append `piece` (which contains no `\n`) to the currently open
    /// assistant line, or start a new open assistant line.
    fn append_or_open(&mut self, piece: &str) {
        let can_extend = matches!(
            self.transcript.last(),
            Some(TranscriptLine {
                kind: LineKind::Assistant { is_open: true },
                ..
            })
        );
        if can_extend {
            if let Some(last) = self.transcript.last_mut() {
                last.text.push_str(piece);
            }
        } else {
            self.transcript.push(TranscriptLine {
                kind: LineKind::Assistant { is_open: true },
                text: piece.to_string(),
            });
        }
    }

    /// Called on Response::Done / Response::Error / stream close so the
    /// last in-flight assistant line is sealed.  Subsequent assistant
    /// chunks (e.g. on a follow-up turn) will start a fresh line.
    fn close_open_assistant_line(&mut self) {
        if let Some(last) = self.transcript.last_mut() {
            if let LineKind::Assistant { ref mut is_open } = last.kind {
                *is_open = false;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Event handling
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum KeyOutcome {
    Continue,
    SubmitInput(String),
    /// User asked to exit the TUI (Ctrl-D, or double-Ctrl-C).
    Exit,
    /// First Ctrl-C while a turn is streaming.  The caller must:
    /// 1. Send `Request::Interrupt` so the daemon stops mid-generation.
    /// 2. Flip `state.steer_pending = true` so subsequent Response
    ///    frames buffer rather than scribble over the user's input.
    /// 3. Push a hint to the transcript explaining the steer protocol.
    InterruptForSteer,
    /// Submit a steer correction (`Request::Steer { message: text }`)
    /// for the in-flight turn.  Carries the text the user typed.
    SubmitSteer(String),
    /// Empty Enter while `steer_pending`: cancel the steer and let
    /// the buffered output flush back to the transcript.
    CancelSteer,
}

fn handle_key(key: KeyEvent, state: &mut AppState) -> KeyOutcome {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let now = std::time::Instant::now();
    match key.code {
        // Ctrl-C semantics — depends on what mode we're in:
        //
        //   • streaming, no steer pending  → InterruptForSteer
        //   • streaming, steer pending     → second Ctrl-C exits
        //   • idle, non-empty input        → clear the input
        //   • idle, empty input            → first arms double-press
        //                                    (within `DOUBLE_CTRLC_WINDOW`),
        //                                    second exits
        //
        // This matches the classic chat (run_chat_loop) contract so
        // muscle memory transfers between the two UIs.  Ctrl-D always
        // exits — no double-press required, no steer interaction.
        KeyCode::Char('d') if ctrl => KeyOutcome::Exit,
        KeyCode::Char('c') if ctrl => {
            if state.streaming {
                if state.steer_pending {
                    // Two cases here:
                    //
                    // - `UserCtrlC` armed the steer.  The user's
                    //   first Ctrl-C (which armed it) already
                    //   committed to wanting out, so a second
                    //   press is a real exit gesture — fall
                    //   straight through.
                    //
                    // - `DaemonWaitingForInput` armed it.  The
                    //   user never pressed Ctrl-C before this
                    //   one; bailing on a single press would be
                    //   surprising (and would silently drop the
                    //   model's open question).  Require the
                    //   same double-press window the idle path
                    //   uses, with the same hint, so the gesture
                    //   is symmetrical between idle and
                    //   daemon-armed-steer states.
                    if state.steer_source == SteerSource::DaemonWaitingForInput {
                        if let Some(prev) = state.last_ctrl_c {
                            if now.duration_since(prev) <= DOUBLE_CTRLC_WINDOW {
                                return KeyOutcome::Exit;
                            }
                        }
                        state.last_ctrl_c = Some(now);
                        if state.transcript.last().map(|tl| tl.text.as_str())
                            != Some(CTRLC_EXIT_HINT)
                        {
                            state.push_system_line(CTRLC_EXIT_HINT.to_string());
                        }
                        return KeyOutcome::Continue;
                    }
                    return KeyOutcome::Exit;
                }
                state.last_ctrl_c = Some(now);
                return KeyOutcome::InterruptForSteer;
            }
            // Idle.  Non-empty input: treat Ctrl-C as a "wipe the
            // current line" rather than an exit gesture, matching
            // shell readline.
            if !state.input.is_empty() {
                state.input.clear();
                state.input_cursor = 0;
                state.last_ctrl_c = None;
                return KeyOutcome::Continue;
            }
            // Empty input + idle: first press arms the double-press
            // window; second press inside the window exits.
            if let Some(prev) = state.last_ctrl_c {
                if now.duration_since(prev) <= DOUBLE_CTRLC_WINDOW {
                    return KeyOutcome::Exit;
                }
            }
            state.last_ctrl_c = Some(now);
            // Show a single-line hint so the user knows another
            // Ctrl-C will exit.  Push at most one of these — checking
            // the previous transcript line keeps Ctrl-C spam clean.
            if state.transcript.last().map(|tl| tl.text.as_str()) != Some(CTRLC_EXIT_HINT) {
                state.push_system_line(CTRLC_EXIT_HINT.to_string());
            }
            KeyOutcome::Continue
        }
        // Emacs-style line editing.  These are the bare minimum any
        // serious user expects in a text input box.  They mirror what
        // reedline (a likely later replacement) would offer; doing them
        // by hand for now keeps the dependency surface small while we
        // shape the rest of the TUI.
        KeyCode::Char('a') if ctrl => {
            state.input_cursor = 0;
            KeyOutcome::Continue
        }
        KeyCode::Char('e') if ctrl => {
            state.input_cursor = state.input.len();
            KeyOutcome::Continue
        }
        KeyCode::Char('b') if ctrl => {
            state.input_cursor = prev_char_boundary(&state.input, state.input_cursor);
            KeyOutcome::Continue
        }
        KeyCode::Char('f') if ctrl => {
            state.input_cursor = next_char_boundary(&state.input, state.input_cursor);
            KeyOutcome::Continue
        }
        // Ctrl-K kills from cursor to end of line — handy for clearing
        // a partially-typed input without holding Backspace.
        KeyCode::Char('k') if ctrl => {
            state.input.truncate(state.input_cursor);
            KeyOutcome::Continue
        }
        // Ctrl-U kills from start to cursor (kill backward to BOL).
        KeyCode::Char('u') if ctrl => {
            state.input.replace_range(..state.input_cursor, "");
            state.input_cursor = 0;
            KeyOutcome::Continue
        }
        KeyCode::Enter => {
            // Mid-stream Enter (no steer pending) is a no-op: the
            // event loop wouldn't dispatch SubmitInput anyway, and
            // taking the buffer would silently drop whatever the
            // user was typing for the next turn.  Leave it intact.
            if state.streaming && !state.steer_pending {
                return KeyOutcome::Continue;
            }
            let text = std::mem::take(&mut state.input);
            state.input_cursor = 0;
            // Reset the double-Ctrl-C window any time Enter is pressed —
            // the user is doing something other than confirming an exit.
            state.last_ctrl_c = None;
            // While a steer is pending, Enter has different semantics:
            //   - non-empty → submit as Request::Steer correction
            //   - empty     → cancel the steer (drain buffer, resume)
            if state.steer_pending {
                if text.trim().is_empty() {
                    return KeyOutcome::CancelSteer;
                }
                return KeyOutcome::SubmitSteer(text);
            }
            KeyOutcome::SubmitInput(text)
        }
        KeyCode::Backspace => {
            // Step back one Unicode scalar regardless of byte width
            // (CJK chars are 3 bytes each in UTF-8 but should still
            // come out as one Backspace).
            if state.input_cursor > 0 {
                let prev = prev_char_boundary(&state.input, state.input_cursor);
                state.input.replace_range(prev..state.input_cursor, "");
                state.input_cursor = prev;
            }
            KeyOutcome::Continue
        }
        KeyCode::Delete => {
            // Forward delete: remove the scalar starting at the cursor
            // (if any).  Cursor stays put; characters to the right
            // shift left.
            if state.input_cursor < state.input.len() {
                let next = next_char_boundary(&state.input, state.input_cursor);
                state.input.replace_range(state.input_cursor..next, "");
            }
            KeyOutcome::Continue
        }
        KeyCode::Left => {
            state.input_cursor = prev_char_boundary(&state.input, state.input_cursor);
            KeyOutcome::Continue
        }
        KeyCode::Right => {
            state.input_cursor = next_char_boundary(&state.input, state.input_cursor);
            KeyOutcome::Continue
        }
        KeyCode::Home => {
            state.input_cursor = 0;
            KeyOutcome::Continue
        }
        KeyCode::End => {
            state.input_cursor = state.input.len();
            KeyOutcome::Continue
        }
        KeyCode::Up => {
            // ↑ recalls older history entries (same direction as
            // shell readline).  The first press from a non-history
            // input also snapshots the current draft so ↓ past the
            // most-recent entry can restore it.
            state.history_prev();
            KeyOutcome::Continue
        }
        KeyCode::Down => {
            state.history_next();
            KeyOutcome::Continue
        }
        KeyCode::PageUp => {
            state.transcript_scroll_back =
                state.transcript_scroll_back.saturating_add(PAGE_STEP_ROWS);
            KeyOutcome::Continue
        }
        KeyCode::PageDown => {
            state.transcript_scroll_back =
                state.transcript_scroll_back.saturating_sub(PAGE_STEP_ROWS);
            KeyOutcome::Continue
        }
        KeyCode::Char(c) => {
            // Discard any other modifier combos we haven't handled —
            // including Alt/Shift-space — rather than interpret them.
            // This keeps the prototype predictable and leaves room for
            // reedline (a possible later swap-in) to take over.
            if ctrl {
                return KeyOutcome::Continue;
            }
            // Insert at cursor, not at the end.  Encoding in place
            // avoids allocating a temporary String per keystroke.
            let mut buf = [0u8; 4];
            let s = c.encode_utf8(&mut buf);
            state.input.insert_str(state.input_cursor, s);
            state.input_cursor += s.len();
            KeyOutcome::Continue
        }
        _ => KeyOutcome::Continue,
    }
}

/// UTF-8-safe clamp to the largest char boundary ≤ `idx` in `s`.
/// Used so Backspace removes a whole character, not a fractional byte.
fn floor_char_boundary(s: &str, idx: usize) -> usize {
    let mut i = idx.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Return the byte index of the Unicode scalar immediately before
/// `idx`, or 0 if `idx == 0`.  Saturates at 0; never panics.
fn prev_char_boundary(s: &str, idx: usize) -> usize {
    if idx == 0 {
        return 0;
    }
    floor_char_boundary(s, idx - 1)
}

/// Return the byte index of the Unicode scalar immediately after
/// `idx`, or `s.len()` if `idx == s.len()`.  Saturates at `s.len()`;
/// never panics.
fn next_char_boundary(s: &str, idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    let mut i = idx + 1;
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Dispatch a single daemon frame.  Returns a `ResponseOutcome`
/// enumerating any follow-up the caller needs to perform — namely
/// "send this synthesised supervision prompt" once a `/claude` launch
/// completes, since send_prompt is async and can't be invoked from
/// inside this synchronous handler.
fn handle_response(resp: Response, state: &mut AppState) -> ResponseOutcome {
    // While the user is typing a steer correction, buffer streaming-
    // content frames (Text / ToolUse / Compacting — see
    // `is_buffered_frame`) so they don't fight the steer prompt for
    // screen real estate.  Control / state-changing frames (Done /
    // Error / SteerAck / ModelSwitched / PaneAssigned /
    // CapacityError / WaitingForInput) bypass the buffer because
    // they either change the steer-pending state itself or carry
    // information the user needs to see immediately.
    if state.steer_pending && is_buffered_frame(&resp) {
        if state.steer_buffer.len() >= STEER_BUFFER_MAX_FRAMES {
            // Evict oldest first so the freshest output survives.
            // Mark `steer_buffer_truncated` so the eventual flush can
            // tell the user some frames were dropped.
            state.steer_buffer.pop_front();
            state.steer_buffer_truncated = true;
        }
        state.steer_buffer.push_back(resp);
        return ResponseOutcome::Continuing;
    }
    match resp {
        Response::Text { chunk } => {
            // Feed the plan tracker the raw chunk BEFORE we push it
            // through the markdown buffer so the tracker sees the
            // exact `- [ ]` / `- [x]` markers the LLM emitted.
            state.plan_tracker.push(&chunk);
            state.push_assistant_chunk(&chunk);
            ResponseOutcome::Continuing
        }
        Response::Done => {
            // If the user had pressed Ctrl-C and the daemon was
            // still mid-stream when the turn finished cleanly (LLM
            // hit end-of-turn before our Steer landed), Done is the
            // signal that no SteerAck is coming — the steer
            // injection happened past the daemon's turn boundary.
            // Clear steer mode and flush whatever buffered up while
            // the user was typing the correction so the user sees
            // the final output instead of being stuck in a steer
            // prompt forever.  Order matters: clear pending FIRST
            // so flush_steer_buffer doesn't re-buffer.
            if state.steer_pending {
                state.steer_pending = false;
                state.steer_source = SteerSource::Idle;
                state.last_ctrl_c = None;
                flush_steer_buffer(state);
            }
            // Score any final checklist item the model emitted
            // without a trailing newline, then reset the tracker so
            // progress from this turn doesn't leak into the next
            // (e.g. a follow-up turn with no checklist would otherwise
            // keep showing the previous turn's [plan N/M done]).
            state.plan_tracker.finalize_tail();
            let final_progress = state.plan_tracker.latest_progress();
            if let Some((done, total)) = final_progress {
                if total > 0 {
                    state.push_system_line(format!("[plan {done}/{total} done]"));
                }
            }
            state.plan_tracker = crate::client::PlanProgressTracker::new(false);
            state.close_open_assistant_line();
            state.streaming = false;
            // If a /claude launch was in flight, this Done means the
            // daemon has finished assigning panes.  Synthesise the
            // [launched] block and ask the caller to send it as a
            // user turn so the LLM enters supervision mode.  Empty
            // launched list means the launch errored before assigning
            // anything; nothing to synthesise.
            if let Some(pending) = state.pending_claude.take() {
                if !pending.launched.is_empty() {
                    let synth = render_launched_block(&pending.launched);
                    state.push_system_line(String::new());
                    return ResponseOutcome::TurnEndedSendSynth(synth);
                }
            }
            ResponseOutcome::TurnEnded
        }
        Response::Error { message } => {
            // Same as Done: an error before SteerAck means the
            // steer never landed.  Clear steer mode + flush so the
            // user isn't stranded in steer-pending UI on a turn that
            // already failed.
            if state.steer_pending {
                state.steer_pending = false;
                state.steer_source = SteerSource::Idle;
                state.last_ctrl_c = None;
                flush_steer_buffer(state);
            }
            state.close_open_assistant_line();
            state.push_error_line(format!("error: {message}"));
            state.streaming = false;
            // Drop any in-flight /claude launch state on error so
            // a subsequent input doesn't leak it into a synthesised
            // supervision prompt.
            state.pending_claude = None;
            // Reset the plan tracker too — a half-emitted checklist
            // shouldn't survive into the next turn.
            state.plan_tracker = crate::client::PlanProgressTracker::new(false);
            ResponseOutcome::TurnEnded
        }
        Response::ToolUse { name, detail } => {
            // Render the tool name + detail as a single transcript
            // line; the magenta colour + 🔧 glyph in
            // `push_wrapped_transcript_line` make it stand out from
            // model text and amaebi metadata.  We mirror classic
            // chat's tool→glyph mapping where it adds extra signal
            // (read/edit/tmux), and fall back to the generic 🔧
            // for everything else by leaving the kind-level glyph
            // alone.
            state.push_tool_line(format!("{} {detail}", tool_label(&name)));
            ResponseOutcome::Continuing
        }
        Response::Compacting => {
            state.push_compacting_line("compacting conversation history…".to_string());
            ResponseOutcome::Continuing
        }
        Response::SteerAck => {
            // Daemon accepted our steer correction.  Drain the
            // buffered frames (which arrived while the user was
            // typing) back into the transcript, then clear the
            // steer_pending flag.  Order matters: `flush_steer_buffer`
            // re-enters `handle_response`, so we must clear the flag
            // FIRST or it'll re-buffer everything we just popped.
            state.steer_pending = false;
            state.steer_source = SteerSource::Idle;
            state.last_ctrl_c = None;
            state.push_steer_line("steer acknowledged".to_string());
            flush_steer_buffer(state);
            ResponseOutcome::Continuing
        }
        Response::ModelSwitched { model } => {
            // Daemon-side model switch (e.g. the LLM called the
            // `switch_model` tool).  Mirror it locally so the next
            // outgoing Request::Chat carries the new value.  Same
            // behaviour as run_chat_loop's ModelSwitched handler.
            state.push_system_line(format!("[model switched: {} → {}]", state.model, model));
            state.model = model;
            ResponseOutcome::Continuing
        }
        Response::WaitingForInput { prompt } => {
            // The daemon is asking the user for a clarifying reply
            // mid-turn (e.g. the model wrote "which option do you
            // prefer, A or B?").  The protocol expects the user's
            // reply to come back as a Request::Steer so the daemon
            // injects it as the next user message in the SAME
            // agentic loop iteration — NOT as Request::Chat (which
            // would start a fresh turn and break the LLM's
            // context).
            //
            // We reuse the existing steer machinery: arming
            // steer_pending = true makes Enter ship a
            // Request::Steer, and the daemon's eventual SteerAck
            // (or Done if the LLM aborts) clears the flag.
            // streaming stays true so the input title shows the
            // steer prompt instead of "input (Enter to send…)".
            //
            // If steer is ALREADY pending — e.g. the user pressed
            // Ctrl-C and is mid-correction when the model also
            // emits a question — preserve the existing buffer and
            // source.  Wiping them would lose any frames already
            // captured during the user-armed steer (matches the
            // classic chat path which buffers WaitingForInput in
            // that case instead of re-arming).
            if !prompt.is_empty() {
                state.push_steer_line(prompt);
            } else {
                state.push_steer_line(
                    "model is waiting for your reply — type and Enter".to_string(),
                );
            }
            if !state.steer_pending {
                state.steer_pending = true;
                state.steer_source = SteerSource::DaemonWaitingForInput;
                state.steer_buffer.clear();
                state.steer_buffer_truncated = false;
            }
            ResponseOutcome::Continuing
        }
        Response::PaneAssigned {
            tag,
            pane_id,
            session_id: _sid,
            worktree,
            resources,
        } => {
            // Surface the assignment so the user can see what landed.
            let resources_blurb = if resources.is_empty() {
                String::new()
            } else {
                format!(" resources={}", resources.join(","))
            };
            state.push_launch_line(format!("pane {pane_id}: tag={tag}{resources_blurb}"));
            // Buffer for the supervision prompt synthesised on Done.
            // If the user typed /claude but no pending state was set
            // (shouldn't happen on this path), defensively skip the
            // accumulator instead of panicking.
            if let Some(pending) = state.pending_claude.as_mut() {
                let description = pending
                    .descriptions
                    .get(&tag)
                    .cloned()
                    .unwrap_or_else(|| tag.clone());
                pending.launched.push(LaunchedPane {
                    pane_id,
                    description,
                    tag,
                    worktree,
                    resources,
                });
            }
            ResponseOutcome::Continuing
        }
        Response::CapacityError {
            requested,
            max_panes,
            current_busy,
        } => {
            state.push_error_line(format!(
                "[error] capacity limit reached: max_panes={max_panes}, busy={current_busy}, \
                 requested={requested}; free existing panes to continue"
            ));
            // Clear pending state so we don't try to synth a
            // supervision prompt for a launch that never happened.
            state.pending_claude = None;
            ResponseOutcome::Continuing
        }
        Response::TaskReleased {
            pane_id,
            resources_freed,
            tag,
            summary,
            worktree_path,
            worktree_dirty,
            pane_tail,
            elapsed_ms,
        } => {
            // Reuse classic chat's `format_task_released` so the
            // released-pane block looks identical between the two
            // UIs.  Push it as a green Launch-kind block — release
            // is the inverse of /claude launch and benefits from
            // the same visual category.  format_task_released
            // returns a multi-line string, and `push_launch_line`
            // through the new newline-aware path one entry per
            // logical line.
            let formatted = crate::client::format_task_released(
                &pane_id,
                &resources_freed,
                tag.as_deref(),
                summary.as_deref(),
                worktree_path.as_deref(),
                worktree_dirty,
                &pane_tail,
                elapsed_ms,
            );
            // `push_launch_line` splits on '\n' so the multi-line
            // formatted block lands as one transcript entry per
            // logical line.
            state.push_launch_line(formatted);
            ResponseOutcome::Continuing
        }
        other => {
            // Anything we haven't classified surfaces as a debug-
            // dumped system line so it doesn't get silently
            // dropped — covers daemon protocol additions we haven't
            // wired display for yet.
            state.push_system_line(format!("[{other:?}]"));
            ResponseOutcome::Continuing
        }
    }
}

/// Pretty-print a tool name as a short, distinctive label that
/// distinguishes the most-common tools at a glance.  Mirrors classic
/// chat's emoji choices (run_chat_loop's ToolUse handler) for read /
/// edit / shell / tmux variants; everything else falls back to the
/// plain tool name (the kind-level 🔧 prefix carries the "tool"
/// signal).  Returning a String avoids forcing static lifetimes on
/// the tool-name dispatch.
fn tool_label(tool: &str) -> String {
    match tool {
        "shell_command" => "$".to_string(),
        "read_file" => "📄 read".to_string(),
        "edit_file" => "✏️  edit".to_string(),
        "tmux_send_text" => "⌨️  send-text".to_string(),
        "tmux_send_key" => "⌨️  send-key".to_string(),
        "tmux_capture_pane" => "🖥️  capture".to_string(),
        "tmux_wait" => "⏸️  wait".to_string(),
        other => other.to_string(),
    }
}

/// True for `Response` variants we want to buffer while the user is
/// composing a steer correction, false for control frames that need
/// to be processed immediately (state changes, errors, the
/// SteerAck that ends steer mode, the WaitingForInput that ARMS
/// steer mode).
fn is_buffered_frame(resp: &Response) -> bool {
    matches!(
        resp,
        Response::Text { .. } | Response::ToolUse { .. } | Response::Compacting
    )
}

/// What `handle_response` wants the caller (the main run_chat_tui
/// select loop) to do next.  The handler can't perform async work
/// itself, so anything that needs the daemon socket or `send_prompt`
/// is requested here and executed by the caller.
#[derive(Debug, PartialEq, Eq)]
enum ResponseOutcome {
    /// Frame consumed; turn still in progress.  Just redraw.
    Continuing,
    /// Turn finished cleanly — flip `streaming = false`, redraw, do
    /// nothing else.
    TurnEnded,
    /// Turn finished, and the caller should now ship the included
    /// synthesised user prompt as a fresh `Request::Chat` so the LLM
    /// takes over supervision after `/claude`.
    TurnEndedSendSynth(String),
}

/// Push the same startup banner the classic chat prints (logo +
/// version + model + sandbox + session + cwd) into the transcript as
/// system lines.  Classic chat prints to stderr just before entering
/// the read loop, but in --tui we're already on the alternate screen
/// by the time greeting fires, so an `eprintln!` would land on the
/// alt screen and be cleared by the next ratatui draw.  Rendering the
/// banner as transcript content keeps it visible above the input box
/// throughout the session (until it scrolls off the top).
///
/// Logic mirrors `banner::print`; we don't call it directly because
/// the printing is interleaved with `eprint!` / `eprintln!` and
/// extracting just the strings would require restructuring that
/// module.  Keeping a small bespoke version here is the lower-risk
/// path while feat/tui is still being shaped.
fn push_banner(state: &mut AppState, cwd: &std::path::Path) {
    const LOGO: &str = "  ╔═╗╔╦╗╔═╗╔═╗╔╗ ╦
  ╠═╣║║║╠═╣║╣ ╠╩╗║
  ╩ ╩╩ ╩╩ ╩╚═╝╚═╝╩";
    for line in LOGO.lines() {
        state.push_system_line(line.to_string());
    }

    let version = env!("CARGO_PKG_VERSION");
    let commit = env!("AMAEBI_GIT_COMMIT");
    let sandbox = match std::env::var("AMAEBI_SANDBOX").as_deref() {
        Ok("docker") => {
            let image = std::env::var("AMAEBI_SANDBOX_IMAGE")
                .unwrap_or_else(|_| "amaebi-sandbox:bookworm-slim".to_string());
            format!("docker ({image})")
        }
        _ => "off".to_string(),
    };
    // Resolve user aliases the same way the classic banner does so
    // /model output and the banner agree on what the daemon will see.
    let user_aliases = crate::config::Config::load().model_aliases;
    let spec = crate::provider::resolve_with_aliases(&state.model, &user_aliases);
    let model_display =
        if state.model.starts_with("copilot/") || state.model.starts_with("bedrock/") {
            state.model.clone()
        } else if let Some(target) = user_aliases
            .get(state.model.trim_end_matches("[1m]"))
            .filter(|_| {
                !crate::provider::is_builtin_bedrock_alias(state.model.trim_end_matches("[1m]"))
            })
        {
            let needs_1m = state.model.ends_with("[1m]") && !target.ends_with("[1m]");
            if needs_1m {
                format!("{} → {}[1m]", state.model, target)
            } else {
                format!("{} → {}", state.model, target)
            }
        } else {
            format!("{}/{}", spec.provider, state.model)
        };

    state.push_system_line(format!("  version  {version} ({commit})"));
    state.push_system_line(format!("  model    {model_display}"));
    state.push_system_line(format!("  sandbox  {sandbox}"));
    state.push_system_line(format!("  session  {}", state.session_id));
    state.push_system_line(format!("  cwd      {}", cwd.display()));
    state.push_system_line(String::new());

    // Re-emit the unread cron-report bell — the eprintln in main.rs
    // gets cleared by the alt-screen flip in TUI mode (suppressed at
    // the call site for that reason).  Rendering it as a transcript
    // line keeps the notification visible above the input box until
    // it scrolls off naturally.
    if let Some(n) = crate::unread_cron_count() {
        let noun = if n == 1 { "report" } else { "reports" };
        state.push_steer_line(format!(
            "🔔 You have {n} unread cron {noun}. Run `amaebi inbox list` to read."
        ));
        state.push_system_line(String::new());
    }
}

async fn send_prompt(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    state: &mut AppState,
    prompt: String,
) -> Result<()> {
    state.push_user_line(format!("> {prompt}"));
    state.streaming = true;

    // Record in-memory + persist to ~/.amaebi/history.jsonl so this
    // prompt becomes ↑-recallable both within this session and on the
    // next chat invocation in the same cwd.  Disk write is best-
    // effort: a missing / locked / full history file should not break
    // the chat itself.
    state.record_submitted_prompt(&prompt);
    if let Err(e) = crate::client::record_history_line(&prompt) {
        tracing::warn!(error = %e, "failed to persist prompt to history.jsonl");
    }

    let req = Request::Chat {
        prompt,
        tmux_pane: std::env::var("TMUX_PANE").ok(),
        session_id: Some(state.session_id.clone()),
        model: state.model.clone(),
    };
    let mut frame = serde_json::to_string(&req).context("serializing Request::Chat")?;
    frame.push('\n');
    writer
        .write_all(frame.as_bytes())
        .await
        .context("sending Chat request to daemon")?;
    writer.flush().await.ok();
    Ok(())
}

/// Mid-turn Ctrl-C handler: ship `Request::Interrupt` so the daemon
/// stops the agentic loop ASAP, then arm `state.steer_pending` so
/// subsequent `Response::Text` / `ToolUse` / etc. frames buffer
/// instead of fighting the user's correction for screen real estate.
/// Push a one-line breadcrumb so the user knows what mode they're in
/// and how to get out (matches classic chat's prompt).
async fn send_interrupt_and_arm_steer(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    state: &mut AppState,
) -> Result<()> {
    state.steer_pending = true;
    state.steer_source = SteerSource::UserCtrlC;
    state.steer_buffer.clear();
    state.steer_buffer_truncated = false;
    state.push_steer_line(
        "type a correction and press Enter, empty Enter to cancel, \
         Ctrl-C again to exit"
            .to_string(),
    );

    let req = Request::Interrupt {
        session_id: state.session_id.clone(),
    };
    let mut frame = serde_json::to_string(&req).context("serializing Request::Interrupt")?;
    frame.push('\n');
    // The daemon may have already finished the turn before our
    // Interrupt arrives; either way SteerAck-or-Done lands.  Swallow
    // a closed-pipe error so the response loop can drain normally.
    let _ = writer.write_all(frame.as_bytes()).await;
    let _ = writer.flush().await;
    Ok(())
}

/// Send a steer correction for the in-flight turn.  The daemon will
/// drain it as a fresh user message between model turns and reply
/// with `Response::SteerAck`, which is where the buffered output
/// gets flushed back into the transcript (in `handle_response`).
async fn send_steer(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    state: &mut AppState,
    text: String,
) -> Result<()> {
    state.push_user_line(format!("> [steer] {text}"));
    let req = Request::Steer {
        session_id: state.session_id.clone(),
        message: text,
    };
    let mut frame = serde_json::to_string(&req).context("serializing Request::Steer")?;
    frame.push('\n');
    let _ = writer.write_all(frame.as_bytes()).await;
    let _ = writer.flush().await;
    // Stay in steer-pending until we see SteerAck so any in-flight
    // chunks the daemon already shipped before our Steer arrived
    // continue to buffer instead of clobbering the steer prompt.
    Ok(())
}

/// Empty Enter while steer-pending.  Two cases, gated on
/// `state.steer_source`:
///
/// - `UserCtrlC`: the user pressed Ctrl-C and decided not to
///   correct anything.  The first Ctrl-C already shipped
///   `Request::Interrupt`, so we just locally flush the buffer +
///   clear the flags.
///
/// - `DaemonWaitingForInput`: the daemon is parked on
///   `steer_rx.recv()` waiting for a real reply.  Per
///   `daemon.rs:5225` it explicitly ignores interrupt sentinels
///   while in this state — the only way out is a real Steer
///   message, a disconnect, or a 300s timeout.  An empty Enter
///   here is therefore a no-op; we leave steer_pending = true so
///   the user keeps seeing the steer prompt and can type a real
///   reply.  Pushing a faint "(empty Enter ignored …)" hint so
///   the user knows what happened.
///
/// `Idle`: should never reach `cancel_steer` (no steer to cancel),
/// but tolerate gracefully — silent no-op.
fn cancel_steer_local(state: &mut AppState) -> bool {
    match state.steer_source {
        SteerSource::DaemonWaitingForInput => {
            state.push_steer_line(
                "(empty Enter ignored — type a reply for the model's question, or Ctrl-C twice to exit)"
                    .to_string(),
            );
            // No state change; next Enter will be re-evaluated.
            false
        }
        SteerSource::UserCtrlC => {
            // Order matters: clear `steer_pending` BEFORE flushing,
            // otherwise `flush_steer_buffer` re-enters
            // `handle_response` which would re-buffer the very frames
            // we just popped.
            state.steer_pending = false;
            state.steer_source = SteerSource::Idle;
            state.last_ctrl_c = None;
            state.push_steer_line("steer cancelled".to_string());
            flush_steer_buffer(state);
            false
        }
        SteerSource::Idle => false,
    }
}

/// Empty Enter while `steer_pending`: roll back the steer mode and
/// flush the buffered output.  See `cancel_steer_local` for the
/// per-source logic.  Currently no source returns a "needs IPC" hint
/// (true), so this wrapper is just an async-context shim — kept for
/// symmetry with the other action helpers in case a future steer
/// source needs to ship a frame.
async fn cancel_steer(
    _writer: &mut tokio::net::unix::OwnedWriteHalf,
    state: &mut AppState,
) -> Result<()> {
    let _ = cancel_steer_local(state);
    Ok(())
}

/// Replay every frame buffered while `steer_pending` was set, in
/// arrival order, by feeding them back through `handle_response` —
/// the same code path normal frames take, so styling/state stays
/// consistent.  If we evicted any frames at the buffer cap, a
/// truncation notice is prepended so the user knows the buffer
/// scrolled past some output.  Note: `handle_response` writes to
/// `state.steer_buffer` only when `steer_pending` is true, so by
/// the time a flush runs we must already have cleared that flag —
/// callers (`cancel_steer`, the SteerAck arm of `handle_response`)
/// arrange that ordering.
fn flush_steer_buffer(state: &mut AppState) {
    if state.steer_buffer_truncated {
        state.push_steer_line(format!(
            "buffer truncated — dropped older frames past {STEER_BUFFER_MAX_FRAMES}"
        ));
        state.steer_buffer_truncated = false;
    }
    let buffered: VecDeque<Response> = std::mem::take(&mut state.steer_buffer);
    for frame in buffered {
        // Recurse into the normal handler.  Steer-pending is false
        // at this point so frames go to the transcript proper.  We
        // ignore the ResponseOutcome — none of the buffered frames
        // would request an async follow-up that the cancel/SteerAck
        // path could service inline anyway (SendSynth only fires on
        // Done with a /claude in flight, which would have been
        // surfaced before the steer was armed).
        let _ = handle_response(frame, state);
    }
}

/// What `dispatch_input` decided to do with a freshly-submitted line.
///
/// Pulling the slash-command decision out of the async dispatcher
/// keeps the parser part unit-testable without a real Unix socket.
/// `Claude` and `ReplyReview` carry the parser output verbatim;
/// dispatching them runs async work (tag generation, ClaudeLaunch
/// IPC) which we do directly in `dispatch_input`.
// PartialEq only (not Eq) because ReleaseCmd carries a String
// summary and other variants have nested Vecs of structs that
// don't implement Eq either.  Tests use match-and-assert rather
// than `assert_eq!` for variant comparison anyway.
#[derive(Debug, PartialEq)]
enum InputDispatch {
    /// `/model` (no arg): show current model in transcript.
    ShowModel,
    /// `/model <name>`: update `state.model` to this name.
    SwitchModel(String),
    /// `/claude "task" ...` — parser succeeded with the given tasks.
    Claude(Vec<crate::client::ClaudeTask>),
    /// `/replyreview <PR> ...` — parser succeeded with these PR
    /// numbers.  Worktree + description resolution happens
    /// asynchronously via `crate::client::resolve_replyreview_tasks`.
    ReplyReview(Vec<u32>),
    /// `/release %pane` or `/release all` — parser succeeded.
    /// Carries the parsed ReleaseCmd verbatim; dispatch sends a
    /// `Request::ClaudeRelease` and the existing `TaskReleased`
    /// handler in `handle_response` renders each released-pane
    /// block.
    Release(crate::client::ReleaseCmd),
    /// Reserved for future slash commands that aren't yet wired.
    /// Currently unused — every recognised command has a real
    /// dispatch path — but kept so a future addition (say
    /// `/inbox` listing) can park behind a clear UX message
    /// rather than fall through to chat.
    #[allow(dead_code)]
    NotYetWired(&'static str),
    /// Slash command failed to parse; surface the parser error to
    /// the transcript.
    SlashError(String),
    /// Plain text: send as `Request::Chat` to the daemon.
    SendChat,
}

fn classify_input(text: &str) -> InputDispatch {
    match parse_slash_command(text) {
        Some(SlashCommand::Model(None)) => InputDispatch::ShowModel,
        Some(SlashCommand::Model(Some(name))) => InputDispatch::SwitchModel(name),
        Some(SlashCommand::Claude(Ok(tasks))) => InputDispatch::Claude(tasks),
        Some(SlashCommand::Claude(Err(msg))) => InputDispatch::SlashError(msg),
        Some(SlashCommand::ReplyReview(Ok(prs))) => InputDispatch::ReplyReview(prs),
        Some(SlashCommand::ReplyReview(Err(msg))) => InputDispatch::SlashError(msg),
        Some(SlashCommand::Release(Ok(cmd))) => InputDispatch::Release(cmd),
        Some(SlashCommand::Release(Err(msg))) => InputDispatch::SlashError(msg),
        None => InputDispatch::SendChat,
    }
}

/// Dispatch a single Enter-pressed line.  Side-effect wrapper around
/// `classify_input`: applies the resulting `InputDispatch` to `state`
/// and the daemon socket.
async fn dispatch_input(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    state: &mut AppState,
    text: String,
) -> Result<()> {
    match classify_input(&text) {
        InputDispatch::ShowModel => {
            state.push_system_line(format!("[model] current: {}", state.model));
        }
        InputDispatch::SwitchModel(name) => {
            // Local-only switch: the daemon picks up the new model on
            // the next `Request::Chat` because we ship `state.model`
            // in every request.  Same shape as classic chat (see
            // run_chat_loop's /model handling).
            state.push_system_line(format!("[model] {} → {}", state.model, name));
            state.model = name;
        }
        InputDispatch::Claude(tasks) => {
            launch_claude_tasks(writer, state, tasks).await?;
        }
        InputDispatch::ReplyReview(prs) => {
            // /replyreview is normalised into the same Vec<ClaudeTask>
            // shape as /claude after head-branch + worktree
            // resolution (matches classic chat's flow).  This is an
            // async + network-bound step (gh + git), so we surface a
            // breadcrumb so the user knows the TUI hasn't frozen.
            state.push_system_line(format!(
                "[replyreview] resolving {} PR(s) — running gh + git…",
                prs.len()
            ));
            match crate::client::resolve_replyreview_tasks(&prs).await {
                Ok(tasks) => launch_claude_tasks(writer, state, tasks).await?,
                Err(msg) => state.push_error_line(format!("/replyreview: {msg}")),
            }
        }
        InputDispatch::Release(cmd) => {
            release_panes(writer, state, cmd).await?;
        }
        InputDispatch::NotYetWired(name) => {
            state.push_system_line(format!(
                "[error] {name} is not yet wired in --tui; \
                 fall back to classic chat (no --tui flag) for that command."
            ));
        }
        InputDispatch::SlashError(msg) => {
            state.push_error_line(msg);
        }
        InputDispatch::SendChat => {
            send_prompt(writer, state, text).await?;
        }
    }
    Ok(())
}

/// Send `Request::ClaudeLaunch` for the given tasks and arm
/// `state.pending_claude` so the response handler can collect
/// `PaneAssigned` frames and synthesise the supervision prompt on
/// `Response::Done`.  Any pre-launch error (tag generation failure,
/// IPC error) is surfaced to the transcript and `pending_claude`
/// stays None so the next user input isn't blocked.
async fn launch_claude_tasks(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    state: &mut AppState,
    mut tasks: Vec<crate::client::ClaudeTask>,
) -> Result<()> {
    if tasks.is_empty() {
        // Defensive — parser should have errored before getting here.
        return Ok(());
    }
    // Resolve any tasks that arrived without an explicit --tag by
    // shelling a Request::GenerateTag over a one-shot side connection.
    // Same flow as classic chat (see run_chat_loop).
    if let Err(e) =
        crate::client::resolve_missing_tags(&state.socket_path, &mut tasks, &state.cwd_str).await
    {
        state.push_error_line(format!("[error] tag generation failed: {e:#}"));
        return Ok(());
    }

    // Snapshot tag → original description so the response handler
    // can rebuild the [launched] block on Response::Done.
    let descriptions: std::collections::HashMap<String, String> = tasks
        .iter()
        .map(|t| (t.tag.clone(), t.description.clone()))
        .collect();

    // Resolve client_cwd the way classic chat does: prefer the task's
    // --cwd override, else the chat process's cwd; canonicalise so
    // the daemon's notebook lookup matches across symlink/`..` paths.
    let invocation_repo_dir: Option<String> = Some({
        let effective_cwd = tasks
            .iter()
            .find_map(|t| t.cwd.clone())
            .unwrap_or_else(|| state.cwd_str.clone());
        crate::session::canonical_key(std::path::Path::new(&effective_cwd))
    });

    let task_specs: Vec<crate::ipc::TaskSpec> = tasks
        .into_iter()
        .map(|t| crate::ipc::TaskSpec {
            tag: t.tag,
            description: t.description,
            worktree: t.worktree,
            client_cwd: t.cwd.or_else(|| Some(state.cwd_str.clone())),
            auto_enter: t.auto_enter,
            resume_pane: t.resume_pane,
            resources: t.resources,
            resource_timeout_secs: t.resource_timeout_secs,
        })
        .collect();

    let req = Request::ClaudeLaunch {
        tasks: task_specs,
        session_id: Some(state.session_id.clone()),
        repo_dir: invocation_repo_dir,
    };
    let mut frame = serde_json::to_string(&req).context("serializing ClaudeLaunch")?;
    frame.push('\n');
    if let Err(e) = writer.write_all(frame.as_bytes()).await {
        state.push_error_line(format!("[error] sending /claude to daemon: {e}"));
        return Ok(());
    }
    let _ = writer.flush().await;

    // Tell the user we shipped the launch, and hand the rest of the
    // flow to the main response loop via `pending_claude`.
    state.push_system_line(format!(
        "[claude] launching {} task(s); waiting for pane assignment…",
        descriptions.len()
    ));
    state.pending_claude = Some(PendingClaudeLaunch {
        descriptions,
        launched: Vec::new(),
    });
    state.streaming = true;
    Ok(())
}

/// Render the dim one-row status bar between transcript and input.
/// Layout: `model · cwd · session-prefix` truncated/elided to fit
/// `width` so it never wraps.  When the row is too narrow even for
/// the model name alone, just truncate to width with an ellipsis;
/// the user can scroll the banner via PgUp to recover full info.
fn render_status_bar(state: &AppState, width: u16) -> Paragraph<'static> {
    // Left half: model · cwd · session-prefix.  Static, identifying.
    let cwd_short = shorten_cwd(&state.cwd_str);
    let session_short = &state.session_id[..8.min(state.session_id.len())];
    let left = format!("{}  ·  {}  ·  {}", state.model, cwd_short, session_short);

    // Right half: dynamic state.  Steer mode and plan progress are
    // both load-bearing signals while a turn is in flight; without
    // surfacing them the user has to guess from input-box title
    // whether anything is happening.  Pick the most-specific
    // applicable signal so the right half stays short.
    let right = build_status_right(state);

    // Layout: left + (filler whitespace) + right, all on a single
    // row coloured DarkGray.  When width is too small for both,
    // elide the left half — the right half is short and dynamic, it
    // shouldn't truncate.
    let total_w = width as usize;
    let right_w = unicode_width::UnicodeWidthStr::width(right.as_str());
    // Reserve 2 columns of breathing room between left and right so
    // they don't visually collide when the bar is just wide enough.
    let reserved_for_right = if right.is_empty() { 0 } else { right_w + 2 };
    let left_budget = total_w.saturating_sub(reserved_for_right);
    let left_elided = elide_to_width(&left, left_budget);
    let left_w = unicode_width::UnicodeWidthStr::width(left_elided.as_str());
    // Pad between the two halves with spaces so the right edge of
    // the bar lines up exactly with the right border above.
    let gap = total_w.saturating_sub(left_w + right_w);
    let padding = " ".repeat(gap);

    let style = Style::default().fg(Color::DarkGray);
    Paragraph::new(Line::from(vec![
        Span::styled(left_elided, style),
        Span::styled(padding, style),
        Span::styled(right, style),
    ]))
}

/// Build the right-hand status-bar text.  Returns the empty string
/// when there's nothing dynamic to show (idle session, no scroll-
/// back, no steer).  Picks the most specific signal:
///
/// - `steer (Ctrl-C exits)` while steer is pending
/// - `streaming · [plan N/M done]` during a turn with a live plan
/// - `streaming` during a plain turn
///
/// Does NOT show `↑ N rows` here because that's already on the
/// transcript box title and would be redundant.
fn build_status_right(state: &AppState) -> String {
    if state.steer_pending {
        return "steer (Enter submits, Ctrl-C exits)".to_string();
    }
    if state.streaming {
        if let Some((d, t)) = state.plan_tracker.latest_progress() {
            if t > 0 {
                return format!("streaming · [plan {d}/{t} done]");
            }
        }
        return "streaming".to_string();
    }
    String::new()
}

/// Replace a leading `$HOME` with `~` so the status bar fits in the
/// common case (e.g. `~/.amaebi/worktrees/dev/tui-chat` rather than
/// the full `/home/yuankuns/...`).  Falls back to the raw string if
/// HOME isn't set or the cwd is outside it.
fn shorten_cwd(cwd: &str) -> String {
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() && cwd.starts_with(&home) {
            return format!("~{}", &cwd[home.len()..]);
        }
    }
    cwd.to_string()
}

/// Truncate `s` to occupy at most `max_cols` terminal columns,
/// appending `…` when truncation happens.  Returns `s` unchanged
/// when it already fits.
fn elide_to_width(s: &str, max_cols: usize) -> String {
    if max_cols == 0 {
        return String::new();
    }
    if unicode_width::UnicodeWidthStr::width(s) <= max_cols {
        return s.to_string();
    }
    // Reserve one column for the trailing ellipsis.
    let budget = max_cols.saturating_sub(1);
    let mut out = String::with_capacity(s.len());
    let mut col = 0usize;
    for ch in s.chars() {
        let w = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if col + w > budget {
            break;
        }
        out.push(ch);
        col += w;
    }
    out.push('…');
    out
}

/// Send `Request::ClaudeRelease` for the parsed /release command.
/// The daemon responds with one `Response::TaskReleased` per
/// released pane followed by `Response::Done`.  Our existing
/// `handle_response` arms render those frames as green
/// formatted blocks (see TaskReleased arm), so this dispatcher
/// just ships the request and returns; no follow-up state is
/// needed beyond surfacing a breadcrumb so the user knows the
/// release is in flight.
async fn release_panes(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    state: &mut AppState,
    cmd: crate::client::ReleaseCmd,
) -> Result<()> {
    let (target, clean_worktree, summary) = match cmd {
        crate::client::ReleaseCmd::Pane {
            pane_id,
            clean,
            summary,
        } => (
            crate::ipc::ClaudeReleaseTarget::Pane { pane_id },
            clean,
            summary,
        ),
        crate::client::ReleaseCmd::All { clean } => {
            (crate::ipc::ClaudeReleaseTarget::All, clean, None)
        }
    };
    let target_blurb = match &target {
        crate::ipc::ClaudeReleaseTarget::Pane { pane_id } => format!("pane {pane_id}"),
        crate::ipc::ClaudeReleaseTarget::All => "all panes".to_string(),
    };
    state.push_system_line(format!(
        "[release] requesting release of {target_blurb}{}…",
        if clean_worktree {
            " (clean worktree)"
        } else {
            ""
        }
    ));
    let req = Request::ClaudeRelease {
        target,
        clean_worktree,
        summary,
    };
    let mut frame = serde_json::to_string(&req).context("serializing ClaudeRelease")?;
    frame.push('\n');
    if let Err(e) = writer.write_all(frame.as_bytes()).await {
        state.push_error_line(format!("[error] sending /release to daemon: {e}"));
        return Ok(());
    }
    let _ = writer.flush().await;
    Ok(())
}

/// Synthesise the `[launched]` user-turn that classic chat emits
/// after a `/claude` flow lands its `Response::Done`.  The LLM reads
/// this on its next Chat round and takes over supervision per the
/// chat-takeover contract (see `docs/design/claude-chat-takeover.md`).
fn render_launched_block(launched: &[LaunchedPane]) -> String {
    let mut synth = String::new();
    for l in launched {
        if !synth.is_empty() {
            synth.push_str("\n\n---\n\n");
        }
        synth.push_str(l.description.trim_end());
        synth.push_str("\n\n[launched]\n");
        synth.push_str(&format!("  pane: {}\n", l.pane_id));
        if let Some(wt) = l.worktree.as_deref() {
            synth.push_str(&format!("  worktree: {wt}\n"));
        }
        if !l.resources.is_empty() {
            synth.push_str(&format!("  resources: {}\n", l.resources.join(", ")));
        }
        synth.push_str(&format!("  tag: {}\n", l.tag));
    }
    synth
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

// Concrete backend: the only caller is `run_chat_tui`, which always
// constructs the terminal over `CrosstermBackend<Stdout>`.  Keeping the
// function non-generic avoids having to prove `B::Error: Send + Sync +
// StdError` for anyhow, which doesn't hold for arbitrary backends.
fn draw(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    state: &AppState,
) -> Result<()> {
    terminal
        .draw(|frame| {
            // We deliberately do NOT use ratatui's `Wrap { trim: false }`
            // because that uses a word-boundary wrapper (`WordWrapper`)
            // which mishandles CJK: a Chinese sentence has no spaces,
            // so the wrapper either treats it as one un-breakable word
            // and overshoots the right border, or it breaks at the
            // first stray ASCII it finds inside the run (observed
            // 2026-05-15 — a "input box" English fragment in an
            // otherwise-Chinese assistant reply caused an aggressive
            // mid-sentence break).  Pre-wrapping every line ourselves
            // on display columns guarantees each Line we hand to
            // ratatui is already ≤ inner_width, so the renderer never
            // has to wrap anything itself.
            let total_area = frame.area();

            // While streaming, splice the live `[plan N/M done]`
            // The input box title now stays short — `[plan N/M done]`
            // and the streaming indicator moved to the status bar
            // (right-aligned), where they have more room and don't
            // collide with the editing-help hint.
            let input_title: &str = if state.steer_pending {
                " steer (Enter submits, empty Enter cancels, Ctrl-C exits) "
            } else {
                " input (Enter to send, Ctrl-C twice to exit) "
            };

            // Floor at 3 so an empty input still shows a 1-row cavity;
            // cap at half the frame so a paste-bomb can't consume the
            // whole transcript area.
            let max_input_height = (total_area.height / 2).max(3);
            let input_inner_width = total_area.width.saturating_sub(2);
            let input_segments = char_grid_wrap(&state.input, input_inner_width);
            let input_visual_rows = input_segments.len().max(1) as u16;
            let input_height = (input_visual_rows + 2).clamp(3, max_input_height);

            // Three-row layout: transcript (flex) → status bar
            // (1 row, no border) → input box (input_height).  The
            // status bar shows model + cwd + session at a glance so
            // the user doesn't have to scroll back to the banner to
            // see what they're talking to or where.
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(0),
                    Constraint::Length(1),
                    Constraint::Length(input_height),
                ])
                .split(total_area);

            // Transcript: pre-wrap every entry into one Line per
            // visual row so ratatui never has to wrap.
            let transcript_inner_width = chunks[0].width.saturating_sub(2);
            let mut transcript_lines: Vec<Line> = Vec::new();
            for tl in &state.transcript {
                push_wrapped_transcript_line(&mut transcript_lines, tl, transcript_inner_width);
            }
            let transcript_total_rows = transcript_lines.len() as u16;
            let transcript_visible_rows = chunks[0].height.saturating_sub(2);
            // Tail-following base: how many rows we'd hide above the
            // viewport to keep the newest content at the bottom edge.
            let tail_scroll = transcript_total_rows.saturating_sub(transcript_visible_rows);
            // User-requested scrollback subtracts from that base.  The
            // saturating arithmetic handles the boundary cleanly:
            // scroll_back ≥ tail_scroll lands us at the very top.
            let scroll_y = tail_scroll.saturating_sub(state.transcript_scroll_back);
            // Title shows a "↑ N rows" indicator when the user has
            // scrolled away from the tail, so they're never wondering
            // why new content stopped appearing on screen.
            let transcript_title: String = if state.transcript_scroll_back > 0 {
                format!(
                    " amaebi  ↑ {} rows from tail (PgDn / End to follow) ",
                    state.transcript_scroll_back
                )
            } else {
                " amaebi ".to_string()
            };

            let transcript = Paragraph::new(transcript_lines)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(transcript_title),
                )
                .scroll((scroll_y, 0));
            frame.render_widget(transcript, chunks[0]);

            // Status bar — single dim row between transcript and
            // input box.  Shows what model we're talking to, what
            // cwd we're in, and the truncated session id.  Built
            // here (not at startup) so /model switches and any
            // future cwd / session changes reflect immediately.
            let status = render_status_bar(state, chunks[1].width);
            frame.render_widget(status, chunks[1]);

            // Input: same pre-wrap.  Each segment becomes its own
            // Line, so ratatui draws them on consecutive rows
            // verbatim.  An empty input still emits one empty Line
            // so the box renders correctly.
            let mut input_lines: Vec<Line> = Vec::with_capacity(input_segments.len().max(1));
            if input_segments.is_empty() {
                input_lines.push(Line::from(""));
            } else {
                for &(s, e) in &input_segments {
                    input_lines.push(Line::from(state.input[s..e].to_string()));
                }
            }
            let input_para = Paragraph::new(input_lines)
                .block(Block::default().borders(Borders::ALL).title(input_title));
            frame.render_widget(input_para, chunks[2]);

            // Cursor position inside the input box: walk the typed-
            // so-far prefix under the same char-grid wrap so the
            // cursor lands exactly where the renderer placed the
            // matching character.
            let inner_width = chunks[2].width.saturating_sub(2);
            let typed_so_far = &state.input[..state.input_cursor.min(state.input.len())];
            let (cursor_row, cursor_col) = wrapped_cursor_position(typed_so_far, inner_width);
            let visible_rows = chunks[2].height.saturating_sub(2);
            let cursor_row = cursor_row.min(visible_rows.saturating_sub(1));
            frame.set_cursor_position((chunks[2].x + 1 + cursor_col, chunks[2].y + 1 + cursor_row));
        })
        .map_err(|e| anyhow::anyhow!("terminal.draw: {e}"))?;
    Ok(())
}

/// Append the visual rows for one transcript entry to `out`, wrapping
/// at `inner_width` columns under a char-grid wrap.  Preserves the
/// entry's `LineKind` styling on every visual row so a wrapped User
/// line stays cyan all the way down, etc.
/// One styled run after parsing inline markdown.  Plain text uses
/// the `LineKind`'s base style; the variants below add their own
/// modifier on top so a `**bold** text` row renders with the bold
/// half visually distinct.
#[derive(Debug, Clone, PartialEq)]
enum MdToken {
    Plain(String),
    Code(String),
    Bold(String),
    Italic(String),
}

impl MdToken {
    fn text(&self) -> &str {
        match self {
            MdToken::Plain(s) | MdToken::Code(s) | MdToken::Bold(s) | MdToken::Italic(s) => {
                s.as_str()
            }
        }
    }

    /// Build the ratatui `Style` for this token, on top of `base`.
    /// Each variant adds BOTH a modifier AND a distinct foreground
    /// so the styling is visible even when the terminal renders the
    /// modifier subtly (Windows Terminal's bold-on-default-fg is
    /// barely perceptible — observed 2026-05-16 — so we lean on
    /// colour as the primary signal and treat the modifier as
    /// secondary reinforcement).
    fn style_on(&self, base: Style) -> Style {
        match self {
            MdToken::Plain(_) => base,
            MdToken::Code(_) => base.fg(Color::Cyan),
            MdToken::Bold(_) => base.fg(Color::LightYellow).add_modifier(Modifier::BOLD),
            MdToken::Italic(_) => base.fg(Color::LightMagenta).add_modifier(Modifier::ITALIC),
        }
    }
}

/// Parse a line of assistant text into styled tokens.  Recognises:
///
/// - `` `inline code` ``     → `Code(text)`
/// - `**bold**`              → `Bold(text)`
/// - `*italic*`              → `Italic(text)` (also `_italic_`)
///
/// Markers are recognised greedily left-to-right.  An unmatched
/// opener (e.g. a stray `` ` `` with no close before EOL) falls
/// through as plain text — we never panic and never silently drop
/// characters.  Whole-line constructs (headings, fences, lists)
/// are handled at a higher level by `assistant_line_style`, not
/// here.
fn tokenize_inline_markdown(text: &str) -> Vec<MdToken> {
    let bytes = text.as_bytes();
    let mut tokens: Vec<MdToken> = Vec::new();
    let mut plain_start = 0usize;
    let mut i = 0usize;
    let len = bytes.len();

    let flush_plain = |tokens: &mut Vec<MdToken>, src: &str, start: usize, end: usize| {
        if start < end {
            tokens.push(MdToken::Plain(src[start..end].to_string()));
        }
    };

    while i < len {
        // `code` — single backtick run.  Matches the smallest closing
        // backtick (greedy on text, lazy on closer) so `foo` inside
        // `text with foo` doesn't lose the trailing foo.
        if bytes[i] == b'`' {
            // Pick the nearest closing backtick.  We deliberately do
            // NOT honour backslash escaping here — paths like
            // `foo\bar` and shell snippets like `printf '\n'` are far
            // more common in our streamed output than literal
            // backticks inside a code span, and CommonMark itself
            // also doesn't treat `\` as an escape inside `` `…` ``.
            if let Some(close) = (i + 1..len).find(|&j| bytes[j] == b'`') {
                let inside = &text[i + 1..close];
                if !inside.is_empty() {
                    flush_plain(&mut tokens, text, plain_start, i);
                    tokens.push(MdToken::Code(inside.to_string()));
                    i = close + 1;
                    plain_start = i;
                    continue;
                }
            }
        }
        // **bold** — paired double-asterisk.  Lookahead must NOT
        // also match an italic `*x*` immediately by accident; we
        // only fire when the next two bytes are `**` AND a closing
        // `**` exists.
        if i + 1 < len && bytes[i] == b'*' && bytes[i + 1] == b'*' {
            if let Some(close) = find_seq(bytes, i + 2, b"**") {
                let inside = &text[i + 2..close];
                if !inside.is_empty() {
                    flush_plain(&mut tokens, text, plain_start, i);
                    tokens.push(MdToken::Bold(inside.to_string()));
                    i = close + 2;
                    plain_start = i;
                    continue;
                }
            }
        }
        // *italic* / _italic_ — single asterisk or underscore.
        // Reject if the next char is a space (markdown convention:
        // `* not italic *`) or if we're at the boundary of a word
        // for `_` (so `foo_bar_baz` doesn't get italicized in
        // middle), to keep code identifiers sane.
        if (bytes[i] == b'*' || bytes[i] == b'_') && i + 1 < len {
            let marker = bytes[i];
            let next = bytes[i + 1];
            // Skip if it's a closing-only or empty marker.
            if next != b' ' && next != b'\t' && next != b'\n' && next != marker {
                if let Some(close) = (i + 1..len).find(|&j| bytes[j] == marker) {
                    // For `_`, require the closing `_` to also not
                    // be in the middle of a word (`_x_y` shouldn't
                    // italicize `x`).  For `*` we don't enforce
                    // word-boundary.
                    let close_ok = if marker == b'_' {
                        close + 1 == len
                            || !(bytes[close + 1].is_ascii_alphanumeric()
                                || bytes[close + 1] == b'_')
                    } else {
                        true
                    };
                    let inside = &text[i + 1..close];
                    if close_ok && !inside.is_empty() && !inside.starts_with(' ') {
                        flush_plain(&mut tokens, text, plain_start, i);
                        tokens.push(MdToken::Italic(inside.to_string()));
                        i = close + 1;
                        plain_start = i;
                        continue;
                    }
                }
            }
        }
        // Default: advance one byte.  We're in the middle of a
        // potential plain run; the next iteration will check for
        // markers again.
        i += 1;
    }
    flush_plain(&mut tokens, text, plain_start, len);
    tokens
}

/// Find a multi-byte sequence in `haystack` starting at `from`.
/// Inlined so the tokenizer doesn't need a `memchr` dependency.
fn find_seq(haystack: &[u8], from: usize, needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    let last = haystack.len() - needle.len();
    (from..=last).find(|&i| &haystack[i..i + needle.len()] == needle)
}

/// Returns the heading style + the heading text for `# / ## / ###`
/// lines, or `None` for non-headings.  Headings get a tint and
/// keep their leading hashes stripped so the rendered line reads
/// cleanly.
fn assistant_heading(text: &str) -> Option<(String, Style)> {
    let trimmed = text.trim_start();
    let prefix_len = text.len() - trimmed.len();
    let prefix = &text[..prefix_len];
    let (hashes, rest) = if let Some(r) = trimmed.strip_prefix("### ") {
        ("###", r)
    } else if let Some(r) = trimmed.strip_prefix("## ") {
        ("##", r)
    } else if let Some(r) = trimmed.strip_prefix("# ") {
        ("#", r)
    } else {
        return None;
    };
    let style = match hashes {
        "#" => Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD),
        "##" => Style::default()
            .fg(Color::LightYellow)
            .add_modifier(Modifier::BOLD),
        _ => Style::default().add_modifier(Modifier::BOLD),
    };
    // Preserve original leading whitespace so wrapped lists with
    // headings keep their indent.  Hashes themselves go away —
    // they're chrome, not content.
    Some((format!("{prefix}{rest}"), style))
}

/// Markdown-aware wrap for an assistant transcript line.  Recognises
/// inline `code` / `**bold**` / `*italic*` markers via
/// `tokenize_inline_markdown`, plus whole-line headings via
/// `assistant_heading`.  The output uses ratatui's Span composition
/// so different parts of the same visual row can carry different
/// colours / modifiers — important for code-spanned terms in
/// otherwise-plain prose.
///
/// Wrap math: walk tokens left-to-right, accumulating into the
/// current visual row; when adding a token's next character would
/// exceed `inner_width`, emit the row and start fresh.  Within a
/// single token we may also need to break (a long `code` span that
/// spans multiple rows).  Width comes from `unicode_width` so CJK +
/// emoji land correctly.
fn push_wrapped_assistant_line(out: &mut Vec<Line<'static>>, text: &str, inner_width: u16) {
    if text.is_empty() {
        out.push(Line::from(""));
        return;
    }
    if inner_width == 0 {
        // Degenerate viewport — push a single empty row to keep
        // layout intact rather than panicking.
        out.push(Line::from(""));
        return;
    }

    // Heading detection happens before token parsing because the
    // hash characters are chrome, not content — they'd otherwise
    // show up verbatim in the rendered output.  When a heading is
    // found, we still tokenize the rest (so "## **important**"
    // gets bold-on-yellow), but the heading style gets folded into
    // the base.
    let (effective_text, base_style) = if let Some((stripped, style)) = assistant_heading(text) {
        (stripped, style)
    } else {
        (text.to_string(), Style::default())
    };

    let tokens = tokenize_inline_markdown(&effective_text);

    // Walk tokens, splitting each one into character-grid pieces
    // that fit in the remaining width on the current row.  When a
    // row fills up, emit it and start fresh.  Plain tokens don't
    // get a special style; other tokens fold their modifier on top
    // of `base_style`.
    let inner = inner_width as usize;
    let mut current_row: Vec<Span<'static>> = Vec::new();
    let mut current_col: usize = 0;

    for token in &tokens {
        let token_style = token.style_on(base_style);
        let mut piece_start = 0usize;
        let mut piece_col = 0usize;
        let token_text = token.text();
        let token_bytes = token_text.as_bytes();
        for (idx, ch) in token_text.char_indices() {
            let w = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
            if w == 0 {
                continue;
            }
            if current_col + piece_col + w > inner {
                // Emit any accumulated piece into the current row,
                // then push the row out and start a new one.
                if piece_start < idx {
                    current_row.push(Span::styled(
                        token_text[piece_start..idx].to_string(),
                        token_style,
                    ));
                }
                out.push(Line::from(std::mem::take(&mut current_row)));
                current_col = 0;
                piece_start = idx;
                piece_col = 0;
                // Re-evaluate this character on the fresh row (fall
                // through; w + 0 is by construction <= inner unless
                // a single char is wider than the whole viewport,
                // in which case we accept overflow rather than
                // loop forever).
            }
            piece_col += w;
            // Step over the codepoint we just accepted.  We don't
            // need to track the byte offset of `idx + ch.len_utf8()`
            // because the next iteration of char_indices will give
            // us the next codepoint's start.
            let _ = token_bytes;
        }
        // Emit the trailing piece into the current row.
        if piece_start < token_text.len() {
            current_row.push(Span::styled(
                token_text[piece_start..].to_string(),
                token_style,
            ));
            current_col += piece_col;
        }
    }
    // Flush the final row even if it's empty (e.g. text was only
    // markers that all got eaten — vanishingly unlikely).
    out.push(Line::from(current_row));
}

fn push_wrapped_transcript_line(
    out: &mut Vec<Line<'static>>,
    tl: &TranscriptLine,
    inner_width: u16,
) {
    // Assistant lines carry inline markdown (`code`, **bold**,
    // *italic*) and may be heading lines.  Route them through the
    // markdown-aware path; everything else stays on the simpler
    // single-style wrap below.
    if matches!(tl.kind, LineKind::Assistant { .. }) {
        push_wrapped_assistant_line(out, &tl.text, inner_width);
        return;
    }
    let style = match tl.kind {
        LineKind::System => Style::default().fg(Color::DarkGray),
        LineKind::User => Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
        LineKind::Assistant { .. } => unreachable!("handled above"),
        LineKind::Error => Style::default().fg(Color::Red),
        LineKind::Tool => Style::default().fg(Color::Magenta),
        LineKind::Compacting => Style::default().fg(Color::Yellow),
        LineKind::Steer => Style::default().fg(Color::Yellow),
        LineKind::Launch => Style::default().fg(Color::Green),
    };
    // Continuation rows of a multi-line block (e.g. the [released]
    // formatted output, where each logical row was split into its
    // own TranscriptLine entry by push_kind_line) start with leading
    // whitespace by convention (`  pane:`, `  | tail line`, etc.).
    // Suppress the kind glyph on those so the visual marker only
    // appears on the leading row of the block, instead of repeating
    // 🚀 on every line.
    let is_continuation = tl.text.starts_with(' ') || tl.text.starts_with('\t');
    let prefix: &'static str = match tl.kind {
        _ if is_continuation => "",
        LineKind::System => "  ",
        LineKind::Error => "! ",
        LineKind::Tool => "🔧 ",
        LineKind::Compacting => "⏳ ",
        LineKind::Steer => "↳ ",
        LineKind::Launch => "🚀 ",
        LineKind::User | LineKind::Assistant { .. } => "",
    };

    // Empty text still produces an empty visual row so a deliberate
    // blank line in the transcript stays a blank line.
    if tl.text.is_empty() {
        out.push(Line::from(vec![
            Span::raw(prefix),
            Span::styled(String::new(), style),
        ]));
        return;
    }

    // Reserve `prefix`'s width on the first visual row only —
    // continuation rows shift left to use the full inner width.
    let prefix_cols =
        unicode_width::UnicodeWidthStr::width(prefix).min(inner_width as usize) as u16;
    let first_inner = inner_width.saturating_sub(prefix_cols);
    let first_segments = char_grid_wrap(&tl.text, first_inner);
    if first_segments.is_empty() {
        out.push(Line::from(vec![
            Span::raw(prefix),
            Span::styled(String::new(), style),
        ]));
        return;
    }

    let (s, e) = first_segments[0];
    out.push(Line::from(vec![
        Span::raw(prefix),
        Span::styled(tl.text[s..e].to_string(), style),
    ]));
    if first_segments.len() == 1 {
        return;
    }

    // Wrap the remaining text at the full inner width — continuation
    // rows have no prefix, so they get more columns.  We can't reuse
    // `first_segments[1..]` because those breaks were computed under
    // the narrower `first_inner`.
    let remainder = &tl.text[e..];
    if remainder.is_empty() {
        return;
    }
    for (rs, re) in char_grid_wrap(remainder, inner_width) {
        out.push(Line::from(Span::styled(
            remainder[rs..re].to_string(),
            style,
        )));
    }
}

/// Return the (row, col) in display-width terms where the cursor
/// should land after typing `typed` into a box of inner width
/// `inner_width`.  Implements a character-grid wrap (no word
/// boundaries): a glyph that won't fit on the current row goes to
/// the next row at column 0.
///
/// Must match `char_grid_wrap`'s break decisions exactly, otherwise
/// the rendered text and the cursor diverge.
fn wrapped_cursor_position(typed: &str, inner_width: u16) -> (u16, u16) {
    if inner_width == 0 {
        return (0, 0);
    }
    let inner = inner_width as usize;
    let mut row: u16 = 0;
    let mut col: usize = 0;
    for ch in typed.chars() {
        let w = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if w == 0 {
            continue;
        }
        if col + w > inner {
            row = row.saturating_add(1);
            col = 0;
        }
        col += w;
    }
    (row, col as u16)
}

/// Hard char-grid wrap: split `text` on display-column boundaries so
/// every emitted segment occupies at most `inner_width` terminal
/// columns.  Unlike ratatui's `Wrap { trim: false }` (which is
/// word-aware via `WordWrapper`), this never tries to break on
/// spaces — important for CJK input, where the WordWrapper treats a
/// whole sentence as one un-breakable word and overshoots the right
/// border, OR makes a poor break at the first ASCII character it
/// finds inside the run.  The output is a list of byte ranges into
/// `text`, one per visual row, allocation-free at call time apart
/// from the Vec itself.  `Vec::is_empty()` if `text` is empty.
fn char_grid_wrap(text: &str, inner_width: u16) -> Vec<(usize, usize)> {
    if inner_width == 0 || text.is_empty() {
        return Vec::new();
    }
    let inner = inner_width as usize;
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut start: usize = 0;
    let mut col: usize = 0;
    for (idx, ch) in text.char_indices() {
        let w = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if w == 0 {
            continue;
        }
        if col + w > inner {
            // Break: emit the current segment, start a new one
            // at this character.
            out.push((start, idx));
            start = idx;
            col = 0;
        }
        col += w;
    }
    // Final segment, if any.
    if start < text.len() {
        out.push((start, text.len()));
    }
    out
}

// ---------------------------------------------------------------------------
// Terminal lifecycle
// ---------------------------------------------------------------------------

/// RAII guard: `enter()` flips raw mode + alternate screen, `Drop`
/// restores both.  Mirrors the `dashboard::TerminalGuard` pattern so a
/// panic inside `run_chat_tui` leaves the user's shell usable.
struct TerminalGuard;

impl TerminalGuard {
    fn enter() -> Result<Self> {
        enable_raw_mode().context("enabling raw mode")?;
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Step-1 smoke tests for the pure-state helpers.  Terminal lifecycle
    // + event loop are covered by manual runs, not cargo test, because
    // they require a real TTY.  The rules we lock here are the ones
    // that a refactor might plausibly break: stream-chunking that
    // respects newlines, backspace UTF-8 safety, and follow-tail math.

    /// Build an AppState with placeholder socket/cwd values that the
    /// pure-state tests don't exercise.
    fn test_state() -> AppState {
        AppState::new(
            "sid".into(),
            "model".into(),
            std::path::PathBuf::from("/tmp/amaebi-test.sock"),
            "/tmp".to_string(),
        )
    }

    #[test]
    fn assistant_chunks_across_newlines_split_into_distinct_lines() {
        let mut state = test_state();
        // A single chunk spanning two logical lines must land as two
        // transcript entries, the first closed (is_open=false) and the
        // second open so the next chunk attaches to it.
        state.push_assistant_chunk("first\nsecond partial");
        assert_eq!(state.transcript.len(), 2);
        assert!(matches!(
            state.transcript[0].kind,
            LineKind::Assistant { is_open: false }
        ));
        assert_eq!(state.transcript[0].text, "first");
        assert!(matches!(
            state.transcript[1].kind,
            LineKind::Assistant { is_open: true }
        ));
        assert_eq!(state.transcript[1].text, "second partial");
    }

    #[test]
    fn assistant_chunk_continues_open_line() {
        // Two streamed chunks with no newline between them must merge
        // into a single line — the common case for token-by-token
        // model output.
        let mut state = test_state();
        state.push_assistant_chunk("hello ");
        state.push_assistant_chunk("world");
        assert_eq!(state.transcript.len(), 1);
        assert_eq!(state.transcript[0].text, "hello world");
        assert!(matches!(
            state.transcript[0].kind,
            LineKind::Assistant { is_open: true }
        ));
    }

    #[test]
    fn handle_response_done_seals_open_line() {
        // Response::Done must close the currently open assistant line
        // so a subsequent turn doesn't accidentally concatenate onto
        // the previous reply.
        let mut state = test_state();
        state.push_assistant_chunk("partial without newline");
        state.streaming = true;

        let outcome = handle_response(Response::Done, &mut state);
        assert_eq!(outcome, ResponseOutcome::TurnEnded);
        assert!(!state.streaming);
        assert!(matches!(
            state.transcript.last().unwrap().kind,
            LineKind::Assistant { is_open: false }
        ));

        // The next assistant chunk must not extend the now-sealed line.
        state.push_assistant_chunk("next turn");
        assert_eq!(state.transcript.len(), 2);
    }

    #[test]
    fn floor_char_boundary_handles_multibyte() {
        // "日本語" is three 3-byte chars.  Stepping back one "character"
        // from the end must land on the start of "語", not in the middle
        // of a UTF-8 sequence.
        let s = "日本語";
        let past_end = s.len();
        // At past_end: nothing to undo.
        assert_eq!(floor_char_boundary(s, past_end), past_end);
        // Asking for past_end-1 is inside "語" — snap back to "語"'s start.
        let guess = past_end - 1;
        let clamped = floor_char_boundary(s, guess);
        assert!(s.is_char_boundary(clamped));
        // And the result must be strictly less than past_end, not
        // equal, otherwise Backspace would be a no-op.
        assert!(clamped < past_end);
    }

    // ── wrapped_cursor_position ────────────────────────────────
    // Drives the multi-row cursor placement when input wraps in the
    // bottom box.  Anchored on the contract that cursor position is
    // measured in display-width terms (CJK = 2 cols), and that
    // hitting the right edge advances row.

    #[test]
    fn wrapped_cursor_position_single_row() {
        // Short input fits on one row; col equals display width, row is 0.
        assert_eq!(wrapped_cursor_position("hello", 80), (0, 5));
        assert_eq!(wrapped_cursor_position("你好", 80), (0, 4));
    }

    #[test]
    fn wrapped_cursor_position_wraps_when_full() {
        // After 5 chars in a 5-wide box, the next char goes to a new
        // row at column 0.  The current cursor (typing-so-far ends
        // at the boundary) reports (1, 0) — the start of the next row.
        // Anything fewer than 5 chars stays on row 0.
        assert_eq!(wrapped_cursor_position("abcd", 5), (0, 4));
        assert_eq!(wrapped_cursor_position("abcde", 5), (0, 5));
        // 5 chars + 1 more → wrap.
        assert_eq!(wrapped_cursor_position("abcdef", 5), (1, 1));
    }

    #[test]
    fn wrapped_cursor_position_zero_width_safe() {
        // Pathological: a 0-wide visible region (window collapsed).
        // Must not divide-by-zero or panic.  We don't care what the
        // value is, only that we get something.
        let _ = wrapped_cursor_position("hello", 0);
    }

    // ── char_grid_wrap ───────────────────────────────────────────
    // The render path now pre-wraps every Line with this function
    // before handing it to ratatui, so it's a load-bearing primitive.
    // The contract: every emitted segment has display width
    // ≤ inner_width, and the segments concatenate back to the input
    // verbatim.  These tests pin both halves.

    fn assert_wrap_round_trips(text: &str, inner: u16) {
        let segs = char_grid_wrap(text, inner);
        // Each segment fits.
        for &(s, e) in &segs {
            let w = unicode_width::UnicodeWidthStr::width(&text[s..e]);
            assert!(
                w <= inner as usize,
                "segment {:?} has width {} > inner {}",
                &text[s..e],
                w,
                inner
            );
        }
        // Segments concatenate back to the original.
        let glued: String = segs.iter().map(|&(s, e)| &text[s..e]).collect();
        assert_eq!(glued, text);
    }

    #[test]
    fn char_grid_wrap_short_input_one_segment() {
        let segs = char_grid_wrap("hi", 10);
        assert_eq!(segs, vec![(0, 2)]);
    }

    #[test]
    fn char_grid_wrap_breaks_cjk_at_column_boundary() {
        // CJK glyphs are 2 cols each; in a 5-wide box we fit 2 per row
        // (4 cols) and the third spills.  Without our wrap, ratatui's
        // word-aware wrapper would either overshoot or break at an
        // unrelated ASCII character.
        let text = "你好世界";
        let segs = char_grid_wrap(text, 5);
        assert_eq!(segs.len(), 2);
        // Each segment has 2 glyphs (4 cols), well within 5.
        assert_eq!(&text[segs[0].0..segs[0].1], "你好");
        assert_eq!(&text[segs[1].0..segs[1].1], "世界");
        assert_wrap_round_trips(text, 5);
    }

    #[test]
    fn char_grid_wrap_mixed_cjk_and_ascii_no_word_breaks() {
        // The window-5 regression: an English fragment ("input box")
        // embedded in Chinese — ratatui's WordWrapper would prefer to
        // break on the space between "input" and "box", leaving the
        // surrounding Chinese running off the right border.  Our
        // char-grid wrap doesn't care about spaces; it just tracks
        // column count and breaks where the next glyph won't fit.
        let text = "应该看到 input box 这里继续";
        // 30 cols is wide enough for several glyphs; the wrap must
        // simply walk left-to-right and break only when full.
        assert_wrap_round_trips(text, 30);
        assert_wrap_round_trips(text, 12);
        assert_wrap_round_trips(text, 4);
    }

    #[test]
    fn char_grid_wrap_empty_and_zero_width() {
        assert!(char_grid_wrap("", 10).is_empty());
        assert!(char_grid_wrap("nonempty", 0).is_empty());
    }

    #[test]
    fn char_grid_wrap_matches_wrapped_cursor_position() {
        // Critical invariant: the cursor walker and the wrapper must
        // agree on break decisions, otherwise the cursor and the
        // rendered text drift apart.  Walk a few realistic inputs
        // under both and confirm the row counts match.
        for (text, inner) in [
            ("你好世界", 5u16),
            ("应该看到 input box 这里", 12),
            ("hello world goodbye world", 10),
            ("aaaaa", 1),
        ] {
            let segs = char_grid_wrap(text, inner);
            let (final_row, _final_col) = wrapped_cursor_position(text, inner);
            // After the full text, the cursor sits at the start of
            // (or partway through) the last visual row.  So the row
            // index equals segs.len() - 1, unless the text is empty.
            if segs.is_empty() {
                assert_eq!(final_row, 0);
            } else {
                assert_eq!(
                    final_row as usize,
                    segs.len() - 1,
                    "wrap and cursor disagree for {text:?} at inner={inner}"
                );
            }
        }
    }

    // Note: the previous `input_cursor_column_*` tests anchored a
    // single-row cursor formula that is now obsolete.  The wrap-aware
    // replacement is `wrapped_cursor_position`, covered by its own
    // tests above; the CJK display-width contract (你 = 2 cols) now
    // lives in `wrapped_cursor_position_single_row`.

    #[test]
    fn classify_input_routes_slash_model() {
        // Bare `/model` shows the current model; `/model <name>`
        // switches.  Anything else with leading text should not be
        // treated as a slash command.
        assert_eq!(classify_input("/model"), InputDispatch::ShowModel);
        assert_eq!(
            classify_input("/model claude-opus-4.7[1m]"),
            InputDispatch::SwitchModel("claude-opus-4.7[1m]".to_string())
        );
    }

    #[test]
    fn classify_input_routes_claude_with_tasks() {
        // /claude carries the parser output verbatim through
        // InputDispatch::Claude so the dispatcher can hand them off
        // to launch_claude_tasks without re-parsing.
        let dispatched = classify_input("/claude \"do X\"");
        match dispatched {
            InputDispatch::Claude(tasks) => {
                assert_eq!(tasks.len(), 1);
                assert_eq!(tasks[0].description, "do X");
            }
            other => panic!("expected Claude(_), got {other:?}"),
        }
    }

    #[test]
    fn classify_input_routes_replyreview_with_prs() {
        let dispatched = classify_input("/replyreview 165 168");
        match dispatched {
            InputDispatch::ReplyReview(prs) => assert_eq!(prs, vec![165, 168]),
            other => panic!("expected ReplyReview(_), got {other:?}"),
        }
    }

    #[test]
    fn classify_input_surfaces_slash_parse_errors() {
        // Bare /claude is a usage error; the user sees it in the
        // transcript via SlashError rather than a silent no-op.
        match classify_input("/claude") {
            InputDispatch::SlashError(msg) => {
                assert!(msg.contains("usage:"), "expected usage hint, got {msg:?}");
            }
            other => panic!("expected SlashError(_), got {other:?}"),
        }
    }

    #[test]
    fn classify_input_routes_release_pane() {
        // /release with a real pane id parses into Release(Pane).
        match classify_input("/release %54") {
            InputDispatch::Release(crate::client::ReleaseCmd::Pane { pane_id, .. }) => {
                assert_eq!(pane_id, "%54");
            }
            other => panic!("expected Release(Pane), got {other:?}"),
        }
    }

    #[test]
    fn classify_input_routes_release_all() {
        match classify_input("/release all --clean") {
            InputDispatch::Release(crate::client::ReleaseCmd::All { clean }) => {
                assert!(clean, "--clean must propagate");
            }
            other => panic!("expected Release(All), got {other:?}"),
        }
    }

    #[test]
    fn classify_input_release_parse_error_is_slash_error() {
        // A parser-level failure (e.g. /release alone with no
        // target) must surface as SlashError, not fall through to
        // SendChat — same shape as /claude error handling.
        match classify_input("/release") {
            InputDispatch::SlashError(_) => {}
            other => panic!("expected SlashError, got {other:?}"),
        }
    }

    #[test]
    fn elide_to_width_returns_unchanged_when_fits() {
        assert_eq!(elide_to_width("abc", 10), "abc");
        assert_eq!(elide_to_width("你好", 10), "你好");
    }

    #[test]
    fn elide_to_width_appends_ellipsis_when_too_long() {
        // 20 ascii chars in a 10-col box → keep 9, add ellipsis.
        let s = "abcdefghijklmnopqrst";
        let out = elide_to_width(s, 10);
        assert!(out.ends_with('…'));
        assert_eq!(unicode_width::UnicodeWidthStr::width(out.as_str()), 10);
    }

    #[test]
    fn elide_to_width_handles_cjk_widths() {
        // 4 CJK glyphs = 8 cols.  In a 5-col box we keep 2 glyphs
        // (4 cols) + ellipsis.
        let s = "你好世界";
        let out = elide_to_width(s, 5);
        assert!(out.ends_with('…'));
        assert!(unicode_width::UnicodeWidthStr::width(out.as_str()) <= 5);
    }

    #[test]
    fn shorten_cwd_replaces_home_prefix() {
        // We can't poke HOME safely from a test (it's process-wide),
        // but we can at least call shorten_cwd and assert that a
        // path beneath the current HOME picks up the ~.  If HOME is
        // unset (CI quirks) the function falls through and returns
        // input verbatim — this test then becomes vacuous, which is
        // acceptable as it can't fail.
        if let Ok(home) = std::env::var("HOME") {
            if !home.is_empty() {
                let inside = format!("{home}/projects/foo");
                let short = shorten_cwd(&inside);
                assert_eq!(short, "~/projects/foo");
            }
        }
        // Outside-HOME path stays verbatim.
        assert_eq!(shorten_cwd("/etc/hosts"), "/etc/hosts");
    }

    #[test]
    fn build_status_right_idle_returns_empty() {
        let s = test_state();
        assert_eq!(build_status_right(&s), "");
    }

    #[test]
    fn build_status_right_streaming_shows_streaming() {
        let mut s = test_state();
        s.streaming = true;
        assert_eq!(build_status_right(&s), "streaming");
    }

    #[test]
    fn build_status_right_streaming_with_plan_shows_progress() {
        // When the LLM has emitted a checklist mid-stream, the
        // status bar splices the live count into the streaming
        // indicator so the user sees plan progress without
        // squinting at the input title.
        let mut s = test_state();
        s.streaming = true;
        s.plan_tracker
            .push("- [x] Step 1\n- [ ] Step 2\n- [ ] Step 3\n");
        let right = build_status_right(&s);
        assert!(
            right.contains("[plan 1/3 done]"),
            "right side must surface live plan progress; got {right:?}"
        );
        assert!(right.contains("streaming"));
    }

    #[test]
    fn build_status_right_steer_overrides_streaming() {
        // Steer mode is more specific than streaming; the right
        // side should pivot to the steer hint rather than show
        // both.
        let mut s = test_state();
        s.streaming = true;
        s.steer_pending = true;
        let right = build_status_right(&s);
        assert!(right.starts_with("steer"));
        assert!(!right.contains("streaming"));
    }

    #[test]
    fn tokenize_inline_markdown_recognises_code_bold_italic() {
        let toks = tokenize_inline_markdown("a `code` b **bold** c *italic* d");
        // Plain runs get separated by the markup tokens.  We don't
        // care about exact whitespace boundaries beyond "no token
        // is empty and the concatenation round-trips".
        let mut all = String::new();
        for t in &toks {
            all.push_str(t.text());
        }
        // Markers themselves are stripped — the recovered text is
        // the rendered visible text, not the source.
        assert_eq!(all, "a code b bold c italic d");
        assert!(toks
            .iter()
            .any(|t| matches!(t, MdToken::Code(s) if s == "code")));
        assert!(toks
            .iter()
            .any(|t| matches!(t, MdToken::Bold(s) if s == "bold")));
        assert!(toks
            .iter()
            .any(|t| matches!(t, MdToken::Italic(s) if s == "italic")));
    }

    #[test]
    fn tokenize_inline_markdown_unmatched_marker_falls_through_as_plain() {
        // A stray backtick with no closer must not panic and must
        // not eat trailing text.
        let toks = tokenize_inline_markdown("foo `unfinished bar");
        let joined: String = toks.iter().map(|t| t.text()).collect();
        assert_eq!(joined, "foo `unfinished bar");
        // No Code token should have been produced.
        assert!(!toks.iter().any(|t| matches!(t, MdToken::Code(_))));
    }

    #[test]
    fn tokenize_inline_markdown_underscore_inside_word_not_italic() {
        // `foo_bar_baz` is a common identifier; the parser must
        // not treat the inner `_bar_` as italic.
        let toks = tokenize_inline_markdown("foo_bar_baz");
        assert!(!toks.iter().any(|t| matches!(t, MdToken::Italic(_))));
        let joined: String = toks.iter().map(|t| t.text()).collect();
        assert_eq!(joined, "foo_bar_baz");
    }

    #[test]
    fn assistant_heading_strips_hashes_and_assigns_style() {
        let (stripped, style) = assistant_heading("# Title").unwrap();
        assert_eq!(stripped, "Title");
        // Bold modifier should be present.
        assert!(style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn assistant_heading_returns_none_for_non_heading() {
        assert!(assistant_heading("just a paragraph").is_none());
        // `#tag` (no space after hash) is not a heading.
        assert!(assistant_heading("#tag at line start").is_none());
    }

    #[test]
    fn push_wrapped_assistant_line_emits_styled_spans_for_inline_code() {
        // Inline `code` between plain text should produce at least
        // three spans on the same row: "before ", "code", " after".
        let mut out: Vec<Line<'static>> = Vec::new();
        push_wrapped_assistant_line(&mut out, "before `code` after", 80);
        assert_eq!(out.len(), 1, "single short line shouldn't wrap");
        let row = &out[0];
        let span_texts: Vec<&str> = row.spans.iter().map(|s| s.content.as_ref()).collect();
        let joined = span_texts.concat();
        assert_eq!(joined, "before code after");
        // At least one span carries Cyan (the Code style).
        let any_cyan = row.spans.iter().any(|s| s.style.fg == Some(Color::Cyan));
        assert!(any_cyan, "code span must render in cyan");
    }

    #[test]
    fn classify_input_passes_through_plain_text() {
        // Non-slash text is a normal chat message.  Leading slashes
        // that don't match any command (e.g. `/notacommand`) also
        // fall through, matching parse_slash_command's None branch.
        assert_eq!(classify_input("hello world"), InputDispatch::SendChat,);
        assert_eq!(classify_input("/notacommand foo"), InputDispatch::SendChat,);
    }

    #[test]
    fn render_launched_block_concatenates_panes_with_separator() {
        // The synthesised user-turn shape is the chat-takeover
        // contract: original description + [launched] block per pane,
        // separated by `---`.  Anchoring on the literal markers so a
        // future refactor can't silently change the format the daemon
        // (and Claude prompts) expect.
        let launched = vec![
            LaunchedPane {
                pane_id: "%41".into(),
                description: "do the thing".into(),
                tag: "thing-1".into(),
                worktree: Some("/tmp/wt-thing".into()),
                resources: vec!["sim-9900".into()],
            },
            LaunchedPane {
                pane_id: "%42".into(),
                description: "do the other".into(),
                tag: "other-2".into(),
                worktree: None,
                resources: vec![],
            },
        ];
        let synth = render_launched_block(&launched);
        assert!(synth.contains("[launched]"));
        assert!(synth.contains("  pane: %41"));
        assert!(synth.contains("  worktree: /tmp/wt-thing"));
        assert!(synth.contains("  resources: sim-9900"));
        assert!(synth.contains("  tag: thing-1"));
        assert!(synth.contains("---"));
        assert!(synth.contains("  pane: %42"));
        assert!(synth.contains("  tag: other-2"));
        // Single-pane case (no separator).
        let single = render_launched_block(&launched[..1]);
        assert!(!single.contains("---"));
    }

    // ── Ctrl-C steer ────────────────────────────────────────────
    // Mid-turn Ctrl-C arms steer mode: subsequent Response frames
    // buffer instead of clobbering the user's correction; SteerAck
    // (or empty-Enter cancel) flushes the buffer back to the
    // transcript.  These tests pin the state machine's contract.

    #[test]
    fn ctrl_c_while_streaming_arms_steer() {
        // First Ctrl-C while streaming: handle_key returns
        // InterruptForSteer and stamps last_ctrl_c so a second press
        // can detect the double-press exit gesture.
        let mut s = test_state();
        s.streaming = true;
        let outcome = handle_key(ctrl_key('c'), &mut s);
        assert!(matches!(outcome, KeyOutcome::InterruptForSteer));
        assert!(s.last_ctrl_c.is_some());
    }

    #[test]
    fn second_ctrl_c_while_steer_pending_exits() {
        // Second Ctrl-C while steer is already armed by Ctrl-C
        // exits — the user's first press already committed to
        // wanting out, so the second is a real exit.
        let mut s = test_state();
        s.streaming = true;
        s.steer_pending = true;
        s.steer_source = SteerSource::UserCtrlC;
        let outcome = handle_key(ctrl_key('c'), &mut s);
        assert!(matches!(outcome, KeyOutcome::Exit));
    }

    #[test]
    fn ctrl_c_when_daemon_waiting_steer_armed_requires_double_press() {
        // When steer was armed by Response::WaitingForInput (not by
        // a user Ctrl-C), the user has NOT pressed Ctrl-C before.
        // The first Ctrl-C must therefore behave like the idle
        // empty-input case (arm double-press window + show hint),
        // not like a "second press" exit.  A second press inside
        // the window finally exits.
        let mut s = test_state();
        s.streaming = true;
        s.steer_pending = true;
        s.steer_source = SteerSource::DaemonWaitingForInput;

        let first = handle_key(ctrl_key('c'), &mut s);
        assert!(matches!(first, KeyOutcome::Continue));
        assert!(s.last_ctrl_c.is_some());
        assert_eq!(s.transcript.last().unwrap().text, CTRLC_EXIT_HINT);
        assert!(
            s.steer_pending,
            "first Ctrl-C must not silently disarm steer mode"
        );

        // Second press while last_ctrl_c is still fresh → Exit.
        let second = handle_key(ctrl_key('c'), &mut s);
        assert!(matches!(second, KeyOutcome::Exit));
    }

    #[test]
    fn ctrl_c_idle_with_text_clears_input() {
        // Ctrl-C with a non-empty input box clears the line (shell
        // readline convention) instead of arming exit.
        let mut s = test_state();
        s.input = "half-typed".into();
        s.input_cursor = s.input.len();
        let outcome = handle_key(ctrl_key('c'), &mut s);
        assert!(matches!(outcome, KeyOutcome::Continue));
        assert!(s.input.is_empty());
        assert_eq!(s.input_cursor, 0);
    }

    #[test]
    fn ctrl_c_idle_empty_arms_double_press() {
        // First Ctrl-C with empty input on an idle session arms the
        // double-press.  A hint line is added to the transcript so
        // the user knows what's about to happen.
        let mut s = test_state();
        let outcome = handle_key(ctrl_key('c'), &mut s);
        assert!(matches!(outcome, KeyOutcome::Continue));
        assert!(s.last_ctrl_c.is_some());
        assert_eq!(s.transcript.last().unwrap().text, CTRLC_EXIT_HINT,);
    }

    #[test]
    fn enter_while_steer_pending_routes_to_submit_steer() {
        // Non-empty Enter while steer-pending submits as a steer
        // correction, not as a fresh chat prompt.
        let mut s = test_state();
        s.streaming = true;
        s.steer_pending = true;
        s.input = "fix this".into();
        s.input_cursor = s.input.len();
        let outcome = handle_key(key(KeyCode::Enter), &mut s);
        match outcome {
            KeyOutcome::SubmitSteer(text) => assert_eq!(text, "fix this"),
            other => panic!("expected SubmitSteer, got {other:?}"),
        }
    }

    #[test]
    fn empty_enter_while_steer_pending_cancels() {
        let mut s = test_state();
        s.streaming = true;
        s.steer_pending = true;
        let outcome = handle_key(key(KeyCode::Enter), &mut s);
        assert!(matches!(outcome, KeyOutcome::CancelSteer));
    }

    #[test]
    fn waiting_for_input_arms_steer_mode_not_turn_end() {
        // Daemon's `Response::WaitingForInput` means the model is
        // asking the user a clarifying question and expects the
        // reply via `Request::Steer` (not a fresh `Request::Chat`).
        // Arming steer_pending = true causes the next Enter to
        // route through send_steer; the existing SteerAck/Done
        // handling clears the flag.
        let mut s = test_state();
        s.streaming = true;
        let outcome = handle_response(
            Response::WaitingForInput {
                prompt: "which option do you prefer?".to_string(),
            },
            &mut s,
        );
        assert!(matches!(outcome, ResponseOutcome::Continuing));
        assert!(s.steer_pending, "WaitingForInput must arm steer mode");
        assert!(s.streaming, "streaming should stay true — same turn");
        // The prompt text shows up as a Steer-kind line so the
        // user sees the question above the input box.
        let last = s.transcript.last().unwrap();
        assert!(matches!(last.kind, LineKind::Steer));
        assert!(last.text.contains("which option"));
    }

    #[test]
    fn done_during_steer_pending_flushes_buffer_and_clears_mode() {
        // Regression for the 2026-05-16 manual-test bug: if the
        // daemon's current turn finishes (Response::Done) before
        // our Steer reaches a turn boundary — common when the LLM
        // emits a long text-only stream with no tools — there'll
        // never be a SteerAck, and the user must not be stranded
        // in steer-pending UI.  Done in steer mode must flush the
        // buffered Text frames AND clear the steer flag.
        let mut s = test_state();
        s.streaming = true;
        s.steer_pending = true;
        let _ = handle_response(
            Response::Text {
                chunk: "queued during steer\n".into(),
            },
            &mut s,
        );
        assert_eq!(s.steer_buffer.len(), 1);
        let _ = handle_response(Response::Done, &mut s);
        assert!(!s.steer_pending);
        assert!(!s.streaming);
        assert!(
            s.transcript
                .iter()
                .any(|tl| tl.text.contains("queued during steer")),
            "buffered chunk must surface on Done so the user sees the final output"
        );
    }

    #[test]
    fn error_during_steer_pending_flushes_buffer_and_clears_mode() {
        // Same shape as the Done test, for the error path: a turn
        // that errors before SteerAck must still clear steer mode.
        let mut s = test_state();
        s.streaming = true;
        s.steer_pending = true;
        let _ = handle_response(
            Response::Text {
                chunk: "queued\n".into(),
            },
            &mut s,
        );
        let _ = handle_response(
            Response::Error {
                message: "bedrock blew up".into(),
            },
            &mut s,
        );
        assert!(!s.steer_pending);
        assert!(!s.streaming);
        assert!(
            s.transcript.iter().any(|tl| tl.text.contains("queued")),
            "buffered frames must flush on Error too"
        );
    }

    #[test]
    fn push_kind_line_splits_multiline_block() {
        // Regression: the synthesised /claude [launched] block was
        // getting jammed onto a single TranscriptLine, then
        // re-flowed by ratatui's wrap into one mashed paragraph.
        // push_kind_line now emits one entry per logical line so
        // the wrap logic respects line boundaries.
        let mut s = test_state();
        s.push_user_line("> a\n\nb\n  c".to_string());
        let kinds_and_texts: Vec<(LineKind, &str)> = s
            .transcript
            .iter()
            .map(|tl| (tl.kind, tl.text.as_str()))
            .collect();
        assert_eq!(kinds_and_texts.len(), 4);
        assert_eq!(kinds_and_texts[0].1, "> a");
        assert_eq!(kinds_and_texts[1].1, "");
        assert_eq!(kinds_and_texts[2].1, "b");
        assert_eq!(kinds_and_texts[3].1, "  c");
        for (kind, _) in &kinds_and_texts {
            assert!(matches!(kind, LineKind::User));
        }
    }

    #[test]
    fn buffered_frames_drain_on_steer_ack_in_order() {
        // While steer_pending, Text frames buffer.  When SteerAck
        // arrives, the buffered chunks should land in the transcript
        // in their original order, exactly as if they'd never been
        // intercepted.
        let mut s = test_state();
        s.streaming = true;
        s.steer_pending = true;

        let _ = handle_response(
            Response::Text {
                chunk: "alpha\n".into(),
            },
            &mut s,
        );
        let _ = handle_response(
            Response::Text {
                chunk: "beta\n".into(),
            },
            &mut s,
        );
        // Nothing in transcript yet — frames are in the buffer.
        assert_eq!(s.steer_buffer.len(), 2);
        assert!(!s
            .transcript
            .iter()
            .any(|tl| tl.text.contains("alpha") || tl.text.contains("beta")));

        // SteerAck flushes.
        let _ = handle_response(Response::SteerAck, &mut s);
        assert!(!s.steer_pending);
        let texts: Vec<&str> = s.transcript.iter().map(|tl| tl.text.as_str()).collect();
        let alpha_pos = texts.iter().position(|t| t.contains("alpha")).unwrap();
        let beta_pos = texts.iter().position(|t| t.contains("beta")).unwrap();
        assert!(alpha_pos < beta_pos, "alpha must precede beta");
    }

    #[test]
    fn pageup_pagedown_steps_scroll_back() {
        // PgUp grows scroll-back by PAGE_STEP_ROWS, PgDn shrinks it,
        // and PgDn past 0 saturates at "follow tail" rather than
        // wrapping or panicking.
        let mut s = test_state();
        let _ = handle_key(key(KeyCode::PageUp), &mut s);
        assert_eq!(s.transcript_scroll_back, PAGE_STEP_ROWS);
        let _ = handle_key(key(KeyCode::PageUp), &mut s);
        assert_eq!(s.transcript_scroll_back, PAGE_STEP_ROWS * 2);
        let _ = handle_key(key(KeyCode::PageDown), &mut s);
        assert_eq!(s.transcript_scroll_back, PAGE_STEP_ROWS);
        let _ = handle_key(key(KeyCode::PageDown), &mut s);
        assert_eq!(s.transcript_scroll_back, 0);
        // Past-zero PgDn must not underflow.
        let _ = handle_key(key(KeyCode::PageDown), &mut s);
        assert_eq!(s.transcript_scroll_back, 0);
    }

    #[test]
    fn tool_label_specialises_known_tools() {
        // Read / edit / shell / tmux pick up distinctive glyphs that
        // make the tool obvious without reading the detail string.
        assert!(tool_label("read_file").contains("📄"));
        assert!(tool_label("edit_file").contains("✏"));
        assert!(tool_label("tmux_send_text").contains("send-text"));
        assert!(tool_label("tmux_capture_pane").contains("capture"));
        // Unknown tool falls back to its bare name; the kind-level
        // 🔧 prefix in `push_wrapped_transcript_line` carries the
        // tool signal.
        assert_eq!(tool_label("some_other_tool"), "some_other_tool");
    }

    #[test]
    fn handle_response_routes_tool_use_to_tool_kind() {
        // Regression guard: a refactor must keep ToolUse on the
        // magenta + 🔧 path rather than dropping it back into the
        // generic dim-grey System bucket.
        let mut s = test_state();
        s.streaming = true;
        let _ = handle_response(
            Response::ToolUse {
                name: "read_file".into(),
                detail: "src/foo.rs".into(),
            },
            &mut s,
        );
        let last = s.transcript.last().unwrap();
        assert!(matches!(last.kind, LineKind::Tool));
        assert!(last.text.contains("src/foo.rs"));
        assert!(last.text.contains("📄"));
    }

    #[test]
    fn handle_response_routes_compacting_to_compacting_kind() {
        // Daemon-side compaction signal — yellow + ⏳ in the
        // transcript so the user knows a background summarisation
        // ran (typically silent otherwise).  Covered here rather
        // than via live test because triggering compaction needs
        // a low AMAEBI_COMPACTION_THRESHOLD on a daemon restart.
        let mut s = test_state();
        s.streaming = true;
        let _ = handle_response(Response::Compacting, &mut s);
        let last = s.transcript.last().unwrap();
        assert!(matches!(last.kind, LineKind::Compacting));
        assert!(last.text.contains("compacting"));
    }

    #[test]
    fn handle_response_routes_model_switched_updates_state() {
        // Daemon-side `switch_model` tool fires Response::ModelSwitched.
        // The TUI must mirror that to state.model so the next
        // outgoing Request::Chat carries the new value, and the
        // status bar reflects it on the next draw.
        let mut s = test_state();
        s.streaming = true;
        let prev = s.model.clone();
        let _ = handle_response(
            Response::ModelSwitched {
                model: "bedrock/claude-opus-4.7[1m]".to_string(),
            },
            &mut s,
        );
        assert_eq!(s.model, "bedrock/claude-opus-4.7[1m]");
        assert_ne!(s.model, prev);
        let last = s.transcript.last().unwrap();
        assert!(matches!(last.kind, LineKind::System));
        assert!(last.text.contains("model switched"));
    }

    #[test]
    fn handle_response_routes_pane_assigned_to_launch_kind() {
        let mut s = test_state();
        s.pending_claude = Some(PendingClaudeLaunch {
            descriptions: [("t".to_string(), "desc".to_string())]
                .into_iter()
                .collect(),
            launched: Vec::new(),
        });
        let _ = handle_response(
            Response::PaneAssigned {
                tag: "t".into(),
                pane_id: "%41".into(),
                session_id: "sid".into(),
                worktree: None,
                resources: vec!["sim-9900".into()],
            },
            &mut s,
        );
        let last = s.transcript.last().unwrap();
        assert!(matches!(last.kind, LineKind::Launch));
        assert!(last.text.contains("%41"));
    }

    #[test]
    fn plan_tracker_updates_on_text_chunk() {
        // Each Response::Text chunk feeds the parser before the
        // chunk lands in the transcript, so the input box title can
        // show live `[plan N/M done]` while streaming.
        let mut s = test_state();
        s.streaming = true;
        let _ = handle_response(
            Response::Text {
                chunk: "- [x] step 1\n- [ ] step 2\n".into(),
            },
            &mut s,
        );
        assert_eq!(s.plan_tracker.latest_progress(), Some((1, 2)));
    }

    #[test]
    fn plan_tracker_resets_on_done_so_next_turn_starts_clean() {
        // Without the reset, a turn with no checklist would keep
        // showing the previous turn's progress count.
        let mut s = test_state();
        s.streaming = true;
        let _ = handle_response(
            Response::Text {
                chunk: "- [x] a\n- [x] b\n".into(),
            },
            &mut s,
        );
        assert_eq!(s.plan_tracker.latest_progress(), Some((2, 2)));
        let _ = handle_response(Response::Done, &mut s);
        assert_eq!(s.plan_tracker.latest_progress(), None);
    }

    #[test]
    fn cancel_steer_local_ctrlc_path_clears_state_and_flushes() {
        // Ctrl-C-armed cancels: first Ctrl-C already shipped
        // Interrupt, so this just flushes the buffer locally and
        // clears the flags.
        let mut s = test_state();
        s.streaming = true;
        s.steer_pending = true;
        s.steer_source = SteerSource::UserCtrlC;
        let _ = handle_response(
            Response::Text {
                chunk: "queued chunk\n".into(),
            },
            &mut s,
        );
        let _ = cancel_steer_local(&mut s);
        assert!(!s.steer_pending);
        assert_eq!(s.steer_source, SteerSource::Idle);
        assert!(
            s.transcript
                .iter()
                .any(|tl| tl.text.contains("queued chunk")),
            "buffered chunk must be flushed to the transcript on cancel"
        );
    }

    #[test]
    fn cancel_steer_local_waiting_for_input_is_ignored_with_hint() {
        // WaitingForInput-armed cancels: daemon ignores Interrupt
        // (per daemon.rs:5225 — it only accepts a real reply or
        // disconnect).  Empty Enter is therefore a no-op; we leave
        // steer_pending = true and push a hint so the user knows
        // they need to type a real reply.
        let mut s = test_state();
        s.streaming = true;
        s.steer_pending = true;
        s.steer_source = SteerSource::DaemonWaitingForInput;
        let _ = cancel_steer_local(&mut s);
        assert!(s.steer_pending, "WaitingForInput steer must persist");
        assert_eq!(s.steer_source, SteerSource::DaemonWaitingForInput);
        assert!(
            s.transcript
                .last()
                .unwrap()
                .text
                .contains("empty Enter ignored"),
            "user must see a hint explaining why nothing happened"
        );
    }

    // ── line editing ─────────────────────────────────────────────
    // These tests verify the byte-offset-based cursor model: every
    // position is a UTF-8 char boundary, and Left/Right/Backspace/
    // Delete/insert all step by full Unicode scalars (so CJK input
    // moves one glyph at a time, not one byte).  Without these the
    // user sees garbled text or silent panics on CJK input.

    fn make_test_state(text: &str, cursor: usize) -> AppState {
        let mut s = test_state();
        s.input = text.to_string();
        s.input_cursor = cursor;
        s
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn ctrl_key(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
    }

    #[test]
    fn left_right_step_by_unicode_scalar_for_cjk() {
        // 你好 = 6 bytes (3 + 3).  Right at cursor=0 should land at
        // byte 3 (after 你), not byte 1 (which would be inside 你's
        // UTF-8 sequence and panic on slicing).
        let mut s = make_test_state("你好", 0);
        let _ = handle_key(key(KeyCode::Right), &mut s);
        assert_eq!(s.input_cursor, 3, "Right past 你 lands at byte 3");
        let _ = handle_key(key(KeyCode::Right), &mut s);
        assert_eq!(s.input_cursor, 6, "Right past 好 lands at end");
        let _ = handle_key(key(KeyCode::Right), &mut s);
        assert_eq!(s.input_cursor, 6, "Right past end clamps");

        let _ = handle_key(key(KeyCode::Left), &mut s);
        assert_eq!(s.input_cursor, 3);
        let _ = handle_key(key(KeyCode::Left), &mut s);
        assert_eq!(s.input_cursor, 0);
        let _ = handle_key(key(KeyCode::Left), &mut s);
        assert_eq!(s.input_cursor, 0, "Left at 0 clamps");
    }

    #[test]
    fn home_end_jump_to_extremes() {
        let mut s = make_test_state("hello world", 5);
        let _ = handle_key(key(KeyCode::Home), &mut s);
        assert_eq!(s.input_cursor, 0);
        let _ = handle_key(key(KeyCode::End), &mut s);
        assert_eq!(s.input_cursor, "hello world".len());
    }

    #[test]
    fn ctrl_a_and_ctrl_e_jump_like_emacs() {
        let mut s = make_test_state("hello", 3);
        let _ = handle_key(ctrl_key('a'), &mut s);
        assert_eq!(s.input_cursor, 0);
        let _ = handle_key(ctrl_key('e'), &mut s);
        assert_eq!(s.input_cursor, 5);
    }

    #[test]
    fn backspace_and_delete_remove_one_scalar() {
        // Cursor between the two CJK glyphs: backspace eats 你, delete
        // eats 好.
        let mut s = make_test_state("你好", 3);
        let _ = handle_key(key(KeyCode::Backspace), &mut s);
        assert_eq!(s.input, "好");
        assert_eq!(s.input_cursor, 0);

        let mut s = make_test_state("你好", 3);
        let _ = handle_key(key(KeyCode::Delete), &mut s);
        assert_eq!(s.input, "你");
        // Cursor stays at byte 3 — but the string is now only 3 bytes
        // long so cursor==input.len(); that's fine.
        assert_eq!(s.input_cursor, 3);
    }

    #[test]
    fn char_inserts_at_cursor_not_at_end() {
        // The whole point of the cursor-aware editor: typing in the
        // middle inserts there, doesn't append at end.
        let mut s = make_test_state("hello", 2);
        let _ = handle_key(key(KeyCode::Char('X')), &mut s);
        assert_eq!(s.input, "heXllo");
        assert_eq!(s.input_cursor, 3);
    }

    #[test]
    fn ctrl_k_and_ctrl_u_kill_to_line_edges() {
        // Ctrl-K from middle removes everything to the right of the
        // cursor; Ctrl-U removes everything to the left.
        let mut s = make_test_state("hello world", 5);
        let _ = handle_key(ctrl_key('k'), &mut s);
        assert_eq!(s.input, "hello");
        assert_eq!(s.input_cursor, 5);

        let mut s = make_test_state("hello world", 5);
        let _ = handle_key(ctrl_key('u'), &mut s);
        assert_eq!(s.input, " world");
        assert_eq!(s.input_cursor, 0);
    }

    // (cursor render byte-slice contract is now covered by
    // wrapped_cursor_position_single_row above — same shape, but
    // also accounts for multi-row wrap.)

    // ── history navigation ───────────────────────────────────────
    // ↑/↓ walk through `state.history` (oldest-first) starting from
    // the most recent entry.  The first ↑ from a non-history input
    // also captures the in-progress draft so ↓ past the most recent
    // entry restores it — the same shell-style readline contract.

    fn populate(history: &[&str]) -> AppState {
        let mut s = test_state();
        s.history = history.iter().map(|s| s.to_string()).collect();
        s
    }

    #[test]
    fn history_prev_walks_back_from_most_recent() {
        let mut s = populate(&["one", "two", "three"]);
        let _ = handle_key(key(KeyCode::Up), &mut s);
        assert_eq!(s.input, "three", "first ↑ shows the most recent prompt");
        let _ = handle_key(key(KeyCode::Up), &mut s);
        assert_eq!(s.input, "two");
        let _ = handle_key(key(KeyCode::Up), &mut s);
        assert_eq!(s.input, "one");
        let _ = handle_key(key(KeyCode::Up), &mut s);
        assert_eq!(s.input, "one", "↑ at oldest stays put");
    }

    #[test]
    fn history_down_past_newest_restores_draft() {
        let mut s = populate(&["one", "two"]);
        // User had typed a half-formed draft, then started arrowing.
        s.input = "wip".into();
        s.input_cursor = s.input.len();

        let _ = handle_key(key(KeyCode::Up), &mut s);
        assert_eq!(s.input, "two");
        let _ = handle_key(key(KeyCode::Up), &mut s);
        assert_eq!(s.input, "one");
        let _ = handle_key(key(KeyCode::Down), &mut s);
        assert_eq!(s.input, "two");
        let _ = handle_key(key(KeyCode::Down), &mut s);
        assert_eq!(
            s.input, "wip",
            "↓ past most recent must restore the captured draft"
        );
        assert!(s.history_pos.is_none());
    }

    #[test]
    fn history_up_with_empty_history_is_noop() {
        let mut s = populate(&[]);
        s.input = "draft".into();
        s.input_cursor = s.input.len();
        let _ = handle_key(key(KeyCode::Up), &mut s);
        // Nothing to recall — input unchanged, no history mode entered.
        assert_eq!(s.input, "draft");
        assert!(s.history_pos.is_none());
    }

    #[test]
    fn record_submitted_prompt_appends_and_dedups_recent() {
        let mut s = populate(&["one"]);
        s.record_submitted_prompt("two");
        assert_eq!(s.history, vec!["one", "two"]);
        // Submitting the same line back-to-back doesn't grow history.
        s.record_submitted_prompt("two");
        assert_eq!(s.history, vec!["one", "two"]);
        // But submitting a non-duplicate after does grow.
        s.record_submitted_prompt("three");
        assert_eq!(s.history, vec!["one", "two", "three"]);
    }
}
