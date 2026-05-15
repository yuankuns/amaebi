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
//!
//! Still on the to-do list (will land in subsequent commits on this
//! same branch):
//!
//! - `/release` (recognised but parked behind a "not yet wired"
//!   message; classic chat covers it for now).
//! - Ctrl-C mid-generation steer + steer buffering.
//! - `--resume` (UUID is accepted but treated like a new session).
//! - Plan progress indicator (`[plan N/M done]`).
//! - Tool-notice colour styling (`ToolUse`, `Compacting`,
//!   `WaitingForInput` are currently rendered as plain `[…]` lines).
//! - Markdown rendering — assistant text is emitted verbatim.
//! - TUI-internal scrollback (PgUp / mouse wheel); we always follow
//!   the tail of the transcript.

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

/// Public entry — called from `main.rs` when `--tui` is set.
///
/// Mirrors the `run_chat_loop` signature so the two paths are
/// interchangeable at the call site.  `resumed_session_id` is accepted
/// but not yet used by the TUI path (Step 1 limitation; see module
/// docstring).
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

/// Where a transcript line originated from — controls colour / prefix.
#[derive(Clone, Copy)]
enum LineKind {
    /// Anything amaebi wants to show that isn't model output — greeting,
    /// `[tool] ...` notices, compaction banners, etc.
    System,
    /// A user prompt we just submitted.  Rendered in a distinct colour
    /// so the transcript is readable when scrolling.
    User,
    /// Streaming assistant reply.  `is_open` means the last chunk did
    /// not end in a newline, so the next text delta should continue
    /// this line rather than start a new one.
    Assistant { is_open: bool },
    /// Hard errors from the daemon or protocol.
    Error,
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
    /// / `ToolUse` / `Compacting` / `WaitingForInput` frames are
    /// buffered into `steer_buffer` instead of going to the
    /// transcript so they don't fight the steer prompt for screen
    /// real estate.  Reserved for the upcoming Ctrl-C-steer commit
    /// on this branch; not yet wired into handle_response.
    #[allow(dead_code)]
    steer_pending: bool,
    /// Frames received while `steer_pending`, replayed through
    /// `handle_response` once steering ends.  Capped at
    /// `STEER_BUFFER_MAX_FRAMES` (oldest-first eviction); when an
    /// eviction has happened the next flush prepends a truncation
    /// notice so the user knows some output was dropped.  Reserved
    /// for the upcoming Ctrl-C-steer commit on this branch.
    #[allow(dead_code)]
    steer_buffer: Vec<Response>,
    #[allow(dead_code)]
    steer_buffer_truncated: bool,
    /// Timestamp of the last Ctrl-C press, used to detect the
    /// double-Ctrl-C-within-window exit gesture.  Cleared every time
    /// we leave the steer/exit-pending state.  Reserved for the
    /// upcoming Ctrl-C-steer commit on this branch.
    #[allow(dead_code)]
    last_ctrl_c: Option<std::time::Instant>,
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
            steer_buffer: Vec::new(),
            steer_buffer_truncated: false,
            last_ctrl_c: None,
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

    fn push_system_line(&mut self, text: String) {
        self.transcript.push(TranscriptLine {
            kind: LineKind::System,
            text,
        });
    }

    fn push_user_line(&mut self, text: String) {
        self.transcript.push(TranscriptLine {
            kind: LineKind::User,
            text,
        });
    }

    fn push_error_line(&mut self, text: String) {
        self.transcript.push(TranscriptLine {
            kind: LineKind::Error,
            text,
        });
    }

    /// Append an assistant text chunk, continuing the previous
    /// assistant line if it was left "open" (no trailing newline).
    fn push_assistant_chunk(&mut self, chunk: &str) {
        // Split off any trailing newlines so we can correctly track
        // which line is still open for the next chunk to append to.
        let mut remaining = chunk;
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

enum KeyOutcome {
    Continue,
    SubmitInput(String),
    Exit,
}

fn handle_key(key: KeyEvent, state: &mut AppState) -> KeyOutcome {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        // Ctrl-C / Ctrl-D exit the TUI.  A future commit on this branch
        // will change Ctrl-C into the steer trigger while a turn is
        // streaming, matching the classic chat semantics.
        KeyCode::Char('c') if ctrl => KeyOutcome::Exit,
        KeyCode::Char('d') if ctrl => KeyOutcome::Exit,
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
            let text = std::mem::take(&mut state.input);
            state.input_cursor = 0;
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
    match resp {
        Response::Text { chunk } => {
            state.push_assistant_chunk(&chunk);
            ResponseOutcome::Continuing
        }
        Response::Done => {
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
            state.close_open_assistant_line();
            state.push_error_line(format!("error: {message}"));
            state.streaming = false;
            // Drop any in-flight /claude launch state on error so
            // a subsequent input doesn't leak it into a synthesised
            // supervision prompt.
            state.pending_claude = None;
            ResponseOutcome::TurnEnded
        }
        Response::ToolUse { name, detail } => {
            state.push_system_line(format!("[{name}] {detail}"));
            ResponseOutcome::Continuing
        }
        Response::Compacting => {
            state.push_system_line("[compacting conversation…]".to_string());
            ResponseOutcome::Continuing
        }
        Response::SteerAck => {
            state.push_system_line("[steer acknowledged]".to_string());
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
            if !prompt.is_empty() {
                state.push_system_line(prompt);
            }
            state.streaming = false;
            ResponseOutcome::TurnEnded
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
            state.push_system_line(format!("[pane {pane_id}] tag={tag}{resources_blurb}"));
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
        other => {
            state.push_system_line(format!("[{other:?}]"));
            ResponseOutcome::Continuing
        }
    }
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

/// What `dispatch_input` decided to do with a freshly-submitted line.
///
/// Pulling the slash-command decision out of the async dispatcher
/// keeps the parser part unit-testable without a real Unix socket.
/// `Claude` and `ReplyReview` carry the parser output verbatim;
/// dispatching them runs async work (tag generation, ClaudeLaunch
/// IPC) which we do directly in `dispatch_input`.
#[derive(Debug, PartialEq, Eq)]
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
    /// `/release` is recognised but not yet ported to --tui.  Tell
    /// the user to fall back rather than silently shipping the
    /// literal text to the daemon (window-6 footgun fixed in #163).
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
        Some(SlashCommand::Release(_)) => InputDispatch::NotYetWired("/release"),
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

            let input_title = if state.streaming {
                " input (streaming… Enter queues for next turn — not yet wired) "
            } else {
                " input (Enter to send, Ctrl-C to exit) "
            };

            // Floor at 3 so an empty input still shows a 1-row cavity;
            // cap at half the frame so a paste-bomb can't consume the
            // whole transcript area.
            let max_input_height = (total_area.height / 2).max(3);
            let input_inner_width = total_area.width.saturating_sub(2);
            let input_segments = char_grid_wrap(&state.input, input_inner_width);
            let input_visual_rows = input_segments.len().max(1) as u16;
            let input_height = (input_visual_rows + 2).clamp(3, max_input_height);

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(0), Constraint::Length(input_height)])
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
            let scroll_y = transcript_total_rows.saturating_sub(transcript_visible_rows);

            let transcript = Paragraph::new(transcript_lines)
                .block(Block::default().borders(Borders::ALL).title(" amaebi "))
                .scroll((scroll_y, 0));
            frame.render_widget(transcript, chunks[0]);

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
            frame.render_widget(input_para, chunks[1]);

            // Cursor position inside the input box: walk the typed-
            // so-far prefix under the same char-grid wrap so the
            // cursor lands exactly where the renderer placed the
            // matching character.
            let inner_width = chunks[1].width.saturating_sub(2);
            let typed_so_far = &state.input[..state.input_cursor.min(state.input.len())];
            let (cursor_row, cursor_col) = wrapped_cursor_position(typed_so_far, inner_width);
            let visible_rows = chunks[1].height.saturating_sub(2);
            let cursor_row = cursor_row.min(visible_rows.saturating_sub(1));
            frame.set_cursor_position((chunks[1].x + 1 + cursor_col, chunks[1].y + 1 + cursor_row));
        })
        .map_err(|e| anyhow::anyhow!("terminal.draw: {e}"))?;
    Ok(())
}

/// Append the visual rows for one transcript entry to `out`, wrapping
/// at `inner_width` columns under a char-grid wrap.  Preserves the
/// entry's `LineKind` styling on every visual row so a wrapped User
/// line stays cyan all the way down, etc.
fn push_wrapped_transcript_line(
    out: &mut Vec<Line<'static>>,
    tl: &TranscriptLine,
    inner_width: u16,
) {
    let style = match tl.kind {
        LineKind::System => Style::default().fg(Color::DarkGray),
        LineKind::User => Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
        LineKind::Assistant { .. } => Style::default(),
        LineKind::Error => Style::default().fg(Color::Red),
    };
    let prefix: &'static str = match tl.kind {
        LineKind::System => "  ",
        LineKind::Error => "! ",
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
    fn classify_input_routes_release_to_not_yet_wired() {
        // /release is recognised but not yet ported.  Must NOT fall
        // through to SendChat (window-6 footgun, #163).
        assert_eq!(
            classify_input("/release %54"),
            InputDispatch::NotYetWired("/release"),
            "/release must not silently send as chat"
        );
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
