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
//!
//! Still on the to-do list (will land in subsequent commits on this
//! same branch):
//!
//! - `/claude` / `/release` / `/replyreview` — recognised but emit a
//!   "not yet wired in --tui" message instead of falling through to
//!   chat (which would re-create the window-6 footgun fixed in #163).
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
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Terminal;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use unicode_width::UnicodeWidthStr;

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

    let mut state = AppState::new(session_id.clone(), model.clone());
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
                                if handle_response(resp, &mut state) {
                                    // Turn is done; if we have a pending
                                    // queued submission this is where a
                                    // future step would dispatch it.
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
}

impl AppState {
    fn new(session_id: String, model: String) -> Self {
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

/// Compute the cursor's terminal column inside the input box (relative
/// to the inside-border origin), clamped to the visible width.
///
/// Must use *display width* — terminal columns occupied — rather than
/// `chars().count()` because CJK / fullwidth / emoji glyphs each
/// occupy two columns but only one Unicode scalar.  Without this fix
/// Chinese input parks the cursor halfway through the user's text
/// (observed 2026-05-15).
fn input_cursor_column(input: &str, visible_width: u16) -> u16 {
    UnicodeWidthStr::width(input).min(visible_width as usize) as u16
}

/// Dispatch a single daemon frame.  Returns `true` when the turn has
/// completed (Done / Error), so callers can clear `streaming` and
/// perform any queued work.
fn handle_response(resp: Response, state: &mut AppState) -> bool {
    match resp {
        Response::Text { chunk } => {
            state.push_assistant_chunk(&chunk);
            false
        }
        Response::Done => {
            state.close_open_assistant_line();
            state.streaming = false;
            true
        }
        Response::Error { message } => {
            state.close_open_assistant_line();
            state.push_error_line(format!("error: {message}"));
            state.streaming = false;
            true
        }
        Response::ToolUse { name, detail } => {
            // Step 1 just stringifies these; Step 2 will colour-code
            // them like the classic UI does.
            state.push_system_line(format!("[{name}] {detail}"));
            false
        }
        Response::Compacting => {
            state.push_system_line("[compacting conversation…]".to_string());
            false
        }
        Response::SteerAck => {
            state.push_system_line("[steer acknowledged]".to_string());
            false
        }
        Response::ModelSwitched { model } => {
            // Daemon-side model switch (e.g. the LLM called the
            // `switch_model` tool).  Mirror it locally so the next
            // outgoing Request::Chat carries the new value.  Same
            // behaviour as run_chat_loop's ModelSwitched handler.
            state.push_system_line(format!("[model switched: {} → {}]", state.model, model));
            state.model = model;
            false
        }
        Response::WaitingForInput { prompt } => {
            if !prompt.is_empty() {
                state.push_system_line(prompt);
            }
            state.streaming = false;
            // We treat this as a turn end for Step 1 — the user can
            // just type their reply in the input box.
            true
        }
        // All other variants surface once we expose the features that
        // generate them (detach, memory, PaneAssigned, etc.).
        other => {
            state.push_system_line(format!("[{other:?}]"));
            false
        }
    }
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
/// Pulling the decision out of the side-effect-laden async function
/// keeps it unit-testable without a real Unix socket.  The decision is
/// pure-state-readable: we look at `parse_slash_command(text)` and
/// nothing else.
#[derive(Debug, PartialEq, Eq)]
enum InputDispatch {
    /// `/model` (no arg): show current model in transcript.
    ShowModel,
    /// `/model <name>`: update `state.model` to this name.
    SwitchModel(String),
    /// `/claude` / `/release` / `/replyreview` — recognised but not
    /// yet ported to the TUI flow.  Tell the user to fall back to
    /// classic chat instead of silently sending the literal text as a
    /// chat prompt (window-6 footgun, fixed in #163).
    NotYetWired,
    /// Plain text: send as `Request::Chat` to the daemon.
    SendChat,
}

fn classify_input(text: &str) -> InputDispatch {
    match parse_slash_command(text) {
        Some(SlashCommand::Model(None)) => InputDispatch::ShowModel,
        Some(SlashCommand::Model(Some(name))) => InputDispatch::SwitchModel(name),
        Some(_) => InputDispatch::NotYetWired,
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
        InputDispatch::NotYetWired => {
            state.push_system_line(
                "[error] /claude /release /replyreview are not yet wired in --tui; \
                 fall back to classic chat (no --tui flag) for those commands."
                    .to_string(),
            );
        }
        InputDispatch::SendChat => {
            send_prompt(writer, state, text).await?;
        }
    }
    Ok(())
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
            // Transcript gets all remaining rows; input box pinned at
            // the bottom with a fixed 3-row height (border + one line +
            // border).  `Min(0)` rather than `Min(1)` means the
            // transcript can collapse to zero height on a tiny window
            // without causing layout errors.
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(0), Constraint::Length(3)])
                .split(frame.area());

            let lines: Vec<Line> = state
                .transcript
                .iter()
                .map(|tl| transcript_line_to_ratatui(tl))
                .collect();

            let transcript = Paragraph::new(lines)
                .block(Block::default().borders(Borders::ALL).title(" amaebi "))
                .wrap(Wrap { trim: false })
                .scroll(scroll_offset(&state.transcript, chunks[0].height));

            frame.render_widget(transcript, chunks[0]);

            let input_title = if state.streaming {
                " input (streaming… Enter queues for next turn — not yet wired) "
            } else {
                " input (Enter to send, Ctrl-C to exit) "
            };
            let input = Paragraph::new(Line::from(state.input.as_str()))
                .block(Block::default().borders(Borders::ALL).title(input_title));
            frame.render_widget(input, chunks[1]);

            // Position the terminal cursor inside the input box at the
            // logical cursor position (`input_cursor`), not at the end
            // of the input — that's how Left/Right/Home/End work.
            // `input_cursor_column` measures display width up to that
            // byte offset so CJK glyphs land on column boundaries.
            let visible = chunks[1].width.saturating_sub(2);
            let cursor_col = input_cursor_column(
                &state.input[..state.input_cursor.min(state.input.len())],
                visible,
            );
            frame.set_cursor_position((chunks[1].x + 1 + cursor_col, chunks[1].y + 1));
        })
        .map_err(|e| anyhow::anyhow!("terminal.draw: {e}"))?;
    Ok(())
}

fn transcript_line_to_ratatui(tl: &TranscriptLine) -> Line<'static> {
    let (prefix, style) = match tl.kind {
        LineKind::System => ("  ", Style::default().fg(Color::DarkGray)),
        LineKind::User => (
            "",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        LineKind::Assistant { .. } => ("", Style::default()),
        LineKind::Error => ("! ", Style::default().fg(Color::Red)),
    };
    Line::from(vec![
        Span::raw(prefix),
        Span::styled(tl.text.clone(), style),
    ])
}

/// Follow-tail scroll: if the transcript has more logical lines than
/// fit inside the visible area, shift the viewport so the newest line
/// is on the bottom edge.  Step 1 does no wrap-aware accounting — with
/// `Wrap { trim: false }` a single long line may occupy multiple visual
/// rows, so the clamp is a lower bound on the right answer and can
/// leave a line or two of stale content at the top of a resized view.
/// Proper scroll + PgUp support arrives with Step 5.
fn scroll_offset(transcript: &[TranscriptLine], viewport_height: u16) -> (u16, u16) {
    // Subtract 2 for the borders of the enclosing Block.
    let visible = viewport_height.saturating_sub(2) as usize;
    let overflow = transcript.len().saturating_sub(visible);
    (overflow as u16, 0)
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

    #[test]
    fn assistant_chunks_across_newlines_split_into_distinct_lines() {
        let mut state = AppState::new("sid".into(), "model".into());
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
        let mut state = AppState::new("sid".into(), "model".into());
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
        let mut state = AppState::new("sid".into(), "model".into());
        state.push_assistant_chunk("partial without newline");
        state.streaming = true;

        let turn_ended = handle_response(Response::Done, &mut state);
        assert!(turn_ended);
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

    #[test]
    fn scroll_offset_follows_tail_when_overflowing() {
        // A transcript of 10 lines in a viewport of 5 visible rows
        // (plus 2 for borders → viewport_height=7) must scroll by 5
        // rows so the newest line is at the bottom.
        let tl = TranscriptLine {
            kind: LineKind::System,
            text: "x".into(),
        };
        let transcript: Vec<_> = std::iter::repeat_with(|| TranscriptLine {
            kind: tl.kind,
            text: tl.text.clone(),
        })
        .take(10)
        .collect();
        let (y, x) = scroll_offset(&transcript, 7);
        assert_eq!(x, 0);
        assert_eq!(y, 5);
    }

    #[test]
    fn scroll_offset_zero_when_below_viewport() {
        let tl = TranscriptLine {
            kind: LineKind::System,
            text: "x".into(),
        };
        let transcript = vec![tl];
        let (y, _) = scroll_offset(&transcript, 20);
        assert_eq!(y, 0);
    }

    #[test]
    fn input_cursor_column_uses_display_width_for_cjk() {
        // 你好 = 2 chars, but each CJK glyph occupies 2 terminal
        // columns, so the cursor must land at column 4.  The
        // pre-fix code used chars().count() and would have parked
        // the cursor at column 2, halfway through the user's text.
        assert_eq!(input_cursor_column("你好", 80), 4);
        // Ascii baseline.
        assert_eq!(input_cursor_column("hi", 80), 2);
        // Mixed CJK + ASCII.
        assert_eq!(input_cursor_column("你hi好", 80), 6);
    }

    #[test]
    fn input_cursor_column_clamps_at_visible_width() {
        // A long input must not park the cursor past the right edge
        // of the visible area; clamp at `visible_width`.  Without
        // wrap awareness this is a single-line approximation, but it
        // at least keeps the cursor on screen.
        let s = "你".repeat(50); // 100 columns wide
        assert_eq!(input_cursor_column(&s, 20), 20);
    }

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
    fn classify_input_routes_unported_slashes_to_not_yet_wired() {
        // The window-6 regression (#163) was that slash commands typed
        // mid-turn got shipped to the daemon as plain text.  In
        // --tui we have a similar risk: /claude needs the
        // ClaudeLaunch flow which isn't wired yet, so we must NOT
        // fall through to SendChat.
        for input in [
            "/claude --resource sim-9901 do the thing",
            "/release %54",
            "/replyreview 165",
        ] {
            assert_eq!(
                classify_input(input),
                InputDispatch::NotYetWired,
                "input {input:?} must not silently send as chat"
            );
        }
    }

    #[test]
    fn classify_input_passes_through_plain_text() {
        // Non-slash text is a normal chat message.  Leading slashes
        // that don't match any command (e.g. `/notacommand`) also
        // fall through, matching parse_slash_command's None branch.
        assert_eq!(classify_input("hello world"), InputDispatch::SendChat,);
        assert_eq!(classify_input("/notacommand foo"), InputDispatch::SendChat,);
    }

    // ── line editing ─────────────────────────────────────────────
    // These tests verify the byte-offset-based cursor model: every
    // position is a UTF-8 char boundary, and Left/Right/Backspace/
    // Delete/insert all step by full Unicode scalars (so CJK input
    // moves one glyph at a time, not one byte).  Without these the
    // user sees garbled text or silent panics on CJK input.

    fn make_test_state(text: &str, cursor: usize) -> AppState {
        let mut s = AppState::new("sid".into(), "model".into());
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

    #[test]
    fn input_cursor_column_uses_byte_slice_in_render() {
        // Render-side regression: the cursor column must measure
        // display width of `input[..cursor]`, not of the whole input.
        // For "你hi好" with cursor in the middle (byte 5, after "你hi"),
        // display width is 2 (你) + 2 (hi) = 4 columns.
        assert_eq!(input_cursor_column("你hi", 80), 4);
    }

    // ── history navigation ───────────────────────────────────────
    // ↑/↓ walk through `state.history` (oldest-first) starting from
    // the most recent entry.  The first ↑ from a non-history input
    // also captures the in-progress draft so ↓ past the most recent
    // entry restores it — the same shell-style readline contract.

    fn populate(history: &[&str]) -> AppState {
        let mut s = AppState::new("sid".into(), "model".into());
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
