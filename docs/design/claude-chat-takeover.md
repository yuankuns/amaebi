# /claude chat-takeover — design contract

Status: **agreed 2026-04-30 / 2026-05-02, not yet implemented.**
This document is the frozen contract for a multi-PR refactor (PRs B–F below).
Implementation PRs reference this file instead of re-arguing the decisions.

## Problem

Today `/claude` runs in two halves:

1. **Launch** — `Request::ClaudeLaunch` acquires pane + worktree + resource
   leases, starts `claude` in the pane.
2. **Supervise** — `Request::SupervisePanes` runs a ~1500-line Rust polling
   loop (`handle_supervision_inner` at `src/daemon.rs:2705`) that captures the
   pane, calls an LLM for a `WAIT / STEER / DONE` verdict, parses the verdict
   against `SUPERVISION_VERDICT_SCHEMA`, then sleeps.  Rust counts `K=3`
   drift blocks, folds sustained-WAIT spans, enforces a 10 s idle threshold
   and a 5 min poll ceiling, etc.

In practice an LLM-driven loop out-steered this Rust loop ("指哪打哪" —
point-and-shoot).  The triggering incident was a long Phase-8 chat where the
LLM issued a summary without a tool call; `finish_reason="stop"` returned;
the old supervision concept had no answer and the client treated the next
idle as session-end.  PR #148 patched the client symptom; this document
targets the underlying mismatch: **Rust is trying to own the supervision
loop, and it shouldn't.**

## Principle

> **Rust is the harness; the LLM is the brain.**
> Rust owns pre-ops + post-ops + atomic invariants + IPC + HTTP/tmux
> subprocess.  The LLM owns the loop body — judgement, pacing, summary,
> task-done-or-not.

Rust cannot be trusted with judgement; prompts iterate faster and generalise
better.  The LLM cannot be trusted with invariants; race-safe lease /
worktree / socket / SQLite operations stay in Rust.

Concrete translation: **delete `handle_supervision_inner`.**  `/claude`
becomes a pre-op that feeds one synthesised user turn into the existing
chat agentic loop, which then drives the pane until the LLM calls
`task_done` or the user releases.

## Three layers, strictly separated

| layer | what | lifetime | "exit" means |
|---|---|---|---|
| **chat session** | one `amaebi chat` run + its LLM history | user Ctrl-D / Ctrl-C×2 | LLM context ends |
| **loop (one /claude task)** | the pane/resource/task_lease held for a single `/claude` | `task_done` / `/release` / chat exit | release amaebi's ownership of the pane (pane + `claude` process keep running) |
| **pane / claude process** | tmux pane + Claude Code inside | only user `tmux kill-pane` ends it | never touched by amaebi |

Releasing a task **never** kills the pane, terminates `claude`, or removes
the worktree.  `--clean` on `/release` is opt-in worktree removal, nothing
more.

## Flow

### Pre-op (Rust, synchronous during `/claude`)

`handle_claude_launch` acquires (in order, all-or-nothing):

1. Task notebook lease (`tasks.db`, holder id stable across launch + chat).
2. Pane lease (auto-expanding tmux pool).
3. Resource leases (`resource-state.json`, canonical ordering).
4. Worktree directory + branch (`~/.amaebi/worktrees/<repo>/<tag>-<uuid8>/`).
5. AGENTS.md written into the worktree.
6. `claude` launched into the pane (but **the description is NOT pasted** —
   that becomes the LLM's job so the LLM sees the same text the pane sees).
7. Insert into `HeldResources[conn_id]`.

The daemon then writes `Response::PaneAssigned { tag, pane_id, session_id,
worktree, resources }` per pane, then `Response::Done`.

`Response::PaneAssigned` gains two new fields:

```rust
PaneAssigned {
    tag: String,
    pane_id: String,
    session_id: String,
    worktree: Option<String>,     // NEW
    resources: Vec<String>,       // NEW — resolved names, empty if --resource not passed
}
```

### Handoff (client, one user turn)

Client collects all `PaneAssigned` frames for this `/claude`, then
synthesises **one** user turn and sends it via `Request::Chat` on the same
`session_id`:

```
{desc}

[launched]
  pane: %54
  worktree: /home/you/.amaebi/worktrees/amaebi/kernel-opt-ab12cd34
  resource: xesim-9902
  tag: kernel-opt
```

- Multiple tasks in one `/claude`: one `[launched]` block per pane,
  concatenated, still as one user turn.
- The synthesis happens **client-side** so there's exactly one user turn in
  history (auditable, replayable on resume).  The daemon does not inject
  system text behind the LLM's back.
- This synthetic user turn is also rendered to the terminal output stream,
  so the human sees what the LLM saw.  The user does **not** re-press Enter.

### Loop (LLM, via the existing chat agentic loop)

The LLM uses existing tools: `tmux_capture_pane`, `tmux_send_text`,
`tmux_send_key`, `tmux_wait`, `shell_command`, `read_file`, `edit_file`,
plus **one new lifecycle tool**:

```rust
task_done {
    pane_id: String,   // REQUIRED — no default, avoids ambiguity with multiple tasks
    summary: String,   // LLM's final summary, surfaced to user + inbox
}
```

#### Pane-alive invariant (PROMPT-enforced, NOT Rust-enforced)

The chat system prompt states: while any pane is alive (in
`HeldResources[conn]`, not yet `task_done` / `/release`'d), **every LLM
turn must call at least one tool.**  Pure-text replies are forbidden in
that state.  The LLM's vocabulary for "I want to wait" is `tmux_wait` —
not a text reply saying "let me wait".

Violation — LLM returns `finish_reason=stop` with no tool call while
HeldResources is non-empty — is a prompt bug, fixed in the prompt.  Rust's
agentic loop retains its current terminate-on-no-tool-call behaviour.  No
Rust nudge / counter / timer is added to compensate.  Diagnostic: Rust
MAY log a `warn!` on this transition ("agentic loop exited with pane
%54 still alive"), strictly for debugging — it must not trigger any
daemon action.

This is the single most important principle of this redesign.  Do not
re-add Rust-side nudging without this doc being amended first.

### Post-op (five paths, ONE Rust release function)

| source | trigger | scope | summary | receipt to | chat exits? |
|---|---|---|---|---|---|
| LLM judgement | `task_done(pane, summary)` | that pane | LLM-authored | screen + inbox | no |
| user single pane | `/release %54 [--clean] [--summary "..."]` | that pane | optional | screen + inbox | no |
| user all | `/release all [--clean]` | all under chat | empty | screen + inbox | no |
| chat normal exit | Ctrl-D / Ctrl-C×2 | all under conn | empty | screen + inbox | yes |
| socket abnormal | terminal kill / crash | all under conn | `[abandoned]` | inbox only | yes (passive) |

Plus a **24 h lease TTL** as disaster backstop (daemon crash, lost conn
before cleanup hook, stale `--resume-pane` reconnection window, etc.).

Release for a single pane:

```
pane_lease::release_lease(pane)
  + resource_lease::release_all_for_pane(pane)
  + release_task_leases_for_holder(task_holder)
  + collect pane tail  (tmux capture-pane -p)
  + collect worktree   (git status --short)
  + write inbox entry  (~/.amaebi/inbox/<session>-<tag>.md)
  + remove HeldResources[conn] entry
```

Release does **NOT**:
- `tmux kill-pane`
- `git worktree remove` (unless `--clean`)
- terminate `claude` process

Release is **idempotent** — `task_done` followed by socket-break must not
double-release, and double-release must not panic or corrupt state.

## Resume semantics

A pane in `HeldResources[conn]` is held **exclusively by that conn**.  When
the conn drops (chat exit or socket break) Rust releases all entries under
that conn.  On the next `amaebi chat -r <sid>` the new conn's
`HeldResources` starts empty — the previous pane is no longer amaebi-owned.

To re-claim it, the user runs `/claude --resume-pane %54` in the new chat.
This goes through the normal pre-op path: acquire pane + resource + task
leases afresh, insert into the new conn's `HeldResources`, synthesise a
`[launched]` user turn.  The pane + `claude` process were never killed, so
conversation state inside the pane is preserved.

Grace period for resume is covered by the 24 h lease TTL.  No daemon-side
conn buffering, no "grab-back" semantics, no cross-conn ownership transfer.

## IPC surface

### Modify

- `Response::PaneAssigned` — add `worktree: Option<String>`, `resources:
  Vec<String>`.

### Add

```rust
enum ClaudeReleaseTarget {
    Pane(String),   // "%54"
    All,
}

Request::ClaudeRelease {
    target: ClaudeReleaseTarget,
    clean_worktree: bool,
    summary: Option<String>,
}

Response::TaskReleased {
    pane_id: String,
    resources_freed: Vec<String>,
    tag: Option<String>,
    summary: Option<String>,
    worktree_path: Option<String>,
    worktree_dirty: bool,
    pane_tail: String,
}
```

### Delete (PR F)

- `Request::SupervisePanes`, `SupervisionTarget`, `Response::Heartbeat`.
- `handle_supervision`, `handle_supervision_inner` (~1500 lines).
- `SUPERVISION_VERDICT_SCHEMA`, `SUPERVISION_SYSTEM_PROMPT`,
  `parse_supervision_verdict`, `SupervisionParsed`.
- `render_event_stream_history`, `capture_git_evidence`,
  `build_notebook_context`.
- `trigger_hard_boundary`, `SUPERVISION_DRIFT_BLOCK_K`.
- `tasks.rs` verdict layer: `VerdictRecord`, `append_verdict_record`,
  `all_verdict_records`, `kind='verdict'` rows.  Notebook text itself
  stays — the LLM can still `shell_command "sqlite3 ..."` to read it.

## Daemon state

```rust
struct DaemonState {
    // ...existing fields...
    held: Mutex<HashMap<ConnId, Vec<TaskEntry>>>,
}

struct TaskEntry {
    pane_id: String,
    resources: Vec<String>,
    worktree: Option<PathBuf>,
    tag: Option<String>,
    task_lease_holder: Option<String>,
    created_at: Instant,
    declared_done_at: Option<Instant>,
}
```

Keyed by **ConnId, not session_id.**  Socket break == release even though
the session_id might be resumable later.  The `--resume-pane` rare case is
covered by the 24 h TTL, not by daemon-side conn buffering.

## Client slash commands (PR E)

- `/release %54 [--clean] [--summary "..."]` — one pane.
- `/release all [--clean]` — every pane under this chat conn.

No `tag=` granularity (not needed yet).  No cross-session release (a user
in chat A cannot release a pane held by chat B — they're different conns).

## Decided trade-offs (do not relitigate without new evidence)

1. **Pre-op delivery**: client synthesises `[launched]` block into one user
   message; not a separate prompt, not an auto-injected system text.
   `/claude` description and resource facts travel together as one turn so
   history is replay-safe.
2. **Pre-op does NOT paste prompt into pane.**  The LLM pastes via
   `tmux_send_text`.  Consequence: the LLM sees the same text the pane sees
   — useful when diagnosing transcription issues.
3. **`task_done` requires `pane_id` explicitly**, no default.  Single-pane
   chats and multi-pane chats use the same call shape.
4. **LLM cannot call `release_*` directly.**  `task_done` is the only
   lifecycle tool it sees; it **signals** release, Rust executes.
   `/release` goes through the same Rust release function.  Idempotent.
5. **Socket break → immediate release.**  Grace for `--resume-pane`
   handled by 24 h TTL, not conn buffering.
6. **Worktree preserved by default**; `--clean` opt-in.
7. **`task_done` / `/release` / chat exit / socket break / TTL all funnel
   to the same Rust release function** — no per-source policy branches.
8. **Pane-alive invariant is enforced by prompt, not Rust.**  If the LLM
   exits the agentic loop with pane alive, that's a prompt bug; the pane
   sits until user `/release` or 24 h TTL.  Rust does NOT nudge.
9. **No parallel-path / two-week validation period.**  The old
   `Request::SupervisePanes` path has exactly one caller (amaebi's own
   client); once the client stops sending it, nothing uses it.  PR F
   removes the code in the same sequence.

## PR sequence

| PR | scope | blocks | status |
|---|---|---|---|
| **A** | this design doc | B, C | **this PR** |
| **B** | `Response::PaneAssigned` gains `worktree` + `resources`. Daemon populates them; client ignores them for now (field-expansion only). | F | blocked by A |
| **C** | `task_done` tool + `Request::ClaudeRelease` + `Response::TaskReleased` + `HeldResources` table + socket-disconnect hook + idempotent release fn. `handle_supervision` cleanup path also routes through the new release fn (double-release safe). Does not change the client cutover. | D, E | blocked by A |
| **D** | chat system prompt: pane-alive invariant. | F | blocked by C |
| **E** | client `/release %pane` + `/release all`. | – | blocked by C |
| **F** | **atomic cutover + delete.**  In one PR: client synthesises `[launched]` and sends `Request::Chat` instead of `Request::SupervisePanes`; delete `handle_supervision*` + `Request::SupervisePanes` + verdict schema + verdict layer in `tasks.rs`.  Big minus diff. | – | blocked by B, C, D |

B and C are parallelisable after A.  D depends on C (prompt must reference
the `task_done` tool).  E depends on C (needs `HeldResources` and
`Request::ClaudeRelease`).

**Why F is atomic instead of split:** if the client cutover (stop sending
`SupervisePanes`) lands before `HeldResources` + socket-break hook is live,
panes leak until the 24 h TTL because the old `handle_supervision` cleanup
path stops running and no new path has replaced it.  F does the cutover
and the deletion in one commit so the release path is always live.

## Anti-patterns (non-exhaustive)

- Rust counters that trigger escalation (`SUPERVISION_DRIFT_BLOCK_K = 3`
  is the canonical example to avoid).
- Rust regex / keyword matching over pane content.
- Rust pre-computing "is this drift" / "is this done".
- Rust timers as policy (`poll every 5 min`, `idle threshold 10 s`).  LLM
  calls `tmux_wait` / `tmux_capture_pane` when it wants to.
- Rust folding / summarising verdict history before the LLM sees it.
- Rust nudging the LLM back into the loop when it returns without a tool
  call.  The prompt forbids pure-text turns while pane-alive; a violation
  is a prompt bug.

## Why this is the right direction

User validated LLM-driven supervision feels 指哪打哪 — steering lands
exactly where intended.  Rust policy ages poorly, is rigid, hard to tune.
Prompts iterate faster and generalise better.  Rust cannot be trusted with
judgement; only with invariants.

The observation that triggered this redesign: a long Phase-8 chat ended
because the LLM issued a long text summary with no tool call,
`finish_reason="stop"` returned, and the old (Rust-owned) supervision
concept had no answer — the client treated the next idle as session-end
and exited.  PR #148 fixed the client symptom; this redesign targets the
architectural mismatch that made the symptom possible.
