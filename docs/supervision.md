# Supervision loop

Every `/claude` launch (see [claude.md](claude.md)) is followed by a
supervision loop in the daemon. The loop runs until the task is done, times
out, or is interrupted. Its job is to read the pane, decide whether the
Claude subprocess is making progress, and nudge it when it isn't.

Implementation: `handle_supervision` / `handle_supervision_inner` in
`src/daemon.rs:2207`.

## Structured verdict (drift-detection rework)

Each iteration the supervisor LLM sees:

1. Task description pinned at the top
2. Hard constraints from any resource leases the pane holds
3. Task notebook preamble (for `--tag` runs) as an **event stream** — non-WAIT
   rounds kept verbatim, runs of identical-signature WAITs folded into
   `[N ticks] sustained WAIT: …` markers so a 24 h session doesn't drown
   real drift events in hundreds of `reading files` lines
4. **Filesystem ground truth** — per-pane `git diff` / `git status --short`
   (from the pane's worktree) so the supervisor can compare Claude's
   self-narration against what actually changed on disk
5. In-session verdict history + the most recent STEER
6. The current pane snapshots

and returns a single JSON object:

```json
{
  "stated_intent":   "<what Claude claims to be doing, distilled from pane>",
  "observed_action": "<what actually changed on disk this turn>",
  "verdict":         "WAIT" | "STEER" | "DONE",
  "rationale":       "<one sentence why>",
  "steer_message":   "<only present when verdict=STEER>",
  "claude_responded_to_last_steer": true | false | null
}
```

`claude_responded_to_last_steer` judges whether Claude acted on the most
recent prior STEER — reading, thinking, or course-correcting all count
as `true`; ignoring the STEER is `false`; no prior STEER exists yet is
`null`. This field drives the **hard-boundary escalation** described
below.

All five fields land in `~/.amaebi/tasks.db` for every tick, so drift
trajectories can be reconstructed across resumes. Rows written before
this schema existed read back with `verdict="LEGACY"` and the raw string
in `rationale` — no migration required.

Verdict semantics:

| Verdict | Condition | Effect |
|---------|-----------|--------|
| `WAIT` | `stated_intent` and `observed_action` consistent, in scope, progressing | Re-snapshot on next tick |
| `STEER` | Any drift signal: self-narration disagrees with diff, touched files outside task scope, hard-constraint violation, stuck at a prompt | `tmux send-keys` the `steer_message` into the pane |
| `DONE` | Explicit completion signal + diff covers the deliverables + Claude is idle | Stream summary to the client, exit |

Assembly: `build_supervision_user_content` (prompt) and
`parse_supervision_verdict` (response), both in `src/daemon.rs`. An
unparseable response degrades to a WAIT whose rationale carries the raw
text, so a misbehaving model never silently skips a tick. The completion
is capped at `MAX_SUPERVISION_TOKENS` — enough for the JSON object, not
enough for a long narrative.

## Timing

All durations are in `handle_supervision_inner`:

| Knob | Default | Env override | Purpose |
|------|---------|--------------|---------|
| Poll-interval ceiling | 24 h | `AMAEBI_SUPERVISION_INTERVAL_SECS` | Rarely reached; the idle threshold is the real trigger. Kept as an escape hatch for manual rate-limiting. |
| Idle threshold | 10 s | (compile-time constant `IDLE_SECS`) | Pane must be unchanged this long before the LLM is called — the primary pacing knob |
| Idle poll period | 2 s | (compile-time constant `IDLE_POLL_SECS`) | How often to snapshot the pane while waiting for idle |
| Hard timeout | 24 h | `AMAEBI_SUPERVISION_TIMEOUT_SECS` | Wall-clock ceiling; after this, supervision exits regardless.  Matches the pane, resource, and task-notebook lease TTLs (all 24 h) so supervision never outlives the leases it holds. |

The drift-detection rework dropped the prior 5-minute ceiling: Claude
pauses briefly between tool calls even during active work, so a short
ceiling would let the supervisor judge only after Claude was genuinely
stuck — too late to catch drift in flight. Defaulting the ceiling to
the full supervision timeout means a supervision turn fires as soon as
the pane has been stable for `IDLE_SECS` (10 s), which is the real
trigger we want. Effect (catching drift) is explicitly prioritised over
cost (extra LLM calls).

The hard timeout protects against runaway tasks. If supervision is still
polling after 24 hours, it emits a single `[supervision] timeout after …`
message and exits; the normal cleanup path then releases the pane and any
resource/task leases. No DONE summary is produced on this path — those
only come from a `DONE` verdict.

## What STEER does

When the supervisor returns `verdict: "STEER"`, the daemon injects the
`steer_message` into the first supervised pane via `tmux send-keys`,
including Enter so Claude processes it. The pane sees the message as if
you typed it.

Steering is used when the LLM determines Claude is stuck (waiting at a
confirmation prompt, looping on the same failing command, chasing a wrong
hypothesis) or has drifted off the task (touching files outside scope,
silently reinterpreting the goal). The supervision prompt explicitly
prefers STEER over WAIT when in doubt — a wrong STEER costs one
keystroke, a missed STEER costs hours of wrong work.

## Hard boundary: ESC + forced message after K ignored STEERs

A plain STEER only works if Claude reads it. When Claude ignores
`SUPERVISION_DRIFT_BLOCK_K` consecutive STEERs — judged by the
supervisor's `claude_responded_to_last_steer: false` on each turn — the
daemon escalates:

1. **ESC keystroke** (`\x1b`) into the pane via `tmux send-keys`, which
   cancels Claude's current tool call without exiting the TUI.
2. **Forced message** injected as a new user turn, quoting the recent
   ignored STEERs verbatim and re-stating the original task:
   ```
   Stop.  You have ignored the supervisor repeatedly.  Return to the original task.

   Original task: <desc>

   Recent steering you did not act on:
     - <steer 1>
     - <steer 2>
     - <steer 3>

   Re-read the task above and resume work on it now.
   ```
3. A dedicated notebook row is written: `rationale` starts with
   `HARD_BOUNDARY: K=N drift escalation`, so prior-session history
   (event stream) surfaces escalations distinctly.
4. Counter + quoted-STEER FIFO are reset. The forced message is itself
   a STEER; if Claude ignores it too, the counter accumulates fresh.

`K` is a compile-time constant (`SUPERVISION_DRIFT_BLOCK_K` in
`src/daemon.rs`), currently `3`. It is not surfaced as an env var — the
right value depends on how the supervisor calibrates "responded" and is
a tuning knob, not an operator parameter.

The counter resets on any non-STEER verdict, on a STEER marked
`responded=true`, and on a STEER the daemon could not dispatch (tmux
failure — not Claude's fault). Scattered unresponsive STEERs separated
by WAIT therefore never accumulate; only a *consecutive* run trips the
boundary.

The hard boundary does **not** roll back files (no `git checkout --`) or
exit supervision. It is a soft-block: Claude is interrupted and
re-prompted with the original task, but the loop continues so legitimate
follow-up work can resume.

## What DONE does

On DONE, the daemon:

1. Streams the `<summary>` text to the client as `Response::Text` frames.
2. For `--tag` runs, writes a resume hint (`src/daemon.rs:2230`):
   ```
   [supervision] to resume any of these panes:
     pane %41 (tag=kernel-opt)
       continue task:  /claude --tag kernel-opt
       reuse pane:     /claude --resume-pane %41
   ```
3. Writes `Response::Done` and closes the frame stream.
4. Runs the cleanup block (release panes + task leases).

## How to interrupt

- **Ctrl-C twice in `chat`**: the first Ctrl-C interrupts the current response
  (including supervision). The second Ctrl-C exits the `chat` process.
- **Client disconnect**: closing the client socket is treated as an interrupt.
  The daemon drains the current iteration, exits supervision, and runs
  cleanup.
- **`AMAEBI_SUPERVISION_TIMEOUT_SECS`**: hard wall-clock ceiling.

## Release guarantees

Cleanup is **unconditional**. Every exit path — DONE, STEER-failure, timeout,
Ctrl-C interrupt, model error, client disconnect, or an unhandled inner `Err`
— runs the wrapper at `src/daemon.rs:2250`:

```rust
release_supervised_panes(&panes).await;
release_task_leases_for_holder(state, &holder).await;
```

This means:

- tmux pane leases never stay `Busy` past the supervision lifetime. The 24 h
  lease TTL (`src/pane_lease.rs`) is only a safety net for daemon crashes.
- Task notebook leases (`~/.amaebi/tasks.db`) are freed so the next
  `/claude --tag <same-tag>` is not rejected as a duplicate holder.
- Resource leases (`src/resource_lease.rs:59`, `LEASE_TTL_SECS`) are released
  via the pane cleanup — the resource lifetime is tied to the pane that holds
  it.

If the daemon itself crashes mid-supervision, the TTL-based reclaim path takes
over: heartbeats are stale, the lease is marked effective-Idle, and the next
acquirer can take over (see `effective_status` at
`src/resource_lease.rs:159`).
