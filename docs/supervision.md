# Supervision loop

Every `/claude` launch (see [claude.md](claude.md)) is followed by a
supervision loop in the daemon. The loop runs until the task is done, times
out, or is interrupted. Its job is to read the pane, decide whether the
Claude subprocess is making progress, and nudge it when it isn't.

Implementation: `handle_supervision` / `handle_supervision_inner` in
`src/daemon.rs:2207`.

## WAIT / STEER / DONE

Each iteration the supervision LLM sees the pane contents plus the task
notebook preamble (for `--tag` runs) and must return exactly one of:

| Verdict | Meaning | Effect |
|---------|---------|--------|
| `WAIT` | Still working, check again later | Sleep, re-snapshot on next tick |
| `STEER: <pane_id>: <message>` | Pane is stuck or off-track | `tmux_send_keys` the message into the pane |
| `DONE: <summary>` | Task is complete | Stream the summary to the client, exit |

The prompt that asks for this verdict is built from the pane capture plus the
notebook preamble (`build_notebook_context`, `src/daemon.rs:2305`). The
completion is capped at `MAX_SUPERVISION_TOKENS` — enough for a short verdict,
not enough for a long narrative.

## Timing

All durations are in `handle_supervision_inner`:

| Knob | Default | Env override | Purpose |
|------|---------|--------------|---------|
| Poll-interval ceiling | 5 min | `AMAEBI_SUPERVISION_INTERVAL_SECS` | Maximum wait between LLM calls |
| Idle threshold | 10 s | (compile-time constant `IDLE_SECS`) | Pane must be unchanged this long before the LLM is called |
| Idle poll period | 2 s | (compile-time constant `IDLE_POLL_SECS`) | How often to snapshot the pane while waiting for idle |
| Hard timeout | 10 h | `AMAEBI_SUPERVISION_TIMEOUT_SECS` | Wall-clock ceiling; after this, supervision exits regardless |

The 5-minute ceiling (`src/daemon.rs:2418`) is the *maximum* gap between LLM
calls. Each iteration also waits for 10 seconds of pane stability
(`src/daemon.rs:2431`) before taking a snapshot — this avoids calling the LLM
every 2-5 seconds during active tool output. Net effect: supervision LLM cost
drops ~70-80 % in typical sessions.

The hard timeout (`src/daemon.rs:2437`) protects against runaway tasks. If
supervision is still polling after 10 hours, it gives up, writes a summary,
and releases the pane.

## What STEER does

STEER verdicts are formatted as `STEER: <pane_id>: <message>`. The daemon
parses the pane id, resolves it to a live tmux target, and sends the message
via `tmux send-keys`, including Enter so Claude processes it. The pane sees
the message as if you typed it.

Steering is used when the LLM determines Claude is stuck (waiting at a
confirmation prompt, looping on the same failing command, or chasing a wrong
hypothesis). The supervision prompt is explicitly instructed to prefer WAIT
over STEER when in doubt, to avoid thrashing.

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
