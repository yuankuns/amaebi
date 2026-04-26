# amaebi dashboard

Live TUI aggregating pane, session, inbox, cron, and resource state. Read-only
— does not touch the daemon via IPC, does not modify any on-disk file.

```bash
amaebi dashboard
```

Implementation: `src/dashboard.rs`, entry point at `src/dashboard.rs:511`.

## Refresh

- Auto-refresh every 2 seconds (`REFRESH` constant at `src/dashboard.rs:39`).
- Press `r` to force an immediate re-read.

Every tick re-reads the underlying files from scratch: `sessions.json`,
`tmux-state.json`, `resource-state.json`, `inbox.db`, `cron.db`,
`memory.db`. Sources that are missing or fail to open render as empty — the
dashboard works on a fresh install.

## Keys

| Key | Action |
|-----|--------|
| `q` / `Q` | Exit |
| `Esc` | Exit |
| `Ctrl-C` | Exit |
| `r` / `R` | Force re-read |

No mouse interaction. No navigation between panels — it is strictly a
read-only monitor.

## Panels

| Panel | Source | Shows |
|-------|--------|-------|
| Environment | runtime | cwd, git branch, sandbox mode |
| Task summary | aggregated | Pane counts (busy / idle / starting), cron counts, unread inbox |
| Panes | `~/.amaebi/tmux-state.json` | Every known pane with status, tag, and age |
| Sessions | `memory.db` | Newest turn per session UUID (prompt snippet, cap 80 chars) |
| Resources | `~/.amaebi/resource-state.json` + `resources.toml` | Every resource with class, status, holder *(requires PR β merged for the dedicated resource panel; currently resources appear in the Activity stream)* |
| Inbox | `inbox.db` | Recent arrivals (unread first) |
| Cron | `cron.db` | `last_run` events and schedules |
| Activity | unified | Merged event stream, newest first, capped at 20 entries (`ACTIVITY_CAP` at `src/dashboard.rs:40`) |

The activity stream is the most useful panel day-to-day: it merges pane
state changes, session updates, inbox arrivals, and cron `last_run` events
into one time-ordered tail.

## Typical use

- During a long `/claude --tag kernel-opt "…"` run, keep the dashboard open in
  a second pane to watch the supervision verdicts accumulate and see when the
  Activity stream marks the task DONE.
- After a cron tick, check the Inbox counter for new reports.
- While debugging the resource pool, watch the Panes panel to see which pane
  is holding which resource.

## Limitations

- Read-only. You cannot kill panes, release leases, or mark inbox items read
  from the dashboard. Use the corresponding subcommands (`amaebi tag
  release`, `amaebi inbox read`, etc.).
- Best on an 80×24 or larger terminal. The three vertical sections
  (environment, task summary, activity) will clip on very small windows.
- Does not show the daemon's live LLM token stream — supervision output only
  appears in the client that initiated `/claude`.
