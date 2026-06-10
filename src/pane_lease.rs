//! Pane lease management for tmux-based parallel Claude sessions.
//!
//! Provides atomic acquisition, release, and auto-expansion of tmux panes so
//! that multiple `amaebi` instances launched via `/claude` cannot steal the
//! same pane from each other.
//!
//! ## State file
//!
//! Pane state is persisted in `~/.amaebi/tmux-state.json` (a `HashMap<pane_id,
//! PaneLease>`) protected by an exclusive `flock` on
//! `~/.amaebi/tmux-state.lock`.  The JSON format is intentional: the file is a
//! runtime/operational state store similar to `sessions.json`, where multiple
//! CLI processes may need to check the lock quickly.
//!
//! ## Thread / process safety
//!
//! Every mutating operation acquires `LOCK_EX` for the duration of the
//! read-modify-write cycle.  [`ensure_idle_panes`] holds the lock for the
//! entire expansion loop (including any `tmux new-window` calls) so two
//! concurrent `ClaudeLaunch` requests never create duplicate panes.
//!
//! All public functions are **synchronous** and must be called from inside
//! `tokio::task::spawn_blocking` when used from async code.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use anyhow::{Context, Result};
use fs2::FileExt;
use serde::{Deserialize, Serialize};

use crate::auth::amaebi_home;
use crate::ipc::AgentKind;

/// Maximum total number of panes (Idle + Busy) that the daemon will manage.
pub const MAX_PANES: usize = 16;

/// Seconds after which a Busy pane whose heartbeat has not been refreshed is
/// treated as Idle (allows dead processes to release their pane automatically).
///
/// Set to 24 hours so long-running `claude` sessions are not mistakenly
/// reclaimed.  Proper heartbeat support (periodic refresh while a lease is
/// active) is a future enhancement; until then TTL serves only as a
/// crash-recovery safety net.
pub const LEASE_TTL_SECS: u64 = 86_400;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PaneStatus {
    Idle,
    Busy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaneLease {
    /// tmux pane ID, e.g. `"%3"`.
    pub pane_id: String,
    /// tmux window ID that contains this pane, e.g. `"@2"`.
    pub window_id: String,
    pub status: PaneStatus,
    /// User-supplied task label (e.g. `"pr-123"`).
    pub tag: Option<String>,
    /// amaebi session UUID assigned when the lease was acquired.
    pub session_id: Option<String>,
    /// Absolute path to the git worktree for this task (uniqueness key).
    pub worktree: Option<String>,
    /// Unix timestamp of the last heartbeat (or acquisition time).
    pub heartbeat_at: u64,
    /// Whether `claude` has been started in this pane.  Idle panes with
    /// `has_claude = true` are preferred over blank panes when assigning tasks:
    /// the scheduler injects just the prompt rather than launching a new session.
    #[serde(default)]
    pub has_claude: bool,
    /// Agent runtime currently active in this pane.  Older state files only
    /// have `has_claude`; those deserialize with `active_agent = None` and are
    /// treated as Claude Code when `has_claude` is true.
    #[serde(default)]
    pub active_agent: Option<AgentKind>,
    /// Full task description last injected into this pane, persisted so
    /// `/claude --resume-pane <pid>` or `/codex --resume-pane <pid>` can
    /// reuse it without the user retyping.
    /// Preserved by `release_lease` (alongside `worktree` / `has_claude`) so
    /// the pane remembers what it was working on across supervision exits.
    #[serde(default)]
    pub task_description: Option<String>,
}

impl PaneLease {
    pub fn new_idle(pane_id: String, window_id: String) -> Self {
        Self {
            pane_id,
            window_id,
            status: PaneStatus::Idle,
            tag: None,
            session_id: None,
            worktree: None,
            heartbeat_at: now_secs(),
            has_claude: false,
            active_agent: None,
            task_description: None,
        }
    }

    /// Returns the *effective* status, treating expired Busy leases as Idle.
    pub fn effective_status(&self) -> PaneStatus {
        if self.status == PaneStatus::Busy
            && now_secs().saturating_sub(self.heartbeat_at) > LEASE_TTL_SECS
        {
            PaneStatus::Idle
        } else {
            self.status.clone()
        }
    }

    pub fn effective_agent(&self) -> Option<AgentKind> {
        self.active_agent
            .or_else(|| self.has_claude.then_some(AgentKind::ClaudeCode))
    }
}

/// Keyed by `pane_id`.
pub type PaneState = HashMap<String, PaneLease>;

// ---------------------------------------------------------------------------
// Error type for capacity violations
// ---------------------------------------------------------------------------

/// Returned by [`ensure_idle_panes`] when adding the requested panes would
/// push the total past [`MAX_PANES`].
#[derive(Debug)]
pub struct CapacityError {
    pub requested: usize,
    pub max_panes: usize,
    pub current_busy: usize,
}

impl std::fmt::Display for CapacityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "capacity limit reached: max_panes={}, busy={}, requested={}; \
             free existing panes to continue",
            self.max_panes, self.current_busy, self.requested
        )
    }
}

impl std::error::Error for CapacityError {}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn state_path() -> Result<PathBuf> {
    Ok(amaebi_home()?.join("tmux-state.json"))
}

fn lock_path() -> Result<PathBuf> {
    Ok(amaebi_home()?.join("tmux-state.lock"))
}

fn open_lock_file() -> Result<File> {
    let dir = amaebi_home()?;
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating directory {}", dir.display()))?;
    let path = lock_path()?;
    let mut opts = OpenOptions::new();
    opts.create(true).truncate(false).write(true);
    #[cfg(unix)]
    opts.mode(0o600);
    opts.open(&path)
        .with_context(|| format!("opening lock file {}", path.display()))
}

// ---------------------------------------------------------------------------
// Raw (unlocked) state I/O
// ---------------------------------------------------------------------------

fn read_state_unlocked() -> Result<PaneState> {
    let path = state_path()?;
    if !path.exists() {
        return Ok(PaneState::new());
    }
    let contents =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    if contents.trim().is_empty() {
        return Ok(PaneState::new());
    }
    // Tolerate corrupt JSON: reset to empty state rather than hard-erroring.
    // A corrupt file is treated the same as a missing one; persisted lease
    // state is discarded and rebuilt through normal pane creation / lease
    // updates going forward.
    Ok(serde_json::from_str(&contents).unwrap_or_default())
}

fn write_state_unlocked(state: &PaneState) -> Result<()> {
    let path = state_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating directory {}", parent.display()))?;
    }
    let contents = serde_json::to_string_pretty(state)?;

    // Atomic write: write to a temp file then rename so readers never see a
    // partially-written file.  Restrict permissions to 0o600 so session IDs
    // and worktree paths are not world-readable.
    let tmp_path = path.with_extension("tmp");
    std::fs::write(&tmp_path, &contents)
        .with_context(|| format!("writing tmp {}", tmp_path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp_path, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("setting 0o600 permissions on {}", tmp_path.display()))?;
    }
    std::fs::rename(&tmp_path, &path)
        .with_context(|| format!("renaming {} → {}", tmp_path.display(), path.display()))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Read the current pane state (acquires a shared lock, read-only).
///
/// Used by `amaebi pane list` (P1) and for heartbeat monitoring.
#[allow(dead_code)]
pub fn read_state() -> Result<PaneState> {
    let lock = open_lock_file()?;
    lock.lock_shared()
        .context("acquiring shared flock for read_state")?;
    let state = read_state_unlocked()?;
    lock.unlock().context("releasing flock after read_state")?;
    Ok(state)
}

/// Acquire a lease on the **first available idle pane** for the given task.
///
/// Returns `(pane_id, had_agent)`. If `worktree` is provided it is checked
/// against all currently Busy panes for uniqueness. Prefer
/// [`ensure_and_acquire_idle`] for production use to avoid TOCTOU races.
#[allow(dead_code)]
pub fn acquire_first_idle(
    tag: &str,
    session_id: &str,
    worktree: Option<&str>,
) -> Result<(String, bool)> {
    acquire_first_idle_for_agent(AgentKind::ClaudeCode, tag, session_id, worktree)
}

pub fn acquire_first_idle_for_agent(
    agent: AgentKind,
    tag: &str,
    session_id: &str,
    worktree: Option<&str>,
) -> Result<(String, bool)> {
    let lock = open_lock_file()?;
    lock.lock_exclusive()
        .context("acquiring flock for acquire_first_idle")?;

    let result = acquire_first_idle_locked(agent, tag, session_id, worktree);

    lock.unlock()
        .context("releasing flock after acquire_first_idle")?;
    result
}

/// Returns `(pane_id, had_agent)` where `had_agent` indicates whether the
/// requested agent was already running in the pane. Callers use this to decide
/// whether to inject only the prompt (`had_agent = true`) or to launch a fresh
/// agent session (`had_agent = false`).
///
/// Priority: idle panes with a matching active agent are preferred so existing
/// Claude/Codex sessions absorb new tasks before blank panes are used.
fn acquire_first_idle_locked(
    agent: AgentKind,
    tag: &str,
    session_id: &str,
    worktree: Option<&str>,
) -> Result<(String, bool)> {
    let mut state = read_state_unlocked()?;

    // Worktree uniqueness check.
    if let Some(wt) = worktree {
        for (pid, l) in &state {
            if l.effective_status() == PaneStatus::Busy && l.worktree.as_deref() == Some(wt) {
                anyhow::bail!(
                    "worktree {} already held by task {:?} on pane {}",
                    wt,
                    l.tag,
                    pid
                );
            }
        }
    }

    // Priority order for pane selection:
    //
    // 1. Idle pane with the requested agent running in the *same* worktree → safe to
    //    inject a prompt directly (after /compact).
    // 2. Idle pane with no agent running (blank shell) → fresh launch.
    // 3. Idle pane with an agent running in a *different* worktree → skip.
    //    Sending shell commands to a pane where an agent is already intercepting
    //    input would deliver them as chat messages, not shell commands.  Leave
    //    those panes alone and let auto-expansion create a new blank one.
    // Use state.iter() so the HashMap key is carried through the selection,
    // avoiding a second get_mut lookup (and the unwrap it would require).
    // Tier-1 reuse requires a known, non-None worktree to match against.
    // When worktree is None (auto-creation failed), None==None would match any
    // pane with worktree=None, injecting a prompt into an arbitrary agent
    // session with unknown directory context.  Guard with worktree.is_some().
    let pane_id = state
        .iter()
        .find(|(_, l)| {
            l.effective_status() == PaneStatus::Idle
                && l.has_claude
                && l.effective_agent() == Some(agent)
                && worktree.is_some()
                && l.worktree.as_deref() == worktree
        })
        .or_else(|| {
            state
                .iter()
                .find(|(_, l)| l.effective_status() == PaneStatus::Idle && !l.has_claude)
        })
        .map(|(k, _)| k.clone())
        .ok_or_else(|| anyhow::anyhow!("no idle panes available"))?;

    let lease = state
        .get_mut(&pane_id)
        .ok_or_else(|| anyhow::anyhow!("pane {pane_id} disappeared after selection"))?;
    // `had_agent` is only true when the pane has a known matching worktree.
    // worktree.is_some() prevents None==None from triggering tier-1 reuse.
    let had_agent = lease.effective_agent() == Some(agent)
        && worktree.is_some()
        && lease.worktree.as_deref() == worktree;
    lease.status = PaneStatus::Busy;
    lease.tag = Some(tag.to_string());
    lease.session_id = Some(session_id.to_string());
    lease.worktree = worktree.map(str::to_string);
    lease.heartbeat_at = now_secs();

    write_state_unlocked(&state)?;
    Ok((pane_id, had_agent))
}

/// Acquire a lease on a **specific pane** by ID.
///
/// Returns `Err` if the pane is already Busy (and not expired), or if the
/// `worktree` conflicts with another Busy pane.  Used by
/// `/claude --resume-pane <pane_id>` to target a specific pane instead of
/// letting the scheduler pick one.
///
/// `worktree = None` means "do not change the existing worktree field".
/// Passing `Some(_)` overwrites it.  This matters for the
/// `Request::ReservePane` pre-flight (used by agent distillation):
/// the client doesn't know the resume-pane's worktree, and a previous
/// `release_lease` deliberately preserved the field so resume-pane could
/// find the worktree later.  An unconditional overwrite would clear that
/// hint and break the subsequent `ClaudeLaunch` resume-pane probe.
#[allow(dead_code)]
pub fn acquire_lease(
    pane_id: &str,
    tag: &str,
    session_id: &str,
    worktree: Option<&str>,
) -> Result<()> {
    acquire_lease_inner(None, pane_id, tag, session_id, worktree)
}

/// Acquire a specific pane only when it already hosts the requested agent.
///
/// Used by `ReservePane` for `--resume-pane` pre-flight so a `/codex` launch
/// cannot temporarily lock a Claude pane, or vice versa, during distillation.
pub fn acquire_lease_for_agent(
    agent: AgentKind,
    pane_id: &str,
    tag: &str,
    session_id: &str,
    worktree: Option<&str>,
) -> Result<()> {
    acquire_lease_inner(Some(agent), pane_id, tag, session_id, worktree)
}

fn acquire_lease_inner(
    required_agent: Option<AgentKind>,
    pane_id: &str,
    tag: &str,
    session_id: &str,
    worktree: Option<&str>,
) -> Result<()> {
    let lock = open_lock_file()?;
    lock.lock_exclusive()
        .context("acquiring flock for acquire_lease")?;

    let result = (|| {
        let mut state = read_state_unlocked()?;

        let lease = state
            .get(pane_id)
            .ok_or_else(|| anyhow::anyhow!("pane {pane_id} not found in state"))?;

        if lease.effective_status() == PaneStatus::Busy {
            anyhow::bail!("pane {} is busy (task: {:?})", pane_id, lease.tag);
        }

        if let Some(agent) = required_agent {
            if lease.effective_agent() != Some(agent) {
                anyhow::bail!(
                    "pane {} is not marked as having `{}` started; drop --resume-pane and let the scheduler pick or start a new pane",
                    pane_id,
                    agent.command()
                );
            }
        }

        // Worktree uniqueness check.
        if let Some(wt) = worktree {
            for (pid, l) in &state {
                if pid != pane_id
                    && l.effective_status() == PaneStatus::Busy
                    && l.worktree.as_deref() == Some(wt)
                {
                    anyhow::bail!(
                        "worktree {} already held by task {:?} on pane {}",
                        wt,
                        l.tag,
                        pid
                    );
                }
            }
        }

        let lease = state.get_mut(pane_id).unwrap();
        lease.status = PaneStatus::Busy;
        lease.tag = Some(tag.to_string());
        lease.session_id = Some(session_id.to_string());
        if let Some(wt) = worktree {
            lease.worktree = Some(wt.to_string());
        }
        lease.heartbeat_at = now_secs();

        write_state_unlocked(&state)?;
        Ok(())
    })();

    lock.unlock()
        .context("releasing flock after acquire_lease")?;
    result
}

/// Release the lease on a pane, marking it as Idle and clearing task metadata.
#[allow(dead_code)]
pub fn release_lease(pane_id: &str) -> Result<()> {
    let lock = open_lock_file()?;
    lock.lock_exclusive()
        .context("acquiring flock for release_lease")?;

    let result = (|| {
        let mut state = read_state_unlocked()?;
        if let Some(lease) = state.get_mut(pane_id) {
            lease.status = PaneStatus::Idle;
            lease.tag = None;
            lease.session_id = None;
            // Intentionally keep `worktree`, `has_claude`, and
            // `task_description` so `/claude --resume-pane <pid>` or
            // `/codex --resume-pane <pid>` can re-acquire the pane and
            // continue the same work with `/compact + original
            // description` (tier-1 reuse) when those fields still describe the
            // current pane state.  Depending on why the lease was released
            // (e.g. early startup or tmux injection failures), these fields may
            // be absent or stale rather than proving that the agent is still
            // running on the same task — callers of resume-pane must treat
            // them as hints and validate before reuse.  They are cleared only
            // when the pane is destroyed or explicitly reset to a blank shell.
            lease.heartbeat_at = now_secs();
        }
        write_state_unlocked(&state)?;
        Ok(())
    })();

    lock.unlock()
        .context("releasing flock after release_lease")?;
    result
}

/// Remove a pane entry entirely from the lease map.
///
/// Used when a tmux pane no longer exists (e.g. "unknown pane" error from
/// `tmux send-keys`).  Unlike [`release_lease`], this removes the entry so
/// the scheduler does not keep selecting the same stale pane.
pub fn remove_pane(pane_id: &str) -> Result<()> {
    let lock = open_lock_file()?;
    lock.lock_exclusive()
        .context("acquiring flock for remove_pane")?;

    let result = (|| {
        let mut state = read_state_unlocked()?;
        state.remove(pane_id);
        write_state_unlocked(&state)
    })();

    lock.unlock().context("releasing flock after remove_pane")?;
    result
}

/// Refresh the heartbeat timestamp for a Busy pane to prevent TTL expiry.
#[allow(dead_code)]
pub fn heartbeat(pane_id: &str) -> Result<()> {
    let lock = open_lock_file()?;
    lock.lock_exclusive()
        .context("acquiring flock for heartbeat")?;

    let result = (|| {
        let mut state = read_state_unlocked()?;
        if let Some(lease) = state.get_mut(pane_id) {
            lease.heartbeat_at = now_secs();
        }
        write_state_unlocked(&state)?;
        Ok(())
    })();

    lock.unlock().context("releasing flock after heartbeat")?;
    result
}

/// Rename a tmux pane's title.
///
/// Uses `tmux select-pane -t <pane_id> -T <title>`.  Returns `Err` when tmux
/// is not running or returns a non-zero exit code.  Callers may choose to
/// ignore the error in non-tmux environments.
pub fn rename_pane(pane_id: &str, title: &str) -> Result<()> {
    let output = std::process::Command::new("tmux")
        .args(["select-pane", "-t", pane_id, "-T", title])
        .output()
        .context("spawning tmux select-pane")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("tmux select-pane -T failed: {stderr}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Auto-expansion
// ---------------------------------------------------------------------------

/// Ensure at least `needed` Idle panes exist.
///
/// Prefer [`ensure_and_acquire_idle`] for single-request use to avoid a
/// TOCTOU race.  This function is retained for bulk pre-warming use cases.
///
/// If the current number of Idle panes is insufficient, this function creates
/// new tmux windows to host new panes (up to [`MAX_PANES`] total).
/// The entire operation runs inside a single `LOCK_EX`, so concurrent calls
/// never create duplicate panes.
///
/// **Priority order for window creation:**
/// 1. Sessions owning windows with *no* Busy panes (least disruptive).
/// 2. Any remaining window's session (contains at least one Busy pane).
/// 3. If total + deficit would exceed `MAX_PANES` → `Err(CapacityError)`.
///
/// # Errors
///
/// - [`CapacityError`] (as a boxed `anyhow::Error`) when capacity would be
///   exceeded.
/// - `anyhow::Error` with `"not in a tmux session"` when `tmux list-windows`
///   fails (no `$TMUX` set).
/// - `anyhow::Error` when all `tmux new-window` attempts fail.
#[allow(dead_code)]
pub fn ensure_idle_panes(needed: usize) -> Result<()> {
    if needed == 0 {
        return Ok(());
    }

    let lock = open_lock_file()?;
    lock.lock_exclusive()
        .context("acquiring flock for ensure_idle_panes")?;

    let result = ensure_idle_panes_locked(needed);

    lock.unlock()
        .context("releasing flock after ensure_idle_panes")?;
    result
}

/// Read the pane state, drop records whose `pane_id` is no longer in
/// tmux, and persist the cleaned state if anything was removed.  Used as
/// the first step of any capacity / availability decision so downstream
/// math is never inflated by panes the user closed behind amaebi's back.
///
/// Tmux failure (not installed, no server, transient error) degrades to
/// "keep existing ledger": we'd rather preserve a stale ledger than
/// wipe it based on an empty / wrong list.  Callers see the pre-fix
/// behavior on that edge, which is strictly no-worse than the old code.
///
/// Must be called with the pane-state flock held LOCK_EX (same contract
/// as `write_state_unlocked`).
fn reap_zombies_and_load_state_unlocked() -> Result<PaneState> {
    let mut state = read_state_unlocked()?;
    match tmux_list_all_panes_sync() {
        Ok(live) => {
            let zombies = reap_zombies(&mut state, &live);
            if !zombies.is_empty() {
                tracing::info!(
                    count = zombies.len(),
                    ids = ?zombies,
                    "reaped zombie pane lease records"
                );
                write_state_unlocked(&state)?;
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "skipping zombie reap; keeping existing ledger");
        }
    }
    Ok(state)
}

fn ensure_idle_panes_locked(needed: usize) -> Result<()> {
    // Reap zombie lease records BEFORE any capacity math.  If the user
    // closed tmux panes directly (e.g. `:kill-pane`, window closed in
    // another tool), `tmux-state.json` still carries those entries until
    // a heartbeat / release event touches them — and in the typical
    // workflow that never happens, so they accumulate forever.  Without
    // reaping first, `total` below would include dead panes and we'd
    // reject a fresh `/claude` with a misleading "capacity reached"
    // even when tmux itself has plenty of room.
    let mut state = reap_zombies_and_load_state_unlocked()?;

    let idle_count = state
        .values()
        .filter(|l| l.effective_status() == PaneStatus::Idle)
        .count();

    if idle_count >= needed {
        return Ok(());
    }

    let mut deficit = needed - idle_count;
    let total = state.len();
    let busy_count = state
        .values()
        .filter(|l| l.effective_status() == PaneStatus::Busy)
        .count();

    if total + deficit > MAX_PANES {
        return Err(anyhow::Error::new(CapacityError {
            requested: deficit,
            max_panes: MAX_PANES,
            current_busy: busy_count,
        }));
    }

    let all_windows = tmux_list_windows_sync().context("listing tmux windows")?;

    if all_windows.is_empty() {
        anyhow::bail!("not in a tmux session; /claude requires tmux");
    }

    // Window IDs that have at least one Busy pane (owned strings to avoid
    // borrowing `state` while we mutate it below).
    let busy_window_ids: std::collections::HashSet<String> = state
        .values()
        .filter(|l| l.effective_status() == PaneStatus::Busy)
        .map(|l| l.window_id.clone())
        .collect();

    // Step 1: prefer windows with no Busy panes.
    for win in all_windows.iter().filter(|w| !busy_window_ids.contains(*w)) {
        if deficit == 0 || state.len() >= MAX_PANES {
            break;
        }
        match tmux_new_window_sync(win) {
            Ok((new_pane, new_win)) => {
                state.insert(new_pane.clone(), PaneLease::new_idle(new_pane, new_win));
                deficit -= 1;
            }
            Err(e) => {
                tracing::warn!(window = %win, error = %e, "failed to create new tmux window");
            }
        }
    }

    // Step 2: fall back to windows with Busy panes.
    if deficit > 0 {
        for win in all_windows.iter().filter(|w| busy_window_ids.contains(*w)) {
            if deficit == 0 || state.len() >= MAX_PANES {
                break;
            }
            match tmux_new_window_sync(win) {
                Ok((new_pane, new_win)) => {
                    state.insert(new_pane.clone(), PaneLease::new_idle(new_pane, new_win));
                    deficit -= 1;
                }
                Err(e) => {
                    tracing::warn!(window = %win, error = %e, "failed to create new tmux window");
                }
            }
        }
    }

    write_state_unlocked(&state)?;

    if deficit > 0 {
        if state.len() >= MAX_PANES {
            anyhow::bail!(
                "pane capacity reached: {}/{MAX_PANES} panes in use, \
                 need {deficit} more; free existing panes to continue",
                state.len()
            );
        } else {
            anyhow::bail!(
                "unable to create a new tmux window (need {deficit} more pane(s)); \
                 ensure a tmux session is active"
            );
        }
    }

    Ok(())
}

/// Atomically ensure at least one idle pane exists **and** acquire it for
/// the given task — all within a single `LOCK_EX`.
///
/// Returns `(pane_id, had_agent)`. `had_agent = true` means the pane already
/// had the requested agent running; the caller should inject just the prompt.
/// `had_agent = false` means the pane is blank; the caller should launch the
/// requested agent.
///
/// This eliminates the TOCTOU race between `ensure_idle_panes` and
/// `acquire_first_idle` when multiple `ClaudeLaunch` requests arrive
/// concurrently.
#[allow(dead_code)]
pub fn ensure_and_acquire_idle(
    tag: &str,
    session_id: &str,
    worktree: Option<&str>,
) -> Result<(String, bool)> {
    ensure_and_acquire_idle_for_agent(AgentKind::ClaudeCode, tag, session_id, worktree)
}

pub fn ensure_and_acquire_idle_for_agent(
    agent: AgentKind,
    tag: &str,
    session_id: &str,
    worktree: Option<&str>,
) -> Result<(String, bool)> {
    let lock = open_lock_file()?;
    lock.lock_exclusive()
        .context("acquiring flock for ensure_and_acquire_idle")?;

    let result = (|| {
        // Count panes that can actually serve this request:
        //   - blank panes (has_claude = false): always usable
        //   - same-worktree matching-agent panes: reusable via /compact + inject,
        //     but only when worktree.is_some() — when worktree is None,
        //     None==None would match any no-worktree agent pane, but
        //     acquire_first_idle_locked guards tier-1 reuse with
        //     worktree.is_some(), so those panes won't actually be selected.
        //     Counting them as available suppresses expansion and leads to
        //     "no idle panes available" instead.
        // Reap zombies first so `available` / `total_idle` reflect only
        // panes tmux still knows about — otherwise a ledger bloated with
        // zombie Idle entries would make `ensure_idle_panes_locked`
        // receive a `needed = total_idle + 1` way larger than we really
        // want, producing spurious `CapacityError` (or over-creation)
        // once the reap inside that function shrinks `idle_count`.
        let state = reap_zombies_and_load_state_unlocked()?;
        let available = state
            .values()
            .filter(|l| {
                l.effective_status() == PaneStatus::Idle
                    && (!l.has_claude
                        || (l.effective_agent() == Some(agent)
                            && worktree.is_some()
                            && l.worktree.as_deref() == worktree))
            })
            .count();
        if available == 0 {
            // No usable idle pane exists (all idle panes have agents running in
            // different worktrees and cannot receive shell commands). Force
            // ensure_idle_panes_locked to create a new blank pane by requesting
            // one more than the current total idle count — it would otherwise
            // see the existing (unusable) idle panes and skip expansion.
            let total_idle = state
                .values()
                .filter(|l| l.effective_status() == PaneStatus::Idle)
                .count();
            ensure_idle_panes_locked(total_idle + 1)?;
        }
        // Acquire the first usable idle pane.
        acquire_first_idle_locked(agent, tag, session_id, worktree)
    })();

    lock.unlock()
        .context("releasing flock after ensure_and_acquire_idle")?;
    result
}

/// Record the full task description on the pane's lease.
///
/// Called by `handle_claude_launch` after successfully injecting the
/// description, so subsequent `/claude --resume-pane <pid>` or
/// `/codex --resume-pane <pid>` calls can retrieve it without the user
/// retyping. Best-effort: if the pane is no longer in the state map (e.g.
/// removed by a concurrent cleanup), the call is a no-op.
pub fn set_task_description(pane_id: &str, description: &str) -> Result<()> {
    let lock = open_lock_file()?;
    lock.lock_exclusive()
        .context("acquiring flock for set_task_description")?;

    let result = (|| {
        let mut state = read_state_unlocked()?;
        if let Some(lease) = state.get_mut(pane_id) {
            lease.task_description = Some(description.to_string());
            write_state_unlocked(&state)
        } else {
            Ok(())
        }
    })();

    lock.unlock()
        .context("releasing flock after set_task_description")?;
    result
}

/// Correct the session ID stored in the pane lease after the real
/// `session::get_or_create` UUID is known.
///
/// `ensure_and_acquire_idle` is called with a placeholder UUID so the pane is
/// secured before the (potentially fallible) session creation step.  This
/// function patches the stored value to the real ID.
pub fn update_session_id(pane_id: &str, session_id: &str) -> Result<()> {
    let lock = open_lock_file()?;
    lock.lock_exclusive()
        .context("acquiring flock for update_session_id")?;

    let result = (|| {
        let mut state = read_state_unlocked()?;
        if let Some(lease) = state.get_mut(pane_id) {
            lease.session_id = Some(session_id.to_string());
        }
        write_state_unlocked(&state)
    })();

    lock.unlock()
        .context("releasing flock after update_session_id")?;
    result
}

/// Mark a pane as having an active `claude` session.
///
/// Called after successfully launching `claude` into a blank pane so
/// that future task assignments can inject prompts directly instead of
/// launching a new session.
#[allow(dead_code)]
pub fn mark_claude_started(pane_id: &str) -> Result<()> {
    mark_agent_started(pane_id, AgentKind::ClaudeCode)
}

pub fn mark_agent_started(pane_id: &str, agent: AgentKind) -> Result<()> {
    let lock = open_lock_file()?;
    lock.lock_exclusive()
        .context("acquiring flock for mark_claude_started")?;

    let result = (|| {
        let mut state = read_state_unlocked()?;
        if let Some(lease) = state.get_mut(pane_id) {
            lease.has_claude = true;
            lease.active_agent = Some(agent);
        }
        write_state_unlocked(&state)
    })();

    lock.unlock()
        .context("releasing flock after mark_claude_started")?;
    result
}

// ---------------------------------------------------------------------------
// Private tmux helpers (sync, safe to call inside spawn_blocking / flock)
// ---------------------------------------------------------------------------

/// Run `tmux list-panes -a -F "#{pane_id}"` and return every live pane ID
/// across all sessions as a set.  Used by `reap_zombies` to drop lease
/// records that point at panes the user closed behind amaebi's back —
/// without this, `tmux-state.json` accumulates entries indefinitely and
/// eventually saturates `MAX_PANES` even though the tmux server has
/// nothing left on them.
///
/// Tests (unit and integration) can bypass the real tmux call by setting
/// the `AMAEBI_TEST_TMUX_PANES_OVERRIDE` env var to a whitespace-separated
/// list of pane ids (e.g. `"%0 %1 %2"`).  Empty string means "tmux
/// reports no live panes", which lets tests exercise the all-zombies
/// path without shelling out.  Available in all builds so integration
/// tests (which spawn the real binary) can also control the reap path.
fn tmux_list_all_panes_sync() -> Result<std::collections::HashSet<String>> {
    if let Ok(override_str) = std::env::var("AMAEBI_TEST_TMUX_PANES_OVERRIDE") {
        return Ok(override_str
            .split_whitespace()
            .map(str::to_string)
            .collect());
    }
    let output = std::process::Command::new("tmux")
        .args(["list-panes", "-a", "-F", "#{pane_id}"])
        .output()
        .context("spawning tmux list-panes -a")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("tmux list-panes -a failed: {stderr}");
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect())
}

/// Drop lease records whose `pane_id` is no longer present in `live`.
/// Returns the removed pane ids in unspecified order (the implementation
/// iterates `state.keys()`, and `std::collections::HashMap` iteration
/// order is arbitrary and may change across runs / hash-seed rotations).
/// Unit tests that care about membership assert with sets / `contains`,
/// not positional equality.
///
/// Pure function — takes the live-pane set as a parameter so tests can
/// inject any set they like without touching a real tmux server.  The
/// caller is responsible for obtaining `live` (production path uses
/// [`tmux_list_all_panes_sync`]; tests build it by hand).
fn reap_zombies(state: &mut PaneState, live: &std::collections::HashSet<String>) -> Vec<String> {
    let zombies: Vec<String> = state
        .keys()
        .filter(|k| !live.contains(*k))
        .cloned()
        .collect();
    for z in &zombies {
        state.remove(z);
    }
    zombies
}

/// Run `tmux list-windows -F "#{window_id}"` and return a list of window IDs.
fn tmux_list_windows_sync() -> Result<Vec<String>> {
    let output = std::process::Command::new("tmux")
        .args(["list-windows", "-F", "#{window_id}"])
        .output()
        .context("spawning tmux list-windows")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("tmux list-windows failed: {stderr}");
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect())
}

/// Create a new tmux window in the session that owns `window_id`, then return
/// the new window's pane ID and window ID as `(pane_id, window_id)`.
///
/// Uses `tmux new-window` rather than `split-window` so each Claude session
/// gets its own full-size window instead of a split pane — easier to navigate
/// and gives Claude the full terminal width.
fn tmux_new_window_sync(window_id: &str) -> Result<(String, String)> {
    // Resolve the session that owns this window so we can target it.
    let sess_out = std::process::Command::new("tmux")
        .args(["display-message", "-t", window_id, "-p", "#{session_id}"])
        .output()
        .context("spawning tmux display-message")?;
    if !sess_out.status.success() {
        let stderr = String::from_utf8_lossy(&sess_out.stderr);
        anyhow::bail!("tmux display-message failed for {window_id}: {stderr}");
    }
    let session_id = String::from_utf8_lossy(&sess_out.stdout).trim().to_string();
    if session_id.is_empty() {
        anyhow::bail!("could not resolve session for window {window_id}");
    }

    let output = std::process::Command::new("tmux")
        .args([
            "new-window",
            "-t",
            &session_id,
            "-d", // don't switch focus to the new window
            "-P",
            "-F",
            "#{pane_id} #{window_id}",
        ])
        .output()
        .context("spawning tmux new-window")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("tmux new-window failed: {stderr}");
    }
    let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let mut parts = raw.split_whitespace();
    let pane_id = parts
        .next()
        .filter(|s| !s.is_empty())
        .context("tmux new-window produced no pane ID")?
        .to_string();
    let new_window_id = parts
        .next()
        .filter(|s| !s.is_empty())
        .context("tmux new-window produced no window ID")?
        .to_string();
    Ok((pane_id, new_window_id))
}

// ---------------------------------------------------------------------------
// Test-only helpers (accessible from other modules' tests)
// ---------------------------------------------------------------------------

/// Insert or overwrite a single lease in the pane state file.  Test-only.
///
/// Reads the current state, upserts `lease` keyed by `pane_id` (other entries
/// are preserved), then writes the result back — so callers can build up
/// multi-pane fixtures with repeated calls.
///
/// Exposed so tests outside this module (e.g. supervision-loop tests in
/// `daemon.rs`) can set up pane state without reaching into the private
/// `write_state_unlocked` helper.  Must be called inside a `with_temp_home`
/// scope so the write lands under a temp `$HOME`.
#[cfg(test)]
pub(crate) fn seed_state_for_test(lease: PaneLease) -> Result<()> {
    let lock = open_lock_file()?;
    lock.lock_exclusive()
        .context("acquiring flock for seed_state_for_test")?;
    let result = (|| {
        let mut state = read_state_unlocked()?;
        state.insert(lease.pane_id.clone(), lease);
        write_state_unlocked(&state)
    })();
    lock.unlock()
        .context("releasing flock after seed_state_for_test")?;
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// RAII guard that sets `AMAEBI_TEST_TMUX_PANES_OVERRIDE` on construction
    /// and restores the prior value (or removes it if unset) on drop — even
    /// when the drop is caused by a panicking assertion.  Tests that use
    /// this guard are also tagged `#[serial_test::serial]` because the env
    /// var is process-global; two parallel holders would clobber each
    /// other regardless of RAII cleanup.  Capturing the prior value means
    /// nested holders and pre-set environments (CI harness, dev shell) are
    /// left untouched, so cleanup is never a silent mutation of the outer
    /// scope.
    struct TmuxOverride {
        prior: Option<std::ffi::OsString>,
    }

    impl TmuxOverride {
        fn set(panes: &str) -> Self {
            let prior = std::env::var_os("AMAEBI_TEST_TMUX_PANES_OVERRIDE");
            std::env::set_var("AMAEBI_TEST_TMUX_PANES_OVERRIDE", panes);
            Self { prior }
        }
    }

    impl Drop for TmuxOverride {
        fn drop(&mut self) {
            match self.prior.take() {
                Some(v) => std::env::set_var("AMAEBI_TEST_TMUX_PANES_OVERRIDE", v),
                None => std::env::remove_var("AMAEBI_TEST_TMUX_PANES_OVERRIDE"),
            }
        }
    }

    fn make_idle(pane_id: &str, window_id: &str) -> PaneLease {
        PaneLease::new_idle(pane_id.to_string(), window_id.to_string())
    }

    fn make_busy(pane_id: &str, window_id: &str, worktree: Option<&str>) -> PaneLease {
        PaneLease {
            pane_id: pane_id.to_string(),
            window_id: window_id.to_string(),
            status: PaneStatus::Busy,
            tag: Some("task-1".to_string()),
            session_id: Some("sess-1".to_string()),
            worktree: worktree.map(str::to_string),
            heartbeat_at: now_secs(),
            has_claude: false,
            active_agent: None,
            task_description: None,
        }
    }

    // ── effective_status ──────────────────────────────────────────────────

    #[test]
    fn effective_status_idle_is_idle() {
        let l = make_idle("%0", "@0");
        assert_eq!(l.effective_status(), PaneStatus::Idle);
    }

    #[test]
    fn effective_status_busy_fresh_is_busy() {
        let l = make_busy("%0", "@0", None);
        assert_eq!(l.effective_status(), PaneStatus::Busy);
    }

    #[test]
    fn effective_status_busy_expired_becomes_idle() {
        let mut l = make_busy("%0", "@0", None);
        // Expire the heartbeat.
        l.heartbeat_at = now_secs().saturating_sub(LEASE_TTL_SECS + 1);
        assert_eq!(l.effective_status(), PaneStatus::Idle);
    }

    #[test]
    fn effective_status_busy_at_boundary_still_busy() {
        let mut l = make_busy("%0", "@0", None);
        // Exactly at TTL boundary — should still be Busy.
        l.heartbeat_at = now_secs().saturating_sub(LEASE_TTL_SECS);
        assert_eq!(l.effective_status(), PaneStatus::Busy);
    }

    // ── serialization ─────────────────────────────────────────────────────

    #[test]
    fn pane_lease_round_trip() {
        let l = make_busy("%3", "@2", Some("/home/user/repo-wt/task1"));
        let json = serde_json::to_string(&l).expect("serialize");
        let back: PaneLease = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.pane_id, "%3");
        assert_eq!(back.window_id, "@2");
        assert_eq!(back.worktree.as_deref(), Some("/home/user/repo-wt/task1"));
        assert_eq!(back.status, PaneStatus::Busy);
    }

    #[test]
    fn pane_state_round_trip() {
        let mut state: PaneState = HashMap::new();
        state.insert("%0".to_string(), make_idle("%0", "@0"));
        state.insert("%1".to_string(), make_busy("%1", "@0", None));

        let json = serde_json::to_string_pretty(&state).expect("serialize");
        let back: PaneState = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.len(), 2);
        assert_eq!(back["%0"].effective_status(), PaneStatus::Idle);
        assert_eq!(back["%1"].effective_status(), PaneStatus::Busy);
    }

    // ── CapacityError display ─────────────────────────────────────────────

    #[test]
    fn capacity_error_display() {
        let e = CapacityError {
            requested: 3,
            max_panes: 16,
            current_busy: 14,
        };
        let s = e.to_string();
        assert!(s.contains("max_panes=16"), "got: {s}");
        assert!(s.contains("busy=14"), "got: {s}");
        assert!(s.contains("requested=3"), "got: {s}");
    }

    // ── file-based state ops (use tempdir via HOME override) ───────────────

    // Helper: redirect HOME to a tempdir, run f(), restore HOME.

    #[test]
    fn acquire_lease_on_idle_pane_succeeds() {
        {
            let _guard = crate::test_utils::with_temp_home();
            // Seed one idle pane directly.
            let mut state: PaneState = HashMap::new();
            state.insert("%0".to_string(), make_idle("%0", "@0"));
            write_state_unlocked(&state).expect("seed state");

            acquire_lease("%0", "task-x", "sess-x", None).expect("acquire");

            let s = read_state_unlocked().expect("read back");
            assert_eq!(s["%0"].effective_status(), PaneStatus::Busy);
            assert_eq!(s["%0"].tag.as_deref(), Some("task-x"));
        }
    }

    #[test]
    fn acquire_lease_on_busy_pane_returns_err() {
        {
            let _guard = crate::test_utils::with_temp_home();
            let mut state: PaneState = HashMap::new();
            state.insert("%0".to_string(), make_busy("%0", "@0", None));
            write_state_unlocked(&state).expect("seed state");

            let err = acquire_lease("%0", "task-y", "sess-y", None);
            assert!(err.is_err(), "expected Err for busy pane");
        }
    }

    #[test]
    fn acquire_lease_on_expired_busy_pane_succeeds() {
        {
            let _guard = crate::test_utils::with_temp_home();
            let mut expired = make_busy("%0", "@0", None);
            expired.heartbeat_at = now_secs().saturating_sub(LEASE_TTL_SECS + 1);
            let mut state: PaneState = HashMap::new();
            state.insert("%0".to_string(), expired);
            write_state_unlocked(&state).expect("seed state");

            acquire_lease("%0", "task-new", "sess-new", None).expect("should succeed after expiry");

            let s = read_state_unlocked().expect("read back");
            assert_eq!(s["%0"].tag.as_deref(), Some("task-new"));
        }
    }

    #[test]
    fn acquire_lease_with_none_worktree_preserves_existing() {
        // Regression: `Request::ReservePane` (used by agent distillation)
        // calls `acquire_lease` with `worktree=None` because the client
        // doesn't know the resume-pane's worktree.  An unconditional
        // overwrite would clear the worktree set by a previous launch's
        // `release_lease` (which intentionally preserves it) and break the
        // subsequent `--resume-pane` probe with "pane has no associated
        // worktree".
        let _guard = crate::test_utils::with_temp_home();
        let mut released = make_idle("%0", "@0");
        released.worktree = Some("/tmp/wt-from-previous-launch".to_string());
        released.has_claude = true;
        let mut state: PaneState = HashMap::new();
        state.insert("%0".to_string(), released);
        write_state_unlocked(&state).expect("seed state");

        acquire_lease("%0", "task-new", "sess-new", None).expect("acquire");

        let s = read_state_unlocked().expect("read back");
        assert_eq!(
            s["%0"].worktree.as_deref(),
            Some("/tmp/wt-from-previous-launch"),
            "worktree must be preserved when caller passes None"
        );
        assert_eq!(s["%0"].tag.as_deref(), Some("task-new"));
        assert_eq!(s["%0"].effective_status(), PaneStatus::Busy);
    }

    #[test]
    fn acquire_lease_with_some_worktree_overwrites() {
        let _guard = crate::test_utils::with_temp_home();
        let mut prior = make_idle("%0", "@0");
        prior.worktree = Some("/tmp/old".to_string());
        let mut state: PaneState = HashMap::new();
        state.insert("%0".to_string(), prior);
        write_state_unlocked(&state).expect("seed state");

        acquire_lease("%0", "task-new", "sess-new", Some("/tmp/new")).expect("acquire");

        let s = read_state_unlocked().expect("read back");
        assert_eq!(s["%0"].worktree.as_deref(), Some("/tmp/new"));
    }

    #[test]
    fn acquire_lease_for_agent_rejects_mismatched_agent() {
        let _guard = crate::test_utils::with_temp_home();
        let mut claude = make_idle("%0", "@0");
        claude.has_claude = true;
        claude.active_agent = Some(AgentKind::ClaudeCode);
        let mut state: PaneState = HashMap::new();
        state.insert("%0".to_string(), claude);
        write_state_unlocked(&state).expect("seed state");

        let err = acquire_lease_for_agent(AgentKind::Codex, "%0", "task", "sess", None)
            .expect_err("codex must not reserve a claude pane");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("codex"),
            "error should name expected agent: {msg}"
        );
        assert!(
            msg.contains("--resume-pane"),
            "error should explain recovery: {msg}"
        );

        let s = read_state_unlocked().expect("read back");
        assert_eq!(s["%0"].effective_status(), PaneStatus::Idle);
    }

    #[test]
    fn acquire_lease_for_agent_accepts_legacy_claude_marker() {
        let _guard = crate::test_utils::with_temp_home();
        let mut legacy = make_idle("%0", "@0");
        legacy.has_claude = true;
        legacy.active_agent = None;
        let mut state: PaneState = HashMap::new();
        state.insert("%0".to_string(), legacy);
        write_state_unlocked(&state).expect("seed state");

        acquire_lease_for_agent(AgentKind::ClaudeCode, "%0", "task", "sess", None)
            .expect("legacy has_claude marker should reserve as Claude Code");

        let s = read_state_unlocked().expect("read back");
        assert_eq!(s["%0"].effective_status(), PaneStatus::Busy);
        assert_eq!(s["%0"].tag.as_deref(), Some("task"));
    }

    #[test]
    fn acquire_first_idle_finds_idle_pane() {
        {
            let _guard = crate::test_utils::with_temp_home();
            let mut state: PaneState = HashMap::new();
            state.insert("%0".to_string(), make_busy("%0", "@0", None));
            state.insert("%1".to_string(), make_idle("%1", "@0"));
            write_state_unlocked(&state).expect("seed state");

            let (pane, _had_agent) =
                acquire_first_idle("t", "s", None).expect("acquire first idle");
            assert_eq!(pane, "%1");
        }
    }

    #[test]
    fn acquire_first_idle_rejects_duplicate_worktree() {
        {
            let _guard = crate::test_utils::with_temp_home();
            let mut state: PaneState = HashMap::new();
            state.insert(
                "%0".to_string(),
                make_busy("%0", "@0", Some("/repo/wt/task1")),
            );
            state.insert("%1".to_string(), make_idle("%1", "@0"));
            write_state_unlocked(&state).expect("seed state");

            let err = acquire_first_idle("t2", "s2", Some("/repo/wt/task1"));
            assert!(err.is_err(), "expected Err for duplicate worktree");
        }
    }

    #[test]
    fn acquire_first_idle_prefers_has_claude_pane_with_matching_worktree() {
        {
            let _guard = crate::test_utils::with_temp_home();
            let mut state: PaneState = HashMap::new();
            // Pane with an agent already running in the target worktree — should be preferred.
            let mut has_claude_matching = make_idle("%0", "@0");
            has_claude_matching.has_claude = true;
            has_claude_matching.worktree = Some("/repo/wt/task1".to_string());
            // Plain idle pane (no agent).
            let blank = make_idle("%1", "@0");
            state.insert("%0".to_string(), has_claude_matching);
            state.insert("%1".to_string(), blank);
            write_state_unlocked(&state).expect("seed state");

            let (pane, had_agent) =
                acquire_first_idle("t", "s", Some("/repo/wt/task1")).expect("acquire");
            assert_eq!(pane, "%0", "should prefer matching has_claude pane");
            assert!(had_agent, "had_agent should be true for reused pane");
        }
    }

    #[test]
    fn acquire_first_idle_for_agent_requires_matching_agent() {
        {
            let _guard = crate::test_utils::with_temp_home();
            let mut state: PaneState = HashMap::new();
            let mut claude = make_idle("%0", "@0");
            claude.has_claude = true;
            claude.active_agent = Some(AgentKind::ClaudeCode);
            claude.worktree = Some("/repo/wt/task1".to_string());
            let blank = make_idle("%1", "@0");
            state.insert("%0".to_string(), claude);
            state.insert("%1".to_string(), blank);
            write_state_unlocked(&state).expect("seed state");

            let (pane, had_agent) =
                acquire_first_idle_for_agent(AgentKind::Codex, "t", "s", Some("/repo/wt/task1"))
                    .expect("acquire");
            assert_eq!(pane, "%1", "codex must not reuse a claude pane");
            assert!(!had_agent);
        }
    }

    #[test]
    fn acquire_first_idle_for_agent_reuses_matching_codex_pane() {
        {
            let _guard = crate::test_utils::with_temp_home();
            let mut state: PaneState = HashMap::new();
            let mut codex = make_idle("%0", "@0");
            codex.has_claude = true;
            codex.active_agent = Some(AgentKind::Codex);
            codex.worktree = Some("/repo/wt/task1".to_string());
            let blank = make_idle("%1", "@0");
            state.insert("%0".to_string(), codex);
            state.insert("%1".to_string(), blank);
            write_state_unlocked(&state).expect("seed state");

            let (pane, had_agent) =
                acquire_first_idle_for_agent(AgentKind::Codex, "t", "s", Some("/repo/wt/task1"))
                    .expect("acquire");
            assert_eq!(pane, "%0", "codex should reuse matching codex pane");
            assert!(had_agent);
        }
    }

    #[test]
    fn acquire_first_idle_skips_has_claude_pane_with_different_worktree() {
        {
            let _guard = crate::test_utils::with_temp_home();
            let mut state: PaneState = HashMap::new();
            // Pane with an agent in a *different* worktree — must not be preferred.
            let mut has_claude_other = make_idle("%0", "@0");
            has_claude_other.has_claude = true;
            has_claude_other.worktree = Some("/repo/wt/other".to_string());
            // Plain idle pane (no agent, no worktree).
            let blank = make_idle("%1", "@0");
            state.insert("%0".to_string(), has_claude_other);
            state.insert("%1".to_string(), blank);
            write_state_unlocked(&state).expect("seed state");

            let (pane, had_agent) =
                acquire_first_idle("t", "s", Some("/repo/wt/task1")).expect("acquire");
            // Must select the blank pane (%1), not the mismatched agent pane (%0).
            // Sending shell commands to a pane where an agent is running in a different
            // worktree would deliver them as chat messages, not shell commands.
            assert_eq!(
                pane, "%1",
                "must skip has_claude pane with different worktree"
            );
            assert!(!had_agent, "had_agent must be false for blank pane");
        }
    }

    #[test]
    fn ensure_and_acquire_expands_when_worktree_none_and_only_claude_panes_exist() {
        // When worktree=None and the only idle pane has has_claude=true with
        // worktree=None, None==None must NOT suppress expansion.
        // acquire_first_idle_locked requires worktree.is_some() for tier-1
        // reuse, so that pane is not selectable — expansion must be attempted.
        //
        // Skip inside a live tmux session: ensure_and_acquire_idle calls
        // tmux split-window when expanding, which would create real panes.
        if std::env::var("TMUX").is_ok() {
            eprintln!("skipping: live tmux session detected (run via scripts/test.sh --docker)");
            return;
        }
        {
            let _guard = crate::test_utils::with_temp_home();
            let mut state: PaneState = HashMap::new();
            let mut pane = make_idle("%0", "@0");
            pane.has_claude = true;
            pane.worktree = None; // matches None==None but must NOT count as available
            state.insert("%0".to_string(), pane);
            write_state_unlocked(&state).expect("seed state");

            match ensure_and_acquire_idle("t", "s", None) {
                Ok((_, had_agent)) => {
                    assert!(
                        !had_agent,
                        "None-worktree pane must not trigger tier-1 reuse"
                    );
                }
                Err(e) => {
                    let msg = format!("{e:#}");
                    assert!(
                        !msg.contains("no idle panes available"),
                        "should attempt expansion, not short-circuit: {msg}"
                    );
                }
            }
        }
    }

    #[test]
    fn ensure_and_acquire_expands_when_only_mismatched_claude_panes_exist() {
        // All idle panes have an agent running in a different worktree — none
        // are usable for a fresh task. ensure_and_acquire_idle must attempt
        // expansion rather than returning "no idle panes available" immediately.
        //
        // Skip inside a live tmux session: ensure_and_acquire_idle calls
        // tmux split-window when expanding, which would create real panes.
        if std::env::var("TMUX").is_ok() {
            eprintln!("skipping: live tmux session detected (run via scripts/test.sh --docker)");
            return;
        }
        {
            let _guard = crate::test_utils::with_temp_home();
            let mut state: PaneState = HashMap::new();
            let mut pane = make_idle("%0", "@0");
            pane.has_claude = true;
            pane.worktree = Some("/repo/wt/other".to_string());
            state.insert("%0".to_string(), pane);
            write_state_unlocked(&state).expect("seed state");

            match ensure_and_acquire_idle("t", "s", Some("/repo/wt/task1")) {
                Ok((pane_id, had_agent)) => {
                    assert_ne!(pane_id, "%0", "must not acquire mismatched claude pane");
                    assert!(!had_agent, "new pane must not have had_agent");
                }
                Err(e) => {
                    let msg = format!("{e:#}");
                    assert!(
                        !msg.contains("no idle panes available"),
                        "should attempt expansion, not short-circuit: {msg}"
                    );
                }
            }
        }
    }

    #[test]
    fn release_lease_marks_pane_idle() {
        {
            let _guard = crate::test_utils::with_temp_home();
            let mut state: PaneState = HashMap::new();
            state.insert("%0".to_string(), make_busy("%0", "@0", None));
            write_state_unlocked(&state).expect("seed state");

            release_lease("%0").expect("release");

            let s = read_state_unlocked().expect("read back");
            assert_eq!(s["%0"].effective_status(), PaneStatus::Idle);
            assert!(s["%0"].tag.is_none());
        }
    }

    #[test]
    fn release_lease_preserves_worktree_and_has_claude() {
        // worktree and has_claude must survive release so the scheduler can
        // reuse the pane for a future task in the same worktree (tier-1 reuse).
        {
            let _guard = crate::test_utils::with_temp_home();
            let mut lease = make_busy("%0", "@0", Some("/repo/wt/task1"));
            lease.has_claude = true;
            let mut state: PaneState = HashMap::new();
            state.insert("%0".to_string(), lease);
            write_state_unlocked(&state).expect("seed state");

            release_lease("%0").expect("release");

            let s = read_state_unlocked().expect("read back");
            assert_eq!(s["%0"].effective_status(), PaneStatus::Idle);
            assert_eq!(
                s["%0"].worktree.as_deref(),
                Some("/repo/wt/task1"),
                "worktree must be preserved so tier-1 reuse can match it"
            );
            assert!(s["%0"].has_claude, "has_claude must be preserved");
        }
    }

    #[test]
    fn heartbeat_updates_timestamp() {
        {
            let _guard = crate::test_utils::with_temp_home();
            let mut lease = make_busy("%0", "@0", None);
            // Set an old heartbeat.
            lease.heartbeat_at = 1000;
            let mut state: PaneState = HashMap::new();
            state.insert("%0".to_string(), lease);
            write_state_unlocked(&state).expect("seed state");

            heartbeat("%0").expect("heartbeat");

            let s = read_state_unlocked().expect("read back");
            assert!(
                s["%0"].heartbeat_at > 1000,
                "heartbeat should have been updated"
            );
        }
    }

    #[test]
    #[serial_test::serial]
    fn ensure_idle_panes_capacity_exceeded_returns_err() {
        {
            let _guard = crate::test_utils::with_temp_home();
            // Fill all MAX_PANES slots with Busy leases.
            let mut state: PaneState = HashMap::new();
            let mut ids = Vec::with_capacity(MAX_PANES);
            for i in 0..MAX_PANES {
                let pid = format!("%{i}");
                state.insert(pid.clone(), make_busy(&pid, "@0", None));
                ids.push(pid);
            }
            write_state_unlocked(&state).expect("seed state");

            // Tell the zombie-reap step that every seeded pane is still
            // alive in tmux — without this it would drop them all and
            // the capacity check would pass instead of failing.  RAII
            // guard clears the env var even if the asserts below panic.
            let _override = TmuxOverride::set(&ids.join(" "));

            // Acquiring the lock before calling the internal helper so we can
            // test the logic without real tmux.
            let lock = open_lock_file().expect("open lock");
            lock.lock_exclusive().expect("flock");
            let result = ensure_idle_panes_locked(1);
            lock.unlock().expect("unlock");

            let err = result.expect_err("should be Err");
            let cap = err.downcast_ref::<CapacityError>().expect("CapacityError");
            assert_eq!(cap.max_panes, MAX_PANES);
            assert_eq!(cap.requested, 1);
        }
    }

    #[test]
    #[serial_test::serial]
    fn ensure_idle_panes_noop_when_enough_idle() {
        {
            let _guard = crate::test_utils::with_temp_home();
            let mut state: PaneState = HashMap::new();
            state.insert("%0".to_string(), make_idle("%0", "@0"));
            state.insert("%1".to_string(), make_idle("%1", "@0"));
            write_state_unlocked(&state).expect("seed state");

            // Mark the seeded panes as live so the reap leaves them alone.
            let _override = TmuxOverride::set("%0 %1");

            let lock = open_lock_file().expect("open lock");
            lock.lock_exclusive().expect("flock");
            // 2 idle panes, need 2 → no-op, no new tmux windows created.
            let result = ensure_idle_panes_locked(2);
            lock.unlock().expect("unlock");

            result.expect("should succeed without creating new windows");
        }
    }

    #[test]
    #[serial_test::serial]
    fn ensure_idle_panes_reaps_zombies_before_capacity_check() {
        // End-to-end regression for the window-6 bug (2026-05-10):
        // ledger has MAX_PANES entries, but only a handful are still
        // alive in tmux.  Before the fix, this would reject with
        // CapacityError because `total == MAX_PANES`; after the fix,
        // the zombies are reaped first and capacity is judged against
        // what's actually live.  We don't exercise the tmux-window
        // creation path (that requires a real tmux); instead we ask
        // for `needed = <idle count>` so no deficit remains and the
        // function short-circuits at the idle-count check.
        let _guard = crate::test_utils::with_temp_home();
        let mut state: PaneState = HashMap::new();
        // 2 live + 14 zombies = 16 entries total (saturated ledger).
        for i in 0..2 {
            let pid = format!("%live{i}");
            state.insert(pid.clone(), make_idle(&pid, "@0"));
        }
        for i in 0..14 {
            let pid = format!("%zombie{i}");
            state.insert(pid.clone(), make_idle(&pid, "@0"));
        }
        assert_eq!(state.len(), MAX_PANES, "seed must saturate MAX_PANES");
        write_state_unlocked(&state).expect("seed state");

        // tmux only reports the 2 live panes — the 14 zombies were
        // closed behind amaebi's back.  RAII guard clears the env var
        // even if the asserts below panic.
        let _override = TmuxOverride::set("%live0 %live1");

        let lock = open_lock_file().expect("open lock");
        lock.lock_exclusive().expect("flock");
        // Need 2 idle panes; we have 2 live idle after reap → Ok.
        let result = ensure_idle_panes_locked(2);
        lock.unlock().expect("unlock");

        result.expect("reap should drop the 14 zombies; 2 live idle is enough");

        // Persisted ledger must now have only the live entries — if
        // the reap didn't write through, the next launch would see a
        // saturated state again.
        let reloaded = read_state_unlocked().expect("reload state");
        assert_eq!(reloaded.len(), 2);
        assert!(reloaded.contains_key("%live0"));
        assert!(reloaded.contains_key("%live1"));
    }

    #[test]
    #[serial_test::serial]
    fn ensure_and_acquire_idle_reaps_zombies_before_counting_available() {
        // Regression for the round-3 review comment: before the fix,
        // `ensure_and_acquire_idle` computed `available` / `total_idle`
        // from the pre-reap state, so a ledger bloated with zombie Idle
        // entries would (a) mask the fact that no live pane is usable
        // or (b) pass `needed = total_idle + 1` through to
        // `ensure_idle_panes_locked` with an inflated value, producing
        // spurious CapacityError once that function's internal reap
        // shrank idle_count.
        //
        // This test pins the "reap before counting" contract in the
        // cheap end-to-end case: 2 zombies + 1 live blank idle pane.
        // After reap, `available = 1` (the live blank), so no expansion
        // is needed and `acquire_first_idle_locked` selects the live
        // pane without touching real tmux.
        let _guard = crate::test_utils::with_temp_home();
        let mut state: PaneState = HashMap::new();
        state.insert("%live".into(), make_idle("%live", "@0"));
        state.insert("%zombie0".into(), make_idle("%zombie0", "@0"));
        state.insert("%zombie1".into(), make_idle("%zombie1", "@0"));
        write_state_unlocked(&state).expect("seed state");

        let _override = TmuxOverride::set("%live");

        let (pane_id, had_agent) = ensure_and_acquire_idle("t", "s", None).expect("acquire");

        assert_eq!(pane_id, "%live", "must pick the live pane, not a zombie");
        assert!(!had_agent);

        let reloaded = read_state_unlocked().expect("reload");
        assert!(
            !reloaded.contains_key("%zombie0") && !reloaded.contains_key("%zombie1"),
            "zombies must be reaped before acquisition"
        );
        assert_eq!(
            reloaded["%live"].effective_status(),
            PaneStatus::Busy,
            "acquired pane must be marked busy"
        );
    }

    // ── reap_zombies ─────────────────────────────────────────────────────
    //
    // `reap_zombies` is the pure predicate behind the zombie cleanup now
    // performed at the top of `ensure_idle_panes_locked`.  Before this
    // fix, `tmux-state.json` accumulated lease records for panes the
    // user had closed directly in tmux — the ledger eventually hit 16
    // entries and blocked every new `/claude` with a misleading
    // "capacity reached" even though tmux had plenty of room.  These
    // tests pin the three interesting cases.

    #[test]
    fn reap_zombies_removes_entries_not_in_live_set() {
        let mut state: PaneState = HashMap::new();
        state.insert("%0".into(), make_idle("%0", "@0"));
        state.insert("%1".into(), make_busy("%1", "@1", None));
        state.insert("%2".into(), make_idle("%2", "@2"));
        // Only %0 and %2 are still alive in tmux.  %1 is a zombie
        // (user closed it in tmux without releasing through amaebi).
        let live: std::collections::HashSet<String> =
            ["%0".to_string(), "%2".to_string()].into_iter().collect();

        let removed = reap_zombies(&mut state, &live);

        assert_eq!(removed, vec!["%1".to_string()]);
        assert!(!state.contains_key("%1"));
        assert!(state.contains_key("%0"));
        assert!(state.contains_key("%2"));
    }

    #[test]
    fn reap_zombies_noop_when_state_matches_live() {
        // If every ledger entry is still alive, reap must not touch
        // anything and must return an empty vec.  This is the common
        // hot path — we don't want to pay a write on every
        // `ensure_idle_panes_locked` call.
        let mut state: PaneState = HashMap::new();
        state.insert("%0".into(), make_idle("%0", "@0"));
        state.insert("%1".into(), make_idle("%1", "@0"));
        let live: std::collections::HashSet<String> =
            ["%0".to_string(), "%1".to_string()].into_iter().collect();

        let removed = reap_zombies(&mut state, &live);

        assert!(removed.is_empty());
        assert_eq!(state.len(), 2);
    }

    #[test]
    fn reap_zombies_clears_all_when_tmux_is_empty() {
        // Defensive: if tmux was restarted and returns no panes at all,
        // every ledger entry is a zombie.  reap must empty the state
        // rather than leaving stale records that would immediately
        // re-saturate capacity on the next launch.
        let mut state: PaneState = HashMap::new();
        state.insert("%9".into(), make_busy("%9", "@0", None));
        state.insert("%8".into(), make_idle("%8", "@0"));
        let live: std::collections::HashSet<String> = std::collections::HashSet::new();

        let removed = reap_zombies(&mut state, &live);

        assert_eq!(removed.len(), 2);
        assert!(state.is_empty());
    }

    #[test]
    fn set_task_description_persists_to_disk_for_existing_pane() {
        let _guard = crate::test_utils::with_temp_home();
        let mut state: PaneState = HashMap::new();
        let mut lease = make_idle("%7", "@0");
        lease.task_description = None;
        state.insert("%7".to_string(), lease);
        write_state_unlocked(&state).expect("seed state");

        set_task_description("%7", "resume this exact task").expect("persist");

        let s = read_state_unlocked().expect("read back");
        assert_eq!(
            s["%7"].task_description.as_deref(),
            Some("resume this exact task"),
            "description must be persisted on the existing lease"
        );
    }

    #[test]
    fn set_task_description_is_noop_for_missing_pane() {
        let _guard = crate::test_utils::with_temp_home();
        let mut state: PaneState = HashMap::new();
        let mut lease = make_idle("%7", "@0");
        lease.task_description = Some("keep me".to_string());
        state.insert("%7".to_string(), lease);
        write_state_unlocked(&state).expect("seed state");

        set_task_description("%999", "should not be written").expect("noop");

        let s = read_state_unlocked().expect("read back");
        assert_eq!(s.len(), 1, "no new entries must be added");
        assert!(
            !s.contains_key("%999"),
            "missing pane must not be auto-inserted"
        );
        assert_eq!(
            s["%7"].task_description.as_deref(),
            Some("keep me"),
            "existing pane's description must be left untouched"
        );
    }
}
