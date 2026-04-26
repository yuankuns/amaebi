//! Resource lease management for externally-arbitrated resources (GPUs,
//! simulator containers, serial devices, etc.) shared across parallel Claude
//! sessions launched via `/claude`.
//!
//! ## Two files, two roles
//!
//! * `~/.amaebi/resources.toml` — the **pool**: static definitions of what
//!   resources exist.  Hand-edited or written by a future
//!   `amaebi resource register` CLI.  Each resource carries `metadata`
//!   (machine-readable key/value, feeds placeholder substitution), `env`
//!   (injected into the pane shell), and `prompt_hint` (prepended to the task
//!   description so the LLM knows what resource is assigned and how to use it).
//!
//! * `~/.amaebi/resource-state.json` — the **lease table**: runtime state
//!   (who holds what, when).  Protected by an exclusive `flock` on
//!   `~/.amaebi/resource-state.lock`.
//!
//! ## Acquisition semantics
//!
//! - **Named** requests (`ResourceRequest::Named("sim-9900")`) target a
//!   specific resource.  Fails / waits if the resource is busy.
//! - **Class** requests (`ResourceRequest::Class("simulator")`) pick any
//!   idle resource of that class.
//! - **All-or-nothing**: [`acquire_all`] either returns every requested
//!   lease or rolls back partial holds.  Prevents deadlocks where two
//!   callers each hold half of what the other wants.
//! - **Canonical ordering**: requests are processed in a deterministic
//!   `(class, name)` order regardless of caller input order, so concurrent
//!   callers cannot interleave into a cycle.
//! - **Wait vs. Nowait**: callers choose via [`WaitPolicy`].  Wait uses
//!   `tokio::sync::Notify`; [`release_all`] wakes pending waiters.
//!
//! ## Not yet supported
//!
//! Dynamic discovery (nvidia-smi etc.), `amaebi resource register`, and the
//! standalone `amaebi with-resource -- <cmd>` entrypoint are deliberately
//! deferred; this module currently serves the `/claude --resource` path.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use anyhow::{Context, Result};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;

use crate::auth::amaebi_home;

/// Seconds after which a Busy lease whose heartbeat has not been refreshed is
/// treated as Idle.  Mirrors [`crate::pane_lease::LEASE_TTL_SECS`] so a pane
/// whose claude session died without releasing its resources will eventually
/// return those resources to the pool.
pub const LEASE_TTL_SECS: u64 = 86_400;

// ---------------------------------------------------------------------------
// Pool definition (TOML)
// ---------------------------------------------------------------------------

/// One resource as declared in `~/.amaebi/resources.toml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResourceDef {
    pub name: String,
    pub class: String,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    /// Environment variables to inject into the pane shell for the holder.
    /// Values may reference `{metadata_key}` placeholders and `{name}`.
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Optional text prepended to the task description so the LLM knows
    /// what resource it has and how to use it.  Supports the same
    /// placeholders as `env`.
    #[serde(default)]
    pub prompt_hint: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct PoolFile {
    #[serde(default, rename = "resource")]
    resources: Vec<ResourceDef>,
}

fn pool_path() -> Result<PathBuf> {
    Ok(amaebi_home()?.join("resources.toml"))
}

/// Load the resource pool from disk.  Missing file → empty pool (no error).
pub fn load_pool() -> Result<Vec<ResourceDef>> {
    let path = pool_path()?;
    if !path.exists() {
        return Ok(Vec::new());
    }
    let contents =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    if contents.trim().is_empty() {
        return Ok(Vec::new());
    }
    let parsed: PoolFile =
        toml::from_str(&contents).with_context(|| format!("parsing {}", path.display()))?;

    // Validate: names must be unique.  Duplicate names are a configuration
    // bug that would silently shadow leases; surface it loudly at load time.
    let mut seen = std::collections::HashSet::new();
    for r in &parsed.resources {
        if !seen.insert(r.name.clone()) {
            anyhow::bail!(
                "duplicate resource name {:?} in {}; resource names must be unique",
                r.name,
                path.display()
            );
        }
    }

    Ok(parsed.resources)
}

// ---------------------------------------------------------------------------
// Lease state (JSON)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResourceStatus {
    Idle,
    Busy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLease {
    pub name: String,
    pub class: String,
    pub status: ResourceStatus,
    /// tmux pane that holds this lease (resource lifetime = pane lifetime).
    pub pane_id: Option<String>,
    pub tag: Option<String>,
    pub session_id: Option<String>,
    pub heartbeat_at: u64,
}

impl ResourceLease {
    fn new_idle(def: &ResourceDef) -> Self {
        Self {
            name: def.name.clone(),
            class: def.class.clone(),
            status: ResourceStatus::Idle,
            pane_id: None,
            tag: None,
            session_id: None,
            heartbeat_at: now_secs(),
        }
    }

    pub fn effective_status(&self) -> ResourceStatus {
        if self.status == ResourceStatus::Busy
            && now_secs().saturating_sub(self.heartbeat_at) > LEASE_TTL_SECS
        {
            ResourceStatus::Idle
        } else {
            self.status.clone()
        }
    }
}

/// Keyed by resource `name`.
pub type ResourceState = HashMap<String, ResourceLease>;

// ---------------------------------------------------------------------------
// Requests & results
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceRequest {
    /// Acquire a specific resource by name.
    Named(String),
    /// Acquire any idle resource of this class.
    Class(String),
}

impl ResourceRequest {
    /// Parse a CLI-style spec.
    ///
    /// - `"class:gpu"` → [`Self::Class`] (`"gpu"`)
    /// - `"any:gpu"`   → [`Self::Class`] (`"gpu"`) — convenience alias
    /// - anything else → [`Self::Named`] (the input verbatim)
    ///
    /// The explicit `class:` prefix is intentional: a resource pool can have
    /// both a resource named `"gpu"` and a class named `"gpu"` (e.g. a single
    /// gpu with its class also called `gpu`), and we want the caller to
    /// express intent unambiguously rather than guessing.
    pub fn parse(spec: &str) -> Self {
        if let Some(rest) = spec.strip_prefix("class:") {
            return Self::Class(rest.to_string());
        }
        if let Some(rest) = spec.strip_prefix("any:") {
            return Self::Class(rest.to_string());
        }
        Self::Named(spec.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitPolicy {
    Nowait,
    Wait { timeout: Duration },
}

/// Metadata about the lease holder.  Stored on the lease for observability
/// and to ensure `release_all` only releases what this holder actually owns.
#[derive(Debug, Clone)]
pub struct Holder {
    pub pane_id: String,
    pub tag: String,
    pub session_id: String,
}

#[derive(Debug)]
pub enum AcquireError {
    /// A Named request asked for a resource that does not exist in the pool.
    UnknownName(String),
    /// A Class request asked for a class with zero members in the pool.
    UnknownClass(String),
    /// All requests tried but capacity is exhausted (Nowait) or the wait
    /// timed out.  Message names which request could not be satisfied.
    Unavailable(String),
    /// Other I/O / filesystem error.
    Io(anyhow::Error),
}

impl std::fmt::Display for AcquireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownName(n) => write!(
                f,
                "resource {n:?} is not defined in ~/.amaebi/resources.toml"
            ),
            Self::UnknownClass(c) => write!(
                f,
                "no resources of class {c:?} defined in ~/.amaebi/resources.toml"
            ),
            Self::Unavailable(msg) => write!(f, "resource unavailable: {msg}"),
            Self::Io(e) => write!(f, "resource lease I/O error: {e:#}"),
        }
    }
}

impl std::error::Error for AcquireError {}

// ---------------------------------------------------------------------------
// On-disk state I/O (flock-protected)
// ---------------------------------------------------------------------------

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn state_path() -> Result<PathBuf> {
    Ok(amaebi_home()?.join("resource-state.json"))
}

fn lock_path() -> Result<PathBuf> {
    Ok(amaebi_home()?.join("resource-state.lock"))
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

fn read_state_unlocked() -> Result<ResourceState> {
    let path = state_path()?;
    if !path.exists() {
        return Ok(ResourceState::new());
    }
    let contents =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    if contents.trim().is_empty() {
        return Ok(ResourceState::new());
    }
    match serde_json::from_str(&contents) {
        Ok(state) => Ok(state),
        Err(e) => {
            // Corrupt state is dangerous for resource leases: treating it as
            // empty would drop every Busy lease and allow the next acquirer
            // to grab a resource another pane still holds in memory (double
            // allocation of single-holder hardware).  Quarantine the file
            // instead so the next write starts clean, and log loudly so
            // operators see it in the daemon log.  Acquisitions continue
            // against an empty state — correct when the corruption predates
            // any in-flight leases, and the quarantined file preserves
            // evidence for post-mortem review.
            let quarantine = path.with_extension("json.corrupt");
            match std::fs::rename(&path, &quarantine) {
                Ok(()) => tracing::error!(
                    state_file = %path.display(),
                    quarantined = %quarantine.display(),
                    error = %e,
                    "resource-state.json is corrupt; quarantined and resetting to empty state. \
                     Review the quarantined file and reconcile with `amaebi resource list` \
                     to confirm no leases were lost."
                ),
                Err(rename_err) => tracing::error!(
                    state_file = %path.display(),
                    error = %e,
                    rename_error = %rename_err,
                    "resource-state.json is corrupt AND could not be quarantined; \
                     continuing with empty in-memory state"
                ),
            }
            Ok(ResourceState::new())
        }
    }
}

fn write_state_unlocked(state: &ResourceState) -> Result<()> {
    let path = state_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating directory {}", parent.display()))?;
    }
    let contents = serde_json::to_string_pretty(state)?;
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

/// Read the current lease state (shared lock, read-only).
pub fn read_state() -> Result<ResourceState> {
    let lock = open_lock_file()?;
    lock.lock_shared()
        .context("acquiring shared flock for read_state")?;
    let state = read_state_unlocked()?;
    lock.unlock().context("releasing flock after read_state")?;
    Ok(state)
}

// ---------------------------------------------------------------------------
// Notify registry (in-process waiters)
// ---------------------------------------------------------------------------

/// Single process-wide `Notify` for release events.  Every waiter observes
/// the same signal and re-checks lease state under flock.  A per-resource
/// notify would be lower contention but would also need to synchronise with
/// the flock-held JSON file across multiple threads; using one wakeup and
/// letting everyone retry keeps the code simple while still correct.
fn release_notifier() -> &'static Arc<Notify> {
    static NOTIFIER: OnceLock<Arc<Notify>> = OnceLock::new();
    NOTIFIER.get_or_init(|| Arc::new(Notify::new()))
}

// ---------------------------------------------------------------------------
// Acquire / release
// ---------------------------------------------------------------------------

/// Merge the TOML pool into the on-disk state, adding Idle entries for any
/// newly-defined resources.  Called inside the flock before each acquisition
/// attempt so edits to `resources.toml` take effect without restarting the
/// daemon.  Resources removed from the pool are left alone if still Busy (to
/// avoid yanking a lease out from under a running holder) and pruned only
/// when Idle.
fn reconcile_pool_locked(pool: &[ResourceDef], state: &mut ResourceState) {
    let defined: std::collections::HashSet<&str> = pool.iter().map(|d| d.name.as_str()).collect();

    for def in pool {
        state
            .entry(def.name.clone())
            .and_modify(|lease| {
                // Class may have been edited in the TOML; follow the latest.
                lease.class = def.class.clone();
            })
            .or_insert_with(|| ResourceLease::new_idle(def));
    }

    // Drop Idle leases for resources no longer in the pool.  Busy leases
    // survive until their holder releases (or TTL expires).
    state.retain(|name, lease| {
        defined.contains(name.as_str()) || lease.effective_status() == ResourceStatus::Busy
    });
}

/// Attempt one acquisition pass under an already-held flock.  Returns the
/// list of acquired names on success, or the canonical name of the first
/// request that could not be satisfied on failure (the caller decides
/// whether to wait or error out).
///
/// All matches follow canonical ordering: Named requests are handled first
/// (in name order), then Class requests (in (class, name) order).  Within a
/// Class request, resources are chosen in deterministic name order so two
/// concurrent callers always converge on the same "first idle".
fn try_acquire_locked(
    pool: &[ResourceDef],
    requests_sorted: &[ResourceRequest],
    holder: &Holder,
    state: &mut ResourceState,
) -> std::result::Result<Vec<String>, AcquireError> {
    reconcile_pool_locked(pool, state);

    let mut acquired: Vec<String> = Vec::with_capacity(requests_sorted.len());

    for req in requests_sorted {
        let picked: Option<String> = match req {
            ResourceRequest::Named(name) => {
                let Some(lease) = state.get(name) else {
                    // Roll back any partial holds on this pass before
                    // returning — the outer loop hasn't finished, so nothing
                    // has been written to disk yet, but `acquired` tracks
                    // what we mutated in `state`.
                    for n in &acquired {
                        if let Some(l) = state.get_mut(n) {
                            l.status = ResourceStatus::Idle;
                            l.pane_id = None;
                            l.tag = None;
                            l.session_id = None;
                        }
                    }
                    return Err(AcquireError::UnknownName(name.clone()));
                };
                if lease.effective_status() == ResourceStatus::Idle {
                    Some(name.clone())
                } else {
                    None
                }
            }
            ResourceRequest::Class(class) => {
                // Pool membership check: unknown class is a config error.
                let class_has_members = pool.iter().any(|d| d.class == *class);
                if !class_has_members {
                    for n in &acquired {
                        if let Some(l) = state.get_mut(n) {
                            l.status = ResourceStatus::Idle;
                            l.pane_id = None;
                            l.tag = None;
                            l.session_id = None;
                        }
                    }
                    return Err(AcquireError::UnknownClass(class.clone()));
                }
                // Pick the lowest-name idle resource in this class, skipping
                // any already acquired on this pass so `Class("gpu")` twice
                // in one request picks two different GPUs.
                let mut candidates: Vec<&ResourceLease> = state
                    .values()
                    .filter(|l| {
                        l.class == *class
                            && l.effective_status() == ResourceStatus::Idle
                            && !acquired.contains(&l.name)
                    })
                    .collect();
                candidates.sort_by(|a, b| a.name.cmp(&b.name));
                candidates.first().map(|l| l.name.clone())
            }
        };

        match picked {
            Some(name) => {
                let now = now_secs();
                let lease = state.get_mut(&name).expect("candidate just found in state");
                lease.status = ResourceStatus::Busy;
                lease.pane_id = Some(holder.pane_id.clone());
                lease.tag = Some(holder.tag.clone());
                lease.session_id = Some(holder.session_id.clone());
                lease.heartbeat_at = now;
                acquired.push(name);
            }
            None => {
                // Partial hold — roll back in-memory and signal the outer
                // loop which request blocked.  Nothing has been written to
                // disk yet.
                for n in &acquired {
                    if let Some(l) = state.get_mut(n) {
                        l.status = ResourceStatus::Idle;
                        l.pane_id = None;
                        l.tag = None;
                        l.session_id = None;
                    }
                }
                let blocking = match req {
                    ResourceRequest::Named(n) => format!("name={n}"),
                    ResourceRequest::Class(c) => format!("class={c}"),
                };
                return Err(AcquireError::Unavailable(blocking));
            }
        }
    }

    Ok(acquired)
}

/// Sort requests into canonical order so concurrent callers never interleave
/// into a cycle.  Named first (by name), then Class (by class name).  Stable
/// within equal keys to preserve multiplicity (`Class("gpu")` twice yields
/// two slots).
fn canonicalize(requests: &[ResourceRequest]) -> Vec<ResourceRequest> {
    let mut sorted: Vec<ResourceRequest> = requests.to_vec();
    sorted.sort_by(|a, b| {
        let key = |r: &ResourceRequest| match r {
            ResourceRequest::Named(n) => (0u8, n.clone()),
            ResourceRequest::Class(c) => (1u8, c.clone()),
        };
        key(a).cmp(&key(b))
    });
    sorted
}

/// Acquire all requested resources or none (all-or-nothing).  Writes state
/// to disk only on full success.  On Nowait failure, returns immediately.
/// On Wait failure, awaits the release notifier (with timeout) and retries.
pub async fn acquire_all(
    requests: &[ResourceRequest],
    holder: Holder,
    wait: WaitPolicy,
) -> std::result::Result<Vec<ResourceLease>, AcquireError> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    let canonical = canonicalize(requests);
    let deadline = match wait {
        WaitPolicy::Nowait => None,
        WaitPolicy::Wait { timeout } => Some(tokio::time::Instant::now() + timeout),
    };

    loop {
        // Subscribe to release notifications BEFORE the flock attempt so we
        // don't miss a release that happens between the failed attempt and
        // the .notified() await.
        let notifier = release_notifier().clone();
        let notified = notifier.notified();
        tokio::pin!(notified);

        let holder_clone = holder.clone();
        let canonical_clone = canonical.clone();

        // All filesystem I/O + TOML parse + JSON parse + flock happens
        // inside one spawn_blocking so the async executor thread never
        // stalls on disk.  `load_pool` is called here (under the flock)
        // so it also picks up TOML edits made between retries — matches
        // the reconcile-on-every-attempt contract documented on
        // `reconcile_pool_locked`.
        let attempt = tokio::task::spawn_blocking(
            move || -> std::result::Result<Vec<ResourceLease>, AcquireError> {
                let lock = open_lock_file().map_err(AcquireError::Io)?;
                lock.lock_exclusive()
                    .context("acquiring flock for acquire_all")
                    .map_err(AcquireError::Io)?;

                let result = (|| -> std::result::Result<Vec<ResourceLease>, AcquireError> {
                    let pool = load_pool().map_err(AcquireError::Io)?;
                    let mut state = read_state_unlocked().map_err(AcquireError::Io)?;
                    let acquired_names =
                        try_acquire_locked(&pool, &canonical_clone, &holder_clone, &mut state)?;
                    write_state_unlocked(&state).map_err(AcquireError::Io)?;
                    let leases = acquired_names
                        .iter()
                        .filter_map(|n| state.get(n).cloned())
                        .collect();
                    Ok(leases)
                })();

                let _ = lock.unlock();
                result
            },
        )
        .await
        .map_err(|e| AcquireError::Io(anyhow::anyhow!("acquire_all task panicked: {e}")))?;

        match attempt {
            Ok(leases) => return Ok(leases),
            Err(AcquireError::Unavailable(which)) => match deadline {
                None => return Err(AcquireError::Unavailable(which)),
                Some(d) => {
                    let now = tokio::time::Instant::now();
                    if now >= d {
                        return Err(AcquireError::Unavailable(format!(
                            "{which} (waited up to timeout)"
                        )));
                    }
                    let remaining = d - now;
                    // Poll tick bounds the wait even when no explicit release
                    // happens — e.g. the holder crashed without running the
                    // supervision-exit release path.  In that case the only
                    // path back to Idle is `effective_status`'s TTL check,
                    // which is observed only when `try_acquire_locked` runs.
                    // Without this tick, such leases stay blocked until
                    // `timeout` elapses even if TTL already expired minutes
                    // into the wait.  5 s keeps CPU noise negligible while
                    // cutting TTL-recovery latency from hours to seconds.
                    let tick = remaining.min(std::time::Duration::from_secs(5));
                    tokio::select! {
                        _ = notified.as_mut() => {
                            // A lease was released somewhere — retry.
                            continue;
                        }
                        _ = tokio::time::sleep(tick) => {
                            if tokio::time::Instant::now() >= d {
                                return Err(AcquireError::Unavailable(format!(
                                    "{which} (waited up to timeout)"
                                )));
                            }
                            // Tick fired short of the deadline — retry in case
                            // a TTL expiry made a resource reclaimable.
                            continue;
                        }
                    }
                }
            },
            Err(other) => return Err(other),
        }
    }
}

/// Release every lease currently held by `pane_id`, best-effort.  Matches
/// by pane because `release_all` is called from the supervision-exit path
/// which only knows the pane id, not which resources it held.  Releasing by
/// pane (rather than by a caller-supplied list) makes the path robust to
/// partial acquisition failures where some leases were taken and some
/// weren't.
pub async fn release_all_for_pane(pane_id: &str) -> Result<Vec<String>> {
    let pane = pane_id.to_string();
    let released: Vec<String> = tokio::task::spawn_blocking(move || -> Result<Vec<String>> {
        let lock = open_lock_file()?;
        lock.lock_exclusive()
            .context("acquiring flock for release_all_for_pane")?;
        let result = (|| -> Result<Vec<String>> {
            let mut state = read_state_unlocked()?;
            let mut released = Vec::new();
            for lease in state.values_mut() {
                if lease.pane_id.as_deref() == Some(pane.as_str()) {
                    lease.status = ResourceStatus::Idle;
                    lease.pane_id = None;
                    lease.tag = None;
                    lease.session_id = None;
                    lease.heartbeat_at = now_secs();
                    released.push(lease.name.clone());
                }
            }
            write_state_unlocked(&state)?;
            Ok(released)
        })();
        let _ = lock.unlock();
        result
    })
    .await
    .map_err(|e| anyhow::anyhow!("release_all_for_pane task panicked: {e}"))??;

    // Wake any waiters so they can retry acquisition.
    if !released.is_empty() {
        release_notifier().notify_waiters();
    }
    Ok(released)
}

// ---------------------------------------------------------------------------
// Placeholder substitution (env vars + prompt_hint)
// ---------------------------------------------------------------------------

/// Substitute `{placeholder}` tokens in `template` from a resource's
/// metadata plus the implicit `{name}` and `{class}` keys.  Unknown
/// placeholders are left as-is so a typo surfaces in the pane/prompt
/// instead of silently rendering an empty string.
pub fn render_placeholders(template: &str, def: &ResourceDef) -> String {
    let mut out = String::with_capacity(template.len());
    // Iterate over char boundaries so multi-byte UTF-8 (e.g. CJK in
    // `prompt_hint`) passes through unbroken.  ASCII-only indexing would
    // panic when a placeholder is embedded in non-ASCII surroundings.
    let mut rest = template;
    while !rest.is_empty() {
        if let Some(open) = rest.find('{') {
            out.push_str(&rest[..open]);
            let after_open = &rest[open + 1..];
            if let Some(close) = after_open.find('}') {
                let key = &after_open[..close];
                let replacement: Option<&str> = match key {
                    "name" => Some(def.name.as_str()),
                    "class" => Some(def.class.as_str()),
                    k => def.metadata.get(k).map(String::as_str),
                };
                if let Some(v) = replacement {
                    out.push_str(v);
                    rest = &after_open[close + 1..];
                    continue;
                }
                // Unknown placeholder: keep `{key}` literal and continue
                // scanning after the closing brace.
                out.push('{');
                out.push_str(key);
                out.push('}');
                rest = &after_open[close + 1..];
            } else {
                // Unclosed `{` — emit verbatim and stop scanning.
                out.push('{');
                rest = after_open;
            }
        } else {
            out.push_str(rest);
            break;
        }
    }
    out
}

/// True iff `name` is a valid POSIX-style environment variable identifier
/// (`[A-Za-z_][A-Za-z0-9_]*`).  Keys that don't match are rejected by
/// [`render_env`] rather than injected into the shell command — injecting
/// names containing spaces, `;`, `&`, or `$(...)` would let a malformed
/// `resources.toml` entry break pane launch or, worse, run arbitrary shell
/// code via the `export KEY=VALUE && ...` prefix the daemon builds.
fn is_valid_env_key(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Render env-var assignments for a collection of leased resources.
/// Returns `(KEY, VALUE)` pairs suitable for `KEY=VALUE` shell prefixes.
/// Resources with no `env` map contribute nothing; variables missing from
/// the pool (i.e. the lease references a resource removed after
/// acquisition) are skipped with a warning.  Keys that aren't valid POSIX
/// env-var identifiers are rejected with an error log and dropped — the
/// rendered value is shell-interpolated by the daemon, so letting a bad
/// key through would break pane launch or open an injection vector.
pub fn render_env(leases: &[ResourceLease], pool: &[ResourceDef]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for lease in leases {
        let Some(def) = pool.iter().find(|d| d.name == lease.name) else {
            tracing::warn!(
                resource = %lease.name,
                "resource in lease state but missing from pool; skipping env injection"
            );
            continue;
        };
        for (k, v) in &def.env {
            if !is_valid_env_key(k) {
                tracing::error!(
                    resource = %lease.name,
                    key = %k,
                    "invalid env var name in resources.toml (must match [A-Za-z_][A-Za-z0-9_]*); \
                     skipping this entry to avoid breaking the pane launch command"
                );
                continue;
            }
            out.push((k.clone(), render_placeholders(v, def)));
        }
    }
    out
}

/// Render the concatenated prompt hint for a collection of leased
/// resources.  Returns empty string when no resource has a `prompt_hint`,
/// which lets callers cheaply skip the preamble branch.
pub fn render_prompt_hint(leases: &[ResourceLease], pool: &[ResourceDef]) -> String {
    let mut out = String::new();
    for lease in leases {
        let Some(def) = pool.iter().find(|d| d.name == lease.name) else {
            continue;
        };
        let Some(template) = def.prompt_hint.as_deref() else {
            continue;
        };
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&render_placeholders(template, def));
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn def(name: &str, class: &str) -> ResourceDef {
        ResourceDef {
            name: name.to_string(),
            class: class.to_string(),
            metadata: HashMap::new(),
            env: HashMap::new(),
            prompt_hint: None,
        }
    }

    fn holder(pane: &str) -> Holder {
        Holder {
            pane_id: pane.to_string(),
            tag: "t".to_string(),
            session_id: "s".to_string(),
        }
    }

    fn seed_pool_file(pool: &[ResourceDef]) -> Result<()> {
        let path = pool_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut s = String::new();
        for r in pool {
            s.push_str("[[resource]]\n");
            s.push_str(&format!("name = {:?}\n", r.name));
            s.push_str(&format!("class = {:?}\n", r.class));
            if !r.metadata.is_empty() {
                s.push_str("metadata = { ");
                let parts: Vec<_> = r
                    .metadata
                    .iter()
                    .map(|(k, v)| format!("{k} = {v:?}"))
                    .collect();
                s.push_str(&parts.join(", "));
                s.push_str(" }\n");
            }
            if !r.env.is_empty() {
                s.push_str("env = { ");
                let parts: Vec<_> = r.env.iter().map(|(k, v)| format!("{k} = {v:?}")).collect();
                s.push_str(&parts.join(", "));
                s.push_str(" }\n");
            }
            if let Some(hint) = &r.prompt_hint {
                s.push_str(&format!("prompt_hint = {hint:?}\n"));
            }
            s.push('\n');
        }
        std::fs::write(path, s)?;
        Ok(())
    }

    // ── ResourceLease serialization ────────────────────────────────────────

    #[test]
    fn lease_round_trip() {
        let l = ResourceLease {
            name: "sim-9900".into(),
            class: "simulator".into(),
            status: ResourceStatus::Busy,
            pane_id: Some("%3".into()),
            tag: Some("t1".into()),
            session_id: Some("s1".into()),
            heartbeat_at: 1234,
        };
        let json = serde_json::to_string(&l).expect("serialize");
        let back: ResourceLease = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.name, "sim-9900");
        assert_eq!(back.status, ResourceStatus::Busy);
    }

    #[test]
    fn effective_status_expires_to_idle() {
        let mut l = ResourceLease::new_idle(&def("r", "c"));
        l.status = ResourceStatus::Busy;
        l.heartbeat_at = now_secs().saturating_sub(LEASE_TTL_SECS + 1);
        assert_eq!(l.effective_status(), ResourceStatus::Idle);
    }

    // ── Pool loading ───────────────────────────────────────────────────────

    #[test]
    fn load_pool_missing_file_returns_empty() {
        let _guard = crate::test_utils::with_temp_home();
        let pool = load_pool().expect("load");
        assert!(pool.is_empty());
    }

    #[test]
    fn load_pool_rejects_duplicate_names() {
        let _guard = crate::test_utils::with_temp_home();
        let dup = vec![def("r0", "gpu"), def("r0", "gpu")];
        seed_pool_file(&dup).expect("seed");
        let err = load_pool().expect_err("must reject");
        assert!(format!("{err:#}").contains("duplicate"), "got: {err:#}");
    }

    // ── Canonical ordering ─────────────────────────────────────────────────

    #[test]
    fn resource_request_parse_distinguishes_named_and_class() {
        assert_eq!(
            ResourceRequest::parse("sim-9900"),
            ResourceRequest::Named("sim-9900".into())
        );
        assert_eq!(
            ResourceRequest::parse("class:gpu"),
            ResourceRequest::Class("gpu".into())
        );
        assert_eq!(
            ResourceRequest::parse("any:gpu"),
            ResourceRequest::Class("gpu".into())
        );
    }

    #[test]
    fn canonicalize_sorts_named_before_class() {
        let reqs = vec![
            ResourceRequest::Class("gpu".into()),
            ResourceRequest::Named("sim-9900".into()),
            ResourceRequest::Class("gpu".into()),
            ResourceRequest::Named("abc".into()),
        ];
        let sorted = canonicalize(&reqs);
        assert_eq!(
            sorted,
            vec![
                ResourceRequest::Named("abc".into()),
                ResourceRequest::Named("sim-9900".into()),
                ResourceRequest::Class("gpu".into()),
                ResourceRequest::Class("gpu".into()),
            ]
        );
    }

    // ── acquire/release integration (single task runtime to keep Notify) ───

    #[tokio::test(flavor = "current_thread")]
    async fn acquire_named_then_release_roundtrips() {
        let _guard = crate::test_utils::with_temp_home();
        seed_pool_file(&[def("sim-9900", "simulator")]).expect("seed");

        let leases = acquire_all(
            &[ResourceRequest::Named("sim-9900".into())],
            holder("%3"),
            WaitPolicy::Nowait,
        )
        .await
        .expect("acquire");
        assert_eq!(leases.len(), 1);
        assert_eq!(leases[0].name, "sim-9900");
        assert_eq!(leases[0].status, ResourceStatus::Busy);

        let released = release_all_for_pane("%3").await.expect("release");
        assert_eq!(released, vec!["sim-9900".to_string()]);

        let state = read_state().expect("read");
        assert_eq!(state["sim-9900"].effective_status(), ResourceStatus::Idle);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn acquire_class_picks_any_idle() {
        let _guard = crate::test_utils::with_temp_home();
        seed_pool_file(&[def("sim-9900", "simulator"), def("sim-9901", "simulator")])
            .expect("seed");

        let first = acquire_all(
            &[ResourceRequest::Class("simulator".into())],
            holder("%3"),
            WaitPolicy::Nowait,
        )
        .await
        .expect("acquire first");
        let second = acquire_all(
            &[ResourceRequest::Class("simulator".into())],
            holder("%4"),
            WaitPolicy::Nowait,
        )
        .await
        .expect("acquire second");

        assert_ne!(
            first[0].name, second[0].name,
            "must pick different resources"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn acquire_class_full_capacity_nowait_fails() {
        let _guard = crate::test_utils::with_temp_home();
        seed_pool_file(&[def("sim-9900", "simulator")]).expect("seed");

        acquire_all(
            &[ResourceRequest::Class("simulator".into())],
            holder("%3"),
            WaitPolicy::Nowait,
        )
        .await
        .expect("acquire first");

        let err = acquire_all(
            &[ResourceRequest::Class("simulator".into())],
            holder("%4"),
            WaitPolicy::Nowait,
        )
        .await
        .expect_err("second must fail");
        assert!(matches!(err, AcquireError::Unavailable(_)), "got: {err:?}");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn acquire_all_or_nothing_rolls_back_on_partial() {
        // Two resources in separate classes; caller asks for one of each.
        // Occupy the second one in advance so the Class(b) request fails,
        // and verify the Named(a) request is not left Busy.
        let _guard = crate::test_utils::with_temp_home();
        seed_pool_file(&[def("a", "classA"), def("b", "classB")]).expect("seed");
        acquire_all(
            &[ResourceRequest::Named("b".into())],
            holder("%8"),
            WaitPolicy::Nowait,
        )
        .await
        .expect("pre-occupy b");

        let err = acquire_all(
            &[
                ResourceRequest::Named("a".into()),
                ResourceRequest::Class("classB".into()),
            ],
            holder("%9"),
            WaitPolicy::Nowait,
        )
        .await
        .expect_err("mixed must fail");
        assert!(matches!(err, AcquireError::Unavailable(_)));

        let state = read_state().expect("read");
        assert_eq!(
            state["a"].effective_status(),
            ResourceStatus::Idle,
            "a must be rolled back after partial failure"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn acquire_unknown_name_errors() {
        let _guard = crate::test_utils::with_temp_home();
        seed_pool_file(&[def("sim-9900", "simulator")]).expect("seed");
        let err = acquire_all(
            &[ResourceRequest::Named("nope".into())],
            holder("%3"),
            WaitPolicy::Nowait,
        )
        .await
        .expect_err("unknown name");
        assert!(matches!(err, AcquireError::UnknownName(_)));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn wait_times_out_when_no_release() {
        let _guard = crate::test_utils::with_temp_home();
        seed_pool_file(&[def("sim-9900", "simulator")]).expect("seed");
        acquire_all(
            &[ResourceRequest::Named("sim-9900".into())],
            holder("%3"),
            WaitPolicy::Nowait,
        )
        .await
        .expect("prime");

        let err = acquire_all(
            &[ResourceRequest::Named("sim-9900".into())],
            holder("%4"),
            WaitPolicy::Wait {
                timeout: Duration::from_millis(60),
            },
        )
        .await
        .expect_err("must time out");
        assert!(matches!(err, AcquireError::Unavailable(_)));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn wait_wakes_on_release() {
        let _guard = crate::test_utils::with_temp_home();
        seed_pool_file(&[def("sim-9900", "simulator")]).expect("seed");
        acquire_all(
            &[ResourceRequest::Named("sim-9900".into())],
            holder("%3"),
            WaitPolicy::Nowait,
        )
        .await
        .expect("prime");

        // Spawn the waiter first, then release after a short delay so the
        // notification fires while the waiter is parked.
        let waiter = tokio::spawn(async {
            acquire_all(
                &[ResourceRequest::Named("sim-9900".into())],
                Holder {
                    pane_id: "%4".into(),
                    tag: "t".into(),
                    session_id: "s".into(),
                },
                WaitPolicy::Wait {
                    timeout: Duration::from_secs(2),
                },
            )
            .await
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        release_all_for_pane("%3").await.expect("release");

        let leases = waiter.await.expect("join").expect("acquire after wait");
        assert_eq!(leases.len(), 1);
        assert_eq!(leases[0].pane_id.as_deref(), Some("%4"));
    }

    // ── render_placeholders ────────────────────────────────────────────────

    #[test]
    fn render_placeholders_substitutes_metadata_and_implicits() {
        let mut d = def("sim-9900", "simulator");
        d.metadata.insert("port".into(), "9900".into());
        d.metadata.insert("host".into(), "127.0.0.1".into());
        let rendered = render_placeholders("{name} on {host}:{port} (class={class})", &d);
        assert_eq!(rendered, "sim-9900 on 127.0.0.1:9900 (class=simulator)");
    }

    #[test]
    fn render_placeholders_leaves_unknown_keys_intact() {
        let d = def("r", "c");
        // Unknown placeholder stays verbatim so typos surface.
        assert_eq!(render_placeholders("hi {nobody}", &d), "hi {nobody}");
    }

    #[test]
    fn render_env_rejects_invalid_env_keys() {
        // Guards against a `resources.toml` entry that would break the
        // pane launch command or inject shell code via the `export K=V &&`
        // prefix (e.g. a key with `;` or spaces).  Invalid keys must be
        // dropped rather than propagated.
        let mut d = def("sim", "simulator");
        d.env.insert("GOOD_NAME".into(), "ok".into());
        d.env.insert("BAD NAME".into(), "nope".into());
        d.env.insert("also;bad".into(), "nope".into());
        d.env.insert("".into(), "nope".into());
        d.env.insert("1STARTS_WITH_DIGIT".into(), "nope".into());
        let lease = ResourceLease {
            name: "sim".into(),
            class: "simulator".into(),
            status: ResourceStatus::Busy,
            pane_id: Some("%3".into()),
            tag: None,
            session_id: None,
            heartbeat_at: 0,
        };
        let pairs = render_env(&[lease], &[d]);
        assert_eq!(pairs, vec![("GOOD_NAME".to_string(), "ok".to_string())]);
    }

    #[test]
    fn render_env_produces_expanded_pairs() {
        let mut d = def("sim-9900", "simulator");
        d.metadata.insert("port".into(), "9900".into());
        d.env.insert("SIM_PORT".into(), "{port}".into());
        d.env.insert("SIM_NAME".into(), "{name}".into());
        let lease = ResourceLease {
            name: "sim-9900".into(),
            class: "simulator".into(),
            status: ResourceStatus::Busy,
            pane_id: Some("%3".into()),
            tag: None,
            session_id: None,
            heartbeat_at: 0,
        };
        let mut pairs = render_env(&[lease], &[d]);
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                ("SIM_NAME".to_string(), "sim-9900".to_string()),
                ("SIM_PORT".to_string(), "9900".to_string()),
            ]
        );
    }

    #[test]
    fn render_prompt_hint_joins_multiple_leases() {
        let mut a = def("sim-9900", "simulator");
        a.prompt_hint = Some("A={name}".into());
        let mut b = def("gpu-0", "gpu");
        b.prompt_hint = Some("B={name}".into());
        let leases = vec![
            ResourceLease {
                name: "sim-9900".into(),
                class: "simulator".into(),
                status: ResourceStatus::Busy,
                pane_id: None,
                tag: None,
                session_id: None,
                heartbeat_at: 0,
            },
            ResourceLease {
                name: "gpu-0".into(),
                class: "gpu".into(),
                status: ResourceStatus::Busy,
                pane_id: None,
                tag: None,
                session_id: None,
                heartbeat_at: 0,
            },
        ];
        let hint = render_prompt_hint(&leases, &[a, b]);
        assert_eq!(hint, "A=sim-9900\nB=gpu-0");
    }

    // ── Functional regression tests ────────────────────────────────────────

    /// Two `class:gpu` requests in one call must acquire two **different**
    /// GPUs — otherwise both panes end up pointing at the same device and
    /// the whole feature is pointless.  This is the user's core scenario
    /// (5 simulator containers, 6 panes).
    #[tokio::test(flavor = "current_thread")]
    async fn acquire_multiple_class_requests_pick_distinct_resources() {
        let _guard = crate::test_utils::with_temp_home();
        seed_pool_file(&[
            def("gpu-0", "gpu"),
            def("gpu-1", "gpu"),
            def("gpu-2", "gpu"),
        ])
        .expect("seed");

        let leases = acquire_all(
            &[
                ResourceRequest::Class("gpu".into()),
                ResourceRequest::Class("gpu".into()),
            ],
            holder("%7"),
            WaitPolicy::Nowait,
        )
        .await
        .expect("two gpus");

        assert_eq!(leases.len(), 2);
        assert_ne!(
            leases[0].name, leases[1].name,
            "two Class requests in one acquire must pick different resources"
        );
    }

    /// `release_all_for_pane("%A")` must leave panes `%B`'s leases untouched.
    /// A bug here (e.g. matching the wrong field) would mass-release every
    /// lease on any supervision exit.
    #[tokio::test(flavor = "current_thread")]
    async fn release_by_pane_is_scoped_to_that_pane() {
        let _guard = crate::test_utils::with_temp_home();
        seed_pool_file(&[def("r-a", "x"), def("r-b", "x")]).expect("seed");

        acquire_all(
            &[ResourceRequest::Named("r-a".into())],
            holder("%A"),
            WaitPolicy::Nowait,
        )
        .await
        .expect("a");
        acquire_all(
            &[ResourceRequest::Named("r-b".into())],
            holder("%B"),
            WaitPolicy::Nowait,
        )
        .await
        .expect("b");

        let released = release_all_for_pane("%A").await.expect("release A");
        assert_eq!(released, vec!["r-a".to_string()]);

        let state = read_state().expect("read");
        assert_eq!(state["r-a"].effective_status(), ResourceStatus::Idle);
        assert_eq!(
            state["r-b"].effective_status(),
            ResourceStatus::Busy,
            "r-b must still be held by %B after releasing %A"
        );
        assert_eq!(state["r-b"].pane_id.as_deref(), Some("%B"));
    }

    /// A lease held past `LEASE_TTL_SECS` must be reclaimable — otherwise a
    /// crashed holder pins the resource for 24 h with no recovery path.
    /// This verifies the TTL downgrade is wired into the acquisition check,
    /// not just the `effective_status` accessor.
    #[tokio::test(flavor = "current_thread")]
    async fn expired_busy_lease_is_reclaimable_by_next_acquirer() {
        let _guard = crate::test_utils::with_temp_home();
        seed_pool_file(&[def("sim-9900", "simulator")]).expect("seed");

        // Seed a Busy lease with an expired heartbeat directly into state,
        // simulating a dead holder that never got to release.
        let lock = open_lock_file().expect("lock");
        lock.lock_exclusive().expect("flock");
        let mut state = read_state_unlocked().expect("read");
        state.insert(
            "sim-9900".into(),
            ResourceLease {
                name: "sim-9900".into(),
                class: "simulator".into(),
                status: ResourceStatus::Busy,
                pane_id: Some("%ghost".into()),
                tag: Some("dead".into()),
                session_id: Some("dead".into()),
                heartbeat_at: now_secs().saturating_sub(LEASE_TTL_SECS + 1),
            },
        );
        write_state_unlocked(&state).expect("write");
        lock.unlock().expect("unlock");

        let leases = acquire_all(
            &[ResourceRequest::Named("sim-9900".into())],
            holder("%new"),
            WaitPolicy::Nowait,
        )
        .await
        .expect("reclaim must succeed past TTL");
        assert_eq!(leases[0].pane_id.as_deref(), Some("%new"));
    }

    /// End-to-end pipeline: a TOML pool with the user's simulator shape
    /// (`{port}` metadata → `$SIM_PORT` env + prompt_hint), followed by a
    /// realistic `parse`-based request list, renders env/hint correctly
    /// and releases everything on pane exit.  This is the one test that
    /// exercises every seam between module boundaries.
    #[tokio::test(flavor = "current_thread")]
    async fn end_to_end_simulator_scenario() {
        let _guard = crate::test_utils::with_temp_home();
        let mut sim0 = def("sim-9900", "simulator");
        sim0.metadata.insert("port".into(), "9900".into());
        sim0.env.insert("SIM_PORT".into(), "{port}".into());
        sim0.prompt_hint = Some("use port {port} only".into());
        let mut sim1 = def("sim-9901", "simulator");
        sim1.metadata.insert("port".into(), "9901".into());
        sim1.env.insert("SIM_PORT".into(), "{port}".into());
        sim1.prompt_hint = Some("use port {port} only".into());
        seed_pool_file(&[sim0, sim1]).expect("seed");

        // Specs as they would arrive from `parse_claude` (strings).
        let specs = ["class:simulator"];
        let requests: Vec<ResourceRequest> =
            specs.iter().map(|s| ResourceRequest::parse(s)).collect();
        assert_eq!(requests, vec![ResourceRequest::Class("simulator".into())]);

        let leases = acquire_all(&requests, holder("%42"), WaitPolicy::Nowait)
            .await
            .expect("acquire");
        assert_eq!(leases.len(), 1);
        let picked_name = leases[0].name.clone();
        assert!(picked_name.starts_with("sim-"), "got {picked_name}");

        // Env + prompt_hint must render the *picked* resource's values.
        let pool = load_pool().expect("reload pool");
        let env = render_env(&leases, &pool);
        assert_eq!(env.len(), 1);
        assert_eq!(env[0].0, "SIM_PORT");
        assert!(
            env[0].1 == "9900" || env[0].1 == "9901",
            "SIM_PORT should be the picked port, got {:?}",
            env[0].1
        );
        let hint = render_prompt_hint(&leases, &pool);
        assert!(hint.contains(&env[0].1), "hint must mention the port");
        assert!(hint.contains("use port"));

        // Release by pane — full lifecycle closes.
        let released = release_all_for_pane("%42").await.expect("release");
        assert_eq!(released, vec![picked_name.clone()]);
        let state = read_state().expect("read");
        assert_eq!(
            state[&picked_name].effective_status(),
            ResourceStatus::Idle,
            "resource must be available for the next caller"
        );
    }

    /// `Wait` must observe TTL-based recoveries even when no explicit
    /// `release_all_for_pane` fires — regression for the "holder crashed
    /// without releasing" case.  We seed an already-expired Busy lease
    /// and kick off a waiter with a timeout longer than the poll tick;
    /// without the periodic re-check the waiter would only notice at the
    /// deadline.
    #[tokio::test(flavor = "current_thread")]
    async fn wait_observes_ttl_expiry_without_explicit_release() {
        let _guard = crate::test_utils::with_temp_home();
        seed_pool_file(&[def("sim-9900", "simulator")]).expect("seed");

        // Seed an expired Busy lease directly so no one will ever release
        // it explicitly — the only way the waiter can succeed is by
        // observing that `effective_status` has downgraded it to Idle.
        let lock = open_lock_file().expect("lock");
        lock.lock_exclusive().expect("flock");
        let mut state = read_state_unlocked().expect("read");
        state.insert(
            "sim-9900".into(),
            ResourceLease {
                name: "sim-9900".into(),
                class: "simulator".into(),
                status: ResourceStatus::Busy,
                pane_id: Some("%ghost".into()),
                tag: Some("dead".into()),
                session_id: Some("dead".into()),
                heartbeat_at: now_secs().saturating_sub(LEASE_TTL_SECS + 1),
            },
        );
        write_state_unlocked(&state).expect("write");
        lock.unlock().expect("unlock");

        // The waiter's timeout (30 s) is much longer than the poll tick
        // (5 s), so if the tick works we succeed within a few seconds.
        // Wrap in an outer watchdog so a broken tick doesn't hang CI.
        // Keep `requests` in a let-binding so its lifetime covers the
        // tokio::time::timeout await below.
        let requests = vec![ResourceRequest::Named("sim-9900".into())];
        let acquire_fut = acquire_all(
            &requests,
            holder("%new"),
            WaitPolicy::Wait {
                timeout: Duration::from_secs(30),
            },
        );
        let leases = tokio::time::timeout(Duration::from_secs(10), acquire_fut)
            .await
            .expect("TTL tick must fire before outer watchdog")
            .expect("acquire must reclaim expired lease");
        assert_eq!(leases[0].pane_id.as_deref(), Some("%new"));
    }

    /// Corrupt `resource-state.json` must be quarantined (not silently
    /// reset) so double-allocation cannot happen and operators retain
    /// evidence for post-mortem.  Subsequent acquisitions continue against
    /// a fresh empty state.
    #[tokio::test(flavor = "current_thread")]
    async fn corrupt_state_file_is_quarantined_not_silently_dropped() {
        let _guard = crate::test_utils::with_temp_home();
        seed_pool_file(&[def("sim-9900", "simulator")]).expect("seed pool");

        // Write garbage to the state file where valid JSON is expected.
        let state = state_path().expect("path");
        std::fs::create_dir_all(state.parent().unwrap()).expect("dir");
        std::fs::write(&state, b"{not json at all").expect("write garbage");

        // Acquisition must succeed (reading corrupt state returns empty).
        let leases = acquire_all(
            &[ResourceRequest::Named("sim-9900".into())],
            holder("%1"),
            WaitPolicy::Nowait,
        )
        .await
        .expect("acquire succeeds against reset state");
        assert_eq!(leases[0].name, "sim-9900");

        // The corrupt file must have been renamed to `.corrupt` — so an
        // operator can inspect it — and the live state must be fresh JSON.
        let quarantine = state.with_extension("json.corrupt");
        assert!(
            quarantine.exists(),
            "corrupt file must be quarantined for post-mortem"
        );
        let live = std::fs::read_to_string(&state).expect("read live state");
        assert!(
            live.contains("sim-9900"),
            "live state must be rewritten with the new lease, got: {live}"
        );
    }

    /// Adding a resource to the TOML while the daemon is running must make
    /// it available on the next acquisition attempt (reconciliation under
    /// flock).  Regression: a caching bug here would force a daemon
    /// restart after every `resources.toml` edit.
    #[tokio::test(flavor = "current_thread")]
    async fn new_pool_entry_becomes_available_after_toml_edit() {
        let _guard = crate::test_utils::with_temp_home();
        seed_pool_file(&[def("r-a", "x")]).expect("initial seed");

        // Baseline: only r-a exists, r-b is unknown.
        let err = acquire_all(
            &[ResourceRequest::Named("r-b".into())],
            holder("%1"),
            WaitPolicy::Nowait,
        )
        .await
        .expect_err("r-b unknown");
        assert!(matches!(err, AcquireError::UnknownName(_)));

        // Edit the pool: add r-b.  Simulates the user editing resources.toml.
        seed_pool_file(&[def("r-a", "x"), def("r-b", "x")]).expect("re-seed");

        let leases = acquire_all(
            &[ResourceRequest::Named("r-b".into())],
            holder("%1"),
            WaitPolicy::Nowait,
        )
        .await
        .expect("r-b available after pool edit");
        assert_eq!(leases[0].name, "r-b");
    }
}
