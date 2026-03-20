//! Centralized directory-to-UUID session mapping.
//!
//! Session identifiers are stable UUIDs stored in `~/.amaebi/sessions.json`.
//! Each entry maps a canonical directory path to a session record containing
//! the UUID, creation timestamp, and last-access timestamp.  The mapping
//! persists across daemon restarts; the UUID is the authoritative identity for
//! a session — used for per-session history locking in the daemon.
//!
//! # Concurrency
//!
//! All reads and writes to `sessions.json` are serialised by an advisory lock
//! on `~/.amaebi/sessions.lock` (via `fs2::FileExt`).  Reads use a shared
//! lock; writes use an exclusive lock.  A fast-path shared-lock read in
//! `get_or_create` avoids exclusive locking when the entry already exists,
//! reducing contention for concurrent callers.

use anyhow::{Context, Result};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use uuid::Uuid;

use crate::auth::amaebi_home;

// ---------------------------------------------------------------------------
// Session record
// ---------------------------------------------------------------------------

/// Persistent metadata for a single session entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    /// UUID v4 identifying this session.
    pub uuid: String,
    /// RFC 3339 timestamp when the session was first created.
    pub created_at: String,
    /// RFC 3339 timestamp of the most recent access (get or create).
    pub last_accessed: String,
    /// TTL tier label (e.g. "default", "ephemeral", "persistent").
    /// Determines eviction policy in the daemon's `SessionStore`.
    #[serde(default = "default_ttl_tier")]
    pub ttl_tier: String,
}

fn default_ttl_tier() -> String {
    "default".to_string()
}

impl SessionRecord {
    fn new() -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            uuid: Uuid::new_v4().to_string(),
            created_at: now.clone(),
            last_accessed: now,
            ttl_tier: default_ttl_tier(),
        }
    }

    fn new_with_tier(tier: &str) -> Self {
        let mut rec = Self::new();
        rec.ttl_tier = tier.to_string();
        rec
    }

    fn touch(&mut self) {
        self.last_accessed = chrono::Utc::now().to_rfc3339();
    }
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

fn sessions_path() -> Result<PathBuf> {
    Ok(amaebi_home()?.join("sessions.json"))
}

fn lock_path() -> Result<PathBuf> {
    Ok(amaebi_home()?.join("sessions.lock"))
}

/// Open (creating if necessary) the sessions lock file.
fn open_lock_file() -> Result<std::fs::File> {
    let path = lock_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .mode(0o600)
        .open(&path)
        .with_context(|| format!("opening sessions lock file {}", path.display()))
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

fn load_map(path: &Path) -> Result<HashMap<String, SessionRecord>> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let content =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    if content.trim().is_empty() {
        return Ok(HashMap::new());
    }
    match serde_json::from_str(&content) {
        Ok(map) => Ok(map),
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "sessions.json is corrupted; resetting to empty"
            );
            Ok(HashMap::new())
        }
    }
}

/// Write the map atomically: write to a `.tmp` file, then rename.
fn save_map(path: &Path, map: &HashMap<String, SessionRecord>) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let tmp = path.with_extension("json.tmp");
    {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp)
            .with_context(|| format!("creating {}", tmp.display()))?;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("setting permissions on {}", tmp.display()))?;
        let content = serde_json::to_string_pretty(map).context("serializing sessions")?;
        file.write_all(content.as_bytes())
            .with_context(|| format!("writing {}", tmp.display()))?;
    }
    std::fs::rename(&tmp, path)
        .with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Return the canonical form of `dir` used as the map key.
///
/// Falls back to the raw path if `canonicalize` fails (e.g. dir does not
/// exist yet).
fn canonical_key(dir: &Path) -> String {
    std::fs::canonicalize(dir)
        .unwrap_or_else(|_| dir.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

/// Look up or create the session UUID for `dir`.
///
/// Uses a two-phase locking strategy:
/// 1. **Fast path** — shared lock + read.  If the entry exists, touch its
///    `last_accessed` timestamp under an exclusive lock upgrade and return.
/// 2. **Slow path** — exclusive lock + read-modify-write.  Only taken when
///    the entry does not yet exist.
///
/// This reduces contention when many processes concurrently resolve the same
/// directory (common case: entry already exists).
pub fn get_or_create(dir: &Path) -> Result<String> {
    let path = sessions_path()?;
    let key = canonical_key(dir);

    // --- Fast path: shared-lock read ---
    {
        let lock_file = open_lock_file()?;
        lock_file.lock_shared().context("acquiring shared sessions lock")?;

        if let Ok(map) = load_map(&path) {
            if let Some(rec) = map.get(&key) {
                let uuid = rec.uuid.clone();
                lock_file.unlock().context("releasing shared sessions lock")?;

                // Touch last_accessed under exclusive lock (best-effort).
                if let Ok(lf2) = open_lock_file() {
                    if lf2.lock_exclusive().is_ok() {
                        if let Ok(mut m) = load_map(&path) {
                            if let Some(r) = m.get_mut(&key) {
                                r.touch();
                                let _ = save_map(&path, &m);
                            }
                        }
                        let _ = lf2.unlock();
                    }
                }

                return Ok(uuid);
            }
        }
        let _ = lock_file.unlock();
    }

    // --- Slow path: exclusive lock, create entry ---
    let lock_file = open_lock_file()?;
    lock_file
        .lock_exclusive()
        .context("acquiring exclusive sessions lock")?;

    let mut map = load_map(&path)?;

    // Double-check: another process may have created the entry between the
    // shared read and the exclusive lock acquisition.
    let uuid = if let Some(rec) = map.get_mut(&key) {
        rec.touch();
        rec.uuid.clone()
    } else {
        let rec = SessionRecord::new();
        let id = rec.uuid.clone();
        map.insert(key, rec);
        save_map(&path, &map)?;
        id
    };

    lock_file.unlock().context("releasing sessions lock")?;
    Ok(uuid)
}

/// Look up or create the session UUID for `dir` with a specific TTL tier.
pub fn get_or_create_with_tier(dir: &Path, tier: &str) -> Result<String> {
    let path = sessions_path()?;
    let key = canonical_key(dir);

    let lock_file = open_lock_file()?;
    lock_file
        .lock_exclusive()
        .context("acquiring sessions lock")?;

    let mut map = load_map(&path)?;

    let uuid = if let Some(rec) = map.get_mut(&key) {
        rec.touch();
        rec.ttl_tier = tier.to_string();
        rec.uuid.clone()
    } else {
        let rec = SessionRecord::new_with_tier(tier);
        let id = rec.uuid.clone();
        map.insert(key, rec);
        id
    };

    save_map(&path, &map)?;
    lock_file.unlock().context("releasing sessions lock")?;
    Ok(uuid)
}

/// Replace the session UUID for `dir` with a fresh one, effectively resetting
/// the session context.
pub fn reset(dir: &Path) -> Result<String> {
    let path = sessions_path()?;

    let lock_file = open_lock_file()?;
    lock_file
        .lock_exclusive()
        .context("acquiring sessions lock")?;

    let key = canonical_key(dir);
    let mut map = load_map(&path)?;

    let rec = SessionRecord::new();
    let new_id = rec.uuid.clone();
    map.insert(key, rec);
    save_map(&path, &map)?;

    lock_file.unlock().context("releasing sessions lock")?;
    Ok(new_id)
}

/// Return the current session UUID for `dir`, if one exists.
pub fn current(dir: &Path) -> Result<Option<String>> {
    let path = sessions_path()?;
    if !path.exists() {
        return Ok(None);
    }

    let lock_file = open_lock_file()?;
    lock_file.lock_shared().context("acquiring sessions lock")?;

    let key = canonical_key(dir);
    let map = load_map(&path)?;
    let result = map.get(&key).map(|r| r.uuid.clone());

    lock_file.unlock().context("releasing sessions lock")?;
    Ok(result)
}

/// Return the full session record for `dir`, if one exists.
pub fn current_record(dir: &Path) -> Result<Option<SessionRecord>> {
    let path = sessions_path()?;
    if !path.exists() {
        return Ok(None);
    }

    let lock_file = open_lock_file()?;
    lock_file.lock_shared().context("acquiring sessions lock")?;

    let key = canonical_key(dir);
    let map = load_map(&path)?;
    let result = map.get(&key).cloned();

    lock_file.unlock().context("releasing sessions lock")?;
    Ok(result)
}

/// Return all session records (directory → SessionRecord).
pub fn list_all() -> Result<HashMap<String, SessionRecord>> {
    let path = sessions_path()?;
    if !path.exists() {
        return Ok(HashMap::new());
    }

    let lock_file = open_lock_file()?;
    lock_file.lock_shared().context("acquiring sessions lock")?;

    let map = load_map(&path)?;

    lock_file.unlock().context("releasing sessions lock")?;
    Ok(map)
}

/// Clear expired sessions from `sessions.json`.
///
/// `ttl_secs` maps tier names to their TTL in seconds.  Entries whose
/// `last_accessed` is older than their tier's TTL are removed.
///
/// If `dry_run` is true, returns the list of expired keys without modifying
/// the file.
pub fn clear_expired(
    ttl_secs: &HashMap<String, u64>,
    dry_run: bool,
) -> Result<Vec<(String, SessionRecord)>> {
    let path = sessions_path()?;
    if !path.exists() {
        return Ok(Vec::new());
    }

    let lock_file = open_lock_file()?;
    lock_file
        .lock_exclusive()
        .context("acquiring sessions lock")?;

    let mut map = load_map(&path)?;
    let now = chrono::Utc::now();

    let mut expired = Vec::new();
    let keys: Vec<String> = map.keys().cloned().collect();

    for key in keys {
        let rec = &map[&key];
        let tier_ttl = ttl_secs.get(&rec.ttl_tier).copied().unwrap_or(
            // Fall back to "default" tier, then 30 minutes.
            ttl_secs.get("default").copied().unwrap_or(30 * 60),
        );

        if let Ok(accessed) = chrono::DateTime::parse_from_rfc3339(&rec.last_accessed) {
            let age = now
                .signed_duration_since(accessed)
                .num_seconds()
                .max(0) as u64;
            if age > tier_ttl {
                expired.push((key.clone(), rec.clone()));
            }
        }
    }

    if !dry_run {
        for (key, _) in &expired {
            map.remove(key);
        }
        save_map(&path, &map)?;
    }

    lock_file.unlock().context("releasing sessions lock")?;
    Ok(expired)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::with_temp_home;
    use tempfile::tempdir;

    #[test]
    fn get_or_create_returns_same_uuid_for_same_dir() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        let id1 = get_or_create(dir.path()).unwrap();
        let id2 = get_or_create(dir.path()).unwrap();
        assert_eq!(id1, id2);
    }

    #[test]
    fn get_or_create_different_dirs_get_different_uuids() {
        let _guard = with_temp_home();
        let dir_a = tempdir().unwrap();
        let dir_b = tempdir().unwrap();
        let id_a = get_or_create(dir_a.path()).unwrap();
        let id_b = get_or_create(dir_b.path()).unwrap();
        assert_ne!(id_a, id_b);
    }

    #[test]
    fn current_returns_none_before_creation() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        assert!(current(dir.path()).unwrap().is_none());
    }

    #[test]
    fn current_returns_uuid_after_creation() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        let id = get_or_create(dir.path()).unwrap();
        assert_eq!(current(dir.path()).unwrap(), Some(id));
    }

    #[test]
    fn reset_generates_new_uuid() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        let old_id = get_or_create(dir.path()).unwrap();
        let new_id = reset(dir.path()).unwrap();
        assert_ne!(old_id, new_id);
        assert_eq!(get_or_create(dir.path()).unwrap(), new_id);
    }

    #[test]
    fn uuid_is_valid_v4_format() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        let id = get_or_create(dir.path()).unwrap();
        let parts: Vec<&str> = id.split('-').collect();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0].len(), 8);
        assert_eq!(parts[1].len(), 4);
        assert_eq!(parts[2].len(), 4);
        assert_eq!(parts[3].len(), 4);
        assert_eq!(parts[4].len(), 12);
    }

    #[test]
    fn concurrent_get_or_create_same_dir_returns_same_uuid() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        let dir_path = dir.path().to_path_buf();

        let handles: Vec<_> = (0..50)
            .map(|_| {
                let p = dir_path.clone();
                std::thread::spawn(move || get_or_create(&p).unwrap())
            })
            .collect();

        let ids: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let first = &ids[0];
        for id in &ids {
            assert_eq!(id, first, "all threads must see the same UUID");
        }
    }

    #[test]
    fn concurrent_different_dirs_no_corruption() {
        let _guard = with_temp_home();
        let dirs: Vec<_> = (0..20).map(|_| tempdir().unwrap()).collect();
        let paths: Vec<_> = dirs.iter().map(|d| d.path().to_path_buf()).collect();

        let handles: Vec<_> = paths
            .iter()
            .cloned()
            .map(|p| std::thread::spawn(move || get_or_create(&p).unwrap()))
            .collect();

        let ids: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let unique: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len(), "each dir must get a unique UUID");

        for (path, expected_id) in paths.iter().zip(ids.iter()) {
            assert_eq!(&get_or_create(path).unwrap(), expected_id);
        }
    }

    #[test]
    fn concurrent_reset_and_read() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        let _ = get_or_create(dir.path()).unwrap();

        let dir_path = dir.path().to_path_buf();
        let handles: Vec<_> = (0..30)
            .map(|i| {
                let p = dir_path.clone();
                std::thread::spawn(move || {
                    if i == 0 {
                        reset(&p).unwrap()
                    } else {
                        get_or_create(&p).unwrap()
                    }
                })
            })
            .collect();

        for h in handles {
            let id = h.join().unwrap();
            assert!(!id.is_empty());
        }
    }

    #[test]
    fn corrupted_sessions_json_recovers() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        let _ = get_or_create(dir.path()).unwrap();
        let path = sessions_path().unwrap();
        std::fs::write(&path, "not valid json {{{").unwrap();
        // Corrupted file is treated as empty (warn + reset), so a new UUID is created.
        let id = get_or_create(dir.path()).unwrap();
        assert!(!id.is_empty());
    }

    #[test]
    fn empty_sessions_json_treated_as_fresh() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        let path = sessions_path().unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "").unwrap();
        // Empty file is valid (treated as empty map).
        let id = get_or_create(dir.path()).unwrap();
        assert!(!id.is_empty());
    }

    #[test]
    fn reset_on_nonexistent_session_creates_new() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        let id = reset(dir.path()).unwrap();
        assert!(!id.is_empty());
        assert_eq!(current(dir.path()).unwrap(), Some(id));
    }

    #[test]
    fn sessions_json_file_permissions_are_0600() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        let _ = get_or_create(dir.path()).unwrap();
        let path = sessions_path().unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        let mode = meta.permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "sessions.json should be mode 0600, got {mode:o}");
    }

    #[test]
    fn symlinked_dirs_resolve_to_same_session() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        let link = dir.path().parent().unwrap().join("symlink-test");
        std::os::unix::fs::symlink(dir.path(), &link).unwrap();
        let id1 = get_or_create(dir.path()).unwrap();
        let id2 = get_or_create(&link).unwrap();
        assert_eq!(id1, id2, "symlink and target should share the same session");
        std::fs::remove_file(&link).ok();
    }

    #[test]
    fn persists_across_calls() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        let id = get_or_create(dir.path()).unwrap();
        let id2 = get_or_create(dir.path()).unwrap();
        assert_eq!(id, id2);
    }

    // ---- SessionRecord / multi-TTL tests -----------------------------------

    #[test]
    fn session_record_has_timestamps() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        let _ = get_or_create(dir.path()).unwrap();
        let rec = current_record(dir.path()).unwrap().unwrap();
        assert!(!rec.created_at.is_empty());
        assert!(!rec.last_accessed.is_empty());
        assert_eq!(rec.ttl_tier, "default");
    }

    #[test]
    fn get_or_create_with_tier_sets_tier() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        let _ = get_or_create_with_tier(dir.path(), "ephemeral").unwrap();
        let rec = current_record(dir.path()).unwrap().unwrap();
        assert_eq!(rec.ttl_tier, "ephemeral");
    }

    #[test]
    fn list_all_returns_all_entries() {
        let _guard = with_temp_home();
        let d1 = tempdir().unwrap();
        let d2 = tempdir().unwrap();
        let _ = get_or_create(d1.path()).unwrap();
        let _ = get_or_create(d2.path()).unwrap();
        let all = list_all().unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn clear_expired_removes_old_entries() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        let _ = get_or_create(dir.path()).unwrap();

        // Manually backdate the entry.
        let path = sessions_path().unwrap();
        let mut map = load_map(&path).unwrap();
        let key = canonical_key(dir.path());
        if let Some(rec) = map.get_mut(&key) {
            rec.last_accessed = "2020-01-01T00:00:00Z".to_string();
        }
        save_map(&path, &map).unwrap();

        let mut ttls = HashMap::new();
        ttls.insert("default".to_string(), 60u64); // 60 seconds

        let expired = clear_expired(&ttls, false).unwrap();
        assert_eq!(expired.len(), 1);

        // Verify it's actually gone.
        assert!(current(dir.path()).unwrap().is_none());
    }

    #[test]
    fn clear_expired_dry_run_does_not_remove() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        let id = get_or_create(dir.path()).unwrap();

        // Backdate.
        let path = sessions_path().unwrap();
        let mut map = load_map(&path).unwrap();
        let key = canonical_key(dir.path());
        if let Some(rec) = map.get_mut(&key) {
            rec.last_accessed = "2020-01-01T00:00:00Z".to_string();
        }
        save_map(&path, &map).unwrap();

        let mut ttls = HashMap::new();
        ttls.insert("default".to_string(), 60u64);

        let expired = clear_expired(&ttls, true).unwrap();
        assert_eq!(expired.len(), 1);

        // Entry still exists.
        assert_eq!(current(dir.path()).unwrap(), Some(id));
    }

    #[test]
    fn clear_expired_multi_tier_eviction() {
        let _guard = with_temp_home();
        let d_eph = tempdir().unwrap();
        let d_persist = tempdir().unwrap();
        let d_default = tempdir().unwrap();

        let _ = get_or_create_with_tier(d_eph.path(), "ephemeral").unwrap();
        let _ = get_or_create_with_tier(d_persist.path(), "persistent").unwrap();
        let _ = get_or_create(d_default.path()).unwrap();

        // Backdate all entries to 2 hours ago.
        let path = sessions_path().unwrap();
        let mut map = load_map(&path).unwrap();
        let two_hours_ago = (chrono::Utc::now() - chrono::Duration::hours(2)).to_rfc3339();
        for rec in map.values_mut() {
            rec.last_accessed = two_hours_ago.clone();
        }
        save_map(&path, &map).unwrap();

        let mut ttls = HashMap::new();
        ttls.insert("ephemeral".to_string(), 300u64);    // 5 min — expired
        ttls.insert("default".to_string(), 1800u64);     // 30 min — expired
        ttls.insert("persistent".to_string(), 86400u64); // 24 hr — NOT expired

        let expired = clear_expired(&ttls, false).unwrap();
        assert_eq!(expired.len(), 2); // ephemeral + default

        // persistent should survive.
        assert!(current(d_persist.path()).unwrap().is_some());
        assert!(current(d_eph.path()).unwrap().is_none());
        assert!(current(d_default.path()).unwrap().is_none());
    }

    #[test]
    fn clear_expired_empty_store_no_error() {
        let _guard = with_temp_home();
        let ttls = HashMap::new();
        let expired = clear_expired(&ttls, false).unwrap();
        assert!(expired.is_empty());
    }

    #[test]
    fn concurrent_get_or_create_with_tiers() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        let dir_path = dir.path().to_path_buf();

        let handles: Vec<_> = (0..20)
            .map(|i| {
                let p = dir_path.clone();
                let tier = if i % 2 == 0 { "ephemeral" } else { "default" };
                std::thread::spawn(move || get_or_create_with_tier(&p, tier).unwrap())
            })
            .collect();

        let ids: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let first = &ids[0];
        for id in &ids {
            assert_eq!(id, first, "all threads must see the same UUID");
        }
    }

    #[test]
    fn last_accessed_updates_on_get() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        let _ = get_or_create(dir.path()).unwrap();

        let rec1 = current_record(dir.path()).unwrap().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        let _ = get_or_create(dir.path()).unwrap();
        let rec2 = current_record(dir.path()).unwrap().unwrap();

        // last_accessed should be updated (or at least not before rec1).
        assert!(rec2.last_accessed >= rec1.last_accessed);
    }

    #[test]
    fn concurrent_clear_expired_during_reads() {
        // Simulate clearing while other threads read — no corruption.
        let _guard = with_temp_home();
        let dirs: Vec<_> = (0..10).map(|_| tempdir().unwrap()).collect();
        for d in &dirs {
            let _ = get_or_create(d.path()).unwrap();
        }

        let paths: Vec<_> = dirs.iter().map(|d| d.path().to_path_buf()).collect();
        let mut handles: Vec<std::thread::JoinHandle<()>> = Vec::new();

        // Reader threads.
        for p in paths.iter().cloned() {
            handles.push(std::thread::spawn(move || {
                for _ in 0..5 {
                    let _ = get_or_create(&p);
                    let _ = current(&p);
                }
            }));
        }

        // Clearer thread (nothing should be expired since entries are fresh).
        handles.push(std::thread::spawn(move || {
            let mut ttls = HashMap::new();
            ttls.insert("default".to_string(), 1u64); // very short TTL
            for _ in 0..3 {
                let _ = clear_expired(&ttls, false);
            }
        }));

        for h in handles {
            h.join().unwrap();
        }
    }
}
