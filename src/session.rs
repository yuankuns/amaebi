//! Centralized directory-to-UUID session mapping.
//!
//! Session identifiers are stable UUIDs stored in `~/.amaebi/sessions.json`.
//! Each entry maps a canonical directory path to a UUID v4.  The mapping
//! persists across daemon restarts; the UUID is the authoritative identity for
//! a session — used for per-session history locking in the daemon.
//!
//! # Concurrency
//!
//! All reads and writes to `sessions.json` are serialised by an exclusive
//! advisory lock on `~/.amaebi/sessions.lock` (via `fs2::FileExt`).  This is
//! safe across concurrent processes (e.g., two terminals running `amaebi ask`
//! at the same moment).

use anyhow::{Context, Result};
use fs2::FileExt;
use std::collections::HashMap;
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use uuid::Uuid;

use crate::auth::amaebi_home;

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

fn load_map(path: &Path) -> Result<HashMap<String, String>> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let content =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&content).with_context(|| format!("parsing {}", path.display()))
}

/// Write the map atomically: write to a `.tmp` file, then rename.
fn save_map(path: &Path, map: &HashMap<String, String>) -> Result<()> {
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
/// The canonical absolute path of `dir` is used as the map key.  If no entry
/// exists a new UUID v4 is generated, persisted to `~/.amaebi/sessions.json`,
/// and returned.
pub fn get_or_create(dir: &Path) -> Result<String> {
    let path = sessions_path()?;

    let lock_file = open_lock_file()?;
    lock_file
        .lock_exclusive()
        .context("acquiring sessions lock")?;

    let key = canonical_key(dir);
    let mut map = load_map(&path)?;

    let uuid = if let Some(id) = map.get(&key) {
        id.clone()
    } else {
        let id = Uuid::new_v4().to_string();
        map.insert(key, id.clone());
        save_map(&path, &map)?;
        id
    };

    lock_file.unlock().context("releasing sessions lock")?;
    Ok(uuid)
}

/// Replace the session UUID for `dir` with a fresh one, effectively resetting
/// the session context.
///
/// Used by `amaebi session new`.  Returns the new UUID.
pub fn reset(dir: &Path) -> Result<String> {
    let path = sessions_path()?;

    let lock_file = open_lock_file()?;
    lock_file
        .lock_exclusive()
        .context("acquiring sessions lock")?;

    let key = canonical_key(dir);
    let mut map = load_map(&path)?;

    let new_id = Uuid::new_v4().to_string();
    map.insert(key, new_id.clone());
    save_map(&path, &map)?;

    lock_file.unlock().context("releasing sessions lock")?;
    Ok(new_id)
}

/// Return the current session UUID for `dir`, if one exists.
///
/// Returns `None` if no session has been created for `dir` yet.
/// Used by `amaebi session show`.
pub fn current(dir: &Path) -> Result<Option<String>> {
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
        // get_or_create now returns the new one
        assert_eq!(get_or_create(dir.path()).unwrap(), new_id);
    }

    #[test]
    fn uuid_is_valid_v4_format() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        let id = get_or_create(dir.path()).unwrap();
        // UUID v4 format: 8-4-4-4-12 hex chars separated by hyphens
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
        // Stress test: 50 threads all calling get_or_create on the same dir.
        // All must receive the same UUID — only one thread creates, the rest read.
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
        // 20 threads, each with a unique directory — no data races or corruption.
        let _guard = with_temp_home();
        let dirs: Vec<_> = (0..20).map(|_| tempdir().unwrap()).collect();
        let paths: Vec<_> = dirs.iter().map(|d| d.path().to_path_buf()).collect();

        let handles: Vec<_> = paths
            .iter()
            .cloned()
            .map(|p| std::thread::spawn(move || get_or_create(&p).unwrap()))
            .collect();

        let ids: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        // All UUIDs must be unique.
        let unique: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len(), "each dir must get a unique UUID");

        // Verify they all persisted correctly.
        for (path, expected_id) in paths.iter().zip(ids.iter()) {
            assert_eq!(&get_or_create(path).unwrap(), expected_id);
        }
    }

    #[test]
    fn concurrent_reset_and_read() {
        // One thread resets, others read — no panics or corruption.
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        let _ = get_or_create(dir.path()).unwrap(); // seed

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

        // All threads must complete without error.
        for h in handles {
            let id = h.join().unwrap();
            assert!(!id.is_empty());
        }
    }

    #[test]
    fn persists_across_calls() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        // First call creates and saves.
        let id = get_or_create(dir.path()).unwrap();
        // Second call should load from disk and return same id.
        drop(id.clone()); // just to be explicit we're not relying on in-memory state
        let id2 = get_or_create(dir.path()).unwrap();
        assert_eq!(id, id2);
    }
}
