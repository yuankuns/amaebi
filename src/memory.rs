use anyhow::{Context, Result};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::io::{BufRead, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::PathBuf;

use crate::auth::amaebi_home;

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub timestamp: String,
    pub user: String,
    pub assistant: String,
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

pub fn memory_path() -> Result<PathBuf> {
    Ok(amaebi_home()?.join("memory.jsonl"))
}

/// Path to the advisory lock file used to serialise all memory I/O.
fn lock_path() -> Result<PathBuf> {
    Ok(amaebi_home()?.join("memory.lock"))
}

/// Open (creating if necessary) the memory lock file.
///
/// All callers must call [`FileExt::lock_exclusive`] or
/// [`FileExt::lock_shared`] on the returned handle before accessing
/// `memory.jsonl`.
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
        .with_context(|| format!("opening lock file {}", path.display()))
}

fn truncate(s: &str) -> String {
    if s.len() <= 200 {
        s.to_string()
    } else {
        let end = s.floor_char_boundary(200);
        format!("{}…", &s[..end])
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum size of `memory.jsonl` before it is rotated to `memory.jsonl.old`.
pub const MAX_FILE_BYTES: u64 = 1024 * 1024; // 1 MiB

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Append a user/assistant pair to the memory file.
///
/// Returns the [`MemoryEntry`] that was written so callers can update
/// an in-memory cache without re-reading the file.
///
/// If adding the new entry would cause `memory.jsonl` to reach or exceed
/// [`MAX_FILE_BYTES`] it is atomically renamed to `memory.jsonl.old` and a
/// fresh file is started.  The entire rotate-then-write sequence is protected
/// by an exclusive lock on `memory.lock` so concurrent writers cannot race.
pub fn append(user_prompt: &str, assistant_response: &str) -> Result<MemoryEntry> {
    let path = memory_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }

    let entry = MemoryEntry {
        timestamp: chrono::Utc::now().to_rfc3339(),
        user: truncate(user_prompt),
        assistant: truncate(assistant_response),
    };

    // Build the complete JSONL line before acquiring any lock so the critical
    // section is as short as possible.
    let mut line = serde_json::to_string(&entry)?;
    line.push('\n');

    // Acquire an exclusive advisory lock via the dedicated lock file.  This
    // serialises all writers (including across processes) and covers the entire
    // size-check → optional-rotate → write sequence, eliminating every TOCTOU
    // race between concurrent appenders.
    let lock_file = open_lock_file()?;
    lock_file
        .lock_exclusive()
        .context("acquiring exclusive memory lock")?;

    // Check current file size under the lock.  Include the bytes we are about
    // to write so that the post-write size stays within MAX_FILE_BYTES.
    let current_size = if path.exists() {
        std::fs::metadata(&path)
            .with_context(|| format!("checking size of {}", path.display()))?
            .len()
    } else {
        0
    };

    if current_size + line.len() as u64 >= MAX_FILE_BYTES {
        // Rotate while holding the lock — no other writer can sneak in.
        let old_path = path.with_file_name("memory.jsonl.old");
        std::fs::rename(&path, &old_path)
            .with_context(|| format!("rotating {} to {}", path.display(), old_path.display()))?;
        // Enforce 0600 on the rotated file — the original may have had broader
        // permissions set before this code path existed.
        std::fs::set_permissions(&old_path, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("setting permissions on {}", old_path.display()))?;
        tracing::info!(
            bytes = current_size,
            old_path = %old_path.display(),
            "memory file exceeded limit; rotated"
        );
    }

    // Open (or create) the data file and write the entry.
    let mut data_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open(&path)
        .with_context(|| format!("opening {}", path.display()))?;

    // Enforce 0600 even if the file pre-existed with broader permissions.
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("setting permissions on {}", path.display()))?;

    data_file.write_all(line.as_bytes())?;

    // Release the advisory lock.
    lock_file
        .unlock()
        .context("releasing exclusive memory lock")?;

    Ok(entry)
}

/// Load the last `n` entries from the memory file.
///
/// Uses a fixed-capacity ring buffer so the entire file is never held in
/// memory — only the last `n` entries are kept at any point.
pub fn load_recent(n: usize) -> Result<Vec<MemoryEntry>> {
    let path = memory_path()?;

    // Acquire a shared lock before opening the data file so readers cannot
    // observe the file in a partially-rotated state during concurrent writes.
    let lock_file = open_lock_file()?;
    lock_file
        .lock_shared()
        .context("acquiring shared memory lock")?;

    let result = (|| {
        if !path.exists() {
            return Ok(vec![]);
        }

        let file =
            std::fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;

        let mut ring: VecDeque<MemoryEntry> = VecDeque::new();
        let mut line_no: u64 = 0;

        for line in std::io::BufReader::new(&file).lines() {
            line_no += 1;
            let line = line.with_context(|| format!("reading {}", path.display()))?;
            let l = line.trim();
            if l.is_empty() {
                continue;
            }
            match serde_json::from_str::<MemoryEntry>(l) {
                Ok(entry) => {
                    ring.push_back(entry);
                    if ring.len() > n {
                        ring.pop_front();
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        line_no = line_no,
                        bytes = l.len(),
                        "skipping malformed memory entry"
                    );
                }
            }
        }

        Ok(ring.into_iter().collect())
    })();

    lock_file.unlock().context("releasing shared memory lock")?;

    result
}

/// Return all entries whose user or assistant text contains `query` (case-insensitive).
pub fn search(query: &str) -> Result<Vec<MemoryEntry>> {
    let path = memory_path()?;

    let lock_file = open_lock_file()?;
    lock_file
        .lock_shared()
        .context("acquiring shared memory lock")?;

    let result = (|| {
        if !path.exists() {
            return Ok(vec![]);
        }

        let file =
            std::fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;

        let query_lower = query.to_lowercase();
        let mut results = Vec::new();
        let mut line_no: u64 = 0;

        for line in std::io::BufReader::new(&file).lines() {
            line_no += 1;
            let line = line.with_context(|| format!("reading {}", path.display()))?;
            let l = line.trim();
            if l.is_empty() {
                continue;
            }
            match serde_json::from_str::<MemoryEntry>(l) {
                Ok(entry) => {
                    if entry.user.to_lowercase().contains(&query_lower)
                        || entry.assistant.to_lowercase().contains(&query_lower)
                    {
                        results.push(entry);
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        line_no = line_no,
                        bytes = l.len(),
                        "skipping malformed memory entry"
                    );
                }
            }
        }

        Ok(results)
    })();

    lock_file.unlock().context("releasing shared memory lock")?;

    result
}

/// Delete the memory files (`memory.jsonl` and `memory.jsonl.old` if present).
pub fn clear() -> Result<()> {
    let path = memory_path()?;
    let old_path = path.with_file_name("memory.jsonl.old");

    let lock_file = open_lock_file()?;
    lock_file
        .lock_exclusive()
        .context("acquiring exclusive memory lock")?;

    if path.exists() {
        std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
    }
    if old_path.exists() {
        std::fs::remove_file(&old_path)
            .with_context(|| format!("removing {}", old_path.display()))?;
    }

    lock_file
        .unlock()
        .context("releasing exclusive memory lock")?;

    Ok(())
}

/// Return the total number of entries in the memory file.
pub fn count() -> Result<usize> {
    let path = memory_path()?;

    let lock_file = open_lock_file()?;
    lock_file
        .lock_shared()
        .context("acquiring shared memory lock")?;

    let result = (|| {
        if !path.exists() {
            return Ok(0usize);
        }

        let file =
            std::fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;

        let mut count = 0usize;
        for line in std::io::BufReader::new(&file).lines() {
            let line = line.with_context(|| format!("reading {}", path.display()))?;
            if !line.trim().is_empty() {
                count += 1;
            }
        }

        Ok(count)
    })();

    lock_file.unlock().context("releasing shared memory lock")?;

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    static HOME_LOCK: Mutex<()> = Mutex::new(());
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn with_temp_home<F: FnOnce() -> R, R>(f: F) -> R {
        let _guard = HOME_LOCK.lock().unwrap();
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let tmp = std::env::temp_dir().join(format!("amaebi_test_{}_{}", std::process::id(), id));
        std::fs::create_dir_all(&tmp).unwrap();

        let old_home = std::env::var("HOME").ok();
        // SAFETY: tests are serialized via HOME_LOCK
        unsafe { std::env::set_var("HOME", &tmp) };

        let result = f();

        match old_home {
            Some(h) => unsafe { std::env::set_var("HOME", h) },
            None => unsafe { std::env::remove_var("HOME") },
        }
        std::fs::remove_dir_all(&tmp).ok();
        result
    }

    #[test]
    fn test_empty_when_no_file() {
        with_temp_home(|| {
            assert_eq!(load_recent(20).unwrap().len(), 0);
            assert_eq!(count().unwrap(), 0);
            assert_eq!(search("anything").unwrap().len(), 0);
        });
    }

    #[test]
    fn test_append_and_load() {
        with_temp_home(|| {
            append("hello world", "I can help with that").unwrap();
            let entries = load_recent(20).unwrap();
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].user, "hello world");
            assert_eq!(entries[0].assistant, "I can help with that");
        });
    }

    #[test]
    fn test_count() {
        with_temp_home(|| {
            assert_eq!(count().unwrap(), 0);
            append("a", "b").unwrap();
            assert_eq!(count().unwrap(), 1);
            append("c", "d").unwrap();
            assert_eq!(count().unwrap(), 2);
        });
    }

    #[test]
    fn test_load_recent_limits() {
        with_temp_home(|| {
            for i in 0..10 {
                append(&format!("prompt {i}"), &format!("response {i}")).unwrap();
            }
            assert_eq!(count().unwrap(), 10);
            let entries = load_recent(3).unwrap();
            assert_eq!(entries.len(), 3);
            assert_eq!(entries[0].user, "prompt 7");
        });
    }

    #[test]
    fn test_search_user_field() {
        with_temp_home(|| {
            append("how to install rust", "use rustup").unwrap();
            append("what is python", "a programming language").unwrap();

            let results = search("rust").unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].user, "how to install rust");
        });
    }

    #[test]
    fn test_search_assistant_field() {
        with_temp_home(|| {
            append("how to install rust", "use rustup").unwrap();
            append("what is python", "a programming language").unwrap();

            // "rustup" is in the assistant field
            let results = search("rustup").unwrap();
            assert_eq!(results.len(), 1);

            // No match
            assert_eq!(search("javascript").unwrap().len(), 0);
        });
    }

    #[test]
    fn test_search_case_insensitive() {
        with_temp_home(|| {
            append("How to use Rust", "See the docs").unwrap();
            assert_eq!(search("rust").unwrap().len(), 1);
            assert_eq!(search("RUST").unwrap().len(), 1);
        });
    }

    #[test]
    fn test_truncation() {
        with_temp_home(|| {
            let long = "a".repeat(300);
            append(&long, "short").unwrap();
            let entries = load_recent(20).unwrap();
            // 200 bytes of 'a' + "…" (3 UTF-8 bytes) = 203 bytes
            assert!(entries[0].user.len() <= 203);
            assert!(entries[0].user.ends_with('…'));
            assert_eq!(entries[0].assistant, "short");
        });
    }

    #[test]
    fn test_clear() {
        with_temp_home(|| {
            append("test", "response").unwrap();
            assert_eq!(count().unwrap(), 1);
            clear().unwrap();
            assert_eq!(count().unwrap(), 0);
            // clear on non-existent file is a no-op
            clear().unwrap();
        });
    }

    #[test]
    fn test_clear_removes_old_file() {
        with_temp_home(|| {
            // Manually create a .old file to simulate a prior rotation.
            // The amaebi home directory does not exist yet so create it first.
            let path = memory_path().unwrap();
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let old_path = path.with_file_name("memory.jsonl.old");
            std::fs::write(&old_path, b"old data\n").unwrap();
            assert!(old_path.exists());

            append("test", "response").unwrap();
            clear().unwrap();

            assert!(!memory_path().unwrap().exists());
            assert!(
                !old_path.exists(),
                "clear() must also remove memory.jsonl.old"
            );
        });
    }
}
