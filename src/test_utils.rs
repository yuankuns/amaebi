//! Shared test utilities.
//!
//! Only compiled when running tests (`#[cfg(test)]` in `main.rs`).
//!
//! # Why a single `HOME_LOCK`?
//!
//! Both `auth` and `memory` tests temporarily mutate the `HOME` environment
//! variable.  If each module uses its own `Mutex`, tests from different
//! modules can run concurrently and observe each other's `HOME` mutations,
//! causing spurious failures or, on some platforms, undefined behaviour.
//! Sharing one process-wide lock eliminates that race.

use std::path::Path;
use std::sync::Mutex;

/// Process-wide lock that serialises all tests that mutate `$HOME`.
///
/// All tests that call [`with_home`] or [`with_temp_home`] hold this lock
/// for the duration of their body.
pub static HOME_LOCK: Mutex<()> = Mutex::new(());

/// RAII guard that restores `$HOME` to its original value when dropped.
///
/// Restoring `$HOME` in `Drop` (rather than inline) ensures the environment
/// variable is reset even if the test body panics.  Without this, a panicking
/// test would leave `$HOME` pointing at the temp directory for every subsequent
/// test that runs in the same thread.
///
/// Note: this does **not** prevent [`HOME_LOCK`] from being poisoned — a panic
/// while the mutex is held still marks it poisoned.  [`with_home`] recovers
/// from a poisoned mutex via `unwrap_or_else(|p| p.into_inner())` so later
/// tests are not permanently blocked.
///
/// When `_tmpdir` is `Some`, it is dropped **after** the `Drop` body restores
/// `$HOME` (Rust drops fields in declaration order after the `drop` fn
/// returns), so `$HOME` is always restored before the directory is deleted.
pub struct HomeGuard {
    old: Option<String>,
    /// Keeps the temporary directory alive until `$HOME` has been restored.
    _tmpdir: Option<tempfile::TempDir>,
}

impl Drop for HomeGuard {
    fn drop(&mut self) {
        // SAFETY: HOME_LOCK is held for the entire lifetime of this guard.
        match &self.old {
            Some(h) => unsafe { std::env::set_var("HOME", h) },
            None => unsafe { std::env::remove_var("HOME") },
        }
        // _tmpdir drops here (after this fn returns), deleting the directory.
    }
}

/// Run `f` with `$HOME` temporarily pointing at `dir`.
///
/// Acquires [`HOME_LOCK`] for the full duration so concurrent tests from
/// *any* module cannot race on the environment variable.  The original
/// `$HOME` is restored via [`HomeGuard::drop`] even if `f` panics.
///
/// # Synchronous closures only
///
/// `f` **must** be a synchronous closure.  Passing an `async` block would
/// return an unawaited `Future` without executing any of its body, meaning the
/// test logic would never run while the lock is held — a silent no-op.  For
/// async tests, use [`with_temp_home`] and hold the guard manually.
pub fn with_home<F: FnOnce() -> R, R>(dir: &Path, f: F) -> R {
    let _lock = HOME_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let old = std::env::var("HOME").ok();
    // SAFETY: HOME_LOCK is held above; only one thread mutates HOME at a time.
    unsafe { std::env::set_var("HOME", dir) };
    let _guard = HomeGuard { old, _tmpdir: None };
    f()
}

/// Run `f` with `$HOME` pointing at a fresh temporary directory.
///
/// Uses [`tempfile::TempDir`] for RAII cleanup: the directory is deleted when
/// the [`HomeGuard`] drops (which also restores `$HOME`), even if `f` panics.
pub fn with_temp_home<F: FnOnce() -> R, R>(f: F) -> R {
    let tmp = tempfile::TempDir::new().expect("creating temp home dir");
    let _lock = HOME_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let old = std::env::var("HOME").ok();
    // SAFETY: HOME_LOCK is held above; only one thread mutates HOME at a time.
    unsafe { std::env::set_var("HOME", tmp.path()) };
    let _guard = HomeGuard {
        old,
        _tmpdir: Some(tmp),
    };
    f()
}
