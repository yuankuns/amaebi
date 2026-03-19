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
use std::sync::{Mutex, MutexGuard};

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
/// Fields are dropped in declaration order *after* the `drop` body runs, so
/// the sequence is: HOME restored → temp directory deleted → mutex released.
pub struct HomeGuard {
    old: Option<String>,
    /// Keeps the temporary directory alive until `$HOME` has been restored.
    _tmpdir: Option<tempfile::TempDir>,
    /// Holds [`HOME_LOCK`] for the lifetime of this guard.
    ///
    /// `None` when the guard is created by [`with_home`], which holds the lock
    /// as a local variable instead (the closure completes before `with_home`
    /// returns, so the lock scope is identical).
    _lock: Option<MutexGuard<'static, ()>>,
}

impl Drop for HomeGuard {
    fn drop(&mut self) {
        // SAFETY: `set_var` / `remove_var` are unsafe because *any* concurrent
        // access to the environment — including reads — is undefined behaviour,
        // not only concurrent mutations.  HOME_LOCK serialises all test code
        // in this process that *mutates* HOME, but cannot guard against the
        // operating system, C extensions, or other threads that may read the
        // environment at any time.  This is a known limitation accepted for
        // test-only code.  Production code uses `amaebi_home()`, which reads
        // HOME only at explicit call sites and is never called concurrently
        // with these tests.
        match &self.old {
            Some(h) => unsafe { std::env::set_var("HOME", h) },
            None => unsafe { std::env::remove_var("HOME") },
        }
        // _tmpdir and _lock drop here (after this fn returns), in field order:
        // temp directory is deleted, then the mutex is released.
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
/// return an unawaited `Future` without executing any of its body — the lock
/// would be held and released without any test logic running.  For async
/// tests, use [`with_temp_home`], which returns a [`HomeGuard`] that the
/// caller drops at the end of the test.
pub fn with_home<F: FnOnce() -> R, R>(dir: &Path, f: F) -> R {
    let _lock = HOME_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let old = std::env::var("HOME").ok();
    // SAFETY: see HomeGuard::drop.
    unsafe { std::env::set_var("HOME", dir) };
    let _guard = HomeGuard {
        old,
        _tmpdir: None,
        _lock: None, // lock held by _lock local above for the duration of f()
    };
    f()
}

/// Point `$HOME` at a fresh temporary directory and return a [`HomeGuard`].
///
/// The guard restores `$HOME` and deletes the directory when dropped.  Because
/// this function returns a guard instead of taking a closure it works in both
/// synchronous and async tests:
///
/// ```ignore
/// let _guard = with_temp_home();
/// // ... test body (sync or async) ...
/// // guard dropped here → $HOME restored, temp dir deleted, lock released
/// ```
///
/// Uses [`tempfile::TempDir`] for RAII cleanup so the directory is removed
/// even if the test panics before the guard is explicitly dropped.
pub fn with_temp_home() -> HomeGuard {
    let tmp = tempfile::TempDir::new().expect("creating temp home dir");
    let lock = HOME_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let old = std::env::var("HOME").ok();
    // SAFETY: see HomeGuard::drop.
    unsafe { std::env::set_var("HOME", tmp.path()) };
    HomeGuard {
        old,
        _tmpdir: Some(tmp),
        _lock: Some(lock),
    }
}
