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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Process-wide lock that serialises all tests that mutate `$HOME`.
///
/// All tests that call [`with_home`] or [`with_temp_home`] hold this lock
/// for the duration of their body.
pub static HOME_LOCK: Mutex<()> = Mutex::new(());

/// RAII guard that restores `$HOME` to its original value when dropped.
///
/// Using a `Drop` guard (rather than restoring in-line) ensures `$HOME` is
/// reset even if the test body panics, preventing `HOME_LOCK` poisoning and
/// leaked env state.
pub struct HomeGuard {
    old: Option<String>,
}

impl Drop for HomeGuard {
    fn drop(&mut self) {
        // SAFETY: HOME_LOCK is held for the entire lifetime of this guard.
        match &self.old {
            Some(h) => unsafe { std::env::set_var("HOME", h) },
            None => unsafe { std::env::remove_var("HOME") },
        }
    }
}

/// Run `f` with `$HOME` temporarily pointing at `dir`.
///
/// Acquires [`HOME_LOCK`] for the full duration so concurrent tests from
/// *any* module cannot race on the environment variable.  The original
/// `$HOME` is restored via [`HomeGuard::drop`] even if `f` panics.
pub fn with_home<F: FnOnce() -> R, R>(dir: &Path, f: F) -> R {
    let _lock = HOME_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let old = std::env::var("HOME").ok();
    // SAFETY: HOME_LOCK is held above; only one thread mutates HOME at a time.
    unsafe { std::env::set_var("HOME", dir) };
    let _guard = HomeGuard { old };
    f()
}

/// Run `f` with `$HOME` pointing at a fresh temporary directory.
///
/// The directory is removed after `f` returns.  If `f` panics, `$HOME` is
/// still restored (via [`HomeGuard`]) though the temp directory may be left
/// on disk — that is acceptable for a test failure scenario.
pub fn with_temp_home<F: FnOnce() -> R, R>(f: F) -> R {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::SeqCst);
    let tmp = std::env::temp_dir().join(format!("amaebi_test_{}_{}", std::process::id(), id));
    std::fs::create_dir_all(&tmp).expect("creating temp home dir");
    let result = with_home(&tmp, f);
    std::fs::remove_dir_all(&tmp).ok();
    result
}
