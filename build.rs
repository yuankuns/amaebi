//! Build script: inject the current git commit short hash as `AMAEBI_GIT_COMMIT`.
//!
//! Falls back to "unknown" if `git` is unavailable or the build is not in a
//! git worktree (e.g. built from a source tarball). Re-runs when the real
//! git `HEAD` or `index` paths change so the hash stays fresh across commits,
//! including worktrees, submodules, and similar setups where `.git` is a file.

use std::process::Command;

fn git_path(name: &str) -> Option<String> {
    Command::new("git")
        .args(["rev-parse", "--git-path", name])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn main() {
    // Re-run when HEAD moves (new commits, checkouts) or the working tree
    // gets a new index (amends/rebases update the index too).
    // Use `git rev-parse --git-path` so we find the real paths in worktrees
    // and submodules where `.git` is a file pointing elsewhere.
    if let Some(path) = git_path("HEAD") {
        println!("cargo:rerun-if-changed={path}");
    }
    if let Some(path) = git_path("index") {
        println!("cargo:rerun-if-changed={path}");
    }

    let commit = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=AMAEBI_GIT_COMMIT={commit}");
}
