use anyhow::{Context, Result};
use async_trait::async_trait;
use tokio::process::Command;

use super::{Sandbox, SandboxOutput};

/// No-op sandbox backend: runs commands directly via `sh -c` with no isolation.
///
/// Behaviour is identical to the previous hard-coded `Command::new("sh")` calls
/// so existing functionality is preserved when the sandbox is disabled.
pub struct NoopSandbox;

#[async_trait]
impl Sandbox for NoopSandbox {
    fn name(&self) -> &str {
        "noop"
    }

    fn available(&self) -> bool {
        true
    }

    async fn spawn(&self, cmd: &str, cwd: &std::path::Path) -> Result<SandboxOutput> {
        let output = Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .current_dir(cwd)
            .output()
            .await
            .with_context(|| format!("noop sandbox: spawning shell command: {cmd}"))?;

        Ok(SandboxOutput {
            status: output.status,
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn noop() -> NoopSandbox {
        NoopSandbox
    }

    #[test]
    fn name_is_noop() {
        assert_eq!(noop().name(), "noop");
    }

    #[test]
    fn always_available() {
        assert!(noop().available());
    }

    #[tokio::test]
    async fn captures_stdout() {
        let tmp = TempDir::new().unwrap();
        let out = noop().spawn("echo hello", tmp.path()).await.unwrap();
        assert_eq!(out.stdout.trim(), "hello");
        assert!(out.status.success());
    }

    #[tokio::test]
    async fn captures_stderr() {
        let tmp = TempDir::new().unwrap();
        let out = noop().spawn("echo err >&2", tmp.path()).await.unwrap();
        assert_eq!(out.stderr.trim(), "err");
        assert!(out.status.success());
    }

    #[tokio::test]
    async fn propagates_exit_code() {
        let tmp = TempDir::new().unwrap();
        let out = noop().spawn("exit 42", tmp.path()).await.unwrap();
        assert_eq!(out.status.code(), Some(42));
    }

    #[tokio::test]
    async fn empty_command_still_runs() {
        let tmp = TempDir::new().unwrap();
        let out = noop().spawn("true", tmp.path()).await.unwrap();
        assert!(out.status.success());
        assert!(out.stdout.is_empty());
        assert!(out.stderr.is_empty());
    }

    #[tokio::test]
    async fn respects_cwd() {
        let tmp = TempDir::new().unwrap();
        // Write a known file; verify pwd matches.
        let out = noop().spawn("pwd", tmp.path()).await.unwrap();
        // On macOS, /tmp is a symlink; compare canonical paths.
        let printed = std::fs::canonicalize(out.stdout.trim()).unwrap_or_default();
        let expected = std::fs::canonicalize(tmp.path()).unwrap_or_default();
        assert_eq!(printed, expected, "cwd should be the requested directory");
    }
}
