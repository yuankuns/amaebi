#![allow(dead_code)]

use std::path::Path;

use anyhow::{Context, Result};
use tokio::process::Command;

// ---------------------------------------------------------------------------
// SandboxOutput
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SandboxOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

// ---------------------------------------------------------------------------
// Sandbox trait
// ---------------------------------------------------------------------------

/// A sandbox backend that can execute shell commands in an isolated environment.
#[async_trait::async_trait]
pub trait Sandbox: Send + Sync {
    /// Spawn a shell command inside the sandbox.
    async fn spawn(&self, cmd: &str, cwd: &Path) -> Result<SandboxOutput>;
    /// Returns true if this backend is available on the current system.
    fn available(&self) -> bool;
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// NoopSandbox
// ---------------------------------------------------------------------------

/// No-op sandbox: runs commands directly via sh -c. Used in tests and when
/// sandbox is disabled.
pub struct NoopSandbox;

#[async_trait::async_trait]
impl Sandbox for NoopSandbox {
    async fn spawn(&self, cmd: &str, cwd: &Path) -> Result<SandboxOutput> {
        let output = Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .current_dir(cwd)
            .output()
            .await
            .with_context(|| format!("spawning command: {cmd}"))?;

        Ok(SandboxOutput {
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            exit_code: output.status.code().unwrap_or(-1),
        })
    }

    fn available(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "noop"
    }
}

// ---------------------------------------------------------------------------
// Sub-modules
// ---------------------------------------------------------------------------

pub mod docker;
#[allow(unused_imports)]
pub use docker::{DockerSandbox, DockerSandboxConfig};
