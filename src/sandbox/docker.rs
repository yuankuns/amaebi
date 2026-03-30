#![allow(dead_code)]

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tokio::process::Command;

use super::{Sandbox, SandboxOutput};

// ---------------------------------------------------------------------------
// DockerSandboxConfig
// ---------------------------------------------------------------------------

pub struct DockerSandboxConfig {
    pub image: String,
    /// Mounted read-write at the same path inside the container.
    pub workspace: PathBuf,
    /// Additional paths mounted read-only at the same path inside the container.
    pub ro_paths: Vec<PathBuf>,
    /// Additional paths mounted read-write at the same path inside the container.
    pub rw_paths: Vec<PathBuf>,
}

// ---------------------------------------------------------------------------
// DockerSandbox
// ---------------------------------------------------------------------------

pub struct DockerSandbox {
    config: DockerSandboxConfig,
}

impl DockerSandbox {
    pub fn new(config: DockerSandboxConfig) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl Sandbox for DockerSandbox {
    async fn spawn(&self, cmd: &str, cwd: &Path) -> Result<SandboxOutput> {
        let cwd_str = cwd
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("cwd is not valid UTF-8"))?;
        let workspace_str = self
            .config
            .workspace
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("workspace path is not valid UTF-8"))?;

        let mut docker = Command::new("docker");
        docker.args(["run", "--rm", "--network", "none"]);
        docker.args(["-w", cwd_str]);
        docker.args(["-v", &format!("{workspace_str}:{workspace_str}:rw")]);

        for rw_path in &self.config.rw_paths {
            let s = rw_path
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("rw_path is not valid UTF-8"))?;
            docker.args(["-v", &format!("{s}:{s}:rw")]);
        }

        for ro_path in &self.config.ro_paths {
            let s = ro_path
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("ro_path is not valid UTF-8"))?;
            docker.args(["-v", &format!("{s}:{s}:ro")]);
        }

        docker.arg(&self.config.image);
        docker.args(["sh", "-c", cmd]);

        let output = docker.output().await.context("spawning docker run")?;

        Ok(SandboxOutput {
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            exit_code: output.status.code().unwrap_or(-1),
        })
    }

    fn available(&self) -> bool {
        std::process::Command::new("docker")
            .arg("info")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    fn name(&self) -> &str {
        "docker"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::sandbox::NoopSandbox;

    // ---- NoopSandbox tests -------------------------------------------------

    #[tokio::test]
    async fn noop_sandbox_captures_stdout() {
        let sandbox = NoopSandbox;
        let dir = TempDir::new().unwrap();
        let out = sandbox.spawn("echo hello", dir.path()).await.unwrap();
        assert!(
            out.stdout.contains("hello"),
            "expected 'hello' in stdout, got: {:?}",
            out.stdout
        );
    }

    #[tokio::test]
    async fn noop_sandbox_exit_code_on_failure() {
        let sandbox = NoopSandbox;
        let dir = TempDir::new().unwrap();
        let out = sandbox.spawn("exit 42", dir.path()).await.unwrap();
        let display = format!("[exit {}]", out.exit_code);
        assert!(
            display.contains("[exit 42]"),
            "expected exit code 42, got: {display}"
        );
    }

    // ---- DockerSandbox tests (require Docker) ------------------------------

    fn alpine_sandbox(workspace: &TempDir) -> DockerSandbox {
        DockerSandbox::new(DockerSandboxConfig {
            image: "alpine".to_string(),
            workspace: workspace.path().to_path_buf(),
            ro_paths: vec![],
            rw_paths: vec![],
        })
    }

    #[tokio::test]
    #[ignore]
    async fn docker_sandbox_runs_command() {
        let dir = TempDir::new().unwrap();
        let sandbox = alpine_sandbox(&dir);
        let out = sandbox.spawn("echo hello", dir.path()).await.unwrap();
        assert!(
            out.stdout.contains("hello"),
            "expected 'hello' in stdout, got: {:?}",
            out.stdout
        );
    }

    #[tokio::test]
    #[ignore]
    async fn docker_sandbox_isolates_tmp() {
        let dir = TempDir::new().unwrap();
        let sandbox = alpine_sandbox(&dir);

        // First container writes a marker to /tmp.
        sandbox
            .spawn("echo run1 > /tmp/isolation_marker", dir.path())
            .await
            .unwrap();

        // Second container should have a fresh /tmp — no marker.
        let out = sandbox
            .spawn(
                "test -e /tmp/isolation_marker && echo exists || echo absent",
                dir.path(),
            )
            .await
            .unwrap();
        assert!(
            out.stdout.contains("absent"),
            "containers should have isolated /tmp, got: {:?}",
            out.stdout
        );
    }

    #[tokio::test]
    #[ignore]
    async fn docker_sandbox_credential_dirs_absent() {
        let dir = TempDir::new().unwrap();
        let sandbox = alpine_sandbox(&dir);
        let out = sandbox
            .spawn(
                "test -e /root/.claude && echo exists || echo absent",
                dir.path(),
            )
            .await
            .unwrap();
        assert!(
            out.stdout.contains("absent"),
            "credential dir /root/.claude should not be present, got: {:?}",
            out.stdout
        );
    }

    #[tokio::test]
    #[ignore]
    async fn docker_sandbox_cannot_access_other_workspace() {
        let workspace = TempDir::new().unwrap();
        let other_workspace = TempDir::new().unwrap();

        // Write a secret file in the unmounted workspace.
        let secret_file = other_workspace.path().join("secret.txt");
        std::fs::write(&secret_file, "secret").unwrap();

        // Sandbox only mounts `workspace`, not `other_workspace`.
        let sandbox = DockerSandbox::new(DockerSandboxConfig {
            image: "alpine".to_string(),
            workspace: workspace.path().to_path_buf(),
            ro_paths: vec![],
            rw_paths: vec![],
        });

        let cmd = format!(
            "test -e {} && echo exists || echo absent",
            secret_file.display()
        );
        let out = sandbox.spawn(&cmd, workspace.path()).await.unwrap();
        assert!(
            out.stdout.contains("absent"),
            "other workspace should not be accessible, got: {:?}",
            out.stdout
        );
    }
}
