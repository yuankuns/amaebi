use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tokio::process::Command;
use tokio::sync::Mutex;

use super::{Sandbox, SandboxOutput};

// ---------------------------------------------------------------------------
// DockerSandboxConfig
// ---------------------------------------------------------------------------

#[allow(dead_code)]
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
    container_id: Mutex<Option<String>>,
}

impl DockerSandbox {
    #[allow(dead_code)]
    pub fn new(config: DockerSandboxConfig) -> Self {
        Self {
            config,
            container_id: Mutex::new(None),
        }
    }

    /// Ensure the long-running container is started, returning its ID.
    async fn ensure_container(&self) -> Result<String> {
        let mut guard = self.container_id.lock().await;
        if let Some(id) = guard.as_ref() {
            return Ok(id.clone());
        }

        let workspace_str = self
            .config
            .workspace
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("workspace path is not valid UTF-8"))?;

        let mut docker = Command::new("docker");
        docker.args(["run", "-d", "--rm", "--network", "none"]);
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
        docker.args(["sleep", "infinity"]);

        let output = docker
            .output()
            .await
            .context("starting persistent docker container")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("docker run -d failed: {}", stderr.trim());
        }

        let id = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if id.is_empty() {
            anyhow::bail!("docker run -d returned an empty container ID");
        }

        *guard = Some(id.clone());
        Ok(id)
    }
}

#[async_trait::async_trait]
impl Sandbox for DockerSandbox {
    async fn spawn(&self, cmd: &str, cwd: &Path) -> Result<SandboxOutput> {
        // Validate cwd is inside the workspace.
        let cwd_canon = cwd
            .canonicalize()
            .with_context(|| format!("canonicalizing cwd: {}", cwd.display()))?;
        let workspace_canon = self.config.workspace.canonicalize().with_context(|| {
            format!(
                "canonicalizing workspace: {}",
                self.config.workspace.display()
            )
        })?;
        if cwd_canon.strip_prefix(&workspace_canon).is_err() {
            anyhow::bail!(
                "cwd {} is not under workspace {}",
                cwd.display(),
                self.config.workspace.display()
            );
        }

        let cwd_str = cwd_canon
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("cwd is not valid UTF-8"))?;

        let container_id = self.ensure_container().await?;

        let output = Command::new("docker")
            .args(["exec", "-w", cwd_str, &container_id, "sh", "-c", cmd])
            .output()
            .await
            .context("spawning docker exec")?;

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

impl Drop for DockerSandbox {
    fn drop(&mut self) {
        if let Ok(guard) = self.container_id.try_lock() {
            if let Some(id) = guard.as_ref() {
                let _ = std::process::Command::new("docker")
                    .args(["rm", "-f", id])
                    .output();
            }
        }
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
    // Requires: amaebi-sandbox:bookworm-slim Docker image
    // Build with: ./scripts/build-sandbox-image.sh

    fn test_sandbox(workspace: &TempDir) -> DockerSandbox {
        DockerSandbox::new(DockerSandboxConfig {
            image: "amaebi-sandbox:bookworm-slim".to_string(),
            workspace: workspace.path().to_path_buf(),
            ro_paths: vec![],
            rw_paths: vec![],
        })
    }

    #[tokio::test]
    #[ignore]
    async fn docker_sandbox_runs_command() {
        let dir = TempDir::new().unwrap();
        let sandbox = test_sandbox(&dir);
        let out = sandbox.spawn("echo hello", dir.path()).await.unwrap();
        assert!(
            out.stdout.contains("hello"),
            "expected 'hello' in stdout, got: {:?}",
            out.stdout
        );
    }

    /// Two spawn() calls on the *same* sandbox share /tmp state.
    #[tokio::test]
    #[ignore]
    async fn docker_sandbox_shares_state_within_session() {
        let dir = TempDir::new().unwrap();
        let sandbox = test_sandbox(&dir);

        sandbox
            .spawn("echo session_data > /tmp/shared_file", dir.path())
            .await
            .unwrap();

        let out = sandbox
            .spawn("cat /tmp/shared_file", dir.path())
            .await
            .unwrap();
        assert!(
            out.stdout.contains("session_data"),
            "second spawn should see file written by first spawn, got: {:?}",
            out.stdout
        );
    }

    /// Two *separate* DockerSandbox instances have isolated /tmp.
    #[tokio::test]
    #[ignore]
    async fn docker_sandbox_isolates_tmp() {
        let dir = TempDir::new().unwrap();
        let sandbox1 = test_sandbox(&dir);
        let sandbox2 = test_sandbox(&dir);

        // First sandbox writes a marker to /tmp.
        sandbox1
            .spawn("echo run1 > /tmp/isolation_marker", dir.path())
            .await
            .unwrap();

        // Second sandbox (separate container) should have a fresh /tmp — no marker.
        let out = sandbox2
            .spawn(
                "test -e /tmp/isolation_marker && echo exists || echo absent",
                dir.path(),
            )
            .await
            .unwrap();
        assert!(
            out.stdout.contains("absent"),
            "separate sandbox instances should have isolated /tmp, got: {:?}",
            out.stdout
        );
    }

    #[tokio::test]
    #[ignore]
    async fn docker_sandbox_credential_dirs_absent() {
        let dir = TempDir::new().unwrap();
        let sandbox = test_sandbox(&dir);
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
            image: "amaebi-sandbox:bookworm-slim".to_string(),
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
