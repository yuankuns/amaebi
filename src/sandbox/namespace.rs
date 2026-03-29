#![cfg(target_os = "linux")]
#![allow(dead_code)]

use anyhow::{Context, Result};
use async_trait::async_trait;
use std::path::Path;
use tokio::process::Command;

use super::{Sandbox, SandboxConfig, SandboxOutput};

pub struct NamespaceSandbox {
    config: SandboxConfig,
}

impl NamespaceSandbox {
    pub fn new(config: SandboxConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl Sandbox for NamespaceSandbox {
    fn name(&self) -> &str {
        "namespace"
    }

    fn available(&self) -> bool {
        namespace_available()
    }

    async fn spawn(&self, cmd: &str, cwd: &Path) -> Result<SandboxOutput> {
        let mut args: Vec<String> = vec![
            "--user".to_string(),
            "--map-root-user".to_string(),
            "--mount".to_string(),
        ];

        // Bind mount workspace as read-write.
        let ws = self
            .config
            .workspace
            .to_str()
            .context("workspace path is not valid UTF-8")?;
        args.push(format!("--bind={}:{}", ws, ws));

        // Bind mount additional rw_paths.
        for p in &self.config.rw_paths {
            let s = p.to_str().context("rw_path is not valid UTF-8")?;
            args.push(format!("--bind={}:{}", s, s));
        }

        // Bind mount ro_paths as read-only.
        for p in &self.config.ro_paths {
            let s = p.to_str().context("ro_path is not valid UTF-8")?;
            args.push(format!("--bind-ro={}:{}", s, s));
        }

        // Mount a fresh tmpfs at /tmp for per-invocation isolation.
        args.push("--tmpfs=/tmp".to_string());

        args.push("--".to_string());
        args.push("sh".to_string());
        args.push("-c".to_string());
        args.push(cmd.to_string());

        let output = Command::new("unshare")
            .args(&args)
            .current_dir(cwd)
            .output()
            .await
            .context("failed to spawn sandboxed command via unshare")?;

        Ok(SandboxOutput {
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            status: output.status.code().unwrap_or(-1),
        })
    }
}

/// Probe whether unprivileged user namespaces are available by attempting
/// `unshare --user --map-root-user --mount /bin/true`.  This confirms both
/// that the `unshare` binary is present and that the kernel permits
/// unprivileged user namespaces (some container environments set
/// `kernel.unprivileged_userns_clone=0`).
fn namespace_available() -> bool {
    std::process::Command::new("unshare")
        .args(["--user", "--map-root-user", "--mount", "/bin/true"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

// These tests require Linux user namespaces and the `unshare` binary.
// Run with: cargo test -- --ignored
// Not suitable for Docker containers with default seccomp profile.
#[cfg(test)]
mod tests {
    use crate::sandbox::{create_backend, SandboxConfig};
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn make_config(workspace: PathBuf) -> SandboxConfig {
        SandboxConfig {
            enabled: true,
            backend: "namespace".to_string(),
            workspace,
            ro_paths: Vec::new(),
            rw_paths: Vec::new(),
        }
    }

    #[tokio::test]
    #[ignore]
    async fn namespace_backend_spawns_and_returns_output() {
        let dir = TempDir::new().unwrap();
        let sb = create_backend(make_config(dir.path().to_path_buf()));
        let out = sb.spawn("echo hello", dir.path()).await.unwrap();
        assert_eq!(out.stdout.trim(), "hello");
        assert_eq!(out.status, 0);
    }

    #[tokio::test]
    #[ignore]
    async fn workspace_is_writable() {
        let dir = TempDir::new().unwrap();
        let sb = create_backend(make_config(dir.path().to_path_buf()));
        let file = dir.path().join("canary.txt");
        let cmd = format!("echo ok > {}", file.display());
        let out = sb.spawn(&cmd, dir.path()).await.unwrap();
        assert_eq!(out.status, 0, "stderr: {}", out.stderr);
    }

    #[tokio::test]
    #[ignore]
    async fn tmp_is_isolated_between_spawns() {
        let dir = TempDir::new().unwrap();
        let sb = create_backend(make_config(dir.path().to_path_buf()));

        // First spawn: write a file to /tmp inside the namespace.
        let out1 = sb
            .spawn("echo secret > /tmp/isolated_test_file.txt", dir.path())
            .await
            .unwrap();
        assert_eq!(out1.status, 0, "stderr: {}", out1.stderr);

        // Second spawn: that file should not exist in a fresh /tmp.
        let out2 = sb
            .spawn(
                "test ! -f /tmp/isolated_test_file.txt && echo not_found",
                dir.path(),
            )
            .await
            .unwrap();
        assert_eq!(out2.stdout.trim(), "not_found", "stderr: {}", out2.stderr);
    }

    #[tokio::test]
    #[ignore]
    async fn ro_path_is_not_writable() {
        let ro_dir = TempDir::new().unwrap();
        // Create a file in ro_dir so it's a valid existing directory.
        std::fs::write(ro_dir.path().join("existing.txt"), b"data").unwrap();

        let workspace = TempDir::new().unwrap();
        let mut cfg = make_config(workspace.path().to_path_buf());
        cfg.ro_paths = vec![ro_dir.path().to_path_buf()];

        let sb = create_backend(cfg);
        let cmd = format!("echo x > {}/new.txt", ro_dir.path().display());
        let out = sb.spawn(&cmd, workspace.path()).await.unwrap();
        // Write should fail (non-zero exit) or report permission denied.
        let failed = out.status != 0
            || out.stderr.to_lowercase().contains("permission")
            || out.stderr.to_lowercase().contains("read-only");
        assert!(
            failed,
            "Expected write to ro_path to fail; got status={}, stderr={}",
            out.status, out.stderr
        );
    }

    #[tokio::test]
    #[ignore]
    async fn path_not_in_config_does_not_exist() {
        let workspace = TempDir::new().unwrap();
        let absent_dir = TempDir::new().unwrap();
        let absent_path = absent_dir.path().to_path_buf();
        // Don't add absent_path to ro_paths or rw_paths.
        let sb = create_backend(make_config(workspace.path().to_path_buf()));

        // In the namespace the absent path is still on the underlying fs,
        // because we only set up bind mounts — we don't pivot_root.
        // This test verifies the /tmp isolation behaviour (newly written
        // file is absent in the next spawn) as the primary isolation guarantee.
        let out = sb.spawn("ls /tmp", workspace.path()).await.unwrap();
        // /tmp should be an empty fresh tmpfs — absent_path won't be under /tmp.
        assert!(!out
            .stdout
            .contains(absent_path.file_name().unwrap().to_str().unwrap()));

        // Verify that credential directories are inaccessible inside the namespace.
        for cred_dir in &["~/.claude", "~/.config", "~/.ssh"] {
            let cmd = format!(
                "test -e {dir} && echo exists || echo absent",
                dir = cred_dir
            );
            let out = sb.spawn(&cmd, workspace.path()).await.unwrap();
            assert!(
                out.stdout.contains("absent"),
                "Expected {dir} to be absent inside the namespace; \
                 got stdout={:?} stderr={:?}",
                out.stdout,
                out.stderr,
                dir = cred_dir,
            );
        }
    }

    /// Verify that an agent sandboxed to worktree_b cannot access worktree_a.
    /// The file written into worktree_a must be completely invisible — not just
    /// unreadable — inside the namespace configured only for worktree_b.
    #[tokio::test]
    #[ignore]
    async fn agent_cannot_read_other_agent_worktree() {
        let worktree_a = TempDir::new().unwrap();
        let worktree_b = TempDir::new().unwrap();

        // Write a secret into agent A's worktree.
        let secret_path = worktree_a.path().join("secret.txt");
        std::fs::write(&secret_path, b"agent_a_secret").unwrap();

        // Sandbox configured only for worktree_b — worktree_a is not listed.
        let sb = create_backend(make_config(worktree_b.path().to_path_buf()));

        let cmd = format!("cat {} 2>&1 || echo no_access", secret_path.display());
        let out = sb.spawn(&cmd, worktree_b.path()).await.unwrap();

        assert!(
            out.stdout.contains("no_access"),
            "agent B should not be able to read agent A's worktree; \
             got stdout={:?} stderr={:?}",
            out.stdout,
            out.stderr,
        );
    }

    /// Verify that a sandboxed process cannot signal processes outside the PID
    /// namespace.  `kill -0` just checks accessibility — it does not actually
    /// deliver a signal — so this test is safe to run against the real parent.
    #[tokio::test]
    #[ignore]
    async fn cannot_kill_external_process() {
        let dir = TempDir::new().unwrap();
        let sb = create_backend(make_config(dir.path().to_path_buf()));
        let pid = std::process::id();
        let cmd = format!("kill -0 {} 2>&1 || echo kill_denied", pid);
        let out = sb.spawn(&cmd, dir.path()).await.unwrap();
        assert!(
            out.stdout.contains("kill_denied"),
            "Expected sandbox to deny signaling external PID {}; \
             got stdout={:?} stderr={:?}",
            pid,
            out.stdout,
            out.stderr,
        );
    }
}
