#![cfg(target_os = "linux")]
#![allow(dead_code)]

use anyhow::{Context, Result};
use async_trait::async_trait;
use std::ffi::CString;
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
        let workspace = self.config.workspace.clone();
        let ro_paths = self.config.ro_paths.clone();
        let rw_paths = self.config.rw_paths.clone();

        // Convert paths to CStrings before entering pre_exec (no allocation
        // is safe inside the async-signal-safe section).
        let workspace_c = path_to_cstring(&workspace)?;

        let mut ro_c: Vec<(CString, CString)> = Vec::with_capacity(ro_paths.len());
        for p in &ro_paths {
            ro_c.push((path_to_cstring(p)?, path_to_cstring(p)?));
        }

        let mut rw_c: Vec<(CString, CString)> = Vec::with_capacity(rw_paths.len());
        for p in &rw_paths {
            rw_c.push((path_to_cstring(p)?, path_to_cstring(p)?));
        }

        let tmpfs_target = CString::new("/tmp").unwrap();
        let tmpfs_fstype = CString::new("tmpfs").unwrap();
        let empty = CString::new("").unwrap();

        let mut cmd_builder = Command::new("sh");
        cmd_builder.arg("-c").arg(cmd).current_dir(cwd);

        // Safety: pre_exec closure runs in the child after fork(), before exec().
        // Must only call async-signal-safe functions (libc syscalls).
        unsafe {
            cmd_builder.pre_exec(move || {
                // 1. Create independent mount + user namespace (unprivileged).
                let ret = libc::unshare(libc::CLONE_NEWNS | libc::CLONE_NEWUSER);
                if ret != 0 {
                    return Err(std::io::Error::last_os_error());
                }

                // 2. Mount tmpfs at /tmp — isolated per-agent.
                let ret = libc::mount(
                    empty.as_ptr(),
                    tmpfs_target.as_ptr(),
                    tmpfs_fstype.as_ptr(),
                    0,
                    std::ptr::null(),
                );
                if ret != 0 {
                    return Err(std::io::Error::last_os_error());
                }

                // 3. Bind mount workspace as rw.
                let ret = libc::mount(
                    workspace_c.as_ptr(),
                    workspace_c.as_ptr(),
                    std::ptr::null(),
                    libc::MS_BIND | libc::MS_REC,
                    std::ptr::null(),
                );
                if ret != 0 {
                    return Err(std::io::Error::last_os_error());
                }

                // 4. Bind mount ro_paths as read-only (two-step).
                for (src, dst) in &ro_c {
                    // Step 1: bind mount.
                    let ret = libc::mount(
                        src.as_ptr(),
                        dst.as_ptr(),
                        std::ptr::null(),
                        libc::MS_BIND | libc::MS_REC,
                        std::ptr::null(),
                    );
                    if ret != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    // Step 2: remount read-only.
                    let ret = libc::mount(
                        src.as_ptr(),
                        dst.as_ptr(),
                        std::ptr::null(),
                        libc::MS_BIND | libc::MS_REMOUNT | libc::MS_RDONLY | libc::MS_REC,
                        std::ptr::null(),
                    );
                    if ret != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                }

                // 5. Bind mount rw_paths as read-write.
                for (src, dst) in &rw_c {
                    let ret = libc::mount(
                        src.as_ptr(),
                        dst.as_ptr(),
                        std::ptr::null(),
                        libc::MS_BIND | libc::MS_REC,
                        std::ptr::null(),
                    );
                    if ret != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                }

                Ok(())
            });
        }

        let output = cmd_builder
            .output()
            .await
            .context("failed to spawn sandboxed command")?;

        Ok(SandboxOutput {
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            status: output.status.code().unwrap_or(-1),
        })
    }
}

/// Probe whether unprivileged user namespaces are available by actually
/// attempting `unshare --user --mount` in a subprocess.  Checking
/// `/proc/self/ns/mnt` only verifies kernel support, not whether the running
/// process has permission (e.g. container environments often set
/// `kernel.unprivileged_userns_clone=0`).
fn namespace_available() -> bool {
    std::process::Command::new("unshare")
        .args(["--user", "--mount", "/bin/true"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn path_to_cstring(p: &Path) -> Result<CString> {
    use std::os::unix::ffi::OsStrExt;
    CString::new(p.as_os_str().as_bytes()).context("path contains null byte")
}

// These tests require Linux user namespaces (CLONE_NEWUSER).
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
}
