use anyhow::Result;
use async_trait::async_trait;
use landlock::{
    Access as _, AccessFs, PathBeneath, PathFd, RulesetAttr as _, RulesetCreatedAttr as _,
    RulesetStatus, ABI as LandlockABI,
};
use std::path::PathBuf;
use tokio::process::Command;

use super::{Access, Sandbox, SandboxConfig, SandboxOutput};

/// Landlock-backed sandbox: applies filesystem access rules before executing
/// the child command.
///
/// Uses `tokio::process::Command::pre_exec` (an unsafe hook that runs in the
/// child process after `fork()` but before `exec()`) to install Landlock
/// rules in the child only, so the parent daemon is unaffected.
///
/// `available()` checks whether the running kernel supports Landlock ABI v1.
/// It does **not** reflect the outcome of the last `spawn` call.  If the
/// kernel does not support Landlock the restriction degrades gracefully and
/// errors may be logged to stderr, but the command still runs without
/// isolation rather than failing hard.
pub struct LandlockSandbox {
    config: SandboxConfig,
}

impl LandlockSandbox {
    pub fn new(config: SandboxConfig) -> Self {
        Self { config }
    }

    /// Check whether Landlock ABI v1 is supported by the running kernel.
    #[allow(dead_code)]
    fn kernel_supports_landlock() -> bool {
        landlock::Ruleset::default()
            .handle_access(AccessFs::from_all(LandlockABI::V1))
            .is_ok()
    }
}

#[async_trait]
impl Sandbox for LandlockSandbox {
    fn name(&self) -> &str {
        "landlock"
    }

    fn available(&self) -> bool {
        Self::kernel_supports_landlock()
    }

    async fn spawn(&self, cmd: &str, cwd: &std::path::Path) -> Result<SandboxOutput> {
        // Clone config data we need inside the child closure.
        let workspace = self.config.workspace.clone();
        let workspace_access = self.config.workspace_access.clone();
        let allowed_paths = self.config.allowed_paths.clone();

        // Build the list of (PathBuf, rw_flag) to allow, evaluated in the parent.
        let mut path_rules: Vec<(PathBuf, bool)> = Vec::new();

        // Standard read-only paths needed by virtually every shell command.
        for ro in &[
            "/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/proc", "/dev",
        ] {
            let p = PathBuf::from(ro);
            if p.exists() {
                path_rules.push((p, false));
            }
        }

        // Workspace
        match workspace_access {
            Access::Rw => path_rules.push((workspace, true)),
            Access::Ro => path_rules.push((workspace, false)),
            Access::None => {} // no access to workspace
        }

        // Configured allowed paths
        for (p, acc) in &allowed_paths {
            match acc {
                Access::None => {} // skip — don't grant any access
                Access::Ro => {
                    if p.exists() {
                        path_rules.push((p.clone(), false));
                    }
                }
                Access::Rw => {
                    if p.exists() {
                        path_rules.push((p.clone(), true));
                    }
                }
            }
        }

        // Precompute PathFd handles and rule access levels in the PARENT process
        // before fork. This avoids heap allocations and path.exists() calls inside
        // pre_exec (which runs in the child after fork and must be async-signal-safe).
        // Only the final restrict_self() and ruleset creation happen in pre_exec.
        let precomputed: Vec<(PathFd, bool)> = path_rules
            .into_iter()
            .filter_map(|(p, rw)| PathFd::new(&p).ok().map(|fd| (fd, rw)))
            .collect();

        let mut command = Command::new("sh");
        command.arg("-c").arg(cmd).current_dir(cwd);

        // Safety: pre_exec runs in the child after fork, before exec.
        // `precomputed` contains owned PathFd (file descriptors) and bool flags —
        // plain data with no Mutexes or async resources shared with the parent.
        // IMPORTANT: Do not call tracing/logging macros inside pre_exec —
        // they are not async-signal-safe and can deadlock after fork.
        //
        // The closure is defined *outside* the unsafe block so that `libc::write`
        // inside it is not covered by the outer unsafe scope — requiring its own
        // explicit unsafe block as a visible signal that it is async-signal-safe.
        let pre_exec_fn = move || {
            if let Err(_e) = apply_landlock_rules(&precomputed) {
                // Write to stderr using a raw libc::write — async-signal-safe.
                let msg = b"[sandbox/landlock] restriction failed (degraded)\n";
                let _ = unsafe { libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len()) };
            }
            Ok(())
        };
        unsafe {
            command.pre_exec(pre_exec_fn);
        }

        let output = command.output().await.map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("landlock sandbox: spawning shell command: {cmd}: {e}"),
            )
        })?;

        Ok(SandboxOutput {
            status: output.status,
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

/// Install Landlock filesystem rules for the given precomputed (PathFd, rw) list.
///
/// Called inside the child process (inside `pre_exec`). PathFd handles are
/// opened in the parent before fork, so no heap allocation or path lookups
/// are needed here — only the ruleset creation and restrict_self() call.
///
/// IMPORTANT: This function must not call tracing macros or any code that
/// acquires locks held by other threads — it runs after `fork()` in a
/// single-threaded child and such calls can deadlock.
fn apply_landlock_rules(rules: &[(PathFd, bool)]) -> std::io::Result<()> {
    // Compose the full set of filesystem accesses we handle.
    let handled = AccessFs::from_all(LandlockABI::V1);

    let ruleset = landlock::Ruleset::default()
        .handle_access(handled)
        .map_err(std::io::Error::other)?;

    let mut ruleset_created = ruleset.create().map_err(std::io::Error::other)?;

    for (fd, rw) in rules {
        let access = if *rw {
            AccessFs::from_all(LandlockABI::V1)
        } else {
            AccessFs::from_read(LandlockABI::V1)
        };

        // PathFd implements AsFd; PathBeneath::new takes AsFd.
        ruleset_created = ruleset_created
            .add_rule(PathBeneath::new(fd, access))
            .map_err(std::io::Error::other)?;
    }

    let status = ruleset_created
        .restrict_self()
        .map_err(std::io::Error::other)?;

    // Do NOT use tracing here — we are inside pre_exec (post-fork child).
    // Use raw libc::write for any stderr output.
    match status.ruleset {
        RulesetStatus::FullyEnforced => {}
        RulesetStatus::PartiallyEnforced => {
            let msg = b"[sandbox/landlock] partially enforced (older kernel ABI)\n";
            unsafe {
                libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len());
            }
        }
        RulesetStatus::NotEnforced => {
            let msg = b"[sandbox/landlock] not enforced (kernel lacks support)\n";
            unsafe {
                libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len());
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::process::CommandExt;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn default_sandbox() -> LandlockSandbox {
        LandlockSandbox::new(SandboxConfig::default())
    }

    #[test]
    fn name_is_landlock() {
        assert_eq!(default_sandbox().name(), "landlock");
    }

    #[test]
    fn available_does_not_panic() {
        // Just check it runs without panicking; result depends on kernel.
        let _ = default_sandbox().available();
    }

    #[tokio::test]
    async fn runs_basic_command() {
        let tmp = TempDir::new().unwrap();
        let cfg = SandboxConfig {
            enabled: true,
            backend: "landlock".into(),
            workspace: tmp.path().to_path_buf(),
            workspace_access: Access::Rw,
            ..SandboxConfig::default()
        };
        let sb = LandlockSandbox::new(cfg);
        let out = sb.spawn("echo hello", tmp.path()).await.unwrap();
        assert_eq!(out.stdout.trim(), "hello");
        assert!(out.status.success());
    }

    // ------------------------------------------------------------------
    // New regression tests for PR #21
    // ------------------------------------------------------------------

    /// Helper: create a LandlockSandbox with a single allowed_path entry.
    fn sandbox_with_allowed(
        tmp_workspace: &TempDir,
        path: PathBuf,
        access: Access,
    ) -> LandlockSandbox {
        LandlockSandbox::new(SandboxConfig {
            enabled: true,
            backend: "landlock".into(),
            workspace: tmp_workspace.path().to_path_buf(),
            workspace_access: Access::Rw,
            allowed_paths: vec![(path, access)],
            ..SandboxConfig::default()
        })
    }

    /// Check Landlock fully-enforced status for the running kernel by probing
    /// in a subprocess. This avoids applying the irreversible restrict_self()
    /// call to the test runner process itself.
    /// Returns `true` iff we can rely on restrictions being honored.
    fn landlock_fully_enforced() -> bool {
        use landlock::{Access as _, AccessFs, ABI as LandlockABI};
        // Spawn a child that calls restrict_self and exits with code:
        //   2 = FullyEnforced, 1 = PartiallyEnforced, 0 = NotEnforced, 99 = error
        let mut probe = std::process::Command::new("sh");
        probe.arg("-c").arg("exit 0");
        unsafe {
            probe.pre_exec(move || {
                let handled = AccessFs::from_all(LandlockABI::V1);
                if let Ok(rs) = landlock::Ruleset::default().handle_access(handled) {
                    if let Ok(rc) = rs.create() {
                        if let Ok(status) = rc.restrict_self() {
                            let code = match status.ruleset {
                                RulesetStatus::FullyEnforced => 2,
                                RulesetStatus::PartiallyEnforced => 1,
                                RulesetStatus::NotEnforced => 0,
                            };
                            unsafe { libc::_exit(code) };
                        }
                    }
                }
                unsafe { libc::_exit(99) };
                #[allow(unreachable_code)]
                Ok(())
            });
        }
        let status = probe
            .status()
            .unwrap_or_else(|_| std::process::Command::new("false").status().unwrap());
        status.code().unwrap_or(0) == 2
    }

    #[tokio::test]
    async fn allowed_path_ro_permits_read() {
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("data.txt");
        std::fs::write(&file, "read-me").unwrap();

        let work = TempDir::new().unwrap();
        let sb = sandbox_with_allowed(&work, tmp.path().to_path_buf(), Access::Ro);

        let out = sb
            .spawn(&format!("cat '{}'", file.display()), work.path())
            .await
            .unwrap();
        assert!(out.status.success(), "ro read failed: {:?}", out.stderr);
        assert_eq!(out.stdout.trim(), "read-me");
    }

    #[tokio::test]
    async fn allowed_path_ro_denies_write() {
        if !landlock_fully_enforced() {
            eprintln!("landlock not fully enforced — skipping allowed_path_ro_denies_write");
            return;
        }

        let secret_base = std::path::Path::new("/var/tmp");
        if !secret_base.exists() {
            eprintln!("allowed_path_ro_denies_write: /var/tmp not available — skipping");
            return;
        }
        let ro_dir = match TempDir::new_in(secret_base) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("allowed_path_ro_denies_write: cannot create temp dir in /var/tmp ({e}) — skipping");
                return;
            }
        };
        let file = ro_dir.path().join("out.txt");
        std::fs::write(&file, "original").unwrap();

        let work = TempDir::new().unwrap();
        let sb = sandbox_with_allowed(&work, ro_dir.path().to_path_buf(), Access::Ro);

        let out = sb
            .spawn(
                &format!("echo overwrite > '{}'", file.display()),
                work.path(),
            )
            .await
            .unwrap();
        assert!(
            !out.status.success(),
            "write to ro path should fail; stderr={:?}",
            out.stderr
        );
    }

    #[tokio::test]
    async fn allowed_path_rw_permits_write() {
        let rw_dir = TempDir::new().unwrap();
        let file = rw_dir.path().join("out.txt");

        let work = TempDir::new().unwrap();
        let sb = sandbox_with_allowed(&work, rw_dir.path().to_path_buf(), Access::Rw);

        let out = sb
            .spawn(&format!("echo written > '{}'", file.display()), work.path())
            .await
            .unwrap();
        assert!(out.status.success(), "rw write failed: {:?}", out.stderr);
        assert_eq!(std::fs::read_to_string(&file).unwrap().trim(), "written");
    }

    #[tokio::test]
    async fn access_none_skips_rule() {
        if !landlock_fully_enforced() {
            eprintln!("landlock not fully enforced — skipping access_none_skips_rule");
            return;
        }

        let secret_base = std::path::Path::new("/var/tmp");
        if !secret_base.exists() {
            eprintln!("access_none_skips_rule: /var/tmp not available — skipping");
            return;
        }
        let none_dir = match TempDir::new_in(secret_base) {
            Ok(d) => d,
            Err(e) => {
                eprintln!(
                    "access_none_skips_rule: cannot create temp dir in /var/tmp ({e}) — skipping"
                );
                return;
            }
        };
        let file = none_dir.path().join("secret.txt");
        std::fs::write(&file, "invisible").unwrap();

        let work = TempDir::new().unwrap();
        // Access::None means no rule is added for this path → denied by Landlock allowlist
        let sb = sandbox_with_allowed(&work, none_dir.path().to_path_buf(), Access::None);

        let out = sb
            .spawn(&format!("cat '{}'", file.display()), work.path())
            .await
            .unwrap();
        assert!(
            !out.status.success(),
            "Access::None path should be inaccessible; stdout={:?}",
            out.stdout
        );
    }

    #[tokio::test]
    async fn workspace_auto_added_to_allowed_paths() {
        let workspace = TempDir::new().unwrap();
        let file = workspace.path().join("ws_file.txt");
        std::fs::write(&file, "workspace-data").unwrap();

        let cfg = SandboxConfig {
            enabled: true,
            backend: "landlock".into(),
            workspace: workspace.path().to_path_buf(),
            workspace_access: Access::Ro,
            allowed_paths: vec![],
            ..SandboxConfig::default()
        };
        let sb = LandlockSandbox::new(cfg);

        let out = sb
            .spawn(&format!("cat '{}'", file.display()), workspace.path())
            .await
            .unwrap();
        assert!(
            out.status.success(),
            "workspace read failed: {:?}",
            out.stderr
        );
        assert_eq!(out.stdout.trim(), "workspace-data");
    }

    #[tokio::test]
    async fn multiple_allowed_paths() {
        let ro_dir = TempDir::new().unwrap();
        let rw_dir = TempDir::new().unwrap();
        let ro_file = ro_dir.path().join("ro.txt");
        std::fs::write(&ro_file, "readonly").unwrap();

        let work = TempDir::new().unwrap();
        let cfg = SandboxConfig {
            enabled: true,
            backend: "landlock".into(),
            workspace: work.path().to_path_buf(),
            workspace_access: Access::Rw,
            allowed_paths: vec![
                (ro_dir.path().to_path_buf(), Access::Ro),
                (rw_dir.path().to_path_buf(), Access::Rw),
            ],
            ..SandboxConfig::default()
        };
        let sb = LandlockSandbox::new(cfg);

        // Read from ro_dir
        let out = sb
            .spawn(&format!("cat '{}'", ro_file.display()), work.path())
            .await
            .unwrap();
        assert!(
            out.status.success(),
            "multi-path ro read failed: {:?}",
            out.stderr
        );
        assert_eq!(out.stdout.trim(), "readonly");

        // Write to rw_dir
        let rw_file = rw_dir.path().join("new.txt");
        let out = sb
            .spawn(
                &format!("echo multi > '{}'", rw_file.display()),
                work.path(),
            )
            .await
            .unwrap();
        assert!(
            out.status.success(),
            "multi-path rw write failed: {:?}",
            out.stderr
        );
        assert_eq!(std::fs::read_to_string(&rw_file).unwrap().trim(), "multi");
    }

    #[tokio::test]
    async fn command_inherits_env() {
        let work = TempDir::new().unwrap();
        let cfg = SandboxConfig {
            enabled: true,
            backend: "landlock".into(),
            workspace: work.path().to_path_buf(),
            workspace_access: Access::Rw,
            ..SandboxConfig::default()
        };
        let sb = LandlockSandbox::new(cfg);

        // Save and restore the env var to avoid cross-test interference.
        let prev_val = std::env::var("SANDBOX_TEST_VAR").ok();
        std::env::set_var("SANDBOX_TEST_VAR", "hello_sandbox");
        let out = sb
            .spawn("echo $SANDBOX_TEST_VAR", work.path())
            .await
            .unwrap();
        match prev_val {
            Some(v) => std::env::set_var("SANDBOX_TEST_VAR", v),
            None => std::env::remove_var("SANDBOX_TEST_VAR"),
        }
        assert!(out.status.success(), "env command failed: {:?}", out.stderr);
        assert!(
            out.stdout.contains("hello_sandbox"),
            "env not inherited; stdout={:?}",
            out.stdout
        );
    }

    // ------------------------------------------------------------------
    // End new regression tests
    // ------------------------------------------------------------------

    /// Landlock uses an allowlist-only model: any path not in `allowed_paths`
    /// is implicitly denied.  This test verifies that a path omitted from the
    /// allowlist is inaccessible — it does NOT test `denied_paths` enforcement
    /// (which the Landlock backend does not implement).
    #[tokio::test]
    async fn path_not_in_allowlist_is_implicitly_denied() {
        // Create a "secret" directory under /var/tmp so it's not covered by
        // the default /tmp allowed path.
        let secret_base = std::path::Path::new("/var/tmp");
        if !secret_base.exists() {
            eprintln!("landlock denial test: /var/tmp not available — skipping");
            return;
        }

        let tmp = TempDir::new_in(secret_base).unwrap();
        let secret = tmp.path().join("secret");
        std::fs::create_dir_all(&secret).unwrap();
        std::fs::write(secret.join("file.txt"), "secret data").unwrap();

        // Work dir is separate; use a /tmp dir that IS explicitly allowed.
        let work_tmp = TempDir::new().unwrap();
        let work = work_tmp.path().to_path_buf();

        let cfg = SandboxConfig {
            enabled: true,
            backend: "landlock".into(),
            workspace: work.clone(),
            workspace_access: Access::Rw,
            // Only allow /tmp — not /var/tmp where secret lives.
            allowed_paths: vec![(PathBuf::from("/tmp"), Access::Rw)],
            denied_paths: vec![secret.clone()],
            ..SandboxConfig::default()
        };
        let sb = LandlockSandbox::new(cfg);

        if !sb.available() {
            eprintln!("landlock not available — skipping denial test");
            return;
        }

        // Verify Landlock fully enforced before trusting the test result.
        // If restriction is only partial or not enforced, skip rather than
        // risk a flaky failure.
        if !landlock_fully_enforced() {
            eprintln!("landlock denial test: restrict_self not fully enforced — skipping");
            return;
        }

        // The secret dir is NOT in allowed_paths, so it should be denied
        // implicitly by the Landlock allowlist.
        let out = sb
            .spawn(
                &format!("cat '{}'", secret.join("file.txt").display()),
                &work,
            )
            .await
            .unwrap();
        // Under Landlock the cat should fail (non-zero exit).
        assert!(
            !out.status.success(),
            "reading path absent from allowlist should fail; stdout={:?} stderr={:?}",
            out.stdout,
            out.stderr
        );
    }
}
