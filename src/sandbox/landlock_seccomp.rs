use anyhow::Result;
use landlock::{
    Access as _, AccessFs, PathBeneath, PathFd, RulesetAttr as _, RulesetCreatedAttr as _,
    RulesetStatus, ABI as LandlockABI,
};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::Command;

use super::{Access, Sandbox, SandboxConfig, SandboxOutput};

/// Landlock-backed sandbox: applies filesystem access rules before executing
/// the child command.
///
/// Uses `std::process::Command::pre_exec` (an unsafe hook that runs in the
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

impl Sandbox for LandlockSandbox {
    fn name(&self) -> &str {
        "landlock"
    }

    fn available(&self) -> bool {
        Self::kernel_supports_landlock()
    }

    fn spawn(&self, cmd: &str, cwd: &std::path::Path) -> Result<SandboxOutput> {
        // Clone config data we need inside the child closure.
        let workspace = self.config.workspace.clone();
        let workspace_access = self.config.workspace_access.clone();
        let allowed_paths = self.config.allowed_paths.clone();

        // Build the list of (PathBuf, rw_flag) to allow, evaluated in the parent.
        // We collect into a Vec so the pre_exec closure is 'static-compatible.
        let mut rules: Vec<(PathBuf, bool)> = Vec::new();

        // Standard read-only paths needed by virtually every shell command.
        for ro in &[
            "/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/proc", "/dev",
        ] {
            let p = PathBuf::from(ro);
            if p.exists() {
                rules.push((p, false));
            }
        }

        // Workspace
        match workspace_access {
            Access::Rw => rules.push((workspace, true)),
            Access::Ro => rules.push((workspace, false)),
            Access::None => {} // no access to workspace
        }

        // Configured allowed paths
        for (p, acc) in &allowed_paths {
            match acc {
                Access::None => {} // skip — don't grant any access
                Access::Ro => {
                    if p.exists() {
                        rules.push((p.clone(), false));
                    }
                }
                Access::Rw => {
                    if p.exists() {
                        rules.push((p.clone(), true));
                    }
                }
            }
        }

        let mut command = Command::new("sh");
        command.arg("-c").arg(cmd).current_dir(cwd);

        // Safety: pre_exec runs in the child after fork, before exec.
        // All values captured here are cloned plain data (Vec<(PathBuf, bool)>),
        // so no async resources or Mutexes are shared.
        // IMPORTANT: Do not call tracing/logging macros inside pre_exec —
        // they are not async-signal-safe and can deadlock after fork.
        unsafe {
            command.pre_exec(move || {
                if let Err(_e) = apply_landlock_rules(&rules) {
                    // Write to stderr using a raw libc::write — async-signal-safe.
                    let msg = b"[sandbox/landlock] restriction failed (degraded)\n";
                    libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len());
                }
                Ok(())
            });
        }

        let output = command.output().map_err(|e| {
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

/// Install Landlock filesystem rules for the given path list.
///
/// Called inside the child process (inside `pre_exec`).  Returns `Ok(())`
/// on success or graceful degradation, `Err` only on unexpected I/O errors.
///
/// IMPORTANT: This function must not call tracing macros or any code that
/// acquires locks held by other threads — it runs after `fork()` in a
/// single-threaded child and such calls can deadlock.
fn apply_landlock_rules(rules: &[(PathBuf, bool)]) -> std::io::Result<()> {
    // Compose the full set of filesystem accesses we handle.
    let handled = AccessFs::from_all(LandlockABI::V1);

    let ruleset = landlock::Ruleset::default()
        .handle_access(handled)
        .map_err(std::io::Error::other)?;

    let mut ruleset_created = ruleset.create().map_err(std::io::Error::other)?;

    for (path, rw) in rules {
        if !path.exists() {
            continue;
        }
        let fd = match PathFd::new(path) {
            Ok(f) => f,
            Err(_) => continue, // path not accessible — skip
        };

        let access = if *rw {
            AccessFs::from_all(LandlockABI::V1)
        } else {
            AccessFs::from_read(LandlockABI::V1)
        };

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

    #[test]
    fn runs_basic_command() {
        let tmp = TempDir::new().unwrap();
        let cfg = SandboxConfig {
            enabled: true,
            backend: "landlock".into(),
            workspace: tmp.path().to_path_buf(),
            workspace_access: Access::Rw,
            ..SandboxConfig::default()
        };
        let sb = LandlockSandbox::new(cfg);
        let out = sb.spawn("echo hello", tmp.path()).unwrap();
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

    /// Check Landlock fully-enforced status for the running kernel.
    /// Returns `true` iff we can rely on restrictions being honored.
    fn landlock_fully_enforced() -> bool {
        use landlock::{Access as _, AccessFs, ABI as LandlockABI};
        let result = (|| -> std::result::Result<bool, Box<dyn std::error::Error>> {
            let handled = AccessFs::from_all(LandlockABI::V1);
            let rc = landlock::Ruleset::default()
                .handle_access(handled)?
                .create()?;
            let status = rc.restrict_self()?;
            Ok(matches!(status.ruleset, RulesetStatus::FullyEnforced))
        })();
        result.unwrap_or(false)
    }

    #[test]
    fn allowed_path_ro_permits_read() {
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("data.txt");
        std::fs::write(&file, "read-me").unwrap();

        let work = TempDir::new().unwrap();
        let sb = sandbox_with_allowed(&work, tmp.path().to_path_buf(), Access::Ro);

        let out = sb
            .spawn(&format!("cat '{}'", file.display()), work.path())
            .unwrap();
        assert!(out.status.success(), "ro read failed: {:?}", out.stderr);
        assert_eq!(out.stdout.trim(), "read-me");
    }

    #[test]
    fn allowed_path_ro_denies_write() {
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
            .unwrap();
        assert!(
            !out.status.success(),
            "write to ro path should fail; stderr={:?}",
            out.stderr
        );
    }

    #[test]
    fn allowed_path_rw_permits_write() {
        let rw_dir = TempDir::new().unwrap();
        let file = rw_dir.path().join("out.txt");

        let work = TempDir::new().unwrap();
        let sb = sandbox_with_allowed(&work, rw_dir.path().to_path_buf(), Access::Rw);

        let out = sb
            .spawn(&format!("echo written > '{}'", file.display()), work.path())
            .unwrap();
        assert!(out.status.success(), "rw write failed: {:?}", out.stderr);
        assert_eq!(std::fs::read_to_string(&file).unwrap().trim(), "written");
    }

    #[test]
    fn access_none_skips_rule() {
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
            .unwrap();
        assert!(
            !out.status.success(),
            "Access::None path should be inaccessible; stdout={:?}",
            out.stdout
        );
    }

    #[test]
    fn workspace_auto_added_to_allowed_paths() {
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
            .unwrap();
        assert!(
            out.status.success(),
            "workspace read failed: {:?}",
            out.stderr
        );
        assert_eq!(out.stdout.trim(), "workspace-data");
    }

    #[test]
    fn multiple_allowed_paths() {
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
            .unwrap();
        assert!(
            out.status.success(),
            "multi-path rw write failed: {:?}",
            out.stderr
        );
        assert_eq!(std::fs::read_to_string(&rw_file).unwrap().trim(), "multi");
    }

    #[test]
    fn command_inherits_env() {
        let work = TempDir::new().unwrap();
        let cfg = SandboxConfig {
            enabled: true,
            backend: "landlock".into(),
            workspace: work.path().to_path_buf(),
            workspace_access: Access::Rw,
            ..SandboxConfig::default()
        };
        let sb = LandlockSandbox::new(cfg);

        // Use a fixed env var that should be inherited from the test process.
        // We set it in the test process and check it appears in child output.
        std::env::set_var("SANDBOX_TEST_VAR", "hello_sandbox");
        let out = sb.spawn("echo $SANDBOX_TEST_VAR", work.path()).unwrap();
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
    #[test]
    fn path_not_in_allowlist_is_implicitly_denied() {
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
        let enforcement_check = {
            use landlock::{Access as _, AccessFs, ABI as LandlockABI};
            let rules: Vec<(PathBuf, bool)> = vec![];
            // Probe restrict_self on a throwaway ruleset in a temp child.
            // We do this by spawning a child that just exits so we can check
            // stdout for enforcement level.
            let mut probe = std::process::Command::new("sh");
            probe.arg("-c").arg("exit 0");
            // We can't easily call restrict_self without forking, so we use a
            // best-effort check: if available() is true the kernel handle_access
            // call succeeded, but restrict_self may yield less-than-full
            // enforcement. We use a separate spawned probe via pre_exec.
            let enforced = std::sync::Arc::new(std::sync::Mutex::new(None::<RulesetStatus>));
            let enforced_clone = enforced.clone();
            unsafe {
                probe.pre_exec(move || {
                    let handled = AccessFs::from_all(LandlockABI::V1);
                    if let Ok(rs) = landlock::Ruleset::default().handle_access(handled) {
                        if let Ok(rc) = rs.create() {
                            if let Ok(status) = rc.restrict_self() {
                                // Store status then exit child immediately.
                                let level = match status.ruleset {
                                    RulesetStatus::FullyEnforced => 2,
                                    RulesetStatus::PartiallyEnforced => 1,
                                    RulesetStatus::NotEnforced => 0,
                                };
                                // Encode in exit code: 2=full, 1=partial, 0=none
                                unsafe { libc::_exit(level) };
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
            status.code().unwrap_or(0)
        };

        if enforcement_check < 2 {
            eprintln!(
                "landlock denial test: restrict_self not fully enforced (code={enforcement_check}) — skipping"
            );
            return;
        }

        // The secret dir is NOT in allowed_paths, so it should be denied
        // implicitly by the Landlock allowlist.
        let out = sb
            .spawn(
                &format!("cat '{}'", secret.join("file.txt").display()),
                &work,
            )
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
