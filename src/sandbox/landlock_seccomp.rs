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
