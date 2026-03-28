use anyhow::{Context, Result};
use landlock::{
    Access as LLAccess, AccessFs, PathBeneath, PathFd, RulesetAttr, RulesetCreatedAttr,
    RulesetStatus, ABI as LandlockABI,
};
use std::os::unix::process::CommandExt;
use std::process::Command;

use super::{Access, Sandbox, SandboxConfig, SandboxOutput};

/// Landlock-backed sandbox: applies filesystem access rules before executing
/// the child command.
///
/// Uses `std::process::Command::pre_exec` (an unsafe hook that runs in the
/// child process after `fork()` but before `exec()`) to install Landlock
/// rules in the child only, so the parent daemon is unaffected.
///
/// If the running kernel does not support Landlock (kernel < 5.13, or the
/// feature is compiled out), the restriction silently degrades and the command
/// runs without isolation rather than failing.  `available()` reflects whether
/// full enforcement was possible on the last `spawn` call; it is advisory only.
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

        // Build the list of (path, rw_flag) to allow, evaluated in the parent.
        // We collect into a Vec so the pre_exec closure is 'static-compatible.
        let mut rules: Vec<(String, bool)> = Vec::new();

        // Standard read-only paths needed by virtually every shell command.
        for ro in &[
            "/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/proc", "/dev",
        ] {
            if std::path::Path::new(ro).exists() {
                rules.push((ro.to_string(), false));
            }
        }

        // Workspace
        let ws_str = workspace.to_string_lossy().into_owned();
        match workspace_access {
            Access::Rw => rules.push((ws_str, true)),
            Access::Ro => rules.push((ws_str, false)),
            Access::None => {} // no access to workspace
        }

        // Configured allowed paths
        for (p, acc) in &allowed_paths {
            if p.exists() {
                let rw = *acc == Access::Rw;
                rules.push((p.to_string_lossy().into_owned(), rw));
            }
        }

        let mut command = Command::new("sh");
        command.arg("-c").arg(cmd).current_dir(cwd);

        // Safety: pre_exec runs in the child after fork, before exec.
        // All values captured here are cloned plain data (Vec<(String, bool)>),
        // so no async resources or Mutexes are shared.
        unsafe {
            command.pre_exec(move || {
                if let Err(e) = apply_landlock_rules(&rules) {
                    // Log to stderr (visible in daemon logs if the parent
                    // captures stderr) but do not abort — degrade gracefully.
                    eprintln!("[sandbox/landlock] restriction failed (degraded): {e}");
                }
                Ok(())
            });
        }

        let output = command
            .output()
            .with_context(|| format!("landlock sandbox: spawning shell command: {cmd}"))?;

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
fn apply_landlock_rules(rules: &[(String, bool)]) -> Result<()> {
    // Compose the full set of filesystem accesses we handle.
    let handled = AccessFs::from_all(LandlockABI::V1);

    let ruleset = landlock::Ruleset::default()
        .handle_access(handled)
        .context("landlock: creating ruleset")?;

    let mut ruleset_created = ruleset
        .create()
        .context("landlock: creating ruleset (create)")?;

    for (path_str, rw) in rules {
        let path = std::path::Path::new(path_str);
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

        // add_rule consumes self and returns Result<Self, _>.
        // Propagate errors — the caller (pre_exec) handles them gracefully.
        ruleset_created = ruleset_created
            .add_rule(PathBeneath::new(fd, access))
            .with_context(|| format!("landlock: add_rule for {path_str}"))?;
    }

    let status = ruleset_created
        .restrict_self()
        .context("landlock: restrict_self")?;

    match status.ruleset {
        RulesetStatus::FullyEnforced => {
            tracing::debug!("landlock: fully enforced");
        }
        RulesetStatus::PartiallyEnforced => {
            tracing::debug!("landlock: partially enforced (older kernel ABI)");
        }
        RulesetStatus::NotEnforced => {
            tracing::debug!("landlock: not enforced (kernel lacks support)");
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

    #[test]
    fn denied_paths_are_blocked_when_landlock_available() {
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

        // The secret dir is NOT in allowed_paths, so it should be denied.
        let out = sb
            .spawn(
                &format!("cat '{}'", secret.join("file.txt").display()),
                &work,
            )
            .unwrap();
        // Under Landlock the cat should fail (non-zero exit).
        assert!(
            !out.status.success(),
            "reading denied path should fail; stdout={:?} stderr={:?}",
            out.stdout,
            out.stderr
        );
    }
}
