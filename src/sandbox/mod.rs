#[cfg(target_os = "linux")]
pub mod landlock_seccomp;
pub mod noop;

use anyhow::Result;
use std::path::PathBuf;
use std::process::ExitStatus;

// ---------------------------------------------------------------------------
// Access level
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum Access {
    None,
    Ro,
    Rw,
}
// ---------------------------------------------------------------------------
// SandboxConfig
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    pub enabled: bool,
    /// Backend to use: `"landlock"` | `"noop"`.
    pub backend: String,
    pub workspace: PathBuf,
    pub workspace_access: Access,
    pub network: bool,
    pub allowed_paths: Vec<(PathBuf, Access)>,
    /// Note: Landlock uses an allowlist-only approach.  Paths listed here are
    /// **not enforced** by the Landlock backend — access is denied implicitly
    /// for any path not in `allowed_paths`.  This field is reserved for future
    /// backends (e.g. seccomp) that support explicit denials.
    pub denied_paths: Vec<PathBuf>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            backend: "noop".into(),
            workspace: PathBuf::from("."),
            workspace_access: Access::Rw,
            network: true,
            allowed_paths: vec![(PathBuf::from("/tmp"), Access::Rw)],
            denied_paths: vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// SandboxOutput
// ---------------------------------------------------------------------------

pub struct SandboxOutput {
    pub status: ExitStatus,
    pub stdout: String,
    pub stderr: String,
}

// ---------------------------------------------------------------------------
// Sandbox trait
// ---------------------------------------------------------------------------

pub trait Sandbox: Send + Sync {
    #[allow(dead_code)]
    fn name(&self) -> &str;
    #[allow(dead_code)]
    /// Return `true` when this backend is usable on the current kernel/OS.
    fn available(&self) -> bool;
    /// Run `sh -c <cmd>` in `cwd`, applying the configured isolation policy.
    fn spawn(&self, cmd: &str, cwd: &std::path::Path) -> Result<SandboxOutput>;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Create the appropriate sandbox backend from `config`.
///
/// If `config.enabled` is `false` the noop backend is always returned,
/// providing identical behaviour to running commands without any sandbox.
pub fn create_backend(config: &SandboxConfig) -> Box<dyn Sandbox> {
    if !config.enabled {
        return Box::new(noop::NoopSandbox);
    }
    // Warn if denied_paths is set — the Landlock backend uses allowlist-only
    // enforcement and will not actively block these paths.
    if !config.denied_paths.is_empty() {
        tracing::warn!(
            count = config.denied_paths.len(),
            "sandbox: denied_paths is set but the current backend uses allowlist-only \
             enforcement; denied_paths will NOT be actively blocked"
        );
    }
    match config.backend.as_str() {
        #[cfg(target_os = "linux")]
        "landlock" => Box::new(landlock_seccomp::LandlockSandbox::new(config.clone())),
        #[cfg(not(target_os = "linux"))]
        "landlock" => {
            tracing::warn!(
                "sandbox: backend=\"landlock\" requested but Landlock is only available on Linux; \
                 falling back to noop"
            );
            Box::new(noop::NoopSandbox)
        }
        _ => Box::new(noop::NoopSandbox),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_disabled_noop() {
        let cfg = SandboxConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.backend, "noop");
    }

    #[test]
    fn create_backend_disabled_returns_noop() {
        let cfg = SandboxConfig::default();
        let backend = create_backend(&cfg);
        assert_eq!(backend.name(), "noop");
    }

    #[test]
    fn create_backend_unknown_backend_returns_noop() {
        let cfg = SandboxConfig {
            enabled: true,
            backend: "unicorn".into(),
            ..SandboxConfig::default()
        };
        let backend = create_backend(&cfg);
        assert_eq!(backend.name(), "noop");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn create_backend_landlock_returns_landlock() {
        let cfg = SandboxConfig {
            enabled: true,
            backend: "landlock".into(),
            ..SandboxConfig::default()
        };
        let backend = create_backend(&cfg);
        assert_eq!(backend.name(), "landlock");
    }

    #[test]
    #[cfg(not(target_os = "linux"))]
    fn create_backend_landlock_falls_back_to_noop_on_non_linux() {
        let cfg = SandboxConfig {
            enabled: true,
            backend: "landlock".into(),
            ..SandboxConfig::default()
        };
        let backend = create_backend(&cfg);
        assert_eq!(backend.name(), "noop");
    }

    // ------------------------------------------------------------------
    // New regression tests for PR #21
    // ------------------------------------------------------------------

    #[test]
    #[cfg(target_os = "linux")]
    fn enabled_config_creates_correct_backend() {
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        let cfg = SandboxConfig {
            enabled: true,
            backend: "landlock".into(),
            workspace: tmp.path().to_path_buf(),
            workspace_access: Access::Rw,
            allowed_paths: vec![(PathBuf::from("/tmp"), Access::Rw)],
            denied_paths: vec![],
            network: false,
        };
        let backend = create_backend(&cfg);
        assert_eq!(backend.name(), "landlock");
        // Verify spawn works end-to-end through the config flow
        let out = backend.spawn("echo cfg-ok", tmp.path()).unwrap();
        assert!(out.status.success());
        assert_eq!(out.stdout.trim(), "cfg-ok");
    }

    #[test]
    fn noop_backend_allows_everything() {
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        let secret = tmp.path().join("secret.txt");
        std::fs::write(&secret, "secret").unwrap();

        // noop backend — no sandbox restrictions at all
        let cfg = SandboxConfig {
            enabled: true,
            backend: "noop".into(),
            workspace: tmp.path().to_path_buf(),
            ..SandboxConfig::default()
        };
        let backend = create_backend(&cfg);
        assert_eq!(backend.name(), "noop");

        let out = backend
            .spawn(&format!("cat '{}'", secret.display()), tmp.path())
            .unwrap();
        assert!(
            out.status.success(),
            "noop should allow reads; stderr={:?}",
            out.stderr
        );
        assert_eq!(out.stdout.trim(), "secret");

        let write_file = tmp.path().join("written.txt");
        let out = backend
            .spawn(
                &format!("echo noop-write > '{}'", write_file.display()),
                tmp.path(),
            )
            .unwrap();
        assert!(
            out.status.success(),
            "noop should allow writes; stderr={:?}",
            out.stderr
        );
        assert_eq!(
            std::fs::read_to_string(&write_file).unwrap().trim(),
            "noop-write"
        );
    }

    // ------------------------------------------------------------------
    // End new regression tests
    // ------------------------------------------------------------------
}
