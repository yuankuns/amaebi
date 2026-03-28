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
    match config.backend.as_str() {
        "landlock" => Box::new(landlock_seccomp::LandlockSandbox::new(config.clone())),
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
    fn create_backend_landlock_returns_landlock() {
        let cfg = SandboxConfig {
            enabled: true,
            backend: "landlock".into(),
            ..SandboxConfig::default()
        };
        let backend = create_backend(&cfg);
        assert_eq!(backend.name(), "landlock");
    }
}
