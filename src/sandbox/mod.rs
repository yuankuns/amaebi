#![allow(dead_code)]

use anyhow::Result;
use async_trait::async_trait;
use std::path::{Path, PathBuf};

#[cfg(target_os = "linux")]
pub mod namespace;

#[derive(Debug, Clone)]
pub struct SandboxConfig {
    pub enabled: bool,
    pub backend: String,
    pub workspace: PathBuf,
    pub ro_paths: Vec<PathBuf>,
    pub rw_paths: Vec<PathBuf>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            backend: "noop".to_string(),
            workspace: PathBuf::from("."),
            ro_paths: Vec::new(),
            rw_paths: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct SandboxOutput {
    pub stdout: String,
    pub stderr: String,
    pub status: i32,
}

#[async_trait]
pub trait Sandbox: Send + Sync {
    fn name(&self) -> &str;
    fn available(&self) -> bool;
    async fn spawn(&self, cmd: &str, cwd: &Path) -> Result<SandboxOutput>;
}

pub fn create_backend(config: SandboxConfig) -> Box<dyn Sandbox> {
    match config.backend.as_str() {
        #[cfg(target_os = "linux")]
        "namespace" => Box::new(namespace::NamespaceSandbox::new(config)),
        _ => Box::new(NoopSandbox),
    }
}

pub struct NoopSandbox;

#[async_trait]
impl Sandbox for NoopSandbox {
    fn name(&self) -> &str {
        "noop"
    }

    fn available(&self) -> bool {
        true
    }

    async fn spawn(&self, cmd: &str, cwd: &Path) -> Result<SandboxOutput> {
        let output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .current_dir(cwd)
            .output()
            .await?;
        Ok(SandboxOutput {
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            status: output.status.code().unwrap_or(-1),
        })
    }
}
