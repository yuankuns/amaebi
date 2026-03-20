//! User configuration loaded from `~/.amaebi/config.json`.
//!
//! Supports per-directory TTL overrides for session eviction.
//!
//! # Example `~/.amaebi/config.json`
//!
//! ```json
//! {
//!   "ttl_minutes": {
//!     "default": 30,
//!     "/home/syk/projectX": 120,
//!     "/tmp/scratch": 15
//!   }
//! }
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use crate::auth::amaebi_home;

/// Default session TTL when no configuration is present.
const DEFAULT_TTL_MINUTES: u64 = 30;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    /// Per-directory TTL overrides.  The key `"default"` sets the global
    /// default; other keys are canonical directory paths.
    #[serde(default)]
    pub ttl_minutes: HashMap<String, u64>,
}

impl Config {
    /// Load configuration from `~/.amaebi/config.json`.
    ///
    /// Returns `Config::default()` if the file does not exist or is invalid.
    pub fn load() -> Self {
        Self::load_inner().unwrap_or_default()
    }

    fn load_inner() -> Result<Self> {
        let path = amaebi_home()?.join("config.json");
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(&path)?;
        if content.trim().is_empty() {
            return Ok(Self::default());
        }
        Ok(serde_json::from_str(&content)?)
    }

    /// Resolve the TTL duration for a given directory path.
    ///
    /// Checks for an exact canonical-path match first, then falls back to
    /// the `"default"` key, and finally to `DEFAULT_TTL_MINUTES`.
    pub fn ttl_for(&self, dir: &str) -> Duration {
        // Exact match on canonical directory path.
        if let Some(&minutes) = self.ttl_minutes.get(dir) {
            return Duration::from_secs(minutes * 60);
        }
        // Longest-prefix match: find the most specific configured ancestor.
        let mut best: Option<(&str, u64)> = None;
        for (key, &minutes) in &self.ttl_minutes {
            if key == "default" {
                continue;
            }
            if dir.starts_with(key.as_str()) {
                match best {
                    Some((prev_key, _)) if key.len() > prev_key.len() => {
                        best = Some((key.as_str(), minutes));
                    }
                    None => {
                        best = Some((key.as_str(), minutes));
                    }
                    _ => {}
                }
            }
        }
        if let Some((_, minutes)) = best {
            return Duration::from_secs(minutes * 60);
        }
        // Global default from config, or hardcoded fallback.
        let default_minutes = self
            .ttl_minutes
            .get("default")
            .copied()
            .unwrap_or(DEFAULT_TTL_MINUTES);
        Duration::from_secs(default_minutes * 60)
    }

    /// Return the default TTL (for sessions without a known directory).
    pub fn default_ttl(&self) -> Duration {
        let minutes = self
            .ttl_minutes
            .get("default")
            .copied()
            .unwrap_or(DEFAULT_TTL_MINUTES);
        Duration::from_secs(minutes * 60)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_30_minutes() {
        let cfg = Config::default();
        assert_eq!(cfg.default_ttl(), Duration::from_secs(30 * 60));
    }

    #[test]
    fn custom_default_ttl() {
        let mut cfg = Config::default();
        cfg.ttl_minutes.insert("default".into(), 60);
        assert_eq!(cfg.default_ttl(), Duration::from_secs(60 * 60));
    }

    #[test]
    fn exact_path_match() {
        let mut cfg = Config::default();
        cfg.ttl_minutes.insert("/home/syk/projectX".into(), 120);
        assert_eq!(
            cfg.ttl_for("/home/syk/projectX"),
            Duration::from_secs(120 * 60)
        );
    }

    #[test]
    fn prefix_match() {
        let mut cfg = Config::default();
        cfg.ttl_minutes.insert("/home/syk/projectX".into(), 120);
        assert_eq!(
            cfg.ttl_for("/home/syk/projectX/subdir"),
            Duration::from_secs(120 * 60)
        );
    }

    #[test]
    fn longest_prefix_wins() {
        let mut cfg = Config::default();
        cfg.ttl_minutes.insert("/home/syk".into(), 60);
        cfg.ttl_minutes.insert("/home/syk/projectX".into(), 120);
        assert_eq!(
            cfg.ttl_for("/home/syk/projectX/subdir"),
            Duration::from_secs(120 * 60)
        );
    }

    #[test]
    fn falls_back_to_default_key() {
        let mut cfg = Config::default();
        cfg.ttl_minutes.insert("default".into(), 45);
        assert_eq!(
            cfg.ttl_for("/some/random/path"),
            Duration::from_secs(45 * 60)
        );
    }

    #[test]
    fn falls_back_to_hardcoded_default() {
        let cfg = Config::default();
        assert_eq!(
            cfg.ttl_for("/some/random/path"),
            Duration::from_secs(30 * 60)
        );
    }

    #[test]
    fn roundtrip_serde() {
        let mut cfg = Config::default();
        cfg.ttl_minutes.insert("default".into(), 30);
        cfg.ttl_minutes.insert("/projectX".into(), 120);
        let json = serde_json::to_string_pretty(&cfg).unwrap();
        let cfg2: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg2.ttl_minutes.get("default"), Some(&30));
        assert_eq!(cfg2.ttl_minutes.get("/projectX"), Some(&120));
    }
}
