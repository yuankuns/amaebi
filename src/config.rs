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
    /// TTL overrides keyed by one of three kinds of identifier (minutes):
    ///
    /// * `"default"` — global fallback when no other key matches.
    /// * tier name (e.g. `"ephemeral"`, `"persistent"`) — overrides the
    ///   built-in TTL for all sessions whose `ttl_tier` matches that name.
    /// * canonical directory path (e.g. `"/home/syk/projectX"`) — overrides
    ///   the TTL for the session associated with that exact directory.
    ///   Longest-prefix matching applies: the most specific ancestor path wins.
    ///
    /// Resolution order (highest priority first):
    /// 1. exact directory-path match
    /// 2. longest ancestor directory-path prefix
    /// 3. tier-name match
    /// 4. `"default"` key
    /// 5. built-in 30-minute fallback
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
    /// Resolution order (highest priority first):
    /// 1. Exact match on `dir` in `ttl_minutes`.
    /// 2. Longest ancestor-prefix match: the most specific configured path
    ///    that is a true parent of `dir` (i.e. `dir` starts with the key
    ///    followed by `/`, or equals the key exactly).
    /// 3. The `"default"` key in `ttl_minutes`.
    /// 4. `DEFAULT_TTL_MINUTES` (30 minutes).
    #[allow(dead_code)]
    pub fn ttl_for(&self, dir: &str) -> Duration {
        // Exact match on canonical directory path.
        if let Some(&minutes) = self.ttl_minutes.get(dir) {
            return Duration::from_secs(minutes * 60);
        }
        // Longest-prefix match: find the most specific configured ancestor.
        let mut best: Option<(&str, u64)> = None;
        for (key, &minutes) in &self.ttl_minutes {
            // Skip the global default key and any tier-name keys (e.g.
            // "ephemeral", "persistent").  Only canonical path keys (those
            // beginning with '/') are eligible as ancestor prefixes.
            if !key.starts_with('/') {
                continue;
            }
            // Require a path-separator boundary so that e.g. key
            // "/home/user/projects" does not match "/home/user/projects-old".
            let is_ancestor = dir == key.as_str()
                || dir.starts_with(key.as_str()) && dir.as_bytes().get(key.len()) == Some(&b'/');
            if is_ancestor {
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
    fn sibling_path_does_not_match() {
        // "/home/syk/projects-old" must NOT inherit the TTL for "/home/syk/projects".
        let mut cfg = Config::default();
        cfg.ttl_minutes.insert("/home/syk/projects".into(), 120);
        assert_eq!(
            cfg.ttl_for("/home/syk/projects-old"),
            Duration::from_secs(30 * 60), // falls back to hardcoded default
        );
    }

    #[test]
    fn tier_name_key_not_used_as_path_prefix() {
        // "ephemeral" and "persistent" are tier names, not directory paths.
        // They must never act as ancestor prefixes for real paths.
        let mut cfg = Config::default();
        cfg.ttl_minutes.insert("ephemeral".into(), 5);
        cfg.ttl_minutes.insert("persistent".into(), 1440);
        // A real path should fall through to the hardcoded default.
        assert_eq!(
            cfg.ttl_for("/home/syk/anything"),
            Duration::from_secs(30 * 60),
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
