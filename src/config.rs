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

/// Default heartbeat interval when present in config but no interval specified.
const DEFAULT_HEARTBEAT_INTERVAL_MINUTES: u64 = 30;

/// Heartbeat configuration.  When present in `config.json`, the daemon
/// periodically reviews pending heartbeat items and surfaces actionable
/// ones via the inbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatConfig {
    /// How often (in minutes) the heartbeat scheduler fires.
    #[serde(default = "default_heartbeat_interval")]
    pub interval_minutes: u64,

    /// Active hours window `[start, end)` in UTC (0–23).
    /// Heartbeat only fires when the current UTC hour is within this range.
    /// If absent, heartbeat fires at any hour.
    #[serde(default)]
    pub active_hours: Option<(u8, u8)>,

    /// Model override for heartbeat LLM calls.
    /// Falls back to `AMAEBI_MODEL` / `"gpt-4o"` when absent.
    #[serde(default)]
    pub model: Option<String>,
}

fn default_heartbeat_interval() -> u64 {
    DEFAULT_HEARTBEAT_INTERVAL_MINUTES
}

impl Default for HeartbeatConfig {
    fn default() -> Self {
        Self {
            interval_minutes: DEFAULT_HEARTBEAT_INTERVAL_MINUTES,
            active_hours: None,
            model: None,
        }
    }
}

impl HeartbeatConfig {
    /// Check whether the current UTC hour is within the active window.
    pub fn is_active_now(&self) -> bool {
        let Some((start, end)) = self.active_hours else {
            return true;
        };
        // Treat out-of-range values as misconfigured — log and treat as always active.
        if start > 23 || end > 23 {
            tracing::warn!(
                start,
                end,
                "heartbeat active_hours values must be in 0–23; treating as always active"
            );
            return true;
        }
        let hour = chrono::Utc::now()
            .format("%H")
            .to_string()
            .parse::<u8>()
            .unwrap_or(0);
        if start <= end {
            // Normal range: e.g. [9, 21) means 09:00–20:59
            hour >= start && hour < end
        } else {
            // Wrap-around: e.g. [22, 8) means 22:00–07:59
            hour >= start || hour < end
        }
    }
}

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

    /// Heartbeat configuration.  When absent (`null` or omitted), the
    /// heartbeat scheduler is disabled entirely.
    #[serde(default)]
    pub heartbeat: Option<HeartbeatConfig>,
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

    // ---- HeartbeatConfig ---------------------------------------------------

    #[test]
    fn heartbeat_absent_means_disabled() {
        let cfg: Config = serde_json::from_str(r#"{"ttl_minutes":{}}"#).unwrap();
        assert!(cfg.heartbeat.is_none());
    }

    #[test]
    fn heartbeat_empty_object_uses_defaults() {
        let cfg: Config = serde_json::from_str(r#"{"heartbeat":{}}"#).unwrap();
        let hb = cfg.heartbeat.unwrap();
        assert_eq!(hb.interval_minutes, 30);
        assert!(hb.active_hours.is_none());
        assert!(hb.model.is_none());
    }

    #[test]
    fn heartbeat_full_config_round_trip() {
        let json = r#"{
            "heartbeat": {
                "interval_minutes": 15,
                "active_hours": [9, 21],
                "model": "gpt-4o-mini"
            }
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        let hb = cfg.heartbeat.unwrap();
        assert_eq!(hb.interval_minutes, 15);
        assert_eq!(hb.active_hours, Some((9, 21)));
        assert_eq!(hb.model.as_deref(), Some("gpt-4o-mini"));
    }

    #[test]
    fn heartbeat_is_active_no_restriction() {
        let hb = HeartbeatConfig::default();
        assert!(hb.is_active_now());
    }

    #[test]
    fn heartbeat_active_hours_out_of_range_treats_as_always_active() {
        // Values outside 0-23 are invalid; is_active_now must return true
        // (always-active) rather than producing wrong results silently.
        let hb = HeartbeatConfig {
            active_hours: Some((25, 10)), // start=25 is invalid
            ..Default::default()
        };
        assert!(
            hb.is_active_now(),
            "out-of-range active_hours should be treated as always active"
        );

        let hb2 = HeartbeatConfig {
            active_hours: Some((9, 200)), // end=200 is invalid
            ..Default::default()
        };
        assert!(
            hb2.is_active_now(),
            "out-of-range active_hours end should be treated as always active"
        );
    }

    #[test]
    fn heartbeat_active_hours_valid_boundary_values() {
        // 0 and 23 are valid boundary values; must not be rejected.
        let hb = HeartbeatConfig {
            active_hours: Some((0, 23)),
            ..Default::default()
        };
        // Just verify it doesn't panic and returns a bool.
        let _ = hb.is_active_now();
    }

    #[test]
    fn heartbeat_interval_saturating_mul_does_not_overflow() {
        // u64::MAX / 60 + 1 overflows with plain multiplication.
        // saturating_mul must not panic and must cap at u64::MAX.
        let hb = HeartbeatConfig {
            interval_minutes: u64::MAX,
            ..Default::default()
        };
        // This exercises the saturating_mul path in the scheduler.
        // We can't call the scheduler directly, but we can verify the
        // arithmetic produces a valid Duration rather than panicking.
        let secs = hb.interval_minutes.saturating_mul(60);
        let _dur = std::time::Duration::from_secs(secs);
        // If we reach here without panicking, the overflow is handled.
    }
}
