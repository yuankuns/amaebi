use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

// ---------------------------------------------------------------------------
// Copilot OAuth token — read from disk
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct HostEntry {
    oauth_token: String,
}

/// Returns `~/.amaebi/` — the primary config directory for amaebi.
pub(crate) fn amaebi_home() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("$HOME is not set")?;
    Ok(PathBuf::from(home).join(".amaebi"))
}

/// Read the long-lived GitHub OAuth token from the Copilot config files.
///
/// Checks `~/.amaebi/hosts.json` first, then falls back to
/// `~/.config/github-copilot/hosts.json` and `apps.json` for VS Code / neovim
/// compatibility.
pub(crate) fn read_oauth_token() -> Result<String> {
    let home = std::env::var("HOME").context("$HOME is not set")?;
    let home = PathBuf::from(home);

    let copilot_dir = home.join(".config/github-copilot");
    let candidates = [
        amaebi_home()?.join("hosts.json"),
        copilot_dir.join("hosts.json"),
        copilot_dir.join("apps.json"),
    ];

    for path in &candidates {
        if !path.exists() {
            continue;
        }
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let map: HashMap<String, HostEntry> =
            serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
        if let Some(entry) = map.get("github.com") {
            return Ok(entry.oauth_token.clone());
        }
    }

    anyhow::bail!(
        "GitHub Copilot OAuth token not found. \
         Expected ~/.amaebi/hosts.json or ~/.config/github-copilot/hosts.json / apps.json \
         with a \"github.com\" entry."
    )
}

/// Write (or merge) an OAuth token into `~/.amaebi/hosts.json`.
///
/// If the file already exists its other top-level keys are preserved; only the
/// `"github.com"` entry is created or replaced.
pub(crate) fn save_hosts_json(oauth_token: &str, username: &str) -> Result<()> {
    let dir = amaebi_home()?;
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating config dir {}", dir.display()))?;
    // Restrict the config directory so other users cannot list its contents.
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))
        .with_context(|| format!("setting permissions on {}", dir.display()))?;

    let path = dir.join("hosts.json");

    // Start from the existing file content (if any) so we don't clobber other
    // providers that may have written their own keys.
    let mut root: serde_json::Value = if path.exists() {
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&raw)
            .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()))
    } else {
        serde_json::Value::Object(serde_json::Map::new())
    };

    let obj = root
        .as_object_mut()
        .context("hosts.json root is not a JSON object")?;

    obj.insert(
        "github.com".to_string(),
        serde_json::json!({
            "user":        username,
            "oauth_token": oauth_token,
            "git_protocol": "https"
        }),
    );

    let json = serde_json::to_string_pretty(&root).context("serialising hosts.json")?;
    std::fs::write(&path, json).with_context(|| format!("writing {}", path.display()))?;
    // Restrict the token file so only the owning user can read it.
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("setting permissions on {}", path.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Short-lived API token — fetched from GitHub and cached
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ApiTokenResponse {
    token: String,
    /// Seconds until the token expires.
    #[serde(rename = "refresh_in")]
    refresh_in: u64,
}

/// Default Copilot chat-completions base URL, used when `proxy-ep` is absent.
pub const DEFAULT_COPILOT_BASE_URL: &str = "https://api.githubcopilot.com";

/// Extract the chat-completions base URL from the Copilot API token.
///
/// The token returned by the Copilot token endpoint is a semicolon-delimited
/// key=value string.  One field is `proxy-ep=<url>`.  GitHub's convention is
/// that the actual API host is obtained by replacing the `proxy.` sub-domain
/// with `api.`.  For example:
///   `proxy-ep=https://proxy.individual.githubcopilot.com`
///   → `https://api.individual.githubcopilot.com`
///
/// Returns the derived URL, or [`DEFAULT_COPILOT_BASE_URL`] when the field is
/// absent or cannot be parsed.
pub fn base_url_from_token(token: &str) -> String {
    for field in token.split(';') {
        let field = field.trim();
        let Some(val) = field.strip_prefix("proxy-ep=") else {
            continue;
        };
        let val = val.trim();
        if val.is_empty() {
            continue;
        }
        // Strip the scheme, swap proxy. → api., re-attach https://.
        // Normalise to lowercase for the prefix check so that unexpected
        // capitalisation (e.g. "Proxy.") is handled correctly.
        let host = val
            .trim_start_matches("https://")
            .trim_start_matches("http://");
        let host_lower = host.to_lowercase();
        let api_host = if let Some(rest) = host_lower.strip_prefix("proxy.") {
            format!("api.{rest}")
        } else {
            host_lower
        };
        let derived = format!("https://{api_host}");
        tracing::debug!(proxy_ep = %val, base_url = %derived, "derived Copilot base URL from proxy-ep");
        return derived;
    }
    DEFAULT_COPILOT_BASE_URL.to_string()
}

/// A cached Copilot API token with its expiry time and the derived base URL.
struct CachedToken {
    value: String,
    /// Chat-completions base URL derived from `proxy-ep` in the token.
    base_url: String,
    valid_until: Instant,
}

/// The token value and its associated base URL, returned together so callers
/// always use the correct endpoint for the token they hold.
pub struct CopilotToken {
    pub value: String,
    pub base_url: String,
}

/// Thread-safe token cache stored in daemon shared state.
pub struct TokenCache {
    inner: Mutex<Option<CachedToken>>,
}

impl TokenCache {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }

    /// Evict the cached token, forcing the next [`get`] call to fetch a fresh one.
    ///
    /// Call this when the API returns a 4xx response that suggests the token
    /// was invalidated server-side before its nominal `valid_until` time.
    pub async fn invalidate(&self) {
        let mut guard = self.inner.lock().await;
        *guard = None;
    }

    /// Return a valid API token and its base URL, fetching a fresh one if the
    /// cache is empty or the cached token is about to expire.
    pub async fn get(&self, http: &reqwest::Client) -> Result<CopilotToken> {
        // Allow callers to inject a pre-baked token via env var, bypassing
        // the full OAuth flow.  Still parse proxy-ep from the token so that
        // a real Copilot JWT passed this way gets the correct per-account
        // base URL rather than always falling back to the default.
        if let Ok(tok) = std::env::var("AMAEBI_COPILOT_TOKEN") {
            if !tok.trim().is_empty() {
                let value = tok.trim().to_string();
                let base_url = base_url_from_token(&value);
                return Ok(CopilotToken { value, base_url });
            }
        }

        let mut guard = self.inner.lock().await;

        // Return cached token if it has more than 60 s left.
        if let Some(ref cached) = *guard {
            if cached.valid_until > Instant::now() {
                return Ok(CopilotToken {
                    value: cached.value.clone(),
                    base_url: cached.base_url.clone(),
                });
            }
            tracing::debug!("Copilot API token expired; refreshing");
        }

        let cached = fetch_api_token(http).await?;
        let tok = CopilotToken {
            value: cached.value.clone(),
            base_url: cached.base_url.clone(),
        };
        *guard = Some(cached);
        Ok(tok)
    }
}

async fn fetch_api_token(http: &reqwest::Client) -> Result<CachedToken> {
    let oauth_token = read_oauth_token()?;

    tracing::debug!("fetching Copilot API token");

    let resp = http
        .get("https://api.github.com/copilot_internal/v2/token")
        .header("Authorization", format!("token {oauth_token}"))
        .header("Accept", "application/json")
        .header("User-Agent", concat!("amaebi/", env!("CARGO_PKG_VERSION")))
        .send()
        .await
        .context("fetching Copilot API token")?
        .error_for_status()
        .context("Copilot token endpoint returned an error")?;

    let body: ApiTokenResponse = resp
        .json()
        .await
        .context("parsing Copilot API token response")?;

    // Subtract 60 s as a safety margin so we never use an about-to-expire token.
    let ttl = Duration::from_secs(body.refresh_in.saturating_sub(60));
    tracing::debug!(ttl_secs = ttl.as_secs(), "Copilot API token refreshed");

    let base_url = base_url_from_token(&body.token);
    Ok(CachedToken {
        base_url,
        value: body.token,
        valid_until: Instant::now() + ttl,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::with_home;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::time::Duration;
    use tempfile::TempDir;

    // ---- base_url_from_token -----------------------------------------------

    #[test]
    fn base_url_from_token_extracts_proxy_ep() {
        // Standard individual Copilot account.
        let tok = "tid=abc;proxy-ep=https://proxy.individual.githubcopilot.com;sku=pro";
        assert_eq!(
            base_url_from_token(tok),
            "https://api.individual.githubcopilot.com"
        );
    }

    #[test]
    fn base_url_from_token_enterprise_host() {
        let tok = "tid=xyz;proxy-ep=https://proxy.business.githubcopilot.com";
        assert_eq!(
            base_url_from_token(tok),
            "https://api.business.githubcopilot.com"
        );
    }

    #[test]
    fn base_url_from_token_no_proxy_ep_uses_default() {
        assert_eq!(
            base_url_from_token("tid=abc;sku=pro"),
            DEFAULT_COPILOT_BASE_URL
        );
    }

    #[test]
    fn base_url_from_token_empty_token_uses_default() {
        assert_eq!(base_url_from_token(""), DEFAULT_COPILOT_BASE_URL);
    }

    #[test]
    fn base_url_from_token_case_insensitive_prefix() {
        // "Proxy." (capitalised) must still be swapped to "api.".
        let tok = "proxy-ep=https://Proxy.individual.githubcopilot.com";
        assert_eq!(
            base_url_from_token(tok),
            "https://api.individual.githubcopilot.com"
        );
    }

    // ---- TokenCache::invalidate -------------------------------------------

    #[tokio::test]
    async fn token_cache_invalidate_clears_cached_value() {
        let cache = TokenCache::new();
        // Manually populate the cache with a far-future token.
        {
            let mut guard = cache.inner.lock().await;
            *guard = Some(CachedToken {
                value: "test-token".into(),
                base_url: DEFAULT_COPILOT_BASE_URL.to_string(),
                valid_until: Instant::now() + Duration::from_secs(3600),
            });
        }
        // Invalidate should clear it.
        cache.invalidate().await;
        let guard = cache.inner.lock().await;
        assert!(guard.is_none(), "cache should be empty after invalidate");
    }

    /// Path to `~/.amaebi/` inside a temp home.
    fn amaebi_dir(home: &std::path::Path) -> std::path::PathBuf {
        home.join(".amaebi")
    }

    /// Path to the legacy `~/.config/github-copilot/` inside a temp home.
    fn copilot_dir(home: &std::path::Path) -> std::path::PathBuf {
        home.join(".config/github-copilot")
    }

    // ---- amaebi_home ------------------------------------------------------

    #[test]
    fn amaebi_home_is_home_dot_amaebi() {
        let tmp = TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let dir = amaebi_home().unwrap();
            assert_eq!(dir, tmp.path().join(".amaebi"));
        });
    }

    // ---- read_oauth_token -------------------------------------------------

    #[test]
    fn read_token_from_amaebi_hosts_json() {
        // Primary candidate: ~/.amaebi/hosts.json
        let tmp = TempDir::new().unwrap();
        let dir = amaebi_dir(tmp.path());
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("hosts.json"),
            r#"{"github.com": {"oauth_token": "tok123", "user": "alice"}}"#,
        )
        .unwrap();

        with_home(tmp.path(), || {
            assert_eq!(read_oauth_token().unwrap(), "tok123");
        });
    }

    #[test]
    fn read_token_falls_back_to_copilot_hosts_json() {
        // Second candidate: ~/.config/github-copilot/hosts.json
        let tmp = TempDir::new().unwrap();
        let dir = copilot_dir(tmp.path());
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("hosts.json"),
            r#"{"github.com": {"oauth_token": "copilottok", "user": "bob"}}"#,
        )
        .unwrap();

        with_home(tmp.path(), || {
            assert_eq!(read_oauth_token().unwrap(), "copilottok");
        });
    }

    #[test]
    fn read_token_falls_back_to_apps_json() {
        // Third candidate: ~/.config/github-copilot/apps.json
        let tmp = TempDir::new().unwrap();
        let dir = copilot_dir(tmp.path());
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("apps.json"),
            r#"{"github.com": {"oauth_token": "appstoken"}}"#,
        )
        .unwrap();

        with_home(tmp.path(), || {
            assert_eq!(read_oauth_token().unwrap(), "appstoken");
        });
    }

    #[test]
    fn read_token_hosts_json_missing_github_com_falls_through() {
        let tmp = TempDir::new().unwrap();
        let dir = copilot_dir(tmp.path());
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("hosts.json"),
            r#"{"gitlab.com": {"oauth_token": "gl"}}"#,
        )
        .unwrap();

        with_home(tmp.path(), || {
            assert!(read_oauth_token().is_err());
        });
    }

    #[test]
    fn read_token_no_files_returns_err() {
        let tmp = TempDir::new().unwrap();
        with_home(tmp.path(), || {
            let err = read_oauth_token().unwrap_err();
            assert!(format!("{err}").contains("not found"));
        });
    }

    // ---- save_hosts_json --------------------------------------------------

    #[test]
    fn save_creates_new_hosts_json() {
        let tmp = TempDir::new().unwrap();
        with_home(tmp.path(), || {
            save_hosts_json("newtoken", "bob").unwrap();

            // save writes to ~/.amaebi/hosts.json
            let path = amaebi_dir(tmp.path()).join("hosts.json");
            let raw = fs::read_to_string(path).unwrap();
            let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
            assert_eq!(v["github.com"]["oauth_token"], "newtoken");
            assert_eq!(v["github.com"]["user"], "bob");
            assert_eq!(v["github.com"]["git_protocol"], "https");
        });
    }

    #[test]
    fn save_merges_other_top_level_keys() {
        let tmp = TempDir::new().unwrap();
        // Pre-populate ~/.amaebi/hosts.json with an existing key.
        let dir = amaebi_dir(tmp.path());
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("hosts.json"),
            r#"{"gitlab.com": {"oauth_token": "gl"}, "github.com": {"oauth_token": "old"}}"#,
        )
        .unwrap();

        with_home(tmp.path(), || {
            save_hosts_json("updated", "carol").unwrap();

            let raw = fs::read_to_string(dir.join("hosts.json")).unwrap();
            let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
            // Pre-existing key must survive.
            assert_eq!(v["gitlab.com"]["oauth_token"], "gl");
            // github.com entry is replaced.
            assert_eq!(v["github.com"]["oauth_token"], "updated");
            assert_eq!(v["github.com"]["user"], "carol");
        });
    }

    #[test]
    fn save_overwrites_existing_github_com_token() {
        let tmp = TempDir::new().unwrap();
        with_home(tmp.path(), || {
            save_hosts_json("first", "u1").unwrap();
            save_hosts_json("second", "u2").unwrap();

            let path = amaebi_dir(tmp.path()).join("hosts.json");
            let raw = fs::read_to_string(path).unwrap();
            let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
            assert_eq!(v["github.com"]["oauth_token"], "second");
            assert_eq!(v["github.com"]["user"], "u2");
        });
    }

    #[test]
    fn save_sets_dir_permissions_to_0700() {
        let tmp = TempDir::new().unwrap();
        with_home(tmp.path(), || {
            save_hosts_json("tok", "user").unwrap();
            let mode = amaebi_dir(tmp.path())
                .metadata()
                .unwrap()
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o700, "~/.amaebi/ must be mode 0700");
        });
    }

    #[test]
    fn save_sets_file_permissions_to_0600() {
        let tmp = TempDir::new().unwrap();
        with_home(tmp.path(), || {
            save_hosts_json("tok", "user").unwrap();
            let mode = amaebi_dir(tmp.path())
                .join("hosts.json")
                .metadata()
                .unwrap()
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600, "hosts.json must be mode 0600");
        });
    }
}
