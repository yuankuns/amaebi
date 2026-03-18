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

/// A cached Copilot API token with its expiry time.
struct CachedToken {
    value: String,
    valid_until: Instant,
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

    /// Return a valid API token, fetching a fresh one if the cache is empty
    /// or the cached token is about to expire.
    pub async fn get(&self, http: &reqwest::Client) -> Result<String> {
        let mut guard = self.inner.lock().await;

        // Return cached token if it has more than 60 s left.
        if let Some(ref cached) = *guard {
            if cached.valid_until > Instant::now() {
                return Ok(cached.value.clone());
            }
            tracing::debug!("Copilot API token expired; refreshing");
        }

        let cached = fetch_api_token(http).await?;
        let value = cached.value.clone();
        *guard = Some(cached);
        Ok(value)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use tempfile::TempDir;

    // HOME_LOCK serialises all tests that mutate $HOME so they don't race.
    static HOME_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Run `f` with `$HOME` temporarily pointing at `dir`.
    fn with_home(dir: &std::path::Path, f: impl FnOnce()) {
        let _guard = HOME_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let old = std::env::var("HOME").ok();
        // SAFETY: HOME_LOCK ensures only one thread modifies HOME at a time.
        unsafe { std::env::set_var("HOME", dir) };
        f();
        unsafe {
            match old {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
        }
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

    Ok(CachedToken {
        value: body.token,
        valid_until: Instant::now() + ttl,
    })
}
