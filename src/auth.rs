use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
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

fn copilot_config_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("$HOME is not set")?;
    Ok(PathBuf::from(home).join(".config/github-copilot"))
}

/// Read the long-lived GitHub OAuth token from the Copilot config files.
///
/// Tries `hosts.json` first (VS Code / neovim plugin), then `apps.json`
/// (GitHub CLI plugin).
fn read_oauth_token() -> Result<String> {
    let dir = copilot_config_dir()?;
    let candidates = ["hosts.json", "apps.json"];

    for name in candidates {
        let path = dir.join(name);
        if !path.exists() {
            continue;
        }
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let map: HashMap<String, HostEntry> =
            serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
        if let Some(entry) = map.get("github.com") {
            return Ok(entry.oauth_token.clone());
        }
    }

    anyhow::bail!(
        "GitHub Copilot OAuth token not found. \
         Expected ~/.config/github-copilot/hosts.json or apps.json with a \
         \"github.com\" entry."
    )
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

async fn fetch_api_token(http: &reqwest::Client) -> Result<CachedToken> {
    let oauth_token = read_oauth_token()?;

    tracing::debug!("fetching Copilot API token");

    let resp = http
        .get("https://api.github.com/copilot_internal/v2/token")
        .header("Authorization", format!("token {oauth_token}"))
        .header("Accept", "application/json")
        .header(
            "User-Agent",
            concat!("amaebi/", env!("CARGO_PKG_VERSION")),
        )
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
