//! GitHub Copilot authentication via the OAuth 2.0 Device Authorization Flow.
//!
//! # API assumptions
//!
//! * Device-code endpoint: POST `https://github.com/login/device/code`
//!   form fields: `client_id`, `scope` (empty string works for Copilot).
//!   GitHub always responds HTTP 200 even on errors; check the body.
//!
//! * Token-poll endpoint: POST `https://github.com/login/oauth/access_token`
//!   form fields: `client_id`, `device_code`,
//!   `grant_type=urn:ietf:params:oauth:grant-type:device_code`.
//!   Returns `access_token` on success, or `error` + `error_description`.
//!
//! * Copilot validation: GET `https://api.github.com/copilot_internal/v2/token`
//!   with `Authorization: token <oauth_token>`. 200 = active subscription.
//!
//! * Client ID `Iv1.b507a08c87ecfe98` is the public OAuth App ID used by the
//!   neovim copilot.vim plugin and other open-source clients. It is a *public*
//!   client (no secret required) and is overridable via `--client-id`.

use std::io::Write as IoWrite;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::Deserialize;
use tokio::process::Command as Proc;

use crate::auth::{read_oauth_token, save_hosts_json};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default GitHub OAuth App client ID (neovim copilot.vim, public client).
pub const DEFAULT_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";

const DEVICE_CODE_URL: &str = "https://github.com/login/device/code";
const TOKEN_URL: &str = "https://github.com/login/oauth/access_token";
const USER_API_URL: &str = "https://api.github.com/user";
const COPILOT_TOKEN_URL: &str = "https://api.github.com/copilot_internal/v2/token";

/// grant_type value, percent-encoded for application/x-www-form-urlencoded.
const DEVICE_GRANT_TYPE: &str = "urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code";

/// Never poll faster than this, regardless of server's `slow_down` responses.
const MAX_POLL_INTERVAL_SECS: u64 = 30;

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct DeviceCodeResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    expires_in: u64,
    interval: u64,
}

#[derive(Deserialize)]
struct TokenPollResponse {
    #[serde(default)]
    access_token: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    error_description: Option<String>,
}

#[derive(Deserialize)]
struct GitHubUser {
    login: String,
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Check for an existing token; run the device flow if absent or invalid.
///
/// This is the top-level handler for `amaebi auth`.
pub async fn ensure_authenticated(
    http: &reqwest::Client,
    client_id: &str,
    skip_validate: bool,
) -> Result<()> {
    match read_oauth_token() {
        Ok(token) => {
            println!("Found existing GitHub Copilot token.");
            if skip_validate {
                println!("Skipping validation (--skip-validate).");
                return Ok(());
            }
            print!("Validating Copilot access... ");
            flush_stdout();
            match validate_copilot_access(http, &token).await {
                Ok(()) => {
                    println!("✓ confirmed.");
                    println!("Run: amaebi daemon");
                    Ok(())
                }
                Err(e) => {
                    println!("✗ failed: {e:#}");
                    println!("Starting fresh login...\n");
                    run_and_save(http, client_id, skip_validate).await
                }
            }
        }
        Err(_) => {
            println!("No GitHub Copilot token found. Starting device login...\n");
            run_and_save(http, client_id, skip_validate).await
        }
    }
}

/// Validate an existing OAuth token against the Copilot API.
///
/// Returns `Ok(())` if the account has an active Copilot subscription.
pub async fn validate_copilot_access(http: &reqwest::Client, oauth_token: &str) -> Result<()> {
    let resp = http
        .get(COPILOT_TOKEN_URL)
        .header("Authorization", format!("token {oauth_token}"))
        .header("Accept", "application/json")
        .header("User-Agent", concat!("amaebi/", env!("CARGO_PKG_VERSION")))
        .send()
        .await
        .context("calling Copilot token endpoint")?
        .error_for_status()
        .context(
            "Copilot access validation failed — \
             check that your GitHub account has an active Copilot subscription",
        )?;

    // Drain body politely.
    let _ = resp.bytes().await;
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal flow
// ---------------------------------------------------------------------------

/// Run the device flow end-to-end and persist the result.
async fn run_and_save(http: &reqwest::Client, client_id: &str, skip_validate: bool) -> Result<()> {
    let (oauth_token, username) = run_device_flow(http, client_id).await?;

    save_hosts_json(&oauth_token, &username).context("saving token to hosts.json")?;
    println!("\n✓ Token saved to ~/.amaebi/hosts.json");

    if !skip_validate {
        print!("Validating Copilot access... ");
        flush_stdout();
        match validate_copilot_access(http, &oauth_token).await {
            Ok(()) => println!("✓ confirmed.\nRun: amaebi daemon"),
            Err(e) => println!(
                "⚠  {e:#}\n\
                 Token saved, but Copilot access could not be confirmed.\n\
                 Your account may not have an active Copilot subscription."
            ),
        }
    }

    Ok(())
}

/// Request a device code, display it to the user, and poll until authorised.
///
/// Returns `(oauth_token, github_username)`.
async fn run_device_flow(http: &reqwest::Client, client_id: &str) -> Result<(String, String)> {
    // Step 1 — request a device + user code pair.
    let dc = request_device_code(http, client_id).await?;

    // Step 2 — show the code to the user (terminal + tmux + browser).
    display_code(&dc.user_code, &dc.verification_uri).await;

    // Step 3 — poll until the user authorises or the code expires.
    let oauth_token =
        poll_for_token(http, client_id, &dc.device_code, dc.interval, dc.expires_in).await?;

    // Clear the polling status line.
    println!("\r✓ Authorised!                              ");

    // Step 4 — fetch the GitHub username (best-effort; fall back to "unknown").
    let username = fetch_github_username(http, &oauth_token)
        .await
        .unwrap_or_else(|e| {
            tracing::warn!(error = %e, "could not fetch GitHub username; using 'unknown'");
            "unknown".to_string()
        });

    Ok((oauth_token, username))
}

// ---------------------------------------------------------------------------
// Step 1 — device code request
// ---------------------------------------------------------------------------

async fn request_device_code(
    http: &reqwest::Client,
    client_id: &str,
) -> Result<DeviceCodeResponse> {
    // GitHub's device code endpoint expects form-encoded data, not JSON.
    let body = format!("client_id={client_id}&scope=");

    let resp = http
        .post(DEVICE_CODE_URL)
        .header("Accept", "application/json")
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("User-Agent", concat!("amaebi/", env!("CARGO_PKG_VERSION")))
        .body(body)
        .send()
        .await
        .context("requesting device code from GitHub")?
        .error_for_status()
        .context("GitHub device code endpoint returned an error")?;

    resp.json::<DeviceCodeResponse>()
        .await
        .context("parsing device code response")
}

// ---------------------------------------------------------------------------
// Step 2 — display
// ---------------------------------------------------------------------------

/// Show the user code and verification URL in the terminal, and optionally in
/// a tmux status-bar message / popup and the system browser.
async fn display_code(user_code: &str, verification_uri: &str) {
    println!("  ┌─────────────────────────────────────────────┐");
    println!("  │         Amaebi — GitHub Copilot Login        │");
    println!("  ├─────────────────────────────────────────────┤");
    println!("  │                                             │");
    println!("  │  1. Open:  {verification_uri:<33} │");
    println!("  │  2. Enter: {user_code:<33} │");
    println!("  │                                             │");
    println!("  └─────────────────────────────────────────────┘");
    println!();

    // Try to open the browser silently in the background.
    let _ = Proc::new("xdg-open").arg(verification_uri).spawn();

    // If running inside tmux, use two display mechanisms:
    if std::env::var("TMUX").is_ok() {
        // (a) Persist the code in the status bar until we clear it.
        let status_msg = format!("amaebi login — visit {verification_uri}  enter: {user_code}");
        let _ = Proc::new("tmux")
            .args(["display-message", "-d", "0", &status_msg])
            .output()
            .await;

        // (b) Show a popup with the code (tmux ≥ 3.2).  Fire-and-forget —
        //     the popup auto-dismisses after 30 s or on 'q'.  We write to a
        //     temp file to avoid quoting issues with the shell -c argument.
        let tmp = "/tmp/amaebi-auth.txt";
        let popup_text = format!(
            "\n  Amaebi — GitHub Copilot Login\n\n\
             \r  Visit:  {verification_uri}\n\
             \r  Enter:  {user_code}\n\n\
             \r  Waiting for authorization in your terminal...\n\
             \r  (press q to dismiss this popup)\n"
        );
        if std::fs::write(tmp, popup_text).is_ok() {
            let popup_cmd = format!("cat {tmp}; sleep 30");
            let _ = Proc::new("tmux")
                .args([
                    "display-popup",
                    "-T",
                    " Amaebi Login ",
                    "-w",
                    "62",
                    "-h",
                    "12",
                    "-x",
                    "C",
                    "-y",
                    "C",
                    "--",
                    "sh",
                    "-c",
                    &popup_cmd,
                ])
                .spawn();
        }
    }
}

// ---------------------------------------------------------------------------
// Step 3 — polling loop
// ---------------------------------------------------------------------------

async fn poll_for_token(
    http: &reqwest::Client,
    client_id: &str,
    device_code: &str,
    initial_interval: u64,
    expires_in: u64,
) -> Result<String> {
    // Leave a 10 s margin so we don't attempt a final poll on an expired code.
    let deadline = Instant::now() + Duration::from_secs(expires_in.saturating_sub(10));
    let mut interval = initial_interval.min(MAX_POLL_INTERVAL_SECS);

    loop {
        if Instant::now() >= deadline {
            anyhow::bail!("device code expired — please run `amaebi auth` again");
        }

        // Show a live countdown on the same terminal line.
        let remaining = deadline.saturating_duration_since(Instant::now());
        print!(
            "\r  Polling GitHub... ({:>2}m {:02}s remaining)  ",
            remaining.as_secs() / 60,
            remaining.as_secs() % 60,
        );
        flush_stdout();

        tokio::time::sleep(Duration::from_secs(interval)).await;

        let body = format!(
            "client_id={client_id}\
             &device_code={device_code}\
             &grant_type={DEVICE_GRANT_TYPE}"
        );

        let resp = http
            .post(TOKEN_URL)
            .header("Accept", "application/json")
            .header("Content-Type", "application/x-www-form-urlencoded")
            .header("User-Agent", concat!("amaebi/", env!("CARGO_PKG_VERSION")))
            .body(body)
            .send()
            .await
            .context("polling GitHub access token endpoint")?;

        // GitHub returns HTTP 200 for both success and pending — parse the body.
        let poll: TokenPollResponse = resp.json().await.context("parsing token poll response")?;

        if let Some(token) = poll.access_token.filter(|t| !t.is_empty()) {
            return Ok(token);
        }

        match poll.error.as_deref() {
            None | Some("authorization_pending") => {
                // Normal — user hasn't approved yet.
            }
            Some("slow_down") => {
                // Server asks us to back off; increase the interval by 5 s.
                interval = (interval + 5).min(MAX_POLL_INTERVAL_SECS);
                tracing::debug!(interval, "received slow_down, backing off");
            }
            Some("expired_token") => {
                anyhow::bail!("device code expired — please run `amaebi auth` again");
            }
            Some("access_denied") => {
                anyhow::bail!("authorization denied — the user clicked 'Deny' on GitHub");
            }
            Some(other) => {
                let desc = poll.error_description.as_deref().unwrap_or("");
                anyhow::bail!("GitHub error: {other} — {desc}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Step 4 — GitHub username lookup
// ---------------------------------------------------------------------------

async fn fetch_github_username(http: &reqwest::Client, oauth_token: &str) -> Result<String> {
    let resp = http
        .get(USER_API_URL)
        .header("Authorization", format!("token {oauth_token}"))
        .header("Accept", "application/json")
        .header("User-Agent", concat!("amaebi/", env!("CARGO_PKG_VERSION")))
        .send()
        .await
        .context("fetching GitHub user info")?
        .error_for_status()
        .context("GitHub user API returned an error")?;

    let user: GitHubUser = resp.json().await.context("parsing GitHub user response")?;
    Ok(user.login)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn flush_stdout() {
    let _ = std::io::stdout().flush();
}
