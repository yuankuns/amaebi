use anyhow::{Context, Result};
use serde::Deserialize;

const MODELS_ENDPOINT: &str = "https://api.githubcopilot.com/models";

#[derive(Deserialize, Debug)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(Deserialize, Debug)]
struct ModelEntry {
    id: String,
    #[serde(default)]
    vendor: Option<String>,
    #[serde(default)]
    capabilities: Option<ModelCapabilities>,
    #[serde(default)]
    model_picker_enabled: bool,
}

#[derive(Deserialize, Debug)]
struct ModelCapabilities {
    #[serde(default)]
    family: Option<String>,
    #[serde(rename = "type", default)]
    kind: Option<String>,
}

/// Derive a display category from the model ID / family.
fn category(id: &str, family: Option<&str>) -> &'static str {
    let haystack = family.unwrap_or(id).to_lowercase();
    let id_lower = id.to_lowercase();
    if haystack.contains("opus")
        || id_lower.contains("opus")
        || haystack.contains("o1-preview")
        || (haystack.starts_with("gpt-4") && !haystack.contains("mini") && !haystack.contains("4o"))
    {
        "powerful"
    } else if haystack.contains("mini")
        || haystack.contains("haiku")
        || id_lower.contains("mini")
        || id_lower.contains("haiku")
        || id_lower.contains("flash")
    {
        "lightweight"
    } else {
        "versatile"
    }
}

/// List available models for all providers.
pub async fn run() -> Result<()> {
    // ── Bedrock aliases ───────────────────────────────────────────────────
    println!("Bedrock (use as: bedrock/<alias> or just <alias>):");
    let aliases = crate::provider::bedrock_aliases();
    let alias_width = aliases.iter().map(|(a, _)| a.len()).max().unwrap_or(0);
    for &(alias, full_id) in aliases {
        println!(
            "  {:<alias_width$}  → {full_id}",
            alias,
            alias_width = alias_width
        );
    }
    println!();

    // ── Copilot models ────────────────────────────────────────────────────
    println!("Copilot (use as: copilot/<model>):");
    match run_copilot_models().await {
        Ok(()) => {}
        Err(e) => {
            println!("  (could not fetch Copilot models: {e:#})");
        }
    }

    Ok(())
}

/// Fetch models from the Copilot API and print a table.
async fn run_copilot_models() -> Result<()> {
    let oauth_token = crate::auth::read_oauth_token()?;
    let http = reqwest::Client::new();

    // Exchange OAuth token for a short-lived Copilot API token.
    #[derive(Deserialize)]
    struct TokenResp {
        token: String,
    }
    let token_resp: TokenResp = http
        .get("https://api.github.com/copilot_internal/v2/token")
        .header("Authorization", format!("token {oauth_token}"))
        .header("Accept", "application/json")
        .header("User-Agent", concat!("amaebi/", env!("CARGO_PKG_VERSION")))
        .send()
        .await
        .context("fetching Copilot API token")?
        .error_for_status()
        .context("Copilot token endpoint returned an error")?
        .json()
        .await
        .context("parsing Copilot API token response")?;

    let api_token = token_resp.token;

    let resp: ModelsResponse = http
        .get(MODELS_ENDPOINT)
        .header("Authorization", format!("Bearer {api_token}"))
        .header("Copilot-Integration-Id", "vscode-chat")
        .header("Editor-Version", "vscode/1.90.0")
        .header("Accept", "application/json")
        .header("User-Agent", concat!("amaebi/", env!("CARGO_PKG_VERSION")))
        .send()
        .await
        .context("fetching Copilot models")?
        .error_for_status()
        .context("Copilot models endpoint returned an error")?
        .json()
        .await
        .context("parsing Copilot models response")?;

    // Only show models that are picker-enabled (available for use).
    let mut models: Vec<&ModelEntry> = resp
        .data
        .iter()
        .filter(|m| m.model_picker_enabled)
        .collect();
    models.sort_by_key(|m| m.id.as_str());

    if models.is_empty() {
        println!("No models available.");
        return Ok(());
    }

    // Compute column widths.
    let id_width = models.iter().map(|m| m.id.len()).max().unwrap_or(0).max(5);
    let vendor_width = models
        .iter()
        .map(|m| m.vendor.as_deref().unwrap_or("").len())
        .max()
        .unwrap_or(0)
        .max(6);

    for m in models {
        let vendor = m.vendor.as_deref().unwrap_or("");
        let family = m.capabilities.as_ref().and_then(|c| c.family.as_deref());
        let cat = category(&m.id, family);

        // Check if non-chat (embedding etc.) to mark as such.
        let kind_tag = m
            .capabilities
            .as_ref()
            .and_then(|c| c.kind.as_deref())
            .filter(|k| *k != "chat")
            .map(|k| format!("  [{k}]"))
            .unwrap_or_default();

        println!(
            "  {:<id_width$}  {:<vendor_width$}  {}{}",
            m.id,
            vendor,
            cat,
            kind_tag,
            id_width = id_width,
            vendor_width = vendor_width,
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::category;

    #[test]
    fn category_powerful() {
        assert_eq!(category("claude-opus-4.6", Some("claude-opus")), "powerful");
        assert_eq!(category("gpt-4-turbo", Some("gpt-4-turbo")), "powerful");
    }

    #[test]
    fn category_lightweight() {
        assert_eq!(
            category("claude-haiku-3.5", Some("claude-haiku")),
            "lightweight"
        );
        assert_eq!(category("gpt-4o-mini", Some("gpt-4o-mini")), "lightweight");
        assert_eq!(category("gemini-flash", None), "lightweight");
    }

    #[test]
    fn category_versatile() {
        assert_eq!(
            category("claude-sonnet-4.6", Some("claude-sonnet")),
            "versatile"
        );
        assert_eq!(category("gpt-4o", Some("gpt-4o")), "versatile");
        assert_eq!(category("o1", None), "versatile");
    }
}
