//! Model provider resolution.
//!
//! Parses the `provider/model` syntax (e.g. `bedrock/claude-sonnet-4.6`,
//! `copilot/gpt-4o`) and resolves Bedrock-friendly aliases to full model IDs.
//!
//! When no provider prefix is given, the default provider is **Bedrock**.

/// Supported model providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    /// GitHub Copilot — routes through the Copilot API gateway.
    Copilot,
    /// Amazon Bedrock — calls ConverseStream with bearer-token auth.
    Bedrock,
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Copilot => write!(f, "copilot"),
            Self::Bedrock => write!(f, "bedrock"),
        }
    }
}

/// A resolved model specification ready for use by the daemon.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// Which backend to send the request to.
    pub provider: ProviderKind,
    /// The actual model ID sent to the provider's API.
    ///
    /// For Copilot this is the model name as-is (e.g. `gpt-4o`).
    /// For Bedrock this is the full model ID after alias resolution
    /// (e.g. `us.anthropic.claude-sonnet-4-6-v1:0`).
    pub model_id: String,
    /// The raw string the user typed — used for logging and display.
    pub display_name: String,
    /// Whether the user requested 1M context via the `[1m]` suffix.
    ///
    /// Only meaningful for Bedrock — Copilot does not support 1M context.
    pub use_1m: bool,
}

// ---------------------------------------------------------------------------
// Bedrock model alias table
// ---------------------------------------------------------------------------

/// Friendly alias → full Bedrock model ID.
///
/// **Maintenance note**: when new Claude models are released on Bedrock,
/// add a new entry here.  The alias is the short name users type on the
/// CLI (e.g. `claude-sonnet-4.6`); the value is the cross-region
/// inference model ID (with `us.` prefix).
///
/// To find the correct model ID, check the Bedrock console or run:
///   `aws bedrock list-foundation-models --by-provider Anthropic`
const BEDROCK_ALIASES: &[(&str, &str)] = &[
    // Claude 4.7 family
    ("claude-opus-4.7", "us.anthropic.claude-opus-4-7"),
    // Claude 4.6 family (cross-region inference profiles — no `:0` suffix)
    ("claude-sonnet-4.6", "us.anthropic.claude-sonnet-4-6"),
    ("claude-opus-4.6", "us.anthropic.claude-opus-4-6-v1"),
    // Claude 4.5 family
    (
        "claude-sonnet-4.5",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ),
    (
        "claude-opus-4.5",
        "us.anthropic.claude-opus-4-5-20251101-v1:0",
    ),
    (
        "claude-haiku-4.5",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    ),
    // Claude 4.1 family
    (
        "claude-opus-4.1",
        "us.anthropic.claude-opus-4-1-20250805-v1:0",
    ),
    // Claude 4 family
    (
        "claude-sonnet-4",
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
    ),
    ("claude-opus-4", "us.anthropic.claude-opus-4-20250514-v1:0"),
    // Claude 3.5 family
    (
        "claude-haiku-3.5",
        "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    ),
];

/// Default model when no `--model` or `AMAEBI_MODEL` is specified.
pub const DEFAULT_MODEL: &str = "claude-sonnet-4.6";

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

/// Parse a raw model string into a [`ModelSpec`].
///
/// Format: `[provider/]model`
///
/// - `bedrock/claude-sonnet-4.6` → Bedrock, alias resolved
/// - `copilot/gpt-4o`            → Copilot, model as-is
/// - `claude-sonnet-4.6`         → Bedrock (default), alias resolved
///
/// Unknown provider prefixes (e.g. `azure/gpt-4o`) are treated as the
/// full model name with the default provider (Bedrock), so the `/` becomes
/// part of the model ID.  This avoids a hard error for forward-compatible
/// model names that happen to contain a slash.
pub fn resolve(raw: &str) -> ModelSpec {
    let display_name = raw.to_owned();

    // Strip the `[1m]` opt-in suffix before routing and alias resolution.
    // The flag is preserved in `display_name` and `use_1m`.
    let use_1m = raw.ends_with("[1m]");
    let bare: &str = if use_1m {
        &raw[..raw.len() - "[1m]".len()]
    } else {
        raw
    };

    if let Some((prefix, model)) = bare.split_once('/') {
        match prefix {
            "copilot" => {
                return ModelSpec {
                    provider: ProviderKind::Copilot,
                    model_id: model.to_owned(),
                    display_name,
                    // Preserve the user's opt-in so the daemon can warn and log
                    // accurately; the Copilot execution path ignores it at
                    // runtime because Copilot does not support 1M context.
                    use_1m,
                };
            }
            "bedrock" => {
                return ModelSpec {
                    provider: ProviderKind::Bedrock,
                    model_id: resolve_bedrock_alias(model),
                    display_name,
                    use_1m,
                };
            }
            _ => {
                // Unknown prefix — fall through to default provider with the
                // entire raw string as the model name.
            }
        }
    }

    // No recognised prefix → default provider (Bedrock) with alias resolution.
    ModelSpec {
        provider: ProviderKind::Bedrock,
        model_id: resolve_bedrock_alias(bare),
        display_name,
        use_1m,
    }
}

/// Resolve a Bedrock-friendly alias to the full model ID.
///
/// Returns the input unchanged when no alias matches, allowing users to
/// pass full model IDs directly (e.g. `us.anthropic.claude-sonnet-4-6-v1:0`).
fn resolve_bedrock_alias(name: &str) -> String {
    for &(alias, full_id) in BEDROCK_ALIASES {
        if name == alias {
            return full_id.to_owned();
        }
    }
    // No alias match — pass through as-is.
    name.to_owned()
}

/// Return the Bedrock alias table for display (e.g. `amaebi models`).
pub fn bedrock_aliases() -> &'static [(&'static str, &'static str)] {
    BEDROCK_ALIASES
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- resolve() parsing ------------------------------------------------

    #[test]
    fn resolve_bedrock_prefix() {
        let spec = resolve("bedrock/claude-sonnet-4.6");
        assert_eq!(spec.provider, ProviderKind::Bedrock);
        assert_eq!(spec.model_id, "us.anthropic.claude-sonnet-4-6");
        assert_eq!(spec.display_name, "bedrock/claude-sonnet-4.6");
    }

    #[test]
    fn resolve_copilot_prefix() {
        let spec = resolve("copilot/gpt-4o");
        assert_eq!(spec.provider, ProviderKind::Copilot);
        assert_eq!(spec.model_id, "gpt-4o");
        assert_eq!(spec.display_name, "copilot/gpt-4o");
    }

    #[test]
    fn resolve_no_prefix_defaults_to_bedrock() {
        let spec = resolve("claude-sonnet-4.6");
        assert_eq!(spec.provider, ProviderKind::Bedrock);
        assert_eq!(spec.model_id, "us.anthropic.claude-sonnet-4-6");
    }

    #[test]
    fn resolve_unknown_provider_prefix_treated_as_model_name() {
        // "azure/gpt-4o" — unknown prefix, entire string becomes model name
        let spec = resolve("azure/gpt-4o");
        assert_eq!(spec.provider, ProviderKind::Bedrock);
        assert_eq!(spec.model_id, "azure/gpt-4o");
    }

    #[test]
    fn resolve_full_bedrock_model_id_passthrough() {
        let spec = resolve("bedrock/us.anthropic.claude-sonnet-4-6-v1:0");
        assert_eq!(spec.provider, ProviderKind::Bedrock);
        assert_eq!(spec.model_id, "us.anthropic.claude-sonnet-4-6-v1:0");
    }

    #[test]
    fn resolve_copilot_model_id_passthrough() {
        // No alias resolution for copilot — model name is passed as-is.
        let spec = resolve("copilot/claude-opus-4.6");
        assert_eq!(spec.provider, ProviderKind::Copilot);
        assert_eq!(spec.model_id, "claude-opus-4.6");
    }

    // ---- Alias resolution -------------------------------------------------

    #[test]
    fn alias_claude_sonnet_4_6() {
        assert_eq!(
            resolve_bedrock_alias("claude-sonnet-4.6"),
            "us.anthropic.claude-sonnet-4-6"
        );
    }

    #[test]
    fn alias_claude_opus_4_7() {
        assert_eq!(
            resolve_bedrock_alias("claude-opus-4.7"),
            "us.anthropic.claude-opus-4-7"
        );
    }

    #[test]
    fn alias_claude_opus_4_6() {
        assert_eq!(
            resolve_bedrock_alias("claude-opus-4.6"),
            "us.anthropic.claude-opus-4-6-v1"
        );
    }

    #[test]
    fn alias_claude_haiku_3_5() {
        assert_eq!(
            resolve_bedrock_alias("claude-haiku-3.5"),
            "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        );
    }

    #[test]
    fn alias_unknown_passthrough() {
        assert_eq!(
            resolve_bedrock_alias("some-future-model"),
            "some-future-model"
        );
    }

    #[test]
    fn alias_full_id_passthrough() {
        let full = "us.anthropic.claude-sonnet-4-6-v1:0";
        assert_eq!(resolve_bedrock_alias(full), full);
    }

    // ---- 1M suffix --------------------------------------------------------

    #[test]
    fn resolve_1m_suffix_bedrock_alias() {
        let spec = resolve("claude-sonnet-4.6[1m]");
        assert_eq!(spec.provider, ProviderKind::Bedrock);
        assert_eq!(spec.model_id, "us.anthropic.claude-sonnet-4-6");
        assert_eq!(spec.display_name, "claude-sonnet-4.6[1m]");
        assert!(spec.use_1m);
    }

    #[test]
    fn resolve_1m_suffix_bedrock_prefix() {
        let spec = resolve("bedrock/us.anthropic.claude-sonnet-4-6[1m]");
        assert_eq!(spec.provider, ProviderKind::Bedrock);
        assert_eq!(spec.model_id, "us.anthropic.claude-sonnet-4-6");
        assert_eq!(
            spec.display_name,
            "bedrock/us.anthropic.claude-sonnet-4-6[1m]"
        );
        assert!(spec.use_1m);
    }

    #[test]
    fn resolve_1m_suffix_copilot_preserves_flag() {
        // Copilot does not support 1M at runtime, but use_1m is preserved so
        // the daemon can warn and log accurately.
        let spec = resolve("copilot/claude-opus-4.6[1m]");
        assert_eq!(spec.provider, ProviderKind::Copilot);
        assert_eq!(spec.model_id, "claude-opus-4.6");
        assert_eq!(spec.display_name, "copilot/claude-opus-4.6[1m]");
        assert!(spec.use_1m);
    }

    #[test]
    fn resolve_no_1m_suffix_use_1m_false() {
        let spec = resolve("claude-sonnet-4.6");
        assert!(!spec.use_1m);
    }

    // ---- Display ----------------------------------------------------------

    #[test]
    fn provider_kind_display() {
        assert_eq!(ProviderKind::Copilot.to_string(), "copilot");
        assert_eq!(ProviderKind::Bedrock.to_string(), "bedrock");
    }

    // ---- bedrock_aliases() ------------------------------------------------

    #[test]
    fn bedrock_aliases_not_empty() {
        assert!(!bedrock_aliases().is_empty());
    }
}
