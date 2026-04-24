//! Model provider resolution.
//!
//! Parses the `provider/model` syntax (e.g. `bedrock/claude-sonnet-4.6`,
//! `copilot/gpt-4o`) and resolves Bedrock-friendly aliases to full model IDs.
//!
//! When no provider prefix is given, the default provider is **Bedrock**.

use std::collections::HashMap;

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
///
/// Includes the `[1m]` opt-in suffix so Bedrock requests carry the
/// `context-1m-2025-08-07` beta header, letting sessions grow past the
/// 200k window.  Small requests cost the same as plain
/// `claude-sonnet-4.6`.  Copilot does not accept the 1M beta, so callers
/// targeting Copilot must use [`default_model_for_provider`] (or
/// otherwise strip the suffix).
pub const DEFAULT_MODEL: &str = "claude-sonnet-4.6[1m]";

/// The bare default model name, without the `[1m]` suffix.  Used when the
/// target provider does not support 1M context (currently: Copilot).
pub const DEFAULT_MODEL_BARE: &str = "claude-sonnet-4.6";

/// Return the appropriate default model string for `provider_prefix`.
///
/// `bedrock` (or any unknown prefix routed to Bedrock) gets the `[1m]`
/// suffix; `copilot` gets the bare name because Copilot does not accept
/// the Bedrock 1M beta opt-in.
pub fn default_model_for_provider(provider_prefix: &str) -> &'static str {
    match provider_prefix {
        "copilot" => DEFAULT_MODEL_BARE,
        _ => DEFAULT_MODEL,
    }
}

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
    resolve_with_aliases(raw, &HashMap::new())
}

/// Parse a raw model string into a [`ModelSpec`], consulting user-defined
/// aliases from the config file before falling back to the built-in table.
///
/// Resolution order when the bare name has no recognised provider prefix:
/// 1. Built-in `BEDROCK_ALIASES` — wins on name conflict.
/// 2. `user_aliases` — the value replaces `bare` and re-enters the normal
///    prefix-parsing path (so aliases may expand to `copilot/...`,
///    `bedrock/...`, or another bare name).  Alias values are resolved
///    exactly once; no chain expansion.
/// 3. Default provider (Bedrock), passing the bare name through unchanged.
pub fn resolve_with_aliases(raw: &str, user_aliases: &HashMap<String, String>) -> ModelSpec {
    let display_name = raw.to_owned();

    // Strip the `[1m]` opt-in suffix before routing and alias resolution.
    // The flag is preserved in `display_name` and `use_1m`.
    let use_1m = raw.ends_with("[1m]");
    let bare: &str = if use_1m {
        &raw[..raw.len() - "[1m]".len()]
    } else {
        raw
    };

    // Expand user alias only when `bare` is a plain name (no provider prefix)
    // AND does not collide with a built-in Bedrock alias.  Built-in wins on
    // conflict so the LLM-visible `claude-opus-4.6` etc. are stable.
    let expanded: String;
    let bare: &str = if bare.contains('/') || is_builtin_bedrock_alias(bare) {
        bare
    } else if let Some(target) = user_aliases.get(bare) {
        expanded = target.clone();
        &expanded
    } else {
        bare
    };

    // Alias targets in config.json may themselves carry a `[1m]` suffix
    // (e.g. `"sonnet": "bedrock/claude-sonnet-4.6[1m]"`).  Normalize again
    // after alias expansion so the suffix is always recorded on `use_1m`
    // and never left embedded inside the Bedrock model id.
    let (bare, use_1m) = if let Some(stripped) = bare.strip_suffix("[1m]") {
        (stripped, true)
    } else {
        (bare, use_1m)
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

/// Returns `true` when `name` is a key in the built-in Bedrock alias table.
///
/// Used by the daemon's user-alias expansion logic so that built-in aliases
/// always win on name conflicts.
pub fn is_builtin_bedrock_alias(name: &str) -> bool {
    BEDROCK_ALIASES.iter().any(|(a, _)| *a == name)
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

    // ---- default_model_for_provider ---------------------------------------

    #[test]
    fn default_model_for_copilot_strips_1m() {
        assert_eq!(default_model_for_provider("copilot"), DEFAULT_MODEL_BARE);
        assert!(!default_model_for_provider("copilot").ends_with("[1m]"));
    }

    #[test]
    fn default_model_for_bedrock_and_unknown_keep_1m() {
        assert_eq!(default_model_for_provider("bedrock"), DEFAULT_MODEL);
        assert_eq!(default_model_for_provider("unknown"), DEFAULT_MODEL);
        assert!(default_model_for_provider("bedrock").ends_with("[1m]"));
    }

    // ---- resolve_with_aliases() -------------------------------------------

    #[test]
    fn resolve_with_aliases_expands_user_alias() {
        let mut map = HashMap::new();
        map.insert("opus".into(), "bedrock/claude-opus-4.7".into());
        let spec = resolve_with_aliases("opus", &map);
        assert_eq!(spec.provider, ProviderKind::Bedrock);
        assert_eq!(spec.model_id, "us.anthropic.claude-opus-4-7");
        // display_name preserves what the user typed.
        assert_eq!(spec.display_name, "opus");
    }

    #[test]
    fn user_alias_does_not_override_builtin() {
        // Built-in alias `claude-opus-4.6` must win even when the user tries
        // to point it at something else.
        let mut map = HashMap::new();
        map.insert("claude-opus-4.6".into(), "bedrock/claude-sonnet-4.6".into());
        let spec = resolve_with_aliases("claude-opus-4.6", &map);
        assert_eq!(spec.model_id, "us.anthropic.claude-opus-4-6-v1");
    }

    #[test]
    fn user_alias_preserves_1m_suffix() {
        let mut map = HashMap::new();
        map.insert("opus".into(), "bedrock/claude-opus-4.7".into());
        let spec = resolve_with_aliases("opus[1m]", &map);
        assert!(spec.use_1m);
        assert_eq!(spec.model_id, "us.anthropic.claude-opus-4-7");
        assert_eq!(spec.display_name, "opus[1m]");
    }

    #[test]
    fn user_alias_target_can_include_1m_suffix() {
        // config.json may carry `[1m]` inside the target; it must be parsed
        // as the opt-in flag, not left embedded in the resolved model id.
        let mut map = HashMap::new();
        map.insert("sonnet".into(), "bedrock/claude-sonnet-4.6[1m]".into());
        let spec = resolve_with_aliases("sonnet", &map);
        assert!(spec.use_1m);
        assert_eq!(spec.model_id, "us.anthropic.claude-sonnet-4-6");
    }

    #[test]
    fn user_alias_target_1m_and_input_1m_both_enable_use_1m() {
        // User typed `sonnet[1m]` AND the alias target also has `[1m]` —
        // resolution stays idempotent.
        let mut map = HashMap::new();
        map.insert("sonnet".into(), "bedrock/claude-sonnet-4.6[1m]".into());
        let spec = resolve_with_aliases("sonnet[1m]", &map);
        assert!(spec.use_1m);
        assert_eq!(spec.model_id, "us.anthropic.claude-sonnet-4-6");
    }

    #[test]
    fn user_alias_target_is_not_chain_resolved() {
        // `a -> b`, `b -> bedrock/...`.  Expanding "a" must stop at "b".
        let mut map = HashMap::new();
        map.insert("a".into(), "b".into());
        map.insert("b".into(), "bedrock/claude-opus-4.7".into());
        let spec = resolve_with_aliases("a", &map);
        // "b" is not a built-in alias, so it passes through unchanged.
        assert_eq!(spec.model_id, "b");
        assert_eq!(spec.provider, ProviderKind::Bedrock);
    }

    #[test]
    fn user_alias_can_target_copilot() {
        let mut map = HashMap::new();
        map.insert("mini".into(), "copilot/gpt-4o-mini".into());
        let spec = resolve_with_aliases("mini", &map);
        assert_eq!(spec.provider, ProviderKind::Copilot);
        assert_eq!(spec.model_id, "gpt-4o-mini");
    }

    #[test]
    fn resolve_wrapper_passes_empty_aliases() {
        // The zero-config wrapper must behave exactly like the old resolve().
        let spec = resolve("claude-opus-4.7");
        assert_eq!(spec.model_id, "us.anthropic.claude-opus-4-7");
    }

    #[test]
    fn user_alias_ignored_when_input_has_prefix() {
        // `bedrock/foo` should never try to expand `bedrock/foo` as a user alias.
        let mut map = HashMap::new();
        map.insert(
            "bedrock/claude-opus-4.7".into(),
            "bedrock/claude-sonnet-4.6".into(),
        );
        let spec = resolve_with_aliases("bedrock/claude-opus-4.7", &map);
        assert_eq!(spec.model_id, "us.anthropic.claude-opus-4-7");
    }
}
