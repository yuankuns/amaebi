use std::io::IsTerminal as _;
use std::path::Path;

/// Startup logo rendered in Calvin S box-drawing style.
const LOGO: &str = "
  ╔═╗╔╦╗╔═╗╔═╗╔╗ ╦
  ╠═╣║║║╠═╣║╣ ╠╩╗║
  ╩ ╩╩ ╩╩ ╩╚═╝╚═╝╩
";

/// Inner logic for [`should_show`], parameterised so tests can inject values
/// without touching process-global env vars.
fn should_show_impl(no_banner_env: Option<&str>, stderr_is_tty: bool) -> bool {
    // Opt-out: only the exact value "1" suppresses.
    if no_banner_env == Some("1") {
        return false;
    }
    stderr_is_tty
}

/// Returns `true` when the startup banner should be printed.
///
/// Suppressed when either:
/// - `AMAEBI_NO_BANNER=1` is set in the environment, or
/// - stderr is not a terminal (non-interactive / pipe / redirect mode).
pub fn should_show() -> bool {
    should_show_impl(
        std::env::var("AMAEBI_NO_BANNER").ok().as_deref(),
        std::io::stderr().is_terminal(),
    )
}

/// Print the banner and a compact runtime status block to stderr.
///
/// Callers are responsible for gating this behind [`should_show`].
pub fn print(model: &str, session_id: &str, cwd: &Path) {
    let version = env!("CARGO_PKG_VERSION");
    let commit = env!("AMAEBI_GIT_COMMIT");

    let sandbox = match std::env::var("AMAEBI_SANDBOX").as_deref() {
        Ok("docker") => {
            let image = std::env::var("AMAEBI_SANDBOX_IMAGE")
                .unwrap_or_else(|_| "amaebi-sandbox:bookworm-slim".to_string());
            format!("docker ({image})")
        }
        _ => "off".to_string(),
    };

    // Always show the resolved provider/ prefix so users see which backend
    // will be hit.  Unknown prefixes (e.g. `azure/`) route to Bedrock by
    // default — reflect that rather than echoing the raw input.
    let spec = crate::provider::resolve(model);
    let model_display = if model.starts_with("copilot/") || model.starts_with("bedrock/") {
        model.to_string()
    } else {
        format!("{}/{}", spec.provider, model)
    };

    eprint!("{LOGO}");
    eprintln!("  version  {version} ({commit})");
    eprintln!("  model    {model_display}");
    eprintln!("  sandbox  {sandbox}");
    eprintln!("  session  {session_id}");
    eprintln!("  cwd      {}", cwd.display());
    eprintln!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logo_is_non_empty() {
        assert!(!LOGO.trim().is_empty());
    }

    #[test]
    fn suppressed_when_no_banner_env_is_one_regardless_of_tty() {
        assert!(!should_show_impl(Some("1"), true));
        assert!(!should_show_impl(Some("1"), false));
    }

    #[test]
    fn suppressed_when_not_tty() {
        assert!(!should_show_impl(None, false));
        assert!(!should_show_impl(Some("0"), false));
    }

    #[test]
    fn shown_when_tty_and_no_env_override() {
        assert!(should_show_impl(None, true));
    }

    #[test]
    fn env_var_value_other_than_one_does_not_suppress() {
        assert!(should_show_impl(Some("true"), true));
        assert!(should_show_impl(Some("yes"), true));
        assert!(should_show_impl(Some(""), true));
    }
}
