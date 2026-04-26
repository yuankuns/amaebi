//! LLM-based tag generator for the task notebook.
//!
//! When the user doesn't pass `--tag <tag>`, the daemon asks Haiku
//! for a short ASCII hyphen-separated tag derived from the task
//! description.  Haiku is used unconditionally here — tagging is a
//! short, cheap call that doesn't need Opus/Sonnet, and isolating it
//! from the user's chat model avoids burning through a more expensive
//! quota on housekeeping work.
//!
//! The tagger also receives the list of existing tags in the same
//! `repo_dir` so it can collapse semantically similar tasks onto a
//! prior tag (enabling cross-invocation resume for a continuing
//! effort described with different wording each time).
//!
//! Failures (model unreachable, quota, network, malformed output)
//! fall back to a slug-with-date tag so the notebook path keeps
//! working — we never block task launch on tag generation.

use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::Local;
use rusqlite::params;

use crate::copilot::Message;
use crate::daemon::DaemonState;

/// Maximum completion tokens for the tag call.  A tag is a handful of
/// words — 32 is ample and keeps Haiku fast.
const TAGGER_MAX_TOKENS: usize = 32;

/// How many of the most recent existing tags to show Haiku as context.
/// Keeps the prompt short while still covering the "ongoing work"
/// window a user is likely to want to resume.
const EXISTING_TAG_CONTEXT_LIMIT: usize = 20;

/// Derive the Haiku model id from the user's chat model, preserving
/// provider.  If the user runs Bedrock we tag with Bedrock Haiku; if
/// they run Copilot we tag with Copilot Haiku.  Keeps auth / billing
/// on the same rails they already set up.  `None` → bedrock default.
fn tagger_model_for(chat_model: Option<&str>) -> String {
    let provider = chat_model
        .and_then(|m| {
            let (head, rest) = m.split_once('/')?;
            // Only accept non-empty head and explicit provider prefix;
            // without a `/` we don't know the provider and fall back
            // to bedrock below.
            if head.is_empty() || rest.is_empty() {
                None
            } else {
                Some(head)
            }
        })
        .unwrap_or("bedrock");
    format!("{provider}/claude-haiku-4.5")
}

/// Fallback tag when the LLM call fails: slug of the description
/// (keeping CJK characters, not just ASCII) plus a `MMDD` suffix so
/// the same description on different days doesn't collide.
pub fn fallback_tag(description: &str) -> String {
    let slug: String = description
        .chars()
        .take(32)
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();
    let collapsed = {
        let mut out = String::new();
        let mut prev_dash = false;
        for c in slug.chars() {
            if c == '-' {
                if !prev_dash {
                    out.push('-');
                }
                prev_dash = true;
            } else {
                out.push(c);
                prev_dash = false;
            }
        }
        out.trim_matches('-').to_string()
    };
    let date = Local::now().format("%m%d").to_string();
    if collapsed.is_empty() {
        format!("task-{date}")
    } else {
        format!("{collapsed}-{date}")
    }
}

/// Return the most recent tags in `repo_dir` paired with the latest
/// `desc` row for each tag, up to [`EXISTING_TAG_CONTEXT_LIMIT`].
/// Giving Haiku BOTH the tag name and its desc lets it judge
/// semantic continuity — without the desc, a tag name is opaque and
/// Haiku tends to blindly reuse whatever it sees.
fn existing_tag_context(state: &Arc<DaemonState>, repo_dir: &str) -> Result<Vec<(String, String)>> {
    let guard = state
        .tasks_db
        .lock()
        .map_err(|e| anyhow::anyhow!("tasks_db mutex poisoned: {e}"))?;
    let Some(conn) = guard.as_ref() else {
        return Ok(Vec::new());
    };
    // For each tag under this repo, pick the most recent `desc` row.
    // Correlated subquery is cheapest for the small scale expected
    // (typically a handful of tags per repo).
    let mut stmt = conn
        .prepare(
            "SELECT tag, COALESCE(( \
                 SELECT content FROM task_notes d \
                 WHERE d.repo_dir = task_notes.repo_dir \
                   AND d.tag = task_notes.tag \
                   AND d.kind = 'desc' \
                 ORDER BY d.timestamp DESC LIMIT 1 \
             ), '') AS latest_desc \
             FROM task_notes \
             WHERE repo_dir = ?1 \
             GROUP BY tag \
             ORDER BY MAX(timestamp) DESC \
             LIMIT ?2",
        )
        .context("preparing existing_tag_context query")?;
    let rows = stmt
        .query_map(
            params![repo_dir, EXISTING_TAG_CONTEXT_LIMIT as i64],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .context("running existing_tag_context query")?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .context("collecting existing_tag_context rows")
}

/// Ask Haiku for a tag.  Synchronous round-trip; callers await it.
/// On any failure returns the slug-with-date fallback so downstream
/// paths (lease / verdict / desc persistence) always get a usable tag.
pub async fn generate_tag(
    state: &Arc<DaemonState>,
    chat_model: Option<&str>,
    description: &str,
    repo_dir: &str,
) -> String {
    // Ensure the DB is open before we read from it; cheap when already
    // initialised.  Running inside spawn_blocking here keeps file I/O
    // off the async executor on first call.
    let state_for_ensure = Arc::clone(state);
    let _ = tokio::task::spawn_blocking(move || crate::daemon::ensure_tasks_db(&state_for_ensure))
        .await;

    let existing = {
        let state_for_list = Arc::clone(state);
        let repo = repo_dir.to_string();
        tokio::task::spawn_blocking(move || existing_tag_context(&state_for_list, &repo))
            .await
            .unwrap_or_else(|e| Err(anyhow::anyhow!("existing_tag_context panicked: {e}")))
            .unwrap_or_else(|e| {
                tracing::warn!(error = %e, "failed to list existing tags; proceeding with empty context");
                Vec::new()
            })
    };

    let system = "You are naming a long-running supervision task for a local notebook. \
                  Output exactly one tag and nothing else.\n\n\
                  Default behaviour: CREATE A NEW TAG derived from the new task description. \
                  Only reuse an existing tag when the new description is clearly a direct \
                  continuation of the SAME underlying task (same feature being iterated, \
                  same bug being hunted, same migration being advanced).  Different \
                  subsystems or unrelated work in the same repo MUST get a new tag.\n\n\
                  Rules:\n\
                  - 2 to 4 words, lowercase, ASCII only, hyphen-separated \
                  (e.g. `fmha4-paged-migration`).\n\
                  - No punctuation other than hyphens.  No quotes, no explanation.\n\
                  - Keep it short; the tag is a lookup key, not a summary.";

    let existing_block = if existing.is_empty() {
        "No existing tags in this repo.".to_string()
    } else {
        let mut s = String::from("Existing tags in this repo (most recently active first):\n");
        for (tag, desc) in &existing {
            // Truncate historical desc to keep the prompt compact.
            let desc_preview: String = desc.chars().take(120).collect();
            let ellipsis = if desc.chars().count() > 120 {
                "…"
            } else {
                ""
            };
            if desc.is_empty() {
                s.push_str(&format!("- `{tag}` (no recorded desc)\n"));
            } else {
                s.push_str(&format!("- `{tag}`: {desc_preview}{ellipsis}\n"));
            }
        }
        s
    };

    let user = format!(
        "{existing_block}\n\
         New task description:\n{description}\n\n\
         Output only the tag."
    );

    let messages = vec![Message::system(system), Message::user(user)];
    let model = tagger_model_for(chat_model);

    let mut sink = tokio::io::sink();
    let response = match crate::daemon::invoke_model_for_tagger(
        state,
        &model,
        &messages,
        TAGGER_MAX_TOKENS,
        &mut sink,
    )
    .await
    {
        Ok(r) => r.text,
        Err(e) => {
            tracing::warn!(error = %e, model, "tagger LLM call failed; using fallback");
            return fallback_tag(description);
        }
    };

    let cleaned = sanitise_tag(&response);
    if cleaned.is_empty() {
        tracing::warn!(raw = %response, "tagger returned empty/invalid tag; using fallback");
        return fallback_tag(description);
    }
    cleaned
}

/// Strip the LLM output down to the tag shape we want.  Takes the
/// first whitespace-delimited token, lowercases it, and keeps only
/// ASCII alphanumerics and `-`.  Chinese / emoji / markdown fences
/// all get scrubbed out here so a loosely-formatted response still
/// produces a valid key.
fn sanitise_tag(raw: &str) -> String {
    // Try each whitespace-delimited token; pick the first one that
    // yields non-empty ASCII after filtering.  Handles LLM outputs
    // that lead with decoration (emoji, quote marks) before the
    // actual tag.
    for token in raw.split_whitespace() {
        let lower = token.to_lowercase();
        let filtered: String = lower
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
            .collect();
        let mut out = String::with_capacity(filtered.len());
        let mut prev_dash = false;
        for c in filtered.chars() {
            if c == '-' {
                if !prev_dash {
                    out.push(c);
                }
                prev_dash = true;
            } else {
                out.push(c);
                prev_dash = false;
            }
        }
        let trimmed = out.trim_matches('-');
        let bounded: String = trimmed.chars().take(48).collect();
        let result = bounded.trim_end_matches('-').to_string();
        if !result.is_empty() {
            return result;
        }
    }
    String::new()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_tag_ascii_desc() {
        let tag = fallback_tag("fix the lint error");
        // Starts with a recognisable slug.
        assert!(tag.starts_with("fix-the-lint-error"), "got {tag}");
        // Ends with MMDD (4 digits after final `-`).
        let suffix = tag.rsplit('-').next().unwrap();
        assert_eq!(suffix.len(), 4);
        assert!(suffix.chars().all(|c| c.is_ascii_digit()));
    }

    #[test]
    fn fallback_tag_cjk_desc_keeps_unicode_slug() {
        let tag = fallback_tag("继续 fmha4 迁移");
        // Contains CJK characters — SQLite TEXT handles them fine.
        assert!(tag.contains("继续"), "got {tag}");
        assert!(tag.contains("fmha4"), "got {tag}");
        // Still date-suffixed.
        assert!(tag
            .rsplit('-')
            .next()
            .unwrap()
            .chars()
            .all(|c| c.is_ascii_digit()));
    }

    #[test]
    fn fallback_tag_empty_desc_uses_task_date() {
        let tag = fallback_tag("");
        assert!(tag.starts_with("task-"), "got {tag}");
    }

    #[test]
    fn sanitise_tag_strips_quotes_and_extra_text() {
        assert_eq!(
            sanitise_tag("`fmha4-paged-migration`"),
            "fmha4-paged-migration"
        );
        assert_eq!(
            sanitise_tag("\"fmha4-paged-migration\" — continues prior work"),
            "fmha4-paged-migration"
        );
    }

    #[test]
    fn sanitise_tag_uppercase_and_underscore() {
        assert_eq!(sanitise_tag("Kernel_Opt"), "kernelopt");
    }

    #[test]
    fn sanitise_tag_collapses_dashes_and_caps_length() {
        let tag = sanitise_tag(&"a-".repeat(50));
        assert!(!tag.contains("--"));
        assert!(tag.chars().count() <= 48);
    }

    #[test]
    fn sanitise_tag_drops_emoji_and_cjk() {
        assert_eq!(sanitise_tag("🚀 launch-speed 🚀"), "launch-speed");
        assert_eq!(sanitise_tag("迁移-lint"), "lint");
    }

    #[test]
    fn tagger_model_preserves_provider() {
        assert_eq!(
            tagger_model_for(Some("bedrock/claude-opus-4.7")),
            "bedrock/claude-haiku-4.5"
        );
        assert_eq!(
            tagger_model_for(Some("copilot/gpt-4o")),
            "copilot/claude-haiku-4.5"
        );
        // No provider prefix → default to bedrock.
        assert_eq!(
            tagger_model_for(Some("claude-sonnet-4.6")),
            "bedrock/claude-haiku-4.5"
        );
        assert_eq!(tagger_model_for(None), "bedrock/claude-haiku-4.5");
    }
}
