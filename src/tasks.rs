//! Per-tag supervision notebook.
//!
//! Persists supervision-loop verdicts and task descriptions across
//! supervision sessions so a long-running task can resume with history
//! intact when `/claude --tag <tag>` is invoked again.
//!
//! One table, keyed by `(repo_dir, tag)`:
//!
//! * `kind = "desc"` — the `<desc>` passed on the CLI.  Written once per
//!   `/claude --tag <tag> "<desc>"` invocation.  On resume without a
//!   `<desc>`, the supervisor falls back to the most recent `desc` row.
//! * `kind = "verdict"` — one row per supervision turn.  Content is a
//!   JSON-serialised [`VerdictRecord`] capturing the four-field
//!   verdict (`stated_intent`, `observed_action`, `verdict`,
//!   `rationale`) plus an optional `steer_message`.  Rows written
//!   before this schema existed are read back as
//!   [`VerdictKind::Legacy`] with the raw text in `rationale` — no
//!   migration required.
//! * `kind = "lease"` — at most one live row per `(repo_dir, tag)`.
//!   Guarantees same-tag supervision is serialised: parallel `/claude
//!   --tag <tag>` invocations synchronously reject while another
//!   session still holds the lease.  TTL is 24 h; a stale lease is
//!   auto-replaced on the next acquire.
//!
//! DB path: `~/.amaebi/tasks.db`.

use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use anyhow::{Context, Result};
use rusqlite::{params, Connection};

use crate::auth::amaebi_home;

/// Seconds after which a `lease` row is considered stale and may be
/// replaced without the holder having released it.  Matches
/// `pane_lease::LEASE_TTL_SECS` so a crashed holder is reclaimable on
/// the same timescale.
pub const LEASE_TTL_SECS: i64 = 86_400;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS task_notes (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_dir  TEXT    NOT NULL,
    tag       TEXT    NOT NULL,
    kind      TEXT    NOT NULL,
    content   TEXT    NOT NULL,
    timestamp INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_task_notes_lookup
    ON task_notes(repo_dir, tag, kind, timestamp DESC);
"#;

/// Returns `~/.amaebi/tasks.db`.
pub fn db_path() -> Result<PathBuf> {
    Ok(amaebi_home()?.join("tasks.db"))
}

/// Open the task-notebook database and apply schema.  Creates the file
/// and parent directory if missing.  Sets `0600` permissions on Unix.
pub fn init_db(path: &Path) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }

    let conn = Connection::open(path)
        .with_context(|| format!("opening tasks DB at {}", path.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Err(e) = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)) {
            tracing::debug!(error = %e, "could not set permissions on tasks DB");
        }
        // Make sure the file inherits 0600 even when open(2) created it
        // just now (umask dependent).
        let _ = OpenOptions::new().mode(0o600).write(true).open(path);
    }

    conn.busy_timeout(std::time::Duration::from_millis(5000))
        .context("setting SQLite busy timeout")?;

    conn.execute_batch(SCHEMA)
        .context("applying tasks schema")?;

    Ok(conn)
}

fn now_epoch() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Lease
// ---------------------------------------------------------------------------

/// Outcome of [`acquire_lease`].
#[derive(Debug)]
pub enum AcquireLeaseResult {
    /// Lease granted; caller must call [`release_lease`] before exiting.
    Acquired,
    /// Another live holder owns this `(repo_dir, tag)` — caller must
    /// abort the current request with a user-visible error.
    Held { holder: String, age_secs: i64 },
}

/// Try to acquire the lease for `(repo_dir, tag)`.
///
/// Semantics:
/// - No existing lease row → insert one, return `Acquired`.
/// - Existing row within [`LEASE_TTL_SECS`] → return `Held` with the
///   incumbent `holder` and the row's age.
/// - Existing row older than [`LEASE_TTL_SECS`] (stale) → replace it
///   atomically and return `Acquired`.
///
/// All three paths run inside one `IMMEDIATE` transaction so two
/// concurrent acquires cannot both win.
pub fn acquire_lease(
    conn: &mut Connection,
    repo_dir: &str,
    tag: &str,
    holder: &str,
) -> Result<AcquireLeaseResult> {
    let now = now_epoch();
    let tx = conn
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .context("beginning acquire_lease transaction")?;

    let existing: Option<(i64, String, i64)> = tx
        .query_row(
            "SELECT id, content, timestamp FROM task_notes
             WHERE repo_dir = ?1 AND tag = ?2 AND kind = 'lease'
             ORDER BY timestamp DESC
             LIMIT 1",
            params![repo_dir, tag],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .or_else(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => Ok((0, String::new(), 0)),
            other => Err(other),
        })
        .map(|(id, content, ts)| {
            if id == 0 {
                None
            } else {
                Some((id, content, ts))
            }
        })
        .context("reading existing lease")?;

    if let Some((_id, incumbent, ts)) = existing {
        let age = now.saturating_sub(ts);
        if age <= LEASE_TTL_SECS {
            tx.rollback().ok();
            return Ok(AcquireLeaseResult::Held {
                holder: incumbent,
                age_secs: age,
            });
        }
        // Stale — drop **all** lease rows for this key, not just the
        // most recent.  Repeated crashes-and-reacquires could otherwise
        // leave older stale rows accumulating indefinitely.
        tx.execute(
            "DELETE FROM task_notes WHERE repo_dir = ?1 AND tag = ?2 AND kind = 'lease'",
            params![repo_dir, tag],
        )
        .context("deleting stale leases")?;
    }

    tx.execute(
        "INSERT INTO task_notes (repo_dir, tag, kind, content, timestamp)
         VALUES (?1, ?2, 'lease', ?3, ?4)",
        params![repo_dir, tag, holder, now],
    )
    .context("inserting lease row")?;

    tx.commit().context("committing acquire_lease")?;
    Ok(AcquireLeaseResult::Acquired)
}

/// Release the lease held by `holder` for `(repo_dir, tag)`.  No-op if
/// no matching row exists (e.g. a previous release already ran).
pub fn release_lease(conn: &Connection, repo_dir: &str, tag: &str, holder: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM task_notes
         WHERE repo_dir = ?1 AND tag = ?2 AND kind = 'lease' AND content = ?3",
        params![repo_dir, tag, holder],
    )
    .context("releasing lease")?;
    Ok(())
}

/// Force-release every live lease matching `holder`, regardless of
/// repo_dir / tag.  Used by `handle_supervision` cleanup when the
/// holder id (e.g. session UUID) is known but the tag is not
/// conveniently in scope.  No-op if the holder never acquired anything.
pub fn release_all_by_holder(conn: &Connection, holder: &str) -> Result<usize> {
    let n = conn
        .execute(
            "DELETE FROM task_notes WHERE kind = 'lease' AND content = ?1",
            params![holder],
        )
        .context("releasing leases by holder")?;
    Ok(n)
}

// ---------------------------------------------------------------------------
// Desc
// ---------------------------------------------------------------------------

/// Append a new `desc` row.  Previous descs are kept as history; readers
/// pick the most recent one.
pub fn append_desc(conn: &Connection, repo_dir: &str, tag: &str, desc: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO task_notes (repo_dir, tag, kind, content, timestamp)
         VALUES (?1, ?2, 'desc', ?3, ?4)",
        params![repo_dir, tag, desc, now_epoch()],
    )
    .context("appending desc")?;
    Ok(())
}

/// Return the most recent `desc` content for `(repo_dir, tag)`, or
/// `None` when no desc has ever been written.
pub fn latest_desc(conn: &Connection, repo_dir: &str, tag: &str) -> Result<Option<String>> {
    conn.query_row(
        "SELECT content FROM task_notes
         WHERE repo_dir = ?1 AND tag = ?2 AND kind = 'desc'
         ORDER BY timestamp DESC
         LIMIT 1",
        params![repo_dir, tag],
        |row| row.get::<_, String>(0),
    )
    .map(Some)
    .or_else(|e| match e {
        rusqlite::Error::QueryReturnedNoRows => Ok(None),
        other => Err(anyhow::Error::from(other).context("reading latest desc")),
    })
}

// ---------------------------------------------------------------------------
// Verdicts
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Structured verdicts
// ---------------------------------------------------------------------------

/// One supervision turn's full verdict record.  Serialised as JSON into
/// the existing `content` column so no schema migration is required;
/// rows written before this rework (plain `"WAIT: ..."` / `"STEER: ..."`
/// strings) are read back as [`VerdictKind::Legacy`] with the raw text
/// placed in `rationale`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct VerdictRecord {
    /// What Claude claims to be doing this turn — distilled by the
    /// supervisor from pane capture, not regex-scraped.  Empty string
    /// when the supervisor produced no field value (including LEGACY
    /// rows written before this schema existed).
    #[serde(default)]
    pub stated_intent: String,
    /// Ground-truth observation: git diff delta, touched files, tool
    /// calls this turn.
    #[serde(default)]
    pub observed_action: String,
    /// The classification.
    pub verdict: VerdictKind,
    /// One-sentence reason for the verdict.
    #[serde(default)]
    pub rationale: String,
    /// STEER payload actually sent into the pane — only present when
    /// `verdict == Steer` AND the message was dispatched successfully.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steer_message: Option<String>,
    /// Unix epoch seconds — server-assigned on append, populated on read.
    /// Not serialised into the JSON blob (the row's `timestamp` column
    /// is the source of truth).
    #[serde(skip)]
    pub timestamp: i64,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum VerdictKind {
    Wait,
    Steer,
    Done,
    /// Placeholder for rows written before the structured schema existed.
    /// Reader maps any non-JSON `content` to this variant.
    Legacy,
}

impl VerdictRecord {
    /// Render as a compact single-line summary suitable for
    /// `amaebi tag list` and legacy prompt sections that still expect a
    /// string form.
    pub fn to_line(&self) -> String {
        match self.verdict {
            VerdictKind::Wait => {
                if self.rationale.is_empty() {
                    "WAIT".to_string()
                } else {
                    format!("WAIT: {}", self.rationale)
                }
            }
            VerdictKind::Steer => match &self.steer_message {
                Some(m) => format!("STEER: {m}"),
                None => format!("STEER: {}", self.rationale),
            },
            VerdictKind::Done => format!("DONE: {}", self.rationale),
            VerdictKind::Legacy => self.rationale.clone(),
        }
    }
}

/// Append a structured verdict.  Stored as JSON in `content`.
pub fn append_verdict_record(
    conn: &Connection,
    repo_dir: &str,
    tag: &str,
    record: &VerdictRecord,
) -> Result<()> {
    let json = serde_json::to_string(record).context("serialising verdict record")?;
    conn.execute(
        "INSERT INTO task_notes (repo_dir, tag, kind, content, timestamp)
         VALUES (?1, ?2, 'verdict', ?3, ?4)",
        params![repo_dir, tag, json, now_epoch()],
    )
    .context("appending verdict record")?;
    Ok(())
}

/// Return every verdict row for `(repo_dir, tag)`, oldest first.
///
/// Used by the event-stream history renderer which needs to run its
/// own filtering (keep all non-WAIT, dedupe sustained WAIT, fold runs)
/// rather than the fixed time-window in [`recent_verdicts`].  Legacy
/// plain-string rows are surfaced as [`VerdictKind::Legacy`] so their
/// content still makes it into the prompt.
pub fn all_verdict_records(
    conn: &Connection,
    repo_dir: &str,
    tag: &str,
) -> Result<Vec<VerdictRecord>> {
    let mut stmt = conn
        .prepare(
            "SELECT content, timestamp FROM task_notes
             WHERE repo_dir = ?1 AND tag = ?2 AND kind = 'verdict'
             ORDER BY timestamp ASC, id ASC",
        )
        .context("preparing all_verdict_records query")?;
    let rows = stmt
        .query_map(params![repo_dir, tag], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })
        .context("running all_verdict_records query")?;
    let mut out = Vec::new();
    for row in rows {
        let (content, ts) = row.context("decoding all_verdict_records row")?;
        let mut rec = parse_verdict_content(&content);
        rec.timestamp = ts;
        out.push(rec);
    }
    Ok(out)
}

/// Parse a `task_notes.content` string into a [`VerdictRecord`],
/// tolerating legacy plain-text rows from before the JSON schema.
fn parse_verdict_content(raw: &str) -> VerdictRecord {
    if let Ok(parsed) = serde_json::from_str::<VerdictRecord>(raw) {
        return parsed;
    }
    VerdictRecord {
        stated_intent: String::new(),
        observed_action: String::new(),
        verdict: VerdictKind::Legacy,
        rationale: raw.to_string(),
        steer_message: None,
        timestamp: 0,
    }
}

// ---------------------------------------------------------------------------
// Observability — `amaebi tag list`
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ActiveLease {
    pub repo_dir: String,
    pub tag: String,
    pub holder: String,
    pub age_secs: i64,
}

/// Return every live (non-stale) lease.  Stale rows are filtered out so
/// `amaebi tag list` does not show dead holders; they are cleaned up
/// on the next acquire.
pub fn list_active_leases(conn: &Connection) -> Result<Vec<ActiveLease>> {
    let now = now_epoch();
    let mut stmt = conn
        .prepare(
            "SELECT repo_dir, tag, content, timestamp FROM task_notes
             WHERE kind = 'lease'
             ORDER BY timestamp DESC",
        )
        .context("preparing list_active_leases query")?;
    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })
        .context("running list_active_leases query")?;
    let mut out = Vec::new();
    for row in rows {
        let (repo_dir, tag, holder, ts) = row.context("decoding list_active_leases row")?;
        let age = now.saturating_sub(ts);
        if age <= LEASE_TTL_SECS {
            out.push(ActiveLease {
                repo_dir,
                tag,
                holder,
                age_secs: age,
            });
        }
    }
    Ok(out)
}

/// Force-release any live lease for `(repo_dir, tag)`, regardless of
/// holder.  Used by `amaebi tag release <tag>` to recover from a
/// stranded holder without waiting out the 24 h TTL.  Returns the
/// number of rows deleted.
pub fn force_release(conn: &Connection, repo_dir: &str, tag: &str) -> Result<usize> {
    let n = conn
        .execute(
            "DELETE FROM task_notes
             WHERE repo_dir = ?1 AND tag = ?2 AND kind = 'lease'",
            params![repo_dir, tag],
        )
        .context("force-releasing lease")?;
    Ok(n)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn fresh_db() -> (Connection, TempDir) {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("tasks.db");
        let conn = init_db(&path).expect("init");
        (conn, dir)
    }

    // ── desc round-trip ─────────────────────────────────────────────────

    #[test]
    fn desc_not_present_initially() {
        let (conn, _g) = fresh_db();
        assert_eq!(latest_desc(&conn, "/proj", "kernel").unwrap(), None);
    }

    #[test]
    fn append_and_read_latest_desc() {
        let (conn, _g) = fresh_db();
        append_desc(&conn, "/proj", "kernel", "first desc").unwrap();
        assert_eq!(
            latest_desc(&conn, "/proj", "kernel").unwrap().as_deref(),
            Some("first desc")
        );
    }

    #[test]
    fn latest_desc_returns_most_recent() {
        let (conn, _g) = fresh_db();
        append_desc(&conn, "/proj", "kernel", "v1").unwrap();
        // Force later timestamp (seconds resolution) — use an INSERT with
        // explicit timestamp since test runs faster than 1s.
        conn.execute(
            "INSERT INTO task_notes (repo_dir, tag, kind, content, timestamp) \
             VALUES ('/proj', 'kernel', 'desc', 'v2', ?1)",
            params![now_epoch() + 1],
        )
        .unwrap();
        assert_eq!(
            latest_desc(&conn, "/proj", "kernel").unwrap().as_deref(),
            Some("v2")
        );
    }

    #[test]
    fn desc_isolated_by_repo_dir_and_tag() {
        let (conn, _g) = fresh_db();
        append_desc(&conn, "/proj-a", "kernel", "a-kernel").unwrap();
        append_desc(&conn, "/proj-b", "kernel", "b-kernel").unwrap();
        append_desc(&conn, "/proj-a", "lint", "a-lint").unwrap();

        assert_eq!(
            latest_desc(&conn, "/proj-a", "kernel").unwrap().as_deref(),
            Some("a-kernel")
        );
        assert_eq!(
            latest_desc(&conn, "/proj-b", "kernel").unwrap().as_deref(),
            Some("b-kernel")
        );
        assert_eq!(
            latest_desc(&conn, "/proj-a", "lint").unwrap().as_deref(),
            Some("a-lint")
        );
    }

    // ── verdicts ─────────────────────────────────────────────────────────

    #[test]
    fn structured_verdict_roundtrips() {
        let (conn, _g) = fresh_db();
        let rec = VerdictRecord {
            stated_intent: "read src/auth.rs to understand sessions".into(),
            observed_action: "modified src/daemon.rs".into(),
            verdict: VerdictKind::Steer,
            rationale: "touched file outside scope".into(),
            steer_message: Some("stop, return to reading src/auth.rs".into()),
            timestamp: 0,
        };
        append_verdict_record(&conn, "/proj", "kernel", &rec).unwrap();

        let all = all_verdict_records(&conn, "/proj", "kernel").unwrap();
        assert_eq!(all.len(), 1);
        let back = &all[0];
        assert_eq!(back.stated_intent, rec.stated_intent);
        assert_eq!(back.observed_action, rec.observed_action);
        assert_eq!(back.verdict, VerdictKind::Steer);
        assert_eq!(
            back.steer_message.as_deref(),
            Some(rec.steer_message.as_deref().unwrap())
        );
        assert!(back.timestamp > 0);
    }

    #[test]
    fn legacy_plain_string_verdict_reads_as_legacy() {
        let (conn, _g) = fresh_db();
        // Simulate a row written before the JSON schema existed.
        conn.execute(
            "INSERT INTO task_notes (repo_dir, tag, kind, content, timestamp) \
             VALUES ('/proj', 'kernel', 'verdict', ?1, ?2)",
            params!["WAIT: still reading files", now_epoch()],
        )
        .unwrap();

        let all = all_verdict_records(&conn, "/proj", "kernel").unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].verdict, VerdictKind::Legacy);
        assert_eq!(all[0].rationale, "WAIT: still reading files");
        // Legacy render must still produce a non-empty line for the prompt.
        assert_eq!(all[0].to_line(), "WAIT: still reading files");
    }

    #[test]
    fn verdicts_isolated_by_repo_dir_and_tag() {
        let (conn, _g) = fresh_db();
        let mk = |rationale: &str| VerdictRecord {
            stated_intent: String::new(),
            observed_action: String::new(),
            verdict: VerdictKind::Wait,
            rationale: rationale.into(),
            steer_message: None,
            timestamp: 0,
        };
        append_verdict_record(&conn, "/proj-a", "kernel", &mk("a-v")).unwrap();
        append_verdict_record(&conn, "/proj-b", "kernel", &mk("b-v")).unwrap();
        append_verdict_record(&conn, "/proj-a", "lint", &mk("lint-v")).unwrap();

        let reads = |r, t| -> Vec<String> {
            all_verdict_records(&conn, r, t)
                .unwrap()
                .into_iter()
                .map(|v| v.rationale)
                .collect()
        };

        assert_eq!(reads("/proj-a", "kernel"), vec!["a-v".to_string()]);
        assert_eq!(reads("/proj-b", "kernel"), vec!["b-v".to_string()]);
        assert_eq!(reads("/proj-a", "lint"), vec!["lint-v".to_string()]);
    }

    // ── lease ────────────────────────────────────────────────────────────

    #[test]
    fn acquire_fresh_lease_succeeds() {
        let (mut conn, _g) = fresh_db();
        let r = acquire_lease(&mut conn, "/proj", "kernel", "sess-1").unwrap();
        assert!(matches!(r, AcquireLeaseResult::Acquired));
    }

    #[test]
    fn acquire_live_lease_rejected() {
        let (mut conn, _g) = fresh_db();
        acquire_lease(&mut conn, "/proj", "kernel", "sess-1").unwrap();
        let r = acquire_lease(&mut conn, "/proj", "kernel", "sess-2").unwrap();
        match r {
            AcquireLeaseResult::Held { holder, age_secs } => {
                assert_eq!(holder, "sess-1");
                assert!(age_secs <= 5, "acquire just happened, got age={age_secs}");
            }
            other => panic!("expected Held, got {other:?}"),
        }
    }

    #[test]
    fn release_frees_lease_for_next_acquire() {
        let (mut conn, _g) = fresh_db();
        acquire_lease(&mut conn, "/proj", "kernel", "sess-1").unwrap();
        release_lease(&conn, "/proj", "kernel", "sess-1").unwrap();
        let r = acquire_lease(&mut conn, "/proj", "kernel", "sess-2").unwrap();
        assert!(matches!(r, AcquireLeaseResult::Acquired));
    }

    #[test]
    fn release_wrong_holder_noop() {
        let (mut conn, _g) = fresh_db();
        acquire_lease(&mut conn, "/proj", "kernel", "sess-1").unwrap();
        release_lease(&conn, "/proj", "kernel", "stranger").unwrap();
        // sess-2 still cannot acquire.
        let r = acquire_lease(&mut conn, "/proj", "kernel", "sess-2").unwrap();
        assert!(matches!(r, AcquireLeaseResult::Held { .. }));
    }

    #[test]
    fn stale_lease_is_reclaimable_past_ttl() {
        let (mut conn, _g) = fresh_db();
        // Seed a very old lease directly.
        conn.execute(
            "INSERT INTO task_notes (repo_dir, tag, kind, content, timestamp)
             VALUES ('/proj', 'kernel', 'lease', 'ghost', ?1)",
            params![now_epoch() - (LEASE_TTL_SECS + 1)],
        )
        .unwrap();

        let r = acquire_lease(&mut conn, "/proj", "kernel", "fresh").unwrap();
        assert!(matches!(r, AcquireLeaseResult::Acquired));

        // The ghost row is gone, fresh row is live.
        let leases = list_active_leases(&conn).unwrap();
        assert_eq!(leases.len(), 1);
        assert_eq!(leases[0].holder, "fresh");
    }

    #[test]
    fn release_all_by_holder_clears_multiple_tags() {
        let (mut conn, _g) = fresh_db();
        acquire_lease(&mut conn, "/proj", "a", "sess-1").unwrap();
        acquire_lease(&mut conn, "/proj", "b", "sess-1").unwrap();
        acquire_lease(&mut conn, "/proj", "c", "sess-2").unwrap();
        let n = release_all_by_holder(&conn, "sess-1").unwrap();
        assert_eq!(n, 2);
        let live = list_active_leases(&conn).unwrap();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].tag, "c");
    }

    #[test]
    fn force_release_drops_any_live_lease() {
        let (mut conn, _g) = fresh_db();
        acquire_lease(&mut conn, "/proj", "kernel", "sess-stuck").unwrap();
        let n = force_release(&conn, "/proj", "kernel").unwrap();
        assert_eq!(n, 1);
        // Can now acquire as a different holder.
        let r = acquire_lease(&mut conn, "/proj", "kernel", "sess-new").unwrap();
        assert!(matches!(r, AcquireLeaseResult::Acquired));
    }

    #[test]
    fn list_active_leases_filters_stale() {
        let (mut conn, _g) = fresh_db();
        acquire_lease(&mut conn, "/proj", "live", "sess-live").unwrap();
        conn.execute(
            "INSERT INTO task_notes (repo_dir, tag, kind, content, timestamp)
             VALUES ('/proj', 'stale-tag', 'lease', 'ghost', ?1)",
            params![now_epoch() - (LEASE_TTL_SECS + 100)],
        )
        .unwrap();

        let live = list_active_leases(&conn).unwrap();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].tag, "live");
    }

    // ── schema migration (idempotency) ──────────────────────────────────

    #[test]
    fn init_db_twice_is_idempotent() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("tasks.db");
        let _c1 = init_db(&path).expect("first init");
        let c2 = init_db(&path).expect("second init");
        // Still writable after second init.
        append_desc(&c2, "/proj", "kernel", "hi").unwrap();
    }
}
