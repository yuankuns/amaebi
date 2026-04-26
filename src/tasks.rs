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
//! * `kind = "verdict"` — one row per supervision turn, containing the
//!   supervisor LLM's `WAIT: ...` / `STEER: ...` / `DONE: ...` line.
//!   Read back as context for subsequent supervision runs on the same
//!   tag.
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

/// Maximum history rows returned to the supervisor per turn.
pub const RECENT_VERDICTS_WINDOW: usize = 10;

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

/// Append a supervision verdict row.  Timestamp is derived server-side
/// so callers need not order them.
pub fn append_verdict(conn: &Connection, repo_dir: &str, tag: &str, verdict: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO task_notes (repo_dir, tag, kind, content, timestamp)
         VALUES (?1, ?2, 'verdict', ?3, ?4)",
        params![repo_dir, tag, verdict, now_epoch()],
    )
    .context("appending verdict")?;
    Ok(())
}

/// Return up to [`RECENT_VERDICTS_WINDOW`] most recent verdicts, oldest
/// first (so the supervisor sees them in chronological order when
/// rendered into a prompt).
pub fn recent_verdicts(conn: &Connection, repo_dir: &str, tag: &str) -> Result<Vec<String>> {
    let mut stmt = conn
        .prepare(
            "SELECT content FROM task_notes
             WHERE repo_dir = ?1 AND tag = ?2 AND kind = 'verdict'
             ORDER BY timestamp DESC
             LIMIT ?3",
        )
        .context("preparing recent_verdicts query")?;
    let rows = stmt
        .query_map(
            params![repo_dir, tag, RECENT_VERDICTS_WINDOW as i64],
            |row| row.get::<_, String>(0),
        )
        .context("running recent_verdicts query")?;
    let mut out: Vec<String> = rows
        .collect::<rusqlite::Result<Vec<_>>>()
        .context("collecting recent_verdicts rows")?;
    out.reverse(); // oldest first for prompt rendering
    Ok(out)
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
    fn recent_verdicts_empty_initially() {
        let (conn, _g) = fresh_db();
        let v = recent_verdicts(&conn, "/proj", "kernel").unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn append_verdicts_and_read_oldest_first() {
        let (conn, _g) = fresh_db();
        // Use explicit timestamps so ordering is deterministic.
        for (i, v) in ["first", "second", "third"].iter().enumerate() {
            conn.execute(
                "INSERT INTO task_notes (repo_dir, tag, kind, content, timestamp) \
                 VALUES ('/proj', 'kernel', 'verdict', ?1, ?2)",
                params![v, now_epoch() + i as i64],
            )
            .unwrap();
        }
        let v = recent_verdicts(&conn, "/proj", "kernel").unwrap();
        assert_eq!(v, vec!["first", "second", "third"]);
    }

    #[test]
    fn recent_verdicts_caps_at_window() {
        let (conn, _g) = fresh_db();
        let total = RECENT_VERDICTS_WINDOW + 5;
        for i in 0..total {
            conn.execute(
                "INSERT INTO task_notes (repo_dir, tag, kind, content, timestamp) \
                 VALUES ('/proj', 'kernel', 'verdict', ?1, ?2)",
                params![format!("v{i}"), now_epoch() + i as i64],
            )
            .unwrap();
        }
        let v = recent_verdicts(&conn, "/proj", "kernel").unwrap();
        assert_eq!(v.len(), RECENT_VERDICTS_WINDOW);
        // Oldest-first means we should see the last 10 inserted, in order.
        assert_eq!(
            v.first().unwrap(),
            &format!("v{}", total - RECENT_VERDICTS_WINDOW)
        );
        assert_eq!(v.last().unwrap(), &format!("v{}", total - 1));
    }

    #[test]
    fn verdicts_isolated_by_repo_dir_and_tag() {
        let (conn, _g) = fresh_db();
        append_verdict(&conn, "/proj-a", "kernel", "a-v").unwrap();
        append_verdict(&conn, "/proj-b", "kernel", "b-v").unwrap();
        append_verdict(&conn, "/proj-a", "lint", "lint-v").unwrap();

        assert_eq!(
            recent_verdicts(&conn, "/proj-a", "kernel").unwrap(),
            vec!["a-v".to_string()]
        );
        assert_eq!(
            recent_verdicts(&conn, "/proj-b", "kernel").unwrap(),
            vec!["b-v".to_string()]
        );
        assert_eq!(
            recent_verdicts(&conn, "/proj-a", "lint").unwrap(),
            vec!["lint-v".to_string()]
        );
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
