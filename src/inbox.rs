//! Persistent inbox for cron task reports.
//!
//! Stores completed cron job outputs in an SQLite database at
//! `~/.amaebi/inbox.db`.  The CLI checks for unread entries on every
//! invocation and prints a bell notification when any are present.
//!
//! # Schema
//!
//! ```sql
//! CREATE TABLE inbox (
//!     id               INTEGER PRIMARY KEY AUTOINCREMENT,
//!     session_id       TEXT    NOT NULL,
//!     task_description TEXT    NOT NULL,
//!     output           TEXT    NOT NULL,
//!     created_at       TEXT    NOT NULL,
//!     read             INTEGER NOT NULL DEFAULT 0
//! );
//! ```
//!
//! # Concurrency
//!
//! SQLite's WAL mode is enabled so concurrent readers do not block writers.
//! The database is opened with a 5-second busy timeout so concurrent CLI
//! invocations wait rather than failing immediately.

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use crate::auth::amaebi_home;

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

/// A single cron-task report stored in the inbox.
#[derive(Debug, Clone)]
pub struct InboxReport {
    /// Autoincrement primary key.
    pub id: i64,
    /// Session UUID of the project that triggered the cron task.
    pub session_id: String,
    /// Human-readable description of the task that was executed.
    pub task_description: String,
    /// The agent's full output text.
    pub output: String,
    /// RFC 3339 creation timestamp.
    pub created_at: String,
    /// Whether this report has been read by the user.
    pub read: bool,
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// SQLite-backed store for inbox reports.
///
/// Each method opens the database at the configured path; the connection is
/// not kept open between calls so the store is cheap to construct and does not
/// hold a file descriptor when idle.
pub struct InboxStore {
    db_path: PathBuf,
}

impl InboxStore {
    /// Open (or create) the inbox database at `~/.amaebi/inbox.db`.
    pub fn open() -> Result<Self> {
        let db_path = amaebi_home()?.join("inbox.db");
        let store = Self { db_path };
        store.init()?;
        Ok(store)
    }

    /// Open the inbox database at an explicit path (used in tests and cron integration).
    #[allow(dead_code)]
    pub fn open_at(db_path: PathBuf) -> Result<Self> {
        let store = Self { db_path };
        store.init()?;
        Ok(store)
    }

    fn connect(&self) -> Result<Connection> {
        if let Some(parent) = self.db_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }

        let conn = Connection::open(&self.db_path)
            .with_context(|| format!("opening inbox.db at {}", self.db_path.display()))?;

        // Enforce 0600 permissions on the database file (may contain user output).
        std::fs::set_permissions(&self.db_path, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("setting permissions on {}", self.db_path.display()))?;

        // WAL mode: readers don't block writers; writers don't block readers.
        conn.execute_batch("PRAGMA journal_mode=WAL;")
            .context("enabling WAL mode")?;
        // 5-second busy timeout to handle concurrent CLI invocations gracefully.
        conn.busy_timeout(std::time::Duration::from_secs(5))
            .context("setting busy timeout")?;

        Ok(conn)
    }

    /// Create the inbox table if it does not already exist.
    fn init(&self) -> Result<()> {
        let conn = self.connect()?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS inbox (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id       TEXT    NOT NULL,
                task_description TEXT    NOT NULL,
                output           TEXT    NOT NULL,
                created_at       TEXT    NOT NULL,
                read             INTEGER NOT NULL DEFAULT 0
            );",
        )
        .context("creating inbox table")?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Write operations
    // -----------------------------------------------------------------------

    /// Persist a completed cron task report.
    ///
    /// Returns the row `id` of the newly inserted report.
    /// Called by the daemon's cron scheduler when a task completes (Phase 3).
    #[allow(dead_code)]
    pub fn save_report(
        &self,
        session_id: &str,
        task_description: &str,
        output: &str,
    ) -> Result<i64> {
        let conn = self.connect()?;
        let created_at = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO inbox (session_id, task_description, output, created_at, read)
             VALUES (?1, ?2, ?3, ?4, 0)",
            params![session_id, task_description, output, created_at],
        )
        .context("inserting inbox report")?;
        Ok(conn.last_insert_rowid())
    }

    /// Mark a single report as read by its `id`.
    pub fn mark_read(&self, id: i64) -> Result<()> {
        let conn = self.connect()?;
        conn.execute("UPDATE inbox SET read = 1 WHERE id = ?1", params![id])
            .context("marking report as read")?;
        Ok(())
    }

    /// Mark all unread reports as read.
    pub fn mark_all_read(&self) -> Result<()> {
        let conn = self.connect()?;
        conn.execute("UPDATE inbox SET read = 1 WHERE read = 0", [])
            .context("marking all reports as read")?;
        Ok(())
    }

    /// Delete a single report by its `id`.
    #[allow(dead_code)]
    pub fn delete(&self, id: i64) -> Result<()> {
        let conn = self.connect()?;
        conn.execute("DELETE FROM inbox WHERE id = ?1", params![id])
            .context("deleting inbox report")?;
        Ok(())
    }

    /// Delete all reports from the inbox.
    pub fn clear(&self) -> Result<()> {
        let conn = self.connect()?;
        conn.execute("DELETE FROM inbox", [])
            .context("clearing inbox")?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Read operations
    // -----------------------------------------------------------------------

    /// Return the number of unread reports.
    ///
    /// Returns `0` if the database does not exist yet (no cron tasks have
    /// ever completed).
    pub fn unread_count(&self) -> Result<usize> {
        if !self.db_path.exists() {
            return Ok(0);
        }
        let conn = self.connect()?;
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM inbox WHERE read = 0", [], |row| {
                row.get(0)
            })
            .context("counting unread inbox reports")?;
        Ok(count as usize)
    }

    /// Return all unread reports, oldest first.
    pub fn get_unread(&self) -> Result<Vec<InboxReport>> {
        if !self.db_path.exists() {
            return Ok(vec![]);
        }
        let conn = self.connect()?;
        self.query_reports(&conn, "WHERE read = 0 ORDER BY id ASC")
    }

    /// Return all reports (read and unread), newest first.
    pub fn get_all(&self) -> Result<Vec<InboxReport>> {
        if !self.db_path.exists() {
            return Ok(vec![]);
        }
        let conn = self.connect()?;
        self.query_reports(&conn, "ORDER BY id DESC")
    }

    /// Return a single report by its `id`, if it exists.
    pub fn get_by_id(&self, id: i64) -> Result<Option<InboxReport>> {
        if !self.db_path.exists() {
            return Ok(None);
        }
        let conn = self.connect()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, task_description, output, created_at, read
                 FROM inbox WHERE id = ?1",
            )
            .context("preparing get_by_id statement")?;

        let mut rows = stmt.query(params![id]).context("querying report by id")?;
        if let Some(row) = rows.next().context("reading row")? {
            Ok(Some(row_to_report(row)?))
        } else {
            Ok(None)
        }
    }

    fn query_reports(&self, conn: &Connection, where_clause: &str) -> Result<Vec<InboxReport>> {
        let sql = format!(
            "SELECT id, session_id, task_description, output, created_at, read
             FROM inbox {where_clause}"
        );
        let mut stmt = conn.prepare(&sql).context("preparing inbox query")?;
        let reports: Result<Vec<_>, _> = stmt
            .query_map([], |row| {
                Ok(InboxReport {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    task_description: row.get(2)?,
                    output: row.get(3)?,
                    created_at: row.get(4)?,
                    read: row.get::<_, i64>(5)? != 0,
                })
            })
            .context("executing inbox query")?
            .collect();
        reports.context("reading inbox rows")
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn row_to_report(row: &rusqlite::Row<'_>) -> rusqlite::Result<InboxReport> {
    Ok(InboxReport {
        id: row.get(0)?,
        session_id: row.get(1)?,
        task_description: row.get(2)?,
        output: row.get(3)?,
        created_at: row.get(4)?,
        read: row.get::<_, i64>(5)? != 0,
    })
}

/// Return the path to the inbox database, for use in diagnostic output.
#[allow(dead_code)]
pub fn db_path() -> Result<PathBuf> {
    Ok(amaebi_home()?.join("inbox.db"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::with_temp_home;
    use tempfile::tempdir;

    /// Returns both the store and the `TempDir` that must be kept alive.
    fn make_store() -> (InboxStore, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let store = InboxStore::open_at(dir.path().join("inbox.db")).unwrap();
        (store, dir)
    }

    #[test]
    fn unread_count_zero_when_empty() {
        let _guard = with_temp_home();
        let (store, _dir) = make_store();
        assert_eq!(store.unread_count().unwrap(), 0);
    }

    #[test]
    fn save_and_count_unread() {
        let _guard = with_temp_home();
        let (store, _dir) = make_store();
        store
            .save_report("session-1", "Run tests", "All passed")
            .unwrap();
        store
            .save_report("session-1", "Build release", "Success")
            .unwrap();
        assert_eq!(store.unread_count().unwrap(), 2);
    }

    #[test]
    fn get_unread_returns_unread_only() {
        let _guard = with_temp_home();
        let (store, _dir) = make_store();
        let id1 = store.save_report("s1", "Task A", "output A").unwrap();
        let _id2 = store.save_report("s1", "Task B", "output B").unwrap();
        store.mark_read(id1).unwrap();

        let unread = store.get_unread().unwrap();
        assert_eq!(unread.len(), 1);
        assert_eq!(unread[0].task_description, "Task B");
    }

    #[test]
    fn get_all_returns_all() {
        let _guard = with_temp_home();
        let (store, _dir) = make_store();
        store.save_report("s1", "Task A", "output A").unwrap();
        store.save_report("s1", "Task B", "output B").unwrap();
        store.mark_read(store.get_unread().unwrap()[0].id).unwrap();

        let all = store.get_all().unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn mark_read_clears_unread() {
        let _guard = with_temp_home();
        let (store, _dir) = make_store();
        let id = store.save_report("s1", "Task", "output").unwrap();
        assert_eq!(store.unread_count().unwrap(), 1);
        store.mark_read(id).unwrap();
        assert_eq!(store.unread_count().unwrap(), 0);
    }

    #[test]
    fn mark_all_read_clears_all() {
        let _guard = with_temp_home();
        let (store, _dir) = make_store();
        store.save_report("s1", "Task A", "out").unwrap();
        store.save_report("s1", "Task B", "out").unwrap();
        store.mark_all_read().unwrap();
        assert_eq!(store.unread_count().unwrap(), 0);
    }

    #[test]
    fn get_by_id_returns_correct_report() {
        let _guard = with_temp_home();
        let (store, _dir) = make_store();
        let id = store.save_report("sess", "My task", "My output").unwrap();
        let report = store.get_by_id(id).unwrap().unwrap();
        assert_eq!(report.task_description, "My task");
        assert_eq!(report.output, "My output");
        assert_eq!(report.session_id, "sess");
        assert!(!report.read);
    }

    #[test]
    fn get_by_id_returns_none_for_missing() {
        let _guard = with_temp_home();
        let (store, _dir) = make_store();
        assert!(store.get_by_id(999).unwrap().is_none());
    }

    #[test]
    fn delete_removes_report() {
        let _guard = with_temp_home();
        let (store, _dir) = make_store();
        let id = store.save_report("s1", "Task", "output").unwrap();
        store.delete(id).unwrap();
        assert_eq!(store.unread_count().unwrap(), 0);
        assert!(store.get_by_id(id).unwrap().is_none());
    }

    #[test]
    fn clear_removes_all_reports() {
        let _guard = with_temp_home();
        let (store, _dir) = make_store();
        store.save_report("s1", "A", "a").unwrap();
        store.save_report("s1", "B", "b").unwrap();
        store.clear().unwrap();
        assert_eq!(store.get_all().unwrap().len(), 0);
        assert_eq!(store.unread_count().unwrap(), 0);
    }

    #[test]
    fn get_unread_ordered_oldest_first() {
        let _guard = with_temp_home();
        let (store, _dir) = make_store();
        store.save_report("s1", "First", "a").unwrap();
        store.save_report("s1", "Second", "b").unwrap();
        store.save_report("s1", "Third", "c").unwrap();
        let unread = store.get_unread().unwrap();
        assert_eq!(unread[0].task_description, "First");
        assert_eq!(unread[2].task_description, "Third");
    }

    #[test]
    fn report_read_field_updates_after_mark_read() {
        let _guard = with_temp_home();
        let (store, _dir) = make_store();
        let id = store.save_report("s1", "Task", "out").unwrap();
        let before = store.get_by_id(id).unwrap().unwrap();
        assert!(!before.read);
        store.mark_read(id).unwrap();
        let after = store.get_by_id(id).unwrap().unwrap();
        assert!(after.read);
    }

    #[test]
    fn db_file_permissions_are_0600() {
        let _guard = with_temp_home();
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("inbox.db");
        let store = InboxStore::open_at(db_path.clone()).unwrap();
        store.save_report("s", "t", "o").unwrap();
        let mode = std::fs::metadata(&db_path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "inbox.db should be mode 0600, got {mode:o}");
    }
}
