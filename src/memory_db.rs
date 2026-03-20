//! SQLite-backed memory storage with FTS5 full-text search.
//!
//! Replaces the append-only JSONL store as the primary persistence layer.
//! Stores individual messages (one row per turn) with role metadata, enabling
//! both recency-based retrieval and semantic full-text search.
//!
//! # Concurrency
//!
//! [`rusqlite::Connection`] is `!Send`, so connections are never stored in
//! shared state.  Callers open a fresh connection inside each
//! `tokio::task::spawn_blocking` closure and close it when the closure returns.
//! SQLite's WAL mode allows concurrent readers without blocking.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rusqlite::{params, Connection};

use crate::auth::amaebi_home;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single message row from the `memories` table.
#[derive(Debug, Clone)]
pub struct DbMemoryEntry {
    pub id: i64,
    pub timestamp: String,
    /// Groups messages from the same conversation.  Stored for future use;
    /// callers may leave this as an empty string.
    #[allow(dead_code)]
    pub session_id: String,
    /// `"user"` or `"assistant"`.
    pub role: String,
    pub content: String,
    /// Optional AI-generated summary for compact context injection.
    /// Stored for future use; callers may leave this as an empty string.
    #[allow(dead_code)]
    pub summary: String,
}

// ---------------------------------------------------------------------------
// Path helper
// ---------------------------------------------------------------------------

/// Path to the SQLite memory database (`~/.amaebi/memory.db`).
pub fn db_path() -> Result<PathBuf> {
    Ok(amaebi_home()?.join("memory.db"))
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

const SCHEMA: &str = "
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT    NOT NULL,
    session_id TEXT    NOT NULL DEFAULT '',
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    summary    TEXT    NOT NULL DEFAULT ''
);

-- FTS5 content table mirrors the base table; triggers keep it in sync.
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    summary,
    content  = 'memories',
    content_rowid = 'id'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, summary)
    VALUES (new.id, new.content, new.summary);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary)
    VALUES ('delete', old.id, old.content, old.summary);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary)
    VALUES ('delete', old.id, old.content, old.summary);
    INSERT INTO memories_fts(rowid, content, summary)
    VALUES (new.id, new.content, new.summary);
END;
";

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Open (or create) the SQLite memory database at `path`.
///
/// Applies WAL mode and creates the schema on first run.
/// Sets a 5-second busy timeout so concurrent processes retry instead of
/// failing immediately on lock contention.
/// On Unix, attempts to set `0600` permissions on the database file
/// (best-effort: failures are logged at debug level and not propagated).
pub fn init_db(path: &Path) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }

    let conn = Connection::open(path)
        .with_context(|| format!("opening memory DB at {}", path.display()))?;

    // Enforce 0600 so conversation history is readable only by the owner.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Err(e) = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)) {
            tracing::debug!(error = %e, "could not set permissions on memory DB");
        }
    }

    // Retry for up to 5 s when the DB is locked by another process.
    conn.busy_timeout(std::time::Duration::from_millis(5000))
        .context("setting SQLite busy timeout")?;

    conn.execute_batch(SCHEMA)
        .context("applying memory DB schema")?;

    Ok(conn)
}

/// Insert a single message into the `memories` table.
pub fn store_memory(
    conn: &Connection,
    timestamp: &str,
    session_id: &str,
    role: &str,
    content: &str,
    summary: &str,
) -> Result<()> {
    conn.execute(
        "INSERT INTO memories (timestamp, session_id, role, content, summary)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![timestamp, session_id, role, content, summary],
    )
    .context("inserting memory entry")?;
    Ok(())
}

/// Full-text search over `content` and `summary` fields using FTS5.
///
/// Returns up to `limit` entries ordered by FTS5 relevance rank.
/// Returns an empty list when the query is blank or yields no results.
/// Propagates SQLite errors (e.g. schema issues) as `Err`.
pub fn search_relevant(conn: &Connection, query: &str, limit: usize) -> Result<Vec<DbMemoryEntry>> {
    if query.trim().is_empty() {
        return Ok(vec![]);
    }
    let safe_query = escape_fts5_query(query);
    let mut stmt = conn
        .prepare(
            "SELECT m.id, m.timestamp, m.session_id, m.role, m.content, m.summary
             FROM memories m
             JOIN memories_fts ON memories_fts.rowid = m.id
             WHERE memories_fts MATCH ?1
             ORDER BY memories_fts.rank
             LIMIT ?2",
        )
        .context("preparing FTS5 search query")?;

    let entries = stmt
        .query_map(params![safe_query, limit as i64], row_to_entry)
        .context("executing FTS5 search")?
        .collect::<rusqlite::Result<Vec<_>>>()
        .context("collecting FTS5 results")?;
    Ok(entries)
}

/// Return the most recent `limit` messages in chronological order.
pub fn get_recent(conn: &Connection, limit: usize) -> Result<Vec<DbMemoryEntry>> {
    let mut stmt = conn
        .prepare(
            "SELECT id, timestamp, session_id, role, content, summary
             FROM memories
             ORDER BY id DESC
             LIMIT ?1",
        )
        .context("preparing get_recent query")?;

    let mut entries = stmt
        .query_map(params![limit as i64], row_to_entry)
        .context("executing get_recent")?
        .collect::<rusqlite::Result<Vec<_>>>()
        .context("collecting get_recent results")?;

    entries.reverse(); // chronological order
    Ok(entries)
}

/// Return the total number of rows in the `memories` table.
pub fn count(conn: &Connection) -> Result<usize> {
    conn.query_row("SELECT COUNT(*) FROM memories", [], |r| r.get::<_, i64>(0))
        .context("counting memories")
        .map(|n| n as usize)
}

/// Delete all rows from `memories` (and rebuild the FTS index).
pub fn clear(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "DELETE FROM memories;
         INSERT INTO memories_fts(memories_fts) VALUES ('rebuild');",
    )
    .context("clearing memories")
}

// ---------------------------------------------------------------------------
// Retrieval helper used by build_messages
// ---------------------------------------------------------------------------

/// Combined retrieval: last `recent_n` turns + up to `search_n` FTS matches.
///
/// Deduplicates by `id` and returns entries sorted chronologically.  Recent
/// turns always appear; FTS matches that overlap with the recency window are
/// deduplicated out.
pub fn retrieve_context(
    conn: &Connection,
    prompt: &str,
    recent_n: usize,
    search_n: usize,
) -> Result<Vec<DbMemoryEntry>> {
    let recent = get_recent(conn, recent_n)?;
    let recent_ids: HashSet<i64> = recent.iter().map(|e| e.id).collect();

    let mut relevant = search_relevant(conn, prompt, search_n)?;
    relevant.retain(|e| !recent_ids.contains(&e.id));

    // Merge: relevant (older context) first, then recent (continuity).
    // Sort by id so the sequence is always chronological when injected.
    let mut combined = relevant;
    combined.extend(recent);
    combined.sort_by_key(|e| e.id);
    Ok(combined)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn row_to_entry(row: &rusqlite::Row<'_>) -> rusqlite::Result<DbMemoryEntry> {
    Ok(DbMemoryEntry {
        id: row.get(0)?,
        timestamp: row.get(1)?,
        session_id: row.get(2)?,
        role: row.get(3)?,
        content: row.get(4)?,
        summary: row.get(5)?,
    })
}

/// Wrap `s` as a safe FTS5 phrase query.
///
/// Encloses the string in double-quotes and escapes any embedded `"` as `""`.
/// This prevents FTS5 syntax errors when the user's text contains operators
/// like `OR`, `AND`, `-`, `*`, or stray quotes.
fn escape_fts5_query(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn open_test_db() -> (Connection, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("memory.db");
        let conn = init_db(&path).unwrap();
        (conn, dir)
    }

    #[test]
    fn test_store_and_get_recent() {
        let (conn, _dir) = open_test_db();
        store_memory(&conn, "2026-01-01T00:00:00Z", "s1", "user", "hello", "").unwrap();
        store_memory(
            &conn,
            "2026-01-01T00:00:01Z",
            "s1",
            "assistant",
            "world",
            "",
        )
        .unwrap();

        let entries = get_recent(&conn, 10).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].role, "user");
        assert_eq!(entries[0].content, "hello");
        assert_eq!(entries[1].role, "assistant");
        assert_eq!(entries[1].content, "world");
    }

    #[test]
    fn test_get_recent_limit_and_order() {
        let (conn, _dir) = open_test_db();
        for i in 0..10 {
            store_memory(
                &conn,
                "2026-01-01T00:00:00Z",
                "",
                "user",
                &format!("msg {i}"),
                "",
            )
            .unwrap();
        }
        let entries = get_recent(&conn, 4).unwrap();
        assert_eq!(entries.len(), 4);
        // Most recent 4 in chronological order: msg 6, 7, 8, 9
        assert_eq!(entries[3].content, "msg 9");
        assert_eq!(entries[0].content, "msg 6");
    }

    #[test]
    fn test_fts_search_relevant() {
        let (conn, _dir) = open_test_db();
        store_memory(
            &conn,
            "2026-01-01T00:00:00Z",
            "",
            "user",
            "how to install rust toolchain",
            "",
        )
        .unwrap();
        store_memory(
            &conn,
            "2026-01-01T00:00:01Z",
            "",
            "assistant",
            "use rustup installer",
            "",
        )
        .unwrap();
        store_memory(
            &conn,
            "2026-01-02T00:00:00Z",
            "",
            "user",
            "what is python used for",
            "",
        )
        .unwrap();

        let results = search_relevant(&conn, "rust toolchain", 10).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|e| e.content.contains("rust")));
        // "python" entry should not match
        assert!(!results.iter().any(|e| e.content.contains("python")));
    }

    #[test]
    fn test_fts_empty_query_returns_empty() {
        let (conn, _dir) = open_test_db();
        store_memory(&conn, "2026-01-01T00:00:00Z", "", "user", "hello", "").unwrap();
        let results = search_relevant(&conn, "", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_fts_special_chars_do_not_panic() {
        let (conn, _dir) = open_test_db();
        store_memory(&conn, "2026-01-01T00:00:00Z", "", "user", "some text", "").unwrap();
        // These would normally break FTS5 syntax.
        let r1 = search_relevant(&conn, "OR AND NOT", 5);
        let r2 = search_relevant(&conn, "\"unclosed quote", 5);
        let r3 = search_relevant(&conn, "*wildcard*", 5);
        // All should succeed (possibly returning empty, not panicking).
        assert!(r1.is_ok());
        assert!(r2.is_ok());
        assert!(r3.is_ok());
    }

    #[test]
    fn test_count_and_clear() {
        let (conn, _dir) = open_test_db();
        assert_eq!(count(&conn).unwrap(), 0);
        store_memory(&conn, "2026-01-01T00:00:00Z", "", "user", "a", "").unwrap();
        store_memory(&conn, "2026-01-01T00:00:01Z", "", "assistant", "b", "").unwrap();
        assert_eq!(count(&conn).unwrap(), 2);
        clear(&conn).unwrap();
        assert_eq!(count(&conn).unwrap(), 0);
    }

    #[test]
    fn test_full_text_preserved() {
        let (conn, _dir) = open_test_db();
        let long_text = "a".repeat(5000);
        store_memory(&conn, "2026-01-01T00:00:00Z", "", "user", &long_text, "").unwrap();
        let entries = get_recent(&conn, 1).unwrap();
        assert_eq!(
            entries[0].content, long_text,
            "full content must be stored without truncation"
        );
    }

    #[test]
    fn test_retrieve_context_deduplicates() {
        let (conn, _dir) = open_test_db();
        // Insert enough rows so recent and search windows overlap.
        for i in 0..6 {
            store_memory(
                &conn,
                "2026-01-01T00:00:00Z",
                "",
                "user",
                &format!("rust cargo question {i}"),
                "",
            )
            .unwrap();
        }
        let ctx = retrieve_context(&conn, "rust cargo", 4, 10).unwrap();
        // No duplicate ids.
        let ids: HashSet<i64> = ctx.iter().map(|e| e.id).collect();
        assert_eq!(
            ids.len(),
            ctx.len(),
            "context must not contain duplicate entries"
        );
    }

    #[test]
    fn test_escape_fts5_query() {
        assert_eq!(escape_fts5_query("hello world"), "\"hello world\"");
        assert_eq!(escape_fts5_query("say \"hi\""), "\"say \"\"hi\"\"\"");
        assert_eq!(escape_fts5_query(""), "\"\"");
    }
}
