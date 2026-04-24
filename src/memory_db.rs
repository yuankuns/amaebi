//! SQLite-backed memory storage with FTS5 full-text search and session compaction.
//!
//! Two tables are maintained in `~/.amaebi/memory.db`:
//!
//! * **`memories`** — one row per conversation turn (user or assistant message),
//!   indexed by FTS5 for semantic search.  The daemon loads full history and trims
//!   it to a token budget, keeping a "hot tail" of recent turns verbatim.
//!
//! * **`session_summaries`** — one compact LLM-generated summary per session UUID,
//!   written lazily by `compact_session` in two cases:
//!   - **Cross-session**: when a new session starts, old sessions without a summary
//!     are compacted so future sessions can learn from them.
//!   - **Within-session**: when a session's token budget is exhausted, older turns
//!     are summarised while the recent "hot tail" is kept verbatim.
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
    /// Groups messages from the same conversation.
    #[allow(dead_code)]
    pub session_id: String,
    /// `"user"` or `"assistant"`.
    pub role: String,
    pub content: String,
    /// Reserved column in the `memories` table.  Per-session compacted summaries
    /// are stored in the separate `session_summaries` table, not here.
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
    role       TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT    NOT NULL,
    summary    TEXT    NOT NULL DEFAULT '',
    archived   INTEGER NOT NULL DEFAULT 0
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

-- One compacted summary per session.  Written lazily by compact_session when:
-- (a) a new session starts in the same folder (cross-session learning), or
-- (b) the session history grows beyond the sliding-window cap (within-session).
-- Provides cross-session context without injecting full raw history.
CREATE TABLE IF NOT EXISTS session_summaries (
    session_id TEXT PRIMARY KEY,
    summary    TEXT NOT NULL,
    timestamp  TEXT NOT NULL
);
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
               AND m.archived = 0
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
             WHERE archived = 0
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

/// Return the most-recent entry for each distinct `session_id`, newest first.
///
/// Used by the dashboard to show one "latest activity" line per session
/// without being swamped by intra-session chatter.  Honors `archived = 0`.
pub fn get_latest_per_session(conn: &Connection, limit: usize) -> Result<Vec<DbMemoryEntry>> {
    let mut stmt = conn
        .prepare(
            "SELECT id, timestamp, session_id, role, content, summary
             FROM memories
             WHERE archived = 0 AND id IN (
                 SELECT MAX(id) FROM memories WHERE archived = 0 GROUP BY session_id
             )
             ORDER BY id DESC
             LIMIT ?1",
        )
        .context("preparing get_latest_per_session query")?;

    let entries = stmt
        .query_map(params![limit as i64], row_to_entry)
        .context("executing get_latest_per_session")?
        .collect::<rusqlite::Result<Vec<_>>>()
        .context("collecting get_latest_per_session results")?;
    Ok(entries)
}

/// Return all messages for a given `session_id` in chronological order.
///
/// Used by `--resume` to reload full conversation history from the DB
/// without FTS5 filtering.  Returns an empty list when no rows match.
pub fn get_session_history(conn: &Connection, session_id: &str) -> Result<Vec<DbMemoryEntry>> {
    let mut stmt = conn
        .prepare(
            "SELECT id, timestamp, session_id, role, content, summary
             FROM memories
             WHERE session_id = ?1 AND archived = 0
             ORDER BY id ASC",
        )
        .context("preparing get_session_history query")?;

    let entries = stmt
        .query_map(params![session_id], row_to_entry)
        .context("executing get_session_history")?
        .collect::<rusqlite::Result<Vec<_>>>()
        .context("collecting session history results")?;
    Ok(entries)
}

/// Return the oldest `limit` messages for a given `session_id` in chronological order.
///
/// Used by `compact_session` to summarise only the turns that the sliding window
/// is dropping, so the summary and the raw history window do not overlap.
pub fn get_session_oldest(
    conn: &Connection,
    session_id: &str,
    limit: usize,
) -> Result<Vec<DbMemoryEntry>> {
    if limit == 0 {
        return Ok(vec![]);
    }
    let mut stmt = conn
        .prepare(
            "SELECT id, timestamp, session_id, role, content, summary
             FROM memories
             WHERE session_id = ?1 AND archived = 0
             ORDER BY id ASC
             LIMIT ?2",
        )
        .context("preparing get_session_oldest query")?;

    let entries = stmt
        .query_map(params![session_id, limit as i64], row_to_entry)
        .context("executing get_session_oldest")?
        .collect::<rusqlite::Result<Vec<_>>>()
        .context("collecting session oldest results")?;

    Ok(entries) // already in chronological (ASC) order
}

/// Return the most recent `limit` messages for a given `session_id` in
/// chronological order.  Only used in tests.
#[cfg(test)]
pub fn get_session_recent(
    conn: &Connection,
    session_id: &str,
    limit: usize,
) -> Result<Vec<DbMemoryEntry>> {
    let mut stmt = conn
        .prepare(
            "SELECT id, timestamp, session_id, role, content, summary
             FROM memories
             WHERE session_id = ?1
             ORDER BY id DESC
             LIMIT ?2",
        )
        .context("preparing get_session_recent query")?;

    let mut entries = stmt
        .query_map(params![session_id, limit as i64], row_to_entry)
        .context("executing get_session_recent")?
        .collect::<rusqlite::Result<Vec<_>>>()
        .context("collecting session recent results")?;

    entries.reverse(); // chronological order
    Ok(entries)
}

/// Return the total number of rows in the `memories` table.
pub fn count(conn: &Connection) -> Result<usize> {
    conn.query_row("SELECT COUNT(*) FROM memories", [], |r| r.get::<_, i64>(0))
        .context("counting memories")
        .map(|n| n as usize)
}

/// Delete all rows from `memories` (and rebuild the FTS index) and all session summaries.
pub fn clear(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "DELETE FROM memories;
         INSERT INTO memories_fts(memories_fts) VALUES ('rebuild');
         DELETE FROM session_summaries;",
    )
    .context("clearing memories")
}

/// Upsert a compacted summary for `session_id`.
///
/// Called by `compact_session` in the daemon.  `timestamp` should be an RFC 3339 UTC string.
pub fn store_session_summary(
    conn: &Connection,
    session_id: &str,
    summary: &str,
    timestamp: &str,
) -> Result<()> {
    conn.execute(
        "INSERT INTO session_summaries (session_id, summary, timestamp)
         VALUES (?1, ?2, ?3)
         ON CONFLICT(session_id) DO UPDATE SET
             summary   = excluded.summary,
             timestamp = excluded.timestamp",
        params![session_id, summary, timestamp],
    )
    .context("storing session summary")?;
    Ok(())
}

/// Return the total number of message rows for `session_id`.
///
/// Used to decide whether the session history is long enough to warrant
/// within-session compaction (i.e., the sliding window is dropping turns).
pub fn count_session_turns(conn: &Connection, session_id: &str) -> Result<usize> {
    conn.query_row(
        "SELECT COUNT(*) FROM memories WHERE session_id = ?1 AND archived = 0",
        params![session_id],
        |r| r.get::<_, i64>(0),
    )
    .context("counting session turns")
    .map(|n| n as usize)
}

/// Mark a set of memory rows as archived so they are excluded from future history loads.
///
/// Called by `compact_session` after a summary is successfully stored.
/// Archived turns are kept in the DB for audit purposes but never re-loaded
/// into the context window or re-compacted.
pub fn archive_session_turns(conn: &Connection, ids: &[i64]) -> Result<()> {
    if ids.is_empty() {
        return Ok(());
    }
    // SQLite has a hard limit of 999 bound parameters per statement.
    // Process the ids in chunks to stay within that limit.
    const CHUNK_SIZE: usize = 999;
    for chunk in ids.chunks(CHUNK_SIZE) {
        let placeholders = chunk
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!("UPDATE memories SET archived = 1 WHERE id IN ({placeholders})");
        let mut stmt = conn
            .prepare(&sql)
            .context("preparing archive_session_turns")?;
        stmt.execute(rusqlite::params_from_iter(chunk))
            .context("archiving session turns")?;
    }
    Ok(())
}

/// Return the compacted summary for `session_id` if one exists, otherwise `None`.
///
/// Used to inject an ongoing session's own running summary before its hot-tail
/// turns when the full history no longer fits the token budget.
pub fn get_session_own_summary(conn: &Connection, session_id: &str) -> Result<Option<String>> {
    let mut stmt = conn
        .prepare("SELECT summary FROM session_summaries WHERE session_id = ?1 LIMIT 1")
        .context("preparing get_session_own_summary")?;
    let mut rows = stmt
        .query(params![session_id])
        .context("executing get_session_own_summary")?;
    if let Some(row) = rows.next().context("reading session summary row")? {
        Ok(Some(row.get(0).context("reading summary column")?))
    } else {
        Ok(None)
    }
}

/// Return session IDs that have conversation history but no compacted summary yet,
/// excluding `exclude_session`.  Used to find old sessions to compact when a new
/// session starts.
pub fn get_sessions_without_summary(
    conn: &Connection,
    exclude_session: &str,
    limit: usize,
) -> Result<Vec<String>> {
    let mut stmt = conn
        .prepare(
            "SELECT session_id FROM (
                 SELECT m.session_id, MIN(m.id) AS first_id
                 FROM memories m
                 WHERE m.session_id != ?1
                   AND m.session_id != ''
                   AND m.session_id NOT IN (SELECT session_id FROM session_summaries)
                 GROUP BY m.session_id
             ) ORDER BY first_id ASC
             LIMIT ?2",
        )
        .context("preparing get_sessions_without_summary")?;

    let sessions = stmt
        .query_map(params![exclude_session, limit as i64], |row| {
            row.get::<_, String>(0)
        })
        .context("executing get_sessions_without_summary")?
        .collect::<rusqlite::Result<Vec<_>>>()
        .context("collecting uncompacted sessions")?;

    Ok(sessions)
}

/// Return up to `limit` summaries from sessions other than `exclude_session`,
/// ordered oldest-first so the caller can inject them chronologically.
///
/// Excludes the current session so the model does not see a stale summary of
/// the conversation it is actively participating in.
pub fn get_recent_summaries(
    conn: &Connection,
    exclude_session: &str,
    limit: usize,
) -> Result<Vec<String>> {
    // Fetch most-recent first, then reverse so injection is chronological.
    let mut stmt = conn
        .prepare(
            "SELECT summary FROM session_summaries
             WHERE session_id != ?1
             ORDER BY timestamp DESC
             LIMIT ?2",
        )
        .context("preparing get_recent_summaries")?;

    let mut summaries = stmt
        .query_map(params![exclude_session, limit as i64], |row| {
            row.get::<_, String>(0)
        })
        .context("executing get_recent_summaries")?
        .collect::<rusqlite::Result<Vec<_>>>()
        .context("collecting summaries")?;

    summaries.reverse(); // oldest first → model reads history in order
    Ok(summaries)
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

    #[test]
    fn test_get_session_history() {
        let (conn, _dir) = open_test_db();
        store_memory(&conn, "2026-01-01T00:00:00Z", "sess-1", "user", "hello", "").unwrap();
        store_memory(
            &conn,
            "2026-01-01T00:00:01Z",
            "sess-1",
            "assistant",
            "hi",
            "",
        )
        .unwrap();
        store_memory(&conn, "2026-01-01T00:00:02Z", "sess-2", "user", "other", "").unwrap();

        let history = get_session_history(&conn, "sess-1").unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].content, "hello");
        assert_eq!(history[1].content, "hi");

        let other = get_session_history(&conn, "sess-2").unwrap();
        assert_eq!(other.len(), 1);

        let empty = get_session_history(&conn, "nonexistent").unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_get_session_recent() {
        let (conn, _dir) = open_test_db();
        for i in 0..10 {
            store_memory(
                &conn,
                "2026-01-01T00:00:00Z",
                "s1",
                "user",
                &format!("msg {i}"),
                "",
            )
            .unwrap();
        }
        let recent = get_session_recent(&conn, "s1", 4).unwrap();
        assert_eq!(recent.len(), 4);
        assert_eq!(recent[0].content, "msg 6");
        assert_eq!(recent[3].content, "msg 9");
    }

    #[test]
    fn store_session_summary_upserts() {
        let (conn, _dir) = open_test_db();

        store_session_summary(&conn, "s1", "first summary", "2026-01-01T00:00:00Z").unwrap();
        let s1 = get_session_own_summary(&conn, "s1").unwrap();
        assert_eq!(s1.as_deref(), Some("first summary"));

        // Upsert — should update, not create a second row.
        store_session_summary(&conn, "s1", "updated summary", "2026-01-02T00:00:00Z").unwrap();
        let s2 = get_session_own_summary(&conn, "s1").unwrap();
        assert_eq!(s2.as_deref(), Some("updated summary"));

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM session_summaries WHERE session_id = 's1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1, "upsert must not create duplicate rows");
    }

    #[test]
    fn get_session_own_summary_returns_none_when_absent() {
        let (conn, _dir) = open_test_db();
        let result = get_session_own_summary(&conn, "no-such-session").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn get_sessions_without_summary_excludes_current_and_summarised() {
        let (conn, _dir) = open_test_db();
        store_memory(&conn, "2026-01-01T00:00:00Z", "s1", "user", "hi", "").unwrap();
        store_memory(&conn, "2026-01-01T00:00:00Z", "s2", "user", "hi", "").unwrap();
        store_memory(&conn, "2026-01-01T00:00:00Z", "s3", "user", "hi", "").unwrap();
        store_session_summary(&conn, "s2", "summary for s2", "2026-01-01T00:00:00Z").unwrap();

        // s1 has no summary; s2 has a summary; s3 is the "current" session (excluded).
        let unsummarised = get_sessions_without_summary(&conn, "s3", 10).unwrap();
        assert_eq!(unsummarised, vec!["s1"]);
    }

    #[test]
    fn get_recent_summaries_excludes_current_and_orders_oldest_first() {
        let (conn, _dir) = open_test_db();
        store_session_summary(&conn, "s1", "summary A", "2026-01-01T00:00:00Z").unwrap();
        store_session_summary(&conn, "s2", "summary B", "2026-01-03T00:00:00Z").unwrap();
        store_session_summary(&conn, "s3", "summary C", "2026-01-02T00:00:00Z").unwrap();

        // Exclude s3 (current session); expect oldest-first order.
        let summaries = get_recent_summaries(&conn, "s3", 10).unwrap();
        assert_eq!(summaries, vec!["summary A", "summary B"]);
    }

    #[test]
    fn get_latest_per_session_returns_one_row_per_session_newest_first() {
        let (conn, _dir) = open_test_db();
        // s1: two messages; we expect the later one (id 2).
        store_memory(&conn, "2026-01-01T00:00:00Z", "s1", "user", "hi s1", "").unwrap();
        store_memory(
            &conn,
            "2026-01-01T00:00:01Z",
            "s1",
            "assistant",
            "s1 reply",
            "",
        )
        .unwrap();
        // s2: single message (id 3).
        store_memory(&conn, "2026-01-01T00:00:02Z", "s2", "user", "hi s2", "").unwrap();
        // s3: single message (id 4) but archive it — must be excluded.
        store_memory(&conn, "2026-01-01T00:00:03Z", "s3", "user", "archived", "").unwrap();
        conn.execute(
            "UPDATE memories SET archived = 1 WHERE session_id = 's3'",
            [],
        )
        .unwrap();

        let rows = get_latest_per_session(&conn, 10).unwrap();
        // Expect: s2 (id 3) then s1 (id 2). s3 is archived → excluded.
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].session_id, "s2");
        assert_eq!(rows[0].content, "hi s2");
        assert_eq!(rows[1].session_id, "s1");
        assert_eq!(rows[1].content, "s1 reply");
    }

    #[test]
    fn get_latest_per_session_respects_limit() {
        let (conn, _dir) = open_test_db();
        store_memory(&conn, "2026-01-01T00:00:00Z", "s1", "user", "a", "").unwrap();
        store_memory(&conn, "2026-01-01T00:00:01Z", "s2", "user", "b", "").unwrap();
        store_memory(&conn, "2026-01-01T00:00:02Z", "s3", "user", "c", "").unwrap();
        let rows = get_latest_per_session(&conn, 2).unwrap();
        assert_eq!(rows.len(), 2);
        // Newest first → s3, s2.
        assert_eq!(rows[0].session_id, "s3");
        assert_eq!(rows[1].session_id, "s2");
    }
}
