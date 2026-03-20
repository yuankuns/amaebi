//! Cron job store and scheduler.
//!
//! Jobs are persisted in `~/.amaebi/cron.db` (SQLite).  The daemon ticks
//! every minute and fires any jobs whose schedule matches the current UTC
//! wall-clock time.  Results land in the inbox via [`crate::inbox::InboxStore`].
//!
//! # Schema
//!
//! ```sql
//! CREATE TABLE IF NOT EXISTS cron_jobs (
//!     id         TEXT PRIMARY KEY,
//!     description TEXT NOT NULL,
//!     schedule   TEXT NOT NULL,
//!     created_at TEXT NOT NULL,
//!     last_run   TEXT
//! );
//! ```
//!
//! # Concurrency
//!
//! WAL mode is enabled so CLI reads do not block daemon writes.
//! A 5-second busy timeout prevents immediate failure when the daemon and
//! CLI collide on the same row.

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use crate::auth::amaebi_home;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A registered cron job.
#[derive(Debug, Clone)]
pub struct CronJob {
    /// UUID v4 identifier.
    pub id: String,
    /// Human-readable task description, used as the LLM prompt.
    pub description: String,
    /// 5-field cron expression (`min hour dom mon dow`).
    pub schedule: String,
    /// RFC 3339 creation timestamp.
    pub created_at: String,
    /// RFC 3339 timestamp of the last successful run, or `None`.
    pub last_run: Option<String>,
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// SQLite-backed store for cron jobs.
///
/// The connection is opened fresh on each operation so the store is cheap to
/// construct and holds no file descriptor when idle.
pub struct CronStore {
    db_path: PathBuf,
}

impl CronStore {
    /// Open (or create) the cron database at `~/.amaebi/cron.db`.
    pub fn open() -> Result<Self> {
        let db_path = amaebi_home()?.join("cron.db");
        let store = Self { db_path };
        store.init()?;
        Ok(store)
    }

    /// Open the cron database at an explicit path (used in tests).
    #[cfg(test)]
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
            .with_context(|| format!("opening cron.db at {}", self.db_path.display()))?;

        // Enforce 0600 permissions (may contain sensitive task descriptions).
        std::fs::set_permissions(&self.db_path, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("setting permissions on {}", self.db_path.display()))?;

        conn.execute_batch("PRAGMA journal_mode=WAL;")
            .context("enabling WAL mode")?;
        conn.busy_timeout(std::time::Duration::from_secs(5))
            .context("setting busy timeout")?;

        Ok(conn)
    }

    fn init(&self) -> Result<()> {
        let conn = self.connect()?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS cron_jobs (
                id          TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                schedule    TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                last_run    TEXT
            );",
        )
        .context("creating cron_jobs table")?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Read operations
    // -----------------------------------------------------------------------

    /// Return all registered cron jobs, ordered by creation time.
    pub fn list(&self) -> Result<Vec<CronJob>> {
        let conn = self.connect()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, description, schedule, created_at, last_run
                 FROM cron_jobs
                 ORDER BY created_at ASC",
            )
            .context("preparing cron list query")?;

        let rows = stmt
            .query_map([], |row| {
                Ok(CronJob {
                    id: row.get(0)?,
                    description: row.get(1)?,
                    schedule: row.get(2)?,
                    created_at: row.get(3)?,
                    last_run: row.get(4)?,
                })
            })
            .context("executing cron list query")?;

        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("collecting cron job rows")
    }

    // -----------------------------------------------------------------------
    // Write operations
    // -----------------------------------------------------------------------

    /// Register a new cron job.  Returns the new job's UUID.
    ///
    /// The schedule is validated before writing so an unparseable expression
    /// is never persisted.
    pub fn add(&self, description: &str, schedule: &str) -> Result<String> {
        parse_schedule(schedule).with_context(|| format!("invalid cron schedule: {schedule:?}"))?;

        let id = uuid::Uuid::new_v4().to_string();
        let created_at = chrono::Utc::now().to_rfc3339();
        let conn = self.connect()?;
        conn.execute(
            "INSERT INTO cron_jobs (id, description, schedule, created_at, last_run)
             VALUES (?1, ?2, ?3, ?4, NULL)",
            params![id, description, schedule, created_at],
        )
        .context("inserting cron job")?;
        Ok(id)
    }

    /// Remove a job by ID.  Returns `true` if a row was deleted.
    pub fn delete(&self, id: &str) -> Result<bool> {
        let conn = self.connect()?;
        let n = conn
            .execute("DELETE FROM cron_jobs WHERE id = ?1", params![id])
            .context("deleting cron job")?;
        Ok(n > 0)
    }

    /// Record that job `id` ran at `timestamp` (RFC 3339).
    pub fn update_last_run(&self, id: &str, timestamp: &str) -> Result<()> {
        let conn = self.connect()?;
        conn.execute(
            "UPDATE cron_jobs SET last_run = ?1 WHERE id = ?2",
            params![timestamp, id],
        )
        .context("updating cron job last_run")?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Free-function shims used by daemon.rs and main.rs
// ---------------------------------------------------------------------------
//
// These thin wrappers keep daemon.rs and main.rs callers unchanged while the
// storage backend is fully SQLite.

/// Load all cron jobs from the store.
pub fn load_jobs() -> Result<Vec<CronJob>> {
    CronStore::open()?.list()
}

/// Register a new cron job; returns the UUID.
pub fn add_job(description: &str, schedule: &str) -> Result<String> {
    CronStore::open()?.add(description, schedule)
}

/// Remove a job by ID; returns `true` if found and deleted.
pub fn delete_job(id: &str) -> Result<bool> {
    CronStore::open()?.delete(id)
}

/// Update the `last_run` timestamp for a job.
pub fn update_last_run(id: &str, timestamp: &str) -> Result<()> {
    CronStore::open()?.update_last_run(id, timestamp)
}

// ---------------------------------------------------------------------------
// Schedule parsing
// ---------------------------------------------------------------------------

/// Parsed 5-field cron expression.
#[derive(Debug)]
pub struct Schedule {
    pub minute: FieldSpec,
    pub hour: FieldSpec,
    pub dom: FieldSpec,
    pub month: FieldSpec,
    pub dow: FieldSpec,
}

/// A single field's allowed values.
#[derive(Debug)]
pub enum FieldSpec {
    /// Wildcard — matches every value in the field's range.
    Any,
    /// Explicit set of allowed values.
    Values(Vec<u32>),
}

impl FieldSpec {
    pub fn matches(&self, value: u32) -> bool {
        match self {
            FieldSpec::Any => true,
            FieldSpec::Values(vals) => vals.contains(&value),
        }
    }
}

/// Parse a 5-field cron expression (`min hour dom mon dow`).
///
/// Each field supports:
/// - `*`       — any value
/// - `n`       — exact value
/// - `n-m`     — inclusive range
/// - `*/n`     — every n-th value across the full range
/// - `a,b,...` — comma-separated list of any of the above
pub fn parse_schedule(expr: &str) -> Result<Schedule> {
    let parts: Vec<&str> = expr.split_whitespace().collect();
    anyhow::ensure!(
        parts.len() == 5,
        "expected 5 fields (min hour dom mon dow), got {}: {:?}",
        parts.len(),
        expr
    );
    Ok(Schedule {
        minute: parse_field(parts[0], 0, 59)?,
        hour: parse_field(parts[1], 0, 23)?,
        dom: parse_field(parts[2], 1, 31)?,
        month: parse_field(parts[3], 1, 12)?,
        dow: parse_field(parts[4], 0, 6)?,
    })
}

fn parse_field(s: &str, min: u32, max: u32) -> Result<FieldSpec> {
    if s == "*" {
        return Ok(FieldSpec::Any);
    }
    let mut values = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part == "*" {
            return Ok(FieldSpec::Any);
        } else if let Some(step_str) = part.strip_prefix("*/") {
            let step: u32 = step_str
                .parse()
                .with_context(|| format!("invalid step in {part:?}"))?;
            anyhow::ensure!(step > 0, "step must be > 0 in {part:?}");
            let mut v = min;
            while v <= max {
                values.push(v);
                v = v.saturating_add(step);
            }
        } else if let Some((lo_s, hi_s)) = part.split_once('-') {
            let lo: u32 = lo_s
                .parse()
                .with_context(|| format!("invalid range start in {part:?}"))?;
            let hi: u32 = hi_s
                .parse()
                .with_context(|| format!("invalid range end in {part:?}"))?;
            anyhow::ensure!(lo <= hi, "range start > end in {part:?}");
            anyhow::ensure!(
                lo >= min && hi <= max,
                "range {lo}-{hi} out of bounds [{min},{max}] in {part:?}"
            );
            values.extend(lo..=hi);
        } else {
            let n: u32 = part
                .parse()
                .with_context(|| format!("invalid number {part:?}"))?;
            anyhow::ensure!(
                n >= min && n <= max,
                "value {n} out of bounds [{min},{max}]"
            );
            values.push(n);
        }
    }
    Ok(FieldSpec::Values(values))
}

// ---------------------------------------------------------------------------
// Due-job detection
// ---------------------------------------------------------------------------

/// Return `true` if `schedule` fires at the given UTC datetime.
///
/// All 5 fields must match simultaneously.  The day-of-week mapping follows
/// Unix cron convention: 0 = Sunday, 1 = Monday, …, 6 = Saturday.
pub fn is_due(schedule: &Schedule, dt: &chrono::DateTime<chrono::Utc>) -> bool {
    use chrono::{Datelike as _, Timelike as _};
    schedule.minute.matches(dt.minute())
        && schedule.hour.matches(dt.hour())
        && schedule.dom.matches(dt.day())
        && schedule.month.matches(dt.month())
        && schedule.dow.matches(dt.weekday().num_days_from_sunday())
}

/// Filter `jobs` to those that are due to run at `now`.
///
/// A job is skipped (even if the schedule matches) when its `last_run`
/// timestamp is within the last 30 seconds — this prevents double-firing
/// if the daemon tick runs slightly within the same wall-clock minute.
pub fn due_jobs(jobs: &[CronJob], now: &chrono::DateTime<chrono::Utc>) -> Vec<CronJob> {
    let mut due = Vec::new();
    for job in jobs {
        let sched = match parse_schedule(&job.schedule) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    id = %job.id,
                    schedule = %job.schedule,
                    error = %e,
                    "skipping cron job with invalid schedule"
                );
                continue;
            }
        };
        if !is_due(&sched, now) {
            continue;
        }
        // Guard: don't re-fire within the same minute.
        if let Some(last) = &job.last_run {
            if let Ok(last_dt) = chrono::DateTime::parse_from_rfc3339(last) {
                let elapsed = now.signed_duration_since(last_dt.with_timezone(&chrono::Utc));
                if elapsed < chrono::Duration::seconds(30) {
                    continue;
                }
            }
        }
        due.push(job.clone());
    }
    due
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone as _;

    fn utc(year: i32, month: u32, day: u32, hour: u32, min: u32) -> chrono::DateTime<chrono::Utc> {
        chrono::Utc
            .with_ymd_and_hms(year, month, day, hour, min, 0)
            .unwrap()
    }

    // ---- parse_schedule ---------------------------------------------------

    #[test]
    fn parse_all_stars_returns_any_fields() {
        let s = parse_schedule("* * * * *").unwrap();
        assert!(matches!(s.minute, FieldSpec::Any));
        assert!(matches!(s.hour, FieldSpec::Any));
        assert!(matches!(s.dom, FieldSpec::Any));
        assert!(matches!(s.month, FieldSpec::Any));
        assert!(matches!(s.dow, FieldSpec::Any));
    }

    #[test]
    fn parse_specific_values_single_field() {
        let s = parse_schedule("30 9 15 3 1").unwrap();
        let FieldSpec::Values(ref mins) = s.minute else {
            panic!("expected Values for minute")
        };
        assert_eq!(mins, &[30]);
        let FieldSpec::Values(ref hours) = s.hour else {
            panic!("expected Values for hour")
        };
        assert_eq!(hours, &[9]);
    }

    #[test]
    fn parse_step_every_15_minutes() {
        let s = parse_schedule("*/15 * * * *").unwrap();
        let FieldSpec::Values(ref mins) = s.minute else {
            panic!("expected Values")
        };
        assert_eq!(mins, &[0, 15, 30, 45]);
    }

    #[test]
    fn parse_range_9_to_17() {
        let s = parse_schedule("0 9-17 * * *").unwrap();
        let FieldSpec::Values(ref hours) = s.hour else {
            panic!("expected Values")
        };
        assert_eq!(hours, &(9u32..=17).collect::<Vec<_>>());
    }

    #[test]
    fn parse_comma_list_dom() {
        let s = parse_schedule("0 0 1,15 * *").unwrap();
        let FieldSpec::Values(ref doms) = s.dom else {
            panic!("expected Values")
        };
        assert_eq!(doms, &[1, 15]);
    }

    #[test]
    fn parse_wrong_field_count_errors() {
        assert!(parse_schedule("* * * *").is_err(), "4 fields should error");
        assert!(
            parse_schedule("* * * * * *").is_err(),
            "6 fields should error"
        );
    }

    #[test]
    fn parse_out_of_range_minute_errors() {
        assert!(parse_schedule("60 * * * *").is_err());
    }

    #[test]
    fn parse_out_of_range_hour_errors() {
        assert!(parse_schedule("* 24 * * *").is_err());
    }

    #[test]
    fn parse_invalid_step_zero_errors() {
        assert!(parse_schedule("*/0 * * * *").is_err());
    }

    #[test]
    fn parse_inverted_range_errors() {
        assert!(parse_schedule("0 17-9 * * *").is_err());
    }

    // ---- is_due -----------------------------------------------------------

    #[test]
    fn is_due_all_stars_always_matches() {
        let s = parse_schedule("* * * * *").unwrap();
        assert!(is_due(&s, &chrono::Utc::now()));
    }

    #[test]
    fn is_due_specific_time_matches_exactly() {
        let s = parse_schedule("30 9 * * *").unwrap();
        let yes = utc(2026, 3, 20, 9, 30);
        let no = utc(2026, 3, 20, 9, 31);
        assert!(is_due(&s, &yes));
        assert!(!is_due(&s, &no));
    }

    #[test]
    fn is_due_step_every_15_minutes() {
        let s = parse_schedule("*/15 * * * *").unwrap();
        for min in [0u32, 15, 30, 45] {
            let dt = utc(2026, 3, 20, 12, min);
            assert!(is_due(&s, &dt), "minute {min} should be due");
        }
        for min in [1u32, 14, 16, 29, 31, 44, 46] {
            let dt = utc(2026, 3, 20, 12, min);
            assert!(!is_due(&s, &dt), "minute {min} should not be due");
        }
    }

    #[test]
    fn is_due_day_of_week_matches() {
        // 2026-03-20 is a Friday (dow=5 in chrono, which is also 5 in 0=Sun encoding).
        let s = parse_schedule("0 0 * * 5").unwrap(); // every Friday at midnight
        let fri = utc(2026, 3, 20, 0, 0);
        let sat = utc(2026, 3, 21, 0, 0);
        assert!(is_due(&s, &fri), "Friday midnight should match");
        assert!(!is_due(&s, &sat), "Saturday should not match");
    }

    // ---- due_jobs ---------------------------------------------------------

    #[test]
    fn due_jobs_skips_recently_run() {
        let now = utc(2026, 3, 20, 9, 30);
        let just_ran = (now - chrono::Duration::seconds(10)).to_rfc3339();
        let long_ago = (now - chrono::Duration::minutes(10)).to_rfc3339();

        let jobs = vec![
            CronJob {
                id: "a".into(),
                description: "task A".into(),
                schedule: "30 9 * * *".into(),
                created_at: "2026-01-01T00:00:00Z".into(),
                last_run: Some(just_ran),
            },
            CronJob {
                id: "b".into(),
                description: "task B".into(),
                schedule: "30 9 * * *".into(),
                created_at: "2026-01-01T00:00:00Z".into(),
                last_run: Some(long_ago),
            },
            CronJob {
                id: "c".into(),
                description: "task C".into(),
                schedule: "30 9 * * *".into(),
                created_at: "2026-01-01T00:00:00Z".into(),
                last_run: None,
            },
        ];

        let due = due_jobs(&jobs, &now);
        let ids: Vec<&str> = due.iter().map(|j| j.id.as_str()).collect();
        assert!(!ids.contains(&"a"), "recently-run job must be skipped");
        assert!(ids.contains(&"b"), "old job must be due");
        assert!(ids.contains(&"c"), "never-run job must be due");
    }

    #[test]
    fn due_jobs_skips_non_matching_schedule() {
        let now = utc(2026, 3, 20, 9, 31); // 9:31, not 9:30
        let jobs = vec![CronJob {
            id: "a".into(),
            description: "task".into(),
            schedule: "30 9 * * *".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            last_run: None,
        }];
        assert!(due_jobs(&jobs, &now).is_empty());
    }

    #[test]
    fn due_jobs_skips_invalid_schedule_gracefully() {
        let now = utc(2026, 3, 20, 9, 30);
        let jobs = vec![CronJob {
            id: "bad".into(),
            description: "broken".into(),
            schedule: "not a cron expression".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            last_run: None,
        }];
        // Must not panic; bad job is simply skipped.
        assert!(due_jobs(&jobs, &now).is_empty());
    }

    // ---- CronStore SQLite roundtrip ---------------------------------------

    fn open_temp() -> (CronStore, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("cron.db");
        let store = CronStore::open_at(path).unwrap();
        // Return dir alongside the store so the caller keeps it alive.
        (store, dir)
    }

    #[test]
    fn store_add_and_list_roundtrip() {
        let (store, _dir) = open_temp();
        let id = store.add("daily report", "0 9 * * *").unwrap();
        let jobs = store.list().unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].id, id);
        assert_eq!(jobs[0].description, "daily report");
        assert_eq!(jobs[0].schedule, "0 9 * * *");
        assert!(jobs[0].last_run.is_none());
    }

    #[test]
    fn store_multiple_jobs_ordered_by_created_at() {
        let (store, _dir) = open_temp();
        let id1 = store.add("first", "0 1 * * *").unwrap();
        let id2 = store.add("second", "0 2 * * *").unwrap();
        let jobs = store.list().unwrap();
        assert_eq!(jobs.len(), 2);
        // Insertion order preserved via created_at ASC.
        assert_eq!(jobs[0].id, id1);
        assert_eq!(jobs[1].id, id2);
    }

    #[test]
    fn store_delete_removes_job() {
        let (store, _dir) = open_temp();
        let id = store.add("task", "* * * * *").unwrap();
        assert!(store.delete(&id).unwrap());
        assert!(store.list().unwrap().is_empty());
    }

    #[test]
    fn store_delete_nonexistent_returns_false() {
        let (store, _dir) = open_temp();
        assert!(!store.delete("no-such-id").unwrap());
    }

    #[test]
    fn store_update_last_run() {
        let (store, _dir) = open_temp();
        let id = store.add("task", "* * * * *").unwrap();
        store.update_last_run(&id, "2026-03-20T09:00:00Z").unwrap();
        let jobs = store.list().unwrap();
        assert_eq!(jobs[0].last_run.as_deref(), Some("2026-03-20T09:00:00Z"));
    }

    #[test]
    fn store_rejects_invalid_schedule() {
        let (store, _dir) = open_temp();
        assert!(store.add("bad", "60 * * * *").is_err());
        assert!(store.list().unwrap().is_empty(), "nothing written on error");
    }

    #[test]
    fn store_empty_list_when_no_jobs() {
        let (store, _dir) = open_temp();
        assert!(store.list().unwrap().is_empty());
    }
}
