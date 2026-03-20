//! Cron job store and scheduler.
//!
//! Jobs are persisted in `~/.amaebi/cron.json` (plain JSON, human-readable).
//! The daemon ticks every minute and fires any jobs whose schedule matches the
//! current UTC wall-clock time.  Results land in the inbox via [`InboxStore`].

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::auth::amaebi_home;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A registered cron job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CronJob {
    /// UUID v4 identifier.
    pub id: String,
    /// Human-readable task description, used as the LLM prompt.
    pub description: String,
    /// 5-field cron expression (`min hour dom mon dow`).
    pub schedule: String,
    /// RFC 3339 creation timestamp.
    pub created_at: String,
    /// RFC 3339 timestamp of the last successful run, or `null`.
    pub last_run: Option<String>,
}

// ---------------------------------------------------------------------------
// Path helper
// ---------------------------------------------------------------------------

/// Path to the cron job store (`~/.amaebi/cron.json`).
pub fn cron_path() -> Result<PathBuf> {
    Ok(amaebi_home()?.join("cron.json"))
}

// ---------------------------------------------------------------------------
// CRUD operations
// ---------------------------------------------------------------------------

/// Load all cron jobs from disk.  Returns an empty list if the file does not
/// exist yet (first run before any jobs are added).
pub fn load_jobs() -> Result<Vec<CronJob>> {
    let path = cron_path()?;
    if !path.exists() {
        return Ok(vec![]);
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

/// Persist the job list to disk using an atomic write (temp file → rename).
pub fn save_jobs(jobs: &[CronJob]) -> Result<()> {
    let path = cron_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(jobs).context("serialising cron jobs")?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &json).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &path)
        .with_context(|| format!("renaming cron tmp to {}", path.display()))?;
    Ok(())
}

/// Register a new cron job after validating the schedule expression.
///
/// Returns the new job's UUID.
pub fn add_job(description: &str, schedule: &str) -> Result<String> {
    // Validate first — never write an unparseable expression to disk.
    parse_schedule(schedule)
        .with_context(|| format!("invalid cron schedule: {schedule:?}"))?;

    let mut jobs = load_jobs()?;
    let id = uuid::Uuid::new_v4().to_string();
    jobs.push(CronJob {
        id: id.clone(),
        description: description.to_owned(),
        schedule: schedule.to_owned(),
        created_at: chrono::Utc::now().to_rfc3339(),
        last_run: None,
    });
    save_jobs(&jobs)?;
    Ok(id)
}

/// Remove a job by ID.  Returns `true` if removed, `false` if not found.
pub fn delete_job(id: &str) -> Result<bool> {
    let mut jobs = load_jobs()?;
    let before = jobs.len();
    jobs.retain(|j| j.id != id);
    if jobs.len() == before {
        return Ok(false);
    }
    save_jobs(&jobs)?;
    Ok(true)
}

/// Record that a job ran at `timestamp` (RFC 3339).
pub fn update_last_run(id: &str, timestamp: &str) -> Result<()> {
    let mut jobs = load_jobs()?;
    for job in &mut jobs {
        if job.id == id {
            job.last_run = Some(timestamp.to_owned());
        }
    }
    save_jobs(&jobs)?;
    Ok(())
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
pub fn due_jobs(
    jobs: &[CronJob],
    now: &chrono::DateTime<chrono::Utc>,
) -> Vec<CronJob> {
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
                let elapsed =
                    now.signed_duration_since(last_dt.with_timezone(&chrono::Utc));
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

    // ---- save/load roundtrip ---------------------------------------------

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("cron.json");

        let jobs = vec![CronJob {
            id: "test-id-1".into(),
            description: "daily report".into(),
            schedule: "0 9 * * *".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            last_run: Some("2026-03-20T09:00:00Z".into()),
        }];

        let json = serde_json::to_string_pretty(&jobs).unwrap();
        std::fs::write(&path, json).unwrap();

        let loaded: Vec<CronJob> =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "test-id-1");
        assert_eq!(loaded[0].schedule, "0 9 * * *");
        assert_eq!(loaded[0].last_run.as_deref(), Some("2026-03-20T09:00:00Z"));
    }
}
