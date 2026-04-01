//! Deferred follow-up: relative-time parsing and cron-expression generation.
//!
//! Converts natural-language relative durations ("in 30 minutes", "in 2h") into
//! UTC [`DateTime`] values and one-shot 5-field cron expressions used by the
//! `schedule_followup` tool.

use anyhow::{bail, Result};
use chrono::{DateTime, Datelike, Timelike, Utc};

/// Minimum follow-up delay in minutes.
const MIN_MINUTES: u64 = 5;

/// Parse a relative time string like `"in 2 hours"` or `"in 30m"` into a
/// future UTC datetime.
///
/// The returned datetime is ceil'd to the next whole minute so the generated
/// cron expression never fires earlier than requested.
///
/// # Minimum
/// 5 minutes.  Returns `Err` for anything shorter, unrecognised formats, or
/// zero-duration inputs.
///
/// # Supported forms
/// - `in N minutes` / `in N minute` / `in Nm` / `in N min` / `in N mins`
/// - `in N hours`   / `in N hour`   / `in Nh` / `in N hr`  / `in N hrs`
pub fn parse_relative_time(when: &str) -> Result<DateTime<Utc>> {
    let input = when.trim().to_lowercase();

    // Must start with "in "
    let rest = input
        .strip_prefix("in ")
        .ok_or_else(|| {
            anyhow::anyhow!(
                "unrecognised time format {:?}; expected e.g. 'in 30 minutes' or 'in 2h'",
                when
            )
        })?
        .trim_start();

    // Parse leading digits
    let digit_end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    if digit_end == 0 {
        bail!(
            "unrecognised time format {:?}; expected a positive integer after 'in'",
            when
        );
    }
    let n: u64 = rest[..digit_end].parse()?;
    if n == 0 {
        bail!("duration must be positive, got 0 in {:?}", when);
    }

    let unit = rest[digit_end..].trim();
    let minutes: u64 = match unit {
        "m" | "min" | "mins" | "minute" | "minutes" => n,
        "h" | "hr" | "hrs" | "hour" | "hours" => n
            .checked_mul(60)
            .ok_or_else(|| anyhow::anyhow!("duration overflow in {:?}", when))?,
        "" => bail!(
            "unrecognised time format {:?}; missing unit (use 'm', 'h', 'minutes', or 'hours')",
            when
        ),
        other => bail!(
            "unrecognised unit {:?} in {:?}; expected 'minutes', 'hours', 'm', or 'h'",
            other,
            when
        ),
    };

    if minutes < MIN_MINUTES {
        bail!(
            "minimum follow-up delay is {} minutes, got {} in {:?}",
            MIN_MINUTES,
            minutes,
            when
        );
    }

    let now = Utc::now();
    let target = now
        + chrono::Duration::try_minutes(minutes as i64)
            .ok_or_else(|| anyhow::anyhow!("duration overflow"))?;

    // Ceil to next whole minute: advance if any sub-minute component is set.
    let ceiled = if target.second() > 0 || target.nanosecond() > 0 {
        target
            .with_second(0)
            .and_then(|t| t.with_nanosecond(0))
            .map(|t| t + chrono::Duration::try_minutes(1).expect("1 minute is always valid"))
            .ok_or_else(|| anyhow::anyhow!("failed to ceil datetime to minute boundary"))?
    } else {
        target
    };

    Ok(ceiled)
}

/// Convert a UTC datetime to a one-shot 5-field cron expression.
///
/// The expression matches exactly the given minute/hour/day/month, with `*`
/// for day-of-week.  Suitable for use as a one-shot job: fire once at the
/// given wall-clock moment, then auto-delete.
///
/// # Example
/// `2026-04-01 14:35 UTC` → `"35 14 1 4 *"`
pub fn datetime_to_cron_expr(dt: &DateTime<Utc>) -> String {
    format!(
        "{} {} {} {} *",
        dt.minute(),
        dt.hour(),
        dt.day(),
        dt.month()
    )
}

/// Validate and convert a `when` string into a `(cron_expr, fires_at)` pair.
///
/// Combines [`parse_relative_time`] and [`datetime_to_cron_expr`].
/// Returns an error if the input is unrecognised or below the 5-minute minimum.
pub fn resolve_when(when: &str) -> Result<(String, DateTime<Utc>)> {
    let fires_at = parse_relative_time(when)?;
    let cron = datetime_to_cron_expr(&fires_at);
    Ok((cron, fires_at))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    // ---- parse_relative_time: valid inputs ----

    #[test]
    fn parse_in_minutes_long_form() {
        let before = Utc::now();
        let dt = parse_relative_time("in 30 minutes").unwrap();
        let min_expected = before + chrono::Duration::minutes(30);
        let max_expected = Utc::now() + chrono::Duration::minutes(31); // +1 for ceil
        assert!(dt >= min_expected, "got {dt}, expected >= {min_expected}");
        assert!(dt <= max_expected, "got {dt}, expected <= {max_expected}");
    }

    #[test]
    fn parse_in_hours_long_form() {
        let before = Utc::now();
        let dt = parse_relative_time("in 2 hours").unwrap();
        let min_expected = before + chrono::Duration::hours(2);
        let max_expected = Utc::now() + chrono::Duration::hours(2) + chrono::Duration::minutes(1);
        assert!(dt >= min_expected, "got {dt}, expected >= {min_expected}");
        assert!(dt <= max_expected, "got {dt}, expected <= {max_expected}");
    }

    #[test]
    fn parse_in_minutes_short_m() {
        let before = Utc::now();
        let dt = parse_relative_time("in 30m").unwrap();
        let min_expected = before + chrono::Duration::minutes(30);
        assert!(dt >= min_expected, "got {dt}, expected >= {min_expected}");
    }

    #[test]
    fn parse_in_hours_short_h() {
        let before = Utc::now();
        let dt = parse_relative_time("in 2h").unwrap();
        let min_expected = before + chrono::Duration::hours(2);
        assert!(dt >= min_expected, "got {dt}, expected >= {min_expected}");
    }

    #[test]
    fn parse_in_1_hour_singular() {
        let before = Utc::now();
        let dt = parse_relative_time("in 1 hour").unwrap();
        let min_expected = before + chrono::Duration::hours(1);
        assert!(dt >= min_expected, "got {dt}, expected >= {min_expected}");
    }

    #[test]
    fn parse_in_120_min_large_minutes() {
        let before = Utc::now();
        let dt = parse_relative_time("in 120 min").unwrap();
        let min_expected = before + chrono::Duration::minutes(120);
        assert!(dt >= min_expected, "got {dt}, expected >= {min_expected}");
    }

    #[test]
    fn parse_exact_5_minute_boundary_accepted() {
        // Exactly 5 minutes is the minimum — must succeed.
        let result = parse_relative_time("in 5 minutes");
        assert!(
            result.is_ok(),
            "5 minutes should be accepted, got: {result:?}"
        );
    }

    #[test]
    fn parse_in_hrs_abbreviation() {
        let before = Utc::now();
        let dt = parse_relative_time("in 3 hrs").unwrap();
        let min_expected = before + chrono::Duration::hours(3);
        assert!(dt >= min_expected, "got {dt}, expected >= {min_expected}");
    }

    // ---- parse_relative_time: result is ceil'd to whole minute ----

    #[test]
    fn result_is_ceil_to_whole_minute() {
        // Regardless of sub-minute offset in `now`, the result must land on
        // an exact minute boundary (seconds == 0).
        let dt = parse_relative_time("in 10 minutes").unwrap();
        assert_eq!(dt.second(), 0, "result must be on a whole minute: {dt}");
    }

    // ---- parse_relative_time: rejection cases ----

    #[test]
    fn parse_4_minutes_rejects() {
        let err = parse_relative_time("in 4 minutes").unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("5") || msg.contains("minimum"),
            "error should mention 5-minute minimum, got: {msg}"
        );
    }

    #[test]
    fn parse_zero_hours_rejects() {
        assert!(
            parse_relative_time("in 0 hours").is_err(),
            "0 hours must be rejected"
        );
    }

    #[test]
    fn parse_zero_minutes_rejects() {
        assert!(
            parse_relative_time("in 0 minutes").is_err(),
            "0 minutes must be rejected"
        );
    }

    #[test]
    fn parse_absolute_time_rejects() {
        assert!(
            parse_relative_time("tomorrow 9am").is_err(),
            "absolute time must be rejected"
        );
    }

    #[test]
    fn parse_empty_string_rejects() {
        assert!(
            parse_relative_time("").is_err(),
            "empty string must be rejected"
        );
    }

    #[test]
    fn parse_bare_number_rejects() {
        assert!(
            parse_relative_time("30").is_err(),
            "bare number must be rejected"
        );
    }

    #[test]
    fn parse_negative_duration_rejects() {
        assert!(
            parse_relative_time("in -5 minutes").is_err(),
            "negative duration must be rejected"
        );
    }

    // ---- datetime_to_cron_expr ----

    #[test]
    fn cron_expr_format_correct() {
        // 2026-04-01 14:35 UTC  →  "35 14 1 4 *"
        use chrono::TimeZone;
        let dt = chrono::Utc.with_ymd_and_hms(2026, 4, 1, 14, 35, 0).unwrap();
        assert_eq!(datetime_to_cron_expr(&dt), "35 14 1 4 *");
    }

    #[test]
    fn cron_expr_midnight() {
        use chrono::TimeZone;
        let dt = chrono::Utc.with_ymd_and_hms(2026, 12, 31, 0, 0, 0).unwrap();
        assert_eq!(datetime_to_cron_expr(&dt), "0 0 31 12 *");
    }

    #[test]
    fn cron_expr_single_digit_fields_no_leading_zero() {
        // e.g. 2026-03-05 09:05 → "5 9 5 3 *"  (no leading zeros)
        use chrono::TimeZone;
        let dt = chrono::Utc.with_ymd_and_hms(2026, 3, 5, 9, 5, 0).unwrap();
        assert_eq!(datetime_to_cron_expr(&dt), "5 9 5 3 *");
    }

    #[test]
    fn cron_expr_dow_field_is_star() {
        use chrono::TimeZone;
        let dt = chrono::Utc.with_ymd_and_hms(2026, 6, 15, 9, 5, 0).unwrap();
        let expr = datetime_to_cron_expr(&dt);
        let fields: Vec<&str> = expr.split_whitespace().collect();
        assert_eq!(fields.len(), 5, "must be 5 fields: {expr}");
        assert_eq!(fields[4], "*", "dow must be '*': {expr}");
    }

    // ---- resolve_when ----

    #[test]
    fn resolve_when_returns_cron_and_future_datetime() {
        let (cron, fires_at) = resolve_when("in 30 minutes").unwrap();
        let fields: Vec<&str> = cron.split_whitespace().collect();
        assert_eq!(fields.len(), 5, "cron must be 5 fields: {cron}");
        assert!(fires_at > Utc::now(), "fires_at must be in the future");
    }

    #[test]
    fn resolve_when_cron_matches_fires_at() {
        // The cron expression minute/hour/day/month fields must agree with fires_at.
        let (cron, fires_at) = resolve_when("in 10 minutes").unwrap();
        let fields: Vec<&str> = cron.split_whitespace().collect();
        assert_eq!(fields.len(), 5);
        let expected_min = fires_at.format("%-M").to_string();
        let expected_hour = fires_at.format("%-H").to_string();
        let expected_day = fires_at.format("%-d").to_string();
        let expected_month = fires_at.format("%-m").to_string();
        assert_eq!(fields[0], expected_min, "minute mismatch in cron {cron}");
        assert_eq!(fields[1], expected_hour, "hour mismatch in cron {cron}");
        assert_eq!(fields[2], expected_day, "day mismatch in cron {cron}");
        assert_eq!(fields[3], expected_month, "month mismatch in cron {cron}");
    }

    #[test]
    fn resolve_when_rejects_below_minimum() {
        assert!(resolve_when("in 3 minutes").is_err());
    }

    #[test]
    fn resolve_when_rejects_unknown_format() {
        assert!(resolve_when("next tuesday").is_err());
    }
}
