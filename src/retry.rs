//! Shared HTTP retry utilities used by all provider backends.
//!
//! Extracted from `copilot.rs` so that `bedrock.rs` and `responses.rs` can
//! share the same exponential back-off formula and `Retry-After` header
//! parsing without going through `copilot`.

use std::time::Duration;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Base delay for exponential backoff: attempt 0 → 1 s, 1 → 2 s, 2 → 4 s.
pub const BACKOFF_BASE_MS: u64 = 1_000;

/// Hard ceiling on a `Retry-After` header value.  Prevents hanging for
/// unreasonably long server-imposed back-off windows.
pub const MAX_RETRY_AFTER_SECS: u64 = 30;

// ---------------------------------------------------------------------------
// Retry helpers
// ---------------------------------------------------------------------------

/// Exponential back-off delay for `attempt` (0-indexed).
///
/// Returns 1 s, 2 s, 4 s for attempts 0, 1, 2.  The exponent is capped at
/// 10, so the delay saturates at `BACKOFF_BASE_MS << 10` (about 17 minutes),
/// but in practice `MAX_RETRIES` is 3 so the maximum used delay is 4 s.
pub fn backoff_delay(attempt: u32) -> Duration {
    Duration::from_millis(BACKOFF_BASE_MS << attempt.min(10))
}

/// Parse the `Retry-After` response header from a [`reqwest::header::HeaderMap`]
/// into a [`Duration`].
///
/// Accepts an integer number of seconds only (date-form is not handled).
/// Returns `None` when the header is absent, non-numeric, or zero.
/// Caps the returned delay at [`MAX_RETRY_AFTER_SECS`] to prevent the daemon
/// from sleeping for unreasonably long periods.
pub fn parse_retry_after_header(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let secs = headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&s| s > 0)?;
    Some(Duration::from_secs(secs.min(MAX_RETRY_AFTER_SECS)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderValue, RETRY_AFTER};

    #[test]
    fn backoff_delay_increases_exponentially() {
        assert_eq!(backoff_delay(0), Duration::from_millis(1_000));
        assert_eq!(backoff_delay(1), Duration::from_millis(2_000));
        assert_eq!(backoff_delay(2), Duration::from_millis(4_000));
    }

    #[test]
    fn backoff_delay_saturates_at_high_attempt() {
        // Must not panic on large attempt numbers.
        let _ = backoff_delay(30);
        let _ = backoff_delay(u32::MAX);
    }

    #[test]
    fn parse_retry_after_header_missing() {
        let headers = HeaderMap::new();
        assert_eq!(parse_retry_after_header(&headers), None);
    }

    #[test]
    fn parse_retry_after_header_invalid_value() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("not-a-number"));
        assert_eq!(parse_retry_after_header(&headers), None);
    }

    #[test]
    fn parse_retry_after_header_zero_returns_none() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("0"));
        assert_eq!(parse_retry_after_header(&headers), None);
    }

    #[test]
    fn parse_retry_after_header_valid_value() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("10"));
        let dur = parse_retry_after_header(&headers).expect("expected Some(Duration)");
        assert_eq!(dur, Duration::from_secs(10));
    }

    #[test]
    fn parse_retry_after_header_caps_large_values() {
        let mut headers = HeaderMap::new();
        let large = (MAX_RETRY_AFTER_SECS.saturating_add(10)).to_string();
        headers.insert(
            RETRY_AFTER,
            HeaderValue::from_str(&large).expect("valid header value"),
        );
        let dur = parse_retry_after_header(&headers).expect("expected Some(Duration)");
        assert_eq!(dur, Duration::from_secs(MAX_RETRY_AFTER_SECS));
    }
}
