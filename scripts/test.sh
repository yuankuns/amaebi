#!/usr/bin/env bash
# scripts/test.sh — standardized test runner for amaebi
#
# Usage:
#   ./scripts/test.sh [--ignored] [--filter <pattern>] [extra cargo test args]
#
# Flags:
#   --ignored   Also run #[ignore] tests (requires host with CLONE_NEWUSER)
#               On Ubuntu 24.04+: sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
#   --filter    Pass a test filter pattern to cargo test

set -euo pipefail

# Change to repo root regardless of where the script is called from
cd "$(dirname "$0")/.."

IGNORED=0
FILTER=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ignored) IGNORED=1; shift ;;
        --filter)  FILTER="$2"; shift 2 ;;
        *)         EXTRA_ARGS+=("$1"); shift ;;
    esac
done

PASS=0
FAIL=0

step() {
    echo ""
    echo "==> Step $1: $2"
}

ok() {
    echo "    ✓ passed"
    PASS=$((PASS + 1))
}

fail() {
    echo "    ✗ failed"
    FAIL=$((FAIL + 1))
    exit 1
}

# Step 1: cargo check
step 1 "cargo check"
cargo check 2>&1 && ok || fail

# Step 2: cargo test
step 2 "cargo test"
if [[ -n "$FILTER" ]]; then
    cargo test "$FILTER" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" 2>&1 && ok || fail
else
    cargo test "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" 2>&1 && ok || fail
fi

# Step 3: clippy
step 3 "cargo clippy -- -D warnings"
cargo clippy -- -D warnings 2>&1 && ok || fail

# Step 4 (optional): #[ignore] tests
if [[ "$IGNORED" == "1" ]]; then
    step 4 "cargo test -- --ignored  (namespace/capability tests)"

    APPARMOR_FILE="/proc/sys/kernel/apparmor_restrict_unprivileged_userns"
    if [[ -f "$APPARMOR_FILE" ]] && [[ "$(cat "$APPARMOR_FILE" | tr -d '[:space:]')" == "1" ]]; then
        echo ""
        echo "    ⚠️  AppArmor restricts unprivileged user namespaces (apparmor_restrict_unprivileged_userns=1)"
        echo "       Skipping namespace tests. To enable:"
        echo "       sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0"
    else
        cargo test -- --ignored 2>&1 && ok || fail
    fi
fi

echo ""
echo "All checks passed ✓  ($PASS steps)"
