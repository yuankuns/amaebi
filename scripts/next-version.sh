#!/usr/bin/env bash
#
# Derive the project version from git history on the current branch.
#
# Rule (see CLAUDE.md): calendar versioning, `YYYY.M.N`.
#   - MAJOR = year of latest commit on branch
#   - MINOR = month of latest commit (1..12, no leading zero)
#   - PATCH (N) = count of commits on branch whose subject starts with
#                 `feat(` / `feat:` / `fix(` / `fix:` / `docs(` / `docs:`,
#                 counted WITHIN the current (YYYY, M) month only.
#                 Other prefixes (`refactor`, `chore`, `test`, `revert`,
#                 `spike`, merges, etc.) do not bump N.
#   - Each new month resets N to 0.
#   - Initial month (no qualifying commits yet): `YYYY.M.0`.
#
# Examples:
#   2026-04-01  feat(...):   -> 2026.4.1
#   2026-04-02  fix(...):    -> 2026.4.2
#   2026-04-03  refactor:    -> 2026.4.2  (no bump)
#   2026-05-05  feat(...):   -> 2026.5.1  (month rollover; N reset then +1)
#
# Usage:
#   scripts/next-version.sh            # print the expected version
#   scripts/next-version.sh --check    # exit 1 if Cargo.toml disagrees
#
# The script only reads git; it never writes to the repo.

set -euo pipefail

die() { echo "next-version: $*" >&2; exit 1; }

# --- locate repo root (script may be called from anywhere) ---
repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || die "not a git repo"
cd "$repo_root"

cargo_toml="$repo_root/Cargo.toml"
[[ -f "$cargo_toml" ]] || die "Cargo.toml not found at $cargo_toml"

# --- read version from Cargo.toml (first `version = "X.Y.Z"` line) ---
cargo_version=$(awk -F'"' '/^version = "/ { print $2; exit }' "$cargo_toml")
[[ -n "$cargo_version" ]] || die "could not parse version from $cargo_toml"

# --- walk every commit on the current branch (chronological), tracking the
#     (year, month) of the latest commit and N within that month.  We use
#     committer date (%cI) since merges land at merge-time and that is what
#     a human reader treats as "when the change landed on master".
expected=$(git log --reverse --format='%cI%x09%s' | awk '
    BEGIN { year = 0; month = 0; n = 0 }
    {
        # %cI is a full ISO-8601 timestamp; the first 7 chars are YYYY-MM.
        y = substr($1, 1, 4) + 0
        m = substr($1, 6, 2) + 0
        # Subject is everything after the tab.
        tab = index($0, "\t")
        subj = substr($0, tab + 1)

        if (y != year || m != month) {
            year = y; month = m; n = 0
        }
        if (subj ~ /^feat[(:]/ || subj ~ /^fix[(:]/ || subj ~ /^docs[(:]/) {
            n++
        }
    }
    END {
        if (year == 0) {
            # Empty history — fall back to the current wallclock date so
            # the very first commit on a fresh repo has a sensible base.
            "date -u +%Y.%-m" | getline today
            printf "%s.0\n", today
        } else {
            printf "%d.%d.%d\n", year, month, n
        }
    }
')

if [[ "${1:-}" == "--check" ]]; then
    if [[ "$cargo_version" != "$expected" ]]; then
        cat >&2 <<EOF
next-version: Cargo.toml version mismatch.
  Cargo.toml says: $cargo_version
  History implies: $expected
  (Rule: YYYY.M.N — year/month from latest master commit; N is the count
   of feat/fix/docs commits in the current month on master.  Other
   prefixes do not bump N.)
Fix: edit Cargo.toml to '$expected' and commit.
EOF
        exit 1
    fi
    echo "OK $expected"
else
    echo "$expected"
fi
