#!/usr/bin/env bash
#
# Derive the project version from the git history of the checked-out
# branch (HEAD) — on a PR branch this includes the PR's own commits;
# on master it is the master history.
#
# Rule (see CLAUDE.md): calendar versioning, `YYYY.M.N`.
#   - MAJOR = year of latest commit on HEAD
#   - MINOR = month of latest commit (1..12, no leading zero)
#   - PATCH (N) = count of commits on HEAD whose subject starts with
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

# --- walk every commit on the checked-out branch (HEAD), chronological,
#     tracking the (year, month) of the latest commit and N within that
#     month.  We use committer date (%cI) since merges land at merge-time
#     and that is what a human reader treats as "when the change landed".
#     On a fresh repo with no commits (`git log HEAD` fails under
#     `pipefail`), we short-circuit to today's YYYY.M.0.
if ! git rev-parse --verify --quiet HEAD >/dev/null; then
    today_year=$(date -u +%Y)
    today_month_padded=$(date -u +%m)
    # Strip any leading zero without relying on GNU `%-m` (BSD/macOS safe).
    today_month=$((10#$today_month_padded))
    expected="${today_year}.${today_month}.0"
else
    expected=$(git log --reverse --format='%cI%x09%s' | awk '
        BEGIN { year = 0; month = 0; n = 0 }
        {
            # %cI is a full ISO-8601 timestamp; the first 7 chars are YYYY-MM.
            y = substr($1, 1, 4) + 0
            m = substr($1, 6, 2) + 0
            # Subject is everything after the tab.
            tab = index($0, "\t")
            subj = substr($0, tab + 1)

            # Skip merge commits outright.  They never bump N (they are
            # not feat/fix/docs) and — critically — they must not
            # advance the (year, month) either.  GitHub synthesises a
            # preview merge commit for every pull-request CI run with a
            # timestamp of "now in UTC"; without this guard, a PR that
            # ran even a second into a new month would roll the calver
            # forward to that month even though no real commit on the
            # branch falls there.  PR #143 broke exactly this way on
            # 2026-05-01.
            if (subj ~ /^Merge /) {
                next
            }

            if (y != year || m != month) {
                year = y; month = m; n = 0
            }
            if (subj ~ /^feat[(:]/ || subj ~ /^fix[(:]/ || subj ~ /^docs[(:]/) {
                n++
            }
        }
        END { printf "%d.%d.%d\n", year, month, n }
    ')
fi

if [[ "${1:-}" == "--check" ]]; then
    if [[ "$cargo_version" != "$expected" ]]; then
        cat >&2 <<EOF
next-version: Cargo.toml version mismatch.
  Cargo.toml says: $cargo_version
  History implies: $expected
  (Rule: YYYY.M.N — year/month from the latest commit on the current
   branch (HEAD); N is the count of feat/fix/docs commits in that month
   on the same branch.  Other prefixes do not bump N.)
Fix: edit Cargo.toml to '$expected' and commit.
EOF
        exit 1
    fi
    echo "OK $expected"
else
    echo "$expected"
fi
