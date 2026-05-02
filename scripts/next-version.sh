#!/usr/bin/env bash
#
# Derive the project version from the git history of the checked-out
# branch (HEAD) — on a PR branch this includes the PR's own commits;
# on master it is the master history.
#
# Rule (see CLAUDE.md): calendar versioning, `YYYY.M.N`.
#   - MAJOR = year of the latest **non-merge** commit on HEAD
#   - MINOR = month of the latest non-merge commit (1..12, no leading zero)
#   - PATCH (N) = count of non-merge commits on HEAD whose subject starts
#                 with `feat(` / `feat:` / `fix(` / `fix:` / `docs(` /
#                 `docs:`, counted WITHIN the current (YYYY, M) month
#                 only.  Other prefixes (`refactor`, `chore`, `test`,
#                 `revert`, `spike`, etc.) do not bump N.
#   - Merge commits (detected by **topology**: `%P` has 2+ parents, not
#     by subject text) are ignored entirely — they never bump N and
#     never advance (year, month).  This matters because GitHub's PR
#     checks run against a synthetic `refs/pull/<N>/merge` merge commit
#     whose committer date is "now in UTC", which would otherwise roll
#     the calver into the current month even when no real commit on the
#     branch belongs there (PR #143 broke exactly this way on
#     2026-05-01).
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
#     tracking the (year, month) of the latest non-merge commit and N
#     within that month.  We use committer date (%cI) because that is
#     what a human reader treats as "when the change landed".  Merge
#     commits are skipped by topology (see the awk body).  On a fresh
#     repo with no commits (`git log HEAD` fails under `pipefail`),
#     short-circuit to today's YYYY.M.0.
if ! git rev-parse --verify --quiet HEAD >/dev/null; then
    today_year=$(date -u +%Y)
    today_month_padded=$(date -u +%m)
    # Strip any leading zero without relying on GNU `%-m` (BSD/macOS safe).
    today_month=$((10#$today_month_padded))
    expected="${today_year}.${today_month}.0"
else
    # Columns are TAB-separated to avoid colliding with spaces in commit
    # subjects: `%cI\t%P\t%s`.  `%P` is a space-separated list of parent
    # hashes — one parent = normal commit, 2+ parents = merge commit
    # (detected here by topology, **not** by subject text, so a normal
    # commit whose message happens to start with "Merge " is NOT
    # skipped, and a merge commit with a custom non-"Merge …" subject
    # IS skipped).
    expected=$(git log --reverse --format='%cI%x09%P%x09%s' | awk -F'\t' '
        BEGIN { year = 0; month = 0; n = 0 }
        {
            # $1 = committer ISO-8601, $2 = parents, $3 = subject.
            ts = $1
            parents = $2
            subj = $3

            # Topology check: a merge commit has 2+ parents, which means
            # `%P` contains at least one space.  Skip these entirely —
            # they do not bump N and do not advance (year, month).  See
            # the header docs for why this matters for GitHub PR checks.
            if (index(parents, " ") > 0) {
                next
            }

            # %cI is a full ISO-8601 timestamp; the first 7 chars are YYYY-MM.
            y = substr(ts, 1, 4) + 0
            m = substr(ts, 6, 2) + 0

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
  (Rule: YYYY.M.N — year/month from the latest **non-merge** commit on
   the current branch (HEAD); N is the count of non-merge feat/fix/docs
   commits in that month on the same branch.  Other prefixes do not
   bump N.  Merge commits are detected by topology (2+ parents) and are
   ignored entirely, so GitHub's synthetic pull-request merge commits
   cannot flip the calver into a month no real commit belongs to.)
Fix: edit Cargo.toml to '$expected' and commit.
EOF
        exit 1
    fi
    echo "OK $expected"
else
    echo "$expected"
fi
