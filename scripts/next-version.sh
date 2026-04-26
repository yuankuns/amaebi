#!/usr/bin/env bash
#
# Derive the project version from git history on the current branch.
#
# Rule (see CLAUDE.md):
#   - Baseline: 0.0.0
#   - Walk commits in chronological order:
#       feat(...): MINOR += 1, PATCH := 0
#       fix(...):  PATCH += 1
#       anything else: ignored
#   - MAJOR is never bumped automatically; a human bumps `Cargo.toml`
#     manually when they decide a release warrants MAJOR+1.  Once MAJOR
#     is > 0 in `Cargo.toml`, the scanner uses it verbatim as the floor:
#     only MINOR/PATCH are derived from history SINCE the commit that
#     last changed MAJOR in `Cargo.toml`.
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

# --- read MAJOR.MINOR.PATCH from Cargo.toml (first `version = "X.Y.Z"` line) ---
cargo_version=$(awk -F'"' '/^version = "/ { print $2; exit }' "$cargo_toml")
[[ -n "$cargo_version" ]] || die "could not parse version from $cargo_toml"

if [[ ! "$cargo_version" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
    die "Cargo.toml version '$cargo_version' is not MAJOR.MINOR.PATCH"
fi
cargo_major=${BASH_REMATCH[1]}
cargo_minor=${BASH_REMATCH[2]}
cargo_patch=${BASH_REMATCH[3]}

# --- pick the scan window ---
# When MAJOR == 0 we scan the full history.  Otherwise we find the latest
# commit that touched the MAJOR digit in Cargo.toml and only count commits
# after it — that way a human-chosen MAJOR bump resets the counters.
if [[ "$cargo_major" == "0" ]]; then
    base_range=""   # scan everything
else
    # Find the most recent commit whose diff of Cargo.toml changed the
    # leading MAJOR digit.  `git log -L` would be overkill; a simple
    # `git log -G'^version = "N\.'` over `Cargo.toml` is enough.
    base_commit=$(git log -n 1 --format=%H -G"^version = \"${cargo_major}\." -- Cargo.toml 2>/dev/null || true)
    if [[ -z "$base_commit" ]]; then
        # MAJOR > 0 but no commit ever introduced that MAJOR — treat as
        # "starting fresh from this MAJOR": no history counts, so the
        # expected version is MAJOR.0.0.
        echo "${cargo_major}.0.0"
        if [[ "${1:-}" == "--check" ]]; then
            if [[ "$cargo_version" != "${cargo_major}.0.0" ]]; then
                echo "next-version: Cargo.toml=$cargo_version but expected ${cargo_major}.0.0" >&2
                exit 1
            fi
        fi
        exit 0
    fi
    base_range="${base_commit}..HEAD"
fi

# --- scan commit subjects, chronological order ---
# Conventional-commit prefixes we care about.  Accept both `feat(scope):`
# and `feat:` forms.  Everything else is ignored (refactor, chore, docs,
# test, spike, merge commits, ...).
minor=0
patch=0
while IFS= read -r subject; do
    case "$subject" in
        feat\(*\):*|feat:*)
            minor=$((minor + 1))
            patch=0
            ;;
        fix\(*\):*|fix:*)
            patch=$((patch + 1))
            ;;
    esac
done < <(git log $base_range --reverse --format=%s)

expected="${cargo_major}.${minor}.${patch}"

if [[ "${1:-}" == "--check" ]]; then
    if [[ "$cargo_version" != "$expected" ]]; then
        cat >&2 <<EOF
next-version: Cargo.toml version mismatch.
  Cargo.toml says: $cargo_version
  History implies: $expected
  (MAJOR=$cargo_major fixed by Cargo.toml; MINOR/PATCH derived from
   commit history — feat bumps MINOR, fix bumps PATCH.)
Fix: edit Cargo.toml to '$expected', or if this PR should bump MAJOR,
     also edit Cargo.toml MAJOR and re-run this check.
EOF
        exit 1
    fi
    echo "OK $expected"
else
    echo "$expected"
fi
