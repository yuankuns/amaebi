#!/usr/bin/env bash
#
# Derive the project version for the CURRENT commit (HEAD) based on its
# parent's Cargo.toml plus a delta determined by HEAD's commit type.
#
# Rule (see CLAUDE.md): calendar versioning, `YYYY.M.N`.
#   - YYYY / M = year and month of HEAD's committer date.
#   - N = HEAD^'s Cargo.toml N + delta (if same month as HEAD^) or
#         starts fresh on month rollover:
#           * 1  if HEAD is a qualifying commit (feat/fix/docs)
#           * 0  if HEAD is anything else
#   - delta (same month):
#           * +1 if HEAD's subject starts with `feat(` / `feat:` / `fix(` /
#                `fix:` / `docs(` / `docs:`
#           *  0 otherwise (refactor, chore, test, revert, spike, …)
#   - Merge commits (2+ parents) are transparent: they are not themselves
#     validated.  `--check` walks down from HEAD through any merge
#     commits, preferring `^2` (GitHub convention: `refs/pull/<N>/merge`
#     has `^1` = base branch tip, `^2` = incoming PR tip) until it finds
#     a non-merge "anchor" ancestor, then validates THAT commit's
#     Cargo.toml against the anchor's own parent's Cargo.toml + delta.
#     Squash merges on master are 1-parent (linear), so they are treated
#     as regular commits.
#
# New-invariant rationale: the pre-2026.5 script counted total qualifying
# commits across HEAD's history and required Cargo.toml == that count.
# That broke on squash merges: a PR with several internal commits would
# bump Cargo.toml several times, then squash would preserve only the last
# value, creating a "跳号" gap (e.g. master went from .2 → .13 in one
# merge).  The delta-based invariant tolerates those gaps because it only
# checks neighbor-to-neighbor increments, not the absolute count.
#
# Usage:
#   scripts/next-version.sh            # print the expected version for HEAD
#   scripts/next-version.sh --check    # exit 1 if HEAD's Cargo.toml disagrees
#
# The script only reads git; it never writes to the repo.

set -euo pipefail

die() { echo "next-version: $*" >&2; exit 1; }

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || die "not a git repo"
cd "$repo_root"

cargo_toml="$repo_root/Cargo.toml"
[[ -f "$cargo_toml" ]] || die "Cargo.toml not found at $cargo_toml"

# Return the version string embedded in `$1`'s Cargo.toml, or empty if
# that tree has no Cargo.toml.  Uses `git show` so no working-tree state
# is consulted — important for walking history.
cargo_version_at() {
    local ref="$1"
    git show "${ref}:Cargo.toml" 2>/dev/null |
        awk -F'"' '/^version = "/ { print $2; exit }'
}

# Read the current working-tree Cargo.toml version (what `--check`
# compares against).  Uses awk like `cargo_version_at` so formatting
# quirks are handled the same way.
current_cargo_version() {
    awk -F'"' '/^version = "/ { print $2; exit }' "$cargo_toml"
}

# Parse "YYYY.M.N" into three space-separated fields.  Missing / malformed
# versions return "0 0 0" so callers can still compute a sensible delta
# (month rollover will apply, N starts fresh).
split_version() {
    local v="$1"
    if [[ -z "$v" ]]; then
        echo "0 0 0"
        return
    fi
    # awk allows fractional handling but calver is integer-only.
    echo "$v" | awk -F'.' '{ printf "%d %d %d\n", $1, $2, $3 }'
}

# Return 1 if `$1` (subject) starts with feat/fix/docs (with `(` or `:`),
# else 0.  Conservative prefix match so `features` / `fixup!` /
# `document` never fire; the `(` / `:` separator is the second anchor
# that keeps `feat…` words that aren't conventional-commits prefixes
# from being mistaken for qualifying prefixes.
is_qualifying() {
    local subj="$1"
    if [[ "$subj" =~ ^(feat|fix|docs)[\(:] ]]; then
        echo 1
    else
        echo 0
    fi
}

# Return 1 if `$1` (ref) is a merge commit (2+ parents), else 0.
is_merge() {
    local ref="$1"
    local parents
    parents=$(git log -1 --format='%P' "$ref")
    if [[ "$parents" =~ " " ]]; then
        echo 1
    else
        echo 0
    fi
}

# Walk from `$1` down through merge commits until a non-merge ancestor
# is found.  For merge commits we follow `^2` (the "incoming" branch —
# GitHub convention: `^1` = base branch tip, `^2` = PR/feature tip) so
# that on a synthetic `refs/pull/<N>/merge` check, the validated
# Cargo.toml is the PR head's, not the base branch's.  Returns the hash
# of that ancestor, or empty if none found.
find_non_merge_ancestor() {
    local ref="$1"
    while [[ -n "$ref" ]] && [[ "$(is_merge "$ref")" == "1" ]]; do
        # Try ^2 first (incoming branch).  Fall back to ^1 if ^2 doesn't
        # exist (defensive — a 2-parent commit must have ^2, but handle
        # gracefully anyway).
        local next
        next=$(git log -1 --format='%H' "${ref}^2" 2>/dev/null || echo "")
        if [[ -z "$next" ]]; then
            next=$(git log -1 --format='%H' "${ref}^1" 2>/dev/null || echo "")
        fi
        ref="$next"
    done
    echo "$ref"
}

# Compute the expected Cargo.toml version for commit `$1` based on its
# parent's Cargo.toml + delta derived from `$1`'s type and committer date.
#
# This is the core invariant: every non-merge commit's Cargo.toml equals
# its parent's Cargo.toml plus a delta determined by the commit's type
# and whether we've crossed a month boundary.
expected_version_for() {
    local ref="$1"
    local parent_hash parent_ver parent_year parent_month parent_n
    local ref_iso ref_year ref_month ref_subject qualifying delta

    parent_hash=$(git log -1 --format='%H' "${ref}^1" 2>/dev/null || echo "")
    if [[ -z "$parent_hash" ]]; then
        # Root commit — no parent to base delta off of.  Fall back to the
        # committer date and start the month at N=1 (for qualifying) or 0.
        ref_iso=$(git log -1 --format='%cI' "$ref")
        ref_year=$(echo "$ref_iso" | cut -c1-4)
        ref_month=$((10#$(echo "$ref_iso" | cut -c6-7)))
        ref_subject=$(git log -1 --format='%s' "$ref")
        qualifying=$(is_qualifying "$ref_subject")
        if [[ "$qualifying" == "1" ]]; then
            printf "%d.%d.1\n" "$ref_year" "$ref_month"
        else
            printf "%d.%d.0\n" "$ref_year" "$ref_month"
        fi
        return
    fi

    parent_ver=$(cargo_version_at "$parent_hash")
    read -r parent_year parent_month parent_n <<<"$(split_version "$parent_ver")"

    ref_iso=$(git log -1 --format='%cI' "$ref")
    ref_year=$(echo "$ref_iso" | cut -c1-4)
    ref_month=$((10#$(echo "$ref_iso" | cut -c6-7)))
    ref_subject=$(git log -1 --format='%s' "$ref")
    qualifying=$(is_qualifying "$ref_subject")

    # Month rollover: N resets to 1 (qualifying) or 0 (non-qualifying),
    # keyed off HEAD's committer month, NOT parent's.  This means a June
    # `chore:` lands at 2026.6.0 even if master was at 2026.5.42.
    if [[ "$ref_year" != "$parent_year" ]] || [[ "$ref_month" != "$parent_month" ]]; then
        if [[ "$qualifying" == "1" ]]; then
            printf "%d.%d.1\n" "$ref_year" "$ref_month"
        else
            printf "%d.%d.0\n" "$ref_year" "$ref_month"
        fi
        return
    fi

    # Same-month: add delta to parent's N.
    if [[ "$qualifying" == "1" ]]; then
        delta=1
    else
        delta=0
    fi
    printf "%d.%d.%d\n" "$ref_year" "$ref_month" "$((parent_n + delta))"
}

# --- main ---

if ! git rev-parse --verify --quiet HEAD >/dev/null; then
    # Fresh repo with no commits — fall back to today's YYYY.M.0.
    today_year=$(date -u +%Y)
    today_month=$((10#$(date -u +%m)))
    expected="${today_year}.${today_month}.0"
else
    # If HEAD is a merge commit (GitHub synthetic PR merge
    # refs/pull/<N>/merge is the usual case), walk down to the first
    # non-merge ancestor — that's the commit whose Cargo.toml is
    # authoritative for the branch's head state.
    anchor=$(find_non_merge_ancestor HEAD)
    if [[ -z "$anchor" ]]; then
        die "could not find a non-merge ancestor of HEAD"
    fi
    expected=$(expected_version_for "$anchor")
fi

if [[ "${1:-}" == "--check" ]]; then
    current=$(current_cargo_version)
    # Distinguish "Cargo.toml unreadable / malformed" from "version
    # disagrees".  Reporting a parse failure as a mismatch would point
    # the user at the wrong fix (edit Cargo.toml vs fix the format).
    if [[ -z "$current" ]]; then
        die "could not parse 'version = \"…\"' from $cargo_toml"
    fi
    if [[ -z "$expected" ]]; then
        die "could not compute expected version (parent Cargo.toml parse failed?)"
    fi
    if [[ "$current" != "$expected" ]]; then
        cat >&2 <<EOF
next-version: Cargo.toml version mismatch.
  Cargo.toml says: $current
  Expected:        $expected
  (Rule: YYYY.M from HEAD's committer date; N = parent's N + 1 when HEAD
   is feat/fix/docs and same month as parent, else parent's N; N resets
   to 1 or 0 on month rollover.  Merge commits are transparent — expected
   equals the first non-merge ancestor's expected value.  See scripts/
   next-version.sh header for full rationale.)
Fix: edit Cargo.toml to '$expected' and commit.
EOF
        exit 1
    fi
    echo "OK $expected"
else
    echo "$expected"
fi
