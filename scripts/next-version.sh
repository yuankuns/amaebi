#!/usr/bin/env bash
#
# Derive the project's expected Cargo.toml version for the CURRENT commit
# (HEAD) and, with --check, fail CI if Cargo.toml disagrees.
#
# Calendar versioning: YYYY.M.N.
#   - YYYY / M = year and month of HEAD's committer date (master mode),
#     or of the latest non-merge commit in `master_ref..HEAD` (branch
#     mode — so CI's synthetic `refs/pull/<N>/merge` commit-time of
#     "now" doesn't trigger a spurious month rollover).
#   - Branch-mode rollover compares those (YYYY, M) to the master
#     Cargo.toml *version*'s (YYYY, M), not to master's committer date.
#     Using the version is important because an inactive master can sit
#     on stale content-dates for weeks while still being "the current
#     state of master" for purposes of anchoring the PR's expected N.
#
# The rule depends on which MODE we're in:
#
#   MASTER MODE — HEAD is on master (or any ancestor of `origin/master`).
#     Expected = parent's Cargo.toml + delta(HEAD.subject).
#       * delta = 1 if HEAD's subject starts with feat(/feat:/fix(/fix:/
#         docs(/docs:.  These are the "qualifying" conventional-commits
#         prefixes that represent user-visible change.
#       * delta = 0 for everything else (refactor, chore, test, revert,
#         spike, …).
#     On month rollover, N resets to 1 (qualifying) or 0 (non-qualifying).
#     Squash-merges on master are regular 1-parent commits whose subject
#     is the PR title, so the delta is determined by the PR title prefix.
#
#   BRANCH MODE — HEAD is off to the side of master (PR head, feature
#     branch).  Expected = master's Cargo.toml + branch_delta, where
#     branch_delta = 1 iff ANY non-merge commit in
#     `origin/master..HEAD` has a qualifying subject; otherwise 0.
#     Every commit on a single PR therefore shares ONE expected value —
#     a PR lands as exactly one bump (or zero) on master regardless of
#     how many internal commits it has.  Month rollover: if HEAD's
#     committer month differs from master's, expected is YYYY.M.1 (PR
#     contains a qualifier) or YYYY.M.0 (non-qualifying PR).
#
# Why two modes: before this file, a feature branch revalidated every
# commit against its parent's Cargo.toml, so each follow-up fix in a
# review cycle had to bump again (.17 → .18 → .19 → .20 as Copilot
# rounds piled up) even though the eventual squash to master would
# collapse the whole PR into a single delta.  The branch-mode rule
# pins the PR to a single expected value from the moment it diverges,
# so follow-up commits don't churn the version number.  Master-mode
# stays linear-per-commit so the squash subject continues to govern
# the landed version.
#
# Merge-commit handling: master MUST be linear (squash or rebase merges
# only).  Branch mode scans `master_ref..HEAD` with `--no-merges`, so a
# user merging master back into their branch adds a merge commit that
# is transparent to the qualifier scan (only the original PR work is
# counted).  On master itself, a 2+-parent commit breaks the rule: the
# standard "Create a merge commit" PR subject is `Merge pull request
# ...` (non-qualifying → expected delta 0), but the merge tree
# typically contains the PR's Cargo.toml bump, so `--check` would
# report a false mismatch.  Rather than silently papering over that,
# master-mode explicitly fails on merge commits with a clear error so
# the operator fixes their merge style (switch to squash / rebase)
# instead of learning about the gap later.  GitHub's synthetic
# `refs/pull/<N>/merge` commits are 2-parent but NOT ancestors of
# `origin/master`, so they fall into branch mode and are validated via
# the `master_ref..HEAD` scan — this guard only trips for merges that
# actually land on master.
#
# Usage:
#   scripts/next-version.sh            # print the expected version for HEAD
#   scripts/next-version.sh --check    # exit 1 if Cargo.toml disagrees
#
# The script only reads git; it never writes to the repo.

set -euo pipefail

die() { echo "next-version: $*" >&2; exit 1; }

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || die "not a git repo"
cd "$repo_root"

cargo_toml="$repo_root/Cargo.toml"
[[ -f "$cargo_toml" ]] || die "Cargo.toml not found at $cargo_toml"

# The master ref to diff the branch against.  Configurable via env so
# the test harness can point at a local branch instead of `origin/master`.
# Default matches CI's `actions/checkout@v4 fetch-depth: 0` convention.
master_ref="${NEXT_VERSION_MASTER_REF:-origin/master}"

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

# Return "master" if `$1` is an ancestor of (or equal to) $master_ref,
# else "branch".  If $master_ref doesn't resolve (e.g. shallow clone
# with no `origin/master`), fall back to "master" mode so the script
# still works in legacy contexts.
determine_mode() {
    local ref="$1"
    if ! git rev-parse --verify --quiet "$master_ref" >/dev/null; then
        echo "master"
        return
    fi
    if git merge-base --is-ancestor "$ref" "$master_ref" 2>/dev/null; then
        echo "master"
    else
        echo "branch"
    fi
}

# Compute expected version for `$1` under MASTER MODE (per-commit delta
# against parent's Cargo.toml).  Root commits fall back to committer
# month with N=1/0 depending on qualifier.  Multi-parent commits on
# master are rejected: the "Create a merge commit" button produces a
# `Merge pull request ...` subject (non-qualifying → expected delta 0)
# while the merge tree carries the PR's bumped Cargo.toml, so --check
# would flag a false mismatch.  We fail fast with actionable guidance
# instead.
expected_version_master_mode() {
    local ref="$1"
    local parent_hash parent_ver parent_year parent_month parent_n
    local ref_iso ref_year ref_month ref_subject qualifying delta
    local parent_count

    # `%P` is a space-separated list of parent hashes; `wc -w` counts them.
    parent_count=$(git log -1 --format='%P' "$ref" | wc -w)
    if [[ "$parent_count" -ge 2 ]]; then
        die "merge commit on master ($ref has $parent_count parents).  \
This rule requires linear history on master; use squash or rebase merges \
instead of \"Create a merge commit\".  Branch-mode is happy to scan through \
merges inside a PR, but once a multi-parent commit lands on master the \
delta rule cannot distinguish a non-qualifying merge subject from a \
qualifying Cargo.toml bump in the incoming tree."
    fi

    parent_hash=$(git log -1 --format='%H' "${ref}^1" 2>/dev/null || echo "")
    if [[ -z "$parent_hash" ]]; then
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

    if [[ "$ref_year" != "$parent_year" ]] || [[ "$ref_month" != "$parent_month" ]]; then
        if [[ "$qualifying" == "1" ]]; then
            printf "%d.%d.1\n" "$ref_year" "$ref_month"
        else
            printf "%d.%d.0\n" "$ref_year" "$ref_month"
        fi
        return
    fi

    if [[ "$qualifying" == "1" ]]; then
        delta=1
    else
        delta=0
    fi
    printf "%d.%d.%d\n" "$ref_year" "$ref_month" "$((parent_n + delta))"
}

# Compute expected version for `$1` under BRANCH MODE.  Expected = master's
# Cargo.toml + 1 if any non-merge commit in `master_ref..ref` is a
# qualifier, else master's Cargo.toml unchanged.  Month rollover is
# keyed off HEAD's committer date relative to master's version's month,
# so a PR opened in June against a May master lands at 2026.6.1 (or .0)
# regardless of master's N.
expected_version_branch_mode() {
    local ref="$1"
    local master_ver master_year master_month master_n
    local ref_iso ref_year ref_month
    local any_qualifier=0
    local subj qualifying

    master_ver=$(cargo_version_at "$master_ref")
    read -r master_year master_month master_n <<<"$(split_version "$master_ver")"

    # Derive HEAD's year/month from the latest non-merge commit in
    # `master_ref..ref`, NOT from `ref` itself.  On GitHub Actions PR
    # builds, `HEAD` is a synthetic `refs/pull/<N>/merge` whose committer
    # date is the workflow run time, not the PR head's time — using that
    # directly can trigger a spurious month rollover when CI runs after
    # month-end.  The latest author commit on the branch is the right
    # anchor for "when the PR's work was actually done".  If the branch
    # has no non-merge commits (pathological: empty PR), fall back to
    # `ref`'s own date.
    ref_iso=$(git log --no-merges -1 --format='%cI' "${master_ref}..${ref}" 2>/dev/null)
    if [[ -z "$ref_iso" ]]; then
        ref_iso=$(git log -1 --format='%cI' "$ref")
    fi
    ref_year=$(echo "$ref_iso" | cut -c1-4)
    ref_month=$((10#$(echo "$ref_iso" | cut -c6-7)))

    # Scan every non-merge commit in master_ref..ref.  --no-merges skips
    # any merge commits inside the branch (e.g. user merged master back
    # in), so `branch_delta` reflects only real work by the PR author.
    while IFS= read -r subj; do
        [[ -z "$subj" ]] && continue
        qualifying=$(is_qualifying "$subj")
        if [[ "$qualifying" == "1" ]]; then
            any_qualifier=1
            break
        fi
    done < <(git log --no-merges --format='%s' "${master_ref}..${ref}" 2>/dev/null)

    # Month rollover: if HEAD's committer month diverges from master's
    # version month, the branch's expected version moves to HEAD's
    # month with N reset to 1 (qualifying PR) or 0 (non-qualifying).
    if [[ "$ref_year" != "$master_year" ]] || [[ "$ref_month" != "$master_month" ]]; then
        if [[ "$any_qualifier" == "1" ]]; then
            printf "%d.%d.1\n" "$ref_year" "$ref_month"
        else
            printf "%d.%d.0\n" "$ref_year" "$ref_month"
        fi
        return
    fi

    printf "%d.%d.%d\n" "$master_year" "$master_month" \
        "$((master_n + any_qualifier))"
}

expected_version_for() {
    local ref="$1"
    local mode
    mode=$(determine_mode "$ref")
    if [[ "$mode" == "branch" ]]; then
        expected_version_branch_mode "$ref"
    else
        expected_version_master_mode "$ref"
    fi
}

# --- main ---

if ! git rev-parse --verify --quiet HEAD >/dev/null; then
    # Fresh repo with no commits — fall back to today's YYYY.M.0.
    today_year=$(date -u +%Y)
    today_month=$((10#$(date -u +%m)))
    expected="${today_year}.${today_month}.0"
else
    # HEAD itself is the anchor — no walking.  In both modes the
    # computation is robust against HEAD being a merge commit:
    #   * BRANCH MODE scans `master_ref..HEAD --no-merges` (so any merge
    #     commits inside the PR are transparent — a merge-master-into-
    #     branch doesn't re-count master's fixes) and reads HEAD's
    #     Cargo.toml directly.
    #   * MASTER MODE on a merge commit uses `HEAD^1`, which is the
    #     first-parent (base-branch tip) — the right parent for
    #     computing the delta introduced by the merge.
    expected=$(expected_version_for HEAD)
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
  (Rule: on master, expected = parent's Cargo.toml + 1 if HEAD is
   feat/fix/docs, else parent's Cargo.toml.  On a feature branch
   (non-ancestor of $master_ref), expected = master's Cargo.toml + 1
   if any branch commit is feat/fix/docs, else master's Cargo.toml
   unchanged — so a single PR lands as one bump regardless of how
   many follow-up commits it has.  See scripts/next-version.sh header
   for full rationale.)
Fix: edit Cargo.toml to '$expected' and commit.
EOF
        exit 1
    fi
    echo "OK $expected"
else
    echo "$expected"
fi
