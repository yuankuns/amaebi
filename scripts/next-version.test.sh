#!/usr/bin/env bash
#
# Tests for scripts/next-version.sh.  Each `scenario_*` function spins up
# a disposable git repo in `mktemp -d`, replays a specific history, and
# asserts that `next-version.sh` produces the expected output.  No
# network access; `origin/master` is simulated by a local branch and
# passed into the script via `NEXT_VERSION_MASTER_REF`.

set -euo pipefail

script_path="$(cd "$(dirname "$0")" && pwd)/next-version.sh"
[[ -x "$script_path" ]] || { echo "test: next-version.sh not executable at $script_path" >&2; exit 1; }

pass=0
fail=0
failures=()

# Each test runs in its own temp dir so state doesn't leak between
# scenarios.  mktemp gives absolute paths so `cd` doesn't need care.
make_repo() {
    local dir
    dir=$(mktemp -d -t next-version-test.XXXXXX)
    (
        cd "$dir"
        git init -q -b master
        git config user.email test@example.com
        git config user.name tester
        # Block `next-version.sh`'s default `origin/master` lookup — we
        # use a local simulated master instead.  The script reads this
        # env var (documented in its header).
        :
    )
    echo "$dir"
}

# Write $2 to $1/Cargo.toml as a minimal crate manifest with the given
# version, preserving a stable filename the script can grep.
write_cargo() {
    local dir="$1" ver="$2"
    cat >"$dir/Cargo.toml" <<EOF
[package]
name = "testcrate"
version = "$ver"
edition = "2021"
EOF
}

# Commit with a fixed date so month-rollover tests are deterministic.
# $3 (date, ISO-ish) is optional; defaults to "now".
commit_all() {
    local dir="$1" subject="$2" when="${3:-}"
    (
        cd "$dir"
        git add -A
        if [[ -n "$when" ]]; then
            GIT_AUTHOR_DATE="$when" GIT_COMMITTER_DATE="$when" \
                git commit -q -m "$subject"
        else
            git commit -q -m "$subject"
        fi
    )
}

# Run the script in $1 with NEXT_VERSION_MASTER_REF=$2 and no --check.
# Echoes the expected-version line.
run_script() {
    local dir="$1" master_ref="$2"
    (
        cd "$dir"
        NEXT_VERSION_MASTER_REF="$master_ref" "$script_path"
    )
}

# Run with --check; echoes exit code and any stderr.  Captures both so
# scenarios can assert either "OK <ver>" on stdout or a mismatch exit.
run_check() {
    local dir="$1" master_ref="$2"
    (
        cd "$dir"
        set +e
        NEXT_VERSION_MASTER_REF="$master_ref" "$script_path" --check
        echo "__EXIT__=$?"
    )
}

assert_eq() {
    local label="$1" expected="$2" actual="$3"
    if [[ "$expected" == "$actual" ]]; then
        pass=$((pass + 1))
        echo "  ok   $label"
    else
        fail=$((fail + 1))
        failures+=("$label: expected '$expected', got '$actual'")
        echo "  FAIL $label"
        echo "       expected: $expected"
        echo "       actual:   $actual"
    fi
}

# ---------------------------------------------------------------------

scenario_master_mode_fix_bumps_n() {
    echo "scenario: master mode — fix(…) on master bumps N"
    local dir; dir=$(make_repo)
    write_cargo "$dir" "2026.5.0"
    commit_all "$dir" "chore: seed" "2026-05-01T00:00:00"
    write_cargo "$dir" "2026.5.1"
    commit_all "$dir" "fix(x): first fix" "2026-05-02T00:00:00"
    # `master` branch is its own ref; simulate `origin/master` by pointing
    # at the same commit.  HEAD is-ancestor-of-master, so master mode.
    (cd "$dir" && git branch simulated-master master)
    local out; out=$(run_script "$dir" "simulated-master")
    assert_eq "master fix bumps .0 → .1" "2026.5.1" "$out"
    rm -rf "$dir"
}

scenario_master_mode_refactor_no_bump() {
    echo "scenario: master mode — refactor on master doesn't bump"
    local dir; dir=$(make_repo)
    write_cargo "$dir" "2026.5.5"
    commit_all "$dir" "fix: prior" "2026-05-01T00:00:00"
    # Same Cargo.toml version; change some unrelated content so the
    # commit isn't empty.
    echo "cleanup" >"$dir/noise.txt"
    commit_all "$dir" "refactor: cleanup" "2026-05-02T00:00:00"
    (cd "$dir" && git branch simulated-master master)
    local out; out=$(run_script "$dir" "simulated-master")
    assert_eq "refactor keeps .5" "2026.5.5" "$out"
    rm -rf "$dir"
}

scenario_branch_mode_single_fix_bumps_once() {
    echo "scenario: branch mode — single fix on branch → master + 1"
    local dir; dir=$(make_repo)
    write_cargo "$dir" "2026.5.10"
    commit_all "$dir" "fix(core): lands on master" "2026-05-01T00:00:00"
    (cd "$dir" && git branch simulated-master master && git checkout -q -b feat/x)
    write_cargo "$dir" "2026.5.11"
    commit_all "$dir" "fix(x): branch work" "2026-05-02T00:00:00"
    local out; out=$(run_script "$dir" "simulated-master")
    assert_eq "branch fix → master.N + 1" "2026.5.11" "$out"
    # --check should pass since Cargo.toml matches.
    local check; check=$(run_check "$dir" "simulated-master" | tail -1)
    assert_eq "check exit code 0" "__EXIT__=0" "$check"
    rm -rf "$dir"
}

scenario_branch_mode_fix_plus_refactors_still_one_bump() {
    echo "scenario: branch mode — one fix + three refactors still = master + 1"
    local dir; dir=$(make_repo)
    write_cargo "$dir" "2026.5.17"
    commit_all "$dir" "fix(prior): lands on master" "2026-05-01T00:00:00"
    (cd "$dir" && git branch simulated-master master && git checkout -q -b feat/review)
    # First commit on branch: qualifying + bump.
    write_cargo "$dir" "2026.5.18"
    commit_all "$dir" "fix(x): main work" "2026-05-02T00:00:00"
    # Second commit on branch: refactor, version stays at .18 (the key
    # behavior the whole PR exists to enable).
    echo "noise1" >"$dir/noise1.txt"
    commit_all "$dir" "refactor(x): review cleanup 1" "2026-05-03T00:00:00"
    local out; out=$(run_script "$dir" "simulated-master")
    assert_eq "branch fix+refactor stays at .18" "2026.5.18" "$out"
    # Add two more refactors; still .18.
    echo "noise2" >"$dir/noise2.txt"
    commit_all "$dir" "refactor(x): review cleanup 2" "2026-05-04T00:00:00"
    echo "noise3" >"$dir/noise3.txt"
    commit_all "$dir" "chore(x): polish" "2026-05-05T00:00:00"
    local out2; out2=$(run_script "$dir" "simulated-master")
    assert_eq "branch fix+3 noise commits stays at .18" "2026.5.18" "$out2"
    rm -rf "$dir"
}

scenario_branch_mode_all_refactors_no_bump() {
    echo "scenario: branch mode — all refactors on branch → master + 0"
    local dir; dir=$(make_repo)
    write_cargo "$dir" "2026.5.17"
    commit_all "$dir" "fix: prior" "2026-05-01T00:00:00"
    (cd "$dir" && git branch simulated-master master && git checkout -q -b refactor/foo)
    echo "r1" >"$dir/r1.txt"
    commit_all "$dir" "refactor: only refactor" "2026-05-02T00:00:00"
    local out; out=$(run_script "$dir" "simulated-master")
    assert_eq "all-refactor branch keeps .17" "2026.5.17" "$out"
    rm -rf "$dir"
}

scenario_branch_mode_double_bump_caught() {
    echo "scenario: branch mode — double bump is caught by --check"
    local dir; dir=$(make_repo)
    write_cargo "$dir" "2026.5.20"
    commit_all "$dir" "fix: prior" "2026-05-01T00:00:00"
    (cd "$dir" && git branch simulated-master master && git checkout -q -b feat/over)
    write_cargo "$dir" "2026.5.21"
    commit_all "$dir" "fix(a): legitimate bump" "2026-05-02T00:00:00"
    # Author bumps again even though branch mode forbids it.
    write_cargo "$dir" "2026.5.22"
    commit_all "$dir" "fix(b): accidental second bump" "2026-05-03T00:00:00"
    local out; out=$(run_check "$dir" "simulated-master" | tail -1)
    assert_eq "double bump → exit 1" "__EXIT__=1" "$out"
    rm -rf "$dir"
}

scenario_branch_mode_month_rollover() {
    echo "scenario: branch mode — HEAD in June against May master rolls month"
    local dir; dir=$(make_repo)
    write_cargo "$dir" "2026.5.42"
    commit_all "$dir" "fix: prior" "2026-05-31T00:00:00"
    (cd "$dir" && git branch simulated-master master && git checkout -q -b feat/jun)
    write_cargo "$dir" "2026.6.1"
    commit_all "$dir" "fix(x): opened in June" "2026-06-03T00:00:00"
    local out; out=$(run_script "$dir" "simulated-master")
    assert_eq "June-on-May branch with qualifier → YYYY.M.1" "2026.6.1" "$out"
    # Same repo, extra refactor in June: must still be .1 (not .2).
    echo "r" >"$dir/r.txt"
    commit_all "$dir" "refactor(x): month rollover + noise" "2026-06-04T00:00:00"
    local out2; out2=$(run_script "$dir" "simulated-master")
    assert_eq "June rollover + refactor stays at .1" "2026.6.1" "$out2"
    rm -rf "$dir"
}

scenario_branch_mode_no_qualifier_month_rollover() {
    echo "scenario: branch mode — non-qualifying PR rolling into new month"
    local dir; dir=$(make_repo)
    write_cargo "$dir" "2026.5.10"
    commit_all "$dir" "fix: prior" "2026-05-31T00:00:00"
    (cd "$dir" && git branch simulated-master master && git checkout -q -b refactor/jun)
    write_cargo "$dir" "2026.6.0"
    commit_all "$dir" "refactor: no qualifier" "2026-06-03T00:00:00"
    local out; out=$(run_script "$dir" "simulated-master")
    assert_eq "June refactor-only branch → YYYY.M.0" "2026.6.0" "$out"
    rm -rf "$dir"
}

scenario_branch_mode_synthetic_pr_merge_routes_via_caret2() {
    echo "scenario: branch mode — synthetic refs/pull/<N>/merge resolves via ^2"
    local dir; dir=$(make_repo)
    write_cargo "$dir" "2026.5.5"
    commit_all "$dir" "fix: master prior" "2026-05-01T00:00:00"
    (cd "$dir" && git branch simulated-master master && git checkout -q -b feat/topic)
    write_cargo "$dir" "2026.5.6"
    commit_all "$dir" "fix(t): branch work" "2026-05-02T00:00:00"
    # Fabricate a GitHub-style merge commit: master + feat/topic merged,
    # with ^1 = master tip and ^2 = feat/topic tip.  `git merge --no-ff`
    # from a detached master gives exactly that.
    (
        cd "$dir"
        git checkout -q master
        git merge --no-ff -q -m "Merge pull request #99" feat/topic
    )
    local out; out=$(run_script "$dir" "simulated-master")
    # Anchor walks ^2 to feat/topic tip → branch mode → master + 1.
    assert_eq "synthetic PR merge routes to branch rule" "2026.5.6" "$out"
    rm -rf "$dir"
}

scenario_branch_mode_with_internal_merge_is_transparent() {
    echo "scenario: branch mode — merge commit inside the branch is ignored"
    local dir; dir=$(make_repo)
    write_cargo "$dir" "2026.5.3"
    commit_all "$dir" "fix: master prior" "2026-05-01T00:00:00"
    (cd "$dir" && git branch simulated-master master && git checkout -q -b feat/long)
    # One qualifier commit.
    write_cargo "$dir" "2026.5.4"
    commit_all "$dir" "fix(y): original branch work" "2026-05-02T00:00:00"
    # Simulate merging master back into the branch later (the commit's
    # subject might say "Merge master" but --no-merges filters it out
    # of the qualifier scan).
    (
        cd "$dir"
        git checkout -q master
        echo "later" >"later.txt"
        git add -A
        GIT_AUTHOR_DATE=2026-05-03T00:00:00 \
            GIT_COMMITTER_DATE=2026-05-03T00:00:00 \
            git commit -q -m "chore: master moves on"
        git checkout -q feat/long
        GIT_AUTHOR_DATE=2026-05-04T00:00:00 \
            GIT_COMMITTER_DATE=2026-05-04T00:00:00 \
            git merge --no-ff -q -m "Merge master into feat/long" master
        # Move simulated master forward to the post-merge state on master
        # (so `master_ref..HEAD` is non-empty).
        git branch -f simulated-master master
    )
    local out; out=$(run_script "$dir" "simulated-master")
    assert_eq "internal merge doesn't double-count" "2026.5.4" "$out"
    rm -rf "$dir"
}

# ---------------------------------------------------------------------

scenario_master_mode_fix_bumps_n
scenario_master_mode_refactor_no_bump
scenario_branch_mode_single_fix_bumps_once
scenario_branch_mode_fix_plus_refactors_still_one_bump
scenario_branch_mode_all_refactors_no_bump
scenario_branch_mode_double_bump_caught
scenario_branch_mode_month_rollover
scenario_branch_mode_no_qualifier_month_rollover
scenario_branch_mode_synthetic_pr_merge_routes_via_caret2
scenario_branch_mode_with_internal_merge_is_transparent

echo
echo "passed: $pass   failed: $fail"
if [[ "$fail" -gt 0 ]]; then
    printf '  - %s\n' "${failures[@]}"
    exit 1
fi
