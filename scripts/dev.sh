#!/usr/bin/env bash
# Create or reuse a git worktree and run Claude Code to write/fix code.
# Usage:
#   scripts/dev.sh <task-name> "<prompt>"
#
# Creates worktree at ~/amaebi-wt/<task-name> on branch feat/<task-name>.
# If the worktree already exists, reuses it (for fix iterations).
# Claude Code writes code and commits. Does NOT run tests or push.

set -euo pipefail

TASK="${1:-}"
PROMPT="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKTREE_BASE="$HOME/amaebi-wt"
WORKDIR="$WORKTREE_BASE/$TASK"
BRANCH="feat/$TASK"
CLAUDE="${CLAUDE:-$HOME/.local/bin/claude}"

if [[ -z "$TASK" || -z "$PROMPT" ]]; then
    echo "Usage: scripts/dev.sh <task-name> \"<prompt>\""
    echo "  task-name: short slug (e.g. fix-context-limit)"
    echo "  prompt:    task description for Claude Code"
    exit 1
fi

step() { echo ""; echo "==> $1"; }

# Step 1: setup worktree
step "worktree: $WORKDIR (branch: $BRANCH)"
if [[ -d "$WORKDIR" ]]; then
    echo "    reusing existing worktree"
else
    cd "$REPO_ROOT"
    git worktree add "$WORKDIR" -b "$BRANCH"
    echo "    created"
fi

# Step 2: run Claude Code
step "coder (Claude Code)"
cd "$WORKDIR"
exec "$CLAUDE" --permission-mode bypassPermissions --print "$PROMPT"
