#!/usr/bin/env bash
# .claude/hooks/pre-commit.sh
#
# PreToolUse hook: runs scripts/test.sh --docker before every `git commit`.
# Called by Claude Code with CLAUDE_TOOL_INPUT set to the Bash tool's JSON input.
# Exit non-zero to block the commit and surface the failure to Claude.

set -euo pipefail

# Extract the bash command from the tool input.
TOOL_CMD=$(python3 -c "
import os, json
raw = os.environ.get('CLAUDE_TOOL_INPUT', '{}')
try:
    print(json.loads(raw).get('command', ''))
except Exception:
    print('')
" 2>/dev/null || true)

# Only intercept actual commit invocations; skip --no-verify and --help.
case "$TOOL_CMD" in
    *"git commit"*)
        if echo "$TOOL_CMD" | grep -qE '(--no-verify|-n[[:space:]]|--help|-h[[:space:]])'; then
            exit 0
        fi
        ;;
    *)
        exit 0
        ;;
esac

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$REPO_ROOT"
echo "[hook] running scripts/test.sh --docker before commit..."
exec ./scripts/test.sh --docker
