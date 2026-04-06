#!/usr/bin/env bash
# .claude/hooks/pre-commit.sh
#
# PreToolUse hook: runs scripts/test.sh --docker before every `git commit`.
# Called by Claude Code with CLAUDE_TOOL_INPUT set to the Bash tool's JSON input.
# Exit non-zero to block the commit and surface the failure to Claude.

set -euo pipefail

# python3 is required to parse CLAUDE_TOOL_INPUT; fail closed if missing.
if ! command -v python3 >/dev/null 2>&1; then
    echo "[hook] error: python3 is required to parse CLAUDE_TOOL_INPUT" >&2
    exit 1
fi

# Extract the bash command from the tool input; fail closed on parse error.
if ! TOOL_CMD=$(python3 -c "
import json, os, sys
raw = os.environ.get('CLAUDE_TOOL_INPUT', '{}')
try:
    payload = json.loads(raw)
except Exception:
    sys.exit(1)
if not isinstance(payload, dict):
    sys.exit(1)
command = payload.get('command')
if not isinstance(command, str):
    sys.exit(1)
print(command)
"); then
    echo "[hook] error: failed to parse CLAUDE_TOOL_INPUT" >&2
    exit 1
fi

# Use token-level parsing to detect 'git commit', correctly handling global
# git options before the subcommand (e.g. git -c foo=bar commit, git --no-pager commit).
SHOULD_RUN=$(python3 -c "
import shlex, sys

try:
    argv = shlex.split('''$TOOL_CMD''')
except Exception:
    print('0')
    sys.exit(0)

if not argv or argv[0] != 'git':
    print('0')
    sys.exit(0)

# Global git options that consume the next token as a value.
global_opts_with_value = {'-c', '-C', '--git-dir', '--work-tree',
                          '--namespace', '--super-prefix', '--config-env'}

i = 1
while i < len(argv):
    token = argv[i]
    if token == 'commit':
        break
    if token == '--':
        print('0')
        sys.exit(0)
    if token.startswith('-'):
        opt = token.split('=', 1)[0]
        if opt in global_opts_with_value and '=' not in token:
            i += 2
        else:
            i += 1
        continue
    # Non-option, non-'commit' token — some other subcommand.
    print('0')
    sys.exit(0)

if i >= len(argv) or argv[i] != 'commit':
    print('0')
    sys.exit(0)

# Check commit-level bypass flags as exact tokens.
commit_args = argv[i + 1:]
if any(arg in ('--no-verify', '-n', '--help', '-h') for arg in commit_args):
    print('0')
else:
    print('1')
" 2>/dev/null || echo "0")

if [ "$SHOULD_RUN" != "1" ]; then
    exit 0
fi

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$REPO_ROOT"
echo "[hook] running scripts/test.sh --docker before commit..."
exec ./scripts/test.sh --docker
