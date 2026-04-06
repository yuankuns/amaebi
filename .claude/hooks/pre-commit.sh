#!/usr/bin/env bash
# .claude/hooks/pre-commit.sh
#
# PreToolUse hook: runs scripts/test.sh --docker before every `git commit`.
# Called by Claude Code with hook JSON piped to stdin.
# Exit non-zero to block the commit and surface the failure to Claude.

set -euo pipefail

# python3 is required to parse stdin JSON; fail closed if missing.
if ! command -v python3 >/dev/null 2>&1; then
    echo "[hook] error: python3 is required" >&2
    exit 1
fi

# Read hook input from stdin; extract the bash command. Fail closed on error.
if ! TOOL_CMD=$(python3 -c "
import json, sys
try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(1)
if not isinstance(payload, dict):
    sys.exit(1)
tool_input = payload.get('tool_input', {})
if not isinstance(tool_input, dict):
    sys.exit(1)
command = tool_input.get('command')
if not isinstance(command, str):
    sys.exit(1)
print(command)
"); then
    echo "[hook] error: failed to parse hook input from stdin" >&2
    exit 1
fi

# Use token-level parsing to detect 'git commit', correctly handling global
# git options before the subcommand (e.g. git -c foo=bar commit, git -C /path commit).
# TOOL_CMD is passed via environment to avoid shell-interpolation injection.
# Outputs two lines: "0" or "1" (should run), then the effective -C work dir (may be empty).
# Fails with exit 1 on unparseable input so the hook fails closed.
PARSE_RESULT=$(TOOL_CMD="$TOOL_CMD" python3 -c "
import os, shlex, sys

cmd = os.environ.get('TOOL_CMD', '')
try:
    argv = shlex.split(cmd)
except Exception:
    sys.stderr.write('[hook] error: could not tokenize git command\n')
    sys.exit(1)

if not argv or argv[0] != 'git':
    print('0')
    print('')
    sys.exit(0)

# Global git options that consume the next token as a value.
global_opts_with_value = {'-c', '-C', '--git-dir', '--work-tree',
                          '--namespace', '--super-prefix', '--config-env'}

i = 1
work_dir = ''
while i < len(argv):
    token = argv[i]
    if token == 'commit':
        break
    if token == '--':
        print('0')
        print('')
        sys.exit(0)
    if token.startswith('-'):
        opt = token.split('=', 1)[0]
        if opt in global_opts_with_value and '=' not in token:
            if opt == '-C' and i + 1 < len(argv):
                work_dir = argv[i + 1]
            i += 2
        else:
            if opt == '-C' and len(token) > 2:
                work_dir = token[2:]
            i += 1
        continue
    # Non-option, non-'commit' token — some other git subcommand.
    print('0')
    print('')
    sys.exit(0)

if i >= len(argv) or argv[i] != 'commit':
    print('0')
    print('')
    sys.exit(0)

# Check commit-level bypass flags as exact tokens.
commit_args = argv[i + 1:]
if any(arg in ('--no-verify', '-n', '--help', '-h') for arg in commit_args):
    print('0')
else:
    print('1')
print(work_dir)
") || {
    echo "[hook] error: failed to parse git command, blocking commit" >&2
    exit 1
}

SHOULD_RUN=$(echo "$PARSE_RESULT" | sed -n '1p')
WORK_DIR=$(echo "$PARSE_RESULT" | sed -n '2p')

if [ "$SHOULD_RUN" != "1" ]; then
    exit 0
fi

if [ -n "$WORK_DIR" ]; then
    REPO_ROOT=$(git -C "$WORK_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$WORK_DIR")
else
    REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
fi
cd "$REPO_ROOT"
echo "[hook] running scripts/test.sh --docker before commit..."
exec ./scripts/test.sh --docker
