#!/usr/bin/env bash
# scripts/test.sh — standardized test runner for amaebi
#
# Step 1 runs inside openclaw-sandbox-dev:bookworm-slim (per DEV_WORKFLOW.md).
# Step 2 (--docker) runs on the host since it needs access to the Docker daemon.
#
# Usage (from repo root or any subdirectory):
#   ./scripts/test.sh              # cargo check + test + clippy (in container)
#   ./scripts/test.sh --docker     # also run Docker integration tests (#[ignore]) on host
#   ./scripts/test.sh --filter <p> # filter test by pattern
#
# Docker tests require:
#   ./scripts/build-sandbox-image.sh  (builds amaebi-sandbox:bookworm-slim)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKDIR="$REPO_ROOT"

DEV_IMAGE="openclaw-sandbox-dev:bookworm-slim"
HOST_CARGO="${CARGO:-$HOME/.cargo/bin/cargo}"
RUN_DOCKER=0
FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --docker)  RUN_DOCKER=1; shift ;;
        --filter)  FILTER="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

pass=0

step() { echo ""; echo "==> Step $1: $2"; }
ok()   { echo "    ✓ passed"; pass=$((pass + 1)); }
fail() { echo "    ✗ failed"; exit 1; }

# Step 1: run inside container
step 1 "cargo check + test + clippy (in $DEV_IMAGE)"
# Note: --user 0:0 (root) is intentional. The image's Cargo/Rustup toolchains
# live under /home/user/.cargo and /home/user/.rustup, owned by the container's
# 'user' account. Root can write there; a host-mapped UID (id -u):(id -g) cannot,
# which breaks `cargo build`. Build artifacts written into the bind-mounted
# workspace will be root-owned, but that is an acceptable trade-off for a local
# dev-test runner.
if docker run --rm \
    -w "$WORKDIR" \
    --user 0:0 \
    -v "$REPO_ROOT:$REPO_ROOT:rw" \
    -v "$(dirname "$REPO_ROOT"):$(dirname "$REPO_ROOT"):rw" \
    -e HOME=/root \
    -e PATH=/home/user/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    -e RUSTUP_HOME=/home/user/.rustup \
    -e CARGO_HOME=/home/user/.cargo \
    -e RUST_BACKTRACE=1 \
    "$DEV_IMAGE" \
    sh -c "cargo check && cargo test${FILTER:+ $FILTER} && cargo clippy -- -D warnings"
then ok; else fail; fi

# Step 2 (optional): Docker integration tests — run on host (needs docker daemon)
if [[ "$RUN_DOCKER" == "1" ]]; then
    step 2 "Docker integration tests -- --ignored (on host)"

    if ! docker image inspect amaebi-sandbox:bookworm-slim &>/dev/null; then
        echo ""
        echo "    ⚠️  amaebi-sandbox:bookworm-slim not found. Build it first:"
        echo "       ./scripts/build-sandbox-image.sh"
        exit 1
    fi

    if "$HOST_CARGO" test -- --ignored; then ok; else fail; fi
fi

echo ""
echo "All checks passed ✓  ($pass steps)"
