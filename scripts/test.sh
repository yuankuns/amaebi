#!/usr/bin/env bash
# scripts/test.sh — standardized test runner for amaebi
#
# Step 1 runs inside the amaebi-dev:bookworm-slim Docker image.
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

# Log file: logs/test-YYYY-MM-DD-HHMMSS.log
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/test-$(date +%Y-%m-%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "==> Log: $LOG_FILE"

# Set AMAEBI_DEV_IMAGE to override the default dev image.
# Build the default image locally with: ./scripts/build-dev-image.sh
DEV_IMAGE="${AMAEBI_DEV_IMAGE:-amaebi-dev:bookworm-slim}"
HOST_CARGO="${CARGO:-$HOME/.cargo/bin/cargo}"
RUN_DOCKER=0
FILTER=""

usage() {
    echo "Usage:"
    echo "  ./scripts/test.sh              # cargo check + test + clippy (in container)"
    echo "  ./scripts/test.sh --docker     # also run Docker integration tests on host"
    echo "  ./scripts/test.sh --filter <p> # filter test by pattern"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --docker)  RUN_DOCKER=1; shift ;;
        --filter)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --filter"
                usage
                exit 1
            fi
            FILTER="$2"
            shift 2
            ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

pass=0

step() { echo ""; echo "==> Step $1: $2"; }
ok()   { echo "    ✓ passed"; pass=$((pass + 1)); }
fail() { echo "    ✗ failed"; exit 1; }

# Step 1: run inside container
step 1 "cargo check + test + clippy (in $DEV_IMAGE)"
if ! docker image inspect "$DEV_IMAGE" &>/dev/null; then
    echo ""
    echo "    ✗ Dev image '$DEV_IMAGE' not found."
    echo "      Build it first:  ./scripts/build-dev-image.sh"
    echo "      Or set AMAEBI_DEV_IMAGE to point to an existing local image."
    exit 1
fi
# Note: --user 0:0 (root) is intentional. The image's Cargo/Rustup toolchains
# live under /root/.cargo and /root/.rustup. Build artifacts written into the
# bind-mounted workspace will be root-owned, but that is an acceptable trade-off
# for a local dev-test runner.
echo "    image:   $DEV_IMAGE"
echo "    workdir: $WORKDIR"
# Capture the host user's UID/GID so we can restore ownership of any files
# the container creates as root in the bind-mounted workspace (primarily
# target/).  Without this, the host user cannot delete or modify those
# files without sudo.
HOST_UID=$(id -u)
HOST_GID=$(id -g)
# Ensure the container is always removed, even if the script is interrupted
# (Ctrl-C) or exits early due to set -e.  CONTAINER_ID is initialised to ""
# so the trap is a no-op before docker run assigns it.
CONTAINER_ID=""
cleanup_container() {
    if [[ -n "$CONTAINER_ID" ]]; then
        docker rm -f "$CONTAINER_ID" &>/dev/null || true
    fi
}
trap cleanup_container EXIT INT TERM
# Omit --rm so the container is not auto-deleted before `docker wait` reads
# its exit code.  With --rm, the container is removed the instant it exits,
# creating a race where `docker wait` (and sometimes `docker logs -f`) can
# no longer find it.  The explicit cleanup below handles the normal path,
# while the trap provides cleanup on interrupts/early exits.
CONTAINER_ID=$(docker run -d \
    -w "$WORKDIR" \
    --user 0:0 \
    -v "$REPO_ROOT:$REPO_ROOT:rw" \
    -v "$(dirname "$REPO_ROOT"):$(dirname "$REPO_ROOT"):rw" \
    -e HOME=/root \
    -e PATH=/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    -e RUSTUP_HOME=/root/.rustup \
    -e CARGO_HOME=/root/.cargo \
    -e RUST_BACKTRACE=1 \
    -e HOST_UID="$HOST_UID" \
    -e HOST_GID="$HOST_GID" \
    "$DEV_IMAGE" \
    sh -c "cargo check && cargo test${FILTER:+ $FILTER} && cargo clippy -- -D warnings && echo __TESTS_PASSED__
           rc=\$?
           chown -hR \"\$HOST_UID:\$HOST_GID\" \"$WORKDIR/target\" 2>/dev/null || true
           exit \$rc")
echo "    container: $CONTAINER_ID"
docker logs -f "$CONTAINER_ID"
EXIT_CODE=$(docker wait "$CONTAINER_ID")
docker rm "$CONTAINER_ID" &>/dev/null || true
CONTAINER_ID=""
if [[ "$EXIT_CODE" == "0" ]]; then ok; else fail; fi

# Step 2 (optional): Docker integration tests — run on host (needs docker daemon)
if [[ "$RUN_DOCKER" == "1" ]]; then
    step 2 "Docker integration tests -- --ignored (on host)"

    if ! docker image inspect amaebi-sandbox:bookworm-slim &>/dev/null; then
        echo ""
        echo "    ⚠️  amaebi-sandbox:bookworm-slim not found. Build it first:"
        echo "       ./scripts/build-sandbox-image.sh"
        exit 1
    fi

    # Preflight: ensure target/ is writable by the host user.
    # Step 1 runs cargo as root inside Docker and can leave root-owned artifacts.
    # The inline chown above handles the normal path; this catches edge cases
    # (e.g. the user ran Step 1 manually with an older script version).
    if [[ -d "$WORKDIR/target" ]] && { ! [[ -w "$WORKDIR/target" ]] || find "$WORKDIR/target" ! -w -print -quit 2>/dev/null | grep -q .; }; then
        echo "    detected non-writable paths under target/, repairing ownership via docker..."
        HOST_UID=$(id -u)
        HOST_GID=$(id -g)
        if docker run --rm \
            --user 0:0 \
            -v "$WORKDIR:/repo" \
            "$DEV_IMAGE" \
            sh -c "chown -hR ${HOST_UID}:${HOST_GID} /repo/target || true"; then
            if [[ -w "$WORKDIR/target" ]]; then
                echo "    target ownership repaired"
            else
                echo "    ✗ target/ is still not writable after repair attempt."
                echo "      Run manually: sudo chown -R ${HOST_UID}:${HOST_GID} $WORKDIR/target"
                exit 1
            fi
        else
            echo "    ✗ docker command failed while repairing target/ ownership."
            echo "      Run manually: sudo chown -R ${HOST_UID}:${HOST_GID} $WORKDIR/target"
            exit 1
        fi
    fi

    echo "    image:   amaebi-sandbox:bookworm-slim"
    echo "    running: cargo test -- --ignored"
    if "$HOST_CARGO" test -- --ignored; then ok; else fail; fi
fi

echo ""
echo "All checks passed ✓  ($pass steps)"
