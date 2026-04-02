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
# Run as the current host user so build artifacts in the bind-mounted workspace
# are owned by the invoking user, not root — same approach as DockerSandbox
# (libc::getuid()/getgid()).
#
# Read CARGO_HOME and RUSTUP_HOME from the image's own environment.
# {{println .}} prints one var=value per line, handling values that contain
# spaces.  sed -n '...//p' is used instead of grep so that "no match" exits 0
# under set -euo pipefail (grep would exit 1 and abort the script).
IMAGE_ENV=$(docker inspect "$DEV_IMAGE" --format '{{range .Config.Env}}{{println .}}{{end}}')
DEV_CARGO_HOME=$(echo "$IMAGE_ENV" | sed -n 's/^CARGO_HOME=//p')
DEV_RUSTUP_HOME=$(echo "$IMAGE_ENV" | sed -n 's/^RUSTUP_HOME=//p')
# Only fail fast when the image explicitly sets /root-based paths.
# When the vars are unset in the image, leave them empty so the container uses
# its own defaults (avoids misdiagnosing custom images that don't set them).
if [[ "$(id -u)" -ne 0 ]] && \
   { { [[ -n "$DEV_CARGO_HOME" ]] && [[ "$DEV_CARGO_HOME" == /root/* ]]; } || \
     { [[ -n "$DEV_RUSTUP_HOME" ]] && [[ "$DEV_RUSTUP_HOME" == /root/* ]]; }; }; then
    echo ""
    echo "    ✗ Dev image '$DEV_IMAGE' has a /root-based Rust toolchain:"
    echo "      CARGO_HOME='$DEV_CARGO_HOME'"
    echo "      RUSTUP_HOME='$DEV_RUSTUP_HOME'"
    echo "      This is incompatible with running the container as a non-root user."
    echo "      Rebuild the dev image:  ./scripts/build-dev-image.sh"
    exit 1
fi
echo "    image:   $DEV_IMAGE"
echo "    workdir: $WORKDIR"
# Omit --rm so the container is not auto-deleted before `docker wait` reads
# its exit code.  With --rm, the container is removed the instant it exits,
# creating a race where `docker wait` (and sometimes `docker logs -f`) can
# no longer find it.  The explicit `docker rm` below does the cleanup.
#
# FILTER is passed via env var (AMAEBI_FILTER) rather than interpolated into
# the sh -c string, preventing shell injection from a crafted filter value.
#
# Only pass CARGO_HOME/RUSTUP_HOME when the image explicitly sets them; for
# images that don't declare these vars, let the container use its own defaults.
EXTRA_ENV=()
[[ -n "$DEV_CARGO_HOME" ]] && EXTRA_ENV+=(-e "CARGO_HOME=$DEV_CARGO_HOME")
[[ -n "$DEV_RUSTUP_HOME" ]] && EXTRA_ENV+=(-e "RUSTUP_HOME=$DEV_RUSTUP_HOME")
CONTAINER_ID=$(docker run -d \
    -w "$WORKDIR" \
    --user "$(id -u):$(id -g)" \
    -v "$REPO_ROOT:$REPO_ROOT:rw" \
    "${EXTRA_ENV[@]}" \
    -e RUST_BACKTRACE=1 \
    -e RUST_TEST_THREADS=1 \
    -e AMAEBI_FILTER="$FILTER" \
    "$DEV_IMAGE" \
    sh -c 'cargo check && \
           if [ -n "$AMAEBI_FILTER" ]; then cargo test "$AMAEBI_FILTER"; else cargo test; fi && \
           cargo clippy -- -D warnings && \
           echo __TESTS_PASSED__')
echo "    container: $CONTAINER_ID"
docker logs -f "$CONTAINER_ID"
EXIT_CODE=$(docker wait "$CONTAINER_ID")
docker rm "$CONTAINER_ID" &>/dev/null || true
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

    echo "    image:   amaebi-sandbox:bookworm-slim"
    echo "    running: cargo test -- --ignored"
    if RUST_TEST_THREADS=1 "$HOST_CARGO" test -- --ignored; then ok; else fail; fi
fi

echo ""
echo "All checks passed ✓  ($pass steps)"
