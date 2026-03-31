#!/usr/bin/env bash
# scripts/build-dev-image.sh — build the amaebi development Docker image
#
# Builds docker/dev/Dockerfile into amaebi-dev:bookworm-slim.
# This image provides a Rust toolchain for running scripts/test.sh in a container.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Building amaebi-dev:bookworm-slim from docker/dev/Dockerfile ..."
docker build -t amaebi-dev:bookworm-slim "$REPO_ROOT/docker/dev"
echo "==> Success: amaebi-dev:bookworm-slim is ready."
