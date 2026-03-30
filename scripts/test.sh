#!/usr/bin/env bash
set -euo pipefail

echo "Running unit tests..."
cargo test

echo "Checking for amaebi-sandbox:latest image before running Docker integration tests..."
if ! docker image inspect amaebi-sandbox:latest &>/dev/null; then
    echo "  ⚠️  amaebi-sandbox:latest not found. Build it first:"
    echo "      ./scripts/build-sandbox-image.sh"
    exit 1
fi

echo "Running Docker integration tests..."
cargo test -- --ignored
