#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -t amaebi-sandbox:latest docker/sandbox-test/
echo "Built amaebi-sandbox:latest"
