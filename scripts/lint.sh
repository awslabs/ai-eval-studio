#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Run the Python lint gate over the backend, infra and scripts trees.
#
# Gates the hard-error rule set configured in the repo-root ruff.toml — most
# importantly undefined names (F821): a name that is never bound fails only
# when its line runs, which can be at startup, so an offline check is the one
# thing that catches it first.
#
# Paths are resolved from this script's own location, not the caller's cwd, so
# the root ruff.toml is always the config that applies.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if command -v ruff >/dev/null 2>&1; then
  RUFF=(ruff)
elif [ -x backend/.venv/bin/ruff ]; then
  RUFF=(backend/.venv/bin/ruff)
elif [ -x "${VIRTUAL_ENV:-/nonexistent}/bin/ruff" ]; then
  RUFF=("$VIRTUAL_ENV/bin/ruff")
else
  echo "ERROR: ruff not found. Install the backend dev dependencies:" >&2
  echo "  uv sync --group dev   (from backend/)" >&2
  exit 1
fi

echo "==> ruff check backend infra scripts"
"${RUFF[@]}" check backend infra scripts

# Optional extra check: runs only when that script is present, and nothing
# above depends on it.
if [ -x "$REPO_ROOT/scripts/check-public-safe.sh" ]; then
  echo "==> public-safety check"
  "$REPO_ROOT/scripts/check-public-safe.sh"
fi

echo "lint gate: OK"
