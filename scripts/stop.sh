#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Kill processes from PID file
if [ -f .dev-pids ]; then
    echo "Stopping backend and frontend (from PID file)..."
    # Read into an array: the file holds several space-separated PIDs, so a
    # quoted substitution would hand kill one argument instead of many.
    read -ra dev_pids < .dev-pids
    kill "${dev_pids[@]}" 2>/dev/null || true
    rm -f .dev-pids
fi

# Also kill anything on our ports as fallback
echo "Clearing ports 8100 and 3100..."
lsof -ti:8100 | xargs kill 2>/dev/null || true
lsof -ti:3100 | xargs kill 2>/dev/null || true

echo "Stack stopped."
