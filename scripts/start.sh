#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Kill any existing processes on our ports
lsof -ti:8100 | xargs kill 2>/dev/null || true
lsof -ti:3100 | xargs kill 2>/dev/null || true

# Claude Agent SDK must use Bedrock
export CLAUDE_CODE_USE_BEDROCK=1

# Both processes tee to a file as well as the terminal. Without this the only
# copy of the log lives in whichever terminal launched the stack, so "reproduce
# it and read the log" is impossible for anyone who wasn't at that keyboard.
# Appended, not truncated, with a session banner: restarting the stack to
# retry a bug must not destroy the log of the failure being chased. Safe to
# delete at any time — gitignored, and nothing reads it back.
LOG_DIR="$PROJECT_ROOT/.dev-logs"
mkdir -p "$LOG_DIR"
SESSION_BANNER="===== stack started $(date '+%Y-%m-%d %H:%M:%S %Z') (pid $$) ====="
echo "$SESSION_BANNER" >> "$LOG_DIR/backend.log"
echo "$SESSION_BANNER" >> "$LOG_DIR/frontend.log"

echo "Starting backend..."
cd "$PROJECT_ROOT/backend"
# Watch only the ``app/`` directory. Whitelist > blacklist: anything
# outside ``app/`` is invisible to the reloader, so eval result writes
# under ``../projects`` and pytest temp files under the venv never
# trigger a reload, regardless of where the venv lives.
#
# Process substitution rather than a ``| tee`` pipeline: in a pipeline ``$!``
# is the LAST command's pid, so BACKEND_PID would be tee's and both the
# cleanup trap and stop.sh would leave uvicorn running on port 8100.
#
# ``--ws-max-size`` matches the deployed server, so a client frame the cloud
# would refuse is refused here too rather than first failing in a deployment.
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8100 \
    --ws-max-size 65536 \
    --reload-dir app > >(tee -a "$LOG_DIR/backend.log") 2>&1 &
BACKEND_PID=$!

echo "Starting frontend..."
cd "$PROJECT_ROOT/frontend"
npm run dev > >(tee -a "$LOG_DIR/frontend.log") 2>&1 &
FRONTEND_PID=$!

echo "$BACKEND_PID $FRONTEND_PID" > "$PROJECT_ROOT/.dev-pids"
echo ""
echo "Stack running:"
echo "  Frontend: http://localhost:3100"
echo "  Backend:  http://localhost:8100"
echo "  Bedrock:  CLAUDE_CODE_USE_BEDROCK=1"
echo "  Logs:     .dev-logs/backend.log, .dev-logs/frontend.log"
echo ""
echo "Press Ctrl+C to stop..."

cleanup() {
    echo ""
    echo "Stopping backend and frontend..."
    kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
    rm -f "$PROJECT_ROOT/.dev-pids"
    echo "Stopped."
}

trap cleanup INT TERM
wait
