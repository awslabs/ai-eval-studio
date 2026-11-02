# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for update_state durable owner-key normalization (optimization-path evals).

An optimization-path eval Job carries ``config.owner_id`` but NO
``config.user_id`` (the manual eval path sets ``user_id``). Before this fix,
``JobTracker.update_state``'s durable write-through guarded on
``config.user_id`` only, so optimization-path evals never mirrored their
terminal ``state``/``progress``/``result`` to DynamoDB — the record stayed
frozen at ``queued``.

The fix resolves the durable owner as
``config.owner_id or config.user_id`` (the same coalesce
``hard_cancel_project_eval`` and the reconciler already use), so both dispatch
paths mirror terminal state. These tests trip the guard behaviorally:

* An optimization-path eval (owner_id only, arch_split=True) mirrors terminal
  state to DDB under the OWNER partition.
* A manual-path eval (user_id) still mirrors (the ``or`` falls back to
  user_id) — no regression.
* arch_split=False fires NO write-through (byte-unchanged old path).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.jobs import Job, JobTracker


@pytest.fixture
def tracker():
    """Fresh JobTracker with a clean in-memory job map."""
    t = JobTracker()
    t._reset()
    return t


def _register(tracker, job):
    tracker._jobs[job.job_id] = job


@pytest.mark.asyncio
async def test_optimization_path_eval_mirrors_terminal_state_under_owner(tracker):
    """owner_id-only eval mirrors terminal state to DDB under the owner partition."""
    mock_store = AsyncMock()

    job = Job(
        job_id="opt-eval-1",
        project_id="proj-1",
        type="eval",
        # Optimization path: owner_id set, NO user_id (this is the live shape).
        config={"arch_split": True, "owner_id": "owner-abc"},
        state="running",
        progress={"message": "Running"},
        created_at="2026-07-16T12:00:00Z",
    )
    _register(tracker, job)

    with patch("app.platform.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = mock_store
        # Silence the terminal fan-out side effects (WS publish etc.).
        with patch("app.services.jobs._publish_per_job_done"), \
             patch("app.services.jobs._publish_project_event"):
            tracker.update_state("opt-eval-1", "completed", result={"foo": "bar"})
            await asyncio.sleep(0.2)

    mock_store.update_job_state.assert_called()
    call = mock_store.update_job_state.call_args
    # First positional is the durable owner partition key — must be owner_id.
    assert call.args[0] == "owner-abc", (
        "optimization-path eval must mirror under config.owner_id when user_id is absent"
    )
    assert call.args[1] == "proj-1"
    assert call.args[2] == "opt-eval-1"
    assert call.kwargs.get("state") == "completed"


@pytest.mark.asyncio
async def test_manual_path_eval_still_mirrors_under_user_id(tracker):
    """user_id-only eval still mirrors (or-coalesce falls back to user_id)."""
    mock_store = AsyncMock()

    job = Job(
        job_id="manual-eval-1",
        project_id="proj-1",
        type="eval",
        # Manual path: user_id set, no owner_id.
        config={"arch_split": True, "user_id": "user-xyz"},
        state="running",
        progress={"message": "Running"},
        created_at="2026-07-16T12:00:00Z",
    )
    _register(tracker, job)

    with patch("app.platform.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = mock_store
        with patch("app.services.jobs._publish_per_job_done"), \
             patch("app.services.jobs._publish_project_event"):
            tracker.update_state("manual-eval-1", "completed", result={"foo": "bar"})
            await asyncio.sleep(0.2)

    mock_store.update_job_state.assert_called()
    call = mock_store.update_job_state.call_args
    assert call.args[0] == "user-xyz"


@pytest.mark.asyncio
async def test_arch_split_false_fires_no_write_through(tracker):
    """arch_split=False: no durable write-through (byte-unchanged old path)."""
    mock_store = AsyncMock()

    job = Job(
        job_id="dev-eval-1",
        project_id="proj-1",
        type="eval",
        config={"owner_id": "owner-abc"},  # No arch_split key → False
        state="running",
        progress={"message": "Running"},
        created_at="2026-07-16T12:00:00Z",
    )
    _register(tracker, job)

    with patch("app.platform.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = mock_store
        with patch("app.services.jobs._publish_per_job_done"), \
             patch("app.services.jobs._publish_project_event"):
            tracker.update_state("dev-eval-1", "completed", result={"foo": "bar"})
            await asyncio.sleep(0.2)

    mock_store.update_job_state.assert_not_called()
