# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test: the progress poller must not claim completion it can't fire.

``claim_completion_fired`` is a ONE-WAY, exactly-once gate. The reconciler's
completion-recovery block skips any job whose ``completion_fired`` is set
(``eval_reconciler.py``: ``if job_record.get("completion_fired"): continue``), so
a claim taken and then abandoned permanently loses the completion turn — the
"reconciler is the safety net" comment does not hold for a claimed job.

The poller used to claim FIRST and materialize second, so a
``RunOutputNotAvailable`` (worker upload hadn't landed yet — a routine, transient
condition) burned the claim and broke out of the loop. The reconciler in the same
codebase already does it in the safe order: materialize, then claim only when
ready to fire.

This test pins the safe order by asserting the claim is NOT taken when
materialization fails.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.platform import get_platform
from app.services.job_progress_republisher import poll_and_republish_progress
from app.services.run_output_snapshot import RunOutputNotAvailable


async def _seed_terminal_job(store, job_id: str, project_id: str, owner_id: str) -> None:
    """Seed a durable job record plus a terminal ``results`` event."""
    await store.put_job(
        {
            "job_id": job_id,
            "project_id": project_id,
            "type": "eval",
            "state": "running",
            "owner_id": owner_id,
            "config": {"user_id": owner_id, "owner_id": owner_id},
        }
    )
    ttl = int(time.time()) + (7 * 24 * 3600)
    await store.append_job_event(
        job_id,
        1,
        {
            "stream": f"job:{job_id}",
            "seq": 1,
            "kind": "progress",
            "data": {"event": "results", "file": "eval_results_v1_real.json"},
        },
        ttl=ttl,
    )


async def _run_one_cycle(job_id: str, owner_id: str, project_id: str) -> None:
    """Drive the poller through terminal detection, then stop it."""
    task = asyncio.create_task(
        poll_and_republish_progress(job_id, owner_id, project_id, poll_interval=0.05),
        name=f"test-poller-{job_id[:8]}",
    )
    # The poller `break`s out of the loop on terminal; give it room to do so.
    for _ in range(40):
        await asyncio.sleep(0.02)
        if task.done():
            break
    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_claim_not_taken_when_materialize_unavailable() -> None:
    """Transient RunOutputNotAvailable must leave the completion claim UNTAKEN.

    Against the buggy order the claim is already set when materialization
    fails, and the reconciler then skips the job forever — the completion turn
    is permanently lost.
    """
    job_id = f"test-job-{uuid.uuid4().hex[:8]}"
    project_id = "test-project"
    owner_id = "test-owner"
    store = get_platform().metadata_store
    await _seed_terminal_job(store, job_id, project_id, owner_id)

    claim_calls: list[str] = []
    real_claim = store.claim_completion_fired

    async def _spy_claim(user_id, proj_id, jid):
        claim_calls.append(jid)
        return await real_claim(user_id, proj_id, jid)

    async def _unavailable(*_args, **_kwargs):
        raise RunOutputNotAvailable("S3 object has not landed yet")

    with (
        patch("app.services.job_progress_republisher.get_ws_hub") as mock_hub,
        patch.object(store, "claim_completion_fired", side_effect=_spy_claim),
        patch(
            "app.services.run_output_snapshot.materialize_detached_run",
            side_effect=_unavailable,
        ),
    ):
        mock_hub.return_value.publish_to_user = AsyncMock()
        await _run_one_cycle(job_id, owner_id, project_id)

    assert claim_calls == [], (
        "Completion claim must NOT be taken when the run output cannot be "
        f"materialized — the claim is one-way and the reconciler skips claimed "
        f"jobs, so this permanently loses the completion turn. Claims: {claim_calls}"
    )

    # The durable record must still look unclaimed so recovery can fire later.
    record = await store.get_job(owner_id, project_id, job_id)
    assert not record.get("completion_fired"), (
        "Job record must remain unclaimed so the reconciler can recover it."
    )


@pytest.mark.asyncio
async def test_claim_taken_and_turn_fired_when_materialize_succeeds() -> None:
    """Happy path still claims exactly once and fires the completion turn.

    Counterpart assertion: reordering must not disable the claim, or two firers
    could both fire the same completion turn.
    """
    job_id = f"test-job-{uuid.uuid4().hex[:8]}"
    project_id = "test-project"
    owner_id = "test-owner"
    store = get_platform().metadata_store
    await _seed_terminal_job(store, job_id, project_id, owner_id)

    claim_calls: list[str] = []
    real_claim = store.claim_completion_fired

    async def _spy_claim(user_id, proj_id, jid):
        claim_calls.append(jid)
        return await real_claim(user_id, proj_id, jid)

    async def _materialized(*_args, **_kwargs):
        return "/tmp/eval_results_v1_real.json"

    fired: list[tuple[str, object]] = []

    async def _capture_turn(proj_id, job):
        fired.append((proj_id, job))

    with (
        patch("app.services.job_progress_republisher.get_ws_hub") as mock_hub,
        patch.object(store, "claim_completion_fired", side_effect=_spy_claim),
        patch(
            "app.services.run_output_snapshot.materialize_detached_run",
            side_effect=_materialized,
        ),
        patch(
            "app.services.run_output_snapshot.load_materialized_results",
            return_value=[{"model": "m", "cases": []}],
        ),
        # Patch on the republisher module: it binds the symbol at module scope
        # (`from app.services.completion_turn import request_completion_turn`),
        # so patching the source module would not affect the live reference.
        patch(
            "app.services.job_progress_republisher.request_completion_turn",
            side_effect=_capture_turn,
        ),
        patch("app.services.jobs.eval_executor._dispatch_run_report", AsyncMock()),
        patch(
            "app.services.jobs.eval_executor._dispatch_exec_if_winner_changed",
            AsyncMock(),
        ),
    ):
        mock_hub.return_value.publish_to_user = AsyncMock()
        await _run_one_cycle(job_id, owner_id, project_id)

    assert claim_calls == [job_id], (
        f"Happy path must claim exactly once, got: {claim_calls}"
    )
    assert len(fired) == 1, f"Completion turn must fire exactly once, got {len(fired)}"
    fired_job = fired[0][1]
    assert fired_job.result == [{"model": "m", "cases": []}]
    assert fired_job.output_file == "eval_results_v1_real.json"
