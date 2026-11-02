# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the reconciler's ``pass_failed`` attribution flag.

``reconcile_once`` wraps its triple-duty body in a blanket ``except`` so a
transient AWS error can never kill the periodic loop. That safety has a
cost: a pass that aborted on its first call and a pass that legitimately
found nothing to reap both return ``stuck_jobs_failed=0``, and the operator
log line reads identically for each.

That ambiguity misleads: a log showing ``stuck_jobs_failed=0`` for a whole
window reads as "the reconciler looked and found nothing" when a thrown pass
would look the same. ``pass_failed`` is what
separates the two, so these tests pin BOTH directions: a raising pass must
report True, and a clean pass must report False. A test covering only one
direction would pass against a field hardcoded to that value.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _fargate_compute_mock() -> MagicMock:
    """Build a compute mock that passes the ``isinstance`` dev-mode gate.

    ``reconcile_once`` returns early unless compute is a ``FargateTaskCompute``
    (dev has no detached workers), so the spec must satisfy that check for the
    body under test to run at all.
    """
    from app.platform.aws.fargate_task_compute import FargateTaskCompute

    return MagicMock(spec=FargateTaskCompute)


@pytest.mark.asyncio
async def test_thrown_pass_reports_pass_failed_true():
    """A pass that raises mid-body must report ``pass_failed=True``.

    Without the flag this returns the same ``stuck_jobs_failed=0`` as a
    healthy pass, which lets a silently failing reconciler look healthy.
    """
    from app.services import eval_reconciler

    compute = _fargate_compute_mock()
    # Fail the first await inside the guarded body — the same shape as a
    # transient ECS/IAM error in production.
    compute.list_running_workers = AsyncMock(
        side_effect=RuntimeError("ECS ListTasks denied")
    )

    with (
        patch.object(eval_reconciler, "get_job_compute", return_value=compute),
        patch.object(eval_reconciler, "get_platform") as mock_platform,
        patch.object(
            eval_reconciler, "is_cost_kill_switch_active", return_value=False
        ),
        patch.object(eval_reconciler, "logger"),
    ):
        mock_platform.return_value.metadata_store = MagicMock()
        result = await eval_reconciler.reconcile_once()

    assert result["pass_failed"] is True, (
        "a pass that raised must be distinguishable from a clean zero-reap pass"
    )
    # The counters stay zero — which is precisely why the flag is needed.
    assert result["stuck_jobs_failed"] == 0


@pytest.mark.asyncio
async def test_clean_pass_reports_pass_failed_false():
    """A pass that completes with nothing to reap must report False.

    This is the other half of the contract: the flag must track the actual
    outcome, not be pinned True by construction.
    """
    from app.services import eval_reconciler

    compute = _fargate_compute_mock()
    compute.list_running_workers = AsyncMock(return_value=[])  # no workers

    store = MagicMock()
    store.list_non_terminal_jobs = AsyncMock(return_value=[])  # nothing stuck

    with (
        patch.object(eval_reconciler, "get_job_compute", return_value=compute),
        patch.object(eval_reconciler, "get_platform") as mock_platform,
        patch.object(
            eval_reconciler, "is_cost_kill_switch_active", return_value=False
        ),
        patch.object(eval_reconciler, "logger"),
    ):
        mock_platform.return_value.metadata_store = store
        result = await eval_reconciler.reconcile_once()

    assert result["pass_failed"] is False, (
        "a completed pass with nothing to reap must not look like a failure"
    )
    assert result["stuck_jobs_failed"] == 0


@pytest.mark.asyncio
async def test_pass_failed_is_rendered_in_the_operator_log():
    """The flag must reach the log line, not just the return value.

    The return value is consumed by the periodic loop, which discards it —
    the operator only ever sees the log. A flag that never renders would
    leave the ambiguity exactly as it was.
    """
    from app.services import eval_reconciler

    compute = _fargate_compute_mock()
    compute.list_running_workers = AsyncMock(
        side_effect=RuntimeError("ECS ListTasks denied")
    )

    with (
        patch.object(eval_reconciler, "get_job_compute", return_value=compute),
        patch.object(eval_reconciler, "get_platform") as mock_platform,
        patch.object(
            eval_reconciler, "is_cost_kill_switch_active", return_value=False
        ),
        patch.object(eval_reconciler, "logger") as mock_logger,
    ):
        mock_platform.return_value.metadata_store = MagicMock()
        await eval_reconciler.reconcile_once()

        # Find the pass-complete summary line and confirm the flag is one of
        # its arguments. Asserting on the args (not the rendered prose) keeps
        # this from breaking on harmless wording changes.
        rendered = [
            (call.args[0], call.args[1:])
            for call in mock_logger.info.call_args_list
            if call.args and "Pass complete" in str(call.args[0])
        ]
        assert rendered, "the pass-complete summary must be logged"
        fmt, fmt_args = rendered[0]
        assert "pass_failed" in fmt, (
            "the operator log must name pass_failed so a thrown pass is legible"
        )
        assert True in fmt_args, (
            "the logged pass_failed value must reflect the thrown pass"
        )
