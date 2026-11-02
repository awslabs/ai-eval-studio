# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mid-eval control-plane restart: surviving workers are re-adopted.

After a restart, ``_reattach_surviving_workers`` hands recovery to the adopt
sweep that re-establishes pollers for surviving workers. The second half shows
the first-wins contract the poller relies on to fire the completion turn
exactly once (``claim_completion_fired``), against a stand-in store.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import at module level — app.main is already loaded by conftest and other
# tests. We drive behavior purely by patching without reloading the module.
from app.main import _reattach_surviving_workers


@pytest.mark.asyncio
async def test_mid_eval_deploy_survival():
    """Restart + reattach runs the adopt sweep; the completion claim admits one winner."""
    # What this test checks:
    # 1. After a simulated restart (_reattach_surviving_workers), the adopt sweep
    #    that re-establishes surviving workers' pollers runs.
    # 2. The first-wins claim contract: the first claim_completion_fired call wins
    #    and a second observer's call loses. This half runs against a stand-in
    #    store, not a real one.

    job_id = "test-job-123"
    project_id = "test-proj-456"
    owner_id = "test-user-789"
    compute_handle = "arn:aws:ecs:us-east-1:123456789012:task/cluster/test-task"

    # Mock job record (running, with ECS ARN compute_handle)
    mock_job = {
        "job_id": job_id,
        "project_id": project_id,
        "owner_id": owner_id,
        "user_id": owner_id,
        "compute_handle": compute_handle,
        "state": "running",
    }

    # Mock store
    mock_store = MagicMock()
    mock_store.list_non_terminal_jobs = AsyncMock(return_value=[mock_job])

    # Track adopt_unpolled_jobs calls (the reattach action now delegates to the sweep)
    adopt_calls = []

    with (
        patch("app.platform.get_platform") as mock_get_platform,
        patch("app.services.job_progress_republisher.adopt_unpolled_jobs", new_callable=AsyncMock) as mock_adopt,
    ):
        mock_platform = MagicMock()
        mock_platform.metadata_store = mock_store
        mock_get_platform.return_value = mock_platform

        # Record adopt sweep calls
        async def record_adopt():
            adopt_calls.append(True)
            return 1  # Return count of adopted jobs

        mock_adopt.side_effect = record_adopt

        # Call the reattach function directly (simulates startup recovery)
        await _reattach_surviving_workers()

        # ASSERTION 1: Adopt sweep was called for startup recovery
        assert len(adopt_calls) == 1, f"Expected 1 adopt sweep call, got {len(adopt_calls)}"

    # The first-wins contract the poller relies on, shown against a stand-in
    # store: a re-attached poller and a concurrent observer must not both fire.
    mock_store_claim = MagicMock()

    # Track claim_completion_fired calls
    claim_calls = []
    async def mock_claim(user_id, project_id_arg, job_id_arg):
        """Return True on first call, False on subsequent (exactly-once)."""
        claim_calls.append((user_id, project_id_arg, job_id_arg))
        # First caller wins
        return len(claim_calls) == 1

    mock_store_claim.claim_completion_fired = AsyncMock(side_effect=mock_claim)

    with patch("app.services.job_progress_republisher.get_platform") as mock_get_platform_claim:
        mock_get_platform_claim.return_value.metadata_store = mock_store_claim

        # Simulate two observers (re-attached poller + a second concurrent observer)
        # Both try to claim, only one wins
        won_first = await mock_store_claim.claim_completion_fired(owner_id, project_id, job_id)
        won_second = await mock_store_claim.claim_completion_fired(owner_id, project_id, job_id)

        # ASSERTION 2: First claim wins, second claim loses
        assert won_first is True, "First observer should win the claim"
        assert won_second is False, "Second observer should lose the claim (exactly-once)"
        assert len(claim_calls) == 2, "Both observers should call claim_completion_fired"

    # Only the claim winner fires the completion turn. That branch lives in
    # poll_and_republish_progress and is covered by
    # test_manual_terminal_lost_claim_does_not_fire in test_job_progress_republisher.py.
