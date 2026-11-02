# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for startup recovery gating and poller reattachment.

Tests that the new-path startup branch:
- DISABLES the dev-only hooks (_recover_orphaned_evals, _fail_stale_dev_job_records)
- REATTACHES pollers to surviving workers via _reattach_surviving_workers
- STARTS the reconciler loop

And the old-path branch keeps existing behavior intact.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import at module level — app.main is already loaded by conftest and other
# tests. We drive behavior purely by patching the functions lifespan calls
# and the os.environ discriminator, without reloading the module.
from app.main import lifespan, _reattach_surviving_workers
from fastapi import FastAPI


@pytest.mark.asyncio
async def test_startup_recovery_gated_new_path():
    """New-path (cloud mode): killers disabled, reattach + reconciler enabled."""
    # Use patch.dict to set EVAL_STUDIO_MODE=cloud cleanly for lifespan's discriminator
    with patch.dict("os.environ", {"EVAL_STUDIO_MODE": "cloud"}):
        with (
            patch("app.main._recover_orphaned_evals", new_callable=AsyncMock) as mock_evals,
            patch("app.main._fail_stale_dev_job_records", new_callable=AsyncMock) as mock_fail_stale,
            patch("app.main._reattach_surviving_workers", new_callable=AsyncMock) as mock_reattach,
            patch("app.services.eval_reconciler.start_reconciler_loop") as mock_reconciler_start,
            patch("app.services.eval_reconciler.stop_reconciler_loop") as mock_reconciler_stop,
            patch("app.main._recover_pending_deletes", new_callable=AsyncMock),
            patch("app.services.ws_hub.get_ws_hub"),
            patch("app.services.bedrock_mantle_model._verify_mantle_tls", new_callable=AsyncMock),
            patch("app.services.mcp_server_cache.pre_warm_mcp_server"),
            # Patched at its own module: lifespan imports it function-locally, so
            # an ``app.main.`` target would not bind.
            patch(
                "app.services.agentcore_session_reconciler.reconcile_orphaned_agentcore_claims",
                new_callable=AsyncMock,
                return_value={"claims_cleared": 1},
            ) as mock_claim_reconcile,
            # Patched at its own module: lifespan imports it function-locally.
            patch(
                "app.services.optimization_loop_orchestrator.adopt_orphan_optimization_loops",
                new_callable=AsyncMock,
                return_value=0,
            ) as mock_adopt_loops,
        ):
            app = FastAPI()
            async with lifespan(app):
                # Assertions: new-path branch behavior
                # Dev-only hooks NOT called
                mock_evals.assert_not_called()
                mock_fail_stale.assert_not_called()

                # ...but loop ADOPTION must run here (dispatch-to-owner,
                # successor of the deleted flip-to-cancelled durable recovery).
                # A record left live by a restart or a microVM chat dispatch
                # gets its loop back at boot; without this, the record wedges
                # the project (the dispatch fence refuses every later eval and
                # optimization) until the reconciler's first pass — or, before
                # the seam, until someone edited the table by hand.
                mock_adopt_loops.assert_awaited_once()

                # Reattach + reconciler CALLED
                mock_reattach.assert_awaited_once()
                mock_reconciler_start.assert_called_once()

                # The AgentCore claim reconcile belongs here: cloud is the only
                # place a relay claim can exist. A non-zero cleared count is
                # deliberate — the success log line only runs when claims were
                # cleared, so a zero count would not exercise it and this test
                # would pass against an unbound logger name.
                mock_claim_reconcile.assert_awaited_once()

            # After exiting the context, reconciler stop should be called
            mock_reconciler_stop.assert_called_once()


@pytest.mark.asyncio
async def test_cloud_startup_survives_raising_claim_reconciler():
    """A raising AgentCore claim reconcile degrades to a warning, never blocking boot.

    Covers the ``except`` branch, whose own body logs — so an unbound logger name
    there turns a swallowed reconcile failure into a failed startup.
    """
    with patch.dict("os.environ", {"EVAL_STUDIO_MODE": "cloud"}):
        with (
            patch("app.main._reattach_surviving_workers", new_callable=AsyncMock),
            patch("app.services.eval_reconciler.start_reconciler_loop"),
            patch("app.services.eval_reconciler.stop_reconciler_loop"),
            patch("app.main._recover_pending_deletes", new_callable=AsyncMock),
            patch("app.services.ws_hub.get_ws_hub"),
            patch("app.services.bedrock_mantle_model._verify_mantle_tls", new_callable=AsyncMock),
            patch("app.services.mcp_server_cache.pre_warm_mcp_server"),
            patch(
                "app.services.agentcore_session_reconciler.reconcile_orphaned_agentcore_claims",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ) as mock_claim_reconcile,
        ):
            app = FastAPI()
            # Completing this block at all is the assertion: the failure must not
            # propagate out of lifespan.
            async with lifespan(app):
                mock_claim_reconcile.assert_awaited_once()


@pytest.mark.asyncio
async def test_startup_recovery_gated_old_path():
    """Old-path (dev mode): killers enabled, reattach + reconciler NOT called."""
    with patch.dict("os.environ", {"EVAL_STUDIO_MODE": "dev"}):
        with (
            patch("app.main._recover_orphaned_evals", new_callable=AsyncMock) as mock_evals,
            patch("app.main._fail_stale_dev_job_records", new_callable=AsyncMock) as mock_fail_stale,
            patch("app.main._reattach_surviving_workers", new_callable=AsyncMock) as mock_reattach,
            patch("app.services.eval_reconciler.start_reconciler_loop") as mock_reconciler_start,
            patch("app.main._recover_pending_deletes", new_callable=AsyncMock),
            # Patched at its own module: the dev branch imports it function-locally.
            patch(
                "app.services.optimization_loop_orchestrator.adopt_orphan_optimization_loops",
                new_callable=AsyncMock,
                return_value=0,
            ) as mock_adopt_loops,
            patch("app.services.ws_hub.get_ws_hub"),
            patch("app.main.PROJECTS_DIR", MagicMock(mkdir=MagicMock())),
            patch("app.services.bedrock_mantle_model._verify_mantle_tls", new_callable=AsyncMock),
            patch("app.services.mcp_server_cache.pre_warm_mcp_server"),
            patch(
                "app.services.agentcore_session_reconciler.reconcile_orphaned_agentcore_claims",
                new_callable=AsyncMock,
            ) as mock_claim_reconcile,
        ):
            app = FastAPI()
            async with lifespan(app):
                # Assertions: old-path branch behavior
                # Dev hooks CALLED: orphan-eval killer, job-corpse failer,
                # then the SAME adoption sweep cloud runs (dev adopts
                # stale loops instead of cancelling them).
                mock_evals.assert_awaited_once()
                mock_fail_stale.assert_awaited_once()
                mock_adopt_loops.assert_awaited_once()

                # Reattach + reconciler NOT called
                mock_reattach.assert_not_called()
                mock_reconciler_start.assert_not_called()

                # Wiring assertion for the relocation: dev can never hold a relay
                # claim (the relay resolver is forced off outside cloud), so
                # reconciling here would only ever scan an empty table.
                mock_claim_reconcile.assert_not_called()


@pytest.mark.asyncio
async def test_reattach_poller():
    """_reattach_surviving_workers delegates to adopt_unpolled_jobs.

    The ECS-ARN filtering happens inside adopt_unpolled_jobs, which is mocked
    here; the seeded jobs only document the input shape.
    """
    # Seed three non-terminal jobs: a Fargate handle, no handle, and a PID handle
    mock_store = MagicMock()
    mock_store.list_non_terminal_jobs = AsyncMock(return_value=[
        {
            "job_id": "job-with-arn-123",
            "project_id": "proj-123",
            "owner_id": "user-456",
            "compute_handle": {
                "kind": "fargate_task",
                "value": "arn:aws:ecs:us-east-1:123456789012:task/cluster/abc123",
            },
            "config": {"source": "manual"},
            "state": "running",
        },
        {
            "job_id": "job-no-handle-456",
            "project_id": "proj-789",
            "owner_id": "user-999",
            # No compute_handle
            "config": {"source": "manual"},
            "state": "running",
        },
        {
            "job_id": "job-pid-handle-789",
            "project_id": "proj-abc",
            "owner_id": "user-xyz",
            "compute_handle": {
                "kind": "subprocess",
                "value": "12345",
            },
            "config": {"source": "manual"},
            "state": "running",
        },
    ])

    with (
        patch("app.platform.get_platform") as mock_get_platform,
        patch("app.services.job_progress_republisher.adopt_unpolled_jobs", new_callable=AsyncMock) as mock_adopt,
    ):
        mock_platform = MagicMock()
        mock_platform.metadata_store = mock_store
        mock_get_platform.return_value = mock_platform

        # Mock adopt to return 1 (one job adopted - the Fargate one)
        mock_adopt.return_value = 1

        # Call the reattach function directly — no module reload needed
        await _reattach_surviving_workers()

        # Assert adopt_unpolled_jobs called once
        mock_adopt.assert_awaited_once_with()
