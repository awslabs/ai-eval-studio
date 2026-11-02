# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests: both dispatch sites route through JobCompute.

Verifies that dispatch_eval (standalone path) AND dispatch_eval_job
(optimization path) both call JobCompute.launch when the arch_split flag is ON,
and both preserve the in-process path when the flag is OFF.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.platform.job_compute import JobComputeHandle


@pytest.mark.asyncio
async def test_dispatch_eval_routes_through_job_compute_when_flag_on():
    """With arch_split=True, dispatch_eval calls JobCompute.launch + persists compute_handle."""
    from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
    dispatch_eval = _dispatch_eval_tool.handler

    # Mock dependencies
    fake_handle = JobComputeHandle(kind="fargate_task", value="arn:aws:ecs:us-east-1:123456789012:task/abc")
    fake_compute = AsyncMock()
    fake_compute.launch = AsyncMock(return_value=fake_handle)

    with (
        patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve,
        patch("app.services.mcp_tools.dispatch.get_job_tracker") as mock_tracker,
        patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
        patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
        patch("app.services.mcp_tools.dispatch._preflight_models_invocable", return_value=[]),
        patch("app.platform.arch_split.resolve_arch_split", return_value=True),
        patch("app.platform.job_compute.get_job_compute", return_value=fake_compute),
        patch("app.services.mcp_tools.dispatch._publish_project_event"),
        patch("app.platform.types.current_request") as mock_req,
    ):
        mock_resolve.return_value = MagicMock()
        mock_resolve.return_value.__truediv__ = lambda s, o: MagicMock(exists=lambda: True)

        mock_tracker_inst = MagicMock()
        mock_job = MagicMock(job_id="job-123", project_id="proj-abc")
        mock_tracker_inst.create_job.return_value = mock_job
        mock_tracker_inst.get_active_job_by_type.return_value = None
        mock_tracker.return_value = mock_tracker_inst

        mock_req.return_value = MagicMock(user_id="user-xyz")

        # Mock the metadata store to capture the compute_handle write
        captured_handle = None

        async def mock_update_job_state(owner_id, project_id, job_id, **kwargs):
            nonlocal captured_handle
            captured_handle = kwargs.get("compute_handle")

        mock_store_inst = AsyncMock()
        mock_store_inst.update_job_state = AsyncMock(side_effect=mock_update_job_state)
        # NEW PATH store contract (dispatch guards)
        mock_store_inst.count_active_jobs_for_owner = AsyncMock(return_value=0)
        mock_store_inst.list_non_terminal_jobs = AsyncMock(return_value=[])
        mock_store_inst.claim_eval_lock = AsyncMock(return_value=True)
        # Idempotency path not reached when key omitted, but set defensively
        mock_store_inst.claim_idempotency_key = AsyncMock(return_value=(True, None))
        # The optimization-live fence reads this first. An AsyncMock would hand
        # back a MagicMock the state loader then treats as a real record.
        mock_store_inst.get_optimization_state = AsyncMock(return_value=None)
        # Unclaimed launch gate: echo the claimant back, as the real store
        # does for the first caller. An AsyncMock would invent a MagicMock
        # here, which reads as someone else's claim and silently skips the
        # launch this test is about.
        mock_store_inst.claim_launch_gate = AsyncMock(
            side_effect=lambda *a, claimant, **k: claimant
        )

        mock_platform = MagicMock()
        mock_platform.metadata_store = mock_store_inst

        with patch("app.platform.get_platform", return_value=mock_platform):
            result = await dispatch_eval({
                "project_id": "proj-abc",
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["anthropic.claude-3-sonnet-20240229-v1:0"],
            })

        # Assertions - extract JSON from MCP envelope
        import json
        result_json = json.loads(result["content"][0]["text"])
        assert result_json.get("job_id") == "job-123"
        fake_compute.launch.assert_called_once()
        assert captured_handle == str(fake_handle)


@pytest.mark.asyncio
async def test_dispatch_eval_preserves_old_path_when_flag_off():
    """With arch_split=False, dispatch_eval uses asyncio.create_task (old path)."""
    from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
    dispatch_eval = _dispatch_eval_tool.handler

    fake_compute = AsyncMock()
    fake_compute.launch = AsyncMock()  # Should NOT be called

    with (
        patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve,
        patch("app.services.mcp_tools.dispatch.get_job_tracker") as mock_tracker,
        patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
        patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
        patch("app.services.mcp_tools.dispatch._preflight_models_invocable", return_value=[]),
        patch("app.platform.arch_split.resolve_arch_split", return_value=False),
        patch("app.platform.job_compute.get_job_compute", return_value=fake_compute),
        patch("app.services.mcp_tools.dispatch.asyncio.create_task") as mock_create_task,
        patch("app.services.mcp_tools.dispatch.attach_error_sink"),
        patch("app.services.mcp_tools.dispatch._publish_project_event"),
        patch("app.platform.types.current_request") as mock_req,
    ):
        mock_resolve.return_value = MagicMock()
        mock_resolve.return_value.__truediv__ = lambda s, o: MagicMock(exists=lambda: True)

        mock_tracker_inst = MagicMock()
        mock_job = MagicMock(job_id="job-123", project_id="proj-abc")
        mock_tracker_inst.create_job.return_value = mock_job
        mock_tracker_inst.get_active_job_by_type.return_value = None
        mock_tracker.return_value = mock_tracker_inst

        mock_req.return_value = MagicMock(user_id="user-xyz")

        result = await dispatch_eval({
            "project_id": "proj-abc",
            "data_file": "test.csv",
            "prompt_version": "baseline",
            "target_model_ids": ["anthropic.claude-3-sonnet-20240229-v1:0"],
        })

        # Old path assertions - extract JSON from MCP envelope
        import json
        result_json = json.loads(result["content"][0]["text"])
        assert result_json.get("job_id") == "job-123"
        fake_compute.launch.assert_not_called()
        mock_create_task.assert_called_once()


@pytest.mark.asyncio
async def test_optimization_dispatch_routes_through_job_compute_when_flag_on(tmp_path):
    """With arch_split=True, dispatch_eval_job calls JobCompute.launch + persists compute_handle."""
    from app.services.optimization_orchestrator import dispatch_eval_job, OptimizationConfig
    from pathlib import Path

    fake_handle = JobComputeHandle(kind="fargate_task", value="arn:aws:ecs:us-east-1:123456789012:task/xyz")
    fake_compute = AsyncMock()
    fake_compute.launch = AsyncMock(return_value=fake_handle)

    config = OptimizationConfig(
        project_id="proj-abc",
        owner_id="user-xyz",
        model_ids=["model-1"],
        data_file="train.csv",
        field_mappings={},
        prompt_version="v1",
        judge_model_id=None,
        target_metric_scores={},
        max_iterations=5,
        plateau_window=3,
    )

    project_dir = tmp_path / "workspace" / "proj-abc"

    captured_handle = None

    async def mock_update_job_state(owner_id, project_id, job_id, **kwargs):
        nonlocal captured_handle
        captured_handle = kwargs.get("compute_handle")

    mock_store_inst = AsyncMock()
    mock_store_inst.update_job_state = AsyncMock(side_effect=mock_update_job_state)
    # Unclaimed launch gate: echo the claimant back, as the real store
    # does for the first caller. An AsyncMock would invent a MagicMock
    # here, which reads as someone else's claim and silently skips the
    # launch this test is about.
    mock_store_inst.claim_launch_gate = AsyncMock(
        side_effect=lambda *a, claimant, **k: claimant
    )

    mock_platform = MagicMock()
    mock_platform.metadata_store = mock_store_inst

    with (
        patch("app.platform.job_compute.get_job_compute", return_value=fake_compute),
        patch("app.platform.get_platform", return_value=mock_platform),
        patch("pathlib.Path.exists", return_value=True),
    ):
        # Pass arch_split=True explicitly (thread resolved flag, never re-resolve)
        job_id = await dispatch_eval_job(project_dir, config, "v1", arch_split=True)

        # Assertions
        # Job ID is generated by JobTracker, so just verify seam was called
        assert job_id  # non-empty
        fake_compute.launch.assert_called_once()
        call_kwargs = fake_compute.launch.call_args.kwargs
        assert call_kwargs["job_id"] == job_id
        assert call_kwargs["owner_id"] == "user-xyz"
        assert call_kwargs["project_id"] == "proj-abc"
        assert captured_handle == str(fake_handle)


@pytest.mark.asyncio
async def test_optimization_dispatch_preserves_old_path_when_flag_off(tmp_path):
    """With arch_split=False, dispatch_eval_job uses asyncio.create_task (old path)."""
    from app.services.optimization_orchestrator import dispatch_eval_job, OptimizationConfig
    from pathlib import Path

    fake_compute = AsyncMock()
    fake_compute.launch = AsyncMock()  # Should NOT be called

    config = OptimizationConfig(
        project_id="proj-abc",
        owner_id="user-xyz",
        model_ids=["model-1"],
        data_file="train.csv",
        field_mappings={},
        prompt_version="v1",
        judge_model_id=None,
        target_metric_scores={},
        max_iterations=5,
        plateau_window=3,
    )

    project_dir = tmp_path / "workspace" / "proj-abc"

    with (
        patch("app.platform.job_compute.get_job_compute", return_value=fake_compute),
        patch("app.services.optimization_orchestrator.asyncio.create_task") as mock_create_task,
        patch("pathlib.Path.exists", return_value=True),
    ):
        # Pass arch_split=False explicitly (thread resolved flag, never re-resolve)
        job_id = await dispatch_eval_job(project_dir, config, "v1", arch_split=False)

        # Old path assertions
        assert job_id  # non-empty
        fake_compute.launch.assert_not_called()
        mock_create_task.assert_called_once()


# ---------------------------------------------------------------------------
# Cancel via compute_handle (StopTask) for Fargate jobs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_fargate_job_routes_through_stop_task():
    """A job with a task-ARN compute_handle cancels via JobCompute.cancel (StopTask)."""
    from app.api import eval_jobs

    task_arn = "arn:aws:ecs:us-east-1:123456789012:task/cluster/abc123"
    fake_compute = AsyncMock()
    fake_compute.cancel = AsyncMock()
    # StopTask returning is only ACCEPTANCE; the cancel path now confirms
    # lastStatus == STOPPED before treating the worker as gone, because ECS allows
    # a 30s stopTimeout during which the task keeps running.
    fake_compute.status = AsyncMock(return_value="STOPPED")

    mock_store = AsyncMock()
    mock_store.get_job = AsyncMock(return_value={"compute_handle": task_arn})
    mock_platform = MagicMock()
    mock_platform.metadata_store = mock_store

    mock_job = MagicMock(status="running", pid=None)
    mock_tracker = MagicMock()
    mock_tracker.get_job_for_project.return_value = mock_job

    with (
        patch("app.api.eval_jobs.get_project_dir_or_404", return_value=MagicMock()),
        patch("app.services.eval_cancel.get_job_tracker", return_value=mock_tracker),
        patch("app.platform.get_platform", return_value=mock_platform),
        patch("app.platform.job_compute.get_job_compute", return_value=fake_compute),
        patch("app.services.harness_runner.kill_harness_subprocess") as mock_kill,
        patch("app.services.optimization_state.OptimizationState.load", return_value=None),
    ):
        result = await eval_jobs.cancel_eval_job("proj-abc", "job-123", "user-xyz")

        assert result["status"] == "cancelled"
        # Fargate path: StopTask called with the full ARN, PID kill NOT called
        fake_compute.cancel.assert_called_once()
        cancel_handle = fake_compute.cancel.call_args.args[0]
        assert cancel_handle.value == task_arn
        assert cancel_handle.kind == "fargate_task"
        mock_kill.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_local_job_routes_through_pid_kill():
    """A job with a PID and no task-ARN compute_handle cancels via kill_harness_subprocess."""
    from app.api import eval_jobs

    fake_compute = AsyncMock()
    fake_compute.cancel = AsyncMock()
    # StopTask returning is only ACCEPTANCE; the cancel path now confirms
    # lastStatus == STOPPED before treating the worker as gone, because ECS allows
    # a 30s stopTimeout during which the task keeps running.
    fake_compute.status = AsyncMock(return_value="STOPPED")

    mock_store = AsyncMock()
    mock_store.get_job = AsyncMock(return_value={"compute_handle": None})
    mock_platform = MagicMock()
    mock_platform.metadata_store = mock_store

    mock_job = MagicMock(status="running", pid=4242)
    mock_tracker = MagicMock()
    mock_tracker.get_job_for_project.return_value = mock_job

    with (
        patch("app.api.eval_jobs.get_project_dir_or_404", return_value=MagicMock()),
        patch("app.services.eval_cancel.get_job_tracker", return_value=mock_tracker),
        patch("app.platform.get_platform", return_value=mock_platform),
        patch("app.platform.job_compute.get_job_compute", return_value=fake_compute),
        patch("app.services.harness_runner.kill_harness_subprocess") as mock_kill,
        patch("app.services.optimization_state.OptimizationState.load", return_value=None),
    ):
        result = await eval_jobs.cancel_eval_job("proj-abc", "job-123", "user-xyz")

        assert result["status"] == "cancelled"
        # Local path: PID kill called, seam NOT called
        mock_kill.assert_called_once_with(4242)
        fake_compute.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_survives_control_plane_restart():
    """With the in-memory Job absent, cancel still reads compute_handle from the store.

    Simulates a control-plane restart: the durable job record carries the
    task ARN even though the in-memory tracker was rebuilt. Cancel resolves
    the handle from the store and StopTasks.
    """
    from app.api import eval_jobs

    task_arn = "arn:aws:ecs:us-east-1:123456789012:task/cluster/restart-xyz"
    fake_compute = AsyncMock()
    fake_compute.cancel = AsyncMock()
    # StopTask returning is only ACCEPTANCE; the cancel path now confirms
    # lastStatus == STOPPED before treating the worker as gone, because ECS allows
    # a 30s stopTimeout during which the task keeps running.
    fake_compute.status = AsyncMock(return_value="STOPPED")

    # Durable store still has the handle after a restart
    mock_store = AsyncMock()
    mock_store.get_job = AsyncMock(return_value={"compute_handle": task_arn})
    mock_platform = MagicMock()
    mock_platform.metadata_store = mock_store

    # The rebuilt in-memory Job has no PID (worker runs remotely on Fargate)
    mock_job = MagicMock(status="running", pid=None)
    mock_tracker = MagicMock()
    mock_tracker.get_job_for_project.return_value = mock_job

    with (
        patch("app.api.eval_jobs.get_project_dir_or_404", return_value=MagicMock()),
        patch("app.services.eval_cancel.get_job_tracker", return_value=mock_tracker),
        patch("app.platform.get_platform", return_value=mock_platform),
        patch("app.platform.job_compute.get_job_compute", return_value=fake_compute),
        patch("app.services.harness_runner.kill_harness_subprocess") as mock_kill,
        patch("app.services.optimization_state.OptimizationState.load", return_value=None),
    ):
        result = await eval_jobs.cancel_eval_job("proj-abc", "job-123", "user-xyz")

        assert result["status"] == "cancelled"
        # Cancel read the durable handle and StopTask'd — no PID kill
        fake_compute.cancel.assert_called_once()
        mock_kill.assert_not_called()


def test_is_fargate_handle_discriminates_on_arn_prefix():
    """_is_fargate_handle recognizes task ARNs and rejects PID/None handles."""
    from app.api.eval_jobs import _is_fargate_handle

    assert _is_fargate_handle("arn:aws:ecs:us-east-1:123456789012:task/c/abc") is True
    assert _is_fargate_handle(None) is False
    assert _is_fargate_handle("") is False
    assert _is_fargate_handle("4242") is False  # PID-shaped
    assert _is_fargate_handle("job-abc-123") is False  # local job_id handle


# ---------------------------------------------------------------------------
# dispatch_optimization + executor cancel-poll route through DynamoDB
# ---------------------------------------------------------------------------


def _make_dispatch_opt_mocks(existing_state, put_return=1):
    """Build the common mock set for dispatch_optimization DDB-path tests.

    Returns (patches_ctx_factory, mock_store) where mock_store exposes
    get_optimization_state (returns ``existing_state``) and
    put_optimization_state (returns ``put_return``).
    """
    from unittest.mock import AsyncMock, MagicMock

    mock_store = AsyncMock()
    mock_store.get_optimization_state = AsyncMock(return_value=existing_state)
    mock_store.put_optimization_state = AsyncMock(return_value=put_return)
    # Dispatch-guard store contract
    mock_store.count_active_jobs_for_owner = AsyncMock(return_value=0)
    mock_store.list_non_terminal_jobs = AsyncMock(return_value=[])
    mock_store.get_eval_lock = AsyncMock(return_value=None)
    return mock_store


@pytest.mark.asyncio
async def test_dispatch_optimization_arch_split_true_creates_via_ddb_no_existing(tmp_path):
    """arch_split=True with NO existing record: create routes to put_optimization_state (expected_version=None)."""
    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock, patch

    from app.services.mcp_tools.dispatch import dispatch_optimization as _tool
    dispatch_optimization = _tool.handler

    mock_store = _make_dispatch_opt_mocks(existing_state=None, put_return=1)
    mock_platform = MagicMock()
    mock_platform.metadata_store = mock_store

    # Harness config with valid fields
    harness = MagicMock()
    harness.models = ["model-1"]
    harness.field_mappings = {"input": "q", "expected_output": "a"}
    harness.target_metric_scores = {"ExactMatch": 0.85}
    harness.max_iterations = 5
    harness.plateau_window = 3
    harness.plateau_threshold = None
    harness.data_file = "seed.csv"
    harness.judge_model_id = None

    with (
        patch("app.services.mcp_tools.dispatch.resolve_project_dir", return_value=tmp_path / "proj"),
        patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
        patch("app.platform.arch_split.resolve_arch_split", new_callable=AsyncMock, return_value=True),
        patch("app.platform.arch_split.is_cost_kill_switch_active", return_value=False),
        patch("app.services.mcp_tools.dispatch.load_harness_config", return_value=harness),
        patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
        patch("app.services.mcp_tools.dispatch._preflight_models_invocable", new_callable=AsyncMock, return_value=[]),
        patch("app.services.mcp_tools.dispatch.get_job_tracker") as mock_tracker,
        patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="user-xyz"),
        patch("app.platform.types.current_request") as mock_req,
        patch("app.platform.get_platform", return_value=mock_platform),
        patch("app.services.optimization_loop_orchestrator.run_optimization_loop", new_callable=AsyncMock),
        patch("app.services.mcp_tools.dispatch.attach_error_sink"),
        patch("app.services.optimization_orchestrator.detect_latest_prompt", return_value="baseline"),
    ):
        mock_tracker.return_value = MagicMock()
        mock_req.return_value = MagicMock(user_id="user-xyz")

        result = await dispatch_optimization({"project_id": "proj-abc"})

        # Verify create routed to put_optimization_state with expected_version=None
        assert mock_store.put_optimization_state.called, "create must route to put_optimization_state on arch_split=True"
        call_kwargs = mock_store.put_optimization_state.call_args.kwargs
        assert call_kwargs.get("expected_version") is None
        # Not an mcp_error
        assert "is_error" not in result or not result["is_error"]


@pytest.mark.asyncio
async def test_dispatch_optimization_arch_split_true_overwrites_terminal_record(tmp_path):
    """arch_split=True with existing TERMINAL record: existing-check allows, create threads expected_version=2."""
    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock, patch

    from app.services.mcp_tools.dispatch import dispatch_optimization as _tool
    dispatch_optimization = _tool.handler

    existing = {
        "version": 2,
        "status": "converged",  # terminal
        "iterations": [],
        "target_metric_scores": {"ExactMatch": 0.85},
        "max_iterations": 5,
        "plateau_window": 3,
        "max_iter_cap": 5,
    }
    mock_store = _make_dispatch_opt_mocks(existing_state=existing, put_return=3)
    mock_platform = MagicMock()
    mock_platform.metadata_store = mock_store

    harness = MagicMock()
    harness.models = ["model-1"]
    harness.field_mappings = {"input": "q", "expected_output": "a"}
    harness.target_metric_scores = {"ExactMatch": 0.85}
    harness.max_iterations = 5
    harness.plateau_window = 3
    harness.plateau_threshold = None
    harness.data_file = "seed.csv"
    harness.judge_model_id = None

    with (
        patch("app.services.mcp_tools.dispatch.resolve_project_dir", return_value=tmp_path / "proj"),
        patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
        patch("app.platform.arch_split.resolve_arch_split", new_callable=AsyncMock, return_value=True),
        patch("app.platform.arch_split.is_cost_kill_switch_active", return_value=False),
        patch("app.services.mcp_tools.dispatch.load_harness_config", return_value=harness),
        patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
        patch("app.services.mcp_tools.dispatch._preflight_models_invocable", new_callable=AsyncMock, return_value=[]),
        patch("app.services.mcp_tools.dispatch.get_job_tracker") as mock_tracker,
        patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="user-xyz"),
        patch("app.platform.types.current_request") as mock_req,
        patch("app.platform.get_platform", return_value=mock_platform),
        patch("app.services.optimization_loop_orchestrator.run_optimization_loop", new_callable=AsyncMock),
        patch("app.services.mcp_tools.dispatch.attach_error_sink"),
        patch("app.services.optimization_orchestrator.detect_latest_prompt", return_value="baseline"),
    ):
        mock_tracker.return_value = MagicMock()
        mock_req.return_value = MagicMock(user_id="user-xyz")

        result = await dispatch_optimization({"project_id": "proj-abc"})

        # Verify put received expected_version=2 (overwrite terminal), no error
        assert mock_store.put_optimization_state.called
        call_kwargs = mock_store.put_optimization_state.call_args.kwargs
        assert call_kwargs.get("expected_version") == 2
        assert "is_error" not in result or not result["is_error"]


@pytest.mark.asyncio
async def test_dispatch_optimization_arch_split_true_running_record_rejects(tmp_path):
    """arch_split=True with a running DDB record: existing-check returns mcp_error (read via get, not EFS)."""
    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock, patch

    from app.services.mcp_tools.dispatch import dispatch_optimization as _tool
    dispatch_optimization = _tool.handler

    existing = {
        "version": 1,
        "status": "running",
        "iterations": [],
        "target_metric_scores": {"ExactMatch": 0.85},
        "max_iterations": 5,
        "plateau_window": 3,
        "max_iter_cap": 5,
    }
    mock_store = _make_dispatch_opt_mocks(existing_state=existing)
    mock_platform = MagicMock()
    mock_platform.metadata_store = mock_store

    with (
        patch("app.services.mcp_tools.dispatch.resolve_project_dir", return_value=tmp_path / "proj"),
        # OptimizationState.load must NOT be the source — if it's called it would return None and let it through
        patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
        patch("app.platform.arch_split.resolve_arch_split", new_callable=AsyncMock, return_value=True),
        patch("app.platform.get_platform", return_value=mock_platform),
        patch("app.platform.types.current_request") as mock_req,
        patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="user-xyz"),
    ):
        mock_req.return_value = MagicMock(user_id="user-xyz")

        result = await dispatch_optimization({"project_id": "proj-abc"})

        # Verify the existing-check read via get_optimization_state and rejected
        assert mock_store.get_optimization_state.called, "existing-check must read via get_optimization_state on arch_split=True"
        assert result.get("is_error") is True
        content = result.get("content", [])
        assert any("already running" in c.get("text", "").lower() for c in content)


@pytest.mark.asyncio
async def test_wait_for_eval_arch_split_true_reads_and_cancels_via_ddb(tmp_path):
    """_wait_for_eval on arch_split=True reads cancel signal via get_optimization_state and flips to cancelled via put."""
    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock, patch

    from app.services.jobs.optimization_executor import _wait_for_eval
    from app.services.jobs import Job

    # Parent job carries arch_split=True + user_id
    parent_job = Job(
        job_id="parent-1",
        project_id="proj-abc",
        type="optimization_iteration",
        config={"arch_split": True, "user_id": "user-xyz", "project_id": "proj-abc"},
        state="running",
        progress={},
    )

    # DDB record with status "stopping" (cancel signal)
    mock_store = AsyncMock()
    mock_store.get_optimization_state = AsyncMock(return_value={
        "version": 3,
        "status": "stopping",
        "iterations": [],
        "target_metric_scores": {"ExactMatch": 0.85},
        "max_iterations": 5,
        "plateau_window": 3,
        "max_iter_cap": 5,
    })
    mock_store.put_optimization_state = AsyncMock(return_value=4)
    mock_platform = MagicMock()
    mock_platform.metadata_store = mock_store

    mock_tracker = MagicMock()
    mock_tracker.cancel_job = MagicMock()
    mock_tracker.update_state = MagicMock()

    with (
        # Speed up the poll so the test doesn't sleep 10s
        patch("app.services.jobs.optimization_executor.asyncio.sleep", new_callable=AsyncMock),
        patch("app.platform.get_platform", return_value=mock_platform),
        # Deliberately NO repository patch: this test's SUBJECT is that the cancel
        # poll performs the real durable read and the real conditional write. It
        # previously patched OptimizationState.load to prove EFS was not the source;
        # with the repository owning the branch, the store mock below is what proves
        # it — patching the repository here would bypass the very thing under test.
    ):
        result = await _wait_for_eval(
            eval_job_id="eval-1",
            project_dir=tmp_path / "proj",
            project_id="proj-abc",
            parent_job=parent_job,
            job_tracker=mock_tracker,
        )

        # Cancel detected → returns None
        assert result is None
        # Read via get_optimization_state (NOT OptimizationState.load)
        assert mock_store.get_optimization_state.called, "cancel poll must read DDB on arch_split=True"
        # Flip stopping → cancelled via put_optimization_state (version threaded)
        assert mock_store.put_optimization_state.called, "cancel poll must write cancelled to DDB on arch_split=True"
        put_kwargs = mock_store.put_optimization_state.call_args.kwargs
        assert put_kwargs.get("expected_version") == 3
        # Eval + parent cancelled
        mock_tracker.cancel_job.assert_called_with("eval-1")


# ---------------------------------------------------------------------------
# Integration round-trip: loop finds non-None running state after start
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_finds_running_state_after_start_write():
    """After an arch_split=True start-equivalent write, the loop's first read returns a
    non-None running record.

    If start wrote EFS state.json while the loop read DDB, get would return None and
    `run_optimization_loop` would hit its `State=None, exiting` break and never
    iterate. This proves the round-trip: create/save via DDB (arch_split=True) records
    the state, and the loop's subsequent get_optimization_state returns it with
    status="running" and a version.
    """
    import tempfile
    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock, patch

    from app.services.optimization_state import OptimizationState, OPTIMIZATION_STATE_FILE

    # A stateful in-memory store: put records the last written payload (with a
    # version), get returns it. Mirrors the AsyncMock side_effect idiom used
    # elsewhere in this file — not a new fixture framework.
    stored: dict = {}

    async def _put(user_id, project_id, data, expected_version=None):
        # Assign/bump a version the way the real store does
        new_version = (stored.get("version", 0) + 1) if stored else 1
        stored.clear()
        stored.update(data)
        stored["version"] = new_version
        return new_version

    async def _get(user_id, project_id):
        return dict(stored) if stored else None

    mock_store = MagicMock()
    mock_store.put_optimization_state = AsyncMock(side_effect=_put)
    mock_store.get_optimization_state = AsyncMock(side_effect=_get)
    mock_platform = MagicMock()
    mock_platform.metadata_store = mock_store

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)

        with patch("app.platform.get_platform", return_value=mock_platform):
            # Start-equivalent write: create with arch_split=True (first create,
            # expected_version=None) records status="running" to the store.
            OptimizationState.create(
                project_dir=project_dir,
                target_metric_scores={"ExactMatch": 0.85},
                max_iterations=5,
                arch_split=True,
                user_id="user-xyz",
                project_id="proj-abc",
                expected_version=None,
            )

            # No state.json written on the new path (proves EFS was NOT the writer)
            assert not (project_dir / OPTIMIZATION_STATE_FILE).exists()

            # The loop's first read in run_optimization_loop — get_optimization_state
            state_data = await mock_store.get_optimization_state("user-xyz", "proj-abc")

            # NOT None → the loop would NOT hit `State=None, exiting`
            assert state_data is not None, "loop's first read must return non-None after start write"
            assert state_data["status"] == "running"
            assert state_data.get("version") is not None and state_data["version"] >= 1
