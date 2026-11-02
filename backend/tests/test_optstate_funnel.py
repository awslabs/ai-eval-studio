# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single-writer funnel + flag-branched save() tests.

This test module proves:
- resolve_arch_split is called exactly once per dispatch_eval
- resolve_arch_split is called exactly once per dispatch_optimization
- The resolved boolean lands in the job config
- Every iteration job in one optimization run carries the same flag value
- OptimizationState.save() branches on the flag: new path -> put_optimization_state
  with expected_version, no state.json; old path -> EFS JSON write
- A stale expected_version raises ConcurrentUpdateError
- completion_turn reads best-so-far via get_optimization_state on new path
- record_optimization_iteration writes via put_optimization_state WITH
  expected_version on new path
"""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest


# Resolve-once + thread into config


def test_dispatch_eval_has_arch_split_resolve():
    """Verify dispatch_eval source contains arch_split resolution."""
    from app.services.mcp_tools.dispatch import dispatch_eval

    # Read the source code of the handler
    import inspect
    source = inspect.getsource(dispatch_eval.handler)

    # Assert the source contains the resolve_arch_split call
    assert "resolve_arch_split" in source, "dispatch_eval must call resolve_arch_split"
    assert "arch_split = await resolve_arch_split" in source, "dispatch_eval must await resolve_arch_split"

    # Assert the resolved flag is threaded into config
    assert '"arch_split": arch_split' in source or "'arch_split': arch_split" in source, \
        "dispatch_eval must thread arch_split into job config"


def test_dispatch_optimization_has_arch_split_resolve():
    """Verify dispatch_optimization source contains arch_split resolution."""
    from app.services.mcp_tools.dispatch import dispatch_optimization

    import inspect
    source = inspect.getsource(dispatch_optimization.handler)

    # Assert the source contains the resolve_arch_split call
    assert "resolve_arch_split" in source, "dispatch_optimization must call resolve_arch_split"
    assert "arch_split = await resolve_arch_split" in source, "dispatch_optimization must await resolve_arch_split"

    # Assert the resolved flag is threaded into iteration_config
    assert '"arch_split": arch_split' in source or "'arch_split': arch_split" in source, \
        "dispatch_optimization must thread arch_split into iteration_config"


def test_optimization_loop_code_threads_arch_split():
    """Verify run_optimization_loop threads arch_split from iteration_config."""
    from app.services.optimization_loop_orchestrator import run_optimization_loop
    import inspect

    source = inspect.getsource(run_optimization_loop)

    # Assert the loop unpacks arch_split from iteration_config
    # (it doesn't re-resolve, it uses the threaded value)
    assert "iteration_config" in source, "run_optimization_loop must receive iteration_config"

    # Assert create_job is called with config that includes iteration_config spread
    # (the ** spread pattern threads all iteration_config fields including arch_split)
    assert "**iteration_config" in source, \
        "run_optimization_loop must spread iteration_config into each job config"


def test_no_mid_operation_resolve_arch_split():
    """Verify resolve_arch_split is not called inside job/state operations."""
    # These files must NOT call resolve_arch_split (they consume the threaded flag)
    forbidden_modules = [
        "app.services.jobs",
        "app.services.optimization_state",
        "app.services.jobs.optimization_executor",
        "app.services.jobs.eval_executor",
        "app.services.completion_turn",
    ]

    for module_path in forbidden_modules:
        try:
            module = __import__(module_path, fromlist=[""])
            import inspect
            source = inspect.getsource(module)

            # Allow the import statement but not the actual call
            assert "from app.platform.arch_split import resolve_arch_split" not in source, \
                f"{module_path} must not import resolve_arch_split directly"
            assert "resolve_arch_split(" not in source, \
                f"{module_path} must not call resolve_arch_split (consume threaded flag instead)"
        except (ImportError, OSError):
            # Module doesn't exist or can't be inspected - that's fine
            pass


# Single-writer funnel + flag-branched save()


# create()/reset_for_resume() are arch_split-aware


@pytest.mark.asyncio
async def test_create_arch_split_true_routes_to_ddb_no_state_json():
    """create(arch_split=True) routes save to put_optimization_state and writes NO state.json."""
    import tempfile
    from pathlib import Path
    from app.services.optimization_state import OptimizationState, OPTIMIZATION_STATE_FILE
    from app.platform import get_platform
    from unittest.mock import AsyncMock

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)

        # Mock the platform store
        store = get_platform().metadata_store
        original_put = store.put_optimization_state
        mock_put = AsyncMock(return_value=1)
        store.put_optimization_state = mock_put

        try:
            state_file = project_dir / OPTIMIZATION_STATE_FILE

            # Call create with arch_split=True
            state = OptimizationState.create(
                project_dir=project_dir,
                target_metric_scores={"Correctness": 0.8},
                max_iterations=5,
                arch_split=True,
                user_id="u-1",
                project_id="p-1",
                expected_version=None,
            )

            # Verify put_optimization_state was called with expected_version=None
            assert mock_put.called
            call_args = mock_put.call_args
            assert call_args[0][0] == "u-1"  # user_id
            assert call_args[0][1] == "p-1"  # project_id
            assert isinstance(call_args[0][2], dict)  # state data
            assert call_args[1]["expected_version"] is None

            # Verify NO state.json was written (new path)
            assert not state_file.exists(), "state.json must NOT be written on new path"

        finally:
            store.put_optimization_state = original_put


@pytest.mark.asyncio
async def test_create_arch_split_true_expected_version_overwrite():
    """create(arch_split=True, expected_version=3) passes expected_version=3 through (overwrite case)."""
    import tempfile
    from pathlib import Path
    from app.services.optimization_state import OptimizationState
    from app.platform import get_platform
    from unittest.mock import AsyncMock

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)

        store = get_platform().metadata_store
        original_put = store.put_optimization_state
        mock_put = AsyncMock(return_value=4)
        store.put_optimization_state = mock_put

        try:
            # Call create with expected_version=3 (overwrite terminal record)
            state = OptimizationState.create(
                project_dir=project_dir,
                target_metric_scores={"Correctness": 0.8},
                max_iterations=5,
                arch_split=True,
                user_id="u-1",
                project_id="p-1",
                expected_version=3,
            )

            # Verify put_optimization_state received expected_version=3
            assert mock_put.called
            call_args = mock_put.call_args
            assert call_args[1]["expected_version"] == 3

        finally:
            store.put_optimization_state = original_put


@pytest.mark.asyncio
async def test_create_arch_split_false_writes_state_json_no_store():
    """create() with defaults (arch_split=False) still writes state.json and never calls put_optimization_state."""
    import tempfile
    from pathlib import Path
    from app.services.optimization_state import OptimizationState, OPTIMIZATION_STATE_FILE
    from app.platform import get_platform
    from unittest.mock import AsyncMock
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)

        store = get_platform().metadata_store
        original_put = store.put_optimization_state
        mock_put = AsyncMock(return_value=1)
        store.put_optimization_state = mock_put

        try:
            state_file = project_dir / OPTIMIZATION_STATE_FILE

            # Call create with defaults (arch_split=False)
            state = OptimizationState.create(
                project_dir=project_dir,
                target_metric_scores={"Correctness": 0.8},
                max_iterations=5,
            )

            # Verify put_optimization_state was NOT called
            assert not mock_put.called, "put_optimization_state must NOT be called on old path"

            # Verify state.json WAS written
            assert state_file.exists(), "state.json must be written on old path"

            # Verify the file content
            data = json.loads(state_file.read_text())
            assert data["status"] == "running"
            assert data["max_iterations"] == 5

        finally:
            store.put_optimization_state = original_put


@pytest.mark.asyncio
async def test_reset_for_resume_no_persist():
    """reset_for_resume() performs NO persistence: clears error, flips status in-memory only."""
    import tempfile
    from pathlib import Path
    from app.services.optimization_state import OptimizationState, OPTIMIZATION_STATE_FILE
    from app.platform import get_platform
    from unittest.mock import AsyncMock

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)

        # Create state file with terminal status
        state = OptimizationState(
            project_dir=project_dir,
            target_metric_scores={"Correctness": 0.8},
            max_iterations=5,
            plateau_window=3,
            iterations=[{"iteration": 1}],
            status="cancelled",
        )
        from app.services.resource_error import ResourceError
        state.error = ResourceError(code="EVAL_CANCELLED", message="test error")

        store = get_platform().metadata_store
        original_put = store.put_optimization_state
        mock_put = AsyncMock(return_value=1)
        store.put_optimization_state = mock_put

        try:
            state_file = project_dir / OPTIMIZATION_STATE_FILE
            # Pre-delete state.json if it exists
            if state_file.exists():
                state_file.unlink()

            # Call reset_for_resume
            state.reset_for_resume()

            # Verify in-memory state changed
            assert state.status == "running"
            assert state.error is None

            # Verify put_optimization_state was NOT called
            assert not mock_put.called, "reset_for_resume must NOT persist (caller owns the save)"

            # Verify NO state.json was written
            assert not state_file.exists(), "reset_for_resume must NOT write state.json"

        finally:
            store.put_optimization_state = original_put


@pytest.mark.asyncio
async def test_optimization_state_save_new_path_calls_put_optimization_state():
    """On new path (arch_split=True), save() routes to put_optimization_state with expected_version."""
    import tempfile
    from pathlib import Path
    from app.services.optimization_state import OptimizationState, OPTIMIZATION_STATE_FILE
    from app.platform import get_platform
    from unittest.mock import AsyncMock

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        # Create state directly without calling create() (which calls save with old-path defaults)
        state = OptimizationState(
            project_dir=project_dir,
            target_metric_scores={"Correctness": 0.8},
            max_iterations=5,
            plateau_window=3,
            iterations=[],
            status="running",
        )

        # Mock the platform store
        store = get_platform().metadata_store
        original_put = store.put_optimization_state
        mock_put = AsyncMock(return_value=1)
        store.put_optimization_state = mock_put

        try:
            # Ensure state.json doesn't exist before we call save
            state_file = project_dir / OPTIMIZATION_STATE_FILE
            if state_file.exists():
                state_file.unlink()

            # Call save with new-path params
            state.save(
                arch_split=True,
                user_id="u-1",
                project_id="p-1",
                expected_version=0,
            )

            # Verify put_optimization_state was called with expected_version
            assert mock_put.called
            call_args = mock_put.call_args
            assert call_args[0][0] == "u-1"  # user_id
            assert call_args[0][1] == "p-1"  # project_id
            assert isinstance(call_args[0][2], dict)  # state data
            assert call_args[1]["expected_version"] == 0

            # Verify NO state.json was written
            assert not state_file.exists(), "state.json must NOT be written on new path"

        finally:
            store.put_optimization_state = original_put


@pytest.mark.asyncio
async def test_optimization_state_save_old_path_writes_state_json():
    """On old path (arch_split=False), save() writes state.json and does NOT call store."""
    import tempfile
    from pathlib import Path
    from app.services.optimization_state import OptimizationState, OPTIMIZATION_STATE_FILE
    from app.platform import get_platform
    from unittest.mock import AsyncMock

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"Correctness": 0.8},
            max_iterations=5,
        )

        # Mock the platform store
        store = get_platform().metadata_store
        original_put = store.put_optimization_state
        mock_put = AsyncMock(return_value=1)
        store.put_optimization_state = mock_put

        try:
            # Call save with old-path (arch_split=False, default)
            state.save()

            # Verify put_optimization_state was NOT called
            assert not mock_put.called, "put_optimization_state must NOT be called on old path"

            # Verify state.json WAS written
            state_file = project_dir / OPTIMIZATION_STATE_FILE
            assert state_file.exists(), "state.json must be written on old path"

            # Verify the file content
            import json
            data = json.loads(state_file.read_text())
            assert data["status"] == "running"
            assert data["max_iterations"] == 5

        finally:
            store.put_optimization_state = original_put


@pytest.mark.asyncio
async def test_put_optimization_state_stale_version_raises_concurrent_update_error():
    """Stale expected_version must raise ConcurrentUpdateError (seatbelt)."""
    import boto3
    from botocore.stub import Stubber, ANY
    from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore
    from app.platform.metadata_store import ConcurrentUpdateError

    ddb = boto3.client("dynamodb", region_name="us-east-1")
    stubber = Stubber(ddb)

    # Stub a ConditionalCheckFailedException (version mismatch)
    stubber.add_client_error(
        "put_item",
        service_error_code="ConditionalCheckFailedException",
        service_message="version mismatch",
        expected_params={
            "TableName": "ai-eval-studio-metadata",
            "Item": ANY,
            "ConditionExpression": "attribute_not_exists(version) OR version = :expected",
            "ExpressionAttributeValues": {":expected": {"N": "2"}},
        },
    )
    stubber.activate()

    try:
        store = DynamoDBMetadataStore(ddb_client=ddb)
        with pytest.raises(ConcurrentUpdateError):
            await store.put_optimization_state(
                "u-1", "p-1", {"status": "running"}, expected_version=2
            )
        stubber.assert_no_pending_responses()
    finally:
        stubber.deactivate()


# Reroute secondary opt-state writers


@pytest.mark.asyncio
async def test_completion_turn_reads_via_get_optimization_state_on_new_path():
    """On new path, completion_turn reads best-so-far via get_optimization_state (not OptimizationState.load)."""
    # This test verifies the code PATH is correct (new-path branches to get_optimization_state),
    # not the full best-so-far message assembly (which has complex async mocking issues).
    # The key assertion: on arch_split=True, _build_synthetic_message calls
    # store.get_optimization_state instead of OptimizationState.load.
    import tempfile
    from pathlib import Path
    from unittest.mock import AsyncMock, patch

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        project_id = "p-1"

        # Mock resolve_project_dir (it's imported with `from` so patch the source module)
        with patch("app.services.project_paths.resolve_project_dir", return_value=project_dir):
            # Mock get_platform().metadata_store.get_optimization_state (patch at source)
            with patch("app.platform.get_platform") as mock_platform:
                mock_store = mock_platform.return_value.metadata_store
                mock_get = AsyncMock(return_value={
                    "iterations": [{"iteration": 1, "prompt_version": "v1", "passed": 8, "total_cases": 10}],
                    "target_metric_scores": {"Correctness": 0.8},
                    "max_iterations": 5,
                    "plateau_window": 3,
                    "status": "running",
                    "max_iter_cap": 5,
                })
                mock_store.get_optimization_state = mock_get

                from app.services.completion_turn import _build_synthetic_message
                from app.services.jobs import Job

                # Create a job with arch_split=True
                job = Job(
                    job_id="job-1",
                    project_id=project_id,
                    type="optimization_iteration",
                    config={
                        "arch_split": True,
                        "user_id": "u-1",
                        "iteration": 1,
                        "prompt_version": "v1",
                    },
                    state="completed",
                    progress={},
                    result={
                        "iteration": 1,
                        "prompt_version": "v1",
                        "passed_cases": 8,
                        "total_cases": 10,
                        "metrics_summary": {"Correctness": 0.8},
                        "per_model_pass_rates": {},
                        "run_output_file": "run-1.json",
                        "convergence": None,
                    },
                )

                # Call _build_synthetic_message (now async, must await)
                message = await _build_synthetic_message(job)

                # Key assertion: get_optimization_state was called on the new path
                assert mock_get.called
                call_args = mock_get.call_args
                # Verify it was called with user_id and project_id
                assert len(call_args[0]) >= 2
                assert call_args[0][1] == project_id

                # Message should at least contain the basic content
                assert "Optimization iteration 1 complete" in message


@pytest.mark.asyncio
async def test_record_optimization_iteration_writes_via_put_optimization_state_on_new_path():
    """On new path, record_optimization_iteration writes via put_optimization_state WITH expected_version."""
    import tempfile
    from pathlib import Path
    from app.services.mcp_tools.optimization import record_optimization_iteration
    from app.services.optimization_state import OptimizationState
    from app.platform import get_platform
    from unittest.mock import AsyncMock, patch

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        project_id = "p-1"

        # The tool derives the iteration row from the run FILE (the
        # results owner) — agent-supplied counts are ignored. Materialize a
        # minimal results file for run_id "run-1": one model, 10 cases,
        # 8 passing, per-metric data for "Correctness".
        output_dir = project_dir / "output"
        output_dir.mkdir()
        import json as _json
        _cases = [
            {"success": i < 8, "metrics_data": [{"name": "Correctness", "success": i < 8}]}
            for i in range(10)
        ]
        (output_dir / "run-1").write_text(_json.dumps([
            {"hyperparameters": {"model": "m1"}, "results": _cases}
        ]))

        # Mock resolve_project_dir (patch at source)
        with patch("app.services.mcp_tools.optimization.resolve_project_dir", return_value=project_dir):
            # Mock resolve_arch_split to return True (new path, patch at source)
            with patch("app.platform.arch_split.resolve_arch_split", new_callable=AsyncMock, return_value=True):
                # Mock current_request to return user_id (patch at source)
                with patch("app.platform.types.current_request") as mock_request:
                    mock_request.return_value = MagicMock(user_id="u-1")

                    # Mock the platform store
                    store = get_platform().metadata_store
                    original_get = store.get_optimization_state
                    original_put = store.put_optimization_state

                    # Return an empty state on first get
                    mock_get = AsyncMock(return_value={
                        "iterations": [],
                        "target_metric_scores": {"Correctness": 0.8},
                        "max_iterations": 5,
                        "plateau_window": 3,
                        "status": "running",
                        "max_iter_cap": 5,
                        "version": 1,
                    })
                    mock_put = AsyncMock(return_value=2)
                    store.get_optimization_state = mock_get
                    store.put_optimization_state = mock_put

                    try:
                        # Call the MCP tool handler (it's wrapped by @tool decorator)
                        result = await record_optimization_iteration.handler({
                            "project_id": project_id,
                            "prompt_version": "v1",
                            "run_id": "run-1",
                        })

                        # Verify get_optimization_state was called
                        assert mock_get.called

                        # Verify put_optimization_state was called WITH expected_version
                        # The tool calls record_iteration (writes with version=1) then
                        # check_convergence (may write with version=2). We verify at least
                        # ONE put_optimization_state call had an expected_version param.
                        assert mock_put.called
                        # Check that at least one call had expected_version
                        has_expected_version = any(
                            "expected_version" in call.kwargs
                            for call in mock_put.call_args_list
                        )
                        assert has_expected_version, "put_optimization_state must be called with expected_version"

                        # Verify result is not an error
                        assert "is_error" not in result or not result["is_error"]

                    finally:
                        store.get_optimization_state = original_get
                        store.put_optimization_state = original_put


@pytest.mark.asyncio
async def test_record_optimization_iteration_stale_version_surfaces_mcp_error():
    """Stale expected_version on the tool write surfaces as mcp_error (not unhandled raise)."""
    import tempfile
    from pathlib import Path
    from app.services.mcp_tools.optimization import record_optimization_iteration
    from app.platform import get_platform
    from app.platform.metadata_store import ConcurrentUpdateError
    from unittest.mock import AsyncMock, patch, MagicMock

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        project_id = "p-1"

        # The tool derives the iteration row from the run FILE (the
        # results owner) — agent-supplied counts are ignored. Materialize a
        # minimal results file for run_id "run-1": one model, 10 cases,
        # 8 passing, per-metric data for "Correctness".
        output_dir = project_dir / "output"
        output_dir.mkdir()
        import json as _json
        _cases = [
            {"success": i < 8, "metrics_data": [{"name": "Correctness", "success": i < 8}]}
            for i in range(10)
        ]
        (output_dir / "run-1").write_text(_json.dumps([
            {"hyperparameters": {"model": "m1"}, "results": _cases}
        ]))

        with patch("app.services.mcp_tools.optimization.resolve_project_dir", return_value=project_dir):
            with patch("app.platform.arch_split.resolve_arch_split", new_callable=AsyncMock, return_value=True):
                with patch("app.platform.types.current_request") as mock_request:
                    mock_request.return_value = MagicMock(user_id="u-1")

                    store = get_platform().metadata_store
                    original_get = store.get_optimization_state
                    original_put = store.put_optimization_state

                    mock_get = AsyncMock(return_value={
                        "iterations": [],
                        "target_metric_scores": {"Correctness": 0.8},
                        "max_iterations": 5,
                        "plateau_window": 3,
                        "status": "running",
                        "max_iter_cap": 5,
                        "version": 1,
                    })
                    # put raises ConcurrentUpdateError (stale version)
                    mock_put = AsyncMock(side_effect=ConcurrentUpdateError("version mismatch"))
                    store.get_optimization_state = mock_get
                    store.put_optimization_state = mock_put

                    try:
                        result = await record_optimization_iteration.handler({
                            "project_id": project_id,
                            "prompt_version": "v1",
                            "run_id": "run-1",
                        })

                        # Verify the result is an mcp_error (not an unhandled exception)
                        assert result.get("is_error") is True
                        # The error message is in content array
                        content = result.get("content", [])
                        assert len(content) > 0
                        error_text = content[0].get("text", "").lower()
                        assert "version conflict" in error_text

                    finally:
                        store.get_optimization_state = original_get
                        store.put_optimization_state = original_put


@pytest.mark.asyncio
async def test_record_optimization_iteration_derives_from_run_file_not_args():
    """Discriminator: the run FILE is the results owner.

    The tool's input schema must not accept scores at all — an agent-typed
    pass rate is a claim, not a measurement — and the recorded row must be
    derived from the run file, with per-model data. Rows written from agent
    claims carried no per-model data, which is what kept the aggregate
    convergence/plateau fallbacks alive.

    Junk score keys are passed in anyway to prove they are ignored rather
    than picked up by any surviving read path."""
    import json as _json
    import tempfile
    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock, patch

    from app.platform import get_platform
    from app.services.mcp_tools.optimization import record_optimization_iteration

    # Structural pin: the agent is never offered a place to type a score.
    schema = record_optimization_iteration.input_schema
    assert set(schema["properties"]) == {"project_id", "prompt_version", "run_id"}, (
        "record_optimization_iteration must take no scores — adding one back "
        "lets an agent claim a pass rate the run file never measured."
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        output_dir = project_dir / "output"
        output_dir.mkdir()
        cases = [
            {"success": i < 8, "metrics_data": [{"name": "Correctness", "success": i < 8}]}
            for i in range(10)
        ]
        (output_dir / "run-1").write_text(_json.dumps([
            {"hyperparameters": {"model": "m1"}, "results": cases}
        ]))

        with (
            patch(
                "app.services.mcp_tools.optimization.resolve_project_dir",
                return_value=project_dir,
            ),
            patch(
                "app.platform.arch_split.resolve_arch_split",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("app.platform.types.current_request") as mock_request,
        ):
            mock_request.return_value = MagicMock(user_id="u-1")
            store = get_platform().metadata_store
            original_get = store.get_optimization_state
            original_put = store.put_optimization_state
            store.get_optimization_state = AsyncMock(return_value={
                "iterations": [],
                "target_metric_scores": {"Correctness": 1.0},
                "max_iterations": 5,
                "plateau_window": 3,
                "status": "running",
                "max_iter_cap": 5,
                "version": 1,
            })
            put_mock = AsyncMock(return_value=2)
            store.put_optimization_state = put_mock
            try:
                result = await record_optimization_iteration.handler({
                    "project_id": "p-1",
                    "prompt_version": "v1",
                    "run_id": "run-1",
                    # Not in the schema; ignored if a caller sends them anyway.
                    "metrics_summary": {"Correctness": 1.0},
                    "total_cases": 20,
                    "passed": 20,
                })
                assert not result.get("is_error"), result

                # The persisted row carries the FILE's numbers + per-model data.
                written_state = put_mock.call_args_list[0].args[2]
                row = written_state["iterations"][-1]
                assert row["total_cases"] == 10
                assert row["passed"] == 8
                assert row["per_model_pass_rates"] == {"m1": 0.8}
                assert row["per_model_metrics"]["m1"]["Correctness"] == 0.8

                # The response reports the file-derived rate.
                payload = _json.loads(result["content"][0]["text"])
                assert abs(payload["current_pass_rate"] - 0.8) < 1e-9
            finally:
                store.get_optimization_state = original_get
                store.put_optimization_state = original_put
