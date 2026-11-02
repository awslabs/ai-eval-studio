# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for optimization loop REST endpoints (backend/app/api/optimization.py)."""

import json

import pytest
from httpx import ASGITransport, AsyncClient
from app.main import create_app

from .async_wait import until


def allow_launch_admission(mock_store):
    """Wire a mocked store so the launch admission gate admits.

    The start and resume routes now run the shared admission checks — cost kill
    switch, per-user cap, account cap — and then claim the project's durable eval
    lock, all against the metadata store. A ``MagicMock`` store returns
    non-awaitable attributes for those, so a test that only stubs the
    optimization-state methods fails on the await rather than on its subject.

    Tests about admission itself should override these afterwards.
    """
    from unittest.mock import AsyncMock

    mock_store.count_active_jobs_for_owner = AsyncMock(return_value=0)
    mock_store.list_non_terminal_jobs = AsyncMock(return_value=[])
    mock_store.claim_eval_lock = AsyncMock(return_value=True)
    mock_store.release_eval_lock = AsyncMock(return_value=None)
    return mock_store


@pytest.fixture
def app(tmp_path):
    import app.main as main_module
    original = main_module.PROJECTS_DIR
    main_module.PROJECTS_DIR = tmp_path / "projects"
    main_module.PROJECTS_DIR.mkdir()
    main_module._project_manager = None
    application = create_app()
    yield application
    main_module._project_manager = None
    main_module.PROJECTS_DIR = original


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _scaffold_project(projects_dir, project_id: str, with_synthetic: bool) -> None:
    """Build a minimum viable project on disk for optimization/start tests.

    Creates ``config.json``, ``run_eval.py`` (harness sentinel),
    ``src/config.py`` with parseable fields, raw + train data files, and
    optionally a synthetic batch with one approved case.
    """
    project_dir = projects_dir / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "config.json").write_text(json.dumps({
        "id": project_id,
        "name": "Test project",
        "description": "",
        "created_at": "2026-01-01T00:00:00Z",
        "owner_id": "local-dev",
    }))
    (project_dir / "run_eval.py").write_text("# harness sentinel\n")

    (project_dir / "src").mkdir(exist_ok=True)
    (project_dir / "src" / "config.py").write_text("""
from dataclasses import dataclass, field

@dataclass
class EvalConfig:
    models: list = field(default_factory=lambda: ["us.anthropic.claude-haiku-4-5-20251001-v1:0"])
    metrics: list = field(default_factory=lambda: [{"name": "ExactMatch", "type": "exact_match"}])
    data_file: str = "seed.csv"
    field_mappings: dict = field(default_factory=lambda: {
        "input": "question",
        "expected_output": "answer",
    })
    target_metric_scores: dict = field(default_factory=lambda: {"ExactMatch": 0.85})
    max_iterations: int = 10
    plateau_window: int = 3
""")

    raw_dir = project_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "seed.csv").write_text(
        "question,answer\nWhat is 2+2?,4\nCapital of France?,Paris\n"
    )

    (project_dir / "prompts").mkdir(exist_ok=True)
    (project_dir / "prompts" / "baseline.md").write_text("You are a helpful assistant.\n")

    if with_synthetic:
        synth_dir = project_dir / "data" / "synthetic"
        synth_dir.mkdir(parents=True, exist_ok=True)
        batch_id = "2026-04-22T10-00-00_paraphrase"
        (synth_dir / f"{batch_id}.csv").write_text(
            "question,answer\nq1,a1\nq2,a2\n"
        )
        manifest = {
            "batch_id": batch_id,
            "timestamp": batch_id.split("_")[0],
            "strategy": "paraphrase",
            "total_cases": 2,
            "field_mappings": {"input": "question", "expected_output": "answer"},
            "csv_file": f"{batch_id}.csv",
            "cases": [
                {"index": 0, "status": "approved"},
                {"index": 1, "status": "pending"},
            ],
        }
        (synth_dir / f"{batch_id}.manifest.json").write_text(json.dumps(manifest))


def _stub_terminal_message(monkeypatch) -> None:
    """Neutralize the agent notification every optimization endpoint fires.

    Split out of :func:`_stub_spawned_work` for the tests that need the real
    dispatch of the optimization loop — they patch ``run_optimization_loop``
    themselves to capture its argument, so they cannot use the full stub, but they
    still must not leave a CLI subprocess behind. Without this, waiting long
    enough to observe the loop dispatch also lets the notification start, and loop
    teardown then blocks on it for roughly three minutes per test.
    """
    import asyncio

    import app.services.completion_turn as ct_mod

    monkeypatch.setattr(
        ct_mod, "request_terminal_message", lambda *a, **k: asyncio.sleep(0),
    )


def _stub_spawned_work(monkeypatch) -> None:
    """Stop an endpoint leaving real background work behind after it responds.

    Every optimization endpoint notifies the agent on its way out
    (``request_terminal_message`` → a real Claude CLI subprocess) and /start and
    /resume also spawn the loop as a task. Nothing awaits them, so the event
    loop's teardown does — roughly three idle minutes per test, which reads as
    a hung suite.

    Patched at the DEFINING modules because both are imported function-locally
    at the spawn site.
    """
    import asyncio

    import app.services.completion_turn as ct_mod
    import app.services.optimization_loop_orchestrator as loop_mod

    monkeypatch.setattr(
        loop_mod, "run_optimization_loop", lambda iteration_config: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        ct_mod, "request_terminal_message", lambda *a, **k: asyncio.sleep(0),
    )


class TestOptimizationStartDataSource:
    """Tests for ``data_source`` wiring in ``POST /optimization/start``."""

    async def test_start_with_synthetic_sets_data_file_override(
        self, client, app, monkeypatch,
    ):
        import asyncio
        import app.main as main_module
        # The loop driver moved to services/optimization_loop_orchestrator and
        # dispatch imports it function-locally at the spawn site, so the patch must
        # target the DEFINING module — a dispatch attribute would not exist.
        import app.services.optimization_loop_orchestrator as loop_mod

        project_id = "proj-synthetic-src"
        _scaffold_project(main_module.PROJECTS_DIR, project_id, with_synthetic=True)
        main_module._project_manager = None  # re-init with new project on disk

        captured: dict = {}

        async def fake_run_loop(iteration_config):
            captured["config"] = iteration_config

        # Stub the OTHER unawaited spawn before waiting on this one. The endpoint
        # also fires ``request_terminal_message``, which reaches a real CLI
        # subprocess nothing awaits — so any wait long enough to be robust is also
        # long enough to let that start, and loop teardown then blocks on it for
        # minutes. See :func:`_stub_spawned_work`.
        _stub_terminal_message(monkeypatch)
        monkeypatch.setattr(loop_mod, "run_optimization_loop", fake_run_loop)

        resp = await client.post(
            f"/api/projects/{project_id}/optimization/start",
            json={"data_source": "synthetic"},
        )
        assert resp.status_code == 200, resp.text
        await until(
            lambda: "config" in captured,
            message="run_optimization_loop was never called",
        )

        cfg = captured["config"]
        assert cfg["data_file_override"] is not None
        assert "synthetic_only_" in cfg["data_file_override"]

    async def test_start_without_body_defaults_to_raw(
        self, client, app, monkeypatch,
    ):
        import asyncio
        import app.main as main_module
        # The loop driver moved to services/optimization_loop_orchestrator and
        # dispatch imports it function-locally at the spawn site, so the patch must
        # target the DEFINING module — a dispatch attribute would not exist.
        import app.services.optimization_loop_orchestrator as loop_mod

        project_id = "proj-default-raw"
        _scaffold_project(main_module.PROJECTS_DIR, project_id, with_synthetic=False)
        main_module._project_manager = None

        captured: dict = {}

        async def fake_run_loop(iteration_config):
            captured["config"] = iteration_config

        _stub_terminal_message(monkeypatch)
        monkeypatch.setattr(loop_mod, "run_optimization_loop", fake_run_loop)

        resp = await client.post(f"/api/projects/{project_id}/optimization/start")
        assert resp.status_code == 200, resp.text
        await until(
            lambda: "config" in captured,
            message="run_optimization_loop was never called",
        )
        assert captured["config"]["data_file_override"] is None

    async def test_start_synthetic_rejects_when_no_approvals(
        self, client, app,
    ):
        import app.main as main_module

        project_id = "proj-no-approvals"
        _scaffold_project(main_module.PROJECTS_DIR, project_id, with_synthetic=False)
        main_module._project_manager = None

        resp = await client.post(
            f"/api/projects/{project_id}/optimization/start",
            json={"data_source": "synthetic"},
        )
        assert resp.status_code == 400
        assert "approved synthetic" in resp.json()["detail"].lower()


class TestOptimizationStartEvalGuard:
    """A loop must not start/resume while an eval is still active.

    The duplicate-loop fence lives at the REST layer (the UI starts
    loops here), and it trusts the eval job's liveness — NOT the on-disk
    optimization status, which a stale/racy terminal write would defeat.
    Two live loops over one project race for the harness subprocess and
    leave the tracker holding an active eval no live loop owns (the exact
    drift /activity now refuses to paper over)."""

    def _register_active_eval(self, project_id: str):
        from app.services.jobs import get_job_tracker
        tracker = get_job_tracker()
        job = tracker.create_job(
            project_id, "eval",
            config={"model_ids": ["m"], "prompt_version": "baseline",
                    "source": "optimization"},
        )
        tracker.update_progress(job.job_id, message="running", models_total=1)
        return job

    async def test_start_rejected_when_eval_active(self, client, app):
        import app.main as main_module
        from app.services.jobs import get_job_tracker

        get_job_tracker()._reset()
        project_id = "proj-start-eval-guard"
        _scaffold_project(main_module.PROJECTS_DIR, project_id, with_synthetic=False)
        main_module._project_manager = None

        job = self._register_active_eval(project_id)
        try:
            resp = await client.post(
                f"/api/projects/{project_id}/optimization/start"
            )
            assert resp.status_code == 409, resp.text
            assert job.job_id[:8] in resp.json()["detail"]
        finally:
            get_job_tracker()._reset()

    async def test_resume_rejected_when_eval_active(self, client, app):
        import app.main as main_module
        from app.services.jobs import get_job_tracker
        from app.services.optimization_state import OptimizationState

        get_job_tracker()._reset()
        project_id = "proj-resume-eval-guard"
        _scaffold_project(main_module.PROJECTS_DIR, project_id, with_synthetic=False)
        main_module._project_manager = None

        # Seed a resumable (cancelled) optimization state so we get past
        # the status guard and reach the eval-active guard.
        project_dir = main_module.PROJECTS_DIR / project_id
        state = OptimizationState.create(
            project_dir, target_metric_scores={"ExactMatch": 0.85}
        )
        state.status = "cancelled"
        state.save()

        job = self._register_active_eval(project_id)
        try:
            resp = await client.post(
                f"/api/projects/{project_id}/optimization/resume"
            )
            assert resp.status_code == 409, resp.text
            assert job.job_id[:8] in resp.json()["detail"]
        finally:
            get_job_tracker()._reset()


class TestOptimizationIterationConfig:
    """REST endpoints inject arch_split + user_id into iteration_config.

    The REST optimization paths (start + resume) build an iteration_config dict
    consumed by run_optimization_loop (services/optimization_loop_orchestrator.py).
    Without arch_split + user_id in it, every UI/REST-launched loop runs on the
    in-process path regardless of the project's arch_split_enabled flag AND the
    worker's durable event-log writes are misfiled under a null user. The MCP path
    (dispatch_optimization in services/mcp_tools/dispatch.py) sets both; these
    tests hold the REST endpoints to the same contract."""

    async def test_start_injects_arch_split_and_user_id(
        self, client, app, monkeypatch,
    ):
        """POST /optimization/start includes arch_split (bool) + user_id (non-null) in iteration_config."""
        import asyncio
        import app.main as main_module
        # The loop driver moved to services/optimization_loop_orchestrator and
        # dispatch imports it function-locally at the spawn site, so the patch must
        # target the DEFINING module — a dispatch attribute would not exist.
        import app.services.optimization_loop_orchestrator as loop_mod

        project_id = "proj-start-injection"
        _scaffold_project(main_module.PROJECTS_DIR, project_id, with_synthetic=False)
        main_module._project_manager = None

        captured: dict = {}

        async def fake_run_loop(iteration_config):
            captured["config"] = iteration_config

        _stub_terminal_message(monkeypatch)
        monkeypatch.setattr(loop_mod, "run_optimization_loop", fake_run_loop)

        resp = await client.post(f"/api/projects/{project_id}/optimization/start")
        assert resp.status_code == 200, resp.text
        await until(
            lambda: "config" in captured,
            message="run_optimization_loop was never called",
        )

        cfg = captured["config"]
        assert "arch_split" in cfg
        assert isinstance(cfg["arch_split"], bool)
        assert cfg["user_id"] is not None

    async def test_resume_injects_arch_split_and_user_id(
        self, client, app, monkeypatch,
    ):
        """POST /optimization/resume includes arch_split (bool) + user_id (non-null) in iteration_config."""
        import asyncio
        import app.main as main_module
        # The loop driver moved to services/optimization_loop_orchestrator and
        # dispatch imports it function-locally at the spawn site, so the patch must
        # target the DEFINING module — a dispatch attribute would not exist.
        import app.services.optimization_loop_orchestrator as loop_mod
        from app.services.jobs import get_job_tracker
        from app.services.optimization_state import OptimizationState

        get_job_tracker()._reset()
        project_id = "proj-resume-injection"
        _scaffold_project(main_module.PROJECTS_DIR, project_id, with_synthetic=False)
        main_module._project_manager = None

        # Seed a resumable state (cancelled, no active eval) so the handler
        # reaches the success path and calls the loop.
        project_dir = main_module.PROJECTS_DIR / project_id
        state = OptimizationState.create(
            project_dir, target_metric_scores={"ExactMatch": 0.85}
        )
        state.status = "cancelled"
        state.save()

        captured: dict = {}

        async def fake_run_loop(iteration_config):
            captured["config"] = iteration_config

        _stub_terminal_message(monkeypatch)
        monkeypatch.setattr(loop_mod, "run_optimization_loop", fake_run_loop)

        try:
            resp = await client.post(f"/api/projects/{project_id}/optimization/resume")
            assert resp.status_code == 200, resp.text
            await until(
                lambda: "config" in captured,
                message="run_optimization_loop was never called",
            )

            cfg = captured["config"]
            assert "arch_split" in cfg
            assert isinstance(cfg["arch_split"], bool)
            assert cfg["user_id"] is not None
        finally:
            get_job_tracker()._reset()


# REST start + resume read/write DynamoDB on arch_split=True


class TestOptimizationStartDynamoDBPath:
    """Tests for start_optimization_rest routing to DynamoDB on arch_split=True."""

    @pytest.mark.asyncio
    async def test_start_arch_split_true_persists_to_ddb(self, client, app, tmp_path, monkeypatch):
        """On arch_split=True, start persists to DynamoDB and does NOT write state.json."""
        from pathlib import Path
        from unittest.mock import AsyncMock, patch
        import asyncio
        from app.services.jobs import get_job_tracker
        from app.services.mcp_tools import dispatch as dispatch_mod
        # The loop driver moved out of dispatch; patch its defining module.
        import app.services.optimization_loop_orchestrator as loop_mod

        project_id = "test-start-ddb"
        _scaffold_project(tmp_path / "projects", project_id, with_synthetic=False)
        project_dir = tmp_path / "projects" / project_id

        # Mock resolve_arch_split to return True (patch the bound name in optimization.py)
        with patch("app.api.optimization.resolve_arch_split", new_callable=AsyncMock, return_value=True):
            # Mock get_platform().metadata_store
            with patch("app.platform.get_platform") as mock_platform:
                mock_store = mock_platform.return_value.metadata_store
                mock_put = AsyncMock(return_value=1)
                mock_get = AsyncMock(return_value=None)  # No existing record
                mock_store.put_optimization_state = mock_put
                mock_store.get_optimization_state = mock_get
                allow_launch_admission(mock_store)

                # Monkeypatch the loop to avoid spawning background task
                captured = {}

                async def fake_run_loop(iteration_config):
                    captured["config"] = iteration_config

                monkeypatch.setattr(loop_mod, "run_optimization_loop", fake_run_loop)

                try:
                    resp = await client.post(f"/api/projects/{project_id}/optimization/start")
                    assert resp.status_code == 200, resp.text

                    # Give async tasks a chance to run
                    for _ in range(5):
                        await asyncio.sleep(0)

                    # Verify put_optimization_state was called
                    assert mock_put.called, "put_optimization_state must be called on arch_split=True"

                    # Verify NO state.json was written
                    from app.services.optimization_state import OPTIMIZATION_STATE_FILE
                    state_file = project_dir / OPTIMIZATION_STATE_FILE
                    assert not state_file.exists(), "state.json must NOT be written on new path"

                finally:
                    get_job_tracker()._reset()

    @pytest.mark.asyncio
    async def test_start_arch_split_true_overwrites_terminal_record(self, client, app, tmp_path, monkeypatch):
        """On arch_split=True with existing terminal DDB record, start overwrites without ConcurrentUpdateError."""
        from unittest.mock import AsyncMock, patch
        import asyncio
        from app.services.jobs import get_job_tracker
        from app.services.mcp_tools import dispatch as dispatch_mod
        # The loop driver moved out of dispatch; patch its defining module.
        import app.services.optimization_loop_orchestrator as loop_mod

        project_id = "test-start-overwrite"
        _scaffold_project(tmp_path / "projects", project_id, with_synthetic=False)

        # Mock resolve_arch_split to return True
        with patch("app.api.optimization.resolve_arch_split", new_callable=AsyncMock, return_value=True):
            with patch("app.platform.get_platform") as mock_platform:
                mock_store = mock_platform.return_value.metadata_store
                mock_put = AsyncMock(return_value=5)
                # Existing terminal record with version=4
                mock_get = AsyncMock(return_value={
                    "version": 4,
                    "status": "cancelled",
                    "iterations": [],
                    "target_metric_scores": {"Correctness": 0.8},
                    "max_iterations": 5,
                    "plateau_window": 3,
                    "max_iter_cap": 5,
                })
                mock_store.put_optimization_state = mock_put
                mock_store.get_optimization_state = mock_get
                allow_launch_admission(mock_store)

                captured = {}

                async def fake_run_loop(iteration_config):
                    captured["config"] = iteration_config

                monkeypatch.setattr(loop_mod, "run_optimization_loop", fake_run_loop)

                try:
                    resp = await client.post(f"/api/projects/{project_id}/optimization/start")
                    assert resp.status_code == 200, resp.text

                    for _ in range(5):
                        await asyncio.sleep(0)

                    # Verify put_optimization_state received expected_version=4
                    assert mock_put.called
                    call_kwargs = mock_put.call_args.kwargs
                    assert "expected_version" in call_kwargs
                    assert call_kwargs["expected_version"] == 4

                finally:
                    get_job_tracker()._reset()

    @pytest.mark.asyncio
    async def test_start_arch_split_true_first_time_expected_version_none(self, client, app, tmp_path, monkeypatch):
        """On arch_split=True first-time start, expected_version is None."""
        from unittest.mock import AsyncMock, patch
        import asyncio
        from app.services.jobs import get_job_tracker
        from app.services.mcp_tools import dispatch as dispatch_mod
        # The loop driver moved out of dispatch; patch its defining module.
        import app.services.optimization_loop_orchestrator as loop_mod

        project_id = "test-start-first"
        _scaffold_project(tmp_path / "projects", project_id, with_synthetic=False)

        with patch("app.api.optimization.resolve_arch_split", new_callable=AsyncMock, return_value=True):
            with patch("app.platform.get_platform") as mock_platform:
                mock_store = mock_platform.return_value.metadata_store
                mock_put = AsyncMock(return_value=1)
                mock_get = AsyncMock(return_value=None)  # No existing record
                mock_store.put_optimization_state = mock_put
                mock_store.get_optimization_state = mock_get
                allow_launch_admission(mock_store)

                captured = {}

                async def fake_run_loop(iteration_config):
                    captured["config"] = iteration_config

                monkeypatch.setattr(loop_mod, "run_optimization_loop", fake_run_loop)

                try:
                    resp = await client.post(f"/api/projects/{project_id}/optimization/start")
                    assert resp.status_code == 200, resp.text

                    for _ in range(5):
                        await asyncio.sleep(0)

                    # Verify expected_version=None on first write
                    assert mock_put.called
                    call_kwargs = mock_put.call_args.kwargs
                    assert "expected_version" in call_kwargs
                    assert call_kwargs["expected_version"] is None

                finally:
                    get_job_tracker()._reset()


class TestOptimizationResumeDynamoDBPath:
    """Tests for resume_optimization routing to DynamoDB on arch_split=True."""

    @pytest.mark.asyncio
    async def test_resume_arch_split_true_reads_and_writes_ddb(self, client, app, tmp_path, monkeypatch):
        """On arch_split=True, resume reads via get_optimization_state and persists reset via put_optimization_state."""
        from unittest.mock import AsyncMock, patch
        import asyncio
        from app.services.jobs import get_job_tracker
        from app.services.mcp_tools import dispatch as dispatch_mod
        # The loop driver moved out of dispatch; patch its defining module.
        import app.services.optimization_loop_orchestrator as loop_mod

        project_id = "test-resume-ddb"
        _scaffold_project(tmp_path / "projects", project_id, with_synthetic=False)
        project_dir = tmp_path / "projects" / project_id

        # Create a terminal state on disk for the harness config loader
        from app.services.optimization_state import OptimizationState
        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"Correctness": 0.8},
            max_iterations=5,
        )
        state.status = "cancelled"
        state.save()

        with patch("app.api.optimization.resolve_arch_split", new_callable=AsyncMock, return_value=True):
            with patch("app.platform.get_platform") as mock_platform:
                mock_store = mock_platform.return_value.metadata_store
                mock_put = AsyncMock(return_value=2)
                mock_get = AsyncMock(return_value={
                    "version": 1,
                    "status": "cancelled",
                    "iterations": [{"iteration": 1}],
                    "target_metric_scores": {"Correctness": 0.8},
                    "max_iterations": 5,
                    "plateau_window": 3,
                    "max_iter_cap": 5,
                    "workload": None,
                })
                mock_store.put_optimization_state = mock_put
                mock_store.get_optimization_state = mock_get
                allow_launch_admission(mock_store)

                captured = {}

                async def fake_run_loop(iteration_config):
                    captured["config"] = iteration_config

                monkeypatch.setattr(loop_mod, "run_optimization_loop", fake_run_loop)

                try:
                    resp = await client.post(f"/api/projects/{project_id}/optimization/resume")
                    assert resp.status_code == 200, resp.text

                    for _ in range(5):
                        await asyncio.sleep(0)

                    # Verify get_optimization_state was called
                    assert mock_get.called, "get_optimization_state must be called on arch_split=True"

                    # Verify put_optimization_state was called to persist the reset with expected_version=1
                    assert mock_put.called, "put_optimization_state must be called to persist reset"
                    call_kwargs = mock_put.call_args.kwargs
                    assert "expected_version" in call_kwargs
                    assert call_kwargs["expected_version"] == 1

                finally:
                    get_job_tracker()._reset()

    @pytest.mark.asyncio
    async def test_resume_arch_split_false_unchanged(self, client, app, tmp_path, monkeypatch):
        """On arch_split=False, resume uses OptimizationState.load and save (old path)."""
        from unittest.mock import AsyncMock, patch
        import asyncio
        from app.services.jobs import get_job_tracker
        from app.services.mcp_tools import dispatch as dispatch_mod
        # The loop driver moved out of dispatch; patch its defining module.
        import app.services.optimization_loop_orchestrator as loop_mod

        project_id = "test-resume-old"
        _scaffold_project(tmp_path / "projects", project_id, with_synthetic=False)
        project_dir = tmp_path / "projects" / project_id

        # Create terminal state on disk
        from app.services.optimization_state import OptimizationState
        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"Correctness": 0.8},
            max_iterations=5,
        )
        state.status = "cancelled"
        state.save()

        # arch_split defaults to False (no patch needed)
        with patch("app.platform.get_platform") as mock_platform:
            mock_store = mock_platform.return_value.metadata_store
            mock_put = AsyncMock(return_value=1)
            mock_get = AsyncMock(return_value=None)
            mock_store.put_optimization_state = mock_put
            mock_store.get_optimization_state = mock_get
            allow_launch_admission(mock_store)

            captured = {}

            async def fake_run_loop(iteration_config):
                captured["config"] = iteration_config

            monkeypatch.setattr(loop_mod, "run_optimization_loop", fake_run_loop)

            try:
                resp = await client.post(f"/api/projects/{project_id}/optimization/resume")
                assert resp.status_code == 200, resp.text

                for _ in range(5):
                    await asyncio.sleep(0)

                # Verify get_optimization_state was NOT called (old path)
                assert not mock_get.called, "get_optimization_state must NOT be called on old path"

                # Verify state.json WAS written by resume
                from app.services.optimization_state import OPTIMIZATION_STATE_FILE
                state_file = project_dir / OPTIMIZATION_STATE_FILE
                assert state_file.exists(), "state.json must be written on old path"

            finally:
                get_job_tracker()._reset()


# REST cancel + read endpoints route through DynamoDB on arch_split=True


class TestOptimizationCancelDynamoDBPath:
    """Tests for cancel_optimization routing to DynamoDB on arch_split=True."""

    @pytest.mark.asyncio
    async def test_cancel_arch_split_true_writes_stopping_to_ddb(
        self, client, app, tmp_path, monkeypatch,
    ):
        """On arch_split=True with a running DDB record, cancel reads via get and writes stopping via put."""
        from unittest.mock import AsyncMock, patch
        from app.services.jobs import get_job_tracker

        _stub_spawned_work(monkeypatch)
        project_id = "test-cancel-ddb"
        _scaffold_project(tmp_path / "projects", project_id, with_synthetic=False)

        with patch("app.api.optimization.resolve_arch_split", new_callable=AsyncMock, return_value=True):
            with patch("app.platform.get_platform") as mock_platform:
                mock_store = mock_platform.return_value.metadata_store
                # First put returns version 3 (stopping-save); second returns 4 (orphan flip if it fires)
                mock_put = AsyncMock(side_effect=[3, 4])
                mock_get = AsyncMock(return_value={
                    "version": 2,
                    "status": "running",
                    "iterations": [],
                    "target_metric_scores": {"Correctness": 0.8},
                    "max_iterations": 5,
                    "plateau_window": 3,
                    "max_iter_cap": 5,
                })
                mock_store.put_optimization_state = mock_put
                mock_store.get_optimization_state = mock_get
                allow_launch_admission(mock_store)
                # The orphan-flip branch also releases the loop's eval lock
                # (holder-fenced no-op when a manual eval holds it instead).
                mock_store.release_eval_lock = AsyncMock()

                try:
                    resp = await client.post(f"/api/projects/{project_id}/optimization/cancel")
                    assert resp.status_code == 200, resp.text

                    # Verify get_optimization_state was called (read, not state.json)
                    assert mock_get.called, "get_optimization_state must be called on arch_split=True"

                    # Verify put_optimization_state was called for the stopping-save
                    assert mock_put.called, "put_optimization_state must be called to write stopping"
                    # First put threaded the read version (2) as expected_version
                    first_call_kwargs = mock_put.call_args_list[0].kwargs
                    assert first_call_kwargs.get("expected_version") == 2

                    # Verify state.json was NOT written
                    from app.services.optimization_state import OPTIMIZATION_STATE_FILE
                    project_dir = tmp_path / "projects" / project_id
                    state_file = project_dir / OPTIMIZATION_STATE_FILE
                    assert not state_file.exists(), "state.json must NOT be written on new path"

                finally:
                    get_job_tracker()._reset()

    @pytest.mark.asyncio
    async def test_cancel_arch_split_true_orphan_flip_threads_version(
        self, client, app, tmp_path, monkeypatch,
    ):
        """On arch_split=True orphan case (no orchestrator task), the direct-flip to cancelled threads the returned version."""
        from unittest.mock import AsyncMock, patch
        from app.services.jobs import get_job_tracker

        _stub_spawned_work(monkeypatch)
        project_id = "test-cancel-orphan"
        _scaffold_project(tmp_path / "projects", project_id, with_synthetic=False)

        with patch("app.api.optimization.resolve_arch_split", new_callable=AsyncMock, return_value=True):
            with patch("app.platform.get_platform") as mock_platform:
                mock_store = mock_platform.return_value.metadata_store
                # stopping-save returns 3; orphan direct-flip must thread 3 (not guess +1)
                mock_put = AsyncMock(side_effect=[3, 4])
                mock_get = AsyncMock(return_value={
                    "version": 2,
                    "status": "running",
                    "iterations": [],
                    "target_metric_scores": {"Correctness": 0.8},
                    "max_iterations": 5,
                    "plateau_window": 3,
                    "max_iter_cap": 5,
                })
                mock_store.put_optimization_state = mock_put
                mock_store.get_optimization_state = mock_get
                allow_launch_admission(mock_store)
                mock_release = AsyncMock()
                mock_store.release_eval_lock = mock_release

                try:
                    # No orchestrator task exists (nothing was started) → orphan direct-flip fires
                    resp = await client.post(f"/api/projects/{project_id}/optimization/cancel")
                    assert resp.status_code == 200, resp.text

                    # Two put calls: stopping-save (version 2) + orphan flip (version 3 returned by first save)
                    assert mock_put.call_count == 2, "orphan case must call put twice (stopping + cancelled)"
                    second_call_kwargs = mock_put.call_args_list[1].kwargs
                    # The orphan flip threads the version RETURNED from the stopping-save (3), not a +1 guess
                    assert second_call_kwargs.get("expected_version") == 3

                    # The orphan flip also releases the loop's eval lock — the
                    # dead (or never-adopted) loop can never run its terminal
                    # tail, and the reconciler's stale-lock duty is keyed off
                    # jobs, so without this the lock leaks forever and wedges the
                    # next dispatch. Holder-fenced: the loop's synthetic holder id.
                    assert mock_release.await_count == 1
                    release_args = mock_release.await_args.args
                    assert release_args[2] == f"optimization-{project_id}"

                finally:
                    get_job_tracker()._reset()

    @pytest.mark.asyncio
    async def test_cancel_no_active_optimization_returns_409(self, client, app, tmp_path):
        """On arch_split=True with no active optimization (get returns None), cancel returns 409."""
        from unittest.mock import AsyncMock, patch
        from app.services.jobs import get_job_tracker

        project_id = "test-cancel-none"
        _scaffold_project(tmp_path / "projects", project_id, with_synthetic=False)

        with patch("app.api.optimization.resolve_arch_split", new_callable=AsyncMock, return_value=True):
            with patch("app.platform.get_platform") as mock_platform:
                mock_store = mock_platform.return_value.metadata_store
                mock_put = AsyncMock(return_value=1)
                mock_get = AsyncMock(return_value=None)  # No record
                mock_store.put_optimization_state = mock_put
                mock_store.get_optimization_state = mock_get
                allow_launch_admission(mock_store)

                try:
                    resp = await client.post(f"/api/projects/{project_id}/optimization/cancel")
                    assert resp.status_code == 409, resp.text
                    assert "No active optimization to cancel" in resp.text

                finally:
                    get_job_tracker()._reset()

    @pytest.mark.asyncio
    async def test_cancel_terminal_status_returns_409(self, client, app, tmp_path):
        """On arch_split=True with a terminal DDB record, cancel returns 409."""
        from unittest.mock import AsyncMock, patch
        from app.services.jobs import get_job_tracker

        project_id = "test-cancel-terminal"
        _scaffold_project(tmp_path / "projects", project_id, with_synthetic=False)

        with patch("app.api.optimization.resolve_arch_split", new_callable=AsyncMock, return_value=True):
            with patch("app.platform.get_platform") as mock_platform:
                mock_store = mock_platform.return_value.metadata_store
                mock_put = AsyncMock(return_value=1)
                mock_get = AsyncMock(return_value={
                    "version": 4,
                    "status": "cancelled",  # terminal
                    "iterations": [],
                    "target_metric_scores": {"Correctness": 0.8},
                    "max_iterations": 5,
                    "plateau_window": 3,
                    "max_iter_cap": 5,
                })
                mock_store.put_optimization_state = mock_put
                mock_store.get_optimization_state = mock_get
                allow_launch_admission(mock_store)

                try:
                    resp = await client.post(f"/api/projects/{project_id}/optimization/cancel")
                    assert resp.status_code == 409, resp.text

                finally:
                    get_job_tracker()._reset()


class TestOptimizationReadDynamoDBPath:
    """Tests for GET /activity reading from DynamoDB on arch_split=True."""

    @pytest.mark.asyncio
    async def test_get_project_activity_reads_ddb(self, client, app, tmp_path):
        """GET /activity on arch_split=True reads the optimization portion via get_optimization_state."""
        from unittest.mock import AsyncMock, patch
        from app.services.jobs import get_job_tracker

        project_id = "test-activity-ddb"
        _scaffold_project(tmp_path / "projects", project_id, with_synthetic=False)

        with patch("app.api.optimization.resolve_arch_split", new_callable=AsyncMock, return_value=True):
            with patch("app.platform.get_platform") as mock_platform:
                mock_store = mock_platform.return_value.metadata_store
                mock_get = AsyncMock(return_value={
                    "version": 2,
                    "status": "running",
                    "iterations": [],
                    "target_metric_scores": {"Correctness": 0.8},
                    "max_iterations": 5,
                    "plateau_window": 3,
                    "max_iter_cap": 5,
                })
                mock_store.get_optimization_state = mock_get
                allow_launch_admission(mock_store)
                # The activity view now rebuilds jobs from the durable store before
                # reading the active eval, so a job dispatched in another process is
                # visible here. Give the mocked store an awaitable job listing so the
                # rebuild is a clean no-op — this test asserts only the optimization
                # portion's read path.
                mock_store.list_jobs = AsyncMock(return_value=[])

                try:
                    resp = await client.get(f"/api/projects/{project_id}/activity")
                    assert resp.status_code == 200, resp.text

                    # Verify the optimization portion was read via get_optimization_state
                    assert mock_get.called, "get_optimization_state must be called for the optimization portion"

                    data = resp.json()
                    assert data["optimization"] is not None
                    assert data["optimization"]["status"] == "running"

                finally:
                    get_job_tracker()._reset()


class TestOptimizationAgentNotifications:
    """Tests for UI-initiated optimization start/resume/cancel agent notifications."""

    async def test_start_delivers_orchestrator_message(self, client, tmp_path, monkeypatch):
        """Start delivers one orchestrator message naming trigger + starting point."""
        from unittest.mock import AsyncMock, patch
        from app.services import optimization_loop_orchestrator as loop_mod
        from app.services import completion_turn

        project_id = "test-start-notify"
        _scaffold_project(tmp_path / "projects", project_id, with_synthetic=False)

        # Patch the loop so it doesn't actually run
        fake_run_loop = AsyncMock()
        monkeypatch.setattr(loop_mod, "run_optimization_loop", fake_run_loop)

        # Capture the message delivered to _submit_orchestrator_message
        captured_messages = []

        async def capture_submit(pid: str, msg: str, *, lifecycle: bool = False):
            captured_messages.append({"project_id": pid, "message": msg, "lifecycle": lifecycle})

        with patch.object(completion_turn, "_submit_orchestrator_message", side_effect=capture_submit):
            resp = await client.post(
                f"/api/projects/{project_id}/optimization/start",
                json={"data_source": "raw"}
            )
            assert resp.status_code == 200

        # Assert exactly one message was delivered
        assert len(captured_messages) == 1
        msg = captured_messages[0]
        assert msg["project_id"] == project_id

        # Message must name: trigger, prompt version, target scores, max_iterations
        text = msg["message"]
        text_lower = text.lower()
        assert "started from the ui" in text_lower
        assert "baseline" in text  # prompt version
        assert "exactmatch" in text_lower or "exact" in text_lower  # target metric
        assert "0.85" in text or "85" in text  # target score

    async def test_start_notification_non_fatal(self, client, tmp_path, monkeypatch):
        """Start still succeeds and spawns loop when notification raises."""
        from unittest.mock import AsyncMock, patch
        from app.services import optimization_loop_orchestrator as loop_mod
        from app.services import completion_turn

        project_id = "test-start-nonfatal"
        _scaffold_project(tmp_path / "projects", project_id, with_synthetic=False)

        fake_run_loop = AsyncMock()
        monkeypatch.setattr(loop_mod, "run_optimization_loop", fake_run_loop)

        # Patch _submit_orchestrator_message to raise
        async def boom(pid: str, msg: str, *, lifecycle: bool = False):
            raise RuntimeError("notification failure")

        with patch.object(completion_turn, "_submit_orchestrator_message", side_effect=boom):
            resp = await client.post(
                f"/api/projects/{project_id}/optimization/start",
                json={"data_source": "raw"}
            )
            # Endpoint still returns 200
            assert resp.status_code == 200
            # Loop still spawned
            assert fake_run_loop.called

    async def test_resume_delivers_orchestrator_message(self, client, tmp_path, monkeypatch):
        """Resume delivers one message naming trigger + iteration + prompt version."""
        from unittest.mock import AsyncMock, patch
        from app.services import optimization_loop_orchestrator as loop_mod
        from app.services import completion_turn
        from app.services.optimization_state import OptimizationState

        project_id = "test-resume-notify"
        project_dir = tmp_path / "projects" / project_id
        _scaffold_project(tmp_path / "projects", project_id, with_synthetic=False)

        # Seed a resumable state with 3 completed iterations
        opt_state = OptimizationState(
            project_dir=project_dir,
            status="cancelled",
            target_metric_scores={"ExactMatch": 0.85},
            max_iterations=10,
            plateau_window=3,
            iterations=[
                {"iteration": 1, "prompt_version": "baseline", "passed": 10, "total_cases": 20},
                {"iteration": 2, "prompt_version": "v1", "passed": 12, "total_cases": 20},
                {"iteration": 3, "prompt_version": "v2", "passed": 15, "total_cases": 20},
            ],
        )
        opt_state.save()

        fake_run_loop = AsyncMock()
        monkeypatch.setattr(loop_mod, "run_optimization_loop", fake_run_loop)

        captured_messages = []

        async def capture_submit(pid: str, msg: str, *, lifecycle: bool = False):
            captured_messages.append({"project_id": pid, "message": msg, "lifecycle": lifecycle})

        with patch.object(completion_turn, "_submit_orchestrator_message", side_effect=capture_submit):
            resp = await client.post(f"/api/projects/{project_id}/optimization/resume")
            assert resp.status_code == 200

        assert len(captured_messages) == 1
        msg = captured_messages[0]
        text = msg["message"]
        text_lower = text.lower()
        assert "resumed from the ui" in text_lower
        assert "iteration 3" in text_lower or "3" in text
        assert "baseline" in text  # prompt version detected on disk

    async def test_cancel_delivers_orchestrator_message(self, client, tmp_path, monkeypatch):
        """Cancel delivers one message naming trigger + iteration, after drains."""
        from unittest.mock import AsyncMock, patch
        from app.services import completion_turn
        from app.services.optimization_state import OptimizationState
        from app.services.jobs import get_job_tracker

        project_id = "test-cancel-notify"
        project_dir = tmp_path / "projects" / project_id
        _scaffold_project(tmp_path / "projects", project_id, with_synthetic=False)

        # Seed a running state at iteration 2
        opt_state = OptimizationState(
            project_dir=project_dir,
            status="running",
            target_metric_scores={"ExactMatch": 0.85},
            max_iterations=10,
            plateau_window=3,
            iterations=[
                {"iteration": 1, "prompt_version": "baseline", "passed": 10, "total_cases": 20},
                {"iteration": 2, "prompt_version": "v1", "passed": 12, "total_cases": 20},
            ],
        )
        opt_state.save()

        captured_messages = []

        async def capture_submit(pid: str, msg: str, *, lifecycle: bool = False):
            captured_messages.append({"project_id": pid, "message": msg, "lifecycle": lifecycle})

        try:
            with patch.object(completion_turn, "_submit_orchestrator_message", side_effect=capture_submit):
                resp = await client.post(f"/api/projects/{project_id}/optimization/cancel")
                assert resp.status_code == 200

            assert len(captured_messages) == 1
            msg = captured_messages[0]
            text = msg["message"]
            text_lower = text.lower()
            assert "cancelled from the ui" in text_lower
            assert "iteration 2" in text_lower or "2" in text
        finally:
            # Reset job tracker for next test
            get_job_tracker()._jobs.clear()

    async def test_cancel_notification_survives_drains(self, client, tmp_path, monkeypatch):
        """Cancel notification is submitted AFTER drains, not before."""
        from unittest.mock import AsyncMock, patch, call
        from app.services import completion_turn
        from app.services.optimization_state import OptimizationState
        from app.services.jobs import get_job_tracker

        project_id = "test-cancel-drain-order"
        project_dir = tmp_path / "projects" / project_id
        _scaffold_project(tmp_path / "projects", project_id, with_synthetic=False)

        opt_state = OptimizationState(
            project_dir=project_dir,
            status="running",
            target_metric_scores={"ExactMatch": 0.85},
            max_iterations=10,
            plateau_window=3,
            iterations=[
                {"iteration": 1, "prompt_version": "baseline", "passed": 10, "total_cases": 20},
            ],
        )
        opt_state.save()

        # Track call order
        call_order = []

        def track_drain_pending(pid):
            call_order.append("drain_pending")
            return 0

        def track_drain_queued(pid, origin):
            call_order.append("drain_queued")
            return 0

        async def track_submit(pid, msg, *, lifecycle=False):
            call_order.append("submit_message")

        try:
            with patch("app.services.completion_turn.drain_pending_completions", side_effect=track_drain_pending), \
                 patch("app.services.agent_turn.get_agent_turn_tracker") as mock_tracker, \
                 patch.object(completion_turn, "_submit_orchestrator_message", side_effect=track_submit):

                mock_tracker.return_value.drain_queued_turns = track_drain_queued
                mock_tracker.return_value.get_active_turn = lambda *args, **kwargs: None

                resp = await client.post(f"/api/projects/{project_id}/optimization/cancel")
                assert resp.status_code == 200

            # Assert the message was submitted AFTER both drains
            assert call_order == ["drain_pending", "drain_queued", "submit_message"]
        finally:
            get_job_tracker()._jobs.clear()

    async def test_start_no_notification_on_early_guards(self, client, tmp_path):
        """Early HTTPException guards do not emit any notification."""
        from unittest.mock import patch
        from app.services import completion_turn

        project_id = "test-early-guard"
        _scaffold_project(tmp_path / "projects", project_id, with_synthetic=False)

        # Remove the harness so validation fails
        (tmp_path / "projects" / project_id / "run_eval.py").unlink()

        captured_messages = []

        async def capture_submit(pid: str, msg: str, *, lifecycle: bool = False):
            captured_messages.append({"project_id": pid, "message": msg, "lifecycle": lifecycle})

        with patch.object(completion_turn, "_submit_orchestrator_message", side_effect=capture_submit):
            resp = await client.post(
                f"/api/projects/{project_id}/optimization/start",
                json={"data_source": "raw"}
            )
            # Expect 400 or 409 (harness missing)
            assert resp.status_code >= 400

        # No message should have been delivered
        assert len(captured_messages) == 0


class TestNoInventedDefaults:
    """/start and /resume never fabricate optimization parameters.

    These endpoints used to substitute ExactMatch@0.85 / 10 iterations /
    plateau 3 when the harness config lacked them — so a half-configured
    project optimized against targets nobody chose. A missing field is now
    a 400 naming it."""

    def _scaffold_without(self, projects_dir, project_id: str, omit: set[str]) -> None:
        """Minimal project whose src/config.py omits the given fields."""
        project_dir = projects_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "config.json").write_text(json.dumps({
            "id": project_id,
            "name": "Test project",
            "description": "",
            "created_at": "2026-01-01T00:00:00Z",
            "owner_id": "local-dev",
        }))
        (project_dir / "run_eval.py").write_text("# harness sentinel\n")
        (project_dir / "src").mkdir(exist_ok=True)

        fields = {
            "models": 'models: list = field(default_factory=lambda: ["m1"])',
            "metrics": 'metrics: list = field(default_factory=lambda: [{"name": "ExactMatch", "type": "exact_match"}])',
            "data_file": 'data_file: str = "seed.csv"',
            "field_mappings": 'field_mappings: dict = field(default_factory=lambda: {"input": "q", "expected_output": "a"})',
            "target_metric_scores": 'target_metric_scores: dict = field(default_factory=lambda: {"ExactMatch": 0.9})',
            "max_iterations": "max_iterations: int = 5",
            "plateau_window": "plateau_window: int = 2",
        }
        body = "\n    ".join(v for k, v in fields.items() if k not in omit)
        (project_dir / "src" / "config.py").write_text(
            "from dataclasses import dataclass, field\n\n"
            "@dataclass\nclass EvalConfig:\n    " + body + "\n"
        )
        raw_dir = project_dir / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "seed.csv").write_text("q,a\nx,y\n")
        (project_dir / "prompts").mkdir(exist_ok=True)
        (project_dir / "prompts" / "baseline.md").write_text("prompt\n")

    @pytest.mark.parametrize("omitted", [
        "target_metric_scores", "max_iterations", "plateau_window",
    ])
    async def test_start_400s_naming_the_missing_field(
        self, client, app, omitted,
    ):
        import app.main as main_module
        project_id = f"proj-missing-{omitted.replace('_', '-')}"
        self._scaffold_without(main_module.PROJECTS_DIR, project_id, {omitted})

        resp = await client.post(f"/api/projects/{project_id}/optimization/start")

        assert resp.status_code == 400
        assert omitted in resp.json()["detail"]

    async def test_metrics_absent_from_parse_does_not_400(
        self, client, app, monkeypatch,
    ):
        """Regression pin: the AST
        parse only sees config.py LITERALS. A builder-scaffolded GEval metric
        lives as code in src/metrics/metrics.py, so parse-absence is NOT
        not-configured — /start must never reject on it. The harness's
        build_metrics owns metrics and fails loudly when truly missing."""
        import app.main as main_module
        _stub_spawned_work(monkeypatch)
        project_id = "proj-code-built-metrics"
        self._scaffold_without(main_module.PROJECTS_DIR, project_id, {"metrics"})

        resp = await client.post(f"/api/projects/{project_id}/optimization/start")

        assert not (
            resp.status_code == 400
            and "metrics" in resp.json().get("detail", "")
        )

    async def test_start_proceeds_with_complete_config(
        self, client, app, monkeypatch,
    ):
        """Control: nothing omitted → the missing-fields guard passes (the
        request advances past it; any later failure is NOT the guard's 400)."""
        import app.main as main_module
        _stub_spawned_work(monkeypatch)
        project_id = "proj-complete-config"
        self._scaffold_without(main_module.PROJECTS_DIR, project_id, set())

        resp = await client.post(f"/api/projects/{project_id}/optimization/start")

        assert not (
            resp.status_code == 400
            and "missing required optimization fields" in resp.json().get("detail", "")
        )


class TestPhaseThroughFullReadPath:
    """``phase`` must survive the endpoint's hand-built response dict — a third
    copy of the field set, between the durable record and the SPA's render
    branch. This test reads through the REAL endpoint so the next display field
    that skips the middle hop fails here instead of in production."""

    @pytest.mark.asyncio
    async def test_phase_survives_endpoint_serialization(self, client, app, tmp_path):
        from unittest.mock import AsyncMock, patch
        from app.services.jobs import get_job_tracker

        project_id = "test-phase-read-path"
        _scaffold_project(tmp_path / "projects", project_id, with_synthetic=False)

        with patch("app.api.optimization.resolve_arch_split", new_callable=AsyncMock, return_value=True):
            with patch("app.platform.get_platform") as mock_platform:
                mock_store = mock_platform.return_value.metadata_store
                mock_store.get_optimization_state = AsyncMock(return_value={
                    "version": 7,
                    "status": "running",
                    "phase": "optimizing_prompt",
                    "iterations": [],
                    "target_metric_scores": {"Correctness": 0.8},
                    "max_iterations": 5,
                    "plateau_window": 3,
                    "max_iter_cap": 5,
                })
                # /activity rebuilds jobs from the store before reading the
                # active eval; an empty listing keeps that a no-op here.
                mock_store.list_jobs = AsyncMock(return_value=[])

                try:
                    resp = await client.get(f"/api/projects/{project_id}/activity")

                    assert resp.status_code == 200, resp.text
                    data = resp.json()["optimization"]
                    assert data["phase"] == "optimizing_prompt", (
                        "phase must survive the endpoint's response dict — dropping "
                        "it there makes the bar show 'working...' while the record "
                        "says 'optimizing_prompt'."
                    )
                finally:
                    get_job_tracker()._reset()


class TestOptimizationStartWorkloadFingerprint:
    """``POST /optimization/start`` must record what the run will execute.

    Resume compares the live config against the fingerprint captured at
    start; a record with no fingerprint gets no drift detection at all.
    This path captured none — every UI-started optimization could be
    resumed against swapped models, changed metrics or a different dataset
    silently. It also dropped the project's ``plateau_threshold``, so a
    UI-started run plateaued on the 0.02 default instead of the configured
    value while an agent-dispatched run of the same project used the
    configured one.
    """

    async def test_start_records_effective_workload_and_threshold(
        self, client, app, monkeypatch,
    ):
        import asyncio
        import app.main as main_module
        import app.services.optimization_loop_orchestrator as loop_mod

        project_id = "proj-workload-fingerprint"
        _scaffold_project(main_module.PROJECTS_DIR, project_id, with_synthetic=True)
        project_dir = main_module.PROJECTS_DIR / project_id
        # A non-default plateau threshold discriminates "config honored" from
        # "DEFAULT_PLATEAU_THRESHOLD substituted".
        cfg = (project_dir / "src" / "config.py").read_text()
        (project_dir / "src" / "config.py").write_text(
            cfg + "    plateau_threshold: float = 0.005\n"
        )
        main_module._project_manager = None

        monkeypatch.setattr(
            loop_mod, "run_optimization_loop", lambda iteration_config: asyncio.sleep(0),
        )
        # Both endpoints notify the agent on the way out, which starts a real
        # CLI turn (and a real subprocess) unless stubbed. Patch the DEFINING
        # module — the endpoint imports the function locally.
        import app.services.completion_turn as ct_mod
        monkeypatch.setattr(
            ct_mod, "request_terminal_message",
            lambda *a, **k: asyncio.sleep(0),
        )

        resp = await client.post(
            f"/api/projects/{project_id}/optimization/start",
            json={"data_source": "synthetic"},
        )
        assert resp.status_code == 200, resp.text
        for _ in range(5):
            await asyncio.sleep(0)

        state = json.loads(
            (project_dir / "output" / "optimization" / "state.json").read_text()
        )
        workload = state.get("workload")
        assert workload is not None, (
            "start recorded no workload fingerprint — resume then has nothing "
            "to diff and reports no drift no matter what the user changed"
        )
        assert workload["models"] == ["us.anthropic.claude-haiku-4-5-20251001-v1:0"]
        assert "metrics" not in workload, "metrics are code, not part of the fingerprint"
        # The EFFECTIVE dataset, not src/config.py's raw seed.csv: switching
        # data source is the most likely edit between cancel and resume, and a
        # fingerprint of the wrong file can never detect it.
        assert "synthetic_only_" in (workload["data_file"] or ""), workload["data_file"]

        assert state["plateau_threshold"] == 0.005, (
            "the project's configured plateau_threshold must reach the state "
            "record; substituting the default changes when the loop stops"
        )

    async def test_resume_detects_data_source_drift(
        self, client, app, monkeypatch,
    ):
        """A resume onto a different dataset must register as drift.

        The whole point of the fingerprint: the resumed iterations are not
        comparable with the prior ones, so ``check_convergence`` withholds
        "target reached" on the first post-drift row. Both sides used to
        read ``src/config.py``'s ``data_file`` and ignore the resolved
        override, which made switching data source — the one edit the
        endpoint itself offers — invisible to the check.
        """
        import asyncio
        import app.main as main_module
        import app.services.optimization_loop_orchestrator as loop_mod

        project_id = "proj-resume-drift"
        _scaffold_project(main_module.PROJECTS_DIR, project_id, with_synthetic=True)
        project_dir = main_module.PROJECTS_DIR / project_id
        main_module._project_manager = None

        monkeypatch.setattr(
            loop_mod, "run_optimization_loop", lambda iteration_config: asyncio.sleep(0),
        )
        # Both endpoints notify the agent on the way out, which starts a real
        # CLI turn (and a real subprocess) unless stubbed. Patch the DEFINING
        # module — the endpoint imports the function locally.
        import app.services.completion_turn as ct_mod
        monkeypatch.setattr(
            ct_mod, "request_terminal_message",
            lambda *a, **k: asyncio.sleep(0),
        )

        # Start on raw, then park the run so it is resumable.
        assert (await client.post(
            f"/api/projects/{project_id}/optimization/start"
        )).status_code == 200
        for _ in range(5):
            await asyncio.sleep(0)
        state_path = project_dir / "output" / "optimization" / "state.json"
        parked = json.loads(state_path.read_text())
        assert parked["workload"]["data_file"] == "seed.csv"
        parked["status"] = "cancelled"
        state_path.write_text(json.dumps(parked))

        # Resume on the synthetic set — a different dataset.
        resp = await client.post(
            f"/api/projects/{project_id}/optimization/resume",
            json={"data_source": "synthetic"},
        )
        assert resp.status_code == 200, resp.text
        for _ in range(5):
            await asyncio.sleep(0)

        resumed = json.loads(state_path.read_text())
        drift = resumed.get("pending_resume_drift")
        assert drift is not None, (
            "resuming onto a different dataset registered no drift — the "
            "fingerprint was compared against the wrong file"
        )
        assert drift["data_file"]["before"] == "seed.csv"
        assert "synthetic_only_" in drift["data_file"]["after"]


@pytest.mark.parametrize(
    "source",
    ["class EvalConfig(:\n", "models = [1, 2]\ndata_file = 'cases.csv'\n"],
    ids=["syntax-error", "models-not-strings"],
)
def test_an_unreadable_config_rejects_start_with_400(tmp_path, source):
    """Start and resume read config.py first; a broken file is a 400 naming it, not a 500."""
    from fastapi import HTTPException

    from app.api.optimization import _load_harness_config_or_400

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "config.py").write_text(source)
    with pytest.raises(HTTPException) as exc:
        _load_harness_config_or_400(tmp_path, "start")
    assert exc.value.status_code == 400
    assert "src/config.py could not be read" in exc.value.detail
