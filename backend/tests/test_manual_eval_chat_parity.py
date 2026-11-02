# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A manual eval says in chat how it ended, the way an optimization run does.

The optimization loop posts to chat at every ending: each iteration's results,
a failure, a stall, a crash, a cancel from the UI. A manual eval reached chat
only when it finished with results. A failure was silent from a detached worker
and described as "complete, pass rate 0/0" in-process, and a cancel from the UI
posted nothing at all.

The detached-worker failure path is covered with the poller, in
``test_job_progress_republisher.py``.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.platform import set_platform
from app.platform.dev_local import FilesystemMetadataStore
from app.services.completion_turn import (
    _build_synthetic_message,
    announce_eval_cancelled_from_ui,
)
from app.services.eval_cancel import cancel_eval_job
from app.services.jobs import Job, _default_progress, get_job_tracker
from app.services.resource_error import Codes, ResourceError

from .async_wait import until


def _eval_job(**overrides) -> Job:
    """A manual eval job on prompt ``v2``, with ``overrides`` applied."""
    fields = {
        "job_id": "job-1234abcd",
        "project_id": "proj-1",
        "type": "eval",
        "state": "failed",
        "config": {"prompt_version": "v2", "source": "manual"},
        "progress": {},
    }
    fields.update(overrides)
    return Job(**fields)


class TestAFailedEvalIsDescribedAsFailed:
    async def test_with_the_error_it_recorded(self):
        job = _eval_job(error=ResourceError(
            code=Codes.HARNESS_SUBPROCESS_FAILED,
            message="The harness exited with code 1",
        ))

        message = await _build_synthetic_message(job)

        assert message.startswith("Eval on prompt 'v2' failed: The harness exited with code 1")
        assert "complete" not in message.lower()
        assert "Pass rate" not in message

    async def test_rebuilt_from_the_durable_record_keeps_its_reason(self):
        # The record persists no error, but it persists the progress text that
        # ``fail_job`` writes.
        job = _eval_job(progress={
            "message": "Failed: Eval completed but its results could not be saved.",
        })

        message = await _build_synthetic_message(job)

        assert message.startswith(
            "Eval on prompt 'v2' failed: Eval completed but its results could not be saved."
        )

    async def test_with_no_recorded_reason_still_says_it_failed(self):
        # What the reconciler leaves when a worker vanishes: the last progress
        # line, which is not a failure reason.
        job = _eval_job(progress={"message": "Scoring: model-a (40/208)"})

        message = await _build_synthetic_message(job)

        assert message.startswith(
            "Eval on prompt 'v2' failed: its worker stopped without reporting a result"
        )


class TestAUiCancelIsAnnounced:
    async def test_it_posts_a_lifecycle_line_naming_how_far_the_eval_got(self):
        job = _eval_job(
            state="cancelled",
            progress={"models_completed": 1, "models_total": 3},
        )

        with patch(
            "app.services.completion_turn.request_terminal_message", new=AsyncMock()
        ) as post:
            announce_eval_cancelled_from_ui("proj-1", job)
            await until(lambda: post.await_count == 1, message="no line was posted")

        post.assert_awaited_once_with(
            "proj-1",
            "Eval of prompt 'v2' cancelled from the UI with 1 of 3 models finished.",
            lifecycle=True,
        )

    async def test_without_progress_it_still_names_the_prompt(self):
        job = _eval_job(state="cancelled")

        with patch(
            "app.services.completion_turn.request_terminal_message", new=AsyncMock()
        ) as post:
            announce_eval_cancelled_from_ui("proj-1", job)
            await until(lambda: post.await_count == 1, message="no line was posted")

        assert post.await_args.args[1] == "Eval of prompt 'v2' cancelled from the UI."

    async def test_a_loop_owned_eval_is_left_to_the_loop(self):
        job = _eval_job(
            state="cancelled",
            config={"prompt_version": "v2", "source": "optimization"},
        )

        with patch(
            "app.services.completion_turn.request_terminal_message", new=AsyncMock()
        ) as post:
            announce_eval_cancelled_from_ui("proj-1", job)

        post.assert_not_called()


class TestADetachedCancelSaysHowFarTheWorkerGot:
    async def test_the_line_counts_the_models_the_worker_finished(self, tmp_path):
        # A detached eval runs in its worker, so this process's copy of the job
        # keeps the progress it was created with. The worker's durable record is
        # what knows one of the four models had finished.
        store = FilesystemMetadataStore(projects_dir=tmp_path)
        await store.put_job({
            "owner_id": "user-1",
            "project_id": "proj-1",
            "job_id": "job-1234abcd",
            "type": "eval",
            "state": "running",
            "compute_handle": "arn:aws:ecs:us-east-1:123456789012:task/cluster/abc123",
            "created_at": "2026-01-01T00:00:00+00:00",
            "config": {"prompt_version": "v2", "source": "manual"},
            "progress": {
                "models_completed": 1,
                "models_total": 4,
                "message": "Complete: model-a (150/208 passed)",
            },
        })
        set_platform(SimpleNamespace(metadata_store=store))
        tracker = get_job_tracker()
        job = _eval_job(
            state="running",
            config={
                "prompt_version": "v2",
                "source": "manual",
                "arch_split": True,
                "user_id": "user-1",
            },
            progress=_default_progress(),
        )
        tracker._jobs[job.job_id] = job
        compute = MagicMock()
        compute.cancel = AsyncMock()
        compute.status = AsyncMock(return_value="STOPPED")

        with patch.object(
            tracker, "rebuild_jobs_from_store", new=AsyncMock()
        ), patch(
            "app.platform.job_compute.get_job_compute", return_value=compute
        ), patch(
            "app.services.eval_cancel._stop_optimization_loop", new=AsyncMock()
        ), patch(
            "app.services.completion_turn.request_terminal_message", new=AsyncMock()
        ) as post:
            outcome, _ = await cancel_eval_job(
                "proj-1", job.job_id, user_id="user-1", project_dir=tmp_path,
                arch_split=True,
            )
            announce_eval_cancelled_from_ui("proj-1", job)
            await until(lambda: post.await_count == 1, message="no line was posted")
            await tracker.drain_pending_writes()

        assert outcome == "cancelled"
        assert post.await_args.args[1] == (
            "Eval of prompt 'v2' cancelled from the UI with 1 of 4 models finished."
        )
        # The cancelled state carries the worker's progress, not this copy's.
        record = await store.get_job("user-1", "proj-1", job.job_id)
        assert record["state"] == "cancelled"
        assert record["progress"]["models_completed"] == 1
        assert record["progress"]["models_total"] == 4


@pytest.fixture
def app(tmp_path):
    import app.main as main_module
    original = main_module.PROJECTS_DIR
    main_module.PROJECTS_DIR = tmp_path / "projects"
    main_module.PROJECTS_DIR.mkdir()
    main_module._project_manager = None
    application = create_app()
    yield application
    get_job_tracker()._reset()
    main_module._project_manager = None
    main_module.PROJECTS_DIR = original


@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def _running_manual_eval(client) -> tuple[str, Job]:
    """A project with a manual eval running in it."""
    resp = await client.post("/api/projects", json={"name": "Cancel From UI"})
    project_id = resp.json()["id"]
    tracker = get_job_tracker()
    job = tracker.create_job(
        project_id,
        job_type="eval",
        config={"model_ids": ["model-a", "model-b"], "prompt_version": "v2", "source": "manual"},
    )
    tracker.update_state(job.job_id, "running")
    job.progress.update({"models_completed": 1, "models_total": 2})
    return project_id, job


class TestTheUiCancelRoute:
    async def test_a_confirmed_cancel_is_announced(self, client):
        project_id, job = await _running_manual_eval(client)

        with patch(
            "app.services.completion_turn.request_terminal_message", new=AsyncMock()
        ) as post:
            resp = await client.post(
                f"/api/projects/{project_id}/eval/jobs/{job.job_id}/cancel"
            )
            assert resp.status_code == 200
            await until(lambda: post.await_count == 1, message="the cancel was not announced")

        post.assert_awaited_once_with(
            project_id,
            "Eval of prompt 'v2' cancelled from the UI with 1 of 2 models finished.",
            lifecycle=True,
        )

    async def test_a_cancel_that_could_not_stop_the_worker_is_not_announced(self, client):
        # The eval is still running, so saying it was cancelled would be false.
        project_id, job = await _running_manual_eval(client)

        with patch(
            "app.services.eval_cancel.cancel_eval_job",
            new=AsyncMock(return_value=("stop_failed", "running")),
        ), patch(
            "app.services.completion_turn.request_terminal_message", new=AsyncMock()
        ) as post:
            resp = await client.post(
                f"/api/projects/{project_id}/eval/jobs/{job.job_id}/cancel"
            )

        assert resp.status_code == 503
        post.assert_not_called()
