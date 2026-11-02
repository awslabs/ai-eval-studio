# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A cancel must not report success until the compute is confirmed stopped.

The old order recorded the terminal state first — publishing "cancelled" to the
SPA and releasing the project's eval lock — and only then attempted StopTask, with
no handling for its failure. A failed stop therefore left the UI saying cancelled
while the worker kept invoking models and billing, the lock free for a second eval
on the same project, and a retry short-circuiting on "already terminal" without
ever trying again.

A handle-less record made it worse: it fell through to the in-process branch,
which killed a pid this process never owned. Handles are written AFTER RunTask
returns, so that window is ordinary, not exotic.
"""
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio

TASK_ARN = "arn:aws:ecs:us-east-1:123456789012:task/cluster/abc123"


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch):
    """Collapse the stop-confirmation and tag-retry backoffs.

    Both are measured in seconds by design — ECS gives the worker a 30s
    stopTimeout and list_tasks takes 5-15s to show a new task — and none of these
    tests are about the durations.
    """
    monkeypatch.setattr("app.services.eval_cancel._STOP_CONFIRM_DELAY_S", 0)
    monkeypatch.setattr("app.services.eval_cancel._TAG_LOOKUP_DELAY_S", 0)


def _tracker(status: str = "running", pid: int | None = None) -> MagicMock:
    tracker = MagicMock()
    tracker.rebuild_jobs_from_store = AsyncMock()
    tracker.get_job_for_project.return_value = MagicMock(status=status, pid=pid)
    tracker.cancel_job = MagicMock()
    return tracker


def _compute(*, stopped: bool = True) -> MagicMock:
    """A job-compute double.

    ``status`` matters as much as ``cancel``: StopTask returning only means ECS
    accepted the request, so the cancel path confirms the containers stopped
    before treating the compute as gone.
    """
    compute = MagicMock()
    compute.cancel = AsyncMock()
    compute.cancel_by_tag = AsyncMock(return_value=TASK_ARN)
    compute.status = AsyncMock(return_value="STOPPED" if stopped else "STOPPING")
    return compute


def _store() -> MagicMock:
    """The durable store, for the cancel mark and the worker's progress read."""
    store = MagicMock()
    store.mark_cancel_requested = AsyncMock()
    store.get_job = AsyncMock(return_value=None)
    return store


async def _cancel(tracker, compute, *, handle, arch_split=True, store=None):
    from app.services.eval_cancel import cancel_eval_job

    platform = MagicMock()
    platform.metadata_store = store if store is not None else _store()

    with patch(
        "app.services.eval_cancel.get_job_tracker", return_value=tracker
    ), patch(
        "app.api.eval_jobs._resolve_compute_handle",
        new=AsyncMock(return_value=handle),
    ), patch(
        "app.platform.job_compute.get_job_compute", return_value=compute
    ), patch(
        "app.services.eval_cancel.get_platform", return_value=platform
    ), patch(
        "app.services.eval_cancel._stop_optimization_loop", new=AsyncMock()
    ):
        return await cancel_eval_job(
            "proj-abc",
            "job-123",
            user_id="user-xyz",
            project_dir=Path("/tmp/nonexistent"),
            arch_split=arch_split,
        )


async def test_worker_is_stopped_before_the_job_is_marked_cancelled():
    """Ordering is the defect: the terminal write must come after the stop."""
    order: list[str] = []
    compute = _compute()
    compute.cancel = AsyncMock(side_effect=lambda _h: order.append("stop"))

    tracker = _tracker()
    tracker.cancel_job = MagicMock(side_effect=lambda _j: order.append("terminal"))

    outcome, _ = await _cancel(tracker, compute, handle=TASK_ARN)

    assert outcome == "cancelled"
    assert order == ["stop", "terminal"], (
        "the job must not read terminal until the worker is stopped"
    )


async def test_the_cancel_is_recorded_before_the_worker_is_stopped():
    """The mark has to be on the record before ECS takes the task off its list.

    StopTask removes the task from the running list at once, and from then until
    the job is marked cancelled the reconciler would read a crashed worker. The
    mark is what it reads instead, so it must already be there.
    """
    order: list[str] = []
    store = _store()
    store.mark_cancel_requested = AsyncMock(side_effect=lambda *_a: order.append("mark"))
    compute = _compute()
    compute.cancel = AsyncMock(side_effect=lambda _h: order.append("stop"))
    tracker = _tracker()
    tracker.cancel_job = MagicMock(side_effect=lambda _j: order.append("terminal"))

    outcome, _ = await _cancel(tracker, compute, handle=TASK_ARN, store=store)

    assert outcome == "cancelled"
    assert order == ["mark", "stop", "terminal"]
    store.mark_cancel_requested.assert_awaited_once_with("user-xyz", "proj-abc", "job-123")


async def test_a_failed_mark_does_not_stop_the_cancel():
    """Stopping the worker matters more than labelling the job."""
    store = _store()
    store.mark_cancel_requested = AsyncMock(side_effect=RuntimeError("throttled"))
    compute = _compute()
    tracker = _tracker()

    outcome, _ = await _cancel(tracker, compute, handle=TASK_ARN, store=store)

    assert outcome == "cancelled"
    compute.cancel.assert_awaited_once()
    tracker.cancel_job.assert_called_once_with("job-123")


async def test_a_failed_stop_does_not_mark_the_job_cancelled():
    """StopTask failing must leave the job running and say so."""
    compute = _compute()
    compute.cancel = AsyncMock(side_effect=RuntimeError("ecs unavailable"))
    compute.cancel_by_tag = AsyncMock(side_effect=RuntimeError("ecs unavailable"))

    tracker = _tracker()
    outcome, status = await _cancel(tracker, compute, handle=TASK_ARN)

    assert outcome == "stop_failed"
    assert status == "running"
    # A job whose worker is still running must not read terminal — the retry and
    # the reconciler both need it to stay visible.
    tracker.cancel_job.assert_not_called()


async def test_a_handle_less_cloud_job_is_stopped_by_its_tag():
    """The handle is written after RunTask, so a cancel can arrive without one.

    Previously this fell into the in-process branch and killed a pid the
    coordinator never owned, leaving the worker running.
    """
    compute = _compute()

    tracker = _tracker(pid=4242)
    outcome, _ = await _cancel(tracker, compute, handle=None)

    assert outcome == "cancelled"
    compute.cancel_by_tag.assert_awaited_once_with("user-xyz", "proj-abc", "job-123")
    compute.cancel.assert_not_awaited()


async def test_stop_task_failure_falls_back_to_the_tag():
    """A stale handle should not prevent stopping a worker that is still tagged."""
    compute = _compute()
    compute.cancel = AsyncMock(side_effect=RuntimeError("no such task"))

    tracker = _tracker()
    outcome, _ = await _cancel(tracker, compute, handle=TASK_ARN)

    assert outcome == "cancelled"
    compute.cancel_by_tag.assert_awaited_once()


async def test_no_running_worker_is_a_successful_cancel():
    """Nothing to stop is not a failure — the job may simply have exited."""
    compute = _compute()
    compute.cancel_by_tag = AsyncMock(return_value=None)

    tracker = _tracker()
    outcome, _ = await _cancel(tracker, compute, handle=None)

    assert outcome == "cancelled"
    tracker.cancel_job.assert_called_once()


async def test_dev_still_marks_cancelled_before_killing_the_subprocess():
    """Dev ordering is load-bearing and deliberately unchanged.

    The harness reads its own job's status after the subprocess dies to decide
    whether to discard the results it produced, so marking cancelled after the
    kill would race that read. The compute is a local child process, not billable
    detached compute, so the cloud exposure does not apply.
    """
    order: list[str] = []
    compute = _compute()

    tracker = _tracker(pid=99)
    tracker.cancel_job = MagicMock(side_effect=lambda _j: order.append("terminal"))

    async def _kill(_pid):
        order.append("kill")

    with patch("app.services.harness_runner.kill_harness_subprocess", new=_kill):
        outcome, _ = await _cancel(
            tracker, compute, handle=None, arch_split=False
        )

    assert outcome == "cancelled"
    assert order == ["terminal", "kill"]
    compute.cancel_by_tag.assert_not_awaited()


async def test_an_accepted_stop_that_never_reaches_stopped_is_not_success():
    """StopTask returning is acceptance, not termination.

    ECS sends SIGTERM and allows a 30s stopTimeout before SIGKILL, so the task
    sits in STOPPING and keeps invoking models. Terminalizing on acceptance
    released the project's lock while the worker was still billing.
    """
    compute = _compute(stopped=False)  # never leaves STOPPING

    tracker = _tracker()
    outcome, _ = await _cancel(tracker, compute, handle=TASK_ARN)

    assert outcome == "stop_failed"
    tracker.cancel_job.assert_not_called()
    assert compute.status.await_count > 1, "it must actually poll, not ask once"


async def test_a_task_whose_containers_stopped_is_confirmed_before_teardown_ends():
    """DEPROVISIONING already means every container has exited.

    It is the network teardown between the containers stopping and the task
    reading STOPPED, and it takes tens of seconds on Fargate. Waiting through it
    ran out the confirmation window on every cancel of a worker that needed its
    SIGKILL, so the cancel reported failure for a worker that was already gone.
    """
    compute = _compute()
    compute.status = AsyncMock(
        side_effect=["RUNNING", "STOPPING", "DEPROVISIONING", "DEPROVISIONING"]
    )

    tracker = _tracker()
    outcome, _ = await _cancel(tracker, compute, handle=TASK_ARN)

    assert outcome == "cancelled"
    tracker.cancel_job.assert_called_once()
    assert compute.status.await_count == 3, "it must stop polling at DEPROVISIONING"


async def test_a_single_tag_miss_is_not_proof_there_is_no_worker():
    """list_tasks is eventually consistent, so one empty result proves nothing.

    A worker in PROVISIONING/PENDING takes 5-15s to appear, and the durable handle
    is written only after RunTask returns — so a cancel in that window sees no
    handle AND may see no tagged task. Concluding "nothing to stop" from one miss
    marked the job cancelled while the worker was starting up.
    """
    compute = _compute()
    compute.cancel_by_tag = AsyncMock(side_effect=[None, None, TASK_ARN])

    tracker = _tracker()
    outcome, _ = await _cancel(tracker, compute, handle=None)

    assert outcome == "cancelled"
    assert compute.cancel_by_tag.await_count == 3, "it must retry the lookup"
    tracker.cancel_job.assert_called_once()


async def test_a_tag_found_worker_must_also_reach_stopped():
    """The tag path carries the same contract as the direct-ARN path.

    It used to return success the moment StopTask was accepted while the ARN path
    went on to confirm — two guarantees behind one word. And this is the path a
    cancel in the launch window actually takes, because the durable handle is
    written only after RunTask returns, so the weaker guarantee applied to the
    majority of real cancels.
    """
    compute = _compute(stopped=False)  # never leaves STOPPING

    tracker = _tracker()
    outcome, _ = await _cancel(tracker, compute, handle=None)

    assert outcome == "stop_failed"
    tracker.cancel_job.assert_not_called()
    assert compute.status.await_count > 1, "the tag path must poll for STOPPED too"


async def test_a_task_that_cannot_be_found_is_not_confirmed_stopped():
    """DescribeTasks returning no task establishes nothing.

    ``status`` reports UNKNOWN for an empty response because that is what ordinary
    eventual consistency looks like moments after StopTask. Reading it as STOPPED
    confirmed the stop instantly and released the project's eval lock underneath a
    worker that was still running.
    """
    compute = _compute()
    compute.status = AsyncMock(return_value="UNKNOWN")

    tracker = _tracker()
    outcome, _ = await _cancel(tracker, compute, handle=TASK_ARN)

    assert outcome == "stop_failed"
    tracker.cancel_job.assert_not_called()


async def test_one_failed_status_read_does_not_decide_the_cancel():
    """A throttled DescribeTasks costs one observation, not the verdict.

    Bailing out on the first error let a transient ECS throttle report a failed
    stop for a task that did stop — and, worse, made a single read decisive in
    either direction.
    """
    compute = _compute()
    compute.status = AsyncMock(side_effect=[RuntimeError("throttled"), "STOPPED"])

    tracker = _tracker()
    outcome, _ = await _cancel(tracker, compute, handle=TASK_ARN)

    assert outcome == "cancelled"
    assert compute.status.await_count == 2, "it must keep polling past an error"


async def test_a_fargate_handle_is_never_treated_as_local():
    """Detachedness follows the compute, not the flag.

    A Fargate handle can outlive the flag that produced it, and treating that job
    as in-process would kill an unrelated pid and leave the worker running.
    """
    compute = _compute()

    tracker = _tracker(pid=4242)
    with patch("app.services.harness_runner.kill_harness_subprocess") as kill:
        outcome, _ = await _cancel(
            tracker, compute, handle=TASK_ARN, arch_split=False
        )

    assert outcome == "cancelled"
    compute.cancel.assert_awaited_once()
    kill.assert_not_called()
