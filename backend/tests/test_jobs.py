# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``services/jobs/__init__.py`` — the JobTracker singleton.

Covers the eval-job lifecycle (create / cancel / complete / fail /
evict / per-project query) end-to-end. Higher-level surface (queue
priorities, completion-turn dispatch, harness-build cancellation) is
covered by ``test_job_manager_behavior.py``.
"""

import pytest

from app.services.jobs import Job, JobTracker, get_job_tracker
from app.services.resource_error import Codes, ResourceError


@pytest.fixture(autouse=True)
def reset_tracker():
    """Reset the singleton tracker before each test to ensure isolation."""
    tracker = get_job_tracker()
    tracker._reset()
    yield
    tracker._reset()


class TestSingleton:
    """Tests for singleton behavior."""

    def test_singleton_returns_same_instance(self):
        """get_job_tracker always returns the same object."""
        tracker_a = get_job_tracker()
        tracker_b = get_job_tracker()
        assert tracker_a is tracker_b

    def test_constructor_returns_same_instance(self):
        """Direct JobTracker() calls also return the singleton."""
        tracker_a = JobTracker()
        tracker_b = JobTracker()
        assert tracker_a is tracker_b


class TestCreateJob:
    """Tests for job creation."""

    def test_create_job_returns_eval_job(self):
        """create_job returns a properly initialized Job."""
        tracker = get_job_tracker()
        job = tracker.create_job("proj-1", "eval", config={"model_ids": ["model-a", "model-b"], "prompt_version": "v1"})

        assert isinstance(job, Job)
        assert job.project_id == "proj-1"
        assert job.status == "pending"
        assert job.progress["models_total"] == 2
        assert job.progress["current_model"] is None
        assert job.result is None
        assert job.output_file is None
        assert job.error is None
        assert job.created_at != ""
        assert job.completed_at is None

    def test_create_job_unique_ids(self):
        """Each created job gets a unique job_id."""
        tracker = get_job_tracker()
        job_a = tracker.create_job("proj-1", "eval", config={"model_ids": ["m1"], "prompt_version": "v1"})
        job_b = tracker.create_job("proj-1", "eval", config={"model_ids": ["m1"], "prompt_version": "v1"})
        assert job_a.job_id != job_b.job_id

    def test_create_job_progress_message_includes_details(self):
        """Progress message on creation mentions model count and prompt version."""
        tracker = get_job_tracker()
        job = tracker.create_job("proj-1", "eval", config={"model_ids": ["a", "b", "c"], "prompt_version": "baseline"})
        assert "3 model(s)" in job.progress["message"]
        assert "baseline" in job.progress["message"]


class TestUpdateProgress:
    """Tests for progress updates."""

    def test_update_progress_merges_fields(self):
        """update_progress merges kwargs into job.progress."""
        tracker = get_job_tracker()
        job = tracker.create_job("proj-1", "eval", config={"model_ids": ["m1"], "prompt_version": "v1"})

        tracker.update_progress(
            job.job_id,
            current_model="m1",
            current_case=5,
            cases_total=100,
            message="Running model m1",
        )

        assert job.progress["current_model"] == "m1"
        assert job.progress["current_case"] == 5
        assert job.progress["cases_total"] == 100
        assert job.progress["message"] == "Running model m1"

    def test_update_progress_transitions_pending_to_running(self):
        """First progress update moves status from pending to running."""
        tracker = get_job_tracker()
        job = tracker.create_job("proj-1", "eval", config={"model_ids": ["m1"], "prompt_version": "v1"})
        assert job.status == "pending"

        tracker.update_progress(job.job_id, message="Starting")
        assert job.status == "running"

    def test_update_progress_keeps_running_status(self):
        """Subsequent progress updates don't change running status."""
        tracker = get_job_tracker()
        job = tracker.create_job("proj-1", "eval", config={"model_ids": ["m1"], "prompt_version": "v1"})

        tracker.update_progress(job.job_id, message="First")
        tracker.update_progress(job.job_id, message="Second")
        assert job.status == "running"

    def test_update_progress_unknown_job_raises_key_error(self):
        """Updating a nonexistent job raises KeyError."""
        tracker = get_job_tracker()
        with pytest.raises(KeyError):
            tracker.update_progress("nonexistent-id", message="oops")


class TestCompleteJob:
    """Tests for job completion."""

    def test_complete_job_sets_fields(self):
        """complete_job sets status, result, output_file, and completed_at."""
        tracker = get_job_tracker()
        job = tracker.create_job("proj-1", "eval", config={"model_ids": ["m1"], "prompt_version": "v1"})
        result_data = [{"model": "m1", "score": 0.95}]

        tracker.complete_job(job.job_id, result=result_data, output_file="results.json")

        assert job.status == "completed"
        assert job.result == result_data
        assert job.output_file == "results.json"
        assert job.completed_at is not None
        assert "Completed" in job.progress["message"]

    def test_complete_job_unknown_raises_key_error(self):
        """Completing a nonexistent job raises KeyError."""
        tracker = get_job_tracker()
        with pytest.raises(KeyError):
            tracker.complete_job("bad-id", result=[], output_file="x.json")


class TestFailJob:
    """Tests for job failure."""

    def test_fail_job_sets_fields(self):
        """fail_job sets status, error, and completed_at."""
        tracker = get_job_tracker()
        job = tracker.create_job("proj-1", "eval", config={"model_ids": ["m1"], "prompt_version": "v1"})

        err = ResourceError(code=Codes.HARNESS_SUBPROCESS_FAILED, message="Model timeout")
        tracker.fail_job(job.job_id, err)

        assert job.status == "failed"
        assert job.error is err
        assert job.error.message == "Model timeout"
        assert job.error.code == Codes.HARNESS_SUBPROCESS_FAILED
        assert job.completed_at is not None
        assert "Failed" in job.progress["message"]
        assert "Model timeout" in job.progress["message"]

    def test_fail_job_unknown_raises_key_error(self):
        """Failing a nonexistent job raises KeyError."""
        tracker = get_job_tracker()
        err = ResourceError(code=Codes.HARNESS_SUBPROCESS_FAILED, message="oops")
        with pytest.raises(KeyError):
            tracker.fail_job("bad-id", err)


class TestGetJob:
    """Tests for get_job lookup."""

    def test_get_existing_job(self):
        """get_job returns the correct Job by ID."""
        tracker = get_job_tracker()
        job = tracker.create_job("proj-1", "eval", config={"model_ids": ["m1"], "prompt_version": "v1"})
        assert tracker.get_job(job.job_id) is job

    def test_get_nonexistent_job_returns_none(self):
        """get_job returns None for unknown IDs."""
        tracker = get_job_tracker()
        assert tracker.get_job("nonexistent") is None


class TestGetActiveJob:
    """Tests for get_active_job query."""

    def test_returns_pending_job(self):
        """get_active_job returns a pending job."""
        tracker = get_job_tracker()
        job = tracker.create_job("proj-1", "eval", config={"model_ids": ["m1"], "prompt_version": "v1"})
        assert tracker.get_active_job("proj-1") is job

    def test_returns_running_job(self):
        """get_active_job returns a running job."""
        tracker = get_job_tracker()
        job = tracker.create_job("proj-1", "eval", config={"model_ids": ["m1"], "prompt_version": "v1"})
        tracker.update_progress(job.job_id, message="Running")
        assert tracker.get_active_job("proj-1") is job

    def test_ignores_completed_jobs(self):
        """get_active_job does not return completed jobs."""
        tracker = get_job_tracker()
        job = tracker.create_job("proj-1", "eval", config={"model_ids": ["m1"], "prompt_version": "v1"})
        tracker.complete_job(job.job_id, result=[], output_file="r.json")
        assert tracker.get_active_job("proj-1") is None

    def test_ignores_failed_jobs(self):
        """get_active_job does not return failed jobs."""
        tracker = get_job_tracker()
        job = tracker.create_job("proj-1", "eval", config={"model_ids": ["m1"], "prompt_version": "v1"})
        tracker.fail_job(job.job_id, ResourceError(
            code=Codes.HARNESS_SUBPROCESS_FAILED, message="boom",
        ))
        assert tracker.get_active_job("proj-1") is None

    def test_returns_latest_active(self):
        """When multiple active jobs exist, returns the most recently created."""
        tracker = get_job_tracker()
        job_old = tracker.create_job("proj-1", "eval", config={"model_ids": ["m1"], "prompt_version": "v1"})
        job_new = tracker.create_job("proj-1", "eval", config={"model_ids": ["m2"], "prompt_version": "v2"})
        active = tracker.get_active_job("proj-1")
        # Both are pending; should return the newer one
        assert active is job_new

    def test_returns_none_for_unknown_project(self):
        """get_active_job returns None for a project with no jobs."""
        tracker = get_job_tracker()
        assert tracker.get_active_job("no-such-project") is None

    def test_scoped_to_project(self):
        """get_active_job only considers jobs for the specified project."""
        tracker = get_job_tracker()
        tracker.create_job("proj-1", "eval", config={"model_ids": ["m1"], "prompt_version": "v1"})
        job_b = tracker.create_job("proj-2", "eval", config={"model_ids": ["m2"], "prompt_version": "v1"})
        assert tracker.get_active_job("proj-2") is job_b


class TestListJobs:
    """Tests for list_jobs query."""

    def test_lists_all_project_jobs(self):
        """list_jobs returns all jobs for a project regardless of status."""
        tracker = get_job_tracker()
        job_a = tracker.create_job("proj-1", "eval", config={"model_ids": ["m1"], "prompt_version": "v1"})
        job_b = tracker.create_job("proj-1", "eval", config={"model_ids": ["m2"], "prompt_version": "v2"})
        tracker.complete_job(job_a.job_id, result=[], output_file="r.json")

        jobs = tracker.list_jobs("proj-1")
        assert len(jobs) == 2
        job_ids = {j.job_id for j in jobs}
        assert job_a.job_id in job_ids
        assert job_b.job_id in job_ids

    def test_newest_first_ordering(self):
        """list_jobs returns jobs newest-first by created_at."""
        tracker = get_job_tracker()
        job_a = tracker.create_job("proj-1", "eval", config={"model_ids": ["m1"], "prompt_version": "v1"})
        job_b = tracker.create_job("proj-1", "eval", config={"model_ids": ["m2"], "prompt_version": "v2"})

        jobs = tracker.list_jobs("proj-1")
        # job_b was created after job_a
        assert jobs[0].job_id == job_b.job_id
        assert jobs[1].job_id == job_a.job_id

    def test_scoped_to_project(self):
        """list_jobs only returns jobs for the specified project."""
        tracker = get_job_tracker()
        tracker.create_job("proj-1", "eval", config={"model_ids": ["m1"], "prompt_version": "v1"})
        tracker.create_job("proj-2", "eval", config={"model_ids": ["m2"], "prompt_version": "v1"})

        jobs = tracker.list_jobs("proj-1")
        assert len(jobs) == 1
        assert jobs[0].project_id == "proj-1"

    def test_empty_for_unknown_project(self):
        """list_jobs returns empty list for a project with no jobs."""
        tracker = get_job_tracker()
        assert tracker.list_jobs("unknown") == []


class TestFullLifecycle:
    """Integration test covering a complete job lifecycle."""

    def test_pending_to_running_to_completed(self):
        """Job transitions through pending -> running -> completed."""
        tracker = get_job_tracker()
        job = tracker.create_job("proj-1", "eval", config={"model_ids": ["m1", "m2"], "prompt_version": "v1"})
        assert job.status == "pending"

        # Start running first model
        tracker.update_progress(
            job.job_id,
            current_model="m1",
            current_case=0,
            cases_total=50,
            message="Evaluating m1",
        )
        assert job.status == "running"

        # Progress through cases
        tracker.update_progress(job.job_id, current_case=25)
        tracker.update_progress(job.job_id, current_case=50, models_completed=1)

        # Switch to second model
        tracker.update_progress(
            job.job_id,
            current_model="m2",
            current_case=0,
            message="Evaluating m2",
        )

        # Complete
        tracker.complete_job(
            job.job_id,
            result=[{"model": "m1"}, {"model": "m2"}],
            output_file="eval_results_20260404.json",
        )
        assert job.status == "completed"
        assert len(job.result) == 2
        assert tracker.get_active_job("proj-1") is None

    def test_pending_to_running_to_failed(self):
        """Job transitions through pending -> running -> failed."""
        tracker = get_job_tracker()
        job = tracker.create_job("proj-1", "eval", config={"model_ids": ["m1"], "prompt_version": "v1"})

        tracker.update_progress(job.job_id, current_model="m1", message="Running")
        assert job.status == "running"

        tracker.fail_job(job.job_id, ResourceError(
            code=Codes.BEDROCK_THROTTLED, message="Bedrock throttling",
        ))
        assert job.status == "failed"
        assert job.error is not None
        assert job.error.message == "Bedrock throttling"
        assert tracker.get_active_job("proj-1") is None


class TestEvictOldJobs:
    """Tests for _evict_old_jobs capacity management."""

    def test_evicts_oldest_completed_jobs_over_capacity(self):
        """Old completed jobs are evicted when exceeding max_terminal."""
        tracker = get_job_tracker()

        # Create 5 completed jobs
        job_ids = []
        for i in range(5):
            job = tracker.create_job("proj-evict", "eval", config={"model_ids": [f"m{i}"], "prompt_version": "v1"})
            tracker.complete_job(job.job_id, result=[], output_file=f"r{i}.json")
            job_ids.append(job.job_id)

        # All 5 should be present before eviction
        assert all(tracker.get_job(jid) is not None for jid in job_ids)

        # Evict with max_terminal=3, should remove the 2 oldest
        tracker._evict_old_jobs(max_terminal=3)

        # Oldest 2 should be gone
        assert tracker.get_job(job_ids[0]) is None
        assert tracker.get_job(job_ids[1]) is None
        # Newest 3 should remain
        assert tracker.get_job(job_ids[2]) is not None
        assert tracker.get_job(job_ids[3]) is not None
        assert tracker.get_job(job_ids[4]) is not None

    def test_keeps_running_jobs_during_eviction(self):
        """Running/pending jobs are never evicted regardless of capacity."""
        tracker = get_job_tracker()

        # Create a running job
        running_job = tracker.create_job("proj-evict", "eval", config={"model_ids": ["m-run"], "prompt_version": "v1"})
        tracker.update_progress(running_job.job_id, message="Running")

        # Create 3 completed jobs
        for i in range(3):
            job = tracker.create_job("proj-evict", "eval", config={"model_ids": [f"m{i}"], "prompt_version": "v1"})
            tracker.complete_job(job.job_id, result=[], output_file=f"r{i}.json")

        # Evict with very low capacity
        tracker._evict_old_jobs(max_terminal=1)

        # Running job must still be present
        assert tracker.get_job(running_job.job_id) is not None
        assert tracker.get_job(running_job.job_id).status == "running"

    def test_no_eviction_under_capacity(self):
        """No jobs evicted when terminal count is within capacity."""
        tracker = get_job_tracker()

        job = tracker.create_job("proj-evict", "eval", config={"model_ids": ["m1"], "prompt_version": "v1"})
        tracker.complete_job(job.job_id, result=[], output_file="r.json")

        tracker._evict_old_jobs(max_terminal=50)
        assert tracker.get_job(job.job_id) is not None


class TestAppendEventStoresEnvelope:
    """``Job.append_event`` stores envelopes (with ``seq``), not raw events.

    ``ws_hub.replay_to_connection`` filters the buffer on
    ``isinstance(env.get("seq"), int)``. Storing raw event dicts (no
    seq) made every replay return zero events — a SPA subscribing
    mid-eval never received the buffered ``scoring_start`` envelope
    that sets the denominator, and the bar froze with a zero
    denominator (``"model 0 of N"``). This test pins the contract.
    """

    @pytest.mark.asyncio
    async def test_buffered_entries_carry_envelope_seq(self):
        from unittest.mock import AsyncMock, patch

        tracker = get_job_tracker()
        job = tracker.create_job(
            "proj-replay", "eval",
            config={"model_ids": ["m1"], "prompt_version": "v1"},
        )

        # Patch the publish so we don't need a live ws_hub / project.
        with patch("app.services.jobs._publish_project_event"), patch(
            "app.main.get_project_manager",
        ) as mock_pm:
            mock_pm.return_value.get_project.return_value = None  # owner_id resolves to None
            await job.append_event({"event": "scoring_start", "model": "m1", "total": 210})
            await job.append_event({"event": "generating", "model": "m1", "current": 5, "total": 210})

        assert len(job.events) == 2
        # Every buffered entry MUST carry a seq so replay_to_connection
        # picks it up; without one the bar freezes at "model 0 of N".
        for env in job.events:
            assert isinstance(env.get("seq"), int), f"missing seq on {env}"
        # Envelope shape preserved — payload sits under .data so live
        # and replay paths deliver the same shape to the SPA.
        assert job.events[0]["kind"] == "progress"
        assert job.events[0]["data"]["event"] == "scoring_start"
        assert job.events[0]["data"]["total"] == 210


class TestTerminalTransitionPublishesDone:
    """Every terminal transition MUST publish a per-job ``done`` envelope.

    The SPA's ``useEvalJobStream`` flips out of "running" only when it
    receives a ``done`` envelope on the ``job:<id>`` stream. Before
    consolidating publish at ``update_state``, only ``complete_job`` /
    ``fail_job`` published — every cancel path (``cancel_job``,
    ``hard_cancel_project_eval``, the optimization-executor's direct
    ``update_state(..., "cancelled")`` calls) went silent and the eval
    bar wedged at "model 0 of N" with a live Stop button forever, until
    the coordinator restarted.

    These tests pin: (1) the chokepoint publishes for completed,
    failed, and cancelled; (2) it doesn't double-publish when callers
    redundantly transition an already-terminal job; (3) the eval-only
    project-event scope doesn't leak to other job types.
    """

    def test_update_state_to_cancelled_publishes_done(self):
        """Cancelling via update_state emits a per-job done envelope.

        The fix for the stuck Stop button — before, the cancel paths
        only set state and never told the SPA the job ended.
        """
        from unittest.mock import patch

        tracker = get_job_tracker()
        job = tracker.create_job(
            "proj-cancel", "eval",
            config={"model_ids": ["m1"], "prompt_version": "v1"},
        )

        with patch("app.services.jobs._publish_per_job_done") as mock_done, patch(
            "app.services.jobs._publish_project_event",
        ):
            tracker.update_state(job.job_id, "cancelled")

        mock_done.assert_called_once()
        kwargs = mock_done.call_args.kwargs
        assert kwargs["status"] == "cancelled"
        assert kwargs["output_file"] is None
        assert kwargs["error"] is None

    def test_cancel_job_publishes_done_once(self):
        """cancel_job goes through update_state → exactly one done publish.

        Guards against both the "no publish" regression (the original
        bug) and the "double publish" regression that would happen if
        cancel_job ever grew its own publish call alongside the
        chokepoint.
        """
        from unittest.mock import patch

        tracker = get_job_tracker()
        job = tracker.create_job(
            "proj-cancel-once", "eval",
            config={"model_ids": ["m1"], "prompt_version": "v1"},
        )

        with patch("app.services.jobs._publish_per_job_done") as mock_done, patch(
            "app.services.jobs._publish_project_event",
        ):
            tracker.cancel_job(job.job_id)

        assert mock_done.call_count == 1
        assert mock_done.call_args.kwargs["status"] == "cancelled"

    def test_complete_job_publishes_done_exactly_once(self):
        """complete_job routes through update_state without double-firing.

        complete_job used to publish directly AND via update_state once
        the chokepoint moved into update_state — this test guards against
        a regression where both paths run.
        """
        from unittest.mock import patch

        tracker = get_job_tracker()
        job = tracker.create_job(
            "proj-complete", "eval",
            config={"model_ids": ["m1"], "prompt_version": "v1"},
        )

        with patch("app.services.jobs._publish_per_job_done") as mock_done, patch(
            "app.services.jobs._publish_project_event",
        ):
            tracker.complete_job(job.job_id, result=[{"m": "m1"}], output_file="r.json")

        assert mock_done.call_count == 1
        kwargs = mock_done.call_args.kwargs
        assert kwargs["status"] == "completed"
        assert kwargs["output_file"] == "r.json"

    def test_fail_job_publishes_done_exactly_once(self):
        """fail_job routes through update_state without double-firing."""
        from unittest.mock import patch

        tracker = get_job_tracker()
        job = tracker.create_job(
            "proj-fail", "eval",
            config={"model_ids": ["m1"], "prompt_version": "v1"},
        )
        err = ResourceError(code=Codes.HARNESS_SUBPROCESS_FAILED, message="boom")

        with patch("app.services.jobs._publish_per_job_done") as mock_done, patch(
            "app.services.jobs._publish_project_event",
        ):
            tracker.fail_job(job.job_id, err)

        assert mock_done.call_count == 1
        kwargs = mock_done.call_args.kwargs
        assert kwargs["status"] == "failed"
        assert kwargs["error"] is err

    def test_redundant_terminal_transition_does_not_double_publish(self):
        """Transitioning an already-terminal job is a no-op publish-wise.

        ``optimization_executor`` cancel paths sometimes route through
        both ``cancel_job`` and a follow-up ``update_state(..., "cancelled")``
        on the same id. Without the prior-state guard, that pair would
        emit two done envelopes per cancel, making the SPA fire its
        completion hook twice.
        """
        from unittest.mock import patch

        tracker = get_job_tracker()
        job = tracker.create_job(
            "proj-double", "eval",
            config={"model_ids": ["m1"], "prompt_version": "v1"},
        )

        with patch("app.services.jobs._publish_per_job_done") as mock_done, patch(
            "app.services.jobs._publish_project_event",
        ):
            tracker.update_state(job.job_id, "cancelled")
            tracker.update_state(job.job_id, "cancelled")

        assert mock_done.call_count == 1

    def test_eval_job_terminal_publishes_project_event(self):
        """Eval jobs publish the dashboard-refresh project event.

        Cancellations need to clear stale "running" cards in the
        EvalDashboard. The project event is what
        ``useProjectEvent({resource: "eval_job"})`` listens on.
        """
        from unittest.mock import patch

        tracker = get_job_tracker()
        job = tracker.create_job(
            "proj-evt", "eval",
            config={"model_ids": ["m1"], "prompt_version": "v1"},
        )

        with patch("app.services.jobs._publish_per_job_done"), patch(
            "app.services.jobs._publish_project_event",
        ) as mock_proj:
            tracker.update_state(job.job_id, "cancelled")

        mock_proj.assert_called_once()
        project_id, event = mock_proj.call_args.args
        assert project_id == "proj-evt"
        assert event["resource"] == "eval_job"
        assert event["type"] == "cancelled"
        assert event["job_id"] == job.job_id

    def test_non_eval_job_terminal_skips_eval_job_event(self):
        """Non-eval jobs (harness_build, optimization_iteration) don't
        leak ``eval_job`` project events — those listeners only know
        about real eval runs and would refetch garbage otherwise.
        """
        from unittest.mock import patch

        tracker = get_job_tracker()
        job = tracker.create_job(
            "proj-non-eval", "optimization_iteration",
            config={"iteration": 1, "prompt_version": "v1"},
        )

        with patch("app.services.jobs._publish_per_job_done"), patch(
            "app.services.jobs._publish_project_event",
        ) as mock_proj:
            tracker.update_state(job.job_id, "cancelled")

        mock_proj.assert_not_called()
