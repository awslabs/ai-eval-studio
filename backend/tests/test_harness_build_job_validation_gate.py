# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The harness-build job's validation gate.

A harness build runs scaffold, then validate. When validation REJECTS the
harness the job must fail, because the alternative is a user running a full
eval — hundreds of judge calls — against a harness already known to be broken.

The gate read the wrong level of the response::

    content = validate_result.get("content", [])
    has_error = any(item.get("is_error") for item in content if isinstance(item, dict))

``mcp_error`` returns ``{"content": [{"type": "text", "text": ...}],
"is_error": True}`` — the flag sits beside ``content``, never inside its items
(``mcp_response.py``). So ``has_error`` was unconditionally False and every
rejected harness was reported as a COMPLETED build.

Both directions are pinned here. A gate that always fires is as broken as one
that never does, and only the never-fires half was the bug.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.jobs import Job
from app.services.jobs.harness_executor import run_harness_build_job
from app.services.mcp_response import mcp_error, mcp_success


@pytest.fixture
def job() -> Job:
    return Job(
        job_id="job-1234abcd",
        project_id="proj-5678",
        type="harness_build",
        config={"project_id": "proj-5678", "scaffold": False, "validate": True},
    )


async def _run(job: Job, validate_result: dict):
    """Run the executor with ``validate_harness`` stubbed to one response.

    Returns:
        Tuple of (JobResult, the tracker mock, the publish mock).
    """
    tracker = MagicMock()
    validate = MagicMock()
    validate.handler = AsyncMock(return_value=validate_result)

    with (
        patch("app.services.jobs.harness_executor.get_job_tracker", return_value=tracker),
        patch("app.services.jobs.harness_executor._publish_project_event") as publish,
        # The executor imports the tools inside the function, so the patch has
        # to land on the tool module's attribute rather than on a name the
        # executor holds.
        patch("app.services.mcp_tools.harness.validate_harness", validate),
    ):
        result = await run_harness_build_job(job)
    return result, tracker, publish


class TestValidationFailureFailsTheJob:
    @pytest.mark.asyncio
    async def test_top_level_is_error_fails_the_job(self, job):
        """The regression: a real ``mcp_error`` response used to pass the gate."""
        result, tracker, publish = await _run(
            job, mcp_error("Harness validation failed: metric import error")
        )

        assert result.status == "failed"
        assert result.error is not None
        assert "metric import error" in result.error.message

    @pytest.mark.asyncio
    async def test_the_failure_reaches_the_tracker_and_the_event_bus(self, job):
        """A failed job the UI never hears about is the same outage to a user."""
        _, tracker, publish = await _run(job, mcp_error("boom"))

        state_calls = [c for c in tracker.update_state.call_args_list]
        assert state_calls, "the tracker was never told the job's terminal state"
        assert state_calls[-1].args[1] == "failed"

        assert publish.called
        assert publish.call_args.args[1]["type"] == "failed"

    @pytest.mark.asyncio
    async def test_the_error_text_comes_from_the_validation_message(self, job):
        """Not a generic string — the operator needs the actual reason."""
        result, _, _ = await _run(job, mcp_error("uv sync failed: no such package"))
        assert "uv sync failed: no such package" in result.error.message


class TestValidationSuccessCompletesTheJob:
    @pytest.mark.asyncio
    async def test_a_success_response_completes(self, job):
        result, tracker, publish = await _run(job, mcp_success("All metrics OK"))

        assert result.status == "completed"
        assert result.error is None
        assert tracker.update_state.call_args_list[-1].args[1] == "completed"
        assert publish.call_args.args[1]["type"] == "completed"

    @pytest.mark.asyncio
    async def test_content_text_mentioning_failure_does_not_fail_the_job(self, job):
        """The gate must key on the flag, not on words in the text.

        A passing validation legitimately reports warnings that read like
        failures ("2 warning(s)", "could not be probed"). Matching text here
        would fail builds that validation deliberately let through.
        """
        result, _, _ = await _run(
            job,
            mcp_success(
                "metric self-test: 2 metric(s) probed, 0 blocking, 2 warning(s); "
                "Answer Semantic Equivalence: could not be probed"
            ),
        )
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_a_falsy_is_error_completes(self, job):
        result, _, _ = await _run(
            job, {"content": [{"type": "text", "text": "ok"}], "is_error": False}
        )
        assert result.status == "completed"
