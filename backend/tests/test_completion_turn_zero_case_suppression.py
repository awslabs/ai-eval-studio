# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for zero-case completion suppression.

Proves the degenerate-result guard in request_completion_turn suppresses
completed jobs with zero scored cases at the publisher (before enqueuing),
while allowing failures with zero cases to fire normally.

Assertions:
1. optimization_iteration, completed, total_cases==0 → NO turn enqueued, suppression logged
2. optimization_iteration, completed, total_cases>0 → turn enqueued
3. failed with zero cases → turn IS enqueued (guard must not swallow failures)
4. Manual eval result-not-ready (neutral seed) → turn enqueued
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.services.jobs import Job

pytestmark = pytest.mark.asyncio


async def test_optimization_iteration_zero_cases_suppressed():
    """optimization_iteration, completed, total_cases==0 → NO turn enqueued."""
    from app.services import completion_turn

    job = Job(
        job_id="job-zero",
        project_id="proj-A",
        type="optimization_iteration",
        state="completed",
        config={"source": "optimization", "iteration": 1},
        result={"total_cases": 0, "passed_cases": 0, "iteration": 1},
    )

    # Clear any prior state
    completion_turn._pending_completions.clear()

    with patch.object(completion_turn, "logger") as mock_logger:
        await completion_turn.request_completion_turn("proj-A", job)

    # Assert: NO turn was enqueued (_pending_completions never populated)
    assert "proj-A" not in completion_turn._pending_completions

    # Assert: suppression was logged
    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args[0]
    assert "Suppressed degenerate zero-case completion" in call_args[0]
    assert "job-zero" in str(call_args)
    assert "optimization_iteration" in str(call_args)
    assert "completed" in str(call_args)


async def test_optimization_iteration_nonzero_cases_enqueued():
    """optimization_iteration, completed, total_cases>0 → turn enqueued."""
    from app.services import completion_turn

    job = Job(
        job_id="job-nonzero",
        project_id="proj-B",
        type="optimization_iteration",
        state="completed",
        config={"source": "optimization", "iteration": 1},
        result={"total_cases": 5, "passed_cases": 3, "iteration": 1},
    )

    # Clear any prior state
    completion_turn._pending_completions.clear()

    with patch.object(completion_turn, "logger") as mock_logger, \
         patch.object(completion_turn, "_build_synthetic_message", new_callable=AsyncMock, return_value="iteration complete"), \
         patch("asyncio.create_task") as mock_create_task:
        await completion_turn.request_completion_turn("proj-B", job)

    # Assert: turn WAS enqueued (_pending_completions populated, task created)
    assert "proj-B" in completion_turn._pending_completions
    mock_create_task.assert_called_once()

    # Assert: NO suppression log
    for call in mock_logger.info.call_args_list:
        assert "Suppressed degenerate" not in str(call)

    # Cleanup
    completion_turn._pending_completions.clear()


async def test_failed_job_zero_cases_still_enqueued():
    """failed with zero cases → turn IS enqueued (must not swallow failures)."""
    from app.services import completion_turn

    job = Job(
        job_id="job-failed",
        project_id="proj-C",
        type="optimization_iteration",
        state="failed",
        config={"source": "optimization", "iteration": 1},
        result={"total_cases": 0, "passed_cases": 0, "iteration": 1},
    )

    # Clear any prior state
    completion_turn._pending_completions.clear()

    with patch.object(completion_turn, "logger") as mock_logger, \
         patch.object(completion_turn, "_build_synthetic_message", new_callable=AsyncMock, return_value="iteration failed"), \
         patch("asyncio.create_task") as mock_create_task:
        await completion_turn.request_completion_turn("proj-C", job)

    # Assert: turn WAS enqueued (failures must fire even with zero cases)
    assert "proj-C" in completion_turn._pending_completions
    mock_create_task.assert_called_once()

    # Assert: NO suppression log
    for call in mock_logger.info.call_args_list:
        assert "Suppressed degenerate" not in str(call)

    # Cleanup
    completion_turn._pending_completions.clear()


async def test_manual_eval_not_ready_neutral_seed_enqueued():
    """Manual eval result-not-ready → turn enqueued with neutral seed.

    Design decision: the neutral seed is the user's only signal that their
    single eval finished. Suppressing it would leave them with silence, which
    is worse than a vague message. So eval-type jobs with total==0 return False
    from _is_degenerate_result, preserving the neutral-seed behavior.
    """
    from app.services import completion_turn

    job = Job(
        job_id="job-eval-not-ready",
        project_id="proj-D",
        type="eval",
        state="completed",
        config={"source": "manual", "prompt_version": "baseline"},
        result=[],  # empty list → total==0 in _build_synthetic_message
        output_file="output/baseline.json",
    )

    # Clear any prior state
    completion_turn._pending_completions.clear()

    with patch.object(completion_turn, "logger") as mock_logger, \
         patch.object(completion_turn, "_build_synthetic_message", new_callable=AsyncMock, return_value="Eval complete. Results finalizing."), \
         patch("asyncio.create_task") as mock_create_task:
        await completion_turn.request_completion_turn("proj-D", job)

    # Assert: turn WAS enqueued (neutral seed preserved for manual eval)
    assert "proj-D" in completion_turn._pending_completions
    mock_create_task.assert_called_once()

    # Assert: NO suppression log
    for call in mock_logger.info.call_args_list:
        assert "Suppressed degenerate" not in str(call)

    # Cleanup
    completion_turn._pending_completions.clear()
