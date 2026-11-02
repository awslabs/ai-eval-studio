# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for ``optimization_executor._wait_for_eval`` hard ceiling.

Defense in depth: the harness has no single subprocess timeout, so a
stuck eval could otherwise pin the optimization loop forever in a
``while True: await asyncio.sleep(10)``. The ceiling fails the iteration
with a structured error after a configurable wall-clock budget.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from app.services.jobs.optimization_executor import (
    _DEFAULT_WAIT_FOR_EVAL_MAX_S,
    _wait_for_eval,
    _wait_for_eval_max_s,
)
from app.services.optimization_state_repository import OptimizationStateHandle
from app.services.resource_error import Codes, ResourceError


async def _empty_handle_read(project_id, project_dir, **kwargs):
    """Stand in for a repository read that finds no optimization state.

    Returns a real handle with ``state=None`` rather than mocking the store, so the
    executor exercises its genuine "no state" branch.
    """
    return OptimizationStateHandle(
        state=None,
        version=None,
        arch_split=kwargs.get("arch_split", False),
        owner_id=kwargs.get("owner_id"),
        project_id=project_id,
        project_dir=project_dir,
    )


@dataclass
class _FakeJob:
    """Minimal Job stand-in for the wait_for_eval poll-loop tests."""

    job_id: str
    project_id: str = "proj-1"
    state: str = "running"
    error: ResourceError | None = None
    # Mirror the real Job.config (default_factory=dict). _wait_for_eval reads
    # parent_job.config for arch_split + user_id; an empty config keeps these
    # ceiling tests on the old (EFS) path — status quo for this file.
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return self.state


@dataclass
class _FakeTracker:
    """In-memory tracker that records calls; never schedules real work."""

    jobs: dict[str, _FakeJob] = field(default_factory=dict)
    cancel_calls: list[str] = field(default_factory=list)
    state_updates: list[tuple[str, str, ResourceError | None]] = field(default_factory=list)

    def get_job(self, job_id: str) -> _FakeJob | None:
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> None:
        self.cancel_calls.append(job_id)

    def update_state(
        self,
        job_id: str,
        state: str,
        *,
        error: ResourceError | None = None,
    ) -> None:
        self.state_updates.append((job_id, state, error))
        if job_id in self.jobs:
            self.jobs[job_id].state = state
            if error is not None:
                self.jobs[job_id].error = error


def test_wait_for_eval_max_s_default():
    """Resolver returns the 4-hour default when the env var is unset."""
    assert _wait_for_eval_max_s() == _DEFAULT_WAIT_FOR_EVAL_MAX_S


def test_wait_for_eval_max_s_env_override(monkeypatch: pytest.MonkeyPatch):
    """A positive integer in ``OPTIMIZATION_WAIT_FOR_EVAL_MAX_S`` wins."""
    monkeypatch.setenv("OPTIMIZATION_WAIT_FOR_EVAL_MAX_S", "120")
    assert _wait_for_eval_max_s() == 120


def test_wait_for_eval_max_s_invalid_falls_back(monkeypatch: pytest.MonkeyPatch):
    """A non-integer or non-positive value falls back to the default."""
    monkeypatch.setenv("OPTIMIZATION_WAIT_FOR_EVAL_MAX_S", "not-an-int")
    assert _wait_for_eval_max_s() == _DEFAULT_WAIT_FOR_EVAL_MAX_S
    monkeypatch.setenv("OPTIMIZATION_WAIT_FOR_EVAL_MAX_S", "0")
    assert _wait_for_eval_max_s() == _DEFAULT_WAIT_FOR_EVAL_MAX_S


@pytest.mark.asyncio
async def test_wait_for_eval_ceiling_fires(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """When the eval never reaches a terminal state within the ceiling,
    the helper cancels the eval, marks the parent failed with a
    structured error, and returns ``None``."""
    # Tight ceiling so the test runs in ~one fake "tick". The poll
    # interval inside the helper is 10s, so a 10s ceiling overflows
    # after exactly one iteration.
    monkeypatch.setenv("OPTIMIZATION_WAIT_FOR_EVAL_MAX_S", "10")

    # Fast-forward asyncio.sleep so the test doesn't actually sleep 10s.
    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(
        "app.services.jobs.optimization_executor.asyncio.sleep", _no_sleep
    )

    # The repository read returns an empty handle (no state) — keeps the
    # cancellation branch dormant so the ceiling branch can fire. Patched at the
    # repository, which now owns the read; the executor no longer touches
    # OptimizationState directly.
    monkeypatch.setattr(
        "app.services.optimization_state_repository.read", _empty_handle_read
    )

    eval_job = _FakeJob(job_id="eval-1")
    parent_job = _FakeJob(job_id="parent-1")
    tracker = _FakeTracker(jobs={"eval-1": eval_job, "parent-1": parent_job})

    result = await _wait_for_eval(
        eval_job_id="eval-1",
        project_dir=tmp_path,
        project_id="proj-1",
        parent_job=parent_job,  # type: ignore[arg-type] — duck-typed Job
        job_tracker=tracker,  # type: ignore[arg-type] — duck-typed JobTracker
    )

    assert result is None
    # The eval job was cancelled and the parent failed with the wait-exceeded code.
    assert tracker.cancel_calls == ["eval-1"]
    parent_updates = [
        (jid, st, err) for (jid, st, err) in tracker.state_updates
        if jid == "parent-1" and st == "failed"
    ]
    assert len(parent_updates) == 1, parent_updates
    error = parent_updates[0][2]
    assert isinstance(error, ResourceError)
    assert error.code == Codes.OPTIMIZATION_EVAL_WAIT_EXCEEDED
    assert error.details["max_wait_s"] == 10


@pytest.mark.asyncio
async def test_wait_for_eval_returns_when_eval_completes_before_ceiling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """Sanity: a normally-completing eval returns its result before the
    ceiling and never trips the ceiling path."""
    monkeypatch.setenv("OPTIMIZATION_WAIT_FOR_EVAL_MAX_S", "120")

    sleep_count = {"n": 0}

    async def _no_sleep(_seconds: float) -> None:
        sleep_count["n"] += 1
        return None

    monkeypatch.setattr(
        "app.services.jobs.optimization_executor.asyncio.sleep", _no_sleep
    )
    monkeypatch.setattr(
        "app.services.optimization_state_repository.read", _empty_handle_read
    )

    eval_job = _FakeJob(job_id="eval-2")
    parent_job = _FakeJob(job_id="parent-2")
    tracker = _FakeTracker(jobs={"eval-2": eval_job, "parent-2": parent_job})

    # Patch extract_iteration_metrics so the post-loop helper doesn't
    # blow up on our empty fake results list.
    monkeypatch.setattr(
        "app.services.jobs.iteration_metrics.extract_iteration_metrics",
        lambda _results: {"passed_cases": 0, "total_cases": 0, "metrics_summary": {}},
    )

    # Mark the eval completed before the first poll completes.
    eval_job.state = "completed"

    # Give the harness-runner-style result so the post-loop branch can
    # compute iteration metrics.
    eval_job.__dict__["result"] = []  # type: ignore[attr-defined]
    eval_job.__dict__["output_file"] = "eval_results_v1_test.json"  # type: ignore[attr-defined]

    result = await _wait_for_eval(
        eval_job_id="eval-2",
        project_dir=tmp_path,
        project_id="proj-1",
        parent_job=parent_job,  # type: ignore[arg-type]
        job_tracker=tracker,  # type: ignore[arg-type]
    )

    # Helper produced a metrics dict, no ceiling-fired error appears.
    assert result is not None
    assert tracker.cancel_calls == []
    # No ``failed`` update for the parent.
    assert not any(
        st == "failed" for (jid, st, _err) in tracker.state_updates
        if jid == "parent-2"
    )
