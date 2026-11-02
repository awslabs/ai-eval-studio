# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for ``_wait_for_eval`` detached-path terminal precedence.

On the detached (``arch_split=True``) path ``_wait_for_eval`` derives terminal
from TWO durable sources that a successful eval sets **seconds apart**:

1. the ``results`` event in the durable event log (the harness's own success
   signal, written by ``save_results``), and
2. ``state="completed"`` on the durable job record (mirrored by the coordinator
   and the worker).

A 10s poll tick that lands after BOTH have landed observes both signals in the
same iteration. The record-terminal branch must therefore only be treated as
"the eval died without emitting results" when the results event is genuinely
absent — otherwise a perfectly successful eval is force-failed: StopTask on the
worker, eval marked ``failed``, iteration failed.

These tests pin the precedence so the success signal always wins when both are
visible in one poll window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.jobs.optimization_executor import _wait_for_eval
from app.services.resource_error import ResourceError


@dataclass
class _FakeJob:
    """Minimal Job stand-in carrying the detached-path config."""

    job_id: str
    project_id: str = "proj-1"
    state: str = "running"
    error: ResourceError | None = None
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


def _detached_parent(job_id: str = "parent-1") -> _FakeJob:
    """Parent iteration job configured for the detached (arch_split) path."""
    return _FakeJob(
        job_id=job_id,
        config={"arch_split": True, "user_id": "owner-1", "owner_id": "owner-1"},
    )


@pytest.fixture
def _no_sleep(monkeypatch: pytest.MonkeyPatch):
    """Fast-forward the poll loop's 10s sleep."""

    async def _sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(
        "app.services.jobs.optimization_executor.asyncio.sleep", _sleep
    )


@pytest.mark.asyncio
async def test_results_event_wins_when_record_also_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _no_sleep,
):
    """Both signals visible in ONE poll window → success, not force-fail.

    This is the exact live-path race: the worker emits ``results`` and the
    record is mirrored to ``state="completed"`` seconds apart, so any tick
    landing after both sees both. Against the buggy ordering (record-terminal
    checked first, unconditionally) this eval is reaped and the iteration
    fails despite a fully successful run.
    """
    store = MagicMock()
    # Durable event log already carries the harness success signal.
    store.events_since = AsyncMock(
        return_value=[
            {"seq": 1, "kind": "progress", "data": {"event": "progress", "case": 1}},
            {
                "seq": 2,
                "kind": "progress",
                "data": {"event": "results", "file": "eval_results_v1_real.json"},
            },
        ]
    )
    # ...and the record has ALREADY been mirrored terminal.
    store.get_job = AsyncMock(return_value={"state": "completed"})
    store.get_optimization_state = AsyncMock(return_value=None)
    store.update_job_state = AsyncMock()
    store.claim_completion_fired = AsyncMock(return_value=True)

    platform = MagicMock()
    platform.metadata_store = store

    reap_calls: list[str] = []

    async def _fake_reap(_owner, _project, eval_job_id):
        reap_calls.append(eval_job_id)

    monkeypatch.setattr(
        "app.services.jobs.optimization_executor._reap_detached_eval", _fake_reap
    )
    monkeypatch.setattr(
        "app.services.jobs.optimization_executor.detect_infra_failure",
        lambda _results: None,
    )
    monkeypatch.setattr(
        "app.services.jobs.optimization_executor.detect_generation_unreliable",
        lambda _results: None,
    )
    monkeypatch.setattr(
        "app.services.jobs.optimization_executor._detect_judge_unreliable",
        lambda _results: None,
    )

    async def _fake_materialize(_project, _job, _filename):
        return tmp_path / "eval_results_v1_real.json"

    monkeypatch.setattr(
        "app.services.run_output_snapshot.materialize_detached_run", _fake_materialize
    )
    monkeypatch.setattr(
        "app.services.run_output_snapshot.load_materialized_results",
        lambda _path: [{"model": "m", "cases": []}],
    )
    monkeypatch.setattr(
        "app.services.jobs.optimization_executor._extract_iteration_data",
        lambda results, output_file: {
            "results": results, "run_output_file": output_file
        },
    )
    monkeypatch.setattr(
        "app.services.jobs.optimization_executor.build_generation_caveats",
        lambda _results: [],
    )

    parent_job = _detached_parent()
    tracker = _FakeTracker(jobs={"parent-1": parent_job})

    with patch("app.platform.get_platform", return_value=platform):
        result = await _wait_for_eval(
            eval_job_id="eval-1",
            project_dir=tmp_path,
            project_id="proj-1",
            parent_job=parent_job,  # type: ignore[arg-type]
            job_tracker=tracker,  # type: ignore[arg-type]
        )

    assert result is not None, (
        "A successful eval whose record was already mirrored terminal must NOT "
        "be treated as 'died without results'."
    )
    assert result["run_output_file"] == "eval_results_v1_real.json"
    assert reap_calls == [], (
        f"Successful eval must not be reaped, got reap calls: {reap_calls}"
    )
    assert not any(
        st == "failed" for (jid, st, _e) in tracker.state_updates if jid == "parent-1"
    ), f"Parent must not be failed: {tracker.state_updates}"


@pytest.mark.asyncio
async def test_record_terminal_without_results_still_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _no_sleep,
):
    """The genuine 'worker died' case keeps its fail-fast behavior.

    Record says terminal, event log has NO ``results`` event → the eval really
    did die (credential expiry, task kill, reconciler reap). It must still be
    reaped and the iteration failed. This is the counterpart assertion that
    keeps the fix from silently disabling the second release condition.
    """
    store = MagicMock()
    store.events_since = AsyncMock(
        return_value=[
            {"seq": 1, "kind": "progress", "data": {"event": "progress", "case": 1}},
        ]
    )
    store.get_job = AsyncMock(return_value={"state": "failed"})
    store.get_optimization_state = AsyncMock(return_value=None)
    store.update_job_state = AsyncMock()

    platform = MagicMock()
    platform.metadata_store = store

    reap_calls: list[str] = []

    async def _fake_reap(_owner, _project, eval_job_id):
        reap_calls.append(eval_job_id)

    monkeypatch.setattr(
        "app.services.jobs.optimization_executor._reap_detached_eval", _fake_reap
    )

    parent_job = _detached_parent("parent-2")
    tracker = _FakeTracker(jobs={"parent-2": parent_job})

    with patch("app.platform.get_platform", return_value=platform):
        result = await _wait_for_eval(
            eval_job_id="eval-2",
            project_dir=tmp_path,
            project_id="proj-1",
            parent_job=parent_job,  # type: ignore[arg-type]
            job_tracker=tracker,  # type: ignore[arg-type]
        )

    assert result is None, "A record-terminal eval with no results event must fail."
    assert reap_calls == ["eval-2"], (
        f"The dead worker must be reaped, got: {reap_calls}"
    )
