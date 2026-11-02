# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test: results.json byte-parity.

Locks the contract: the arch split moves WHERE eval code runs (in-process vs. an
isolated Fargate worker), not WHAT it computes. The worker runs the exact same
``run_eval.py`` harness against the exact same snapshotted working tree; the only
runner-side difference is ``skip_sync`` (the worker's deps are baked into the
image, so it skips ``uv sync``). ``skip_sync`` therefore MUST NOT change the
produced ``results.json`` by a single byte.

This test drives one stub harness through ``run_eval_via_harness`` twice —
once with ``skip_sync=False`` (the in-process path) and once with
``skip_sync=True`` (the worker path) — for identical inputs, and asserts the
two ``results.json`` files are byte-identical AND carry the established
per-case shape (``list[dict]``).

Marked ``@pytest.mark.integration`` so it skips by default and only runs
under ``pytest -m integration``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.jobs import get_job_tracker

from ..test_harness_defer_completion import _GOLDEN_RESULTS, write_stub_project


async def _run_and_read_results(project_dir: Path, *, skip_sync: bool) -> bytes:
    """Run the stub harness via run_eval_via_harness and return results.json bytes."""
    from app.services.harness_runner import run_eval_via_harness

    tracker = get_job_tracker()
    job = tracker.create_job(
        project_id="parity-project",
        job_type="eval",
        config={"project_dir": str(project_dir)},
    )
    await run_eval_via_harness(
        job_id=job.job_id,
        project_dir=project_dir,
        model_ids=["stub-model"],
        prompt_version="baseline",
        max_test_cases=None,
        judge_model_id=None,
        skip_sync=skip_sync,
    )
    return (project_dir / "output" / "results.json").read_bytes()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_results_json_byte_identical_local_vs_worker(tmp_path):
    """The worker path (skip_sync=True) produces a byte-identical results.json.

    Runs the same harness for the same inputs on the in-process path
    (skip_sync=False) and the worker path (skip_sync=True); the produced
    results.json must be byte-for-byte identical and carry the established
    per-case list[dict] shape.
    """
    import shutil

    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH — required to run the harness subprocess.")

    # In-process path
    local_dir = tmp_path / "local"
    write_stub_project(local_dir)
    local_bytes = await _run_and_read_results(local_dir, skip_sync=False)

    # Worker path (deps baked into image → skip uv sync)
    worker_dir = tmp_path / "worker"
    write_stub_project(worker_dir)
    worker_bytes = await _run_and_read_results(worker_dir, skip_sync=True)

    # Byte-parity: the split changed only WHERE the code ran
    assert local_bytes == worker_bytes, "results.json differs between local and worker paths"

    # Shape: the established per-case list[dict]
    parsed = json.loads(worker_bytes)
    assert isinstance(parsed, list), "results.json must be a list"
    assert all(isinstance(item, dict) for item in parsed), "each case must be a dict"
    assert parsed == _GOLDEN_RESULTS, "per-case shape/content drifted from golden"
