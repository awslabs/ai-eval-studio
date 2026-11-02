# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``defer_completion`` must hand the results back without recording completion.

Driven through the REAL harness rather than a mock, because the thing under test
is a side effect a mocked harness cannot have. ``complete_job`` is the terminal
chokepoint: it schedules the durable "completed" write, publishes the per-job
``done`` envelope and the project event, and releases the project's eval lock --
immediately, and irreversibly, because the fan-out is skipped once the job is
already terminal. The detached worker uploads its only copy of the results AFTER
the harness returns, so completing here announced success while those results were
still on a disk about to disappear, and a later ``fail_job`` could not retract it.

This runs in the DEFAULT suite. It was marked ``integration`` and therefore
deselected, which meant the fix it pins was never exercised by a normal run -- and
the marker is defined for tests that hit real external services (AWS, network),
which this does not: the harness it drives is a stdlib-only stub written into
``tmp_path``. It needs only ``uv`` on PATH, which is how the harness is always
invoked.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from app.services.jobs import get_job_tracker

# The per-case shape the harness produces: a list of dicts with a stable key set.
_GOLDEN_RESULTS = [
    {"test_case": "case1", "input": "q1", "actual_output": "a1", "score": 0.9, "passed": True},
    {"test_case": "case2", "input": "q2", "actual_output": "a2", "score": 0.7, "passed": True},
    {"test_case": "case3", "input": "q3", "actual_output": "a3", "score": 0.4, "passed": False},
]


def write_stub_project(project_dir: Path) -> None:
    """Write a minimal standalone project whose run_eval.py emits golden results.

    The harness writes ``output/results.json`` with the canonical per-case
    list-of-dicts shape and emits the ``results`` progress event
    ``run_eval_via_harness`` looks for. It uses only the stdlib, so ``uv run``
    needs no dependency resolution and no network — which is what lets the
    ``skip_sync=True`` path run without a baked venv while staying faithful to the
    real harness contract.

    Args:
        project_dir: Directory to populate. Created if absent.
    """
    (project_dir / "output").mkdir(parents=True, exist_ok=True)
    (project_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (project_dir / "prompts" / "baseline.md").write_text("Answer the question.")

    golden_literal = json.dumps(_GOLDEN_RESULTS)
    run_eval = project_dir / "run_eval.py"
    run_eval.write_text(
        "import argparse, json, sys\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--models')\n"
        "parser.add_argument('--prompt-version')\n"
        "parser.add_argument('--json-progress', action='store_true')\n"
        "parser.add_argument('--source')\n"
        "parser.add_argument('--max-examples', type=int)\n"
        "parser.add_argument('--judge-model')\n"
        "parser.add_argument('--data-file')\n"
        "parser.add_argument('--max-concurrent', type=int)\n"
        "args, _ = parser.parse_known_args()\n"
        f"results = json.loads(r'''{golden_literal}''')\n"
        "with open('output/results.json', 'w') as f:\n"
        "    json.dump(results, f)\n"
        "print(json.dumps({'event': 'scoring', 'done': 3, 'total': 3}))\n"
        "print(json.dumps({'event': 'results', 'file': 'output/results.json'}))\n"
        "print(json.dumps({'event': 'complete', 'done': 3, 'total': 3}))\n"
        "sys.stdout.flush()\n"
    )
    # Minimal pyproject so `uv run` treats the dir as a project (stdlib only)
    (project_dir / "pyproject.toml").write_text(
        "[project]\n"
        'name = "stub-harness"\n'
        'version = "0.0.0"\n'
        'requires-python = ">=3.11"\n'
        "dependencies = []\n"
    )


requires_uv = pytest.mark.skipif(
    shutil.which("uv") is None,
    reason="uv not on PATH — the harness is always invoked as `uv run`.",
)


@requires_uv
@pytest.mark.asyncio
async def test_defer_completion_returns_results_without_terminalizing(tmp_path):
    """The harness hands the payload back and leaves the terminal write to the caller.

    The unit tests around the worker mock the harness, so they can assert that the
    worker ASKS to defer but not that deferring works. This asserts the effect: the
    results come back, and the job is still non-terminal afterwards.
    """
    from app.services.harness_runner import run_eval_via_harness

    write_stub_project(tmp_path)
    tracker = get_job_tracker()
    job = tracker.create_job(
        project_id="defer-project",
        job_type="eval",
        config={"project_dir": str(tmp_path)},
    )

    run = await run_eval_via_harness(
        job_id=job.job_id,
        project_dir=tmp_path,
        model_ids=["stub-model"],
        prompt_version="baseline",
        max_test_cases=None,
        judge_model_id=None,
        skip_sync=True,
        defer_completion=True,
    )

    assert run.outcome == "completed"
    assert run.results is not None, "the payload must come back to the caller"
    assert run.output_file
    # The job is NOT terminal: nothing has been announced or released yet.
    assert tracker.get_job(job.job_id).status not in (
        "completed", "failed", "cancelled"
    ), "defer_completion must leave the terminal transition to the caller"

    # And the caller can then complete it with what it was handed.
    tracker.complete_job(job.job_id, run.results, run.output_file)
    assert tracker.get_job(job.job_id).status == "completed"


@requires_uv
@pytest.mark.asyncio
async def test_without_defer_the_harness_still_completes_the_job(tmp_path):
    """The default is unchanged, so the flag is what moved the transition.

    Without this the test above would pass just as well against a harness that
    never completes anything.
    """
    from app.services.harness_runner import run_eval_via_harness

    write_stub_project(tmp_path)
    tracker = get_job_tracker()
    job = tracker.create_job(
        project_id="no-defer-project",
        job_type="eval",
        config={"project_dir": str(tmp_path)},
    )

    await run_eval_via_harness(
        job_id=job.job_id,
        project_dir=tmp_path,
        model_ids=["stub-model"],
        prompt_version="baseline",
        max_test_cases=None,
        judge_model_id=None,
        skip_sync=True,
    )

    assert tracker.get_job(job.job_id).status == "completed"
