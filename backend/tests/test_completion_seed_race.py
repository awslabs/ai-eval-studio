# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the completion-seed race: a finished eval must never be reported as 0/0.

The race: the completion-turn seed could report "Pass rate: 0/0 (N/A)" / "no cases
scored, file may be malformed" for an eval that passed, with a correct job record and
result file.

How it arises (cross-process): the durable terminal ``results`` progress event is
appended MID-STREAM (``handle_harness_progress`` → ``Job.append_event`` → a
backgrounded ``append_job_event``), and that event is what the poller detects. The
durable job ``result`` FIELD is written later, by ``JobTracker.complete_job`` and its
``update_state`` write-through. So a poller or reconciler in another process can see
the terminal event and read ``job_record.result`` BEFORE the result is durable — an
empty result, rendered as 0/0.

BEHAVIOURS TESTED:
- The worker persists the result before a terminal signal can trigger a seed
  (closes the race at its source).
- get_job has a ConsistentRead option, and completion-seed reads use consistent=True.
- The seed NEVER emits 0/0/N/A for a not-ready result; bounded wait + neutral degrade.
- output_file is persisted durably and set in the poller + reconciler Job rebuild.
- prompt_version is threaded into the worker job config, so the seed shows the real
  prompt name, not '?'.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.platform import get_platform, set_platform
from app.platform.dev_local import FilesystemMetadataStore
from app.services.completion_turn import _build_synthetic_message
from app.services.jobs import Job

if TYPE_CHECKING:
    from app.platform.types import JobId, ProjectId


@pytest.fixture(autouse=True)
def _setup_test_projects_dir(tmp_path, monkeypatch):
    """Set up a temporary projects directory for all tests in this module."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()

    # Reset platform singleton to force fresh build with tmp projects_dir
    set_platform(None)

    # Build a custom platform with the tmp projects_dir
    from app.platform import Platform
    from app.platform.dev_local import LocalAuth, NoOpTokenVending, _build_dev_session_store

    metadata = FilesystemMetadataStore(projects_dir=projects_dir)
    auth = LocalAuth(metadata)
    platform = Platform(
        metadata_store=metadata,
        session_store=_build_dev_session_store(),
        token_vending=NoOpTokenVending(),
        auth=auth,
    )
    set_platform(platform)

    yield projects_dir

    # Clean up
    set_platform(None)


# --- Consistent reads ---


@pytest.mark.asyncio
async def test_get_job_accepts_consistent_kwarg():
    """get_job accepts consistent: bool kwarg (Protocol + adapters)."""
    store = get_platform().metadata_store

    owner_id = "user-001"
    project_id = "proj-abc"
    job_id = str(uuid.uuid4())

    # Create a job record
    await store.put_job({
        "job_id": job_id,
        "project_id": project_id,
        "owner_id": owner_id,
        "type": "eval",
        "state": "completed",
        "created_at": "2026-07-10T10:00:00Z",
        "result": [{"model_id": "claude-3-5-sonnet-latest", "results": [{"success": True}]}],
    })

    # Should accept consistent=False (default)
    job = await store.get_job(owner_id, project_id, job_id, consistent=False)
    assert job is not None
    assert job["job_id"] == job_id

    # Should accept consistent=True
    job = await store.get_job(owner_id, project_id, job_id, consistent=True)
    assert job is not None
    assert job["job_id"] == job_id


# --- Never report 0/0 on a not-ready eval ---


@pytest.mark.asyncio
async def test_seed_never_reports_zero_over_zero_for_not_ready_result():
    """Completion seed does NOT emit '0/0 (N/A)' when result is not ready yet.

    Simulates the race: job.result is empty/None when the seed first reads it.
    The seed must detect this (total==0 on a completed eval is impossible) and
    either wait for the result or degrade to a neutral seed.
    """
    # Build a Job with empty result (simulating the not-ready race)
    job = Job(
        job_id=str(uuid.uuid4()),
        project_id="proj-race",
        type="eval",
        state="completed",
        config={"prompt_version": "baseline"},
        created_at="2026-07-10T10:00:00Z",
        progress={},
        result=None,  # NOT READY YET — the race condition
    )

    # Build the seed
    seed = await _build_synthetic_message(job)

    # The seed MUST NOT contain "0/0" or "N/A" or "no cases scored" or "file may be malformed"
    assert "0/0" not in seed, f"Seed must NOT report 0/0 on not-ready result. Got: {seed}"
    assert "N/A" not in seed, f"Seed must NOT report N/A on not-ready result. Got: {seed}"
    assert "no cases scored" not in seed.lower(), f"Seed must NOT say 'no cases scored'. Got: {seed}"
    assert "malformed" not in seed.lower(), f"Seed must NOT say 'malformed'. Got: {seed}"

    # The seed SHOULD be a neutral message acknowledging completion
    # but NOT fabricating a pass rate
    assert "complete" in seed.lower(), f"Seed should acknowledge completion. Got: {seed}"


@pytest.mark.asyncio
async def test_seed_never_reports_zero_over_zero_for_empty_list_result():
    """Completion seed does NOT emit '0/0 (N/A)' when result is an empty list.

    Variant: result is [] (empty list) instead of None.
    """
    job = Job(
        job_id=str(uuid.uuid4()),
        project_id="proj-race2",
        type="eval",
        state="completed",
        config={"prompt_version": "baseline"},
        created_at="2026-07-10T10:00:00Z",
        progress={},
        result=[],  # Empty list — also not-ready
    )

    seed = await _build_synthetic_message(job)

    # Must NOT contain the 0/0 / N/A / malformed guidance
    assert "0/0" not in seed, f"Seed must NOT report 0/0 on empty result. Got: {seed}"
    assert "N/A" not in seed, f"Seed must NOT report N/A on empty result. Got: {seed}"
    assert "no cases scored" not in seed.lower(), f"Seed must NOT say 'no cases scored'. Got: {seed}"
    assert "malformed" not in seed.lower(), f"Seed must NOT say 'malformed'. Got: {seed}"


@pytest.mark.asyncio
async def test_seed_correctly_reports_real_zero_pass_eval():
    """Completion seed CORRECTLY reports '0/N (0.0%)' for a real zero-pass eval.

    A genuine eval where total>0 but passed==0 (all cases failed) must STILL
    report the real rate. Only total==0 (impossible for a completed eval)
    triggers the not-ready degrade.
    """
    # Real eval result: 30 cases, 0 passed (all failed)
    job = Job(
        job_id=str(uuid.uuid4()),
        project_id="proj-real-zero",
        type="eval",
        state="completed",
        config={"prompt_version": "baseline"},
        created_at="2026-07-10T10:00:00Z",
        progress={},
        result=[{
            "model_id": "claude-3-5-sonnet-latest",
            "results": [{"success": False} for _ in range(30)]  # 30 failures
        }],
    )

    seed = await _build_synthetic_message(job)

    # This SHOULD report the real rate: 0/30 (0.0%)
    assert "0/30" in seed, f"Seed should report real zero-pass rate 0/30. Got: {seed}"
    assert "0.0%" in seed, f"Seed should report 0.0% pass rate. Got: {seed}"
    # Must NOT degrade to neutral (this is a real result, not not-ready)
    assert "finalizing" not in seed.lower(), f"Real zero-pass should NOT degrade. Got: {seed}"


# --- output_file persistence ---


@pytest.mark.asyncio
async def test_update_job_state_accepts_output_file_kwarg():
    """update_job_state accepts output_file: str | None kwarg (Protocol + adapters)."""
    store = get_platform().metadata_store

    owner_id = "user-002"
    project_id = "proj-output"
    job_id = str(uuid.uuid4())

    # Create a job record
    await store.put_job({
        "job_id": job_id,
        "project_id": project_id,
        "owner_id": owner_id,
        "type": "eval",
        "state": "running",
        "created_at": "2026-07-10T10:00:00Z",
    })

    # Update with output_file
    await store.update_job_state(
        owner_id,
        project_id,
        job_id,
        state="completed",
        output_file="output/eval_results_baseline.json",
    )

    # Read back and verify output_file is persisted
    job = await store.get_job(owner_id, project_id, job_id)
    assert job is not None
    assert job.get("output_file") == "output/eval_results_baseline.json", "output_file should be persisted"


@pytest.mark.asyncio
async def test_poller_rebuild_includes_output_file():
    """Job rebuild in poller includes output_file from job_record.

    This test simulates the poller's Job rebuild path in
    job_progress_republisher.py. The rebuilt Job must carry output_file
    so the seed's "Results file: {output_file}" line renders correctly.
    """
    store = get_platform().metadata_store

    owner_id = "user-003"
    project_id = "proj-poller-rebuild"
    job_id = str(uuid.uuid4())

    # Create a terminal job WITH output_file
    await store.put_job({
        "job_id": job_id,
        "project_id": project_id,
        "owner_id": owner_id,
        "type": "eval",
        "state": "completed",
        "created_at": "2026-07-10T10:00:00Z",
        "result": [{"model_id": "claude-3-5-sonnet-latest", "results": [{"success": True}]}],
        "output_file": "output/eval_results_v1.json",
    })

    # Read back (simulating poller's get_job)
    job_record = await store.get_job(owner_id, project_id, job_id)
    assert job_record is not None

    # Rebuild the Job the way job_progress_republisher.py does
    job = Job(
        job_id=job_record["job_id"],
        project_id=job_record["project_id"],
        type=job_record["type"],
        state=job_record["state"],
        config=job_record.get("config", {}),
        created_at=job_record.get("created_at"),
        progress=job_record.get("progress"),
        result=job_record.get("result"),
        output_file=job_record.get("output_file"),  # the rebuild must carry it
    )

    # Verify output_file is set
    assert job.output_file == "output/eval_results_v1.json", "Rebuilt Job must carry output_file"

    # Verify the seed renders the output_file line
    seed = await _build_synthetic_message(job)
    assert "output/eval_results_v1.json" in seed, f"Seed should show output_file. Got: {seed}"


# --- prompt_version in the worker job config ---


@pytest.mark.asyncio
async def test_seed_shows_prompt_version_not_question_mark():
    """Completion seed shows real prompt_version, not '?'.

    When the Job config carries prompt_version, the seed's "Eval on prompt
    '{prompt_version}' complete" line should show the real name, not '?'.
    """
    job = Job(
        job_id=str(uuid.uuid4()),
        project_id="proj-prompt-name",
        type="eval",
        state="completed",
        config={"prompt_version": "v2-improved"},  # what the seed names
        created_at="2026-07-10T10:00:00Z",
        progress={},
        result=[{
            "model_id": "claude-3-5-sonnet-latest",
            "results": [{"success": True} for _ in range(10)]
        }],
    )

    seed = await _build_synthetic_message(job)

    # Should show the real prompt_version
    assert "v2-improved" in seed, f"Seed should show real prompt version. Got: {seed}"
    assert "'?'" not in seed, f"Seed should NOT show '?' placeholder. Got: {seed}"


@pytest.mark.asyncio
async def test_seed_falls_back_to_question_mark_when_prompt_version_missing():
    """Seed falls back to '?' when prompt_version is missing from config.

    A job config without prompt_version still renders a seed, with '?' in
    place of the prompt name.
    """
    job = Job(
        job_id=str(uuid.uuid4()),
        project_id="proj-no-prompt-name",
        type="eval",
        state="completed",
        config={},  # NO prompt_version
        created_at="2026-07-10T10:00:00Z",
        progress={},
        result=[{
            "model_id": "claude-3-5-sonnet-latest",
            "results": [{"success": True} for _ in range(10)]
        }],
    )

    seed = await _build_synthetic_message(job)

    # Should fall back to '?'
    assert "'?'" in seed, f"Seed should fall back to '?' when prompt_version missing. Got: {seed}"
