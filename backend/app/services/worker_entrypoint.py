# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker entrypoint — boot, snapshot pull, harness execution, durable event log.

The worker bootstraps from tiny env vars (job_id + owner_id + project_id),
reads the full job spec from DynamoDB, pulls the S3 snapshot, runs the harness
WITHOUT uv sync (deps baked into image), and drives progress/results/terminal
state to the durable event log.

Called by ``app.scripts.worker_main`` (the Fargate RunTask entrypoint).
"""
from __future__ import annotations

import asyncio
import logging
import os
from decimal import Decimal
from pathlib import Path

logger = logging.getLogger(__name__)

#: Root for per-job scratch space. The Fargate worker's container has a writable
#: ``/tmp``, so this is correct in production. It is a module constant rather than
#: three inline f-strings so tests can point it at a temp dir — the literals made
#: four tests depend on being able to write to a fixed absolute path, which they
#: silently could not under a sandboxed shell.
WORKSPACE_ROOT = Path("/tmp/workspace")

#: Attempts to get a completed run's results into S3 before giving up.
#:
#: Worth retrying because the bytes are still on local disk and the usual causes
#: (a throttle, a dropped connection) clear in seconds. Worth bounding because
#: the task is billing the whole time and the reconciler is a better answer than
#: an indefinite spin.
_OUTPUT_UPLOAD_ATTEMPTS = 3
_OUTPUT_UPLOAD_BACKOFF_S = 2.0


async def _commit_run_output(
    *, job_id: str, tracker, s3_client, snapshot_bucket: str
) -> bool:
    """Upload a completed run's results to S3, retrying while the bytes exist.

    Returns True when the results are durable in S3. On failure, records the
    reason on the job record and returns False so the caller can report failure
    rather than a completion whose output no longer exists anywhere.
    """
    from app.services.resource_error import Codes, ResourceError
    from app.services.run_output_snapshot import upload_run_output_to_s3

    output_dir = WORKSPACE_ROOT / job_id / "output"
    last_error: Exception | None = None

    for attempt in range(1, _OUTPUT_UPLOAD_ATTEMPTS + 1):
        try:
            uploaded = await upload_run_output_to_s3(
                output_dir, s3_client, snapshot_bucket, job_id
            )
        except Exception as e:  # noqa: BLE001 - retried, then reported below
            last_error = e
            logger.warning(
                "Run output upload attempt failed",
                exc_info=e,
                extra={"job_id": job_id, "attempt": attempt},
            )
        else:
            if uploaded:
                logger.info(
                    "Worker uploaded run output",
                    extra={"job_id": job_id, "files": uploaded},
                )
                return True
            # Nothing matched. The harness reported completion, so a missing
            # results file is a real inconsistency, not a transient fault —
            # retrying cannot conjure the file.
            tracker.fail_job(job_id, ResourceError(
                code=Codes.HARNESS_SUBPROCESS_FAILED,
                message="Eval completed but produced no results file to upload.",
                hint=(
                    "The harness reported success and the output directory is "
                    "empty. Check the per-job log for what it wrote."
                ),
                details={"output_dir": str(output_dir)},
            ))
            logger.error(
                "Completed run had no output files to upload",
                extra={"job_id": job_id, "output_dir": str(output_dir)},
            )
            return False

        if attempt < _OUTPUT_UPLOAD_ATTEMPTS:
            await asyncio.sleep(_OUTPUT_UPLOAD_BACKOFF_S * attempt)

    tracker.fail_job(job_id, ResourceError(
        code=Codes.HARNESS_SUBPROCESS_FAILED,
        message="Eval completed but its results could not be saved.",
        hint=(
            "The results existed only on the worker's temporary disk and the "
            "upload to S3 failed. Re-run the eval."
        ),
        details={
            "attempts": _OUTPUT_UPLOAD_ATTEMPTS,
            "exception": type(last_error).__name__ if last_error else "unknown",
        },
    ))
    logger.error(
        "Run output upload failed after every attempt — reporting failure",
        exc_info=last_error,
        extra={"job_id": job_id},
    )
    return False


async def run_worker(job_id: str, owner_id: str, project_id: str) -> int:
    """Worker bootstrap: read job spec → pull snapshot → run harness → report.

    1. Boots the platform via ``get_platform()``, which builds it from
       ``EVAL_STUDIO_MODE`` on first use. The worker's AWS calls resolve
       credentials through boto3's default chain; in cloud,
       ``app.scripts.worker_main`` has already pointed ``AWS_CONFIG_FILE`` at a
       ``credential_process`` helper that mints this job's scoped,
       self-refreshing credentials (see ``worker_credentials_config``).
    2. Reads the full job spec from ``metadata_store.get_job``; if missing,
       log + return exit code 1.
    3. Pulls + safely unpacks the S3 snapshot to ``WORKSPACE_ROOT/{job_id}``
       via ``download_and_unpack_snapshot`` (traversal-safe unpack).
    4. Registers a Job in the worker's own ``JobTracker`` with
       ``config={..., "arch_split": True, "owner_id": owner_id}`` (so
       ``Job.append_event`` durably appends to DDB via allocate_job_seq /
       append_job_event — reusing the existing wiring).
    5. Runs the harness: ``await run_eval_via_harness(..., skip_sync=True)``.
       The harness writes its ``output/eval_results_*.json`` file (the same
       shape an in-process run produces) and drives terminal state through
       the JobTracker → durable event log. Its RETURN
       VALUE is the outcome; a failed eval returns normally rather than raising.
    6. On the completed path, uploads the results to S3 while they still exist on
       local disk, and only then records completion. The harness is asked to
       DEFER completion (``defer_completion=True``) so there is exactly one
       terminal transition and it happens after the output is durable — the EFS
       mount is read-only here, so ``/tmp`` holds the only copy, and a job must
       never announce completion with its results unreachable.
    7. Drains pending durable writes (``await tracker.drain_pending_writes()``)
       so the terminal state/event is persisted before the event loop closes.

    Returns:
        Exit code 0 when the eval completed with durable results, or was
        cancelled as asked. 1 when the eval failed, its results could not be
        saved, or the terminal state could not be persisted.
    """
    from claude_agent_sdk import project_key_for_directory

    from app.platform import get_platform
    from app.services.harness_runner import (
        HarnessOutcome,
        HarnessRun,
        run_eval_via_harness,
    )
    from app.services.jobs import get_job_tracker
    from app.services.working_tree_snapshot import (
        SnapshotTraversalError,
        download_and_unpack_snapshot,
    )

    logger.info(
        "Worker starting",
        extra={"job_id": job_id, "owner_id": owner_id, "project_id": project_id},
    )

    # 1. Boot the cloud platform
    try:
        platform = get_platform()
        store = platform.metadata_store
    except Exception as e:
        logger.error("Failed to get platform", exc_info=e)
        return 1

    # Build the S3 client with boto3's default credential chain. In cloud,
    # ``app.scripts.worker_main`` has already set AWS_CONFIG_FILE to a
    # credential_process helper and removed the static AWS_ACCESS_KEY_ID /
    # AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN variables (see
    # ``worker_credentials_config``), so this client gets the job's scoped,
    # self-refreshing credentials. The Platform object exposes no S3 client,
    # so the worker builds its own.
    import boto3
    from botocore.config import Config

    try:
        region = os.environ.get("AWS_REGION", "us-east-1")
        boto_session = boto3.session.Session(region_name=region)
        s3_client = boto_session.client("s3", config=Config(max_pool_connections=20))

        # Read snapshot bucket from env (loud KeyError if unset — no hardcoded fallback)
        snapshot_bucket = os.environ["S3_SNAPSHOTS_BUCKET"]
    except KeyError as e:
        logger.error(f"Missing required env var: {e}", extra={"job_id": job_id})
        return 1

    # 2. Read the full job spec from DDB
    try:
        job = await store.get_job(owner_id, project_id, job_id)
        if job is None:
            logger.error("Job not found in store", extra={"job_id": job_id})
            return 1
    except Exception as e:
        logger.error("Failed to read job spec", exc_info=e, extra={"job_id": job_id})
        return 1

    # Extract harness params from the nested config.
    # Harness params (model_ids, prompt_version, metrics, max_test_cases,
    # judge_model_id, data_file_override, source) are nested under 'config'
    # because dispatch_eval's create_job stores them there, and get_job
    # returns the DynamoDB record unflattened. Top-level keys are only
    # owner_id/project_id/job_id/type/state/progress/created_at.
    cfg = job.get("config", {})

    # Cast numeric config fields so Decimal (from DDB _from_dynamo decode) does
    # not leak into run_eval_via_harness / json.dumps. _from_dynamo returns
    # Decimal for numeric strings, which breaks JSON serialization downstream.
    if "max_test_cases" in cfg and cfg["max_test_cases"] is not None:
        cfg["max_test_cases"] = int(cfg["max_test_cases"]) if isinstance(cfg["max_test_cases"], Decimal) else cfg["max_test_cases"]

    # 3. Pull + safely unpack the snapshot under WORKSPACE_ROOT/{job_id}
    workspace = WORKSPACE_ROOT / job_id
    workspace.mkdir(parents=True, exist_ok=True)

    # Compute snapshot key via shared helper (launch + worker MUST use same key)
    from app.services.snapshot_key import snapshot_key_for_job

    snapshot_key = snapshot_key_for_job(job_id)

    try:
        logger.info("Downloading snapshot", extra={"key": snapshot_key})
        await download_and_unpack_snapshot(
            s3_client, snapshot_bucket, snapshot_key, workspace
        )
    except SnapshotTraversalError as e:
        logger.error(
            "Snapshot traversal error",
            exc_info=e,
            extra={"job_id": job_id, "snapshot_key": snapshot_key},
        )
        return 1
    except Exception as e:
        logger.error(
            "Failed to download snapshot",
            exc_info=e,
            extra={"job_id": job_id, "snapshot_key": snapshot_key},
        )
        return 1

    # 4. Register a Job in the worker's JobTracker with arch_split config.
    #
    # durable_write=False: the coordinator owns the durable job record; the
    # worker owns the event log. The worker must NOT full-item put_job over
    # the coordinator's record — doing so ran tens of seconds after the
    # coordinator's compute_handle write and clobbered it (wiping the handle,
    # re-freezing state to queued, and replacing config with this sparse
    # worker shape). The worker still needs the in-memory Job (for
    # append_event / progress) and the replay provider — both are registered
    # regardless of the flag. The worker's RESULT + event-log writes (via
    # update_state / append_event during the harness run) are UNAFFECTED;
    # durable_write only gates create_job's put_job.
    #
    # owner_id is threaded so update_state's durable write-through (now keyed
    # on config.owner_id or user_id) can still mirror the worker's own
    # terminal state to DDB if it fires; the coordinator additionally writes
    # the record terminal authoritatively on the event-log results signal.
    tracker = get_job_tracker()
    try:
        tracker.create_job(
            project_id=project_id,
            job_id=job_id,
            job_type=job.get("type", "eval"),
            config={
                "arch_split": True,
                "owner_id": owner_id,
                "project_id": project_id,
                "user_id": cfg.get("user_id"),
                # The completion seed names the prompt from this; without it
                # the seed reads '?'.
                "prompt_version": cfg.get("prompt_version"),
            },
            durable_write=False,
        )
    except Exception as e:
        logger.error("Failed to create job in tracker", exc_info=e)
        return 1

    # 5. Run the harness with skip_sync=True (deps baked into image)
    #
    # The outcome is the harness's OWN verdict, not "did it raise". An ordinary
    # eval failure is recorded on the job record and returns normally, so
    # branching on exceptions treated every failed eval as a success and exited
    # zero.
    outcome: HarnessOutcome
    run: HarnessRun | None = None
    try:
        run = await run_eval_via_harness(
            job_id=job_id,
            project_dir=workspace,
            model_ids=cfg["model_ids"],
            prompt_version=cfg["prompt_version"],
            max_test_cases=cfg.get("max_test_cases"),
            judge_model_id=cfg.get("judge_model_id"),
            data_file_override=cfg.get("data_file_override"),
            source=cfg.get("source", "manual"),
            skip_sync=True,  # Worker: deps baked, skip uv sync
            # The harness must NOT record completion — this function does, once,
            # after the results are durable in S3. See step 6.
            defer_completion=True,
        )
        outcome = run.outcome
    except Exception as e:
        logger.error("Harness execution failed", exc_info=e, extra={"job_id": job_id})
        outcome = "failed"

    # 6. Commit the output, THEN record the single terminal transition.
    #
    # This worker's EFS mount is read-only, so ``/tmp`` holds the ONLY copy of
    # the results until they reach S3 — and ``/tmp`` dies with the task.
    #
    # The ordering has to include the terminal WRITE, not just the drain.
    # ``complete_job`` is the terminal chokepoint: it schedules the durable
    # ``completed`` write, publishes the per-job ``done`` envelope and the project
    # event, and releases the project's eval lock, all immediately. Letting the
    # harness call it before this upload announced success while the results were
    # still only on local disk — and a later ``fail_job`` could not retract it,
    # because the fan-out is gated on the job not already being terminal. The SPA
    # kept "completed" while the durable record went to "failed", and the lock was
    # already free for another eval.
    #
    # So the harness defers, and exactly one transition happens here: completed
    # only once the bytes are in S3, failed otherwise.
    #
    # Only the completed path uploads. Cancelled runs have already had their
    # /tmp output deleted by the harness, and a failed run must not publish
    # partial output as if it were a result.
    if outcome == "completed":
        if await _commit_run_output(
            job_id=job_id,
            tracker=tracker,
            s3_client=s3_client,
            snapshot_bucket=snapshot_bucket,
        ):
            # Results are durable — now, and only now, the job is complete.
            tracker.complete_job(
                job_id,
                run.results if run else [],
                (run.output_file if run else None) or "results.json",
            )
        else:
            # ``_commit_run_output`` already recorded the failure on the job.
            outcome = "failed"

    # 7. Drain pending durable writes (state + events) before the event loop
    # closes. Backgrounded writes are tracked in tracker._pending_writes and
    # module-level _durable_append_tasks; drain awaits them to completion so the
    # terminal state/event is durably persisted (not cancelled at loop
    # shutdown). This is the commit point for whichever terminal state stands
    # after the output check above.
    try:
        await tracker.drain_pending_writes()
    except Exception as e:
        logger.warning("Drain pending writes failed", exc_info=e, extra={"job_id": job_id})
        # A terminal state that never reached the store leaves the job
        # non-terminal for the reconciler to pick up. Do not report success.
        outcome = "failed"

    logger.info("Worker finished", extra={"job_id": job_id, "outcome": outcome})
    # Cancelled is not a worker failure — the cancel was honoured.
    return 0 if outcome in ("completed", "cancelled") else 1


__all__ = ["run_worker"]
