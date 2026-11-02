# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reverse-sync of a completed detached eval's ``output/`` results.

WHY this reverse leg exists: the detached Fargate eval worker mounts the
tenant EFS tree READ-ONLY (isolation boundary = read-only mount + /tmp
snapshot), so it CANNOT write its results back to the coordinator's project
directory. The completed results file therefore travels
worker → S3 → coordinator → EFS:

1. On the completed path (harness exit 0) the worker uploads its
   ``/tmp/workspace/{job_id}/output/eval_results_*.json`` file(s) to S3 under
   the job-derived ``output/`` key (``upload_run_output_to_s3``).
2. At the shared detached-terminal point (optimization loop, manual
   republisher, reconciler orphan-recovery) the coordinator downloads that
   object to the project's EFS ``output/`` dir with a bounded
   poll-until-present loop (``download_run_output_to_efs``) and materializes
   the run (``materialize_detached_run``).
3. Downstream readers (metrics extraction, run-detail API, report generation)
   read the materialized FILE via the normalizing ``load_materialized_results``
   companion — the DDB ``result`` field is unreliable at scale (a ~10 MB
   208-case write silently exceeds the 400 KB DynamoDB item cap).

This module mirrors ``working_tree_snapshot.py`` (which holds both directions
of the working-tree transfer). It reuses the worker's already-constructed S3
client + snapshot bucket rather than inventing new infra.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.services.snapshot_key import output_key_for_job

logger = logging.getLogger(__name__)


class RunOutputNotAvailable(Exception):
    """Raised when a detached run's ``output/`` file cannot be materialized.

    Signals that the worker-uploaded object never landed within the bounded
    download ceiling, that materialization was attempted off the detached path
    in a non-Fargate (dev) context, or that the materialized file is not a
    non-empty list. Every caller (optimization loop, manual republisher,
    reconciler) fails or skips CLEANLY on this — it never hangs.
    """


async def upload_run_output_to_s3(
    output_dir: Path, s3_client, bucket: str, job_id: str
) -> list[str]:
    """Upload a completed eval's ``eval_results_*.json`` file(s) to S3.

    Finds every ``eval_results_*.json`` file in ``output_dir`` and uploads each
    under ``output_key_for_job(job_id, filename)``. Mirrors
    ``upload_snapshot_to_s3`` — the blocking boto3 put runs in a thread.

    Args:
        output_dir: The worker's ``/tmp/workspace/{job_id}/output`` directory.
        s3_client: Low-level boto3 S3 client (the worker's own).
        bucket: Snapshot S3 bucket name.
        job_id: Job ID (drives the S3 key prefix).

    Returns:
        The list of uploaded filenames (bare names, not paths). Empty when
        ``output_dir`` has no matching results file.
    """

    def _find_and_upload() -> list[str]:
        if not output_dir.is_dir():
            return []
        uploaded: list[str] = []
        for path in sorted(output_dir.glob("eval_results_*.json")):
            if not path.is_file():
                continue
            key = output_key_for_job(job_id, path.name)
            with open(path, "rb") as f:
                s3_client.put_object(Bucket=bucket, Key=key, Body=f)
            uploaded.append(path.name)
            logger.info(
                "Uploaded run output to S3",
                extra={"key": key, "job_id": job_id},
            )
        return uploaded

    return await asyncio.to_thread(_find_and_upload)


async def download_run_output_to_efs(
    s3_client,
    bucket: str,
    job_id: str,
    run_filename: str,
    dest_output_dir: Path,
    *,
    max_attempts: int = 10,
    delay_s: float = 0.25,
) -> Path:
    """Download the worker-uploaded output object to EFS (poll-until-present).

    The worker's upload lags the terminal ``results`` event (same window the
    coordinator's DDB consistent-read already tolerates), so this bounded
    poll-download retries on a missing key before giving up. On the ceiling it
    raises ``RunOutputNotAvailable`` rather than hanging.

    ``run_filename`` originates from the harness stdout (``data.file``), which
    is attacker-adjacent input crossing the worker→coordinator boundary. It is
    path-sanitized through ``safe_basename_path`` BEFORE any write — a rejected
    name (separators, hidden, NUL, traversal) raises ``RunOutputNotAvailable``,
    so the destination can never escape ``output/`` on the shared EFS tree.

    Args:
        s3_client: Low-level boto3 S3 client (the coordinator's).
        bucket: Snapshot S3 bucket name.
        job_id: Job ID (drives the S3 key prefix).
        run_filename: Bare results filename from the terminal event.
        dest_output_dir: The coordinator project's EFS ``output/`` directory.
        max_attempts: Poll attempts before ceiling (default 10).
        delay_s: Sleep between attempts on a missing key (default 0.25s).

    Returns:
        The EFS path the object was written to.

    Raises:
        RunOutputNotAvailable: On a rejected filename or an exhausted ceiling.
    """
    from app.services.project_paths import safe_basename_path

    # Sanitize BEFORE any filesystem write (attacker-adjacent harness stdout).
    dest_path = safe_basename_path(dest_output_dir, run_filename)
    if dest_path is None:
        raise RunOutputNotAvailable(
            f"Rejected unsafe run_filename {run_filename!r} (job={job_id})"
        )

    key = output_key_for_job(job_id, run_filename)

    def _get_bytes() -> bytes | None:
        """Return the object bytes, or None if the key is not present yet."""
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except Exception as exc:  # noqa: BLE001 — treat any read miss as not-yet-present
            # boto3/moto raise ClientError(NoSuchKey/404); match structurally on
            # the not-found signal without pattern-matching English: read the
            # structured error code, never third-party message prose.
            error_code = None
            if hasattr(exc, "response") and isinstance(exc.response, dict):
                error_code = exc.response.get("Error", {}).get("Code")
            if error_code in ("NoSuchKey", "404", "NotFound"):
                return None
            # Unknown/transient error — log and treat as not-yet-present so the
            # bounded loop retries rather than dying on a flaky read.
            logger.warning(
                "get_object failed for run output (job=%s key=%s): %s",
                job_id, key, exc,
            )
            return None

    for attempt in range(max_attempts):
        data = await asyncio.to_thread(_get_bytes)
        if data is not None:
            dest_output_dir.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(data)
            return dest_path
        if attempt < max_attempts - 1:
            await asyncio.sleep(delay_s)

    raise RunOutputNotAvailable(
        f"Run output {run_filename!r} did not appear in S3 within "
        f"{max_attempts} attempts (job={job_id})"
    )


async def materialize_detached_run(
    project_id: str,
    job_id: str,
    run_filename: str,
) -> Path:
    """Download a detached run's ``output/`` file from S3 to the coordinator EFS.

    The ONE shared helper called from all three terminal points (optimization
    loop, manual republisher, reconciler orphan-recovery) so no third copy of
    the "on detached terminal → download output/ from S3 to EFS" logic exists.

    Materialization is a cloud/detached-only operation: it resolves
    ``get_job_compute()`` and only proceeds when it is a ``FargateTaskCompute``
    (reusing its ``.s3`` client + ``.snapshot_bucket``, same isinstance-guard
    pattern as ``_reap_detached_eval``). In any other context (dev/in-process)
    it raises ``RunOutputNotAvailable``.

    Args:
        project_id: Project UUID (resolves the coordinator project dir).
        job_id: Job ID (drives the S3 key prefix).
        run_filename: Bare results filename from the terminal event.

    Returns:
        The EFS path the run was materialized to.

    Raises:
        RunOutputNotAvailable: When not on the Fargate/detached path, or when
            the download ceilings out / the filename is rejected.
    """
    from app.platform.aws.fargate_task_compute import FargateTaskCompute
    from app.platform.job_compute import get_job_compute
    from app.services.project_paths import resolve_project_dir

    compute = get_job_compute()
    if not isinstance(compute, FargateTaskCompute):
        raise RunOutputNotAvailable(
            "materialize_detached_run is a cloud/detached-only operation "
            f"(compute={type(compute).__name__})"
        )

    project_dir = resolve_project_dir(project_id)
    dest_output_dir = project_dir / "output"

    return await download_run_output_to_efs(
        compute.s3,
        compute.snapshot_bucket,
        job_id,
        run_filename,
        dest_output_dir,
    )


def load_materialized_results(efs_path: Path) -> list[dict]:
    """Load + NORMALIZE a materialized run file into a per-model run list.

    Wraps ``EvalRunner().load_results`` (typed ``list | dict``, a bare
    ``json.loads`` with NO normalization) and mirrors the harness runner's
    ``_load_harness_results`` normalization: a dict return is wrapped as
    ``[data]``. Downstream readers (``_extract_iteration_data``,
    ``detect_infra_failure``, ``_detect_judge_unreliable``) all iterate
    ``for model_run in results`` and ``isinstance(model_run, dict)``-filter, so
    an un-normalized dict return would silently drop every model run.

    All three terminal points consume THIS, not raw ``load_results``, so they
    share one normalization.

    Args:
        efs_path: Path to the materialized results JSON on EFS.

    Returns:
        The per-model run list (non-empty).

    Raises:
        RunOutputNotAvailable: When the file is not a non-empty list after
            normalization (consistent with the download ceiling-fail path).
    """
    from app.services.eval_runner import EvalRunner

    data = EvalRunner().load_results(efs_path)
    results = [data] if isinstance(data, dict) else data
    if not isinstance(results, list) or len(results) == 0:
        raise RunOutputNotAvailable(
            f"Materialized run {efs_path.name} is not a non-empty results list"
        )
    return results


__all__ = [
    "RunOutputNotAvailable",
    "upload_run_output_to_s3",
    "download_run_output_to_efs",
    "materialize_detached_run",
    "load_materialized_results",
]
