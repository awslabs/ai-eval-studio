# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable off-EFS copies of report sidecars.

WHY THIS EXISTS. ``write_report_json`` writes a report to
``{project_dir}/output/reports/`` on EFS and nothing else. That was the only
copy: the run file travels to S3 (``run_output_snapshot``), the chat transcript
lives in S3 (``S3SessionStore``), and the report — the artifact a user actually
reads and shares — existed once, on one file system, with no backup and no
out-of-band way to fetch it. Losing or unmounting the EFS tree lost every
report, and an operator debugging a bad report had no way to see what was
written without going through the app.

WHICH BUCKET, and why not the snapshots bucket. Reports go to the SESSIONS
bucket:

* The snapshots bucket is wrong. It carries a ``snapshots-1d-expiry`` lifecycle
  rule (``storage_stack.py``), so a report copied there would silently
  disappear after a day — a backup that looks present and isn't is worse than
  no backup.
* The sessions bucket deliberately does NOT expire objects, is already granted
  to the coordinator task role for the whole bucket
  (``iam_stack.py`` ``_attach_shared_policies``), and already holds
  project-owned data that cascades on project delete. Same lifetime, same
  tenancy, same cleanup story.

WHY A TOP-LEVEL ``reports/`` PREFIX rather than nesting under the project's
session prefix: ``S3SessionStore.list_sessions`` paginates
``{project_key}/`` and treats each child as a session, so a ``reports/`` child
there would surface in the UI as a phantom chat session. Keeping reports out of
that key space entirely is the only way that cannot happen.

FAILURE POLICY IS ASYMMETRIC, on purpose. Upload failures WARN and return:
losing the durable copy must never fail the report the user asked for, and the
EFS copy is still there. Delete failures RAISE, because
``_complete_project_deletion`` keeps the ``deleting`` marker and retries — silently
skipping the delete would orphan tenant data in S3 forever.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: Top-level key prefix. Deliberately outside every ``{project_key}/`` space —
#: see the module docstring.
_REPORT_PREFIX = "reports"


def report_key_for_project(project_id: str, filename: str) -> str:
    """S3 key for one report sidecar.

    Uses the on-disk FILENAME rather than re-deriving scope + run name, so the
    S3 copy and the EFS copy cannot disagree about what a report is called.

    Args:
        project_id: Owning project.
        filename: Sidecar file name, e.g. ``executive.report.json`` or
            ``eval_results_v4_20260909_224651.report.json``.

    Returns:
        The object key.
    """
    return f"{_REPORT_PREFIX}/{project_id}/{filename}"


def _store():
    """The platform session store, or ``None`` in dev mode.

    Reuses the store's already-constructed client and bucket rather than
    building a second S3 client, following ``run_output_snapshot``'s precedent.
    In dev this is ``None`` and every function here becomes a no-op — reports
    live under the project tree and there is nothing to back up to.
    """
    from app.platform import get_platform

    return get_platform().session_store


async def upload_report_to_s3(project_id: str, report_path: Path) -> str | None:
    """Copy a freshly-written report sidecar to S3.

    Best-effort by design: a failure here leaves the EFS copy intact and must
    not fail the caller's ``save_report``. It logs a WARNING rather than staying
    silent, because a durable copy that quietly stopped happening is the same
    class of bug this module was written to fix.

    Args:
        project_id: Owning project.
        report_path: Path to the sidecar just written on EFS.

    Returns:
        The S3 key on success, ``None`` in dev mode or on failure.
    """
    store = _store()
    if store is None:
        return None

    key = report_key_for_project(project_id, report_path.name)

    def _put() -> None:
        with open(report_path, "rb") as f:
            store.s3.put_object(
                Bucket=store.bucket,
                Key=key,
                Body=f,
                ContentType="application/json",
            )

    try:
        await asyncio.to_thread(_put)
    except Exception as exc:  # noqa: BLE001 — durability is best-effort
        logger.warning(
            "[report-snapshot] project=%s could not copy %s to S3 (%s). The EFS "
            "copy is intact, but this report has no durable backup.",
            project_id[:8], report_path.name, exc,
        )
        return None

    logger.info(
        "[report-snapshot] project=%s copied %s to s3://%s/%s",
        project_id[:8], report_path.name, store.bucket, key,
    )
    return key


async def delete_project_reports_from_s3(project_id: str) -> int:
    """Remove every S3 report copy for a project.

    Called from the project-delete cascade. Raises on failure so the caller
    keeps its ``deleting`` marker and retries — see the module docstring.

    Args:
        project_id: Project being deleted.

    Returns:
        The number of objects deleted (0 in dev mode or when none exist).
    """
    store = _store()
    if store is None:
        return 0

    prefix = f"{_REPORT_PREFIX}/{project_id}/"

    def _delete() -> int:
        client = store.s3
        paginator = client.get_paginator("list_objects_v2")
        removed = 0
        for page in paginator.paginate(Bucket=store.bucket, Prefix=prefix):
            keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if not keys:
                continue
            client.delete_objects(Bucket=store.bucket, Delete={"Objects": keys})
            removed += len(keys)
        return removed

    removed = await asyncio.to_thread(_delete)
    if removed:
        logger.info(
            "[report-snapshot] project=%s deleted %d report object(s) from S3",
            project_id[:8], removed,
        )
    return removed


__all__ = [
    "delete_project_reports_from_s3",
    "report_key_for_project",
    "upload_report_to_s3",
]
