# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic data batch review endpoints.

REST surface the frontend Synthesis tab consumes. These endpoints are
intentionally separate from the agent-facing MCP tools in
``services/mcp_tools/synthesis.py`` — the UI needs enriched responses
(rows zipped with CSV values) that the agent doesn't.

All file I/O goes through ``safe_path`` so no ``batch_id`` supplied by
the client can escape ``data/synthetic/``.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.platform.api_authz import require_project_read, require_project_write
from app.platform.types import UserId
from app.models.schemas import (
    SyntheticApproveRequest,
    SyntheticBatchDetail,
    SyntheticBatchStatusResponse,
    SyntheticBatchSummary,
    SyntheticCase,
    SyntheticRejectRequest,
)
from app.services.project_paths import safe_path

from app.services.project_manager import get_project_dir_or_404

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["synthetic"])


def _synthetic_dir(project_dir: Path) -> Path:
    """Return the project's ``data/synthetic/`` directory (not created)."""
    return project_dir / "data" / "synthetic"


def _load_manifest(synthetic_dir: Path, batch_id: str) -> dict:
    """Load and validate a batch manifest, raising 404 on anything missing.

    ``batch_id`` is untrusted user input — we route it through
    ``safe_path`` so it can't escape the synthetic dir via ``../`` etc.

    Args:
        synthetic_dir: Project-scoped ``data/synthetic/`` directory.
        batch_id: Timestamped batch identifier from the URL.

    Returns:
        Parsed manifest dict.

    Raises:
        HTTPException 404: Manifest is missing or batch_id is invalid.
        HTTPException 500: Manifest exists but fails to parse.
    """
    manifest_path = safe_path(synthetic_dir, f"{batch_id}.manifest.json")
    if manifest_path is None or not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")
    try:
        return json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        # Full parser detail (line/column, file content snippet) stays
        # in CloudWatch — the client gets a generic message.
        logger.exception("[synthetic] corrupt manifest at %s", manifest_path)
        raise HTTPException(
            status_code=500,
            detail="Failed to read synthetic manifest.",
        ) from exc


def _write_manifest(synthetic_dir: Path, batch_id: str, manifest: dict) -> None:
    """Persist a batch manifest, re-validating the path on write."""
    manifest_path = safe_path(synthetic_dir, f"{batch_id}.manifest.json")
    if manifest_path is None:
        # Should be unreachable — _load_manifest rejected the same id already.
        raise HTTPException(status_code=400, detail=f"Invalid batch_id: {batch_id}")
    manifest_path.write_text(json.dumps(manifest, indent=2))


def _count_statuses(cases: list[dict]) -> dict[str, int]:
    """Aggregate per-status counts from a manifest ``cases`` array."""
    counts = {"pending": 0, "approved": 0, "rejected": 0}
    for case in cases:
        status = case.get("status", "pending")
        if status in counts:
            counts[status] += 1
    return counts


def _read_csv_rows(synthetic_dir: Path, csv_name: str) -> tuple[list[str], list[dict[str, str]]]:
    """Read the companion CSV for a batch and return columns + rows.

    Returns empty (``[]``, ``[]``) if the CSV is missing so the UI can
    still render manifest metadata when the CSV was deleted out-of-band.
    """
    csv_path = safe_path(synthetic_dir, csv_name)
    if csv_path is None or not csv_path.exists():
        return [], []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = [{col: (row.get(col) or "") for col in columns} for row in reader]
    return columns, rows


@router.get("/{project_id}/synthetic/batches", response_model=list[SyntheticBatchSummary])
async def list_synthetic_batches(
    project_id: str,
    _user: UserId = Depends(require_project_read),
) -> list[SyntheticBatchSummary]:
    """List every synthetic batch for a project, newest first.

    Reads each ``*.manifest.json`` under ``data/synthetic/`` and returns
    a summary that includes the counts the left-pane batch list needs.
    Returns an empty list when the synthetic dir doesn't exist yet.

    Args:
        project_id: Project UUID from the URL.

    Returns:
        Summaries sorted by ``timestamp`` descending.
    """
    project_dir = get_project_dir_or_404(project_id)
    synthetic_dir = _synthetic_dir(project_dir)
    if not synthetic_dir.exists():
        return []

    summaries: list[SyntheticBatchSummary] = []
    for manifest_path in synthetic_dir.glob("*.manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            # One bad file shouldn't block the whole list; the agent-facing
            # MCP tool does the same thing.
            logger.warning("[synthetic-api] skipping %s: %s", manifest_path.name, exc)
            continue

        cases = manifest.get("cases") or []
        counts = _count_statuses(cases)

        summaries.append(
            SyntheticBatchSummary(
                batch_id=manifest.get("batch_id", ""),
                strategy=manifest.get("strategy", ""),
                timestamp=manifest.get("timestamp", ""),
                seed_source=manifest.get("seed_source"),
                seed_run_id=manifest.get("seed_run_id"),
                total_cases=manifest.get("total_cases", len(cases)),
                pending=counts["pending"],
                approved=counts["approved"],
                rejected=counts["rejected"],
                csv_file=manifest.get("csv_file", ""),
            )
        )

    # Newest first — matches the eval-runs list contract. Timestamps are
    # filesystem-safe strings, so lexicographic sort is correct.
    summaries.sort(key=lambda s: s.timestamp, reverse=True)
    return summaries


@router.get(
    "/{project_id}/synthetic/batches/{batch_id}",
    response_model=SyntheticBatchDetail,
)
async def get_synthetic_batch(
    project_id: str,
    batch_id: str,
    _user: UserId = Depends(require_project_read),
) -> SyntheticBatchDetail:
    """Return a batch manifest plus its per-case rows from the CSV.

    The CSV is read each request rather than cached — batches are small
    and an agent may rewrite the file.

    Args:
        project_id: Project UUID from the URL.
        batch_id: Timestamped batch identifier (manifest filename stem).

    Returns:
        Full batch manifest with ``rows`` enriched from the CSV.

    Raises:
        HTTPException 404: Batch is missing.
    """
    project_dir = get_project_dir_or_404(project_id)
    synthetic_dir = _synthetic_dir(project_dir)
    manifest = _load_manifest(synthetic_dir, batch_id)

    cases = manifest.get("cases") or []
    csv_name = manifest.get("csv_file", "")
    _, csv_rows = _read_csv_rows(synthetic_dir, csv_name)

    # Zip manifest cases with CSV rows by index so the UI can render each
    # card in one pass. If the CSV has fewer rows than the manifest (e.g.
    # truncated), fall back to empty values for the missing rows.
    rows: list[SyntheticCase] = []
    for case in cases:
        idx = int(case.get("index", 0))
        values = csv_rows[idx] if 0 <= idx < len(csv_rows) else {}
        rows.append(
            SyntheticCase(
                index=idx,
                status=case.get("status", "pending"),
                values=values,
            )
        )

    return SyntheticBatchDetail(
        batch_id=manifest.get("batch_id", batch_id),
        timestamp=manifest.get("timestamp", ""),
        strategy=manifest.get("strategy", ""),
        seed_source=manifest.get("seed_source"),
        seed_run_id=manifest.get("seed_run_id"),
        total_cases=manifest.get("total_cases", len(cases)),
        field_mappings=manifest.get("field_mappings") or {},
        csv_file=csv_name,
        rows=rows,
    )


@router.post(
    "/{project_id}/synthetic/batches/{batch_id}/approve",
    response_model=SyntheticBatchStatusResponse,
)
async def approve_synthetic_batch(
    project_id: str,
    batch_id: str,
    body: SyntheticApproveRequest,
    _user: UserId = Depends(require_project_write),
) -> SyntheticBatchStatusResponse:
    """Approve individual cases or every pending case in a batch.

    Mirrors the semantics of ``approve_synthetic_cases`` (MCP tool): for
    ``approve_all=True``, only cases currently in ``pending`` flip — we
    never silently undo a prior rejection. Explicit ``indices`` always
    override current status.

    Args:
        project_id: Project UUID from the URL.
        batch_id: Batch identifier from the URL.
        body: Either a non-empty ``indices`` list or ``approve_all=True``.

    Returns:
        Updated counts + the raw cases array for optimistic reconcile.

    Raises:
        HTTPException 400: Neither indices nor approve_all was supplied.
        HTTPException 404: Batch is missing.
    """
    indices = body.indices or []
    if not indices and not body.approve_all:
        raise HTTPException(
            status_code=400,
            detail="Provide a non-empty 'indices' list or 'approve_all=true'.",
        )

    project_dir = get_project_dir_or_404(project_id)
    synthetic_dir = _synthetic_dir(project_dir)
    manifest = _load_manifest(synthetic_dir, batch_id)

    cases = manifest.get("cases") or []
    approve_set = set(indices)

    for case in cases:
        idx = case.get("index")
        if body.approve_all and case.get("status") == "pending":
            case["status"] = "approved"
        if idx in approve_set:
            case["status"] = "approved"

    manifest["cases"] = cases
    _write_manifest(synthetic_dir, batch_id, manifest)

    counts = _count_statuses(cases)
    return SyntheticBatchStatusResponse(
        batch_id=batch_id,
        pending=counts["pending"],
        approved=counts["approved"],
        rejected=counts["rejected"],
        cases=cases,
    )


@router.post(
    "/{project_id}/synthetic/batches/{batch_id}/reject",
    response_model=SyntheticBatchStatusResponse,
)
async def reject_synthetic_batch(
    project_id: str,
    batch_id: str,
    body: SyntheticRejectRequest,
    _user: UserId = Depends(require_project_write),
) -> SyntheticBatchStatusResponse:
    """Reject one or more cases by index.

    Args:
        project_id: Project UUID from the URL.
        batch_id: Batch identifier from the URL.
        body: Non-empty ``indices`` list to reject.

    Returns:
        Updated counts + the raw cases array.

    Raises:
        HTTPException 400: ``indices`` is empty.
        HTTPException 404: Batch is missing.
    """
    if not body.indices:
        raise HTTPException(
            status_code=400, detail="Provide a non-empty 'indices' list."
        )

    project_dir = get_project_dir_or_404(project_id)
    synthetic_dir = _synthetic_dir(project_dir)
    manifest = _load_manifest(synthetic_dir, batch_id)

    cases = manifest.get("cases") or []
    reject_set = set(body.indices)

    for case in cases:
        if case.get("index") in reject_set:
            case["status"] = "rejected"

    manifest["cases"] = cases
    _write_manifest(synthetic_dir, batch_id, manifest)

    counts = _count_statuses(cases)
    return SyntheticBatchStatusResponse(
        batch_id=batch_id,
        pending=counts["pending"],
        approved=counts["approved"],
        rejected=counts["rejected"],
        cases=cases,
    )
