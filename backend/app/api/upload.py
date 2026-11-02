# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""File upload API endpoint for project resources.

Accepts one or more files per request. Zips are unpacked into
``data/uploads/<stem>/`` and the original archive is preserved at
``data/uploads/<name>.zip``; loose files land directly under
``data/uploads/``. The ``data-analyzer`` subagent later classifies each
file and moves it into ``data/raw/`` or ``docs/``.

Large uploads (≥50 MB) use a two-call presigned-PUT flow: the browser
PUTs directly to an S3 staging bucket, then the backend copies into
the EFS project tree. Avoids streaming gigabytes through the
coordinator process and unblocks 1 GB+ datasets.
"""
from __future__ import annotations

import io
import logging
import os
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from app.models.schemas import (
    PresignedUploadCompleteRequest,
    PresignedUploadInitRequest,
    PresignedUploadInitResponse,
    UploadResponse,
)
from app.platform.api_authz import require_project_write
from app.platform.types import UserId

router = APIRouter(prefix="/api/projects", tags=["upload"])

logger = logging.getLogger(__name__)

# Maximum cumulative upload size per request (100 MB). Uncompressed zip
# contents must also fit within MAX_UNPACK_SIZE to prevent zip bombs.
MAX_UPLOAD_SIZE = 100 * 1024 * 1024
MAX_UNPACK_SIZE = 5 * MAX_UPLOAD_SIZE

# Per-upload cap on the presigned-PUT path. Bigger than the inline
# multi-file cap because the network bytes go straight to S3 — the
# coordinator process never holds the full payload in memory.
MAX_PRESIGNED_UPLOAD_SIZE = 5 * 1024 * 1024 * 1024  # 5 GB

# Presigned URL TTL. Long enough for a 1 GB upload on a slow line; short
# enough that a leaked URL doesn't matter for very long.
PRESIGNED_URL_TTL_SEC = 15 * 60

# OS-level noise we strip while extracting zips: macOS resource forks
# (``__MACOSX/*`` and ``._*``) and Finder metadata (``.DS_Store``).
_FILTERED_NAMES = {".DS_Store"}


def _is_filtered_entry(name: str) -> bool:
    """Return ``True`` if a zip entry should be silently skipped.

    Covers macOS resource forks (``__MACOSX/...`` and ``._*``) and Finder
    metadata (``.DS_Store``) so project trees stay clean.

    Args:
        name: The ``ZipInfo.filename`` of the candidate entry.

    Returns:
        True when the entry is OS detritus we never want on disk.
    """
    parts = name.replace("\\", "/").split("/")
    for part in parts:
        if part == "__MACOSX":
            return True
        if part in _FILTERED_NAMES:
            return True
        if part.startswith("._"):
            return True
    return False


def _unique_dirname(parent: Path, stem: str) -> Path:
    """Return a sibling-safe extraction path under ``parent``.

    If ``parent/stem`` already exists, append ``_2``, ``_3``, ... until
    we find an unused name. Used so re-uploading the same filename
    doesn't clobber the first extraction.

    Args:
        parent: Directory the extraction target will live under.
        stem: Desired folder name (typically the zip's filename stem).

    Returns:
        A path of the form ``parent/<stem>`` or ``parent/<stem>_N``.
    """
    candidate = parent / stem
    n = 2
    while candidate.exists():
        candidate = parent / f"{stem}_{n}"
        n += 1
    return candidate


def _unique_filepath(parent: Path, filename: str) -> Path:
    """Return a sibling-safe file path under ``parent``.

    Used for both the raw zip archive and loose-file uploads so that
    multiple uploads of the same filename don't overwrite each other.

    Args:
        parent: Directory the file will live under.
        filename: Desired filename (preserved verbatim on first write).

    Returns:
        A path of the form ``parent/<filename>`` or
        ``parent/<stem>_N<suffix>``.
    """
    candidate = parent / filename
    if not candidate.exists():
        return candidate
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    n = 2
    while True:
        candidate = parent / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def _extract_zip(
    content: bytes,
    extraction_dir: Path,
    remaining_budget: int,
) -> int:
    """Extract ``content`` into ``extraction_dir`` with safety guards.

    Enforces zip-slip, symlink, and zip-bomb protections. Skips entries
    flagged by :func:`_is_filtered_entry` so OS detritus never reaches
    disk.

    Args:
        content: Raw zip bytes.
        extraction_dir: Target directory (created by the caller).
        remaining_budget: Maximum additional uncompressed bytes permitted.

    Returns:
        Bytes written to disk (excluding filtered entries).

    Raises:
        HTTPException: 400 on malformed zip or path escape, 413 on size
            overrun.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip archive")

    # Budget is measured against the *retained* entries only — filtered
    # OS detritus doesn't count against the user's quota.
    retained_total = sum(
        max(zi.file_size, 0)
        for zi in zf.infolist()
        if not zi.is_dir() and not _is_filtered_entry(zi.filename)
    )
    if retained_total > remaining_budget:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Zip uncompresses to {retained_total} bytes, exceeding "
                f"remaining budget of {remaining_budget} bytes."
            ),
        )

    extraction_resolved = extraction_dir.resolve()
    written = 0
    for info in zf.infolist():
        # Skip directory entries; the tree is recreated via parents below.
        if info.is_dir():
            continue
        if _is_filtered_entry(info.filename):
            continue
        # Skip symlinks / non-regular entries to avoid pointer abuse.
        # External attrs high 16 bits encode Unix file mode; the top nibble
        # is the S_IFMT type (0xA000 == symlink, 0x8000 == regular file,
        # 0x4000 == dir, others are devices/sockets/FIFOs).
        #
        # Many archivers (including Python's ``zipfile.writestr``) emit
        # regular files with the permission bits set but NO S_IFMT type
        # encoded. When ``mode & S_IFMT == 0`` we treat it as "unspecified"
        # and accept — combined with the ``is_relative_to`` check below,
        # the worst case is writing a regular file inside the extraction
        # dir. We only explicitly reject when S_IFMT is present AND not a
        # regular file / directory (so symlinks, devices, FIFOs go away).
        mode = info.external_attr >> 16
        ifmt = mode & 0o170000
        if ifmt and ifmt not in (0o100000, 0o040000):
            continue
        target = (extraction_dir / info.filename).resolve()
        if not target.is_relative_to(extraction_resolved):
            raise HTTPException(
                status_code=400,
                detail=f"Zip entry escapes extraction dir: {info.filename!r}",
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as dst:
            chunk = src.read()
            dst.write(chunk)
            written += len(chunk)
    return written


@router.post("/{project_id}/upload", response_model=UploadResponse)
async def upload_files(
    project_id: str,
    files: list[UploadFile] = File(...),
    _user: UserId = Depends(require_project_write),
) -> UploadResponse:
    """Upload one or more files into ``data/uploads/``.

    Zips are preserved verbatim and unpacked into a sibling directory
    named after the archive stem (with ``_2``, ``_3`` suffixes on
    collisions). Loose files land directly under ``data/uploads/``. The
    ``data-analyzer`` subagent sorts the resulting tree into
    ``data/raw/`` or ``docs/``.

    Args:
        project_id: Project UUID.
        files: One or more multipart file parts.

    Returns:
        UploadResponse: written paths (project-relative) and total bytes.

    Raises:
        HTTPException: 404 if project is missing, 413 on size overrun,
            400 on invalid filename or malformed zip.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    from app.main import get_project_manager
    from app.services.project_manager import get_project_dir_or_404
    pm = get_project_manager()

    project = pm.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # Cloud-mode-safe project dir (owner-prefixed under EFS).
    project_root = get_project_dir_or_404(project_id)
    upload_root = project_root / "data" / "uploads"
    upload_root.mkdir(parents=True, exist_ok=True)

    written_total = 0
    written_paths: list[str] = []
    project_root = project_root.resolve()

    for file in files:
        safe_name = Path(file.filename).name if file.filename else "upload"
        if not safe_name or safe_name in (".", ".."):
            raise HTTPException(status_code=400, detail="Invalid filename")

        content = await file.read()
        if written_total + len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Cumulative upload size exceeds {MAX_UPLOAD_SIZE} bytes."
                ),
            )

        if safe_name.lower().endswith(".zip"):
            # Preserve the archive verbatim so the user (or the agent)
            # can re-extract later as a reset path.
            archive_path = _unique_filepath(upload_root, safe_name)
            archive_path.write_bytes(content)
            written_total += len(content)
            written_paths.append(
                str(archive_path.relative_to(project_root))
            )

            extraction_dir = _unique_dirname(upload_root, archive_path.stem)
            extraction_dir.mkdir(parents=True, exist_ok=True)
            bytes_extracted = _extract_zip(
                content,
                extraction_dir,
                remaining_budget=MAX_UNPACK_SIZE - written_total,
            )
            written_total += bytes_extracted
            # If the archive was entirely filtered junk, drop the empty dir.
            if not any(extraction_dir.rglob("*")):
                extraction_dir.rmdir()
            else:
                for p in sorted(extraction_dir.rglob("*")):
                    if p.is_file():
                        written_paths.append(
                            str(p.relative_to(project_root))
                        )
        else:
            target = _unique_filepath(upload_root, safe_name)
            target.write_bytes(content)
            written_total += len(content)
            written_paths.append(str(target.relative_to(project_root)))

    return UploadResponse(
        files=written_paths,
        total_bytes=written_total,
    )


# ---------------------------------------------------------------------------
# Presigned-PUT upload flow (large uploads ≥50 MB)
# ---------------------------------------------------------------------------


def _staging_bucket() -> str | None:
    """Return the configured staging bucket, or None when the flow isn't
    available (dev mode, or cloud mode where the env var hasn't been
    provisioned). Caller should 503 if needed."""
    return os.environ.get("S3_UPLOADS_STAGING_BUCKET", "").strip() or None


def _staging_key(owner_id: str, project_id: str, upload_id: str, filename: str) -> str:
    """Compose the staging-bucket object key for a presigned upload.

    Layout matches the per-user prefix convention so a future
    bucket-policy fence can scope by ``users/{sub}/`` if we tighten it.
    """
    safe_name = Path(filename).name or "upload"
    return f"users/{owner_id}/projects/{project_id}/{upload_id}/{safe_name}"


@router.post(
    "/{project_id}/uploads/init",
    response_model=PresignedUploadInitResponse,
)
async def init_presigned_upload(
    project_id: str,
    req: PresignedUploadInitRequest,
    user_id: UserId = Depends(require_project_write),
) -> PresignedUploadInitResponse:
    """Mint a presigned PUT URL for ≥50 MB direct-to-S3 uploads.

    Browser PUTs directly to S3 staging using the returned URL, then
    calls ``/uploads/complete`` to trigger the backend copy into the
    project's EFS data directory. Avoids streaming large bodies through
    the coordinator.

    Args:
        project_id: Project UUID.
        req: Filename + optional size hint.
        user_id: Resolved by the auth middleware via the project-write
            chokepoint.

    Returns:
        PresignedUploadInitResponse with the URL and a TTL.

    Raises:
        HTTPException: 503 when staging isn't configured (dev, or a cloud
            deploy without ``S3_UPLOADS_STAGING_BUCKET``), 413 if the
            declared size exceeds the cap, 400 on invalid filename.
    """
    bucket = _staging_bucket()
    if bucket is None:
        raise HTTPException(
            status_code=503,
            detail="Presigned uploads are unavailable in this environment.",
        )

    safe_name = Path(req.filename).name
    if not safe_name or safe_name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if req.size_bytes <= 0:
        raise HTTPException(
            status_code=400,
            detail="size_bytes must be a positive integer.",
        )
    if req.size_bytes > MAX_PRESIGNED_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Declared size {req.size_bytes} exceeds maximum "
                f"{MAX_PRESIGNED_UPLOAD_SIZE} bytes."
            ),
        )

    # Use the requesting user as the owner_id for the staging prefix.
    # require_project_write already enforced access; the prefix scoping
    # is just a tidy-up so the staging bucket layout mirrors the EFS
    # tree.
    upload_id = str(uuid.uuid4())
    target_key = _staging_key(user_id, project_id, upload_id, safe_name)

    import boto3

    # Use the platform's task-role boto3. Staging-bucket writes are
    # constrained by the presigned URL's signature, which pins the exact
    # key, rather than by a per-request scoped session.
    s3 = boto3.client("s3")
    params = {
        "Bucket": bucket,
        "Key": target_key,
        # Sign ``Content-Length`` into the presigned URL: S3 rejects
        # any PUT whose body length doesn't match exactly, so an authed
        # attacker can't replay the URL with a giant body to fill the
        # staging bucket up to the 24h lifecycle window. The backend's
        # ``complete_presigned_upload`` size check stays as a defense-
        # in-depth guard for the unlikely case that S3 ever drops this
        # enforcement.
        "ContentLength": req.size_bytes,
    }
    if req.content_type:
        params["ContentType"] = req.content_type
    try:
        url = s3.generate_presigned_url(
            "put_object",
            Params=params,
            ExpiresIn=PRESIGNED_URL_TTL_SEC,
        )
    except Exception as e:  # noqa: BLE001 - AWS SDK errors are varied
        # Full exception (often carries bucket name + AWS error codes)
        # goes to CloudWatch via ``logger.exception``; the client sees
        # a generic message so we don't leak internal AWS detail.
        logger.exception("[upload-init] presign failed for project=%s", project_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate upload URL.",
        ) from e

    expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=PRESIGNED_URL_TTL_SEC)
    ).isoformat()

    return PresignedUploadInitResponse(
        upload_id=upload_id,
        presigned_url=url,
        target_key=target_key,
        expires_at=expires_at,
        max_size_bytes=MAX_PRESIGNED_UPLOAD_SIZE,
    )


@router.post(
    "/{project_id}/uploads/complete",
    response_model=UploadResponse,
)
async def complete_presigned_upload(
    project_id: str,
    req: PresignedUploadCompleteRequest,
    user_id: UserId = Depends(require_project_write),
) -> UploadResponse:
    """Stream a staged upload from S3 into the project's EFS data dir.

    Browser already PUT the bytes to S3 staging; this call moves the
    object into the project tree and deletes the staging object.

    Args:
        project_id: Project UUID.
        req: Upload ID + filename returned by ``/uploads/init``.
        user_id: Resolved by the project-write chokepoint.

    Returns:
        UploadResponse describing the file written to EFS.

    Raises:
        HTTPException: 503 if staging not configured, 404 if the staged
            object doesn't exist, 400 on bad filename, 413 if the
            staged object exceeds the cap.
    """
    bucket = _staging_bucket()
    if bucket is None:
        raise HTTPException(
            status_code=503,
            detail="Presigned uploads are unavailable in this environment.",
        )

    safe_name = Path(req.filename).name
    if not safe_name or safe_name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    from app.services.project_manager import get_project_dir_or_404
    project_root = get_project_dir_or_404(project_id).resolve()
    upload_root = project_root / "data" / "uploads"
    upload_root.mkdir(parents=True, exist_ok=True)

    staged_key = _staging_key(user_id, project_id, req.upload_id, safe_name)

    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3")

    # 1. Verify the staged object exists + check size.
    try:
        head = s3.head_object(Bucket=bucket, Key=staged_key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            raise HTTPException(
                status_code=404,
                detail="Staged upload not found — was the PUT completed?",
            ) from e
        logger.exception("[upload-complete] head_object failed: %s", staged_key)
        raise HTTPException(
            status_code=500,
            detail="Failed to verify staged upload.",
        ) from e

    size_bytes = int(head.get("ContentLength", 0))
    if size_bytes > MAX_PRESIGNED_UPLOAD_SIZE:
        # Defense in depth: the presigned URL pins Content-Length to a size
        # already checked against this cap, so this only fires if that
        # enforcement failed. Delete the oversized object rather than copy it.
        try:
            s3.delete_object(Bucket=bucket, Key=staged_key)
        except ClientError:
            pass
        raise HTTPException(
            status_code=413,
            detail=(
                f"Staged upload size {size_bytes} exceeds "
                f"{MAX_PRESIGNED_UPLOAD_SIZE} bytes."
            ),
        )

    # 2. Pick a non-colliding target path under data/uploads.
    target = _unique_filepath(upload_root, safe_name)

    # 3. Stream S3 → EFS. boto3's get_object body is a streaming reader;
    # write 8 MB chunks so we don't load gigabytes into memory.
    try:
        obj = s3.get_object(Bucket=bucket, Key=staged_key)
        body = obj["Body"]
        with open(target, "wb") as out:
            while True:
                chunk = body.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except (ClientError, OSError) as e:
        logger.exception(
            "[upload-complete] copy S3→EFS failed: %s → %s", staged_key, target
        )
        # Best-effort cleanup of partial EFS file.
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        # Generic client message — exception detail (often carries
        # bucket name + EFS path) stays in CloudWatch via the
        # ``logger.exception`` above.
        raise HTTPException(
            status_code=500,
            detail="Failed to finalize upload.",
        ) from e

    # 4. Delete the staging object so it doesn't linger past the 24h TTL.
    try:
        s3.delete_object(Bucket=bucket, Key=staged_key)
    except ClientError:
        # Lifecycle rule will sweep it; not an error worth surfacing.
        logger.info(
            "[upload-complete] staging delete failed for %s; lifecycle will sweep",
            staged_key,
        )

    return UploadResponse(
        files=[str(target.relative_to(project_root))],
        total_bytes=size_bytes,
    )
