# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Working-tree snapshot pack/unpack with path-traversal defense.

Snapshots the exact project working tree at dispatch (prompts, src/,
run_eval.py, config, dataset) to S3 under the tenant prefix; the worker
pulls that snapshot. The snapshot bundles the dataset dirs (data/raw,
data/processed, data/synthetic) so the worker harness can read its data
file relative to the unpacked working tree; transient data/uploads is
excluded. Harness deps are baked into the image (not snapshotted).

Path-traversal defense: ``unpack_snapshot`` validates every tar member
before extraction — rejects absolute paths, parent refs (``..``),
symlinks/hardlinks/devices, and any member whose resolved target escapes
the dest dir. This is attacker-adjacent input (user-uploaded / agent-written
filenames crossing a code-execution boundary on the worker).

Which bucket and which key to use is decided by the callers —
``FargateTaskCompute`` on the launch side and the worker entrypoint on the
other — and passed in. ``pack_working_tree`` and ``unpack_snapshot`` are
pure local tar operations, unit-testable without S3; ``upload_snapshot_to_s3``
and ``download_and_unpack_snapshot`` are thin async wrappers that add the S3
transfer around them.
"""
from __future__ import annotations

import tarfile
from pathlib import Path


class SnapshotTraversalError(ValueError):
    """Raised when a tar member would escape the dest directory.

    Subclasses ``ValueError`` so existing callers that catch ``ValueError``
    continue to work. Mirrors ``ProjectTraversalError`` from
    ``project_paths.py``.
    """


class SnapshotTooLargeError(ValueError):
    """Raised when an archive would exceed a resource ceiling on extraction.

    Separate from :class:`SnapshotTraversalError` because the two say different
    things to an operator: traversal means the archive tried to write outside its
    destination, this means it was honest about where but not about how much.
    Both subclass ``ValueError`` for the same caller-compatibility reason.
    """


#: Ceilings applied to an archive as it is extracted.
#:
#: Path validation answers "could this write outside dest_dir". It says nothing
#: about volume, so a well-formed archive could still fill the worker's disk:
#: gzip reaches roughly 1000:1 on repetitive input, which turns a 10 MB snapshot
#: into 10 GB of writes. These are the limits that make that a rejection instead
#: of an outage.
#:
#: Sized against what a real snapshot holds — prompts, ``src/``, a harness and a
#: dataset — with a wide margin, because the failure mode of a limit set too
#: tight is a legitimate eval that cannot dispatch. Raise them deliberately if a
#: real project ever trips one; do not remove them.
_MAX_MEMBERS = 20_000
_MAX_TOTAL_UNCOMPRESSED_BYTES = 2 * 1024**3  # 2 GiB across all members
_MAX_MEMBER_BYTES = 512 * 1024**2  # 512 MiB for any single member
_MAX_PATH_DEPTH = 32
_MAX_PATH_LENGTH = 1024


# Files/dirs included in the snapshot (small, fast-changing tree + dataset)
# Dataset dirs (raw/processed/synthetic) are bundled so the worker harness
# can read its data file relative to the unpacked working tree. Transient
# data/uploads (upload staging) is excluded. Harness deps are baked into
# the image (not snapshotted); output/ not needed.
_SNAPSHOT_INCLUDE = (
    "prompts",
    "src",
    "run_eval.py",
    "pyproject.toml",
    "README.md",
    "config.json",
    "data/raw",
    "data/processed",
    "data/synthetic",
    # reference material the harness may legitimately read at runtime.
    # Measured failure: a generated deterministic validator read
    # docs/column-definitions.csv relative to cwd — present on EFS,
    # absent in the worker workspace — and every case died on
    # FileNotFoundError while the run reported 0%. The harness working
    # tree IS the contract; a dir the agent can see while writing code
    # must exist where that code runs. Reference docs are small
    # (user-uploaded specs, not datasets).
    "docs",
)


def pack_working_tree(project_dir: Path, tar_path: Path) -> None:
    """Pack the working tree to a tar.gz snapshot (include-set + dataset).

    Packs the small, fast-changing tree (prompts, src/, harness, config)
    plus the dataset dirs (data/raw, data/processed, data/synthetic) so the
    worker harness can read its data file relative to the unpacked working
    tree. Transient data/uploads (upload staging) is excluded. Harness deps
    are baked into the image (not snapshotted); output/ not needed.

    WHY the include-set is minimal: cheap snapshot (fast pack/upload,
    small S3 object, fast download/unpack) and immutability (snapshot
    taken at dispatch cannot be mutated by agent mid-run).

    WHY dataset dirs are bundled: the worker harness runs at
    /tmp/workspace/{job_id} and expects to find its data file relative
    to that cwd (e.g., data/raw/<file>.csv). Large-dataset snapshot
    bloat (multi-hundred-MB datasets inflating every dispatch
    pack/upload) is a known, deferred scale-path concern.

    Args:
        project_dir: Project directory to snapshot.
        tar_path: Destination tar.gz path.
    """
    with tarfile.open(tar_path, "w:gz") as tar:
        for pattern in _SNAPSHOT_INCLUDE:
            path = project_dir / pattern
            if path.exists():
                # Add with arcname relative to project root (never absolute)
                tar.add(path, arcname=pattern)


def unpack_snapshot(tar_path: Path, dest_dir: Path) -> None:
    """Extract tarfile with path-traversal defense (Python 3.11).

    Validates every tar member before extraction:
    - Reject absolute paths (name starts with ``/``)
    - Reject parent directory references (any part is ``..``)
    - Reject symlinks, hardlinks, device files
    - Reject members whose resolved target escapes dest_dir

    Only after ALL members validate does extraction proceed. This is
    attacker-adjacent input — user-uploaded / agent-written filenames
    crossing a code-execution boundary on the worker.

    ``filter='data'`` is passed as well, so the stdlib re-checks each member as
    it is written. The manual loop stays because it is what raises
    ``SnapshotTraversalError`` with the offending member name — the stdlib
    filter raises its own exception types and names only the first refusal — and
    because a second check at write time covers anything between validation and
    extraction. The filter argument reached 3.11 in 3.11.4, which is the floor
    ``requires-python`` states.

    Resource ceilings are checked in the same pass — member count, total and
    per-member uncompressed size, path depth and path length. Path validation
    alone cannot catch volume: a well-formed archive that expands 1000:1 fills
    the worker's disk without ever naming a path outside ``dest_dir``.

    Args:
        tar_path: Path to the tar.gz snapshot.
        dest_dir: Destination directory (must exist).

    Raises:
        SnapshotTraversalError: If any member would escape dest_dir.
        SnapshotTooLargeError: If the archive exceeds a ceiling above.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tar:
        total_bytes = 0
        member_count = 0
        # Validate ALL members before extracting ANY — walking the archive one
        # header at a time, not through ``getmembers()``. That call reads the WHOLE
        # archive before returning anything: on a gzip stream it decompresses
        # every member's data to reach the next header, and keeps a ``TarInfo``
        # per member. The ceilings below would then run only after the cost they
        # exist to bound had been paid. Iterating lets the first refusal stop the
        # read; ``extractall`` below reuses the members this loop already loaded.
        for member in tar:
            member_count += 1
            if member_count > _MAX_MEMBERS:
                raise SnapshotTooLargeError(
                    f"Archive has more than {_MAX_MEMBERS} members"
                )

            # ``member.size`` is the size the archive CLAIMS. It is checked
            # before extracting rather than measured after, which is the whole
            # point: the cost of a lie is bounded by refusing to start. A member
            # whose real size exceeds its header is caught by the stdlib filter.
            if member.size > _MAX_MEMBER_BYTES:
                raise SnapshotTooLargeError(
                    f"Member exceeds {_MAX_MEMBER_BYTES} bytes: "
                    f"{member.name} claims {member.size}"
                )
            total_bytes += member.size
            if total_bytes > _MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise SnapshotTooLargeError(
                    f"Archive expands past {_MAX_TOTAL_UNCOMPRESSED_BYTES} "
                    f"bytes (reached {total_bytes} at {member.name})"
                )

            if len(member.name) > _MAX_PATH_LENGTH:
                raise SnapshotTooLargeError(
                    f"Member path longer than {_MAX_PATH_LENGTH} characters: "
                    f"{member.name[:80]}..."
                )
            if len(Path(member.name).parts) > _MAX_PATH_DEPTH:
                raise SnapshotTooLargeError(
                    f"Member nested deeper than {_MAX_PATH_DEPTH}: {member.name}"
                )

            # Reject absolute paths
            if member.name.startswith("/"):
                raise SnapshotTraversalError(
                    f"Absolute path rejected: {member.name}"
                )

            # Reject parent directory references
            if ".." in Path(member.name).parts:
                raise SnapshotTraversalError(
                    f"Parent ref (..) rejected: {member.name}"
                )

            # Reject symlinks, hardlinks, device files
            if member.issym() or member.islnk() or member.isdev():
                raise SnapshotTraversalError(
                    f"Non-regular file (symlink/device) rejected: {member.name}"
                )

            # Verify resolved path stays under dest_dir
            # (catches escape attempts like "foo/../../etc/passwd")
            target = (dest_dir / member.name).resolve()
            if not target.is_relative_to(dest_dir.resolve()):
                raise SnapshotTraversalError(
                    f"Escape attempt: {member.name} resolves to {target}"
                )

        # All members validated — safe to extract
        tar.extractall(dest_dir, filter="data")


async def upload_snapshot_to_s3(
    project_dir: Path, s3_client, bucket: str, key: str
) -> int:
    """Pack working tree and upload to S3 (async orchestration wrapper).

    Thin async wrapper around ``pack_working_tree`` + boto3 S3 upload. Packs
    to a temp ``tar.gz``, uploads via ``s3_client.put_object``, logs the
    snapshot byte size, and cleans up the temp file in a ``finally`` block.

    WHY the key must live under the tenant canonical prefix: the worker's
    minted STS-scoped credentials are scoped to
    ``project_key_for_directory(cwd)`` prefix — a key outside that prefix
    will be denied by IAM.

    Args:
        project_dir: Project directory to snapshot.
        s3_client: Low-level boto3 S3 client (injected for testing).
        bucket: S3 bucket name.
        key: S3 key for the snapshot (MUST be under tenant canonical prefix).

    Returns:
        Snapshot size in bytes.
    """
    import asyncio
    import logging
    import tempfile

    logger = logging.getLogger(__name__)

    def _pack_and_upload() -> int:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_tar:
            # Nothing is written inside this block: it reserves a name, and the
            # closed file is what pack_working_tree below writes into.
            # nosemgrep: tempfile-without-flush
            tmp_path = Path(tmp_tar.name)

        try:
            # Pack to temp tar.gz
            pack_working_tree(project_dir, tmp_path)
            size = tmp_path.stat().st_size

            # Upload to S3
            with open(tmp_path, "rb") as f:
                s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=f,
                )

            logger.info(
                "Uploaded snapshot to S3",
                extra={"key": key, "size_bytes": size},
            )

            return size
        finally:
            # Clean up temp file
            if tmp_path.exists():
                tmp_path.unlink()

    return await asyncio.to_thread(_pack_and_upload)


async def download_and_unpack_snapshot(
    s3_client, bucket: str, key: str, dest_dir: Path
) -> None:
    """Download S3 snapshot and unpack to dest_dir (async orchestration wrapper).

    Thin async wrapper around boto3 S3 download + ``unpack_snapshot``. Downloads
    to a temp ``tar.gz``, calls the traversal-safe unpack, and
    cleans up the temp file in a ``finally`` block.

    WHY the key must live under the tenant canonical prefix: the worker's
    minted STS-scoped credentials are scoped to
    ``project_key_for_directory(cwd)`` prefix — a key outside that prefix
    will be denied by IAM.

    Args:
        s3_client: Low-level boto3 S3 client (injected for testing).
        bucket: S3 bucket name.
        key: S3 key for the snapshot (MUST be under tenant canonical prefix).
        dest_dir: Destination directory (will be created if missing).

    Raises:
        SnapshotTraversalError: If any tar member would escape dest_dir.
    """
    import asyncio
    import tempfile

    def _download_and_unpack() -> None:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_tar:
            # Nothing is written inside this block: it reserves a name, and the
            # closed file is what the S3 download below writes into.
            # nosemgrep: tempfile-without-flush
            tmp_path = Path(tmp_tar.name)

        try:
            # Download from S3
            s3_client.download_file(bucket, key, str(tmp_path))

            # Unpack (traversal-safe)
            unpack_snapshot(tmp_path, dest_dir)
        finally:
            # Clean up temp file
            if tmp_path.exists():
                tmp_path.unlink()

    await asyncio.to_thread(_download_and_unpack)


__all__ = [
    "SnapshotTooLargeError",
    "SnapshotTraversalError",
    "pack_working_tree",
    "unpack_snapshot",
    "upload_snapshot_to_s3",
    "download_and_unpack_snapshot",
]
