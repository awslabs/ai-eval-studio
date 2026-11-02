# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP tools for inspecting and organizing uploaded files.

Uploads land in ``projects/{id}/data/uploads/`` — zip archives are
preserved verbatim alongside their extracted sibling directories, and
loose files sit at the top level. The agent calls ``list_upload`` to see
what's there, ``move_file`` to promote each file into ``data/raw/`` or
``docs/``, and ``finalize_upload`` to clean up once sorting is done.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from claude_agent_sdk import ToolAnnotations, tool

from app.services.mcp_response import mcp_error, mcp_json
from app.services.project_paths import resolve_project_dir

logger = logging.getLogger(__name__)

# Destinations the agent is allowed to move files into. Kept intentionally
# narrow — each entry is a project-relative path that exists (or will be
# created) under the project root.
ALLOWED_DEST_SUBDIRS = {"data/raw", "data/processed", "docs", "prompts"}

# Extension sniffer for the suggested_dest hint. Deliberately small — the
# AI makes the final call; this is just a starting point.
_TABULAR_EXTS = {".csv", ".json", ".jsonl", ".xlsx", ".xls", ".parquet", ".tsv"}
_DOCUMENT_EXTS = {".md", ".txt", ".pdf", ".rst", ".docx", ".html"}
_CODE_EXTS = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java"}


def _classify(path: Path) -> tuple[str, str | None]:
    """Return ``(kind, suggested_dest)`` for a file based on its extension.

    Args:
        path: File path to classify (only the suffix is read).

    Returns:
        Tuple of kind label and project-relative destination hint, or
        ``(kind, None)`` when the extension is unknown.
    """
    ext = path.suffix.lower()
    if ext in _TABULAR_EXTS:
        return "tabular", "data/raw"
    if ext in _DOCUMENT_EXTS:
        return "document", "docs"
    if ext in _CODE_EXTS:
        return "code", "docs"
    return "unknown", None


def _upload_root(project_dir: Path) -> Path:
    """Return the project's upload landing directory.

    The folder is lazily created by the upload API and removed by
    :func:`finalize_upload` when it ends up empty, so callers must handle
    its absence gracefully.
    """
    return project_dir / "data" / "uploads"


@tool(
    "list_upload",
    "List every file currently under data/uploads/ for a project, each with a "
    "suggested destination under data/raw/ or docs/. Use this after a user "
    "uploads files (or when they ask to analyze their data) to decide where each "
    "file should go. Zip archives surface as `.zip` entries alongside their "
    "extracted contents; move the extracted files, not the archive.",
    {"project_id": str},
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_upload(args: dict[str, Any]) -> dict[str, Any]:
    """Return the tree under ``data/uploads/`` for classification.

    Args:
        args: Dict with ``project_id``.

    Returns:
        MCP response with ``files``: list of ``{relpath, size, kind,
        suggested_dest}`` entries. ``relpath`` is relative to
        ``data/uploads/``. An empty list is returned when the upload
        directory does not exist.
    """
    try:
        project_dir = resolve_project_dir(args["project_id"])
    except ValueError as exc:
        return mcp_error(str(exc))

    upload_root = _upload_root(project_dir)
    if not upload_root.is_dir():
        return mcp_json({"files": []})

    upload_resolved = upload_root.resolve()
    files: list[dict[str, Any]] = []
    for entry in sorted(upload_root.rglob("*")):
        if not entry.is_file():
            continue
        # Defence in depth against unexpected symlinks escaping the root.
        if not entry.resolve().is_relative_to(upload_resolved):
            continue
        kind, suggested = _classify(entry)
        files.append({
            "relpath": str(entry.relative_to(upload_root)),
            "size": entry.stat().st_size,
            "kind": kind,
            "suggested_dest": suggested,
        })

    logger.info(
        "[list_upload] project=%s files=%d",
        args["project_id"][:12], len(files),
    )
    return mcp_json({"files": files})


@tool(
    "move_file",
    "Move one file from data/uploads/ into its final destination "
    "(data/raw, data/processed, docs, or prompts). Call once per file after "
    "classifying. Do NOT move the .zip archive itself — move the files that "
    "were extracted from it.",
    {
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "src_relpath": {
                "type": "string",
                "description": "Path relative to data/uploads/, including any subdirectory",
            },
            "dest_subdir": {
                "type": "string",
                "description": f"One of: {sorted(ALLOWED_DEST_SUBDIRS)}",
            },
            "dest_filename": {
                "type": "string",
                "description": "Optional rename; defaults to the source basename",
            },
        },
        "required": ["project_id", "src_relpath", "dest_subdir"],
    },
)
async def move_file(args: dict[str, Any]) -> dict[str, Any]:
    """Move a staged file into an allowlisted project subdirectory.

    Args:
        args: Dict with ``project_id``, ``src_relpath``, ``dest_subdir``,
            and optional ``dest_filename``.

    Returns:
        MCP response with ``{from, to, size}`` on success or an error.
    """
    dest_subdir = args.get("dest_subdir", "")
    if dest_subdir not in ALLOWED_DEST_SUBDIRS:
        return mcp_error(
            f"dest_subdir must be one of {sorted(ALLOWED_DEST_SUBDIRS)}; got {dest_subdir!r}"
        )

    try:
        project_dir = resolve_project_dir(args["project_id"])
    except ValueError as exc:
        return mcp_error(str(exc))

    # Resolve and bounds-check the source inside the upload root.
    upload_base = _upload_root(project_dir).resolve()
    if not upload_base.is_dir():
        return mcp_error("No upload directory for this project")
    src = (upload_base / args["src_relpath"]).resolve()
    if not src.is_relative_to(upload_base):
        return mcp_error(f"src_relpath escapes upload dir: {args['src_relpath']}")
    if not src.is_file():
        return mcp_error(f"Source file not found: {args['src_relpath']}")

    # Resolve and bounds-check the destination inside the allowlisted subdir.
    dest_dir = (project_dir / dest_subdir).resolve()
    if not dest_dir.is_relative_to(project_dir.resolve()):
        return mcp_error("Invalid dest_subdir")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_name = args.get("dest_filename") or src.name
    # Prevent a rename that injects path segments.
    if "/" in dest_name or "\\" in dest_name or dest_name in ("", ".", ".."):
        return mcp_error(f"Invalid dest_filename: {dest_name!r}")
    dest = (dest_dir / dest_name).resolve()
    if not dest.is_relative_to(dest_dir):
        return mcp_error("Invalid dest_filename")
    if dest.exists():
        return mcp_error(f"Destination already exists: {dest_subdir}/{dest_name}")

    size = src.stat().st_size
    shutil.move(str(src), str(dest))

    logger.info(
        "[move_file] project=%s %s -> %s/%s (%d bytes)",
        args["project_id"][:12], args["src_relpath"], dest_subdir, dest_name, size,
    )
    return mcp_json({
        "from": f"data/uploads/{args['src_relpath']}",
        "to": f"{dest_subdir}/{dest_name}",
        "size": size,
    })


def _is_empty_dir(path: Path) -> bool:
    """Return True when ``path`` has no direct children.

    Paired with a depth-first walk in :func:`finalize_upload`: by the
    time we consider a parent directory, any empty children have
    already been removed, so a single ``iterdir()`` call is sufficient.
    """
    try:
        return not any(path.iterdir())
    except OSError:
        return False


@tool(
    "finalize_upload",
    "Clean up data/uploads/ after sorting. Call with no delete_zips to inspect "
    "what remains — the response lists any preserved .zip archives so you can "
    "ask the user whether to keep them. Then call again with delete_zips=[names] "
    "to remove the ones the user declines. Empty extraction folders are always "
    "removed; data/uploads/ itself is removed when it becomes empty.",
    {
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "delete_zips": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Filenames (no path) of .zip archives under data/uploads/ "
                    "to delete. Omit on the first call; supply after the user "
                    "confirms."
                ),
            },
        },
        "required": ["project_id"],
    },
)
async def finalize_upload(args: dict[str, Any]) -> dict[str, Any]:
    """Cleanup pass over ``data/uploads/`` after the agent's sort loop.

    Behaviour:
      * Every subdirectory that is now empty (recursively) is removed —
        these are extraction folders whose contents have been moved out.
      * If ``delete_zips`` is provided, each listed ``.zip`` file sitting
        directly under ``data/uploads/`` is deleted.
      * Any ``.zip`` files that remain are reported back so the agent can
        ask the user whether to keep them.
      * If ``data/uploads/`` ends up empty, the directory itself is
        removed so fresh projects continue to show no ``data/`` tree.

    Args:
        args: Dict with ``project_id`` and optional ``delete_zips``.

    Returns:
        MCP response with ``{removed_dirs, deleted_zips, remaining_zips,
        upload_dir_removed}``.
    """
    try:
        project_dir = resolve_project_dir(args["project_id"])
    except ValueError as exc:
        return mcp_error(str(exc))

    upload_root = _upload_root(project_dir)
    if not upload_root.is_dir():
        return mcp_json({
            "removed_dirs": [],
            "deleted_zips": [],
            "remaining_zips": [],
            "upload_dir_removed": False,
        })

    upload_resolved = upload_root.resolve()
    deleted_zips: list[str] = []
    for name in args.get("delete_zips") or []:
        # Validate: filename only, must live directly under data/uploads/.
        if "/" in name or "\\" in name or name in ("", ".", "..") or not name.lower().endswith(".zip"):
            return mcp_error(f"Invalid zip name: {name!r}")
        candidate = (upload_root / name).resolve()
        if not candidate.is_relative_to(upload_resolved) or candidate.parent != upload_resolved:
            return mcp_error(f"Zip not in data/uploads/: {name!r}")
        if not candidate.is_file():
            return mcp_error(f"Zip not found: {name!r}")
        candidate.unlink()
        deleted_zips.append(name)

    # Remove empty extraction subfolders. Walk deepest-first so nested
    # empties collapse cleanly.
    removed_dirs: list[str] = []
    subdirs = sorted(
        (p for p in upload_root.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for sub in subdirs:
        if _is_empty_dir(sub):
            rel = str(sub.relative_to(upload_root))
            sub.rmdir()
            removed_dirs.append(rel)

    # Report zips still present after the optional delete pass.
    remaining_zips = sorted(
        p.name
        for p in upload_root.iterdir()
        if p.is_file() and p.suffix.lower() == ".zip"
    )

    # If the upload root is now completely empty, drop it too.
    upload_dir_removed = False
    if not any(upload_root.iterdir()):
        upload_root.rmdir()
        upload_dir_removed = True

    logger.info(
        "[finalize_upload] project=%s removed_dirs=%d deleted_zips=%d remaining_zips=%d upload_dir_removed=%s",
        args["project_id"][:12], len(removed_dirs), len(deleted_zips),
        len(remaining_zips), upload_dir_removed,
    )
    return mcp_json({
        "removed_dirs": removed_dirs,
        "deleted_zips": deleted_zips,
        "remaining_zips": remaining_zips,
        "upload_dir_removed": upload_dir_removed,
    })
