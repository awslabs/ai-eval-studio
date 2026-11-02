# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""File browsing, content retrieval, saving, and harness config parsing endpoints.

Provides API endpoints for navigating project file trees, reading and writing
file contents, and parsing structured configuration from harness config.py files.
"""
import csv
import io
import logging
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import (
    FileContentResponse,
    FileNode,
    FileSaveRequest,
    HarnessConfig,
)
from app.platform.api_authz import require_project_read, require_project_write
from app.platform.types import UserId
from app.services.harness_config import parse_harness_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["files"])

#: Ceiling on the uncompressed bytes the harness download will archive.
#:
#: The archive is assembled entirely in memory and ``data/`` is deliberately
#: included, so without a ceiling one request against a project holding an
#: accepted multi-gigabyte dataset could exhaust the coordinator. Chosen to sit
#: well above a realistic harness plus dataset while staying a small fraction of
#: the task's memory.
HARNESS_ZIP_MAX_BYTES = 512 * 1024 * 1024

#: Ceiling on a single file's size for the text/CSV preview endpoint.
#:
#: The preview decodes the whole file into one string before anything trims it,
#: so this is the real memory cost of one request. Comfortably above any prompt,
#: config or source file, and above a dataset a human would page through.
FILE_PREVIEW_MAX_BYTES = 32 * 1024 * 1024

# Directory names to exclude from file tree listings
EXCLUDED_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "__pypackages__",
    # App-internal chat-history home (transcript.jsonl + cursor sidecar).
    # State, not harness content, so the file tree hides it.
    "chat",
}

# Directory names excluded from the harness download zip — non-portable or
# regenerable cruft. Distinct from ``EXCLUDED_DIRS`` (which is for the file
# tree): the zip must stay runnable after ``uv sync`` locally, so it drops
# virtualenvs and caches that ``uv sync`` rebuilds. ``*.pyc`` files are
# filtered separately (suffix match, not a dir name).
HARNESS_ZIP_EXCLUDED_DIRS = {
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "__pypackages__",
    ".uv-cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    # Server-side derived cache of eval results (compact projection + per-case
    # detail sidecars under ``output/_derived/``). Regenerated on read from the
    # immutable run files, so it's cruft in a downloaded harness.
    "_derived",
    # App-internal chat-history home — a downloaded harness must not ship the
    # user's chat transcript.
    "chat",
    # DeepEval's local result cache; regenerable eval cruft (same category
    # as _derived).
    ".deepeval",
}

# Map file extensions to language identifiers for syntax highlighting
EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".json": "json",
    ".md": "markdown",
    ".csv": "csv",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".txt": "text",
    ".html": "html",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescriptreact",
    ".jsx": "javascriptreact",
    ".sh": "shell",
    ".cfg": "ini",
    ".ini": "ini",
}


def _safe_project_path(project_id: str, subpath: str = "") -> Path:
    """Resolve a path within a project directory, preventing traversal attacks.

    Validates that the resolved path stays within the project's root directory
    to prevent directory traversal (e.g., "../../etc/passwd").

    Args:
        project_id: UUID of the project.
        subpath: Optional relative path within the project directory.

    Returns:
        Resolved absolute Path to the target file or directory.

    Raises:
        HTTPException: 404 if project directory does not exist.
        HTTPException: 400 if the resolved path escapes the project directory.
    """
    from app.services.project_manager import get_project_dir_or_404

    # ``get_project_dir_or_404`` enforces existence + path-traversal
    # safety AND respects the cloud-mode owner-prefixed layout.
    project_dir = get_project_dir_or_404(project_id)
    if subpath:
        target = (project_dir / subpath).resolve()
        # ``is_relative_to``, not a string prefix. Both resolve() first, so no
        # traversal or symlink currently escapes either way — the prefix form was
        # safe here only because project directories are fixed-length uuid4
        # strings, so no sibling name can be a prefix of another. That is a
        # property of the naming scheme rather than of this check, and it is not
        # the kind of thing a path guard should depend on. The rest of the
        # codebase already uses this form (project_paths.safe_path,
        # get_project_dir_or_404).
        if not target.is_relative_to(project_dir.resolve()):
            raise HTTPException(status_code=400, detail="Invalid path")
        return target
    return project_dir


def _build_file_tree(directory: Path, base_path: Path) -> list[FileNode]:
    """Recursively build a file tree structure from a directory.

    Walks the directory tree, excluding directories in EXCLUDED_DIRS, and returns
    a sorted list of FileNode objects (directories first, then alphabetical).

    Args:
        directory: Absolute path to the directory to scan.
        base_path: Project root path, used to compute relative paths.

    Returns:
        Sorted list of FileNode objects representing the directory contents.
    """
    nodes: list[FileNode] = []

    try:
        entries = sorted(directory.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    except PermissionError:
        return nodes

    for entry in entries:
        if entry.name in EXCLUDED_DIRS:
            continue

        # Refuse symlinks rather than resolving them. ``is_dir()`` follows a
        # link, so a directory link made this listing walk whatever it pointed
        # at and report every name and size underneath — on the shared EFS
        # mount, that includes other tenants' project trees. Link targets are
        # also where recursion could cycle, bounded only by the kernel's
        # symlink limit and paid for in response size.
        if entry.is_symlink():
            continue

        relative_path = str(entry.relative_to(base_path))

        if entry.is_dir():
            children = _build_file_tree(entry, base_path)
            nodes.append(FileNode(
                name=entry.name,
                path=relative_path,
                type="dir",
                size=None,
                children=children,
            ))
        elif entry.is_file():
            try:
                file_size = entry.stat().st_size
            except OSError:
                file_size = None
            nodes.append(FileNode(
                name=entry.name,
                path=relative_path,
                type="file",
                size=file_size,
                children=None,
            ))

    return nodes


def _detect_language(file_path: Path) -> str:
    """Detect the programming language from a file's extension.

    Args:
        file_path: Path to the file.

    Returns:
        Language identifier string (e.g., "python", "json"). Defaults to "text".
    """
    return EXTENSION_LANGUAGE_MAP.get(file_path.suffix.lower(), "text")


def _parse_csv_preview(
    content: str, max_rows: int = 5000,
) -> tuple[list[str], list[dict], int]:
    """Parse CSV content into columns and row dicts, limited to max_rows.

    The client-side preview paginates through whatever we return, so the
    cap is set generously — enough to cover typical eval datasets end
    to end — while still bounding the JSON payload at predictable size.
    Larger files are still truncated to protect the renderer; callers
    should use the streaming file endpoints for exports.

    Args:
        content: Raw CSV string content.
        max_rows: Maximum number of data rows to return.

    Returns:
        Tuple of (column_names, row_dicts, total_rows). ``total_rows`` is
        the count in the underlying CSV regardless of the cap, so the
        caller can detect truncation (``len(rows) < total_rows``) and
        surface it to the user.
    """
    reader = csv.DictReader(io.StringIO(content))
    columns = reader.fieldnames or []
    rows: list[dict] = []
    total_rows = 0
    for i, row in enumerate(reader):
        total_rows = i + 1
        if i < max_rows:
            rows.append(dict(row))
    return list(columns), rows, total_rows


@router.get("/{project_id}/files")
async def list_files(
    project_id: str,
    _user: UserId = Depends(require_project_read),
) -> dict:
    """List all files in a project directory as a recursive tree.

    Walks the project directory, excluding common non-essential directories
    like __pycache__ and .git. Returns a nested tree structure sorted with
    directories first, then alphabetically by name.

    Args:
        project_id: UUID of the project to list files for.

    Returns:
        Dict with "tree" key containing a list of FileNode objects.
    """
    project_dir = _safe_project_path(project_id)
    tree = _build_file_tree(project_dir, project_dir)
    return {"tree": tree}


@router.get("/{project_id}/files/{file_path:path}")
async def get_file_content(
    project_id: str,
    file_path: str,
    _user: UserId = Depends(require_project_read),
) -> FileContentResponse:
    """Read and return the content of a file within a project.

    Reads the file as UTF-8 text and detects the language from the file extension.
    For CSV files, also parses the content into rows and columns for structured access.
    Binary files that cannot be decoded as UTF-8 return a 400 error.

    Args:
        project_id: UUID of the project.
        file_path: Relative path to the file within the project directory.

    Returns:
        FileContentResponse with content, size, language, and optional CSV data.

    Raises:
        HTTPException: 404 if the file does not exist.
        HTTPException: 400 if the file cannot be read as UTF-8 text.
        HTTPException: 413 if the file is larger than ``FILE_PREVIEW_MAX_BYTES``.
    """
    target = _safe_project_path(project_id, file_path)

    if target.is_symlink() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Check the size BEFORE reading. The read pulls the whole file into memory as
    # one string, and uploads are accepted well past what the coordinator can
    # afford to hold; the CSV row cap further down bounds the RESPONSE, not this
    # read, because it runs on the already-materialised content.
    file_size = target.stat().st_size
    if file_size > FILE_PREVIEW_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "This file is too large to preview "
                f"({file_size // (1024 * 1024)} MB; limit "
                f"{FILE_PREVIEW_MAX_BYTES // (1024 * 1024)} MB). Download it instead."
            ),
        )

    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Binary file cannot be displayed")
    language = _detect_language(target)

    rows = None
    columns = None
    total_rows = None
    if language == "csv":
        columns, rows, total_rows = _parse_csv_preview(content)

    return FileContentResponse(
        path=file_path,
        content=content,
        size=file_size,
        language=language,
        rows=rows,
        columns=columns,
        total_rows=total_rows,
    )


@router.put("/{project_id}/files/{file_path:path}")
async def save_file_content(
    project_id: str,
    file_path: str,
    body: FileSaveRequest,
    _user: UserId = Depends(require_project_write),
) -> None:
    """Reject direct file edits — source editing is disabled.

    Hand-editing harness source (config.py, metrics, the ``--data-file``
    loader) breaks contracts the agent maintains and re-validates. All
    harness changes go through chat, where the agent edits with knowledge
    of those contracts and re-runs validation. Browsing/viewing/downloading
    files stays available; only writes are blocked.

    Agent-driven writes do NOT use this endpoint — they go through the MCP
    tools (``scaffold_harness``, ``save_prompt``), which write to disk
    directly. This endpoint is user-only, so disabling it is safe.

    Args:
        project_id: UUID of the project.
        file_path: Relative path to the file within the project directory.
        body: Request body containing the file content to write.

    Raises:
        HTTPException: 403 always — source editing is disabled.
    """
    raise HTTPException(
        status_code=403,
        detail="Source editing is disabled; ask in chat to modify the harness.",
    )


@router.get("/{project_id}/harness/config")
async def get_harness_config(
    project_id: str,
    _user: UserId = Depends(require_project_read),
) -> HarnessConfig | None:
    """Parse and return structured configuration from a project's harness config.py.

    Reads the src/config.py file from the project directory and uses AST parsing
    to extract configuration fields without executing the code.

    Args:
        project_id: UUID of the project.

    Returns:
        HarnessConfig with parsed field values, or ``null`` when the project
        has no ``src/config.py`` yet. Not-configured-yet is a NORMAL state
        for a new project, not an error: the SPA fetches this on every page
        load, and a 404 here put a browser network-error line in the console
        on every refresh of every fresh project (browsers log non-2xx
        fetches unconditionally; it cannot be suppressed from JS).

    Raises:
        HTTPException: 400 if config.py has a syntax error or a field whose
            value has the wrong shape.
    """
    project_dir = _safe_project_path(project_id)
    config_path = project_dir / "src" / "config.py"

    if not config_path.is_file():
        return None

    source_code = config_path.read_text(encoding="utf-8")

    try:
        config = parse_harness_config(source_code)
    except (SyntaxError, ValueError) as exc:
        # A syntax error or a wrongly shaped value: the agent wrote the file, so
        # it is the user's to fix in chat and is reported, not a server fault.
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse config.py: {exc}",
        ) from exc

    # readiness = what a dispatch actually needs, not parse success.
    # config.py exists from the moment the scaffold lands, minutes before
    # the metric implementation and baseline prompt do — gating UI actions
    # on parse success alone would offer an optimization that fails
    # immediately, with no prompt yet.
    #
    # Metrics are not checked here: a harness defines them as code in
    # src/metrics/metrics.py, which this parse does not read. The baseline
    # prompt is written after the harness and its metrics are built, so a
    # present prompt file already means that stage is done.
    prompt_file_exists = bool(
        config.prompt_version
        and (project_dir / "prompts" / f"{config.prompt_version}.md").is_file()
    )
    config.ready = bool(
        config.models
        and config.data_file
        and prompt_file_exists
    )
    return config


def _build_harness_zip(project_dir: Path) -> io.BytesIO:
    """Zip a project's runnable harness into an in-memory buffer.

    Includes everything needed to run the eval locally — ``run_eval.py``,
    ``src/``, ``pyproject.toml``, ``prompts/``, ``README``, the dataset in
    ``data/raw/``, and prior runs in ``output/``. Excludes non-portable /
    regenerable cruft (``.venv/``, ``__pycache__/``, ``*.pyc``, caches) so
    the recipient can ``uv sync && uv run python run_eval.py`` cleanly.

    Symlinks are refused, never followed. ``Path.is_file()`` follows a link and
    ``ZipFile.write()`` then opens the link's TARGET, so a single symlink planted
    in a project tree turned this endpoint into a read primitive for any file
    the backend process can reach — including, on the shared EFS mount, another
    tenant's project. Nothing stops one being planted: the coordinator agent has
    unjailed Bash with the project as its cwd, and the worker runs
    agent-authored harness code.

    Args:
        project_dir: Absolute path to the project working tree.

    Returns:
        A seeked-to-zero ``BytesIO`` holding the zip archive.

    Raises:
        HTTPException: 413 when the tree exceeds ``HARNESS_ZIP_MAX_BYTES``.
    """
    project_root = project_dir.resolve()
    buffer = io.BytesIO()
    total_bytes = 0
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(project_dir.rglob("*")):
            # Skip any path under an excluded directory.
            rel = path.relative_to(project_dir)
            if any(part in HARNESS_ZIP_EXCLUDED_DIRS for part in rel.parts):
                continue
            # lstat, before anything that follows the link.
            if path.is_symlink():
                logger.warning(
                    "[harness-zip] skipping symlink %s in %s", rel, project_dir.name
                )
                continue
            if not path.is_file() or path.suffix == ".pyc":
                continue
            # Belt and braces: even with links refused above, only archive paths
            # that genuinely live under this project.
            if not path.resolve().is_relative_to(project_root):
                logger.warning(
                    "[harness-zip] skipping out-of-tree path %s in %s",
                    rel, project_dir.name,
                )
                continue
            # Root config.json is the app's INTERNAL project record (owner
            # id, session id, lifecycle flags) — not part of the runnable
            # harness (the shipped template never reads it) and not the
            # recipient's business, so it is excluded from the zip.
            if str(rel) == "config.json":
                continue
            # The whole archive is buffered in memory, and ``data/`` is not
            # excluded, so an accepted multi-gigabyte dataset would otherwise be
            # a single-request way to exhaust the coordinator.
            total_bytes += path.stat().st_size
            if total_bytes > HARNESS_ZIP_MAX_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "This project's files are too large to download as a "
                        f"single archive (limit {HARNESS_ZIP_MAX_BYTES // (1024 * 1024)} MB). "
                        "Download the large files individually."
                    ),
                )
            archive.write(path, arcname=str(rel))
    buffer.seek(0)
    return buffer


@router.get("/{project_id}/harness/download")
async def download_harness(
    project_id: str,
    _user: UserId = Depends(require_project_read),
) -> StreamingResponse:
    """Stream the project's runnable harness as a zip download.

    The archive is built from the project's working tree on disk (EFS in
    cloud, local in dev). The user unzips it and runs the eval locally.

    Args:
        project_id: UUID of the project.

    Returns:
        A ``StreamingResponse`` carrying the zip with an attachment
        ``Content-Disposition`` so the browser saves it.

    Raises:
        HTTPException: 404 if the project directory does not exist.
    """
    project_dir = _safe_project_path(project_id)
    buffer = _build_harness_zip(project_dir)
    filename = f"harness-{project_id}.zip"
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
