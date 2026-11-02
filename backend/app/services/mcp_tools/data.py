# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP tools for dataset operations: describe_data.

``describe_data`` inspects an uploaded file and returns schema/sample info so the
agent can understand the dataset before configuring field mappings and metrics.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from claude_agent_sdk import ToolAnnotations, tool

from app.services.dataset_service import DatasetService
from app.services.mcp_response import mcp_error, mcp_json
from app.services.project_paths import resolve_project_dir

logger = logging.getLogger(__name__)


@tool(
    "describe_data",
    "Analyze a data file under the project directory. Returns column names, row count, "
    "and sample rows. Pass a bare filename (e.g. 'cases.csv') to read from data/raw/, "
    "or a project-relative path (e.g. 'docs/schema.csv') to read from elsewhere. "
    "Use this to understand the dataset before configuring field mappings and metrics.",
    {"project_id": str, "filename": str},
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def describe_data(args: dict[str, Any]) -> dict[str, Any]:
    """Analyze a data file and return schema/sample information.

    Resolves ``filename`` in this order:
      1. If it contains a ``/``, treat it as a project-relative path.
      2. Otherwise, default to ``data/raw/<filename>`` — the common case.

    In both cases the resolved path is bounds-checked against the project
    root to prevent traversal (``..``) escape.

    Args:
        args: Dict with ``project_id`` and ``filename`` (bare name or
            project-relative path).

    Returns:
        MCP response with file description (columns, rows, samples), or
        an error when the path escapes the project root or the file
        doesn't exist.
    """
    filename = args.get("filename", "")
    logger.info("[describe_data] project=%s file=%s", args.get("project_id", "?")[:12], filename)
    try:
        project_dir = resolve_project_dir(args["project_id"])
        project_root = project_dir.resolve()

        # Bare name → default to data/raw/. Any path separator → treat as
        # already project-relative (so the agent can pass 'docs/foo.csv'
        # or 'data/processed/foo.csv' without us mangling it).
        rel_path = filename if "/" in filename or "\\" in filename else f"data/raw/{filename}"
        file_path = (project_dir / rel_path).resolve()

        if not file_path.is_relative_to(project_root):
            return mcp_error(f"filename escapes project dir: {filename!r}")
        if not file_path.is_file():
            logger.warning("[describe_data] File not found: %s", file_path)
            return mcp_error(f"File not found: {filename}")

        svc = DatasetService()
        info = svc.describe_file(file_path)
        return mcp_json(info)
    except Exception as e:
        return mcp_error(f"Error: {e}")


