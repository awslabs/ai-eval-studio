# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the data MCP tools (``describe_data``).

``resolve_project_dir`` is patched so we don't need a booted
``ProjectManager``. Each test writes a tiny CSV and drives the handler
directly.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest  # noqa: F401

from app.services.mcp_tools.data import describe_data as _describe_tool

describe_data = _describe_tool.handler


def _text(result: dict) -> str:
    """Extract the text body from an MCP tool response for assertions."""
    return result["content"][0]["text"]


def _seed_csv(path: Path, content: str = "a,b\n1,2\n3,4\n") -> None:
    """Write a CSV at ``path``, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class TestDescribeData:
    @patch("app.services.mcp_tools.data.resolve_project_dir")
    async def test_bare_filename_defaults_to_data_raw(self, mock_resolve, tmp_path):
        project_dir = tmp_path / "p"
        _seed_csv(project_dir / "data" / "raw" / "cases.csv")
        mock_resolve.return_value = project_dir

        result = await describe_data({"project_id": "p", "filename": "cases.csv"})
        assert "is_error" not in result
        payload = json.loads(_text(result))
        assert "columns" in payload or "rows" in payload

    @patch("app.services.mcp_tools.data.resolve_project_dir")
    async def test_relative_path_does_not_double_prefix(self, mock_resolve, tmp_path):
        """A ``data/raw/`` prefixed path should resolve to that path, not
        to ``data/raw/data/raw/``. Regression for the path-doubling bug."""
        project_dir = tmp_path / "p"
        _seed_csv(project_dir / "data" / "raw" / "cases.csv")
        mock_resolve.return_value = project_dir

        result = await describe_data({
            "project_id": "p",
            "filename": "data/raw/cases.csv",
        })
        assert "is_error" not in result

    @patch("app.services.mcp_tools.data.resolve_project_dir")
    async def test_docs_path_reads_from_docs(self, mock_resolve, tmp_path):
        """The agent can pass ``docs/foo.csv`` to inspect a reference CSV."""
        project_dir = tmp_path / "p"
        _seed_csv(project_dir / "docs" / "schema.csv")
        mock_resolve.return_value = project_dir

        result = await describe_data({
            "project_id": "p",
            "filename": "docs/schema.csv",
        })
        assert "is_error" not in result

    @patch("app.services.mcp_tools.data.resolve_project_dir")
    async def test_missing_file_returns_error(self, mock_resolve, tmp_path):
        project_dir = tmp_path / "p"
        project_dir.mkdir()
        mock_resolve.return_value = project_dir

        result = await describe_data({"project_id": "p", "filename": "ghost.csv"})
        assert result.get("is_error") is True
        assert "not found" in _text(result).lower()

    @patch("app.services.mcp_tools.data.resolve_project_dir")
    async def test_traversal_rejected(self, mock_resolve, tmp_path):
        project_dir = tmp_path / "p"
        project_dir.mkdir()
        # A file outside the project root.
        outside = tmp_path / "secret.csv"
        outside.write_text("s,v\n1,2\n")
        mock_resolve.return_value = project_dir

        result = await describe_data({
            "project_id": "p",
            "filename": "../secret.csv",
        })
        assert result.get("is_error") is True
