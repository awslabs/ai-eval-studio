# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the uploads MCP tools (``list_upload``, ``move_file``,
``finalize_upload``).

``resolve_project_dir`` is patched so the tests don't need a booted
``ProjectManager``. Each test builds a minimal ``data/uploads/`` tree
under ``tmp_path`` and drives the tool handlers directly.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.mcp_tools.uploads import (
    finalize_upload as _finalize_tool,
    list_upload as _list_tool,
    move_file as _move_tool,
)

# SdkMcpTool wraps the handler — unwrap for direct pytest calls.
list_upload = _list_tool.handler
move_file = _move_tool.handler
finalize_upload = _finalize_tool.handler


def _make_upload_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Build ``projects/{id}/data/uploads/`` and return project + upload root."""
    project_dir = tmp_path / "test-project"
    upload_root = project_dir / "data" / "uploads"
    upload_root.mkdir(parents=True)
    return project_dir, upload_root


def _text(result: dict) -> str:
    """Extract the text body from an MCP tool response for assertions."""
    return result["content"][0]["text"]


class TestListUpload:
    @patch("app.services.mcp_tools.uploads.resolve_project_dir")
    async def test_happy_path_mixed_files(self, mock_resolve, tmp_path):
        project_dir, upload_root = _make_upload_dir(tmp_path)
        mock_resolve.return_value = project_dir
        (upload_root / "cases.csv").write_bytes(b"a,b\n1,2\n")
        (upload_root / "manual.pdf").write_bytes(b"%PDF-1.4")
        (upload_root / "notes.md").write_text("hello")
        (upload_root / "nested").mkdir()
        (upload_root / "nested" / "schema.json").write_text("{}")

        result = await list_upload({"project_id": "test-project"})
        assert "is_error" not in result
        payload = json.loads(_text(result))
        rels = {f["relpath"]: f for f in payload["files"]}
        assert set(rels) == {"cases.csv", "manual.pdf", "notes.md", "nested/schema.json"}
        assert rels["cases.csv"]["kind"] == "tabular"
        assert rels["cases.csv"]["suggested_dest"] == "data/raw"
        assert rels["manual.pdf"]["kind"] == "document"
        assert rels["manual.pdf"]["suggested_dest"] == "docs"
        assert rels["nested/schema.json"]["kind"] == "tabular"

    @patch("app.services.mcp_tools.uploads.resolve_project_dir")
    async def test_missing_upload_dir_returns_empty(self, mock_resolve, tmp_path):
        """A fresh project has no ``data/uploads/`` yet — not an error."""
        project_dir = tmp_path / "test-project"
        project_dir.mkdir()
        mock_resolve.return_value = project_dir

        result = await list_upload({"project_id": "test-project"})
        assert "is_error" not in result
        payload = json.loads(_text(result))
        assert payload == {"files": []}

    @patch("app.services.mcp_tools.uploads.resolve_project_dir")
    async def test_zip_archive_appears_in_listing(self, mock_resolve, tmp_path):
        """Zip archives surface as plain entries (agent is told to skip them)."""
        project_dir, upload_root = _make_upload_dir(tmp_path)
        mock_resolve.return_value = project_dir
        (upload_root / "bundle.zip").write_bytes(b"PK\x03\x04junk")
        (upload_root / "bundle").mkdir()
        (upload_root / "bundle" / "cases.csv").write_text("a,b\n")

        result = await list_upload({"project_id": "test-project"})
        payload = json.loads(_text(result))
        rels = {f["relpath"] for f in payload["files"]}
        assert "bundle.zip" in rels
        assert "bundle/cases.csv" in rels


class TestMoveFile:
    @patch("app.services.mcp_tools.uploads.resolve_project_dir")
    async def test_happy_path_to_data_raw(self, mock_resolve, tmp_path):
        project_dir, upload_root = _make_upload_dir(tmp_path)
        mock_resolve.return_value = project_dir
        (upload_root / "cases.csv").write_bytes(b"a,b\n1,2\n")

        result = await move_file({
            "project_id": "test-project",
            "src_relpath": "cases.csv",
            "dest_subdir": "data/raw",
        })
        assert "is_error" not in result
        payload = json.loads(_text(result))
        assert payload["to"] == "data/raw/cases.csv"
        assert payload["from"] == "data/uploads/cases.csv"
        assert (project_dir / "data" / "raw" / "cases.csv").is_file()
        assert not (upload_root / "cases.csv").exists()

    @patch("app.services.mcp_tools.uploads.resolve_project_dir")
    async def test_happy_path_rename(self, mock_resolve, tmp_path):
        project_dir, upload_root = _make_upload_dir(tmp_path)
        mock_resolve.return_value = project_dir
        (upload_root / "readme.md").write_text("hi")

        result = await move_file({
            "project_id": "test-project",
            "src_relpath": "readme.md",
            "dest_subdir": "docs",
            "dest_filename": "overview.md",
        })
        assert "is_error" not in result
        assert (project_dir / "docs" / "overview.md").is_file()

    @patch("app.services.mcp_tools.uploads.resolve_project_dir")
    async def test_nested_source_path(self, mock_resolve, tmp_path):
        """``src_relpath`` may point into an extraction subdirectory."""
        project_dir, upload_root = _make_upload_dir(tmp_path)
        mock_resolve.return_value = project_dir
        (upload_root / "dataset").mkdir()
        (upload_root / "dataset" / "cases.csv").write_text("a,b")

        result = await move_file({
            "project_id": "test-project",
            "src_relpath": "dataset/cases.csv",
            "dest_subdir": "data/raw",
        })
        assert "is_error" not in result
        assert (project_dir / "data" / "raw" / "cases.csv").is_file()

    @patch("app.services.mcp_tools.uploads.resolve_project_dir")
    async def test_non_allowlisted_dest_rejected(self, mock_resolve, tmp_path):
        project_dir, upload_root = _make_upload_dir(tmp_path)
        mock_resolve.return_value = project_dir
        (upload_root / "x.csv").write_text("a,b")

        result = await move_file({
            "project_id": "test-project",
            "src_relpath": "x.csv",
            "dest_subdir": "etc",
        })
        assert result.get("is_error") is True
        assert "dest_subdir" in _text(result)
        assert (upload_root / "x.csv").exists()

    @patch("app.services.mcp_tools.uploads.resolve_project_dir")
    async def test_source_traversal_rejected(self, mock_resolve, tmp_path):
        project_dir, _ = _make_upload_dir(tmp_path)
        mock_resolve.return_value = project_dir
        outside = project_dir / "secret.txt"
        outside.write_text("nope")

        result = await move_file({
            "project_id": "test-project",
            "src_relpath": "../../secret.txt",
            "dest_subdir": "docs",
        })
        assert result.get("is_error") is True
        assert outside.exists()

    @patch("app.services.mcp_tools.uploads.resolve_project_dir")
    async def test_dest_filename_with_slash_rejected(self, mock_resolve, tmp_path):
        project_dir, upload_root = _make_upload_dir(tmp_path)
        mock_resolve.return_value = project_dir
        (upload_root / "x.md").write_text("hi")

        result = await move_file({
            "project_id": "test-project",
            "src_relpath": "x.md",
            "dest_subdir": "docs",
            "dest_filename": "../etc/passwd",
        })
        assert result.get("is_error") is True

    @patch("app.services.mcp_tools.uploads.resolve_project_dir")
    async def test_overwrite_rejected(self, mock_resolve, tmp_path):
        project_dir, upload_root = _make_upload_dir(tmp_path)
        mock_resolve.return_value = project_dir
        (upload_root / "cases.csv").write_text("new")
        existing = project_dir / "data" / "raw" / "cases.csv"
        existing.parent.mkdir(parents=True)
        existing.write_text("existing")

        result = await move_file({
            "project_id": "test-project",
            "src_relpath": "cases.csv",
            "dest_subdir": "data/raw",
        })
        assert result.get("is_error") is True
        assert "already exists" in _text(result)
        assert existing.read_text() == "existing"
        assert (upload_root / "cases.csv").read_text() == "new"

    @patch("app.services.mcp_tools.uploads.resolve_project_dir")
    async def test_missing_source(self, mock_resolve, tmp_path):
        project_dir, _ = _make_upload_dir(tmp_path)
        mock_resolve.return_value = project_dir

        result = await move_file({
            "project_id": "test-project",
            "src_relpath": "nope.csv",
            "dest_subdir": "data/raw",
        })
        assert result.get("is_error") is True
        assert "not found" in _text(result).lower()

    @patch("app.services.mcp_tools.uploads.resolve_project_dir")
    async def test_missing_upload_dir(self, mock_resolve, tmp_path):
        project_dir = tmp_path / "test-project"
        project_dir.mkdir()
        mock_resolve.return_value = project_dir

        result = await move_file({
            "project_id": "test-project",
            "src_relpath": "cases.csv",
            "dest_subdir": "data/raw",
        })
        assert result.get("is_error") is True


class TestFinalizeUpload:
    @patch("app.services.mcp_tools.uploads.resolve_project_dir")
    async def test_empty_extraction_dir_removed(self, mock_resolve, tmp_path):
        """After the agent moves every file out, the extraction dir collapses."""
        project_dir, upload_root = _make_upload_dir(tmp_path)
        mock_resolve.return_value = project_dir
        (upload_root / "dataset").mkdir()
        (upload_root / "dataset" / "nested").mkdir()
        # No files — simulates post-move state.
        (upload_root / "dataset.zip").write_bytes(b"archive")

        result = await finalize_upload({"project_id": "test-project"})
        assert "is_error" not in result
        payload = json.loads(_text(result))

        # Both the parent and the nested empty child must collapse.
        assert "dataset" in payload["removed_dirs"]
        assert "dataset/nested" in payload["removed_dirs"]
        assert payload["remaining_zips"] == ["dataset.zip"]
        assert payload["upload_dir_removed"] is False
        assert not (upload_root / "dataset").exists()
        assert (upload_root / "dataset.zip").exists()

    @patch("app.services.mcp_tools.uploads.resolve_project_dir")
    async def test_non_empty_extraction_preserved(self, mock_resolve, tmp_path):
        project_dir, upload_root = _make_upload_dir(tmp_path)
        mock_resolve.return_value = project_dir
        (upload_root / "kit").mkdir()
        (upload_root / "kit" / "leftover.txt").write_text("still here")

        result = await finalize_upload({"project_id": "test-project"})
        assert "is_error" not in result
        payload = json.loads(_text(result))
        assert payload["removed_dirs"] == []
        assert (upload_root / "kit" / "leftover.txt").exists()

    @patch("app.services.mcp_tools.uploads.resolve_project_dir")
    async def test_delete_zips_removes_listed_archives(self, mock_resolve, tmp_path):
        project_dir, upload_root = _make_upload_dir(tmp_path)
        mock_resolve.return_value = project_dir
        (upload_root / "keep.zip").write_bytes(b"a")
        (upload_root / "drop.zip").write_bytes(b"b")
        (upload_root / "drop").mkdir()  # empty extraction → will be cleaned

        result = await finalize_upload({
            "project_id": "test-project",
            "delete_zips": ["drop.zip"],
        })
        assert "is_error" not in result
        payload = json.loads(_text(result))
        assert payload["deleted_zips"] == ["drop.zip"]
        assert payload["remaining_zips"] == ["keep.zip"]
        assert not (upload_root / "drop.zip").exists()
        assert (upload_root / "keep.zip").exists()

    @patch("app.services.mcp_tools.uploads.resolve_project_dir")
    async def test_empty_upload_dir_is_removed(self, mock_resolve, tmp_path):
        """When everything's been moved + cleaned, ``data/uploads/`` disappears."""
        project_dir, upload_root = _make_upload_dir(tmp_path)
        mock_resolve.return_value = project_dir
        (upload_root / "kit").mkdir()  # empty

        result = await finalize_upload({"project_id": "test-project"})
        assert "is_error" not in result
        payload = json.loads(_text(result))
        assert payload["upload_dir_removed"] is True
        assert not upload_root.exists()

    @patch("app.services.mcp_tools.uploads.resolve_project_dir")
    async def test_missing_upload_dir_is_no_op(self, mock_resolve, tmp_path):
        project_dir = tmp_path / "test-project"
        project_dir.mkdir()
        mock_resolve.return_value = project_dir

        result = await finalize_upload({"project_id": "test-project"})
        assert "is_error" not in result
        payload = json.loads(_text(result))
        assert payload == {
            "removed_dirs": [],
            "deleted_zips": [],
            "remaining_zips": [],
            "upload_dir_removed": False,
        }

    @patch("app.services.mcp_tools.uploads.resolve_project_dir")
    async def test_delete_zips_validates_name(self, mock_resolve, tmp_path):
        project_dir, upload_root = _make_upload_dir(tmp_path)
        mock_resolve.return_value = project_dir

        # Path separators rejected.
        bad = await finalize_upload({
            "project_id": "test-project",
            "delete_zips": ["../etc/passwd"],
        })
        assert bad.get("is_error") is True

    @patch("app.services.mcp_tools.uploads.resolve_project_dir")
    async def test_delete_zips_rejects_non_zip(self, mock_resolve, tmp_path):
        project_dir, upload_root = _make_upload_dir(tmp_path)
        mock_resolve.return_value = project_dir
        (upload_root / "notes.md").write_text("hi")

        result = await finalize_upload({
            "project_id": "test-project",
            "delete_zips": ["notes.md"],
        })
        assert result.get("is_error") is True
        assert (upload_root / "notes.md").exists()

    @patch("app.services.mcp_tools.uploads.resolve_project_dir")
    async def test_delete_zips_missing_file(self, mock_resolve, tmp_path):
        project_dir, _ = _make_upload_dir(tmp_path)
        mock_resolve.return_value = project_dir

        result = await finalize_upload({
            "project_id": "test-project",
            "delete_zips": ["ghost.zip"],
        })
        assert result.get("is_error") is True
