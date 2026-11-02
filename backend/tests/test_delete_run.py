# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``delete_run`` MCP tool's project-event publish.

The run list the SPA renders is a pull-only REST resource. Deleting a run
without announcing it left the Evaluations tab showing a run whose file was
already gone, until some unrelated navigation happened to refetch. These tests
pin the announcement itself — that the event fires on success with the shape
the frontend subscribes to, and that a failed delete stays silent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.mcp_tools.eval import delete_eval_run as _delete_run_tool

_delete_run = _delete_run_tool.handler


@pytest.fixture
def project(tmp_path: Path) -> dict:
    """A project tree with one eval-results file in ``output/``."""
    project_id = "delete-run-test"
    project_dir = tmp_path / project_id
    (project_dir / "output").mkdir(parents=True)
    filename = "eval_results_v1_20260504_120000.json"
    (project_dir / "output" / filename).write_text("[]")
    return {"project_id": project_id, "project_dir": project_dir, "filename": filename}


@pytest.fixture
def resolved(project, monkeypatch) -> dict:
    """``project`` with ``resolve_project_dir`` pointed at its tree."""
    monkeypatch.setattr(
        "app.services.mcp_tools.eval.resolve_project_dir",
        lambda pid: project["project_dir"],
    )
    return project


class TestDeleteRunPublishesEvent:
    async def test_publishes_eval_run_deleted_on_success(self, resolved, monkeypatch):
        """Success publishes ``eval_run/deleted`` carrying the run filename."""
        published: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            "app.services.mcp_tools.eval._publish_project_event",
            lambda pid, event: published.append((pid, event)),
        )

        result = await _delete_run({
            "project_id": resolved["project_id"],
            "run_filename": resolved["filename"],
        })

        assert "is_error" not in result
        assert not (resolved["project_dir"] / "output" / resolved["filename"]).exists()
        assert len(published) == 1
        project_id, event = published[0]
        assert project_id == resolved["project_id"]
        assert event["resource"] == "eval_run"
        assert event["type"] == "deleted"
        assert event["run_filename"] == resolved["filename"]

    async def test_missing_file_publishes_nothing(self, resolved, monkeypatch):
        """A run that doesn't exist changes no list, so it must stay silent."""
        published: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            "app.services.mcp_tools.eval._publish_project_event",
            lambda pid, event: published.append((pid, event)),
        )

        result = await _delete_run({
            "project_id": resolved["project_id"],
            "run_filename": "nonexistent.json",
        })

        assert result.get("is_error") is True
        assert published == []

    async def test_invalid_filename_publishes_nothing(self, resolved, monkeypatch):
        """A rejected filename deletes nothing, so it must stay silent."""
        published: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            "app.services.mcp_tools.eval._publish_project_event",
            lambda pid, event: published.append((pid, event)),
        )

        result = await _delete_run({
            "project_id": resolved["project_id"],
            "run_filename": "../config.json",
        })

        assert result.get("is_error") is True
        assert published == []

    async def test_unlink_failure_publishes_nothing(self, resolved, monkeypatch):
        """An OSError on unlink means the run survives — announcing its
        deletion would make the SPA drop a run that is still there."""
        published: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            "app.services.mcp_tools.eval._publish_project_event",
            lambda pid, event: published.append((pid, event)),
        )

        def fail_unlink(self, *args, **kwargs):
            raise PermissionError(f"[Errno 13] Permission denied: {self}")

        monkeypatch.setattr(Path, "unlink", fail_unlink)

        result = await _delete_run({
            "project_id": resolved["project_id"],
            "run_filename": resolved["filename"],
        })

        assert result.get("is_error") is True
        assert published == []
