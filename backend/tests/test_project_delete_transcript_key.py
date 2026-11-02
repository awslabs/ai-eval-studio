# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cloud transcript delete must target the prefix the transcript is under.

The SDK keys its S3 session store by the agent's cwd, i.e. the project directory.
Cloud projects live at ``users/{owner}/projects/{pid}``, but by the time deletion
cleanup runs the project is flagged ``deleting`` — which makes ordinary
resolution report it as gone and fall back to the flat ``{projects_dir}/{pid}``.
That produced a different project key, and deleting an S3 prefix that matches
nothing succeeds silently, so the transcript survived its own project's deletion
with no lifecycle rule to reap it.

These tests drive the REAL key derivation. The pre-existing cascade test patches
``_delete_session_transcript`` out entirely, so it cannot see a wrong key.
"""
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

OWNER = "sub-3f9c1a"
PROJECT_ID = "abc-123"
SESSION_ID = "sess-77"


def _project_manager(tmp_path: Path):
    """A real ProjectManager rooted at a temp dir, so layout logic is genuine."""
    from app.services.project_manager import ProjectManager

    return ProjectManager(tmp_path / "projects")


def _main_module():
    """Import ``app.main`` and hand it back for patching.

    Imported here, before a test switches to cloud mode: ``app.main`` calls
    ``create_app()`` at module scope, and the cloud bootstrap demands the Cognito
    env vars. Patching by dotted string would trigger that import from inside the
    cloud-mode block and fail for a reason unrelated to what is under test.
    """
    import app.main

    return app.main


def _captured_store():
    """A session store double that records the delete descriptor it received."""
    store = MagicMock()
    store.delete = AsyncMock(return_value=None)
    return store


@pytest.mark.asyncio
async def test_cloud_transcript_delete_uses_the_owner_aware_project_key(
    tmp_path, monkeypatch
):
    """The key must come from ``users/{owner}/projects/{pid}``, not the flat path."""
    from claude_agent_sdk import project_key_for_directory

    from app.api import projects as projects_api

    main = _main_module()
    monkeypatch.setenv("EVAL_STUDIO_MODE", "cloud")
    pm = _project_manager(tmp_path)
    store = _captured_store()

    with patch.object(main, "get_project_manager", return_value=pm), patch(
        "app.platform.get_platform",
        return_value=MagicMock(session_store=store),
    ):
        await projects_api._delete_session_transcript(
            PROJECT_ID, SESSION_ID, owner_id=OWNER
        )

    store.delete.assert_awaited_once()
    descriptor = store.delete.await_args.args[0]

    owner_aware = pm.projects_dir / "users" / OWNER / "projects" / PROJECT_ID
    assert descriptor["project_key"] == project_key_for_directory(
        str(owner_aware.resolve())
    )
    assert descriptor["session_id"] == SESSION_ID

    # The specific wrong answer this defect produced. Asserting the positive
    # alone would still pass if both paths happened to canonicalise the same.
    flat = pm.projects_dir / PROJECT_ID
    assert descriptor["project_key"] != project_key_for_directory(str(flat.resolve()))


@pytest.mark.asyncio
async def test_cloud_transcript_delete_refuses_without_an_owner(tmp_path, monkeypatch):
    """No owner means the real prefix is unknown, so fail and keep the marker.

    Deleting *something* would be worse than failing: the caller keeps the
    ``deleting`` marker on an exception, so the startup sweep retries.
    """
    from app.api import projects as projects_api

    main = _main_module()
    monkeypatch.setenv("EVAL_STUDIO_MODE", "cloud")
    pm = _project_manager(tmp_path)
    store = _captured_store()

    with patch.object(main, "get_project_manager", return_value=pm), patch(
        "app.platform.get_platform",
        return_value=MagicMock(session_store=store),
    ), pytest.raises(ValueError):
        await projects_api._delete_session_transcript(
            PROJECT_ID, SESSION_ID, owner_id=None
        )

    store.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_dev_mode_transcript_delete_is_still_a_no_op(tmp_path, monkeypatch):
    """Dev keeps transcripts under the tree, so there is no S3 prefix to remove."""
    from app.api import projects as projects_api

    monkeypatch.setenv("EVAL_STUDIO_MODE", "dev")

    with patch(
        "app.platform.get_platform", return_value=MagicMock(session_store=None)
    ):
        await projects_api._delete_session_transcript(
            PROJECT_ID, SESSION_ID, owner_id=None
        )
    # Reaching here without raising is the assertion: dev must not need an owner.


@pytest.mark.asyncio
async def test_cascade_threads_the_owner_into_transcript_cleanup(monkeypatch):
    """The owner has always been in scope one frame up; it must be passed down."""
    from app.api import projects as projects_api

    monkeypatch.setenv("EVAL_STUDIO_MODE", "dev")

    with patch.object(
        projects_api, "_delete_session_transcript", new=AsyncMock()
    ) as transcript, patch(
        "app.services.report_snapshot.delete_project_reports_from_s3",
        new=AsyncMock(),
    ), patch.object(
        projects_api, "_get_pm", return_value=MagicMock()
    ), patch(
        "app.platform.get_platform", return_value=MagicMock()
    ):
        await projects_api._complete_project_deletion(
            PROJECT_ID, owner_id=OWNER, session_id=SESSION_ID
        )

    transcript.assert_awaited_once_with(PROJECT_ID, SESSION_ID, owner_id=OWNER)
