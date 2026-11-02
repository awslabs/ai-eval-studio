# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the agentcore_session_registry helper service."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from app.services.agentcore_session_registry import (
    claim_turn,
    release_turn,
    resolve_resume,
    record_session,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.asyncio


@pytest.fixture
def dev_platform_for_registry(tmp_path: Path, projects_dir: Path):
    """Set up dev platform with FilesystemMetadataStore for registry tests."""
    from app.platform import Platform, set_platform
    from app.platform.dev_local import (
        FilesystemMetadataStore,
        LocalAuth,
        NoOpTokenVending,
        _build_dev_session_store,
    )

    set_platform(None)

    # Build a dev platform with the tmp projects_dir
    metadata_store = FilesystemMetadataStore(projects_dir=projects_dir)
    session_store = _build_dev_session_store()
    auth = LocalAuth(metadata_store=metadata_store)
    token_vending = NoOpTokenVending()

    platform = Platform(
        metadata_store=metadata_store,
        session_store=session_store,
        auth=auth,
        token_vending=token_vending,
    )
    set_platform(platform)

    yield platform

    # Cleanup
    set_platform(None)


@pytest.fixture
def test_project_with_session():
    """Provide a mock project with a persisted session_id."""
    from app.models.schemas import ProjectConfig

    project = ProjectConfig(
        id="test-proj-001",
        name="Test Project",
        description="Test",
        created_at="2026-01-01T00:00:00",
        session_id="test-session-abc",  # The persisted session_id
    )
    return project


class TestClaimRelease:
    """Tests for claim_turn + release_turn."""

    async def test_claim_then_second_claim_false(self, dev_platform_for_registry):
        """Second claim on the same (user, project) returns False while held."""
        user_id = "user-1"
        project_id = "proj-1"

        # First claim wins
        won1 = await claim_turn(user_id, project_id)
        assert won1 is True

        # Second claim on the same (user, project) returns False
        won2 = await claim_turn(user_id, project_id)
        assert won2 is False

    async def test_release_then_reclaim_true(self, dev_platform_for_registry):
        """After release, the same (user, project) can be claimed again."""
        user_id = "user-1"
        project_id = "proj-1"

        # Claim, release, reclaim
        won1 = await claim_turn(user_id, project_id)
        assert won1 is True

        await release_turn(user_id, project_id)

        won2 = await claim_turn(user_id, project_id)
        assert won2 is True

    async def test_release_never_raises_on_store_error(self, tmp_path: Path, monkeypatch):
        """release_turn swallows store exceptions (best-effort, non-raising)."""
        # Build a platform with a store that raises on release
        from app.platform import Platform, set_platform
        from app.platform.dev_local import (
            FilesystemMetadataStore,
            LocalAuth,
            NoOpTokenVending,
            _build_dev_session_store,
        )

        set_platform(None)

        # Inject a raising fake store
        class RaisingStore(FilesystemMetadataStore):
            async def release_agentcore_session_turn(self, user_id, project_id):
                raise RuntimeError("Store exploded")

        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()

        metadata_store = RaisingStore(projects_dir=projects_dir)
        session_store = _build_dev_session_store()
        auth = LocalAuth(metadata_store=metadata_store)
        token_vending = NoOpTokenVending()

        platform = Platform(
            metadata_store=metadata_store,
            session_store=session_store,
            auth=auth,
            token_vending=token_vending,
        )
        set_platform(platform)

        user_id = "user-1"
        project_id = "proj-1"

        # release_turn should NOT raise, just log and return
        await release_turn(user_id, project_id)

        # Cleanup
        set_platform(None)

    async def test_grain_per_user_project_not_per_user(self, dev_platform_for_registry):
        """Grain test: (userA, projX) held does NOT block (userA, projY)."""
        user_id = "user-1"
        proj_x = "proj-x"
        proj_y = "proj-y"

        # Claim projX
        won_x = await claim_turn(user_id, proj_x)
        assert won_x is True

        # Claim projY for the SAME user — should also win (per-(user,project) grain)
        won_y = await claim_turn(user_id, proj_y)
        assert won_y is True


class TestResolveResume:
    """Tests for resolve_resume durable resume-id resolution."""

    async def test_resolve_resume_returns_durable_ids_when_present(
        self, dev_platform_for_registry
    ):
        """When a durable session exists, resolve_resume returns its stored ids."""
        user_id = "user-1"
        project_id = "proj-1"
        runtime_id = "runtime-123"
        app_id = "app-456"

        # Record a durable session
        await record_session(
            user_id, project_id, runtime_session_id=runtime_id, app_session_id=app_id
        )

        # Resolve should return the durable ids
        resolved_runtime, resolved_app = await resolve_resume(
            user_id, project_id, runtime_session_id="ignored-caller-runtime"
        )

        assert resolved_runtime == runtime_id
        assert resolved_app == app_id

    async def test_resolve_resume_fallback_when_no_durable_session(
        self, dev_platform_for_registry, test_project_with_session, monkeypatch
    ):
        """When no durable session exists, resolve_resume falls back to persisted session_id."""
        from unittest.mock import MagicMock

        user_id = "user-1"
        project_id = test_project_with_session.id
        caller_runtime = "runtime-789"

        # Mock get_project_manager to return our test project
        mock_pm = MagicMock()
        mock_pm.get_project = MagicMock(return_value=test_project_with_session)
        monkeypatch.setattr("app.main.get_project_manager", lambda: mock_pm)

        # No durable session recorded — should fallback to the project's persisted session_id
        # test_project_with_session fixture gives us a project with a persisted session_id
        resolved_runtime, resolved_app = await resolve_resume(
            user_id, project_id, runtime_session_id=caller_runtime
        )

        # Should return the caller-supplied runtime_session_id and the project's persisted session_id
        assert resolved_runtime == caller_runtime
        assert resolved_app == test_project_with_session.session_id

    async def test_resolve_resume_fallback_when_claim_placeholder_exists(
        self, dev_platform_for_registry, test_project_with_session, monkeypatch
    ):
        """Regression test: claim placeholder (item present, app_session_id empty) falls through to persisted session_id.

        When claim_agentcore_session_turn creates the item with resume ids unset,
        resolve_resume must NOT short-circuit the persisted-session fallback just
        because the item exists — it falls through when app_session_id is empty.
        This is the integration-level behavior the original isolated unit test missed.
        """
        from unittest.mock import MagicMock

        user_id = "user-1"
        project_id = test_project_with_session.id
        caller_runtime = "runtime-789"

        # Mock get_project_manager to return our test project
        mock_pm = MagicMock()
        mock_pm.get_project = MagicMock(return_value=test_project_with_session)
        monkeypatch.setattr("app.main.get_project_manager", lambda: mock_pm)

        # Claim creates the item with resume ids unset (the scenario)
        won = await claim_turn(user_id, project_id)
        assert won is True

        # Now resolve_resume — the item exists but app_session_id is empty
        # Must fall through to project.session_id (not return (_, None))
        resolved_runtime, resolved_app = await resolve_resume(
            user_id, project_id, runtime_session_id=caller_runtime
        )

        # Should return the caller-supplied runtime and the project's persisted session_id
        assert resolved_runtime == caller_runtime
        assert resolved_app == test_project_with_session.session_id


class TestRecordSession:
    """Tests for record_session durable upsert."""

    async def test_record_session_upserts_without_forging_claim(self, dev_platform_for_registry):
        """record_session persists mapping WITHOUT setting status=running."""
        user_id = "user-1"
        project_id = "proj-1"
        runtime_id = "runtime-abc"
        app_id = "app-def"

        # Record a session WITHOUT claiming
        await record_session(
            user_id, project_id, runtime_session_id=runtime_id, app_session_id=app_id
        )

        # Verify the mapping is readable
        resolved_runtime, resolved_app = await resolve_resume(
            user_id, project_id, runtime_session_id=None
        )
        assert resolved_runtime == runtime_id
        assert resolved_app == app_id

        # Verify we can still claim (proves record_session did NOT forge a claim)
        won = await claim_turn(user_id, project_id)
        assert won is True
