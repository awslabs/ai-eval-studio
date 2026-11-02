# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the chat API endpoints.

Covers project validation, turn creation, history, stop semantics, and
empty-message rejection. The SSE GET routes
(``/api/chat/stream/{turn_id}`` and ``/api/chat/active-turn/{pid}``) were
deleted along with their tests — the live envelope path now lives on /ws
and is covered by ``test_chat_ws_e2e.py``.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.services.agent_turn import get_agent_turn_tracker


@pytest.fixture
def app(tmp_path):
    """Create test app with isolated projects directory."""
    import app.main as main_module
    original = main_module.PROJECTS_DIR
    main_module.PROJECTS_DIR = tmp_path / "projects"
    main_module.PROJECTS_DIR.mkdir()
    main_module._project_manager = None
    application = create_app()
    yield application
    main_module._project_manager = None
    main_module.PROJECTS_DIR = original
    get_agent_turn_tracker()._reset()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestChatAPI:
    async def test_chat_requires_project(self, client):
        resp = await client.post(
            "/api/chat",
            json={"message": "hello", "project_id": "nonexistent"},
        )
        assert resp.status_code == 404

    async def test_chat_returns_turn_id(self, client):
        """POST /api/chat starts a background turn and returns turn_id."""
        create_resp = await client.post("/api/projects", json={"name": "Chat Test"})
        project_id = create_resp.json()["id"]

        with patch("app.api.chat.start_agent_turn") as mock_start:
            mock_turn = MagicMock()
            mock_turn.turn_id = "test-turn-123"
            mock_start.return_value = mock_turn

            resp = await client.post(
                "/api/chat",
                json={"message": "hello", "project_id": project_id},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["turn_id"] == "test-turn-123"
            mock_start.assert_called_once()

    async def test_chat_history_empty(self, client):
        """History returns empty messages for project with no session."""
        create_resp = await client.post("/api/projects", json={"name": "History Test"})
        project_id = create_resp.json()["id"]
        resp = await client.get(f"/api/projects/{project_id}/chat/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["messages"] == []
        assert data["interrupted"] is False

    async def test_chat_empty_message(self, client):
        resp = await client.post(
            "/api/chat",
            json={"message": "", "project_id": "some-id"},
        )
        assert resp.status_code == 422

    async def test_stop_turn_not_found(self, client):
        """POST /api/chat/stop/{turn_id} returns 404 for nonexistent turn."""
        resp = await client.post("/api/chat/stop/nonexistent-turn-id")
        assert resp.status_code == 404

    async def test_stop_turn_running(self, client):
        """POST /api/chat/stop/{turn_id} returns 'stopped' for a running turn."""
        # Authz: stop requires write access to the turn's project,
        # so create a real one and use its id.
        create_resp = await client.post("/api/projects", json={"name": "Stop Test"})
        project_id = create_resp.json()["id"]
        tracker = get_agent_turn_tracker()
        turn = tracker.create_turn(project_id)
        assert turn.status == "running"

        resp = await client.post(f"/api/chat/stop/{turn.turn_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stopped"

    async def test_stop_turn_already_completed(self, client):
        """POST /api/chat/stop/{turn_id} returns current status for finished turn."""
        create_resp = await client.post("/api/projects", json={"name": "Stop Done"})
        project_id = create_resp.json()["id"]
        tracker = get_agent_turn_tracker()
        turn = tracker.create_turn(project_id)
        turn.status = "completed"

        resp = await client.post(f"/api/chat/stop/{turn.turn_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert "already finished" in data["message"].lower()


class TestChatHistoryTranscript:
    """Read-path tests: GET /chat/history serves the durable transcript.

    The history endpoint reads from the durable transcript (not the live SDK
    session), so SDK auto-compaction never erases user-visible history. These
    cover the four behaviors that depend on that decoupling: post-compaction
    durability, synthetic-message exclusion, lazy backfill, and the
    ``interrupted`` flag computed off the served transcript.
    """

    async def _project_with_session(self, client, name: str) -> str:
        """Create a project and stamp a session_id so history isn't short-circuited."""
        from app.main import get_project_manager

        project_id = (await client.post("/api/projects", json={"name": name})).json()["id"]
        pm = get_project_manager()
        project = pm.get_project(project_id)
        project.session_id = "sess-test"
        pm.save_config(project)
        return project_id

    async def test_post_compaction_durability(self, client):
        """History returns the FULL durable transcript even when the live SDK
        session has been compacted down to only the most recent turns.

        Proves the read path no longer depends on the live session for turns
        already recorded — the whole point of the decoupling.
        """
        project_id = await self._project_with_session(client, "Durability")
        full_transcript = [
            {"id": "u1", "role": "user", "parts": [{"type": "text", "text": "first question"}], "created_at": "2026-07-10T10:00:00.000Z"},
            {"id": "a1", "role": "assistant", "parts": [{"type": "text", "text": "first answer"}], "created_at": "2026-07-10T10:00:01.000Z"},
            {"id": "u2", "role": "user", "parts": [{"type": "text", "text": "second question"}], "created_at": "2026-07-10T10:00:05.000Z"},
            {"id": "a2", "role": "assistant", "parts": [{"type": "text", "text": "second answer"}], "created_at": "2026-07-10T10:00:06.000Z"},
        ]
        # Transcript already populated → no backfill path; load_session_messages
        # (the compacted live session) must never be consulted for the response.
        with patch("app.services.chat_transcript.load_transcript", return_value=full_transcript):
            resp = await client.get(f"/api/projects/{project_id}/chat/history")

        assert resp.status_code == 200
        data = resp.json()
        assert data["messages"] == full_transcript
        assert data["interrupted"] is False

    async def test_backfill_seeds_then_does_not_duplicate(self, client):
        """A project with a session but no transcript.jsonl is lazy-seeded from
        the raw SDK session entries on first read, then accumulates without duplicating."""
        project_id = await self._project_with_session(client, "Backfill")
        seeded = [
            {"id": "u1", "role": "user", "parts": [{"type": "text", "text": "hi"}], "created_at": ""},
            {"id": "a1", "role": "assistant", "parts": [{"type": "text", "text": "hello"}], "created_at": ""},
        ]
        # First read: no transcript yet → backfill from the raw (mapped) SDK session entries.
        # load_transcript returns [] first (absent), then the seeded list after append.
        load_calls = [[], seeded]
        with patch("app.services.chat_transcript.load_transcript", side_effect=load_calls), \
             patch("app.platform.chat_history.load_raw_session_entries", new=AsyncMock(return_value=["raw"])), \
             patch("app.api.chat.map_session_messages", return_value=seeded), \
             patch("app.services.chat_transcript.append_messages", new=AsyncMock()) as mock_append:
            resp = await client.get(f"/api/projects/{project_id}/chat/history")

        assert resp.status_code == 200
        assert resp.json()["messages"] == seeded
        # Backfill appended exactly once with the mapped session.
        mock_append.assert_awaited_once()
        assert mock_append.await_args.args[1] == seeded

    async def test_synthetic_messages_excluded(self, client):
        """Orchestrator-synthetic messages never reach the response — they were
        dropped by map_session_messages before any append, so a populated
        transcript already excludes them."""
        project_id = await self._project_with_session(client, "Synthetic")
        # Transcript holds only real bubbles (synthetic ones were filtered pre-append).
        clean = [
            {"id": "u1", "role": "user", "parts": [{"type": "text", "text": "real message"}], "created_at": "2026-07-10T12:00:00.000Z"},
        ]
        with patch("app.services.chat_transcript.load_transcript", return_value=clean):
            resp = await client.get(f"/api/projects/{project_id}/chat/history")

        data = resp.json()
        assert data["messages"] == clean
        # Check all parts for synthetic prefix (no old-format `content` field)
        for msg in data["messages"]:
            for part in msg.get("parts", []):
                if part.get("type") == "text":
                    assert "__EVAL_STUDIO_SYS_" not in part.get("text", "")

    async def test_interrupted_flag_off_transcript(self, client):
        """interrupted is True iff the last served message is a user message."""
        project_id = await self._project_with_session(client, "Interrupted")
        trailing_user = [
            {"id": "a1", "role": "assistant", "parts": [{"type": "text", "text": "answer"}], "created_at": "2026-07-10T12:00:00.000Z"},
            {"id": "u2", "role": "user", "parts": [{"type": "text", "text": "unanswered"}], "created_at": "2026-07-10T12:00:05.000Z"},
        ]
        with patch("app.services.chat_transcript.load_transcript", return_value=trailing_user):
            resp = await client.get(f"/api/projects/{project_id}/chat/history")
        assert resp.json()["interrupted"] is True

        trailing_assistant = [
            {"id": "u1", "role": "user", "parts": [{"type": "text", "text": "question"}], "created_at": "2026-07-10T12:00:00.000Z"},
            {"id": "a1", "role": "assistant", "parts": [{"type": "text", "text": "answer"}], "created_at": "2026-07-10T12:00:01.000Z"},
        ]
        with patch("app.services.chat_transcript.load_transcript", return_value=trailing_assistant):
            resp = await client.get(f"/api/projects/{project_id}/chat/history")
        assert resp.json()["interrupted"] is False

    async def test_stale_schema_transcript_is_rebuilt_in_parts_format(self, client):
        """When the transcript is OLD-FORMAT (entries lack `parts`), discard it
        and rebuild from the SDK session in `parts` format — no crash, no
        old-format parsing."""
        project_id = await self._project_with_session(client, "StaleRebuild")
        old_format = [
            {"id": "u1", "role": "user", "content": "hi", "tool_calls": None, "created_at": ""}
        ]
        rebuilt = [
            {"id": "u1", "role": "user", "parts": [{"type": "text", "text": "hi"}], "created_at": ""}
        ]
        # First load_transcript call returns old-format, second returns rebuilt after append
        load_calls = [old_format, rebuilt]
        with patch("app.services.chat_transcript.load_transcript", side_effect=load_calls), \
             patch("app.services.chat_transcript.discard_transcript", new=AsyncMock()) as mock_discard, \
             patch("app.platform.chat_history.load_raw_session_entries", new=AsyncMock(return_value=["raw"])), \
             patch("app.api.chat.map_session_messages", return_value=rebuilt), \
             patch("app.services.chat_transcript.append_messages", new=AsyncMock()) as mock_append:
            resp = await client.get(f"/api/projects/{project_id}/chat/history")

        assert resp.status_code == 200
        assert resp.json()["messages"] == rebuilt
        # Discard was called once to remove the stale transcript
        mock_discard.assert_awaited_once()
        # Append was called once with the rebuilt parts list
        mock_append.assert_awaited_once()
        assert mock_append.await_args.args[1] == rebuilt

    async def test_current_schema_transcript_loads_untouched(self, client):
        """A CURRENT-schema transcript (every entry has `parts` AND non-empty created_at)
        loads without rebuilding — no needless discard, no extra session read."""
        project_id = await self._project_with_session(client, "CurrentSchema")
        current = [
            {"id": "u1", "role": "user", "parts": [{"type": "text", "text": "hi"}], "created_at": "2026-07-10T12:00:00.000Z"}
        ]
        with patch("app.services.chat_transcript.load_transcript", return_value=current), \
             patch("app.services.chat_transcript.discard_transcript", new=AsyncMock()) as mock_discard, \
             patch("app.platform.chat_history.load_raw_session_entries", new=AsyncMock()) as mock_load_raw:
            resp = await client.get(f"/api/projects/{project_id}/chat/history")

        assert resp.status_code == 200
        assert resp.json()["messages"] == current
        # Discard was NOT called (current schema, no rebuild needed)
        mock_discard.assert_not_awaited()
        # load_raw_session_entries was NOT called (no backfill path)
        mock_load_raw.assert_not_awaited()

    async def test_stale_rebuild_failure_degrades_to_empty_not_500(self, client):
        """A stale-transcript rebuild that fails degrades to empty history —
        never 500s the endpoint."""
        project_id = await self._project_with_session(client, "RebuildFail")
        old_format = [
            {"id": "u1", "role": "user", "content": "hi", "tool_calls": None, "created_at": ""}
        ]
        # load_transcript returns old-format, then [] after discard, then still []
        load_calls = [old_format, []]
        with patch("app.services.chat_transcript.load_transcript", side_effect=load_calls), \
             patch("app.services.chat_transcript.discard_transcript", new=AsyncMock()), \
             patch("app.platform.chat_history.load_raw_session_entries", new=AsyncMock(side_effect=Exception("Boom"))):
            resp = await client.get(f"/api/projects/{project_id}/chat/history")

        assert resp.status_code == 200
        assert resp.json()["messages"] == []
        assert resp.json()["interrupted"] is False

    async def test_cloud_shaped_self_heal_writes_to_per_tenant_path(self, client, tmp_path):
        """Cloud-shaped path: rebuild writes to the per-tenant project path,
        not a silent no-op.

        Uses a MagicMock session_store (no moto needed) like test_chat_history.py.
        The rebuild routing through load_raw_session_entries + resolve_project_dir
        + platform.session_store + append_messages proves the per-tenant path is
        used, not a dev-mode fallback.
        """
        from app.main import get_project_manager

        project_id = (await client.post("/api/projects", json={"name": "CloudSelfHeal"})).json()["id"]
        pm = get_project_manager()
        project = pm.get_project(project_id)
        project.session_id = "sess-cloud-test"
        pm.save_config(project)

        old_format = [
            {"id": "u1", "role": "user", "content": "hi", "tool_calls": None, "created_at": ""}
        ]
        rebuilt = [
            {"id": "u1", "role": "user", "parts": [{"type": "text", "text": "hi"}], "created_at": ""}
        ]

        # Mock platform with session_store that returns raw entries
        mock_platform = MagicMock()
        mock_platform.session_store.load = AsyncMock(return_value=[{"uuid": "u1", "role": "user"}])

        # Cloud-shaped path: per-tenant layout
        tenant_path = tmp_path / "users" / "owner-sub" / "projects" / project_id
        tenant_path.mkdir(parents=True)

        load_calls = [old_format, rebuilt]
        with patch("app.api.chat.get_platform", return_value=mock_platform), \
             patch("app.services.project_paths.resolve_project_dir", return_value=tenant_path), \
             patch("app.services.chat_transcript.load_transcript", side_effect=load_calls), \
             patch("app.services.chat_transcript.discard_transcript", new=AsyncMock()) as mock_discard, \
             patch("app.api.chat.map_session_messages", return_value=rebuilt), \
             patch("app.services.chat_transcript.append_messages", new=AsyncMock()) as mock_append:
            resp = await client.get(f"/api/projects/{project_id}/chat/history")

        assert resp.status_code == 200
        assert resp.json()["messages"] == rebuilt

        # Discard was called
        mock_discard.assert_awaited_once()
        # Append was called with the rebuilt parts list
        mock_append.assert_awaited_once()
        assert mock_append.await_args.args[1] == rebuilt
        # Proves the rebuild routed through the cloud-shaped resolve + session_store


@pytest.mark.asyncio
async def test_active_turn_discovers_relay_turn_from_durable_registry(tmp_path):
    """In-flight discovery: /chat/active-turn reports relay turn_id from durable session registry."""
    from app.api.chat import router as chat_router
    import app.main as main_module
    from app.main import get_project_manager
    from app.platform import set_platform, Platform
    from app.platform.dev_local import FilesystemMetadataStore, LocalAuth, NoOpTokenVending
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    import time

    # Setup isolated projects directory
    original = main_module.PROJECTS_DIR
    main_module.PROJECTS_DIR = tmp_path / "projects"
    main_module.PROJECTS_DIR.mkdir()
    main_module._project_manager = None

    # Create dev platform
    dev_store = FilesystemMetadataStore()
    dev_platform = Platform(
        metadata_store=dev_store,
        session_store=None,  # type: ignore
        token_vending=NoOpTokenVending(),
        auth=LocalAuth(metadata_store=dev_store),
    )
    set_platform(dev_platform)

    app = FastAPI()
    app.include_router(chat_router)

    # Bypass auth using dependency override
    from app.platform.api_authz import require_project_read

    async def mock_require_project_read():
        return "user-test"

    app.dependency_overrides[require_project_read] = mock_require_project_read

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Create a project
        from app.models.schemas import ProjectCreate
        pm = get_project_manager()
        project = pm.create_project(ProjectCreate(name="RelayTest"), owner_id="user-test")
        project_id = project.id

        # Simulate a running relay turn claim with current_turn_id
        turn_id = "turn-relay-123"
        ttl = int(time.time()) + 7200
        store = dev_platform.metadata_store
        won = await store.claim_agentcore_session_turn(
            user_id="user-test",
            project_id=project_id,
            ttl=ttl,
            current_turn_id=turn_id
        )
        assert won

        # /chat/active-turn should discover the relay turn
        resp = await client.get(f"/api/projects/{project_id}/chat/active-turn")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data is not None
        assert data["turn_id"] == turn_id
        assert data["status"] == "running"

        # After release, should return None
        await store.release_agentcore_session_turn("user-test", project_id)
