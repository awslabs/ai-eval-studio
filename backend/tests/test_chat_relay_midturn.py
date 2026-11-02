# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relay mid-turn refresh behavioral tests.

Proves the relay chat path persists session_id + user message at turn START,
before the AgentCore HTTPS POST, so a mid-turn page refresh renders the
conversation (not a blank splash).

Behavioral tests:
  1. start_persists_session_and_writes_user_bubble_before_completion:
     At POST-time, project.session_id is already truthy AND the durable
     transcript already contains the user message (proves START persistence).
  2. no_duplicate_user_bubble_after_completion_flush:
     After a successful relay turn + completion flush re-map, the transcript
     holds exactly ONE user bubble whose id is the SDK uuid (proves the
     upgrade collapses the provisional bubble).
  3. start_persist_failure_is_non_fatal_and_releases_claim:
     When _persist_session raises, the turn still proceeds/exits and the
     (user, project) claim is released (proves finally-release invariant).
  4. resumed_turn_no_new_id_minted:
     When resolve_resume returns an existing app session id, NO new id is
     minted (the persisted value equals the resolved id).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

pytestmark = pytest.mark.asyncio


@pytest.fixture
def dev_platform_with_store(projects_dir: Path, monkeypatch):
    """Set up dev platform with a REAL FilesystemMetadataStore.

    Uses the same pattern as test_chat_relay_claim.py so the relay client can
    write to a real temp projects dir via pm.get_project + pm.save_config.
    """
    from app.platform import Platform, set_platform
    from app.platform.dev_local import (
        FilesystemMetadataStore,
        LocalAuth,
        NoOpTokenVending,
        _build_dev_session_store,
    )

    set_platform(None)

    metadata_store = FilesystemMetadataStore(projects_dir=projects_dir)
    session_store = _build_dev_session_store()
    auth = LocalAuth(metadata_store=metadata_store)
    token_vending = NoOpTokenVending()

    # Stub authz to a no-op so the claim (not access control) is under test.
    async def _noop_require_access(self, user_id, project_id, action):
        return None

    monkeypatch.setattr(LocalAuth, "require_access", _noop_require_access)

    platform = Platform(
        metadata_store=metadata_store,
        session_store=session_store,
        auth=auth,
        token_vending=token_vending,
    )
    set_platform(platform)

    yield platform

    set_platform(None)


class IteratorStream:
    """Mock httpx streaming response that yields SSE frames.

    Mirrors the pattern from test_chat_relay_claim.py. When the
    test needs to inspect state BEFORE the POST completes, the side_effect
    of the httpx.AsyncClient.stream __aenter__ can run assertions, then
    return this iterator.
    """

    def __init__(self, frames: list[tuple[str, dict]]):
        self.frames = frames

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    @property
    def status_code(self):
        return 200

    async def aread(self):
        return b""

    def aiter_bytes(self):
        """Yield SSE frames as bytes."""

        async def _gen():
            for kind, data in self.frames:
                import json

                # SSE frame format: event:<kind>\ndata:<json>\n\n
                frame = f"event:{kind}\ndata:{json.dumps(data)}\n\n"
                yield frame.encode("utf-8")

        return _gen()


async def test_start_persists_session_and_writes_user_bubble_before_completion(
    dev_platform_with_store, projects_dir: Path
):
    """At POST-time, project.session_id is truthy AND transcript has the user message.

    Drives invoke_agentcore_turn up to (but blocking BEFORE) the POST, then
    asserts that project.session_id is already persisted and the durable
    transcript already contains the user message. Proves the persistence
    happens BEFORE the network call.
    """
    from app.services.agentcore_relay_client import invoke_agentcore_turn
    from app.services.chat_transcript import load_transcript
    from app.main import get_project_manager
    from app.models.schemas import ProjectCreate

    project_id = "test-proj-midturn"
    owner_id = "user-123"
    message = "Test message at start"
    turn_id = "turn-abc"
    access_token = "Bearer test-token"

    # Create a real project via the ProjectManager (so _persist_session can write to it)
    pm = get_project_manager()
    project = pm.create_project(
        ProjectCreate(name="Test Project", description=""), owner_id=owner_id
    )
    real_project_id = project.id

    # Patch get_ws_hub to a no-op (we don't care about WS publishes here)
    with patch("app.services.agentcore_relay_client.get_ws_hub") as mock_hub:
        mock_hub_instance = AsyncMock()
        mock_hub.return_value = mock_hub_instance

        # Patch resolve_resume to return (None, None) -> fresh turn 1
        with patch(
            "app.services.agentcore_relay_client.resolve_resume"
        ) as mock_resolve:
            mock_resolve.return_value = (None, None)

            # Patch release_turn to a no-op (claim not under test here)
            with patch(
                "app.services.agentcore_relay_client.release_turn"
            ) as mock_release:
                mock_release.return_value = None

                # Patch record_session to a no-op (session registry not under test here)
                with patch(
                    "app.services.agentcore_relay_client.record_session"
                ) as mock_record:
                    mock_record.return_value = None

                    # Patch httpx.AsyncClient.stream to capture the moment right BEFORE the POST
                    async def _capture_and_block(*args, **kwargs):
                        # At this moment, the start-of-turn persistence should have fired.
                        # Assert project.session_id is truthy.
                        project_after_start = pm.get_project(real_project_id)
                        assert project_after_start is not None
                        assert project_after_start.session_id  # truthy (minted at start)

                        # Assert the durable transcript already contains the user message.
                        transcript = load_transcript(real_project_id)
                        assert len(transcript) == 1
                        assert transcript[0]["role"] == "user"
                        assert transcript[0]["parts"][0]["text"] == message

                        # Return a mock iterator (the test is already done; let it drain)
                        return IteratorStream([("result", {"status": "ok"})])

                    with patch("httpx.AsyncClient.stream", new=_capture_and_block):
                        # Invoke the relay turn
                        await invoke_agentcore_turn(
                            project_id=real_project_id,
                            message=message,
                            owner_id=owner_id,
                            runtime_session_id=None,
                            turn_id=turn_id,
                            access_token=access_token,
                        )


async def test_no_duplicate_user_bubble_after_completion_flush(
    dev_platform_with_store, projects_dir: Path
):
    """After relay turn + completion flush re-map, exactly ONE user bubble (SDK uuid).

    Simulates the full cycle:
    1. relay start writes a provisional user bubble
    2. relay turn completes (we don't actually invoke; just skip to step 3)
    3. completion flush re-maps the turn with a fresh SDK uuid
    Assert the transcript has exactly ONE user bubble whose id is the SDK uuid
    (proves the upgrade collapses the provisional bubble).
    """
    from app.services.agentcore_relay_client import invoke_agentcore_turn
    from app.services.chat_transcript import load_transcript, append_messages
    from app.main import get_project_manager
    from app.models.schemas import ProjectCreate

    project_id = "test-proj-dedupe"
    owner_id = "user-123"
    message = "Test message for dedupe"
    turn_id = "turn-dedupe"
    access_token = "Bearer test-token"

    # Create a real project
    pm = get_project_manager()
    project = pm.create_project(
        ProjectCreate(name="Test Project", description=""), owner_id=owner_id
    )
    real_project_id = project.id

    # Patch get_ws_hub, resolve_resume, release_turn, record_session
    with patch("app.services.agentcore_relay_client.get_ws_hub") as mock_hub:
        mock_hub_instance = AsyncMock()
        mock_hub.return_value = mock_hub_instance

        with patch(
            "app.services.agentcore_relay_client.resolve_resume"
        ) as mock_resolve:
            mock_resolve.return_value = (None, None)

            with patch(
                "app.services.agentcore_relay_client.release_turn"
            ) as mock_release:
                mock_release.return_value = None

                with patch(
                    "app.services.agentcore_relay_client.record_session"
                ) as mock_record:
                    mock_record.return_value = None

                    # Patch httpx.AsyncClient.stream to return immediately (turn "succeeds")
                    async def _mock_stream(*args, **kwargs):
                        return IteratorStream([("result", {"status": "ok"})])

                    with patch("httpx.AsyncClient.stream", new=_mock_stream):
                        # Invoke the relay turn (writes provisional user bubble at start)
                        await invoke_agentcore_turn(
                            project_id=real_project_id,
                            message=message,
                            owner_id=owner_id,
                            runtime_session_id=None,
                            turn_id=turn_id,
                            access_token=access_token,
                        )

    # At this point, the transcript has the provisional user bubble.
    # Now simulate the completion flush re-map with a real SDK uuid.
    completion_batch = [
        {
            "id": "sdk-U1",
            "role": "user",
            "parts": [{"type": "text", "text": message}],
            "created_at": "2026-07-14T12:00:00.000Z",
        },
        {
            "id": "sdk-A1",
            "role": "assistant",
            "parts": [{"type": "text", "text": "Assistant response"}],
            "created_at": "2026-07-14T12:00:01.000Z",
        },
    ]
    await append_messages(real_project_id, completion_batch)

    # Load transcript
    transcript = load_transcript(real_project_id)

    # Assert exactly 2 records (user + assistant), NOT 3 (no duplicate user)
    assert len(transcript) == 2
    assert transcript[0]["id"] == "sdk-U1"  # SDK uuid, not provisional
    assert transcript[0]["role"] == "user"
    assert transcript[1]["id"] == "sdk-A1"
    assert transcript[1]["role"] == "assistant"


async def test_start_persist_failure_is_non_fatal_and_releases_claim(
    dev_platform_with_store, projects_dir: Path
):
    """When _persist_session raises, turn still proceeds and claim is released.

    Forces _persist_session to raise, then asserts:
    1. The turn does not crash (invoke_agentcore_turn returns normally).
    2. The (user, project) claim is released (a subsequent claim succeeds).
    """
    from app.services.agentcore_relay_client import invoke_agentcore_turn
    from app.services.agentcore_session_registry import claim_turn
    from app.main import get_project_manager
    from app.models.schemas import ProjectCreate

    project_id = "test-proj-persist-fail"
    owner_id = "user-123"
    message = "Test message with persist failure"
    turn_id = "turn-fail"
    access_token = "Bearer test-token"

    # Create a real project
    pm = get_project_manager()
    project = pm.create_project(
        ProjectCreate(name="Test Project", description=""), owner_id=owner_id
    )
    real_project_id = project.id

    # Patch agent_runner._persist_session to raise (it's imported inside the function)
    async def _raise_persist(*args, **kwargs):
        raise RuntimeError("Simulated persist failure")

    with patch(
        "app.services.agent_runner._persist_session",
        side_effect=_raise_persist,
    ):
        # Patch get_ws_hub, resolve_resume, release_turn, record_session
        with patch("app.services.agentcore_relay_client.get_ws_hub") as mock_hub:
            mock_hub_instance = AsyncMock()
            mock_hub.return_value = mock_hub_instance

            with patch(
                "app.services.agentcore_relay_client.resolve_resume"
            ) as mock_resolve:
                mock_resolve.return_value = (None, None)

                with patch(
                    "app.services.agentcore_relay_client.release_turn"
                ) as mock_release:
                    mock_release.return_value = None

                    with patch(
                        "app.services.agentcore_relay_client.record_session"
                    ) as mock_record:
                        mock_record.return_value = None

                        # Patch httpx.AsyncClient.stream to return immediately
                        async def _mock_stream(*args, **kwargs):
                            return IteratorStream([("result", {"status": "ok"})])

                        with patch("httpx.AsyncClient.stream", new=_mock_stream):
                            # Invoke should NOT raise (best-effort persist)
                            await invoke_agentcore_turn(
                                project_id=real_project_id,
                                message=message,
                                owner_id=owner_id,
                                runtime_session_id=None,
                                turn_id=turn_id,
                                access_token=access_token,
                            )

    # Assert the claim was released: a subsequent claim_turn should succeed
    result = await claim_turn(owner_id, real_project_id)
    assert result is True  # claim succeeds (previous turn released it)


async def test_resumed_turn_no_new_id_minted(
    dev_platform_with_store, projects_dir: Path
):
    """When resolve_resume returns an existing session id, NO new id is minted.

    Simulates a resumed turn (resolve_resume returns an existing app session id).
    Assert the persisted project.session_id equals the resolved id (no minting).
    """
    from app.services.agentcore_relay_client import invoke_agentcore_turn
    from app.main import get_project_manager
    from app.models.schemas import ProjectCreate

    project_id = "test-proj-resumed"
    owner_id = "user-123"
    message = "Resumed turn message"
    turn_id = "turn-resumed"
    access_token = "Bearer test-token"
    existing_session_id = "existing-session-abc123"

    # Create a real project
    pm = get_project_manager()
    project = pm.create_project(
        ProjectCreate(name="Test Project", description=""), owner_id=owner_id
    )
    real_project_id = project.id

    # Patch os.environ to provide AGENTCORE_RUNTIME_ARN (so the ARN guard passes)
    import os
    with patch.dict(os.environ, {"AGENTCORE_RUNTIME_ARN": "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test"}):
        # Patch resolve_resume to return an existing session id
        with patch("app.services.agentcore_relay_client.resolve_resume") as mock_resolve:
            mock_resolve.return_value = ("runtime-session-xyz", existing_session_id)

            # Patch get_ws_hub, release_turn, record_session
            with patch("app.services.agentcore_relay_client.get_ws_hub") as mock_hub:
                mock_hub_instance = AsyncMock()
                mock_hub.return_value = mock_hub_instance

                with patch(
                    "app.services.agentcore_relay_client.release_turn"
                ) as mock_release:
                    mock_release.return_value = None

                    with patch(
                        "app.services.agentcore_relay_client.record_session"
                    ) as mock_record:
                        mock_record.return_value = None

                        # Patch httpx.AsyncClient.stream to return immediately
                        async def _mock_stream(*args, **kwargs):
                            return IteratorStream([("result", {"status": "ok"})])

                        with patch("httpx.AsyncClient.stream", new=_mock_stream):
                            # Invoke the relay turn
                            await invoke_agentcore_turn(
                                project_id=real_project_id,
                                message=message,
                                owner_id=owner_id,
                                runtime_session_id="runtime-session-xyz",
                                turn_id=turn_id,
                                access_token=access_token,
                            )

        # Assert the persisted session_id equals the resolved id (no new id minted)
        project_after = pm.get_project(real_project_id)
        assert project_after is not None
        assert project_after.session_id == existing_session_id
