# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test: POST /api/chat → envelopes appear on /ws.

This is the single integration test that proves the full
SSE-replacement path: an HTTP request kicks off a real agent turn, the
turn runner emits envelopes through ``ws_hub.publish_to_user``, and
those envelopes reach the SPA's WebSocket. We mock the Claude SDK so
the turn finishes deterministically without a real Bedrock call —
identical pattern to ``test_agent_service.py``.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import create_app
from app.platform import set_platform
from app.services.agent_turn import get_agent_turn_tracker
from app.services.ws_envelope import reset_counters
from app.services.ws_hub import get_ws_hub
from tests.conftest import authed_ws


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Wipe cross-test state on the runner / hub / envelope counters."""
    reset_counters()
    get_ws_hub()._reset()
    get_agent_turn_tracker()._reset()
    yield
    reset_counters()
    get_ws_hub()._reset()
    get_agent_turn_tracker()._reset()


@pytest.fixture
def app(tmp_path: Path, monkeypatch):
    """Boot the real app against a tmp projects dir + dev mode platform."""
    monkeypatch.setenv("EVAL_STUDIO_MODE", "dev")
    monkeypatch.setenv("LOCAL_DEV_USER_ID", "e2e-user")
    set_platform(None)  # force fresh build under the new env

    original = main_module.PROJECTS_DIR
    main_module.PROJECTS_DIR = tmp_path / "projects"
    main_module.PROJECTS_DIR.mkdir()
    main_module._project_manager = None

    application = create_app()
    yield application

    main_module._project_manager = None
    main_module.PROJECTS_DIR = original
    set_platform(None)


def _make_text_event(text: str):
    """SDK StreamEvent that emits a text_delta block."""
    m = MagicMock()
    m.event = {
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": text},
    }
    m.session_id = None
    m.num_turns = None
    m.parent_tool_use_id = None
    return m


def _make_result(session_id: str = "sess-e2e"):
    """SDK ResultMessage marking a successful turn."""
    m = MagicMock()
    m.subtype = "success"
    m.result = "Done"
    m.session_id = session_id
    m.total_cost_usd = 0.0
    m.num_turns = 1
    m.event = None
    m.content = None
    m.tool_use_result = None
    return m


def _build_mock_sdk(messages: list):
    """Build a MagicMock claude_agent_sdk that yields ``messages``."""

    async def fake_receive():
        for m in messages:
            yield m

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.receive_response = MagicMock(return_value=fake_receive())

    mock_sdk = MagicMock()
    mock_sdk.ClaudeSDKClient = MagicMock(return_value=mock_client)
    mock_sdk.ClaudeAgentOptions = MagicMock()
    return mock_sdk


def test_chat_post_emits_envelopes_on_ws(app):
    """POST /api/chat → background turn → envelopes on /ws.

    The frontend is the only consumer that observes these envelopes in
    production; here we open a TestClient WS, fire the POST, and assert
    we receive at least one ``text_delta`` plus the final ``result``,
    and that ``seq`` is strictly monotonic across them.
    """
    client = TestClient(app)
    # Create a project the chat endpoint can write to.
    create_resp = client.post("/api/projects", json={"name": "E2E Test"})
    assert create_resp.status_code in (200, 201)
    project_id = create_resp.json()["id"]

    # Mock the Claude SDK so the turn completes without Bedrock.
    text_evt_a = _make_text_event("Hello ")
    text_evt_b = _make_text_event("world")
    result_msg = _make_result()
    mock_sdk = _build_mock_sdk([text_evt_a, text_evt_b, result_msg])

    with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}), \
         patch("app.services.agent_runner._persist_session", new_callable=AsyncMock):
        with authed_ws(client) as ws:
            chat_resp = client.post(
                "/api/chat",
                json={"message": "hi", "project_id": project_id},
            )
            assert chat_resp.status_code == 200
            turn_id = chat_resp.json()["turn_id"]

            # Drain envelopes up to a small budget. The runner emits a
            # leading ``status`` envelope (``state="connecting"``), then
            # the two text deltas, then the final ``result``. We only
            # need to see one text_delta + one result to prove the path.
            collected: list[dict] = []
            for _ in range(20):
                env = ws.receive_json()
                collected.append(env)
                if env["kind"] == "result":
                    break

    # The turn's stream id is derived from the turn_id.
    turn_stream = f"turn:{turn_id}"
    turn_envelopes = [e for e in collected if e["stream"] == turn_stream]
    assert len(turn_envelopes) >= 2, (
        f"expected >=2 envelopes for {turn_stream}, got {turn_envelopes}"
    )
    # At least one text_delta and one result.
    kinds = {e["kind"] for e in turn_envelopes}
    assert "text_delta" in kinds
    assert "result" in kinds

    # Seq strictly monotonic within the turn stream.
    seqs = [e["seq"] for e in turn_envelopes]
    assert seqs == sorted(seqs), f"seq not monotonic: {seqs}"
    assert len(set(seqs)) == len(seqs), f"seq not unique: {seqs}"
