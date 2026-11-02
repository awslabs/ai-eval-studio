# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AgentCore relay client and flag resolver.

This module tests:
- relay flag resolver (an environment check: cloud resolves True, dev False;
  the old kill-switch and per-project flag are ignored)
- invoke_agentcore_turn (server-side AgentCore invocation + SSE drain + republish)
- POST /chat relay gate (with the relay on, the relay path replaces the
  in-process turn)
"""
import os
import uuid
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.platform.arch_split import resolve_agentcore_relay


class TestResolveAgentcoreRelay:
    """The relay resolver is an ENVIRONMENT check — cloud=True, dev=False.

    The per-project agentcore_relay_enabled field and the
    AGENTCORE_RELAY_KILL_SWITCH are DEAD (the same pattern that killed
    arch_split's per-project read): stored values are ignored, no store is
    consulted. Forcing cloud turns onto the in-process path would be an
    untested escape hatch; the real rollback is redeploying."""

    @pytest.mark.asyncio
    async def test_dev_resolves_false(self):
        with patch.dict(os.environ, {"EVAL_STUDIO_MODE": "dev"}):
            assert await resolve_agentcore_relay("proj-any") is False

    @pytest.mark.asyncio
    async def test_unset_mode_resolves_false(self):
        with patch.dict(os.environ, {}, clear=True):
            assert await resolve_agentcore_relay("proj-any") is False

    @pytest.mark.asyncio
    async def test_cloud_resolves_true(self):
        with patch.dict(os.environ, {"EVAL_STUDIO_MODE": "cloud"}):
            assert await resolve_agentcore_relay("proj-any") is True

    @pytest.mark.asyncio
    async def test_cloud_ignores_stored_false_and_dead_kill_switch(self):
        """Regression pin: older records may carry explicit False, and old
        runbooks may set the kill-switch — neither may route a cloud turn
        onto the in-process path again."""
        with patch.dict(
            os.environ,
            {"EVAL_STUDIO_MODE": "cloud", "AGENTCORE_RELAY_KILL_SWITCH": "1"},
        ):
            assert await resolve_agentcore_relay("proj-with-stored-false") is True

    @pytest.mark.asyncio
    async def test_cloud_performs_no_store_read(self):
        """The resolver must not touch the platform/store at all — any read
        reintroduces per-project routing and its owner-resolution bugs."""
        def _explode():
            raise AssertionError("resolve_agentcore_relay must not read the platform")

        with patch.dict(os.environ, {"EVAL_STUDIO_MODE": "cloud"}), \
             patch("app.platform.arch_split.get_platform", _explode):
            assert await resolve_agentcore_relay("proj-any") is True


class _ChunkedByteIterator:
    """Async byte iterator that delivers bytes in fixed-size chunks.

    Tracks how many chunks have been pulled so a test can prove frames are
    yielded BEFORE the whole stream is exhausted.
    """

    def __init__(self, data: bytes, chunk_size: int):
        self._data = data
        self._chunk_size = chunk_size
        self._pos = 0
        self.chunks_pulled = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._pos >= len(self._data):
            raise StopAsyncIteration

        limit = min(self._chunk_size, len(self._data) - self._pos)
        chunk = self._data[self._pos:self._pos + limit]
        self._pos += len(chunk)
        self.chunks_pulled += 1
        return chunk



class TestParseSseFramesIncremental:
    """Relay must parse the async byte stream incrementally, not whole-body."""

    @pytest.mark.asyncio
    async def test_frames_yielded_incrementally_before_stream_exhausted(self):
        """Frames are yielded as boundaries arrive — first frame before stream exhausted.

        Against the buggy whole-body read, the entire stream is drained before ANY
        frame is yielded. The incremental read yields the first frame while bytes
        remain unread.
        """
        from app.services.agentcore_relay_client import _parse_sse_frames

        canned_sse = (
            b"event: text_delta\n"
            b'data: {"delta": "Hello"}\n'
            b"\n"
            b"event: text_delta\n"
            b'data: {"delta": " world"}\n'
            b"\n"
            b"event: result\n"
            b'data: {"status": "completed"}\n'
            b"\n"
        )
        # Small chunks so the stream is NOT fully consumed at the first frame.
        byte_iter = _ChunkedByteIterator(canned_sse, chunk_size=8)

        total_chunks = -(-len(canned_sse) // 8)  # Ceiling division
        frames = []
        chunks_at_first_frame = None
        async for kind, data in _parse_sse_frames(byte_iter):
            if chunks_at_first_frame is None:
                chunks_at_first_frame = byte_iter.chunks_pulled
            frames.append((kind, data))

        # All three frames parsed correctly
        assert frames == [
            ("text_delta", {"delta": "Hello"}),
            ("text_delta", {"delta": " world"}),
            ("result", {"status": "completed"}),
        ]
        # The first frame must have been yielded BEFORE the whole stream was exhausted.
        assert chunks_at_first_frame < total_chunks, (
            f"First frame was yielded only after all {total_chunks} chunks were pulled — "
            f"relay is not streaming. Chunks at first yield: {chunks_at_first_frame}"
        )

    @pytest.mark.asyncio
    async def test_frame_split_across_chunk_boundary_parses(self):
        """A frame whose bytes span multiple chunks still parses correctly."""
        from app.services.agentcore_relay_client import _parse_sse_frames

        canned_sse = (
            b"event: text_delta\n"
            b'data: {"delta": "split-across-boundary"}\n'
            b"\n"
            b"event: result\n"
            b'data: {"ok": true}\n'
            b"\n"
        )
        # chunk_size deliberately tiny (3 bytes) so every frame is split many times.
        byte_iter = _ChunkedByteIterator(canned_sse, chunk_size=3)

        frames = []
        async for kind, data in _parse_sse_frames(byte_iter):
            frames.append((kind, data))

        assert frames == [
            ("text_delta", {"delta": "split-across-boundary"}),
            ("result", {"ok": True}),
        ]


class TestInvokeAgentcoreTurn:
    """Test the invoke_agentcore_turn relay function."""

    @pytest.mark.asyncio
    async def test_invoke_parses_sse_frames_and_republishes(self, httpx_mock):
        """Canned SSE frames are parsed and republished on stream turn:<id>.

        Also pins the raw-HTTPS httpx request the relay sends: URL, headers
        and body.
        """
        import json
        from unittest.mock import MagicMock, patch
        from urllib.parse import quote
        import pytest_httpx

        from app.services.agentcore_relay_client import invoke_agentcore_turn

        # Canned SSE response (matches agentcore_runtime.py _sse_frames format)
        canned_sse = (
            b"event: text_delta\n"
            b'data: {"delta": "Hello"}\n'
            b"\n"
            b"event: text_delta\n"
            b'data: {"delta": " world"}\n'
            b"\n"
            b"event: result\n"
            b'data: {"status": "completed"}\n'
            b"\n"
        )

        # Mock httpx response with SSE stream
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent-runtime/test"
        expected_url = f"https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/{quote(arn, safe='')}/invocations?qualifier=DEFAULT"

        httpx_mock.add_response(
            method="POST",
            url=expected_url,
            status_code=200,
            stream=pytest_httpx.IteratorStream([canned_sse]),
            headers={"Content-Type": "text/event-stream"},
        )

        # Mock ws_hub.publish_to_user
        published_envelopes = []

        async def mock_publish(owner_id, envelope):
            published_envelopes.append((owner_id, envelope))

        with patch(
            "app.services.agentcore_relay_client.get_ws_hub"
        ) as mock_hub_getter:
            mock_hub = MagicMock()
            mock_hub.publish_to_user = mock_publish
            mock_hub_getter.return_value = mock_hub

            # Mock durable registry helpers (resolve_resume returns a tuple;
            # record_session + release_turn are AsyncMocks so we don't hit the store)
            with patch(
                "app.services.agentcore_relay_client.resolve_resume",
                AsyncMock(return_value=(None, None)),
            ), patch(
                "app.services.agentcore_relay_client.record_session",
                AsyncMock(),
            ), patch(
                "app.services.agentcore_relay_client.release_turn",
                AsyncMock(),
            ):
                # Mock env vars
                with patch.dict(
                    os.environ,
                    {
                        "AGENTCORE_RUNTIME_ARN": arn,
                        "AWS_REGION": "us-east-1",
                    },
                ):
                    # Invoke the relay with access_token
                    await invoke_agentcore_turn(
                        project_id="proj-test",
                        message="test message",
                        owner_id="user-123",
                        runtime_session_id=None,
                        turn_id="turn-abc",
                        access_token="Bearer test-token-123",
                    )

        # Verify the HTTP request shape
        requests = httpx_mock.get_requests()
        assert len(requests) == 1
        request = requests[0]

        # Request line
        assert request.method == "POST"
        assert str(request.url) == expected_url

        # Headers
        assert request.headers["Authorization"] == "Bearer test-token-123"
        assert request.headers["Accept"] == "text/event-stream"
        assert request.headers["Content-Type"] == "application/json"
        session_id_header = request.headers.get("X-Amzn-Bedrock-AgentCore-Runtime-Session-Id")
        assert session_id_header is not None
        assert len(session_id_header) >= 33

        # Body
        body = json.loads(request.content.decode())
        assert body["message"] == "test message"
        assert body["project_id"] == "proj-test"
        # session_id is now always minted/persisted at start (no longer omitted).
        # Canonical DASHED UUID form — the CLI rejects dashless ``--session-id``
        # ("Invalid session ID. Must be a valid UUID.").
        assert "session_id" in body
        assert isinstance(body["session_id"], str)
        assert body["session_id"] == str(uuid.UUID(body["session_id"]))

        # Verify three envelopes were published
        assert len(published_envelopes) == 3
        assert all(owner_id == "user-123" for owner_id, _ in published_envelopes)

        # Verify envelope structure
        _, env1 = published_envelopes[0]
        assert env1["stream"] == "turn:turn-abc"
        assert env1["kind"] == "text_delta"
        assert env1["data"] == {"delta": "Hello"}

        _, env2 = published_envelopes[1]
        assert env2["kind"] == "text_delta"
        assert env2["data"] == {"delta": " world"}

        _, env3 = published_envelopes[2]
        assert env3["kind"] == "result"
        assert env3["data"] == {"status": "completed"}

    @pytest.mark.asyncio
    async def test_resolve_resume_consulted(self, httpx_mock):
        """Resolve_resume(owner_id, project_id, runtime_session_id) is consulted before invoke."""
        import json
        from unittest.mock import patch
        from urllib.parse import quote
        import pytest_httpx

        from app.services.agentcore_relay_client import invoke_agentcore_turn

        # Canned SSE response
        canned_sse = b"event: result\ndata: {}\n\n"

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent-runtime/test"
        expected_url = f"https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/{quote(arn, safe='')}/invocations?qualifier=DEFAULT"

        httpx_mock.add_response(
            method="POST",
            url=expected_url,
            status_code=200,
            stream=pytest_httpx.IteratorStream([canned_sse]),
        )

        with patch(
            "app.services.agentcore_relay_client.get_ws_hub"
        ) as mock_hub_getter:
            mock_hub = Mock()
            mock_hub.publish_to_user = AsyncMock()
            mock_hub_getter.return_value = mock_hub

            # resolve_resume returns (runtime_session_id_to_use, app_session_id)
            mock_resume = AsyncMock(return_value=("rt-session-456", "session-resolved-123"))
            with patch(
                "app.services.agentcore_relay_client.resolve_resume",
                mock_resume,
            ), patch(
                "app.services.agentcore_relay_client.record_session",
                AsyncMock(),
            ), patch(
                "app.services.agentcore_relay_client.release_turn",
                AsyncMock(),
            ):
                with patch.dict(
                    os.environ,
                    {
                        "AGENTCORE_RUNTIME_ARN": arn,
                        "AWS_REGION": "us-east-1",
                    },
                ):
                    await invoke_agentcore_turn(
                        project_id="proj-test",
                        message="test message",
                        owner_id="user-123",
                        runtime_session_id="rt-session-456",
                        turn_id="turn-abc",
                        access_token="Bearer test-token",
                    )

        # Verify resolve_resume was called with the right args (owner_id, project_id, runtime_session_id)
        mock_resume.assert_awaited_once_with("user-123", "proj-test", "rt-session-456")

        # Verify the resolved session_id was included in the payload
        requests = httpx_mock.get_requests()
        assert len(requests) == 1
        payload = json.loads(requests[0].content.decode())
        assert payload["session_id"] == "session-resolved-123"

    @pytest.mark.asyncio
    async def test_first_turn_generates_conforming_runtime_session_id(self, httpx_mock):
        """First turn (no session id) → non-empty X-Amzn-Bedrock-AgentCore-Runtime-Session-Id, length >= 33.

        Sending "" is rejected by AgentCore (documented min length ~33) and the
        ValidationException is swallowed → first relay turn silently produces
        nothing. The relay must generate a fresh conforming id instead.
        """
        from unittest.mock import patch
        from urllib.parse import quote
        import pytest_httpx

        from app.services.agentcore_relay_client import invoke_agentcore_turn

        canned_sse = b"event: result\ndata: {}\n\n"

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent-runtime/test"
        expected_url = f"https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/{quote(arn, safe='')}/invocations?qualifier=DEFAULT"

        httpx_mock.add_response(
            method="POST",
            url=expected_url,
            status_code=200,
            stream=pytest_httpx.IteratorStream([canned_sse]),
        )

        with patch(
            "app.services.agentcore_relay_client.get_ws_hub"
        ) as mock_hub_getter:
            mock_hub = Mock()
            mock_hub.publish_to_user = AsyncMock()
            mock_hub_getter.return_value = mock_hub

            with patch(
                "app.services.agentcore_relay_client.resolve_resume",
                AsyncMock(return_value=(None, None)),
            ), patch(
                "app.services.agentcore_relay_client.record_session",
                AsyncMock(),
            ), patch(
                "app.services.agentcore_relay_client.release_turn",
                AsyncMock(),
            ):
                with patch.dict(
                    os.environ,
                    {
                        "AGENTCORE_RUNTIME_ARN": arn,
                        "AWS_REGION": "us-east-1",
                    },
                ):
                    # First turn: runtime_session_id is None
                    await invoke_agentcore_turn(
                        project_id="proj-test",
                        message="test message",
                        owner_id="user-123",
                        runtime_session_id=None,
                        turn_id="turn-abc",
                        access_token="Bearer test-token",
                    )

        # The invoke must NOT send an empty X-Amzn-Bedrock-AgentCore-Runtime-Session-Id.
        requests = httpx_mock.get_requests()
        assert len(requests) == 1
        sent_session_id = requests[0].headers.get("X-Amzn-Bedrock-AgentCore-Runtime-Session-Id")
        assert sent_session_id, "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id must not be empty on first turn"
        assert len(sent_session_id) >= 33, (
            f"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id must be >= 33 chars (AgentCore min), "
            f"got {len(sent_session_id)}: {sent_session_id!r}"
        )

    @pytest.mark.asyncio
    async def test_first_turn_writes_invoke_session_id_back_to_relay_tasks(self, httpx_mock):
        """First-turn cancel: the real invoke_session_id lands in _relay_tasks.

        The spawn site seeds _relay_tasks[turn_id]["runtime_session_id"] with
        req.session_id, which is None on a first turn. The runtime knows the turn
        by the generated invoke_session_id (the header value). Without the
        write-back, stop_turn reads None and skips StopRuntimeSession — the
        cancelled microVM runs to Completed (observed: 305 events, $1.66). This
        asserts the entry holds the SAME real id that was sent in the header.
        """
        from unittest.mock import patch
        from urllib.parse import quote
        import pytest_httpx

        from app.api import chat
        from app.services.agentcore_relay_client import invoke_agentcore_turn

        canned_sse = b"event: result\ndata: {}\n\n"
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent-runtime/test"
        expected_url = f"https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/{quote(arn, safe='')}/invocations?qualifier=DEFAULT"
        httpx_mock.add_response(
            method="POST",
            url=expected_url,
            status_code=200,
            stream=pytest_httpx.IteratorStream([canned_sse]),
        )

        turn_id = "turn-firstcancel"
        # Seed the entry exactly as the POST /chat spawn site does on a first turn:
        # runtime_session_id is None (req.session_id was None).
        chat._relay_tasks[turn_id] = {
            "task": Mock(),
            "runtime_session_id": None,
            "project_id": "proj-test",
            "owner_id": "user-123",
            "access_token": "Bearer test-token",
        }
        try:
            with patch("app.services.agentcore_relay_client.get_ws_hub") as mock_hub_getter:
                mock_hub = Mock()
                mock_hub.publish_to_user = AsyncMock()
                mock_hub_getter.return_value = mock_hub
                with patch(
                    "app.services.agentcore_relay_client.resolve_resume",
                    AsyncMock(return_value=(None, None)),
                ), patch(
                    "app.services.agentcore_relay_client.record_session", AsyncMock(),
                ), patch(
                    "app.services.agentcore_relay_client.release_turn", AsyncMock(),
                ):
                    with patch.dict(
                        os.environ,
                        {"AGENTCORE_RUNTIME_ARN": arn, "AWS_REGION": "us-east-1"},
                    ):
                        await invoke_agentcore_turn(
                            project_id="proj-test",
                            message="first message",
                            owner_id="user-123",
                            runtime_session_id=None,
                            turn_id=turn_id,
                            access_token="Bearer test-token",
                        )

            # The header the runtime received.
            sent_session_id = httpx_mock.get_requests()[0].headers.get(
                "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"
            )
            # The entry must now hold the SAME real id — not None.
            stored = chat._relay_tasks[turn_id]["runtime_session_id"]
            assert stored is not None, (
                "invoke_session_id was not written back — stop_turn would skip "
                "StopRuntimeSession on a first-turn cancel"
            )
            assert stored == sent_session_id, (
                f"stored id {stored!r} must equal the header id {sent_session_id!r} "
                "so StopRuntimeSession targets the session the runtime knows"
            )
            assert len(stored) >= 33
        finally:
            chat._relay_tasks.pop(turn_id, None)

    @pytest.mark.asyncio
    async def test_every_republish_targets_owner_id_only(self, httpx_mock):
        """Every publish call targets owner_id only (cross-user isolation)."""
        from unittest.mock import patch
        from urllib.parse import quote
        import pytest_httpx

        from app.services.agentcore_relay_client import invoke_agentcore_turn

        # Canned SSE response with multiple frames
        canned_sse = (
            b"event: text_delta\n"
            b'data: {"delta": "A"}\n'
            b"\n"
            b"event: text_delta\n"
            b'data: {"delta": "B"}\n'
            b"\n"
            b"event: result\n"
            b'data: {}\n'
            b"\n"
        )

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent-runtime/test"
        expected_url = f"https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/{quote(arn, safe='')}/invocations?qualifier=DEFAULT"

        httpx_mock.add_response(
            method="POST",
            url=expected_url,
            status_code=200,
            stream=pytest_httpx.IteratorStream([canned_sse]),
        )

        published_users = []

        async def mock_publish(owner_id, envelope):
            published_users.append(owner_id)

        with patch(
            "app.services.agentcore_relay_client.get_ws_hub"
        ) as mock_hub_getter:
            mock_hub = Mock()
            mock_hub.publish_to_user = mock_publish
            mock_hub_getter.return_value = mock_hub

            with patch(
                "app.services.agentcore_relay_client.resolve_resume",
                AsyncMock(return_value=(None, None)),
            ), patch(
                "app.services.agentcore_relay_client.record_session",
                AsyncMock(),
            ), patch(
                "app.services.agentcore_relay_client.release_turn",
                AsyncMock(),
            ):
                with patch.dict(
                    os.environ,
                    {
                        "AGENTCORE_RUNTIME_ARN": arn,
                        "AWS_REGION": "us-east-1",
                    },
                ):
                    await invoke_agentcore_turn(
                        project_id="proj-test",
                        message="test message",
                        owner_id="user-owner",
                        runtime_session_id=None,
                        turn_id="turn-abc",
                        access_token="Bearer test-token",
                    )

        # Verify all publishes targeted the same owner_id
        assert len(published_users) == 3
        assert all(user == "user-owner" for user in published_users)

    @pytest.mark.asyncio
    async def test_tenant_scope_cross_user_isolation(self, httpx_mock):
        """A turn for owner U produces zero publishes to a different user."""
        from unittest.mock import patch
        from urllib.parse import quote
        import pytest_httpx

        from app.services.agentcore_relay_client import invoke_agentcore_turn

        # Canned SSE response
        canned_sse = b"event: result\ndata: {}\n\n"

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent-runtime/test"
        expected_url = f"https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/{quote(arn, safe='')}/invocations?qualifier=DEFAULT"

        httpx_mock.add_response(
            method="POST",
            url=expected_url,
            status_code=200,
            stream=pytest_httpx.IteratorStream([canned_sse]),
        )

        # Track all publish calls
        publish_calls = []

        async def mock_publish(owner_id, envelope):
            publish_calls.append((owner_id, envelope))

        with patch(
            "app.services.agentcore_relay_client.get_ws_hub"
        ) as mock_hub_getter:
            mock_hub = Mock()
            mock_hub.publish_to_user = mock_publish
            mock_hub_getter.return_value = mock_hub

            with patch(
                "app.services.agentcore_relay_client.resolve_resume",
                AsyncMock(return_value=(None, None)),
            ), patch(
                "app.services.agentcore_relay_client.record_session",
                AsyncMock(),
            ), patch(
                "app.services.agentcore_relay_client.release_turn",
                AsyncMock(),
            ):
                with patch.dict(
                    os.environ,
                    {
                        "AGENTCORE_RUNTIME_ARN": arn,
                        "AWS_REGION": "us-east-1",
                    },
                ):
                    # Invoke turn for user-A
                    await invoke_agentcore_turn(
                        project_id="proj-a",
                        message="test message",
                        owner_id="user-a",
                        runtime_session_id=None,
                        turn_id="turn-abc",
                        access_token="Bearer test-token",
                    )

        # Verify NO publishes went to user-b
        user_b_publishes = [call for call in publish_calls if call[0] == "user-b"]
        assert len(user_b_publishes) == 0

        # Verify publishes went ONLY to user-a
        user_a_publishes = [call for call in publish_calls if call[0] == "user-a"]
        assert len(user_a_publishes) == len(publish_calls)

    @pytest.mark.asyncio
    async def test_missing_token_makes_zero_http_calls(self, httpx_mock):
        """Missing access_token logs and returns with zero HTTP calls (non-fatal, same discipline as ARN guard)."""
        from unittest.mock import patch
        from urllib.parse import quote

        from app.services.agentcore_relay_client import invoke_agentcore_turn

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent-runtime/test"
        expected_url = f"https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/{quote(arn, safe='')}/invocations?qualifier=DEFAULT"

        # No httpx_mock.add_response — we're asserting zero HTTP calls

        with patch(
            "app.services.agentcore_relay_client.get_ws_hub"
        ) as mock_hub_getter:
            mock_hub = Mock()
            mock_hub.publish_to_user = AsyncMock()
            mock_hub.publish_project = AsyncMock()
            mock_hub_getter.return_value = mock_hub

            with patch(
                "app.services.agentcore_relay_client.resolve_resume",
                AsyncMock(return_value=(None, None)),
            ), patch(
                "app.services.agentcore_relay_client.record_session",
                AsyncMock(),
            ), patch(
                "app.services.agentcore_relay_client.release_turn",
                AsyncMock(),
            ):
                with patch.dict(
                    os.environ,
                    {
                        "AGENTCORE_RUNTIME_ARN": arn,
                        "AWS_REGION": "us-east-1",
                    },
                ):
                    # Invoke with access_token=None
                    await invoke_agentcore_turn(
                        project_id="proj-test",
                        message="test message",
                        owner_id="user-123",
                        runtime_session_id=None,
                        turn_id="turn-abc",
                        access_token=None,
                    )

        # Verify zero HTTP requests were made
        requests = httpx_mock.get_requests()
        assert len(requests) == 0








class TestRelayFailureTerminal:
    """Every relay failure exit publishes a terminal error envelope."""

    @pytest.mark.asyncio
    async def test_missing_arn_publishes_terminal_error(self):
        """Missing AGENTCORE_RUNTIME_ARN publishes terminal error envelope + project event."""
        from unittest.mock import patch, AsyncMock, Mock

        from app.services.agentcore_relay_client import invoke_agentcore_turn

        # Track publishes
        published_envelopes = []
        published_events = []

        async def mock_publish(owner_id, envelope):
            published_envelopes.append((owner_id, envelope))

        async def mock_publish_project(project_id, owner_id, event):
            published_events.append((project_id, owner_id, event))

        with patch(
            "app.services.agentcore_relay_client.get_ws_hub"
        ) as mock_hub_getter:
            mock_hub = Mock()
            mock_hub.publish_to_user = mock_publish
            mock_hub.publish_project = mock_publish_project
            mock_hub_getter.return_value = mock_hub

            with patch(
                "app.services.agentcore_relay_client.release_turn",
                AsyncMock(),
            ):
                with patch.dict(os.environ, {}, clear=True):
                    # Invoke with missing AGENTCORE_RUNTIME_ARN
                    await invoke_agentcore_turn(
                        project_id="proj-test",
                        message="test message",
                        owner_id="user-123",
                        runtime_session_id=None,
                        turn_id="turn-abc",
                        access_token="Bearer test-token",
                    )

        # Verify terminal error envelope on turn:<id>
        assert len(published_envelopes) == 1
        owner_id, envelope = published_envelopes[0]
        assert owner_id == "user-123"
        assert envelope["stream"] == "turn:turn-abc"
        assert envelope["kind"] == "result"
        assert envelope["data"]["subtype"] == "error"
        assert envelope["data"]["result"] is None
        assert envelope["data"]["error"] is not None
        assert "AGENTCORE_RUNTIME_ARN" in envelope["data"]["error"] or "runtime" in envelope["data"]["error"].lower()

        # Verify project error event
        assert len(published_events) == 1
        proj_id, owner_id, event = published_events[0]
        assert proj_id == "proj-test"
        assert owner_id == "user-123"
        assert event["type"] == "error"
        assert event["turn_id"] == "turn-abc"

    @pytest.mark.asyncio
    async def test_non_2xx_publishes_terminal_error(self, httpx_mock):
        """Non-2xx response publishes terminal error envelope + project event."""
        from unittest.mock import patch, AsyncMock, Mock
        from urllib.parse import quote
        import pytest_httpx

        from app.services.agentcore_relay_client import invoke_agentcore_turn

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent-runtime/test"
        expected_url = f"https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/{quote(arn, safe='')}/invocations?qualifier=DEFAULT"

        # Mock 503 response
        httpx_mock.add_response(
            method="POST",
            url=expected_url,
            status_code=503,
            content=b'{"message": "Service temporarily unavailable"}',
        )

        published_envelopes = []
        published_events = []

        async def mock_publish(owner_id, envelope):
            published_envelopes.append((owner_id, envelope))

        async def mock_publish_project(project_id, owner_id, event):
            published_events.append((project_id, owner_id, event))

        with patch(
            "app.services.agentcore_relay_client.get_ws_hub"
        ) as mock_hub_getter:
            mock_hub = Mock()
            mock_hub.publish_to_user = mock_publish
            mock_hub.publish_project = mock_publish_project
            mock_hub_getter.return_value = mock_hub

            with patch(
                "app.services.agentcore_relay_client.resolve_resume",
                AsyncMock(return_value=(None, None)),
            ), patch(
                "app.services.agentcore_relay_client.release_turn",
                AsyncMock(),
            ):
                with patch.dict(
                    os.environ,
                    {
                        "AGENTCORE_RUNTIME_ARN": arn,
                        "AWS_REGION": "us-east-1",
                    },
                ):
                    await invoke_agentcore_turn(
                        project_id="proj-test",
                        message="test message",
                        owner_id="user-123",
                        runtime_session_id=None,
                        turn_id="turn-abc",
                        access_token="Bearer test-token",
                    )

        # Verify terminal error envelope
        assert len(published_envelopes) == 1
        owner_id, envelope = published_envelopes[0]
        assert envelope["stream"] == "turn:turn-abc"
        assert envelope["kind"] == "result"
        assert envelope["data"]["subtype"] == "error"
        assert envelope["data"]["result"] is None
        assert "503" in envelope["data"]["error"] or "unavailable" in envelope["data"]["error"].lower()

        # Verify project error event
        assert len(published_events) == 1
        proj_id, owner_id, event = published_events[0]
        assert event["type"] == "error"

    @pytest.mark.asyncio
    async def test_http_error_publishes_terminal_error(self, httpx_mock):
        """Httpx.HTTPError publishes terminal error envelope + project event."""
        from unittest.mock import patch, AsyncMock, Mock
        from urllib.parse import quote
        import httpx

        from app.services.agentcore_relay_client import invoke_agentcore_turn

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent-runtime/test"
        expected_url = f"https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/{quote(arn, safe='')}/invocations?qualifier=DEFAULT"

        # Mock a connection error
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))

        published_envelopes = []
        published_events = []

        async def mock_publish(owner_id, envelope):
            published_envelopes.append((owner_id, envelope))

        async def mock_publish_project(project_id, owner_id, event):
            published_events.append((project_id, owner_id, event))

        with patch(
            "app.services.agentcore_relay_client.get_ws_hub"
        ) as mock_hub_getter:
            mock_hub = Mock()
            mock_hub.publish_to_user = mock_publish
            mock_hub.publish_project = mock_publish_project
            mock_hub_getter.return_value = mock_hub

            with patch(
                "app.services.agentcore_relay_client.resolve_resume",
                AsyncMock(return_value=(None, None)),
            ), patch(
                "app.services.agentcore_relay_client.release_turn",
                AsyncMock(),
            ):
                with patch.dict(
                    os.environ,
                    {
                        "AGENTCORE_RUNTIME_ARN": arn,
                        "AWS_REGION": "us-east-1",
                    },
                ):
                    await invoke_agentcore_turn(
                        project_id="proj-test",
                        message="test message",
                        owner_id="user-123",
                        runtime_session_id=None,
                        turn_id="turn-abc",
                        access_token="Bearer test-token",
                    )

        # Verify terminal error envelope
        assert len(published_envelopes) == 1
        owner_id, envelope = published_envelopes[0]
        assert envelope["stream"] == "turn:turn-abc"
        assert envelope["kind"] == "result"
        assert envelope["data"]["subtype"] == "error"
        assert envelope["data"]["result"] is None
        assert "connection" in envelope["data"]["error"].lower() or "refused" in envelope["data"]["error"].lower()

        # Verify project error event
        assert len(published_events) == 1

    @pytest.mark.asyncio
    async def test_auth_expiry_distinguishable_message(self, httpx_mock):
        """401 or expired token surfaces distinguishable 'session expired' message."""
        from unittest.mock import patch, AsyncMock, Mock
        from urllib.parse import quote
        import pytest_httpx

        from app.services.agentcore_relay_client import invoke_agentcore_turn

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent-runtime/test"
        expected_url = f"https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/{quote(arn, safe='')}/invocations?qualifier=DEFAULT"

        # Mock 401 response
        httpx_mock.add_response(
            method="POST",
            url=expected_url,
            status_code=401,
            content=b'{"message": "Unauthorized"}',
        )

        published_envelopes = []

        async def mock_publish(owner_id, envelope):
            published_envelopes.append((owner_id, envelope))

        with patch(
            "app.services.agentcore_relay_client.get_ws_hub"
        ) as mock_hub_getter:
            mock_hub = Mock()
            mock_hub.publish_to_user = mock_publish
            mock_hub.publish_project = AsyncMock()
            mock_hub_getter.return_value = mock_hub

            with patch(
                "app.services.agentcore_relay_client.resolve_resume",
                AsyncMock(return_value=(None, None)),
            ), patch(
                "app.services.agentcore_relay_client.release_turn",
                AsyncMock(),
            ):
                with patch.dict(
                    os.environ,
                    {
                        "AGENTCORE_RUNTIME_ARN": arn,
                        "AWS_REGION": "us-east-1",
                    },
                ):
                    await invoke_agentcore_turn(
                        project_id="proj-test",
                        message="test message",
                        owner_id="user-123",
                        runtime_session_id=None,
                        turn_id="turn-abc",
                        access_token="Bearer test-token",
                    )

        # Verify distinguishable auth-expiry message
        assert len(published_envelopes) == 1
        owner_id, envelope = published_envelopes[0]
        error_msg = envelope["data"]["error"].lower()
        assert "session" in error_msg or "expired" in error_msg or "sign in" in error_msg or "retry" in error_msg


class TestDurableRelayAppend:
    """Test durable per-frame append in relay drain loop."""

    @pytest.mark.asyncio
    async def test_each_frame_durably_appended_with_ddb_seq(self, httpx_mock):
        """Each drained frame is durably appended via append_turn_event with allocate_turn_seq seq."""
        from unittest.mock import patch, AsyncMock, Mock
        from urllib.parse import quote
        import pytest_httpx

        from app.services.agentcore_relay_client import invoke_agentcore_turn

        canned_sse = (
            b"event: text_delta\n"
            b'data: {"delta": "Hello"}\n'
            b"\n"
            b"event: text_delta\n"
            b'data: {"delta": " world"}\n'
            b"\n"
            b"event: result\n"
            b'data: {"status": "completed"}\n'
            b"\n"
        )

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent-runtime/test"
        expected_url = f"https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/{quote(arn, safe='')}/invocations?qualifier=DEFAULT"

        httpx_mock.add_response(
            method="POST",
            url=expected_url,
            status_code=200,
            stream=pytest_httpx.IteratorStream([canned_sse]),
            headers={"Content-Type": "text/event-stream"},
        )

        # Track durable appends
        allocated_seqs = []
        appended_events = []

        async def mock_allocate(turn_id):
            seq = len(allocated_seqs) + 1
            allocated_seqs.append(seq)
            return seq

        async def mock_append(turn_id, seq, event, *, ttl):
            appended_events.append((turn_id, seq, event))

        with patch(
            "app.services.agentcore_relay_client.get_ws_hub"
        ) as mock_hub_getter:
            mock_hub = Mock()
            mock_hub.publish_to_user = AsyncMock()
            mock_hub.publish_project = AsyncMock()
            mock_hub_getter.return_value = mock_hub

            with patch(
                "app.services.agentcore_relay_client.resolve_resume",
                AsyncMock(return_value=(None, None)),
            ), patch(
                "app.services.agentcore_relay_client.record_session",
                AsyncMock(),
            ), patch(
                "app.services.agentcore_relay_client.release_turn",
                AsyncMock(),
            ), patch(
                "app.services.agentcore_relay_client.get_platform"
            ) as mock_platform:
                mock_store = Mock()
                mock_store.allocate_turn_seq = mock_allocate
                mock_store.append_turn_event = mock_append
                mock_platform.return_value.metadata_store = mock_store

                with patch.dict(
                    os.environ,
                    {
                        "AGENTCORE_RUNTIME_ARN": arn,
                        "AWS_REGION": "us-east-1",
                    },
                ):
                    await invoke_agentcore_turn(
                        project_id="proj-test",
                        message="test message",
                        owner_id="user-123",
                        runtime_session_id=None,
                        turn_id="turn-abc",
                        access_token="Bearer test-token",
                    )

        # Wait for background tasks to complete
        import asyncio
        await asyncio.sleep(0.1)

        # Verify allocate_turn_seq was called 3 times (one per frame)
        assert len(allocated_seqs) == 3
        assert allocated_seqs == [1, 2, 3]

        # Verify append_turn_event was called 3 times
        assert len(appended_events) == 3

        # Verify each event carries the DDB seq (not make_envelope's in-memory seq)
        for i, (turn_id, seq, event) in enumerate(appended_events):
            assert turn_id == "turn-abc"
            assert seq == i + 1  # DDB-allocated seq
            assert event["seq"] == i + 1  # Durable payload carries the DDB seq
            assert event["stream"] == "turn:turn-abc"
            assert event["kind"] in ["text_delta", "result"]

    @pytest.mark.asyncio
    async def test_terminal_frame_appended_durably(self, httpx_mock):
        """The terminal result frame is appended durably."""
        from unittest.mock import patch, AsyncMock, Mock
        from urllib.parse import quote
        import pytest_httpx

        from app.services.agentcore_relay_client import invoke_agentcore_turn

        canned_sse = (
            b"event: result\n"
            b'data: {"status": "completed"}\n'
            b"\n"
        )

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent-runtime/test"
        expected_url = f"https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/{quote(arn, safe='')}/invocations?qualifier=DEFAULT"

        httpx_mock.add_response(
            method="POST",
            url=expected_url,
            status_code=200,
            stream=pytest_httpx.IteratorStream([canned_sse]),
            headers={"Content-Type": "text/event-stream"},
        )

        appended_events = []

        async def mock_append(turn_id, seq, event, *, ttl):
            appended_events.append((turn_id, seq, event))

        with patch(
            "app.services.agentcore_relay_client.get_ws_hub"
        ) as mock_hub_getter:
            mock_hub = Mock()
            mock_hub.publish_to_user = AsyncMock()
            mock_hub.publish_project = AsyncMock()
            mock_hub_getter.return_value = mock_hub

            with patch(
                "app.services.agentcore_relay_client.resolve_resume",
                AsyncMock(return_value=(None, None)),
            ), patch(
                "app.services.agentcore_relay_client.record_session",
                AsyncMock(),
            ), patch(
                "app.services.agentcore_relay_client.release_turn",
                AsyncMock(),
            ), patch(
                "app.services.agentcore_relay_client.get_platform"
            ) as mock_platform:
                mock_store = Mock()
                mock_store.allocate_turn_seq = AsyncMock(return_value=1)
                mock_store.append_turn_event = mock_append
                mock_platform.return_value.metadata_store = mock_store

                with patch.dict(
                    os.environ,
                    {
                        "AGENTCORE_RUNTIME_ARN": arn,
                        "AWS_REGION": "us-east-1",
                    },
                ):
                    await invoke_agentcore_turn(
                        project_id="proj-test",
                        message="test message",
                        owner_id="user-123",
                        runtime_session_id=None,
                        turn_id="turn-abc",
                        access_token="Bearer test-token",
                    )

        # Wait for background tasks
        import asyncio
        await asyncio.sleep(0.1)

        # Verify the terminal result was appended
        assert len(appended_events) == 1
        turn_id, seq, event = appended_events[0]
        assert event["kind"] == "result"
        assert event["data"] == {"status": "completed"}

    @pytest.mark.asyncio
    async def test_append_failure_non_fatal(self, httpx_mock):
        """Append_turn_event failure is non-fatal — live publish still happens."""
        from unittest.mock import patch, AsyncMock, Mock
        from urllib.parse import quote
        import pytest_httpx

        from app.services.agentcore_relay_client import invoke_agentcore_turn

        canned_sse = (
            b"event: text_delta\n"
            b'data: {"delta": "Hello"}\n'
            b"\n"
            b"event: result\n"
            b'data: {"status": "completed"}\n'
            b"\n"
        )

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent-runtime/test"
        expected_url = f"https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/{quote(arn, safe='')}/invocations?qualifier=DEFAULT"

        httpx_mock.add_response(
            method="POST",
            url=expected_url,
            status_code=200,
            stream=pytest_httpx.IteratorStream([canned_sse]),
            headers={"Content-Type": "text/event-stream"},
        )

        # Track live publishes
        published_envelopes = []

        async def mock_publish(owner_id, envelope):
            published_envelopes.append(envelope)

        # Make append_turn_event always raise
        async def mock_append_fail(turn_id, seq, event, *, ttl):
            raise Exception("DDB write failed")

        with patch(
            "app.services.agentcore_relay_client.get_ws_hub"
        ) as mock_hub_getter:
            mock_hub = Mock()
            mock_hub.publish_to_user = mock_publish
            mock_hub.publish_project = AsyncMock()
            mock_hub_getter.return_value = mock_hub

            with patch(
                "app.services.agentcore_relay_client.resolve_resume",
                AsyncMock(return_value=(None, None)),
            ), patch(
                "app.services.agentcore_relay_client.record_session",
                AsyncMock(),
            ), patch(
                "app.services.agentcore_relay_client.release_turn",
                AsyncMock(),
            ), patch(
                "app.services.agentcore_relay_client.get_platform"
            ) as mock_platform:
                mock_store = Mock()
                mock_store.allocate_turn_seq = AsyncMock(return_value=1)
                mock_store.append_turn_event = mock_append_fail
                mock_platform.return_value.metadata_store = mock_store

                with patch.dict(
                    os.environ,
                    {
                        "AGENTCORE_RUNTIME_ARN": arn,
                        "AWS_REGION": "us-east-1",
                    },
                ):
                    await invoke_agentcore_turn(
                        project_id="proj-test",
                        message="test message",
                        owner_id="user-123",
                        runtime_session_id=None,
                        turn_id="turn-abc",
                        access_token="Bearer test-token",
                    )

        # Verify live publishes still happened
        assert len(published_envelopes) == 2
        assert published_envelopes[0]["kind"] == "text_delta"
        assert published_envelopes[1]["kind"] == "result"


class TestPostChatRelayGate:
    """The POST /chat relay gate: with the relay on, the relay path replaces the in-process turn."""

    @pytest.mark.asyncio
    async def test_relay_on_invokes_relay_path_only(self):
        """With relay flag ON, invoke_agentcore_turn is called and start_agent_turn is NOT.

        This verifies the replace contract: when the relay flag is on,
        the in-process SDK turn does NOT also run — the relay path replaces it.
        """
        from unittest.mock import AsyncMock, MagicMock, Mock, patch

        async def mock_relay_true(*args, **kwargs):
            return True

        with patch(
            "app.platform.arch_split.resolve_agentcore_relay",
            side_effect=mock_relay_true,
        ):
            with patch(
                "app.services.agentcore_relay_client.invoke_agentcore_turn",
                new_callable=AsyncMock,
            ) as mock_invoke:
                with patch(
                    "app.api.chat.start_agent_turn"
                ) as mock_start:
                    with patch("app.api.chat.current_user_id", return_value="user-123"):
                        with patch("app.api.chat.get_platform") as mock_platform:
                            mock_auth = AsyncMock()
                            mock_platform.return_value.auth.require_access = mock_auth
                            with patch("app.main.get_project_manager") as mock_pm_getter:
                                mock_pm = MagicMock()
                                mock_project = MagicMock()
                                mock_project.session_id = "session-123"
                                mock_pm.get_project.return_value = mock_project
                                mock_pm_getter.return_value = mock_pm

                                from app.api.chat import chat
                                from app.models.schemas import ChatRequest

                                # Mock Request with Authorization header
                                mock_request = Mock()
                                mock_request.headers.get.return_value = "Bearer test-access-token"

                                req = ChatRequest(
                                    project_id="proj-test",
                                    message="test message",
                                    session_id="rt-session-456",
                                )
                                response_on = await chat(req, mock_request)

        # Verify relay path fired
        assert mock_invoke.call_count == 1, "invoke_agentcore_turn should be called when relay is on"
        assert mock_start.call_count == 0, "start_agent_turn should NOT be called when relay is on (the relay path replaces it)"
        assert "turn_id" in response_on
        assert response_on["status"] == "running"

    @pytest.mark.asyncio
    async def test_access_token_threaded_from_authorization_header(self):
        """POST /chat's relay branch forwards the incoming Authorization header verbatim to invoke_agentcore_turn."""
        from unittest.mock import AsyncMock, MagicMock, Mock, patch

        async def mock_relay_true(*args, **kwargs):
            return True

        with patch(
            "app.platform.arch_split.resolve_agentcore_relay",
            side_effect=mock_relay_true,
        ):
            with patch(
                "app.services.agentcore_relay_client.invoke_agentcore_turn",
                new_callable=AsyncMock,
            ) as mock_invoke:
                with patch("app.api.chat.current_user_id", return_value="user-123"):
                    with patch("app.api.chat.get_platform") as mock_platform:
                        mock_auth = AsyncMock()
                        mock_platform.return_value.auth.require_access = mock_auth
                        with patch("app.main.get_project_manager") as mock_pm_getter:
                            mock_pm = MagicMock()
                            mock_project = MagicMock()
                            mock_project.session_id = "session-123"
                            mock_pm.get_project.return_value = mock_project
                            mock_pm_getter.return_value = mock_pm

                            from app.api.chat import chat
                            from app.models.schemas import ChatRequest

                            # Mock Request with specific Authorization header
                            mock_request = Mock()
                            mock_request.headers.get.return_value = "Bearer caller-token-xyz"

                            req = ChatRequest(
                                project_id="proj-test",
                                message="test message",
                                session_id="rt-session-456",
                            )
                            await chat(req, mock_request)

        # Verify access_token was forwarded verbatim
        assert mock_invoke.call_count == 1
        call_kwargs = mock_invoke.call_args.kwargs
        assert call_kwargs["access_token"] == "Bearer caller-token-xyz"


class TestTurnMetadataPersistedAtStart:
    """Regression coverage: the turn-metadata write must actually run.

    ``invoke_agentcore_turn`` writes turn metadata (PK=TURN#{turn_id}, SK=META)
    at turn start so the ws owner-guard can later resolve the turn's owner. That
    write used ``store`` before it was bound (``store`` was only assigned deep in
    the drain loop, making it a function-local unbound at the earlier call), and
    the write is wrapped in a best-effort try/except — so the UnboundLocalError
    was swallowed EVERY turn and metadata never persisted, silently breaking
    durable turn replay in cloud mode. Prior tests seeded metadata directly and
    never exercised this real write path, so they stayed green.
    """

    @pytest.mark.asyncio
    async def test_put_turn_metadata_actually_called_on_relay_start(self, httpx_mock):
        """The real store.put_turn_metadata runs (no swallowed UnboundLocalError)."""
        from unittest.mock import MagicMock
        from urllib.parse import quote
        import pytest_httpx

        from app.platform import get_platform
        from app.services.agentcore_relay_client import invoke_agentcore_turn

        canned_sse = (
            b"event: result\n"
            b'data: {"status": "completed"}\n'
            b"\n"
        )
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent-runtime/test"
        expected_url = (
            f"https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/"
            f"{quote(arn, safe='')}/invocations?qualifier=DEFAULT"
        )
        httpx_mock.add_response(
            method="POST",
            url=expected_url,
            status_code=200,
            stream=pytest_httpx.IteratorStream([canned_sse]),
            headers={"Content-Type": "text/event-stream"},
        )

        async def mock_publish(owner_id, envelope):
            pass

        # Spy on the REAL metadata store's put_turn_metadata so we can assert the
        # write path executed (an UnboundLocalError would skip it silently).
        store = get_platform().metadata_store
        put_spy = AsyncMock(wraps=store.put_turn_metadata)

        with patch(
            "app.services.agentcore_relay_client.get_ws_hub"
        ) as mock_hub_getter, patch.object(
            store, "put_turn_metadata", put_spy
        ):
            mock_hub = MagicMock()
            mock_hub.publish_to_user = mock_publish
            mock_hub_getter.return_value = mock_hub

            with patch(
                "app.services.agentcore_relay_client.resolve_resume",
                AsyncMock(return_value=(None, None)),
            ), patch(
                "app.services.agentcore_relay_client.record_session",
                AsyncMock(),
            ), patch(
                "app.services.agentcore_relay_client.release_turn",
                AsyncMock(),
            ), patch.dict(
                os.environ,
                {"AGENTCORE_RUNTIME_ARN": arn, "AWS_REGION": "us-east-1"},
            ):
                await invoke_agentcore_turn(
                    project_id="proj-turn-meta",
                    message="hi",
                    owner_id="user-turn-meta",
                    runtime_session_id=None,
                    turn_id="turn-meta-1",
                    access_token="Bearer test-token",
                )

        # The metadata write MUST have run with the owner + project so the ws
        # owner-guard can later resolve this turn. Before the fix this spy
        # was never called (UnboundLocalError swallowed by the try/except).
        assert put_spy.await_count == 1
        kwargs = put_spy.await_args.kwargs
        assert kwargs["user_id"] == "user-turn-meta"
        assert kwargs["project_id"] == "proj-turn-meta"
        # TTL must be an absolute epoch expiry in the future, not a bare
        # duration or a monotonic-clock value near 1970.
        import time as _time
        assert kwargs["ttl"] > int(_time.time())
