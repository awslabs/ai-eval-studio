# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``/ws`` route + ``WsHub`` + ``ws_envelope``.

Covers the full lifecycle of a WebSocket connection:
- Token validation at connect time (cloud + dev modes).
- Registration into the per-user hub on accept; removal on disconnect.
- Outbound envelope pumping via the per-connection asyncio queue.
- Keepalive cadence (uses an injected short interval to keep the test fast).
- Pure-unit coverage of envelope construction (sequence numbering,
  per-stream isolation).

The WS route uses FastAPI's native WebSocket support; we drive it via
:class:`fastapi.testclient.TestClient` which is the only TestClient that
speaks the WebSocket portion of the ASGI spec — ``httpx.AsyncClient``
with ``ASGITransport`` does not.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api import ws as ws_module
from app.api.ws import router as ws_router
from app.platform import set_platform, Platform
from app.platform.aws.auth_cognito_bearer import CognitoBearerJwtAuth
from app.platform.dev_local import LocalAuth, NoOpTokenVending
from app.services.ws_envelope import make_envelope, make_envelope_with_seq, reset_counters
from app.services.ws_hub import WsConnection, get_ws_hub
from tests.conftest import authed_ws


# --- Helpers / fixtures ----------------------------------------------


def _build_app() -> FastAPI:
    """Tiny FastAPI app exposing only the WS route — fastest possible."""
    app = FastAPI()
    app.include_router(ws_router)
    return app


@pytest.fixture(autouse=True)
def _reset_ws_state():
    """Clear hub registry + envelope counters between tests.

    Singletons leak otherwise: a connection registered in one test would
    persist into the next, and seq counters would carry over and break
    monotonicity assertions in tests that pin to absolute values.
    """
    reset_counters()
    get_ws_hub()._reset()
    yield
    reset_counters()
    get_ws_hub()._reset()


def _make_dev_platform() -> Platform:
    """A Platform whose Auth is :class:`LocalAuth` — no JWT required."""
    return Platform(
        metadata_store=None,  # type: ignore[arg-type]
        session_store=None,
        token_vending=NoOpTokenVending(),
        auth=LocalAuth(metadata_store=None),  # type: ignore[arg-type]
    )


def _make_cloud_platform(cognito_env) -> Platform:
    """A Platform whose Auth is :class:`CognitoBearerJwtAuth`."""
    return Platform(
        metadata_store=None,  # type: ignore[arg-type]
        session_store=None,
        token_vending=NoOpTokenVending(),
        auth=CognitoBearerJwtAuth(
            metadata_store=None,  # type: ignore[arg-type]
            region=cognito_env["region"],
            user_pool_id=cognito_env["user_pool_id"],
            client_id=cognito_env["spa_client_id"],
        ),
    )


# --- WS route — connect + auth ---------------------------------------


class TestWsRouteAuth:
    """Token validation at WS-connect time."""

    def test_connect_dev_mode_succeeds(self, monkeypatch):
        """Dev mode: any token (or none) accepts and registers the conn."""
        monkeypatch.setenv("EVAL_STUDIO_MODE", "dev")
        monkeypatch.setenv("LOCAL_DEV_USER_ID", "ws-dev-user")
        set_platform(_make_dev_platform())
        try:
            app = _build_app()
            client = TestClient(app)
            with authed_ws(client):
                # Inside the with-block, hub has registered the conn.
                conns = get_ws_hub()._by_user.get("ws-dev-user", set())
                assert len(conns) == 1
            # After exit, route's finally cleans up.
            assert get_ws_hub()._by_user.get("ws-dev-user") is None
        finally:
            set_platform(None)

    def test_connect_cloud_mode_with_valid_token_succeeds(
        self, cognito_env, cognito_jwks_inject, valid_access_token
    ):
        set_platform(_make_cloud_platform(cognito_env))
        try:
            app = _build_app()
            client = TestClient(app)
            with authed_ws(client, valid_access_token):
                conns = get_ws_hub()._by_user.get(cognito_env["sub"], set())
                assert len(conns) == 1
        finally:
            set_platform(None)

    def test_connect_cloud_mode_with_bad_token_refused(
        self, cognito_env, cognito_jwks_inject
    ):
        """Invalid token in the auth frame → ``ws.close(code=4401)``.

        The socket accepts first now, so the rejection arrives as a close after the
        handshake rather than a refused upgrade. What must not change is that
        nothing reaches the hub: the assertion below is the security property, and
        the close code is only how the client learns of it."""
        set_platform(_make_cloud_platform(cognito_env))
        try:
            app = _build_app()
            client = TestClient(app)
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect("/ws") as ws:
                    ws.send_json({"action": "auth", "token": "not.a.jwt"})
                    ws.receive_json()  # raises: the server closed instead
            # Application-defined close code 4401 = "auth required" — see
            # ``app.api.ws._CLOSE_AUTH_REQUIRED``.
            assert exc_info.value.code == 4401
            # Hub is empty — connection never reached register.
            assert not get_ws_hub()._by_user
        finally:
            set_platform(None)

    def test_dev_mode_accepts_the_empty_token_the_browser_sends(self, monkeypatch):
        """The dev SPA sends ``{"action": "auth", "token": ""}`` — no login exists.

        Spelled out rather than left to ``authed_ws``'s default because this is the
        exact frame that was refused: every dev socket closed 4401, so turns
        completed on the server while the UI never received a frame.
        """
        monkeypatch.setenv("EVAL_STUDIO_MODE", "dev")
        monkeypatch.setenv("LOCAL_DEV_USER_ID", "ws-dev-user")
        set_platform(_make_dev_platform())
        try:
            client = TestClient(_build_app())
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"action": "auth", "token": ""})
                ack = ws.receive_json()
                assert ack.get("kind") == "auth_ok", f"dev socket refused: {ack}"
                assert len(get_ws_hub()._by_user.get("ws-dev-user", set())) == 1
        finally:
            set_platform(None)

    def test_cloud_mode_refuses_an_empty_token(self, cognito_env, cognito_jwks_inject):
        """Moving the emptiness check into Auth must not open cloud to ``""``.

        Cloud's validator rejects an empty token itself; this pins that the socket
        still closes 4401 and never reaches the hub.
        """
        set_platform(_make_cloud_platform(cognito_env))
        try:
            client = TestClient(_build_app())
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect("/ws") as ws:
                    ws.send_json({"action": "auth", "token": ""})
                    ws.receive_json()
            assert exc_info.value.code == 4401
            assert not get_ws_hub()._by_user
        finally:
            set_platform(None)

    @pytest.mark.parametrize("token", [None, 0, ["jwt"], {"t": 1}])
    def test_a_non_string_token_is_refused_in_any_mode(self, monkeypatch, token):
        """The shape check that stays in the route: a token must be a string."""
        monkeypatch.setenv("EVAL_STUDIO_MODE", "dev")
        set_platform(_make_dev_platform())
        try:
            client = TestClient(_build_app())
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect("/ws") as ws:
                    ws.send_json({"action": "auth", "token": token})
                    ws.receive_json()
            assert exc_info.value.code == 4401
            assert not get_ws_hub()._by_user
        finally:
            set_platform(None)

    def test_connect_cloud_mode_missing_token_refused(self, cognito_env, cognito_jwks_inject):
        """A first frame that is not an auth action is refused.

        Covers the shape an attacker would actually try: skip the handshake and go
        straight to ``subscribe``. The server reads exactly one frame and rejects
        anything that is not the credential, so an unauthenticated subscribe cannot
        be honoured by construction."""
        set_platform(_make_cloud_platform(cognito_env))
        try:
            app = _build_app()
            client = TestClient(app)
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect("/ws") as ws:
                    ws.send_json({"action": "subscribe", "stream": "s", "since_seq": -1})
                    ws.receive_json()
            assert exc_info.value.code == 4401
            assert not get_ws_hub()._by_user
        finally:
            set_platform(None)

    def test_a_binary_first_frame_is_refused(self, cognito_env, cognito_jwks_inject):
        """A binary frame where the credential belongs closes 4401, like any other
        unusable first frame.

        ``receive_text`` indexes ``message["text"]``, which a binary frame does not
        carry, so this used to escape the route as a ``KeyError`` — an ERROR-level
        traceback any anonymous caller could produce with a single frame."""
        set_platform(_make_cloud_platform(cognito_env))
        try:
            client = TestClient(_build_app())
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect("/ws") as ws:
                    ws.send_bytes(b"\x00not-text")
                    ws.receive_json()
            assert exc_info.value.code == 4401
            assert not get_ws_hub()._by_user
        finally:
            set_platform(None)

    def test_connect_cloud_mode_expired_token_refused(
        self, cognito_env, cognito_jwks_inject, expired_access_token
    ):
        set_platform(_make_cloud_platform(cognito_env))
        try:
            app = _build_app()
            client = TestClient(app)
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect("/ws") as ws:
                    ws.send_json({"action": "auth", "token": expired_access_token})
                    ws.receive_json()
            assert exc_info.value.code == 4401
        finally:
            set_platform(None)


class TestWsRouteLifecycle:
    """Connection registration / unregistration / outbound pump."""

    def test_unregisters_on_client_close(self, monkeypatch):
        monkeypatch.setenv("EVAL_STUDIO_MODE", "dev")
        monkeypatch.setenv("LOCAL_DEV_USER_ID", "ws-life-user")
        set_platform(_make_dev_platform())
        try:
            app = _build_app()
            client = TestClient(app)
            with authed_ws(client):
                assert len(get_ws_hub()._by_user.get("ws-life-user", set())) == 1
            # Block exit triggered WebSocketDisconnect → finally → unregister.
            assert get_ws_hub()._by_user.get("ws-life-user") is None
        finally:
            set_platform(None)

    def test_disconnect_log_says_why_the_socket_closed(self, monkeypatch):
        """The disconnect line carries the client's close code and reason.

        Without them a reconnect in the coordinator log cannot be attributed: a
        token rotation, a page refresh and a dropped network all read the same.
        The close here is the one the browser client sends on token rotation.
        """
        monkeypatch.setenv("EVAL_STUDIO_MODE", "dev")
        monkeypatch.setenv("LOCAL_DEV_USER_ID", "ws-close-user")
        set_platform(_make_dev_platform())
        try:
            client = TestClient(_build_app())
            with patch.object(ws_module, "logger") as logger:
                with authed_ws(client) as websocket:
                    websocket.close(code=1000, reason="token-rotation")
            lines = [
                call.args[0] % call.args[1:]
                for call in logger.info.call_args_list
                if call.args and str(call.args[0]).startswith("[ws] disconnect")
            ]
            assert len(lines) == 1, lines
            assert "code=1000" in lines[0]
            assert "reason='token-rotation'" in lines[0]
        finally:
            set_platform(None)

    def test_publish_to_user_reaches_connected_client(self, monkeypatch):
        """End-to-end: publish through hub → pump task → ws.send_json."""
        monkeypatch.setenv("EVAL_STUDIO_MODE", "dev")
        monkeypatch.setenv("LOCAL_DEV_USER_ID", "ws-pub-user")
        set_platform(_make_dev_platform())
        try:
            app = _build_app()
            client = TestClient(app)
            with authed_ws(client) as ws:
                envelope = make_envelope(
                    stream="turn:t-1", kind="text_delta", data={"text": "hi"}
                )
                # The pump task lives on the WS session's portal loop.
                # We use that same portal to enqueue from the test thread
                # so the put_nowait runs on the right event loop.
                ws.portal.call(
                    get_ws_hub().publish_to_user, "ws-pub-user", envelope
                )
                received = ws.receive_json()
                assert received["stream"] == "turn:t-1"
                assert received["kind"] == "text_delta"
                assert received["data"] == {"text": "hi"}
        finally:
            set_platform(None)

    def test_a_binary_frame_after_auth_is_discarded(self, monkeypatch):
        """The protocol is text-only, and an unusable frame is dropped rather than
        ending the connection — a binary one included.

        The subscribe sent AFTER the binary frame is the proof. Frames are read in
        order, so its replay arriving means the binary frame was consumed without
        taking the read loop down with it."""
        monkeypatch.setenv("EVAL_STUDIO_MODE", "dev")
        monkeypatch.setenv("LOCAL_DEV_USER_ID", "ws-bin-user")
        set_platform(_make_dev_platform())
        stream = "test:binary-frame"
        hub = get_ws_hub()

        class _Provider:
            events = [
                make_envelope_with_seq(
                    stream=stream, kind="text_delta", data={"text": "after"}, seq=1,
                )
            ]

        hub.register_replay_provider(stream, _Provider(), owner_id="ws-bin-user")
        try:
            client = TestClient(_build_app())
            with authed_ws(client) as ws:
                ws.send_bytes(b"\x00not-text")
                ws.send_json({"action": "subscribe", "stream": stream, "since_seq": -1})
                envelope = ws.receive_json()
                assert envelope["stream"] == stream
                assert envelope["data"] == {"text": "after"}
        finally:
            hub.unregister_replay_provider(stream)
            set_platform(None)

    def test_keepalive_emits_at_configured_cadence(self, monkeypatch):
        """Short interval → keepalive fires within the test window."""
        monkeypatch.setenv("EVAL_STUDIO_MODE", "dev")
        monkeypatch.setenv("LOCAL_DEV_USER_ID", "ws-ka-user")
        set_platform(_make_dev_platform())
        try:
            hub = get_ws_hub()
            hub.set_keepalive_interval(0.05)

            app = _build_app()
            client = TestClient(app)
            with authed_ws(client) as ws:
                # Start keepalive AFTER the conn is registered, on the
                # same loop the pump runs on.
                ws.portal.call(_start_keepalive_now, hub)
                env = ws.receive_json()
                assert env["kind"] == "keepalive"
                assert env["stream"] == "user:ws-ka-user"
                ws.portal.call(_stop_keepalive_now, hub)
        finally:
            set_platform(None)


async def _start_keepalive_now(hub):
    """Run hub.start_keepalive on the running loop."""
    hub.start_keepalive()


async def _stop_keepalive_now(hub):
    hub.stop_keepalive()


# --- Hub unit tests --------------------------------------------------


class TestWsHubFanout:
    """Direct unit tests on :class:`WsHub` — no FastAPI."""

    @pytest.mark.asyncio
    async def test_register_unregister_round_trip(self):
        hub = get_ws_hub()
        conn = WsConnection(connection_id="c1", user_id="u-rr", websocket=None)
        hub.register(conn)
        assert hub._by_user["u-rr"] == {conn}
        hub.unregister(conn)
        # Empty buckets are removed so the dict doesn't grow forever.
        assert "u-rr" not in hub._by_user

    @pytest.mark.asyncio
    async def test_publish_to_user_fans_out_to_all_user_connections(self):
        hub = get_ws_hub()
        a1 = WsConnection(connection_id="a1", user_id="u-A", websocket=None)
        a2 = WsConnection(connection_id="a2", user_id="u-A", websocket=None)
        b1 = WsConnection(connection_id="b1", user_id="u-B", websocket=None)
        hub.register(a1)
        hub.register(a2)
        hub.register(b1)

        envelope = make_envelope("turn:tx", "text_delta", {"text": "fan"})
        await hub.publish_to_user("u-A", envelope)

        assert a1.queue.qsize() == 1
        assert a2.queue.qsize() == 1
        assert b1.queue.qsize() == 0
        # Both A connections see the same envelope object.
        assert a1.queue.get_nowait() is envelope
        assert a2.queue.get_nowait() is envelope

    @pytest.mark.asyncio
    async def test_publish_to_unknown_user_is_noop(self):
        hub = get_ws_hub()
        # No registrations — should not raise.
        await hub.publish_to_user("nobody", make_envelope("turn:x", "result", {}))

    @pytest.mark.asyncio
    async def test_replay_to_connection_returns_buffered_envelopes_past_since_seq(self):
        """The hub replays only envelopes whose seq is greater than ``since_seq``.

        Locks in the contract the replay path depends on: a SPA reconnecting
        with ``since_seq=N`` receives envelopes ``seq > N`` from the
        registered provider's buffer, and nothing earlier.
        """

        class FakeProvider:
            def __init__(self, events):
                self.events = events
                self.events_dropped = 0

        hub = get_ws_hub()
        provider = FakeProvider([
            {"stream": "turn:t1", "seq": 0, "kind": "status", "data": {}},
            {"stream": "turn:t1", "seq": 1, "kind": "text_delta", "data": {"text": "Hi"}},
            {"stream": "turn:t1", "seq": 2, "kind": "text_delta", "data": {"text": " there"}},
        ])
        hub.register_replay_provider("turn:t1", provider, owner_id="u-rep")

        conn = WsConnection(connection_id="c-replay", user_id="u-rep", websocket=None)
        hub.register(conn)

        # Client says it has already seen seq=0; only 1 and 2 should replay.
        replayed = await hub.replay_to_connection(conn, "turn:t1", since_seq=0)
        assert replayed == 2
        env_a = conn.queue.get_nowait()
        env_b = conn.queue.get_nowait()
        assert env_a["seq"] == 1
        assert env_b["seq"] == 2

    @pytest.mark.asyncio
    async def test_replay_with_no_provider_is_noop(self):
        """An idle stream (resource finalized + evicted) replays nothing."""
        hub = get_ws_hub()
        conn = WsConnection(connection_id="c-empty", user_id="u-empty", websocket=None)
        hub.register(conn)
        replayed = await hub.replay_to_connection(conn, "turn:gone", since_seq=-1)
        assert replayed == 0
        assert conn.queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_replay_with_since_seq_minus_one_replays_full_buffer(self):
        """``since_seq=-1`` (fresh client) replays every buffered envelope."""

        class FakeProvider:
            def __init__(self, events):
                self.events = events
                self.events_dropped = 0

        hub = get_ws_hub()
        provider = FakeProvider([
            {"stream": "job:j1", "seq": 5, "kind": "progress", "data": {}},
            {"stream": "job:j1", "seq": 6, "kind": "progress", "data": {}},
        ])
        hub.register_replay_provider("job:j1", provider, owner_id="u-fresh")
        conn = WsConnection(connection_id="c-fresh", user_id="u-fresh", websocket=None)
        hub.register(conn)
        replayed = await hub.replay_to_connection(conn, "job:j1", since_seq=-1)
        assert replayed == 2

    @pytest.mark.asyncio
    async def test_replay_rejects_cross_tenant_subscribe(self):
        """A connection authenticated as user A must NOT receive replay
        for a stream registered to user B.

        Without this gate, ``stream`` is a client-supplied lookup key
        and any authenticated user who learned another tenant's
        ``turn:<uuid>`` / ``job:<uuid>`` (logs, support handoff,
        leaked screenshot) would dump that tenant's buffered chat
        transcript.
        """

        class FakeProvider:
            def __init__(self, events):
                self.events = events
                self.events_dropped = 0

        hub = get_ws_hub()
        provider = FakeProvider([
            {"stream": "turn:victim", "seq": 0, "kind": "text_delta", "data": {"text": "secret"}},
        ])
        hub.register_replay_provider(
            "turn:victim", provider, owner_id="victim-user",
        )

        # Attacker connection authenticated as a different user.
        attacker = WsConnection(
            connection_id="c-attacker", user_id="attacker-user", websocket=None,
        )
        hub.register(attacker)

        replayed = await hub.replay_to_connection(
            attacker, "turn:victim", since_seq=-1,
        )
        assert replayed == 0
        assert attacker.queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_publish_project_emits_project_event_envelope(self):
        hub = get_ws_hub()
        conn = WsConnection(connection_id="c-proj", user_id="owner-1", websocket=None)
        hub.register(conn)
        await hub.publish_project(
            "proj-7", "owner-1", {"resource": "filesystem", "type": "changed"}
        )
        envelope = await asyncio.wait_for(conn.queue.get(), timeout=1.0)
        assert envelope["kind"] == "project_event"
        assert envelope["stream"] == "project:proj-7"
        assert envelope["data"]["resource"] == "filesystem"
        assert envelope["data"]["type"] == "changed"


# --- Envelope unit tests ---------------------------------------------


class TestWsEnvelope:
    """Pure-data tests on :func:`make_envelope`."""

    def test_make_envelope_includes_required_fields(self):
        env = make_envelope("turn:abc", "text_delta", {"text": "hi"})
        assert set(env.keys()) == {"stream", "seq", "kind", "data"}
        assert env["stream"] == "turn:abc"
        assert env["kind"] == "text_delta"
        assert env["data"] == {"text": "hi"}
        assert isinstance(env["seq"], int)

    def test_seq_monotonic_within_stream(self):
        seqs = [
            make_envelope("turn:mono", "text_delta", {})["seq"]
            for _ in range(3)
        ]
        # Counters reset by the autouse fixture, so this stream starts at 0.
        assert seqs == [0, 1, 2]

    def test_seq_isolated_across_streams(self):
        a0 = make_envelope("turn:A", "text_delta", {})["seq"]
        b0 = make_envelope("turn:B", "text_delta", {})["seq"]
        a1 = make_envelope("turn:A", "text_delta", {})["seq"]
        b1 = make_envelope("turn:B", "text_delta", {})["seq"]
        a2 = make_envelope("turn:A", "text_delta", {})["seq"]
        # Each stream's counter is independent.
        assert (a0, a1, a2) == (0, 1, 2)
        assert (b0, b1) == (0, 1)

    def test_make_envelope_canonical_json_round_trip(self):
        """Envelopes must be JSON-serializable (WS uses ``send_json``)."""
        import json

        # Project streams draw from the durable allocator, so the
        # seq is supplied rather than counter-minted. This test is about JSON
        # round-tripping, so any fixed seq serves.
        env = make_envelope_with_seq(
            "project:p-1",
            "project_event",
            {"resource": "filesystem", "type": "changed", "tool": "Write"},
            seq=1,
        )
        roundtrip = json.loads(json.dumps(env))
        assert roundtrip == env


# --- turn: durable replay routing + owner-guard -----


@pytest.mark.asyncio
async def test_turn_durable_replay_from_moto():
    """Turn: replay from moto DDB — append N events, replay the suffix."""
    from moto import mock_aws
    import boto3
    import time
    from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore
    from app.api.ws import _replay_turn_stream_from_durable_log
    from app.services.ws_hub import WsConnection

    def _create_metadata_table(ddb_client, table_name: str) -> None:
        """Create the single-table schema."""
        ddb_client.create_table(
            TableName=table_name,
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
                {"AttributeName": "GSI1PK", "AttributeType": "S"},
                {"AttributeName": "GSI1SK", "AttributeType": "S"},
            ],
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "GSI1",
                    "KeySchema": [
                        {"AttributeName": "GSI1PK", "KeyType": "HASH"},
                        {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
            BillingMode="PAY_PER_REQUEST",
        )

    with mock_aws():
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        turn_id = "turn-abc"
        user_id = "user-replay"
        project_id = "test-project"
        ttl = int(time.time()) + (7 * 24 * 3600)

        # Store turn metadata for owner resolution
        await store.put_turn_metadata(turn_id, user_id=user_id, project_id=project_id, ttl=ttl)

        # Append 5 events to the durable turn log
        for i in range(1, 6):
            seq = await store.allocate_turn_seq(turn_id)
            event = {
                "stream": f"turn:{turn_id}",
                "seq": seq,
                "kind": "status" if i < 5 else "result",
                "data": {"step": i},
            }
            await store.append_turn_event(turn_id, seq, event, ttl=ttl)

        # Create a mock connection
        conn = WsConnection(
            connection_id="conn-1",
            user_id=user_id,
            websocket=None,  # type: ignore
        )

        # Patch platform to return our store
        test_platform = Platform(
            metadata_store=store,
            session_store=None,  # type: ignore
            token_vending=NoOpTokenVending(),
            auth=LocalAuth(metadata_store=store),
        )
        set_platform(test_platform)

        # Mock project manager to resolve owner
        from unittest.mock import patch, MagicMock
        mock_project = MagicMock()
        mock_project.owner_id = user_id

        with patch("app.main.get_project_manager") as mock_pm:
            mock_pm.return_value.get_project.return_value = mock_project

            # Now replay should work
            count, last_seq = await _replay_turn_stream_from_durable_log(conn, f"turn:{turn_id}", 2)

            assert count == 3
            assert last_seq == 5
            envelopes = []
            while not conn.queue.empty():
                envelopes.append(conn.queue.get_nowait())
            assert len(envelopes) == 3
            assert envelopes[0]["seq"] == 3
            assert envelopes[2]["seq"] == 5


@pytest.mark.asyncio
async def test_turn_replay_owner_mismatch_returns_zero():
    """Owner-guard: mismatched owner replays 0 events."""
    from moto import mock_aws
    import boto3
    import time
    from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore
    from app.api.ws import _replay_turn_stream_from_durable_log
    from app.services.ws_hub import WsConnection

    def _create_metadata_table(ddb_client, table_name: str) -> None:
        ddb_client.create_table(
            TableName=table_name,
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
                {"AttributeName": "GSI1PK", "AttributeType": "S"},
                {"AttributeName": "GSI1SK", "AttributeType": "S"},
            ],
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "GSI1",
                    "KeySchema": [
                        {"AttributeName": "GSI1PK", "KeyType": "HASH"},
                        {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
            BillingMode="PAY_PER_REQUEST",
        )

    with mock_aws():
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        turn_id = "turn-xyz"
        ttl = int(time.time()) + (7 * 24 * 3600)

        # Seed turn metadata (turn owned by project-A, which is owned by user-correct)
        await store.put_turn_metadata(
            turn_id=turn_id,
            user_id="user-correct",
            project_id="project-A",
            ttl=ttl
        )

        # Append 3 events
        for i in range(1, 4):
            seq = await store.allocate_turn_seq(turn_id)
            event = {"stream": f"turn:{turn_id}", "seq": seq, "kind": "status", "data": {}}
            await store.append_turn_event(turn_id, seq, event, ttl=ttl)

        # Connection for a DIFFERENT user
        # queue created by WsConnection
        conn = WsConnection(
            connection_id="conn-2",
            user_id="user-wrong",
            websocket=None,  # type: ignore
            
        )

        test_platform = Platform(
            metadata_store=store,
            session_store=None,  # type: ignore
            token_vending=NoOpTokenVending(),
            auth=LocalAuth(metadata_store=store),
        )
        set_platform(test_platform)

        from unittest.mock import patch, MagicMock
        with patch("app.main.get_project_manager") as mock_pm:
            mock_project = MagicMock()
            mock_project.owner_id = "user-correct"  # Different from conn.user_id
            mock_pm.return_value.get_project.return_value = mock_project

            # Owner-guard rejects on mismatch → 0 events
            count, last_seq = await _replay_turn_stream_from_durable_log(conn, f"turn:{turn_id}", 0)
            assert count == 0
            assert last_seq == 0
            assert conn.queue.empty()


@pytest.mark.asyncio
async def test_turn_replay_source_is_ddb_not_ws_hub():
    """Replay source is turn_events_since (DDB), not ws_hub memory."""
    from moto import mock_aws
    import boto3
    import time
    from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore
    from app.api.ws import _replay_turn_stream_from_durable_log
    from app.services.ws_hub import WsConnection, get_ws_hub

    def _create_metadata_table(ddb_client, table_name: str) -> None:
        ddb_client.create_table(
            TableName=table_name,
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
                {"AttributeName": "GSI1PK", "AttributeType": "S"},
                {"AttributeName": "GSI1SK", "AttributeType": "S"},
            ],
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "GSI1",
                    "KeySchema": [
                        {"AttributeName": "GSI1PK", "KeyType": "HASH"},
                        {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
            BillingMode="PAY_PER_REQUEST",
        )

    with mock_aws():
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        turn_id = "turn-ddb"
        ttl = int(time.time()) + (7 * 24 * 3600)

        # Seed turn metadata (turn owned by project-X, which is owned by user-ddb)
        await store.put_turn_metadata(
            turn_id=turn_id,
            user_id="user-ddb",
            project_id="project-X",
            ttl=ttl
        )

        # Append 2 events to DDB
        for i in range(1, 3):
            seq = await store.allocate_turn_seq(turn_id)
            event = {"stream": f"turn:{turn_id}", "seq": seq, "kind": "status", "data": {"i": i}}
            await store.append_turn_event(turn_id, seq, event, ttl=ttl)

        # Register a DIFFERENT provider in ws_hub memory (should NOT be consulted)
        hub = get_ws_hub()

        class FakeProvider:
            def get_buffered(self, since_seq):
                # This should NEVER be called for turn: replay
                return [{"seq": 999, "kind": "fake", "data": {}}]

        hub.register_replay_provider(f"turn:{turn_id}", FakeProvider(), owner_id="user-ddb")

        # Connection
        # queue created by WsConnection
        conn = WsConnection(
            connection_id="conn-3",
            user_id="user-ddb",
            websocket=None,  # type: ignore
            
        )

        test_platform = Platform(
            metadata_store=store,
            session_store=None,  # type: ignore
            token_vending=NoOpTokenVending(),
            auth=LocalAuth(metadata_store=store),
        )
        set_platform(test_platform)

        from unittest.mock import patch, MagicMock
        with patch("app.main.get_project_manager") as mock_pm:
            mock_project = MagicMock()
            mock_project.owner_id = "user-ddb"
            mock_pm.return_value.get_project.return_value = mock_project

            # Replay from DDB, not ws_hub memory (correct-by-construction)
            count, _ = await _replay_turn_stream_from_durable_log(conn, f"turn:{turn_id}", 0)

            # Must have replayed from DDB, not the fake provider
            assert count == 2
            envelopes = []
            while not conn.queue.empty():
                envelopes.append(conn.queue.get_nowait())
            assert len(envelopes) == 2
            assert envelopes[0]["data"]["i"] == 1
            assert envelopes[1]["data"]["i"] == 2
            # If it had used the fake provider, we'd see seq=999


# --- Job-stream replay fallback (dev / empty durable log) -------------------


@pytest.mark.asyncio
async def test_job_subscribe_falls_back_to_in_memory_when_durable_empty():
    """A job: subscribe with an EMPTY durable log replays the in-memory buffer.

    On dev / in-process (arch_split=False) ``Job.append_event`` never writes
    the durable log, so a durable-only replay served zero events — a mid-run
    refresh rebuilt the progress bars from nothing. The fallback must consult
    the in-memory replay provider whenever the durable log yields nothing.
    """
    import uuid
    from unittest.mock import MagicMock, patch

    from app.api.ws import _handle_inbound
    from app.services.jobs import Job

    job_id = f"job-{uuid.uuid4().hex[:8]}"
    owner_id = "dev-user"
    stream = f"job:{job_id}"
    hub = get_ws_hub()

    # In-memory Job with buffered envelopes (the dev-path source of truth).
    job = Job(job_id=job_id, project_id="proj-x", type="eval", state="running")
    job.events = [
        make_envelope_with_seq(
            stream=stream, kind="progress",
            data={"event": "generating", "model": "m1", "current": i, "total": 3},
            seq=i,
        )
        for i in range(1, 4)
    ]
    hub.register_replay_provider(stream, job, owner_id=owner_id)

    conn = WsConnection(
        connection_id=f"conn-{uuid.uuid4().hex[:8]}",
        user_id=owner_id,
        websocket=MagicMock(),
    )

    try:
        # Durable path returns zero (empty log / dev store) — force it so the
        # test does not depend on platform store contents.
        with patch(
            "app.api.ws._replay_job_stream_from_durable_log", return_value=(0, -1),
        ) as durable:
            import json as _json
            await _handle_inbound(
                conn,
                _json.dumps(
                    {"action": "subscribe", "stream": stream, "since_seq": -1}
                ),
            )
            durable.assert_awaited_once()

        got = []
        while not conn.queue.empty():
            got.append(conn.queue.get_nowait())
        seqs = [e["seq"] for e in got]
        assert seqs == [1, 2, 3], (
            f"Expected in-memory fallback to replay seqs [1, 2, 3], got {seqs}"
        )
    finally:
        hub.unregister_replay_provider(stream)


@pytest.mark.asyncio
async def test_job_subscribe_seq_merges_in_memory_tail_after_durable():
    """When the durable log serves events, the in-memory replay STILL runs —
    but only for seqs past the durable leg's last seq (seq-merge).

    A PARTIALLY populated durable log used to suppress the in-memory
    fallback entirely, permanently truncating the replayed stream. The merge
    closes that gap; starting the in-memory replay at the durable last seq
    means no seq is ever re-sent, so the client's backwards-jump
    domain-reset heuristic can never misread the overlap.
    """
    import uuid
    from unittest.mock import AsyncMock, MagicMock, patch

    from app.api.ws import _handle_inbound

    job_id = f"job-{uuid.uuid4().hex[:8]}"
    stream = f"job:{job_id}"
    conn = WsConnection(
        connection_id=f"conn-{uuid.uuid4().hex[:8]}",
        user_id="cloud-user",
        websocket=MagicMock(),
    )

    with patch(
        "app.api.ws._replay_job_stream_from_durable_log",
        return_value=(5, 5),
    ):
        with patch.object(
            get_ws_hub(), "replay_to_connection", new=AsyncMock(),
        ) as mem_replay:
            import json as _json
            await _handle_inbound(
                conn,
                _json.dumps(
                    {"action": "subscribe", "stream": stream, "since_seq": -1}
                ),
            )
            mem_replay.assert_awaited_once_with(conn, stream, 5)


@pytest.mark.asyncio
async def test_turn_subscribe_falls_back_to_in_memory_when_durable_empty():
    """turn: streams get the same zero-count fallback as job: streams.

    The durable turn log is only written on the relay/arch_split path, so on
    dev it is always empty and durable-only replay lost the whole in-flight
    turn on refresh (an empty assistant bubble with no tool pills).
    """
    import uuid
    from unittest.mock import MagicMock, patch

    from app.api.ws import _handle_inbound

    turn_id = f"turn-{uuid.uuid4().hex[:8]}"
    owner_id = "dev-user"
    stream = f"turn:{turn_id}"
    hub = get_ws_hub()

    class _Provider:
        events = [
            make_envelope_with_seq(
                stream=stream, kind="text_delta", data={"text": f"t{i}"}, seq=i,
            )
            for i in range(1, 4)
        ]

    hub.register_replay_provider(stream, _Provider(), owner_id=owner_id)
    conn = WsConnection(
        connection_id=f"conn-{uuid.uuid4().hex[:8]}",
        user_id=owner_id,
        websocket=MagicMock(),
    )

    try:
        with patch(
            "app.api.ws._replay_turn_stream_from_durable_log", return_value=(0, -1),
        ) as durable:
            import json as _json
            await _handle_inbound(
                conn,
                _json.dumps(
                    {"action": "subscribe", "stream": stream, "since_seq": -1}
                ),
            )
            durable.assert_awaited_once()

        got = []
        while not conn.queue.empty():
            got.append(conn.queue.get_nowait())
        assert [e["seq"] for e in got] == [1, 2, 3], (
            "in-memory fallback must replay the buffered turn envelopes when "
            "the durable log serves zero"
        )
    finally:
        hub.unregister_replay_provider(stream)


class TestTheCredentialIsNotInTheUrl:
    """The credential must not reach anything that records a URL.

    The token used to be a query parameter, and uvicorn's access log writes the
    full request line, so a live access token was recoverable from log storage
    for the retention period long after the socket closed.

    The other half — that the client no longer PUTS a token in the URL — is
    asserted in ``frontend/src/lib/ws-client.test.ts``. Both halves are needed: a
    server that ignores the parameter still leaks it if the client keeps sending
    one.
    """

    def test_a_token_in_the_query_string_is_ignored(
        self, cognito_env, cognito_jwks_inject, valid_access_token, monkeypatch
    ):
        """A valid token supplied the OLD way does not authenticate.

        Asserted because leaving the query-string path working "for compatibility"
        would mean the token keeps being sent, keeps being logged, and the fix
        would be cosmetic. A client that has not been updated must fail loudly.

        The server closes this socket only when its auth-frame deadline passes, so
        the deadline is shortened here; what is under test is the outcome, not how
        long the server waits for it.
        """
        from app.api import ws as ws_module

        monkeypatch.setattr(ws_module, "_AUTH_FRAME_TIMEOUT_S", 0.5)
        set_platform(_make_cloud_platform(cognito_env))
        try:
            app = _build_app()
            client = TestClient(app)
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect(f"/ws?token={valid_access_token}") as ws:
                    # No auth frame: the query string is all the client offered.
                    ws.receive_json()
            assert exc_info.value.code == 4401
            assert not get_ws_hub()._by_user
        finally:
            set_platform(None)
