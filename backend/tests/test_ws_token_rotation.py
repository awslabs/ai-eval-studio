# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token-rotation reconnect: close-then-reopen sequence.

The SPA refreshes its access token ~5 min before expiry, then closes
the WebSocket with code 1000 and reconnects with the new token. The
backend side: ``WebSocketDisconnect`` triggers ``ws_hub.unregister``,
the new ``$connect`` runs through the same ``ws_hub.register`` path,
and the registry ends up holding exactly one entry under the user with
no leak.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.ws import router as ws_router
from app.platform import set_platform, Platform
from app.platform.dev_local import LocalAuth, NoOpTokenVending
from app.services.ws_envelope import reset_counters
from app.services.ws_hub import get_ws_hub
from tests.conftest import authed_ws


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(ws_router)
    return app


def _make_dev_platform() -> Platform:
    return Platform(
        metadata_store=None,  # type: ignore[arg-type]
        session_store=None,
        token_vending=NoOpTokenVending(),
        auth=LocalAuth(metadata_store=None),  # type: ignore[arg-type]
    )


@pytest.fixture(autouse=True)
def _reset_ws_state():
    reset_counters()
    get_ws_hub()._reset()
    yield
    reset_counters()
    get_ws_hub()._reset()


def test_close_then_reconnect_unregisters_then_registers(monkeypatch):
    """Close 1000 → reconnect → unregister-then-register."""
    monkeypatch.setenv("EVAL_STUDIO_MODE", "dev")
    monkeypatch.setenv("LOCAL_DEV_USER_ID", "rotate-user")
    set_platform(_make_dev_platform())
    try:
        app = _build_app()
        client = TestClient(app)

        # First connection — token A. Capture its connection_id by
        # poking into the hub's registry.
        with authed_ws(client, "token-A"):
            conns_a = get_ws_hub()._by_user.get("rotate-user", set())
            assert len(conns_a) == 1
            conn_a = next(iter(conns_a))
            conn_a_id = conn_a.connection_id

        # After the first context exits, unregister has run; bucket gone.
        assert get_ws_hub()._by_user.get("rotate-user") is None

        # Second connection — token B. Different connection_id, exactly
        # one entry in the registry.
        with authed_ws(client, "token-B"):
            conns_b = get_ws_hub()._by_user.get("rotate-user", set())
            assert len(conns_b) == 1
            conn_b = next(iter(conns_b))
            conn_b_id = conn_b.connection_id

        # The two connection_ids are distinct UUIDs — confirms a fresh
        # registration happened, not an accidental reuse.
        assert conn_a_id != conn_b_id
        # And after the second exit, no leak.
        assert get_ws_hub()._by_user.get("rotate-user") is None
    finally:
        set_platform(None)


def test_concurrent_two_connections_for_same_user(monkeypatch):
    """Two open WS for one user — both registered, no fan-out collapse."""
    monkeypatch.setenv("EVAL_STUDIO_MODE", "dev")
    monkeypatch.setenv("LOCAL_DEV_USER_ID", "concurrent-user")
    set_platform(_make_dev_platform())
    try:
        app = _build_app()
        client = TestClient(app)

        with authed_ws(client, "t1") as ws1, \
             authed_ws(client, "t2") as ws2:
            conns = get_ws_hub()._by_user.get("concurrent-user", set())
            assert len(conns) == 2
            # Sanity-check: distinct connection_ids.
            ids = {c.connection_id for c in conns}
            assert len(ids) == 2
        # Both close → registry empty.
        assert get_ws_hub()._by_user.get("concurrent-user") is None
    finally:
        set_platform(None)
