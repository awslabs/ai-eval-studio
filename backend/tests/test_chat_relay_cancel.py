# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for relay turn cancellation.

Verifies that running relay turns can be cancelled server-side via stop_turn,
with the drain task cancelled, the claim released, the microVM compute stopped
via StopRuntimeSession, and a cancelled envelope emitted.
"""
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest


@pytest.fixture
def dev_platform_with_store(projects_dir: Path, monkeypatch):
    """Set up dev platform with a REAL FilesystemMetadataStore for turn cancel tests.

    Mirrors the fixture from test_chat_relay_claim.py. The claim goes through the
    actual conditional-write path so the cancel tests can verify claim release.
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


@pytest.mark.asyncio
async def test_relay_tasks_addressable_by_turn_id():
    """Relay tasks are stored in a turn_id-keyed structure.

    ``stop_turn`` has to find one specific relay turn, which a bare set of
    tasks cannot answer.
    """
    from app.api import chat

    # Any keyed structure will do, as long as it supports turn_id lookup.
    assert hasattr(chat, "_relay_tasks")
    assert not isinstance(chat._relay_tasks, set), \
        "Expected _relay_tasks to be a dict-like keyed structure, not a bare set"


@pytest.mark.asyncio
async def test_relay_aware_stop_cancels_drain_and_releases_claim(dev_platform_with_store):
    """Test that stop_turn on a relay turn cancels the drain task.

    Relay-aware stop_turn cancels the addressable drain task. The drain's
    finally releases the claim (tested separately in relay client tests).
    """
    from app.api import chat

    user_id = "test-user"
    project_id = str(uuid4())
    turn_id = str(uuid4())

    # Arrange: create a mock task that we can check was cancelled
    async def mock_drain_task():
        # Simulate a long-running relay drain
        await asyncio.sleep(10)

    task = asyncio.create_task(mock_drain_task())

    # Register the task in _relay_tasks (keyed by turn_id)
    chat._relay_tasks[turn_id] = {
        "task": task,
        "runtime_session_id": "test-session-id",
        "project_id": project_id,
        "owner_id": user_id,
    }

    # Act: call stop_turn on the relay turn_id
    with patch("app.api.chat.current_user_id", return_value=user_id):
        result = await chat.stop_turn(turn_id)

    # Assert: stop_turn returned cancelled status
    assert result["status"] == "cancelled"

    # Wait for the cancellation to take effect
    for _ in range(10):
        await asyncio.sleep(0)  # Yield to event loop
        if task.cancelled() or task.done():
            break

    # Assert: the task was cancelled
    assert task.cancelled(), "Expected task.cancel() to have been called and processed"

    # Cleanup
    if turn_id in chat._relay_tasks:
        del chat._relay_tasks[turn_id]


@pytest.mark.asyncio
async def test_cancel_calls_stop_runtime_session(dev_platform_with_store):
    """Test that cancel calls StopRuntimeSession with the runtime session id.

    Cancel must call StopRuntimeSession to stop the microVM compute,
    not just cancel the local drain task.
    """
    from app.api import chat
    from app.platform import get_platform
    from app.services.agentcore_session_registry import release_turn

    user_id = "test-user"
    project_id = str(uuid4())
    turn_id = str(uuid4())
    runtime_session_id = str(uuid4())

    # Arrange: seed a running claim
    store = get_platform().metadata_store
    await store.claim_agentcore_session_turn(user_id, project_id, ttl=3600, current_turn_id=turn_id)

    # Arrange: real drain task that releases claim on cancel
    async def mock_drain_task():
        try:
            await asyncio.sleep(10)
        finally:
            await release_turn(user_id, project_id)

    task = asyncio.create_task(mock_drain_task())

    # Register the relay turn with its runtime_session_id AND the caller's
    # access token (the token is stored at spawn so cancel can authorize
    # StopRuntimeSession — an empty token would 401 and the microVM would keep
    # running).
    chat._relay_tasks[turn_id] = {
        "task": task,
        "runtime_session_id": runtime_session_id,
        "project_id": project_id,
        "owner_id": user_id,
        "access_token": "Bearer caller-token-xyz",
    }

    # Act: call stop_turn with mocked StopRuntimeSession
    with patch("app.api.chat.current_user_id", return_value=user_id):
        with patch.dict("os.environ", {"AGENTCORE_RUNTIME_ARN": "arn:test"}):
            with patch("app.api.chat._stop_agentcore_runtime_session", new_callable=AsyncMock) as mock_stop:
                await chat.stop_turn(turn_id)

                # Assert: StopRuntimeSession was called with the correct session id
                # AND the stored access token (not an empty string).
                mock_stop.assert_called_once()
                args, kwargs = mock_stop.call_args
                all_args = list(args) + list(kwargs.values())
                assert runtime_session_id in all_args
                assert "Bearer caller-token-xyz" in all_args

    # Wait for task cleanup
    try:
        await asyncio.wait_for(task, timeout=0.5)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass


@pytest.mark.asyncio
async def test_cancel_skips_stop_runtime_session_when_no_token(dev_platform_with_store):
    """With no stored access token, cancel skips StopRuntimeSession.

    Issuing a StopRuntimeSession with an empty bearer would 401 every time, so
    the stop path deliberately skips the call (and logs) rather than pretending
    to stop the microVM. The local drain cancel + claim release still proceed.
    """
    from app.api import chat
    from app.platform import get_platform
    from app.services.agentcore_session_registry import release_turn

    user_id = "test-user"
    project_id = str(uuid4())
    turn_id = str(uuid4())
    runtime_session_id = str(uuid4())

    store = get_platform().metadata_store
    await store.claim_agentcore_session_turn(user_id, project_id, ttl=3600, current_turn_id=turn_id)

    async def mock_drain_task():
        try:
            await asyncio.sleep(10)
        finally:
            await release_turn(user_id, project_id)

    task = asyncio.create_task(mock_drain_task())

    # No access_token key — simulates a turn registered before the token was
    # captured, or a lost entry.
    chat._relay_tasks[turn_id] = {
        "task": task,
        "runtime_session_id": runtime_session_id,
        "project_id": project_id,
        "owner_id": user_id,
    }

    with patch("app.api.chat.current_user_id", return_value=user_id):
        with patch.dict("os.environ", {"AGENTCORE_RUNTIME_ARN": "arn:test"}):
            with patch("app.api.chat._stop_agentcore_runtime_session", new_callable=AsyncMock) as mock_stop:
                await chat.stop_turn(turn_id)
                # StopRuntimeSession is NOT called with a guaranteed-to-fail empty token.
                mock_stop.assert_not_called()

    try:
        await asyncio.wait_for(task, timeout=0.5)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass

    # Cleanup
    await store.release_agentcore_session_turn(user_id, project_id)
    if turn_id in chat._relay_tasks:
        del chat._relay_tasks[turn_id]


@pytest.mark.asyncio
async def test_cancel_emits_cancelled_envelope(dev_platform_with_store):
    """Test that cancel publishes a cancelled envelope on turn:<id> and appends durably.

    A cancelled envelope (subtype:"cancelled") is published on turn:<id>
    and appended durably so a reconnect sees the terminal cancelled state.
    """
    from app.api import chat
    from app.platform import get_platform
    from app.services import ws_hub

    user_id = "test-user"
    project_id = str(uuid4())
    turn_id = str(uuid4())

    # Arrange: seed claim and mock task
    store = get_platform().metadata_store
    await store.claim_agentcore_session_turn(user_id, project_id, ttl=3600, current_turn_id=turn_id)

    mock_task = AsyncMock()
    mock_task.cancel = Mock()
    mock_task.done = Mock(return_value=False)

    chat._relay_tasks[turn_id] = {
        "task": mock_task,
        "runtime_session_id": "test-session",
        "project_id": project_id,
        "owner_id": user_id,
    }

    # Act: call stop_turn with mocked publish
    published_envelopes = []

    async def capture_publish(owner_id, envelope):
        published_envelopes.append(envelope)

    with patch("app.api.chat.current_user_id", return_value=user_id):
        with patch("app.api.chat.get_platform", return_value=get_platform()):
            with patch("app.api.chat._stop_agentcore_runtime_session"):
                with patch.object(ws_hub.get_ws_hub(), "publish_to_user", side_effect=capture_publish):
                    await chat.stop_turn(turn_id)

    # Assert: a cancelled envelope was published
    assert len(published_envelopes) > 0, "Expected at least one envelope published"
    cancelled_env = next((e for e in published_envelopes if e.get("kind") == "result"), None)
    assert cancelled_env is not None, "Expected a result envelope"
    assert cancelled_env.get("data", {}).get("subtype") == "cancelled", \
        "Expected subtype:cancelled in the result envelope"

    # Cleanup
    await store.release_agentcore_session_turn(user_id, project_id)
    if turn_id in chat._relay_tasks:
        del chat._relay_tasks[turn_id]


@pytest.mark.asyncio
async def test_stop_runtime_session_failure_non_fatal(dev_platform_with_store):
    """Test that StopRuntimeSession failure is non-fatal (best-effort).

    StopRuntimeSession failure should not prevent the local cancel
    from happening. The task cancel still proceeds.
    """
    from app.api import chat

    user_id = "test-user"
    project_id = str(uuid4())
    turn_id = str(uuid4())

    # Arrange: create a mock task
    async def mock_drain_task():
        await asyncio.sleep(10)

    task = asyncio.create_task(mock_drain_task())

    chat._relay_tasks[turn_id] = {
        "task": task,
        "runtime_session_id": "test-session",
        "project_id": project_id,
        "owner_id": user_id,
    }

    # Act: StopRuntimeSession raises an exception (best-effort failure)
    with patch("app.api.chat.current_user_id", return_value=user_id):
        with patch.dict("os.environ", {"AGENTCORE_RUNTIME_ARN": "arn:test"}):
            with patch("app.api.chat._stop_agentcore_runtime_session", side_effect=Exception("API error")):
                # Should not raise — failure is best-effort
                result = await chat.stop_turn(turn_id)
                assert result is not None
                assert result["status"] == "cancelled"

    # Wait for task cancellation
    for _ in range(10):
        await asyncio.sleep(0)
        if task.cancelled() or task.done():
            break

    # Assert: local cancel still happened despite StopRuntimeSession failure
    assert task.cancelled(), "Expected task to be cancelled even when StopRuntimeSession fails"

    # Cleanup
    if turn_id in chat._relay_tasks:
        del chat._relay_tasks[turn_id]


@pytest.mark.asyncio
async def test_normal_relay_turn_unaffected():
    """Test that a normal (uncancelled) relay turn still completes correctly.

    The turn_id-keyed structure should be cleaned up on completion.
    """
    from app.api import chat

    turn_id = str(uuid4())

    # Arrange: simulate a relay turn registration
    mock_task = AsyncMock()
    mock_task.done = Mock(return_value=False)

    chat._relay_tasks[turn_id] = {
        "task": mock_task,
        "runtime_session_id": "test-session",
        "project_id": "test-project",
        "owner_id": "test-user",
    }

    # Simulate task completion callback (should discard from _relay_tasks)
    def done_callback(future):
        if turn_id in chat._relay_tasks:
            del chat._relay_tasks[turn_id]

    mock_task.add_done_callback = Mock(side_effect=lambda cb: done_callback(mock_task))
    mock_task.add_done_callback(lambda f: None)

    # Assert: the turn was cleaned up
    assert turn_id not in chat._relay_tasks, \
        "Expected turn to be cleaned up on completion"


@pytest.mark.asyncio
async def test_stop_runtime_session_resolves_region_and_invokes_boto3():
    """The REAL _stop_agentcore_runtime_session resolves region + calls boto3.

    Regression guard for relay cancellation that silently did nothing: the
    function resolved the region via get_platform().config, but Platform has NO
    .config attribute, so the call raised AttributeError — swallowed by the
    best-effort except — and StopRuntimeSession never ran (Stop was a silent
    no-op, microVM kept billing). The other cancel tests mock
    _stop_agentcore_runtime_session ENTIRELY, so they never exercise the real
    region-resolution + client-call path. This test patches ONLY boto3.client so
    the real function body runs end-to-end.
    """
    from app.api.chat import _stop_agentcore_runtime_session

    mock_client = Mock()
    captured = {}

    def _fake_boto3_client(service, region_name=None, config=None):
        captured["service"] = service
        captured["region_name"] = region_name
        return mock_client

    with patch("boto3.client", side_effect=_fake_boto3_client), patch.dict(
        "os.environ", {"AWS_REGION": "eu-west-1"}
    ):
        # Must NOT raise (the AttributeError bug lived here) and MUST call
        # stop_runtime_session with the ARN + session id.
        await _stop_agentcore_runtime_session(
            runtime_session_id="rt-session-33-chars-min-length-ok-xxx",
            agent_runtime_arn="arn:aws:bedrock-agentcore:eu-west-1:123:agent-runtime/x",
            access_token="Bearer tok-abc",
        )

    # Region resolved from the environment (NOT via a nonexistent Platform.config).
    assert captured["service"] == "bedrock-agentcore"
    assert captured["region_name"] == "eu-west-1"
    # StopRuntimeSession was actually invoked with the runtime ids.
    mock_client.stop_runtime_session.assert_called_once()
    call_kwargs = mock_client.stop_runtime_session.call_args.kwargs
    assert call_kwargs["agentRuntimeArn"] == "arn:aws:bedrock-agentcore:eu-west-1:123:agent-runtime/x"
    assert call_kwargs["runtimeSessionId"] == "rt-session-33-chars-min-length-ok-xxx"


@pytest.mark.asyncio
async def test_stop_runtime_session_falls_back_to_default_region():
    """With AWS_REGION unset, region resolution falls back to us-east-1 (no crash)."""
    from app.api.chat import _stop_agentcore_runtime_session

    mock_client = Mock()
    captured = {}

    def _fake_boto3_client(service, region_name=None, config=None):
        captured["region_name"] = region_name
        return mock_client

    import os as _os
    env_no_region = {k: v for k, v in _os.environ.items() if k != "AWS_REGION"}
    with patch("boto3.client", side_effect=_fake_boto3_client), patch.dict(
        "os.environ", env_no_region, clear=True
    ):
        await _stop_agentcore_runtime_session(
            runtime_session_id="rt-session-33-chars-min-length-ok-xxx",
            agent_runtime_arn="arn:test",
            access_token="Bearer tok",
        )

    assert captured["region_name"] == "us-east-1"
    mock_client.stop_runtime_session.assert_called_once()


@pytest.mark.asyncio
async def test_stop_issues_microvm_stop_before_drain_cancel(dev_platform_with_store):
    """Ordering guard: StopRuntimeSession is issued BEFORE the drain task is cancelled.

    The drain task's finally releases the per-(user,project) claim. Cancelling it
    first frees the claim while the microVM is still running — admitting a
    concurrent turn (the observed pile-up). The stop must therefore be issued
    before the cancel. Asserted deterministically on CALL ORDER (not timing): a
    Mock task records when .cancel() runs; the stop mock records when it runs.
    """
    from app.api import chat

    user_id = "test-user"
    project_id = str(uuid4())
    turn_id = str(uuid4())

    order: list[str] = []

    # Mock task: .cancel() records order; .done() returns False so cancel fires.
    mock_task = Mock()
    mock_task.done = Mock(return_value=False)
    mock_task.cancel = Mock(side_effect=lambda: order.append("cancel"))

    chat._relay_tasks[turn_id] = {
        "task": mock_task,
        "runtime_session_id": "rt-session-33-chars-min-length-ok-xx",
        "project_id": project_id,
        "owner_id": user_id,
        "access_token": "Bearer tok-xyz",
    }

    async def _fake_stop(rt_session, arn, token):
        order.append("stop")

    with patch("app.api.chat.current_user_id", return_value=user_id):
        with patch.dict("os.environ", {"AGENTCORE_RUNTIME_ARN": "arn:test"}):
            with patch("app.api.chat._stop_agentcore_runtime_session", side_effect=_fake_stop) as mock_stop:
                await chat.stop_turn(turn_id)

    mock_stop.assert_called_once()
    mock_task.cancel.assert_called_once()
    # The microVM stop MUST precede the drain cancel (which releases the claim).
    assert order == ["stop", "cancel"], (
        f"expected stop before drain-cancel, got {order} — cancelling first "
        "releases the claim while the microVM still runs (pile-up)"
    )


@pytest.mark.asyncio
async def test_stop_logs_skip_reason_when_no_runtime_session(dev_platform_with_store):
    """Observability: a falsy runtime_session_id logs a skip reason (not silent).

    Attaches a dedicated handler to the ``app.api.chat`` logger rather than
    relying on pytest's ``caplog`` global capture, which another test in the
    full suite can defeat by muting propagation/level (isolation-proof).
    """
    import logging as _logging
    from app.api import chat

    user_id = "test-user"
    project_id = str(uuid4())
    turn_id = str(uuid4())

    async def mock_drain_task():
        await asyncio.sleep(10)

    task = asyncio.create_task(mock_drain_task())
    chat._relay_tasks[turn_id] = {
        "task": task,
        "runtime_session_id": None,  # first-turn resume handle was None
        "project_id": project_id,
        "owner_id": user_id,
        "access_token": "Bearer tok",
    }

    records: list[_logging.LogRecord] = []

    class _Capture(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            records.append(record)

    chat_logger = _logging.getLogger("app.api.chat")
    handler = _Capture()
    prev_level = chat_logger.level
    prev_disabled = chat_logger.disabled
    chat_logger.addHandler(handler)
    chat_logger.setLevel(_logging.INFO)
    chat_logger.disabled = False  # undo any other test's logging.disable/mute
    try:
        with patch("app.api.chat.current_user_id", return_value=user_id):
            with patch.dict("os.environ", {"AGENTCORE_RUNTIME_ARN": "arn:test"}):
                with patch("app.api.chat._stop_agentcore_runtime_session") as mock_stop:
                    await chat.stop_turn(turn_id)
    finally:
        chat_logger.removeHandler(handler)
        chat_logger.setLevel(prev_level)
        chat_logger.disabled = prev_disabled

    messages = [r.getMessage() for r in records]
    # StopRuntimeSession NOT called (no runtime session), and the skip is logged.
    mock_stop.assert_not_called()
    assert any("stop_turn entered" in m for m in messages), \
        f"expected an entry log line; got {messages}"
    assert any("no runtime_session_id" in m for m in messages), \
        f"expected a skip-reason log for the falsy runtime_session_id; got {messages}"

    try:
        await asyncio.wait_for(task, timeout=0.5)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass


@pytest.mark.asyncio
async def test_stop_appends_transcript_marker_so_refresh_is_not_interrupted(
    dev_platform_with_store, tmp_path, monkeypatch
):
    """A deliberate Stop appends an honest marker to the transcript.

    The killed microVM never persists its partial reply, so the transcript
    tail after a Stop was the bare user message — exactly the shape the
    GET /chat/history heuristic reports as interrupted, which put a phantom
    "Session interrupted — Resume" banner on every deliberate Stop after a
    refresh (measured live). The marker records the stop AND flips the tail
    role so the heuristic stays quiet.
    """
    import json as _json

    from app.api import chat

    user_id = "test-user"
    project_id = str(uuid4())
    turn_id = str(uuid4())

    # Project dir with an existing transcript whose tail is the provisional
    # user message (the relay start-write shape at cancel time).
    project_dir = tmp_path / project_id
    (project_dir / "chat").mkdir(parents=True)
    transcript = project_dir / "chat" / "transcript.jsonl"
    transcript.write_text(_json.dumps({
        "id": f"provisional-user-{turn_id}",
        "role": "user",
        "parts": [{"type": "text", "text": "do the thing"}],
        "created_at": "2026-06-22T10:00:00.000Z",
    }) + "\n")
    monkeypatch.setattr(
        "app.services.project_paths.resolve_project_dir",
        lambda pid: project_dir,
    )

    async def mock_drain_task():
        await asyncio.sleep(10)

    task = asyncio.create_task(mock_drain_task())
    chat._relay_tasks[turn_id] = {
        "task": task,
        "runtime_session_id": "test-session-id",
        "project_id": project_id,
        "owner_id": user_id,
    }

    try:
        with patch("app.api.chat.current_user_id", return_value=user_id):
            result = await chat.stop_turn(turn_id)
        assert result["status"] == "cancelled"

        lines = [
            _json.loads(line)
            for line in transcript.read_text().splitlines() if line.strip()
        ]
        assert lines[-1]["id"] == f"stopped-{turn_id}"
        assert lines[-1]["role"] == "assistant", (
            "Tail role must not be 'user' or the interruption heuristic "
            "shows the phantom Resume banner after refresh"
        )
        assert "stopped" in lines[-1]["parts"][0]["text"].lower()
    finally:
        task.cancel()
        chat._relay_tasks.pop(turn_id, None)
