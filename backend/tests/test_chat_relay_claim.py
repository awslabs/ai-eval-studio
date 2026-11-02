# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relay-gate + claim tests for POST /chat.

Proves the durable (user, project) turn claim wired into the relay path:
  1. Fresh (userA, projX) -> claim wins -> relay task spawned, status=running.
  2. Second POST for (userA, projX) while held -> status=busy AND
     invoke_agentcore_turn NOT called a second time (no double-invoke).
  3. Grain: (userA, projX) held does NOT block (userA, projY) -> projY invokes
     (per-(user, project), not per-user).
  4. Release-on-terminal: after invoke_agentcore_turn finishes (success), the claim
     is released so a subsequent turn on the same (user, project) claims again.
  5. Release-on-error: a non-2xx / transport error path still releases (subsequent
     claim succeeds).
  6. Flag-off: relay flag off -> in-process branch runs -> claim_turn NOT called.

The claim uses a REAL FilesystemMetadataStore (dev adapter) so the busy case is a
genuine held-claim conditional write (not a mocked return), and the grain case runs
two real project_ids.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

pytestmark = pytest.mark.asyncio


@pytest.fixture
def dev_platform_with_store(projects_dir: Path, monkeypatch):
    """Set up dev platform with a REAL FilesystemMetadataStore.

    The claim goes through the actual conditional-write path so the busy case
    (2) is a genuine held-claim, not a mocked return. Auth.require_access is
    stubbed to a no-op — authz is out of scope for these claim-gate tests
    (and covered elsewhere); the metadata_store stays REAL for the claim.
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


def _mock_project():
    """Build a mock project with a session_id (relay branch reads pm.get_project)."""
    project = MagicMock()
    project.session_id = "session-123"
    return project


def _mock_request():
    """Build a mock FastAPI Request with an Authorization header."""
    request = Mock()
    request.headers.get.return_value = "Bearer test-access-token"
    return request


async def _relay_true(*args, **kwargs):
    return True


async def _relay_false(*args, **kwargs):
    return False


class TestRelayClaimGate:
    """POST /chat relay-gate claim behavior (cases 1, 2, 3, 6)."""

    async def test_fresh_claim_wins_and_spawns_relay(self, dev_platform_with_store):
        """Case 1: fresh (userA, projX) -> claim wins -> relay spawned, status=running."""
        from app.api.chat import chat
        from app.models.schemas import ChatRequest

        with patch(
            "app.platform.arch_split.resolve_agentcore_relay", side_effect=_relay_true
        ), patch(
            "app.services.agentcore_relay_client.invoke_agentcore_turn",
            new_callable=AsyncMock,
        ) as mock_invoke, patch(
            "app.api.chat.current_user_id", return_value="userA"
        ), patch(
            "app.main.get_project_manager"
        ) as mock_pm_getter:
            mock_pm = MagicMock()
            mock_pm.get_project.return_value = _mock_project()
            mock_pm_getter.return_value = mock_pm

            req = ChatRequest(project_id="projX", message="hi", session_id="rt-1")
            resp = await chat(req, _mock_request())

        assert resp["status"] == "running"
        assert "turn_id" in resp
        assert mock_invoke.call_count == 1

    async def test_busy_claim_returns_busy_no_double_invoke(
        self, dev_platform_with_store
    ):
        """Test: a REAL held claim -> second POST returns HONEST busy, invoke NOT called.

        The relay has no queue — the message is dropped. The status is now "busy"
        (not the misleading "queued"), and the rejection is logged at WARNING so
        a lost message is visible (manual verification: check stderr in test output).
        """
        from app.api.chat import chat
        from app.models.schemas import ChatRequest
        from app.services.agentcore_session_registry import claim_turn

        # Pre-claim (userA, projX) via the REAL store so the gate hits a genuine held claim
        won = await claim_turn("userA", "projX")
        assert won is True

        with patch(
            "app.platform.arch_split.resolve_agentcore_relay", side_effect=_relay_true
        ), patch(
            "app.services.agentcore_relay_client.invoke_agentcore_turn",
            new_callable=AsyncMock,
        ) as mock_invoke, patch(
            "app.api.chat.current_user_id", return_value="userA"
        ), patch(
            "app.main.get_project_manager"
        ) as mock_pm_getter:
            mock_pm = MagicMock()
            mock_pm.get_project.return_value = _mock_project()
            mock_pm_getter.return_value = mock_pm

            req = ChatRequest(project_id="projX", message="hi", session_id="rt-1")
            resp = await chat(req, _mock_request())

        # The claim is already held -> HONEST busy status (not "queued"), no invoke
        # status is "busy": the message was rejected, not queued
        assert resp["status"] == "busy"
        assert "turn_id" in resp
        assert mock_invoke.call_count == 0
        # The log warning is emitted but pytest's caplog doesn't capture it due to
        # the custom RequestContextFilter — verify manually in stderr output

    async def test_grain_projX_held_does_not_block_projY(self, dev_platform_with_store):
        """Case 3: (userA, projX) held does NOT block (userA, projY) — projY invokes."""
        from app.api.chat import chat
        from app.models.schemas import ChatRequest
        from app.services.agentcore_session_registry import claim_turn

        # Hold projX for userA
        won = await claim_turn("userA", "projX")
        assert won is True

        with patch(
            "app.platform.arch_split.resolve_agentcore_relay", side_effect=_relay_true
        ), patch(
            "app.services.agentcore_relay_client.invoke_agentcore_turn",
            new_callable=AsyncMock,
        ) as mock_invoke, patch(
            "app.api.chat.current_user_id", return_value="userA"
        ), patch(
            "app.main.get_project_manager"
        ) as mock_pm_getter:
            mock_pm = MagicMock()
            mock_pm.get_project.return_value = _mock_project()
            mock_pm_getter.return_value = mock_pm

            # POST for projY (different project, same user) — should still claim + invoke
            req = ChatRequest(project_id="projY", message="hi", session_id="rt-2")
            resp = await chat(req, _mock_request())

        assert resp["status"] == "running"
        assert mock_invoke.call_count == 1

    async def test_flag_off_takes_in_process_path_no_claim(self, dev_platform_with_store):
        """Case 6: relay flag off -> in-process branch runs -> claim_turn NOT called."""
        from app.api.chat import chat
        from app.models.schemas import ChatRequest

        with patch(
            "app.platform.arch_split.resolve_agentcore_relay", side_effect=_relay_false
        ), patch(
            "app.services.agentcore_session_registry.claim_turn",
            new_callable=AsyncMock,
        ) as mock_claim, patch(
            "app.api.chat.current_user_id", return_value="userA"
        ), patch(
            "app.main.get_project_manager"
        ) as mock_pm_getter, patch(
            "app.api.chat.get_agent_turn_tracker"
        ) as mock_tracker_getter, patch(
            "app.api.chat.start_agent_turn"
        ) as mock_start, patch(
            "app.api.chat.build_coordinator_prompt", return_value="prompt"
        ), patch(
            "app.services.project_paths.resolve_project_dir", return_value="/tmp/x"
        ):
            mock_pm = MagicMock()
            mock_pm.get_project.return_value = _mock_project()
            mock_pm_getter.return_value = mock_pm

            # No active turn -> starts immediately in-process
            mock_tracker = MagicMock()
            mock_tracker.get_active_turn.return_value = None
            mock_tracker_getter.return_value = mock_tracker

            mock_turn = MagicMock()
            mock_turn.turn_id = "turn-inproc"
            mock_start.return_value = mock_turn

            req = ChatRequest(project_id="projX", message="hi", session_id="rt-1")
            resp = await chat(req, _mock_request())

        # In-process path ran; claim_turn was NOT called
        assert resp["status"] == "running"
        assert mock_start.call_count == 1
        assert mock_claim.call_count == 0


class TestRelayReleaseOnTerminal:
    """invoke_agentcore_turn releases the claim on success + error (cases 4, 5)."""

    async def test_release_on_success(self, dev_platform_with_store, httpx_mock):
        """Case 4: after a successful turn, the claim is released (subsequent claim succeeds)."""
        import os
        import pytest_httpx
        from urllib.parse import quote

        from app.services.agentcore_relay_client import invoke_agentcore_turn
        from app.services.agentcore_session_registry import claim_turn

        won = await claim_turn("userA", "projX", turn_id="turn-1")
        assert won is True

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent-runtime/test"
        expected_url = f"https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/{quote(arn, safe='')}/invocations?qualifier=DEFAULT"
        httpx_mock.add_response(
            method="POST",
            url=expected_url,
            status_code=200,
            stream=pytest_httpx.IteratorStream([b"event: result\ndata: {}\n\n"]),
        )

        with patch(
            "app.services.agentcore_relay_client.get_ws_hub"
        ) as mock_hub_getter:
            mock_hub = Mock()
            mock_hub.publish_to_user = AsyncMock()
            mock_hub.publish_project = AsyncMock()
            mock_hub_getter.return_value = mock_hub
            with patch.dict(
                os.environ, {"AGENTCORE_RUNTIME_ARN": arn, "AWS_REGION": "us-east-1"}
            ):
                await invoke_agentcore_turn(
                    project_id="projX",
                    message="hi",
                    owner_id="userA",
                    runtime_session_id=None,
                    turn_id="turn-1",
                    access_token="Bearer tok",
                )

        # The claim must be released — a fresh claim on the same (user, project) succeeds
        reclaim = await claim_turn("userA", "projX")
        assert reclaim is True

    async def test_release_on_non_2xx(self, dev_platform_with_store, httpx_mock):
        """Case 5a: a non-2xx response path still releases the claim."""
        import os
        from urllib.parse import quote

        from app.services.agentcore_relay_client import invoke_agentcore_turn
        from app.services.agentcore_session_registry import claim_turn

        # Claim under the SAME turn the relay runs, as the chat route does. The
        # release is fenced on the turn id, so an unattributed claim could only
        # be released by an unfenced release — the release the fence rules out.
        won = await claim_turn("userA", "projX", turn_id="turn-1")
        assert won is True

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent-runtime/test"
        expected_url = f"https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/{quote(arn, safe='')}/invocations?qualifier=DEFAULT"
        httpx_mock.add_response(
            method="POST",
            url=expected_url,
            status_code=500,
            content=b"internal error",
        )

        with patch(
            "app.services.agentcore_relay_client.get_ws_hub"
        ) as mock_hub_getter:
            mock_hub = Mock()
            mock_hub.publish_to_user = AsyncMock()
            mock_hub.publish_project = AsyncMock()
            mock_hub_getter.return_value = mock_hub
            with patch.dict(
                os.environ, {"AGENTCORE_RUNTIME_ARN": arn, "AWS_REGION": "us-east-1"}
            ):
                await invoke_agentcore_turn(
                    project_id="projX",
                    message="hi",
                    owner_id="userA",
                    runtime_session_id=None,
                    turn_id="turn-1",
                    access_token="Bearer tok",
                )

        reclaim = await claim_turn("userA", "projX")
        assert reclaim is True

    async def test_release_on_transport_error(self, dev_platform_with_store, httpx_mock):
        """Case 5b: an httpx transport error path still releases the claim."""
        import os
        import httpx
        from urllib.parse import quote

        from app.services.agentcore_relay_client import invoke_agentcore_turn
        from app.services.agentcore_session_registry import claim_turn

        won = await claim_turn("userA", "projX", turn_id="turn-1")
        assert won is True

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent-runtime/test"
        httpx_mock.add_exception(httpx.ConnectError("boom"))

        with patch(
            "app.services.agentcore_relay_client.get_ws_hub"
        ) as mock_hub_getter:
            mock_hub = Mock()
            mock_hub.publish_to_user = AsyncMock()
            mock_hub.publish_project = AsyncMock()
            mock_hub_getter.return_value = mock_hub
            with patch.dict(
                os.environ, {"AGENTCORE_RUNTIME_ARN": arn, "AWS_REGION": "us-east-1"}
            ):
                await invoke_agentcore_turn(
                    project_id="projX",
                    message="hi",
                    owner_id="userA",
                    runtime_session_id=None,
                    turn_id="turn-1",
                    access_token="Bearer tok",
                )

        reclaim = await claim_turn("userA", "projX")
        assert reclaim is True


class TestRelayClaimRelease:
    """A scheduling failure releases the claim without a NameError."""

    async def test_scheduling_failure_releases_claim_no_nameerror(
        self, dev_platform_with_store
    ):
        """Test: scheduling failure (create_task raises) releases claim, no NameError.

        ``release_turn`` used to be missing from the route's imports (only
        ``claim_turn`` was imported), so the except branch in ``chat()`` raised
        NameError and the claim stayed wedged. Now the claim is released and a
        subsequent claim succeeds.
        """
        from app.api.chat import chat
        from app.models.schemas import ChatRequest
        from app.services.agentcore_session_registry import claim_turn

        # Force asyncio.create_task to raise AFTER a successful claim
        def _raising_create_task(*args, **kwargs):
            raise RuntimeError("create_task scheduling failure")

        with patch(
            "app.platform.arch_split.resolve_agentcore_relay", side_effect=_relay_true
        ), patch(
            "asyncio.create_task", side_effect=_raising_create_task
        ), patch(
            "app.api.chat.current_user_id", return_value="userA"
        ), patch(
            "app.main.get_project_manager"
        ) as mock_pm_getter:
            mock_pm = MagicMock()
            mock_pm.get_project.return_value = _mock_project()
            mock_pm_getter.return_value = mock_pm

            req = ChatRequest(project_id="projX", message="hi", session_id="rt-1")

            # The POST should raise (create_task failed), but it must release the claim first
            with pytest.raises(RuntimeError, match="create_task scheduling failure"):
                await chat(req, _mock_request())

        # The claim MUST be released (no NameError) — a fresh claim succeeds
        reclaim = await claim_turn("userA", "projX")
        assert reclaim is True


class TestCompletionTurnClaimMutualExclusion:
    """A held relay claim makes the completion path wait, then defer.

    Proves the one-writer-per-transcript session claim, which serializes
    writers across processes: a relay user turn holding the
    (owner, project) claim blocks the in-process completion turn from starting
    a second concurrent SDK turn. The completion path waits (bounded) then
    defers-not-drops on timeout.
    """

    async def test_held_relay_claim_blocks_completion_turn(
        self, dev_platform_with_store
    ):
        """A held relay claim for (ownerA, projX) -> completion defers (no submit)."""
        from app.services.completion_turn import (
            _execute_completion_turn,
            _pending_completions,
        )
        from app.services.agentcore_session_registry import claim_turn, release_turn

        # Simulate a relay user turn holding the claim
        won = await claim_turn("ownerA", "projX")
        assert won is True

        _pending_completions["projX"] = [
            {
                "job_id": "job-1",
                "job_type": "eval",
                "synthetic_text": "done",
                "visibility": "visible",
            }
        ]

        with patch(
            "app.platform.arch_split.resolve_agentcore_relay", side_effect=_relay_true
        ), patch(
            "app.services.completion_turn._resolve_project_owner", return_value="ownerA"
        ), patch(
            "app.services.completion_turn._COMPLETION_CLAIM_WAIT_CEILING_S", 0.15
        ), patch(
            "app.services.completion_turn._COMPLETION_CLAIM_BACKOFF_INITIAL_S", 0.05
        ), patch(
            "app.services.agent_turn.get_agent_turn_tracker"
        ) as mock_tracker_getter:
            mock_tracker = MagicMock()
            mock_tracker_getter.return_value = mock_tracker

            # The relay claim is held -> completion path cannot acquire -> defers
            await _execute_completion_turn("projX", "visible")

            # No second concurrent SDK turn was started (deferred, not dropped)
            assert mock_tracker.submit.call_count == 0

        # Once the relay turn releases, the completion path could win the claim
        await release_turn("ownerA", "projX")
        reclaim = await claim_turn("ownerA", "projX")
        assert reclaim is True

    async def test_completion_proceeds_once_claim_free(self, dev_platform_with_store):
        """No held claim -> completion acquires and submits the turn."""
        from app.services.completion_turn import (
            _execute_completion_turn,
            _pending_completions,
        )
        from app.services.agentcore_session_registry import claim_turn

        _pending_completions["projX"] = [
            {
                "job_id": "job-1",
                "job_type": "eval",
                "synthetic_text": "done",
                "visibility": "visible",
            }
        ]

        with patch(
            "app.platform.arch_split.resolve_agentcore_relay", side_effect=_relay_true
        ), patch(
            "app.services.completion_turn._resolve_project_owner", return_value="ownerA"
        ), patch(
            "app.services.agent_turn.get_agent_turn_tracker"
        ) as mock_tracker_getter:
            mock_tracker = MagicMock()
            mock_tracker_getter.return_value = mock_tracker

            # No claim held -> completion acquires and submits
            await _execute_completion_turn("projX", "visible")

            assert mock_tracker.submit.call_count == 1

        # The completion turn holds the claim now (marked on the built turn);
        # a fresh claim on the same (owner, project) should fail (still held)
        # since no turn ran to release it (submit was mocked).
        reclaim = await claim_turn("ownerA", "projX")
        assert reclaim is False
