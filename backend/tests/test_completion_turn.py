# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that completion turns respect the relay's one-writer-per-transcript session claim.

Proves the completion/orchestrator turn path takes the SAME per-(user,project)
DDB claim a relay user turn takes when relay is enabled — the cross-process
serializer that prevents an eval-completion turn and a relayed user turn from
writing session+transcript concurrently. Behavioral (trip-the-branch) tests per
the dispatch guards.

  1. Relay flag OFF -> no claim taken, existing tracker path unchanged.
  2. Relay flag ON -> completion acquires the claim, marks the built turn.
  3. Mutual-exclusion: a held claim makes the completion path WAIT (defer path).
  4. Defer-not-drop: acquire timeout -> turn NOT run, result NOT dropped, warning.
  5. Release-on-all-exits: run_agent_turn finally releases the claim on exit.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from app.services.completion_turn import _claim_release_tasks

from .async_wait import until

pytestmark = pytest.mark.asyncio


@pytest.fixture
def dev_platform_with_store(projects_dir: Path, monkeypatch):
    """Set up dev platform with REAL FilesystemMetadataStore for claim tests.

    The claim goes through the actual conditional-write path so mutual-exclusion
    is a genuine held-claim (not a mocked return).
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


async def _relay_true(*args, **kwargs):
    return True


async def _relay_false(*args, **kwargs):
    return False


class TestCompletionTurnRelayOff:
    """Relay flag OFF: existing path unchanged, no claim taken."""

    async def test_relay_off_no_claim_taken(self, dev_platform_with_store):
        """Relay OFF -> completion submits via tracker WITHOUT taking a claim."""
        from app.services.completion_turn import (
            _execute_completion_turn,
            _pending_completions,
        )

        # Seed a pending completion message so _execute_completion_turn proceeds
        _pending_completions["projX"] = [
            {
                "job_id": "job-1",
                "job_type": "eval",
                "synthetic_text": "done",
                "visibility": "visible",
            }
        ]

        with patch(
            "app.platform.arch_split.resolve_agentcore_relay", side_effect=_relay_false
        ), patch(
            "app.services.agent_turn.get_agent_turn_tracker"
        ) as mock_tracker_getter, patch(
            "app.services.agentcore_session_registry.claim_turn",
            new_callable=AsyncMock,
        ) as mock_claim:
            mock_tracker = MagicMock()
            mock_tracker_getter.return_value = mock_tracker

            await _execute_completion_turn("projX", "visible")

        # Existing tracker path ran; claim_turn was NOT called
        assert mock_tracker.submit.call_count == 1
        assert mock_claim.call_count == 0


class TestCompletionTurnRelayOn:
    """Relay flag ON: completion acquires the shared claim + marks the turn."""

    async def test_relay_on_acquires_claim_and_marks_turn(
        self, dev_platform_with_store
    ):
        """Relay ON -> claim acquired, built turn marked _holds_ddb_claim."""
        from app.services.completion_turn import (
            _execute_completion_turn,
            _pending_completions,
        )

        _pending_completions["projX"] = [
            {
                "job_id": "job-1",
                "job_type": "eval",
                "synthetic_text": "done",
                "visibility": "visible",
            }
        ]

        built_turn = MagicMock()
        built_turn.turn_id = "turn-1"

        # Capture the build_turn callable so we can invoke it and inspect the turn
        captured = {}

        def _fake_submit(
            project_id, origin, build_turn, priority=None, on_drop=None, turn_id=None
        ):
            captured["build_turn"] = build_turn
            captured["on_drop"] = on_drop
            captured["turn_id"] = turn_id
            return turn_id or "turn-1"

        with patch(
            "app.platform.arch_split.resolve_agentcore_relay", side_effect=_relay_true
        ), patch(
            "app.services.completion_turn._resolve_project_owner", return_value="userA"
        ), patch(
            "app.services.agent_turn.get_agent_turn_tracker"
        ) as mock_tracker_getter, patch(
            "app.services.agent_runner.start_agent_turn", return_value=built_turn
        ), patch(
            "app.main.get_project_manager"
        ) as mock_pm, patch(
            "app.services.coordinator_prompt.build_coordinator_prompt",
            return_value="sys",
        ), patch(
            "app.services.project_paths.resolve_project_dir", return_value="/tmp/p"
        ):
            project = MagicMock()
            project.session_id = "s"
            project.agent_model = "opus"
            mock_pm.return_value.get_project.return_value = project

            mock_tracker = MagicMock()
            mock_tracker.submit.side_effect = _fake_submit
            mock_tracker_getter.return_value = mock_tracker

            await _execute_completion_turn("projX", "visible")

            # The claim was won (real store) — invoke build_turn to verify marking
            assert "build_turn" in captured
            # The claim is taken under a reserved id, and that id is what the
            # builder must run under, so the fenced release recognises its own
            # claim rather than releasing whatever is running.
            assert captured["turn_id"], "the claim's turn id must reach submit"
            captured["build_turn"](captured["turn_id"])

        # The built turn was marked as claim-holder (release fires in run_agent_turn)
        assert built_turn._holds_ddb_claim is True

    async def test_held_claim_makes_completion_defer(self, dev_platform_with_store):
        """Defer-not-drop: a held claim past the bound -> turn NOT run.

        Pre-claim (userA, projX), set a tiny wait ceiling so the acquire-or-wait
        loop times out fast, assert the completion turn defers (tracker.submit
        NOT called). The result is NOT dropped — it stays on the durable Job
        record for the next turn/reconciler; the completion turn simply returns.
        """
        from app.services.completion_turn import (
            _execute_completion_turn,
            _pending_completions,
        )
        from app.services.agentcore_session_registry import claim_turn, release_turn

        # Hold the claim so the completion path cannot acquire it
        won = await claim_turn("userA", "projX")
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
            "app.services.completion_turn._resolve_project_owner", return_value="userA"
        ), patch(
            "app.services.completion_turn._COMPLETION_CLAIM_WAIT_CEILING_S", 0.2
        ), patch(
            "app.services.completion_turn._COMPLETION_CLAIM_BACKOFF_INITIAL_S", 0.05
        ), patch(
            "app.services.agent_turn.get_agent_turn_tracker"
        ) as mock_tracker_getter:
            mock_tracker = MagicMock()
            mock_tracker_getter.return_value = mock_tracker

            await _execute_completion_turn("projX", "visible")

            # Deferred: the turn was NOT submitted (claim never acquired)
            assert mock_tracker.submit.call_count == 0

        # The claim is still held by userA (completion did NOT steal it).
        # The holder's release lets a subsequent claim win.
        await release_turn("userA", "projX")
        reclaim = await claim_turn("userA", "projX")
        assert reclaim is True


class TestCompletionTurnClaimRelease:
    """The claim is released on ALL turn exits."""

    async def test_run_agent_turn_finally_releases_claim(
        self, dev_platform_with_store
    ):
        """run_agent_turn finally releases the DDB claim on exit (failure path)."""
        from app.services.agent_runner import run_agent_turn
        from app.services.agent_turn import AgentTurn
        from app.services.agentcore_session_registry import claim_turn

        # Create a turn that holds the DDB claim
        turn = AgentTurn(turn_id="turn-1", project_id="projX", origin="orchestrator")
        turn._holds_ddb_claim = True
        turn.owner_id = "userA"

        # Pre-claim under this turn's id, as the completion path does. The
        # release is fenced on the turn id, so a claim recorded without one
        # could only be cleared by an unfenced release.
        won = await claim_turn("userA", "projX", turn_id="turn-1")
        assert won is True

        # Force run_agent_turn to fail fast so we reach the finally.
        # ClaudeSDKClient is imported inside run_agent_turn — patch it to raise.
        with patch(
            "claude_agent_sdk.ClaudeSDKClient",
            side_effect=RuntimeError("Mock SDK error"),
        ):
            # run_agent_turn catches exceptions internally (turn.fail), so it
            # should not re-raise — but the finally MUST release the claim.
            await run_agent_turn(turn, "test prompt", "projX", "coordinator prompt")

        # The claim MUST be released (reclaim succeeds)
        reclaim = await claim_turn("userA", "projX")
        assert reclaim is True

    async def test_build_failure_releases_claim(self, dev_platform_with_store):
        """If _build fails (queue swallows it), the claim is released.

        The tracker's _try_process_queue swallows build_turn exceptions, so the
        finally-release in run_agent_turn never fires (no turn was built). The
        _build closure must release the claim itself via _schedule_claim_release.
        """
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

        # Use the REAL tracker so _try_process_queue actually runs _build + swallows
        from app.services.agent_turn import get_agent_turn_tracker

        get_agent_turn_tracker()._reset()

        with patch(
            "app.platform.arch_split.resolve_agentcore_relay", side_effect=_relay_true
        ), patch(
            "app.services.completion_turn._resolve_project_owner", return_value="userA"
        ), patch(
            "app.main.get_project_manager"
        ) as mock_pm:
            # Force _build to fail: project not found
            mock_pm.return_value.get_project.return_value = None

            await _execute_completion_turn("projX", "visible")

            # Join the scheduled release rather than guessing how long it needs.
            # ``_schedule_claim_release`` registers the task in
            # ``_claim_release_tasks`` and discards it on completion, so an empty
            # registry is the release having finished.
            await until(
                lambda: not _claim_release_tasks,
                message="the scheduled claim release never completed",
            )

        # The claim MUST have been released by _schedule_claim_release
        reclaim = await claim_turn("userA", "projX")
        assert reclaim is True

    async def test_drained_queued_turn_releases_claim(self, dev_platform_with_store):
        """Cancel of a QUEUED completion turn releases the DDB claim.

        The completion turn acquires the shared claim BEFORE tracker.submit. If a
        turn is already running for the project, the completion turn is QUEUED —
        its _build never runs, so neither run_agent_turn's finally nor the
        _build-failure release fires. Cancel (drain_queued_turns) must invoke the
        on_drop hook so the claim is released — otherwise it leaks until the 2h
        TTL backstop, wedging the project's relay serialization. Acceptance: the
        claim goes IDLE (a fresh claim wins), not merely "hook registered".
        """
        from app.services.completion_turn import (
            _execute_completion_turn,
            _pending_completions,
        )
        from app.services.agentcore_session_registry import claim_turn
        from app.services.agent_turn import AgentTurn, get_agent_turn_tracker

        tracker = get_agent_turn_tracker()
        tracker._reset()

        # Occupy the project with a RUNNING turn so the completion turn queues
        # (submit does not build it immediately).
        running = AgentTurn(turn_id="running-user-turn", project_id="projX", origin="user")
        tracker._turns[running.turn_id] = running
        tracker._by_project["projX"] = running.turn_id
        tracker._processing.add("projX")

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
            "app.services.completion_turn._resolve_project_owner", return_value="userA"
        ):
            await _execute_completion_turn("projX", "visible")

            # The completion turn acquired the claim and QUEUED (did not build).
            queued = tracker.get_queued_turns("projX")
            assert len(queued) == 1
            assert queued[0].origin == "orchestrator"
            assert queued[0].on_drop is not None
            # Claim is currently held (a re-claim by the same owner fails).
            assert await claim_turn("userA", "projX") is False

            # Cancel: drain the queued orchestrator turn (the optimization cancel path).
            removed = tracker.drain_queued_turns("projX", origin="orchestrator")
            assert removed == 1

        # on_drop schedules an async release — join it via the task registry
        # rather than assuming a fixed duration is long enough.
        await until(
            lambda: not _claim_release_tasks,
            message="the scheduled claim release never completed",
        )

        # Acceptance: the claim is now IDLE — a fresh claim wins.
        assert await claim_turn("userA", "projX") is True

    async def test_drained_terminal_message_turn_releases_claim(
        self, dev_platform_with_store
    ):
        """Cancel of a QUEUED terminal-message turn releases the DDB claim.

        ``_submit_orchestrator_message`` takes the same per-(user,project) claim as
        the job-results completion turn, but submitted WITHOUT an ``on_drop`` hook
        — so a queued terminal-message turn drained by cancel released the claim
        nowhere: not via run_agent_turn's finally (the turn never ran), not via the
        _build-failure path (_build never ran), not via a drop hook (absent).

        The claim then survives to its TTL. And per the DynamoDB TTL contract,
        deletion lags expiry by up to a few days, so the "2h TTL backstop" does
        not bound the outage — the pair stays wedged on a long-lived coordinator.

        Acceptance mirrors the job-results sibling: the claim goes IDLE (a fresh
        claim WINS), not merely "a hook was registered".
        """
        from app.services.completion_turn import _submit_orchestrator_message
        from app.services.agentcore_session_registry import claim_turn
        from app.services.agent_turn import AgentTurn, get_agent_turn_tracker

        tracker = get_agent_turn_tracker()
        tracker._reset()

        # Occupy the project with a RUNNING turn so the terminal message queues
        # instead of building immediately.
        running = AgentTurn(
            turn_id="running-user-turn", project_id="projY", origin="user"
        )
        tracker._turns[running.turn_id] = running
        tracker._by_project["projY"] = running.turn_id
        tracker._processing.add("projY")

        with patch(
            "app.platform.arch_split.resolve_agentcore_relay", side_effect=_relay_true
        ), patch(
            "app.services.completion_turn._resolve_project_owner", return_value="userB"
        ):
            await _submit_orchestrator_message("projY", "The loop ended this way.")

            # The terminal message acquired the claim and QUEUED.
            queued = tracker.get_queued_turns("projY")
            assert len(queued) == 1
            assert queued[0].origin == "orchestrator"
            assert queued[0].on_drop is not None, (
                "A queued terminal-message turn holding the claim MUST carry an "
                "on_drop hook — without it a cancel-drain leaks the claim."
            )
            # Claim is genuinely held (a re-claim by the same owner fails).
            assert await claim_turn("userB", "projY") is False

            # Cancel: drain the queued orchestrator turn.
            removed = tracker.drain_queued_turns("projY", origin="orchestrator")
            assert removed == 1

        # on_drop schedules an async release — join it via the task registry
        # rather than assuming a fixed duration is long enough.
        await until(
            lambda: not _claim_release_tasks,
            message="the scheduled claim release never completed",
        )

        # Acceptance: the claim is now IDLE — a fresh claim wins.
        assert await claim_turn("userB", "projY") is True, (
            "The claim must be released when the queued terminal-message turn is "
            "dropped; otherwise the (user, project) pair is wedged until the TTL "
            "item is actually deleted (days, per the DynamoDB TTL contract)."
        )
