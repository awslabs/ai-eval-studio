# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A failed turn must leave a durable explanation, not just a toast.

THE SYMPTOM. Turn errors are published as a ``ResourceError``
over the WebSocket, which becomes a toast. The durable transcript stores only
user/assistant messages. So after a failed turn plus a refresh, the user's own
prompt sits there with no reply and no explanation — an expired-credentials
notice vanished on reload, leaving a conversation that looks like the agent
simply ignored the question.

THE SHAPE OF THE FIX is not new: the relay-cancel path already appends an
assistant-role "stopped" marker for exactly this reason, and deliberately relies
on it changing the transcript's tail role so ``GET /chat/history`` stops
offering "Resume" for a turn that is not resumable-by-retry. Failures reuse that
mechanism through one shared builder rather than a second hand-rolled copy.

WHAT IS ACTUALLY LOAD-BEARING, and therefore tested here:

* the marker is written at the turn's single exit chokepoint, because the four
  ``_publish_cli_error`` sites straddle the transcript append — two fire before
  it, one after, one on a path that never reaches it. Any earlier site covers
  some failures and silently misses others;
* it is SKIPPED inside the AgentCore runtime. The transcript is a one-process
  write target and a second process's appends were silently dropped by the
  other's atomic rename. Getting this wrong is invisible in dev
  and corrupts transcripts in cloud;
* a user-stopped turn gets no failure marker — the cancel path already wrote a
  "stopped" one, and two contradicting markers is worse than none.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from app.services.agent_runner import (
    _persist_turn_failure_marker,
    turn_failure_marker_text,
)
from app.services.chat_transcript import build_turn_marker_message
from app.services.resource_error import ResourceError


class _Turn:
    """Minimal stand-in for AgentTurn — only what the marker logic reads."""

    def __init__(
        self,
        *,
        status: str = "failed",
        failure_text: str | None = None,
        error: ResourceError | None = None,
        stopped_by_user: bool = False,
    ) -> None:
        self.turn_id = "turn-abc123"
        self.status = status
        self.error = error
        self._stopped_by_user = stopped_by_user
        if failure_text is not None:
            self._failure_text = failure_text


class TestMarkerText:
    """Which turns get a marker, and whose wording wins."""

    def test_a_cli_failure_uses_the_CLI_wording(self):
        """The actionable text, not a generic code.

        "Your session has expired. Please reauthenticate." is what the user needs;
        ``authentication_failed`` is not.
        """
        turn = _Turn(
            failure_text=(
                "API Error: Your session has expired. Please reauthenticate. "
                "Re-authenticate the AWS credentials Bedrock is using and retry."
            )
        )
        text = turn_failure_marker_text(turn)
        assert text is not None
        assert "session has expired" in text
        assert text.startswith("Turn failed:")

    def test_a_completed_turn_gets_no_marker(self):
        assert turn_failure_marker_text(_Turn(status="completed")) is None

    def test_a_user_stopped_turn_gets_no_marker(self):
        """The cancel path already wrote a "stopped" marker for this turn.

        A second, contradicting "failed" marker would tell the user their own
        deliberate Stop was an error.
        """
        turn = _Turn(failure_text="something went wrong", stopped_by_user=True)
        assert turn_failure_marker_text(turn) is None

    def test_a_completed_turn_wins_over_a_stashed_error(self):
        """A recoverable mid-turn error must not mark a turn that then succeeded.

        ``_publish_cli_error`` can fire on an AssistantMessage error the turn goes
        on to recover from (the empty-result retry does exactly this). The final
        status is the authority, not the fact that an error was once seen.
        """
        turn = _Turn(status="completed", failure_text="transient blip")
        assert turn_failure_marker_text(turn) is None

    def test_falls_back_to_the_turns_own_resource_error(self):
        """A turn killed by a Python exception has no CLI text, but has an error."""
        turn = _Turn(
            error=ResourceError(
                code="agent.crashed",
                message="Harness subprocess died",
                hint="Check the backend logs.",
            )
        )
        text = turn_failure_marker_text(turn)
        assert text is not None
        assert "Harness subprocess died" in text
        assert "Check the backend logs." in text

    def test_a_failure_with_no_recorded_reason_gets_no_marker(self):
        """A bare "this failed" bubble adds a claim without adding information.

        The failed status is already visible in the UI; inventing prose for it is
        not an improvement, and it would fire on every cancelled-by-teardown turn
        that reaches ``finally`` still marked running.
        """
        assert turn_failure_marker_text(_Turn()) is None


class TestMarkerBuilder:
    """One builder for both marker kinds — stop and failure."""

    def test_the_marker_is_an_assistant_message_in_the_current_schema(self):
        """Assistant role is what makes it render with no new part type, and what
        changes the tail role the interruption heuristic reads."""
        marker = build_turn_marker_message("turn-1", "Turn failed: nope", kind="failed")
        assert marker["role"] == "assistant"
        assert marker["parts"] == [{"type": "text", "text": "_(Turn failed: nope)_"}]

    def test_created_at_is_non_empty(self):
        """LOAD-BEARING: ``is_current_schema`` requires it, so a marker without one
        makes the whole transcript look stale-schema and get DISCARDED on the next
        history load — losing history to add an explanation."""
        from app.services.chat_transcript import is_current_schema

        marker = build_turn_marker_message("turn-1", "Turn failed: nope", kind="failed")
        assert marker["created_at"]
        assert marker["created_at"].endswith("Z")
        assert is_current_schema([marker])

    def test_the_id_is_deterministic_per_turn_and_kind(self):
        """``append_messages`` dedupes on id, so a replay must not double up —
        and stop vs failed must never collide on one turn."""
        a = build_turn_marker_message("turn-1", "x", kind="failed")
        b = build_turn_marker_message("turn-1", "y", kind="failed")
        stopped = build_turn_marker_message("turn-1", "x", kind="stopped")
        assert a["id"] == b["id"]
        assert a["id"] != stopped["id"]

    def test_the_stop_marker_id_is_unchanged_by_the_refactor(self):
        """The cancel path's id was ``stopped-{turn_id}`` when hand-rolled.

        Pinned because the id is the dedupe key: changing it would make a
        replayed cancel append a SECOND stop bubble instead of a no-op.
        """
        marker = build_turn_marker_message("turn-1", "Turn stopped by user.",
                                           kind="stopped")
        assert marker["id"] == "stopped-turn-1"


class TestPersistence:
    """Where the marker is written, and where it must not be."""

    def test_it_appends_the_marker_for_a_failed_turn(self):
        turn = _Turn(failure_text="Your session has expired.")
        appended: list[list[dict]] = []

        async def _fake_append(project_id, msgs):
            appended.append(msgs)

        with patch("app.services.chat_transcript.append_messages", new=_fake_append), \
             patch("app.platform.process_role.is_agentcore_runtime",
                   return_value=False):
            asyncio.run(_persist_turn_failure_marker("proj-1", turn))

        assert len(appended) == 1
        assert appended[0][0]["role"] == "assistant"
        assert "session has expired" in appended[0][0]["parts"][0]["text"]

    def test_it_is_SKIPPED_inside_the_agentcore_runtime(self):
        """Single writer. The transcript is a one-process write target: the
        coordinator and the microVM each serialize behind their OWN process-local
        lock, and the atomic whole-file rename silently dropped lines the other
        process appended after its read. The coordinator's relay
        drain writes this marker for relay turns instead.
        """
        turn = _Turn(failure_text="Your session has expired.")
        appended: list[list[dict]] = []

        async def _fake_append(project_id, msgs):
            appended.append(msgs)

        with patch("app.services.chat_transcript.append_messages", new=_fake_append), \
             patch("app.platform.process_role.is_agentcore_runtime",
                   return_value=True):
            asyncio.run(_persist_turn_failure_marker("proj-1", turn))

        assert appended == [], (
            "the runtime wrote the transcript — that is the two-writer shape that "
            "dropped lines in production"
        )

    def test_a_transcript_failure_never_propagates(self):
        """This runs inside the turn's ``finally``. An exception here would mask
        whatever actually killed the turn, and a transcript write must never
        change how a turn ends."""
        turn = _Turn(failure_text="Your session has expired.")

        async def _boom(project_id, msgs):
            raise OSError("disk full")

        with patch("app.services.chat_transcript.append_messages", new=_boom), \
             patch("app.platform.process_role.is_agentcore_runtime",
                   return_value=False):
            # Must not raise.
            asyncio.run(_persist_turn_failure_marker("proj-1", turn))

    def test_nothing_is_written_when_there_is_no_marker_to_write(self):
        """No append call at all for a completed turn — not an empty one."""
        appended: list[list[dict]] = []

        async def _fake_append(project_id, msgs):
            appended.append(msgs)

        with patch("app.services.chat_transcript.append_messages", new=_fake_append), \
             patch("app.platform.process_role.is_agentcore_runtime",
                   return_value=False):
            asyncio.run(
                _persist_turn_failure_marker("proj-1", _Turn(status="completed"))
            )

        assert appended == []
