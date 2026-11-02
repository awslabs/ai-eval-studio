# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The transcript's ``turn_id`` stamp — the SPA's only handle on "did THIS
turn's reply land?".

Why the field exists: the SPA replaces the whole thread when an orchestrator
turn finalizes, but the reply is appended by a separate best-effort write that
can land AFTER the finalize signal. When it does, the streamed bubble is
replaced by a transcript that does not have the reply yet, and only comes
back on refresh.

The client cannot detect that on its own: it mints its own
``crypto.randomUUID()`` for the finalized bubble, so there is no identity link,
and every count-based proxy breaks somewhere (a marker grows the count without
being the reply; a turn with no reply never grows it; two in-flight turns cannot
be told apart). A real field is the discriminator.

Stamping is deliberately narrow — the trailing assistant message only, and only
when the caller has already established the reply is present. The transcript's
body is immutable on append, so a wrong stamp is permanent.
"""

from __future__ import annotations

from app.services.chat_transcript import (
    build_turn_marker_message,
    tag_turn_on_trailing_reply,
)


def _assistant(text: str = "reply") -> dict:
    return {
        "id": "a1",
        "role": "assistant",
        "parts": [{"type": "text", "text": text}],
        "created_at": "2026-09-08T10:00:02Z",
    }


def _user(text: str = "prompt") -> dict:
    return {
        "id": "u1",
        "role": "user",
        "parts": [{"type": "text", "text": text}],
        "created_at": "2026-09-08T10:00:00Z",
    }


class TestTagTurnOnTrailingReply:
    def test_stamps_the_trailing_assistant_message(self):
        mapped = [_user(), _assistant()]
        tag_turn_on_trailing_reply(mapped, "turn-abc")
        assert mapped[-1]["turn_id"] == "turn-abc"

    def test_stamps_only_the_tail_not_earlier_assistant_messages(self):
        """Earlier assistant messages belong to earlier turns. Stamping them
        would let the guard match a stale transcript on an old message."""
        earlier = _assistant("older reply")
        earlier["id"] = "a0"
        mapped = [earlier, _user(), _assistant("this turn")]
        tag_turn_on_trailing_reply(mapped, "turn-abc")
        assert mapped[-1]["turn_id"] == "turn-abc"
        assert "turn_id" not in earlier

    def test_no_op_when_the_tail_is_not_an_assistant_message(self):
        """The mapper drops hidden prompts, so the tail can belong to the
        PREVIOUS turn. Body-immutability makes a wrong stamp permanent, so a
        non-assistant tail must be left alone."""
        mapped = [_assistant(), _user()]
        tag_turn_on_trailing_reply(mapped, "turn-abc")
        assert "turn_id" not in mapped[-1]

    def test_no_op_on_a_falsy_turn_id(self):
        """A caller without a turn id must not write an empty stamp that a
        client guard could then match against."""
        mapped = [_user(), _assistant()]
        tag_turn_on_trailing_reply(mapped, None)
        tag_turn_on_trailing_reply(mapped, "")
        assert "turn_id" not in mapped[-1]

    def test_no_op_on_an_empty_mapping(self):
        mapped: list[dict] = []
        tag_turn_on_trailing_reply(mapped, "turn-abc")
        assert mapped == []

    def test_does_not_disturb_role_or_parts(self):
        """``append_messages`` keys dedupe equality on ``(role, parts)`` only
        (chat_transcript.py), so the stamp must not touch either — otherwise it
        would look like changed content and trigger a tail rewrite."""
        mapped = [_user(), _assistant("exact text")]
        before_role = mapped[-1]["role"]
        before_parts = [dict(p) for p in mapped[-1]["parts"]]
        tag_turn_on_trailing_reply(mapped, "turn-abc")
        assert mapped[-1]["role"] == before_role
        assert mapped[-1]["parts"] == before_parts


class TestMarkerCarriesTurnId:
    def test_stopped_marker_carries_the_turn_id_as_a_field(self):
        """A turn that legitimately produces no reply ends in a marker. Without
        the field on markers too, the client's guard would never converge for
        stopped/failed turns and would retry to exhaustion every time."""
        msg = build_turn_marker_message("turn-xyz", "Stopped.", kind="stopped")
        assert msg["turn_id"] == "turn-xyz"

    def test_failed_marker_carries_the_turn_id_as_a_field(self):
        msg = build_turn_marker_message("turn-xyz", "Turn failed.", kind="failed")
        assert msg["turn_id"] == "turn-xyz"

    def test_marker_is_still_assistant_role_with_the_italic_convention(self):
        """Guard against the stamp being added by breaking the existing shape."""
        msg = build_turn_marker_message("turn-xyz", "Stopped.", kind="stopped")
        assert msg["role"] == "assistant"
        assert msg["id"] == "stopped-turn-xyz"
        assert msg["parts"] == [{"type": "text", "text": "_(Stopped.)_"}]


class TestBothWritePathsTagAtOneChokepoint:
    """Both write paths must set the field, or the same
    logical reply differs by which process wrote it.

    They do, structurally rather than by duplication: the in-process path
    (``agent_runner`` at the ResultMessage handler) and the coordinator-side
    relay path (``agentcore_relay_client``) BOTH call
    ``_append_turn_to_transcript(..., turn_id=turn_id)``, and the stamp happens
    inside that function. This test pins that single-chokepoint property, since
    a future refactor that gave either path its own append would silently
    reintroduce the split.
    """

    def test_relay_reaches_the_stamp_through_append_turn_to_transcript(self):
        import inspect

        from app.services import agentcore_relay_client

        source = inspect.getsource(agentcore_relay_client)
        assert "_append_turn_to_transcript" in source, (
            "the relay must reach the transcript through the shared append "
            "helper — a private append here would skip the turn_id stamp"
        )

    def test_the_shared_helper_stamps_the_turn(self):
        import inspect

        from app.services import agent_runner

        source = inspect.getsource(agent_runner._append_turn_to_transcript)
        assert "tag_turn_on_trailing_reply" in source, (
            "_append_turn_to_transcript is the one place both write paths "
            "converge; the stamp has to happen here"
        )

    def test_the_backfill_also_stamps(self):
        """The bounded backfill appends this turn's reply on its own, after
        proving the reply flushed — it needs the stamp too or a reply that
        arrives late lands unstamped, which is exactly the race case."""
        import inspect

        from app.services import agent_runner

        source = inspect.getsource(agent_runner._backfill_transcript_when_flushed)
        assert "tag_turn_on_trailing_reply" in source
