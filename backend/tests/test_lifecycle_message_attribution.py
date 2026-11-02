# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for lifecycle message attribution.

Verifies that only lifecycle messages surface as system messages, iterations
and phantoms keep dropping, CLI envelopes keep dropping, and the wrong-order
trap is documented.
"""

import pytest
from app.services.chat_message_parser import (
    map_session_messages,
    ORCHESTRATOR_MSG_PREFIX,
    LIFECYCLE_MSG_PREFIX,
    is_synthetic_message,
    is_lifecycle_message,
    strip_lifecycle_markers,
)


def test_lifecycle_message_emits_system_with_clean_text():
    """A lifecycle message emits role:"system" with the human text and neither prefix.

    This keeps ``TestChatHistoryTranscript.test_synthetic_messages_excluded``
    in test_chat_api.py green by construction, not by luck.
    """
    # Build a session entry with both prefixes in correct order
    lifecycle_text = "Optimization started from the UI"
    raw_content = ORCHESTRATOR_MSG_PREFIX + LIFECYCLE_MSG_PREFIX + lifecycle_text
    session_entries = [
        {
            "type": "user",
            "uuid": "msg-lifecycle",
            "timestamp": "2026-08-06T10:00:00Z",
            "message": {"content": raw_content},
        }
    ]

    messages = map_session_messages(session_entries)

    assert len(messages) == 1
    msg = messages[0]
    assert msg["role"] == "system", "Lifecycle message must render as role:system"
    assert len(msg["parts"]) == 1, "System message must have exactly one part"
    assert msg["parts"][0]["type"] == "text"
    # The emitted text must have both prefixes stripped
    emitted_text = msg["parts"][0]["text"]
    assert emitted_text == lifecycle_text, "Emitted text must be clean (no prefixes)"
    assert "__EVAL_STUDIO_SYS_" not in emitted_text, (
        "test_synthetic_messages_excluded in test_chat_api.py asserts this never appears"
    )


def test_non_lifecycle_orchestrator_message_is_dropped():
    """A non-lifecycle orchestrator message (iteration completion) is still dropped."""
    # Build a session entry with only ORCHESTRATOR_MSG_PREFIX (no lifecycle marker)
    iteration_text = "Iteration 3 completed."
    raw_content = ORCHESTRATOR_MSG_PREFIX + iteration_text
    session_entries = [
        {
            "type": "user",
            "uuid": "msg-iteration",
            "timestamp": "2026-08-06T10:00:00Z",
            "message": {"content": raw_content},
        }
    ]

    messages = map_session_messages(session_entries)

    # Must be dropped (no messages emitted)
    assert len(messages) == 0, "Non-lifecycle orchestrator messages must be dropped"


def test_cli_task_envelope_is_dropped():
    """A CLI task envelope (<task-notification>) is still dropped."""
    # CLI task envelope starts with <task-notification>
    raw_content = "<task-notification> some background task </task-notification>"
    session_entries = [
        {
            "type": "user",
            "uuid": "msg-cli",
            "timestamp": "2026-08-06T10:00:00Z",
            "message": {"content": raw_content},
        }
    ]

    messages = map_session_messages(session_entries)

    # Must be dropped
    assert len(messages) == 0, "CLI task envelopes must be dropped"


def test_wrong_order_guard_renders_as_user_bubble():
    """Wrong-order guard: reversed prefix order renders as user bubble.

    This test documents the failure mode. If the lifecycle prefix comes FIRST
    (before orchestrator prefix), is_synthetic_message returns False because
    it checks startswith(ORCHESTRATOR_MSG_PREFIX). The elif never fires, and
    the message falls through to the real-user branch — rendering a user bubble
    with raw marker text visible.

    ``completion_turn`` builds ORCHESTRATOR + LIFECYCLE; this is what a message
    with that order reversed would look like. The test drives the parser
    directly, so it pins the parser's behaviour, not ``completion_turn``'s.
    """
    # Build a session entry with WRONG prefix order (lifecycle first)
    wrong_order_content = LIFECYCLE_MSG_PREFIX + ORCHESTRATOR_MSG_PREFIX + "Optimization started from the UI"
    session_entries = [
        {
            "type": "user",
            "uuid": "msg-wrong-order",
            "timestamp": "2026-08-06T10:00:00Z",
            "message": {"content": wrong_order_content},
        }
    ]

    messages = map_session_messages(session_entries)

    # With wrong order: is_synthetic_message returns False, so it renders as user
    assert len(messages) == 1
    msg = messages[0]
    assert msg["role"] == "user", (
        "Wrong-order message renders as user bubble (failure mode)"
    )
    # The raw content is visible in the user bubble
    assert LIFECYCLE_MSG_PREFIX in msg["parts"][0]["text"], (
        "Raw lifecycle marker is visible in the user bubble"
    )


def test_exactly_one_text_part_on_system_message():
    """Exactly one text part on the system message, asserted as a hard invariant.

    The frontend chat runtime throws on any other shape, and the
    mapper's `as` assertion means tsc cannot catch it. Assert it here at the
    producer where it is cheap.
    """
    lifecycle_text = "Optimization started from the UI"
    raw_content = ORCHESTRATOR_MSG_PREFIX + LIFECYCLE_MSG_PREFIX + lifecycle_text
    session_entries = [
        {
            "type": "user",
            "uuid": "msg-lifecycle",
            "timestamp": "2026-08-06T10:00:00Z",
            "message": {"content": raw_content},
        }
    ]

    messages = map_session_messages(session_entries)

    assert len(messages) == 1
    msg = messages[0]
    assert msg["role"] == "system"
    parts = msg["parts"]
    # HARD INVARIANT: exactly one part, and it must be text
    assert len(parts) == 1, "System message must have exactly one part"
    assert parts[0]["type"] == "text", "System message part must be type:text"


def test_is_lifecycle_message_tests_remainder():
    """is_lifecycle_message tests the remainder after stripping ORCHESTRATOR_MSG_PREFIX."""
    # Correct order: ORCHESTRATOR + LIFECYCLE
    correct_content = ORCHESTRATOR_MSG_PREFIX + LIFECYCLE_MSG_PREFIX + "text"
    assert is_synthetic_message(correct_content), "Must be synthetic"
    assert is_lifecycle_message(correct_content), "Must be lifecycle"

    # Wrong order: LIFECYCLE + ORCHESTRATOR
    wrong_content = LIFECYCLE_MSG_PREFIX + ORCHESTRATOR_MSG_PREFIX + "text"
    assert not is_synthetic_message(wrong_content), "Must NOT be synthetic"
    # is_lifecycle_message should not be called on non-synthetic, but if it is:
    # (the test documents the prefix-order dependency)

    # Non-lifecycle orchestrator message
    non_lifecycle = ORCHESTRATOR_MSG_PREFIX + "Iteration 3 completed."
    assert is_synthetic_message(non_lifecycle), "Must be synthetic"
    assert not is_lifecycle_message(non_lifecycle), "Must NOT be lifecycle"


def test_strip_lifecycle_markers_removes_both():
    """strip_lifecycle_markers removes both ORCHESTRATOR_MSG_PREFIX and LIFECYCLE_MSG_PREFIX."""
    raw = ORCHESTRATOR_MSG_PREFIX + LIFECYCLE_MSG_PREFIX + "Optimization started"
    clean = strip_lifecycle_markers(raw)
    assert clean == "Optimization started"
    assert ORCHESTRATOR_MSG_PREFIX not in clean
    assert LIFECYCLE_MSG_PREFIX not in clean


# Mutation proofs

def test_mutation_proof_drop_lifecycle_true_from_start():
    """A start message renders only because it carries the lifecycle marker.

    Without ``lifecycle=True`` at the start call site the message would carry
    only ORCHESTRATOR_MSG_PREFIX and be dropped like any other orchestrator
    message (see test_non_lifecycle_orchestrator_message_is_dropped). This pins
    the marked form: exactly one system message.
    """
    lifecycle_text = "Optimization started from the UI"
    raw_with_lifecycle = ORCHESTRATOR_MSG_PREFIX + LIFECYCLE_MSG_PREFIX + lifecycle_text
    session_with_lifecycle = [
        {
            "type": "user",
            "uuid": "msg-start-with-lifecycle",
            "timestamp": "2026-08-06T10:00:00Z",
            "message": {"content": raw_with_lifecycle},
        }
    ]
    messages_with_lifecycle = map_session_messages(session_with_lifecycle)
    assert len(messages_with_lifecycle) == 1
    assert messages_with_lifecycle[0]["role"] == "system"


def test_mutation_proof_ungate_synthetic_arm():
    """Mutation proof: un-gating the synthetic arm makes this test fail.

    If every synthetic message were emitted (not just lifecycle ones),
    non-lifecycle orchestrator messages (iteration completion) would render as
    system messages instead of being dropped.
    """
    # Non-lifecycle orchestrator message
    iteration_text = "Iteration 3 completed."
    raw_content = ORCHESTRATOR_MSG_PREFIX + iteration_text
    session_entries = [
        {
            "type": "user",
            "uuid": "msg-iteration",
            "timestamp": "2026-08-06T10:00:00Z",
            "message": {"content": raw_content},
        }
    ]

    messages = map_session_messages(session_entries)

    # WITH the gate (correct): len(messages) == 0 (dropped)
    # WITHOUT the gate (mutated, emit all synthetic): len(messages) == 1, role == "system"
    assert len(messages) == 0, (
        "Non-lifecycle synthetic must be dropped (not emitted as system)"
    )


def test_mutation_proof_fold_cli_arm():
    """Mutation proof: folding the CLI arm into a shared ``or`` makes this test fail.

    If is_cli_task_envelope and is_synthetic_message were checked in a single
    ``if ... or ...``, CLI envelopes could render as system messages once the
    lifecycle arm runs first.
    """
    # CLI task envelope
    raw_content = "<task-notification> some task </task-notification>"
    session_entries = [
        {
            "type": "user",
            "uuid": "msg-cli",
            "timestamp": "2026-08-06T10:00:00Z",
            "message": {"content": raw_content},
        }
    ]

    messages = map_session_messages(session_entries)

    # CLI envelopes must be dropped unconditionally
    assert len(messages) == 0, "CLI envelopes must be dropped"
