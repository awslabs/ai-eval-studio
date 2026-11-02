# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-process project-event frame contract for AgentCore relay.

This module defines the wire shape for carrying a project event from the AgentCore
runtime microVM to the coordinator Fargate service across the turn's SSE stream.
Both the producer (microVM injection) and the consumer (coordinator re-target) live
in this module so the two execution homes cannot drift apart on the payload structure.

**Why this exists:** MCP tools executing inside the AgentCore microVM need to publish
project events (e.g., ``delete_eval_run`` publishing ``eval_run/deleted``) to the
project owner's browser-facing WebSocket connections. The microVM holds no browser
connections, so those publishes would reach nobody. Instead the microVM appends the event
into the live ``AgentTurn.events`` buffer, ``agentcore_runtime._sse_frames`` emits it
as an SSE frame (unchanged — it is already kind-generic), and the coordinator's drain
in ``agentcore_relay_client`` re-targets it onto the project stream where the SPA's
existing ``useBackendEvents`` consumer listens.

**Why in app/services/:** This contract is a mechanism for one service (jobs) to hand
an event to another service (agent_turn → agentcore_relay_client → ws_hub). It sits
alongside the other cross-service glue in ``services/``: ``ws_envelope`` (envelope
construction), ``ws_hub`` (WebSocket fan-out), and ``background_work`` (durable work
markers).

**Seq domain preservation (critical):** The microVM ships the RAW event payload, never
a pre-minted envelope. The coordinator calls ``ws_hub.publish_project`` which mints the
``project:<id>`` envelope using the coordinator's own in-memory counter, so the project
stream keeps exactly one seq source. Had the microVM minted a full ``project:`` envelope,
its counter (fresh at 0 each microVM session) would collide with the coordinator's, and
the client would silently drop frames as duplicates or burn a visible domain reset.
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# The envelope kind string for project-event frames.
# Reuses the existing kind from the envelope vocabulary (see ws_envelope module docstring)
# so the kind reads as "this is a project event" while the destination stream is
# reconstructed on the coordinator side.
PROJECT_EVENT_FRAME_KIND = "project_event"

# Retained fire-and-forget tasks so they aren't GC'd mid-flight
_forwarding_tasks: set[asyncio.Task] = set()


def build_project_event_frame(project_id: str, event: dict[str, Any]) -> dict[str, Any]:
    """Build the envelope data payload for a project-event frame.

    The project_id travels across the boundary not as a routing input, but as an
    assertion the coordinator checks against the id it already holds. This prevents
    a frame from naming a project the invocation was not for.

    Args:
        project_id: Project GUID the event belongs to.
        event: The raw event dict (``{"resource": "...", "type": "...", ...}``).

    Returns:
        The frame's ``data`` payload: ``{"project_id": <id>, "event": <dict>}``.
    """
    return {"project_id": project_id, "event": event}


def read_project_event_frame(
    data: dict[str, Any] | Any, expected_project_id: str
) -> dict[str, Any] | None:
    """Decode a project-event frame and validate tenancy.

    Returns the inner event dict when the frame is well-formed and its ``project_id``
    matches the expected one. Returns None (with a warning) on any rejection: non-dict
    payload, missing ``event`` key, missing ``project_id``, or a mismatched project_id.

    Fail closed: an absent or differing ``project_id`` is a rejection, never a fallback
    to the expected id.

    Args:
        data: The envelope's ``data`` field (expected shape: ``{"project_id": ..., "event": ...}``).
        expected_project_id: The project id the relay is running for.

    Returns:
        The inner event dict on success, None on rejection.
    """
    if not isinstance(data, dict):
        logger.warning(
            "[project-event-frame] Rejecting non-dict payload (type=%s)",
            type(data).__name__,
        )
        return None

    event = data.get("event")
    if event is None:
        logger.warning("[project-event-frame] Rejecting frame with missing 'event' key")
        return None

    frame_project_id = data.get("project_id")
    if frame_project_id is None:
        logger.warning("[project-event-frame] Rejecting frame with missing 'project_id'")
        return None

    if frame_project_id != expected_project_id:
        logger.warning(
            "[project-event-frame] Rejecting frame naming project %s; expected %s",
            frame_project_id[:8] if isinstance(frame_project_id, str) else frame_project_id,
            expected_project_id[:8],
        )
        return None

    return event


def forward_project_event_to_coordinator(
    project_id: str, event: dict[str, Any]
) -> bool:
    """Hand a project event from the AgentCore microVM to the coordinator for publishing.

    Appends the event into the live turn's buffer as a ``project_event`` envelope so
    ``agentcore_runtime._sse_frames`` emits it across the SSE boundary. The coordinator's
    drain in ``agentcore_relay_client`` re-targets it onto ``project:<project_id>`` where
    the browser is listening.

    Returns True when the event was successfully routed (a live running turn exists),
    False when unroutable (no turn registered, or the turn already went terminal, or no
    running event loop).

    The seq this envelope carries is inert in the AgentCore process: nothing in the
    AgentCore runtime publishes turn: seqs anywhere, and the coordinator re-mints the
    project: envelope from its own source. This is the same as every envelope
    ``agent_runner.emit`` builds inside the microVM — all are re-minted on the far side.

    **Ordering bound:** A frame appended after the turn's terminal ``result`` envelope
    is not drained, because the drain stops at ``result``. The window is the few
    statements between the result emit and the turn going terminal; the running-status
    check below closes most of it.

    Args:
        project_id: Project the event belongs to.
        event: The raw event dict (``{"resource": "...", "type": "...", ...}``).

    Returns:
        True if the event was appended to the turn buffer, False if unroutable.
    """
    # Lazy import to avoid any import-cycle risk with agent_turn
    from app.services.agent_turn import get_agent_turn_tracker
    from app.services.ws_envelope import make_envelope

    tracker = get_agent_turn_tracker()
    turn = tracker.get_active_turn(project_id, origin=None)

    if turn is None:
        # No live turn for this project, or the turn is not in running state.
        # The caller (jobs._publish_project_event) logs the unroutable case.
        return False

    # Build the frame payload
    data = build_project_event_frame(project_id, event)

    # Construct the envelope on the turn's stream. The seq is inert in this process.
    envelope = make_envelope(
        stream=f"turn:{turn.turn_id}", kind=PROJECT_EVENT_FRAME_KIND, data=data
    )

    # Schedule append on the running loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop. Distinct from the turn-absent case above: a turn WAS
        # resolved, so this branch is unroutable for a different reason. Logged
        # here rather than left to the caller, which cannot tell the two apart.
        logger.warning(
            "[project-event-frame] Unroutable (no running event loop): "
            "turn=%s resource=%s type=%s",
            turn.turn_id[:8],
            event.get("resource"),
            event.get("type"),
        )
        return False

    # Fire-and-forget task, retained so it isn't GC'd mid-flight
    async def _append():
        try:
            await turn.append_event(envelope)
        except Exception as exc:
            # Log but don't propagate — forwarding failures must not break the
            # local publish that happens regardless
            logger.warning(
                "[project-event-frame] Failed to append event to turn %s: %s",
                turn.turn_id[:8],
                exc,
            )

    task = loop.create_task(_append())
    _forwarding_tasks.add(task)
    task.add_done_callback(_forwarding_tasks.discard)

    # Success is not logged. Quiet success here is expected, and it is not evidence
    # of a wrong-destination mode (frames appending to a turn buffer nobody
    # drains): a success line at DEBUG is invisible under the app logger's INFO
    # level, and the unroutable case is logged by the caller. Add a per-call trace
    # only against a concrete symptom; do not treat quiet success as suspicious
    # on its own.
    return True


__all__ = [
    "PROJECT_EVENT_FRAME_KIND",
    "build_project_event_frame",
    "read_project_event_frame",
    "forward_project_event_to_coordinator",
]
