# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backstop for unhandled exceptions in long-lived background tasks.

Every background asyncio.Task spawned by the app already has its own
try/except inside the coroutine body — if execution reaches the top
of the coroutine, errors are caught and routed to the owning resource
(optimization state, eval job, agent turn) via ``ResourceError``.

This module handles the leftover case: an exception that escapes the
coroutine's own handlers, or a failure that happens before the
coroutine's try/except is entered. Today those would die silently
with only Python's "Task exception was never retrieved" warning.

Usage::

    task = asyncio.create_task(run_optimization_loop(cfg), name="...")
    attach_error_sink(task, project_id=cfg["project_id"], resource="optimization")

After the task finishes the callback inspects its result. If an
unexpected exception is present, a :class:`ResourceError` is fanned
out to the project owner's WebSocket connections with
``code="system.task_crashed"``. Cancellation is explicitly treated as
normal exit, not a crash.
"""

from __future__ import annotations

import asyncio
import logging

from app.services.resource_error import Codes, ResourceError
from app.services.ws_hub import get_ws_hub

logger = logging.getLogger(__name__)


def _resolve_owner_id(project_id: str) -> str | None:
    """Return ``owner_id`` for ``project_id`` or ``None`` if unavailable.

    Local copy of the agent_runner helper so this module doesn't import
    from agent_runner (avoids a circular import — agent_runner imports
    ``attach_error_sink``).
    """
    try:
        from app.main import get_project_manager
        project = get_project_manager().get_project(project_id)
        return project.owner_id if project else None
    except Exception:
        return None


def attach_error_sink(
    task: asyncio.Task,
    *,
    project_id: str,
    resource: str,
) -> asyncio.Task:
    """Attach a done-callback that publishes crashes via the WS hub.

    Args:
        task: Task to observe. Returned unchanged so call sites can wrap
            ``asyncio.create_task(...)`` inline without an extra variable.
        project_id: Project GUID for routing the crash event.
        resource: Short label (``"optimization"``, ``"eval_job"``,
            ``"agent_turn"``) attached to the event so the frontend can
            branch on which resource crashed.

    Returns:
        The same task, for chaining.
    """
    def _on_done(completed: asyncio.Task) -> None:
        if completed.cancelled():
            return
        exc = completed.exception()
        if exc is None:
            return
        # The task raised and nothing else retrieved the exception.
        # Emit the full traceback to logs (so operators can triage) and
        # fan out a ResourceError to the project owner's WS connections
        # (so the user knows their work stopped and which resource to
        # look at).
        logger.error(
            "[task-sink] Background task '%s' crashed: %s",
            completed.get_name(), exc, exc_info=exc,
        )
        crash = ResourceError(
            code=Codes.SYSTEM_TASK_CRASHED,
            message=f"Background task crashed: {exc}",
            hint="See backend log for the traceback. Retry the action if it's safe to do so.",
            details={
                "task_name": completed.get_name(),
                "exception": type(exc).__name__,
            },
        )
        owner_id = _resolve_owner_id(project_id)
        if owner_id is None:
            return
        try:
            # Done-callbacks run on the event loop thread (asyncio
            # invariant) so we can safely create_task here. The publish
            # is fire-and-forget — if the loop is mid-shutdown the task
            # may not run, which is acceptable for a crash notification.
            loop = asyncio.get_event_loop()
            loop.create_task(get_ws_hub().publish_project(
                project_id, owner_id, {
                    "resource": resource,
                    "type": "error",
                    **crash.to_dict(),
                },
            ))
        except Exception as publish_exc:
            # Publish failing is worth logging but not recoverable —
            # we're already in an error-handling path.
            logger.error(
                "[task-sink] Failed to publish crash event: %s",
                publish_exc, exc_info=True,
            )

    task.add_done_callback(_on_done)
    return task
