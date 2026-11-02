# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared types for the platform abstraction layer.

Concrete enough to typecheck routes and stores; loose enough to avoid
coupling to any single backend (filesystem, S3, DynamoDB).
"""
from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Literal

logger = logging.getLogger(__name__)

# Strings rather than NewTypes to keep the integration surface minimal —
# the codebase already passes plain strings around for these IDs.
UserId = str
ProjectId = str
TurnId = str
JobId = str

# Action used by ``Auth.require_access``. Only a project's owner is granted
# either one.
AccessAction = Literal["read", "write"]


@dataclass(frozen=True)
class RequestContextValue:
    """Per-request context propagated via ContextVar.

    Read by stores, MCP tools, and the structured logger so every log line and
    S3/DynamoDB call carries the requester's identity without threading it
    through every signature.

    **Set in exactly four places — there is no job-runner setter.** An earlier
    version of this docstring claimed job execution populated it too; that was
    never true, and believing it hides the fact that any server-side code
    spawning its own SDK session starts with NO context (whereupon `mcp_authz`
    fails closed and refuses every MCP tool). The four:

    1. ``platform/auth.py`` middleware — every HTTP request.
    2. ``agentcore_runtime.py`` — the relay serving path.
    3. ``services/agent_runner.py`` — a detached turn, attributed to the owner.
    4. :func:`attribute_detached_context` — detached SDK sessions (report
       narrator, prompt optimizer).

    A task created with ``asyncio.create_task`` inherits a COPY of the context
    as it stands at creation, so whether background work has one depends on its
    task ancestry. Do not assume it is present; use
    :func:`attribute_detached_context`.

    Attributes:
        user_id: Cognito ``sub`` (cloud) or ``LOCAL_DEV_USER_ID`` (dev).
        project_id: Project GUID for routes scoped under ``/projects/{id}/...``.
        request_id: UUID generated per request for log correlation.
        turn_id: Active turn for chat-scoped operations. ``None`` outside chat.
        job_id: Active job for job-execution-scoped operations (eval /
            optimization). ``None`` outside a job.
    """

    user_id: UserId
    request_id: str
    project_id: ProjectId | None = None
    turn_id: TurnId | None = None
    job_id: JobId | None = None


# Process-wide ContextVar. Default is None so an unauthenticated code path
# (e.g., during pytest collection) returns None rather than raising at
# import time. Routes that require a user must call ``current_user_id()``,
# which raises if the var is unset.
request_context: ContextVar[RequestContextValue | None] = ContextVar(
    "request_context", default=None
)


def current_request() -> RequestContextValue:
    """Return the current request context or raise.

    Raises:
        RuntimeError: No request context is active. This means an auth-
            requiring code path was reached without the auth middleware
            running first — almost always a setup bug.
    """
    ctx = request_context.get()
    if ctx is None:
        raise RuntimeError(
            "No request context — auth middleware did not run for this code path."
        )
    return ctx


def current_user_id() -> UserId:
    """Convenience accessor for ``current_request().user_id``."""
    return current_request().user_id


def attribute_detached_context(
    *,
    owner_id: UserId | None,
    project_id: ProjectId | None,
    request_id: str,
) -> object | None:
    """Attribute server-side work with no ambient context to a project owner.

    Server-side code that spawns its own Agent-SDK session (the report
    narrator, the prompt optimizer) runs outside any HTTP request and outside
    ``run_agent_turn``, so no context exists — and ``mcp_authz`` fails CLOSED,
    returning ``mcp_error("not authenticated")`` without invoking the handler.
    Those sessions therefore cannot use any MCP tool until a context is set.

    The semantics are "acting on the owner's behalf for a background action",
    never "the owner made this request".

    Attribution happens only when a context is ABSENT: a caller that already
    has one (an HTTP request, or a turn that attributed itself) keeps the real
    identity rather than being re-attributed to the project owner.

    An existing context is still BOUND to ``project_id`` when it does not
    already carry it, preserving its ``user_id``. Presence is not completeness,
    and the difference is load-bearing: the HTTP middleware sets ``user_id``
    and ``request_id`` before routing, with ``project_id`` left ``None``
    because no project is known that early. Since ``mcp_authz`` pins every
    project-scoped tool call to ``ctx.project_id``, treating that context as
    good enough denies every tool call the background work needs. These
    sessions are reachable straight from HTTP routes — regenerating a report
    narrative, and starting an optimization loop — so this is a live path, not
    a defensive branch.

    **Call this BEFORE constructing the SDK client.** The client's transport
    tasks inherit the ContextVar as it stands when they are created, so a set
    made afterwards is invisible to the in-process MCP tool handlers.

    Args:
        owner_id: Project owner to attribute to. ``None`` disables attribution
            rather than inventing an identity. Irrelevant to binding, which
            never changes who the caller is.
        project_id: Project the background work belongs to. ``None`` cannot
            bind anything, so an existing context is left alone.
        request_id: Correlation id for the logs, e.g. ``detached-report-ab12cd34``.

    Returns:
        The ContextVar token when a context was set or bound, else ``None``.
        Callers MUST reset a non-``None`` token in a ``finally`` block —
        relying on the caller always being a dedicated ``create_task`` would
        leak the identity into a long-lived worker task for every later
        operation.
    """
    ctx = request_context.get()
    if ctx is None:
        if not owner_id:
            return None
        logger.info(
            "[detached-context] %s attributed to project owner for MCP authz "
            "(owner=%s project=%s)",
            request_id, owner_id[:8], str(project_id)[:12],
        )
        return request_context.set(RequestContextValue(
            user_id=owner_id,
            request_id=request_id,
            project_id=project_id,
        ))
    if project_id and ctx.project_id != project_id:
        logger.info(
            "[detached-context] %s bound inherited context to its project for "
            "MCP authz (user=%s project=%s was=%s)",
            request_id, ctx.user_id[:8], project_id[:12],
            str(ctx.project_id)[:12],
        )
        return request_context.set(replace(ctx, project_id=project_id))
    return None


__all__ = [
    "AccessAction",
    "JobId",
    "ProjectId",
    "RequestContextValue",
    "TurnId",
    "UserId",
    "attribute_detached_context",
    "current_request",
    "current_user_id",
    "request_context",
]
