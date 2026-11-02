# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detached orchestrator turns must carry the project owner's identity.

``authorize_tool_call`` gates every MCP tool on the ``request_context``
contextvar. Turns rooted in an HTTP request inherit it through the async
task chain; turns spawned from DETACHED background chains (completion
turns fired by the job-progress republisher / reconciler / startup-adopted
loops) carried None — so every project-scoped MCP tool call inside them
returned "not authenticated" (measured live 3x: post-baseline reporter
turns delivered partial numbers because they couldn't read results).

``run_agent_turn`` now attributes a detached turn to the project's owner.
These tests pin the attribution logic and its safety properties:
1. An inherited context keeps its IDENTITY — a live user's turn is never
   reattributed to the owner, which would launder it as a background action.
2. An inherited context is still BOUND to the turn's project. The HTTP
   middleware sets user_id and request_id before routing and leaves
   project_id None, and ``mcp_authz`` pins every project-scoped tool call to
   it, so a context that is present but unbound denies all of them.
3. Authorization still runs — the owner identity passes ``require_access``
   for the owner's project and cross-project denial is untouched.
"""
from __future__ import annotations

import asyncio
import contextvars
from unittest.mock import MagicMock, patch

from app.platform.types import RequestContextValue, request_context


def _run_turn_prologue(project_id: str, owner_id: str | None) -> RequestContextValue | None:
    """Execute run_agent_turn far enough to observe the context attribution.

    The full turn spins up the SDK; we stop it right after the attribution
    block by making the next dependency raise, then read the contextvar.
    Runs inside a COPIED context so the test never leaks state.
    """
    from app.services import agent_runner

    result: dict[str, RequestContextValue | None] = {}

    class _Stop(Exception):
        pass

    def _boom(*a, **k):
        # Capture the contextvar as seen INSIDE the turn, then halt.
        result["ctx"] = request_context.get()
        raise _Stop()

    async def _drive():
        turn = MagicMock()
        turn.turn_id = "aaaabbbb-cccc-dddd-eeee-ffff00001111"
        with patch.object(agent_runner, "_resolve_owner_id", return_value=owner_id), \
             patch("app.services.project_paths.resolve_project_dir", return_value="/tmp/p"), \
             patch.object(agent_runner, "get_ws_hub", side_effect=_boom):
            try:
                await agent_runner.run_agent_turn(
                    turn, "hi", project_id, "system prompt",
                )
            except _Stop:
                pass
            except Exception:
                # Any later failure still means attribution already ran.
                if "ctx" not in result:
                    result["ctx"] = request_context.get()

    ctx = contextvars.copy_context()
    ctx.run(asyncio.run, _drive())
    return result.get("ctx")


def test_detached_turn_gets_owner_identity() -> None:
    got = _run_turn_prologue("proj-123", owner_id="owner-abc")
    assert got is not None, (
        "a detached turn (no inherited request context) must be attributed "
        "to the project owner so MCP authz can evaluate it"
    )
    assert got.user_id == "owner-abc"
    assert got.project_id == "proj-123"


def test_inherited_context_already_bound_is_left_untouched() -> None:
    inherited = RequestContextValue(
        user_id="real-user", request_id="req-1", project_id="proj-123",
    )

    def _with_inherited():
        request_context.set(inherited)
        return _run_turn_prologue("proj-123", owner_id="owner-abc")

    got = contextvars.copy_context().run(_with_inherited)
    assert got is inherited, (
        "an inherited context that already names this turn's project needs no "
        "rebind, and must not be replaced by owner attribution"
    )


def test_inherited_context_without_a_project_is_bound_to_this_turn() -> None:
    """The shape the HTTP middleware actually produces.

    ``auth_middleware`` runs before routing, so it sets user_id and request_id
    with project_id left None. Guarding on presence alone leaves that context
    unbound, and ``mcp_authz`` then denies every project-scoped tool call on
    every interactive chat turn — file listing, data description, project
    status. Binding is what makes those calls reachable.
    """
    inherited = RequestContextValue(user_id="real-user", request_id="req-1")

    def _with_inherited():
        request_context.set(inherited)
        return _run_turn_prologue("proj-123", owner_id="owner-abc")

    got = contextvars.copy_context().run(_with_inherited)
    assert got is not None
    assert got.project_id == "proj-123", (
        "an inherited context carrying no project must be bound to the "
        "project this turn is for, or mcp_authz denies every tool call in it"
    )
    assert got.user_id == "real-user", (
        "binding the project must NOT reattribute the turn to the owner — "
        "that would launder a live user's turn as a background action"
    )
    assert got.request_id == "req-1", (
        "the request id is the log-correlation key for the originating HTTP "
        "request and must survive the rebind"
    )


def test_inherited_context_naming_another_project_is_rebound() -> None:
    """A stale project binding is corrected, not trusted.

    A context inherited from a task chain that was serving a different project
    would otherwise pin this turn's tool calls to the wrong project.
    """
    inherited = RequestContextValue(
        user_id="real-user", request_id="req-1", project_id="proj-OTHER",
    )

    def _with_inherited():
        request_context.set(inherited)
        return _run_turn_prologue("proj-123", owner_id="owner-abc")

    got = contextvars.copy_context().run(_with_inherited)
    assert got is not None
    assert got.project_id == "proj-123"
    assert got.user_id == "real-user"


def test_unresolvable_owner_leaves_context_unset() -> None:
    got = _run_turn_prologue("proj-123", owner_id=None)
    assert got is None, (
        "no owner -> no synthetic identity; authz keeps failing closed"
    )


def test_owner_identity_still_passes_through_authorize_tool_call() -> None:
    """The attributed identity flows through the REAL authz decorator."""
    from unittest.mock import AsyncMock

    from app.platform.mcp_authz import authorize_tool_call

    calls: list[tuple[str, str, str]] = []

    class _Auth:
        async def require_access(self, user_id, project_id, action):
            calls.append((user_id, project_id, action))
            # Owner accessing own project: allowed (no raise).

    platform = MagicMock()
    platform.auth = _Auth()

    @authorize_tool_call("get_results")
    async def handler(args):
        return {"content": [{"type": "text", "text": "ok"}]}

    async def _drive():
        request_context.set(RequestContextValue(
            user_id="owner-abc", request_id="detached-turn-aaaabbbb",
            project_id="proj-123",
        ))
        with patch("app.platform.mcp_authz.get_platform", return_value=platform):
            return await handler({"project_id": "proj-123"})

    out = contextvars.copy_context().run(asyncio.run, _drive())
    assert out.get("is_error") is not True
    assert calls == [("owner-abc", "proj-123", "read")], (
        "require_access must still run with the attributed owner identity"
    )
