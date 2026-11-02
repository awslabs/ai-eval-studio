# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MCP-tool authz wrapper.

The wrapper sits between the SDK's MCP transport and our tool handlers.
It must:

* Cover every tool registered in ``mcp_tools/__init__.py`` — verified
  against :func:`create_eval_mcp_server`.
* Pass through to the underlying handler when ``require_access`` is OK.
* Return ``mcp_error`` (``is_error: True``) on any denial path so the
  agent can adapt instead of dying mid-turn.
* Tolerate calls with no ``project_id`` in args (the few "global" tools
  like ``list_bedrock_models`` skip the project-scoped check).
* Refuse calls when the request context is unset (auth middleware
  didn't run).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.platform import Platform, set_platform
from app.platform.auth import Auth
from app.platform.dev_local import (
    FilesystemMetadataStore,
    NoOpTokenVending,
)
from app.platform.mcp_authz import (
    TOOL_ACTION_MAP,
    apply_authz_to_tools,
    authorize_tool_call,
)
from app.platform.types import (
    AccessAction,
    ProjectId,
    RequestContextValue,
    UserId,
    request_context,
)
from app.services.mcp_tools import REGISTERED_TOOLS


# --- Fixtures --------------------------------------------------------


class _StubAuth(Auth):
    """Hand-rolled :class:`Auth` so tests can pin allow/deny outcomes.

    LocalAuth needs a real metadata store + project record. For the
    wrapper's behavioral tests we only care that ``require_access``
    raises HTTPException(403) on denial and returns ``None`` on allow.
    """

    def __init__(self, *, allow_pairs: set[tuple[UserId, ProjectId, AccessAction]]):
        self._allow = allow_pairs

    async def extract_user_id(self, request):  # pragma: no cover — unused here
        raise NotImplementedError

    async def require_access(
        self,
        user_id: UserId,
        project_id: ProjectId,
        action: AccessAction,
    ) -> None:
        if (user_id, project_id, action) in self._allow:
            return
        raise HTTPException(status_code=403, detail="Forbidden")


@pytest.fixture
def stub_platform() -> Platform:
    """Build a Platform with :class:`_StubAuth` as the only meaningful piece."""
    return Platform(
        metadata_store=FilesystemMetadataStore(),
        session_store=None,
        token_vending=NoOpTokenVending(),
        auth=_StubAuth(allow_pairs=set()),
    )


@pytest.fixture
def request_ctx():
    """Set/tear down a per-test request context.

    ``project_id`` is populated because every production setter populates it —
    the HTTP middleware, the relay serving path, a detached turn, and
    ``attribute_detached_context``, whose two callers both index the value they
    pass. Omitting it here made the fixture the only place a project-scoped tool
    call ran with no project in context, which is the state the turn pin treats
    as out of scope.
    """
    token = request_context.set(
        RequestContextValue(user_id="alice", request_id="req-1", project_id="p1")
    )
    try:
        yield
    finally:
        request_context.reset(token)


@pytest.fixture(autouse=True)
def _reset_platform_singleton():
    """Each test installs its own platform; clear afterwards."""
    yield
    set_platform(None)


# --- TOOL_ACTION_MAP coverage ---------------------------------------


def test_tool_action_map_matches_registered_tools() -> None:
    """The action map must list exactly the tools registered with the SDK.

    ``REGISTERED_TOOLS`` is the single tool list passed into
    ``create_sdk_mcp_server``; comparing against it catches drift in
    either direction.
    """
    registered_names = {t.name for t in REGISTERED_TOOLS}
    map_names = set(TOOL_ACTION_MAP.keys())
    assert registered_names == map_names, (
        f"TOOL_ACTION_MAP drift.\n"
        f"  Only in map:        {sorted(map_names - registered_names)}\n"
        f"  Only in registered: {sorted(registered_names - map_names)}"
    )


# --- Behavioral tests -----------------------------------------------


@pytest.mark.asyncio
async def test_wrapped_tool_passes_through_for_authorized_user(
    stub_platform: Platform,
    request_ctx,
) -> None:
    """Owner with ``write`` on the project sees the tool's actual return."""
    stub_platform.auth._allow.add(("alice", "p1", "write"))  # type: ignore[attr-defined]
    set_platform(stub_platform)

    @authorize_tool_call("dispatch_eval")
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": "ran"}]}

    result = await handler({"project_id": "p1"})
    assert result == {"content": [{"type": "text", "text": "ran"}]}


@pytest.mark.asyncio
async def test_wrapped_tool_returns_mcp_error_on_denial(
    stub_platform: Platform,
    request_ctx,
) -> None:
    """A caller denied a write tool gets an ``is_error`` response."""
    # Empty allow set — every check denies.
    set_platform(stub_platform)

    @authorize_tool_call("dispatch_eval")
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        pytest.fail("handler must not run when access is denied")

    result = await handler({"project_id": "p1"})
    assert result.get("is_error") is True
    assert "Forbidden" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_wrapped_tool_without_project_id_skips_project_check(
    stub_platform: Platform,
    request_ctx,
) -> None:
    """``list_bedrock_models`` has no project scope — still works for owner."""
    set_platform(stub_platform)

    @authorize_tool_call("list_bedrock_models")
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": "models"}]}

    result = await handler({})  # no project_id
    assert result == {"content": [{"type": "text", "text": "models"}]}


@pytest.mark.asyncio
async def test_wrapped_tool_without_request_context_returns_error(
    stub_platform: Platform,
) -> None:
    """Calls outside an authenticated request fail closed."""
    set_platform(stub_platform)

    @authorize_tool_call("describe_data")
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        pytest.fail("handler must not run without a request context")

    # Note: no request_ctx fixture — context is unset.
    result = await handler({"project_id": "p1"})
    assert result.get("is_error") is True
    assert "not authenticated" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_apply_authz_preserves_metadata(
    stub_platform: Platform,
    request_ctx,
) -> None:
    """``apply_authz_to_tools`` keeps name/description/schema intact."""
    stub_platform.auth._allow.add(("alice", "p1", "read"))  # type: ignore[attr-defined]
    set_platform(stub_platform)

    from claude_agent_sdk import tool

    @tool("describe_data", "desc", {"project_id": str})
    async def describe_data(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": "ok"}]}

    [wrapped] = apply_authz_to_tools([describe_data])
    assert wrapped.name == "describe_data"
    assert wrapped.description == "desc"
    assert wrapped.input_schema == {"project_id": str}

    # The wrapped handler still routes through authz.
    out = await wrapped.handler({"project_id": "p1"})
    assert out == {"content": [{"type": "text", "text": "ok"}]}


def test_apply_authz_rejects_unknown_tool() -> None:
    """A tool whose name isn't in TOOL_ACTION_MAP fails registration loudly."""
    from claude_agent_sdk import tool

    @tool("never_registered_tool", "x", {})
    async def fake(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": []}

    with pytest.raises(ValueError, match="never_registered_tool"):
        apply_authz_to_tools([fake])


# --- Turn pin ---------------------------------------------------------
#
# ``project_id`` is an agent-supplied tool argument. The turn was authorized for
# ONE project before it started, so a call naming a different one is out of
# scope by construction, and the pin rejects it with a string comparison against
# the context the serving path set — no lookup, so no cross-tenant read.
#
# The pin is also the ONLY guard on the filesystem. S3 and DynamoDB refuse a
# cross-tenant call at the AWS layer, but project trees share one EFS access
# point that squashes every operation to a single POSIX identity, so
# ``resolve_project_dir`` on a sibling project's id returns that tree and IAM has
# nothing to say about it.


@pytest.mark.asyncio
async def test_a_tool_call_naming_another_project_is_refused(
    stub_platform: Platform,
    request_ctx,
) -> None:
    """The load-bearing case. Note the user is ALLOWED on the other project —
    shared with them, say — so ``require_access`` would permit this. Being
    permitted to reach a project is not the same as this turn being about it."""
    stub_platform.auth._allow.add(("alice", "p1", "write"))  # type: ignore[attr-defined]
    stub_platform.auth._allow.add(("alice", "p2", "write"))  # type: ignore[attr-defined]
    set_platform(stub_platform)

    ran = False

    @authorize_tool_call("dispatch_eval")
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        nonlocal ran
        ran = True
        return {"content": [{"type": "text", "text": "ran"}]}

    result = await handler({"project_id": "p2"})

    assert result.get("is_error") is True, (
        "a tool call for a project other than the turn's was allowed through"
    )
    assert ran is False, "the handler executed despite the project mismatch"


@pytest.mark.asyncio
async def test_the_turns_own_project_still_runs(
    stub_platform: Platform,
    request_ctx,
) -> None:
    """Negative control. Without this the pin could be denying everything and
    the test above would still pass."""
    stub_platform.auth._allow.add(("alice", "p1", "write"))  # type: ignore[attr-defined]
    set_platform(stub_platform)

    @authorize_tool_call("dispatch_eval")
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": "ran"}]}

    result = await handler({"project_id": "p1"})
    assert result == {"content": [{"type": "text", "text": "ran"}]}


@pytest.mark.asyncio
async def test_a_context_with_no_project_refuses_a_project_scoped_call(
    stub_platform: Platform,
) -> None:
    """Fail closed rather than falling through.

    No production setter leaves ``project_id`` unset, so reaching this state
    means an unrecognised execution path — which is exactly when a permissive
    default would be worst. Falling through to ``require_access`` here would let
    such a path name any project the user can reach.
    """
    stub_platform.auth._allow.add(("alice", "p1", "write"))  # type: ignore[attr-defined]
    set_platform(stub_platform)

    token = request_context.set(
        RequestContextValue(user_id="alice", request_id="req-noproj")
    )
    try:

        @authorize_tool_call("dispatch_eval")
        async def handler(args: dict[str, Any]) -> dict[str, Any]:
            return {"content": [{"type": "text", "text": "ran"}]}

        result = await handler({"project_id": "p1"})
    finally:
        request_context.reset(token)

    assert result.get("is_error") is True


@pytest.mark.asyncio
async def test_the_refusal_is_findable_in_the_log(
    stub_platform: Platform,
    request_ctx,
    monkeypatch,
) -> None:
    """A refused cross-project call should be diagnosable without a repro."""
    import app.platform.mcp_authz as authz_module

    stub_platform.auth._allow.add(("alice", "p2", "write"))  # type: ignore[attr-defined]
    set_platform(stub_platform)

    calls: list[tuple] = []
    fake_logger = MagicMock()
    fake_logger.warning.side_effect = lambda *a, **k: calls.append(a)
    monkeypatch.setattr(authz_module, "logger", fake_logger)

    @authorize_tool_call("dispatch_eval")
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": "ran"}]}

    await handler({"project_id": "p2"})

    joined = " ".join(str(c) for c in calls)
    assert "project-not-this-turn" in joined, (
        f"the refusal does not name its reason: {joined}"
    )
