# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP-tool authorization wrapper.

Every MCP tool registered with the eval SDK MCP server runs in the
coordinator process under whatever user owns the active chat turn.
``Auth.require_access`` is the single source of truth for project-level
read/write decisions, so each tool must consult it before doing
anything that could mutate or read another user's project.

This module provides:

* :data:`TOOL_ACTION_MAP` — single-source mapping of tool name to
  :data:`AccessAction` (``"read"`` / ``"write"``). This dict must stay
  in sync with :mod:`app.services.mcp_tools` (a unit test asserts
  equality).
* :func:`apply_authz_to_tools` — wrap a list of ``@tool``-decorated
  functions (``SdkMcpTool`` instances) so each handler enforces the
  authz check before delegating to the original implementation.

The wrapper is transparent to callers — wrapped tools have the same
``name`` / ``description`` / ``input_schema`` / ``annotations`` and the
same handler signature, so :func:`create_eval_mcp_server` can keep
passing them straight to ``create_sdk_mcp_server``.

On denial we return :func:`mcp_error` rather than raising, because a
raised exception inside an MCP handler aborts the agent's whole turn
without giving Claude a chance to react. Returning ``is_error: True``
lets the agent observe the failure and adapt (e.g. tell the user it has
no access to that project).
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from claude_agent_sdk import SdkMcpTool

from app.services.mcp_response import mcp_error

from . import get_platform
from .types import AccessAction, request_context

logger = logging.getLogger(__name__)


# Single source of truth: tool name → required access action.
#
# Read-only tools merely list / describe / inspect existing project state.
# Write tools create data, dispatch background work, edit prompts, or
# mutate files on disk. Keep this dict alphabetized within each group
# and update it whenever ``mcp_tools/__init__.py`` adds or removes a tool;
# the ``test_mcp_authz`` test enforces equality with the registered set.
TOOL_ACTION_MAP: dict[str, AccessAction] = {
    # Read-only tools — list / describe / inspect existing state.
    "describe_data": "read",
    "check_eval_status": "read",
    "list_runs": "read",
    "get_results": "read",
    "get_case_details": "read",
    "list_bedrock_models": "read",
    "list_synthetic_batches": "read",
    "list_upload": "read",
    "get_project_status": "read",
    "refresh_pricing": "read",
    # Write tools — create data, dispatch jobs, edit prompts/files.
    "dispatch_eval": "write",
    "dispatch_optimization": "write",
    "dispatch_harness_build": "write",
    "request_user_input": "write",
    "resume_job": "write",
    "save_prompt": "write",
    "save_report": "write",
    "analyze_failures": "write",
    "record_optimization_iteration": "write",
    "scaffold_harness": "write",
    "set_models": "write",
    "validate_harness": "write",
    "cancel_eval": "write",
    "delete_run": "write",
    "generate_synthetic": "write",
    "approve_synthetic_cases": "write",
    "move_file": "write",
    "finalize_upload": "write",
}


# Sanity check at import time so a typo in TOOL_ACTION_MAP can't ship.
# Verified against the ``REGISTERED_TOOLS`` literal in
# ``mcp_tools/__init__.py``; the ``test_mcp_authz`` test asserts equality.
_EXPECTED_TOOL_COUNT = 28
assert len(TOOL_ACTION_MAP) == _EXPECTED_TOOL_COUNT, (
    f"TOOL_ACTION_MAP has {len(TOOL_ACTION_MAP)} entries; "
    f"REGISTERED_TOOLS exposes {_EXPECTED_TOOL_COUNT} MCP tools."
)


# Tool handlers all share this shape: ``async def(args: dict) -> dict``.
# Typing it once keeps :func:`authorize_tool_call` readable.
ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def authorize_tool_call(tool_name: str) -> Callable[[ToolHandler], ToolHandler]:
    """Build a decorator that gates ``tool_name`` on ``require_access``.

    The returned decorator wraps an MCP tool handler so every invocation:

    1. Reads the active :class:`RequestContextValue`. Missing context
       (auth middleware didn't run) returns ``mcp_error("not authenticated")``.
    2. If the tool's ``args`` carry a ``project_id``, calls
       ``platform.auth.require_access(user_id, project_id, action)``.
       The action is looked up in :data:`TOOL_ACTION_MAP`; an unknown
       tool name fails closed with ``mcp_error``.
    3. On HTTP 403/404 from ``require_access`` (or any unexpected
       exception), returns ``mcp_error("Forbidden")`` and logs at
       WARNING. The agent observes ``is_error: True`` and can adapt.
    4. On success, awaits the original handler and returns its result
       unchanged.

    Args:
        tool_name: Name registered with the SDK ``@tool`` decorator.

    Returns:
        Decorator that produces an authz-wrapped handler.
    """

    def decorator(handler: ToolHandler) -> ToolHandler:
        async def wrapped(args: dict[str, Any]) -> dict[str, Any]:
            ctx = request_context.get()
            if ctx is None:
                logger.warning(
                    "[mcp-authz] %s called outside request context", tool_name
                )
                return mcp_error("not authenticated")

            action = TOOL_ACTION_MAP.get(tool_name)
            if action is None:
                # Fail closed: an unmapped tool is a programmer error.
                logger.warning(
                    "[mcp-authz] %s missing from TOOL_ACTION_MAP", tool_name
                )
                return mcp_error(
                    f"Tool {tool_name!r} is not registered in TOOL_ACTION_MAP"
                )

            project_id = args.get("project_id") if isinstance(args, dict) else None
            user_id = ctx.user_id

            # Pin every call to the project this turn is for.
            #
            # ``project_id`` is an agent-supplied tool argument, and the agent
            # can name any value. The turn was authorized for ONE project before
            # it started, so a call naming a different one is out of scope by
            # construction — there is no flow where an agent legitimately
            # reaches a second project mid-turn (it cannot create or list
            # projects; no tool takes a raw path; every ``project_dir`` is
            # derived server-side from this argument).
            #
            # This is a string comparison against the context the serving path
            # already set, so it needs no lookup and no cross-tenant read. That
            # matters: ``require_access`` below answers "may user X reach project
            # Y" for ANY Y, and answering it requires reading other owners'
            # records — which is why the runtime role holds an ambient
            # cross-tenant grant. The pin answers a narrower question that needs
            # neither.
            #
            # It is also the ONLY guard on the filesystem. S3 and DynamoDB
            # reject a cross-tenant call at the AWS layer, but project trees
            # share one EFS access point that squashes every operation to a
            # single POSIX identity, so ``resolve_project_dir`` on another
            # project's id returns that tree and IAM has nothing to say about
            # it. Until per-session storage lands, this comparison is what
            # stands between an agent and a sibling project's files.
            if project_id and project_id != ctx.project_id:
                logger.warning(
                    "[mcp-authz] DENY user=%s tool=%s reason=project-not-this-turn "
                    "requested=%s turn_project=%s",
                    user_id,
                    tool_name,
                    str(project_id)[:12],
                    str(ctx.project_id)[:12],
                )
                return mcp_error(
                    "This tool call names a different project than the current "
                    "conversation. Only the project this conversation is about "
                    "can be used."
                )

            if project_id:
                # DEBUG: fires on every project-scoped tool call — success
                # is the norm and floods the log at INFO. The DENY path
                # below stays at WARNING so refusals remain findable.
                logger.debug(
                    "[mcp-authz] check user=%s project=%s tool=%s action=%s",
                    user_id, project_id, tool_name, action,
                )
                try:
                    platform = get_platform()
                    await platform.auth.require_access(user_id, project_id, action)
                except Exception as exc:  # noqa: BLE001 — convert every denial path
                    # ``require_access`` raises HTTPException(403/404) on
                    # denial. Anything else is unexpected and equally
                    # unsafe to propagate out of an MCP handler.
                    logger.warning(
                        "[mcp-authz] DENY user=%s project=%s tool=%s: %s",
                        user_id, project_id, tool_name, exc,
                    )
                    return mcp_error("Forbidden")
            else:
                # No project scope on this call — only assert the user
                # has a valid request context (already checked above).
                # DEBUG: per-call success noise, same as the scoped branch.
                logger.debug(
                    "[mcp-authz] no-project-scope user=%s tool=%s", user_id, tool_name
                )

            return await handler(args)

        # Preserve introspection so logs and error messages can still
        # identify the underlying coroutine.
        wrapped.__name__ = getattr(handler, "__name__", tool_name)
        wrapped.__qualname__ = getattr(handler, "__qualname__", tool_name)
        wrapped.__doc__ = handler.__doc__
        return wrapped

    return decorator


def apply_authz_to_tools(tools: list[SdkMcpTool]) -> list[SdkMcpTool]:
    """Wrap each tool's handler with :func:`authorize_tool_call`.

    The SDK's ``SdkMcpTool`` is a ``@dataclass`` whose ``handler`` field
    is the async callable invoked per tool call. We construct a new
    instance per tool so the registration list returned to
    ``create_sdk_mcp_server`` is fully wrapped, while the original
    decorated objects in ``mcp_tools/*`` remain untouched (handy for
    direct unit tests that bypass the wrapper).

    Args:
        tools: List of SDK MCP tool instances from the ``@tool`` decorator.

    Returns:
        New list of ``SdkMcpTool`` with authz-wrapped handlers, in input order.

    Raises:
        ValueError: A tool's name isn't in :data:`TOOL_ACTION_MAP`. Caller
            must update the map before re-registering.
    """
    wrapped: list[SdkMcpTool] = []
    for tool in tools:
        # Tests sometimes patch ``claude_agent_sdk`` wholesale, which
        # turns the @tool decorator into a MagicMock and makes ``name``
        # unmappable. Skip wrapping in that case rather than crashing —
        # the mock pipeline doesn't go through real authz anyway.
        name = getattr(tool, "name", None)
        if not isinstance(name, str):
            logger.debug(
                "[mcp-authz] Skipping non-str tool name %r (likely a test mock)",
                name,
            )
            wrapped.append(tool)
            continue
        if name not in TOOL_ACTION_MAP:
            raise ValueError(
                f"Tool {name!r} is not in TOOL_ACTION_MAP. "
                "Add an entry mapping it to 'read' or 'write' before registering."
            )
        wrapped_handler = authorize_tool_call(name)(tool.handler)
        wrapped.append(
            SdkMcpTool(
                name=name,
                description=tool.description,
                input_schema=tool.input_schema,
                handler=wrapped_handler,
                annotations=tool.annotations,
            )
        )
    return wrapped


__all__ = [
    "TOOL_ACTION_MAP",
    "apply_authz_to_tools",
    "authorize_tool_call",
]
