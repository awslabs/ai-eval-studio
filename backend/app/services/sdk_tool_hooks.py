# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading an MCP tool call's outcome out of an SDK hook payload.

Two background agents need to know whether their one required tool call
worked: the prompt optimizer (``save_prompt``) and the report narrator
(``save_report``). Both used to inspect the hook payload directly::

    response = input_data.get("tool_response")
    if isinstance(response, dict) and not response.get("is_error"):
        succeeded = True

That never fires for an MCP tool: a ``save_prompt`` call that logged
``SAVED v1.md`` and let the loop advance still left ``succeeded`` False. Since a
plain dict without an error flag WOULD have set it, the payload demonstrably is
not the ``{"is_error": ...}`` dict the check assumed. That was cosmetic for the optimizer, which treats the
filesystem as the verdict and the hooks as diagnostics, but the narrator
RETURNS its flag — so a working ``save_report`` was reported as a failure.

The fix is to stop reading success out of the payload's shape at all. The CLI
already classifies the call for us by choosing which event to fire:
``PostToolUse`` after a call that completed, ``PostToolUseFailure`` after one
that did not (its payload carries ``error`` instead of ``tool_response``). So
register on BOTH and let the event be the answer. An error flag in the payload
is still honoured if one is present, under either spelling — MCP wire format
uses ``isError`` while this codebase's own helpers use ``is_error`` — but
nothing depends on finding one.

What each observation looked like is recorded too, so the next time a payload
changes shape the log says what arrived instead of "the wiring is suspect".
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

#: Register the recorder on both, or a failure is indistinguishable from a
#: tool that was never called: both leave ``succeeded`` False.
TOOL_OUTCOME_HOOK_EVENTS: tuple[str, ...] = ("PostToolUse", "PostToolUseFailure")


class ToolCallOutcome:
    """What the SDK's tool hooks observed for one tool, across a whole session.

    ``attempted`` and ``succeeded`` are deliberately separate: "the agent never
    called the tool" and "the agent called it and it failed" call for different
    operator responses, and collapsing them loses that.
    """

    def __init__(self) -> None:
        self.attempted = False
        self.succeeded = False
        #: Error text from the last failing call, when the payload carried any.
        self.error: str | None = None
        #: One entry per hook invocation, for the diagnostic log.
        self.observations: list[str] = []

    def describe(self) -> str:
        """Compact operator-facing summary of what the hooks actually saw.

        Returns:
            Comma-separated observations, or a phrase saying none arrived —
            never an empty string, since this lands mid-sentence in a log line.
        """
        if not self.observations:
            return "no tool hook fired"
        return ", ".join(self.observations)


def _payload_signals_error(response: Any) -> bool:
    """Whether a ``tool_response`` payload carries an explicit error flag.

    Absence is not evidence of success — most payload shapes have nowhere to
    put the flag. The caller only uses this to override an otherwise
    successful event.

    Args:
        response: The ``tool_response`` value, whatever shape it arrived in.

    Returns:
        True only when a flag is present AND truthy.
    """
    if isinstance(response, dict):
        return bool(response.get("is_error") or response.get("isError"))
    return False


def _describe_payload(response: Any) -> str:
    """Name the payload's shape for the diagnostic log.

    Args:
        response: The ``tool_response`` value.

    Returns:
        Short shape description — the type, plus a dict's keys, since the keys
        are what a shape-sensitive reader would have got wrong.
    """
    if isinstance(response, dict):
        return f"dict{sorted(response)}"
    if isinstance(response, list):
        return f"list[{len(response)}]"
    return type(response).__name__


def make_tool_outcome_hook(
    outcome: ToolCallOutcome,
) -> Callable[[Any, Any, Any], Awaitable[dict[str, Any]]]:
    """Build a hook callback that records tool outcomes into ``outcome``.

    Register the returned callable on every event in
    ``TOOL_OUTCOME_HOOK_EVENTS``. The SDK gives each registration its own
    callback id, so sharing one callable across events is fine.

    Args:
        outcome: The record to mutate. Shared across events and calls.

    Returns:
        An async hook callback returning an empty decision dict (the hooks are
        observers and must not influence the agent).
    """

    async def _hook(input_data: Any, tool_use_id: Any, context: Any) -> dict[str, Any]:
        if not isinstance(input_data, dict):
            outcome.attempted = True
            outcome.observations.append(f"non-dict input ({type(input_data).__name__})")
            return {}

        # Prefer the declared event name; fall back to the payload's own
        # discriminator so a missing field cannot silently turn a failure into
        # a success. PostToolUseFailure carries ``error``; PostToolUse carries
        # ``tool_response``.
        event = input_data.get("hook_event_name")
        if not isinstance(event, str):
            event = "PostToolUseFailure" if "error" in input_data else "PostToolUse"

        outcome.attempted = True

        if event == "PostToolUseFailure":
            error = input_data.get("error")
            outcome.error = error if isinstance(error, str) else None
            outcome.observations.append(f"{event}(error={outcome.error!r})")
            return {}

        response = input_data.get("tool_response")
        if _payload_signals_error(response):
            outcome.observations.append(f"{event}(error flag set)")
            return {}

        outcome.succeeded = True
        outcome.observations.append(f"{event}({_describe_payload(response)})")
        return {}

    return _hook
