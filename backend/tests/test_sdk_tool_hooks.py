# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading an MCP tool's outcome out of an SDK hook payload.

The regression these pin: the optimizer logged ``[save_prompt] SAVED v1.md``,
the loop advanced, and the hook still reported no success. The old check
was::

    if isinstance(response, dict) and not response.get("is_error"):
        succeeded = True

A plain dict without an error flag would have satisfied that, so the observed
failure proves the payload is not that dict. The same check was copied into the
report narrator, where the flag is the RETURN VALUE — so a working
``save_report`` was reported as a failure there rather than merely mislogged.

So these tests deliberately do NOT assert one payload shape. They assert that
every plausible shape resolves correctly, because the shape is the thing that
was wrong and pinning today's guess would just re-freeze the bug.
"""

from __future__ import annotations

import pytest

from app.services.sdk_tool_hooks import (
    TOOL_OUTCOME_HOOK_EVENTS,
    ToolCallOutcome,
    make_tool_outcome_hook,
)


@pytest.fixture
def outcome() -> ToolCallOutcome:
    return ToolCallOutcome()


@pytest.fixture
def hook(outcome):
    return make_tool_outcome_hook(outcome)


class TestSuccessIsReadFromTheEventNotThePayloadShape:
    """Every non-error ``PostToolUse`` payload means the call worked."""

    @pytest.mark.asyncio
    async def test_content_block_list_counts_as_success(self, hook, outcome):
        """The shape the old check silently rejected.

        An MCP tool result reaching the hook as a bare content-block list is
        not a dict, so ``isinstance(response, dict)`` was False and success was
        never recorded — with no error anywhere to explain it.
        """
        await hook(
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "mcp__eval__save_prompt",
                "tool_response": [{"type": "text", "text": "SAVED v1.md"}],
            },
            "tu_1",
            {},
        )
        assert outcome.succeeded is True
        assert outcome.attempted is True

    @pytest.mark.asyncio
    async def test_dict_without_error_flag_counts_as_success(self, hook, outcome):
        await hook(
            {
                "hook_event_name": "PostToolUse",
                "tool_response": {"content": [{"type": "text", "text": "ok"}]},
            },
            "tu_1",
            {},
        )
        assert outcome.succeeded is True

    @pytest.mark.asyncio
    async def test_absent_payload_counts_as_success(self, hook, outcome):
        """``PostToolUse`` fired at all means the CLI did not class it a failure.

        A payload this hook cannot interpret must not be read as an error —
        that is the direction the original bug failed in.
        """
        await hook({"hook_event_name": "PostToolUse"}, "tu_1", {})
        assert outcome.succeeded is True

    @pytest.mark.asyncio
    async def test_string_payload_counts_as_success(self, hook, outcome):
        await hook(
            {"hook_event_name": "PostToolUse", "tool_response": "SAVED v1.md"},
            "tu_1",
            {},
        )
        assert outcome.succeeded is True


class TestErrorFlagsAreHonouredUnderEitherSpelling:
    """MCP's wire format is ``isError``; this codebase's helpers use
    ``is_error``. A payload may arrive either way, so both must block success."""

    @pytest.mark.asyncio
    async def test_snake_case_is_error_blocks_success(self, hook, outcome):
        await hook(
            {
                "hook_event_name": "PostToolUse",
                "tool_response": {"content": [], "is_error": True},
            },
            "tu_1",
            {},
        )
        assert outcome.succeeded is False
        assert outcome.attempted is True

    @pytest.mark.asyncio
    async def test_camel_case_isError_blocks_success(self, hook, outcome):
        await hook(
            {
                "hook_event_name": "PostToolUse",
                "tool_response": {"content": [], "isError": True},
            },
            "tu_1",
            {},
        )
        assert outcome.succeeded is False

    @pytest.mark.asyncio
    async def test_falsy_error_flag_is_not_an_error(self, hook, outcome):
        """``is_error: False`` is what a success looks like when the flag is
        present at all — it must not be read as merely 'a flag exists'."""
        await hook(
            {
                "hook_event_name": "PostToolUse",
                "tool_response": {"content": [], "is_error": False},
            },
            "tu_1",
            {},
        )
        assert outcome.succeeded is True


class TestFailureEvent:
    @pytest.mark.asyncio
    async def test_failure_event_records_the_error_and_not_success(self, hook, outcome):
        await hook(
            {
                "hook_event_name": "PostToolUseFailure",
                "tool_name": "mcp__eval__save_prompt",
                "error": "authz refused",
            },
            "tu_1",
            {},
        )
        assert outcome.attempted is True
        assert outcome.succeeded is False
        assert outcome.error == "authz refused"

    @pytest.mark.asyncio
    async def test_a_later_failure_does_not_erase_an_earlier_success(
        self, hook, outcome
    ):
        """The agent may retry. Once a save has succeeded the artifact exists,
        so a subsequent failed attempt must not retract the verdict — for the
        narrator that flag is the return value."""
        await hook(
            {"hook_event_name": "PostToolUse", "tool_response": ["ok"]}, "tu_1", {}
        )
        await hook(
            {"hook_event_name": "PostToolUseFailure", "error": "dedupe refusal"},
            "tu_2",
            {},
        )
        assert outcome.succeeded is True
        assert outcome.error == "dedupe refusal"

    @pytest.mark.asyncio
    async def test_a_failure_then_a_success_succeeds(self, hook, outcome):
        await hook(
            {"hook_event_name": "PostToolUseFailure", "error": "transient"}, "tu_1", {}
        )
        await hook(
            {"hook_event_name": "PostToolUse", "tool_response": ["ok"]}, "tu_2", {}
        )
        assert outcome.succeeded is True


class TestEventNameFallback:
    """``hook_event_name`` is the primary discriminator, but a payload that
    lacks it must still not turn a failure into a success."""

    @pytest.mark.asyncio
    async def test_an_error_key_without_an_event_name_is_a_failure(
        self, hook, outcome
    ):
        await hook({"error": "boom", "tool_name": "x"}, "tu_1", {})
        assert outcome.succeeded is False
        assert outcome.error == "boom"

    @pytest.mark.asyncio
    async def test_a_response_without_an_event_name_is_a_success(self, hook, outcome):
        await hook({"tool_response": ["ok"], "tool_name": "x"}, "tu_1", {})
        assert outcome.succeeded is True

    @pytest.mark.asyncio
    async def test_a_non_string_event_name_falls_back(self, hook, outcome):
        await hook({"hook_event_name": 7, "error": "boom"}, "tu_1", {})
        assert outcome.succeeded is False


class TestDiagnosticTrace:
    """The old warning could only say "the wiring is suspect". The point of
    recording observations is that the next shape change explains itself."""

    def test_never_called_says_so(self, outcome):
        assert outcome.attempted is False
        assert outcome.describe() == "no tool hook fired"

    @pytest.mark.asyncio
    async def test_describe_names_the_event_and_the_payload_shape(self, hook, outcome):
        await hook(
            {
                "hook_event_name": "PostToolUse",
                "tool_response": [{"type": "text", "text": "ok"}],
            },
            "tu_1",
            {},
        )
        described = outcome.describe()
        assert "PostToolUse" in described
        assert "list" in described

    @pytest.mark.asyncio
    async def test_describe_names_a_dicts_keys(self, hook, outcome):
        """Keys are what a shape-sensitive reader would have got wrong, so they
        are the useful thing to print."""
        await hook(
            {
                "hook_event_name": "PostToolUse",
                "tool_response": {"content": [], "isError": False},
            },
            "tu_1",
            {},
        )
        assert "content" in outcome.describe()

    @pytest.mark.asyncio
    async def test_every_call_is_recorded(self, hook, outcome):
        await hook({"hook_event_name": "PostToolUse", "tool_response": []}, "t1", {})
        await hook({"hook_event_name": "PostToolUseFailure", "error": "x"}, "t2", {})
        assert len(outcome.observations) == 2


class TestHooksStayObservers:
    @pytest.mark.asyncio
    async def test_returns_an_empty_decision(self, hook):
        """A non-empty return would let a diagnostic hook steer the agent."""
        assert await hook({"hook_event_name": "PostToolUse"}, "tu_1", {}) == {}

    @pytest.mark.asyncio
    async def test_a_non_dict_payload_does_not_raise(self, hook, outcome):
        """These run inside the SDK's control loop; raising here would take
        down the agent session over a diagnostic."""
        assert await hook("unexpected", "tu_1", {}) == {}
        assert outcome.attempted is True
        assert outcome.succeeded is False


class TestBothEventsAreRegisteredAtEveryCallSite:
    """Registering only ``PostToolUse`` makes "the tool failed" and "the tool
    was never called" indistinguishable — both leave ``succeeded`` False. The
    optimizer has a filesystem check to fall back on; the narrator does not.
    """

    def test_the_event_tuple_covers_success_and_failure(self):
        assert set(TOOL_OUTCOME_HOOK_EVENTS) == {"PostToolUse", "PostToolUseFailure"}

    @pytest.mark.parametrize(
        "module_name",
        ["optimization_loop_orchestrator", "report_generation"],
    )
    def test_call_sites_register_over_the_whole_event_tuple(self, module_name):
        """Pinned by source because constructing the options needs a live SDK
        session. A call site that hardcodes one event name instead of iterating
        the tuple is the regression this catches.
        """
        import importlib
        import inspect

        module = importlib.import_module(f"app.services.{module_name}")
        source = inspect.getsource(module)
        assert "TOOL_OUTCOME_HOOK_EVENTS" in source, (
            f"{module_name} does not iterate TOOL_OUTCOME_HOOK_EVENTS, so it "
            "registers a hand-picked set of hook events"
        )
        assert "make_tool_outcome_hook" in source
