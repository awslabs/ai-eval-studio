# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SDK subprocess env pins shared by all spawn sites.

The invariant: every SDK spawn site in this backend carries both pins
(CLAUDE_CODE_DISABLE_BACKGROUND_TASKS and DISABLE_GROWTHBOOK). The parity
tests build REAL options via the actual spawn functions rather than grepping
file text — a substring test already proved deletion-blind when a second call
site matched the pattern.
"""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

from app.services.sdk_subprocess_env import sdk_subprocess_env
from app.services.agent_runner import run_agent_turn
from app.services.optimization_loop_orchestrator import _run_prompt_optimizer
from app.services.report_generation import _run_narrator
from app.services.agent_turn import AgentTurn


def _mock_sdk(messages):
    """Helper to construct an SDK mock that yields messages."""
    async def fake_receive():
        for m in messages:
            yield m

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.receive_response = MagicMock(return_value=fake_receive())

    mock_sdk = MagicMock()
    mock_sdk.ClaudeSDKClient = MagicMock(return_value=mock_client)
    mock_sdk.ClaudeAgentOptions = MagicMock()
    return mock_sdk, mock_client


def test_pins_disable_background_tasks_and_growthbook():
    """Both env pins are present in the dict returned by sdk_subprocess_env."""
    result = sdk_subprocess_env()
    assert "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS" in result, \
        "sdk_subprocess_env must include CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"
    assert result["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1", \
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS must be set to '1'"
    assert "DISABLE_GROWTHBOOK" in result, \
        "sdk_subprocess_env must include DISABLE_GROWTHBOOK"
    assert result["DISABLE_GROWTHBOOK"] == "1", \
        "DISABLE_GROWTHBOOK must be set to '1'"


def test_base_env_preserved_and_pins_win():
    """Base env keys survive and conflicting pin values are overridden.

    A caller's base dict may contain unrelated keys (preserved) or
    conflicting values for a pin (overridden). The pins are applied LAST so
    they always win.
    """
    base = {
        "UNRELATED_VAR": "keep-me",
        "DISABLE_GROWTHBOOK": "999",  # conflicting value
    }
    result = sdk_subprocess_env(base)
    assert result["UNRELATED_VAR"] == "keep-me", \
        "Unrelated keys from base must be preserved"
    assert result["DISABLE_GROWTHBOOK"] == "1", \
        "Pin values must override conflicting base values"
    assert result["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1", \
        "Both pins must be present even when base has a conflict"


@pytest.mark.asyncio
async def test_chat_turn_options_carry_pins():
    """The chat turn runner (agent_runner.py) applies both pins.

    This parity test builds REAL options via run_agent_turn and inspects the
    captured kwargs to ensure both pins reach the SDK. Without this test, a
    deletion of the sdk_subprocess_env call in agent_runner would go unnoticed.
    """
    result = MagicMock()
    result.subtype = "success"
    result.result = "Done"
    result.session_id = "sess-pin-test"
    result.total_cost_usd = 0.001
    result.num_turns = 1
    result.event = None
    result.content = None
    result.tool_use_result = None

    mock_sdk, _ = _mock_sdk([result])
    turn = AgentTurn(turn_id="t-pin", project_id="p1")
    captured_kwargs = {}

    def capture_options(**kwargs):
        captured_kwargs.update(kwargs)
        return MagicMock()

    mock_sdk.ClaudeAgentOptions = MagicMock(side_effect=lambda **kw: capture_options(**kw))
    mock_sdk.HookMatcher = MagicMock(
        side_effect=lambda **kw: SimpleNamespace(
            matcher=kw.get("matcher"), hooks=kw.get("hooks") or []
        )
    )

    with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}), \
         patch("app.services.agent_runner._persist_session", new_callable=AsyncMock), \
         patch("app.services.agent_runner._append_turn_to_transcript", new_callable=AsyncMock):
        await run_agent_turn(turn, "hi", "p1", "test")

    env = captured_kwargs.get("env")
    assert env is not None, \
        "ClaudeAgentOptions must receive an env kwarg from run_agent_turn"
    assert "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS" in env, \
        "Chat turn options must carry CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"
    assert env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1", \
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS must be set to '1' in chat turn options"
    assert "DISABLE_GROWTHBOOK" in env, \
        "Chat turn options must carry DISABLE_GROWTHBOOK"
    assert env["DISABLE_GROWTHBOOK"] == "1", \
        "DISABLE_GROWTHBOOK must be set to '1' in chat turn options"


@pytest.mark.asyncio
async def test_prompt_optimizer_options_carry_pins(tmp_path):
    """The prompt-optimizer session (_run_prompt_optimizer) applies both pins.

    This parity test builds REAL options via _run_prompt_optimizer and
    inspects the captured kwargs. The function swallows every exception and
    returns False, so we FIRST assert that options were captured at all —
    without this a vacuously passing test would hide a setup error.
    """
    captured_kwargs = {}

    def capture_options(**kwargs):
        captured_kwargs.update(kwargs)
        return MagicMock()

    mock_sdk = MagicMock()
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async def fake_query(msg):
        return None

    async def fake_receive():
        return
        yield  # make it an async generator

    mock_client.query = AsyncMock(side_effect=fake_query)
    mock_client.receive_response = MagicMock(return_value=fake_receive())
    mock_sdk.ClaudeSDKClient = MagicMock(return_value=mock_client)
    mock_sdk.ClaudeAgentOptions = MagicMock(side_effect=lambda **kw: capture_options(**kw))
    mock_sdk.HookMatcher = MagicMock(
        side_effect=lambda **kw: SimpleNamespace(
            matcher=kw.get("matcher"), hooks=kw.get("hooks") or []
        )
    )

    mock_project = MagicMock()
    mock_project.agent_model = "opus"
    mock_pm = MagicMock()
    mock_pm.get_project = MagicMock(return_value=mock_project)

    mock_mcp_server = MagicMock()
    mock_platform = MagicMock()
    mock_platform.session_store = None

    with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}), \
         patch("app.main.get_project_manager", return_value=mock_pm), \
         patch("app.services.mcp_server_cache.get_eval_mcp_server", return_value=mock_mcp_server), \
         patch("app.platform.get_platform", return_value=mock_platform):
        await _run_prompt_optimizer("p1", tmp_path, 0, "baseline", {"score": 0.5})

    assert captured_kwargs, \
        "Options must have been captured (function swallows exceptions — without this the test is vacuous)"
    env = captured_kwargs.get("env")
    assert env is not None, \
        "ClaudeAgentOptions must receive an env kwarg from _run_prompt_optimizer"
    assert "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS" in env, \
        "Optimizer options must carry CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"
    assert env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1", \
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS must be set to '1' in optimizer options"
    assert "DISABLE_GROWTHBOOK" in env, \
        "Optimizer options must carry DISABLE_GROWTHBOOK"
    assert env["DISABLE_GROWTHBOOK"] == "1", \
        "DISABLE_GROWTHBOOK must be set to '1' in optimizer options"


@pytest.mark.asyncio
async def test_report_narrator_options_carry_pins(tmp_path):
    """The report-narrator session (_run_narrator) applies both pins.

    Same shape as the optimizer parity test: build REAL options, inspect env,
    assert options were captured at all (vacuity guard).
    """
    captured_kwargs = {}

    def capture_options(**kwargs):
        captured_kwargs.update(kwargs)
        return MagicMock()

    mock_sdk = MagicMock()
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async def fake_query(msg):
        return None

    async def fake_receive():
        return
        yield  # make it an async generator

    mock_client.query = AsyncMock(side_effect=fake_query)
    mock_client.receive_response = MagicMock(return_value=fake_receive())
    mock_sdk.ClaudeSDKClient = MagicMock(return_value=mock_client)
    mock_sdk.ClaudeAgentOptions = MagicMock(side_effect=lambda **kw: capture_options(**kw))
    mock_sdk.HookMatcher = MagicMock(
        side_effect=lambda **kw: SimpleNamespace(
            matcher=kw.get("matcher"), hooks=kw.get("hooks") or []
        )
    )

    mock_project = MagicMock()
    mock_project.agent_model = "opus"
    mock_pm = MagicMock()
    mock_pm.get_project = MagicMock(return_value=mock_project)

    mock_mcp_server = MagicMock()
    mock_platform = MagicMock()
    mock_platform.session_store = None

    with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}), \
         patch("app.main.get_project_manager", return_value=mock_pm), \
         patch("app.services.mcp_server_cache.get_eval_mcp_server", return_value=mock_mcp_server), \
         patch("app.platform.get_platform", return_value=mock_platform):
        await _run_narrator(tmp_path, "p1", "narrate the report", None)

    assert captured_kwargs, \
        "Options must have been captured (function swallows exceptions — without this the test is vacuous)"
    env = captured_kwargs.get("env")
    assert env is not None, \
        "ClaudeAgentOptions must receive an env kwarg from _run_narrator"
    assert "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS" in env, \
        "Narrator options must carry CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"
    assert env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1", \
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS must be set to '1' in narrator options"
    assert "DISABLE_GROWTHBOOK" in env, \
        "Narrator options must carry DISABLE_GROWTHBOOK"
    assert env["DISABLE_GROWTHBOOK"] == "1", \
        "DISABLE_GROWTHBOOK must be set to '1' in narrator options"
