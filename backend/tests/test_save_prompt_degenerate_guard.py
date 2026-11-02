# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for save_prompt degenerate-iteration guard.

Effect-level coverage: refuse prompt emission when the optimization
is in a terminal-failure state with no usable iteration data, but still
permit saves in all must-work cases (no state, running loop, real iteration
data, benign terminal, unreadable state).

Every assertion is on the EFFECT: whether the file exists on disk, plus
the is_error flag. No mock-call counts — a mock-call assertion can pass while
the prompt file the user needs was never written.
"""

import json
from pathlib import Path
from unittest.mock import patch
import pytest

from app.services.mcp_tools.prompt import save_prompt as _save_prompt_tool
from app.services.optimization_state import OptimizationState

# Import the handler for async calls
save_prompt = _save_prompt_tool.handler


@pytest.fixture
def mock_project(tmp_path: Path):
    """Create a minimal project directory with config.json and prompts/."""
    config = {
        "project_name": "test-project",
        "models": ["test-model"],
        "judge_model_id": "judge-model",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "prompts").mkdir()
    return tmp_path


@pytest.mark.asyncio
async def test_refusal_generation_unreliable_zero_iterations(mock_project):
    """Assert terminal-failure generation_unreliable + zero iterations refuses."""
    state = OptimizationState.create(mock_project, target_metric_scores={"M": 0.9})
    state.status = "generation_unreliable"
    state.iterations = []
    state.save()

    content = "Test prompt content"
    prompt_path = mock_project / "prompts" / "v1.md"
    project_id = "test-project"

    with patch("app.services.mcp_tools.prompt.resolve_project_dir", return_value=mock_project):
        result = await save_prompt({
            "project_id": project_id,
            "version": "v1",
            "content": content,
        })

    # File must NOT exist
    assert not prompt_path.exists()
    # Response must carry is_error
    assert result.get("is_error") is True


@pytest.mark.asyncio
async def test_refusal_judge_unreliable_zero_iterations(mock_project):
    """Assert terminal-failure judge_unreliable + zero iterations refuses."""
    state = OptimizationState.create(mock_project, target_metric_scores={"M": 0.9})
    state.status = "judge_unreliable"
    state.iterations = []
    state.save()

    content = "Test prompt content"
    prompt_path = mock_project / "prompts" / "v1.md"
    project_id = "test-project"

    with patch("app.services.mcp_tools.prompt.resolve_project_dir", return_value=mock_project):
        result = await save_prompt({
            "project_id": project_id,
            "version": "v1",
            "content": content,
        })

    assert not prompt_path.exists()
    assert result.get("is_error") is True


@pytest.mark.asyncio
async def test_refusal_total_cases_zero(mock_project):
    """Assert terminal-failure whose newest iteration has total_cases == 0 refuses."""
    state = OptimizationState.create(mock_project, target_metric_scores={"M": 0.9})
    state.status = "generation_unreliable"
    state.record_iteration(
        prompt_version="baseline",
        run_id="run-1",
        metrics_summary={},
        total_cases=0,
        passed=0,
    )
    state.save()

    content = "Test prompt content"
    prompt_path = mock_project / "prompts" / "v1.md"
    project_id = "test-project"

    with patch("app.services.mcp_tools.prompt.resolve_project_dir", return_value=mock_project):
        result = await save_prompt({
            "project_id": project_id,
            "version": "v1",
            "content": content,
        })

    assert not prompt_path.exists()
    assert result.get("is_error") is True


@pytest.mark.asyncio
async def test_refusal_failed_zero_iterations(mock_project):
    """Assert status failed + zero iterations refuses."""
    state = OptimizationState.create(mock_project, target_metric_scores={"M": 0.9})
    state.status = "failed"
    state.iterations = []
    state.save()

    content = "Test prompt content"
    prompt_path = mock_project / "prompts" / "v1.md"
    project_id = "test-project"

    with patch("app.services.mcp_tools.prompt.resolve_project_dir", return_value=mock_project):
        result = await save_prompt({
            "project_id": project_id,
            "version": "v1",
            "content": content,
        })

    assert not prompt_path.exists()
    assert result.get("is_error") is True


@pytest.mark.asyncio
async def test_refusal_message_names_reason(mock_project):
    """Assert refusal message names the reason (halted without usable data)."""
    state = OptimizationState.create(mock_project, target_metric_scores={"M": 0.9})
    state.status = "generation_unreliable"
    state.iterations = []
    state.save()

    content = "Test prompt content"
    project_id = "test-project"

    with patch("app.services.mcp_tools.prompt.resolve_project_dir", return_value=mock_project):
        result = await save_prompt({
            "project_id": project_id,
            "version": "v1",
            "content": content,
        })

    assert result.get("is_error") is True
    # Get message from content field (MCP response format)
    content_field = result.get("content", [])
    message = content_field[0].get("text", "") if content_field else ""
    assert "halted" in message.lower() or "iteration" in message.lower()
    assert "invalid" not in message.lower()  # Must be specific


@pytest.mark.asyncio
async def test_must_work_no_state(mock_project):
    """Assert save works when no optimization state exists."""
    # No state file — typical for user/agent saving outside a loop
    content = "Test prompt content"
    prompt_path = mock_project / "prompts" / "v1.md"
    project_id = "test-project"

    with patch("app.services.mcp_tools.prompt.resolve_project_dir", return_value=mock_project):
        result = await save_prompt({
            "project_id": project_id,
            "version": "v1",
            "content": content,
        })

    # File must exist
    assert prompt_path.exists()
    assert prompt_path.read_text() == content
    # No is_error
    assert result.get("is_error") is not True


@pytest.mark.asyncio
async def test_must_work_running_with_healthy_iteration(mock_project):
    """Assert save works during a running loop with real iteration data."""
    state = OptimizationState.create(mock_project, target_metric_scores={"M": 0.9})
    state.status = "running"
    state.record_iteration(
        prompt_version="baseline",
        run_id="run-1",
        metrics_summary={"M": 0.5},
        total_cases=100,
        passed=50,
    )
    state.save()

    content = "Test prompt content"
    prompt_path = mock_project / "prompts" / "v1.md"
    project_id = "test-project"

    with patch("app.services.mcp_tools.prompt.resolve_project_dir", return_value=mock_project):
        result = await save_prompt({
            "project_id": project_id,
            "version": "v1",
            "content": content,
        })

    assert prompt_path.exists()
    assert result.get("is_error") is not True


@pytest.mark.asyncio
async def test_must_work_optimizer_exhausted_with_real_data(mock_project):
    """Assert save works with optimizer_exhausted + real iteration data.

    The documented "edit the prompt to resume" flow must not break.
    """
    state = OptimizationState.create(mock_project, target_metric_scores={"M": 0.9})
    state.status = "optimizer_exhausted"
    state.record_iteration(
        prompt_version="v1",
        run_id="run-1",
        metrics_summary={"M": 0.7},
        total_cases=100,
        passed=70,
    )
    state.save()

    content = "Test prompt content"
    prompt_path = mock_project / "prompts" / "v2.md"
    project_id = "test-project"

    with patch("app.services.mcp_tools.prompt.resolve_project_dir", return_value=mock_project):
        result = await save_prompt({
            "project_id": project_id,
            "version": "v2",
            "content": content,
        })

    assert prompt_path.exists()
    assert result.get("is_error") is not True


@pytest.mark.asyncio
async def test_must_work_benign_terminal_converged(mock_project):
    """Assert save works with benign terminal converged + real data."""
    state = OptimizationState.create(mock_project, target_metric_scores={"M": 0.9})
    state.status = "converged"
    state.record_iteration(
        prompt_version="v1",
        run_id="run-1",
        metrics_summary={"M": 0.95},
        total_cases=100,
        passed=95,
    )
    state.save()

    content = "Test prompt content"
    prompt_path = mock_project / "prompts" / "v2.md"
    project_id = "test-project"

    with patch("app.services.mcp_tools.prompt.resolve_project_dir", return_value=mock_project):
        result = await save_prompt({
            "project_id": project_id,
            "version": "v2",
            "content": content,
        })

    assert prompt_path.exists()
    assert result.get("is_error") is not True


@pytest.mark.asyncio
async def test_must_work_benign_terminal_plateau(mock_project):
    """Assert save works with benign terminal plateau + real data."""
    state = OptimizationState.create(mock_project, target_metric_scores={"M": 0.9})
    state.status = "plateau"
    state.record_iteration(
        prompt_version="v1",
        run_id="run-1",
        metrics_summary={"M": 0.7},
        total_cases=100,
        passed=70,
    )
    state.save()

    content = "Test prompt content"
    prompt_path = mock_project / "prompts" / "v2.md"
    project_id = "test-project"

    with patch("app.services.mcp_tools.prompt.resolve_project_dir", return_value=mock_project):
        result = await save_prompt({
            "project_id": project_id,
            "version": "v2",
            "content": content,
        })

    assert prompt_path.exists()
    assert result.get("is_error") is not True


@pytest.mark.asyncio
async def test_fail_open_unreadable_state_logged(mock_project):
    """Assert state-read failure allows save AND logs warning.

    Fail-open path: an unreadable state must never brick prompt writes.
    Both halves: (a) file created, no is_error; (b) failure logged.
    """
    content = "Test prompt content"
    prompt_path = mock_project / "prompts" / "v1.md"
    project_id = "test-project"

    with patch("app.services.mcp_tools.prompt.resolve_project_dir", return_value=mock_project), \
         patch("app.services.optimization_state.OptimizationState.load", side_effect=RuntimeError("boom")), \
         patch("app.services.mcp_tools.prompt.logger") as mock_logger:
        result = await save_prompt({
            "project_id": project_id,
            "version": "v1",
            "content": content,
        })

    # (a) File created, no is_error
    assert prompt_path.exists()
    assert result.get("is_error") is not True

    # (b) Failure logged
    assert mock_logger.warning.called
    # Check that the exception type or message appears in the log
    call_args = mock_logger.warning.call_args
    logged_text = " ".join(str(arg) for arg in call_args[0]) if call_args else ""
    assert "RuntimeError" in logged_text or "boom" in logged_text


@pytest.mark.asyncio
async def test_interaction_guard_dedupe_still_fires(mock_project):
    """Assert existing identical-content dedupe refusal still fires."""
    # Create an existing prompt with identical content
    (mock_project / "prompts" / "v1.md").write_text("Duplicate content")

    content = "Duplicate content"
    prompt_path = mock_project / "prompts" / "v2.md"
    project_id = "test-project"

    with patch("app.services.mcp_tools.prompt.resolve_project_dir", return_value=mock_project):
        result = await save_prompt({
            "project_id": project_id,
            "version": "v2",
            "content": content,
        })

    # Dedupe should refuse
    assert not prompt_path.exists()
    assert result.get("is_error") is True
    # Get message from content field
    content_field = result.get("content", [])
    message = content_field[0].get("text", "") if content_field else ""
    assert "identical" in message.lower() or "already" in message.lower()
