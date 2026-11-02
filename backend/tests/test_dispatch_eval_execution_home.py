# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch_eval's old path must refuse to run inside the microVM.

The old (``arch_split=False``) branch spawns the harness as an in-process
subprocess of whatever process executes the tool. Executed in the AgentCore
microVM that means the eval runs *there*: invisible to the coordinator's
JobTracker, reconciler and dispatch caps, and killed by the microVM's own
idle/lifetime limits.

Reaching that branch from the microVM means the flags disagree — relay ON with
arch_split OFF — which is a misconfiguration, not a fallback. So it must fail
loud with a structured error the agent can report, rather than silently starting a
second execution home. This is the guard for the repeatedly-observed
"second execution home" bug class.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.platform import process_role
from app.services.jobs import get_job_tracker
from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool

_dispatch_eval = _dispatch_eval_tool.handler

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _reset_tracker():
    get_job_tracker()._reset()
    yield
    get_job_tracker()._reset()


def _project(tmp_path: Path) -> Path:
    """Scaffold the minimum a dispatch needs on disk (data file + prompt)."""
    (tmp_path / "run_eval.py").write_text("# harness\n")
    (tmp_path / "prompts").mkdir(exist_ok=True)
    (tmp_path / "prompts" / "baseline.md").write_text("prompt\n")
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "cases.csv").write_text("q,a\n1,2\n")
    return tmp_path


async def test_old_path_refused_in_agentcore_runtime(tmp_path: Path):
    """In the microVM, the in-process eval path must fail loud.

    Against the unguarded code this returns a normal "dispatched" payload and the
    harness subprocess starts inside the microVM.
    """
    _project(tmp_path)
    spawned: list[str] = []

    def _track_spawn(*_args, **_kwargs):
        spawned.append("create_task")
        return MagicMock()

    with (
        patch.object(
            process_role, "_PROCESS_ROLE", process_role.AGENTCORE_RUNTIME_ROLE
        ),
        patch(
            "app.services.mcp_tools.dispatch.resolve_project_dir",
            return_value=tmp_path,
        ),
        # arch_split OFF while running in the microVM: the misconfigured
        # relay-ON/arch_split-OFF combination.
        patch(
            "app.platform.arch_split.resolve_arch_split",
            AsyncMock(return_value=False),
        ),
        patch(
            "app.services.mcp_tools.dispatch._preflight_models_invocable",
            AsyncMock(return_value=[]),
        ),
        patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
        patch(
            "app.services.mcp_tools.dispatch.asyncio.create_task",
            side_effect=_track_spawn,
        ),
        patch("app.services.mcp_tools.dispatch.attach_error_sink"),
    ):
        result = await _dispatch_eval({
            "project_id": "proj-1",
            "target_model_ids": ["model-a"],
            "data_file": "cases.csv",
            "prompt_version": "baseline",
        })

    assert result.get("is_error"), (
        f"The microVM must REFUSE the in-process eval path, got: {result}"
    )
    assert spawned == [], (
        "No eval task may be spawned inside the AgentCore runtime — the harness "
        f"would run in the microVM, invisible to job tracking. Spawned: {spawned}"
    )


async def test_old_path_allowed_in_coordinator(tmp_path: Path):
    """Counterpart: the coordinator may still run the in-process path.

    Dev mode and old-path cloud projects depend on it, so the guard must be scoped
    to the microVM rather than disabling the branch outright.
    """
    _project(tmp_path)
    spawned: list[str] = []

    def _track_spawn(*_args, **_kwargs):
        spawned.append("create_task")
        return MagicMock()

    with (
        patch.object(process_role, "_PROCESS_ROLE", process_role.COORDINATOR_ROLE),
        patch(
            "app.services.mcp_tools.dispatch.resolve_project_dir",
            return_value=tmp_path,
        ),
        patch(
            "app.platform.arch_split.resolve_arch_split",
            AsyncMock(return_value=False),
        ),
        patch(
            "app.services.mcp_tools.dispatch._preflight_models_invocable",
            AsyncMock(return_value=[]),
        ),
        patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
        patch(
            "app.services.mcp_tools.dispatch.asyncio.create_task",
            side_effect=_track_spawn,
        ),
        patch("app.services.mcp_tools.dispatch.attach_error_sink"),
    ):
        result = await _dispatch_eval({
            "project_id": "proj-1",
            "target_model_ids": ["model-a"],
            "data_file": "cases.csv",
            "prompt_version": "baseline",
        })

    assert not result.get("is_error"), (
        f"The coordinator must still be able to run in-process evals: {result}"
    )
    assert spawned == ["create_task"], (
        f"The coordinator should spawn the eval task: {spawned}"
    )
