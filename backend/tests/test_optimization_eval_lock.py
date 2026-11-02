# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""An optimization run must HOLD the eval lock, not merely check it.

``dispatch_optimization`` checked ``get_eval_lock`` and moved on, with a comment
claiming "the first iteration eval will claim it". It does not:
``claim_eval_lock`` has exactly one call site in the codebase — the manual
``dispatch_eval`` path — and ``optimization_orchestrator.dispatch_eval_job``
contains no claim at all.

So the lock stayed FREE across the ~15-20s invocability preflight that runs
between the check and the loop spawn. A manual eval dispatched in that window
passed both fences (its own check saw no lock; the optimization check had already
returned) and the project ended up running two detached evals at once, violating
one-eval-per-project.

The fix claims the lock before preflight under a synthetic run-scoped holder id,
releases it on every dispatch-side rejection, and hands ownership to the loop task
whose terminal tail releases it.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio


class _FakeStore:
    """Minimal metadata store recording eval-lock claim/release calls."""

    def __init__(self) -> None:
        self.locks: dict[str, tuple[str, str]] = {}  # project -> (owner, holder)
        self.claims: list[tuple[str, str, str]] = []
        self.releases: list[tuple[str, str, str]] = []

    async def claim_eval_lock(self, owner_id, project_id, job_id) -> bool:
        self.claims.append((owner_id, project_id, job_id))
        if project_id in self.locks:
            return False
        self.locks[project_id] = (owner_id, job_id)
        return True

    async def release_eval_lock(self, owner_id, project_id, job_id) -> None:
        self.releases.append((owner_id, project_id, job_id))
        held = self.locks.get(project_id)
        # Holder-conditional, like both real stores.
        if held and held == (owner_id, job_id):
            del self.locks[project_id]

    async def get_eval_lock(self, owner_id, project_id):
        held = self.locks.get(project_id)
        if not held:
            return None
        return {"lock_holder_job_id": held[1], "lock_version": 1}

    async def get_optimization_state(self, *_args, **_kwargs):
        return None

    async def count_active_jobs_for_owner(self, *_args, **_kwargs) -> int:
        return 0

    async def list_non_terminal_jobs(self):
        return []


def _harness(models=("model-a",)):
    """Valid harness config stand-in."""
    h = MagicMock()
    h.models = list(models)
    h.field_mappings = {"input": "q", "expected_output": "a"}
    h.judge_model_id = None
    h.target_metric_scores = {"ExactMatch": 0.85}
    h.max_iterations = 3
    h.plateau_window = 2
    h.plateau_threshold = None
    h.data_file = "data.csv"
    return h


def _base_patches(store, project_dir: Path, *, preflight_bad=None, harness=None):
    """Patch set shared by the dispatch_optimization tests."""
    platform = MagicMock()
    platform.metadata_store = store
    return [
        patch(
            "app.services.mcp_tools.dispatch.resolve_project_dir",
            return_value=project_dir,
        ),
        patch(
            "app.platform.arch_split.resolve_arch_split",
            AsyncMock(return_value=True),
        ),
        patch(
            "app.platform.arch_split.is_cost_kill_switch_active", return_value=False
        ),
        patch("app.platform.get_platform", return_value=platform),
        patch(
            "app.services.agent_turn._resolve_owner_id_for_project",
            return_value="owner-1",
        ),
        patch(
            "app.services.mcp_tools.dispatch.load_harness_config",
            return_value=harness if harness is not None else _harness(),
        ),
        patch(
            "app.services.mcp_tools.dispatch.read_descriptors", return_value={}
        ),
        patch(
            "app.services.mcp_tools.dispatch._preflight_models_invocable",
            AsyncMock(return_value=preflight_bad or []),
        ),
    ]


async def test_lock_is_claimed_before_preflight(tmp_path: Path):
    """The lock must be HELD by the time the invocability preflight runs.

    This is the whole point: the preflight is the ~15-20s window a concurrent
    manual eval used to slip through. Asserted by having the preflight itself
    observe the lock state.
    """
    from app.services.mcp_tools.dispatch import (
        dispatch_optimization as _tool,
    )

    # The decorator wraps the coroutine in an SdkMcpTool; .handler is the
    # callable the agent loop invokes (same access pattern as
    # test_dispatch_preflight.py).
    dispatch_optimization = _tool.handler

    store = _FakeStore()
    lock_state_during_preflight: list[bool] = []

    async def _observing_preflight(*_args, **_kwargs):
        # Record whether the lock is held at preflight time.
        lock_state_during_preflight.append("proj-1" in store.locks)
        return []

    patches = _base_patches(store, tmp_path)
    # Replace the preflight patch with the observing one.
    patches[-1] = patch(
        "app.services.mcp_tools.dispatch._preflight_models_invocable",
        side_effect=_observing_preflight,
    )

    with (
        patches[0], patches[1], patches[2], patches[3], patches[4],
        patches[5], patches[6], patches[7],
        patch(
            "app.services.optimization_state.OptimizationState.create",
            return_value=MagicMock(
                target_metric_scores={"ExactMatch": 0.85}, max_iterations=3
            ),
        ),
        patch(
            "app.services.optimization_orchestrator.detect_latest_prompt",
            return_value="baseline",
        ),
        patch("app.services.mcp_tools.dispatch.asyncio.create_task"),
        patch("app.services.mcp_tools.dispatch.attach_error_sink"),
    ):
        await dispatch_optimization({"project_id": "proj-1"})

    assert lock_state_during_preflight == [True], (
        "The eval lock MUST already be held when the invocability preflight runs "
        "— that window is exactly where a concurrent manual eval slipped in. "
        f"Observed: {lock_state_during_preflight}"
    )
    assert store.claims, "The run must CLAIM the lock, not just check it"


async def test_lock_released_when_preflight_rejects(tmp_path: Path):
    """A dispatch that never starts the loop must not leave the lock held.

    Otherwise a bad model lineup wedges every future dispatch on the project
    until the 24h TTL.
    """
    from app.services.mcp_tools.dispatch import (
        dispatch_optimization as _tool,
    )

    # The decorator wraps the coroutine in an SdkMcpTool; .handler is the
    # callable the agent loop invokes (same access pattern as
    # test_dispatch_preflight.py).
    dispatch_optimization = _tool.handler

    store = _FakeStore()
    patches = _base_patches(store, tmp_path, preflight_bad=["model-a: not invocable"])

    with (
        patches[0], patches[1], patches[2], patches[3], patches[4],
        patches[5], patches[6], patches[7],
    ):
        result = await dispatch_optimization({"project_id": "proj-1"})

    assert result.get("is_error"), "A failed preflight must return an error"
    assert "proj-1" not in store.locks, (
        "The eval lock MUST be released when optimization never starts — "
        f"otherwise every later dispatch is blocked. Releases: {store.releases}"
    )


async def test_lock_released_on_invalid_config(tmp_path: Path):
    """Every dispatch-side validation rejection must release the lock too."""
    from app.services.mcp_tools.dispatch import (
        dispatch_optimization as _tool,
    )

    # The decorator wraps the coroutine in an SdkMcpTool; .handler is the
    # callable the agent loop invokes (same access pattern as
    # test_dispatch_preflight.py).
    dispatch_optimization = _tool.handler

    store = _FakeStore()
    # Out-of-range target -> rejected AFTER the claim.
    patches = _base_patches(store, tmp_path)

    with (
        patches[0], patches[1], patches[2], patches[3], patches[4],
        patches[5], patches[6], patches[7],
    ):
        result = await dispatch_optimization(
            {"project_id": "proj-1", "target_metric_scores": {"ExactMatch": 3000.0}}
        )

    assert result.get("is_error")
    assert "proj-1" not in store.locks, (
        f"Validation rejection must release the lock. Releases: {store.releases}"
    )


async def test_concurrent_manual_eval_is_refused_during_preflight(tmp_path: Path):
    """With the lock held, a manual eval claim during preflight FAILS.

    The behavioral end-state of the fix: the second claimant is refused instead of
    both evals running.
    """
    from app.services.mcp_tools.dispatch import (
        dispatch_optimization as _tool,
    )

    # The decorator wraps the coroutine in an SdkMcpTool; .handler is the
    # callable the agent loop invokes (same access pattern as
    # test_dispatch_preflight.py).
    dispatch_optimization = _tool.handler

    store = _FakeStore()
    manual_claim_result: list[bool] = []

    async def _preflight_with_racing_manual_eval(*_args, **_kwargs):
        # Simulate dispatch_eval trying to claim mid-preflight.
        won = await store.claim_eval_lock("owner-1", "proj-1", "manual-eval-job")
        manual_claim_result.append(won)
        return []

    patches = _base_patches(store, tmp_path)
    patches[-1] = patch(
        "app.services.mcp_tools.dispatch._preflight_models_invocable",
        side_effect=_preflight_with_racing_manual_eval,
    )

    with (
        patches[0], patches[1], patches[2], patches[3], patches[4],
        patches[5], patches[6], patches[7],
        patch(
            "app.services.optimization_state.OptimizationState.create",
            return_value=MagicMock(
                target_metric_scores={"ExactMatch": 0.85}, max_iterations=3
            ),
        ),
        patch(
            "app.services.optimization_orchestrator.detect_latest_prompt",
            return_value="baseline",
        ),
        patch("app.services.mcp_tools.dispatch.asyncio.create_task"),
        patch("app.services.mcp_tools.dispatch.attach_error_sink"),
    ):
        await dispatch_optimization({"project_id": "proj-1"})

    assert manual_claim_result == [False], (
        "A manual eval dispatched during the optimization preflight MUST be "
        "refused the lock — otherwise two detached evals run on one project, "
        f"violating one-eval-per-project. Got: {manual_claim_result}"
    )


async def test_loop_releases_lock_on_terminal_exit(tmp_path: Path):
    """The loop's terminal tail must release the run's lock on every exit.

    Ownership is handed from dispatch to the loop task, so if the loop does not
    release, the lock outlives the run and blocks the project until its TTL.
    """
    from app.services.optimization_loop_orchestrator import run_optimization_loop

    store = _FakeStore()
    await store.claim_eval_lock("owner-1", "proj-1", "optimization-proj-1")
    assert "proj-1" in store.locks

    platform = MagicMock()
    platform.metadata_store = store

    iteration_config = {
        "project_id": "proj-1",
        "project_dir": str(tmp_path),
        "prompt_version": "baseline",
        "max_iterations": 1,
        "arch_split": True,
        "user_id": "owner-1",
        "eval_lock_owner_id": "owner-1",
        "eval_lock_holder_id": "optimization-proj-1",
    }

    with patch("app.platform.get_platform", return_value=platform):
        # get_optimization_state returns None → the loop exits immediately on its
        # first cancellation check, exercising the terminal tail.
        await run_optimization_loop(iteration_config)

    assert "proj-1" not in store.locks, (
        "The loop MUST release the run's eval lock on terminal exit — otherwise "
        f"the lock outlives the run. Releases: {store.releases}"
    )
    assert ("owner-1", "proj-1", "optimization-proj-1") in store.releases


async def test_loop_releases_lock_even_when_terminal_repair_raises(tmp_path: Path):
    """The release must survive a failure in the final state repair.

    The release used to trail the try/except rather than sit in a ``finally``,
    with a comment asserting it ran on every exit "because it sits outside the
    try/except above". The state repair immediately before it does a store read
    and a versioned write that raises on a concurrent update, so a raise there
    jumped past the release and leaked the lock until its 24h TTL — blocking
    every later eval on the project.
    """
    from app.services.optimization_loop_orchestrator import run_optimization_loop

    store = _FakeStore()
    await store.claim_eval_lock("owner-1", "proj-1", "optimization-proj-1")

    platform = MagicMock()
    platform.metadata_store = store

    iteration_config = {
        "project_id": "proj-1",
        "project_dir": str(tmp_path),
        "prompt_version": "baseline",
        "max_iterations": 1,
        "arch_split": True,
        "user_id": "owner-1",
        "eval_lock_owner_id": "owner-1",
        "eval_lock_holder_id": "optimization-proj-1",
    }

    with patch("app.platform.get_platform", return_value=platform), patch(
        "app.services.optimization_state_repository.read",
        new=AsyncMock(side_effect=RuntimeError("store unavailable")),
    ):
        # Must not propagate: the loop owns its own teardown.
        await run_optimization_loop(iteration_config)

    assert "proj-1" not in store.locks, (
        "a failed terminal repair must not strand the run's eval lock; "
        f"releases: {store.releases}"
    )


async def test_loop_crash_is_actually_recorded(tmp_path: Path):
    """A crash must reach the state, the bus and chat — not just the log.

    Worth testing directly rather than only through the loop: the caller wraps
    this in a try/except so a teardown failure cannot escape, which means a
    broken reporter is invisible from the outside. It was — an extraction moved
    this code away from the function-local imports it depended on, and every
    surrounding test still passed because the resulting NameError was swallowed.
    """
    from types import SimpleNamespace

    from app.services import optimization_loop_orchestrator as loop_mod
    from app.services.resource_error import Codes

    state = SimpleNamespace(status="running", error=None)
    handle = SimpleNamespace(state=state, is_live=lambda: True)

    published: list[dict] = []
    chat: list[str] = []

    with patch(
        "app.services.optimization_state_repository.read",
        new=AsyncMock(return_value=handle),
    ), patch(
        "app.services.optimization_state_repository.write", return_value=handle
    ) as write, patch.object(
        loop_mod, "_publish_project_event", side_effect=lambda _pid, ev: published.append(ev)
    ), patch(
        "app.services.completion_turn.request_terminal_message",
        new=AsyncMock(side_effect=lambda _pid, msg: chat.append(msg)),
    ):
        await loop_mod._report_loop_crash(
            RuntimeError("boom"),
            project_id="proj-1",
            project_dir=tmp_path,
            arch_split=True,
            user_id="owner-1",
        )

    assert state.status == "cancelled"
    assert state.error is not None and state.error.code == Codes.SYSTEM_TASK_CRASHED
    write.assert_called_once()
    assert [ev["type"] for ev in published] == ["error", "terminal"]
    assert chat and "boom" in chat[0]


async def test_loop_crash_does_not_overwrite_an_earlier_cause(tmp_path: Path):
    """The FIRST recorded error is the diagnosis; a later symptom must not win."""
    from types import SimpleNamespace

    from app.services import optimization_loop_orchestrator as loop_mod
    from app.services.resource_error import Codes, ResourceError

    first = ResourceError(code=Codes.SYSTEM_TASK_CRASHED, message="the real cause")
    state = SimpleNamespace(status="running", error=first)
    handle = SimpleNamespace(state=state, is_live=lambda: True)

    with patch(
        "app.services.optimization_state_repository.read",
        new=AsyncMock(return_value=handle),
    ), patch(
        "app.services.optimization_state_repository.write", return_value=handle
    ) as write:
        await loop_mod._report_loop_crash(
            RuntimeError("downstream symptom"),
            project_id="proj-1",
            project_dir=tmp_path,
            arch_split=True,
            user_id="owner-1",
        )

    assert state.error is first, "the original diagnosis must survive"
    write.assert_not_called()
