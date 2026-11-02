# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution-home guards: long-lived work must not run in the AgentCore microVM.

The microVM runs the SAME coordinator turn code as the coordinator process but has
different capabilities: zero browser WebSocket connections, no reconciler, no job
tracker the coordinator can see, and a hard idle/max-lifetime ceiling that kills
whatever is still running. Work the agent dispatches from inside a turn is created
in whichever process handled that turn, so it landed in the microVM — where it
appears to work and then silently doesn't.

The structural answer is the dispatch-to-owner seam: work dispatched from the
microVM leaves a durable record, and the coordinator adopts and runs it
(``adopt_orphan_optimization_loops``). These guards turn each remaining silent
wrong-home behavior into something loud:

* the optimization loop REFUSES to run and reports an error (it cannot be made
  correct in the microVM — it outlives the session that created it);
* the poller refuses;
* ``dispatch_eval``'s in-process path refuses;
* a project publish that no live turn can carry is stashed in the durable outbox
  for the coordinator's reconciler to drain, and the local publish is still
  attempted, because suppressing it would erase the only trace that the wrong
  home is in use.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.platform import process_role

pytestmark = pytest.mark.asyncio


def _noop_hub() -> MagicMock:
    """Hub stand-in whose publish_project returns a real coroutine.

    ``loop.create_task`` rejects a MagicMock return value outright, so a bare
    MagicMock hub raises TypeError before the code under test is reached.
    """
    hub = MagicMock()

    async def _publish_project(*_args, **_kwargs):
        return None

    hub.publish_project = _publish_project
    return hub


async def test_optimization_loop_refuses_to_run_in_the_microvm(tmp_path: Path):
    """The loop must refuse in the microVM rather than run where it cannot survive.

    In the microVM the loop's publishes reach a hub with no connections, its child
    evals are invisible to the coordinator's tracker/reconciler/caps, and the whole
    thing dies when the session hits its idle or max-lifetime ceiling — stranding
    optimization state non-terminal with no cleanup.
    """
    from app.services.optimization_loop_orchestrator import run_optimization_loop

    published: list[dict] = []

    def _capture(project_id, event):
        published.append(event)

    with (
        patch.object(
            process_role, "_PROCESS_ROLE", process_role.AGENTCORE_RUNTIME_ROLE
        ),
        patch(
            "app.services.optimization_loop_orchestrator._publish_project_event",
            side_effect=_capture,
        ),
        patch(
            "app.services.optimization_loop_orchestrator.get_job_tracker"
        ) as mock_tracker,
    ):
        await run_optimization_loop({
            "project_id": "proj-1",
            "project_dir": str(tmp_path),
            "prompt_version": "baseline",
            "max_iterations": 3,
            "arch_split": True,
            "user_id": "owner-1",
        })

    # It must not have created a single iteration job.
    mock_tracker.return_value.create_job.assert_not_called()

    # And it must SAY so — a silent return would look identical to a loop that ran
    # and immediately converged.
    assert published, "the refusal must publish an error event, not fail silently"
    assert published[0]["type"] == "error"
    assert "execution_home" in published[0].get("code", ""), (
        f"the error must name the cause; got {published[0]}"
    )


async def test_optimization_loop_runs_in_the_coordinator(tmp_path: Path):
    """Counterpart: the coordinator must still run the loop.

    Scopes the guard to the microVM rather than disabling the loop everywhere — the
    failure mode of an over-broad guard is a feature that silently never runs.
    """
    from app.services.optimization_loop_orchestrator import run_optimization_loop
    from app.services.optimization_state import OptimizationState

    # No state on disk -> the loop exits at its first cancellation check. That is
    # enough to prove it got PAST the execution-home guard, which is the subject.
    published: list[dict] = []

    with (
        patch.object(process_role, "_PROCESS_ROLE", process_role.COORDINATOR_ROLE),
        patch(
            "app.services.optimization_loop_orchestrator._publish_project_event",
            side_effect=lambda p, e: published.append(e),
        ),
        patch("app.services.optimization_loop_orchestrator.get_job_tracker"),
        patch.object(OptimizationState, "load", staticmethod(lambda _d: None)),
    ):
        await run_optimization_loop({
            "project_id": "proj-1",
            "project_dir": str(tmp_path),
            "prompt_version": "baseline",
            "max_iterations": 1,
            "arch_split": False,
            "user_id": None,
        })

    wrong_home = [
        e for e in published
        if "execution_home" in (e.get("code") or "")
    ]
    assert not wrong_home, (
        f"the coordinator must not be refused by the microVM guard: {wrong_home}"
    )


async def test_project_publish_from_microvm_stashes_every_event(tmp_path: Path):
    """Every unroutable microVM publish is stashed.

    This used to pin a once-per-process warning while all five events
    reached nobody. The contract is now the durable outbox: each unroutable
    event is stashed for the coordinator's reconciler to drain — per event,
    not per process, because every one of them is user-visible signal.
    """
    import app.services.jobs as jobs_mod

    project = MagicMock()
    project.owner_id = "owner-1"
    pm = MagicMock()
    pm.get_project.return_value = project

    stashed: list[dict] = []

    async def _stash(owner_id, project_id, event):
        stashed.append(event)

    with (
        patch.object(
            process_role, "_PROCESS_ROLE", process_role.AGENTCORE_RUNTIME_ROLE
        ),
        patch("app.main.get_project_manager", return_value=pm),
        patch("app.services.ws_hub.get_ws_hub", return_value=_noop_hub()),
        patch(
            "app.services.project_event_outbox.stash_unroutable_event",
            side_effect=_stash,
        ),
    ):
        for i in range(5):
            jobs_mod._publish_project_event(
                "proj-1", {"resource": "eval_job", "type": "started", "n": i}
            )
        await asyncio.sleep(0)

    assert len(stashed) == 5, (
        f"every unroutable publish must be stashed; got {len(stashed)}"
    )
    assert [e["n"] for e in stashed] == [0, 1, 2, 3, 4]


async def test_project_publish_from_coordinator_does_not_warn():
    """The coordinator is the correct home — no diagnostic noise there."""
    import app.services.jobs as jobs_mod


    project = MagicMock()
    project.owner_id = "owner-1"
    pm = MagicMock()
    pm.get_project.return_value = project

    with (
        patch.object(process_role, "_PROCESS_ROLE", process_role.COORDINATOR_ROLE),
        patch("app.main.get_project_manager", return_value=pm),
        patch("app.services.ws_hub.get_ws_hub", return_value=_noop_hub()),
        patch.object(jobs_mod, "logger") as mock_logger,
    ):
        jobs_mod._publish_project_event(
            "proj-1", {"resource": "eval_job", "type": "started"}
        )

    warnings = [
        c for c in mock_logger.warning.call_args_list
        if "AgentCore runtime" in str(c)
    ]
    assert warnings == [], f"coordinator publishes must not warn: {warnings}"


async def test_publish_still_happens_from_the_microvm():
    """The guard must not SUPPRESS the publish, only report it.

    Suppressing would change behavior on a path that is already degraded, and would
    erase the trace. Also: if the coordinator and microVM ever share a hub, the
    publish becomes correct again without another code change.
    """
    import app.services.jobs as jobs_mod

    project = MagicMock()
    project.owner_id = "owner-1"
    pm = MagicMock()
    pm.get_project.return_value = project

    hub = MagicMock()
    published: list[tuple] = []

    async def _publish_project(pid, owner, event):
        published.append((pid, owner, event))

    hub.publish_project = _publish_project

    with (
        patch.object(
            process_role, "_PROCESS_ROLE", process_role.AGENTCORE_RUNTIME_ROLE
        ),
        patch("app.main.get_project_manager", return_value=pm),
        patch("app.services.ws_hub.get_ws_hub", return_value=hub),
    ):
        jobs_mod._publish_project_event(
            "proj-1", {"resource": "eval_job", "type": "started"}
        )
        # Drain the fire-and-forget publish task.
        import asyncio
        for _ in range(20):
            await asyncio.sleep(0)

    assert published, "the publish must still be attempted, not suppressed"


async def test_poller_and_in_process_eval_gates_still_hold():
    """The poller and in-process eval microVM gates must remain wired.

    Pinned here alongside the other guards so the execution-home protections are
    inspectable as one set rather than scattered across three suites.
    """
    app_dir = Path(process_role.__file__).resolve().parent.parent

    republisher = (app_dir / "services" / "job_progress_republisher.py").read_text()
    assert republisher.count("is_agentcore_runtime()") >= 2, (
        "both the poller coroutine and its wrapper must gate on process role"
    )

    dispatch = (app_dir / "services" / "mcp_tools" / "dispatch.py").read_text()
    assert "is_agentcore_runtime()" in dispatch, (
        "dispatch_eval's in-process path must refuse to run in the microVM "
        " — the harness would execute inside the session"
    )

    loop = (app_dir / "services" / "optimization_loop_orchestrator.py").read_text()
    assert "is_agentcore_runtime()" in loop, (
        "the optimization loop must gate on process role"
    )
