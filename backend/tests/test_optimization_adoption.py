# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch-to-owner seam: coordinator adoption of orphaned loops.

Covers the failure class the seam prevents: without it, ``dispatch_optimization``
executing in the AgentCore microVM writes the running record and claims the
eval lock, then the spawned loop refuses via the execution-home guard and
returns BEFORE its release tail — chat reports "dispatched", the toast
reports failure, and the project wedges (running record, no loop, leaked
lock).

The seam: the durable running record IS the dispatch request. The microVM
writes it and does not spawn; the coordinator's adoption sweep observes a
live record with no local loop task and spawns the loop there,
reconstructing config from the facts' owners (record + harness config +
deterministic lock identity).
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.services import optimization_loop_orchestrator as orch


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _StubStore:
    def __init__(
        self,
        live_states: list[dict[str, Any]] | None = None,
        non_terminal_jobs: list[dict[str, Any]] | None = None,
    ):
        # Consumed by the patched _iter_live_optimization_records in the seam
        # fixture. Discovery is dir-walk + point-reads in production (the
        # coordinator role deliberately has no dynamodb:Scan), so the store
        # exposes no listing to stub.
        self.live_states = live_states or []
        self.non_terminal_jobs = non_terminal_jobs or []
        self.released_locks: list[tuple[str, str, str]] = []

    async def list_non_terminal_jobs(self):
        return self.non_terminal_jobs

    async def release_eval_lock(self, owner_id, project_id, job_id):
        self.released_locks.append((owner_id, project_id, job_id))


def _record(project_id: str = "proj-1234-abcd", user_id: str = "user-1") -> dict[str, Any]:
    return {
        "user_id": user_id,
        "project_id": project_id,
        "state": {
            "status": "running",
            "target_metric_scores": {"Answer Correctness": 1.0},
            "max_iterations": 10,
            "plateau_window": 3,
            "iterations": [],
            "current_prompt_version": None,
        },
    }


_HARNESS = SimpleNamespace(
    models=["openai.gpt-5.6-luna"],
    metrics=[{"name": "Answer Correctness", "type": "geval"}],
    data_file="data/cases.csv",
    field_mappings={"input": "q", "expected_output": "a"},
    judge_model_id="anthropic.claude-opus-4-8",
)


@pytest.fixture
def seam(monkeypatch, tmp_path):
    """Patch every owner _adopt_one reads from, at its source module.

    Returns a namespace exposing the store stub and the spawn recorder so
    tests assert on what actually happened rather than on internals.
    """
    store = _StubStore()
    spawned: list[dict[str, Any]] = []
    loop_started = asyncio.Event()

    async def fake_loop(iteration_config: dict) -> None:
        spawned.append(iteration_config)
        loop_started.set()
        # Stay alive briefly so a second sweep sees the named task as live.
        await asyncio.sleep(30)

    monkeypatch.setattr(
        "app.platform.get_platform", lambda: SimpleNamespace(metadata_store=store)
    )

    async def fake_discovery():
        return store.live_states

    monkeypatch.setattr(orch, "_iter_live_optimization_records", fake_discovery)
    monkeypatch.setattr(orch, "run_optimization_loop", fake_loop)
    monkeypatch.setattr(
        "app.services.project_paths.resolve_project_dir",
        lambda pid, base_dir=None: tmp_path,
    )
    monkeypatch.setattr(
        "app.services.harness_config.load_harness_config", lambda d: _HARNESS
    )

    async def fake_arch_split(pid):
        return True

    monkeypatch.setattr(
        "app.platform.arch_split.resolve_arch_split", fake_arch_split
    )
    monkeypatch.setattr(
        "app.services.optimization_orchestrator.detect_latest_prompt",
        lambda d: "baseline",
    )
    return SimpleNamespace(store=store, spawned=spawned, loop_started=loop_started)


async def _cancel_adopted_tasks():
    """Reap the fake loop tasks a test spawned so nothing leaks across tests."""
    for t in asyncio.all_tasks():
        if t.get_name().startswith("optimization-"):
            t.cancel()
    await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Adoption
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adopts_running_record_with_no_local_task(seam):
    seam.store.live_states = [_record()]

    adopted = await orch.adopt_orphan_optimization_loops()

    try:
        assert adopted == 1
        await asyncio.wait_for(seam.loop_started.wait(), timeout=2)
        cfg = seam.spawned[0]
        # Config reconstructed from the owners, not from a stashed copy.
        assert cfg["target_metric_scores"] == {"Answer Correctness": 1.0}
        assert cfg["model_ids"] == ["openai.gpt-5.6-luna"]
        assert cfg["prompt_version"] == "baseline"
        assert cfg["eval_lock_holder_id"] == "optimization-proj-1234-abcd"
        assert cfg["eval_lock_owner_id"] == "user-1"
        assert cfg["max_test_cases"] is None  # per-dispatch ephemera, not durable
    finally:
        await _cancel_adopted_tasks()


@pytest.mark.asyncio
async def test_does_not_double_adopt_when_task_already_running(seam):
    """MUTATION DISCRIMINATOR: removing the local-task name check makes the
    second sweep spawn a second loop for the same record — two loops, one
    project, exactly the deploy-overlap hazard the check bounds."""
    seam.store.live_states = [_record()]

    first = await orch.adopt_orphan_optimization_loops()
    await asyncio.wait_for(seam.loop_started.wait(), timeout=2)
    second = await orch.adopt_orphan_optimization_loops()

    try:
        assert first == 1
        assert second == 0
        assert len(seam.spawned) == 1
    finally:
        await _cancel_adopted_tasks()


@pytest.mark.asyncio
async def test_defers_while_optimization_child_job_is_live(seam):
    """A live child eval means the previous loop's iteration may still be
    running in a detached worker; adopting now would double-run it."""
    rec = _record()
    seam.store.live_states = [rec]
    seam.store.non_terminal_jobs = [{
        "job_id": "j1",
        "project_id": rec["project_id"],
        "type": "eval",
        "state": "running",
        "config": {"source": "optimization"},
    }]

    adopted = await orch.adopt_orphan_optimization_loops()

    assert adopted == 0
    assert seam.spawned == []


@pytest.mark.asyncio
async def test_manual_eval_job_does_not_block_adoption(seam):
    """Only optimization-owned children defer adoption. A manual eval on the
    same project is not the loop's iteration."""
    rec = _record()
    seam.store.live_states = [rec]
    seam.store.non_terminal_jobs = [{
        "job_id": "j2",
        "project_id": rec["project_id"],
        "type": "eval",
        "state": "running",
        "config": {"source": "manual"},
    }]

    adopted = await orch.adopt_orphan_optimization_loops()

    try:
        assert adopted == 1
    finally:
        await _cancel_adopted_tasks()


@pytest.mark.asyncio
async def test_microvm_process_never_adopts(seam, monkeypatch):
    seam.store.live_states = [_record()]
    monkeypatch.setattr(
        "app.platform.process_role._PROCESS_ROLE", "agentcore_runtime"
    )

    adopted = await orch.adopt_orphan_optimization_loops()

    assert adopted == 0
    assert seam.spawned == []


@pytest.mark.asyncio
async def test_unadoptable_record_is_failed_and_lock_released(seam, monkeypatch):
    """No harness config -> the record can never be served. It must be flipped
    to failed (not re-logged every sweep) and the lock the dispatch claimed
    must be released — the loop that would have released it will never run."""
    rec = _record()
    seam.store.live_states = [rec]
    monkeypatch.setattr(
        "app.services.harness_config.load_harness_config", lambda d: None
    )

    written: list[str] = []

    class _Handle:
        def __init__(self):
            self.state = SimpleNamespace(status="running", error=None)
            self.arch_split = True
            self.owner_id = rec["user_id"]

        def is_live(self):
            return self.state.status in ("running", "stopping")

    handle = _Handle()

    async def fake_read(pid, pdir, *, arch_split=None, owner_id=None):
        return handle

    def fake_write(h):
        written.append(h.state.status)
        return h

    monkeypatch.setattr(
        "app.services.optimization_state_repository.read", fake_read
    )
    monkeypatch.setattr(
        "app.services.optimization_state_repository.write", fake_write
    )
    published: list[dict] = []
    monkeypatch.setattr(orch, "_publish_project_event", lambda pid, ev: published.append(ev))

    adopted = await orch.adopt_orphan_optimization_loops()

    assert adopted == 0
    assert written == ["failed"]
    assert handle.state.error is not None
    assert seam.store.released_locks == [
        (rec["user_id"], rec["project_id"], f"optimization-{rec['project_id']}")
    ]
    assert published and published[0]["code"] == "optimization.unadoptable"


@pytest.mark.asyncio
async def test_one_bad_record_does_not_block_the_rest(seam, monkeypatch):
    bad = _record(project_id="bad-project-id")
    good = _record(project_id="good-project-1")
    seam.store.live_states = [bad, good]

    real_resolve = lambda pid, base_dir=None: Path("/tmp/does-not-matter")  # noqa: E731

    def resolving(pid, base_dir=None):
        if pid == "bad-project-id":
            raise RuntimeError("boom")
        return real_resolve(pid)

    monkeypatch.setattr("app.services.project_paths.resolve_project_dir", resolving)
    # The good record needs an existing dir for the harness read.
    monkeypatch.setattr(
        "app.services.harness_config.load_harness_config", lambda d: _HARNESS
    )
    monkeypatch.setattr(Path, "exists", lambda self: True)

    adopted = await orch.adopt_orphan_optimization_loops()

    try:
        assert adopted == 1
        await asyncio.wait_for(seam.loop_started.wait(), timeout=2)
        assert seam.spawned[0]["project_id"] == "good-project-1"
    finally:
        await _cancel_adopted_tasks()


# ---------------------------------------------------------------------------
# The shared task-name definition
# ---------------------------------------------------------------------------

def test_loop_task_name_is_the_single_definition():
    """dispatch spawns with it, adoption dedupes on it, cancel scans it. If
    this shape changes, all three must move together — which they do, because
    all three import this function."""
    assert orch._loop_task_name("abcdefgh-rest-of-uuid") == "optimization-abcdefgh"


# ---------------------------------------------------------------------------
# Discovery (dir-walk + point reads — the no-Scan path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_discovery_walks_both_layouts_and_point_reads(monkeypatch, tmp_path):
    """Discovery must parse the owner from the cloud path layout, fall back to
    the dev user for flat dirs, point-read via the store (never a listing),
    and keep only live records. Coverage matters here because the seam
    fixture patches discovery out — without this test the owner-from-path
    parsing has none."""
    # Cloud layout: users/{owner}/projects/{pid}; dev layout: flat {pid}.
    cloud = tmp_path / "users" / "owner-abc" / "projects" / "proj-cloud"
    cloud.mkdir(parents=True)
    (cloud / "config.json").write_text("{}")
    dev = tmp_path / "proj-dev"
    dev.mkdir()
    (dev / "config.json").write_text("{}")
    terminal = tmp_path / "users" / "owner-abc" / "projects" / "proj-done"
    terminal.mkdir(parents=True)
    (terminal / "config.json").write_text("{}")

    monkeypatch.setattr("app.main.PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(
        "app.platform.auth.get_local_dev_user_id", lambda: "dev-user"
    )

    reads: list[tuple[str, str]] = []

    class _PointReadStore:
        async def get_optimization_state(self, user_id, project_id):
            reads.append((user_id, project_id))
            if project_id == "proj-cloud":
                return {"status": "running", "iterations": []}
            if project_id == "proj-dev":
                return {"status": "stopping", "iterations": []}
            return {"status": "cancelled", "iterations": []}

    monkeypatch.setattr(
        "app.platform.get_platform",
        lambda: SimpleNamespace(metadata_store=_PointReadStore()),
    )

    records = await orch._iter_live_optimization_records()

    by_pid = {r["project_id"]: r for r in records}
    assert set(by_pid) == {"proj-cloud", "proj-dev"}  # terminal filtered out
    assert by_pid["proj-cloud"]["user_id"] == "owner-abc"  # owner from path
    assert by_pid["proj-dev"]["user_id"] == "dev-user"  # flat-layout fallback
    assert ("owner-abc", "proj-done") in reads  # read then filtered, not skipped
