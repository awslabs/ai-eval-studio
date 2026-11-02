# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dev startup recovery hooks ``main._recover_orphaned_evals``
and ``main._fail_stale_dev_job_records``.

Stale *optimization loop* records are no longer flipped to cancelled at dev
startup: dev runs the same ``adopt_orphan_optimization_loops`` sweep as
cloud, covered by test_optimization_adoption.py. The corpse-failer tested
here exists to unwedge that sweep — adoption defers while it sees
live-looking optimization-child jobs, and in dev any non-terminal job record
at startup is definitionally a corpse (dev jobs are in-process).

``_recover_orphaned_evals`` is covered by ``TestOrphanEvalKill``:
orphan-subprocess killing is decoupled from optimization status — it reads
the per-project active-eval record and kills manual evals too, not just
optimization-driven ones. These tests invoke it directly with
``PROJECTS_DIR`` patched to a tmp_path.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.main as main_module
from app.main import _fail_stale_dev_job_records
from app.services.optimization_state import OptimizationState


def _make_dev_project(projects_dir: Path, project_id: str, status: str) -> Path:
    """Scaffold a dev-mode project at ``{projects_dir}/{pid}/`` with a state file."""
    project_dir = projects_dir / project_id
    project_dir.mkdir(parents=True)
    (project_dir / "config.json").write_text(json.dumps({
        "id": project_id,
        "name": "test",
        "description": "",
        "created_at": "2026-01-01T00:00:00Z",
        "owner_id": "local-dev",
    }))
    state = OptimizationState.create(
        project_dir=project_dir,
        target_metric_scores={"ExactMatch": 0.9},
    )
    state.status = status
    state.save()
    return project_dir


def _make_cloud_project(
    projects_dir: Path, owner_sub: str, project_id: str, status: str,
) -> Path:
    """Scaffold a cloud-mode project at ``users/{sub}/projects/{pid}/``."""
    project_dir = projects_dir / "users" / owner_sub / "projects" / project_id
    project_dir.mkdir(parents=True)
    (project_dir / "config.json").write_text(json.dumps({
        "id": project_id,
        "name": "test",
        "description": "",
        "created_at": "2026-01-01T00:00:00Z",
        "owner_id": owner_sub,
    }))
    state = OptimizationState.create(
        project_dir=project_dir,
        target_metric_scores={"ExactMatch": 0.9},
    )
    state.status = status
    state.save()
    return project_dir


@pytest.fixture
def projects_root(tmp_path, monkeypatch):
    """Patch ``main.PROJECTS_DIR`` for the duration of the test.

    The recovery function reads the module-global; monkeypatching the
    attribute is the cheapest way to scope each test to its own dir.
    """
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setattr(main_module, "PROJECTS_DIR", root)
    return root


def _run(coro):
    """Run an async coroutine in a fresh event loop for sync test bodies."""
    return asyncio.new_event_loop().run_until_complete(coro)


class TestOrphanEvalKill:
    """``_recover_orphaned_evals`` must SIGKILL orphan harness subprocesses.

    The failure: an eval's harness subprocess keeps running on Bedrock
    after a coordinator restart, because ``JobTracker._jobs`` is in-memory
    and the new coordinator has no PID handle. The on-disk active-eval record (``output/active_eval.json``,
    written for EVERY eval — manual and optimization) is the only handle
    that survives, and recovery scans it for every project.

    These tests target the dedicated eval-recovery pass, which is
    decoupled from optimization status: an orphan manual eval (no
    OptimizationState at all) is now killed too.
    """

    def _write_active_eval(
        self, projects_root: Path, project_id: str, *, pid: int,
    ) -> Path:
        """Scaffold a project with an on-disk active-eval record but NO
        OptimizationState — proving the kill path no longer depends on
        optimization state existing (the manual-eval orphan case)."""
        from app.services.active_eval_state import record_active_eval
        project_dir = projects_root / project_id
        project_dir.mkdir(parents=True)
        (project_dir / "config.json").write_text(json.dumps({
            "id": project_id,
            "name": "test",
            "description": "",
            "created_at": "2026-01-01T00:00:00Z",
            "owner_id": "local-dev",
        }))
        record_active_eval(project_dir, pid, job_id="job-abcd1234")
        return project_dir

    def _record_exists(self, project_dir: Path) -> bool:
        return (project_dir / "output" / "active_eval.json").exists()

    def test_recovery_sigkills_orphan_pid(self, projects_root):
        """Live PID + run_eval.py in cmdline → SIGKILL the process group."""
        from app.main import _recover_orphaned_evals
        project_dir = self._write_active_eval(
            projects_root, "p-orphan-live", pid=12345,
        )

        # ``os.kill(pid, 0)`` returns successfully (process "exists").
        # ``/proc/12345/cmdline`` exists and references run_eval.py.
        # ``os.getpgid`` returns the same pid. ``os.killpg`` is the
        # actual side-effect we assert on.
        with patch("app.main.os.kill") as mock_probe, \
             patch("app.main.os.getpgid", return_value=12345), \
             patch("app.main.os.killpg") as mock_killpg, \
             patch("app.main.Path") as mock_path:
            # Make Path("/proc/12345/cmdline").exists() True and
            # read_bytes() return a cmdline containing run_eval.py.
            # Other Path() calls in main fall through to real Path.
            from pathlib import Path as RealPath
            def _path_factory(arg):
                if str(arg).startswith("/proc/"):
                    fake = MagicMock()
                    fake.exists.return_value = True
                    fake.read_bytes.return_value = (
                        b"uv\x00run\x00python\x00run_eval.py\x00"
                    )
                    return fake
                return RealPath(arg)
            mock_path.side_effect = _path_factory

            _run(_recover_orphaned_evals())

        # Liveness probe was called.
        mock_probe.assert_called_once_with(12345, 0)
        # SIGKILL fired against the process group.
        assert mock_killpg.called
        kill_args = mock_killpg.call_args.args
        assert kill_args[0] == 12345  # pgid
        # SIGKILL is signal 9.
        assert kill_args[1] == 9
        # Record cleared so the next restart doesn't re-probe a stale PID.
        assert not self._record_exists(project_dir)

    def test_recovery_skips_kill_when_pid_already_dead(self, projects_root):
        """``os.kill(pid, 0)`` raises ProcessLookupError → no killpg."""
        from app.main import _recover_orphaned_evals
        project_dir = self._write_active_eval(
            projects_root, "p-orphan-dead", pid=99999,
        )

        with patch(
            "app.main.os.kill", side_effect=ProcessLookupError(),
        ), patch("app.main.os.killpg") as mock_killpg:
            _run(_recover_orphaned_evals())

        mock_killpg.assert_not_called()
        # Record still cleared (process is gone either way).
        assert not self._record_exists(project_dir)

    def test_recovery_skips_kill_when_cmdline_doesnt_match(self, projects_root):
        """Live PID but cmdline doesn't reference run_eval.py — likely the
        OS recycled the PID for an unrelated process. Don't SIGKILL
        someone else's worker."""
        from app.main import _recover_orphaned_evals
        project_dir = self._write_active_eval(
            projects_root, "p-recycled", pid=4242,
        )

        with patch("app.main.os.kill") as mock_probe, \
             patch("app.main.os.killpg") as mock_killpg, \
             patch("app.main.Path") as mock_path:
            from pathlib import Path as RealPath
            def _path_factory(arg):
                if str(arg).startswith("/proc/"):
                    fake = MagicMock()
                    fake.exists.return_value = True
                    # Unrelated cmdline — sshd, postgres, anything.
                    fake.read_bytes.return_value = b"/usr/sbin/sshd"
                    return fake
                return RealPath(arg)
            mock_path.side_effect = _path_factory

            _run(_recover_orphaned_evals())

        # Liveness probe ran but kill skipped.
        mock_probe.assert_called_once_with(4242, 0)
        mock_killpg.assert_not_called()
        # Record still cleared.
        assert not self._record_exists(project_dir)

    def test_recovery_no_record_no_kill_attempt(self, projects_root):
        """A project with no active-eval record never probes /proc."""
        from app.main import _recover_orphaned_evals
        # Dev project with optimization state but NO active-eval record.
        _make_dev_project(projects_root, "p-no-record", "cancelled")

        with patch("app.main.os.kill") as mock_probe, \
             patch("app.main.os.killpg") as mock_killpg:
            _run(_recover_orphaned_evals())

        mock_probe.assert_not_called()
        mock_killpg.assert_not_called()

    def test_recovery_kills_manual_eval_orphan_without_optimization_state(
        self, projects_root,
    ):
        """The core fix: a MANUAL eval (no OptimizationState) whose
        subprocess outlived a restart is now killed. Previously the PID
        lived inside OptimizationState gated on source=='optimization',
        so manual evals leaked an unkillable orphan."""
        from app.main import _recover_orphaned_evals
        project_dir = self._write_active_eval(
            projects_root, "p-manual-orphan", pid=5555,
        )
        # No OptimizationState exists for this project.
        assert OptimizationState.load(project_dir) is None

        with patch("app.main.os.kill"), \
             patch("app.main.os.getpgid", return_value=5555), \
             patch("app.main.os.killpg") as mock_killpg, \
             patch("app.main.Path") as mock_path:
            from pathlib import Path as RealPath
            def _path_factory(arg):
                if str(arg).startswith("/proc/"):
                    fake = MagicMock()
                    fake.exists.return_value = True
                    fake.read_bytes.return_value = b"uv\x00run\x00run_eval.py\x00"
                    return fake
                return RealPath(arg)
            mock_path.side_effect = _path_factory

            _run(_recover_orphaned_evals())

        assert mock_killpg.called
        assert mock_killpg.call_args.args[0] == 5555
        assert not self._record_exists(project_dir)


class TestActiveEvalRecord:
    """The active-eval record round-trips and is the single home for the
    running-eval PID (no longer carried by OptimizationState)."""

    def test_record_round_trips(self, projects_root):
        from app.services.active_eval_state import (
            load_active_eval, record_active_eval,
        )
        project_dir = projects_root / "p-rec-rt"
        project_dir.mkdir()
        record_active_eval(project_dir, pid=7777, job_id="job-deadbeef")

        loaded = load_active_eval(project_dir)
        assert loaded is not None
        assert loaded.pid == 7777
        assert loaded.job_id == "job-deadbeef"
        assert loaded.started_at  # stamped, non-empty

    def test_load_absent_is_none(self, projects_root):
        from app.services.active_eval_state import load_active_eval
        project_dir = projects_root / "p-rec-absent"
        project_dir.mkdir()
        assert load_active_eval(project_dir) is None

    def test_clear_removes_record(self, projects_root):
        from app.services.active_eval_state import (
            clear_active_eval, load_active_eval, record_active_eval,
        )
        project_dir = projects_root / "p-rec-clear"
        project_dir.mkdir()
        record_active_eval(project_dir, pid=8888, job_id="job-1")
        assert load_active_eval(project_dir) is not None

        clear_active_eval(project_dir)
        assert load_active_eval(project_dir) is None
        # Idempotent — clearing an absent record is a no-op.
        clear_active_eval(project_dir)

    def test_malformed_record_loads_as_none(self, projects_root):
        from app.services.active_eval_state import load_active_eval
        project_dir = projects_root / "p-rec-bad"
        (project_dir / "output").mkdir(parents=True)
        (project_dir / "output" / "active_eval.json").write_text("{not json")
        assert load_active_eval(project_dir) is None

    def test_optimization_state_no_longer_carries_pid(self, projects_root):
        """The PID moved out of OptimizationState into active_eval_state —
        guard against a regression that re-adds the coupling."""
        project_dir = projects_root / "p-no-pid-field"
        project_dir.mkdir()
        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.9},
        )
        assert not hasattr(state, "active_eval_pid")
        assert not hasattr(state, "active_eval_started_at")
        # And it's gone from the serialized form.
        loaded = OptimizationState.load(project_dir)
        assert "active_eval_pid" not in loaded.__dict__


class TestDurableOrphanHandling:
    """The flip-to-cancelled durable recovery is deleted — the
    adoption seam owns orphaned live records now, and it RUNS them instead of
    cancelling them (test_optimization_adoption.py covers spawn, dedupe,
    child-job defer, unadoptable flip-to-failed + lock release, and the
    owner-from-path / terminal-filter discovery semantics the old class
    pinned here).

    What this class still owns: the sweep must survive a store that errors
    on one project without losing the rest — the old read-failure pin, now
    against the discovery walk.
    """

    def test_store_read_failure_skips_project_not_sweep(
        self, projects_root, monkeypatch
    ):
        from types import SimpleNamespace

        from app.services import optimization_loop_orchestrator as orch

        _make_cloud_project(projects_root, "owner-sub-1", "p-poisoned", "running")
        _make_cloud_project(projects_root, "owner-sub-1", "p-healthy", "running")
        monkeypatch.setattr("app.main.PROJECTS_DIR", projects_root)

        class _Store:
            async def get_optimization_state(self, user_id, project_id):
                if project_id == "p-poisoned":
                    raise RuntimeError("store exploded for this item")
                return {"status": "running", "iterations": []}

        monkeypatch.setattr(
            "app.platform.get_platform",
            lambda: SimpleNamespace(metadata_store=_Store()),
        )

        records = _run(orch._iter_live_optimization_records())

        assert [r["project_id"] for r in records] == ["p-healthy"], (
            "One unreadable record must be skipped, not abort the sweep"
        )


class TestFailStaleDevJobRecords:
    """``_fail_stale_dev_job_records`` — the corpse-failer.

    Dev jobs are in-process, so any non-terminal job record at startup
    references a process that just died. Flipping them to failed is what
    lets the adoption sweep proceed: adoption defers while it sees
    live-looking optimization-child jobs, and dev runs no reconciler to
    flip them later.
    """

    @staticmethod
    def _store_with(jobs, update_side_effect=None):
        from types import SimpleNamespace
        store = SimpleNamespace(
            list_non_terminal_jobs=AsyncMock(return_value=jobs),
            update_job_state=AsyncMock(side_effect=update_side_effect),
        )
        return SimpleNamespace(metadata_store=store), store

    def test_non_terminal_jobs_flipped_to_failed(self, monkeypatch):
        jobs = [
            {"owner_id": "u1", "project_id": "p1", "job_id": "j1", "state": "running"},
            {"owner_id": "u2", "project_id": "p2", "job_id": "j2", "state": "queued"},
        ]
        platform, store = self._store_with(jobs)
        monkeypatch.setattr("app.platform.get_platform", lambda: platform)

        _run(_fail_stale_dev_job_records())

        assert store.update_job_state.await_count == 2
        store.update_job_state.assert_any_await("u1", "p1", "j1", state="failed")
        store.update_job_state.assert_any_await("u2", "p2", "j2", state="failed")

    def test_no_stale_jobs_no_updates(self, monkeypatch):
        platform, store = self._store_with([])
        monkeypatch.setattr("app.platform.get_platform", lambda: platform)

        _run(_fail_stale_dev_job_records())

        store.update_job_state.assert_not_awaited()

    def test_list_failure_never_raises(self, monkeypatch):
        from types import SimpleNamespace
        store = SimpleNamespace(
            list_non_terminal_jobs=AsyncMock(side_effect=RuntimeError("store down")),
            update_job_state=AsyncMock(),
        )
        platform = SimpleNamespace(metadata_store=store)
        monkeypatch.setattr("app.platform.get_platform", lambda: platform)

        # Must not raise — startup must never be blocked.
        _run(_fail_stale_dev_job_records())
        store.update_job_state.assert_not_awaited()

    def test_one_update_failure_doesnt_stop_others(self, monkeypatch):
        jobs = [
            {"owner_id": "u1", "project_id": "p1", "job_id": "j-bad", "state": "running"},
            {"owner_id": "u1", "project_id": "p1", "job_id": "j-good", "state": "running"},
        ]

        async def _update(user_id, project_id, job_id, *, state):
            if job_id == "j-bad":
                raise RuntimeError("update exploded")

        platform, store = self._store_with(jobs, update_side_effect=_update)
        monkeypatch.setattr("app.platform.get_platform", lambda: platform)

        _run(_fail_stale_dev_job_records())

        # Both attempted; the failure was logged and skipped.
        assert store.update_job_state.await_count == 2
        store.update_job_state.assert_any_await("u1", "p1", "j-good", state="failed")
