# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from app.services.project_manager import ProjectManager
from app.models.schemas import ProjectCreate


class TestProjectManager:
    def test_create_project(self, projects_dir):
        pm = ProjectManager(projects_dir)
        config = pm.create_project(ProjectCreate(name="Test Eval", description="A test"))

        assert config.name == "Test Eval"
        assert config.description == "A test"
        assert (projects_dir / config.id / "config.json").exists()
        assert (projects_dir / config.id / "prompts").exists()
        # ``output/`` (with its dead ``reports/`` child) and
        # ``src/reporting`` are no longer pre-created — nothing read or
        # wrote them at creation time. The runner's ``save_results``
        # mkdirs ``output/`` on the first real run.
        assert not (projects_dir / config.id / "output").exists()
        assert not (projects_dir / config.id / "src" / "reporting").exists()

    def test_create_project_does_not_precreate_data_subdirs(self, projects_dir):
        """``data/`` subfolders are created lazily by the first writer.

        Fresh projects should carry zero empty placeholder directories
        under ``data/`` so the Data tab starts blank.
        """
        pm = ProjectManager(projects_dir)
        config = pm.create_project(ProjectCreate(name="Lazy Dirs"))

        project_dir = projects_dir / config.id
        assert not (project_dir / "data" / "raw").exists()
        assert not (project_dir / "data" / "processed").exists()
        assert not (project_dir / "data" / "synthetic").exists()
        assert not (project_dir / "data" / "uploads").exists()
        # ``data/`` itself must not be created either.
        assert not (project_dir / "data").exists()

    def test_list_projects_empty(self, projects_dir):
        pm = ProjectManager(projects_dir)
        assert pm.list_projects() == []

    def test_list_projects(self, projects_dir):
        pm = ProjectManager(projects_dir)
        pm.create_project(ProjectCreate(name="Project A"))
        pm.create_project(ProjectCreate(name="Project B"))

        projects = pm.list_projects()
        assert len(projects) == 2
        names = {p.name for p in projects}
        assert names == {"Project A", "Project B"}

    def test_list_projects_most_recently_active_first(self, projects_dir):
        """Declared ordering: last_activity desc, created_at fallback.

        Without a declared order the list carries the config-glob's
        directory-name order — UUID hex, effectively random from the user's seat.
        Pin: activity beats recency-of-creation — a fresh eval result file
        moves an OLD project to the top over a NEWER idle one.
        """
        import json as _json

        pm = ProjectManager(projects_dir)
        old = pm.create_project(ProjectCreate(name="Old But Active"))
        # Force distinct created_at ordering regardless of clock resolution.
        cfg_path = projects_dir / old.id / "config.json"
        cfg = _json.loads(cfg_path.read_text())
        cfg["created_at"] = "2026-01-01T00:00:00+00:00"
        cfg_path.write_text(_json.dumps(cfg))
        pm.create_project(ProjectCreate(name="Newer But Idle"))

        # Give the OLD project fresh activity: a result file in output/.
        out = projects_dir / old.id / "output"
        out.mkdir(exist_ok=True)
        (out / "eval_results_20260622_120000.json").write_text("[]")

        ordered = [p.name for p in pm.list_projects()]
        assert ordered == ["Old But Active", "Newer But Idle"], (
            f"most recently ACTIVE project must lead: {ordered}"
        )

    def test_get_project(self, projects_dir):
        pm = ProjectManager(projects_dir)
        created = pm.create_project(ProjectCreate(name="My Eval"))

        fetched = pm.get_project(created.id)
        assert fetched.name == "My Eval"
        assert fetched.id == created.id

    def test_get_project_not_found(self, projects_dir):
        pm = ProjectManager(projects_dir)
        assert pm.get_project("nonexistent-id") is None

    def test_delete_project(self, projects_dir):
        pm = ProjectManager(projects_dir)
        created = pm.create_project(ProjectCreate(name="To Delete"))

        assert pm.delete_project(created.id) is True
        assert pm.get_project(created.id) is None
        assert not (projects_dir / created.id).exists()

    def test_delete_project_not_found(self, projects_dir):
        pm = ProjectManager(projects_dir)
        assert pm.delete_project("nonexistent-id") is False

    def test_update_config(self, projects_dir):
        pm = ProjectManager(projects_dir)
        created = pm.create_project(ProjectCreate(name="Original"))

        created.name = "Updated"
        created.session_id = "sess-123"
        pm.save_config(created)

        fetched = pm.get_project(created.id)
        assert fetched.name == "Updated"
        assert fetched.session_id == "sess-123"


class TestProjectSummaryEvalMetadata:
    """``model_count`` / ``latest_pass_rate`` survive the metric-audit sidecar.

    The list endpoint read the alphabetically-last ``output/*.json`` and treated
    it as eval results. Once runs started shipping a ``.metric-audit.json``
    sidecar the sidecar won that sort — it has no ``model_runs``, so every
    project with a recent run reported zero models and no pass rate, and both
    badges vanished from its card.

    This is the coverage the summary never had: nothing asserted on these fields,
    which is why the regression shipped.
    """

    @staticmethod
    def _write_run(project_dir, name, *, successes, failures):
        """Write one eval-results file with a known pass/fail split."""
        output = project_dir / "output"
        output.mkdir(parents=True, exist_ok=True)
        results = [{"success": True}] * successes + [{"success": False}] * failures
        (output / name).write_text(json.dumps([{"model_id": "m1", "results": results}]))

    @staticmethod
    def _write_sidecar(project_dir, run_name):
        """Write the metric-audit sidecar that shadowed the run in the sort."""
        output = project_dir / "output"
        output.mkdir(parents=True, exist_ok=True)
        stem = Path(run_name).stem
        (output / f"{stem}.metric-audit.json").write_text(
            json.dumps({"schema_version": 2, "status": "ok", "metrics": [], "pairs": []})
        )

    def _summary_for(self, projects_dir, build):
        """Create a project, let ``build`` populate it, return its summary."""
        pm = ProjectManager(projects_dir)
        created = pm.create_project(ProjectCreate(name="Sidecar Project"))
        build(projects_dir / created.id)
        summaries = [p for p in pm.list_projects() if p.id == created.id]
        assert len(summaries) == 1
        return summaries[0]

    def test_models_and_pass_rate_reported_when_a_sidecar_is_present(self, projects_dir):
        run = "eval_results_v2_20260908_150521.json"

        def build(project_dir):
            self._write_run(project_dir, run, successes=3, failures=1)
            self._write_sidecar(project_dir, run)

        summary = self._summary_for(projects_dir, build)
        assert summary.model_count == 1
        assert summary.latest_pass_rate == 0.75

    def test_the_latest_run_wins_not_an_earlier_runs_sidecar(self, projects_dir):
        """Ordering must come from the runs, ignoring interleaved sidecars."""
        older = "eval_results_v1_20260908_145519.json"
        newer = "eval_results_v2_20260908_150521.json"

        def build(project_dir):
            self._write_run(project_dir, older, successes=1, failures=3)
            self._write_sidecar(project_dir, older)
            self._write_run(project_dir, newer, successes=4, failures=0)
            self._write_sidecar(project_dir, newer)

        summary = self._summary_for(projects_dir, build)
        # 1.0 is the newer run; 0.25 would mean the older run was picked.
        assert summary.latest_pass_rate == 1.0

    def test_the_active_eval_marker_is_not_read_as_results(self, projects_dir):
        run = "eval_results_v1_20260908_145519.json"

        def build(project_dir):
            self._write_run(project_dir, run, successes=2, failures=0)
            (project_dir / "output" / "active_eval.json").write_text(
                json.dumps({"pid": 1, "started_at": "2026-09-08T15:00:00Z", "job_id": "j"})
            )

        summary = self._summary_for(projects_dir, build)
        assert summary.model_count == 1
        assert summary.latest_pass_rate == 1.0

    def test_a_project_with_no_runs_reports_no_eval_metadata(self, projects_dir):
        summary = self._summary_for(projects_dir, lambda project_dir: None)
        assert summary.model_count == 0
        assert summary.latest_pass_rate is None

    def test_is_running_is_false_without_an_active_job(self, projects_dir):
        """Liveness comes from the job tracker, not from any file on disk.

        In cloud mode the harness runs in its own task against a snapshot under
        ``/tmp/workspace/{job_id}``, so an on-disk marker never reaches the
        project directory this process can see. A marker present here must not be
        mistaken for liveness.
        """
        run = "eval_results_v1_20260908_145519.json"

        def build(project_dir):
            self._write_run(project_dir, run, successes=1, failures=0)
            (project_dir / "output" / "active_eval.json").write_text(
                json.dumps({"pid": 1, "started_at": "2026-09-08T15:00:00Z", "job_id": "j"})
            )

        assert self._summary_for(projects_dir, build).is_running is False
