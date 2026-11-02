# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which ``output/*.json`` files count as eval runs.

The regression these cover: ``output/`` holds the metric-audit sidecar beside
each run file, and ``.metric-audit.json`` sorts AFTER ``.json`` on a name sort
(``'j' < 'm'``) while also carrying a LATER mtime. So both a name-ordered and an
mtime-ordered "pick the latest" landed on the sidecar, which has no
``model_runs`` — the project list reported zero models and no pass rate for
projects with several finished runs.

Each test here fails if the filter is removed, so they pin the behaviour rather
than the implementation.
"""
import json
import os

import pytest

from app.services.eval_run_files import eval_run_files, is_eval_run_file
from app.services.metric_disagreement import SIDECAR_SUFFIX
from pathlib import Path


def _write_run(output_dir: Path, name: str) -> Path:
    """Write a minimal but realistically-shaped eval-results file."""
    path = output_dir / name
    path.write_text(
        json.dumps(
            [{"model_id": "m", "results": [{"success": True}, {"success": False}]}]
        )
    )
    return path


def _write_sidecar(output_dir: Path, run_name: str) -> Path:
    """Write a metric-audit sidecar shaped like the real one for ``run_name``."""
    path = output_dir / f"{Path(run_name).stem}{SIDECAR_SUFFIX}"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "generated_at": "2026-09-08T15:05:21Z",
                "gating_metrics": [],
                "status": "ok",
                "metrics": [],
                "pairs": [],
                "worst_verdict": "ok",
                "run_file": run_name,
            }
        )
    )
    return path


class TestIsEvalRunFile:
    def test_a_plain_run_file_is_a_run(self):
        assert is_eval_run_file(Path("eval_results_v1_20260908_144154.json"))

    def test_a_harness_named_result_file_is_a_run(self):
        """The filter is a denylist so harness-chosen names still count.

        The platform writes ``eval_results_*`` but a generated harness may write
        ``results.json``; an allowlist on the platform prefix would drop it.
        """
        assert is_eval_run_file(Path("results.json"))
        assert is_eval_run_file(Path("results_2026.json"))

    def test_the_metric_audit_sidecar_is_not_a_run(self):
        assert not is_eval_run_file(
            Path(f"eval_results_v1_20260908_144154{SIDECAR_SUFFIX}")
        )

    def test_the_active_eval_marker_is_not_a_run(self):
        assert not is_eval_run_file(Path("active_eval.json"))

    def test_a_report_sidecar_is_not_a_run(self):
        assert not is_eval_run_file(Path("eval_results_v1.report.json"))


class TestEvalRunFiles:
    def test_a_missing_output_dir_is_not_an_error(self, tmp_path):
        assert eval_run_files(tmp_path / "nope") == []

    def test_an_empty_output_dir_yields_nothing(self, tmp_path):
        (tmp_path / "output").mkdir()
        assert eval_run_files(tmp_path / "output") == []

    def test_the_sidecar_is_excluded(self, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        _write_run(out, "eval_results_v1_20260908_144154.json")
        _write_sidecar(out, "eval_results_v1_20260908_144154.json")

        names = [p.name for p in eval_run_files(out)]
        assert names == ["eval_results_v1_20260908_144154.json"]

    def test_the_last_element_is_the_newest_run_not_its_sidecar(self, tmp_path):
        """The exact regression: ``[-1]`` used to be a sidecar.

        ``.metric-audit.json`` > ``.json`` on a name sort, so the newest run's
        own sidecar displaced it as the last element.
        """
        out = tmp_path / "output"
        out.mkdir()
        for label in ("baseline_20260908_144154", "v1_20260908_145519", "v2_20260908_150521"):
            name = f"eval_results_{label}.json"
            _write_run(out, name)
            _write_sidecar(out, name)

        runs = eval_run_files(out)
        assert len(runs) == 3
        assert runs[-1].name == "eval_results_v2_20260908_150521.json"

    def test_the_newest_by_mtime_is_a_run_not_its_sidecar(self, tmp_path):
        """Covers the mtime-ordered caller (failure-targeted synthesis).

        mtimes are set explicitly rather than relying on write order, so the
        ordering cannot depend on filesystem timestamp granularity.
        """
        out = tmp_path / "output"
        out.mkdir()
        run = _write_run(out, "eval_results_v1_20260908_145519.json")
        sidecar = _write_sidecar(out, "eval_results_v1_20260908_145519.json")
        os.utime(run, (1_700_000_000, 1_700_000_000))
        # The sidecar is written just after the run it audits, so it is newer.
        os.utime(sidecar, (1_700_000_060, 1_700_000_060))

        newest = max(eval_run_files(out), key=lambda p: p.stat().st_mtime)
        assert newest.name == "eval_results_v1_20260908_145519.json"

    def test_the_active_eval_marker_is_excluded(self, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        _write_run(out, "eval_results_v1_20260908_144154.json")
        (out / "active_eval.json").write_text(
            json.dumps({"pid": 123, "started_at": "2026-09-08T15:00:00Z", "job_id": "j"})
        )

        names = [p.name for p in eval_run_files(out)]
        assert names == ["eval_results_v1_20260908_144154.json"]

    def test_directories_are_not_returned(self, tmp_path):
        """``output/`` holds ``reports/``, ``optimization/`` and ``_derived/``."""
        out = tmp_path / "output"
        out.mkdir()
        (out / "reports").mkdir()
        (out / "optimization").mkdir()
        # A directory that happens to end in .json must not be picked up either.
        (out / "odd.json").mkdir()
        _write_run(out, "eval_results_v1_20260908_144154.json")

        names = [p.name for p in eval_run_files(out)]
        assert names == ["eval_results_v1_20260908_144154.json"]

    def test_results_are_name_sorted_ascending(self, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        for name in ("eval_results_v2.json", "eval_results_baseline.json", "eval_results_v1.json"):
            _write_run(out, name)

        names = [p.name for p in eval_run_files(out)]
        assert names == [
            "eval_results_baseline.json",
            "eval_results_v1.json",
            "eval_results_v2.json",
        ]

    @pytest.mark.parametrize("excluded", ["active_eval.json", f"x{SIDECAR_SUFFIX}", "x.report.json"])
    def test_an_output_dir_holding_only_non_runs_yields_nothing(self, tmp_path, excluded):
        out = tmp_path / "output"
        out.mkdir()
        (out / excluded).write_text("{}")
        assert eval_run_files(out) == []
