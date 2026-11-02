# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The post-eval coordinator message carries the metric-trust verdict.

WHY THIS MATTERS MORE THAN THE UI. The coordinator is what decides whether to
cull a model, rewrite the prompt, or declare convergence, and it decided all of
that from pass rates with no way to know a metric was unreliable — the verdict
reached a log line and a sidecar file, neither of which it reads. A badge in the
report warns the user after the fact; this stops the wrong decision being made.

Two failure modes are pinned here because both are silent:

1. The message must never claim a clean run when the verdict is simply absent.
2. An ``error`` verdict must produce a DIRECTIVE, not a polite note. A findings
   list invites a summary that mentions the problem and proceeds anyway.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.completion_turn import _build_synthetic_message, _trust_text
from app.services.jobs import Job
from app.services.run_trust import RunTrustFacts, TrustReason, format_for_agent

GATE = "Judge"
DIAG = "StrictMatch"


def _case(idx: int, *verdicts: tuple[str, bool]) -> dict:
    return {
        "name": f"c{idx}",
        "success": verdicts[0][1],
        "input": "in",
        "actual_output": "out",
        "expected_output": "out",
        "additional_metadata": {"case_index": idx},
        "metrics_data": [
            {"name": n, "score": 1.0 if ok else 0.0, "success": ok,
             "threshold": 0.5, "reason": "r", "error": None}
            for n, ok in verdicts
        ],
    }


@pytest.fixture
def project(tmp_path: Path, monkeypatch) -> Path:
    """A project dir with a run file, wired so ``_trust_text`` resolves to it."""
    (tmp_path / "output").mkdir()
    monkeypatch.setattr(
        "app.services.project_paths.resolve_project_dir", lambda _pid: tmp_path
    )
    return tmp_path


def _write_run(project: Path, name: str, cases: list[dict]) -> str:
    (project / "output" / name).write_text(
        json.dumps([{"model_id": "m1", "test_run_id": "m1", "results": cases}])
    )
    return name


def _audit_the_run(project: Path, name: str, gating: set[str]) -> None:
    """Run the real audit so the sidecar exists, as the executors now do."""
    from app.services.metric_disagreement import audit_run_file

    audit_run_file(project / "output" / name, gating)


class TestFormatForAgent:
    def test_says_nothing_when_there_is_nothing_to_say(self):
        assert format_for_agent(None) == ""
        assert format_for_agent(RunTrustFacts()) == ""

    def test_never_reports_a_clean_bill_of_health(self):
        """Silence, not reassurance — the check may not have run at all."""
        for trust in (None, RunTrustFacts(), RunTrustFacts(severity="ok")):
            text = format_for_agent(trust)
            assert "no problem" not in text.lower()
            assert "clean" not in text.lower()

    def test_an_error_verdict_forbids_culling_and_convergence(self):
        trust = RunTrustFacts()
        trust.add(TrustReason(
            id="gate_saturated", severity="error", metrics=[GATE],
            headline=f"{GATE} decides pass/fail but scores 99% of cases the same",
            impact="Treat the percentages as unproven.",
        ))
        text = format_for_agent(trust)
        assert "ERROR" in text
        assert "do NOT cull" in text
        assert "declare convergence" in text, (
            "an error verdict must block the convergence claim, not footnote it"
        )

    def test_a_warning_verdict_asks_for_a_caveat_not_a_halt(self):
        trust = RunTrustFacts()
        trust.add(TrustReason(
            id="metric_non_discriminating", severity="warning", metrics=[DIAG],
            headline=f"{DIAG} scores 99% of cases the same",
            impact="It cannot separate a good model from a bad one.",
        ))
        text = format_for_agent(trust)
        assert "WARNING" in text
        assert "do NOT cull" not in text, "over-blocking on a display problem"
        assert "limitation" in text

    def test_every_reason_is_listed_with_its_severity(self):
        trust = RunTrustFacts()
        trust.add(TrustReason(id="a", severity="warning", headline="H1", impact="I1"))
        trust.add(TrustReason(id="b", severity="info", headline="H2", impact="I2"))
        text = format_for_agent(trust)
        assert "[warning] H1. I1" in text
        assert "[info] H2. I2" in text
        assert "2 findings" in text

    def test_the_finding_count_is_pluralised(self):
        """The agent quotes this text back to the user; "1 finding(s)" reads badly."""
        one = RunTrustFacts()
        one.add(TrustReason(id="a", severity="warning", headline="H", impact="I"))
        assert "1 finding)" in format_for_agent(one)
        assert "finding(s)" not in format_for_agent(one)


class TestTrustTextFromDisk:
    def test_reads_the_verdict_for_a_real_run(self, project: Path):
        # Degenerate diagnostic against a discriminating gate.
        cases = [_case(i, (GATE, i % 4 != 0), (DIAG, i >= 2)) for i in range(600)]
        name = _write_run(project, "eval_results_v1.json", cases)
        _audit_the_run(project, name, {GATE})

        text = _trust_text("proj-1234", name)
        assert "METRIC TRUST CHECK" in text
        assert DIAG in text

    def test_accepts_an_absolute_path_as_well_as_a_bare_name(self, project: Path):
        """The two call sites pass different shapes; both must work.

        The optimization branch passes ``run_output_file`` and the eval branch
        passes ``job.output_file`` — one of which has historically been a full
        path. Basename resolution lives in the helper for that reason.
        """
        cases = [_case(i, (GATE, i % 4 != 0), (DIAG, i >= 2)) for i in range(600)]
        name = _write_run(project, "eval_results_v1.json", cases)
        _audit_the_run(project, name, {GATE})

        assert _trust_text("proj-1234", f"/abs/output/{name}") == _trust_text(
            "proj-1234", name
        )

    def test_missing_sidecar_yields_silence_not_an_all_clear(self, project: Path):
        """The audit not having run is not evidence the metrics are fine."""
        cases = [_case(i, (GATE, i % 4 != 0), (DIAG, i >= 2)) for i in range(600)]
        name = _write_run(project, "eval_results_v1.json", cases)
        # deliberately NOT audited
        assert _trust_text("proj-1234", name) == ""

    def test_missing_run_file_does_not_raise(self, project: Path):
        assert _trust_text("proj-1234", "eval_results_nope.json") == ""

    def test_a_broken_resolver_does_not_raise(self, monkeypatch):
        """A finished run must reach the coordinator even if trust can't be read."""
        def boom(_pid):
            raise RuntimeError("no such project")

        monkeypatch.setattr(
            "app.services.project_paths.resolve_project_dir", boom
        )
        assert _trust_text("proj-1234", "eval_results_v1.json") == ""


class TestBothCompletionBranches:
    """Standalone evals and optimization iterations are separate code paths.

    Wiring only one leaves half the runs silent, and the optimization half is the
    one where a bad metric feeds every rewrite decision rather than merely
    misleading a reader.
    """

    def _flagged_run(self, project: Path) -> str:
        cases = [_case(i, (GATE, i % 4 != 0), (DIAG, i >= 2)) for i in range(600)]
        name = _write_run(project, "eval_results_v1.json", cases)
        _audit_the_run(project, name, {GATE})
        return name

    @pytest.mark.asyncio
    async def test_optimization_iteration_message_carries_the_verdict(self, project):
        name = self._flagged_run(project)
        job = Job(job_id="j1", type="optimization_iteration", project_id="proj-1234")
        job.config = {"iteration": 2, "prompt_version": "v2"}
        job.result = {
            "iteration": 2, "prompt_version": "v2",
            "passed_cases": 450, "total_cases": 600,
            "metrics_summary": {}, "per_model_pass_rates": {},
            "run_output_file": name, "convergence": None,
        }
        msg = await _build_synthetic_message(job)
        assert "METRIC TRUST CHECK" in msg
        assert msg.index("METRIC TRUST CHECK") < msg.index("Not converged"), (
            "the verdict must precede the convergence instruction, not trail it"
        )

    @pytest.mark.asyncio
    async def test_standalone_eval_message_carries_the_verdict(self, project):
        name = self._flagged_run(project)
        job = Job(job_id="j2", type="eval", project_id="proj-1234")
        job.config = {"prompt_version": "v1"}
        job.output_file = str(project / "output" / name)
        job.result = json.loads((project / "output" / name).read_text())
        msg = await _build_synthetic_message(job)
        assert "METRIC TRUST CHECK" in msg

    @pytest.mark.asyncio
    async def test_a_clean_run_adds_nothing_to_either_message(self, project):
        """No noise on a healthy run, or the agent learns to skip the section.

        Genuinely clean is a narrow target, which is the detector working as
        designed: the diagnostic must fail EVERY case the gate failed (any single
        case where it passes what the gate rejects is the over-lenient direction,
        and that has no ratio floor), and may fail only a couple more (staying
        under the 5% over-strict floor). Both still discriminate at ~75%.
        """
        cases = [
            _case(i, (GATE, i % 4 != 0), (DIAG, i % 4 != 0 and i not in (1, 5)))
            for i in range(600)
        ]
        name = _write_run(project, "eval_results_clean.json", cases)
        _audit_the_run(project, name, {GATE})

        job = Job(job_id="j3", type="eval", project_id="proj-1234")
        job.config = {"prompt_version": "v1"}
        job.output_file = name
        job.result = json.loads((project / "output" / name).read_text())
        msg = await _build_synthetic_message(job)
        assert "METRIC TRUST CHECK" not in msg
