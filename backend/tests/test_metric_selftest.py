# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The metric self-test: does validation notice a metric that measures nothing?

``validate_harness`` proved a harness RUNS and stopped there. Walk a metric
hardcoded to ``success=False, reason="nope"`` against every pre-existing check:
the score is a real float, the verdict a real bool, the reason non-empty — so it
passed validation, and every pass rate the project ever produced was zero for a
reason nobody was told about.

Two layers are tested here, separately:

* the PROBE (``app.services.metric_selftest_probe``), which runs inside a
  project's venv and only measures. It is exercised directly against fake
  metric classes — no subprocess, no project tree — because what matters is
  whether it can tell a stuck metric from a working one.
* the VERDICT (``mcp_tools.harness._read_selftest_findings``), which decides
  what blocks. Kept apart from the probe on purpose: the probe must never
  decide, and the split is what lets a nondeterministic judge-backed result warn
  while a deterministic one blocks.
"""
from __future__ import annotations

import sys
from typing import Any, List

import pytest

from app.services import metric_selftest_probe as probe
from app.services.mcp_tools.harness import _read_selftest_findings


# ---------------------------------------------------------------------------
# Fake metrics. Each one is a real defect shape, not a mock.
# ---------------------------------------------------------------------------


class _Base:
    """Minimal stand-in for a DeepEval metric: measure + is_successful."""

    threshold = 1.0

    def __init__(self) -> None:
        self.score: float | None = None
        self.success: bool | None = None
        self.reason: str | None = None

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:  # noqa: A003 — mirrors BaseMetric's property
        return type(self).__name__


class WorkingExactMatch(_Base):
    """Strict but correct: exact string equality after a strip."""

    _required_params = ["input", "actual_output", "expected_output"]

    def measure(self, case: Any) -> float:
        ok = (case.actual_output or "").strip() == (case.expected_output or "").strip()
        self.score = 1.0 if ok else 0.0
        self.success = ok
        self.reason = "match" if ok else "differs"
        return self.score


class AlwaysFail(_Base):
    """The sharpest defect shape: a hardcoded failure with well-formed output."""

    _required_params = ["input", "actual_output", "expected_output"]

    def measure(self, case: Any) -> float:  # noqa: ARG002
        self.score = 0.0
        self.success = False
        self.reason = "nope"
        return self.score


class AlwaysPass(_Base):
    """The other direction — certifies every run, including a broken model."""

    _required_params = ["input", "actual_output", "expected_output"]

    def measure(self, case: Any) -> float:  # noqa: ARG002
        self.score = 1.0
        self.success = True
        self.reason = "looks fine"
        return self.score


class RaisesOnScore(_Base):
    """A metric that cannot score a well-formed case at all.

    Counts calls so the retry path can prove it does NOT retry this: the failure
    is in the metric's own code, and retrying would turn a fast clear error into
    a slow one.
    """

    _required_params = ["input", "actual_output", "expected_output"]

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def measure(self, case: Any) -> float:  # noqa: ARG002
        self.calls += 1
        raise RuntimeError("normalizer blew up")


#: The verbatim message DeepEval raises when the judge's text will not parse
#: (``deepeval/metrics/utils.py`` ``trimAndLoadJson``). Hardcoded here rather
#: than imported because the classifier matches on it as a string, so a test
#: that shared a constant with the code would assert nothing.
_INVALID_JSON = (
    "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
)


class FlakyJudge(_Base):
    """Fails the first ``fail_times`` samples the way a flaky judge does, then works.

    The failure mode: the judge's HTTP call succeeds and DeepEval cannot parse the
    text it returned. Raised above the model adapter, so the adapter's own retry
    never sees it.
    """

    _required_params = ["input", "actual_output", "expected_output"]

    def __init__(self, fail_times: int = 1) -> None:
        super().__init__()
        self.fail_times = fail_times
        self.calls = 0

    def measure(self, case: Any) -> float:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ValueError(_INVALID_JSON)
        ok = (case.actual_output or "").strip() == (case.expected_output or "").strip()
        self.score = 1.0 if ok else 0.0
        self.success = ok
        self.reason = "match" if ok else "differs"
        return self.score


class AlwaysInvalidJson(_Base):
    """A judge that never returns parseable output — a real finding, not bad luck."""

    _required_params = ["input", "actual_output", "expected_output"]

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def measure(self, case: Any) -> float:  # noqa: ARG002
        self.calls += 1
        raise ValueError(_INVALID_JSON)


class NoBooleanVerdict(_Base):
    """Returns a score but never a boolean — unknown, not a pass."""

    _required_params = ["input", "actual_output", "expected_output"]

    def measure(self, case: Any) -> float:  # noqa: ARG002
        self.score = 0.5
        self.success = None
        return self.score

    def is_successful(self) -> bool:
        raise NotImplementedError


class ToleratesNothing(_Base):
    """Correct on identity, rejects every formatting variant.

    The common strict-comparison defect: a model wraps its answer in backticks
    and the metric scores a right answer wrong.
    """

    _required_params = ["input", "actual_output", "expected_output"]

    def measure(self, case: Any) -> float:
        ok = case.actual_output == case.expected_output
        self.score = 1.0 if ok else 0.0
        self.success = ok
        self.reason = "byte equality"
        return self.score


class ToleratesFormatting(_Base):
    """Strips the artifacts a model adds, then compares."""

    _required_params = ["input", "actual_output", "expected_output"]

    @staticmethod
    def _norm(text: str) -> str:
        out = (text or "").strip()
        for fence in ("```sql", "```"):
            out = out.replace(fence, "")
        out = out.strip().strip("`").strip('"').strip("*")
        if out.lower().startswith("answer:"):
            out = out[len("answer:"):].strip()
        out = out.rstrip(";.").strip()
        return out.replace("’", "'").lower()

    def measure(self, case: Any) -> float:
        ok = self._norm(case.actual_output) == self._norm(case.expected_output)
        self.score = 1.0 if ok else 0.0
        self.success = ok
        self.reason = "match after normalization"
        return self.score


class JudgeBackedAlwaysFail(_Base):
    """Same defect as ``AlwaysFail``, but shaped like a real ``GEval``.

    Two details copied from DeepEval 3.9.9 rather than invented, because both
    decide whether the probe can read this metric at all:

    * ``_required_params`` on the CLASS is a type ANNOTATION
      (``typing.List[SingleTurnParams]``), not a list — reading only the class
      attribute yields something unusable, which would silently skip every
      GEval;
    * the real parameter list arrives per INSTANCE as ``evaluation_params``.
    """

    _required_params = List[str]  # the annotation shape, exactly as GEval has it

    def __init__(self) -> None:
        super().__init__()
        self.model = object()  # what _is_judge_backed reads
        self.evaluation_params = ["input", "actual_output", "expected_output"]

    def measure(self, case: Any) -> float:  # noqa: ARG002
        self.score = 0.0
        self.success = False
        self.reason = "nope"
        return self.score


class RejectsOnlyNonAnswer(_Base):
    """AnswerRelevancy's real shape: relevance to the input, never correctness.

    Passes anything that looks like an answer to the question — including a
    plausible-but-wrong one — and rejects output that is not an answer. Correct
    behaviour, and it must not be reported as degenerate.
    """

    _required_params = ["input", "actual_output"]

    def __init__(self) -> None:
        super().__init__()
        self.model = object()  # judge-backed, like the real metric

    def measure(self, case: Any) -> float:
        ok = not (case.actual_output or "").startswith("ERROR")
        self.score = 1.0 if ok else 0.0
        self.success = ok
        return self.score


class RejectsOnlyContradiction(_Base):
    """Faithfulness's real shape: no claims means nothing to contradict.

    Passes a non-answer (it asserts nothing, so it is faithful) and rejects an
    output that contradicts the context. Also correct, and also not degenerate.
    """

    _required_params = ["input", "actual_output", "retrieval_context"]

    def __init__(self) -> None:
        super().__init__()
        self.model = object()

    def measure(self, case: Any) -> float:
        actual = case.actual_output or ""
        context = " ".join(case.retrieval_context or [])
        makes_claim = "==" in actual
        contradicts = makes_claim and "'Shipped'" not in actual and "Shipped" in context
        self.score = 0.0 if contradicts else 1.0
        self.success = not contradicts
        return self.score


class ReadsRetrievalContext(_Base):
    """Declares ``retrieval_context`` and records whether it actually arrived.

    Stands in for FaithfulnessMetric / ContextualPrecisionMetric /
    ContextualRecallMetric, which all list ``RETRIEVAL_CONTEXT`` in
    ``_required_params`` on DeepEval 3.9.9.
    """

    _required_params = ["input", "actual_output", "expected_output", "retrieval_context"]

    def __init__(self) -> None:
        super().__init__()
        self.seen_context: list[str] = []

    def measure(self, case: Any) -> float:
        self.seen_context = list(case.retrieval_context or [])
        if not self.seen_context:
            raise RuntimeError("retrieval_context missing — DeepEval would raise here")
        ok = (case.actual_output or "").strip() == (case.expected_output or "").strip()
        self.score = 1.0 if ok else 0.0
        self.success = ok
        return self.score


class NeedsToolTrace(_Base):
    """Requires a field the probe cannot synthesize, so it must be reported unprobed."""

    _required_params = ["input", "actual_output", "tools_called"]

    def measure(self, case: Any) -> float:  # noqa: ARG002
        raise AssertionError("must never be scored — the probe cannot build its inputs")


class NoExpectedOutput(_Base):
    """Reads only input + actual — no equivalence notion to test."""

    _required_params = ["input", "actual_output"]

    def measure(self, case: Any) -> float:
        ok = bool((case.actual_output or "").strip())
        self.score = 1.0 if ok else 0.0
        self.success = ok
        return self.score


#: The probe builds a REAL ``LLMTestCase``. There used to be a hand-rolled stub
#: here that accepted exactly ``input``/``actual_output``/``expected_output``,
#: on the reasoning that a plain record with the same fields was enough and it
#: avoided DeepEval's deprecation churn. That stub is what hid a live defect: the
#: probe never set ``retrieval_context``, so every retrieval metric raised
#: ``MissingTestCaseParamsError`` before its first judge call and the self-test
#: was inert on the mainstream DeepEval shape — invisible here, because a fake
#: that ignores the field it does not model cannot fail on it.

INPUT = "Show orders whose status is Shipped"


class TestDegenerateCheck:
    """The one check that blocks, and the reason it can."""

    def test_a_working_metric_is_ok(self):
        assert probe._degenerate_check(WorkingExactMatch(), INPUT)["verdict"] == "ok"

    def test_always_fail_is_caught(self):
        """THE case validation used to pass: score 0.0, success False, reason set."""
        result = probe._degenerate_check(AlwaysFail(), INPUT)
        assert result["verdict"] == "always_fail"
        assert "identical to the expected output" in result["detail"]

    def test_always_pass_is_caught(self):
        result = probe._degenerate_check(AlwaysPass(), INPUT)
        assert result["verdict"] == "always_pass"
        assert "not an answer at all" in result["detail"]

    def test_a_metric_that_raises_is_unknown_not_ok(self):
        """Swallowing the exception would report a broken metric as probed."""
        result = probe._degenerate_check(RaisesOnScore(), INPUT)
        assert result["verdict"] == "unknown"
        assert "normalizer blew up" in result["detail"]

    def test_no_boolean_verdict_is_unknown(self):
        result = probe._degenerate_check(NoBooleanVerdict(), INPUT)
        assert result["verdict"] == "unknown"

    def test_no_wrong_output_is_a_near_miss(self):
        """None of the probe's own fixtures may be scoreable-as-equal by accident.

        If a "wrong" output were a near miss, a lenient-but-correct metric would
        look degenerate.
        """
        assert probe._WRONG_OUTPUTS
        for label, wrong in probe._WRONG_OUTPUTS:
            assert wrong != probe._REFERENCE, label
            assert probe._REFERENCE not in wrong, label

    def test_rejecting_only_one_wrong_output_is_enough(self):
        """The false positive that one wrong output cannot avoid.

        No single wrong answer is wrong for every metric family, so requiring a
        metric to reject BOTH would flag two correct, extremely common metrics:

        * Faithfulness passes a non-answer — an output making no claims
          contradicts no context, so it is faithful;
        * AnswerRelevancy passes a plausible-but-wrong answer — it is a relevant
          answer to the question, and relevancy never reads expected_output.

        Both distinguish something, so both must come back ``ok``.
        """
        for metric in (RejectsOnlyNonAnswer(), RejectsOnlyContradiction()):
            result = probe._degenerate_check(metric, INPUT)
            assert result["verdict"] == "ok", type(metric).__name__
            assert len(result["rejected"]) == 1, type(metric).__name__

    def test_passing_every_wrong_output_is_still_always_pass(self):
        """Requiring only one rejection must not let a stuck metric through."""
        result = probe._degenerate_check(AlwaysPass(), INPUT)
        assert result["verdict"] == "always_pass"

    def test_a_metric_needing_fields_the_probe_cannot_build_says_so_by_name(self):
        """Better than a generic "could not be probed" — and it must not block.

        A metric requiring ``tools_called`` is not defective; the probe simply
        cannot synthesize a tool trace. Naming the field is what tells a user
        that this metric went unchecked and why.
        """
        result = probe._degenerate_check(NeedsToolTrace(), INPUT)
        assert result["verdict"] == "not_probed"
        assert "tools_called" in result["detail"]

    def test_a_retrieval_metric_is_actually_probed(self):
        """The gap the old hand-rolled LLMTestCase stub hid.

        A metric declaring ``retrieval_context`` used to raise
        ``MissingTestCaseParamsError`` inside DeepEval before its first judge
        call, so every retrieval metric came back "unknown" and the self-test
        was inert on the mainstream DeepEval shape. It must now reach a real
        verdict, and it must actually receive the context.
        """
        metric = ReadsRetrievalContext()
        result = probe._degenerate_check(metric, INPUT)
        assert result["verdict"] == "ok"
        assert metric.seen_context, "the probe never populated retrieval_context"


class TestTransientJudgeRetry:
    """A flaky judge sample must not become the verdict on a metric.

    The defect this covers: when the ONLY metric gating a project comes back
    ``unknown`` because one judge sample returned unparseable text, nothing
    checks whether it measures anything.
    """

    @pytest.fixture(autouse=True)
    def _no_real_sleeping(self, monkeypatch):
        """Keep the backoff out of the suite's wall clock."""
        monkeypatch.setattr(probe.time, "sleep", lambda _seconds: None)

    def test_a_single_bad_judge_sample_no_longer_decides_the_verdict(self):
        """The regression itself: this was ``unknown`` before the retry existed."""
        metric = FlakyJudge(fail_times=1)
        result = probe._degenerate_check(metric, INPUT)
        assert result["verdict"] == "ok"

    @staticmethod
    def _score(metric):
        """One ``_score`` call, which is the unit the attempt budget applies to."""
        return probe._score(
            metric, actual="x", expected="x", user_input=INPUT
        )

    def test_the_retry_is_bounded(self):
        metric = AlwaysInvalidJson()
        result = self._score(metric)
        assert not result["ok"]
        assert metric.calls == probe._SCORE_ATTEMPTS, (
            "one _score should stop at the attempt budget, not retry forever"
        )
        assert result["attempts"] == probe._SCORE_ATTEMPTS

    def test_an_exhausted_retry_does_not_read_like_one_unlucky_sample(self):
        assert "still failing after 3 attempts" in self._score(AlwaysInvalidJson())["error"]

    def test_a_permanently_bad_judge_is_reported_as_unprobed(self):
        assert probe._degenerate_check(AlwaysInvalidJson(), INPUT)["verdict"] == "unknown"

    def test_a_metric_whose_own_code_raises_is_not_retried(self):
        """Retrying a broken normalizer only makes a clear failure slow."""
        metric = RaisesOnScore()
        result = self._score(metric)
        assert not result["ok"]
        assert metric.calls == 1
        assert "still failing after" not in result["error"]

    def test_a_verdict_that_needed_retries_says_so(self):
        """Reported even when clean — the same flakiness corrupts the real run."""
        result = probe._degenerate_check(FlakyJudge(fail_times=1), INPUT)
        assert result["verdict"] == "ok"
        assert result["max_attempts"] == 2

    def test_a_clean_judge_reports_one_attempt(self):
        result = probe._degenerate_check(WorkingExactMatch(), INPUT)
        assert result["max_attempts"] == 1

    @pytest.mark.parametrize(
        "message",
        [
            _INVALID_JSON,
            "ThrottlingException: Rate exceeded",
            "An error occurred (ServiceUnavailableException) when calling Converse",
            "ReadTimeoutError: Read timeout on endpoint URL",
        ],
    )
    def test_transient_shapes_are_classified_as_retryable(self, message):
        assert probe._is_transient_judge_failure(ValueError(message))

    @pytest.mark.parametrize(
        "message",
        [
            "AccessDeniedException: not authorized to invoke",
            "ValidationException: model id is invalid",
            "normalizer blew up",
        ],
    )
    def test_real_errors_are_not_retryable(self, message):
        assert not probe._is_transient_judge_failure(ValueError(message))


class TestFormattingBattery:
    """Equivalence UNDER normalization — the check an identity pair cannot do."""

    def test_a_no_op_normalizer_rejects_every_variant(self):
        """This is why the identity/mismatch pair was the WRONG probe.

        ``ToleratesNothing`` passes the degenerate check cleanly — it is correct
        on identity and correct on garbage. Only formatting-only variants expose
        that it scores right answers wrong.
        """
        assert probe._degenerate_check(ToleratesNothing(), INPUT)["verdict"] == "ok"
        result = probe._formatting_check(ToleratesNothing(), INPUT)
        assert "backticks" in result["rejected"]
        assert "code_fence" in result["rejected"]
        assert result["accepted"] == []

    def test_a_tolerant_normalizer_accepts_the_common_artifacts(self):
        result = probe._formatting_check(ToleratesFormatting(), INPUT)
        for expected_ok in ("backticks", "code_fence", "trailing_semicolon",
                            "surrounding_whitespace", "answer_prefix"):
            assert expected_ok in result["accepted"], (
                f"{expected_ok} rejected: {result['rejected']}"
            )

    def test_variant_errors_are_reported_not_swallowed(self):
        result = probe._formatting_check(RaisesOnScore(), INPUT)
        assert result["rejected"] == [] and result["accepted"] == []
        assert len(result["errors"]) == len(probe._FORMATTING_VARIANTS)

    def test_every_variant_differs_from_the_reference_only_in_formatting(self):
        """A variant that changed the ANSWER would make the battery nonsense."""
        for label, actual in probe._FORMATTING_VARIANTS:
            assert actual != probe._REFERENCE, label
            assert ToleratesFormatting._norm(actual) == ToleratesFormatting._norm(
                probe._REFERENCE
            ), f"{label} is not formatting-only: {actual!r}"


class TestBatteryScoping:
    """Who the battery applies to. Applying it to everything would be noise."""

    def test_skipped_for_a_metric_that_never_reads_expected_output(self, tmp_path):
        """No expected/actual comparison means no equivalence notion to test.

        Flagging AnswerRelevancy for "rejecting a code fence" would be a finding
        about nothing — it never looks at the expected output.
        """
        from app.services.metric_selftest_probe import _required_params

        metric = NoExpectedOutput()
        params = _required_params(metric)
        assert "expected_output" not in params
        # The scoping decision lives in ``main()``; assert the input it reads.
        assert params == ["input", "actual_output"]


class TestIntrospection:
    def test_judge_backed_is_read_from_the_instance(self):
        assert probe._is_judge_backed(JudgeBackedAlwaysFail()) is True
        assert probe._is_judge_backed(WorkingExactMatch()) is False

    def test_instance_evaluation_params_win_over_the_class_annotation(self):
        """GEval carries a type ANNOTATION in ``_required_params`` and the real
        list per instance — reading the class attribute alone yields nothing
        usable, which would silently skip the battery for every GEval."""
        metric = JudgeBackedAlwaysFail()
        assert probe._required_params(metric) == [
            "input", "actual_output", "expected_output",
        ]

    def test_missing_declarations_yield_an_empty_list(self):
        class Undeclared(_Base):
            def measure(self, case: Any) -> float:  # noqa: ARG002
                self.success = True
                return 1.0

        assert probe._required_params(Undeclared()) == []


class TestVerdictSplit:
    """What blocks, and what only warns."""

    @staticmethod
    def _payload(**entry: Any) -> dict[str, Any]:
        base = {
            "name": "M", "class": "M", "judge_backed": False,
            "required_params": ["input", "actual_output", "expected_output"],
            "degenerate": {"verdict": "ok", "detail": None},
            "formatting": {"ran": True, "rejected": [], "accepted": [], "errors": []},
        }
        base.update(entry)
        return {"status": "ok", "metrics": [base]}

    def test_a_deterministic_stuck_metric_blocks(self):
        blocking, warnings = _read_selftest_findings(self._payload(
            degenerate={"verdict": "always_fail", "detail": "failed an exact match"},
        ))
        assert blocking and not warnings

    def test_the_same_defect_on_a_judge_metric_only_warns(self):
        """One sample of a nondeterministic scorer is not proof, and blocking on
        it would stop a user's workflow on a coin flip."""
        blocking, warnings = _read_selftest_findings(self._payload(
            judge_backed=True,
            degenerate={"verdict": "always_fail", "detail": "failed an exact match"},
        ))
        assert not blocking
        assert warnings and "nondeterministic" in warnings[0]

    def test_formatting_rejections_never_block(self):
        """Strict-on-purpose and broken-normalizer are indistinguishable here;
        only the metric's author knows which, so this is a question, not a gate."""
        blocking, warnings = _read_selftest_findings(self._payload(
            formatting={"ran": True, "rejected": ["backticks", "code_fence"],
                        "accepted": [], "errors": []},
        ))
        assert not blocking
        assert "backticks" in warnings[0]

    def test_unknown_is_surfaced_rather_than_treated_as_a_pass(self):
        blocking, warnings = _read_selftest_findings(self._payload(
            degenerate={"verdict": "unknown", "detail": "metric raised"},
        ))
        assert not blocking
        assert "could not be probed" in warnings[0]

    def test_a_clean_run_produces_nothing(self):
        assert _read_selftest_findings(self._payload()) == ([], [])

    def test_a_verdict_that_needed_retries_warns_even_though_it_passed(self):
        """The metric is fine; the judge is not, and that costs real cases."""
        blocking, warnings = _read_selftest_findings(self._payload(
            degenerate={"verdict": "ok", "detail": None, "max_attempts": 3},
        ))
        assert not blocking
        assert warnings and "3 attempts" in warnings[0]

    def test_a_single_attempt_does_not_warn(self):
        assert _read_selftest_findings(self._payload(
            degenerate={"verdict": "ok", "detail": None, "max_attempts": 1},
        )) == ([], [])


class TestProbeRunsAsASubprocess:
    """The wiring, not the logic: does the probe actually execute and report?

    Every test above drives the probe's functions in-process. That is the shape
    that hides an integration break — a well-tested core with nothing calling
    it. Here the probe is executed the way validation executes it (source text,
    separate process, cwd = a project root, one JSON object on stdout) against a
    synthetic project whose metrics are deliberately broken.
    """

    @staticmethod
    def _make_project(root, *, metric_body: str) -> None:
        """Write the three modules the probe imports from a project."""
        (root / "src" / "evaluation").mkdir(parents=True)
        (root / "src" / "metrics").mkdir(parents=True)
        (root / "src" / "__init__.py").write_text("")
        (root / "src" / "evaluation" / "__init__.py").write_text("")

        (root / "src" / "config.py").write_text(
            "from dataclasses import dataclass, field\n"
            "@dataclass\n"
            "class EvalConfig:\n"
            "    judge_model_id: str = 'judge-model'\n"
            "    target_metric_scores: dict = field(default_factory=lambda: {})\n"
        )
        # The probe asks the runner for the judge adapter so it uses the same
        # mantle-vs-runtime routing as a real run; a stub is enough here.
        (root / "src" / "evaluation" / "runner.py").write_text(
            "class EvalRunner:\n"
            "    def __init__(self, config, source='manual', **kw):\n"
            "        self.config = config\n"
            "    def _make_model(self, model_id):\n"
            "        return object()\n"
        )
        (root / "src" / "metrics" / "__init__.py").write_text(metric_body)

    @staticmethod
    def _run(root) -> dict:
        import json as _json
        import subprocess

        from app.services.mcp_tools.harness import _probe_source

        proc = subprocess.run(
            [sys.executable, "-u", "-c", _probe_source()],
            cwd=str(root), capture_output=True, text=True, timeout=120,
        )
        assert proc.stdout.strip(), f"probe printed nothing; stderr={proc.stderr[-800:]}"
        return _json.loads(proc.stdout.strip().splitlines()[-1])

    def test_a_hardcoded_failing_metric_is_reported(self, tmp_path):
        self._make_project(tmp_path, metric_body=(
            "class AlwaysFail:\n"
            "    threshold = 1.0\n"
            "    _required_params = ['input', 'actual_output', 'expected_output']\n"
            "    @property\n"
            "    def __name__(self):\n"
            "        return 'Bogus Correctness'\n"
            "    def measure(self, case):\n"
            "        self.score = 0.0\n"
            "        self.success = False\n"
            "        self.reason = 'nope'\n"
            "        return 0.0\n"
            "    def is_successful(self):\n"
            "        return self.success\n"
            "\n"
            "def build_metrics(judge_model):\n"
            "    return [AlwaysFail()]\n"
        ))
        payload = self._run(tmp_path)
        assert payload["status"] == "ok", payload
        assert len(payload["metrics"]) == 1
        entry = payload["metrics"][0]
        assert entry["name"] == "Bogus Correctness"
        assert entry["judge_backed"] is False
        assert entry["degenerate"]["verdict"] == "always_fail"

        # And the verdict layer blocks on it — the two halves agree end to end.
        blocking, _warnings = _read_selftest_findings(payload)
        assert blocking and "Bogus Correctness" in blocking[0]

    def test_no_metrics_at_all_is_an_error(self, tmp_path):
        """The scaffold ships ``build_metrics`` returning ``[]``; validation must
        not pass a project where nothing scores the eval."""
        self._make_project(tmp_path, metric_body=(
            "def build_metrics(judge_model):\n    return []\n"
        ))
        payload = self._run(tmp_path)
        assert payload["status"] == "error"
        assert "no metrics" in payload["errors"][0]

    def test_a_metrics_module_that_raises_is_an_error_with_a_traceback(self, tmp_path):
        self._make_project(tmp_path, metric_body=(
            "def build_metrics(judge_model):\n"
            "    raise ValueError('criteria missing')\n"
        ))
        payload = self._run(tmp_path)
        assert payload["status"] == "error"
        assert "criteria missing" in payload["errors"][0]
        assert "traceback" in payload

    def test_a_working_tolerant_metric_passes_clean(self, tmp_path):
        self._make_project(tmp_path, metric_body=(
            "class Norm:\n"
            "    threshold = 1.0\n"
            "    _required_params = ['input', 'actual_output', 'expected_output']\n"
            "    @property\n"
            "    def __name__(self):\n"
            "        return 'Normalized Match'\n"
            "    @staticmethod\n"
            "    def _n(t):\n"
            "        out = (t or '').strip()\n"
            "        out = out.replace('```sql', '').replace('```', '')\n"
            "        out = out.strip().strip('`').strip('\"').strip('*')\n"
            "        if out.lower().startswith('answer:'):\n"
            "            out = out[7:].strip()\n"
            "        return out.rstrip(';.').strip().replace(chr(8217), chr(39)).lower()\n"
            "    def measure(self, case):\n"
            "        ok = self._n(case.actual_output) == self._n(case.expected_output)\n"
            "        self.score = 1.0 if ok else 0.0\n"
            "        self.success = ok\n"
            "        self.reason = 'normalized'\n"
            "        return self.score\n"
            "    def is_successful(self):\n"
            "        return self.success\n"
            "\n"
            "def build_metrics(judge_model):\n"
            "    return [Norm()]\n"
        ))
        payload = self._run(tmp_path)
        entry = payload["metrics"][0]
        assert entry["degenerate"]["verdict"] == "ok"
        assert entry["formatting"]["ran"] is True
        assert entry["formatting"]["rejected"] == [], entry["formatting"]
        assert _read_selftest_findings(payload) == ([], [])
