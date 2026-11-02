# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metric self-test probe — runs INSIDE the generated project's venv.

WHY THIS EXISTS. ``validate_harness`` proves a harness RUNS: files exist, ``uv
sync`` succeeds, ``EvalConfig`` instantiates, models are invocable, a 1-case
smoke eval completes with well-formed scores. None of that says the metrics
measure anything. Walk it against ``_check_smoke_output``: ``score: 0.0`` is a
float, ``success: False`` is a bool, and a non-empty ``reason`` defeats the
silent-failure branch — so **a metric hardcoded to return
``success=False, reason="nope"`` on every case passes harness validation.**
Nothing asserted that any metric ever passes, and at N=1 the smoke case would
have to happen to contain the artifact.

WHY IT IS A SEPARATE SCRIPT, NOT BACKEND CODE. A project's metrics live in
``src/metrics/metrics.py`` and are built by ``build_metrics(judge_model)``. They
are per-project Python, frequently custom subclasses, and they import DeepEval
from the project's own venv — the backend cannot introspect them in-process.
Real projects also carry NO ``metrics`` field in ``EvalConfig`` (the field is
optional and commonly absent), so the harness config is not a
substitute source either. This module is therefore executed as source text by
``mcp_tools.harness`` via ``uv run python -c``, with cwd set to the project
root, exactly like the existing config-validation step. It lives here as a real
file so it is linted and directly testable, and it is passed by VALUE so an
old project gets the current probe with no template-sync question.

It must not be imported by the backend: its imports (``src.*``) only resolve
inside a project.

WHAT IT CHECKS, and what each check is worth:

1. **Degenerate metric** (fully general — any metric, any output type). Score an
   obviously-correct pair and TWO obviously-wrong pairs, and require the metric
   to pass the first and reject at least one of the others. A metric that
   answers the same way on all three cannot distinguish anything. This is the
   check the identity/mismatch pairs are genuinely FOR, and its only job — see
   below. Two wrong outputs rather than one because no single wrong answer is
   wrong for every metric family: see :data:`_WRONG_OUTPUTS`.
2. **Formatting-artifact battery** (text-equivalence metrics only). Pairs that
   differ ONLY in formatting an LLM routinely adds: code fences, backticks,
   surrounding quotes, trailing punctuation, whitespace, smart quotes, markdown
   emphasis, an ``Answer:`` prefix, casing. Reports which variants each metric
   rejects. It cannot judge them: rejecting a code fence is CORRECT for a
   deliberately strict metric and a defect for one that claims tolerance, and
   only the metric's author knows which. Enumerated artifacts only — it would
   have caught backticks because backticks are on the list, and cannot catch an
   artifact nobody enumerated (that is what the post-run disagreement detector
   is for).

**The rejected first design, recorded so it is not rebuilt:** "identity pair
must pass, mismatched pair must fail" as the whole probe. That does NOT catch a
normalizer that strips nothing — on a literal identity pair the raw strings
already compare equal, so a no-op normalizer goes green and the bug survives.
Equivalence-UNDER-normalization is the only shape that tests a normalizer, which
is why check 2 exists as its own thing and why check 1 is scoped to degeneracy.

JUDGE-BACKED METRICS ARE PROBED THREE TIMES, NOT FOURTEEN. Check 1 costs three
``measure()`` calls per metric — and a DeepEval metric is several judge calls
per ``measure()``, not one (measured on 3.9.9: AnswerRelevancy 3, Faithfulness
4, the contextual metrics 2 each), so a RAG lineup runs to tens of serialized
calls. Check 2 would add eleven more ``measure()`` calls per metric and would be
measuring the judge's nondeterminism rather than a normalizer, so it is skipped
for judge-backed metrics with a recorded reason. This is also why the caller
treats the probe's own timeout as a warning, not a failure: the ceiling is judge
latency, which is not a property of the metric.

NEVER GUESSES A VERDICT. Any metric that raises, or whose scoring cannot be
interpreted, is reported as ``unknown`` with the exception text; one whose
required fields this probe cannot synthesize is ``not_probed`` with those fields
NAMED. The caller decides what blocks; this script only measures.

A TRANSIENT JUDGE FAILURE IS RETRIED, because otherwise it decides the verdict.
Without a retry, a gating metric can come back ``unknown`` with DeepEval's
``ValueError: Evaluation LLM outputted an invalid JSON. Please use a better
evaluation model.`` — the judge's HTTP call SUCCEEDED and DeepEval then failed to
parse its text (``deepeval/metrics/utils.py`` ``trimAndLoadJson``, and
``DeepEvalError`` from ``deepeval/models/llms/utils.py``). One unlucky sample
would then leave the only metric that gates the run unprobed, and the model
layer's own retry cannot help: it is raised ABOVE the adapter, so
``_bedrock_retry`` never sees it. See :data:`_TRANSIENT_JUDGE_MARKERS`.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from typing import Any

#: The reference answer for both checks. Shaped like a short structured string
#: (the common eval output) rather than prose, so the formatting variants below
#: are the kind of thing a model actually emits around it.
_REFERENCE = "status == 'Shipped'"

#: TWO wrong outputs, not one, because no single wrong answer is wrong for every
#: metric family and using one produces false positives on the two most common
#: DeepEval metrics:
#:
#: * a non-answer is correctly rejected by AnswerRelevancy, but Faithfulness
#:   scores it PASS — an output that makes no claims contradicts no context, so
#:   it is perfectly faithful. That is Faithfulness working, not a defect.
#: * a plausible-but-wrong answer is correctly rejected by Faithfulness (it
#:   contradicts the context) and by any expected-output comparison, but
#:   AnswerRelevancy scores it PASS — it is a relevant answer to the question,
#:   and relevancy never reads ``expected_output``.
#:
#: So the verdict below requires a metric to reject AT LEAST ONE of these, not
#: both. A metric that passes the identity pair AND both of these distinguishes
#: nothing whatever it claims to measure; one that rejects either is doing work.
_WRONG_OUTPUTS: list[tuple[str, str]] = [
    ("non_answer", "ERROR: no answer produced"),
    ("contradiction", "status == 'Cancelled'"),
]

#: Grounding document for context-reading metrics, asserting the reference
#: answer. Without it, every retrieval metric raises
#: ``MissingTestCaseParamsError`` before making a single judge call — see
#: :data:`_SUPPLIED_PARAMS`.
_CONTEXT = [
    "Shipped orders are those whose status field equals 'Shipped'. "
    "To select them, filter on status == 'Shipped'."
]

#: Test-case fields this probe can populate. A metric requiring anything outside
#: this set cannot be probed and is reported as such, by name.
#:
#: ``retrieval_context`` and ``context`` are here because omitting them made the
#: self-test INERT on the mainstream DeepEval shape. Verified on 3.9.9:
#: ``FaithfulnessMetric``, ``ContextualPrecisionMetric`` and
#: ``ContextualRecallMetric`` all list ``RETRIEVAL_CONTEXT`` in
#: ``_required_params``, and ``check_llm_test_case_params`` raises
#: ``MissingTestCaseParamsError`` on any required param that is ``None``.
#: Without them every retrieval metric in a RAG project would come back
#: "unknown" — reported honestly, but checked for nothing.
_SUPPLIED_PARAMS = frozenset(
    {"input", "actual_output", "expected_output", "retrieval_context", "context"}
)

#: Formatting variants applied to the reference. Each pair is (label, actual)
#: where ``actual`` differs from the reference ONLY in formatting an LLM adds.
_FORMATTING_VARIANTS: list[tuple[str, str]] = [
    ("code_fence", f"```\n{_REFERENCE}\n```"),
    ("fenced_language", f"```sql\n{_REFERENCE}\n```"),
    ("backticks", f"`{_REFERENCE}`"),
    ("double_quotes_wrapping", f'"{_REFERENCE}"'),
    ("trailing_semicolon", f"{_REFERENCE};"),
    ("trailing_period", f"{_REFERENCE}."),
    ("surrounding_whitespace", f"  {_REFERENCE}\n"),
    ("smart_quotes", _REFERENCE.replace("'", "’")),
    ("markdown_emphasis", f"**{_REFERENCE}**"),
    ("answer_prefix", f"Answer: {_REFERENCE}"),
    ("upper_case", _REFERENCE.upper()),
]


#: Attempts per ``measure()`` call, counting the first. Three rather than more
#: because the retry is bounded by the caller's whole-probe timeout, and a metric
#: whose judge cannot produce parseable output three times running is a real
#: finding rather than bad luck.
_SCORE_ATTEMPTS = 3

#: Base seconds between attempts, multiplied by the attempt number. Small on
#: purpose: invalid-JSON is not a rate problem and needs no real backoff, but the
#: same path catches throttling, where a pause is what helps.
_SCORE_BACKOFF_S = 1.0

#: Lowercased substrings that mean "this sample failed, another may not".
#:
#: Matched against ``"<ExcType>: <message>"`` rather than by exception type,
#: which is deliberate and is why this is not shared with the harness's
#: ``_is_retryable_bedrock_error``: that classifier is typed on botocore/httpx
#: exceptions, and the failure this exists for arrives as a BARE ``ValueError``
#: from inside DeepEval, well above any adapter. A typed classifier cannot see
#: it. The AWS markers are kept alongside because the probe sits above the model
#: layer's retry and sustained throttling can exhaust it.
#:
#: Deliberately narrow. Retrying everything would triple the cost of a genuinely
#: broken metric and turn a fast, clear failure into a slow one — an
#: AccessDenied or a bug in the metric's own code must surface on the first
#: attempt.
_TRANSIENT_JUDGE_MARKERS: tuple[str, ...] = (
    # DeepEval, both spellings of the same underlying JSONDecodeError.
    "outputted an invalid json",
    # Bedrock transient markers, mirroring the harness's retryable subset.
    "throttlingexception",
    "too many requests",
    "rate exceeded",
    "serviceunavailableexception",
    "service unavailable",
    "internalserverexception",
    "internal server error",
    "modeltimeoutexception",
    "model timeout",
    # Transport-level.
    "read timeout",
    "connect timeout",
    "connection reset",
    "connection closed",
)


def _is_transient_judge_failure(exc: BaseException) -> bool:
    """True when re-running the same ``measure()`` could plausibly succeed.

    Args:
        exc: The exception raised by ``metric.measure()``.

    Returns:
        True if the exception matches a :data:`_TRANSIENT_JUDGE_MARKERS` entry.
    """
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in message for marker in _TRANSIENT_JUDGE_MARKERS)


def _metric_name(metric: Any) -> str:
    """Best-effort display name for a metric instance.

    DeepEval exposes ``__name__`` as a property on ``BaseMetric`` and the
    runner reads it directly when building ``MetricData``, so it is the name
    that ends up in run files. Fall back to the class name.
    """
    for attr in ("__name__", "name"):
        value = getattr(metric, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(metric).__name__


def _required_params(metric: Any) -> list[str]:
    """Which test-case fields this metric reads, when it says so.

    DeepEval's concrete metrics carry a class-level ``_required_params`` list
    (verified on 3.9.9: ``ExactMatchMetric`` → input/actual_output/
    expected_output, ``AnswerRelevancyMetric`` → input/actual_output). ``GEval``
    and ``DAGMetric`` instead carry a type ANNOTATION there and take the real
    list per instance as ``evaluation_params``, so both are read and the
    instance wins. Returns ``[]`` when neither is usable — a custom metric need
    not declare anything.
    """
    for attr in ("evaluation_params", "_required_params"):
        raw = getattr(metric, attr, None)
        if isinstance(raw, list) and raw:
            out = []
            for item in raw:
                value = getattr(item, "value", item)
                if isinstance(value, str):
                    out.append(value)
            if out:
                return out
    return []


def _is_judge_backed(metric: Any) -> bool:
    """True when scoring this metric costs a model call.

    Read from the instance rather than a class allowlist: DeepEval sets ``model``
    (and ``using_native_model``) on every LLM-judged metric at construction,
    including ``GEval``, and leaves it absent on deterministic ones like
    ``ExactMatchMetric``. A custom metric that holds a model it does not use is
    treated as judge-backed, which only costs it the cheap probes.
    """
    return getattr(metric, "model", None) is not None


def _score(metric: Any, *, actual: str, expected: str, user_input: str) -> dict[str, Any]:
    """Run one metric against one synthetic case and report what it said.

    Returns a dict with ``ok`` False plus ``error`` when the metric raised — a
    metric that cannot score a well-formed case is itself a finding, and
    swallowing the exception would report it as a passing probe.

    Transient judge failures are retried up to :data:`_SCORE_ATTEMPTS` times (see
    :data:`_TRANSIENT_JUDGE_MARKERS`); anything else fails on the first attempt.
    Either way the returned dict carries ``attempts``, so a verdict that needed
    more than one sample says so instead of looking clean.
    """
    from deepeval.test_case import LLMTestCase

    # ``retrieval_context``/``context`` are always populated, not only for
    # metrics that read them: a metric that ignores them is unaffected, and a
    # metric that requires them raises before any judge call if they are absent.
    case = LLMTestCase(
        input=user_input,
        actual_output=actual,
        expected_output=expected,
        retrieval_context=list(_CONTEXT),
        context=list(_CONTEXT),
    )

    attempts = 0
    while True:
        attempts += 1
        try:
            score = metric.measure(case)
            break
        except Exception as exc:  # noqa: BLE001 — the exception IS the result here
            transient = _is_transient_judge_failure(exc)
            if attempts >= _SCORE_ATTEMPTS or not transient:
                error = f"{type(exc).__name__}: {exc}"
                if transient:
                    # Says "still failing after N" rather than "failed", so the
                    # reader knows this was not one unlucky sample.
                    error += f" (still failing after {attempts} attempts)"
                return {"ok": False, "error": error, "attempts": attempts}
            time.sleep(_SCORE_BACKOFF_S * attempts)

    # ``is_successful()`` is the authoritative verdict (it applies the metric's
    # own threshold); ``success`` is read as a fallback for metrics that set the
    # attribute without implementing the method.
    success: Any = None
    try:
        success = metric.is_successful()
    except Exception:  # noqa: BLE001
        success = getattr(metric, "success", None)
    return {
        "ok": True,
        "score": score if isinstance(score, (int, float)) else None,
        "success": bool(success) if isinstance(success, bool) else None,
        "reason": getattr(metric, "reason", None),
        "attempts": attempts,
    }


def _unsupportable_params(metric: Any) -> list[str]:
    """Required test-case fields this probe cannot synthesize, by name.

    Named rather than reported as a generic failure so a metric requiring, say,
    ``tools_called`` says exactly that instead of surfacing as an opaque
    "could not be probed".
    """
    return sorted(set(_required_params(metric)) - _SUPPLIED_PARAMS)


def _degenerate_check(metric: Any, user_input: str) -> dict[str, Any]:
    """Does this metric distinguish an obviously-correct answer from a wrong one?

    Three cases, no formatting involved: a literal match, and the two wrong
    outputs in :data:`_WRONG_OUTPUTS`. The metric must pass the match and reject
    AT LEAST ONE of the wrong outputs — requiring it to reject both would flag
    Faithfulness (a non-answer is faithful) and AnswerRelevancy (a wrong answer
    is still relevant) as degenerate when both are working correctly.

    A metric that passes all three cannot distinguish anything: always-pass
    certifies every run. One that fails the identity pair cannot either:
    always-fail zeroes every run.
    """
    missing = _unsupportable_params(metric)
    if missing:
        return {
            "verdict": "not_probed",
            "detail": (
                "requires test-case field(s) this probe cannot synthesize: "
                + ", ".join(missing)
            ),
        }

    good = _score(metric, actual=_REFERENCE, expected=_REFERENCE, user_input=user_input)
    wrong: dict[str, dict[str, Any]] = {
        label: _score(
            metric, actual=actual, expected=_REFERENCE, user_input=user_input
        )
        for label, actual in _WRONG_OUTPUTS
    }
    # Worst case across the three probes. Reported even when the verdict is
    # clean: a judge that needed extra samples to answer here is equally
    # unreliable when it scores the real run, which the caller warns about.
    probes = {
        "correct_pair": good,
        "wrong_pairs": wrong,
        "max_attempts": max(r.get("attempts", 1) for r in [good, *wrong.values()]),
    }

    failed = [r for r in [good, *wrong.values()] if not r["ok"]]
    if failed:
        return {
            "verdict": "unknown",
            "detail": failed[0].get("error"),
            **probes,
        }

    good_pass = good["success"]
    wrong_passes = [r["success"] for r in wrong.values()]
    if good_pass is None or any(v is None for v in wrong_passes):
        return {
            "verdict": "unknown",
            "detail": "metric reported no boolean verdict for at least one probe",
            **probes,
        }

    rejected = [label for label, r in wrong.items() if not r["success"]]
    if good_pass and rejected:
        verdict, detail = "ok", None
    elif good_pass:
        verdict, detail = (
            "always_pass",
            "passed output that is not an answer at all AND output that "
            "contradicts the expected answer, so every case passes regardless "
            "of what the model produced",
        )
    elif not any(wrong_passes):
        verdict, detail = (
            "always_fail",
            "failed output identical to the expected output, so no case can "
            "ever pass",
        )
    else:
        verdict, detail = (
            "inverted",
            "failed output identical to the expected output while passing a "
            "wrong answer — the verdict looks inverted",
        )
    return {"verdict": verdict, "detail": detail, "rejected": rejected, **probes}


def _formatting_check(metric: Any, user_input: str) -> dict[str, Any]:
    """Which formatting-only variants of a correct answer does this metric reject?

    Reported, not judged: for a deliberately strict metric every rejection is
    correct and visible in the results; for one that claims tolerance each is a
    case scored wrong for a reason unrelated to the task.
    """
    rejected: list[str] = []
    accepted: list[str] = []
    errors: list[dict[str, str]] = []
    for label, actual in _FORMATTING_VARIANTS:
        result = _score(
            metric, actual=actual, expected=_REFERENCE, user_input=user_input
        )
        if not result["ok"]:
            errors.append({"variant": label, "error": result["error"]})
            continue
        if result["success"] is True:
            accepted.append(label)
        elif result["success"] is False:
            rejected.append(label)
    return {
        "ran": True,
        "rejected": rejected,
        "accepted": accepted,
        "errors": errors,
    }


def main() -> int:
    """Build the project's metrics, probe each one, print one JSON object."""
    result: dict[str, Any] = {"status": "ok", "errors": [], "metrics": []}
    try:
        from src.config import EvalConfig
        from src.evaluation.runner import EvalRunner
        from src.metrics import build_metrics

        config = EvalConfig()
        # Reuse the runner's own adapter routing (mantle vs runtime by
        # descriptor presence) instead of re-deriving it here — a second
        # derivation would drift from the one the real run uses.
        runner = EvalRunner(config, source="validation")
        judge = runner._make_model(config.judge_model_id)
        metrics = build_metrics(judge)
    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        result["traceback"] = traceback.format_exc()[-2000:]
        print(json.dumps(result))
        return 0

    if not metrics:
        result["status"] = "error"
        result["errors"].append(
            "build_metrics() returned no metrics — nothing scores this eval"
        )
        print(json.dumps(result))
        return 0

    user_input = "Show orders whose status is Shipped"
    for metric in metrics:
        entry: dict[str, Any] = {
            "name": _metric_name(metric),
            "class": type(metric).__name__,
            "threshold": getattr(metric, "threshold", None),
            "judge_backed": _is_judge_backed(metric),
            "required_params": _required_params(metric),
        }
        entry["degenerate"] = _degenerate_check(metric, user_input)

        # The battery needs a notion of "the same answer, formatted
        # differently", which only exists for a metric that compares actual
        # against expected. And eleven more ``measure()`` calls per metric
        # would measure the judge, not a normalizer.
        params = entry["required_params"]
        if entry["degenerate"].get("verdict") == "not_probed":
            entry["formatting"] = {
                "ran": False,
                "skipped_reason": entry["degenerate"]["detail"],
            }
        elif entry["judge_backed"]:
            entry["formatting"] = {
                "ran": False,
                "skipped_reason": "judge-backed: eleven more measure() calls per "
                                  "metric would measure judge nondeterminism, "
                                  "not normalization",
            }
        elif params and "expected_output" not in params:
            entry["formatting"] = {
                "ran": False,
                "skipped_reason": "does not read expected_output, so it has no "
                                  "equivalence notion to test",
            }
        else:
            entry["formatting"] = _formatting_check(metric, user_input)

        result["metrics"].append(entry)

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
