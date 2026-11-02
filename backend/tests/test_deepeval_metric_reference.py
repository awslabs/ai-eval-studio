# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The metric reference table must agree with the INSTALLED DeepEval.

WHY THIS IS A TEST AND NOT A DOC NOTE. ``deepeval-metric-reference.md`` is the
coordinator's authoritative source for what each metric reads — the doc says so
in its own header, and tells the agent that web search is fallback only. So the
table being wrong or incomplete does not produce an error; it produces a
confidently-built harness that measures the wrong thing. The doc used to carry a
"verification recipe (re-run on version bump)" snippet, which is a comment
asking a human to remember. This runs it.

TWO FAILURE MODES, both of which have happened:

1. **DRIFT.** DeepEval has rotated field names across releases — the header
   records mapping ``context`` to ContextualRecall when it keys on
   ``retrieval_context``. Every documented row is therefore compared against the
   live ``_required_params``.

2. **OMISSION**, which is the one that bit. ``ExactMatchMetric`` shipped in
   3.9.9 but had no row, so when a project asked for exact-match scoring the
   builder had nothing to consult and hand-rolled a ``BaseMetric`` subclass with
   its own normalizer — a normalizer being the one thing that can be
   over-lenient, which is the error direction nothing downstream catches. An
   absent row is invisible to a drift-only check, so every concrete single-turn
   metric DeepEval ships must be EITHER documented OR listed below with a
   reason. Adding a metric to DeepEval then forces a decision here rather than
   silently widening the gap.

The version pin itself is already gated by
``test_coordinator_prompt.py::test_metric_reference_version_matches_installed_deepeval``
and is deliberately not re-checked here — two copies of one assertion is how the
copies drift.

Note that this file is a MAINTAINER-facing gate while the reference doc is an
AGENT-facing prompt, so the explanation lives here rather than in the doc: a
prompt naming ``backend/...`` points the agent at a path it cannot reach from the
project working tree, which ``test_no_prompt_points_agents_at_repo_paths``
rejects.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import deepeval.metrics as metrics_module
from deepeval.metrics import BaseMetric

REFERENCE = (
    Path(__file__).resolve().parent.parent
    / "app" / "prompts" / "deepeval-metric-reference.md"
)

#: Shipped single-turn metrics with no table row, each with the reason. This is
#: a DECISION LIST, not a suppression list: a new DeepEval metric lands here or
#: in the table, and writing the reason is the point.
UNDOCUMENTED_WITH_REASON = {
    # Abstract / parameterised-at-construction. GEval and DAGMetric DO have
    # rows; they carry a typing annotation in ``_required_params`` and the real
    # list per instance as ``evaluation_params``, so they are exempt from the
    # param comparison rather than from the table.
    "BaseMetric": "abstract base class, not a usable metric",
    "DAGMetric": "params are per-instance (evaluation_params), like GEval",

    # Multimodal — needs a case type the harness does not support.
    "ImageCoherenceMetric": "multimodal; harness is text-only",
    "ImageEditingMetric": "multimodal; harness is text-only",
    "ImageHelpfulnessMetric": "multimodal; harness is text-only",
    "ImageReferenceMetric": "multimodal; harness is text-only",
    "TextToImageMetric": "multimodal; harness is text-only",

    # Agentic / trajectory — needs tool-call or plan traces the harness does not
    # capture, so these stay out until the harness records them.
    "ArgumentCorrectnessMetric": "needs tools_called; harness captures no tool trace",
    "MCPUseMetric": "needs mcp_servers; harness captures no MCP trace",
    "PlanAdherenceMetric": "agentic trajectory metric; no plan trace in the harness",
    "PlanQualityMetric": "agentic trajectory metric; no plan trace in the harness",
    "StepEfficiencyMetric": "agentic trajectory metric; no step trace in the harness",
    "TaskCompletionMetric": "agentic trajectory metric; no task trace in the harness",

    # Guardrail metrics of the SAME shape as the documented Bias/Toxicity rows.
    # Honestly deferred rather than out of scope: usable today, undocumented
    # only because no project has asked. Add a row when one does.
    "MisuseMetric": "guardrail, same shape as BiasMetric; document on first use",
    "NonAdviceMetric": "guardrail, same shape as BiasMetric; document on first use",
    "PIILeakageMetric": "guardrail, same shape as BiasMetric; document on first use",
    "RoleViolationMetric": "guardrail, same shape as BiasMetric; document on first use",
}

#: Documented metrics whose ``_required_params`` is a typing annotation rather
#: than a list, so there is nothing to compare field-by-field.
PARAMS_PER_INSTANCE = {"GEval"}


def _documented_rows() -> dict[str, list[str]]:
    """Parse ``metric name -> documented required params`` out of the table.

    Reads the rendered table rather than a parallel list in this file: a second
    copy of the mapping would be one more thing to drift.
    """
    rows: dict[str, list[str]] = {}
    for line in REFERENCE.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        name_match = re.fullmatch(r"`(\w+)`", cells[0])
        if not name_match:
            continue
        rows[name_match.group(1)] = re.findall(r"`(\w+)`", cells[1])
    return rows


def _shipped_single_turn() -> dict[str, type]:
    """Every ``BaseMetric`` subclass ``deepeval.metrics`` exports."""
    out: dict[str, type] = {}
    for name in dir(metrics_module):
        if name.startswith("_"):
            continue
        obj = getattr(metrics_module, name)
        if inspect.isclass(obj) and issubclass(obj, BaseMetric):
            out[name] = obj
    return out


def test_the_table_parses_at_all():
    """A guard on the parser, so a table format change cannot pass vacuously.

    Without this, a heading tweak that breaks the row regex would make every
    check below iterate an empty dict and report success.
    """
    rows = _documented_rows()
    assert len(rows) >= 14, f"only parsed {len(rows)} rows — did the format change?"
    assert "AnswerRelevancyMetric" in rows
    assert rows["AnswerRelevancyMetric"] == ["input", "actual_output"]


def test_documented_params_match_the_installed_package():
    """Every row's required params equal the live ``_required_params``."""
    mismatches = []
    for name, documented in _documented_rows().items():
        if name in PARAMS_PER_INSTANCE:
            continue
        cls = getattr(metrics_module, name, None)
        assert cls is not None, f"{name} is documented but not exported by DeepEval"
        live_raw = getattr(cls, "_required_params", None)
        assert isinstance(live_raw, list), (
            f"{name} has no _required_params list — add it to PARAMS_PER_INSTANCE "
            "if its params are per-instance"
        )
        live = [getattr(p, "value", str(p)) for p in live_raw]
        if sorted(live) != sorted(documented):
            mismatches.append(f"  {name}: doc={documented} installed={live}")
    assert not mismatches, (
        "the metric reference disagrees with the installed DeepEval:\n"
        + "\n".join(mismatches)
    )


def test_every_shipped_single_turn_metric_is_documented_or_explained():
    """The omission check — an absent row is invisible to a drift-only test.

    This is what ``ExactMatchMetric`` needed: it shipped, it was the right metric
    for a live project, and nothing pointed the builder at it.
    """
    documented = set(_documented_rows())
    shipped = set(_shipped_single_turn())
    unaccounted = sorted(shipped - documented - set(UNDOCUMENTED_WITH_REASON))
    assert not unaccounted, (
        "DeepEval ships single-turn metrics that are neither in the reference "
        "table nor listed with a reason:\n  "
        + "\n  ".join(unaccounted)
        + "\n\nAdd a table row, or an entry in UNDOCUMENTED_WITH_REASON saying "
        "why not. Silence here is how a usable metric goes unnoticed and gets "
        "hand-rolled instead."
    )


def test_the_reason_list_does_not_rot():
    """A metric that DeepEval removes must not linger in the reason list.

    Otherwise the list slowly becomes a record of a package version nobody runs.
    """
    shipped = set(_shipped_single_turn())
    stale = sorted(set(UNDOCUMENTED_WITH_REASON) - shipped)
    assert not stale, (
        "UNDOCUMENTED_WITH_REASON names metrics this DeepEval no longer ships: "
        f"{stale} — drop them"
    )


def test_the_deterministic_metrics_are_flagged_as_free():
    """The steer that prevents the hand-rolled-normalizer failure.

    ``ExactMatchMetric`` and ``PatternMatchMetric`` cost nothing to run, and the
    reason the builder should reach for them first is that a hand-written
    normalizer is the one thing that can be over-LENIENT. The rows have to say
    so, or the table just lists them as two more options.
    """
    content = REFERENCE.read_text(encoding="utf-8")
    for name in ("ExactMatchMetric", "PatternMatchMetric"):
        row = next(
            line for line in content.splitlines() if line.startswith(f"| `{name}`")
        )
        assert "DETERMINISTIC" in row, f"{name} row does not say it needs no judge"

    exact_row = next(
        line for line in content.splitlines()
        if line.startswith("| `ExactMatchMetric`")
    )
    # The exact semantics, because "exact match" is ambiguous about case.
    assert "strip()" in exact_row
    assert "NO case folding" in exact_row

    pattern_row = next(
        line for line in content.splitlines()
        if line.startswith("| `PatternMatchMetric`")
    )
    # The two constraints that decide whether it fits at all.
    assert "fullmatch" in pattern_row
    assert "does not read `expected_output`" in pattern_row


def test_the_documented_exact_match_semantics_are_TRUE_of_the_installed_metric():
    """Verify the CLAIM, not just that two docs agree on the wording.

    "whitespace-trim only, no case folding" is asserted in two places — the table
    row above and the harness template's ``src/metrics/metrics.py`` docstring —
    because they have different audiences (planning vs. writing the file). Two
    copies of a claim about third-party behaviour is a drift risk, and
    string-comparing the two docs would only prove they match each other. This
    runs the metric instead, so if DeepEval ever adds case folding both docs are
    wrong and this fails.

    The direction matters: case folding would make the metric more LENIENT, and
    over-leniency is the error nothing downstream can catch. That is exactly why
    the docs tell the builder to prefer it over a hand-rolled normalizer.
    """
    from deepeval.metrics import ExactMatchMetric
    from deepeval.test_case import LLMTestCase

    def _score(actual: str, expected: str) -> float:
        return ExactMatchMetric().measure(
            LLMTestCase(input="q", actual_output=actual, expected_output=expected),
            _show_indicator=False,
        )

    # Whitespace IS trimmed — both ends, and a trailing newline.
    assert _score("  status == 'Shipped'\n", "status == 'Shipped'") == 1.0

    # Case is NOT folded. If this flips, the metric became over-lenient and the
    # "can never score a wrong answer as correct" claim is void.
    assert _score("STATUS == 'SHIPPED'", "status == 'Shipped'") == 0.0

    # Nothing else is normalized: the artifacts the docs warn about still fail.
    for artifact in ("`status == 'Shipped'`", "status == 'Shipped';",
                     "```\nstatus == 'Shipped'\n```"):
        assert _score(artifact, "status == 'Shipped'") == 0.0, artifact

    # Interior whitespace is NOT collapsed either — only the ends are stripped.
    assert _score("status  ==  'Shipped'", "status == 'Shipped'") == 0.0


def test_pattern_match_requires_a_FULL_match():
    """The constraint that decides whether PatternMatchMetric fits a task.

    Documented in its row because a partial-match assumption is silent: the
    pattern looks right, and every case scores 0 for a reason the pattern does
    not show.
    """
    from deepeval.metrics import PatternMatchMetric
    from deepeval.test_case import LLMTestCase

    metric = PatternMatchMetric(pattern=r"SELECT .*")

    def _score(actual: str) -> float:
        return metric.measure(
            LLMTestCase(input="q", actual_output=actual),
            _show_indicator=False,
        )

    assert _score("SELECT * FROM findings") == 1.0
    # A prefix match is NOT enough — fullmatch, so leading prose fails.
    assert _score("Here you go: SELECT * FROM findings") == 0.0
