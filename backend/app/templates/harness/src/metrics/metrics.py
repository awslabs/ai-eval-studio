# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Metric builder for evaluation.

Constructs the list of DeepEval metrics used during evaluation runs.
The AI agent overwrites this file with project-specific metrics based
on the evaluation requirements.

Example metrics:
- Exact match (string equality)
- Semantic similarity (LLM-as-a-judge)
- Latency threshold
- Custom domain-specific metrics

PREFER DEEPEVAL'S SHIPPED METRICS. ``deepeval.metrics`` provides
``ExactMatchMetric`` and ``PatternMatchMetric`` among others. The shipped
exact-match compares ``expected.strip() == actual.strip()`` — whitespace-trim
only, no case folding — so it can never score a wrong answer as correct. Use it
whenever the task wants a literal match. It IS strict, so a model that wraps its
output in backticks or a ```lang fence, or appends a semicolon, will fail. That
is the safe direction: the failure is visible in the results.

WRITING YOUR OWN DETERMINISTIC METRIC. Subclass ``BaseMetric`` so the contract is
explicit (second example below). If the task genuinely needs tolerance the
shipped metric does not give, you must normalize — and no normalizer is provided
here ON PURPOSE, because what counts as insignificant is domain-specific:

- Case folding is WRONG when the target language is case-sensitive (identifiers,
  field names, column names).
- Whitespace collapsing is WRONG when whitespace is syntax (Python, YAML).
- Punctuation stripping is WRONG when punctuation is syntax (SQL, JSON, regex).

Only this project's own data can tell you which apply.

VALIDATE ANY NORMALIZER AGAINST THIS PROJECT'S DATASET BEFORE USING IT. Run these
over the dataset's own ``expected_output`` values. They need no model calls and no
eval run, so they cost nothing:

1. NO COLLISION — critical. Two DISTINCT expected values must never normalize to
   the same string. A collision means the metric will score a wrong answer as
   CORRECT, and over-leniency is the one error direction nothing downstream can
   catch. This is the check that rejects an over-aggressive rule: lower-casing
   makes ``orderId`` and ``orderid`` collide, so it is unusable on a dataset
   whose field names differ only by case.

2. IDEMPOTENT. ``normalize(normalize(x)) == normalize(x)``. A normalizer that
   keeps changing its own output is not producing a canonical form, and its
   verdicts then depend on how many times it happened to run.

3. FIXED POINT ON GROUND TRUTH — advisory, not pass/fail.
   ``normalize(expected) == expected`` for the dataset's values. Where it differs,
   the normalizer is rewriting ground truth. Sometimes that is correct (the
   dataset carries stray trailing whitespace and ``.strip()`` is the fix);
   sometimes it means the rule is too aggressive. Inspect the differences rather
   than assuming either way — note that a plain ``.strip()`` legitimately fails
   this check, so it must not gate anything on its own.

Report the counts from all three in your summary so the user can judge them.
"""

from src.models.bedrock_runtime_model import BedrockRuntimeModel


def build_metrics(judge_model: BedrockRuntimeModel) -> list:
    """
    Build list of DeepEval metrics for evaluation.

    This is a stub implementation that the AI agent will overwrite with
    project-specific metrics based on the evaluation requirements.

    Parameters:
        judge_model: Initialized BedrockRuntimeModel for LLM-as-a-judge metrics

    Returns:
        List of DeepEval metric instances

    Example — shipped metrics (prefer these):
        >>> from deepeval.metrics import ExactMatchMetric, AnswerRelevancyMetric
        >>> def build_metrics(judge_model):
        ...     return [
        ...         ExactMatchMetric(),
        ...         AnswerRelevancyMetric(model=judge_model),
        ...     ]

    Example — your own deterministic metric, when tolerance is required:
        >>> from deepeval.metrics import BaseMetric
        >>> class NormalizedMatch(BaseMetric):
        ...     def __init__(self, threshold: float = 1.0):
        ...         self.threshold = threshold
        ...     def measure(self, test_case) -> float:
        ...         # normalize() is yours to write AND to validate against this
        ...         # project's dataset — see the module docstring's 3 checks.
        ...         ok = (normalize(test_case.actual_output)
        ...               == normalize(test_case.expected_output))
        ...         self.score = 1.0 if ok else 0.0
        ...         self.success = self.score >= self.threshold
        ...         self.reason = "match" if ok else "differs after normalization"
        ...         return self.score
        ...     async def a_measure(self, test_case) -> float:
        ...         return self.measure(test_case)
        ...     def is_successful(self) -> bool:
        ...         return self.success
        ...     @property
        ...     def __name__(self):
        ...         return "Normalized Match"
    """
    # Stub implementation - agent will overwrite this
    return []
