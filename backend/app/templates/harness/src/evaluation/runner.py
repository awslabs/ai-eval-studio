# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluation runner with parallel generation and DeepEval integration.

This module orchestrates the full evaluation pipeline:
1. Load prompt and dataset
2. Generate outputs in parallel (ThreadPoolExecutor)
3. Score with DeepEval metrics
4. Retry null-scored metrics sequentially (no concurrency = no throttle)
5. Serialize results in DeepEval-native format with a data-quality summary

Key features:
- Parallel generation (max 5 workers) for faster evaluation
- AsyncConfig metric-scoring concurrency of ``EvalConfig.max_concurrent``
  when the config sets it, else 20; ``--max-concurrent`` overrides both.
  The value is sized against DeepEval's gather timeout rather than Bedrock
  throttling, which the adapters' retry decorator absorbs (see
  ``run_single_model``).
- ErrorConfig(ignore_errors=True) to continue on individual case failures
- Sequential retry on null-scored metrics with exponential backoff so
  transient throttling doesn't corrupt the run with silent nulls
- ``data_quality`` field on every model run reports generation and judge
  reliability separately: generation failures (target-model calls) and judge
  reliability (LLM-as-a-judge scoring) are distinct domains with different
  remedies, so the reliability verdict only covers the judge domain
- JSON progress events for platform integration (--json-progress flag)
- Cost estimation from token usage metadata
"""

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.evaluate.types import EvaluationResult
from deepeval.test_case import LLMTestCase

from src.config import EvalConfig
from src.evaluation.dataset_loader import format_user_message, load_test_cases
from src.metrics import build_metrics
from src.models.bedrock_mantle_mappers import MantleReasoningTruncated
from src.models.bedrock_mantle_model import BedrockMantleModel
from src.models.bedrock_runtime_model import BedrockRuntimeModel
from src.models.model_descriptors import MantleDescriptor, read_descriptors


# Placeholder tokens the renderer knows how to fill in the prompt template.
# Any ``{name}`` in the prompt file that isn't here will raise — better to
# fail loudly at render time than silently ship unfilled placeholders to the
# model.
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def render_prompt(template: str, case: LLMTestCase) -> str:
    """Render a prompt template against a single test case.

    Fills well-known placeholders from the ``LLMTestCase``:

    - ``{input}`` → ``case.input`` (the question)
    - ``{context}`` → ``case.context`` joined with ``\n---\n`` (RAG passages)
    - ``{retrieval_context}`` → ``case.retrieval_context`` joined the same way
    - ``{expected_output}`` → ``case.expected_output`` (for few-shot templates
      that want to show a worked example — normally not used at inference)

    Templates without any placeholder render unchanged. Templates with an
    unknown placeholder raise ``ValueError`` so the smoke eval fails loud
    rather than silently shipping an unfilled ``{foo}`` to the model.

    Args:
        template: The prompt text, possibly containing ``{name}`` placeholders.
        case: The test case providing substitution values.

    Returns:
        The fully-rendered prompt string.

    Raises:
        ValueError: If the template references a placeholder we don't
            know how to fill.
    """
    values: dict[str, str] = {
        "input": case.input or "",
        "expected_output": case.expected_output or "",
        "context": "\n---\n".join(case.context) if case.context else "",
        "retrieval_context": "\n---\n".join(case.retrieval_context) if case.retrieval_context else "",
    }

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise ValueError(
                f"Prompt template references unknown placeholder {{{name}}}. "
                f"Supported placeholders: {sorted(values)}."
            )
        return values[name]

    return _PLACEHOLDER_RE.sub(_sub, template)


def _is_generation_failure(case: LLMTestCase) -> bool:
    """Predicate: is this case a generation failure (not send-able to the judge)?

    Returns True when ``case.actual_output`` is None or an empty/whitespace-only
    string, OR ``(case.metadata or {}).get("error")`` is truthy. An empty
    actual_output can never pass the judge ('actual_output' cannot be empty), so
    scoring it is pure waste; this predicate is the fail-fast gate.

    Reasoning-truncated cases have actual_output="" and error set, so they are
    correctly classified as generation failures and skipped — their
    MANTLE_REASONING_TRUNCATED metadata flag is preserved so report renderers
    still show "Reasoning truncated".

    Args:
        case: The generated test case.

    Returns:
        True if the case should be excluded from evaluate() / retry.
    """
    actual = case.actual_output
    if actual is None or (isinstance(actual, str) and not actual.strip()):
        return True
    if (case.metadata or {}).get("error"):
        return True
    return False


class SystemicGenerationFailure(Exception):
    """Raised when a generation error will fail identically for every case.

    This exception signals that continuing the generation loop is pointless —
    the error (credential retrieval, access denial, model not found) will
    repeat for all remaining cases, and the run should abort with ONE error
    instead of building N identical failed test cases.
    """
    pass


def _is_systemic_generation_error(exc: Exception) -> bool:
    """Predicate: will this generation error fail identically for every case?

    Returns True when the exception is a systemic failure — one that will
    repeat for ALL cases because the root cause is independent of case content
    (credential retrieval, authorization, resource not found). Returning True
    triggers an abort: pending futures are cancelled and the run raises once
    instead of building 200+ identical failed cases.

    Classification reads **exception type and structured response code only**,
    never message prose. Message text is provider-controlled and changes
    without notice. If a future case genuinely has no structured signal, that
    is the moment to revisit — this function does NOT match English.

    Deliberate over-inclusion (fail-closed):
    botocore's ProcessProvider wraps *every* helper-subprocess failure in
    CredentialRetrievalError — a ModuleNotFoundError in the helper, an STS
    AccessDenied, fd exhaustion, or a blip reaching the ECS credential endpoint
    all arrive as the same type. This classifier therefore treats ALL
    CredentialRetrievalError as systemic and will abort a run whose credential
    failure was in fact transient. That over-inclusion is intentional: a
    credential that cannot be retrieved is not something a per-case retry
    repairs, the eval is re-dispatchable, and the alternative is the
    208-identical-failures outcome this abort exists to prevent. Do NOT narrow
    it by matching message text — that would swap a recorded, safe
    over-inclusion for an unrecorded dependency on provider-controlled prose.
    If per-cause discrimination is ever needed, the correct move is a
    structured signal from the helper (e.g. a distinct exit code), not prose
    parsing.

    Why this is kept in runner.py and NOT moved to aws_retry.py: aws_retry.py
    is byte-identical-mirrored between app/services/ and
    app/templates/harness/src/models/ (enforced by test_templates_parity.py),
    so editing it forces a mirrored edit. Additionally, aws_retry.py answers a
    different question (is this transient enough to retry?) from ours (will
    this fail identically for every case?). runner.py has NO service twin — so
    keeping the classifier here leaves the parity surface untouched.

    Args:
        exc: The exception raised during generation.

    Returns:
        True if the error is systemic and the run should abort.
    """
    exc_type_name = type(exc).__name__

    # Credential failures: botocore credential retrieval errors
    if exc_type_name in ("CredentialRetrievalError", "NoCredentialsError"):
        return True

    # Access denials / resource not found: ClientError with structured code
    if exc_type_name == "ClientError":
        try:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in (
                "AccessDeniedException",
                "UnrecognizedClientException",
                "InvalidSignatureException",
                "ResourceNotFoundException",
                "ValidationException",
            ):
                return True
        except (AttributeError, TypeError):
            # Malformed response — not systemic
            pass

    # Mantle authorization / model removal: typed failures from mantle model
    # Match by type name (not isinstance) because the template's import graph
    # must not gain a dependency on the mantle model module just for a name
    # check — this precedent already exists in aws_retry._is_retryable_mantle_error.
    if exc_type_name in ("MantleAuthorizationFailure", "ModelRemovedFromMantle"):
        return True

    # Everything else is per-case. In particular:
    # - Throttling (ClientError with ThrottlingException) is NOT systemic
    # - MantleReasoningTruncated is NOT systemic
    # - MantleNotInvocable (environmental-but-transient) is NOT systemic
    return False


_pricing_data: dict[str, Any] | None = None
_pricing_loaded = False


def _load_pricing() -> dict[str, dict[str, float]]:
    """Load src/pricing.json if available, caching the result.

    Returns:
        Dict mapping model IDs to pricing info, or empty dict on failure.
    """
    global _pricing_data, _pricing_loaded
    if _pricing_loaded:
        return _pricing_data or {}

    _pricing_loaded = True
    pricing_path = Path("src") / "pricing.json"
    if pricing_path.exists():
        try:
            raw = json.loads(pricing_path.read_text(encoding="utf-8"))
            _pricing_data = raw.get("models", {})
            return _pricing_data or {}
        except (json.JSONDecodeError, OSError):
            pass
    return {}


# Models we've already warned about missing pricing for — one warning per
# model per process, not one per case.
_warned_no_pricing: set[str] = set()

# Latch for the gating-unconfigured warning — once per harness process (one
# eval run), not once per case. Same shape as ``_warned_no_pricing``.
_warned_no_gating: set[str] = set()


def _warn_gating_unconfigured() -> None:
    """Announce, once per run, that NO gating metrics are configured.

    Why this needs a warning rather than staying silent: with no gating names,
    ``_gated_success`` falls back to DeepEval's ``TestResult.success``, which is
    all-metrics-pass. Every diagnostic metric then vetoes every case it fails —
    the exact failure the gating logic exists to prevent (measured live: a
    validity metric at threshold 1.0 zeroed an entire 832-case run while the
    target metric scored 68-88%).

    The trap is that "display-only" is a property of ``target_metric_scores``
    being READABLE, not a property of the metric. An empty, missing, or
    unparseable config silently promotes every metric to a gate, and the run
    still reports a number — just a wrong one, with no signal. This makes the
    degradation visible without changing what the run does.
    """
    if _warned_no_gating:
        return
    _warned_no_gating.add("warned")
    print(
        "⚠ WARNING: no gating metrics configured "
        "(EvalConfig.target_metric_scores is empty or unreadable).\n"
        "  Falling back to DeepEval's all-metrics aggregate, so EVERY metric "
        "gates case pass/fail —\n"
        "  a display-only metric can now zero the whole run. Case pass rates "
        "from this run may be\n"
        "  far lower than the target metric alone would give. Set "
        "target_metric_scores in src/config.py\n"
        "  to name the metrics that should decide pass/fail.",
        file=sys.stderr,
        flush=True,
    )


def _estimate_cost(usage: dict[str, Any], model_id: str) -> float | None:
    """Estimate generation cost from token usage — or ``None`` when unknown.

    ``src/pricing.json`` (written by ``validate_harness``, refreshed by
    ``refresh_pricing`` on model changes) is the ONLY pricing source. When a
    model has no entry — or the entry lacks input/output rates — the cost is
    UNKNOWN and this returns ``None``; it is never fabricated. A hardcoded
    fallback table used to live here and silently billed unknown models at
    stale or invented rates ($1/$5 per 1M), which is worse than admitting
    ignorance: downstream ranking and reports trusted numbers that were
    fiction. Run output carries ``null`` and the platform's
    ``pricing_available`` classification tells the report why.

    Cache accounting: Bedrock's Converse ``TokenUsage`` reports
    ``inputTokens``, ``cacheReadInputTokens``, and
    ``cacheWriteInputTokens`` as **separate** counts.
    ``inputTokens`` is already the standard-rate (non-cached) portion,
    so cache terms are **added on top** rather than subtracted from
    ``inputTokens``. When a model's pricing entry has no cache rate
    we fall back to the input rate, which keeps the formula
    equivalent to today's no-cache result and avoids silently
    under-billing cache tokens.

    Parameters:
        usage: Usage dict from Bedrock (inputTokens, outputTokens,
            cacheReadInputTokens, cacheWriteInputTokens, ...)
        model_id: Full Bedrock model ID

    Returns:
        Estimated cost in USD, or ``None`` when pricing for ``model_id`` is
        unavailable (unknown cost — reported as null, never invented).
    """
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)
    cache_read = usage.get("cacheReadInputTokens", 0) or 0
    cache_write = usage.get("cacheWriteInputTokens", 0) or 0

    pricing = _load_pricing()
    entry = pricing.get(model_id)
    if not entry or "input" not in entry or "output" not in entry:
        if model_id not in _warned_no_pricing:
            _warned_no_pricing.add(model_id)
            print(
                f"WARNING: no pricing for {model_id!r} in src/pricing.json — "
                "costs for this model will be reported as unknown (null). "
                "Run refresh_pricing (or re-run validate_harness) to fix.",
                file=sys.stderr,
            )
        return None

    input_price = entry["input"]
    output_price = entry["output"]
    # Missing CACHE rates fall back to the model's own input rate — an
    # approximation anchored to this model's real pricing (cache reads cost
    # less than input in practice, so this over-estimates slightly rather
    # than under-billing). Distinct from inventing base rates, which is
    # forbidden above.
    cache_read_price = entry.get("cache_read", input_price)
    cache_write_price = entry.get("cache_write", input_price)
    return (
        input_tokens * input_price
        + cache_read * cache_read_price
        + cache_write * cache_write_price
        + output_tokens * output_price
    ) / 1_000_000


def _log_metric_errors(model_id: str, eval_result: "EvaluationResult") -> None:
    """Summarize per-metric errors surfaced by DeepEval on stderr.

    When ``ErrorConfig(ignore_errors=True)`` is set (the default here),
    DeepEval catches metric exceptions per test case and records them on
    ``MetricData.error``. The final 2% pass rate then looks like a prompt
    regression when the real cause is often an expired AWS token, a judge-
    model quota, or a schema mismatch.

    Walks every test result's metric data, groups error strings by metric
    name, and prints ``ERROR summary`` lines to stderr when any exist. One
    line per (metric, unique error) with a count. Non-fatal: never raises.

    Parameters:
        model_id: The Bedrock model being evaluated (for log context).
        eval_result: DeepEval's EvaluationResult with per-case metric data.
    """
    try:
        # error_text -> {metric_name: count}
        grouped: dict[str, dict[str, int]] = {}
        for tr in eval_result.test_results or []:
            for md in getattr(tr, "metrics_data", None) or []:
                err = getattr(md, "error", None)
                if not err:
                    continue
                name = getattr(md, "name", "?") or "?"
                by_metric = grouped.setdefault(str(err), {})
                by_metric[name] = by_metric.get(name, 0) + 1

        if not grouped:
            return

        total = sum(sum(counts.values()) for counts in grouped.values())
        print(
            f"[metric-errors] model={model_id}: {total} metric errors "
            f"across {len(grouped)} unique message(s)",
            file=sys.stderr,
            flush=True,
        )
        for err_text, by_metric in grouped.items():
            # Collapse error to single line, keep it readable but bounded
            one_line = " ".join(str(err_text).split())[:400]
            breakdown = ", ".join(f"{n}×{c}" for n, c in sorted(by_metric.items()))
            print(
                f"[metric-errors]   {breakdown}: {one_line}",
                file=sys.stderr,
                flush=True,
            )
    except Exception as e:
        # Never let logging break the eval run
        print(f"[metric-errors] (summary failed: {e})", file=sys.stderr, flush=True)


class EvalRunner:
    """
    Orchestrates evaluation runs with parallel generation and DeepEval scoring.

    Parameters:
        config: Evaluation configuration with models, prompt version, etc.
        json_progress: If True, emit JSON progress events to stdout
        source: Whether this run came from an optimization loop ("optimization")
            or a manual eval ("manual"). Written into the result file so the
            UI can tag runs without relying on external state.
        max_concurrent_override: Optional override for DeepEval async metric
            concurrency. Wins over ``EvalConfig.max_concurrent`` (if set) and
            the default of 20. ``validate_harness`` passes 1 here
            so a misconfigured project's smoke can't itself trigger throttling.
    """

    def __init__(
        self,
        config: EvalConfig,
        json_progress: bool = False,
        source: str = "manual",
        max_concurrent_override: int | None = None,
    ):
        self.config = config
        self.json_progress = json_progress
        self.source = source
        self.max_concurrent_override = max_concurrent_override
        # Only TARGET metrics gate case pass/fail. DeepEval's own
        # ``TestResult.success`` is all-metrics-pass, which silently lets a
        # diagnostic/display-only metric veto every case (measured live:
        # a validity metric at threshold 1.0 zeroed an entire 832-case run
        # while the target metric scored 68-88%). The gating set is the
        # config's ``target_metric_scores`` keys — the same names the
        # convergence pipeline compares against — so "display-only" is an
        # enforced property of the aggregation, not a comment.
        self._gating_metric_names: frozenset[str] = frozenset(
            (getattr(config, "target_metric_scores", None) or {}).keys()
        )
        # Read mantle descriptors ONCE per run. The harness subprocess
        # runs with cwd=project_dir so ``Path(".")`` resolves to the
        # project root where ``src/model_descriptors.json`` lives.
        # Returns ``{}`` on a runtime-only project — the runner falls
        # through to the BedrockRuntimeModel path for every id.
        self._descriptors: dict[str, MantleDescriptor] = read_descriptors(Path("."))

    @staticmethod
    def _base_metric_name(name: str) -> str:
        """Strip a trailing `` [MetricType]`` suffix from a metric name.

        DeepEval's ``GEval.__name__`` (version-dependent) appends the metric
        type — ``"Answer Correctness [GEval]"`` — while the config's
        ``target_metric_scores`` keys carry the bare name the user chose.
        An exact-match gating comparison therefore never matches and EVERY
        case fails regardless of score (measured live: 0/416 on a run whose
        target metric scored 51-56%). Normalizing BOTH sides of every
        name comparison — and the serialized names the platform's
        convergence check reads — makes gating robust to either form, so
        DeepEval version drift in per-project venvs can't corrupt pass/fail
        again.
        """
        return re.sub(r"\s*\[[A-Za-z0-9 _-]+\]$", "", name or "")

    def _gated_success(self, test_result) -> bool:
        """Case pass/fail from GATING (target) metrics only.

        Diagnostic metrics still serialize with their own per-metric
        ``success`` for display, but they cannot veto the case. With no
        gating names configured there is nothing to separate targets from
        diagnostics, so every metric gates and the verdict is DeepEval's
        aggregate. When gating names exist but the row carries no gating
        metric at all, the case is NOT passing — an unscored target is
        never a pass (same philosophy as the reconciliation sentinels).
        """
        # getattr: smoke tests (and defensive subclasses) construct via
        # __new__ without __init__ — absent attr means "no gating config".
        gating_names = getattr(self, "_gating_metric_names", frozenset())
        if not gating_names:
            # Semantics unchanged — but say so, because this fallback silently
            # promotes every diagnostic metric to a gate. See
            # ``_warn_gating_unconfigured``.
            _warn_gating_unconfigured()
            return bool(test_result.success)
        gating_base = {self._base_metric_name(n) for n in gating_names}
        gating = [
            m for m in (test_result.metrics_data or [])
            if self._base_metric_name(m.name) in gating_base
        ]
        if not gating:
            return False
        return all(bool(m.success) for m in gating)

    def _make_model(
        self, model_id: str
    ) -> BedrockRuntimeModel | BedrockMantleModel:
        """Construct the right adapter for ``model_id`` by descriptor presence.

        Mantle ids (those with an entry in ``src/model_descriptors.json``)
        route through ``BedrockMantleModel`` against the descriptor's
        api/region. Runtime ids route through ``BedrockRuntimeModel``
        (Converse) using the project's ``aws_region``.

        The mantle adapter signs SigV4 against the descriptor's region
        — passing ``self.config.aws_region`` here would break the
        single-source-region invariant for cross-region
        models like ``xai.grok-4.3`` (us-west-2-only).

        Both adapters honor the same call contract:
        ``generate_with_system(system, user) -> (text, latency_ms, usage)``
        plus ``accumulated_usage`` and ``get_model_name()``. Drop-in
        replacement for every site that previously hardcoded
        ``BedrockRuntimeModel``.
        """
        descriptor = self._descriptors.get(model_id)
        if descriptor is not None:
            return BedrockMantleModel(
                descriptor,
                model_id,
                max_output_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                profile_name=self.config.aws_profile,
            )
        return BedrockRuntimeModel(
            model_id=model_id,
            region=self.config.aws_region,
            profile_name=self.config.aws_profile,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    def _emit_progress(self, event: str, **kwargs: Any) -> None:
        """Emit JSON progress event if json_progress is enabled."""
        if self.json_progress:
            print(json.dumps({"event": event, **kwargs}), file=sys.stdout, flush=True)

    def run(self) -> list[dict[str, Any]]:
        """
        Run evaluation for all configured models.

        Returns:
            List of serialized model results (DeepEval-native format with metadata)
        """
        # Load prompt inline (simple file read)
        prompt_path = Path("prompts") / f"{self.config.prompt_version}.md"
        system_prompt = prompt_path.read_text(encoding="utf-8")

        # Load test cases via agent-written module
        # Pass max_examples as positional to avoid keyword mismatch if the
        # agent named the parameter differently (e.g., max_cases)
        test_cases = load_test_cases(
            self.config.data_file,
            self.config.max_examples,
        )

        results = []
        for model_id in self.config.models:
            result = self.run_single_model(model_id, system_prompt, test_cases)
            results.append(result)

        return results

    def run_single_model(
        self,
        model_id: str,
        system_prompt: str,
        base_cases: list[LLMTestCase],
    ) -> dict[str, Any]:
        """
        Run evaluation for a single model: generate → score → serialize.

        Parameters:
            model_id: Full Bedrock model ID
            system_prompt: System prompt text
            base_cases: List of test cases from load_test_cases()

        Returns:
            Serialized result with DeepEval test results and metadata
        """
        # Route by descriptor: mantle ids → BedrockMantleModel,
        # runtime ids → BedrockRuntimeModel. See ``_make_model``.
        model = self._make_model(model_id)

        # Stage 1: Generate outputs in parallel
        eval_cases = self.generate_outputs(model, system_prompt, base_cases)

        # Partition evaluable vs generation-failed cases. Empty-output
        # cases can never pass the judge ('actual_output' cannot be empty),
        # so scoring them is pure waste. The failed cases still belong in
        # the output so the report/UI shows an honest total and the real
        # error, but they are NOT scored.
        evaluable_cases = [tc for tc in eval_cases if not _is_generation_failure(tc)]
        generation_failed_cases = [tc for tc in eval_cases if _is_generation_failure(tc)]

        # Stage 2: Build metrics and run DeepEval scoring
        judge_model_id = self.config.judge_model_id or model_id
        # Same descriptor-based routing for the judge — a mantle judge
        # (e.g. anthropic.claude-opus-4-8 on the mantle endpoint) must
        # not be smuggled through Converse.
        judge_model = self._make_model(judge_model_id)
        metrics = build_metrics(judge_model)

        # Descriptor-derived routing metadata. Mantle ids (those in
        # ``self._descriptors``) carry the descriptor's endpoint/region;
        # runtime ids implicitly route through Converse from the
        # project's ``aws_region``. Surfacing both means the SPA can
        # group/filter charts by endpoint/region without parsing the
        # raw model id (the source of truth is the
        # descriptor, not the id prefix).
        descriptor = self._descriptors.get(model_id)
        if descriptor is not None:
            endpoint = descriptor["endpoint"]
            region = descriptor["region"]
        else:
            endpoint = "runtime"
            region = self.config.aws_region

        hyperparams = {
            "model": model_id,
            "endpoint": endpoint,
            "region": region,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "prompt_version": self.config.prompt_version,
        }

        if not self.json_progress:
            print(f"\n--- Evaluating metrics for {model_id} ---")

        # Run ``evaluate()`` in a thread pool with a daemon monitor that
        # polls DeepEval's ``global_test_run_manager.test_run.test_cases``
        # length every 2s and emits ``scoring_progress`` so the dashboard
        # shows per-case advancement instead of a frozen "Scoring…" state
        # for the duration of the whole metric pass.
        #
        # Multi-tenant note: ``global_test_run_manager`` is process-local.
        # The harness runs as a per-dispatch subprocess so each tenant's
        # poll reads its own singleton; no cross-tenant pollution.
        import concurrent.futures
        import threading
        from deepeval.test_run import global_test_run_manager

        total_cases = len(evaluable_cases)

        # Concurrency tuned against DeepEval's gather-timeout, NOT
        # Bedrock throttling. The throttling concern is handled by the
        # ``@_bedrock_retry`` decorator in our custom ``BedrockRuntimeModel``
        # (4 attempts by default, waiting 1s, 2s and 4s plus jitter between
        # them — see ``src/models/aws_retry.py``). The actual binding
        # constraint at the runner level is DeepEval's
        # ``get_gather_timeout_seconds()`` which equals
        # ``DEEPEVAL_PER_TASK_TIMEOUT_SECONDS`` (180s default) +
        # ``DEEPEVAL_TASK_GATHER_BUFFER_SECONDS`` (~10-60s auto). When
        # the gather timeout fires, every still-queued task gets
        # cancelled and DeepEval drops those cases from
        # ``test_results`` — which shows up as different case totals per
        # model ("model A 200, model B 198") until reconciliation re-scores
        # the missing cases. At 5-concurrent against 210 cases × 2
        # metrics, ~840s of judge time only got ~200s before the deadline
        # fired.
        # ``backend/app/services/harness_runner.py:_project_subprocess_env``
        # bumps the outer timeout to 600s; 20-concurrent at ~12s/call
        # ≈ 250s of judge work — comfortably inside that budget.
        # 20 rather than a lower value: nulls seen at 20 were
        # gather-timeout cancellations, not Bedrock throttling. Every
        # null record had empty ``error`` text (a caught
        # ThrottlingException under ``ignore_errors=True`` would have
        # populated ``MetricData.error``) and the coordinator log for
        # those runs had no ``ThrottlingException``. Lower concurrency
        # made the timeout cancel MORE cases (86 at 5 vs ~59 at 20)
        # because the queue drained slower.
        # Projects can override via ``EvalConfig.max_concurrent``;
        # ``validate_harness`` passes 1 for the smoke check.
        if self.max_concurrent_override is not None:
            max_concurrent = self.max_concurrent_override
        else:
            max_concurrent = getattr(self.config, "max_concurrent", 20)

        # Observability: surface concurrency on the wire so the
        # dashboard / coordinator log can confirm what actually ran
        # without re-deriving from config + override chain.
        self._emit_progress(
            "scoring_start",
            model=model_id,
            total=total_cases,
            max_concurrent=max_concurrent,
        )
        print(
            f"[scoring] model={model_id} concurrency={max_concurrent} "
            f"cases={total_cases}",
            file=sys.stderr,
            flush=True,
        )

        def _run_eval() -> EvaluationResult:
            return evaluate(
                test_cases=evaluable_cases,
                metrics=metrics,
                hyperparameters=hyperparams,
                async_config=AsyncConfig(max_concurrent=max_concurrent),
                display_config=DisplayConfig(
                    show_indicator=not self.json_progress,
                    print_results=not self.json_progress,
                ),
                error_config=ErrorConfig(ignore_errors=True),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_eval)
            stop_event = threading.Event()

            # Capture the test_run length BEFORE this model's evaluate()
            # appends to it. DeepEval reuses ``global_test_run_manager.test_run``
            # across calls — model 2's evaluate() doesn't reset the list,
            # so naive ``len(test_cases)`` reads ``cases_from_model_1 +
            # cases_from_model_2_so_far``. Subtracting the baseline gives
            # us the per-model delta, which is what the dashboard expects.
            try:
                _baseline_run = global_test_run_manager.test_run
                baseline = len(_baseline_run.test_cases) if _baseline_run is not None else 0
            except Exception:
                baseline = 0

            def _monitor() -> None:
                last_count = 0
                while not stop_event.is_set():
                    stop_event.wait(timeout=2.0)
                    if stop_event.is_set():
                        break
                    try:
                        test_run = global_test_run_manager.test_run
                        if test_run is not None:
                            raw = len(test_run.test_cases)
                            # ``max(0, ...)`` so a DeepEval mid-evaluate
                            # reset (rare, but possible) doesn't push us
                            # negative for one tick.
                            current = max(0, raw - baseline)
                            if current != last_count:
                                last_count = current
                                self._emit_progress(
                                    "scoring_progress",
                                    model=model_id,
                                    current=current,
                                    total=total_cases,
                                )
                    except Exception:
                        pass

            monitor = threading.Thread(
                target=_monitor, daemon=True, name="scoring-progress-monitor",
            )
            monitor.start()
            try:
                eval_result: EvaluationResult = future.result()
            finally:
                stop_event.set()
                monitor.join(timeout=5.0)

        # Summarize per-metric errors. DeepEval's ignore_errors=True swallows
        # per-case metric exceptions into MetricData.error, so a 100% failure
        # that looks like "bad prompt" often turns out to be an AWS token
        # expiry or judge-model quota error. Emit a compact summary to stderr
        # so the platform's harness stderr drain (and standalone users) see
        # the underlying cause instead of just a 2% pass rate.
        _log_metric_errors(model_id, eval_result)

        # Stage 2.4: Reconcile missing cases. ``evaluate()`` can return
        # FEWER ``test_results`` than ``len(eval_cases)`` — when its
        # internal ``asyncio.gather`` hits the gather-timeout it cancels
        # any task that hasn't started yet, and those cases never
        # produce a ``test_results`` entry at all. Different models in
        # the same run can drop different counts, which is why the
        # dashboard shows different "total" per model. We re-score
        # missing cases sequentially before the null-metric retry so
        # the retry pass operates on the full dataset.
        self._reconcile_missing_cases(
            model_id, evaluable_cases, eval_result, judge_model,
        )

        # Stage 2.5: Retry null-scored / errored metrics sequentially.
        # ``evaluate()`` runs N metrics × M cases concurrently; sustained
        # bursts hit Bedrock throttling and DeepEval swallows the
        # ThrottlingException into ``MetricData.error`` with score=None.
        # We re-issue the failures one at a time (no concurrency = no
        # throttling) with exponential backoff so transient rate limits
        # don't silently corrupt the run.
        self._retry_null_metrics(model_id, eval_result, judge_model)

        # Append synthetic TestResult rows for generation-failed cases.
        # These cases are NOT scored (empty actual_output can never pass
        # the judge), but they still belong in the output so the report/UI
        # shows an honest total and the real error.
        from deepeval.evaluate.types import TestResult as _TR
        for tc in generation_failed_cases:
            case_index = (tc.metadata or {}).get("case_index", -1)
            synthetic_result = _TR(
                name=f"test_case_{case_index}",
                success=False,
                metrics_data=[],  # No scored metrics
                conversational=False,
                multimodal=False,
                input=tc.input,
                actual_output=tc.actual_output,
                expected_output=tc.expected_output,
                context=tc.context,
                retrieval_context=tc.retrieval_context,
                metadata={
                    **(tc.metadata or {}),
                    "GENERATION_FAILED": True,
                },
            )
            eval_result.test_results.append(synthetic_result)

        # Count passed cases for progress reporting. Progress counts must
        # agree with the serialized run file, so gate on TARGET metrics
        # only — the same rule as _serialize_result.
        passed = sum(1 for tr in eval_result.test_results if self._gated_success(tr))
        total = len(eval_result.test_results)
        self._emit_progress("scoring_complete", model=model_id, passed=passed, total=total)

        # Judge cost: the shared judge model accumulated token usage
        # across the main pass, reconciliation, and retry. Convert it to
        # USD with the same pricing path generation cost uses, keyed by
        # the judge model id (which may differ from the generator).
        judge_cost_usd = _estimate_cost(
            judge_model.accumulated_usage, judge_model_id,
        )

        # Stage 3: Serialize with cost estimation. ``total_input_cases``
        # is what the user submitted (post-generation, pre-evaluate);
        # the serializer compares it against ``len(test_results)`` to
        # compute ``data_quality.dropped_count`` for reconciliation
        # accounting.
        result = self._serialize_result(
            model_id, eval_result, hyperparams,
            total_input_cases=len(eval_cases),
            judge_cost_usd=judge_cost_usd,
        )

        # Emit complete event after serialization
        self._emit_progress("complete", model=model_id, passed=passed, total=total)

        return result

    def _reconcile_missing_cases(
        self,
        model_id: str,
        eval_cases: list[LLMTestCase],
        eval_result: EvaluationResult,
        judge_model: "BedrockRuntimeModel | BedrockMantleModel",
    ) -> None:
        """Recover cases that ``evaluate()`` silently dropped from test_results.

        DeepEval's async ``evaluate()`` wraps every test-case task in an
        ``asyncio.gather`` with a global gather-timeout. When that fires,
        any task that hadn't started yet is cancelled before its
        ``finally`` block runs — so the case never reaches
        ``test_results.append()`` and the result list is shorter than
        the input. Different models in the same run hit this at
        different counts, which surfaces in the UI as "model A has 200
        cases, model B has 198."

        Identity uses the ``case_index`` we stamp on every
        ``LLMTestCase.metadata`` in :meth:`generate_outputs` — index is
        reliable across duplicate inputs (RAG paraphrase pairs) and
        survives DeepEval's ``create_test_result`` pass-through.

        Each missing case gets up to 3 sequential attempts, pausing 1s and
        then 2s between them. A case that still can't be scored after all
        attempts gets a sentinel ``TestResult`` appended with every metric at
        ``score=None`` and ``error="reconciliation failed"`` — surfaced
        to the optimizer via ``data_quality.unrecovered_count`` so the
        loop halts with ``judge_unreliable``. We never fail-open by
        marking a case as success it didn't actually pass.

        Args:
            model_id: Generator model id (for logs only).
            eval_cases: Original test cases passed to ``evaluate()``.
            eval_result: DeepEval result; ``test_results`` is mutated
                in place so the rest of the pipeline (retry,
                serializer) sees the full dataset.
            judge_model: The shared judge BedrockRuntimeModel created by
                :meth:`run_single_model`. Reused across the main pass,
                reconciliation, and retry so all judge token usage
                accrues into one per-run accumulator. Metric objects are
                still rebuilt fresh per attempt (they're stateful); only
                the stateless model is shared.
        """
        # Build the present-index set from result metadata. ``api_test_case``
        # carries metadata through unchanged, so what we stamped at
        # generation time is what we read back here.
        present_indices: set[int] = set()
        for tr in eval_result.test_results or []:
            md = getattr(tr, "metadata", None) or {}
            idx = md.get("case_index")
            if isinstance(idx, int):
                present_indices.add(idx)

        # Cases that need reconciliation: every input case whose stamped
        # index is missing from the result list.
        missing: list[tuple[int, LLMTestCase]] = []
        for tc in eval_cases:
            md = getattr(tc, "metadata", None) or {}
            idx = md.get("case_index")
            if isinstance(idx, int) and idx not in present_indices:
                missing.append((idx, tc))

        if not missing:
            return

        # Import here so unit tests can stub ``MetricData`` without
        # paying the real DeepEval import in the module-level
        # ``test_harness_template_smoke`` test loader.
        from deepeval.test_run.api import MetricData
        from deepeval.evaluate.types import TestResult as _TR

        print(
            f"[reconcile] model={model_id}: {len(missing)} case(s) missing "
            f"from evaluate() — recovering sequentially",
            file=sys.stderr,
            flush=True,
        )

        recovered = 0
        sentinel = 0
        for case_idx, tc in missing:
            scored_metrics: list = []
            success_flags: list[bool] = []
            for attempt in range(3):
                # Backoff applies BETWEEN attempts only — the first try
                # fires immediately. With many dropped cases (e.g. 90+)
                # a leading sleep would cost ~10 minutes of pure
                # deadtime before any reconciliation work begins.
                if attempt > 0:
                    time.sleep(2 ** (attempt - 1))  # 1s before #1, 2s before #2
                # Fresh metrics per attempt — ``BaseMetric`` carries
                # state from the last ``measure()`` call.
                fresh_metrics = build_metrics(judge_model)
                if not fresh_metrics:
                    break
                try:
                    for metric in fresh_metrics:
                        metric.measure(tc)
                    if all(m.score is not None for m in fresh_metrics):
                        scored_metrics = fresh_metrics
                        break
                except Exception as e:
                    print(
                        f"[reconcile] case={case_idx} attempt={attempt + 1} "
                        f"failed: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

            if scored_metrics:
                # Recovered. Build proper MetricData entries. Compute
                # success once per metric (from explicit success or
                # score-vs-threshold) and reuse for both the wire-side
                # ``MetricData.success`` and the test-level aggregate
                # — bool(None) collapses to False and would force
                # test_success=False even when scores pass threshold.
                metrics_data = []
                for metric in scored_metrics:
                    threshold = metric.threshold
                    if metric.success is not None:
                        metric_success = bool(metric.success)
                    elif metric.score is not None:
                        metric_success = metric.score >= threshold
                    else:
                        metric_success = False
                    metrics_data.append(MetricData(
                        # Base name (suffix stripped): the platform's
                        # per-metric aggregation and convergence check
                        # compare against the bare config key.
                        name=self._base_metric_name(metric.__name__),
                        threshold=threshold,
                        success=metric_success,
                        score=metric.score,
                        reason=getattr(metric, "reason", None),
                        error=None,
                    ))
                    # Only TARGET metrics gate the case — a diagnostic
                    # metric must not veto a reconciled row either. With no
                    # gating names configured every metric gates, same as
                    # _gated_success. Both sides base-normalized (see
                    # _base_metric_name).
                    _gating_names = getattr(self, "_gating_metric_names", frozenset())
                    if not _gating_names or self._base_metric_name(metric.__name__) in {
                        self._base_metric_name(n) for n in _gating_names
                    }:
                        success_flags.append(metric_success)
                test_success = all(success_flags) if success_flags else False
                recovered += 1
            else:
                # Sentinel — couldn't recover. Stamp every metric at
                # score=None with an explicit error so downstream code
                # treats the case as unscored, NOT as a passing zero.
                #
                # ``build_metrics`` may itself throw here (e.g. a 4th
                # throttle event) — guard so the whole reconciliation
                # loop doesn't crash and lose the rest of the missing
                # cases. When the metric list is unavailable we still
                # emit a single placeholder MetricData so the case
                # contributes to ``unrecovered_count`` instead of
                # silently slipping past the per-metric counter in
                # ``_serialize_result``.
                try:
                    fresh_metrics = build_metrics(judge_model)
                except Exception as e:
                    print(
                        f"[reconcile] case={case_idx} sentinel-build "
                        f"failed: {e} — emitting placeholder metric",
                        file=sys.stderr,
                        flush=True,
                    )
                    fresh_metrics = []

                if not fresh_metrics:
                    metrics_data = [MetricData(
                        name="reconciliation_sentinel",
                        threshold=0.5,
                        success=False,
                        score=None,
                        reason=None,
                        error=(
                            "case dropped by evaluate() — reconciliation "
                            "failed (build_metrics unavailable)"
                        ),
                    )]
                else:
                    metrics_data = [
                        MetricData(
                            name=m.__name__,
                            threshold=m.threshold,
                            success=False,
                            score=None,
                            reason=None,
                            error="case dropped by evaluate() — reconciliation failed",
                        )
                        for m in fresh_metrics
                    ]
                test_success = False
                sentinel += 1
                print(
                    f"[reconcile] case={case_idx} unrecoverable after 3 "
                    f"attempts — sentinel inserted",
                    file=sys.stderr,
                    flush=True,
                )

            # Append the synthesized TestResult so the rest of the
            # pipeline sees a full dataset.
            new_tr = _TR(
                name=f"test_case_{case_idx}",
                success=test_success,
                metrics_data=metrics_data,
                conversational=False,
                multimodal=False,
                input=tc.input,
                actual_output=tc.actual_output,
                expected_output=tc.expected_output,
                context=tc.context,
                retrieval_context=tc.retrieval_context,
                metadata={**(tc.metadata or {}), "reconciled": True},
            )
            eval_result.test_results.append(new_tr)

        print(
            f"[reconcile] model={model_id}: {len(missing)} missing → "
            f"{recovered} recovered, {sentinel} sentinel",
            file=sys.stderr,
            flush=True,
        )

    def _retry_null_metrics(
        self,
        model_id: str,
        eval_result: EvaluationResult,
        judge_model: "BedrockRuntimeModel | BedrockMantleModel",
    ) -> None:
        """Re-score metrics that came back null/errored in the parallel pass.

        Mutates ``eval_result.test_results[*].metrics_data[*]`` in place,
        replacing successful retries with their fresh score/success/reason
        and leaving stubborn failures alone (their ``score`` stays None
        and ``error`` is preserved). The serializer reads these back and
        stamps a ``retried`` flag on each repaired record.

        Detection: a metric is considered "needs retry" when ``error`` is
        set or ``score`` is None. Both come from the same DeepEval code
        path under ``ignore_errors=True`` — checking both is cheap and
        survives schema edges.

        Backoff: up to 3 attempts, pausing 1s and then 2s between them.
        Three attempts is enough for transient throttling without dragging
        out a fully-broken run.

        Args:
            model_id: Generator model id (for logs only).
            eval_result: DeepEval result from the parallel evaluate() pass.
            judge_model: The shared judge BedrockRuntimeModel from
                :meth:`run_single_model`, reused so retry judge usage
                accrues into the same per-run accumulator. The metric
                list is still rebuilt fresh per attempt below (metrics
                are stateful); only the stateless model is shared.
        """
        # Collect (case_idx, metric_idx) pairs that need retry.
        targets: list[tuple[int, int]] = []
        for case_idx, tr in enumerate(eval_result.test_results or []):
            for m_idx, md in enumerate(getattr(tr, "metrics_data", None) or []):
                if md.error is not None or md.score is None:
                    targets.append((case_idx, m_idx))

        if not targets:
            return

        print(
            f"[retry] model={model_id}: {len(targets)} null/errored metric(s) "
            f"after parallel pass — retrying sequentially",
            file=sys.stderr,
            flush=True,
        )

        # Rebuild the metric list fresh per attempt below. ``BaseMetric``
        # instances are stateful across calls (last measure() leaves
        # score/reason on self), so we MUST NOT reuse the metric objects
        # from the parallel pass — they carry whatever the final case
        # left behind. The judge *model* is shared (stateless apart from
        # its usage accumulator) so judge cost totals across all phases.
        for case_idx, m_idx in targets:
            tr = eval_result.test_results[case_idx]
            md = tr.metrics_data[m_idx]
            tc = LLMTestCase(
                input=tr.input,
                actual_output=tr.actual_output,
                expected_output=tr.expected_output,
                context=tr.context if isinstance(tr.context, list) else None,
                retrieval_context=tr.retrieval_context if isinstance(tr.retrieval_context, list) else None,
            )
            for attempt in range(3):
                # Backoff applies BETWEEN attempts only — first try
                # fires immediately. (Previously slept 1s before
                # attempt 0, which costs ~10 minutes on a run with
                # 90+ null-scored metrics.)
                if attempt > 0:
                    time.sleep(2 ** (attempt - 1))  # 1s before #1, 2s before #2
                # Fresh metric per attempt — see stateful-metric note above.
                # Index into the rebuilt list by position; ``build_metrics``
                # is project-authored and treated as deterministic for a
                # given judge_model.
                fresh_metrics = build_metrics(judge_model)
                if m_idx >= len(fresh_metrics):
                    # Defensive: build_metrics returned fewer metrics than
                    # the original pass produced. Skip — we can't retry
                    # what we can't reconstruct.
                    break
                metric = fresh_metrics[m_idx]
                try:
                    metric.measure(tc)
                except Exception as e:
                    print(
                        f"[retry] case={case_idx} metric={getattr(metric, 'name', '?')} "
                        f"attempt={attempt + 1} failed: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                if metric.score is not None:
                    # Repair in place. Keep the original ``threshold`` and
                    # ``name`` from the DeepEval record; overwrite the
                    # score/success/reason/error from the fresh measure.
                    md.score = metric.score
                    md.success = bool(metric.success) if metric.success is not None else (metric.score >= md.threshold)
                    md.reason = metric.reason
                    md.error = None
                    break

        # After all retries: refresh ``test_result.success`` since metric
        # success values may have changed. DeepEval defines test success
        # as "all metrics passed" — re-aggregate from the (possibly
        # repaired) metrics_data.
        for tr in eval_result.test_results or []:
            md_list = getattr(tr, "metrics_data", None) or []
            if not md_list:
                continue
            # Test passes only when every metric has a definite True success.
            # A None (still-unscored) metric leaves success=False so the
            # case isn't silently counted as passed.
            tr.success = all(md.success is True for md in md_list)

    def generate_outputs(
        self,
        model: "BedrockRuntimeModel | BedrockMantleModel",
        system_prompt: str,
        base_cases: list[LLMTestCase],
    ) -> list[LLMTestCase]:
        """
        Generate model outputs in parallel using ThreadPoolExecutor.

        Uses format_user_message() from dataset_loader to build the user prompt
        for each test case. Runs up to 5 workers in parallel.

        Parameters:
            model: Initialized adapter — either BedrockRuntimeModel
                (Converse) or BedrockMantleModel (raw HTTPS + SigV4).
                Both honor ``generate_with_system(system, user) ->
                (text, latency_ms, usage)``.
            system_prompt: System prompt text
            base_cases: List of LLMTestCase objects from load_test_cases()

        Returns:
            List of LLMTestCase objects with generated outputs and metadata
        """
        eval_cases = []

        if not self.json_progress:
            print(f"\n--- Generating outputs with {model.get_model_name()} ---")

        # Decide where per-case variables live:
        # - If the prompt template contains ``{input}``/``{context}`` etc., the
        #   template was authored for per-case rendering. Render it for each
        #   case and send as the USER message, with a neutral system prompt
        #   (Bedrock rejects empty ``system`` entries).
        # - Otherwise treat the template as a static system prompt and use
        #   ``format_user_message(case)`` for the user message (classic
        #   non-RAG pattern).
        template_has_placeholders = bool(_PLACEHOLDER_RE.search(system_prompt))
        neutral_system_prompt = "You are a helpful assistant."

        # ThreadPoolExecutor for parallel generation
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all generation tasks
            future_to_idx = {}
            for idx, case in enumerate(base_cases):
                if template_has_placeholders:
                    rendered_system = neutral_system_prompt
                    rendered_user = render_prompt(system_prompt, case)
                else:
                    rendered_system = system_prompt
                    rendered_user = format_user_message(case)
                future = executor.submit(
                    model.generate_with_system,
                    rendered_system,
                    rendered_user,
                )
                future_to_idx[future] = idx

            # Collect results as they complete (counter for monotonic progress)
            completed_count = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                case = base_cases[idx]

                try:
                    output, latency_ms, usage = future.result()

                    if not self.json_progress:
                        print(f"  [{idx + 1}/{len(base_cases)}] Generated in {latency_ms:.0f}ms")

                    completed_count += 1
                    self._emit_progress(
                        "generating",
                        model=model.get_model_name(),
                        current=completed_count,
                        total=len(base_cases),
                    )

                    # Build LLMTestCase with metadata — preserve context
                    # and retrieval_context from the original case (needed by
                    # RAG metrics like Faithfulness)
                    tc = LLMTestCase(
                        input=case.input,
                        actual_output=output,
                        expected_output=case.expected_output,
                        context=case.context,
                        retrieval_context=case.retrieval_context,
                        metadata={
                            **(case.metadata or {}),
                            # ``case_index`` is the stable identity that
                            # ``_reconcile_missing_cases`` matches on.
                            # Don't drop it on the failed path below — we
                            # need it to detect both real drops AND
                            # generation-failure rows.
                            "case_index": idx,
                            "latency_ms": latency_ms,
                            "input_tokens": usage.get("inputTokens", 0),
                            "output_tokens": usage.get("outputTokens", 0),
                            "total_tokens": usage.get("totalTokens", 0),
                            "cache_read_input_tokens": usage.get("cacheReadInputTokens", 0),
                            "cache_write_input_tokens": usage.get("cacheWriteInputTokens", 0),
                            "generation_cost": _estimate_cost(usage, model.get_model_name()),
                        },
                    )
                    eval_cases.append((idx, tc))

                except MantleReasoningTruncated as e:
                    # Mantle Responses returned status='incomplete' with
                    # reason='max_output_tokens' — the model ran the
                    # token budget out before emitting any text. Surface
                    # the typed signal in case metadata so downstream
                    # report renderers can show "Reasoning truncated"
                    # rather than a generic error. NOT silently retried
                    # with a larger cap — the user raises
                    # ``max_output_tokens`` and re-runs.
                    if not self.json_progress:
                        print(
                            f"  [{idx + 1}/{len(base_cases)}] "
                            f"Reasoning truncated: {e}"
                        )
                    # Always emit the stderr line under json_progress so
                    # harness_runner._drain_stderr forwards it to the
                    # coordinator/CloudWatch log. The platform always runs
                    # --json-progress, so this is the ONLY visibility.
                    print(
                        f"[generation-error] case {idx + 1} "
                        f"model={model.get_model_name()} "
                        f"error={type(e).__name__}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    tc = LLMTestCase(
                        input=case.input,
                        actual_output="",
                        expected_output=case.expected_output,
                        context=case.context,
                        retrieval_context=case.retrieval_context,
                        metadata={
                            **(case.metadata or {}),
                            "case_index": idx,
                            "MANTLE_REASONING_TRUNCATED": True,
                            "error": str(e),
                        },
                    )
                    eval_cases.append((idx, tc))

                except Exception as e:
                    # Check if this is a systemic error that will fail for
                    # every case. If so, cancel pending futures and raise once
                    # instead of building 200+ identical failed test cases.
                    if _is_systemic_generation_error(e):
                        # Cancel all pending futures — Future.cancel() returns
                        # False for already-running futures and True for queued
                        # ones, so the bound is "cases already in flight"
                        # (typically max_workers at most, verified 6-18 under
                        # contention at max_workers=5).
                        cancelled_count = sum(
                            1 for f in future_to_idx.keys() if f.cancel()
                        )

                        # Emit ONE systemic-error stderr line marking it as
                        # systemic so harness_runner._drain_stderr forwards it.
                        print(
                            f"[generation-error] SYSTEMIC: case {idx + 1} "
                            f"model={model.get_model_name()} "
                            f"error={type(e).__name__} — cancelled {cancelled_count} "
                            f"pending cases (will fail identically for all)",
                            file=sys.stderr,
                            flush=True,
                        )

                        # Raise immediately, bypassing the per-case append path.
                        # The raise propagates through run_single_model → run →
                        # run_eval.py's except block, which emits a single
                        # {"event":"error"} line and exits non-zero.
                        # harness_runner.py turns non-zero exit into one
                        # HARNESS_SUBPROCESS_FAILED ResourceError with
                        # stderr_tail, published via tracker.fail_job.
                        raise SystemicGenerationFailure(
                            f"Systemic generation failure for model "
                            f"{model.get_model_name()} at case {idx + 1}: "
                            f"{type(e).__name__}: {e}"
                        ) from e

                    # Per-case failure: build a failed test case as today
                    if not self.json_progress:
                        print(f"  [{idx + 1}/{len(base_cases)}] Error: {e}")
                    # Always emit the stderr line under json_progress so
                    # harness_runner._drain_stderr forwards it to the
                    # coordinator/CloudWatch log. The platform always runs
                    # --json-progress, so this is the ONLY visibility.
                    print(
                        f"[generation-error] case {idx + 1} "
                        f"model={model.get_model_name()} "
                        f"error={type(e).__name__}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )

                    # Create failed test case — preserve context for RAG metrics
                    tc = LLMTestCase(
                        input=case.input,
                        actual_output="",
                        expected_output=case.expected_output,
                        context=case.context,
                        retrieval_context=case.retrieval_context,
                        metadata={
                            **(case.metadata or {}),
                            "case_index": idx,
                            "error": str(e),
                        },
                    )
                    eval_cases.append((idx, tc))

        # Sort by original index to maintain order
        eval_cases.sort(key=lambda x: x[0])

        # Emit a generation summary when failures exist
        all_cases = [tc for _, tc in eval_cases]
        failed_cases = [c for c in all_cases if _is_generation_failure(c)]
        if failed_cases:
            first_error = (failed_cases[0].metadata or {}).get("error", "(no message)")
            print(
                f"[generation] {len(failed_cases)}/{len(all_cases)} cases failed generation; "
                f"first error: {first_error}",
                file=sys.stderr,
                flush=True,
            )

        return all_cases

    def _serialize_result(
        self,
        model_id: str,
        eval_result: EvaluationResult,
        hyperparams: dict[str, Any],
        total_input_cases: int,
        judge_cost_usd: float | None = 0.0,
    ) -> dict[str, Any]:
        """
        Serialize evaluation result in DeepEval-native format.

        Parameters:
            model_id: Full Bedrock model ID
            eval_result: DeepEval evaluation result
            hyperparams: Hyperparameters dict
            total_input_cases: Cases submitted to ``evaluate()``. REQUIRED:
                it is the only number that knows how big the run was
                supposed to be, so deriving it from ``test_results``
                instead would report a run that lost cases as a smaller
                run that lost none. When ``_reconcile_missing_cases`` ran
                successfully every input case has a corresponding
                ``test_result`` (real or sentinel) and this equals
                ``len(eval_result.test_results)``.
            judge_cost_usd: Total USD spent on judge (LLM-as-a-judge)
                scoring for this model, derived by the caller from the
                shared judge model's accumulated token usage. Defaults
                to 0.0 so smoke-check callers that have no judge model
                still serialize cleanly. DeepEval never reports this for
                a custom (non-native) model, so the value is captured
                out-of-band on the BedrockRuntimeModel wrapper.

        Returns:
            Serialized result dict matching DeepEval format
        """
        # Structural short-name derivation — strip leading dot-segments
        # until the first segment with a digit, then strip trailing
        # version + date tags. Lives in ``models/model_id_short.py`` so
        # this template stays in step with ``app.services.model_id_short``
        # (parity test enforces byte-equality).
        from src.models.model_id_short import short_model_id
        short_name = short_model_id(model_id)

        # Serialize test results
        results = []
        total_generation_cost = 0.0
        # Counters for the run-level data-quality summary, split by domain:
        #
        # JUDGE DOMAIN (LLM-as-a-judge scoring): Every metric evaluation is
        # one ``(case, metric)`` pair. ``null_after_retry`` is the count
        # still without a score after the sequential retry phase ran.
        # These feed the judge-only ``reliability`` verdict.
        total_metric_evals = 0
        null_after_retry = 0
        errored_after_retry = 0
        # Reconciliation accounting (see ``_reconcile_missing_cases``).
        # ``dropped_count`` = cases that DeepEval lost from test_results
        # and we recovered (or sentinel'd) sequentially.
        # ``unrecovered_count`` = the subset of those that still couldn't
        # be scored after 3 attempts and are now sentinel rows. Any
        # non-zero ``unrecovered_count`` marks the judge domain as degraded
        # because silent drops are unacceptable signal.
        dropped_count = 0
        unrecovered_count = 0
        #
        # GENERATION DOMAIN (target-model calls): ``generation_failed_count``
        # tracks cases excluded from evaluate() because they had empty/errored
        # generation output (can never pass the judge). The backend owns the
        # tolerance decision; the harness only reports the count and ratio.
        generation_failed_count = 0

        # True when ANY case's cost is unknown (no pricing for the model).
        # The run-level total is then null — summing the known subset would
        # present a partial number as the whole spend.
        generation_cost_unknown = False

        for test_result in eval_result.test_results:
            # TestResult fields are directly on the object (not nested under .test_case)
            metadata = test_result.metadata or {}
            gen_cost = metadata.get("generation_cost", 0.0)
            if gen_cost is None:
                generation_cost_unknown = True
            else:
                total_generation_cost += gen_cost

            is_reconciled = bool(metadata.get("reconciled"))
            if is_reconciled:
                dropped_count += 1

            # Skip metric-counting for generation-failed cases. These
            # rows carry empty metrics_data and render with their preserved
            # error, but they don't contribute to the null/errored counters
            # (they were never scored).
            is_generation_failed = bool(metadata.get("GENERATION_FAILED"))
            if is_generation_failed:
                generation_failed_count += 1
                # Append the result row (so it renders) but skip metric loop
                results.append({
                    "name": test_result.name or f"test_{len(results)}",
                    "success": test_result.success,
                    "input": test_result.input,
                    "actual_output": test_result.actual_output,
                    "expected_output": test_result.expected_output,
                    "context": test_result.context if isinstance(test_result.context, list) else None,
                    "retrieval_context": test_result.retrieval_context if isinstance(test_result.retrieval_context, list) else None,
                    "additional_metadata": metadata,
                    "metrics_data": [],
                })
                continue

            # Serialize metrics
            metrics_data = []
            row_unrecovered = False
            for metric_result in test_result.metrics_data:
                total_metric_evals += 1
                if metric_result.score is None:
                    null_after_retry += 1
                if metric_result.error is not None:
                    errored_after_retry += 1
                # A reconciled row whose metric is still null carries
                # the reconciliation-failed sentinel error — count the
                # whole case as unrecovered (don't double-count metrics
                # within the same row).
                if is_reconciled and metric_result.score is None:
                    row_unrecovered = True
                metrics_data.append({
                    # Base name (suffix stripped) so per-metric rates and
                    # the convergence check key on the bare config name.
                    "name": self._base_metric_name(metric_result.name),
                    "score": metric_result.score,
                    "threshold": metric_result.threshold,
                    "success": metric_result.success,
                    "reason": metric_result.reason,
                    # Surface the underlying error text on the wire so the
                    # UI / optimizer can show *why* the score is None
                    # instead of just rendering a blank cell.
                    "error": metric_result.error,
                })
            if row_unrecovered:
                unrecovered_count += 1

            results.append({
                "name": test_result.name or f"test_{len(results)}",
                # gated by TARGET metrics only — a diagnostic metric
                # can never veto the case.
                "success": self._gated_success(test_result),
                "input": test_result.input,
                "actual_output": test_result.actual_output,
                "expected_output": test_result.expected_output,
                "context": test_result.context if isinstance(test_result.context, list) else None,
                "retrieval_context": test_result.retrieval_context if isinstance(test_result.retrieval_context, list) else None,
                "additional_metadata": metadata,
                "metrics_data": metrics_data,
            })

        # Data-quality summary. Target-model generation calls and LLM-as-a-judge
        # scoring fail for unrelated reasons and have different remedies, so
        # they are tracked as separate domains.
        #
        # ``reliability`` is JUDGE-ONLY: "degraded" when EITHER:
        #  * >5% of metric evals stayed null after retry (judge throttling
        #    or flakiness), or
        #  * any case stayed unrecovered after reconciliation (silent drops
        #    are scarier than null scores; even one is a stop signal because
        #    the optimizer would chase noise on the remaining cases).
        #
        # Generation failures are carried separately by ``generation_failed_count``
        # and ``generation_failed_ratio`` — the backend owns the tolerance decision.
        if total_metric_evals > 0:
            null_ratio = null_after_retry / total_metric_evals
        else:
            null_ratio = 0.0
        # Total cases reported to the UI is what was SUBMITTED, not what
        # came back: reconciliation gives every input case a row (real or
        # sentinel), and if it somehow didn't, the shortfall must show up
        # as a shortfall rather than shrink the denominator.
        total_cases = total_input_cases

        # Judge-only reliability verdict: generation failures removed from the formula.
        reliability = "degraded" if (
            null_ratio > 0.05 or unrecovered_count > 0
        ) else "ok"

        # Generation failure ratio for backend tolerance decision.
        if total_cases > 0:
            generation_failed_ratio = round(generation_failed_count / total_cases, 4)
        else:
            generation_failed_ratio = 0.0

        # Per-domain stderr diagnostics so neither domain can describe the other's failure.
        if reliability == "degraded":
            print(
                f"[data-quality:judge] model={model_id}: "
                f"{null_after_retry}/{total_metric_evals} null, "
                f"{dropped_count} dropped, {unrecovered_count} unrecovered — "
                f"judge reliability degraded",
                file=sys.stderr,
                flush=True,
            )

        if generation_failed_count > 0:
            gen_pct = round(generation_failed_ratio * 100, 1)
            print(
                f"[data-quality:generation] model={model_id}: "
                f"{generation_failed_count}/{total_cases} generation calls failed "
                f"after retries ({gen_pct}%)",
                file=sys.stderr,
                flush=True,
            )

        # DeepEval-native format (matches platform ModelRun interface).
        # ``model_id`` is the source of truth — full Bedrock ID, used for
        # per-model aggregation and plateau detection. ``test_run_id`` is
        # a short display name derived from the ID; older readers that
        # join on it still work, but new code should key off ``model_id``.
        # ``source`` is permanent attribution — "optimization" or "manual" —
        # so the UI can tag runs without depending on optimization state.
        return {
            "source": self.source,
            "model_id": model_id,
            "test_run_id": short_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hyperparameters": hyperparams,
            "results": results,
            # Per-model cost totals. ``generation_cost_usd`` sums the
            # per-case ``generation_cost`` (target-model spend);
            # ``judge_cost_usd`` is the judge (LLM-as-a-judge) spend the
            # caller derived from the shared judge model's accumulated
            # usage. Both surfaced top-level so the report reads them
            # without re-summing per-case.
            "generation_cost_usd": (
                None if generation_cost_unknown else round(total_generation_cost, 6)
            ),
            "judge_cost_usd": (
                None if judge_cost_usd is None else round(judge_cost_usd, 6)
            ),
            "data_quality": {
                "null_count": null_after_retry,
                "errored_count": errored_after_retry,
                "total": total_metric_evals,
                "null_ratio": round(null_ratio, 4),
                # Case-level reconciliation accounting. ``total_cases``
                # is what the user submitted; the per-model UI total
                # reads from this so two models in the same run can't
                # show different denominators.
                "total_cases": total_cases,
                "dropped_count": dropped_count,
                "unrecovered_count": unrecovered_count,
                # Generation domain (target-model calls): count and ratio
                # for backend tolerance decision, tracked separately from
                # the judge-only reliability verdict.
                "generation_failed_count": generation_failed_count,
                "generation_failed_ratio": generation_failed_ratio,
                # Judge-only reliability verdict (scoring domain).
                "reliability": reliability,
            },
        }

    def save_results(
        self,
        results: list[dict[str, Any]],
        output_dir: Path,
    ) -> Path:
        """
        Save evaluation results to JSON file.

        Filename format: eval_results_{version}_{timestamp}.json

        Parameters:
            results: List of serialized model results
            output_dir: Output directory path

        Returns:
            Path to saved file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_results_{self.config.prompt_version}_{timestamp}.json"
        output_path = output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Emit filename only (not absolute path) for platform integration
        self._emit_progress("results", file=filename)

        return output_path
