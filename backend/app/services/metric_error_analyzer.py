# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect mass metric-error failures and generation-reliability issues in a completed eval run.

When DeepEval runs with ``ErrorConfig(ignore_errors=True)`` (the default
in the harness template), per-case metric exceptions are swallowed and
recorded on ``metrics_data[*].error``. A single failure is expected
noise — a failure rate near 100%, all with the same exception text,
means the eval is broken at an infrastructure level (expired AWS
token, judge-model access denied, throttling) rather than because the
prompt is bad. Invoking the prompt-optimizer on such a run wastes time
and tokens.

This module covers both the metric-error domain (judge scoring failures)
and the generation-reliability domain (target-model call failures). These
are separate concerns with different remedies: a throttled judge needs
quota or retry; a throttled target model needs generation concurrency
reduction or retry.

This module turns a completed eval job's ``result`` payload into an
optional :class:`ResourceError` the orchestrator can attach to the
optimization state before breaking out of the loop without running the
optimizer step.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from app.services.resource_error import Codes, ResourceError


logger = logging.getLogger(__name__)


# HTTP status code to human-readable label mapping (this repo owns the labels).
# Used for upstream-status extraction in reliability halt messages.
_HTTP_STATUS_LABELS: dict[int, str] = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    408: "Request Timeout",
    409: "Conflict",
    424: "Failed Dependency",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}


# Minimum fraction of metric scores that must be errored for the run
# to be considered "dominantly errored." Chosen at 0.8 so a couple of
# genuine metric regressions don't trip the guard, but a uniform 100%
# failure (token expiry) does.
_ERROR_DOMINANCE_THRESHOLD = 0.8


# Default tolerance for generation failures (target-model call failures).
# Expressed as a ratio of total_cases, not an absolute count, so one or two
# transient failures in a run of a few hundred cases (under 1%) stay tolerable
# while a small run with genuinely bad data (1 of 5 = 20%) still halts. At the
# 2% boundary more than 98% of the dataset is still scored, which cannot move a
# pass rate enough to mislead the optimizer.
# The harness runs generation at high concurrency against Bedrock/mantle
# where transient 529s are expected, so a tolerance is necessary.
_DEFAULT_GENERATION_FAILURE_TOLERANCE_RATIO = 0.02


def _get_generation_failure_tolerance_ratio() -> float:
    """Return the configured generation-failure tolerance ratio.

    Reads ``EVAL_GENERATION_FAILURE_TOLERANCE_RATIO`` from the environment,
    falling back to the default (0.02) when unset or invalid. Logs a warning
    and falls back when the value is non-numeric or non-positive — never raises
    mid-loop so a misconfigured env var doesn't brick optimization.

    Returns:
        Tolerance ratio (0.0-1.0), defaulting to 0.02.
    """
    env_val = os.environ.get("EVAL_GENERATION_FAILURE_TOLERANCE_RATIO")
    if env_val is None:
        return _DEFAULT_GENERATION_FAILURE_TOLERANCE_RATIO

    try:
        ratio = float(env_val)
        if ratio <= 0:
            logger.warning(
                "EVAL_GENERATION_FAILURE_TOLERANCE_RATIO=%s is non-positive; "
                "falling back to default %.2f",
                env_val,
                _DEFAULT_GENERATION_FAILURE_TOLERANCE_RATIO,
            )
            return _DEFAULT_GENERATION_FAILURE_TOLERANCE_RATIO
        return ratio
    except ValueError:
        logger.warning(
            "EVAL_GENERATION_FAILURE_TOLERANCE_RATIO=%s is non-numeric; "
            "falling back to default %.2f",
            env_val,
            _DEFAULT_GENERATION_FAILURE_TOLERANCE_RATIO,
        )
        return _DEFAULT_GENERATION_FAILURE_TOLERANCE_RATIO


def _iter_metric_records(result: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten every ``metrics_data`` record across every test case + model.

    The DeepEval-native output schema (see harness template) nests
    ``results[*].metrics_data[*]``. Returns a flat list so the dominance math
    operates over metric-level atoms regardless of nesting depth.
    """
    records: list[dict[str, Any]] = []
    for model_run in result or []:
        per_case = model_run.get("results") or []
        for tc in per_case:
            md_list = tc.get("metrics_data") or []
            records.extend(m for m in md_list if isinstance(m, dict))
    return records


def _hint_for_error_text(error_text: str) -> str:
    """Return a remediation hint based on error text pattern matching.

    Pattern-matches the provider's error string to suggest actionable
    next steps (credential refresh, access grant, throttling backoff).
    Used by both generation and metric error detectors.

    Args:
        error_text: The error string to analyze (typically truncated).

    Returns:
        A user-facing hint string with remediation guidance.
    """
    lower = error_text.lower()
    # Mantle credential expiry — matches the word `unauthorized` and the
    # HTTP status pair `401 unauthorized` (anchored to prevent a
    # false-positive on incidental `401` digits like "token count 401" or
    # "case index 401"). Mantle reports expiry as a plain HTTP status, not
    # as ExpiredToken/AccessDenied, so matching only those codes misses it.
    if "unauthorized" in lower or "401 unauthorized" in lower:
        return "AWS credentials look expired — refresh with `aws sso login` and resume."
    elif any(
        phrase in lower
        for phrase in [
            "expiredtoken",
            "token included in the request",
            "session has expired",
            "reauthenticate",
            "credentials have changed",
            "aws login",
        ]
    ):
        return "AWS credentials look expired — refresh with `aws sso login` and resume."
    elif "accessdenied" in lower or "is not authorized to perform" in lower:
        return "Check Bedrock model access for the judge + generator models."
    elif "throttl" in lower:
        return "Bedrock throttled the judge — reduce concurrency or retry shortly."
    else:
        return "Inspect the harness log for the full traceback before resuming."


def _extract_dominant_upstream_status(error_texts: list[str]) -> dict[str, Any] | None:
    """Extract the dominant HTTP status code from upstream error texts.

    Parses error strings for HTTP status codes in status-shaped contexts (e.g.,
    'Client error 503 Service Unavailable' from httpx.HTTPStatusError, 'Server error'
    lines from boto3 ClientError). Classifies strictly by the numeric code; the
    human-readable label comes from _HTTP_STATUS_LABELS keyed on that number.

    Returns the most common status code along with observed distribution, or None
    when no parseable status exists. Incidental 3-digit numbers (token counts,
    case indices) do not match — the pattern requires status-shaped anchors.

    Args:
        error_texts: List of truncated error strings to analyze.

    Returns:
        Dict with keys {"status": int, "label": str, "count": int, "observed": dict}
        when a status is found, else None.
    """
    import re

    if not error_texts:
        return None

    # Match 3-digit HTTP status (400-599) in status-shaped context.
    # Anchored on "Client error '503", "Server error '500", or bare "503 " before reason.
    # This prevents false-positive on "token count 503" or "case 401 failed".
    status_pattern = re.compile(
        r"(?:Client error|Server error)\s+'?(\d{3})|"  # httpx / boto3 error prefix
        r"\b(\d{3})\s+(?:Unauthorized|Forbidden|Not Found|Bad Request|"
        r"Internal Server Error|Bad Gateway|Service Unavailable|Gateway Timeout|"
        r"Too Many Requests|Request Timeout|Conflict|Failed Dependency)\b",  # status + reason
        re.IGNORECASE,
    )

    observed: dict[int, int] = {}
    for text in error_texts:
        match = status_pattern.search(text)
        if match:
            # Extract the captured status (group 1 or 2)
            status_str = match.group(1) or match.group(2)
            status = int(status_str)
            # Only count 4xx/5xx
            if 400 <= status <= 599:
                observed[status] = observed.get(status, 0) + 1

    if not observed:
        return None

    # Return the most common status
    dominant_status = max(observed.items(), key=lambda kv: kv[1])
    status_code, count = dominant_status
    label = _HTTP_STATUS_LABELS.get(status_code, f"HTTP {status_code}")

    return {
        "status": status_code,
        "label": label,
        "count": count,
        "observed": observed,
    }


def detect_generation_failure(
    result: list[dict[str, Any]],
) -> ResourceError | None:
    """Return a :class:`ResourceError` when generation failures dominate, else None.

    "Generation failure" means a test case produced empty output (``actual_output``
    missing, None, empty string, or ``"[]"``) OR carries an error in
    ``additional_metadata.error``. "Dominate" means ``_ERROR_DOMINANCE_THRESHOLD``
    (80%) of cases are generation-failed.

    This catches infrastructure failures earlier in the pipeline than
    ``detect_metric_error_dominance`` — expired AWS credentials can cause
    every case to produce empty output with an error, while metrics still
    score successfully (scoring the empty output as a legitimate 0.0).

    Returns None when the run has no test cases, or when generation
    failures are below the dominance threshold.

    Args:
        result: The completed eval result payload (list of model runs).

    Returns:
        A ResourceError with code EVAL_GENERATION_ERRORS_DOMINANT if
        generation failures dominate, else None.
    """
    # Iterate cases the same way as _iter_metric_records but at case level
    cases: list[dict[str, Any]] = []
    for model_run in result or []:
        per_case = model_run.get("results") or []
        cases.extend(c for c in per_case if isinstance(c, dict))

    if not cases:
        return None

    # Count generation-failed cases
    gen_failed = []
    error_texts = []
    for case in cases:
        metadata = case.get("additional_metadata") or {}
        error = metadata.get("error")
        actual = case.get("actual_output")

        # Empty check: missing/None or stripped to "" or "[]"
        is_empty = not actual or str(actual).strip() in ("", "[]")

        if error is not None or is_empty:
            gen_failed.append(case)
            if error is not None:
                error_texts.append(str(error)[:400])

    if not gen_failed:
        return None

    ratio = len(gen_failed) / len(cases)
    if ratio < _ERROR_DOMINANCE_THRESHOLD:
        return None

    # Compute unique error counts and top error for the hint
    if error_texts:
        counts: dict[str, int] = {}
        for text in error_texts:
            counts[text] = counts.get(text, 0) + 1
        top_error, top_count = max(counts.items(), key=lambda kv: kv[1])
        unique_errors = len(counts)
        top_error_preview = top_error
        hint = _hint_for_error_text(top_error)
    else:
        # All failures are empty-output with no error text
        unique_errors = 0
        top_count = 0
        top_error_preview = ""
        hint = "Inspect the harness log for the full traceback before resuming."

    return ResourceError(
        code=Codes.EVAL_GENERATION_ERRORS_DOMINANT,
        message=(
            f"Generation failed for most cases "
            f"({len(gen_failed)}/{len(cases)} cases produced no output)."
        ),
        hint=hint,
        details={
            "generation_failed_cases": len(gen_failed),
            "total_cases": len(cases),
            "error_ratio": round(ratio, 4),
            "unique_errors": unique_errors,
            "top_error_count": top_count,
            "top_error_preview": top_error_preview,
        },
    )


def detect_infra_failure(
    result: list[dict[str, Any]],
) -> ResourceError | None:
    """Return a :class:`ResourceError` when infra failures are detected, else None.

    Combined detector that checks for generation-phase failures first
    (earlier in the pipeline), then falls back to metric-error dominance
    detection. Returns the first non-None error found.

    This is the unified entry point for both eval executor and optimization
    executor — they get a single signal covering both failure modes.

    Args:
        result: The completed eval result payload (list of model runs).

    Returns:
        A ResourceError if either generation or metric errors dominate,
        else None (indicating a genuine low-score run or normal eval).
    """
    # Check generation failures first (earlier pipeline phase)
    gen_error = detect_generation_failure(result)
    if gen_error is not None:
        return gen_error

    # Fall back to metric error dominance
    return detect_metric_error_dominance(result)


def detect_metric_error_dominance(
    result: list[dict[str, Any]],
) -> ResourceError | None:
    """Return a :class:`ResourceError` when metric errors dominate, else None.

    "Dominate" means ``_ERROR_DOMINANCE_THRESHOLD`` (80%) of metric-level
    records carry a non-empty ``error`` field. When true, the returned
    error's ``details`` include the total, the errored count, the
    dominance ratio, a sample of the most common error text, and the
    unique-error count so the UI/log can show operators what's wrong.

    Returns None when the run has no metric records (nothing to judge),
    or when errors are below the dominance threshold (likely a genuine
    prompt regression, not infra).
    """
    records = _iter_metric_records(result)
    if not records:
        return None

    # Check ``is not None`` rather than truthiness so a DeepEval record
    # carrying an empty-string error (``error: ""``) still counts as
    # errored. Today DeepEval only emits ``None`` or a populated string,
    # but the explicit None-check survives schema edges and prevents
    # silent undercounts that would let infra failures through the guard.
    errored = [m for m in records if m.get("error") is not None]
    if not errored:
        return None

    ratio = len(errored) / len(records)
    if ratio < _ERROR_DOMINANCE_THRESHOLD:
        return None

    # Group by exception text so operators see how many distinct failure
    # modes hit them at once (usually just one).
    counts: dict[str, int] = {}
    for m in errored:
        text = str(m["error"])[:400]
        counts[text] = counts.get(text, 0) + 1

    # Most common error wins the preview slot.
    top_error, top_count = max(counts.items(), key=lambda kv: kv[1])

    # Use the shared hint helper
    hint = _hint_for_error_text(top_error)

    return ResourceError(
        code=Codes.EVAL_METRIC_ERRORS_DOMINANT,
        message=(
            f"Metric errors dominated this run "
            f"({len(errored)}/{len(records)} scores errored)."
        ),
        hint=hint,
        details={
            "errored_scores": len(errored),
            "total_scores": len(records),
            "error_ratio": round(ratio, 4),
            "unique_errors": len(counts),
            "top_error_count": top_count,
            "top_error_preview": top_error,
        },
    )


def _assess_generation_reliability(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract per-model generation-reliability assessment from data_quality blocks.

    Iterates the model runs and reads each ``data_quality`` block, extracting
    the generation-failed count, total_cases, and the ratio. The harness always
    writes ``generation_failed_ratio``, so it is read rather than re-derived —
    the count/total fallback existed for runs that lacked the key and computed a
    SECOND version of a number the harness had already decided. Missing fields
    read as 0.0 (not-degraded).

    Also extracts error texts from failed cases and hyperparameters (endpoint, region)
    for upstream status analysis.

    Returns a list of dicts carrying ``model_id``, ``generation_failed_count``,
    ``total_cases``, ``generation_failed_ratio`` (always present — read from
    the harness block, 0.0 when absent), ``error_texts``, ``endpoint``, and
    ``region``.

    Args:
        results: The completed eval result payload (list of model runs).

    Returns:
        List of per-model generation assessments.
    """
    assessments: list[dict[str, Any]] = []
    for model_run in results or []:
        dq = model_run.get("data_quality") or {}
        model_id = model_run.get("model_id", "?")
        hyperparams = model_run.get("hyperparameters") or {}

        # Count and total are carried for the message; the ratio is read, not
        # derived (below). Missing keys read as 0 (not-degraded).
        generation_failed_count = dq.get("generation_failed_count", 0)
        total_cases = dq.get("total_cases", 0)

        # Read, never re-derive: the harness owns this number and rounds it to
        # 4 places, so a locally computed one could differ at the tolerance
        # boundary and make the same run degraded on one path and clean on the
        # other.
        ratio = dq.get("generation_failed_ratio") or 0.0

        # Extract error texts from GENERATION_FAILED results (mirror detect_generation_failure)
        error_texts: list[str] = []
        per_case = model_run.get("results") or []
        for case in per_case:
            metadata = case.get("additional_metadata") or {}
            error = metadata.get("error")
            if error is not None:
                error_texts.append(str(error)[:400])

        assessments.append({
            "model_id": model_id,
            "generation_failed_count": generation_failed_count,
            "total_cases": total_cases,
            "generation_failed_ratio": ratio,
            "error_texts": error_texts,
            "endpoint": hyperparams.get("endpoint", "unknown"),
            "region": hyperparams.get("region", "unknown"),
        })

    return assessments


def detect_generation_unreliable(
    results: list[dict[str, Any]],
) -> ResourceError | None:
    """Return a :class:`ResourceError` when generation failures exceed tolerance, else None.

    "Generation failure" means a target-model call failed to produce valid output
    (after the harness's own retries). The tolerance is ratio-based (default 2%
    of total_cases, overridable via ``EVAL_GENERATION_FAILURE_TOLERANCE_RATIO``)
    so one or two transient failures in a run of a few hundred cases are
    tolerated while a small run with genuinely bad data (1 of 5 = 20%) still
    halts. This is the middle
    tier between the sub-tolerance band (proceed with caveat) and the >=80%
    dominance tier (``detect_generation_failure`` returns a hard-fail error).

    Returns None when no model exceeds the tolerance. When any model is above
    tolerance, returns an error with code ``EVAL_GENERATION_UNRELIABLE``, carrying
    per-model details (count, total, percentage) and a hint pointing at target-model
    throttling remedies.

    The message and hint do NOT mention the judge or scoring — this is the
    generation domain (target-model calls), distinct from judge reliability.

    Args:
        results: The completed eval result payload (list of model runs).

    Returns:
        A ResourceError if any model's generation-failure ratio exceeds the
        tolerance, else None.
    """
    tolerance = _get_generation_failure_tolerance_ratio()
    assessments = _assess_generation_reliability(results)

    above_tolerance = [
        a for a in assessments
        if a["generation_failed_ratio"] > tolerance
    ]

    if not above_tolerance:
        return None

    # Extract upstream status for each above-tolerance model and enhance details
    enhanced_models: list[dict[str, Any]] = []
    observed_statuses: set[int] = set()

    for a in above_tolerance:
        model_details = {
            "model_id": a["model_id"],
            "generation_failed_count": a["generation_failed_count"],
            "total_cases": a["total_cases"],
            "generation_failed_ratio": a["generation_failed_ratio"],
            "endpoint": a["endpoint"],
            "region": a["region"],
        }

        # Extract dominant status from error texts
        status_info = _extract_dominant_upstream_status(a["error_texts"])
        if status_info:
            model_details["status"] = status_info["status"]
            model_details["label"] = status_info["label"]
            model_details["observed"] = status_info["observed"]
            observed_statuses.update(status_info["observed"].keys())

        enhanced_models.append(model_details)

    # Build a user-facing message naming count, total, percentage, model ID,
    # and when available: status + label + endpoint
    parts: list[str] = []
    for m in enhanced_models:
        pct = round(m["generation_failed_ratio"] * 100, 1)
        part = (
            f"{m['generation_failed_count']}/{m['total_cases']} cases ({pct}%) "
            f"on {m['model_id']}"
        )
        if "status" in m:
            part += f" — {m['status']} {m['label']} from endpoint '{m['endpoint']}'"
        parts.append(part)

    # Conditional credentials claim: only when NO 401/403 observed
    has_auth_failure = bool(observed_statuses & {401, 403})
    if has_auth_failure:
        message = (
            f"Generation calls failed for a fraction of cases exceeding the "
            f"tolerated ratio ({int(tolerance * 100)}%): {'; '.join(parts)}. "
            "Auth failures observed."
        )
        hint = "Check credentials and Bedrock model access."
    elif observed_statuses:
        # Non-auth status observed — capacity not credentials
        message = (
            f"Generation calls failed for a fraction of cases exceeding the "
            f"tolerated ratio ({int(tolerance * 100)}%): {'; '.join(parts)}. "
            "Upstream capacity issues, not credentials."
        )
        hint = "Retry the run, or reduce generation concurrency in the harness config."
    else:
        # No parseable status — generic
        message = (
            f"Generation calls failed for a fraction of cases exceeding the "
            f"tolerated ratio ({int(tolerance * 100)}%): {'; '.join(parts)}."
        )
        hint = (
            "Target-model generation throttled or transiently unavailable. "
            "Retry the run, or reduce generation concurrency in the harness config."
        )

    return ResourceError(
        code=Codes.EVAL_GENERATION_UNRELIABLE,
        message=message,
        hint=hint,
        details={
            "models": enhanced_models,
            "tolerance_ratio": tolerance,
        },
    )


def build_generation_caveats(
    results: list[dict[str, Any]],
) -> list[str]:
    """Build user-facing caveat strings for sub-tolerance generation failures.

    Returns one short line per model with a non-zero generation-failed count,
    naming the count, total, and percentage. These are recorded on the iteration
    so a tolerated transient failure survives into history and explains why a
    pass rate moved. Empty list when there is nothing to report.

    Args:
        results: The completed eval result payload (list of model runs).

    Returns:
        List of caveat strings (empty when no generation failures).
    """
    assessments = _assess_generation_reliability(results)
    caveats: list[str] = []

    for a in assessments:
        if a["generation_failed_count"] > 0:
            pct = round(a["generation_failed_ratio"] * 100, 1)
            caveats.append(
                f"{a['generation_failed_count']}/{a['total_cases']} generation calls "
                f"failed ({pct}%) on {a['model_id']} (within tolerance)"
            )

    return caveats
