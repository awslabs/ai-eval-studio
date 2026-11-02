# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured error value attached to long-running resources.

Every long-running resource (optimization runs, eval jobs, chat turns,
synthesis batches) owns its error state. When something goes wrong the
caller builds a :class:`ResourceError`, attaches it to the resource,
and transitions the resource to a terminal status. The frontend reads
the error from the resource like any other field — errors never live
only in log files.

This is the value type. Plumbing (event bus, WebSocket hub) and
attachment to individual resources live in their own modules.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, UTC
from typing import Any


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string with ``Z`` suffix."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass
class ResourceError:
    """A user-visible error attached to a long-running resource.

    Attributes:
        code: Stable, namespaced identifier (``bedrock.token_expired``,
            ``harness.subprocess_failed``, ``eval.metric_errors_dominant``).
            The frontend may branch on the code for specific UI — e.g.,
            showing a "Refresh credentials" button for ``bedrock.*`` codes.
            See :mod:`resource_error` for the registered catalog.
        message: One-line human text describing what happened.
        hint: Optional remediation text, shown as secondary guidance in the
            UI (e.g., "Run ``aws sso login``"). ``None`` when no actionable
            next step is known.
        at: ISO 8601 UTC timestamp of when the error was recorded. Defaults
            to the current time when not supplied.
        details: Optional structured payload (exit code, offending
            model_id, first exception string, etc.). Freeform dict; the
            frontend may surface specific keys when it knows the code.
    """

    code: str
    message: str
    hint: str | None = None
    at: str = field(default_factory=_now_iso)
    details: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate that ``details`` is JSON-serializable at construction time.

        ``ResourceError`` records are persisted to ``state.json`` and
        emitted over the event bus; a non-serializable ``details`` value
        (Path, datetime, custom object, etc.) would only blow up later
        inside the orchestrator's atomic state-save, wedging the
        optimization loop. Fail fast at the call site instead so the
        author can see exactly which field they need to stringify.
        """
        if self.details is not None:
            try:
                json.dumps(self.details)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"ResourceError.details must be JSON-serializable: {exc}"
                ) from exc

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict for API responses and storage.

        Keys match the dataclass field names exactly so ``from_dict`` can
        reconstruct without further translation. ``None`` values are kept
        (not stripped) so consumers can distinguish "field absent from
        payload" from "field explicitly null".
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResourceError:
        """Reconstruct from a dict produced by :meth:`to_dict`.

        Tolerant to missing optional fields (``hint``, ``details``) so
        earlier-format records continue to load after schema evolution.
        Raises :class:`KeyError` when required fields ``code`` or
        ``message`` are missing — callers should handle that as a
        corrupt-record situation.
        """
        return cls(
            code=data["code"],
            message=data["message"],
            hint=data.get("hint"),
            at=data.get("at") or _now_iso(),
            details=data.get("details"),
        )


# ---------------------------------------------------------------------------
# Error code catalog
# ---------------------------------------------------------------------------
# Codes are stable strings. Add new ones here; never rename an existing code
# (frontend branches on them). Group by domain prefix. Keep the list short —
# if a code is used once, inline the string at the call site; promote to a
# constant only when reused or when the frontend needs to pattern-match it.


class Codes:
    """Stable error-code constants.

    Use these rather than bare strings so renames stay caught by typechecks
    and the catalog is discoverable in one place. Add sparingly — codes
    only need to be constants when the frontend reacts to them or multiple
    call sites share them.
    """

    # bedrock.*
    BEDROCK_TOKEN_EXPIRED = "bedrock.token_expired"
    BEDROCK_THROTTLED = "bedrock.throttled"
    BEDROCK_MODEL_ACCESS_DENIED = "bedrock.model_access_denied"

    # harness.*
    HARNESS_SUBPROCESS_FAILED = "harness.subprocess_failed"
    HARNESS_UV_SYNC_FAILED = "harness.uv_sync_failed"
    HARNESS_RESULTS_MISSING = "harness.results_file_missing"

    # eval.*
    EVAL_METRIC_ERRORS_DOMINANT = "eval.metric_errors_dominant"
    EVAL_GENERATION_ERRORS_DOMINANT = "eval.generation_errors_dominant"
    EVAL_GENERATION_UNRELIABLE = "eval.generation_unreliable"
    EVAL_GENERATION_SYSTEMIC_ABORT = "eval.generation_systemic_abort"
    EVAL_NO_TEST_CASES = "eval.no_test_cases"
    EVAL_JUDGE_UNRELIABLE = "eval.judge_unreliable"
    EVAL_RESULTS_UNAVAILABLE = "eval.results_unavailable"

    # optimization.*
    OPTIMIZATION_PROJECT_NOT_FOUND = "optimization.project_not_found"
    OPTIMIZATION_STATE_MISSING = "optimization.state_file_missing"
    OPTIMIZATION_OPTIMIZER_TIMEOUT = "optimization.optimizer_timeout"
    OPTIMIZATION_DISPATCH_FAILED = "optimization.dispatch_failed"
    OPTIMIZATION_OPTIMIZER_EXHAUSTED = "optimization.optimizer_exhausted"
    OPTIMIZATION_OPTIMIZER_STALLED = "optimization.optimizer_stalled"
    OPTIMIZATION_ORPHANED = "optimization.orphaned"
    OPTIMIZATION_EVAL_WAIT_EXCEEDED = "optimization.eval_wait_exceeded"
    EVAL_DIED_WITHOUT_RESULTS = "eval.died_without_results"

    # agent.*
    AGENT_SDK_STREAM_ERROR = "agent.sdk_stream_error"
    AGENT_SUBAGENT_FAILED = "agent.subagent_failed"
    AGENT_CLI_ERROR = "agent.cli_error"

    # system.* — backstop for unhandled task crashes
    SYSTEM_TASK_CRASHED = "system.task_crashed"
