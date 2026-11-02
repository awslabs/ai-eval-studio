# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OptimizationState tracks iterative eval-optimize loops for prompt improvement.

This module provides state management for the optimization workflow, including:
- Recording iteration history (prompt versions, metrics, pass rates)
- Detecting convergence conditions (target reached, max iterations, plateau)
- Persisting state to disk for recovery across restarts
- Identifying the best performing iteration

On the EFS path the state is a JSON file in the project directory; on the
durable path it is a DynamoDB item. Either way it survives restarts.

**Write discipline:** On the durable-state path (arch_split flag True), writes
flow through one funnel, ``optimization_state_repository``: read the
current version → mutate → put_optimization_state(expected_version=<version>).
Several components write — the loop orchestrator, the iteration executor, the
eval-cancel path, the REST optimization/job/project routes, dispatch_optimization
and the record_optimization_iteration tool — so the expected_version conditional
write is the loud seatbelt: it raises ConcurrentUpdateError on a stale version
instead of letting a later write silently overwrite an earlier one. Do NOT add
per-writer retry loops; a writer that loses the race must re-read and decide,
not re-apply its mutation blindly.
"""

import json
import logging
import os
import tempfile
import threading
from decimal import Decimal
from pathlib import Path
from datetime import datetime, UTC
from typing import Any

from app.services.metric_name import base_metric_name as _base_metric_name
from app.services.resource_error import ResourceError

logger = logging.getLogger(__name__)


def find_metric_name_collisions(names: list[str] | dict[str, Any]) -> list[str]:
    """Return base names claimed by MORE than one configured metric name.

    Two config metrics differing only in a trailing ``[...]`` suffix (e.g.
    ``"Correctness [strict]"`` vs ``"Correctness [loose]"``) collapse to the
    same key under suffix normalization — gating and aggregation would merge
    them silently. Ingestion sites call this to reject such configs loudly.
    """
    seen: dict[str, set[str]] = {}
    for n in names:
        seen.setdefault(_base_metric_name(n), set()).add(n)
    return sorted(base for base, originals in seen.items() if len(originals) > 1)

# Per-project save() serialization. Two concurrent writers (e.g., the
# orchestrator + a manual record_optimization_iteration MCP call) each
# create a temp file and race on os.replace. The rename itself is atomic,
# but the later writer silently overwrites the earlier update — losing
# an iteration record. A threading.Lock keyed by absolute state-file
# path makes the write strictly sequential within the process.
_save_locks: dict[str, threading.Lock] = {}
_save_locks_registry = threading.Lock()


def _get_save_lock(state_path: Path) -> threading.Lock:
    """Return the shared save() lock for ``state_path`` (created on first use)."""
    key = str(state_path.resolve())
    with _save_locks_registry:
        lock = _save_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _save_locks[key] = lock
        return lock


def _coerce_decimals_to_float(obj: Any) -> Any:
    """
    Recursively convert Decimal instances to float in dicts and lists.

    The DDB store returns Decimal for fractional numbers (to preserve
    cost_usd precision at the store layer). The optimization convergence
    math mixes these with float fallbacks, and Decimal - float raises
    TypeError. Normalize to float on the OptimizationState DDB read path
    only — do NOT normalize globally in _from_dynamo, which must keep
    Decimal for the cost_usd precision guarantee.

    Builds fresh dicts/lists; primitives (str, bool, int, None) pass through.
    """
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _coerce_decimals_to_float(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_decimals_to_float(v) for v in obj]
    # Primitives: str, bool, int, float, None pass through unchanged
    return obj


OPTIMIZATION_STATE_DIR = Path("output") / "optimization"
OPTIMIZATION_STATE_FILE = OPTIMIZATION_STATE_DIR / "state.json"


# Default plateau threshold: 2% pass-rate spread across the window counts
# as "no improvement". Override per-project via EvalConfig.plateau_threshold
# or the dispatch_optimization MCP tool — long-form GEval tasks often want
# a tighter threshold (e.g. 0.01) to avoid premature termination.
DEFAULT_PLATEAU_THRESHOLD = 0.02


class OptimizationState:
    """
    Manages optimization state for iterative prompt improvement workflow.

    Tracks convergence criteria, iteration history, and provides methods to
    detect when optimization should terminate based on target metrics, max
    iterations, or performance plateau.

    Attributes:
        project_dir: Path to the project directory
        target_metric_scores: Dict mapping metric names to target scores
        max_iterations: Maximum number of optimization iterations per run
        plateau_window: Number of iterations to check for plateau detection
        iterations: List of recorded iteration results
        status: Current optimization status
        max_iter_cap: Absolute iteration number where we auto-stop
    """

    def __init__(
        self,
        project_dir: Path,
        target_metric_scores: dict[str, float],
        max_iterations: int,
        plateau_window: int,
        iterations: list[dict[str, Any]],
        status: str,
        max_iter_cap: int | None = None,
        plateau_threshold: float | None = None,
        workload: dict[str, Any] | None = None,
    ):
        """
        Initialize OptimizationState with given parameters.

        Args:
            project_dir: Path to the project directory
            target_metric_scores: Target scores for each metric
            max_iterations: Max iterations per run (budget added on each max_iter resume)
            plateau_window: Number of iterations for plateau detection
            iterations: List of recorded iteration results
            status: Current optimization status
            max_iter_cap: Absolute iteration cap. Defaults to max_iterations on first create.
            plateau_threshold: Pass-rate spread (0.0–1.0) below which the
                window counts as plateaued. Defaults to
                ``DEFAULT_PLATEAU_THRESHOLD`` (0.02).
        """
        self.project_dir = Path(project_dir)
        self.target_metric_scores = target_metric_scores
        self.max_iterations = max_iterations
        self.plateau_window = plateau_window
        self.iterations = iterations
        self.status = status
        self.max_iter_cap = max_iter_cap if max_iter_cap is not None else max_iterations
        self.plateau_threshold = (
            plateau_threshold if plateau_threshold is not None else DEFAULT_PLATEAU_THRESHOLD
        )
        # Workload fingerprint of what the run will execute, captured by both
        # start paths (REST + the agent's dispatch_optimization) so the resume
        # path can detect that the user edited src/config.py or switched data
        # source between cancel and resume. ``None`` only on records written
        # before the REST path captured it, which gets no drift detection.
        self.workload = workload
        # When resume detects drift, the {before, after} diff goes here.
        # The next ``record_iteration`` call consumes this field — it's
        # attached to that iteration's row and then cleared so it fires
        # at most once per resume event.
        self.pending_resume_drift: dict[str, dict] | None = None
        # Populated by ``check_convergence`` when any model clears every
        # target — the winning model_id so the UI + report agent can
        # recommend it directly.
        self.winning_model_id: str | None = None
        # The prompt version currently being evaluated or just evaluated.
        # Updated at the start of each iteration so the frontend can
        # display the correct prompt without relying on stale iterations[-1].
        self.current_prompt_version: str | None = None
        # Loop phase for honest status display: "evaluating" while an
        # iteration's eval job runs, "optimizing_prompt" while the
        # prompt-optimizer analyzes failures and writes the next version.
        # ``None`` before the orchestrator enters its first phase (the
        # frontend then says only what it knows) and meaningless once
        # status is terminal. Written only by the loop orchestrator, through
        # the same repository funnel every other field goes through.
        self.phase: str | None = None
        # Set when the loop terminates due to a failure the user should
        # see (eval infra collapse, dispatch failure, project not found,
        # optimizer crash, etc.). Paired with ``status`` transitioning
        # to a terminal value. ``None`` means the run ended cleanly or
        # is still live.
        self.error: ResourceError | None = None

    @classmethod
    def create(
        cls,
        project_dir: Path,
        target_metric_scores: dict[str, float],
        max_iterations: int = 10,
        plateau_window: int = 3,
        plateau_threshold: float | None = None,
        workload: dict[str, Any] | None = None,
        *,
        arch_split: bool = False,
        user_id: str | None = None,
        project_id: str | None = None,
        expected_version: int | None = None,
    ) -> "OptimizationState":
        """
        Create a new OptimizationState with default values.

        Args:
            project_dir: Path to the project directory
            target_metric_scores: Target scores for metrics to reach
            max_iterations: Maximum number of iterations (default: 10)
            plateau_window: Window size for plateau detection (default: 3)
            plateau_threshold: Pass-rate spread (0.0–1.0) below which the
                window counts as plateaued. Defaults to
                ``DEFAULT_PLATEAU_THRESHOLD`` (0.02).
            workload: Workload fingerprint (model IDs, metric configs,
                judge model, data file) captured at creation; used by
                ``reset_for_resume`` to detect a workload drift and bias
                convergence accordingly.
            arch_split: If True, route save to DynamoDB via put_optimization_state.
                If False (default), write state.json (old path).
            user_id: Required when arch_split=True for put_optimization_state call.
            project_id: Required when arch_split=True for put_optimization_state call.
            expected_version: Threadable version for conditional DynamoDB writes.
                On the new path (arch_split=True), start/dispatch must overwrite
                any existing TERMINAL record. The store's condition is
                ``attribute_not_exists(version) OR version = :expected`` (see
                ``DynamoDBMetadataStore.put_optimization_state``): passing None uses the
                attribute_not_exists guard (genuine first create); passing an int
                uses the version-match guard (overwrite an existing terminal record
                on the 2nd start). Without threading expected_version, a bare None
                crashes on the 2nd start because a record already exists.

        Returns:
            New OptimizationState instance with empty iteration history.
            The version number (if any) from the durable write is NOT returned —
            callers that need it read via get_optimization_state.
        """
        state = cls(
            project_dir=project_dir,
            target_metric_scores=target_metric_scores,
            max_iterations=max_iterations,
            plateau_window=plateau_window,
            iterations=[],
            status="running",
            max_iter_cap=max_iterations,
            plateau_threshold=plateau_threshold,
            workload=workload,
        )
        state.save(
            arch_split=arch_split,
            user_id=user_id,
            project_id=project_id,
            expected_version=expected_version,
        )
        return state

    @classmethod
    def load(cls, project_dir: Path) -> "OptimizationState | None":
        """Load OptimizationState from ``project_dir / OPTIMIZATION_STATE_FILE``.

        Args:
            project_dir: Path to the project directory.

        Returns:
            OptimizationState instance if file exists, None otherwise.
        """
        project_dir = Path(project_dir)
        state_file = project_dir / OPTIMIZATION_STATE_FILE

        if not state_file.exists():
            return None

        with open(state_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = cls._from_dict(project_dir, data)
        logger.debug("[optimization-state] Loaded: status=%s iter=%d/%d",
                     instance.status, instance.current_iteration, instance.max_iter_cap)
        return instance

    @classmethod
    def _from_dict(cls, project_dir: Path, data: dict) -> "OptimizationState":
        """Construct from a JSON-parsed dict.

        State files written before a field was added load with the
        field's default. State files written by an earlier version with
        a now-removed field (e.g. ``use_prompt_architect``) silently
        ignore the unknown key — JSON-parsed dicts only feed named
        kwargs, so extras don't reach ``__init__``.
        """
        # Normalize Decimal→float so the DDB read path produces the SAME in-memory
        # types as the EFS json.load path. The DDB store returns Decimal for fractional
        # numbers (to preserve cost precision), but convergence math mixes these with
        # float fallbacks and Decimal - float raises TypeError.
        data = _coerce_decimals_to_float(data)

        instance = cls(
            project_dir=project_dir,
            target_metric_scores=data["target_metric_scores"],
            max_iterations=data["max_iterations"],
            plateau_window=data["plateau_window"],
            iterations=data["iterations"],
            status=data["status"],
            max_iter_cap=data.get("max_iter_cap", data["max_iterations"]),
            plateau_threshold=data.get("plateau_threshold"),
            workload=data.get("workload"),
        )
        instance.pending_resume_drift = data.get("pending_resume_drift")
        instance.winning_model_id = data.get("winning_model_id")
        instance.current_prompt_version = data.get("current_prompt_version")
        instance.phase = data.get("phase")
        raw_error = data.get("error")
        if raw_error:
            try:
                instance.error = ResourceError.from_dict(raw_error)
            except (KeyError, TypeError):
                # Malformed on-disk record — log and treat as no error.
                logger.warning(
                    "[optimization-state] Ignoring unreadable error record: %r",
                    raw_error,
                )
        return instance

    @property
    def current_iteration(self) -> int:
        """
        Get the current iteration number.

        Returns:
            Number of completed iterations
        """
        return len(self.iterations)

    def record_iteration(
        self,
        prompt_version: str,
        run_id: str,
        metrics_summary: dict[str, float],
        total_cases: int,
        passed: int,
        per_model_pass_rates: dict[str, float] | None = None,
        per_model_metrics: dict[str, dict[str, float]] | None = None,
        *,
        caveats: list[str] | None = None,
        arch_split: bool = False,
        user_id: str | None = None,
        project_id: str | None = None,
        expected_version: int | None = None,
    ) -> int | None:
        """
        Record a new optimization iteration.

        Args:
            prompt_version: Version identifier for the prompt used (e.g., "v1")
            run_id: Unique identifier for the eval run
            metrics_summary: Dict mapping metric names to scores (aggregate)
            total_cases: Total number of test cases evaluated across all models
            passed: Number of cases that passed across all models
            per_model_pass_rates: Optional dict mapping model_id → pass_rate
                for this iteration. When supplied, plateau detection uses
                per-model data (the honest signal — model A climbing while
                model B regresses should not count as plateaued). When
                omitted, plateau detection falls back to the aggregate.
            per_model_metrics: Optional dict mapping model_id →
                {metric_name: pass_rate} for this iteration. Used by the
                "any model clears every target" convergence check so the
                loop can stop as soon as the best model hits the goal.
            caveats: Optional list of non-fatal run-quality notes (e.g., a
                tolerated transient generation failure). Recorded on the
                iteration when present so the user can see why a pass rate
                moved. Absent key means "nothing to report"; empty list is
                never written (matches resume_drift behavior).
            arch_split: If True, route save to DynamoDB (new path). Default False.
            user_id: Required on new path for put_optimization_state.
            project_id: Required on new path for put_optimization_state.
            expected_version: Required on new path for conditional write.

        Returns:
            On the new path (arch_split=True), returns the new version number.
            On the old path (arch_split=False), returns None. Caller MUST
            thread this forward to the next write (check_convergence).
        """
        pass_rate = passed / total_cases if total_cases > 0 else 0.0

        iteration = {
            "iteration": self.current_iteration + 1,
            "prompt_version": prompt_version,
            "run_id": run_id,
            "metrics_summary": metrics_summary,
            "total_cases": total_cases,
            "passed": passed,
            "pass_rate": pass_rate,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        if per_model_pass_rates is not None:
            iteration["per_model_pass_rates"] = per_model_pass_rates
        if per_model_metrics is not None:
            iteration["per_model_metrics"] = per_model_metrics
        if self.pending_resume_drift:
            iteration["resume_drift"] = self.pending_resume_drift
            self.pending_resume_drift = None
        # Caveats are written only when truthy (mirroring resume_drift pattern).
        # Absent key means "nothing to report"; empty list is never persisted.
        if caveats:
            iteration["caveats"] = caveats

        self.iterations.append(iteration)
        # Persist immediately so state survives crashes
        return self.save(
            arch_split=arch_split,
            user_id=user_id,
            project_id=project_id,
            expected_version=expected_version,
        )

    @staticmethod
    def workload_fingerprint(
        model_ids: list[str],
        judge_model_id: str | None,
        data_file: str | None,
    ) -> dict[str, Any]:
        """Canonicalise the workload parameters a resume must preserve.

        Used by the resume path to detect when the user has edited
        ``src/config.py`` between cancel and resume in ways that would
        make the resumed iterations incomparable with the prior ones
        (swapped models or judge, pointed at a different dataset). Metrics
        are code in ``src/metrics/metrics.py`` and are not part of it.
        Returns a dict; callers diff two dicts to find the drift.

        Prefer :meth:`workload_fingerprint_from_harness` — a fingerprint
        only detects drift if every site derives it the same way.
        """
        return {
            "models": sorted(model_ids),
            "judge_model_id": judge_model_id,
            "data_file": data_file,
        }

    @staticmethod
    def workload_fingerprint_from_harness(
        harness: Any, data_file_override: str | None
    ) -> dict[str, Any]:
        """Fingerprint the workload a run will ACTUALLY execute.

        One derivation for every dispatch and resume entry point. Hand-rolled
        copies drifted three ways: the UI start path captured no fingerprint at
        all (so resume drift went undetected for every UI-started run), and the
        two that did capture one read ``harness.data_file`` even when the caller
        had resolved a synthetic/merged override — the one edit a user is most
        likely to make between cancel and resume.

        Args:
            harness: The project's loaded :class:`HarnessConfig`.
            data_file_override: Resolved dataset path when the caller picked a
                non-raw data source, else ``None``.

        Returns:
            The canonical fingerprint dict.
        """
        return OptimizationState.workload_fingerprint(
            model_ids=harness.models,
            judge_model_id=harness.judge_model_id,
            data_file=data_file_override or harness.data_file,
        )

    def check_convergence(
        self,
        *,
        arch_split: bool = False,
        user_id: str | None = None,
        project_id: str | None = None,
        expected_version: int | None = None,
    ) -> tuple[str | None, int | None]:
        """
        Check if optimization has converged based on configured criteria.

        Convergence conditions checked in order:
        1. Target reached: ANY one target model clears EVERY configured
           metric target on the latest iteration (the "any model wins"
           rule — see the inline comment for the rationale). There is NO
           aggregate fallback: an iteration recorded without per-model
           metrics gets no verdict and logs a warning, because a rate
           pooled across models cannot attest that any single model
           cleared the targets.
        2. Max iterations: reached the absolute iteration cap.
        3. Plateau: EVERY model's per-iteration rate has flatlined within
           the threshold over the last N iterations. Also has no aggregate
           fallback — a window carrying no per-model data at all withholds
           the verdict and lets max_iterations cap the run. (Distinct from
           the per-row fill-in inside the per-model series, which covers a
           model missing from one row of an otherwise per-model window.)

        Args:
            arch_split: If True, route save to DynamoDB (new path). Default False.
            user_id: Required on new path for put_optimization_state.
            project_id: Required on new path for put_optimization_state.
            expected_version: Required on new path for conditional write.

        Returns:
            Tuple of (convergence_reason, new_version). Convergence reason is
            "target_reached", "max_iterations", "plateau", or None if not converged.
            new_version is the version returned by save() on the new path (int) or
            None on the old path. Caller MUST thread the version forward to
            subsequent writes.
        """
        if not self.iterations:
            return (None, None)

        latest = self.iterations[-1]

        def _find_metric(metric_map: dict[str, float], name: str) -> float:
            """Look up metric value by name, tolerating DeepEval's
            ``[GEval]`` suffix on EITHER side (e.g. config key
            ``"Correctness [GEval]"`` matches serialized ``"Correctness"``
            and vice versa). Mirrors the harness template's
            ``_base_metric_name`` regex — keep the two in sync. Returns
            0.0 when absent so a missing metric does NOT accidentally
            clear its target."""
            if name in metric_map:
                return metric_map[name]
            want = _base_metric_name(name)
            for key, val in metric_map.items():
                if _base_metric_name(key) == want:
                    return val
            return 0.0

        # Target-reached: the run converges as soon as **any** model
        # clears every configured target. This is deliberate — the goal
        # of optimization is to find the best model+prompt for the job,
        # so if the strongest model has already hit the targets we stop
        # even when weaker comparison models are still behind.
        #
        # Requires per-model metric data — every writer supplies it now
        # (the MCP recording path derives it from the run file; the
        # executor and orchestrator always did). A row without it gets NO
        # convergence verdict, loudly, rather than a silently different
        # definition of "converged".
        #
        # Post-drift guard: if the latest iteration is the first run
        # after a workload change (resume_drift marker), withhold
        # target_reached for this one iteration. The post-drift score is
        # against a different baseline than the earlier history; a lucky
        # first run against a new dataset shouldn't falsely claim the
        # original goal was met. One more clean iteration unlocks the
        # verdict.
        latest_is_post_drift = bool(latest.get("resume_drift"))

        if not latest_is_post_drift:
            per_model_metrics = latest.get("per_model_metrics")
            if isinstance(per_model_metrics, dict) and per_model_metrics:
                for model_id, model_metrics in per_model_metrics.items():
                    if all(
                        _find_metric(model_metrics, metric) >= target
                        for metric, target in self.target_metric_scores.items()
                    ):
                        self.status = "converged"
                        self.winning_model_id = model_id
                        new_version = self.save(
                            arch_split=arch_split,
                            user_id=user_id,
                            project_id=project_id,
                            expected_version=expected_version,
                        )
                        return ("target_reached", new_version)
            else:
                # No aggregate fallback. "Converged" has ONE definition —
                # some model cleared every target — and an aggregate across
                # models cannot attest that. A row without per-model data is
                # not convergence evidence: withhold the verdict, say so, and
                # let max_iterations/plateau/cancel end the run. Same
                # philosophy as the UI's evidence gating. Every writer derives
                # the row from the run file, so reaching here means the run
                # file itself yielded no per-model results.
                logger.warning(
                    "[optimization-state] Latest iteration (prompt=%s) has no "
                    "per_model_metrics — convergence verdict withheld. The run "
                    "file it was derived from carried no per-model results.",
                    latest.get("prompt_version"),
                )

        # Check max iterations against the absolute cap
        if self.current_iteration >= self.max_iter_cap:
            self.status = "max_iterations"
            new_version = self.save(
                arch_split=arch_split,
                user_id=user_id,
                project_id=project_id,
                expected_version=expected_version,
            )
            return ("max_iterations", new_version)

        # Check plateau using full iteration history.
        #
        # Per-model plateau: the loop plateaus only when EVERY model's
        # per-iteration pass rate has stabilised within the threshold over
        # the recent window. If model A improves 30→35 while model B
        # regresses 30→25, the aggregate pass rate looks flat at 60% but
        # there's still real signal — we want to keep optimizing.
        #
        # ``_aggregate`` below is a per-ROW fill-in, not a fallback for the
        # verdict: it covers a model absent from one row of an otherwise
        # per-model window (a lineup change mid-run). A window with no
        # per-model data anywhere yields no plateau verdict at all — see the
        # ``else`` branch. Two different cases; do not collapse them.
        if self.current_iteration >= self.plateau_window:
            recent = self.iterations[-self.plateau_window :]

            def _aggregate(it: dict) -> float:
                return it.get(
                    "pass_rate",
                    it["passed"] / it["total_cases"] if it.get("total_cases") else 0,
                )

            # Collect the union of model_ids that appear in the window.
            model_ids: set[str] = set()
            for it in recent:
                pm = it.get("per_model_pass_rates")
                if isinstance(pm, dict):
                    model_ids.update(pm.keys())

            if model_ids:
                per_model_spread: dict[str, float] = {}
                for model_id in model_ids:
                    series = [
                        it.get("per_model_pass_rates", {}).get(model_id, _aggregate(it))
                        for it in recent
                    ]
                    per_model_spread[model_id] = max(series) - min(series)

                logger.info(
                    "[optimization-state] Plateau check (per-model): window=%d "
                    "threshold=%.4f spreads=%s",
                    self.plateau_window, self.plateau_threshold,
                    {m: round(s, 4) for m, s in per_model_spread.items()},
                )

                # Plateau only when every model has flatlined.
                if all(s < self.plateau_threshold for s in per_model_spread.values()):
                    self.status = "plateau"
                    new_version = self.save(
                        arch_split=arch_split,
                        user_id=user_id,
                        project_id=project_id,
                        expected_version=expected_version,
                    )
                    return ("plateau", new_version)
            else:
                # No aggregate fallback. A plateau verdict stops the run,
                # so it demands the same evidence quality as convergence —
                # per-model series (model A climbing while model B regresses
                # is flat in aggregate but is real signal). A window with no
                # per-model data at all is not plateau evidence: withhold the
                # verdict, keep iterating, and let max_iterations cap the run.
                # The per-row _aggregate fill-in
                # INSIDE the per-model series above is a different case and
                # stays: it covers a model absent from one row of an
                # otherwise per-model window (lineup change mid-run).
                logger.warning(
                    "[optimization-state] Plateau window has no per-model "
                    "data — plateau verdict withheld; the run is bounded by "
                    "max_iterations.",
                )

        return (None, None)

    def reset_for_resume(self) -> None:
        """
        Reset state for resuming optimization (in-memory transition only).

        Only bumps max_iter_cap when resuming from max_iterations (adds another
        full max_iterations budget). Cancelled/plateau resumes keep the existing cap.
        Plateau detection uses full history — no reset needed.

        **Callers MUST persist after calling this.** The production caller is
        the resume REST endpoint (``api/optimization.py``), which persists once
        afterwards together with any workload-drift annotation, so version
        threading is a single conditional write rather than two chained ones.
        Keeping the save out of this method also keeps it matching its name
        (reset = in-memory transition).
        """
        old_status = self.status
        old_cap = self.max_iter_cap

        # Only extend the cap when resuming from max_iterations
        if old_status == "max_iterations":
            self.max_iter_cap = self.current_iteration + self.max_iterations

        self.status = "running"
        # Clear any prior terminal error so the UI banner doesn't persist
        # into the fresh run. Operators can still inspect the prior error
        # in logs or the iteration history.
        self.error = None
        # No save here — the caller owns the persist (see docstring)
        logger.info("[optimization-state] Reset for resume: status=%s→running cap %d→%d iter=%d",
                    old_status, old_cap, self.max_iter_cap, self.current_iteration)

    def best_iteration(self) -> dict[str, Any] | None:
        """
        Find the iteration with the highest single-model pass rate.

        Matches the "any model converges" semantics of ``check_convergence``
        — the winner is the iteration where some model performed best on
        its own, not the one with the highest aggregate across all models.
        If ``winning_model_id`` is set (convergence fired), we pick the
        iteration where that specific model scored highest. Otherwise we
        scan every iteration's ``per_model_pass_rates`` for the single
        best per-model score and return that iteration.

        Rows without ``per_model_pass_rates`` rank LAST rather than falling back
        to their aggregate rate. The fallback existed as display-only tolerance
        for rows recorded before per-model metrics existed — kept when the
        convergence and plateau fallbacks were deleted, on the grounds that
        blanking those panels was worse than a rough ranking. Pre-release there
        are no such rows to protect, and mixing the two bases was its own defect:
        an aggregate across models and a single model's best score are different
        quantities, so a row ranked on one could beat a row ranked on the other
        and win a comparison it should have lost.

        Returns:
            Iteration dict for the winner, or None if no iterations.
        """
        if not self.iterations:
            return None

        def _best_model_rate(it: dict) -> float:
            """Highest per-model pass rate, or -1 when the row has none.

            Ranking last rather than substituting the aggregate: comparing one
            model's best score against a cross-model average silently compares
            two different quantities.
            """
            pm = it.get("per_model_pass_rates")
            if isinstance(pm, dict) and pm:
                return max(pm.values())
            return -1.0

        # Convergence already picked the winning model — find its best
        # iteration (the highest score that model posted across the run).
        if self.winning_model_id:
            def _winner_rate(it: dict) -> float:
                pm = it.get("per_model_pass_rates")
                if isinstance(pm, dict) and self.winning_model_id in pm:
                    return pm[self.winning_model_id]
                return -1.0  # iterations that didn't include the winner rank last

            candidate = max(self.iterations, key=_winner_rate)
            if _winner_rate(candidate) >= 0.0:
                return candidate

        # No declared winner yet — return whichever iteration showed the
        # strongest single-model performance so far. When NO row carries
        # per-model data nothing is rankable, and returning an arbitrary row
        # (what ``max`` does on an all-equal key) would present a guess as the
        # best result.
        best = max(self.iterations, key=_best_model_rate)
        return best if _best_model_rate(best) >= 0.0 else None

    def save(
        self,
        *,
        arch_split: bool = False,
        user_id: str | None = None,
        project_id: str | None = None,
        expected_version: int | None = None,
    ) -> int | None:
        """
        Persist OptimizationState to disk as JSON (old path) or DynamoDB (new path).

        **Flag-branched behavior:**

        - **New path** (arch_split=True): Routes the serialized data dict to
          ``put_optimization_state(user_id, project_id, data, expected_version)``
          and writes NO ``optimization/state.json`` file. Requires ``user_id``,
          ``project_id``, and ``expected_version`` params.

        - **Old path** (arch_split=False): Writes to ``optimization/state.json``
          via atomic temp+os.replace, exactly as before. Ignores store params.

        Uses atomic write-to-temp-then-rename (old path) so a crash mid-write
        won't leave a corrupt file. A per-state-file lock serializes concurrent
        writers so an orchestrator save doesn't stomp on an overlapping
        ``record_optimization_iteration`` call. Creates the
        ``output/optimization/`` directory if it doesn't exist.

        Args:
            arch_split: If True, route to DynamoDB via put_optimization_state
                (no state.json written). If False, write state.json (old path).
            user_id: Required on new path for put_optimization_state call.
            project_id: Required on new path for put_optimization_state call.
            expected_version: Required on new path; the conditional-write
                guard. Raises ``ConcurrentUpdateError`` on version mismatch.

        Returns:
            On the new path (arch_split=True), returns the new version number
            from the store's conditional write. On the old path (arch_split=False),
            returns None. Callers MUST thread the returned version forward to
            the next write; guessing ``expected_version + 1`` desynchronizes
            the version chain on concurrent writes.
        """
        state_file = self.project_dir / OPTIMIZATION_STATE_FILE
        state_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "target_metric_scores": self.target_metric_scores,
            "max_iterations": self.max_iterations,
            "plateau_window": self.plateau_window,
            "plateau_threshold": self.plateau_threshold,
            "iterations": self.iterations,
            "status": self.status,
            "max_iter_cap": self.max_iter_cap,
            "workload": self.workload,
            "pending_resume_drift": self.pending_resume_drift,
            "winning_model_id": self.winning_model_id,
            "current_prompt_version": self.current_prompt_version,
            "phase": self.phase,
            "error": self.error.to_dict() if self.error else None,
        }

        if arch_split:
            # Durable-state path: route to DynamoDB via put_optimization_state
            if user_id is None or project_id is None:
                raise ValueError(
                    "save() requires user_id and project_id params when writing "
                    "to the durable store — they form the record's key"
                )
            # expected_version is optional on first write (None -> attribute_not_exists guard)
            import asyncio
            from app.platform import get_platform

            store = get_platform().metadata_store

            # Sync wrapper for the async store method (save() is sync-called
            # from record_iteration and check_convergence in the executor)
            def _sync_put() -> int:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    # No running loop — create one and run_until_complete
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(
                            store.put_optimization_state(
                                user_id, project_id, data, expected_version=expected_version
                            )
                        )
                    finally:
                        loop.close()
                        asyncio.set_event_loop(None)
                else:
                    # A loop is already running in this thread, so run the
                    # coroutine on a fresh loop in a worker thread and wait.
                    import threading
                    result = [None]
                    error = [None]
                    def _run():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            result[0] = new_loop.run_until_complete(
                                store.put_optimization_state(
                                    user_id, project_id, data, expected_version=expected_version
                                )
                            )
                        except Exception as e:
                            error[0] = e
                        finally:
                            new_loop.close()
                            asyncio.set_event_loop(None)
                    thread = threading.Thread(target=_run)
                    thread.start()
                    thread.join()
                    if error[0]:
                        raise error[0]
                    return result[0]

            new_version = _sync_put()
            logger.debug(
                "[optimization-state] Saved (DDB): status=%s iter=%d cap=%d version=%d",
                self.status, self.current_iteration, self.max_iter_cap, new_version,
            )
            # Do NOT write state.json on the new path
            return new_version
        else:
            # Old path: write state.json via atomic temp+os.replace
            # Serialize concurrent writers on the same state file. The rename
            # is atomic, but without this two writers can each produce a temp
            # and the later rename silently overwrites the earlier update.
            with _get_save_lock(state_file):
                with tempfile.NamedTemporaryFile(
                    dir=state_file.parent, suffix=".json", delete=False, mode="w"
                ) as f:
                    json.dump(data, f, indent=2)
                    # The name is read here, but the file is replaced only after
                    # this block, where the context manager has already closed
                    # and flushed it.
                    # nosemgrep: tempfile-without-flush
                    tmp_path = f.name
                os.replace(tmp_path, state_file)

            # No per-owner chown: the EFS access point squashes UID, so all
            # files are written as UID 1000 regardless of the project owner.
            # Tenant isolation lives at the app layer (``require_access``).

            logger.debug(
                "[optimization-state] Saved (EFS): status=%s iter=%d cap=%d",
                self.status, self.current_iteration, self.max_iter_cap,
            )
            return None
