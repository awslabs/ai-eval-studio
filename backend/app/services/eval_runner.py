# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# backend/app/services/eval_runner.py
"""Reads evaluation results from a project's output directory.

Evals run in the project's own harness subprocess (``run_eval.py``); this
module loads the run files it writes, for the REST API, reports and the agent's
tools.
"""

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

# Number of characters to include in output/input preview fields for compact projections
_PREVIEW_CHARS = 300


logger = logging.getLogger(__name__)


class RunFileMissingError(FileNotFoundError):
    """A run's results file does not exist on disk.

    Raised by the ``load_*`` readers below so the API layer can map a deleted
    run to a 404 without importing HTTP concerns into this service. Subclasses
    ``FileNotFoundError`` so a caller that only cares about the OS-level meaning
    still catches it.

    This exists because listing a run and reading it are two separate steps: a
    delete landing between them is a normal race, not a server fault. Every
    other IO failure (permissions, a truncated or corrupt file) is a real 500
    and must NOT be funneled through this type.
    """


def _read_run_json(results_path: Path) -> list | dict:
    """Read and parse a run's results JSON, translating absence to a domain error.

    Single place where a run file is read, so the deleted-between-list-and-read
    race is covered for every caller rather than at one route. A missing file
    raises :class:`RunFileMissingError`; a corrupt file's ``JSONDecodeError`` and
    any other ``OSError`` propagate untouched, because those are genuine faults.

    Args:
        results_path: Path to the results JSON file.

    Returns:
        The parsed JSON — an array of per-model run dicts.

    Raises:
        RunFileMissingError: The file does not exist.
    """
    try:
        raw = results_path.read_text()
    except FileNotFoundError as exc:
        raise RunFileMissingError(f"Run file missing: {results_path.name}") from exc
    return json.loads(raw)


def _derived_dir(results_path: Path) -> Path:
    """Return the sidecar directory for a given results file.

    Derived files (compact projections, per-case details) are stored under
    ``output/_derived/<run_stem>/`` to keep them co-located with their
    source run file but isolated from the run file listing glob. Runs are
    immutable once written (the harness subprocess writes them atomically),
    so a sidecar is valid for the life of the run.

    Args:
        results_path: Path to the results JSON file (e.g., output/eval_results_*.json).

    Returns:
        Path to the derived data directory for this run (e.g., output/_derived/eval_results_*/).
    """
    return results_path.parent / "_derived" / results_path.stem


def _atomic_write_json(target: Path, payload) -> None:
    """Write JSON to a file atomically via temp file + os.replace.

    Creates parent directories if they don't exist. Writes to a temp file in the
    SAME directory as the target (so the rename is same-filesystem, guaranteed
    atomic on POSIX), then atomically publishes via os.replace. On crash or
    exception, the target is left in its prior state (old-or-new, never torn).

    Cleans up the temp file in a finally block so a failed write never leaves
    a .tmp orphan.

    Args:
        target: Destination path for the JSON file.
        payload: Python object to serialize (via json.dumps with default=str).

    Raises:
        OSError, IOError: If the write or rename fails.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    tmp_path_obj = Path(tmp_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            # Match existing serialization posture: default=str, no indent for compact storage
            json.dump(payload, f, default=str)
        os.replace(tmp_path, target)
    except Exception:
        # Clean up temp file if write or replace failed
        try:
            tmp_path_obj.unlink(missing_ok=True)
        except Exception:
            pass  # Best effort — log nothing, the outer exception is what matters
        raise


def _case_sidecar_name(case_id: str) -> str:
    """Hash a case_id into a safe filename for per-case sidecar storage.

    Case IDs may contain arbitrary characters (slashes, whitespace, unicode)
    unsafe for filenames. SHA1-hashing produces a fixed-length, filesystem-safe
    string. No index is needed because the endpoint always supplies the case_id
    to retrieve the corresponding detail.

    ``usedforsecurity=False`` records what this digest is for: a name, not a
    signature. Nothing authenticates a case_id from its filename, so SHA1's
    collision weakness costs nothing here, and saying so keeps the call off the
    list of weak-hash findings a scanner would otherwise raise every run.

    Args:
        case_id: Test case identifier (may contain unsafe characters).

    Returns:
        Safe filename: sha1(case_id).json (e.g., "3f2a8b...c4d5.json").
    """
    # nosemgrep: B303-2, insecure-hash-algorithm-sha1
    return hashlib.sha1(
        case_id.encode("utf-8"), usedforsecurity=False
    ).hexdigest() + ".json"


def _missing_required_fields(data: list[dict]) -> str | None:
    """Return a human-readable description of any required-field gap, or None.

    Every model_run in a list_runs payload must carry ``test_run_id``
    and ``hyperparameters`` with ``model``. ``endpoint`` / ``region`` are
    OPTIONAL: in-process / pre-convention projects write result files with
    only ``{model, temperature, max_tokens, prompt_version}`` (no routing
    metadata), and requiring endpoint/region skipped every such run — the
    whole Evaluations/Reports view went empty for real user projects. The
    summary builder defaults the missing routing fields (endpoint→"runtime",
    region→""), so they are not needed to render a run.
    Validating per-row (not just data[0]) prevents partial-write
    multi-model files from silently disappearing the entire run via
    the outer ``except KeyError``.
    """
    if not data:
        return "empty data array"
    for idx, model_run in enumerate(data):
        if not isinstance(model_run, dict):
            return f"model_run[{idx}] is not a dict"
        if "test_run_id" not in model_run:
            return f"model_run[{idx}] missing test_run_id"
        hp = model_run.get("hyperparameters")
        if not isinstance(hp, dict):
            return f"model_run[{idx}] missing hyperparameters"
        if "model" not in hp:
            return f"model_run[{idx}] missing hyperparameters.model"
    return None


class EvalRunner:
    """Loads and lists a project's eval run files.

    Eval execution goes through the project's harness subprocess
    (run_eval.py). This class reads what it writes.
    """

    def load_results(self, results_path: Path) -> list | dict:
        """Load evaluation results from a JSON file.

        Args:
            results_path: Path to the results JSON file.

        Returns:
            Parsed results (array of model run dicts).

        Raises:
            RunFileMissingError: The run file was deleted (or never existed).
        """
        return _read_run_json(results_path)

    def load_compact(self, results_path: Path) -> list[dict]:
        """Load evaluation results in compact form, with write-through sidecar caching.

        Preserves per-model-run metadata (test_run_id, timestamp,
        hyperparameters) but replaces full per-case results with slim
        projections: drops full text fields (input, actual_output,
        expected_output, context, retrieval_context), drops per-metric
        reason/threshold/error, promotes latency/token fields from
        additional_metadata to top level, and adds output_hash + preview
        fields for change detection and grid display.

        On first access, computes the compact projection and writes it to
        ``output/_derived/<stem>/compact.json`` for fast subsequent loads.
        An mtime guard recomputes if the run file is newer than the sidecar
        (self-heals a stale or corrupt cache).

        Args:
            results_path: Path to the results JSON file.

        Returns:
            List of compact model-run dicts, each with slimmed results.

        Raises:
            RunFileMissingError: The run file was deleted (or never existed).
        """
        derived = _derived_dir(results_path)
        compact_path = derived / "compact.json"

        # FAST PATH: read sidecar if it exists and is not stale
        if compact_path.exists():
            try:
                run_mtime = results_path.stat().st_mtime
                sidecar_mtime = compact_path.stat().st_mtime
                if run_mtime <= sidecar_mtime:
                    # Sidecar is fresh — return cached projection
                    return json.loads(compact_path.read_text())
            except (OSError, json.JSONDecodeError):
                # Corrupt sidecar or missing run file — fall through to recompute
                pass

        # SLOW PATH: compute the compact projection from the full run file
        data = _read_run_json(results_path)
        if not isinstance(data, list):
            # Non-list file — return empty and create no sidecar
            return []

        def _num(meta: dict | None, key: str) -> int:
            """Extract numeric field from metadata dict, default to 0."""
            if meta is None:
                return 0
            val = meta.get(key)
            return int(val) if val is not None else 0

        compact_runs = []
        for model_run in data:
            results = model_run.get("results", [])
            compact_results = []
            for case in results:
                meta = case.get("additional_metadata") or {}
                case_id = str(
                    meta.get("id") if meta.get("id") is not None else case.get("name")
                )
                category = meta.get("category")
                actual_output_text = case.get("actual_output") or ""
                input_text = case.get("input") or ""

                # Change detection only: the compare grid asks whether two runs
                # produced the same text, so this is an equality shorthand and
                # not a signature. Hence usedforsecurity=False.
                # nosemgrep: B303-2, insecure-hash-algorithm-sha1
                output_hash = hashlib.sha1(
                    actual_output_text.encode("utf-8"), usedforsecurity=False
                ).hexdigest()
                output_preview = actual_output_text[:_PREVIEW_CHARS]
                input_preview = input_text[:_PREVIEW_CHARS]

                metrics = [
                    {
                        "name": md.get("name"),
                        "success": md.get("success"),
                        "score": md.get("score"),
                    }
                    for md in (case.get("metrics_data") or [])
                ]

                compact_results.append({
                    "name": case.get("name"),
                    "case_id": case_id,
                    "category": category,
                    "success": case.get("success"),
                    "metrics": metrics,
                    "latency_ms": _num(meta, "latency_ms"),
                    "input_tokens": _num(meta, "input_tokens"),
                    "output_tokens": _num(meta, "output_tokens"),
                    "cache_read_input_tokens": _num(meta, "cache_read_input_tokens"),
                    "cache_write_input_tokens": _num(meta, "cache_write_input_tokens"),
                    "output_hash": output_hash,
                    "output_preview": output_preview,
                    "input_preview": input_preview,
                })

            compact_runs.append({
                "test_run_id": model_run.get("test_run_id"),
                "timestamp": model_run.get("timestamp"),
                "hyperparameters": model_run.get("hyperparameters"),
                "results": compact_results,
            })

        # Write sidecar atomically for future reads
        _atomic_write_json(compact_path, compact_runs)
        return compact_runs

    def load_case_detail(self, results_path: Path, case_id: str) -> dict:
        """Load full per-case detail for a single test case, with write-through caching.

        Matches the case by ``additional_metadata.id ?? name`` and returns
        the full input/expected_output (from the first match) plus per-model
        actual_output, token/latency stats, and metric reasons. Used for
        on-demand lazy loading when the user expands a specific case row.

        On first access, parses the run file ONCE and writes ALL per-case detail
        sidecars to ``output/_derived/<stem>/cases/<sha1(case_id)>.json``, then
        returns the requested case. Subsequent loads for ANY case in the same run
        read only the small per-case sidecar without re-parsing the full file.

        An mtime guard recomputes if the run file is newer than any existing sidecar.
        If the requested case_id does not exist in the file, returns the empty-models
        sentinel and writes no sidecar for that id (preserves 404 semantics).

        Args:
            results_path: Path to the results JSON file.
            case_id: Test case identifier to match.

        Returns:
            Dict with ``input``, ``expected_output``, and ``models`` keyed
            by hyperparameters.model. If no model matched, ``models`` is
            empty (sentinel for 404).

        Raises:
            RunFileMissingError: The run file was deleted (or never existed).
        """
        derived = _derived_dir(results_path)
        cases_dir = derived / "cases"
        case_path = cases_dir / _case_sidecar_name(case_id)

        # FAST PATH: read sidecar if it exists and is not stale
        if case_path.exists():
            try:
                run_mtime = results_path.stat().st_mtime
                sidecar_mtime = case_path.stat().st_mtime
                if run_mtime <= sidecar_mtime:
                    # Sidecar is fresh — return cached detail
                    return json.loads(case_path.read_text())
            except (OSError, json.JSONDecodeError):
                # Corrupt sidecar or missing run file — fall through to recompute
                pass

        # SLOW PATH: parse the run file once and write all case sidecars
        data = _read_run_json(results_path)
        if not isinstance(data, list):
            # Non-list file — return sentinel, create no sidecar
            return {"input": "", "expected_output": "", "models": {}}

        def _num(meta: dict | None, key: str) -> int:
            """Extract numeric field from metadata dict, default to 0."""
            if meta is None:
                return 0
            val = meta.get(key)
            return int(val) if val is not None else 0

        # Build detail dict for ALL cases in one pass
        by_case_id: dict[str, dict] = {}
        for model_run in data:
            results = model_run.get("results", [])
            hyperparameters = model_run.get("hyperparameters", {})
            model_id = hyperparameters.get("model", "")

            for case in results:
                meta = case.get("additional_metadata") or {}
                current_case_id = str(
                    meta.get("id") if meta.get("id") is not None else case.get("name")
                )

                # Initialize case entry on first model encounter
                if current_case_id not in by_case_id:
                    by_case_id[current_case_id] = {
                        "input": case.get("input") or "",
                        "expected_output": case.get("expected_output") or "",
                        "models": {},
                    }

                metrics = [
                    {
                        "name": md.get("name"),
                        "success": md.get("success"),
                        "score": md.get("score"),
                        "reason": md.get("reason"),
                    }
                    for md in (case.get("metrics_data") or [])
                ]

                by_case_id[current_case_id]["models"][model_id] = {
                    "actual_output": case.get("actual_output") or "",
                    "latency_ms": _num(meta, "latency_ms"),
                    "input_tokens": _num(meta, "input_tokens"),
                    "output_tokens": _num(meta, "output_tokens"),
                    "cache_read_input_tokens": _num(meta, "cache_read_input_tokens"),
                    "cache_write_input_tokens": _num(meta, "cache_write_input_tokens"),
                    "metrics": metrics,
                }

        # Write all case sidecars atomically
        for cid, detail in by_case_id.items():
            _atomic_write_json(cases_dir / _case_sidecar_name(cid), detail)

        # Return the requested case, or the empty-models sentinel if not found
        if case_id in by_case_id:
            return by_case_id[case_id]
        else:
            # Unknown case_id — return sentinel, do not write a sidecar for it
            return {"input": "", "expected_output": "", "models": {}}

    def list_runs(self, project_dir: Path) -> list[dict]:
        """List all evaluation runs in a project, newest first.

        Each run file is a JSON array of model runs. Returns lightweight
        summaries with model names and counts (no full results).

        Args:
            project_dir: Root directory of the project.

        Returns:
            List of run summary dicts, ordered by timestamp descending.
        """
        output_dir = project_dir / "output"
        if not output_dir.exists():
            return []

        runs = []
        # Scan all JSON files — platform writes eval_results_*.json,
        # harness may write results_*.json or other patterns.
        for path in output_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                if not isinstance(data, list):
                    continue  # Skip non-array files

                # Extract summary from model runs
                first = data[0] if data else {}
                prompt_version = first.get("hyperparameters", {}).get("prompt_version", "")
                timestamp = first.get("timestamp", "")
                source = first.get("source")

                # Validation smoke runs are tagged ``source="validation"``
                # so they never mingle with real evaluations in the
                # dashboard.
                if source == "validation":
                    continue

                # Every model_run in the file must carry the required
                # hyperparameters — checking only data[0] would let a
                # later half-formed model_run silently disappear the
                # whole file via the broad except below. Skip-and-warn
                # at file granularity if ANY model_run is missing the
                # contract, so the user gets a single log line instead
                # of a vanishing run.
                missing = _missing_required_fields(data)
                if missing:
                    logger.warning(
                        "[list_runs] skipping %s: %s",
                        path.name, missing,
                    )
                    continue

                model_summaries = []
                for model_run in data:
                    results = model_run.get("results", [])
                    passed = sum(1 for r in results if r.get("success", False))
                    total = len(results)

                    # Aggregate per-metric pass counts so consumers (the UI's
                    # trophy-selection logic, primarily) can reason about
                    # target-meets-checks on manual runs the same way it does
                    # for optimization-loop iterations. Keys match DeepEval's
                    # metric names (which may carry a "[GEval]" suffix).
                    per_metric: dict[str, dict[str, int]] = {}
                    for r in results:
                        for md in r.get("metrics_data", []) or []:
                            name = md.get("name")
                            if not name:
                                continue
                            bucket = per_metric.setdefault(name, {"passed": 0, "total": 0})
                            bucket["total"] += 1
                            if md.get("success", False):
                                bucket["passed"] += 1
                    per_metric_rates = {
                        name: (stats["passed"] / stats["total"]) if stats["total"] else 0.0
                        for name, stats in per_metric.items()
                    }

                    model_hp = model_run["hyperparameters"]
                    model_summaries.append({
                        "test_run_id": model_run["test_run_id"],
                        "model": model_hp["model"],
                        # Routing metadata sourced from the descriptor
                        # at write time (template runner) — never
                        # parsed back out of the model id. OPTIONAL:
                        # in-process / pre-convention runs omit them, so
                        # default rather than KeyError-and-skip the file.
                        "endpoint": model_hp.get("endpoint", "runtime"),
                        "region": model_hp.get("region", ""),
                        "total": total,
                        "passed": passed,
                        "per_metric": per_metric_rates,
                    })

                runs.append({
                    "_file": path.name,
                    "timestamp": timestamp,
                    "prompt_version": prompt_version,
                    "source": source or "manual",
                    "model_runs": model_summaries,
                })
            except (json.JSONDecodeError, KeyError, IndexError) as exc:
                # Defense in depth: ``_missing_required_fields`` should
                # already have skipped malformed files. Anything that
                # still raises here is a structural surprise — log so
                # operators can investigate instead of silently dropping.
                logger.warning(
                    "[list_runs] skipping %s due to %s: %s",
                    path.name, type(exc).__name__, exc,
                )
                continue

        # Sort by the actual timestamp field (ISO 8601), newest first
        runs.sort(key=lambda r: r["timestamp"], reverse=True)
        return runs

    def short_model_name(self, model_id: str) -> str:
        """Return a friendly short name for a Bedrock model id.

        Thin wrapper around ``app.services.model_id_short.short_model_id``
        so consumers (report rendering, run serialization) keep their
        existing entry point. The structural rule deliberately avoids
        any hardcoded provider/region/model list — see
        ``model_id_short.py`` for the algorithm and verified examples.

        Args:
            model_id: Full Bedrock model identifier.

        Returns:
            Short friendly name (verbatim remainder after prefix/suffix
            strip; dotted versions like ``grok-4.3`` keep their dots).
        """
        from app.services.model_id_short import short_model_id

        return short_model_id(model_id)
