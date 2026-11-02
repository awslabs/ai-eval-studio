# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP tools for harness lifecycle: scaffold_harness, set_models, validate_harness.

``scaffold_harness`` copies boilerplate template files into a project directory,
creating the full harness structure while protecting project-specific files that
already exist.

``set_models`` rewrites the project's ``src/config.py`` ``models`` and
``judge_model_id`` fields after server-side validation against the merged
runtime catalog and fully-resolved mantle descriptors. This is the
load-bearing gate that replaces the agent prompt's "write src/config.py
directly" instruction. All other EvalConfig fields are preserved
byte-for-byte; persistence is atomic (``.tmp`` + ``os.replace``) under the
per-project async lock.

``validate_harness`` performs pre-flight checks: verifies required
files exist, installs dependencies via ``uv sync``, tests that key modules import
correctly, and exercises the target and judge models with a quick generation call.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import re
import shutil
import time
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any


def _project_subprocess_env() -> dict[str, str]:
    """Return a subprocess env with the parent VIRTUAL_ENV stripped.

    Letting the project subprocess inherit the backend's VIRTUAL_ENV makes
    ``uv run`` complain that the active env doesn't match the project's
    ``.venv``. We clear it so uv activates the project's own venv cleanly.

    Also clamps the Bedrock read-timeout + retry knobs for validation
    only: a non-invocable model (one that hangs on Converse) must not
    consume the production 300s × 4 retries budget here, or validation
    itself reproduces the 20-min stall it's trying to catch. Production
    runs use ``harness_runner._project_subprocess_env`` (separate
    function) and keep the full retry budget — this override is
    isolated to the validate-harness path.

    ``MANTLE_*`` env knobs (``MANTLE_MAX_CONNECTIONS``,
    ``MANTLE_MAX_KEEPALIVE``, ``MANTLE_BOOT_SMOKE``, etc.) are
    automatically forwarded via ``os.environ.copy()`` —
    Do NOT add explicit ``MANTLE_*`` env-set
    lines here; that creates a third surface to keep in sync.
    Asserted by
    ``tests/test_subprocess_env_mantle.py::test_validation_factory_propagates_mantle_env``.
    """
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    # Also strip UV_PROJECT_ENVIRONMENT so a child ``uv run``/``uv sync`` in the
    # harness project resolves the project's own environment, not the
    # coordinator's pinned backend venv (which, in dev, is relocated via this
    # same var — a leak would let a project sync re-resolve the backend venv).
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    env["BEDROCK_READ_TIMEOUT_S"] = "30"
    env["BEDROCK_RETRY_MAX_ATTEMPTS"] = "1"
    return env

from claude_agent_sdk import tool

from app.services.bedrock_pricing import write_pricing_file
from app.services.harness_config import parse_harness_config
from app.services.harness_runner import (
    UV_SYNC_TIMEOUT_S,
    VALIDATION_PROBE_TIMEOUT_S,
    SubprocessTimeout,
    run_bounded_subprocess,
)
from app.services.mcp_response import mcp_error, mcp_success
from app.services.project_locks import get_project_lock
from app.services.project_paths import ProjectTraversalError, resolve_project_dir

logger = logging.getLogger(__name__)


#: Timeout for the metric self-test subprocess. Deterministic metrics finish in
#: well under a second; the budget is judge latency, and the earlier 120s was
#: sized on a wrong unit — "two judge calls per judge-backed metric" counted
#: ``measure()`` calls, but ONE ``measure()`` is several judge calls (measured on
#: DeepEval 3.9.9: AnswerRelevancy 3, Faithfulness 4, ContextualPrecision and
#: ContextualRecall 2 each, GEval 1 + steps). The degenerate check scores 3
#: cases, so a four-metric RAG lineup is ~33 serialized judge calls — past 120s
#: at any realistic per-call latency, and further past it under throttling.
#:
#: Exceeding this is a WARNING, not a failure (see the timeout handler): the
#: ceiling measures judge latency, not a property of the metric, and it would be
#: the same false positive this feature already refuses to make when it declines
#: to block on one nondeterministic judge sample.
_SELFTEST_TIMEOUT_S = 300


def _probe_source() -> str:
    """Read the metric self-test probe's source, to run inside the project venv.

    Passed BY VALUE rather than copied into the project tree, so a project
    scaffolded months ago runs the current probe and there is no template-sync
    or overwrite-protection question. Resolved from this module's own location
    (``Path(__file__).resolve()``) because the working directory during a
    validation is the PROJECT, not the backend.
    """
    probe = (
        Path(__file__).resolve().parent.parent / "metric_selftest_probe.py"
    )
    if not probe.exists():  # packaging error — fail loudly, not silently
        raise FileNotFoundError(f"metric self-test probe missing at {probe}")
    return probe.read_text(encoding="utf-8")


def _read_selftest_findings(payload: dict) -> tuple[list[str], list[str]]:
    """Split a probe payload into blocking problems and warnings.

    Only ONE class blocks: a metric that scores locally (no judge call) and gave
    the SAME verdict for a correct answer and for output that is not an answer at
    all. That is conclusive for a deterministic scorer — always-fail zeroes every
    run, always-pass certifies every run — and no task knowledge is needed to
    call it wrong.

    Everything else warns, including the same degeneracy on a JUDGE-backed
    metric: one sample of a nondeterministic scorer is not proof, and blocking a
    user's whole workflow on it would be a false positive waiting to happen.
    The formatting battery cannot block by construction — rejecting a code fence
    is correct for a deliberately strict metric and a defect for one that claims
    tolerance, and only the metric's author knows which.

    Args:
        payload: Parsed probe JSON (``status``/``metrics``).

    Returns:
        ``(blocking, warnings)`` — human-readable lines, empty when clean.
    """
    blocking: list[str] = []
    warnings: list[str] = []

    for entry in payload.get("metrics") or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("class") or "?"
        judge_backed = bool(entry.get("judge_backed"))
        degenerate = entry.get("degenerate") or {}
        verdict = degenerate.get("verdict")
        detail = degenerate.get("detail") or ""

        if verdict in ("always_fail", "always_pass", "inverted"):
            line = f"{name}: {detail}"
            if judge_backed:
                warnings.append(
                    f"{line} (judge-backed, so this is one nondeterministic "
                    "sample rather than proof — re-run or read the judge's "
                    "reason before changing the metric)"
                )
            else:
                blocking.append(line)
        elif verdict == "unknown":
            warnings.append(
                f"{name}: could not be probed — {detail or 'no detail'}. Nothing "
                "here says whether it measures anything."
            )
        elif verdict == "ok" and degenerate.get("max_attempts", 1) > 1:
            # A clean verdict that needed retries is still a finding: the same
            # judge flakiness that cost the probe extra samples corrupts scores
            # on the real run, where nothing retries an uninterpretable verdict.
            warnings.append(
                f"{name}: needed "
                f"{degenerate['max_attempts']} attempts to get a verdict — the "
                "judge failed at least one sample transiently (bad JSON or "
                "throttling). The metric checks out, but expect the same "
                "flakiness to cost you cases in a real run."
            )
        elif verdict == "not_probed":
            warnings.append(
                f"{name}: NOT self-tested — {detail or 'no detail'}. Nothing "
                "here says whether it measures anything; check it by hand."
            )

        formatting = entry.get("formatting") or {}
        rejected = formatting.get("rejected") or []
        if rejected:
            warnings.append(
                f"{name}: rejects formatting-only differences "
                f"({', '.join(rejected)}). Correct if the metric is meant to be "
                "strict — those failures are visible in the results. A defect if "
                "it is meant to tolerate them, and each one scores a right "
                "answer wrong."
            )
        for err in formatting.get("errors") or []:
            if isinstance(err, dict):
                warnings.append(
                    f"{name}: raised on the {err.get('variant')} variant — "
                    f"{err.get('error')}"
                )

    return blocking, warnings

@tool(
    "scaffold_harness",
    "Copy eval harness template files into the project directory. Creates boilerplate "
    "files (run_eval.py, bedrock_runtime_model.py, runner.py, pyproject.toml, tests, README) "
    "so you only need to write project-specific files: src/config.py, "
    "src/evaluation/dataset_loader.py, and src/metrics/metrics.py.",
    {"project_id": str},
)
async def scaffold_harness(args: dict[str, Any]) -> dict[str, Any]:
    """Copy harness template files into a project directory.

    Scaffolds boilerplate files from the template directory, creating the full
    harness structure. Skips project-specific files (config.py, dataset_loader.py,
    src/metrics/metrics.py) if they already exist to avoid overwriting user work.
    Overwrites template files (run_eval.py, runner.py, etc.) to support re-scaffolding.

    Args:
        args: Dict with project_id.

    Returns:
        Dict with text listing created/skipped files and reminder of files to write.
    """
    logger.info("[scaffold_harness] project=%s", args.get("project_id", "?")[:12])
    try:
        project_dir = resolve_project_dir(args["project_id"])
        template_dir = Path(__file__).resolve().parent.parent.parent / "templates" / "harness"

        if not template_dir.exists():
            return mcp_error(f"Template directory not found: {template_dir}")

        # Load project name for placeholder substitution
        config_path = project_dir / "config.json"
        if not config_path.exists():
            return mcp_error("Project config.json not found")

        config_data = json.loads(config_path.read_text())
        raw_project_name = config_data.get("name", "eval-project")
        # PEP 503-compatible slug for pyproject.toml [project].name.
        # Display names can contain spaces and punctuation; uv rejects both in
        # a package name.
        project_name = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_project_name).strip("-._") or "eval-project"

        # Pin the scaffolded project's DeepEval to the exact version the
        # backend is running with, so backend, template, and user-run
        # projects all resolve to byte-identical DeepEval.
        deepeval_version = pkg_version("deepeval")

        # Files that should NOT be overwritten if they already exist
        protect_files = {
            "src/config.py",
            "src/evaluation/dataset_loader.py",
        }

        created_files = []
        skipped_files = []

        # Walk all template files
        for template_path in template_dir.rglob("*"):
            if template_path.is_dir():
                continue

            # Compute relative path from template dir
            rel_path = template_path.relative_to(template_dir)
            target_path = project_dir / rel_path

            # Check if this file should be protected
            rel_path_str = str(rel_path).replace("\\", "/")
            should_skip = False

            # Skip __pycache__ directories and binary files
            if "__pycache__" in template_path.parts or template_path.suffix == ".pyc":
                continue

            # Protect specific files
            if rel_path_str in protect_files and target_path.exists():
                should_skip = True

            # Protect any file in src/metrics/ if it already exists
            if rel_path_str.startswith("src/metrics/") and target_path.exists():
                should_skip = True

            if should_skip:
                skipped_files.append(str(rel_path))
                continue

            # Create parent directories
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy file, substituting {project_name} / {deepeval_version}
            # placeholders in toml and markdown files only.
            content = template_path.read_text(encoding="utf-8")
            if target_path.suffix in {".toml", ".md"}:
                content = content.replace("{project_name}", project_name)
                content = content.replace("{deepeval_version}", deepeval_version)

            target_path.write_text(content, encoding="utf-8")
            created_files.append(str(rel_path))

        # Skills are NOT mirrored per-project. The bundled claude CLI
        # auto-discovers from ``$HOME/.claude/skills/`` (user) AND
        # ``$cwd/.claude/skills/`` (project). No platform skill ships
        # today; one added later belongs at user scope in the runtime
        # image, so scaffolding never writes skills into the project tree.

        # Build response message
        message_parts = []
        if created_files:
            message_parts.append("Created files:")
            message_parts.extend(f"  {f}" for f in sorted(created_files))

        if skipped_files:
            message_parts.append("\nSkipped (already exist):")
            message_parts.extend(f"  {f}" for f in sorted(skipped_files))

        message_parts.append("\nNext steps: Write these project-specific files:")
        message_parts.append("  1. src/config.py - EvalConfig dataclass with model config")
        message_parts.append("  2. src/evaluation/dataset_loader.py - load_test_cases() and format_user_message()")
        message_parts.append("  3. src/metrics/metrics.py - build_metrics() function")
        message_parts.append("  4. src/metrics/__init__.py - re-export: from .builder import build_metrics")

        return mcp_success("\n".join(message_parts))

    except Exception as e:
        logger.error("[scaffold_harness] FAILED: %s", e, exc_info=True)
        return mcp_error(f"Error scaffolding harness: {e}")


def _rewrite_config_models_and_judge(
    source: str,
    model_ids: list[str],
    judge_model_id: str,
) -> str:
    """Rewrite ONLY the ``models`` and ``judge_model_id`` fields of ``EvalConfig``.

    AST-based: parse the source, find the ``EvalConfig`` ``ClassDef``,
    locate its ``models`` and ``judge_model_id`` ``AnnAssign`` nodes, and
    replace each ``.value`` subtree with a fresh literal AST. Every other
    statement in the class body — and every node outside the class body —
    is preserved verbatim.

    ``ast.unparse`` is deterministic for the same input AST, so the
    output is idempotent: calling this helper twice with the same args
    produces byte-identical results.

    The replacement shape mirrors what ``services.harness_config.parse_harness_config``
    recognizes (``field(default_factory=lambda: [...])`` for the list,
    plain string literal for the judge id), so the rewritten file
    re-parses cleanly through the existing AST extractor.

    Args:
        source: Existing ``src/config.py`` source text.
        model_ids: New target model id list — order is preserved
            (cheapest-first contract).
        judge_model_id: New judge model id.

    Returns:
        The rewritten source text (with a trailing newline).

    Raises:
        ValueError: ``EvalConfig`` class not present in the source. The
            caller wraps this in ``mcp_error`` so the agent observes the
            failure instead of the turn dying.
    """
    tree = ast.parse(source)
    eval_config_cls: ast.ClassDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "EvalConfig":
            eval_config_cls = node
            break
    if eval_config_cls is None:
        raise ValueError("EvalConfig class not found in src/config.py")

    # Build the replacement AST nodes ONCE up front. Using ``ast.parse(...,
    # mode='eval').body`` for the field(default_factory=lambda: [...])
    # call gives us a complete Call node whose internals are valid AST.
    models_repr = repr(list(model_ids))
    new_models_value = ast.parse(
        f"field(default_factory=lambda: {models_repr})", mode="eval"
    ).body
    new_judge_value = ast.Constant(value=judge_model_id)

    found_models = False
    found_judge = False
    for item in eval_config_cls.body:
        if not isinstance(item, ast.AnnAssign):
            continue
        if not isinstance(item.target, ast.Name):
            continue
        if item.target.id == "models":
            item.value = new_models_value
            found_models = True
        elif item.target.id == "judge_model_id":
            item.value = new_judge_value
            found_judge = True

    # Defensive append path — the scaffolded EvalConfig always carries
    # both fields, so this branch is only hit if a user hand-edited the
    # template into a degraded state. Build minimal AnnAssign nodes
    # using ``list[str]`` / ``str`` annotations so the rewritten file
    # round-trips through ``parse_harness_config`` cleanly.
    if not found_models:
        eval_config_cls.body.append(
            ast.AnnAssign(
                target=ast.Name(id="models", ctx=ast.Store()),
                annotation=ast.Subscript(
                    value=ast.Name(id="list", ctx=ast.Load()),
                    slice=ast.Name(id="str", ctx=ast.Load()),
                    ctx=ast.Load(),
                ),
                value=new_models_value,
                simple=1,
            )
        )
    if not found_judge:
        eval_config_cls.body.append(
            ast.AnnAssign(
                target=ast.Name(id="judge_model_id", ctx=ast.Store()),
                annotation=ast.Name(id="str", ctx=ast.Load()),
                value=new_judge_value,
                simple=1,
            )
        )

    ast.fix_missing_locations(tree)
    rewritten = ast.unparse(tree)
    if not rewritten.endswith("\n"):
        rewritten += "\n"
    return rewritten


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically write ``text`` to ``path`` via ``.tmp`` + ``os.replace``.

    Same shape as ``model_descriptors.write_descriptors`` and
    ``bedrock_pricing.write_pricing_file``: a partial read of
    ``src/config.py`` would corrupt an eval, so the rename is mandatory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


@tool(
    "set_models",
    "Set the target models and judge model for a project. Server-side "
    "validates that EVERY id has either a runtime catalog entry OR a "
    "fully-resolved mantle descriptor — call list_bedrock_models first to "
    "discover valid ids. ALWAYS call set_models instead of writing "
    "src/config.py directly. Models stay in the order passed (the "
    "optimizer's cheapest-first contract relies on this). Returns mcp_error "
    "with the full bad-id list if any id is unknown. REFUSED while an "
    "optimization loop is running — the lineup is fixed for the duration of a "
    "run; cancel the loop first.",
    {
        "type": "object",
        "properties": {
            "project_id": {
                "type": "string",
                "description": "Project UUID.",
            },
            "model_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Target model ids in cheapest-first order. Each must "
                    "appear in the runtime catalog OR have a fully-resolved "
                    "mantle descriptor."
                ),
            },
            "judge_model_id": {
                "type": "string",
                "description": (
                    "Judge model id for LLM-as-a-judge metric scoring. "
                    "Must also appear in the runtime catalog OR descriptor map."
                ),
            },
        },
        "required": ["project_id", "model_ids", "judge_model_id"],
    },
)
async def set_models(args: dict[str, Any]) -> dict[str, Any]:
    """Validate and persist the project's target + judge models.

    Replaces the agent prompt's "write src/config.py directly" instruction.
    The flow:

      1. Validate the args shape.
      2. Resolve the project directory and confirm ``src/config.py`` exists
         (otherwise direct the agent to ``scaffold_harness`` first).
      2a. Refuse while an optimization loop is live — the lineup is fixed for
         the duration of a run. See the inline comment for why drift detection
         is not a substitute.
      3. Call ``services.bedrock_discovery.discover_models`` for a fresh
         merged catalog + descriptor map (no race
         between list and set).
      4. Reject any id that is neither in the runtime catalog nor in the
         fully-resolved mantle descriptor map; surface the FULL bad-id list
         in one mcp_error so the agent can fix everything in one pass.
      5. AST-rewrite the existing ``src/config.py`` ``models`` +
         ``judge_model_id`` fields under the per-project async lock and
         re-parse the result through ``parse_harness_config`` as a
         defense-in-depth check.

    Args:
        args: ``{"project_id": str, "model_ids": list[str], "judge_model_id": str}``.

    Returns:
        ``mcp_success`` on commit; ``mcp_error`` on any validation or
        write failure (the agent loop survives in every error path).
    """
    project_id = args.get("project_id") if isinstance(args, dict) else None
    if not isinstance(project_id, str) or not project_id:
        return mcp_error("project_id is required for set_models")

    model_ids = args.get("model_ids") if isinstance(args, dict) else None
    if not isinstance(model_ids, list) or not model_ids:
        return mcp_error("model_ids must be a non-empty list of strings")
    if not all(isinstance(m, str) and m for m in model_ids):
        return mcp_error("model_ids must be a non-empty list of non-empty strings")

    judge_model_id = args.get("judge_model_id") if isinstance(args, dict) else None
    if not isinstance(judge_model_id, str) or not judge_model_id:
        return mcp_error("judge_model_id must be a non-empty string")

    try:
        project_dir = resolve_project_dir(project_id)
    except ProjectTraversalError as exc:
        return mcp_error(f"Invalid project_id: {exc}")

    config_path = project_dir / "src" / "config.py"
    if not config_path.exists():
        return mcp_error(
            "Project src/config.py not found — call scaffold_harness first"
        )

    # Fence: a live optimization loop pins the lineup for the whole run.
    #
    # Both convergence rules are comparisons ACROSS iterations. ``target_reached``
    # asks whether any one model cleared every target; plateau asks whether every
    # model's series has flatlined over the window. Swap the lineup mid-run and
    # neither means what it says — a model can be declared the winner on
    # iterations it never ran, and a model added late has no history to flatline,
    # so it silently blocks the plateau verdict. The workload fingerprint does
    # DETECT the change and mark ``resume_drift``, but that is after-the-fact
    # recovery: it cannot reconstruct the comparison the run was making.
    #
    # Checked before discovery so this costs nothing when it trips. Uses the same
    # read and the same ``is_live()`` predicate as the eval-dispatch fence, so
    # "a loop is running" keeps one definition. ``arch_split`` is resolved inside
    # ``read`` because set_models is an operation entry point.
    from app.services import optimization_state_repository as opt_repo

    loop_handle = await opt_repo.read(project_id, project_dir)
    if loop_handle.is_live():
        return mcp_error(
            "Optimization loop is running — the model lineup is fixed for the "
            "duration of a run, because convergence and plateau both compare "
            "models across iterations. Cancel the loop first, then call "
            "set_models."
        )

    # Lazy import — same import-cycle reasoning as ``list_bedrock_models``:
    # ``mcp_tools/harness.py`` is reached early via ``mcp_tools/__init__.py``
    # → ``agent_runner.py``, while ``bedrock_discovery`` pulls in the
    # mantle adapter (DeepEval) on import.
    from app.services.bedrock_discovery import discover_models

    try:
        result = await discover_models(project_id, project_dir)
    except Exception as exc:  # noqa: BLE001 — surface as mcp_error so agent survives
        logger.exception("[set_models] discovery failed")
        return mcp_error(f"Discovery failed: {exc}")

    runtime_ids = {
        e["model_id"] for e in result.catalog if e.get("endpoint") == "runtime"
    }
    mantle_descriptors = result.descriptors  # already mantle-only

    requested = list(model_ids) + [judge_model_id]
    bad = [
        mid for mid in requested
        if mid not in runtime_ids and mid not in mantle_descriptors
    ]
    if bad:
        # Stable order, deduped — show every bad id once.
        seen: set[str] = set()
        unique_bad = [b for b in bad if not (b in seen or seen.add(b))]
        return mcp_error(
            f"Unknown model id(s): {unique_bad} — not in runtime catalog "
            "and no fully-resolved mantle descriptor exists. "
            "Call list_bedrock_models first."
        )

    # Build the rewritten source up front so any rewrite failure surfaces
    # BEFORE we acquire the project lock. The lock-held section only
    # covers the atomic file replace — keeps the lock window minimal.
    existing_source = config_path.read_text(encoding="utf-8")
    try:
        new_source = _rewrite_config_models_and_judge(
            existing_source, list(model_ids), judge_model_id
        )
    except (SyntaxError, ValueError) as exc:
        logger.error("[set_models] AST rewrite failed: %s", exc)
        return mcp_error(f"Failed to rewrite src/config.py: {exc}")

    # Defense-in-depth: re-parse the rewritten source via the existing
    # AST extractor and assert the round-trip preserves the new values.
    # Catches any future ``ast.unparse`` drift before it reaches disk.
    try:
        parsed = parse_harness_config(new_source)
    except Exception as exc:  # noqa: BLE001 — defensive
        logger.error("[set_models] post-rewrite parse failed: %s", exc)
        return mcp_error(
            f"Internal: AST rewrite produced unparseable config.py: {exc}"
        )
    if parsed.models != list(model_ids) or parsed.judge_model_id != judge_model_id:
        logger.error(
            "[set_models] round-trip mismatch — models=%r judge=%r",
            parsed.models, parsed.judge_model_id,
        )
        return mcp_error(
            "Internal: AST rewrite produced inconsistent config — please report"
        )

    lock = await get_project_lock(project_id)
    async with lock:
        await asyncio.to_thread(_atomic_write_text, config_path, new_source)

    return mcp_success(
        f"Set {len(model_ids)} target model(s) + judge: {judge_model_id} "
        "(src/config.py rewritten)"
    )


def _check_smoke_output(runs: Any) -> list[str]:
    """Validate the shape + content of a 1-case smoke eval output file.

    Asserts that the harness produced real metric scores for every test
    case — the exact shape the frontend consumes. Any ``None`` score,
    missing ``success`` flag, or silent ``success=False`` with no
    reason/error surfaces as a validation problem.

    Args:
        runs: The JSON payload parsed from ``output/eval_results_*.json``.
            Expected to be a list of per-model run dicts, each with a
            ``results`` list of DeepEval-serialized test cases.

    Returns:
        List of human-readable problem strings. Empty list means OK.
    """
    problems: list[str] = []

    if not isinstance(runs, list) or not runs:
        problems.append("Output is not a non-empty list of runs.")
        return problems

    for run_idx, run in enumerate(runs):
        if not isinstance(run, dict):
            problems.append(f"Run {run_idx}: not a dict.")
            continue
        run_model_id = run.get("model_id", f"run-{run_idx}")
        results = run.get("results")
        if not isinstance(results, list) or not results:
            problems.append(
                f"{run_model_id}: 'results' is empty or missing."
            )
            continue
        for case_idx, case in enumerate(results):
            prefix = f"{run_model_id} case {case_idx}"
            if not isinstance(case, dict):
                problems.append(f"{prefix}: not a dict.")
                continue
            metadata = case.get("additional_metadata")
            if metadata is None:
                problems.append(f"{prefix}: missing 'additional_metadata' key.")
            # The model MUST have produced non-empty output. Empty output
            # has two distinct causes that need different fix prose:
            #
            #   * MANTLE_REASONING_TRUNCATED: a Mantle Responses-API
            #     reasoning model burned the entire ``max_output_tokens``
            #     cap on hidden reasoning tokens before emitting any
            #     visible text. The fix is to RAISE
            #     ``EvalConfig.max_output_tokens`` (or
            #     ``EvalConfig.max_tokens`` — the runner maps it).
            #     Distinct from generic "empty output" because nothing
            #     about the harness wiring is broken.
            #
            #   * Anything else: generation silently failed (Bedrock
            #     rejected the system prompt, runner template isn't
            #     rendering placeholders, mantle-only id smuggled
            #     through Converse, etc.) — the harness wiring needs
            #     attention.
            #
            # Either way, scoring is meaningless on an empty actual_output;
            # skip the metric checks for this case.
            actual_output = case.get("actual_output")
            if not isinstance(actual_output, str) or not actual_output.strip():
                truncated = bool(
                    isinstance(metadata, dict)
                    and metadata.get("MANTLE_REASONING_TRUNCATED")
                )
                if truncated:
                    err_text = ""
                    if isinstance(metadata, dict):
                        err_text = str(metadata.get("error") or "").strip()
                    problems.append(
                        f"{prefix}: reasoning truncated — model burned the entire "
                        "max_output_tokens budget on hidden reasoning before "
                        "emitting any visible text. Raise "
                        "EvalConfig.max_output_tokens (currently the harness "
                        "default) and re-run validate_harness. "
                        f"Underlying truncation: {err_text or '(no detail)'}"
                    )
                else:
                    problems.append(
                        f"{prefix}: model produced empty output. "
                        "Generation failed — check the harness runner's "
                        "prompt rendering, the Bedrock wrapper's system-prompt "
                        "handling, and that mantle-only model ids are NOT "
                        "being routed through Converse (runner.py "
                        "_make_model dispatches by descriptor)."
                    )
                # If output is empty, metric scores are meaningless — skip them.
                continue
            metrics_data = case.get("metrics_data")
            if not isinstance(metrics_data, list) or not metrics_data:
                problems.append(f"{prefix}: 'metrics_data' is empty — no metrics scored.")
                continue
            for m_idx, metric in enumerate(metrics_data):
                if not isinstance(metric, dict):
                    problems.append(f"{prefix} metric {m_idx}: not a dict.")
                    continue
                m_name = metric.get("name", f"#{m_idx}")
                score = metric.get("score")
                success = metric.get("success")
                reason = metric.get("reason")
                error = metric.get("error")
                if not isinstance(score, (int, float)):
                    problems.append(
                        f"{prefix} metric {m_name!r}: null score "
                        "(metric did not produce a real value — check evaluation_params "
                        "and judge model inference_config)."
                    )
                if not isinstance(success, bool):
                    problems.append(f"{prefix} metric {m_name!r}: 'success' is not a bool.")
                if success is False and not reason and not error:
                    problems.append(
                        f"{prefix} metric {m_name!r}: silent failure "
                        "(success=False with no reason or error — DeepEval likely "
                        "swallowed an exception under ignore_errors=True)."
                    )

    return problems


@tool(
    "validate_harness",
    "Pre-flight check for the eval harness. Verifies all required files exist, "
    "installs dependencies with uv sync, and tests that key modules import correctly. "
    "Call this after writing all project-specific files.",
    {"project_id": str},
)
async def validate_harness(args: dict[str, Any]) -> dict[str, Any]:
    """Pre-flight validation for eval harness.

    Checks that all required files exist, runs uv sync to install dependencies,
    and tests that key modules import correctly. Returns pass/fail with specific
    error details.

    Args:
        args: Dict with project_id.

    Returns:
        Dict with validation results or error details.
    """
    logger.info("[validate_harness] project=%s", args.get("project_id", "?")[:12])
    try:
        project_dir = resolve_project_dir(args["project_id"])

        # Step 1: Check required files exist
        required_files = [
            "run_eval.py",
            "src/config.py",
            "src/evaluation/dataset_loader.py",
            "src/metrics/metrics.py",
            "pyproject.toml",
        ]

        missing_files = []
        for file in required_files:
            if not (project_dir / file).exists():
                missing_files.append(file)

        if missing_files:
            return mcp_error(
                "Missing required files:\n" + "\n".join(f"  - {f}" for f in missing_files)
            )

        # Step 2: Run uv sync
        uv_path = shutil.which("uv")
        if not uv_path:
            return mcp_error("uv not found on PATH. Install uv to validate harness.")

        logger.info("[validate_harness] Running uv sync in %s", project_dir)

        try:
            sync_rc, sync_stdout, sync_stderr = await run_bounded_subprocess(
                [uv_path, "sync"],
                timeout_s=UV_SYNC_TIMEOUT_S,
                cwd=str(project_dir),
                env=_project_subprocess_env(),
                label="uv sync",
            )
        except SubprocessTimeout as timeout:
            return mcp_error(
                f"{timeout}. Dependency resolution is wedged or the package "
                "index is unreachable; the process group was killed."
            )

        if sync_rc != 0:
            error_msg = sync_stderr.decode("utf-8", errors="replace").strip()
            return mcp_error(f"uv sync failed (exit {sync_rc}):\n{error_msg}")

        # Step 3: Validate config — instantiate EvalConfig, check data file,
        # verify model IDs with Bedrock, test generation with first model and
        # judge. The imports at the top of validate_code also serve as the
        # import check (this previously ran as a separate ``uv run`` step;
        # merged here to save one full interpreter/uv-run startup — an
        # ImportError surfaces through the same JSON error path).
        logger.info("[validate_harness] Validating imports, config, and model access")

        validate_code = """
import json, sys
try:
    # These imports double as the import check (a separate uv-run step
    # used to do exactly this before the config check; merged here to
    # save one interpreter/uv-run start — build_metrics is included so
    # the merged step covers the same surface). The Step-5 full-sweep
    # smoke eval (run via the real harness subprocess) is what actually
    # exercises every target + judge end-to-end; this step just confirms
    # imports + config + one-case data load.
    from src.config import EvalConfig
    from src.evaluation.dataset_loader import load_test_cases, format_user_message
    from src.metrics import build_metrics
    from src.models.bedrock_runtime_model import BedrockRuntimeModel  # noqa: F401 — import-check only
    # Mantle adapter + descriptor reader are imported here so a missing
    # symbol in the harness template surfaces at validation time rather
    # than at first dispatch. Loaded for side-effect / import-check only.
    from src.models.bedrock_mantle_model import BedrockMantleModel  # noqa: F401
    from src.models.bedrock_mantle_mappers import MantleAPIRejected  # noqa: F401
    from src.models.model_descriptors import read_descriptors  # noqa: F401

    config = EvalConfig()
    errors = []

    # Check data file exists
    from pathlib import Path
    data_path = Path("data/raw") / config.data_file if not config.data_file.startswith("data/") else Path(config.data_file)
    if not data_path.exists():
        errors.append(f"Data file not found: {data_path}")

    # Check prompt file exists
    prompt_path = Path("prompts") / f"{config.prompt_version}.md"
    if not prompt_path.exists():
        errors.append(f"Prompt file not found: {prompt_path}")

    # Check models list is not empty
    if not config.models:
        errors.append("No target models configured in EvalConfig.models")

    # Check judge model is set
    if not config.judge_model_id:
        errors.append("No judge_model_id configured in EvalConfig")

    if errors:
        print(json.dumps({"status": "error", "errors": errors}))
        sys.exit(0)

    # Test loading one test case
    cases = load_test_cases(config.data_file, 1)
    if not cases:
        errors.append("load_test_cases returned no cases")
    else:
        msg = format_user_message(cases[0])
        if not msg:
            errors.append("format_user_message returned empty string")

    if errors:
        print(json.dumps({"status": "error", "errors": errors}))
    else:
        print(json.dumps({
            "status": "ok",
            "judge_model": config.judge_model_id,
            "test_cases": len(cases),
            "models": len(config.models),
            "all_model_ids": list(set(config.models + [config.judge_model_id])),
            "target_models": list(config.models),
            "aws_region": getattr(config, "aws_region", "us-east-1"),
            "prompt_version": getattr(config, "prompt_version", "baseline"),
            "data_file": config.data_file,
        }))
except Exception as e:
    print(json.dumps({"status": "error", "errors": [str(e)]}))
"""

        try:
            _v_rc, v_stdout, v_stderr = await run_bounded_subprocess(
                [uv_path, "run", "python", "-u", "-c", validate_code],
                timeout_s=VALIDATION_PROBE_TIMEOUT_S,
                cwd=str(project_dir),
                env=_project_subprocess_env(),
                label="harness validation probe",
            )
        except SubprocessTimeout as timeout:
            return mcp_error(
                f"{timeout}. Importing the harness and constructing its metrics "
                "is stuck; the process group was killed."
            )

        v_output = v_stdout.decode("utf-8", errors="replace").strip()
        try:
            v_result = json.loads(v_output)
        except (json.JSONDecodeError, ValueError):
            v_err = v_stderr.decode("utf-8", errors="replace").strip()
            return mcp_error(f"Config validation failed:\n{v_err or v_output}")

        if v_result.get("status") == "error":
            error_list = "\n".join(f"  ✗ {e}" for e in v_result["errors"])
            return mcp_error(
                f"Config validation failed:\n{error_list}\n\n"
                "Fix the issues above and run validate_harness again."
            )

        # Step 5: Full-sweep smoke eval — run the real harness on 1 case
        # against EVERY target model + the judge in a single invocation.
        # The harness is the gate: if any model produces empty output or
        # any metric scores null, validation fails so the user sees the
        # broken model BEFORE a full eval is dispatched.
        #
        # Why not a per-model "Say OK" probe instead? Two
        # reasons. (1) A reachability ping returns OK for reasoning
        # models that burn the entire max_output_tokens budget on hidden
        # reasoning and emit empty visible text — the harness then
        # silently scores 0 across the board. (2) A separate probe
        # duplicates network cost and routing logic that the harness
        # already does correctly via runner.py + the descriptor map.
        # One full-sweep smoke is the honest signal: the harness either
        # runs end-to-end on every model, or it doesn't.
        target_models = v_result["target_models"]
        judge_model_id = v_result["judge_model"]
        prompt_version = v_result.get("prompt_version", "baseline")

        logger.info(
            "[validate_harness] Running 1-case full-sweep smoke eval across %d target models + judge",
            len(target_models),
        )

        # Single harness invocation across every target. Same builder
        # the real run uses — flag drift between smoke and real-run is
        # impossible by construction (``tests/test_harness_flag_contract.py``).
        # Smoke-specific values: 1 example, max-concurrent 1 (sequential
        # scoring — a misconfigured project must NOT trigger Bedrock
        # throttling during validation), exercise the --data-file path
        # against the project's own configured dataset so that flag is
        # always present in the smoke argv.
        from app.services.harness_runner import build_harness_command
        smoke_cmd = build_harness_command(
            uv_path,
            model_ids=target_models,
            prompt_version=prompt_version,
            json_progress=True,
            # ``source="validation"`` is the structural marker that
            # ``EvalRunner.list_runs`` filters out so smoke runs never
            # mingle with real evaluations in the dashboard. The
            # best-effort ``unlink()`` below remains defense-in-depth
            # for the happy-path cleanup; the source tag is the gate
            # that catches every smoke file regardless of subprocess
            # outcome.
            source="validation",
            max_test_cases=1,
            judge_model_id=judge_model_id,
            data_file_override=v_result.get("data_file"),
            max_concurrent=1,
        )
        # Record the mtime floor before launching so we only consider files
        # this subprocess wrote. Any ``eval_results_*.json`` left over from
        # a prior run has an older mtime and must not be picked up as the
        # smoke output.
        smoke_started_at = time.time()
        # Per-model timeout budget. 60s/model is enough for one case at
        # max_output_tokens; reasoning models with generous caps need
        # the headroom. Judge contributes one extra slot (it scores all
        # cases sequentially; same case-count budget applies).
        smoke_timeout_s = 60 * (len(target_models) + 1)
        try:
            smoke_rc, s_stdout, s_stderr = await run_bounded_subprocess(
                smoke_cmd,
                timeout_s=smoke_timeout_s,
                cwd=str(project_dir),
                env=_project_subprocess_env(),
                label="Smoke eval",
            )
        except SubprocessTimeout:
            return mcp_error(
                f"Smoke eval exceeded {smoke_timeout_s}s — one or more "
                f"target models or the judge are hanging. Configured targets: "
                f"{', '.join(target_models)}. Judge: {judge_model_id}."
            )

        if smoke_rc != 0:
            stderr_text = s_stderr.decode("utf-8", errors="replace").strip()
            stdout_text = s_stdout.decode("utf-8", errors="replace").strip()
            excerpt = stderr_text or stdout_text
            # Truncate to keep the tool response readable.
            if len(excerpt) > 2000:
                excerpt = excerpt[-2000:]
            return mcp_error(
                f"Smoke eval subprocess failed (exit {smoke_rc}):\n{excerpt}"
            )

        output_dir = project_dir / "output"
        newest = None
        if output_dir.exists():
            # Only consider files touched at or after the smoke eval started
            # so stale outputs from a prior run can't masquerade as this
            # one's result. Subtract 1 s to tolerate filesystem mtime
            # coarseness (HFS+ / some network mounts round to 1 s).
            floor = smoke_started_at - 1
            fresh = [
                p for p in output_dir.glob("eval_results_*.json")
                if p.stat().st_mtime >= floor
            ]
            if fresh:
                newest = max(fresh, key=lambda p: p.stat().st_mtime)
        if newest is None:
            return mcp_error(
                "Smoke eval completed but wrote no output file in output/. "
                "Check that runner.save_results() is being called."
            )

        # The smoke artifact must be cleaned up regardless of how
        # validation completes — the file is a 1-case-per-model run
        # that has no place in the dashboard. Wrapping the whole
        # parse/check/return block in try/finally ensures failed
        # validations don't leave permanent files.
        try:
            try:
                runs = json.loads(newest.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError) as e:
                return mcp_error(f"Smoke eval output file {newest.name} is not valid JSON: {e}")

            # Coverage check: every target model must have its own run dict
            # in the output. A model that 502'd or got dropped from the
            # harness silently must NOT pass validation.
            run_model_ids = [
                r.get("model_id") for r in runs if isinstance(r, dict)
            ]
            missing_in_output = [
                m for m in target_models if m not in run_model_ids
            ]
            if missing_in_output:
                return mcp_error(
                    "Smoke eval did not produce a run for every target model — "
                    "the harness skipped or crashed on these mid-run:\n"
                    + "\n".join(f"  - {m}" for m in missing_in_output)
                    + "\n\nCheck the harness stdout for per-model errors."
                )

            problems = _check_smoke_output(runs)
            if problems:
                problem_list = "\n".join(f"  - {p}" for p in problems)
                return mcp_error(
                    "Smoke eval produced invalid results — at least one target "
                    "model or metric failed end-to-end:\n" + problem_list + "\n\n"
                    "Common causes: a reasoning model burning its entire "
                    "max_output_tokens budget on hidden reasoning and emitting "
                    "empty text (raise EvalConfig.max_output_tokens); empty "
                    "evaluation_params; judge model rejecting inference_config "
                    "(e.g. temperature on reasoning models); missing "
                    "build_metrics wiring."
                )
        finally:
            # Drop the artifact whether we succeeded or returned early
            # from any of the failure branches above. ``list_runs``
            # already filters validation-tagged files, but leaving them
            # on disk eats EFS quota over time.
            try:
                newest.unlink()
                logger.info("[validate_harness] Removed smoke output %s", newest.name)
            except OSError as cleanup_err:
                logger.warning(
                    "[validate_harness] Could not delete smoke output %s: %s",
                    newest.name, cleanup_err,
                )

        # Per-model summary line: model id + first metric score, so the
        # user sees one OK line per target after a passing sweep.
        smoke_lines: list[str] = []
        for run in runs:
            if not isinstance(run, dict):
                continue
            mid = run.get("model_id", "?")
            try:
                first_case = run["results"][0]
                first_metric = first_case["metrics_data"][0]
                smoke_lines.append(
                    f"  ✓ {mid} scored "
                    f"{first_metric.get('name', '?')}={first_metric.get('score')}"
                )
            except (KeyError, IndexError, TypeError):
                smoke_lines.append(f"  ✓ {mid} ran 1 case successfully")

        # Step 6: metric self-test. The smoke eval proves the harness RUNS;
        # this asks whether its metrics distinguish anything. A metric
        # hardcoded to fail every case passes every check above it — real
        # floats, real bools, a non-empty reason — so nothing until now
        # asserted that any metric ever passes.
        #
        # Runs after the smoke sweep rather than before it on purpose: the
        # probe builds the judge adapter and calls it, so an unreachable judge
        # would surface here as a metric problem instead of in the smoke
        # eval's much clearer per-model message. The cost of that ordering is
        # one wasted sweep when a metric is degenerate.
        # WHAT BLOCKS HERE, and why the list is short. Only a result that
        # positively demonstrates a metric defect fails validation: metrics that
        # cannot be built, and a locally-scoring metric that cannot tell a
        # correct answer from a wrong one. Everything else — the probe timing
        # out, the probe crashing, a judge-backed metric looking degenerate on
        # one sample — is a fact about latency or nondeterminism rather than
        # about the metric, so it warns loudly instead of stopping a user
        # mid-setup on a harness that works.
        selftest_warnings: list[str] = []
        st_payload: dict = {}
        selftest_timed_out = False
        # Every other validation step logs a progress line. This one is by
        # far the most expensive — judge calls against a 300s
        # budget — so without a trace nobody can tell whether it ran, how
        # long it took, or what it concluded, and its warnings never reach
        # CloudWatch.
        logger.info(
            "[validate_harness] Running metric self-test (judge-backed metrics "
            "cost several model calls each; %ds budget)",
            _SELFTEST_TIMEOUT_S,
        )
        selftest_started = time.perf_counter()
        try:
            _st_rc, st_stdout, st_stderr = await run_bounded_subprocess(
                [uv_path, "run", "python", "-u", "-c", _probe_source()],
                timeout_s=_SELFTEST_TIMEOUT_S,
                cwd=str(project_dir),
                env=_project_subprocess_env(),
                label="metric self-test",
            )
        except SubprocessTimeout:
            st_stdout = st_stderr = b""
            selftest_timed_out = True
            selftest_warnings.append(
                f"the metric self-test did not finish within "
                f"{_SELFTEST_TIMEOUT_S}s, so the metrics were NOT checked. Its "
                "cost is judge latency — a judge-backed metric is several model "
                "calls per case — so this usually means a slow or throttled "
                "judge rather than a broken metric. The harness itself passed "
                "every other check. Re-run validate_harness to try again."
            )

        st_text = st_stdout.decode("utf-8", errors="replace").strip()
        if not selftest_timed_out:
            # A probe that cannot report is a gap, not a pass — but it is a gap in
            # the PROBE, which says nothing about the metrics, so it is surfaced
            # rather than used to fail a working harness. Empty stdout counts:
            # the probe's contract is to print one JSON object even on failure, so
            # silence means it died before reaching that line.
            try:
                st_payload = json.loads(st_text.splitlines()[-1])
            except (json.JSONDecodeError, ValueError, IndexError):
                st_err = st_stderr.decode("utf-8", errors="replace").strip()
                selftest_warnings.append(
                    "the metric self-test produced no readable result, so the "
                    "metrics were NOT checked: "
                    f"{(st_err or st_text or 'no output at all')[-800:]}"
                )

        if st_payload.get("status") == "error":
            problems = "\n".join(
                f"  ✗ {e}" for e in (st_payload.get("errors") or ["unknown error"])
            )
            return mcp_error(
                "Metric self-test could not build this project's metrics:\n"
                f"{problems}\n\n"
                "``build_metrics(judge_model)`` in src/metrics/ must return at "
                "least one metric and must construct without raising."
            )

        blocking, probe_warnings = _read_selftest_findings(st_payload)
        selftest_warnings.extend(probe_warnings)
        # The verdict, in one greppable line. Counts rather than prose so a
        # CloudWatch reader can see a regression (blocking>0, or probed
        # dropping to 0) without reading the tool's rendered output.
        logger.info(
            "[validate_harness] Metric self-test finished in %.1fs — "
            "%d metric(s) probed, %d blocking, %d warning(s)%s",
            time.perf_counter() - selftest_started,
            len(st_payload.get("metrics") or []),
            len(blocking),
            len(selftest_warnings),
            " (TIMED OUT — metrics NOT checked)" if selftest_timed_out else "",
        )
        for warning in selftest_warnings:
            logger.warning("[validate_harness] metric self-test: %s", warning)
        if blocking:
            problem_list = "\n".join(f"  ✗ {b}" for b in blocking)
            return mcp_error(
                "Metric self-test FAILED — at least one metric cannot "
                "distinguish a correct answer from a wrong one:\n"
                f"{problem_list}\n\n"
                "This is decided by scoring two synthetic cases: output "
                "identical to the expected output, and output that is not an "
                "answer at all. A metric that answers the same way for both "
                "makes every pass rate in this project meaningless — "
                "always-fail zeroes every run, always-pass certifies every "
                "run. Fix the metric in src/metrics/ and re-run "
                "validate_harness."
            )

        # Fetch pricing and write src/pricing.json (non-fatal)
        all_model_ids = v_result.get("all_model_ids", [])
        pricing_path = None
        if all_model_ids:
            pricing_path = await write_pricing_file(project_dir, all_model_ids)

        if pricing_path:
            pricing_line = "  ✓ Pricing fetched for configured models\n"
        else:
            pricing_line = (
                "  ⚠ Pricing not fetched — costs will show as unknown until "
                "src/pricing.json is written (run refresh_pricing)\n"
            )

        smoke_block = "\n".join(smoke_lines)

        # Warnings are NOT decoration: the formatting findings are the ones that
        # need a human decision (strict on purpose, or a normalizer that misses
        # an artifact), so they ride in the success message where the agent
        # relays them rather than in a log line nobody reads.
        if selftest_warnings:
            selftest_block = "  ⚠ Metric self-test — read before trusting scores:\n" + "\n".join(
                f"      - {w}" for w in selftest_warnings
            ) + "\n"
        else:
            selftest_block = (
                "  ✓ Metric self-test passed (every metric separates a correct "
                "answer from a wrong one)\n"
            )

        return mcp_success(
            "Harness validation passed:\n"
            "  ✓ All required files exist\n"
            "  ✓ Dependencies installed (uv sync)\n"
            "  ✓ All imports successful\n"
            f"  ✓ Data loaded: {v_result['test_cases']} test case(s)\n"
            f"  ✓ Judge model: {judge_model_id}\n"
            f"  ✓ Full-sweep smoke eval passed (1 case × {len(target_models)} target models):\n"
            f"{smoke_block}\n"
            f"{selftest_block}"
            f"{pricing_line}\n"
            "The harness is ready to run evaluations."
        )

    except Exception as e:
        logger.error("[validate_harness] FAILED: %s", e, exc_info=True)
        return mcp_error(f"Validation error: {e}")
