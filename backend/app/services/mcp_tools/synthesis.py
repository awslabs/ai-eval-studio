# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP tools for synthetic data generation and review.

Exposes three tools to the coordinator agent:

    * ``generate_synthetic`` — run one of four DeepEval strategies
      (paraphrase / from_docs / from_scratch / failure_targeted). The
      failure-targeted mode mines the most recent eval run's failures and
      uses them to seed new stress-test cases.
    * ``list_synthetic_batches`` — enumerate batch manifests with
      pending/approved/rejected counts (read-only).
    * ``approve_synthetic_cases`` — flip per-case status in a batch
      manifest.

All file I/O goes through ``resolve_project_dir`` / ``safe_path`` so no
raw path built from tool arguments can escape the project directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from claude_agent_sdk import ToolAnnotations, tool
from deepeval.dataset import Golden

from app.services.bedrock_runtime_model import BedrockRuntimeModel
from app.services.dataset_service import DatasetService
from app.services.eval_run_files import eval_run_files
from app.services.failure_analyzer import analyze_failures
from app.services.failure_targeted_prompt import (
    build_failure_seed_prompt,
    failures_to_seed_goldens,
)
from app.services.harness_config import load_harness_config
from app.services.mcp_response import mcp_error, mcp_json
from app.services.project_paths import resolve_project_dir, safe_path
from app.services.synthesizer import Synthesizer

logger = logging.getLogger(__name__)


SUPPORTED_STRATEGIES = {"paraphrase", "from_docs", "from_scratch", "failure_targeted"}

# Minimum number of failing cases we insist on before running the
# failure-targeted strategy. Below this we'd just be paraphrasing one or
# two outliers, which isn't useful stress-testing.
MIN_FAILURES_FOR_SEEDING = 2


def _load_field_mappings_or_error(project_dir) -> tuple[dict[str, str] | None, dict | None]:
    """Load field_mappings from harness config, or return an MCP error dict.

    Returns a tuple: ``(mappings, error_response)``. Exactly one element
    is non-None. ``input`` and ``expected_output`` must both be present;
    otherwise we return an error so the agent fixes the harness first.
    """
    harness = load_harness_config(project_dir)
    if harness is None:
        return None, mcp_error(
            "No harness config found. Scaffold the harness and configure "
            "field_mappings before generating synthetic data."
        )
    mappings = harness.field_mappings or {}
    missing = [k for k in ("input", "expected_output") if not mappings.get(k)]
    if missing:
        return None, mcp_error(
            "field_mappings must define both 'input' and 'expected_output' "
            f"(missing: {', '.join(missing)}). Update src/config.py first."
        )
    return mappings, None


def _seed_goldens_from_harness_data(
    project_dir, mappings: dict[str, str], max_cases: int | None
) -> tuple[list[Golden] | None, dict | None]:
    """Load the project's harness data_file as a list of ``Golden`` objects.

    Used by the paraphrase strategy. Returns ``(goldens, error_response)``
    with exactly one element non-None.
    """
    harness = load_harness_config(project_dir)
    if harness is None or not harness.data_file:
        return None, mcp_error("No data_file configured in harness.")

    data_path = safe_path(project_dir / "data" / "raw", harness.data_file)
    if data_path is None or not data_path.exists():
        return None, mcp_error(f"Seed data file not found: {harness.data_file}")

    svc = DatasetService()
    cases = svc.load_test_cases(data_path, mappings, max_cases=max_cases)
    # Convert LLMTestCase -> Golden, and give each one a CONTEXT built from the
    # seed pair. The context is not decoration — it decides which DeepEval code
    # path runs, and without it the paraphrase strategy silently stops
    # paraphrasing.
    #
    # Read from DeepEval 3.9.9's own source:
    # ``generate_goldens_from_goldens`` collects ``golden.context`` from every
    # seed, and when that list is EMPTY it calls
    # ``generate_goldens_from_scratch(num_goldens=len(goldens) * max_per)`` —
    # the seeds are then used only to infer prompt STYLING, and
    # ``generate_goldens_from_scratch`` takes no ``include_expected_output``
    # parameter at all. So a seed set with no context yields invented cases with
    # NO ground truth — a batch whose ``expected_output`` values are empty
    # despite passing ``include_expected_output=True``. With contexts present it takes
    # ``generate_goldens_from_contexts``, which honours that flag.
    #
    # Grounding on the seed's own input + expected output is the closest thing
    # to "paraphrase this case" the API offers.
    goldens = []
    for c in cases:
        context = [part for part in (c.input, c.expected_output) if part]
        goldens.append(
            Golden(
                input=c.input,
                expected_output=c.expected_output,
                context=context or None,
                additional_metadata=c.metadata or {},
            )
        )
    return goldens, None


def _build_bedrock_runtime_model(args: dict[str, Any]) -> BedrockRuntimeModel:
    """Construct a BedrockRuntimeModel for generation.

    ``generation_model_id`` is required: the caller (the agent) knows the
    project's settled models and must choose one explicitly. This used to
    fall back to the adapter's hardcoded default model, silently generating
    synthetic data with a model nobody picked.

    Raises:
        ValueError: When ``generation_model_id`` is missing.
    """
    model_id = args.get("generation_model_id")
    if not model_id:
        raise ValueError(
            "generation_model_id is required — pick one of the project's "
            "configured models (src/config.py) for synthetic generation."
        )
    kwargs: dict[str, Any] = {"model_id": model_id}
    if args.get("aws_region"):
        kwargs["region"] = args["aws_region"]
    if args.get("aws_profile"):
        kwargs["profile_name"] = args["aws_profile"]
    return BedrockRuntimeModel(**kwargs)


def _resolve_seed_run_path(
    project_dir: Path, seed_run_filename: str | None,
) -> tuple[Path | None, dict | None]:
    """Locate the eval-results JSON file for failure-targeted seeding.

    If the caller supplied ``seed_run_filename`` we validate it sits
    inside ``output/`` (via ``safe_path``). Otherwise we fall back to the
    most recently modified eval run under ``output/`` (``eval_run_files``,
    which excludes sidecars) so the agent can say "use the last run".

    Returns a tuple ``(path, error_response)`` where exactly one element
    is non-None. The error_response is already MCP-shaped.
    """
    output_dir = project_dir / "output"
    if seed_run_filename:
        candidate = safe_path(output_dir, seed_run_filename)
        if candidate is None:
            return None, mcp_error(f"Invalid seed_run_filename: {seed_run_filename}")
        if not candidate.exists():
            return None, mcp_error(f"Seed run file not found: {seed_run_filename}")
        return candidate, None

    if not output_dir.exists():
        return None, mcp_error(
            "No eval runs found — cannot seed from failures. Run an eval first."
        )
    # Most-recent eval RUN under output/. Runs only: the metric-audit sidecar is
    # written just after the run it audits, so it always carries the newer mtime
    # and a plain ``*.json`` scan picked it every time — then failure analysis
    # found no model runs in it and reported "no analyzable seed run" for a
    # project that had several.
    candidates = sorted(
        eval_run_files(output_dir),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None, mcp_error(
            "No eval run results found in output/ — cannot seed from failures."
        )
    return candidates[0], None


@tool(
    "generate_synthetic",
    "Generate synthetic test cases using DeepEval's Synthesizer. Strategies: "
    "'paraphrase' evolves the project's existing seed data; 'from_docs' generates "
    "cases grounded in uploaded documents (data/raw/); 'from_scratch' generates "
    "purely from a scenario/task description; 'failure_targeted' mines the most "
    "recent eval run's failures and generates stress-test cases that resemble them "
    "(use this after a run when the user wants to harden the model against its "
    "weakest category/metric). Writes the batch to data/synthetic/ with a manifest; "
    "all cases start as 'pending' and must be approved before they feed into eval runs.",
    {
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "strategy": {
                "type": "string",
                "enum": ["paraphrase", "from_docs", "from_scratch", "failure_targeted"],
            },
            "num_cases": {
                "type": "integer",
                "description": "Target number of goldens to generate (from_scratch and "
                               "failure_targeted). Ignored by paraphrase/from_docs, which use "
                               "max_goldens_per_seed.",
            },
            "max_goldens_per_seed": {
                "type": "integer",
                "description": "How many goldens to generate per seed/context (paraphrase, from_docs). Default 2.",
            },
            "subject": {"type": "string", "description": "from_scratch: scenario/domain description"},
            "task": {"type": "string", "description": "from_scratch: what the target model should do"},
            "output_format": {"type": "string", "description": "from_scratch: expected output shape"},
            "input_format": {"type": "string", "description": "from_scratch: optional input shape"},
            "document_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "from_docs: filenames under data/raw/ to ground generation in",
            },
            "max_seed_cases": {
                "type": "integer",
                "description": "paraphrase: cap on seed cases loaded from the harness data_file",
            },
            "seed_run_filename": {
                "type": "string",
                "description": "failure_targeted: filename (under output/) of the eval results "
                               "JSON to mine for failures. Defaults to the most-recent *.json "
                               "in output/ when omitted.",
            },
            "run_index": {
                "type": "integer",
                "description": "failure_targeted: 0-based index into the run's model_analyses "
                               "array when multiple target models were evaluated. Defaults to 0.",
            },
            "generation_model_id": {
                "type": "string",
                "description": (
                    "REQUIRED Bedrock model id for synthetic generation — "
                    "pick one of the project's configured models "
                    "(src/config.py). There is no default."
                ),
            },
            "aws_region": {"type": "string"},
            "aws_profile": {"type": "string"},
        },
        "required": ["project_id", "strategy"],
    },
)
async def generate_synthetic(args: dict[str, Any]) -> dict[str, Any]:
    """Generate a synthetic data batch and persist it for user review.

    Dispatches to the appropriate DeepEval Synthesizer strategy, writes
    the resulting Goldens to ``data/synthetic/`` with a manifest, and
    returns a summary the coordinator can relay to the user.

    Args:
        args: Tool args (see schema). ``project_id`` + ``strategy`` are
            required; remaining fields depend on the chosen strategy.

    Returns:
        MCP response with batch_id, file paths, and total_cases — or
        ``mcp_error`` on any validation / generation failure.
    """
    logger.info(
        "[generate_synthetic] project=%s strategy=%s",
        args.get("project_id", "?")[:12],
        args.get("strategy"),
    )
    try:
        strategy = args.get("strategy", "")
        if strategy not in SUPPORTED_STRATEGIES:
            return mcp_error(
                f"Unsupported strategy: {strategy!r}. "
                f"Expected one of: {', '.join(sorted(SUPPORTED_STRATEGIES))}."
            )

        project_dir = resolve_project_dir(args["project_id"])
        mappings, err = _load_field_mappings_or_error(project_dir)
        if err is not None:
            return err
        assert mappings is not None  # for type checker

        try:
            model = _build_bedrock_runtime_model(args)
        except ValueError as e:
            return mcp_error(str(e))
        synth = Synthesizer(model=model)
        seed_source: str | None = None
        seed_run_id: str | None = None

        if strategy == "paraphrase":
            seed_cases, err = _seed_goldens_from_harness_data(
                project_dir, mappings, args.get("max_seed_cases"),
            )
            if err is not None:
                return err
            assert seed_cases is not None
            if not seed_cases:
                return mcp_error("Seed data file is empty — cannot paraphrase.")
            max_per = args.get("max_goldens_per_seed", 2)
            goldens = synth.generate_from_goldens(
                seed_cases=seed_cases,
                max_goldens_per_golden=max_per,
            )
            # Record the harness data_file as the seed source for traceability.
            harness = load_harness_config(project_dir)
            seed_source = f"data/raw/{harness.data_file}" if harness and harness.data_file else None

        elif strategy == "from_docs":
            doc_names = args.get("document_paths") or []
            if not doc_names:
                return mcp_error("from_docs requires 'document_paths' (non-empty).")
            resolved: list[str] = []
            raw_dir = project_dir / "data" / "raw"
            for name in doc_names:
                p = safe_path(raw_dir, name)
                if p is None:
                    return mcp_error(f"Invalid document path: {name}")
                if not p.exists():
                    return mcp_error(f"Document not found: {name}")
                resolved.append(str(p))
            max_per = args.get("max_goldens_per_seed", 2)
            goldens = synth.generate_from_docs(
                document_paths=resolved,
                max_goldens_per_context=max_per,
            )
            seed_source = "data/raw/" + ",".join(doc_names)

        elif strategy == "from_scratch":
            missing = [
                k for k in ("subject", "task", "output_format")
                if not args.get(k)
            ]
            if missing:
                return mcp_error(
                    "from_scratch requires subject, task, and output_format "
                    f"(missing: {', '.join(missing)})."
                )
            num = args.get("num_cases", 10)
            goldens = synth.generate_from_scratch(
                subject=args["subject"],
                task=args["task"],
                output_format=args["output_format"],
                num_cases=num,
                input_format=args.get("input_format"),
            )

        else:  # failure_targeted
            seed_path, err = _resolve_seed_run_path(
                project_dir, args.get("seed_run_filename")
            )
            if err is not None:
                return err
            assert seed_path is not None

            try:
                analysis = analyze_failures(seed_path)
            except (OSError, ValueError, json.JSONDecodeError) as e:
                return mcp_error(f"Failed to analyze seed run: {e}")
            if "error" in analysis:
                return mcp_error(
                    f"Seed run is not analyzable: {analysis['error']}"
                )

            # Pick the requested model analysis (or default to first).
            run_index = args.get("run_index")
            model_analyses = analysis.get("model_analyses") or []
            if not model_analyses:
                return mcp_error("Seed run contains no model analyses.")
            if run_index is not None and (
                run_index < 0 or run_index >= len(model_analyses)
            ):
                return mcp_error(
                    f"run_index {run_index} out of range "
                    f"(available: 0..{len(model_analyses) - 1})."
                )
            picked_index = run_index if run_index is not None else 0
            picked = model_analyses[picked_index]
            # Trim analysis down to the single picked run so the prompt
            # builder doesn't have to re-guess which one we care about.
            scoped_analysis = {"model_analyses": [picked]}

            try:
                seed_goldens = failures_to_seed_goldens(scoped_analysis, max_seeds=10)
            except ValueError as e:
                return mcp_error(str(e))
            if len(seed_goldens) < MIN_FAILURES_FOR_SEEDING:
                return mcp_error(
                    f"Not enough failures to seed (need at least "
                    f"{MIN_FAILURES_FOR_SEEDING})."
                )

            try:
                prompt_bundle = build_failure_seed_prompt(scoped_analysis, max_examples=6)
            except ValueError as e:
                return mcp_error(str(e))

            # DeepEval's StylingConfig treats ``task`` as free-text prose,
            # so we concatenate the failing-case examples onto it. The
            # subject/output_format still carry the scenario shape.
            composed_task = f"{prompt_bundle['task']}\n\n{prompt_bundle['instructions']}"
            num = args.get("num_cases", 20)
            goldens = synth.generate_from_scratch(
                subject=prompt_bundle["subject"],
                task=composed_task,
                output_format=prompt_bundle["output_format"],
                num_cases=num,
            )

            # Traceability: record where we pulled the seeds from so the
            # UI can link a synthetic batch back to the failing eval run.
            seed_source = f"output/{seed_path.name}"
            seed_run_id = picked.get("test_run_id") or seed_path.stem

        batch = synth.write_batch(
            project_dir=project_dir,
            strategy=strategy,
            goldens=goldens,
            field_mappings=mappings,
            seed_source=seed_source,
            seed_run_id=seed_run_id,
        )

        response = {
            "batch_id": batch["batch_id"],
            "strategy": strategy,
            "total_cases": batch["total_cases"],
            "csv_file": batch["csv_path"].name,
            "manifest_file": batch["manifest_path"].name,
            "status": "pending_review",
        }
        # Surfaced at generation time, not left for the approval call to
        # discover: the user is deciding whether this batch is worth reviewing
        # at all, and "5 of 20 have no ground truth" changes that answer.
        if batch.get("unscoreable_cases"):
            response["unscoreable_cases"] = batch["unscoreable_cases"]
            response["warning"] = (
                f"{batch['unscoreable_cases']} of {batch['total_cases']} case(s) "
                "have an EMPTY expected output. They cannot be approved as-is — "
                "no metric that reads expected_output can pass them. Either fix "
                "them in the batch CSV or reject them."
            )
        return mcp_json(response)

    except Exception as e:
        logger.error("[generate_synthetic] FAILED: %s", e, exc_info=True)
        return mcp_error(f"Synthesis error: {e}")


@tool(
    "list_synthetic_batches",
    "List all synthetic data batches for a project. Returns batch id, strategy, "
    "timestamp, total cases, and counts of pending/approved/rejected cases per batch.",
    {"project_id": str},
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_synthetic_batches(args: dict[str, Any]) -> dict[str, Any]:
    """Enumerate synthetic batches for a project.

    Reads every ``*.manifest.json`` under ``data/synthetic/`` and
    returns a per-batch summary. Manifests that fail to parse are
    skipped with a warning so one bad file doesn't block the tool.

    Args:
        args: Dict with ``project_id``.

    Returns:
        MCP json response with a ``batches`` array (possibly empty).
    """
    try:
        project_dir = resolve_project_dir(args["project_id"])
        synthetic_dir = project_dir / "data" / "synthetic"
        if not synthetic_dir.exists():
            return mcp_json({"batches": []})

        batches: list[dict[str, Any]] = []
        for manifest_path in sorted(synthetic_dir.glob("*.manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("[list_synthetic_batches] skipping %s: %s", manifest_path.name, e)
                continue

            cases = manifest.get("cases") or []
            counts = {"pending": 0, "approved": 0, "rejected": 0}
            for c in cases:
                status = c.get("status", "pending")
                if status in counts:
                    counts[status] += 1

            batches.append({
                "batch_id": manifest.get("batch_id"),
                "strategy": manifest.get("strategy"),
                "timestamp": manifest.get("timestamp"),
                "seed_source": manifest.get("seed_source"),
                "seed_run_id": manifest.get("seed_run_id"),
                "total_cases": manifest.get("total_cases", len(cases)),
                "pending": counts["pending"],
                "approved": counts["approved"],
                "rejected": counts["rejected"],
                "csv_file": manifest.get("csv_file"),
            })

        return mcp_json({"batches": batches})

    except Exception as e:
        return mcp_error(f"Error: {e}")


@tool(
    "approve_synthetic_cases",
    "Approve or reject cases within a synthetic batch. Pass approve_indices or "
    "reject_indices as arrays of row indices, or set approve_all=true to approve "
    "every pending case in the batch. Returns the new per-status counts.",
    {
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "batch_id": {"type": "string"},
            "approve_indices": {"type": "array", "items": {"type": "integer"}},
            "reject_indices": {"type": "array", "items": {"type": "integer"}},
            "approve_all": {"type": "boolean"},
        },
        "required": ["project_id", "batch_id"],
    },
)
async def approve_synthetic_cases(args: dict[str, Any]) -> dict[str, Any]:
    """Update per-case approval status in a synthetic batch manifest.

    Args:
        args: Dict with ``project_id``, ``batch_id``, and at least one
            of ``approve_indices`` / ``reject_indices`` / ``approve_all``.

    Returns:
        MCP json response with the updated ``pending`` / ``approved`` /
        ``rejected`` counts plus the resolved ``batch_id``.
    """
    try:
        project_dir = resolve_project_dir(args["project_id"])
        synthetic_dir = project_dir / "data" / "synthetic"

        batch_id = args.get("batch_id") or ""
        # batch_id is untrusted input — run it through safe_path so we
        # can't escape data/synthetic/.
        manifest_path = safe_path(synthetic_dir, f"{batch_id}.manifest.json")
        if manifest_path is None or not manifest_path.exists():
            return mcp_error(f"Batch not found: {batch_id}")

        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as e:
            return mcp_error(f"Corrupt manifest: {e}")

        cases = manifest.get("cases") or []
        approve_all = bool(args.get("approve_all"))
        approve_set = set(args.get("approve_indices") or [])
        reject_set = set(args.get("reject_indices") or [])

        if not approve_all and not approve_set and not reject_set:
            return mcp_error(
                "Provide approve_indices, reject_indices, or approve_all=true."
            )

        # A case with no ground truth cannot be scored: every metric that reads
        # expected_output fails it whatever the model produces, so approving it
        # books guaranteed failures into the dataset and pulls every pass rate
        # down for a reason unrelated to the model. Human review was the only
        # backstop, and ``approve_all`` bypasses review by design.
        #
        # Two different answers on purpose. ``approve_all`` HOLDS BACK the
        # unscoreable ones and says how many — refusing the whole call would
        # punish a batch that is 18-good/2-bad. An EXPLICIT index is REFUSED,
        # because the caller named that case and silently skipping it would
        # report an approval that did not happen.
        unscoreable_idx = {
            c.get("index") for c in cases if c.get("unscoreable")
        }
        named_unscoreable = sorted(i for i in approve_set if i in unscoreable_idx)
        if named_unscoreable:
            return mcp_error(
                f"Cannot approve case(s) {named_unscoreable}: no expected "
                "output, so no metric that reads it can ever pass them. "
                "Reject them, or fix the expected output in the batch CSV "
                "first."
            )

        held_back = 0
        for c in cases:
            idx = c.get("index")
            is_unscoreable = bool(c.get("unscoreable"))
            if approve_all and c.get("status") == "pending":
                if is_unscoreable:
                    held_back += 1
                else:
                    c["status"] = "approved"
            if idx in approve_set:
                c["status"] = "approved"
            elif idx in reject_set:
                c["status"] = "rejected"

        manifest["cases"] = cases
        manifest_path.write_text(json.dumps(manifest, indent=2))

        counts = {"pending": 0, "approved": 0, "rejected": 0}
        for c in cases:
            status = c.get("status", "pending")
            if status in counts:
                counts[status] += 1

        payload = {
            "batch_id": batch_id,
            "pending": counts["pending"],
            "approved": counts["approved"],
            "rejected": counts["rejected"],
        }
        if held_back:
            payload["held_back_unscoreable"] = held_back
            payload["note"] = (
                f"{held_back} case(s) left pending: they have no expected "
                "output, so no metric that reads it can pass them. Fix the "
                "expected output in the batch CSV or reject them."
            )
        return mcp_json(payload)

    except Exception as e:
        return mcp_error(f"Error: {e}")
