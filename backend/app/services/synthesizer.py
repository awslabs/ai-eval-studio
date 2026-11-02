# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic data generation service wrapping DeepEval's Synthesizer.

This service bridges DeepEval's ``Synthesizer`` with the eval studio's
project layout. Supported strategies:

    * paraphrase        — evolve an existing set of seed Goldens
    * from_docs         — generate from user-supplied documents
    * from_scratch      — generate purely from a scenario/task description
    * failure_targeted  — synthesize cases similar to ones a target model
      failed in a prior run (mines model_analyses from the run's results)

Generated batches are persisted under ``projects/<id>/data/synthetic/`` as
a pair of files: a timestamped CSV (column names driven by the project's
``field_mappings``) and a sibling ``manifest.json`` tracking per-case
approval state.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deepeval.dataset import Golden
from deepeval.synthesizer import Synthesizer as DeepEvalSynthesizer
from deepeval.synthesizer.config import StylingConfig

from app.services.bedrock_runtime_model import BedrockRuntimeModel
from app.services.project_paths import safe_path

logger = logging.getLogger(__name__)


class Synthesizer:
    """Thin wrapper around ``deepeval.synthesizer.Synthesizer``.

    Holds a configured generation model (``BedrockRuntimeModel``) and exposes
    the three generation strategies plus a ``write_batch`` helper
    that serialises the resulting Goldens to disk using the project's
    configured field mappings.
    """

    def __init__(
        self,
        model: BedrockRuntimeModel,
        async_mode: bool = False,
        max_concurrent: int = 10,
    ) -> None:
        """Initialise the synthesizer.

        Args:
            model: Bedrock generation model (DeepEvalBaseLLM-compatible).
                Passed verbatim to DeepEval's ``Synthesizer``.
            async_mode: Whether DeepEval should dispatch generation
                asynchronously. Defaults to ``False`` so we interoperate
                cleanly with existing synchronous call sites.
            max_concurrent: Upper bound on concurrent generations when
                ``async_mode`` is enabled.
        """
        self.model = model
        self._synth = DeepEvalSynthesizer(
            model=model,
            async_mode=async_mode,
            max_concurrent=max_concurrent,
        )

    # ------------------------------------------------------------------
    # Generation strategies
    # ------------------------------------------------------------------

    def generate_from_goldens(
        self,
        seed_cases: list[Golden],
        max_goldens_per_golden: int = 2,
        include_expected_output: bool = True,
    ) -> list[Golden]:
        """Paraphrase/evolve existing seed cases.

        Args:
            seed_cases: Seed ``Golden`` objects to evolve from.
            max_goldens_per_golden: Number of new goldens generated per seed.
            include_expected_output: Whether DeepEval should synthesise
                expected outputs alongside inputs.

        Returns:
            List of newly generated ``Golden`` objects.
        """
        return self._synth.generate_goldens_from_goldens(
            goldens=seed_cases,
            max_goldens_per_golden=max_goldens_per_golden,
            include_expected_output=include_expected_output,
        )

    def generate_from_docs(
        self,
        document_paths: list[str | Path],
        max_goldens_per_context: int = 2,
        include_expected_output: bool = True,
    ) -> list[Golden]:
        """Generate Goldens grounded in user-supplied documents.

        Args:
            document_paths: Absolute paths to source documents. DeepEval
                handles chunking/context construction internally.
            max_goldens_per_context: Goldens generated per extracted
                context chunk.
            include_expected_output: Whether to synthesise expected outputs.

        Returns:
            List of ``Golden`` objects derived from the documents.
        """
        # DeepEval accepts plain strings; normalise Path objects here.
        paths = [str(p) for p in document_paths]
        return self._synth.generate_goldens_from_docs(
            document_paths=paths,
            max_goldens_per_context=max_goldens_per_context,
            include_expected_output=include_expected_output,
        )

    def generate_from_scratch(
        self,
        subject: str,
        task: str,
        output_format: str,
        num_cases: int = 10,
        input_format: str | None = None,
    ) -> list[Golden]:
        """Generate Goldens purely from a scenario/task description.

        Under the hood, DeepEval's ``generate_goldens_from_scratch`` only
        accepts ``num_goldens`` — the scenario/task/input/output formats
        that actually shape generation live on the ``StylingConfig``
        attribute of the synthesizer. We set that directly here so the
        next generation call picks it up.

        Args:
            subject: The scenario description (what/who the cases are about).
            task: What the target model should do with the inputs.
            output_format: Expected output shape/style.
            num_cases: Number of goldens to generate.
            input_format: Optional description of the input shape/style.

        Returns:
            List of newly generated ``Golden`` objects.
        """
        self._synth.styling_config = StylingConfig(
            scenario=subject,
            task=task,
            input_format=input_format,
            expected_output_format=output_format,
        )
        # DeepEval guards styling-gated code paths behind this flag; make
        # sure it's on so the new styling is actually applied.
        self._synth.set_styling_config = True
        return self._synth.generate_goldens_from_scratch(num_goldens=num_cases)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def write_batch(
        self,
        project_dir: Path,
        strategy: str,
        goldens: list[Golden],
        field_mappings: dict[str, str],
        seed_source: str | None = None,
        seed_run_id: str | None = None,
    ) -> dict[str, Any]:
        """Write a batch of generated goldens to ``data/synthetic/``.

        Produces two files:

            * ``{timestamp}_{strategy}.csv`` — rows whose columns are the
              *values* of ``field_mappings`` (so the eval harness can
              consume the file without re-scaffolding).
            * ``{timestamp}_{strategy}.manifest.json`` — batch metadata +
              per-case approval state (all ``pending`` on first write).

        ``field_mappings`` keys recognised: ``input``, ``expected_output``,
        ``category``, ``id``. Any other keys are written as empty string
        columns so downstream tooling can fill them in later if desired.

        Args:
            project_dir: Root directory of the project. ``data/synthetic/``
                is created under this if missing.
            strategy: One of ``paraphrase | from_docs | from_scratch |
                failure_targeted``. Used in filenames + manifest.
            goldens: Generated Goldens to persist.
            field_mappings: The project's harness field mappings. Values
                become CSV column names.
            seed_source: Optional free-text label describing where the
                seed data came from (e.g. ``data/raw/cases.csv``).
            seed_run_id: Optional eval run id this batch was seeded from
                (used by failure-targeted mode; empty otherwise).

        Returns:
            Dict with ``batch_id``, ``timestamp``, ``strategy``,
            ``csv_path``, ``manifest_path``, ``total_cases``, and
            ``unscoreable_cases`` — the number of rows whose mapped
            ``expected_output`` column came out EMPTY. Those carry
            ``unscoreable: "missing_expected_output"`` in the manifest so the
            approval gate can refuse them; a case with no ground truth fails
            every metric that reads it regardless of the model.
        """
        synthetic_dir = project_dir / "data" / "synthetic"
        synthetic_dir.mkdir(parents=True, exist_ok=True)

        # Filesystem-safe timestamp (colons break Windows; also nicer in URLs).
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        batch_id = f"{timestamp}_{strategy}"

        csv_name = f"{batch_id}.csv"
        manifest_name = f"{batch_id}.manifest.json"

        csv_path = safe_path(synthetic_dir, csv_name)
        manifest_path = safe_path(synthetic_dir, manifest_name)
        if csv_path is None or manifest_path is None:
            # Should never happen — batch_id is entirely server-generated.
            raise ValueError(f"Unsafe batch id: {batch_id}")

        # Preserve mapping key order for column layout. Deduplicate values
        # while keeping first occurrence — two keys mapping to the same
        # column name would otherwise produce duplicate headers.
        columns: list[str] = []
        seen: set[str] = set()
        for col in field_mappings.values():
            if col and col not in seen:
                columns.append(col)
                seen.add(col)

        input_col = field_mappings.get("input")
        expected_col = field_mappings.get("expected_output")
        category_col = field_mappings.get("category")
        id_col = field_mappings.get("id")

        cases_meta: list[dict[str, Any]] = []
        unscoreable = 0

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for idx, g in enumerate(goldens):
                row = {col: "" for col in columns}
                if input_col:
                    row[input_col] = g.input or ""
                if expected_col:
                    row[expected_col] = g.expected_output or ""

                # Carry through Golden metadata if the mapping asks for it.
                meta = g.additional_metadata or {}
                if category_col and meta.get("category") is not None:
                    row[category_col] = str(meta["category"])
                if id_col:
                    # Prefer a Golden-supplied id, fall back to a stable
                    # synthetic_<timestamp>_<index> token so downstream
                    # tooling always has something to key on.
                    row[id_col] = str(meta.get("id") or f"synthetic_{batch_id}_{idx}")

                writer.writerow(row)
                # An empty expected output where the mapping declares one is an
                # UNSCOREABLE case: any metric that reads expected_output fails
                # it no matter what the model produces, so approving it books
                # guaranteed failures into the dataset and drags every pass rate
                # down for a reason that has nothing to do with the model.
                # Recorded as a fact here; the approval gate is what refuses it.
                meta_entry: dict[str, Any] = {"index": idx, "status": "pending"}
                if expected_col and not (g.expected_output or "").strip():
                    meta_entry["unscoreable"] = "missing_expected_output"
                    unscoreable += 1
                cases_meta.append(meta_entry)

        manifest: dict[str, Any] = {
            "batch_id": batch_id,
            "timestamp": timestamp,
            "strategy": strategy,
            "seed_source": seed_source,
            "seed_run_id": seed_run_id,
            "total_cases": len(goldens),
            "field_mappings": dict(field_mappings),
            "csv_file": csv_name,
            "unscoreable_cases": unscoreable,
            "cases": cases_meta,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

        logger.info(
            "[synthesizer] Wrote batch=%s strategy=%s cases=%d unscoreable=%d",
            batch_id, strategy, len(goldens), unscoreable,
        )

        return {
            "batch_id": batch_id,
            "timestamp": timestamp,
            "strategy": strategy,
            "csv_path": csv_path,
            "manifest_path": manifest_path,
            "total_cases": len(goldens),
            "unscoreable_cases": unscoreable,
        }
