# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset parsing service for loading evaluation data into DeepEval test cases.

Supports CSV, JSON, JSONL, and Excel files. Handles field mapping, category
filtering, and sampling for large datasets.

Also hosts the synthetic-aware data-source resolution helpers used by eval
and optimization run dispatch. ``resolve_data_file`` translates a
``data_source`` selector (``"raw" | "synthetic" | "merged"``) into a path
that can be handed to the harness as ``--data-file``.
"""

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from deepeval.test_case import LLMTestCase

logger = logging.getLogger(__name__)


class DatasetService:
    """Parses uploaded data files into DeepEval LLMTestCase objects."""

    def load_test_cases(
        self,
        file_path: Path,
        mappings: dict[str, str],
        max_cases: int | None = None,
        category_filter: list[str] | None = None,
    ) -> list[LLMTestCase]:
        """Load a data file and convert rows to LLMTestCase objects.

        Args:
            file_path: Path to the data file (CSV, JSON, JSONL, or Excel).
            mappings: Dict mapping LLMTestCase param names to dataset column names.
                Required keys: ``input``, ``expected_output``.
                Optional keys: ``category``, ``id``.
            max_cases: Optional limit on number of test cases returned.
            category_filter: Optional list of category values to include.

        Returns:
            List of LLMTestCase objects ready for DeepEval evaluation.

        Raises:
            ValueError: If a required mapped column is not found in the data.
        """
        df = self._read_file(file_path)
        self._validate_mapping(df, mappings)

        category_col = mappings.get("category")
        id_col = mappings.get("id")

        if category_filter and category_col:
            df = df[df[category_col].isin(category_filter)]

        if max_cases and len(df) > max_cases:
            df = df.head(max_cases)

        cases = []
        for _, row in df.iterrows():
            metadata = {}
            if id_col and id_col in df.columns:
                metadata["id"] = str(row[id_col])
            if category_col and category_col in df.columns:
                metadata["category"] = str(row[category_col])

            case = LLMTestCase(
                input=str(row[mappings["input"]]),
                actual_output="",
                expected_output=str(row[mappings["expected_output"]]),
                metadata=metadata,
            )
            cases.append(case)

        return cases

    def describe_file(self, file_path: Path) -> dict:
        """Return summary info about a data file.

        Args:
            file_path: Path to the data file.

        Returns:
            Dict with row_count, columns, and sample_rows.
        """
        df = self._read_file(file_path)
        return {
            "row_count": len(df),
            "columns": list(df.columns),
            "sample_rows": df.head(5).to_dict(orient="records"),
        }

    # ------------------------------------------------------------------
    # Synthetic-aware data-source resolution
    # ------------------------------------------------------------------

    def load_approved_synthetic_rows(
        self,
        project_dir: Path,
        field_mappings: dict[str, str],
    ) -> list[dict[str, str]]:
        """Collect every ``approved`` synthetic case across all batches.

        Walks ``data/synthetic/*.manifest.json``, finds cases whose
        ``status == "approved"``, and returns the matching CSV rows
        keyed by the ``field_mappings`` values. Rows are trimmed to the
        set of columns actually referenced by ``field_mappings`` so the
        caller can safely concat them with the raw dataset.

        Missing synthetic directories, unreadable manifests, and missing
        CSVs are tolerated — this function never raises; at worst it
        returns ``[]``.

        Args:
            project_dir: Project root (``data/synthetic/`` lives under this).
            field_mappings: Harness field mappings. Values are the column
                names we care about (anything else in the CSV is dropped).

        Returns:
            List of row dicts (column name -> string value), one per
            approved case. Empty if no approvals exist.
        """
        synthetic_dir = project_dir / "data" / "synthetic"
        if not synthetic_dir.exists():
            return []

        # Deduplicate the set of columns we want to surface. ``dict.fromkeys``
        # keeps insertion order (stable column layout in merged CSVs).
        wanted_cols: list[str] = list(dict.fromkeys(
            col for col in (field_mappings or {}).values() if col
        ))
        if not wanted_cols:
            return []

        collected: list[dict[str, str]] = []
        for manifest_path in sorted(synthetic_dir.glob("*.manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "[dataset-service] skipping %s: %s", manifest_path.name, exc,
                )
                continue

            cases = manifest.get("cases") or []
            approved_indices = {
                int(c.get("index", -1))
                for c in cases
                if c.get("status") == "approved"
            }
            if not approved_indices:
                continue

            csv_name = manifest.get("csv_file") or ""
            csv_path = synthetic_dir / csv_name
            if not csv_name or not csv_path.exists():
                logger.warning(
                    "[dataset-service] manifest=%s references missing csv=%s",
                    manifest_path.name, csv_name,
                )
                continue

            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for idx, row in enumerate(reader):
                    if idx not in approved_indices:
                        continue
                    # Preserve only the mapped columns; fill with empty
                    # string if the CSV is missing one (defensive).
                    trimmed = {col: str(row.get(col) or "") for col in wanted_cols}
                    collected.append(trimmed)

        return collected

    def merge_datasets(
        self,
        raw_path: Path,
        approved_synthetic_rows: list[dict[str, str]],
        output_path: Path,
        field_mappings: dict[str, str],
    ) -> Path:
        """Concatenate raw rows + approved synthetic rows into one CSV.

        The intersection of the raw file's columns and the mapping values
        becomes the output column set — columns present only on one side
        are dropped so the merged file always has a well-defined shape.

        Args:
            raw_path: Path to the project's raw dataset.
            approved_synthetic_rows: Rows returned by
                ``load_approved_synthetic_rows`` (each already trimmed to
                the mapped columns).
            output_path: Where to write the merged CSV. Parent dirs are
                created if they don't exist.
            field_mappings: Harness field mappings. Values determine the
                target column set.

        Returns:
            The ``output_path`` (for call-site convenience).
        """
        raw_df = self._read_file(raw_path)

        mapped_cols = list(dict.fromkeys(
            col for col in (field_mappings or {}).values() if col
        ))
        # Intersection — don't crash if the raw file is missing a mapped
        # column (e.g. optional ``category``). Preserve mapped-column order.
        common_cols = [c for c in mapped_cols if c in raw_df.columns]
        if not common_cols:
            raise ValueError(
                "No common columns between raw dataset and field_mappings. "
                f"Raw columns: {list(raw_df.columns)}, mapped: {mapped_cols}"
            )

        raw_trimmed = raw_df[common_cols].astype(str)
        synthetic_df = pd.DataFrame(
            [{col: row.get(col, "") for col in common_cols} for row in approved_synthetic_rows],
            columns=common_cols,
        )

        merged = pd.concat([raw_trimmed, synthetic_df], ignore_index=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)
        return output_path

    def resolve_data_file(
        self,
        project_dir: Path,
        data_source: str,
        base_data_file: str,
        field_mappings: dict[str, str],
    ) -> str:
        """Resolve a ``data_source`` selector to a harness data-file argument.

        Behaviour:
          * ``"raw"``      — returns ``base_data_file`` unchanged. Caller
            is expected to let the harness resolve it relative to its
            ``data/raw/`` directory.
          * ``"synthetic"``— writes every approved synthetic row to
            ``data/processed/synthetic_only_<timestamp>.csv`` and returns
            the absolute path.
          * ``"merged"``   — writes raw + approved synthetic rows to
            ``data/processed/merged_<timestamp>.csv`` and returns the
            absolute path.

        Args:
            project_dir: Project root.
            data_source: One of ``"raw" | "synthetic" | "merged"``.
            base_data_file: Harness-configured ``data_file`` (relative to
                ``data/raw/``). Used as the fallback for ``"raw"`` and as
                the source for the merge.
            field_mappings: Harness field mappings.

        Returns:
            A string suitable for passing as ``--data-file`` to the harness.
            For ``"raw"`` this is a bare filename; for the other two modes
            it's an absolute path.

        Raises:
            ValueError: If ``data_source`` is unknown, if ``synthetic`` or
                ``merged`` is requested but no approved synthetic rows
                exist, or if the raw file is missing when merging.
        """
        if data_source == "raw":
            return base_data_file

        if data_source not in ("synthetic", "merged"):
            raise ValueError(
                f"Unknown data_source '{data_source}'. "
                "Expected one of: raw, synthetic, merged."
            )

        approved_rows = self.load_approved_synthetic_rows(project_dir, field_mappings)
        if not approved_rows:
            raise ValueError(
                "No approved synthetic cases found. "
                "Approve at least one case before selecting a synthetic data source."
            )

        processed_dir = project_dir / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

        mapped_cols = list(dict.fromkeys(
            col for col in (field_mappings or {}).values() if col
        ))
        if not mapped_cols:
            raise ValueError("field_mappings is empty — cannot resolve columns.")

        if data_source == "synthetic":
            output_path = processed_dir / f"synthetic_only_{timestamp}.csv"
            df = pd.DataFrame(
                [{col: row.get(col, "") for col in mapped_cols} for row in approved_rows],
                columns=mapped_cols,
            )
            df.to_csv(output_path, index=False)
            return str(output_path)

        # data_source == "merged"
        raw_path = project_dir / "data" / "raw" / base_data_file
        if not raw_path.exists():
            raise ValueError(
                f"Raw data file not found: {base_data_file}. "
                "Cannot merge — upload or re-select a raw dataset first."
            )
        output_path = processed_dir / f"merged_{timestamp}.csv"
        self.merge_datasets(raw_path, approved_rows, output_path, field_mappings)
        return str(output_path)

    def _read_file(self, file_path: Path) -> pd.DataFrame:
        """Read a data file into a pandas DataFrame based on extension."""
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(file_path)
        elif suffix == ".json":
            return pd.read_json(file_path)
        elif suffix == ".jsonl":
            return pd.read_json(file_path, lines=True)
        elif suffix in (".xlsx", ".xls"):
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _validate_mapping(self, df: pd.DataFrame, mappings: dict[str, str]) -> None:
        """Check that required mapped columns exist in the DataFrame columns."""
        for key in ("input", "expected_output"):
            col_name = mappings.get(key)
            if not col_name or col_name not in df.columns:
                raise ValueError(
                    f"Column '{col_name}' (mapped as '{key}') not found in data. "
                    f"Available columns: {list(df.columns)}"
                )
