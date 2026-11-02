# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dataset parsing and LLMTestCase creation."""

import pytest
import json
from pathlib import Path

import pandas as pd

from app.services.dataset_service import DatasetService


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data" / "raw"
    d.mkdir(parents=True)
    return d


class TestDatasetService:
    def test_parse_csv(self, data_dir):
        csv_path = data_dir / "test.csv"
        csv_path.write_text("id,input,expected,category\n1,hello,world,greet\n2,bye,farewell,part\n")

        svc = DatasetService()
        mapping = {"input": "input", "expected_output": "expected", "category": "category", "id": "id"}
        cases = svc.load_test_cases(csv_path, mapping)

        assert len(cases) == 2
        assert cases[0].input == "hello"
        assert cases[0].expected_output == "world"
        assert cases[0].metadata["category"] == "greet"
        assert cases[0].metadata["id"] == "1"

    def test_parse_json(self, data_dir):
        json_path = data_dir / "test.json"
        json_path.write_text(json.dumps([
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "Capital of France?", "answer": "Paris"},
        ]))

        svc = DatasetService()
        mapping = {"input": "question", "expected_output": "answer"}
        cases = svc.load_test_cases(json_path, mapping)

        assert len(cases) == 2
        assert cases[0].input == "What is 2+2?"
        assert cases[0].expected_output == "4"

    def test_parse_jsonl(self, data_dir):
        jsonl_path = data_dir / "test.jsonl"
        jsonl_path.write_text('{"q":"a","a":"b"}\n{"q":"c","a":"d"}\n')

        svc = DatasetService()
        mapping = {"input": "q", "expected_output": "a"}
        cases = svc.load_test_cases(jsonl_path, mapping)

        assert len(cases) == 2
        assert cases[0].input == "a"
        assert cases[1].expected_output == "d"

    def test_max_test_cases(self, data_dir):
        csv_path = data_dir / "big.csv"
        rows = "id,input,expected\n" + "\n".join(f"{i},q{i},a{i}" for i in range(100))
        csv_path.write_text(rows)

        svc = DatasetService()
        mapping = {"input": "input", "expected_output": "expected", "id": "id"}
        cases = svc.load_test_cases(csv_path, mapping, max_cases=10)

        assert len(cases) == 10

    def test_category_filter(self, data_dir):
        csv_path = data_dir / "cats.csv"
        csv_path.write_text("input,expected,cat\na,b,x\nc,d,y\ne,f,x\n")

        svc = DatasetService()
        mapping = {"input": "input", "expected_output": "expected", "category": "cat"}
        cases = svc.load_test_cases(csv_path, mapping, category_filter=["x"])

        assert len(cases) == 2
        assert all(c.metadata["category"] == "x" for c in cases)

    def test_missing_field_raises(self, data_dir):
        csv_path = data_dir / "bad.csv"
        csv_path.write_text("col_a,col_b\n1,2\n")

        svc = DatasetService()
        mapping = {"input": "nonexistent", "expected_output": "col_b"}

        with pytest.raises(ValueError, match="nonexistent"):
            svc.load_test_cases(csv_path, mapping)

    def test_describe_dataset(self, data_dir):
        csv_path = data_dir / "desc.csv"
        csv_path.write_text("input,expected,cat\na,b,x\nc,d,y\ne,f,x\ng,h,y\n")

        svc = DatasetService()
        info = svc.describe_file(csv_path)

        assert info["row_count"] == 4
        assert set(info["columns"]) == {"input", "expected", "cat"}
        assert info["sample_rows"] is not None
        assert len(info["sample_rows"]) <= 5


# ---------------------------------------------------------------------------
# Synthetic-aware resolution helpers
# ---------------------------------------------------------------------------

def _seed_synthetic_batch(
    project_dir: Path,
    batch_id: str,
    rows: list[dict[str, str]],
    statuses: list[str],
    field_mappings: dict[str, str],
) -> None:
    """Write a CSV + manifest pair under ``data/synthetic/`` for tests.

    ``rows`` column layout must match the CSV layout we want to persist
    (caller's responsibility). ``statuses`` is 1:1 with ``rows``.
    """
    synthetic_dir = project_dir / "data" / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    csv_path = synthetic_dir / f"{batch_id}.csv"
    manifest_path = synthetic_dir / f"{batch_id}.manifest.json"

    columns = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(c, "")) for c in columns) + "\n")

    manifest = {
        "batch_id": batch_id,
        "timestamp": batch_id.split("_")[0],
        "strategy": "paraphrase",
        "total_cases": len(rows),
        "field_mappings": field_mappings,
        "csv_file": f"{batch_id}.csv",
        "cases": [
            {"index": i, "status": statuses[i]} for i in range(len(rows))
        ],
    }
    manifest_path.write_text(json.dumps(manifest))


class TestLoadApprovedSyntheticRows:
    """Tests for ``DatasetService.load_approved_synthetic_rows``."""

    def test_returns_empty_when_no_synthetic_dir(self, tmp_path):
        svc = DatasetService()
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        rows = svc.load_approved_synthetic_rows(
            project_dir, {"input": "q", "expected_output": "a"},
        )
        assert rows == []

    def test_returns_only_approved_rows_across_batches(self, tmp_path):
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        mappings = {"input": "question", "expected_output": "answer"}

        _seed_synthetic_batch(
            project_dir,
            "2026-04-22T10-00-00_paraphrase",
            rows=[
                {"question": "q1", "answer": "a1"},
                {"question": "q2", "answer": "a2"},
                {"question": "q3", "answer": "a3"},
            ],
            statuses=["approved", "pending", "rejected"],
            field_mappings=mappings,
        )
        _seed_synthetic_batch(
            project_dir,
            "2026-04-22T11-00-00_from_scratch",
            rows=[
                {"question": "q4", "answer": "a4"},
                {"question": "q5", "answer": "a5"},
            ],
            statuses=["approved", "approved"],
            field_mappings=mappings,
        )

        svc = DatasetService()
        rows = svc.load_approved_synthetic_rows(project_dir, mappings)

        # 3 approved cases total (one from batch 1, two from batch 2)
        assert len(rows) == 3
        questions = {r["question"] for r in rows}
        assert questions == {"q1", "q4", "q5"}

    def test_trims_columns_to_field_mappings(self, tmp_path):
        """Extra CSV columns should be dropped from the returned rows."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        mappings = {"input": "question", "expected_output": "answer"}

        _seed_synthetic_batch(
            project_dir,
            "2026-04-22T10-00-00_paraphrase",
            rows=[
                {"question": "q1", "answer": "a1", "extra": "noise"},
            ],
            statuses=["approved"],
            field_mappings=mappings,
        )

        svc = DatasetService()
        rows = svc.load_approved_synthetic_rows(project_dir, mappings)
        assert rows == [{"question": "q1", "answer": "a1"}]


class TestMergeDatasets:
    """Tests for ``DatasetService.merge_datasets``."""

    def test_merges_on_column_intersection(self, tmp_path):
        raw_path = tmp_path / "raw.csv"
        raw_path.write_text("question,answer,extra\nq0,a0,noise\n")

        approved = [
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
        ]
        out = tmp_path / "merged.csv"

        svc = DatasetService()
        svc.merge_datasets(
            raw_path=raw_path,
            approved_synthetic_rows=approved,
            output_path=out,
            field_mappings={"input": "question", "expected_output": "answer"},
        )

        merged = pd.read_csv(out)
        assert list(merged.columns) == ["question", "answer"]
        assert len(merged) == 3
        assert merged.iloc[0]["question"] == "q0"
        assert merged.iloc[2]["answer"] == "a2"

    def test_merges_when_raw_missing_optional_column(self, tmp_path):
        """Merge drops mapped columns the raw file doesn't provide."""
        raw_path = tmp_path / "raw.csv"
        raw_path.write_text("question,answer\nq0,a0\n")

        approved = [{"question": "q1", "answer": "a1", "category": "c1"}]
        out = tmp_path / "merged.csv"

        svc = DatasetService()
        svc.merge_datasets(
            raw_path=raw_path,
            approved_synthetic_rows=approved,
            output_path=out,
            field_mappings={
                "input": "question",
                "expected_output": "answer",
                "category": "category",  # absent from raw — should be dropped
            },
        )
        merged = pd.read_csv(out)
        assert "category" not in merged.columns
        assert len(merged) == 2


class TestResolveDataFile:
    """Tests for ``DatasetService.resolve_data_file``."""

    def test_raw_returns_base_filename(self, tmp_path):
        svc = DatasetService()
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        result = svc.resolve_data_file(
            project_dir=project_dir,
            data_source="raw",
            base_data_file="seed.csv",
            field_mappings={"input": "q", "expected_output": "a"},
        )
        assert result == "seed.csv"

    def test_synthetic_writes_processed_file(self, tmp_path):
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        mappings = {"input": "question", "expected_output": "answer"}
        _seed_synthetic_batch(
            project_dir,
            "2026-04-22T10-00-00_paraphrase",
            rows=[
                {"question": "q1", "answer": "a1"},
                {"question": "q2", "answer": "a2"},
            ],
            statuses=["approved", "approved"],
            field_mappings=mappings,
        )

        svc = DatasetService()
        path_str = svc.resolve_data_file(
            project_dir=project_dir,
            data_source="synthetic",
            base_data_file="seed.csv",
            field_mappings=mappings,
        )
        path = Path(path_str)
        assert path.exists()
        assert path.parent == project_dir / "data" / "processed"
        assert path.name.startswith("synthetic_only_")

        df = pd.read_csv(path)
        assert list(df.columns) == ["question", "answer"]
        assert len(df) == 2

    def test_synthetic_raises_without_approvals(self, tmp_path):
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        svc = DatasetService()
        with pytest.raises(ValueError, match="No approved synthetic"):
            svc.resolve_data_file(
                project_dir=project_dir,
                data_source="synthetic",
                base_data_file="seed.csv",
                field_mappings={"input": "q", "expected_output": "a"},
            )

    def test_merged_concatenates_raw_and_synthetic(self, tmp_path):
        project_dir = tmp_path / "project"
        (project_dir / "data" / "raw").mkdir(parents=True)
        (project_dir / "data" / "raw" / "seed.csv").write_text(
            "question,answer\nq0,a0\n"
        )
        mappings = {"input": "question", "expected_output": "answer"}
        _seed_synthetic_batch(
            project_dir,
            "2026-04-22T10-00-00_paraphrase",
            rows=[{"question": "q1", "answer": "a1"}],
            statuses=["approved"],
            field_mappings=mappings,
        )

        svc = DatasetService()
        path_str = svc.resolve_data_file(
            project_dir=project_dir,
            data_source="merged",
            base_data_file="seed.csv",
            field_mappings=mappings,
        )
        path = Path(path_str)
        assert path.name.startswith("merged_")
        df = pd.read_csv(path)
        assert len(df) == 2
        assert df.iloc[0]["question"] == "q0"
        assert df.iloc[1]["question"] == "q1"

    def test_merged_raises_when_raw_missing(self, tmp_path):
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        mappings = {"input": "q", "expected_output": "a"}
        _seed_synthetic_batch(
            project_dir,
            "2026-04-22T10-00-00_paraphrase",
            rows=[{"q": "q1", "a": "a1"}],
            statuses=["approved"],
            field_mappings=mappings,
        )
        svc = DatasetService()
        with pytest.raises(ValueError, match="Raw data file not found"):
            svc.resolve_data_file(
                project_dir=project_dir,
                data_source="merged",
                base_data_file="seed.csv",
                field_mappings=mappings,
            )

    def test_unknown_data_source_raises(self, tmp_path):
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        svc = DatasetService()
        with pytest.raises(ValueError, match="Unknown data_source"):
            svc.resolve_data_file(
                project_dir=project_dir,
                data_source="bogus",
                base_data_file="seed.csv",
                field_mappings={"input": "q", "expected_output": "a"},
            )
