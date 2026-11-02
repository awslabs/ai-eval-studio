# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the synthesizer service (``app.services.synthesizer``).

DeepEval's ``Synthesizer`` is mocked out entirely — these tests focus on
the wrapper's dispatch logic + the CSV/manifest persistence contract.
"""

from __future__ import annotations

import csv
import json
from unittest.mock import MagicMock, patch

import pytest
from deepeval.dataset import Golden


@pytest.fixture
def mock_deepeval_synth():
    """Patch ``DeepEvalSynthesizer`` in the synthesizer module with a MagicMock.

    Yields the mock instance returned by the patched class constructor so
    tests can configure return values and assert calls.
    """
    with patch("app.services.synthesizer.DeepEvalSynthesizer") as mock_cls:
        instance = MagicMock()
        # DeepEval exposes these as attributes; our wrapper reads them when
        # rewriting styling config. Give them sensible defaults.
        instance.async_mode = False
        instance.max_concurrent = 10
        instance.set_styling_config = False
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def field_mappings():
    """Standard field_mappings used across tests."""
    return {
        "input": "question",
        "expected_output": "answer",
        "category": "topic",
        "id": "case_id",
    }


@pytest.fixture
def sample_goldens():
    """Three Goldens with varied metadata for write_batch tests."""
    return [
        Golden(
            input="What is 2+2?",
            expected_output="4",
            additional_metadata={"category": "math", "id": "m1"},
        ),
        Golden(
            input="Capital of France?",
            expected_output="Paris",
            additional_metadata={"category": "geography"},
        ),
        Golden(input="Spell hello", expected_output="hello"),
    ]


def _make_synth():
    """Import + instantiate Synthesizer with a fake model inside a patched context."""
    from app.services.synthesizer import Synthesizer
    return Synthesizer(model=MagicMock(name="BedrockRuntimeModel"))


def test_generate_from_goldens_forwards_seeds(mock_deepeval_synth):
    """``generate_from_goldens`` should call DeepEval with the seed list."""
    synth = _make_synth()
    seeds = [Golden(input="x", expected_output="y")]
    mock_deepeval_synth.generate_goldens_from_goldens.return_value = ["out"]

    result = synth.generate_from_goldens(seeds, max_goldens_per_golden=5)

    assert result == ["out"]
    mock_deepeval_synth.generate_goldens_from_goldens.assert_called_once_with(
        goldens=seeds,
        max_goldens_per_golden=5,
        include_expected_output=True,
    )


def test_generate_from_docs_forwards_paths(mock_deepeval_synth):
    """``generate_from_docs`` should stringify paths and forward them."""
    from pathlib import Path

    synth = _make_synth()
    mock_deepeval_synth.generate_goldens_from_docs.return_value = ["doc-golden"]

    result = synth.generate_from_docs(
        document_paths=[Path("/tmp/a.pdf"), "/tmp/b.txt"],
        max_goldens_per_context=3,
    )

    assert result == ["doc-golden"]
    mock_deepeval_synth.generate_goldens_from_docs.assert_called_once()
    kwargs = mock_deepeval_synth.generate_goldens_from_docs.call_args.kwargs
    assert kwargs["document_paths"] == ["/tmp/a.pdf", "/tmp/b.txt"]
    assert kwargs["max_goldens_per_context"] == 3


def test_generate_from_scratch_sets_styling(mock_deepeval_synth):
    """``generate_from_scratch`` should install a StylingConfig + call DeepEval."""
    synth = _make_synth()
    mock_deepeval_synth.generate_goldens_from_scratch.return_value = ["g1", "g2"]

    result = synth.generate_from_scratch(
        subject="medical Q&A",
        task="answer concisely",
        output_format="one-sentence plain text",
        num_cases=7,
        input_format="natural-language question",
    )

    assert result == ["g1", "g2"]
    styling = mock_deepeval_synth.styling_config
    assert styling.scenario == "medical Q&A"
    assert styling.task == "answer concisely"
    assert styling.expected_output_format == "one-sentence plain text"
    assert styling.input_format == "natural-language question"
    assert mock_deepeval_synth.set_styling_config is True
    mock_deepeval_synth.generate_goldens_from_scratch.assert_called_once_with(num_goldens=7)


def test_write_batch_creates_directory_and_files(
    tmp_path, mock_deepeval_synth, field_mappings, sample_goldens
):
    """``write_batch`` must create ``data/synthetic/`` + CSV + manifest."""
    synth = _make_synth()
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    # Intentionally do NOT pre-create data/synthetic/ — the method must do it.

    result = synth.write_batch(
        project_dir=project_dir,
        strategy="paraphrase",
        goldens=sample_goldens,
        field_mappings=field_mappings,
        seed_source="data/raw/seed.csv",
    )

    synthetic_dir = project_dir / "data" / "synthetic"
    assert synthetic_dir.is_dir()
    assert result["csv_path"].exists()
    assert result["manifest_path"].exists()
    assert result["total_cases"] == 3
    assert result["strategy"] == "paraphrase"


def test_write_batch_csv_uses_mapping_columns(
    tmp_path, mock_deepeval_synth, field_mappings, sample_goldens
):
    """CSV header + rows must use the mapping *values* as column names."""
    synth = _make_synth()
    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    result = synth.write_batch(
        project_dir=project_dir,
        strategy="paraphrase",
        goldens=sample_goldens,
        field_mappings=field_mappings,
    )

    with result["csv_path"].open() as f:
        rows = list(csv.DictReader(f))

    # Columns are exactly the mapping values (order preserved, deduped).
    with result["csv_path"].open() as f:
        header = f.readline().strip().split(",")
    assert header == ["question", "answer", "topic", "case_id"]

    # Values flow through from Golden + metadata.
    assert rows[0]["question"] == "What is 2+2?"
    assert rows[0]["answer"] == "4"
    assert rows[0]["topic"] == "math"
    assert rows[0]["case_id"] == "m1"

    # Row 1 has no id in metadata -> synthetic_<batch>_<index> fallback.
    assert rows[1]["topic"] == "geography"
    assert rows[1]["case_id"].startswith("synthetic_")
    assert rows[1]["case_id"].endswith("_1")

    # Row 2 has no metadata -> empty category + synthetic fallback id.
    assert rows[2]["topic"] == ""
    assert rows[2]["case_id"].endswith("_2")


def test_write_batch_manifest_shape(
    tmp_path, mock_deepeval_synth, field_mappings, sample_goldens
):
    """Manifest JSON must include metadata + per-case pending status."""
    synth = _make_synth()
    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    result = synth.write_batch(
        project_dir=project_dir,
        strategy="from_scratch",
        goldens=sample_goldens,
        field_mappings=field_mappings,
        seed_source=None,
        seed_run_id="run-42",
    )

    manifest = json.loads(result["manifest_path"].read_text())
    assert manifest["batch_id"] == result["batch_id"]
    assert manifest["strategy"] == "from_scratch"
    assert manifest["seed_source"] is None
    assert manifest["seed_run_id"] == "run-42"
    assert manifest["total_cases"] == 3
    assert manifest["field_mappings"] == field_mappings
    assert manifest["csv_file"] == result["csv_path"].name
    assert len(manifest["cases"]) == 3
    assert all(c["status"] == "pending" for c in manifest["cases"])
    assert [c["index"] for c in manifest["cases"]] == [0, 1, 2]


def test_write_batch_minimal_mapping(tmp_path, mock_deepeval_synth):
    """A mapping without category/id should still write input + expected_output."""
    synth = _make_synth()
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    goldens = [Golden(input="hi", expected_output="hello")]

    result = synth.write_batch(
        project_dir=project_dir,
        strategy="from_docs",
        goldens=goldens,
        field_mappings={"input": "q", "expected_output": "a"},
    )

    with result["csv_path"].open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["q"] == "hi"
    assert rows[0]["a"] == "hello"
    assert list(rows[0].keys()) == ["q", "a"]


class TestUnscoreableFlagAtWriteTime:
    """``write_batch`` records the fact; the approval gate acts on it.

    Kept at this layer too because ``write_batch`` takes arbitrary field
    mappings: the MCP tool happens to require an ``expected_output`` mapping,
    but this method must not flag a dataset that legitimately has no
    ground-truth column — that would be noise, not a finding.
    """

    def test_empty_expected_output_is_flagged_when_the_column_is_mapped(self, tmp_path):
        from deepeval.dataset import Golden

        from app.services.synthesizer import Synthesizer

        # __new__: write_batch touches no instance state, and __init__ would
        # construct a real DeepEval synthesizer (and its model).
        svc = Synthesizer.__new__(Synthesizer)
        batch = svc.write_batch(
            project_dir=tmp_path,
            strategy="paraphrase",
            goldens=[
                Golden(input="a", expected_output="A"),
                Golden(input="b", expected_output="   "),
                Golden(input="c", expected_output=None),
            ],
            field_mappings={"input": "q", "expected_output": "a"},
        )
        assert batch["unscoreable_cases"] == 2

        manifest = json.loads(batch["manifest_path"].read_text())
        assert manifest["unscoreable_cases"] == 2
        assert [c["index"] for c in manifest["cases"] if c.get("unscoreable")] == [1, 2]
        # Whitespace-only counts: a metric comparing it to real ground truth
        # fails the case exactly as an empty string does.
        assert manifest["cases"][1]["unscoreable"] == "missing_expected_output"

    def test_nothing_is_flagged_when_no_expected_column_is_mapped(self, tmp_path):
        from deepeval.dataset import Golden

        from app.services.synthesizer import Synthesizer

        # __new__: write_batch touches no instance state, and __init__ would
        # construct a real DeepEval synthesizer (and its model).
        svc = Synthesizer.__new__(Synthesizer)
        batch = svc.write_batch(
            project_dir=tmp_path,
            strategy="from_scratch",
            goldens=[Golden(input="a", expected_output=None)],
            field_mappings={"input": "q"},
        )
        assert batch["unscoreable_cases"] == 0
        manifest = json.loads(batch["manifest_path"].read_text())
        assert not any(c.get("unscoreable") for c in manifest["cases"])
