# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset loading against the sample fixture dataset.

Uses the rag-qa-10.csv fixture: 10 test cases in two categories (factual,
reasoning).
"""
from pathlib import Path

import pytest

from app.services.dataset_service import DatasetService

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "rag-qa-10.csv"


class TestEvalIntegration:
    """Dataset loading tests using the rag-qa-10 fixture."""

    def test_fixture_exists(self):
        """Verify the fixture file is present and parseable."""
        assert FIXTURE_PATH.exists(), f"Fixture not found at {FIXTURE_PATH}"
        svc = DatasetService()
        info = svc.describe_file(FIXTURE_PATH)
        assert info["row_count"] == 10
        assert set(info["columns"]) == {"id", "category", "input", "context", "expected_output"}

    def test_load_all_cases(self):
        """Load all 10 cases with proper field mapping."""
        svc = DatasetService()
        mapping = {
            "input": "input",
            "expected_output": "expected_output",
            "category": "category",
            "id": "id",
        }
        cases = svc.load_test_cases(FIXTURE_PATH, mapping)

        assert len(cases) == 10
        # Verify first case
        assert "context window" in cases[0].input.lower()
        assert cases[0].expected_output is not None
        assert cases[0].metadata["category"] == "factual"
        assert cases[0].metadata["id"] == "1"

    def test_filter_by_category(self):
        """Filter to only reasoning cases."""
        svc = DatasetService()
        mapping = {
            "input": "input",
            "expected_output": "expected_output",
            "category": "category",
        }
        cases = svc.load_test_cases(FIXTURE_PATH, mapping, category_filter=["reasoning"])

        assert len(cases) == 2
        assert all(c.metadata["category"] == "reasoning" for c in cases)

    def test_sample_limit(self):
        """Limit to 3 cases for a quick smoke test."""
        svc = DatasetService()
        mapping = {
            "input": "input",
            "expected_output": "expected_output",
        }
        cases = svc.load_test_cases(FIXTURE_PATH, mapping, max_cases=3)
        assert len(cases) == 3
