# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``project_paths.safe_path`` — path-traversal prevention.

Cost-estimation tests moved to ``test_eval_runner.py`` when pricing was
consolidated into ``services.bedrock_pricing``.
"""

import pytest
from pathlib import Path

from app.services.project_paths import safe_path


class TestSafePath:
    """Tests for safe_path path traversal prevention."""

    def test_valid_filename(self, tmp_path):
        """Simple filename resolves to a path within base_dir."""
        result = safe_path(tmp_path, "test.csv")
        assert result is not None
        assert result == (tmp_path / "test.csv").resolve()

    def test_valid_nested_path(self, tmp_path):
        """Nested relative path within base_dir is accepted."""
        result = safe_path(tmp_path, "subdir/test.csv")
        assert result is not None
        assert str(result).startswith(str(tmp_path.resolve()))

    def test_traversal_with_dotdot(self, tmp_path):
        """Path traversal with ../../ returns None."""
        result = safe_path(tmp_path, "../../etc/passwd")
        assert result is None

    def test_traversal_with_leading_dotdot(self, tmp_path):
        """Path traversal with leading ../ returns None."""
        result = safe_path(tmp_path, "../secret.txt")
        assert result is None

    def test_traversal_with_deep_escape(self, tmp_path):
        """Deep traversal attempt returns None."""
        result = safe_path(tmp_path, "a/b/../../../../etc/hosts")
        assert result is None

    def test_absolute_path_outside_base(self, tmp_path):
        """Absolute path outside base_dir returns None."""
        result = safe_path(tmp_path, "/etc/passwd")
        assert result is None

    def test_current_dir_dot(self, tmp_path):
        """Current dir reference './file' stays within base_dir."""
        result = safe_path(tmp_path, "./test.csv")
        assert result is not None
        assert result == (tmp_path / "test.csv").resolve()

    def test_empty_name(self, tmp_path):
        """Empty name resolves to base_dir itself (still within base)."""
        result = safe_path(tmp_path, "")
        # Empty string resolves to base_dir itself, which starts with base_dir
        assert result is not None
