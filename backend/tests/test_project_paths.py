# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for project path resolution and safe-path checking."""
from pathlib import Path

import pytest

from app.services.project_paths import (
    ProjectTraversalError,
    resolve_project_dir,
    safe_path,
)


def test_resolve_project_dir_returns_under_base(tmp_path):
    result = resolve_project_dir("abc", base_dir=tmp_path)
    assert result == tmp_path / "abc"


def test_resolve_project_dir_blocks_traversal(tmp_path):
    with pytest.raises(ProjectTraversalError):
        resolve_project_dir("../../etc", base_dir=tmp_path)


def test_safe_path_inside_project_ok(tmp_path):
    (tmp_path / "data").mkdir()
    p = safe_path(tmp_path, "data/file.csv")
    assert p == (tmp_path / "data" / "file.csv").resolve()


def test_safe_path_escape_returns_none(tmp_path):
    assert safe_path(tmp_path, "../../etc/passwd") is None


def test_resolve_project_dir_blocks_same_prefix_sibling(tmp_path):
    # /tmp/abc-sibling should not pass when base is /tmp/abc
    base = tmp_path / "abc"
    base.mkdir()
    with pytest.raises(ProjectTraversalError):
        resolve_project_dir("../abc-sibling", base_dir=base)


def test_safe_path_blocks_same_prefix_sibling(tmp_path):
    # base_dir/data-extra should not be reachable via ../data-extra
    (tmp_path / "data").mkdir()
    result = safe_path(tmp_path / "data", "../data-extra/file.csv")
    assert result is None
