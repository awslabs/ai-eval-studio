# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for harness config convenience loaders."""
from pathlib import Path

import pytest

from app.models.schemas import HarnessConfig
from app.services.harness_config import load_harness_config


def test_load_harness_config_returns_none_when_missing(tmp_path):
    assert load_harness_config(tmp_path) is None


def test_load_harness_config_parses_existing(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    # Use lowercase names matching the field names parse_harness_config extracts
    (src / "config.py").write_text(
        "models = ['anthropic.claude-haiku']\n"
        "data_file = 'data/test.csv'\n"
        "field_mappings = {'input': 'q', 'expected_output': 'a'}\n"
    )
    cfg = load_harness_config(tmp_path)
    assert isinstance(cfg, HarnessConfig)
    assert cfg.models == ["anthropic.claude-haiku"]
    assert cfg.data_file == "data/test.csv"
    assert cfg.field_mappings == {"input": "q", "expected_output": "a"}


def test_load_harness_config_raises_on_syntax_error(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.py").write_text("class Broken(:")
    with pytest.raises(SyntaxError):
        load_harness_config(tmp_path)
