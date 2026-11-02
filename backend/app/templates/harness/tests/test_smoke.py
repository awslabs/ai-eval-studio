# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Smoke tests for evaluation harness.

These tests verify that all modules can be imported without errors.
They do not test functionality, only that the project structure is valid
and dependencies are properly configured.
"""

import pytest


def test_import_config():
    """Verify config module can be imported."""
    from src.config import EvalConfig

    # Check that EvalConfig class exists and can be instantiated
    assert EvalConfig is not None


def test_import_runner():
    """Verify evaluation runner module can be imported."""
    from src.evaluation.runner import EvalRunner

    # Check that EvalRunner class exists
    assert EvalRunner is not None


def test_import_dataset_loader():
    """Verify dataset loader module can be imported."""
    from src.evaluation.dataset_loader import load_test_cases, format_user_message

    # Check that functions exist
    assert load_test_cases is not None
    assert format_user_message is not None


def test_import_bedrock_runtime_model():
    """Verify BedrockRuntimeModel can be imported."""
    from src.models.bedrock_runtime_model import BedrockRuntimeModel

    # Check that BedrockRuntimeModel class exists
    assert BedrockRuntimeModel is not None


def test_import_metrics():
    """Verify metrics module can be imported."""
    from src.metrics import build_metrics

    # Check that build_metrics function exists
    assert build_metrics is not None


def test_bedrock_model_signature():
    """Verify BedrockRuntimeModel has required methods."""
    from src.models.bedrock_runtime_model import BedrockRuntimeModel

    # Check that BedrockRuntimeModel has the required methods
    assert hasattr(BedrockRuntimeModel, "generate")
    assert hasattr(BedrockRuntimeModel, "generate_with_system")
    assert hasattr(BedrockRuntimeModel, "a_generate")
    assert hasattr(BedrockRuntimeModel, "get_model_name")
    assert hasattr(BedrockRuntimeModel, "load_model")


def test_runner_signature():
    """Verify EvalRunner has required methods."""
    from src.evaluation.runner import EvalRunner

    # Check that EvalRunner has the required methods
    assert hasattr(EvalRunner, "run")
    assert hasattr(EvalRunner, "run_single_model")
    assert hasattr(EvalRunner, "generate_outputs")
    assert hasattr(EvalRunner, "save_results")
