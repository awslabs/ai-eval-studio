# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the structural short-name derivation.

Covers every id form discovery emits today (runtime inference profiles,
foundation models, mantle ids — every provider in the catalog) plus the
dotted-version regression that mangled ``xai.grok-4.3`` to ``"3"`` and
``openai.gpt-5.4`` to ``"4"``.

The rule has zero hardcoded provider/region lists — adding a new
provider must NOT require a code change. The parity test enforces that
the platform copy and the harness-template copy stay byte-equal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.model_id_short import short_model_id


@pytest.mark.parametrize(
    "model_id,expected",
    [
        # Dotted versions — the regression this guards against.
        ("xai.grok-4.3", "grok-4.3"),
        ("openai.gpt-5.4", "gpt-5.4"),
        ("openai.gpt-5.5", "gpt-5.5"),
        ("deepseek.v3.2", "v3.2"),
        ("deepseek.v3.1", "v3.1"),
        # Anthropic — claude- preserved.
        ("anthropic.claude-opus-4-8", "claude-opus-4-8"),
        ("anthropic.claude-opus-4-7", "claude-opus-4-7"),
        ("global.anthropic.claude-opus-4-8", "claude-opus-4-8"),
        ("global.anthropic.claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("us.anthropic.claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("eu.anthropic.claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("apac.anthropic.claude-sonnet-4-6", "claude-sonnet-4-6"),
        # Version + date suffixes.
        ("us.anthropic.claude-sonnet-4-6-v1:0", "claude-sonnet-4-6"),
        ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "claude-haiku-4-5"),
        ("us.anthropic.claude-haiku-4-5-20251001", "claude-haiku-4-5"),
        # Amazon.
        ("us.amazon.nova-lite-v1:0", "nova-lite"),
        ("amazon.nova-pro-v1:0", "nova-pro"),
        ("apac.amazon.nova-lite-v1:0", "nova-lite"),
        # Meta.
        ("meta.llama3-70b-instruct-v1:0", "llama3-70b-instruct"),
        ("us.meta.llama3-1-70b-instruct-v1:0", "llama3-1-70b-instruct"),
        # New providers from the mantle catalog — no code change needed.
        ("google.gemma-3-27b-it", "gemma-3-27b-it"),
        ("google.gemma-4-31b", "gemma-4-31b"),
        ("mistral.mistral-large-3-v1:0", "mistral-large-3"),
        ("kimi.kimi-k2.5", "kimi-k2.5"),
        ("glm.glm-4.7", "glm-4.7"),
        ("qwen.qwen3", "qwen3"),
        ("minimax.minimax-m2", "minimax-m2"),
        # Edge cases.
        ("custom-model", "custom-model"),
        ("", ""),
        # Multi-region prefixes never previously listed (au, us-gov).
        ("au.anthropic.claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("us-gov.anthropic.claude-sonnet-4-6", "claude-sonnet-4-6"),
    ],
)
def test_short_model_id(model_id: str, expected: str) -> None:
    """Every id form maps to the expected verbatim remainder."""
    assert short_model_id(model_id) == expected


def test_dotted_version_no_collapse_to_digit() -> None:
    """Regression: split('.')[-1] mangled grok-4.3 to '3' — never again.

    The root of the bug: when bar charts saw test_run_id
    "3", "4", "5", Plotly auto-typed the x-axis as numeric and dropped
    every Claude model (whose names use dashes, not dots). This test fails
    the moment anyone reintroduces a "last segment" extraction.
    """
    for short in (
        short_model_id("xai.grok-4.3"),
        short_model_id("openai.gpt-5.4"),
        short_model_id("openai.gpt-5.5"),
    ):
        assert not short.isdigit(), f"short name {short!r} would collapse Plotly to numeric"


def test_template_and_service_copies_byte_equal() -> None:
    """Templates ↔ services parity: model_id_short.py is duplicated unchanged.

    Same rule as bedrock_runtime_model.py and bedrock_mantle_mappers.py —
    the platform copy and the harness-template copy must stay in step.
    """
    repo_root = Path(__file__).resolve().parents[2]
    service_copy = repo_root / "backend/app/services/model_id_short.py"
    template_copy = (
        repo_root / "backend/app/templates/harness/src/models/model_id_short.py"
    )
    assert service_copy.read_bytes() == template_copy.read_bytes(), (
        "model_id_short.py drift between services and templates — keep them in step"
    )
