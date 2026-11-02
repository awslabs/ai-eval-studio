# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-call ``accumulated_usage`` parity tests for BedrockMantleModel.

Pins that ``BedrockMantleModel.accumulated_usage`` holds the same
{inputTokens, outputTokens} contract as ``BedrockRuntimeModel`` across
multi-call runs, with reasoning tokens counted ONCE (via output_tokens, NOT
additionally via output_tokens_details.reasoning_tokens), with no accrual on
error paths or owner-instance preflight, and with thread-safe summation
under concurrent calls.

These are parity assertions only: they pin the accumulator's contract
against drift from adjacent code (preflight, descriptor invalidation).
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pytest

from app.services import bedrock_pricing
from app.services.bedrock_mantle_mappers import MantleAPIRejected
from app.services.bedrock_mantle_model import (
    BedrockMantleModel,
    _close_shared_mantle_client,
)
from app.services.model_descriptors import MantleDescriptor

FIXTURES = Path(__file__).parent / "fixtures" / "mantle_payloads"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def reset_shared_client():
    """Fresh shared client per test — same pattern as test_bedrock_mantle_model."""
    _close_shared_mantle_client()
    yield
    _close_shared_mantle_client()


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """Shorten retry delays so error-path tests don't spin."""
    monkeypatch.setenv("BEDROCK_RETRY_INITIAL_S", "0.001")
    monkeypatch.setenv("BEDROCK_RETRY_CAP_S", "0.002")


@pytest.fixture(autouse=True)
def aws_creds(monkeypatch):
    """Static creds so SigV4 signing has something to sign with."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATESTACCESSKEY")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture
def descriptor() -> MantleDescriptor:
    return {"endpoint": "mantle", "api": "responses", "region": "us-east-1"}


def _ok_payload(input_tokens: int = 9, output_tokens: int = 12) -> dict:
    """Build a Responses-shaped 200 payload with explicit token counts."""
    return {
        "id": "resp_test_001",
        "object": "response",
        "model": "openai.gpt-5.5",
        "status": "completed",
        "output": [
            {"type": "reasoning", "id": "r_001", "summary": []},
            {
                "type": "message",
                "id": "m_001",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "OK"}],
            },
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


# --------------------------------------------------------------------------- #
# Multi-call accumulator parity                                               #
# --------------------------------------------------------------------------- #


def test_single_call_accrues_usage(httpx_mock, descriptor):
    """1 call → exactly the per-call usage."""
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_ok_payload(input_tokens=10, output_tokens=20),
    )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    model.generate_with_system(None, "Say OK")
    assert model.accumulated_usage == {"inputTokens": 10, "outputTokens": 20}


def test_multi_call_sums_correctly(httpx_mock, descriptor):
    """5 calls with varied usages → sum across all 5."""
    usages = [(5, 7), (11, 13), (1, 1), (100, 200), (0, 9)]
    for inp, out in usages:
        httpx_mock.add_response(
            method="POST",
            url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
            json=_ok_payload(input_tokens=inp, output_tokens=out),
        )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    for _ in usages:
        model.generate_with_system(None, "Say OK")
    expected_in = sum(u[0] for u in usages)
    expected_out = sum(u[1] for u in usages)
    assert model.accumulated_usage == {
        "inputTokens": expected_in,
        "outputTokens": expected_out,
    }


def test_reasoning_tokens_counted_once(httpx_mock, descriptor):
    """Reasoning tokens accrue via output_tokens, NOT in addition.

    The fixture ``responses_200_minimal.json`` carries
    ``output_tokens=12`` (which already INCLUDES any reasoning
    tokens). Asserting we record 12, NOT 12 + reasoning_tokens, is
    the no-double-count gate.
    """
    payload = _load("responses_200_minimal.json")
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=payload,
    )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    model.generate_with_system(None, "Say OK")
    # The fixture's usage.output_tokens is 12. Assert that's what we
    # accrued, not a value reflecting reasoning_tokens summed in.
    assert model.accumulated_usage["outputTokens"] == payload["usage"]["output_tokens"]
    assert model.accumulated_usage["inputTokens"] == payload["usage"]["input_tokens"]


def test_shape_matches_bedrock_runtime_model(httpx_mock, descriptor):
    """The shape is EXACTLY {inputTokens, outputTokens} — no extra/wrong keys."""
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_ok_payload(),
    )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    model.generate_with_system(None, "Say OK")
    keys = set(model.accumulated_usage.keys())
    assert keys == {"inputTokens", "outputTokens"}
    # No snake_case variants leaking through.
    assert "input_tokens" not in model.accumulated_usage
    assert "output_tokens" not in model.accumulated_usage
    # No total/cache fields in the mantle accumulator (BedrockRuntimeModel's
    # surface emits cache fields when they're non-zero, but the mantle
    # adapter's response shape doesn't carry cache breakdowns — the
    # accumulator stays minimal).
    assert "totalTokens" not in model.accumulated_usage
    assert "cacheReadInputTokens" not in model.accumulated_usage


def test_concurrent_calls_thread_safe(httpx_mock, descriptor):
    """100 concurrent calls (4 threads × 25 each) → exact sum, no torn reads."""
    n_calls = 100
    payload = _ok_payload(input_tokens=3, output_tokens=5)
    for _ in range(n_calls):
        httpx_mock.add_response(
            method="POST",
            url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
            json=payload,
        )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")

    def _one_call():
        return model.generate_with_system(None, "Say OK")

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(_one_call) for _ in range(n_calls)]
        for f in as_completed(futures):
            f.result()

    # No off-by-one, no torn reads — exact 100x.
    assert model.accumulated_usage == {
        "inputTokens": 3 * n_calls,
        "outputTokens": 5 * n_calls,
    }


def test_error_path_does_not_accrue(httpx_mock, descriptor, monkeypatch):
    """Mock 400 validation_error → MantleAPIRejected → accumulator stays at zero."""
    monkeypatch.setenv("BEDROCK_RETRY_MAX_ATTEMPTS", "1")
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        status_code=400,
        json={
            "error": {
                "code": "validation_error",
                "message": "does not support",
            }
        },
    )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    with pytest.raises(MantleAPIRejected):
        model.generate_with_system(None, "Say OK")
    # Pre-condition (zero) preserved post-error.
    assert model.accumulated_usage == {"inputTokens": 0, "outputTokens": 0}


def test_preflight_does_not_accrue_on_owner_instance(httpx_mock, descriptor):
    """Cross-check — owner's accumulator clean after preflight().

    Throwaway-adapter pattern is the contract; this is a duplicate of
    a preflight test but lives here as a parity invariant alongside the
    other multi-call assertions.
    """
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_ok_payload(),
    )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    assert model.accumulated_usage == {"inputTokens": 0, "outputTokens": 0}
    model.preflight()
    # Throwaway adapter accrued the call; the owner did NOT.
    assert model.accumulated_usage == {"inputTokens": 0, "outputTokens": 0}


def test_estimate_cost_usd_consumes_accumulated_usage(
    httpx_mock, descriptor, monkeypatch
):
    """Parity invariant — accumulated_usage is a valid input to estimate_cost_usd.

    Prime the LiteLLM cache with a known per-1M rate, then assert the
    cost computed from ``model.accumulated_usage`` matches the formula
    by hand. No key rewriting needed (the accumulator's
    ``inputTokens``/``outputTokens`` shape is what
    ``estimate_cost_usd`` reads).
    """
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_ok_payload(input_tokens=1_000_000, output_tokens=500_000),
    )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    model.generate_with_system(None, "Say OK")
    assert model.accumulated_usage == {
        "inputTokens": 1_000_000,
        "outputTokens": 500_000,
    }

    # Prime the pricing cache for openai/gpt-5.5 — bypassing the
    # network fetch, mirroring the manual prime pattern documented for
    # bedrock_pricing tests.
    bedrock_pricing._clear_pricing_cache()
    monkeypatch.setattr(
        bedrock_pricing,
        "_pricing_cache",
        {"openai/gpt-5.5": {"input": 5.0, "output": 15.0}},
        raising=False,
    )
    monkeypatch.setattr(
        bedrock_pricing,
        "_pricing_cache_timestamp",
        time.time(),
        raising=False,
    )
    try:
        cost = bedrock_pricing.estimate_cost_usd(
            "openai.gpt-5.5", model.accumulated_usage
        )
    finally:
        bedrock_pricing._clear_pricing_cache()

    # 1M input × $5/1M + 0.5M output × $15/1M = 5 + 7.5 = 12.5 USD.
    assert cost == pytest.approx(12.5)
