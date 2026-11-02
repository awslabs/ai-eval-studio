# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Bedrock Mantle mapper functions.

Exercises the request builders and response parsers for all three mantle
APIs (``responses``, ``chat_completions``, ``anthropic_messages``) against
payloads in ``backend/tests/fixtures/mantle_payloads/``. No network or AWS
credentials are touched — pure-function tests over the loaded fixtures.
"""

import json
from pathlib import Path

import pytest

from app.services.bedrock_mantle_mappers import (
    MantleAPIRejected,
    MantleReasoningTruncated,
    MantleSchemaDrift,
    _build_anthropic_messages_request,
    _build_chat_completions_request,
    _build_responses_request,
    _parse_anthropic_messages_response,
    _parse_chat_completions_response,
    _parse_responses_response,
)

FIXTURES = Path(__file__).parent / "fixtures" / "mantle_payloads"


def _load(name: str) -> dict:
    """Load a JSON fixture from the mantle_payloads directory."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# _build_responses_request                                                    #
# --------------------------------------------------------------------------- #


def test_build_responses_request_omits_instructions_when_system_empty():
    """Empty system prompt → ``instructions`` key absent (matches captured request)."""
    req = _build_responses_request(
        model_id="openai.gpt-5.5",
        system_prompt=None,
        user_prompt="Say OK",
        max_output_tokens=16,
        supports_temperature=False,
        temperature=0.0,
    )
    assert req == _load("responses_request.json")
    assert "instructions" not in req


def test_build_responses_request_includes_instructions_when_system_provided():
    """Non-empty system prompt → ``instructions`` carries it verbatim."""
    req = _build_responses_request(
        model_id="openai.gpt-5.5",
        system_prompt="You are helpful",
        user_prompt="Say OK",
        max_output_tokens=16,
        supports_temperature=False,
        temperature=0.0,
    )
    assert req["instructions"] == "You are helpful"


def test_build_responses_request_omits_instructions_when_system_whitespace_only():
    """Whitespace-only system prompt → ``instructions`` omitted (no empty-string send)."""
    req = _build_responses_request(
        model_id="openai.gpt-5.5",
        system_prompt="   \n\t  ",
        user_prompt="Say OK",
        max_output_tokens=16,
        supports_temperature=False,
        temperature=0.0,
    )
    assert "instructions" not in req


def test_build_responses_request_omits_temperature_when_supports_false():
    """``supports_temperature=False`` → no ``temperature`` key."""
    req = _build_responses_request(
        model_id="openai.gpt-5.5",
        system_prompt=None,
        user_prompt="Say OK",
        max_output_tokens=16,
        supports_temperature=False,
        temperature=0.7,
    )
    assert "temperature" not in req


def test_build_responses_request_includes_temperature_when_supports_true():
    """``supports_temperature=True`` → ``temperature`` set to the passed value."""
    req = _build_responses_request(
        model_id="openai.gpt-5.5",
        system_prompt=None,
        user_prompt="Say OK",
        max_output_tokens=16,
        supports_temperature=True,
        temperature=0.7,
    )
    assert req["temperature"] == 0.7


# --------------------------------------------------------------------------- #
# _parse_responses_response                                                   #
# --------------------------------------------------------------------------- #


def test_parse_responses_extracts_text_and_normalizes_usage():
    """Live-captured 200 → ``("OK", {"inputTokens":9, "outputTokens":12})``."""
    payload = _load("responses_200_minimal.json")
    text, usage = _parse_responses_response(payload)
    assert text == "OK"
    assert usage == {"inputTokens": 9, "outputTokens": 12}


def test_parse_responses_skips_reasoning_blocks():
    """Reasoning blocks are skipped — only ``message.content[].output_text`` contributes."""
    # The captured fixture has output[0]=reasoning, output[1]=message.
    # If the parser slipped and picked output[0], it would return ""
    # or raise. Assert the message text wins.
    payload = _load("responses_200_minimal.json")
    text, _ = _parse_responses_response(payload)
    assert text == "OK"


def test_parse_responses_raises_truncation_on_incomplete_max_output_tokens():
    """Status='incomplete' + reason='max_output_tokens' raises (not silent retry)."""
    payload = _load("responses_incomplete_truncation.json")
    with pytest.raises(MantleReasoningTruncated, match="max_output_tokens"):
        _parse_responses_response(payload)


def test_parse_responses_truncation_message_references_model_id():
    """Truncation error message carries the model_id for operator triage."""
    payload = _load("responses_incomplete_truncation.json")
    try:
        _parse_responses_response(payload)
    except MantleReasoningTruncated as e:
        assert "openai.gpt-5.5" in str(e)
    else:
        pytest.fail("MantleReasoningTruncated was not raised")


def test_parse_responses_raises_schema_drift_when_output_missing():
    """No ``output`` key → MantleSchemaDrift mentioning ``output``."""
    payload = {
        "id": "x",
        "status": "completed",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "output": "not-a-list",  # malformed type triggers schema drift
    }
    with pytest.raises(MantleSchemaDrift, match="output"):
        _parse_responses_response(payload)


def test_parse_responses_raises_schema_drift_when_no_output_text_block():
    """Reasoning-only output → MantleSchemaDrift mentioning ``output_text``."""
    payload = {
        "id": "x",
        "status": "completed",
        "output": [{"type": "reasoning", "summary": []}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    with pytest.raises(MantleSchemaDrift, match="output_text"):
        _parse_responses_response(payload)


def test_parse_responses_handles_missing_usage_with_zeros():
    """Missing ``usage`` field → defensive zeros (no crash)."""
    payload = {
        "id": "x",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "OK"}],
            }
        ],
    }
    text, usage = _parse_responses_response(payload)
    assert text == "OK"
    assert usage == {"inputTokens": 0, "outputTokens": 0}


def test_parse_responses_handles_malformed_usage_with_zeros():
    """Malformed usage values (str / None) → defensive zero (mirrors BedrockRuntimeModel._accrue_usage)."""
    payload = {
        "id": "x",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "OK"}],
            }
        ],
        "usage": {"input_tokens": "x", "output_tokens": None},
    }
    text, usage = _parse_responses_response(payload)
    assert usage == {"inputTokens": 0, "outputTokens": 0}


def test_parse_responses_concatenates_multiple_output_text_blocks():
    """Multiple text blocks within one message → concatenated in order."""
    payload = {
        "id": "x",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Hello "},
                    {"type": "output_text", "text": "World"},
                ],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 8},
    }
    text, usage = _parse_responses_response(payload)
    assert text == "Hello World"
    assert usage == {"inputTokens": 5, "outputTokens": 8}


# --------------------------------------------------------------------------- #
# chat_completions mapper — live wire-format pinned                            #
# --------------------------------------------------------------------------- #


def test_build_chat_completions_user_only_message():
    body = _build_chat_completions_request(
        model_id="mistral.mistral-large-3",
        system_prompt=None,
        user_prompt="Say OK",
        max_output_tokens=64,
        supports_temperature=True,
        temperature=0.0,
    )
    assert body == {
        "model": "mistral.mistral-large-3",
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 64,
        "temperature": 0.0,
    }


def test_build_chat_completions_with_system_message_prepended():
    body = _build_chat_completions_request(
        model_id="mistral.mistral-large-3",
        system_prompt="be brief",
        user_prompt="Say OK",
        max_output_tokens=64,
        supports_temperature=False,
        temperature=0.0,
    )
    assert body["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "Say OK"},
    ]
    # supports_temperature=False → no temperature field at all.
    assert "temperature" not in body


def test_build_chat_completions_omits_system_when_whitespace_only():
    body = _build_chat_completions_request(
        model_id="mistral.mistral-large-3",
        system_prompt="   ",
        user_prompt="hi",
        max_output_tokens=64,
        supports_temperature=True,
        temperature=0.7,
    )
    assert all(m["role"] != "system" for m in body["messages"])


def test_parse_chat_completions_extracts_text_and_normalizes_usage():
    payload = _load("chat_completions_200_minimal.json")
    text, usage = _parse_chat_completions_response(payload)
    assert text == "OK"
    # prompt_tokens → inputTokens, completion_tokens → outputTokens
    assert usage == {"inputTokens": 6, "outputTokens": 2}


def test_parse_chat_completions_raises_schema_drift_on_missing_choices():
    with pytest.raises(MantleSchemaDrift):
        _parse_chat_completions_response({"usage": {}})


def test_parse_chat_completions_raises_schema_drift_on_empty_choices():
    with pytest.raises(MantleSchemaDrift):
        _parse_chat_completions_response({"choices": [], "usage": {}})


def test_parse_chat_completions_handles_missing_usage_with_zeros():
    payload = {
        "choices": [{"message": {"content": "OK"}}],
    }
    text, usage = _parse_chat_completions_response(payload)
    assert text == "OK"
    assert usage == {"inputTokens": 0, "outputTokens": 0}


def test_parse_chat_completions_tolerates_null_content():
    """Reasoning models on chat_completions emit content:null while burning
    the cap on hidden reasoning. Permissive parse: treat as empty text,
    do NOT raise. Regression for MiniMax M2 / GLM 4.7 / Kimi K2.5."""
    payload = _load("chat_completions_200_null_content.json")
    text, usage = _parse_chat_completions_response(payload)
    assert text == ""
    assert usage == {"inputTokens": 6, "outputTokens": 64}


def test_parse_chat_completions_raises_schema_drift_on_non_string_non_null_content():
    """A truly-malformed response (content is a list/dict) still raises.
    Only literal None is tolerated — permissive parse, but not blind."""
    payload = {
        "choices": [{"message": {"content": ["chunk1", "chunk2"]}}],
        "usage": {},
    }
    with pytest.raises(MantleSchemaDrift):
        _parse_chat_completions_response(payload)


# --------------------------------------------------------------------------- #
# anthropic_messages mapper — live wire-format pinned                          #
# --------------------------------------------------------------------------- #


def test_build_anthropic_messages_user_only_message():
    body = _build_anthropic_messages_request(
        model_id="anthropic.claude-opus-4-8",
        system_prompt=None,
        user_prompt="Say OK",
        max_output_tokens=64,
        supports_temperature=True,
        temperature=0.0,
    )
    assert body == {
        "model": "anthropic.claude-opus-4-8",
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Say OK"}],
        "temperature": 0.0,
    }


def test_build_anthropic_messages_includes_system_top_level():
    body = _build_anthropic_messages_request(
        model_id="anthropic.claude-opus-4-8",
        system_prompt="be brief",
        user_prompt="Say OK",
        max_output_tokens=64,
        supports_temperature=False,
        temperature=0.0,
    )
    # system is a TOP-LEVEL field (not a message), per the verified live shape.
    assert body["system"] == "be brief"
    assert all(m["role"] != "system" for m in body["messages"])
    assert "temperature" not in body


def test_build_anthropic_messages_omits_system_when_empty():
    body = _build_anthropic_messages_request(
        model_id="anthropic.claude-opus-4-8",
        system_prompt="",
        user_prompt="hi",
        max_output_tokens=64,
        supports_temperature=True,
        temperature=0.5,
    )
    assert "system" not in body


def test_build_anthropic_messages_required_anthropic_version():
    """Without anthropic_version the live API returns 400; the field is non-negotiable."""
    body = _build_anthropic_messages_request(
        model_id="anthropic.claude-opus-4-8",
        system_prompt=None,
        user_prompt="x",
        max_output_tokens=64,
        supports_temperature=True,
        temperature=0.0,
    )
    assert body["anthropic_version"] == "bedrock-2023-05-31"


def test_parse_anthropic_messages_extracts_text_and_normalizes_usage():
    payload = _load("anthropic_messages_200_minimal.json")
    text, usage = _parse_anthropic_messages_response(payload)
    assert text == "OK."
    # input_tokens → inputTokens, output_tokens → outputTokens (note the
    # underscore_case match — anthropic uses snake_case directly).
    assert usage == {"inputTokens": 11, "outputTokens": 5}


def test_parse_anthropic_messages_concatenates_multiple_text_blocks():
    payload = {
        "content": [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world"},
        ],
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    text, _ = _parse_anthropic_messages_response(payload)
    assert text == "Hello world"


def test_parse_anthropic_messages_raises_schema_drift_on_missing_content():
    with pytest.raises(MantleSchemaDrift):
        _parse_anthropic_messages_response({"usage": {}})


def test_parse_anthropic_messages_returns_empty_when_no_text_block():
    """Permissive parse: an anthropic content[] with only non-text blocks
    (e.g. tool_use only, or thought-only when the cap fires before text)
    yields empty text instead of raising MantleSchemaDrift. The
    permissive-discovery policy keeps the model reachable; downstream
    eval code handles empty-string responses cleanly."""
    payload = {"content": [{"type": "tool_use"}], "usage": {}}
    text, usage = _parse_anthropic_messages_response(payload)
    assert text == ""
    assert usage == {"inputTokens": 0, "outputTokens": 0}


def test_mantle_api_rejected_is_exception_subclass():
    """MantleAPIRejected is importable + is an Exception (sanity import-coverage)."""
    assert issubclass(MantleAPIRejected, Exception)
