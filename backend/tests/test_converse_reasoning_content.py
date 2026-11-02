# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Converse response-shape handling for reasoning-capable models.

A reasoning model returns ``output.message.content`` as an ordered,
heterogeneous list whose FIRST block is the reasoning::

    content: [{"reasoningContent": {...}}, {"text": "priority = 'High'"}]

Both adapters used to read ``content[0]["text"]``, which raises
``KeyError('text')`` on exactly that shape — surfacing in logs as a bare
``failed: 'text'``. On a reasoning target that loses most generation calls,
and a GEval judge (the gating metric) errors on every target, so the
reported pass rates mean nothing.

Why THIS is the gate and harness validation is not: both validation steps
do call Bedrock, but the smoke eval makes one judge call and the metric
self-test scores three tiny synthetic cases. None is complex enough to
make a reasoning model emit a reasoning block, so ``content[0]`` happened
to be text every time. A conditional response shape cannot be surfaced by
N=1. These fixtures pin the shape deterministically, with no AWS call.

Both copies are covered because the template copy is what runs inside
every generated project while the services copy serves the coordinator's
own calls. They are deliberately separate files (a generated harness
ships standalone and cannot import from ``app``), so only a test over
both keeps them from drifting.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.services.bedrock_runtime_model import (
    BedrockRuntimeModel,
    ConverseNoTextBlock,
    extract_converse_text,
)
from tests.harness_template_stubs import load_template_bedrock_runtime_model


def _reasoning_then_text(answer: str = "priority = 'High'") -> dict:
    """The shape observed live: reasoning block first, answer second."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"reasoningContent": {"reasoningText": {"text": "Let me think..."}}},
                    {"text": answer},
                ],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 5},
    }


def _reasoning_only() -> dict:
    """Budget exhausted on reasoning — no text block was ever emitted."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"reasoningContent": {"reasoningText": {"text": "Thinking..."}}}
                ],
            }
        },
        "stopReason": "max_tokens",
        "usage": {"inputTokens": 10, "outputTokens": 16},
    }


def _plain_text(answer: str = "ok") -> dict:
    """The non-reasoning shape that already worked — regression guard."""
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": answer}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1},
    }


# ---------------------------------------------------------------- helper ---
# Unit-level coverage of the extractor. The adapter-level tests below are
# what actually prove the call sites use it.


class TestExtractConverseText:
    """Shape-based selection in the services-side extractor."""

    def test_returns_text_after_a_reasoning_block(self):
        """The reasoning-first fixture: block 0 has no 'text', block 1 carries the answer."""
        assert (
            extract_converse_text(_reasoning_then_text(), model_id="m")
            == "priority = 'High'"
        )

    def test_returns_text_when_it_is_the_only_block(self):
        """Non-reasoning models are unaffected."""
        assert extract_converse_text(_plain_text(), model_id="m") == "ok"

    def test_returns_the_first_text_block_when_several_exist(self):
        response = {
            "output": {"message": {"content": [{"text": "first"}, {"text": "second"}]}}
        }
        assert extract_converse_text(response, model_id="m") == "first"

    def test_empty_string_text_block_is_returned_not_raised(self):
        """A model answering with nothing is NOT the same condition as a
        model emitting no text block, and it is what the old code did."""
        response = {"output": {"message": {"content": [{"text": ""}]}}}
        assert extract_converse_text(response, model_id="m") == ""

    def test_reasoning_only_raises_named_error(self):
        """No text anywhere must be named and loud — never an empty string,
        which would score as a legitimate low score and corrupt a pass rate."""
        with pytest.raises(ConverseNoTextBlock) as excinfo:
            extract_converse_text(_reasoning_only(), model_id="global.openai.gpt-5.6")
        assert not isinstance(excinfo.value, KeyError)

    def test_error_message_names_the_model_stop_reason_and_block_types(self):
        """The bare ``KeyError('text')`` is why the live logs were a mystery.
        The replacement has to say what actually came back."""
        with pytest.raises(ConverseNoTextBlock) as excinfo:
            extract_converse_text(_reasoning_only(), model_id="global.openai.gpt-5.6")
        message = str(excinfo.value)
        assert "global.openai.gpt-5.6" in message
        assert "reasoningContent" in message
        assert "max_tokens" in message

    @pytest.mark.parametrize(
        "response",
        [
            {"output": {"message": {"content": []}}},
            {"output": {"message": {}}},
            {"output": {}},
            {},
        ],
        ids=["empty-content", "no-content", "no-message", "no-output"],
    )
    def test_missing_or_empty_content_raises_named_error(self, response):
        """Malformed/truncated envelopes raise the named error too, so no
        caller has to catch KeyError/IndexError/TypeError separately."""
        with pytest.raises(ConverseNoTextBlock):
            extract_converse_text(response, model_id="m")

    def test_non_string_text_value_is_not_treated_as_text(self):
        """Shape check is ``isinstance(str)``, so a nested dict under 'text'
        (a shape we have not seen, but cheap to exclude) can't slip through
        and reach a metric as ``actual_output``."""
        response = {
            "output": {"message": {"content": [{"text": {"nested": "x"}}]}},
        }
        with pytest.raises(ConverseNoTextBlock):
            extract_converse_text(response, model_id="m")


# ------------------------------------------------- services-side adapter ---


class TestServicesAdapterReasoningContent:
    """``app.services.bedrock_runtime_model`` — the coordinator's own calls."""

    @staticmethod
    def _model_with(response: dict) -> BedrockRuntimeModel:
        model = BedrockRuntimeModel(model_id="global.openai.gpt-5.6", region="us-east-1")
        fake_client = MagicMock()
        fake_client.converse.return_value = response
        model._client = fake_client
        return model

    def test_generate_with_system_returns_text_after_reasoning(self):
        model = self._model_with(_reasoning_then_text())
        text, _, _ = model.generate_with_system("sys", "user")
        assert text == "priority = 'High'"

    def test_generate_returns_text_after_reasoning(self):
        """``generate()`` is the DeepEval judge entry point — this is the
        path that made GEval error on all three targets."""
        model = self._model_with(_reasoning_then_text())
        assert model.generate("prompt") == "priority = 'High'"

    def test_plain_text_response_still_works(self):
        model = self._model_with(_plain_text())
        text, _, _ = model.generate_with_system("sys", "user")
        assert text == "ok"

    def test_reasoning_only_raises_named_error_not_keyerror(self):
        model = self._model_with(_reasoning_only())
        with pytest.raises(ConverseNoTextBlock):
            model.generate_with_system("sys", "user")

    def test_usage_is_still_accrued_on_a_reasoning_response(self):
        """Judge-cost totals must not silently lose reasoning-model tokens —
        reasoning output tokens are billed."""
        model = self._model_with(_reasoning_then_text())
        model.generate_with_system("sys", "user")
        assert model.accumulated_usage["outputTokens"] == 5


# ------------------------------------------------- template-side adapter ---


class TestTemplateAdapterReasoningContent:
    """The copy inside every generated project — the one eval runs execute."""

    @staticmethod
    def _model_with(response: dict):
        template = load_template_bedrock_runtime_model()
        model = template.BedrockRuntimeModel(
            model_id="global.openai.gpt-5.6", region="us-east-1"
        )
        fake_client = MagicMock()
        fake_client.converse.return_value = response
        model._client = fake_client
        return template, model

    def test_generate_with_system_returns_text_after_reasoning(self):
        _, model = self._model_with(_reasoning_then_text())
        text, _, _ = model.generate_with_system("sys", "user")
        assert text == "priority = 'High'"

    def test_generate_returns_text_after_reasoning(self):
        _, model = self._model_with(_reasoning_then_text())
        assert model.generate("prompt") == "priority = 'High'"

    def test_plain_text_response_still_works(self):
        _, model = self._model_with(_plain_text())
        text, _, _ = model.generate_with_system("sys", "user")
        assert text == "ok"

    def test_reasoning_only_raises_named_error_not_keyerror(self):
        template, model = self._model_with(_reasoning_only())
        with pytest.raises(template.ConverseNoTextBlock):
            model.generate_with_system("sys", "user")

    def test_reasoning_only_from_generate_raises_named_error(self):
        template, model = self._model_with(_reasoning_only())
        with pytest.raises(template.ConverseNoTextBlock):
            model.generate("prompt")

    def test_usage_is_still_accrued_on_a_reasoning_response(self):
        _, model = self._model_with(_reasoning_then_text())
        model.generate_with_system("sys", "user")
        assert model.accumulated_usage["outputTokens"] == 5


# ----------------------------------------------------------- both copies ---


class TestBothCopiesAgree:
    """Parity guard. The two adapters are separate files by necessity, so
    nothing but a test stops a fix landing in one and not the other."""

    def test_both_expose_the_extractor_and_the_named_error(self):
        template = load_template_bedrock_runtime_model()
        assert hasattr(template, "extract_converse_text")
        assert hasattr(template, "ConverseNoTextBlock")

    def test_both_extract_the_same_text_from_the_reasoning_shape(self):
        template = load_template_bedrock_runtime_model()
        response = _reasoning_then_text()
        assert template.extract_converse_text(
            response, model_id="m"
        ) == extract_converse_text(response, model_id="m")

    def test_neither_raises_keyerror_on_the_reasoning_only_shape(self):
        """The specific regression: ``KeyError('text')`` must be gone from
        BOTH copies, not replaced in one."""
        template = load_template_bedrock_runtime_model()
        for extract, error in (
            (extract_converse_text, ConverseNoTextBlock),
            (template.extract_converse_text, template.ConverseNoTextBlock),
        ):
            with pytest.raises(error) as excinfo:
                extract(_reasoning_only(), model_id="m")
            assert not isinstance(excinfo.value, KeyError)
