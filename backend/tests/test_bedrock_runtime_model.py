# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Bedrock model wrapper (mocked — no AWS calls)."""

import pytest
from unittest.mock import MagicMock, patch
from app.services.bedrock_runtime_model import BedrockRuntimeModel


class TestBedrockRuntimeModel:
    def test_get_model_name(self):
        model = BedrockRuntimeModel(model_id="us.anthropic.claude-sonnet-4-6")
        assert model.get_model_name() == "us.anthropic.claude-sonnet-4-6"

    def test_get_model_name_custom(self):
        model = BedrockRuntimeModel(model_id="us.amazon.nova-lite-v1:0")
        assert model.get_model_name() == "us.amazon.nova-lite-v1:0"

    @patch("app.services.bedrock_runtime_model.boto3.Session")
    def test_load_model_creates_client(self, mock_session_cls):
        from botocore.config import Config
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        model = BedrockRuntimeModel(model_id="test-model", region="us-west-2")
        client = model.load_model()
        mock_session_cls.assert_called_once_with(profile_name=None)
        # ``config=`` carries the read_timeout override that lets large
        # Converse calls run past botocore's default 60s. Pin the
        # contract: client() is called with the right region AND a
        # Config whose read_timeout is the env-tunable default.
        assert mock_session.client.call_count == 1
        args, kwargs = mock_session.client.call_args
        assert args == ("bedrock-runtime",)
        assert kwargs["region_name"] == "us-west-2"
        cfg = kwargs["config"]
        assert isinstance(cfg, Config)
        assert cfg.read_timeout == 300
        assert cfg.connect_timeout == 10

    @patch("app.services.bedrock_runtime_model.boto3.Session")
    def test_load_model_honors_read_timeout_env(self, mock_session_cls, monkeypatch):
        from botocore.config import Config
        monkeypatch.setenv("BEDROCK_READ_TIMEOUT_S", "120")
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        model = BedrockRuntimeModel(model_id="test-model")
        model.load_model()
        cfg = mock_session.client.call_args.kwargs["config"]
        assert isinstance(cfg, Config)
        assert cfg.read_timeout == 120

    @patch("app.services.bedrock_runtime_model.boto3.Session")
    def test_load_model_caches_client(self, mock_session_cls):
        model = BedrockRuntimeModel(model_id="test-model")
        client1 = model.load_model()
        client2 = model.load_model()
        assert mock_session_cls.call_count == 1

    @patch("app.services.bedrock_runtime_model.boto3.Session")
    def test_load_model_with_profile(self, mock_session_cls):
        """Verify profile_name is passed through to boto3.Session."""
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        model = BedrockRuntimeModel(model_id="test-model", profile_name="my-profile")
        model.load_model()
        mock_session_cls.assert_called_once_with(profile_name="my-profile")

    @patch("app.services.bedrock_runtime_model.boto3.Session")
    def test_generate_calls_converse(self, mock_session_cls):
        mock_client = MagicMock()
        mock_session_cls.return_value.client.return_value = mock_client
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "Hello!"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
            "metrics": {"latencyMs": 250},
        }

        model = BedrockRuntimeModel(model_id="test-model")
        result = model.generate(prompt="Say hello")

        assert result == "Hello!"
        mock_client.converse.assert_called_once()

    @patch("app.services.bedrock_runtime_model.boto3.Session")
    def test_generate_with_system_prompt(self, mock_session_cls):
        mock_client = MagicMock()
        mock_session_cls.return_value.client.return_value = mock_client
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "Response"}]}},
            "usage": {"inputTokens": 20, "outputTokens": 10, "totalTokens": 30},
            "metrics": {"latencyMs": 300},
        }

        model = BedrockRuntimeModel(model_id="test-model")
        text, latency_ms, usage = model.generate_with_system(
            system_prompt="You are helpful.",
            user_prompt="Hello",
        )

        assert text == "Response"
        assert latency_ms > 0
        assert usage["inputTokens"] == 20
        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["system"] == [{"text": "You are helpful."}]

    @patch("app.services.bedrock_runtime_model.boto3.Session")
    def test_temperature_stripped_when_bedrock_rejects(self, mock_session_cls):
        """Temperature handling is response-driven — the first
        call attempts with the param; if Bedrock returns a
        ValidationException mentioning temperature, the adapter latches
        the gate and retries (and every subsequent call) without it.
        Replaces the old static-list reasoning-model gate.
        """
        from botocore.exceptions import ClientError

        mock_client = MagicMock()
        mock_session_cls.return_value.client.return_value = mock_client

        ok = {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {},
        }
        temp_reject = ClientError(
            error_response={
                "Error": {
                    "Code": "ValidationException",
                    "Message": "Malformed input request: temperature is not supported.",
                }
            },
            operation_name="Converse",
        )

        for reasoning_id in (
            "us.anthropic.claude-sonnet-4-6",
            "us.anthropic.claude-opus-4-7",
            "us.anthropic.claude-opus-4-8",  # Also Opus 4.8 target
        ):
            mock_client.converse.reset_mock()
            mock_client.converse.side_effect = [temp_reject, ok]
            model = BedrockRuntimeModel(model_id=reasoning_id, temperature=0.5)
            model.generate_with_system(system_prompt="sys", user_prompt="hi")
            # Two calls: rejected (with temp), then self-corrected (without).
            assert mock_client.converse.call_count == 2
            second_kwargs = mock_client.converse.call_args_list[1][1]
            assert "temperature" not in second_kwargs["inferenceConfig"], (
                f"{reasoning_id} must drop temperature after self-correction"
            )

    @patch("app.services.bedrock_runtime_model.boto3.Session")
    def test_temperature_passed_for_regular_models(self, mock_session_cls):
        """Non-reasoning models receive the configured temperature."""
        mock_client = MagicMock()
        mock_session_cls.return_value.client.return_value = mock_client
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {},
        }

        model = BedrockRuntimeModel(
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            temperature=0.7,
        )
        model.generate_with_system(system_prompt="sys", user_prompt="hi")
        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["inferenceConfig"]["temperature"] == 0.7

    @patch("app.services.bedrock_runtime_model.boto3.Session")
    def test_fresh_model_has_zero_usage(self, mock_session_cls):
        model = BedrockRuntimeModel(model_id="test-model")
        usage = model.accumulated_usage
        assert usage["inputTokens"] == 0
        assert usage["outputTokens"] == 0

    @patch("app.services.bedrock_runtime_model.boto3.Session")
    def test_generate_with_system_accrues_usage(self, mock_session_cls):
        mock_client = MagicMock()
        mock_session_cls.return_value.client.return_value = mock_client
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {"inputTokens": 20, "outputTokens": 10, "totalTokens": 30},
        }
        model = BedrockRuntimeModel(model_id="test-model")
        model.generate_with_system(system_prompt="sys", user_prompt="hi")
        usage = model.accumulated_usage
        assert usage["inputTokens"] == 20
        assert usage["outputTokens"] == 10

    @patch("app.services.bedrock_runtime_model.boto3.Session")
    def test_generate_accrues_usage_exactly_once(self, mock_session_cls):
        """Service ``generate()`` delegates to ``generate_with_system()``;
        usage must be accrued once per call, not double-counted."""
        mock_client = MagicMock()
        mock_session_cls.return_value.client.return_value = mock_client
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {"inputTokens": 7, "outputTokens": 3},
        }
        model = BedrockRuntimeModel(model_id="test-model")
        model.generate(prompt="hi")
        usage = model.accumulated_usage
        assert usage["inputTokens"] == 7
        assert usage["outputTokens"] == 3

    @patch("app.services.bedrock_runtime_model.boto3.Session")
    def test_multiple_calls_sum(self, mock_session_cls):
        mock_client = MagicMock()
        mock_session_cls.return_value.client.return_value = mock_client
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {"inputTokens": 5, "outputTokens": 2},
        }
        model = BedrockRuntimeModel(model_id="test-model")
        model.generate(prompt="a")
        model.generate_with_system(system_prompt="sys", user_prompt="b")
        usage = model.accumulated_usage
        assert usage["inputTokens"] == 10
        assert usage["outputTokens"] == 4

    @patch("app.services.bedrock_runtime_model.boto3.Session")
    def test_missing_usage_key_does_not_crash(self, mock_session_cls):
        mock_client = MagicMock()
        mock_session_cls.return_value.client.return_value = mock_client
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "ok"}]}},
        }
        model = BedrockRuntimeModel(model_id="test-model")
        text = model.generate(prompt="hi")
        assert text == "ok"
        assert model.accumulated_usage["inputTokens"] == 0

    @patch("app.services.bedrock_runtime_model.boto3.Session")
    def test_empty_system_prompt_is_not_forwarded(self, mock_session_cls):
        """An empty system prompt must not reach Bedrock (Converse rejects it)."""
        mock_client = MagicMock()
        mock_session_cls.return_value.client.return_value = mock_client
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {},
        }

        model = BedrockRuntimeModel(model_id="test-model")
        # Empty, whitespace-only, and None all treated as "no system".
        for empty in ("", "   \n", None):
            model.generate_with_system(system_prompt=empty, user_prompt="hi")
            call_kwargs = mock_client.converse.call_args[1]
            assert "system" not in call_kwargs, (
                f"Empty system prompt ({empty!r}) should not be forwarded"
            )
