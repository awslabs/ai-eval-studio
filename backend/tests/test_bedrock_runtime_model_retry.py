# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retry-policy tests for the platform's BedrockRuntimeModel.

DeepEval ships a ``@retry_bedrock`` on its bundled ``AmazonBedrockModel``
(see ``deepeval/models/llms/amazon_bedrock_model.py``), but our custom
subclass exists for thread-safety / temperature-guard / empty-system-
prompt reasons (see module docstring). Those reasons make us own the
retry policy too — these tests pin the marker list, attempt count, and
non-retryable behavior so a ``ThrottlingException`` storm during eval
can't silently ride straight into DeepEval's ``ignore_errors`` swallow
again.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from app.services.bedrock_runtime_model import BedrockRuntimeModel


def _throttle_error() -> ClientError:
    return ClientError(
        error_response={
            "Error": {
                "Code": "ThrottlingException",
                "Message": "Too many requests, please slow down",
            }
        },
        operation_name="Converse",
    )


def _access_denied_error() -> ClientError:
    return ClientError(
        error_response={
            "Error": {
                "Code": "AccessDeniedException",
                "Message": "User is not authorized",
            }
        },
        operation_name="Converse",
    )


def _ok_response(text: str = "ok") -> dict:
    return {
        "output": {"message": {"content": [{"text": text}]}},
        "usage": {"inputTokens": 1, "outputTokens": 1},
    }


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch):
    """Shrink retry timing so tests don't sleep real wall-time."""
    monkeypatch.setenv("BEDROCK_RETRY_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("BEDROCK_RETRY_INITIAL_S", "0.0")
    monkeypatch.setenv("BEDROCK_RETRY_CAP_S", "0.0")


def _make_model(client_mock: MagicMock) -> BedrockRuntimeModel:
    model = BedrockRuntimeModel(model_id="us.anthropic.claude-sonnet-4-5-v1:0")
    model._client = client_mock  # Bypass lazy init
    return model


class TestBedrockRetryThrottling:
    def test_retries_throttling_until_success(self):
        """Two throttles then a success — caller sees the success."""
        client = MagicMock()
        client.converse.side_effect = [
            _throttle_error(),
            _throttle_error(),
            _ok_response("third-time-lucky"),
        ]
        model = _make_model(client)

        text, _, _ = model.generate_with_system("sys", "user")

        assert text == "third-time-lucky"
        assert client.converse.call_count == 3

    def test_gives_up_after_max_attempts(self, monkeypatch):
        """All attempts throttle — last one re-raises to the caller."""
        monkeypatch.setenv("BEDROCK_RETRY_MAX_ATTEMPTS", "3")
        client = MagicMock()
        client.converse.side_effect = [_throttle_error()] * 5
        model = _make_model(client)

        with pytest.raises(ClientError):
            model.generate_with_system("sys", "user")

        # Exactly 3 attempts — not 5, even though side_effect has more.
        assert client.converse.call_count == 3

    def test_retries_on_5xx_message_markers(self):
        """ServiceUnavailable / InternalServerError / ModelTimeout markers retry."""
        for code, msg in [
            ("ServiceUnavailableException", "Service unavailable"),
            ("InternalServerException", "internal server error"),
            ("ModelTimeoutException", "model timeout"),
        ]:
            client = MagicMock()
            client.converse.side_effect = [
                ClientError(
                    error_response={"Error": {"Code": code, "Message": msg}},
                    operation_name="Converse",
                ),
                _ok_response(),
            ]
            model = _make_model(client)
            model.generate_with_system("sys", "user")
            assert client.converse.call_count == 2, f"didn't retry on {code}"


class TestBedrockNonRetryable:
    def test_access_denied_fails_immediately(self):
        """AccessDeniedException is non-retryable — surface at attempt 1."""
        client = MagicMock()
        client.converse.side_effect = _access_denied_error()
        model = _make_model(client)

        with pytest.raises(ClientError):
            model.generate_with_system("sys", "user")

        assert client.converse.call_count == 1

    def test_validation_exception_fails_immediately(self):
        """A ValidationException that ISN'T about temperature is non-retryable
        — backing off won't fix the request shape, and the response-driven
        temperature self-correction must NOT trigger here.
        """
        client = MagicMock()
        client.converse.side_effect = ClientError(
            error_response={
                "Error": {
                    "Code": "ValidationException",
                    "Message": "messages: required field is missing",
                }
            },
            operation_name="Converse",
        )
        model = _make_model(client)

        with pytest.raises(ClientError):
            model.generate_with_system("sys", "user")

        assert client.converse.call_count == 1

    def test_out_of_range_temperature_does_not_latch(self):
        """An out-of-range temperature error (caller bug) must propagate
        instead of latching the gate — silently dropping the param
        would mask the misconfiguration."""
        client = MagicMock()
        client.converse.side_effect = ClientError(
            error_response={
                "Error": {
                    "Code": "ValidationException",
                    "Message": "temperature must be between 0 and 1",
                }
            },
            operation_name="Converse",
        )
        model = _make_model(client)

        with pytest.raises(ClientError):
            model.generate_with_system("sys", "user")

        # Single call (no self-correction); gate stays True.
        assert client.converse.call_count == 1
        assert model._supports_temperature_flag is True

    def test_temperature_in_user_prompt_does_not_latch(self):
        """A ValidationException whose body merely echoes the word
        'temperature' (e.g. user prompt about weather) without saying
        the param is unsupported must propagate, not latch."""
        client = MagicMock()
        client.converse.side_effect = ClientError(
            error_response={
                "Error": {
                    "Code": "ValidationException",
                    "Message": (
                        "Input message too long: 'What is the temperature in Paris?'"
                    ),
                }
            },
            operation_name="Converse",
        )
        model = _make_model(client)

        with pytest.raises(ClientError):
            model.generate_with_system("sys", "user")

        assert client.converse.call_count == 1
        assert model._supports_temperature_flag is True

    def test_temperature_rejection_self_corrects_once(self):
        """Temperature-rejection ValidationException → drop temp,
        retry once, latch the gate so subsequent calls skip ``temperature``
        entirely. Replaces the static ``_NO_TEMPERATURE_MODELS`` list.
        """
        client = MagicMock()
        client.converse.side_effect = [
            ClientError(
                error_response={
                    "Error": {
                        "Code": "ValidationException",
                        "Message": (
                            "Malformed input request: temperature is not "
                            "supported for this model"
                        ),
                    }
                },
                operation_name="Converse",
            ),
            _ok_response(),  # Second call succeeds without temperature
        ]
        model = _make_model(client)

        text, _, _ = model.generate_with_system("sys", "user")
        assert text  # the second call succeeded
        # First call: with temperature; second call: without.
        assert client.converse.call_count == 2
        first_kwargs = client.converse.call_args_list[0].kwargs
        second_kwargs = client.converse.call_args_list[1].kwargs
        assert "temperature" in first_kwargs["inferenceConfig"]
        assert "temperature" not in second_kwargs["inferenceConfig"]

        # A subsequent generate call must skip temperature outright —
        # no second self-correction round-trip per call.
        client.converse.reset_mock()
        client.converse.side_effect = [_ok_response()]
        model.generate_with_system("sys", "user-2")
        assert client.converse.call_count == 1
        assert "temperature" not in client.converse.call_args.kwargs["inferenceConfig"]

    def test_temperature_deprecated_self_corrects_once(self):
        """'temperature is deprecated' (Opus 4.8's wording) triggers
        strip+retry under the broad rule. A phrase list that missed
        'deprecated' let every call to such a model fail with a 400, and the
        empty outputs were then scored as if the model had answered.
        """
        client = MagicMock()
        client.converse.side_effect = [
            ClientError(
                error_response={
                    "Error": {
                        "Code": "ValidationException",
                        "Message": "temperature is deprecated for this model",
                    }
                },
                operation_name="Converse",
            ),
            _ok_response("second-call-success"),
        ]
        model = _make_model(client)

        text, _, _ = model.generate_with_system("sys", "user")
        assert text == "second-call-success"
        # First call: with temperature (rejected); second call: without (succeeded).
        assert client.converse.call_count == 2
        first_kwargs = client.converse.call_args_list[0].kwargs
        second_kwargs = client.converse.call_args_list[1].kwargs
        assert "temperature" in first_kwargs["inferenceConfig"]
        assert "temperature" not in second_kwargs["inferenceConfig"]

        # Gate is latched — subsequent calls skip temperature entirely.
        client.converse.reset_mock()
        client.converse.side_effect = [_ok_response()]
        model.generate_with_system("sys", "user-2")
        assert client.converse.call_count == 1
        assert "temperature" not in client.converse.call_args.kwargs["inferenceConfig"]


class TestBedrockNetworkRetry:
    def test_retries_on_endpoint_connection_error(self):
        """Botocore network-level failure is retryable, even though it's not a ClientError."""
        client = MagicMock()
        client.converse.side_effect = [
            EndpointConnectionError(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com/"),
            _ok_response(),
        ]
        model = _make_model(client)

        model.generate_with_system("sys", "user")
        assert client.converse.call_count == 2


class TestBedrockGenerateInheritsRetry:
    def test_generate_routes_through_retried_method(self):
        """``generate()`` delegates to ``generate_with_system()`` — retry is automatic."""
        client = MagicMock()
        client.converse.side_effect = [_throttle_error(), _ok_response("via generate")]
        model = _make_model(client)

        out = model.generate("hello")

        assert out == "via generate"
        assert client.converse.call_count == 2
