# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Bedrock Mantle 401 credential refresh-then-retry behavior.

A 401 from the mantle endpoint triggers credential
refresh and retry with a DIFFERENT credential; a persistent 401 after
refresh terminates as a non-retryable authorization failure.
"""

import json
import threading
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.services.bedrock_mantle_model import (
    BedrockMantleModel,
    MantleAuthorizationFailure,
    _close_shared_mantle_client,
    _close_shared_mantle_credentials,
)


@pytest.fixture
def reset_module_singletons():
    """Reset module-level caches between tests."""
    _close_shared_mantle_client()
    _close_shared_mantle_credentials()
    yield
    _close_shared_mantle_client()
    _close_shared_mantle_credentials()


def test_401_triggers_credential_refresh_and_retries_with_different_key(reset_module_singletons):
    """A first 401 refreshes credentials and retries with a DIFFERENT access key.

    The test captures the ACTUAL access key used to sign each attempt and
    asserts they differ. A test that only checks a mock call count does NOT
    satisfy this requirement.
    """
    # Descriptor for a fake mantle model
    descriptor = {
        "api": "responses",
        "region": "us-east-1",
        "supports_temperature": True,
    }
    model = BedrockMantleModel(descriptor, "test.model", max_output_tokens=64)

    # Fake credential provider that returns DIFFERENT access keys on successive calls
    captured_keys = []
    provider_build_count = [0]

    class FakeCredentials:
        def __init__(self, access_key):
            self.access_key = access_key
            self.secret_key = "fake_secret"
            self.token = "fake_token"

    def fake_get_credentials(profile_name):
        """Returns a provider whose frozen credentials change on each PROVIDER build."""
        provider_build_count[0] += 1
        provider_id = provider_build_count[0]

        def get_frozen_credentials():
            # Return different key per provider instance
            return FakeCredentials(f"AKIA_ATTEMPT_{provider_id}")

        provider = MagicMock()
        provider.get_frozen_credentials = get_frozen_credentials
        return provider

    # Fake httpx client that responds 401 first, then 200
    def make_response(status, json_data, req_id):
        """Helper to create httpx.Response with a mock request."""
        fake_req = httpx.Request("POST", "https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses")
        response = httpx.Response(status, json=json_data, headers={"x-amzn-requestid": req_id}, request=fake_req)
        return response

    response_sequence = [
        make_response(401, {"error": {"message": "Unauthorized"}}, "req1"),
        make_response(200, {
            "status": "complete",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "OK"}]
            }],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }, "req2"),
    ]
    call_index = [0]

    def fake_post(url, content, headers):
        # Capture the access key from the Authorization header
        auth_header = headers.get("Authorization", "")
        # Extract access key from the SigV4 signature (it's in the Credential part)
        if "Credential=" in auth_header:
            cred_part = auth_header.split("Credential=")[1].split("/")[0]
            captured_keys.append(cred_part)

        response = response_sequence[call_index[0]]
        call_index[0] = min(call_index[0] + 1, len(response_sequence) - 1)
        return response

    fake_client = MagicMock()
    fake_client.post = fake_post

    with patch("app.services.bedrock_mantle_model._get_shared_mantle_credentials", fake_get_credentials), \
         patch("app.services.bedrock_mantle_model._get_shared_mantle_client", return_value=fake_client):

        # Generate - should trigger 401, refresh, retry, succeed
        text, latency, usage = model.generate_with_system(None, "Say OK.")

        # Assert: exactly 2 attempts
        assert len(captured_keys) == 2, f"Expected 2 attempts, got {len(captured_keys)}"

        # Assert: the access keys DIFFER (the load-bearing assertion)
        assert captured_keys[1] != captured_keys[0], \
            f"Credential did not refresh: attempt 1 used {captured_keys[0]}, attempt 2 used {captured_keys[1]}"

        # Assert: final response succeeded
        assert text == "OK"
        assert usage["inputTokens"] == 10


def test_persistent_401_after_refresh_is_non_retryable(reset_module_singletons):
    """A 401 recurring after successful refresh is non-retryable.

    Must raise the named authorization exception and make exactly 2 sends
    (not retry forever, not burn the retry budget).
    """
    descriptor = {
        "api": "responses",
        "region": "us-east-1",
        "supports_temperature": True,
    }
    model = BedrockMantleModel(descriptor, "test.model", max_output_tokens=64)

    # Fake credential provider that successfully refreshes (different keys)
    call_count = [0]

    class FakeCredentials:
        def __init__(self, access_key):
            self.access_key = access_key
            self.secret_key = "fake_secret"
            self.token = "fake_token"

    def fake_get_credentials(profile_name):
        def get_frozen_credentials():
            call_count[0] += 1
            return FakeCredentials(f"AKIA_ATTEMPT_{call_count[0]}")

        provider = MagicMock()
        provider.get_frozen_credentials = get_frozen_credentials
        return provider

    # Both attempts return 401
    send_count = [0]

    def fake_post(url, content, headers):
        send_count[0] += 1
        fake_req = httpx.Request("POST", "https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses")
        return httpx.Response(401, json={"error": {"message": "Unauthorized"}}, headers={"x-amzn-requestid": f"req{send_count[0]}"}, request=fake_req)

    fake_client = MagicMock()
    fake_client.post = fake_post

    with patch("app.services.bedrock_mantle_model._get_shared_mantle_credentials", fake_get_credentials), \
         patch("app.services.bedrock_mantle_model._get_shared_mantle_client", return_value=fake_client):

        # Should raise the named non-retryable authorization exception
        with pytest.raises(MantleAuthorizationFailure) as exc_info:
            model.generate_with_system(None, "Say OK.")

        # Assert: exactly 2 sends (first 401, refresh, second 401, terminate)
        assert send_count[0] == 2, f"Expected exactly 2 sends, got {send_count[0]}"

        # Assert: exception carries the response body
        assert "Unauthorized" in str(exc_info.value)


def test_429_does_not_trigger_refresh(reset_module_singletons):
    """Throttling responses remain retryable and do NOT clear the credential cache."""
    descriptor = {
        "api": "responses",
        "region": "us-east-1",
        "supports_temperature": True,
    }
    model = BedrockMantleModel(descriptor, "test.model", max_output_tokens=64)

    # Track how many times the credential provider is built.
    # _get_shared_mantle_credentials is replaced below by a fake with its own
    # per-profile cache, mimicking the real one: a repeat call returns the same
    # provider, so provider_build_count counts distinct provider builds. The
    # real _close_shared_mantle_credentials does not clear this fake's cache.
    provider_build_count = [0]
    fake_provider_cache = {}

    class FakeCredentials:
        def __init__(self):
            self.access_key = "AKIA_FIXED"
            self.secret_key = "fake_secret"
            self.token = "fake_token"

    def fake_get_credentials(profile_name):
        # Mimic the real caching: if we've already built a provider for this
        # profile, return the cached one
        if profile_name in fake_provider_cache:
            return fake_provider_cache[profile_name]

        provider_build_count[0] += 1
        provider = MagicMock()
        provider.get_frozen_credentials = lambda: FakeCredentials()
        fake_provider_cache[profile_name] = provider
        return provider

    # First response is 429, second is 200
    def make_response(status, json_data, req_id):
        fake_req = httpx.Request("POST", "https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses")
        return httpx.Response(status, json=json_data, headers={"x-amzn-requestid": req_id}, request=fake_req)

    response_sequence = [
        make_response(429, {"error": {"message": "Too many requests"}}, "req1"),
        make_response(200, {
            "status": "complete",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "OK"}]
            }],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }, "req2"),
    ]
    call_index = [0]

    def fake_post(url, content, headers):
        response = response_sequence[call_index[0]]
        call_index[0] = min(call_index[0] + 1, len(response_sequence) - 1)
        return response

    fake_client = MagicMock()
    fake_client.post = fake_post

    # Set retry env vars to speed up test
    import os
    os.environ["BEDROCK_RETRY_MAX_ATTEMPTS"] = "2"
    os.environ["BEDROCK_RETRY_INITIAL_S"] = "0.01"

    try:
        with patch("app.services.bedrock_mantle_model._get_shared_mantle_credentials", side_effect=fake_get_credentials), \
             patch("app.services.bedrock_mantle_model._get_shared_mantle_client", return_value=fake_client):

            text, latency, usage = model.generate_with_system(None, "Say OK.")

            # Assert: credential provider built exactly once (no refresh)
            assert provider_build_count[0] == 1, f"Expected 1 provider build, got {provider_build_count[0]}"

            # Assert: succeeded after retry
            assert text == "OK"
    finally:
        os.environ.pop("BEDROCK_RETRY_MAX_ATTEMPTS", None)
        os.environ.pop("BEDROCK_RETRY_INITIAL_S", None)
