# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the public GET /api/config endpoint."""
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the main app."""
    from app.main import create_app
    app = create_app()
    return TestClient(app)


# A recognisable runtime ARN, so a leak anywhere in the body is found by a
# substring search rather than only under one key.
_RUNTIME_ARN = (
    "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/config-leak-probe"
)

# The endpoint is unauthenticated, so its keys are an allowlist. Adding one
# should be a deliberate change to these sets, not a side effect.
_TOP_LEVEL_KEYS = {"auth_required", "cognito", "logout_redirect"}
_COGNITO_KEYS = {"domain", "client_id", "region", "user_pool_id"}


def _get_config(client, mode: str):
    """GET /api/config in ``mode`` with ``AGENTCORE_RUNTIME_ARN`` set."""
    with patch.dict(
        os.environ,
        {"EVAL_STUDIO_MODE": mode, "AGENTCORE_RUNTIME_ARN": _RUNTIME_ARN},
    ):
        response = client.get("/api/config")
    assert response.status_code == 200
    return response


class TestConfigEndpoint:
    """Test the /api/config bootstrap endpoint."""

    @pytest.mark.parametrize("mode", ["dev", "cloud"])
    def test_runtime_arn_is_not_served(self, client, mode):
        """The runtime ARN never reaches this public endpoint, even when the
        coordinator has it: it names the account and the runtime, and the SPA
        has no use for it."""
        response = _get_config(client, mode)

        assert "agentcore" not in response.json()
        assert _RUNTIME_ARN not in response.text

    @pytest.mark.parametrize("mode", ["dev", "cloud"])
    def test_top_level_keys_are_exactly_the_allowlist(self, client, mode):
        """The response carries exactly the keys the SPA needs to sign in."""
        assert set(_get_config(client, mode).json()) == _TOP_LEVEL_KEYS

    def test_cognito_keys_are_exactly_the_allowlist(self, client):
        """In cloud mode the cognito block carries exactly the OIDC pieces."""
        cognito = _get_config(client, "cloud").json()["cognito"]
        assert set(cognito) == _COGNITO_KEYS

    def test_dev_mode_has_no_cognito_block(self, client):
        """Dev mode signs no one in, so there is no cognito block."""
        assert _get_config(client, "dev").json()["cognito"] is None


class TestHealthEndpoint:
    """Test GET /api/health on the real app."""

    def test_health_is_registered_on_the_real_app(self, client):
        """``/api/health`` answers 200 with ``{"status": "ok"}``.

        The ALB target group polls this path (``network_stack.py``), so an
        unregistered route means every task fails its health check and the
        service never comes up. Nothing else asserted it against the app that
        ``create_app`` actually builds: the route is declared inline inside
        ``create_app`` rather than in a router module, and the one other test
        naming this path declares its OWN ``/api/health`` on a synthetic app to
        check middleware short-circuiting, which would keep passing if the real
        route disappeared.
        """
        response = client.get("/api/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
