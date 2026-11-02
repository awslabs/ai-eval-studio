# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the backend test suite.

Includes Cognito-bearer JWT fixtures so every test that needs an
authenticated request or WebSocket can sign a real RS256 token without
standing up a real Cognito user pool.
"""
from __future__ import annotations

import contextlib

from typing import Any

import pytest

from tests.fixtures.jwt_fixtures import (
    generate_rsa_keypair,
    public_key_to_jwks_dict,
    sign_access_token,
)


# Common Cognito environment values reused by every JWT test.
_TEST_REGION = "us-east-1"
_TEST_USER_POOL_ID = "us-east-1_TESTPOOL"
_TEST_SPA_CLIENT_ID = "test-spa-client-id"
_TEST_KID = "test-kid-1"
_TEST_SUB = "test-user-sub-123"
_TEST_ISSUER = (
    f"https://cognito-idp.{_TEST_REGION}.amazonaws.com/{_TEST_USER_POOL_ID}"
)


@pytest.fixture(autouse=True)
def _isolate_aws_credentials(monkeypatch):
    """Scrub the AWS credential resolution chain for every test.

    Without this fixture, tests that instantiate ``boto3.client(...)``
    inherit whatever the developer (or CI runner) has in
    ``~/.aws/config``, ``~/.aws/credentials``, or the ``AWS_PROFILE``
    env var. That works on the implementer's machine but fails
    catastrophically on any other machine — a missing or differently-
    named profile raises ``ProfileNotFound`` before the test code
    even runs, and a *valid* profile would let the tests touch real
    AWS resources.

    Pin every relevant env var to dummy values and point the config
    files at ``/dev/null`` so ``botocore`` resolves a synthetic
    "testing" identity regardless of the host. Tests that need a
    specific stub override these with their own ``Stubber``.
    """
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/dev/null")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/dev/null")


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset Platform and JobTracker singletons before and after each test.

    Cross-file test-isolation bug: tests that patch ``get_platform()``
    or interact with JobTracker can leave global singleton state polluted.
    Subsequent tests then get stale or mocked instances instead of fresh
    bootstraps, causing spurious failures.

    This fixture explicitly clears both singletons before and after each
    test so every test starts with a clean slate.
    """
    from app.platform import set_platform
    from app.services.jobs import JobTracker

    # Reset Platform singleton
    set_platform(None)

    # Reset JobTracker singleton
    JobTracker._instance = None

    yield

    # Cleanup after test
    set_platform(None)
    JobTracker._instance = None


@pytest.fixture
def projects_dir(tmp_path):
    """Provide a temporary projects directory."""
    d = tmp_path / "projects"
    d.mkdir()
    return d


# --- Cognito JWT fixtures --------------------------------------------


@pytest.fixture(scope="session")
def rsa_keypair() -> tuple[bytes, bytes]:
    """Generate exactly one RSA-2048 keypair per test session.

    Session-scoped because keygen is the expensive part of the JWT setup;
    the same keypair signs every token across the suite without affecting
    test independence (each token is signed for its own scenario).

    Returns:
        ``(private_pem, public_pem)`` PEM-encoded keypair.
    """
    return generate_rsa_keypair()


@pytest.fixture(scope="session")
def cognito_jwks_dict(rsa_keypair: tuple[bytes, bytes]) -> dict[str, Any]:
    """JWKS document built from :data:`rsa_keypair`'s public half.

    Loaded into :class:`PyJWKClient.jwk_set_cache` by
    :func:`cognito_jwks_inject` so the validator never makes a network
    call to fetch JWKS during tests.
    """
    _, public_pem = rsa_keypair
    return public_key_to_jwks_dict(public_pem, kid=_TEST_KID)


@pytest.fixture
def cognito_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set the Cognito-related env vars the validator + cloud bootstrap read.

    The validator (``CognitoBearerJwtAuth``) reads the values directly via
    constructor args, so unit tests that instantiate the class don't need
    these env vars at all. They matter for the cloud-mode platform path
    (``EVAL_STUDIO_MODE=cloud`` triggers ``build_cloud_platform`` which
    requires several other vars too — those tests instantiate the class
    directly to bypass that).

    Returns:
        Dict of the env values set, for tests that want to assert against
        them or build matching tokens.
    """
    monkeypatch.setenv("EVAL_STUDIO_MODE", "cloud")
    monkeypatch.setenv("COGNITO_REGION", _TEST_REGION)
    monkeypatch.setenv("COGNITO_USER_POOL_ID", _TEST_USER_POOL_ID)
    monkeypatch.setenv("COGNITO_SPA_CLIENT_ID", _TEST_SPA_CLIENT_ID)
    return {
        "region": _TEST_REGION,
        "user_pool_id": _TEST_USER_POOL_ID,
        "spa_client_id": _TEST_SPA_CLIENT_ID,
        "issuer": _TEST_ISSUER,
        "kid": _TEST_KID,
        "sub": _TEST_SUB,
    }


@pytest.fixture
def cognito_jwks_inject(
    monkeypatch: pytest.MonkeyPatch,
    cognito_jwks_dict: dict[str, Any],
):
    """Patch ``_get_jwks_client`` so it returns a pre-populated client.

    ``CognitoBearerJwtAuth._validate_token`` calls
    ``_get_jwks_client(self._jwks_url)`` and then
    ``client.get_signing_key_from_jwt(token)``. The latter walks
    ``get_jwk_set()`` which checks ``jwk_set_cache`` first; if a cached
    set is present, no HTTP fetch happens.

    This fixture builds a real :class:`PyJWKClient` (so all the cache /
    refresh / kid-matching logic runs untouched) and pre-populates its
    cache with the test JWKS. The result: tokens get verified against our
    keypair as if Cognito had served them, but no socket is opened.

    Yields the patched factory function so tests can introspect call
    counts when needed (e.g. cache-reuse assertions).
    """
    from jwt import PyJWKClient

    import app.platform.aws.auth_cognito_bearer as auth_module

    # Reset PyJWT's lru_cache so test-to-test state doesn't leak.
    auth_module._get_jwks_client.cache_clear()

    call_log: list[str] = []

    def _factory(jwks_url: str) -> PyJWKClient:
        call_log.append(jwks_url)
        client = PyJWKClient(jwks_url, lifespan=86400)
        # Pre-populate the cache: PyJWT's ``fetch_data`` puts the raw
        # JSON dict into ``jwk_set_cache`` (despite the type hints), and
        # ``get_jwk_set`` re-checks ``isinstance(data, dict)`` and rebuilds
        # ``PyJWKSet`` from the dict on every access. So we mirror that
        # contract — store the dict, not the PyJWKSet — to avoid the
        # "The JWKS endpoint did not return a JSON object" error.
        client.jwk_set_cache.put(cognito_jwks_dict)
        return client

    monkeypatch.setattr(auth_module, "_get_jwks_client", _factory)
    yield call_log
    auth_module._get_jwks_client.cache_clear() if hasattr(
        auth_module._get_jwks_client, "cache_clear"
    ) else None


@pytest.fixture
def valid_access_token(rsa_keypair: tuple[bytes, bytes]) -> str:
    """Sign a Cognito-shaped access token with valid claims."""
    private_pem, _ = rsa_keypair
    return sign_access_token(
        private_pem,
        kid=_TEST_KID,
        sub=_TEST_SUB,
        iss=_TEST_ISSUER,
        client_id=_TEST_SPA_CLIENT_ID,
    )


@pytest.fixture
def expired_access_token(rsa_keypair: tuple[bytes, bytes]) -> str:
    """Sign an access token whose ``exp`` is 60 seconds in the past."""
    private_pem, _ = rsa_keypair
    return sign_access_token(
        private_pem,
        kid=_TEST_KID,
        sub=_TEST_SUB,
        iss=_TEST_ISSUER,
        client_id=_TEST_SPA_CLIENT_ID,
        exp_offset_sec=-60,
    )


@pytest.fixture
def wrong_iss_token(rsa_keypair: tuple[bytes, bytes]) -> str:
    """Sign an access token whose ``iss`` points at a different user pool."""
    private_pem, _ = rsa_keypair
    return sign_access_token(
        private_pem,
        kid=_TEST_KID,
        sub=_TEST_SUB,
        iss=f"https://cognito-idp.{_TEST_REGION}.amazonaws.com/us-east-1_OTHERPOOL",
        client_id=_TEST_SPA_CLIENT_ID,
    )


@pytest.fixture
def id_token(rsa_keypair: tuple[bytes, bytes]) -> str:
    """Sign an id-token (``token_use="id"``) — the validator must reject."""
    private_pem, _ = rsa_keypair
    return sign_access_token(
        private_pem,
        kid=_TEST_KID,
        sub=_TEST_SUB,
        iss=_TEST_ISSUER,
        client_id=_TEST_SPA_CLIENT_ID,
        token_use="id",
    )


@pytest.fixture
def wrong_client_id_token(rsa_keypair: tuple[bytes, bytes]) -> str:
    """Sign an access token whose ``client_id`` claim doesn't match."""
    private_pem, _ = rsa_keypair
    return sign_access_token(
        private_pem,
        kid=_TEST_KID,
        sub=_TEST_SUB,
        iss=_TEST_ISSUER,
        client_id="some-other-spa-client",
    )


@pytest.fixture
def bad_signature_token() -> str:
    """Sign a token with a *different* keypair so signature verification fails.

    The header still claims ``kid=_TEST_KID`` so the JWKS lookup succeeds
    (the test keypair's public key is found by kid). Only the signature
    verification fails — exactly what we want to assert.
    """
    other_private_pem, _ = generate_rsa_keypair()
    return sign_access_token(
        other_private_pem,
        kid=_TEST_KID,
        sub=_TEST_SUB,
        iss=_TEST_ISSUER,
        client_id=_TEST_SPA_CLIENT_ID,
    )


def iter_api_routes(app):
    """Yield every ``APIRoute`` reachable from ``app``, following includes.

    ``app.routes`` used to hold the flattened result of every
    ``include_router`` call. FastAPI 0.141 changed that: each call becomes one
    opaque ``_IncludedRouter`` entry that exposes no ``path`` and no ``routes``,
    and keeps the real routes on its ``original_router``. A flat scan therefore
    sees only routes declared with ``@app.<method>`` directly — which is why
    every route-introspecting authz test in this suite started failing at once
    when the framework was upgraded.

    Those tests exist to prove ``require_project_read`` / ``require_project_write``
    is attached to each route, so the scan has to keep working rather than be
    dropped. Walking ``original_router`` is what restores it. The private name
    is unavoidable: there is no public accessor for an included router's routes.

    Args:
        app: The FastAPI application to walk.

    Yields:
        Each ``APIRoute``, including those reached through an include.
    """
    from fastapi.routing import APIRoute

    def _walk(routes):
        for route in routes:
            if isinstance(route, APIRoute):
                yield route
            # ``_IncludedRouter`` (0.141+) keeps its children here. ``routes`` is
            # the pre-0.141 shape and a plain ``APIRouter``'s shape; both are
            # checked so this helper does not itself depend on the version.
            nested = getattr(route, "original_router", None) or route
            child_routes = getattr(nested, "routes", None)
            if child_routes and child_routes is not routes:
                yield from _walk(child_routes)

    yield from _walk(app.routes)


@contextlib.contextmanager
def authed_ws(client, token: str = "", path: str = "/ws"):
    """Open a WebSocket and complete the first-frame auth handshake.

    The credential travels in the first frame rather than the query string, so a
    test that connects and immediately inspects the hub would race the server:
    registration happens only after that frame is read. Waiting for the
    ``auth_ok`` acknowledgement makes the handshake a synchronisation point, so
    assertions after this yields are ordered behind registration.

    Args:
        client: A ``TestClient``.
        token: Token to present. Defaults to ``""``, which is exactly what the
            browser sends in dev mode (``getAccessToken`` returns it when no
            login is configured), so every dev-mode socket test exercises the
            real client's frame. A non-empty placeholder here once hid a server
            check that rejected every dev socket.
        path: Endpoint path, for the rare test that varies it.

    Yields:
        The connected websocket, authenticated and registered.
    """
    with client.websocket_connect(path) as websocket:
        websocket.send_json({"action": "auth", "token": token})
        ack = websocket.receive_json()
        assert ack.get("kind") == "auth_ok", f"handshake did not complete: {ack}"
        yield websocket
