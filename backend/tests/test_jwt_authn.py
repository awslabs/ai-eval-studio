# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the in-app Cognito-bearer JWT validator + middleware.

Covers the seven validation steps the validator enforces, plus the
FastAPI middleware integration paths:

1. Signature against Cognito JWKS (RS256) — the only allowed alg.
2. ``iss`` claim matches the configured user pool.
3. ``token_use=access`` (id-tokens rejected).
4. ``client_id`` claim matches the configured SPA client.
5. ``exp`` not in the past.
6. Bearer header parsed correctly (missing / malformed → 401).
7. Dev mode bypasses JWT validation entirely.

JWKS is served from an in-process pre-populated :class:`PyJWKClient`
cache (see ``tests.conftest.cognito_jwks_inject``) so no network calls
happen during the suite.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.platform.auth import auth_middleware, get_local_dev_user_id
from app.platform.aws.auth_cognito_bearer import CognitoBearerJwtAuth
from app.platform.types import current_user_id


def _make_validator(cognito_env) -> CognitoBearerJwtAuth:
    """Construct a validator wired to the test Cognito env values."""
    # MetadataStore isn't exercised by extract_user_id paths — pass a sentinel.
    return CognitoBearerJwtAuth(
        metadata_store=None,  # type: ignore[arg-type]
        region=cognito_env["region"],
        user_pool_id=cognito_env["user_pool_id"],
        client_id=cognito_env["spa_client_id"],
    )


# --- TestValidateToken: direct unit tests -------------------------------


class TestValidateToken:
    """Direct calls into ``CognitoBearerJwtAuth._validate_token``.

    Faster than the middleware path; exercises the full real-PyJWT
    decode → JWKS → claims-check chain.
    """

    def test_valid_access_token_returns_claims_with_sub(
        self, cognito_env, cognito_jwks_inject, valid_access_token
    ):
        validator = _make_validator(cognito_env)
        claims = validator._validate_token(valid_access_token)
        assert claims is not None
        assert claims["sub"] == cognito_env["sub"]
        assert claims["token_use"] == "access"
        assert claims["client_id"] == cognito_env["spa_client_id"]

    def test_expired_token_returns_none(
        self, cognito_env, cognito_jwks_inject, expired_access_token
    ):
        validator = _make_validator(cognito_env)
        assert validator._validate_token(expired_access_token) is None

    def test_wrong_iss_returns_none(
        self, cognito_env, cognito_jwks_inject, wrong_iss_token
    ):
        validator = _make_validator(cognito_env)
        assert validator._validate_token(wrong_iss_token) is None

    def test_token_use_id_returns_none(
        self, cognito_env, cognito_jwks_inject, id_token
    ):
        validator = _make_validator(cognito_env)
        assert validator._validate_token(id_token) is None

    def test_wrong_client_id_returns_none(
        self, cognito_env, cognito_jwks_inject, wrong_client_id_token
    ):
        validator = _make_validator(cognito_env)
        assert validator._validate_token(wrong_client_id_token) is None

    def test_bad_signature_returns_none(
        self, cognito_env, cognito_jwks_inject, bad_signature_token
    ):
        validator = _make_validator(cognito_env)
        assert validator._validate_token(bad_signature_token) is None

    def test_malformed_token_returns_none(self, cognito_env, cognito_jwks_inject):
        validator = _make_validator(cognito_env)
        assert validator._validate_token("not.a.jwt") is None
        assert validator._validate_token("") is None

    def test_deeply_nested_payload_returns_none(self, cognito_env, cognito_jwks_inject):
        """An unsigned token whose payload nests past the recursion limit is
        rejected, not raised. Looking up the signing key parses the payload
        before any signature check, and PyJWT before 2.15.0 let the parser's
        RecursionError escape, so one unauthenticated header failed the request.
        """
        import base64
        import json
        import sys

        def b64(raw: bytes) -> str:
            return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

        depth = sys.getrecursionlimit() + 100
        header = b64(json.dumps({"alg": "RS256", "typ": "JWT", "kid": "k"}).encode())
        payload = b64(("[" * depth + "]" * depth).encode())
        token = f"{header}.{payload}.{b64(b'not-a-signature')}"

        validator = _make_validator(cognito_env)
        assert validator._validate_token(token) is None

    def test_jwks_factory_called_per_unique_url(
        self, cognito_env, cognito_jwks_inject, valid_access_token
    ):
        """Validation reaches the JWKS factory with a ``/.well-known/jwks.json`` URL.

        Asserts only that the factory was called at least once and that the
        URL it received ends in ``/.well-known/jwks.json``. It cannot observe
        client sharing: in production ``_get_jwks_client`` is wrapped in
        ``lru_cache``, but ``cognito_jwks_inject`` replaces it with an uncached
        factory that logs every call, so both validations below reach it.
        """
        validator = _make_validator(cognito_env)
        validator._validate_token(valid_access_token)
        validator._validate_token(valid_access_token)
        assert len(cognito_jwks_inject) >= 1
        assert cognito_jwks_inject[0].endswith("/.well-known/jwks.json")


# --- TestExtractUserId: HTTP entrypoint --------------------------------


class TestExtractUserId:
    """``CognitoBearerJwtAuth.extract_user_id`` — HTTP request path."""

    @pytest.mark.asyncio
    async def test_valid_bearer_returns_sub(
        self, cognito_env, cognito_jwks_inject, valid_access_token
    ):
        from starlette.requests import Request
        from starlette.types import Scope

        validator = _make_validator(cognito_env)
        scope: Scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/whatever",
            "headers": [
                (b"authorization", f"Bearer {valid_access_token}".encode()),
            ],
        }
        request = Request(scope)
        user_id = await validator.extract_user_id(request)
        assert user_id == cognito_env["sub"]

    @pytest.mark.asyncio
    async def test_missing_header_raises_401(self, cognito_env, cognito_jwks_inject):
        from starlette.requests import Request

        validator = _make_validator(cognito_env)
        request = Request({"type": "http", "method": "GET", "path": "/api/x", "headers": []})
        with pytest.raises(HTTPException) as exc:
            await validator.extract_user_id(request)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_header_without_bearer_prefix_raises_401(
        self, cognito_env, cognito_jwks_inject, valid_access_token
    ):
        from starlette.requests import Request

        validator = _make_validator(cognito_env)
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/x",
            "headers": [(b"authorization", valid_access_token.encode())],
        }
        request = Request(scope)
        with pytest.raises(HTTPException) as exc:
            await validator.extract_user_id(request)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_bearer_with_empty_token_raises_401(
        self, cognito_env, cognito_jwks_inject
    ):
        from starlette.requests import Request

        validator = _make_validator(cognito_env)
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/x",
            "headers": [(b"authorization", b"Bearer ")],
        }
        request = Request(scope)
        with pytest.raises(HTTPException) as exc:
            await validator.extract_user_id(request)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_expired_token_raises_401(
        self, cognito_env, cognito_jwks_inject, expired_access_token
    ):
        from starlette.requests import Request

        validator = _make_validator(cognito_env)
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/x",
            "headers": [
                (b"authorization", f"Bearer {expired_access_token}".encode()),
            ],
        }
        request = Request(scope)
        with pytest.raises(HTTPException) as exc:
            await validator.extract_user_id(request)
        assert exc.value.status_code == 401


# --- TestExtractUserIdFromToken: WebSocket entrypoint ------------------


class TestExtractUserIdFromToken:
    """``CognitoBearerJwtAuth.extract_user_id_from_token`` — WS path.

    Returns ``None`` rather than raising, so the WS route can call
    ``ws.close(code=4401)`` without going through HTTPException unwinding.
    """

    @pytest.mark.asyncio
    async def test_valid_token_returns_sub(
        self, cognito_env, cognito_jwks_inject, valid_access_token
    ):
        validator = _make_validator(cognito_env)
        user_id = await validator.extract_user_id_from_token(valid_access_token)
        assert user_id == cognito_env["sub"]

    @pytest.mark.asyncio
    async def test_empty_token_returns_none(self, cognito_env, cognito_jwks_inject):
        validator = _make_validator(cognito_env)
        assert await validator.extract_user_id_from_token("") is None

    @pytest.mark.asyncio
    async def test_invalid_token_returns_none(
        self, cognito_env, cognito_jwks_inject, expired_access_token
    ):
        validator = _make_validator(cognito_env)
        assert await validator.extract_user_id_from_token(expired_access_token) is None


# --- TestJwtMiddleware: full FastAPI integration -----------------------


def _build_app_with_middleware(auth) -> FastAPI:
    """Tiny FastAPI app with auth_middleware mounted, an echo route,
    and the public health/config short-circuit paths."""
    app = FastAPI()
    app.middleware("http")(auth_middleware(auth))

    @app.get("/api/whoami")
    async def whoami():
        # Reads ``current_user_id`` set by the middleware.
        return {"user_id": current_user_id()}

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/config")
    async def config():
        return {"auth_required": False}

    # Routes that merely START with a public path. They exist so a 401 from
    # them proves the middleware matched exactly, rather than only that no
    # route was there to answer.
    @app.get("/api/config-internal")
    async def config_lookalike():
        return {"leaked": True}

    @app.get("/api/config/x")
    async def config_subpath():
        return {"leaked": True}

    return app


class TestJwtMiddleware:
    """End-to-end middleware tests via FastAPI ``TestClient``."""

    def test_valid_bearer_populates_user_id(
        self, cognito_env, cognito_jwks_inject, valid_access_token
    ):
        validator = _make_validator(cognito_env)
        app = _build_app_with_middleware(validator)
        client = TestClient(app)
        resp = client.get(
            "/api/whoami",
            headers={"Authorization": f"Bearer {valid_access_token}"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"user_id": cognito_env["sub"]}

    def test_missing_header_returns_401(self, cognito_env, cognito_jwks_inject):
        validator = _make_validator(cognito_env)
        app = _build_app_with_middleware(validator)
        client = TestClient(app)
        resp = client.get("/api/whoami")
        assert resp.status_code == 401

    def test_invalid_token_returns_401(
        self, cognito_env, cognito_jwks_inject, expired_access_token
    ):
        validator = _make_validator(cognito_env)
        app = _build_app_with_middleware(validator)
        client = TestClient(app)
        resp = client.get(
            "/api/whoami",
            headers={"Authorization": f"Bearer {expired_access_token}"},
        )
        assert resp.status_code == 401

    def test_health_endpoint_does_not_require_auth(
        self, cognito_env, cognito_jwks_inject
    ):
        """``/api/health`` short-circuits the middleware (the unauthenticated-path
        check in ``app.platform.auth.auth_middleware``).
        """
        validator = _make_validator(cognito_env)
        app = _build_app_with_middleware(validator)
        client = TestClient(app)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_config_endpoint_does_not_require_auth(
        self, cognito_env, cognito_jwks_inject
    ):
        """``/api/config`` short-circuits — frontend reads it pre-login."""
        validator = _make_validator(cognito_env)
        app = _build_app_with_middleware(validator)
        client = TestClient(app)
        resp = client.get("/api/config")
        assert resp.status_code == 200
        assert "auth_required" in resp.json()

    @pytest.mark.parametrize("path", ["/api/config-internal", "/api/config/x"])
    def test_routes_sharing_the_config_prefix_require_auth(
        self, cognito_env, cognito_jwks_inject, path
    ):
        """Only ``/api/config`` itself is public. A route whose path starts
        with it, as a sibling or as a subpath, still needs a token."""
        validator = _make_validator(cognito_env)
        app = _build_app_with_middleware(validator)
        client = TestClient(app)
        resp = client.get(path)
        assert resp.status_code == 401
        assert "leaked" not in resp.text


# --- TestRequireAccess: ownership is the only access rule ---------------


class _OneProjectStore:
    """Metadata store holding one project record, for ``require_access``."""

    def __init__(self, record: dict) -> None:
        self.record = record

    async def find_project_by_id(self, project_id: str) -> dict | None:
        return self.record if project_id == self.record["id"] else None


class TestRequireAccess:
    """Cloud ``require_access`` admits the owner and no one else."""

    def _auth(self, cognito_env, record: dict) -> CognitoBearerJwtAuth:
        return CognitoBearerJwtAuth(
            metadata_store=_OneProjectStore(record),  # type: ignore[arg-type]
            region=cognito_env["region"],
            user_pool_id=cognito_env["user_pool_id"],
            client_id=cognito_env["spa_client_id"],
        )

    @pytest.mark.parametrize("action", ["read", "write"])
    async def test_owner_is_admitted(self, cognito_env, action):
        auth = self._auth(cognito_env, {"id": "p1", "owner_id": "owner"})
        await auth.require_access("owner", "p1", action)

    @pytest.mark.parametrize("action", ["read", "write"])
    async def test_a_listed_viewer_is_denied(self, cognito_env, action):
        """Records written while sharing existed can carry ``viewer_ids``; it grants nothing."""
        auth = self._auth(
            cognito_env, {"id": "p1", "owner_id": "owner", "viewer_ids": ["other"]}
        )
        with pytest.raises(HTTPException) as exc:
            await auth.require_access("other", "p1", action)
        assert exc.value.status_code == 403

    async def test_a_missing_project_is_404(self, cognito_env):
        auth = self._auth(cognito_env, {"id": "p1", "owner_id": "owner"})
        with pytest.raises(HTTPException) as exc:
            await auth.require_access("owner", "p2", "read")
        assert exc.value.status_code == 404


# --- TestDevModeAuth: LocalAuth bypass --------------------------------


class TestDevModeAuth:
    """``LocalAuth`` is the dev-mode :class:`Auth` — no JWT, env-driven."""

    def test_dev_mode_no_token_succeeds(self, monkeypatch):
        from app.platform.dev_local import LocalAuth

        monkeypatch.setenv("EVAL_STUDIO_MODE", "dev")
        monkeypatch.delenv("LOCAL_DEV_USER_ID", raising=False)
        local = LocalAuth(metadata_store=None)  # type: ignore[arg-type]
        app = _build_app_with_middleware(local)
        client = TestClient(app)
        resp = client.get("/api/whoami")
        assert resp.status_code == 200
        # Default sentinel from ``get_local_dev_user_id``.
        assert resp.json() == {"user_id": get_local_dev_user_id()}

    def test_dev_mode_respects_env_override(self, monkeypatch):
        from app.platform.dev_local import LocalAuth

        monkeypatch.setenv("EVAL_STUDIO_MODE", "dev")
        monkeypatch.setenv("LOCAL_DEV_USER_ID", "alice")
        local = LocalAuth(metadata_store=None)  # type: ignore[arg-type]
        app = _build_app_with_middleware(local)
        client = TestClient(app)
        resp = client.get("/api/whoami")
        assert resp.status_code == 200
        assert resp.json() == {"user_id": "alice"}

    @pytest.mark.asyncio
    async def test_dev_mode_extract_user_id_from_token_ignores_token(
        self, monkeypatch
    ):
        """Dev WS path: token value is ignored, returns the local-dev user."""
        from app.platform.dev_local import LocalAuth

        monkeypatch.setenv("EVAL_STUDIO_MODE", "dev")
        monkeypatch.setenv("LOCAL_DEV_USER_ID", "wsuser")
        local = LocalAuth(metadata_store=None)  # type: ignore[arg-type]
        # Any token value, even garbage — the dev impl must still return
        # the configured user id without raising.
        user_id = await local.extract_user_id_from_token("garbage")
        assert user_id == "wsuser"
        # Even an empty token is accepted in dev.
        assert await local.extract_user_id_from_token("") == "wsuser"
