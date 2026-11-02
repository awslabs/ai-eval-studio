# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auth Protocol, FastAPI middleware, and ``require_access``.

Authentication is enforced in-app: the FastAPI middleware reads
``Authorization: Bearer <jwt>`` from each request and validates it
against Cognito JWKS (cloud) or returns the configured local-dev
sentinel (dev). Authorization is enforced here: :func:`require_access`
is called from every API route and every MCP tool wrapper, and allows
``read`` and ``write`` only for the project's ``owner_id``.

In dev mode, the auth middleware short-circuits the JWT extraction and
injects ``LOCAL_DEV_USER_ID`` (defaults to ``"local-dev"``) so the
existing single-user local workflow keeps working without code changes
in routes.

The ``extract_user_id_from_token`` Protocol method exists specifically
for the WebSocket route, which receives the JWT in the connection's first
frame (browsers cannot set ``Authorization`` headers on ``new WebSocket``
calls, and a query string is logged by everything that records a URL). The
HTTP middleware path uses :meth:`Auth.extract_user_id` and the WS path uses
:meth:`Auth.extract_user_id_from_token`; both share a single underlying
validator in the cloud impl.
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Awaitable, Callable, Protocol

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse

from .types import (
    AccessAction,
    ProjectId,
    RequestContextValue,
    UserId,
    request_context,
)

logger = logging.getLogger(__name__)

# Paths the auth middleware serves without a token: the load balancer's
# health check and the SPA's pre-login bootstrap config. Matched exactly.
_PUBLIC_PATHS = frozenset({"/api/health", "/api/config"})

# Dev-mode hardcoded user id. Override via ``LOCAL_DEV_USER_ID`` env var.
DEFAULT_LOCAL_DEV_USER_ID = "local-dev"


class Auth(Protocol):
    """Authorization Protocol — extract user, enforce ``require_access``."""

    async def extract_user_id(self, request: Request) -> UserId:
        """Return the user_id from the request, or raise HTTP 401.

        Cloud implementation parses ``Authorization: Bearer <jwt>`` and
        validates against Cognito JWKS. Dev implementation returns
        ``LOCAL_DEV_USER_ID``.
        """
        ...

    async def extract_user_id_from_token(self, token: str) -> UserId | None:
        """Validate a raw JWT string and return the user_id, or ``None`` on failure.

        Used by the WebSocket route, which receives the JWT in the
        connection's first frame (browsers can't set ``Authorization``
        headers on ``new WebSocket(url)``, and a query string is logged by
        everything that records a URL). Returns ``None`` (rather than
        raising) so the WS handler can ``await ws.close(code=4401)``
        without unwinding through HTTP exception handlers.

        Cloud implementation runs the same JWKS + claim validation as
        :meth:`extract_user_id`. Dev implementation ignores the token
        and returns ``LOCAL_DEV_USER_ID``.
        """
        ...

    async def require_access(
        self,
        user_id: UserId,
        project_id: ProjectId,
        action: AccessAction,
    ) -> None:
        """Raise HTTP 403 unless ``user_id`` may take ``action`` on ``project_id``.

        Owner can read+write. Anyone else is denied. Missing project →
        HTTP 404 (so we don't leak existence to non-owners).
        """
        ...


def auth_middleware(
    auth: Auth,
) -> Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]:
    """Build the FastAPI middleware that populates ``request_context``.

    Wires :meth:`Auth.extract_user_id` and a fresh ``request_id`` into
    the per-request ContextVar, so downstream code can call
    :func:`current_user_id` and structured-log helpers without threading
    the context through call signatures.

    Args:
        auth: The Platform's Auth implementation.

    Returns:
        An ASGI-compatible middleware coroutine.
    """

    async def middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Allow unauthenticated paths so the health endpoint and the
        # frontend bootstrap config endpoint are reachable before login.
        # Exact matches only: a prefix match would also make public any
        # later route that happens to start with one of these paths.
        path = request.url.path
        if path in _PUBLIC_PATHS:
            return await call_next(request)

        # Starlette's BaseHTTPMiddleware sits OUTSIDE ExceptionMiddleware in
        # the ASGI stack, so an HTTPException raised here would propagate
        # past the framework's exception handler and surface as a server
        # error. Convert auth failures to a JSONResponse directly — same
        # body shape FastAPI's ``http_exception_handler`` produces.
        try:
            user_id = await auth.extract_user_id(request)
        except HTTPException as exc:
            return JSONResponse(
                {"detail": exc.detail},
                status_code=exc.status_code,
                headers=exc.headers,
            )
        except Exception:
            logger.exception("[auth] Failed to extract user from request")
            return JSONResponse(
                {"detail": "Unauthorized"},
                status_code=401,
            )

        ctx = RequestContextValue(
            user_id=user_id,
            request_id=str(uuid.uuid4()),
        )
        token = request_context.set(ctx)
        try:
            return await call_next(request)
        finally:
            request_context.reset(token)

    return middleware


def get_local_dev_user_id() -> UserId:
    """Read ``LOCAL_DEV_USER_ID`` from the env (or return the default)."""
    return os.environ.get("LOCAL_DEV_USER_ID", DEFAULT_LOCAL_DEV_USER_ID)


__all__ = [
    "Auth",
    "DEFAULT_LOCAL_DEV_USER_ID",
    "auth_middleware",
    "get_local_dev_user_id",
]
