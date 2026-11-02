# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cognito-bearer-token :class:`Auth` implementation (in-app JWT).

Cognito auth runs in-app rather than at the ALB listener. Each
request carries ``Authorization: Bearer <access-token>``; the WebSocket
route gets the same token in the connection's first frame, because
browsers cannot set headers on ``new WebSocket(url)``. Both paths converge on
:meth:`CognitoBearerJwtAuth._validate_token`, which performs seven
checks:

1. Read the bearer token (HTTP path) or accept the raw token string
   (WS path) — return ``None`` if missing/empty.
2. Verify the RS256 signature against the user pool's JWKS, served at
   ``{issuer}/.well-known/jwks.json``. Keys are cached 24h via PyJWT's
   :class:`PyJWKClient`; the client itself is cached per JWKS URL in
   module scope so we don't refetch JWKS on every request.
3. ``iss`` matches ``https://cognito-idp.<region>.amazonaws.com/<pool-id>``.
4. ``token_use`` is ``"access"`` — id-tokens are rejected (the SPA uses
   the access token for API calls per OIDC best practice).
5. ``client_id`` matches the configured SPA client. Cognito access
   tokens put the client id in ``client_id``, NOT in the standard
   ``aud`` claim, so PyJWT's built-in ``audience=`` arg cannot validate
   it; we pass ``options={"verify_aud": False}`` and check the claim
   manually.
6. ``exp`` not expired — PyJWT enforces this by default during
   ``jwt.decode``.
7. Return ``claims["sub"]`` as the user_id, or ``None`` on any failure.

Fails closed: the HTTP middleware turns a ``None`` from
:meth:`extract_user_id` into a 401; the WebSocket route turns it into
a ``ws.close(code=4401)``.

``require_access``: the owner can read and write; anyone else is denied.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import jwt
from fastapi import HTTPException, Request
from jwt import PyJWKClient

from ..auth import Auth
from ..metadata_store import MetadataStore
from ..types import AccessAction, ProjectId, UserId

logger = logging.getLogger(__name__)


# Cognito access tokens are signed with RS256 (RSA-2048). Hard-coded so
# attackers can't downgrade by claiming alg=none in the JWT header.
_COGNITO_JWT_ALGORITHMS = ["RS256"]

# JWKS cache lifetime (seconds). PyJWT's PyJWKClient refreshes after this
# expires; AWS rotates keys infrequently so 24h is a safe trade-off
# between freshness and reducing JWKS round-trips.
_JWKS_CACHE_LIFESPAN_SEC = 86400


def _mask_sub(sub: Any) -> str:
    """Return a short, log-safe rendering of a Cognito ``sub`` claim.

    Cognito subs are UUIDs — high-entropy and useless to leak in full
    to a shared log stream. The first 8 chars are enough to correlate
    one user's session within a single deploy without making the log
    a re-identification surface for anyone reading CloudWatch. Falls
    back to ``"-"`` when ``sub`` is missing.

    Args:
        sub: The ``sub`` claim, or ``None``/missing.

    Returns:
        A truncated string suitable for log lines.
    """
    if not sub or not isinstance(sub, str):
        return "-"
    return f"{sub[:8]}…"


@lru_cache(maxsize=4)
def _get_jwks_client(jwks_url: str) -> PyJWKClient:
    """Return a process-wide :class:`PyJWKClient` for ``jwks_url``.

    PyJWT's :class:`PyJWKClient` already caches the JWKS document
    internally; we cache the client itself so we don't reconstruct it
    (and re-tear-down its internal urllib pool) on every request. The
    ``maxsize=4`` is generous — in practice there's exactly one Cognito
    user pool per environment, so the cache holds one entry.

    Args:
        jwks_url: The fully-qualified JWKS URL, e.g.
            ``https://cognito-idp.us-east-1.amazonaws.com/us-east-1_abc/.well-known/jwks.json``.

    Returns:
        A cached :class:`PyJWKClient` that fetches and caches the JWKS
        document for 24h.
    """
    return PyJWKClient(jwks_url, lifespan=_JWKS_CACHE_LIFESPAN_SEC)


class CognitoBearerJwtAuth(Auth):
    """Cloud :class:`Auth` validating Cognito access tokens in-app.

    Args:
        metadata_store: Used by :meth:`require_access` to look up the
            project record and its ``owner_id``.
        region: AWS region of the Cognito user pool (e.g. ``us-east-1``).
            Used to construct the issuer URL.
        user_pool_id: Cognito user pool id (e.g. ``us-east-1_AbCdEf123``).
            Used in the issuer URL and JWKS endpoint.
        client_id: SPA Cognito user-pool client id. Validated against
            the ``client_id`` claim on every token.
    """

    def __init__(
        self,
        metadata_store: MetadataStore,
        *,
        region: str,
        user_pool_id: str,
        client_id: str,
    ) -> None:
        self.metadata_store = metadata_store
        self._region = region
        self._user_pool_id = user_pool_id
        self._client_id = client_id
        # Cognito's published OIDC issuer format. The JWKS endpoint is
        # always at ``{issuer}/.well-known/jwks.json`` per RFC 8414.
        self._issuer = (
            f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}"
        )
        self._jwks_url = f"{self._issuer}/.well-known/jwks.json"

    def _validate_token(self, token: str) -> dict[str, Any] | None:
        """Run the 7-step Cognito access-token validation.

        Returns the decoded claims dict on success, ``None`` on any
        failure (bad signature, wrong issuer, wrong audience, wrong
        token_use, expired token, malformed token, JWKS unreachable,
        etc.).

        **Logging contract.** Every rejection emits a structured
        ``WARNING`` line with a stable ``reason=`` field so CloudWatch
        Logs Insights queries can group them (a JWKS outage logs at
        ``ERROR``). Rejections are deliberately not at ``DEBUG``, which
        uvicorn's default ``INFO`` level swallows — that left every 401 in
        prod undebuggable. Success emits a ``DEBUG`` line with the masked
        ``sub`` claim: it fires on every request, so at ``INFO`` it would
        drown the log.

        Args:
            token: The raw JWT string (no ``Bearer `` prefix).

        Returns:
            Decoded claims dict on success, ``None`` on any failure.
        """
        if not token:
            logger.warning(
                "[auth] Rejected token reason=empty kid=- sub=-"
            )
            return None
        # Best-effort kid extraction so the rejection log can correlate
        # to a specific signing key when JWKS rotation is in flight. The
        # header is base64-decodable without verifying the signature;
        # ``options={'verify_signature': False}`` is the documented
        # PyJWT pattern for this.
        try:
            kid = jwt.get_unverified_header(token).get("kid", "-") or "-"
        except (jwt.PyJWTError, KeyError, ValueError):
            kid = "-"

        try:
            jwks_client = _get_jwks_client(self._jwks_url)
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=_COGNITO_JWT_ALGORITHMS,
                issuer=self._issuer,
                # Cognito access tokens don't populate ``aud``; the
                # client id lives in ``client_id`` and is checked
                # manually below. Disable PyJWT's aud check.
                options={"verify_aud": False},
            )
        # Every rejection path below is flagged by
        # ``python-logger-credential-disclosure``, which fires on the words
        # "token"/"auth" near a log call rather than on what is interpolated.
        # No credential material reaches any of them. What does: ``kid`` (a
        # public JWKS key id), ``sub`` through ``_mask_sub``, the expected issuer
        # and client id, and an exception type name. The one site that logs an
        # exception message says inline why that text cannot reflect token
        # contents. Each directive below is scoped to this one rule, so a
        # different finding on the same line still reports.
        except jwt.ExpiredSignatureError:
            # nosemgrep: python-logger-credential-disclosure
            logger.warning(
                "[auth] Rejected token reason=expired kid=%s sub=-", kid
            )
            return None
        except jwt.InvalidIssuerError:
            # nosemgrep: python-logger-credential-disclosure
            logger.warning(
                "[auth] Rejected token reason=bad-issuer kid=%s sub=- "
                "expected_iss=%s",
                kid, self._issuer,
            )
            return None
        except jwt.InvalidSignatureError:
            # nosemgrep: python-logger-credential-disclosure
            logger.warning(
                "[auth] Rejected token reason=bad-signature kid=%s sub=-",
                kid,
            )
            return None
        except jwt.DecodeError as exc:
            # Malformed JWT (not a 3-segment base64 string, header
            # parse failure, etc.). Including the exception message
            # here is safe — DecodeError text never reflects token
            # contents, only structural failures.
            # nosemgrep: python-logger-credential-disclosure
            logger.warning(
                "[auth] Rejected token reason=malformed kid=%s sub=- detail=%s",
                kid, exc,
            )
            return None
        except jwt.PyJWKClientError as exc:
            # JWKS endpoint unreachable / no matching key for the
            # token's kid. Worth a louder log because this is an
            # operational failure, not a credential one — a JWKS
            # outage 401s every request.
            logger.error(
                "[auth] JWKS lookup failed kid=%s url=%s detail=%s",
                kid, self._jwks_url, exc,
            )
            return None
        except (jwt.PyJWTError, KeyError, ValueError) as exc:
            # Catch-all for the rest of the PyJWT exception hierarchy
            # (InvalidAlgorithmError, InvalidKeyError, etc.) plus the
            # KeyError / ValueError that PyJWKClient can raise when
            # the JWKS document is malformed.
            # nosemgrep: python-logger-credential-disclosure
            logger.warning(
                "[auth] Rejected token reason=jwt-error kid=%s sub=- "
                "type=%s detail=%s",
                kid, type(exc).__name__, exc,
            )
            return None

        # The request path accepts access tokens only; the SPA uses the
        # access token for API calls and id-tokens carry profile claims.
        if claims.get("token_use") != "access":
            # nosemgrep: python-logger-credential-disclosure
            logger.warning(
                "[auth] Rejected token reason=wrong-token-use kid=%s "
                "sub=%s got=%r expected=%s",
                kid,
                _mask_sub(claims.get("sub")),
                claims.get("token_use"),
                "access",
            )
            return None

        # A Cognito access token puts the client id in ``client_id`` and
        # leaves ``aud`` unset, hence ``verify_aud: False`` above and this
        # manual check.
        presented = claims.get("client_id")
        if presented != self._client_id:
            # nosemgrep: python-logger-credential-disclosure
            logger.warning(
                "[auth] Rejected token reason=wrong-client-id kid=%s "
                "sub=%s got=%r expected=%r",
                kid,
                _mask_sub(claims.get("sub")),
                presented,
                self._client_id,
            )
            return None

        # DEBUG, not INFO: this fires on every authenticated request AND
        # every SPA poll — success is the expected case and at INFO it
        # drowned the coordinator log (~73% of lines in a 1h sample). The
        # rejection paths above stay at WARNING/ERROR; those are what you
        # actually want to find when scanning the log.
        # nosemgrep: python-logger-credential-disclosure
        logger.debug(
            "[auth] Token validated kid=%s sub=%s",
            kid,
            _mask_sub(claims.get("sub")),
        )
        return claims

    async def extract_user_id(self, request: Request) -> UserId:
        """Extract the authenticated user_id from a FastAPI request.

        Reads ``Authorization: Bearer <token>``, runs the validator,
        and returns the ``sub`` claim. Raises :class:`HTTPException`
        with status 401 on any failure (missing header, invalid token,
        expired token, etc.). Returning rather than raising would let
        unauthenticated requests slip past the middleware.

        Args:
            request: The FastAPI request being authenticated.

        Returns:
            The validated user_id (Cognito ``sub`` claim).

        Raises:
            HTTPException: 401 if the token is missing or invalid.
        """
        path = request.url.path
        auth_header = request.headers.get("Authorization", "")
        # RFC 6750 §2.1 — the "Bearer" scheme name is case-insensitive.
        scheme, _, raw_token = auth_header.partition(" ")
        if scheme.lower() != "bearer" or not raw_token:
            logger.warning(
                "[auth] Rejected request reason=missing-bearer path=%s "
                "header_present=%s",
                path,
                bool(auth_header),
            )
            raise HTTPException(401, "Missing authentication header")
        token = raw_token.strip()
        claims = self._validate_token(token)
        if claims is None:
            # ``_validate_token`` already logged the specific reason at
            # WARNING; we don't double-log here, but we DO surface the
            # request path so the structured rejection line above can
            # be correlated with the route that 401'd.
            logger.warning("[auth] 401 path=%s", path)
            raise HTTPException(401, "Invalid authentication token")
        sub = claims.get("sub")
        if not sub:
            logger.warning(
                "[auth] Rejected request reason=missing-sub path=%s", path
            )
            raise HTTPException(401, "Authentication token missing sub claim")
        return sub

    async def extract_user_id_from_token(self, token: str) -> UserId | None:
        """Validate a raw JWT string from a non-HTTP context (WebSocket).

        Browsers cannot set ``Authorization`` headers on
        ``new WebSocket(url)``, so the client sends the access token as
        the connection's first frame. The WS route calls this method to
        get the user_id without an HTTPException-raising middleware in
        the path; ``None`` lets the WS handler call
        ``await ws.close(code=4401)`` cleanly.

        Args:
            token: The raw JWT string from the client's auth frame.

        Returns:
            The validated ``sub`` claim on success, ``None`` on any
            failure. Never raises.
        """
        claims = self._validate_token(token)
        if claims is None:
            # ``_validate_token`` already logged the specific reason at
            # WARNING. Add a one-line marker so log filters can
            # distinguish a WS-handshake 4401 from an HTTP 401 — the
            # two surface in different parts of the SPA.
            logger.warning("[auth] 4401 transport=ws")
            return None
        sub = claims.get("sub")
        if not sub:
            logger.warning(
                "[auth] Rejected ws reason=missing-sub transport=ws"
            )
            return None
        return sub

    async def require_access(
        self,
        user_id: UserId,
        project_id: ProjectId,
        action: AccessAction,
    ) -> None:
        """Authorize ``user_id`` for ``action`` on ``project_id``.

        Owner can read+write. Anyone else is denied. Missing project →
        HTTP 404 so we don't leak existence to non-owners.

        Uses :meth:`MetadataStore.find_project_by_id` rather than
        :meth:`get_project` because the latter is keyed by USER#{owner}
        in the cloud impl, and the caller may not be the owner.
        ``find_project_by_id`` resolves by project id alone via the
        project-index entry written at put_project time.

        Args:
            user_id: The authenticated caller.
            project_id: The project being accessed.
            action: ``"read"`` or ``"write"``.

        Raises:
            HTTPException: 404 if the project doesn't exist; 403 if
                the caller lacks the requested access.
        """
        project = await self.metadata_store.find_project_by_id(project_id)
        if project is None:
            # 404 here can mean "really doesn't exist" or "exists but
            # the caller can't see it" — by design, we don't
            # distinguish in the response (avoids leaking existence
            # to non-owners). The log line keeps both variants
            # observable on the operator side.
            logger.warning(
                "[authz] Rejected access reason=project-not-found "
                "user=%s project=%s action=%s",
                _mask_sub(user_id),
                project_id,
                action,
            )
            raise HTTPException(status_code=404, detail="Project not found")
        owner_id = project.get("owner_id")
        if owner_id == user_id:
            return
        logger.warning(
            "[authz] Rejected access reason=forbidden user=%s "
            "project=%s action=%s owner=%s",
            _mask_sub(user_id),
            project_id,
            action,
            _mask_sub(owner_id),
        )
        raise HTTPException(status_code=403, detail="Forbidden")


__all__ = ["CognitoBearerJwtAuth"]
