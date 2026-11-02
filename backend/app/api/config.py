# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public ``GET /api/config`` endpoint — bootstrap config for the SPA.

The frontend's :file:`AuthGate` component calls this on first load to
discover whether OIDC is required and, if so, where the Cognito hosted
UI lives. Unauthenticated users must reach it, so the middleware in
:mod:`app.platform.auth` lets exactly this path (and ``/api/health``)
through without a token.

Because it is public, it serves only what the SPA needs to sign in. The
AgentCore runtime ARN is deliberately absent: it names the account and the
runtime, and the SPA never contacts the runtime (every chat turn goes
through the coordinator's ``/ws`` relay, which reads
``AGENTCORE_RUNTIME_ARN`` from its own environment). ``tests/test_config.py``
pins the exact key set, so adding a field here is a deliberate test change.

Response shape::

    {
      "auth_required": true | false,
      "cognito": {
        "domain": "https://<prefix>.auth.<region>.amazoncognito.com",
        "client_id": "<spa client id>",
        "region": "<aws region>",
        "user_pool_id": "<region>_<pool-suffix>"
      } | null,
      "logout_redirect": "<url to redirect to after Cognito sign-out>"
    }

In dev mode (``EVAL_STUDIO_MODE != "cloud"``): ``auth_required`` is
``false``, ``cognito`` is ``null``, ``logout_redirect`` is ``"/"``.

In cloud mode: ``auth_required`` is ``true`` and ``cognito`` is
populated from environment variables set by the CDK (``COGNITO_DOMAIN``,
``COGNITO_SPA_CLIENT_ID``, ``COGNITO_REGION``, ``COGNITO_USER_POOL_ID``).
``logout_redirect`` defaults to ``"/"`` and can be overridden via
``LOGOUT_REDIRECT_URL`` (typically the CloudFront distribution DNS).

The ``user_pool_id`` field exists so the SPA can construct the OIDC
issuer URL (``https://cognito-idp.<region>.amazonaws.com/<pool>``) and
JWKS URL deterministically. Cognito's
``/.well-known/openid-configuration`` endpoint does not set
``Access-Control-Allow-Origin``; a browser fetch from the SPA origin
fails CORS. Returning the pieces here lets the SPA configure
``oidc-client-ts`` with explicit metadata and skip discovery.
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter

# No prefix — this is the canonical ``/api/config`` path. Tag groups it
# under "config" in the auto-generated /docs page.
router = APIRouter(tags=["config"])


@router.get("/api/config")
async def get_config() -> dict[str, Any]:
    """Return SPA bootstrap config (public, unauthenticated).

    Reads ``EVAL_STUDIO_MODE`` to decide whether OIDC is required, then
    reads the Cognito-related env vars (cloud mode only).

    Always returns successfully — a missing env var in cloud mode produces
    an empty string in the response, which the SPA's ``AuthGate`` should
    treat as a configuration error and surface to the user.

    Returns:
        Dict with keys ``auth_required`` (bool), ``cognito``
        (dict | None) and ``logout_redirect`` (str).
    """
    mode = os.environ.get("EVAL_STUDIO_MODE", "dev").lower()
    auth_required = mode == "cloud"

    if not auth_required:
        # Dev mode: no OIDC, redirect to root after any "sign out" the
        # SPA chooses to render (typically a no-op in dev).
        return {
            "auth_required": False,
            "cognito": None,
            "logout_redirect": "/",
        }

    # Cloud mode: read Cognito hosted UI domain + SPA client + region
    # from env. The CDK ``compute_stack.py`` injects these as Fargate
    # task env vars.
    cognito = {
        "domain": os.environ.get("COGNITO_DOMAIN", ""),
        "client_id": os.environ.get("COGNITO_SPA_CLIENT_ID", ""),
        "region": os.environ.get(
            "COGNITO_REGION", os.environ.get("AWS_REGION", "")
        ),
        "user_pool_id": os.environ.get("COGNITO_USER_POOL_ID", ""),
    }
    # ``LOGOUT_REDIRECT_URL`` is the CloudFront DNS name (or eventual
    # custom domain) that Cognito sends the user back to after the
    # hosted-UI ``/logout`` flow. Falls back to ``"/"`` so the endpoint
    # always returns a usable shape.
    logout_redirect = os.environ.get("LOGOUT_REDIRECT_URL", "/")

    return {
        "auth_required": True,
        "cognito": cognito,
        "logout_redirect": logout_redirect,
    }


__all__ = ["router"]
