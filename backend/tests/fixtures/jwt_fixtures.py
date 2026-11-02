# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""RSA keypair + JWKS + JWT signing helpers for the Cognito-bearer auth tests.

``CognitoBearerJwtAuth`` validates Cognito access tokens against a JWKS
document. PyJWT's :class:`PyJWKClient` fetches that
document over plain ``urllib`` — no HTTP-mock library short of
``pytest-httpserver`` (real socket) intercepts that path cleanly.

Strategy chosen here: avoid HTTP entirely. The
:func:`tests.conftest.cognito_jwks_inject` fixture monkeypatches
``app.platform.aws.auth_cognito_bearer._get_jwks_client`` to return a
:class:`PyJWKClient` whose internal ``jwk_set_cache`` is pre-populated with
the JWKS dict built from this module's keypair (the raw dict, which is what
PyJWT's ``fetch_data`` stores there). The cache hit short-circuits
``fetch_data()``, so no network call is ever attempted.

Why not ``responses`` or ``pytest-httpx``: ``responses`` patches
``requests``/``urllib3``, ``pytest-httpx`` patches ``httpx``. PyJWT uses
neither — it calls ``urllib.request.urlopen`` directly. Pre-populating the
client's cache requires no production-code change and no extra dev dep.
"""
from __future__ import annotations

from typing import Any

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm


def generate_rsa_keypair() -> tuple[bytes, bytes]:
    """Generate a fresh RSA-2048 keypair encoded as PEM bytes.

    Returns:
        Tuple ``(private_pem, public_pem)`` — PEM-encoded PKCS#8 private key
        and SubjectPublicKeyInfo public key.
    """
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def public_key_to_jwks_dict(public_pem: bytes, kid: str) -> dict[str, Any]:
    """Build a JWKS document containing one RS256 signing key.

    Mirrors the shape Cognito's ``/.well-known/jwks.json`` returns:
    ``{"keys": [{"kty": "RSA", "kid": "...", "use": "sig", "alg": "RS256",
    "n": "...", "e": "..."}]}``.

    Args:
        public_pem: PEM-encoded public key.
        kid: Key ID to embed (must match the ``kid`` in the JWT header).

    Returns:
        A JWKS dict ready to be loaded into :class:`jwt.PyJWKSet`.
    """
    public_key = serialization.load_pem_public_key(public_pem)
    jwk = RSAAlgorithm.to_jwk(public_key, as_dict=True)
    jwk.update({
        "kid": kid,
        "use": "sig",
        "alg": "RS256",
    })
    return {"keys": [jwk]}


def sign_access_token(
    private_pem: bytes,
    *,
    kid: str,
    sub: str,
    iss: str,
    client_id: str,
    token_use: str = "access",
    exp_offset_sec: int = 3600,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Sign an RS256 JWT mimicking a Cognito access token.

    Args:
        private_pem: PEM-encoded RSA private key (from
            :func:`generate_rsa_keypair`).
        kid: Key ID to put in the JWT header (must match a JWKS key).
        sub: Subject claim — becomes the user_id after validation.
        iss: Issuer claim, e.g.
            ``https://cognito-idp.us-east-1.amazonaws.com/us-east-1_TESTPOOL``.
        client_id: Cognito SPA client id; goes in the ``client_id`` claim.
        token_use: ``"access"`` (default) or ``"id"``.
        exp_offset_sec: ``exp - iat`` in seconds. Default 1h. Pass a negative
            value to produce an already-expired token.
        extra_claims: Additional claims merged into the payload. Useful for
            edge-case tests that need to omit standard claims.

    Returns:
        The signed JWT as a compact-serialized string.
    """
    import time
    now = int(time.time())
    claims: dict[str, Any] = {
        "sub": sub,
        "iss": iss,
        "client_id": client_id,
        "token_use": token_use,
        "iat": now,
        "exp": now + exp_offset_sec,
        # Cognito tokens carry these too; harmless and more realistic.
        "auth_time": now,
        "jti": "test-jti-0001",
    }
    if extra_claims:
        claims.update(extra_claims)
    return jwt.encode(
        claims,
        private_pem,
        algorithm="RS256",
        headers={"kid": kid},
    )


__all__ = [
    "generate_rsa_keypair",
    "public_key_to_jwks_dict",
    "sign_access_token",
]
