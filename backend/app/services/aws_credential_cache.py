# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-(user, project) scoped boto3 Session cache.

Vends short-lived AWS credentials, minted via :class:`STSTokenVending`,
that are narrowed to one project's S3 session prefix and the owner's
DynamoDB partition (``LeadingKeys``). The session tags are built by
:meth:`STSTokenVending.build_session_tags`.

Why this exists:

The platform layer used to construct one ``boto3.client`` at bootstrap
and share it across every request, so cross-tenant safety for those calls
relied entirely on the application's ``require_access`` chokepoint. A call
made through a vended session is also held by IAM: one that names another
tenant's S3 prefix or partition fails with ``AccessDenied``. That is
narrower than tenant isolation — the tenant tag is chosen by whoever mints,
and IAM cannot bind it to the caller (see the README's "Security scope").

Cache shape:

    key   = (user_id, project_id)
    value = (boto3.session.Session, expires_at_epoch)

5-minute TTL on the cache slot regardless of the credentials' actual
expiration (which we set to 1h via STS). 5 min amortizes
``sts:AssumeRole`` cost across bursts of activity for the same user
without holding stale auth state too long.

Dev-mode behavior: ``platform.token_vending.mint`` returns ``None``,
which surfaces as ``None`` from :meth:`get_session` — callers fall
back to ``boto3.Session()`` (ambient creds) so the same code path
works locally without a Cognito identity.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import boto3.session

logger = logging.getLogger(__name__)

# How long we hold a vended session in the cache. Independent of the
# STS credentials' own TTL — we err on the short side here so a user
# whose access was just revoked can't keep operating against the cache.
_CACHE_TTL_SEC = 300


class AwsCredentialCache:
    """LRU-ish cache of STS-vended boto3 Sessions per (user, project).

    Thread-safe: the cache is read by both async (FastAPI) and sync
    (worker subprocess) paths. Eviction is lazy — entries are checked
    against ``expires_at`` on lookup; idle entries linger until they're
    next probed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[
            tuple[str, str], tuple["boto3.session.Session", float]
        ] = {}

    async def get_session(
        self, user_id: str, project_id: str, cwd: str
    ) -> "boto3.session.Session | None":
        """Return a scoped boto3 Session for ``(user_id, project_id)``.

        Returns ``None`` in dev mode (no scoped creds available;
        callers should fall back to ambient credentials).

        Args:
            user_id: Cognito ``sub`` of the requesting user.
            project_id: Project the credentials may access. Used in
                the DynamoDB ``LeadingKeys`` constraint.
            cwd: Filesystem path the agent will run in. The Claude
                Agent SDK derives its session-store S3 prefix from
                this path; the session tags' values resolve the role's
                tag-templated policy to match that exact prefix. See
                :meth:`STSTokenVending.build_session_tags`.

        Returns:
            A boto3 Session populated with the scoped credentials, or
            ``None`` if STS minting isn't available (dev mode).
        """
        cache_key = (user_id, project_id)
        now = time.time()

        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                session, expires_at = cached
                if expires_at > now:
                    return session
                # Expired — drop it before re-minting to avoid two
                # callers each minting fresh creds while the stale
                # entry is still in place.
                del self._cache[cache_key]

        session = await self._mint(user_id, project_id, cwd)
        if session is None:
            return None

        with self._lock:
            self._cache[cache_key] = (session, now + _CACHE_TTL_SEC)
        return session

    async def _mint(
        self, user_id: str, project_id: str, cwd: str
    ) -> "boto3.session.Session | None":
        """Mint a fresh scoped boto3 Session via :class:`STSTokenVending`.

        Returns ``None`` when the platform's TokenVending impl is the
        dev no-op (returns ``None`` from ``mint()``).
        """
        from app.platform import get_platform
        import boto3.session

        platform = get_platform()
        creds = await platform.token_vending.mint(user_id, project_id, cwd)
        if creds is None:
            return None

        return boto3.session.Session(
            aws_access_key_id=creds.access_key_id,
            aws_secret_access_key=creds.secret_access_key,
            aws_session_token=creds.session_token,
        )

    def invalidate(self, user_id: str, project_id: str) -> None:
        """Drop a cached session immediately (e.g. on access revocation)."""
        with self._lock:
            self._cache.pop((user_id, project_id), None)

    def _reset(self) -> None:
        """Clear the cache. For testing only."""
        with self._lock:
            self._cache.clear()


_singleton: AwsCredentialCache | None = None


def get_aws_credential_cache() -> AwsCredentialCache:
    """Return the process-wide :class:`AwsCredentialCache` singleton."""
    global _singleton
    if _singleton is None:
        _singleton = AwsCredentialCache()
    return _singleton


__all__ = ["AwsCredentialCache", "get_aws_credential_cache"]
