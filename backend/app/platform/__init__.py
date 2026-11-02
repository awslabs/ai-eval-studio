# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform abstraction layer for cloud deployment.

The ``Platform`` dataclass is the single dependency-injection point for the
backend's stateful infrastructure: metadata, agent sessions, scoped
credentials, and auth. Each field is a ``Protocol`` so we can swap dev-mode
(filesystem-backed) implementations for cloud-mode (DynamoDB / S3 / STS /
Cognito) without touching call sites. Real-time delivery is not a Platform
field: it goes through ``app.services.ws_hub`` in both modes.

Project files are not a Platform field: they live on the local filesystem
in dev and on the EFS mount in cloud, and call sites reach them directly
through ``app.services.project_paths.resolve_project_dir``.

The mode is selected by ``EVAL_STUDIO_MODE`` (``dev`` | ``cloud``).
``dev`` uses the filesystem-backed adapters in ``platform.dev_local``.
``cloud`` flips every adapter to the AWS-backed implementations from
``platform.aws``.

This module exposes ``get_platform()`` (lazy singleton) and ``bootstrap()``
(explicit construction for tests). Most call sites should depend on the
specific Protocol interface they need rather than the whole Platform.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .auth import Auth
from .metadata_store import MetadataStore
from .session_store import SessionStoreProtocol
from .token_vending import TokenVending


@dataclass(frozen=True)
class Platform:
    """Bundle of platform-level Protocol implementations.

    Held as a process-wide singleton via :func:`get_platform`. Tests can
    construct an instance directly with mock adapters and pass it to
    :func:`set_platform` to override the singleton for the duration of
    the test.

    Attributes:
        metadata_store: Project records, jobs, optimization state, cost rollups.
        session_store: Claude Agent SDK session persistence. Conforms to the
            SDK's ``SessionStore`` Protocol so it plugs into ``ClaudeAgentOptions``.
        token_vending: Mints scoped credentials for per-request data access in
            cloud mode. In dev mode this returns ``None`` (use ambient creds).
        auth: Extract user_id from the request, enforce ``require_access``.
    """

    metadata_store: MetadataStore
    # SessionStore is optional: dev mode returns None and the SDK falls
    # back to its built-in on-disk session store. Cloud mode plugs in
    # S3SessionStore for cross-task durability.
    session_store: SessionStoreProtocol | None
    token_vending: TokenVending
    auth: Auth


_platform: Platform | None = None


def get_platform() -> Platform:
    """Return the process-wide :class:`Platform` singleton.

    Lazily constructs on first call by reading ``EVAL_STUDIO_MODE`` from
    the environment. Subsequent calls return the same instance.

    Returns:
        The configured :class:`Platform`.
    """
    global _platform
    if _platform is None:
        _platform = bootstrap()
    return _platform


def set_platform(platform: Platform | None) -> None:
    """Override the singleton (for tests) or clear it.

    Args:
        platform: Replacement Platform, or ``None`` to force a fresh
            ``bootstrap()`` on the next ``get_platform()`` call.
    """
    global _platform
    _platform = platform


def bootstrap() -> Platform:
    """Build a fresh :class:`Platform` based on ``EVAL_STUDIO_MODE``.

    ``dev`` (default) wires the filesystem-backed adapters in
    :mod:`app.platform.dev_local`. ``cloud`` wires the AWS-backed
    adapters in :mod:`app.platform.aws`.

    Returns:
        A new :class:`Platform`.

    Raises:
        ValueError: ``EVAL_STUDIO_MODE`` is set to an unrecognized value.
    """
    mode = os.environ.get("EVAL_STUDIO_MODE", "dev").lower()
    if mode == "dev":
        from .dev_local import build_dev_platform
        return build_dev_platform()
    if mode == "cloud":
        from .aws import build_cloud_platform
        return build_cloud_platform()
    raise ValueError(
        f"Unknown EVAL_STUDIO_MODE={mode!r}. Expected 'dev' or 'cloud'."
    )


__all__ = [
    "Platform",
    "get_platform",
    "set_platform",
    "bootstrap",
]
