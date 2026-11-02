# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""TokenVending Protocol — STS-issued, per-request scoped credentials.

In cloud mode, the agent turn's transcript store uses scoped credentials
minted via ``sts:AssumeRole`` with session tags (``uid`` and ``proj``, plus
``job`` for an eval worker, which mints the same shape of credential for
itself). The tags resolve the ``${aws:PrincipalTag/*}`` placeholders in the
scoped-session role's policy, narrowing the credentials to one project's S3
prefix (the SDK's canonicalized cwd path) and to DynamoDB ``LeadingKeys``
``USER#{uid}`` and ``PROJECT_SEQ#{project_id}`` (plus ``JOB#{job_id}`` for a
worker). The caller of ``mint`` chooses the tags; IAM does not bind them to
the caller's identity.

In dev mode, this returns ``None`` — boto3 falls back to the developer's
ambient credentials, and there's no multi-tenancy to scope to.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .types import ProjectId, UserId


@dataclass(frozen=True)
class ScopedCredentials:
    """Short-lived AWS credentials returned by :meth:`TokenVending.mint`.

    ``None`` is returned in dev mode; cloud callers should treat ``None``
    as "use ambient creds" rather than crashing, so the same code path
    works locally without changes.
    """

    access_key_id: str
    secret_access_key: str
    session_token: str
    expiration: str  # ISO-8601


@runtime_checkable
class TokenVending(Protocol):
    """Mints scoped AWS credentials for one user's one project."""

    async def mint(
        self,
        user_id: UserId,
        project_id: ProjectId,
        cwd: str,
        *,
        ttl_sec: int = 3600,
        job_id: str | None = None,
    ) -> ScopedCredentials | None:
        """Return scoped credentials, or ``None`` in dev mode.

        Args:
            user_id: User whose data the credentials may access.
            project_id: Project the credentials may access. Used in the
                DynamoDB ``LeadingKeys`` constraint.
            cwd: Filesystem path the agent will run in. The Claude
                Agent SDK's ``S3SessionStore`` keys transcripts under
                ``project_key_for_directory(cwd)`` (every non-alphanumeric
                replaced with ``-``). The role policy's tag template
                rebuilds that prefix from the ``uid`` and ``proj`` tags, so
                the grant only matches while cwd follows the
                ``/mnt/projects/users/{uid}/projects/{pid}`` layout; the STS
                implementation does not read this argument itself.
            ttl_sec: Credential lifetime. Minimum 900s, maximum 3600s
                (1h) for role-chained credentials (the scoped-session role
                is assumed by the coordinator's task role, triggering the
                role-chaining cap). Direct assume-role can reach 12h.
            job_id: Job whose event log the credentials may append to.
                When set, the minted credentials additionally allow
                DynamoDB base-table access under ``JOB#{job_id}`` so a
                worker can append to the per-job event log (keyed
                ``PK=JOB#{job_id}``). When ``None`` (the coordinator path),
                ``USER#{uid}`` and ``PROJECT_SEQ#{project_id}`` are allowed.
        """
        ...


__all__ = ["ScopedCredentials", "TokenVending"]
