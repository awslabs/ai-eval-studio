# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""STS-based :class:`TokenVending` implementation.

Mints scoped credentials via ``sts:assume_role`` with session TAGS that
lock the credentials to one project's S3 prefix and the matching DynamoDB
``LeadingKeys`` values. IAM holds every call made with those credentials to
that scope. It does NOT check that the tags name the caller: whoever may
assume the role chooses the tags, so it can mint for any tenant (see the
README's "Security scope").

The effective minted permission set is determined by the session tag values
(``uid``, ``proj``, ``job``) resolving the ``${aws:PrincipalTag/*}``
placeholders in the fixed ``scoped_session_role`` identity policy. This is
ABAC (Attribute-Based Access Control): per-tenant scoping lives in the
session tags, NOT an inline session policy, so the STS packed size stays
small and FLAT as we add actions, well clear of the ``PackedPolicyTooLarge``
limit.

Tag asymmetry (worker vs coordinator):
- **Worker** (job_id set): tags = [{"Key": "uid", "Value": user_id},
  {"Key": "job", "Value": job_id}, {"Key": "proj", "Value": project_id}].
  The worker fences S3 on the ``job``-derived snapshot prefix
  (``-tmp-workspace-{job_id}/*``). Worker writes to the per-job event log
  (``JOB#{job_id}``) AND can allocate a durable project seq counter
  (``PROJECT_SEQ#{project_id}``). The sessions-bucket statements are
  explicitly gated ``Null: {aws:PrincipalTag/job: "true"}`` so the worker's
  proj tag does NOT gain sessions-bucket access.
- **Coordinator** (job_id=None): tags = [{"Key": "uid", "Value": user_id},
  {"Key": "proj", "Value": project_id}]. The coordinator fences S3 on the
  ``uid``+``proj``-derived sessions prefix
  (``-mnt-projects-users-{uid}-projects-{proj}/*``) for the SDK
  ``S3SessionStore``. Coordinator invokes bedrock under its task role, not
  minted creds.

Worker path carries 3 tags, coordinator carries 2. The asymmetry is necessary
because the durable project seq counter's partition key is derived from the
project id, so a tag-resolved LeadingKeys template (``PROJECT_SEQ#${aws:PrincipalTag/proj}``)
is the only way to authorize it without a wildcard.

The role's fixed ``${aws:PrincipalTag/*}`` identity policy (iam_stack.py)
is where scoping is enforced at authorization time. The prefix shapes
(``-mnt-projects-users-...``, ``-tmp-workspace-...``) come from
:func:`claude_agent_sdk.project_key_for_directory` applied to the
coordinator cwd (``/mnt/projects/users/{owner}/projects/{pid}``) and
worker cwd (``/tmp/workspace/{job_id}``). The tag values must reconstruct
these exact prefixes for the role policy templates to match.

NOTE: the minted credentials carry no GSI grant. The store's ``IndexName=``
queries run under the coordinator or AgentCore runtime role, so for those
queries coordinator code is the only enforcement point.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from ..token_vending import ScopedCredentials, TokenVending
from ..types import ProjectId, UserId

if TYPE_CHECKING:  # pragma: no cover - typing only
    from botocore.client import BaseClient

logger = logging.getLogger(__name__)


# STS hard limits. Role chaining (a task role → the scoped-session role) is
# capped at 3600s BY STS. A higher DurationSeconds makes the call fail, and the
# role's MaxSessionDuration cannot lift it.
_MIN_TTL_SEC = 900
_MAX_TTL_SEC = 3600


class STSTokenVending(TokenVending):
    """Cloud :class:`TokenVending` over ``sts:AssumeRole``.

    Args:
        sts_client: Low-level boto3 STS client. Inject in tests.
        role_arn: ARN of the role assumed for scoped sessions. The role's
            trust policy must allow this app's task role to assume it, and
            its identity policy is templated on ``${aws:PrincipalTag/*}``, so
            the session tags passed at mint time decide the effective scope.
        sessions_bucket: S3 bucket holding the SDK's session JSONL
            part files. The tag-resolved role policy narrows access to one
            project's SDK-canonicalized prefix under this bucket.
        snapshots_bucket: S3 bucket holding per-job working-tree snapshots,
            reached by the worker under its job-derived prefix.
        table_name: DynamoDB single-table name (default ``eval-studio``).
    """

    def __init__(
        self,
        sts_client: "BaseClient",
        role_arn: str,
        sessions_bucket: str,
        snapshots_bucket: str,
        table_name: str = "eval-studio",
    ) -> None:
        """Initialize the STS token vending machine.

        Args:
            sts_client: Boto3 STS client for AssumeRole calls.
            role_arn: ARN of the role to assume (scoped_session_role).
            sessions_bucket: S3 bucket holding SDK session JSONL part files.
            snapshots_bucket: S3 bucket holding per-job working-tree snapshots
                that the worker downloads via minted credentials.
            table_name: DynamoDB table name (default: eval-studio).
        """
        self.sts = sts_client
        self.role_arn = role_arn
        self.sessions_bucket = sessions_bucket
        self.snapshots_bucket = snapshots_bucket
        self.table_name = table_name

    def build_session_tags(
        self, user_id: UserId, project_id: ProjectId, job_id: str | None = None
    ) -> list[dict]:
        """Return the STS session tags list for ``(user_id, project_id, job_id)``.

        Public so unit tests can assert the tag structure without invoking STS.

        Session tags replace the inline session policy — the role's fixed
        ``${aws:PrincipalTag/*}`` identity policy (iam_stack) is where scoping
        is enforced; these tag values resolve the placeholders at authorization
        time.

        Tag asymmetry (intentional and load-bearing):
        - **Worker** (job_id set): tags = [uid, job, proj]. Worker fences S3 on the
          ``job``-derived snapshot prefix (``-tmp-workspace-{job_id}/*``) and can
          allocate durable project seq (``PROJECT_SEQ#{project_id}``). The proj tag
          resolves the LeadingKeys template; sessions-bucket statements are gated
          ``Null: {aws:PrincipalTag/job: "true"}`` to prevent access.
        - **Coordinator** (job_id=None): tags = [uid, proj]. Coordinator fences
          S3 on the ``uid``+``proj``-derived sessions prefix
          (``-mnt-projects-users-{uid}-projects-{proj}/*``) for the SDK
          S3SessionStore.

        Worker carries 3 tags, coordinator carries 2. The asymmetry is necessary
        because the project seq counter's key is derived from project_id, so
        a tag-resolved LeadingKeys entry is the only authorization without a
        wildcard.

        Args:
            user_id: Authenticated owner sub (uid tag).
            project_id: Project ID (proj tag on BOTH paths — the worker needs it
                to resolve the ``PROJECT_SEQ#`` LeadingKeys template).
            job_id: Job ID (job tag for worker, None for coordinator).

        Returns:
            List of tag dicts (Key, Value) for STS Tags kwarg.
        """
        # Base: uid tag is always present
        tags = [{"Key": "uid", "Value": user_id}]

        # Worker path: add job AND proj tags (worker needs proj for PROJECT_SEQ# allocation)
        if job_id:
            tags.append({"Key": "job", "Value": job_id})
            tags.append({"Key": "proj", "Value": project_id})
        # Coordinator path: add proj tag only
        else:
            tags.append({"Key": "proj", "Value": project_id})

        return tags

    @staticmethod
    def _session_name(user_id: UserId, project_id: ProjectId) -> str:
        # STS limits role-session-name to 64 chars and
        # ``[\w+=,.@-]+``. We truncate aggressively to keep within limits
        # for long Cognito sub UUIDs.
        truncated = f"{user_id[:30]}-{project_id[:30]}"
        # Replace anything outside the allowed alphabet.
        allowed = []
        for ch in truncated:
            if ch.isalnum() or ch in "+=,.@-_":
                allowed.append(ch)
            else:
                allowed.append("-")
        return "".join(allowed)[:64] or "scoped-session"

    async def mint(
        self,
        user_id: UserId,
        project_id: ProjectId,
        cwd: str,
        *,
        ttl_sec: int = 3600,
        job_id: str | None = None,
    ) -> ScopedCredentials | None:
        ttl = max(_MIN_TTL_SEC, min(int(ttl_sec), _MAX_TTL_SEC))
        tags = self.build_session_tags(user_id, project_id, job_id=job_id)
        session_name = self._session_name(user_id, project_id)

        def _assume() -> ScopedCredentials:
            # Effective minted permission set = session tag values resolving
            # the ${aws:PrincipalTag/*} placeholders in the fixed
            # scoped_session_role identity policy. This is ABAC: per-tenant
            # scoping lives in the session tags, NOT an inline session policy,
            # so the STS packed size stays small and FLAT as we add actions.
            resp = self.sts.assume_role(
                RoleArn=self.role_arn,
                RoleSessionName=session_name,
                DurationSeconds=ttl,
                Tags=tags,
            )
            # Log the packed policy size so operators can measure real headroom
            # (even with tags, STS returns PackedPolicySize — load-bearing
            # observability).
            packed = resp.get("PackedPolicySize")
            logger.info(
                "[sts] minted session PackedPolicySize=%s%% job_id=%s",
                packed, job_id
            )
            if packed is not None and packed >= 80:
                logger.warning(
                    "[sts] session PackedPolicySize high (%s%%) job_id=%s",
                    packed, job_id
                )
            creds = resp["Credentials"]
            expiration = creds["Expiration"]
            iso = (
                expiration.isoformat()
                if hasattr(expiration, "isoformat")
                else str(expiration)
            )
            return ScopedCredentials(
                access_key_id=creds["AccessKeyId"],
                secret_access_key=creds["SecretAccessKey"],
                session_token=creds["SessionToken"],
                expiration=iso,
            )

        return await asyncio.to_thread(_assume)


__all__ = ["STSTokenVending"]
