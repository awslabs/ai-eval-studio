# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AWS-backed platform adapters (cloud mode).

This package contains the cloud-mode implementations of the
``app.platform`` Protocols, mirroring the dev-mode adapters in
:mod:`app.platform.dev_local`.

Modules:
- :mod:`dynamodb_metadata_store` — :class:`MetadataStore` over single-table
  DynamoDB. The ONLY module allowed to issue raw GSI queries
  (``IndexName=...``).
- :mod:`s3_session_store` — Claude Agent SDK :class:`SessionStore` over S3
- :mod:`sts_token_vending` — :class:`TokenVending` via STS AssumeRole with
  inline session policies.
- :mod:`auth_cognito_bearer` — :class:`Auth` validating Cognito
  ``Authorization: Bearer <jwt>`` access tokens against the user-pool
  JWKS (in-app JWT, not ALB OIDC).

Use :func:`build_cloud_platform` to assemble a :class:`Platform` from
environment-driven configuration. Boto3 clients are constructed lazily
inside the factory so importing this package has near-zero cost.
"""
from __future__ import annotations

import os

from botocore.config import Config

from app.platform.aws.s3_session_store import LOAD_CONCURRENCY


def build_cloud_platform():
    """Construct the cloud :class:`Platform` from environment variables.

    Reads (all required unless noted):
        PROJECTS_DIR: Mount point for the EFS file system that holds
            project working trees, e.g. ``/mnt/projects``. The backend
            resolves ``{PROJECTS_DIR}/users/{owner_sub}/projects/{pid}/``
            per-request via ``app.services.project_paths.resolve_project_dir``.
        S3_SESSIONS_BUCKET: Bucket holding SDK session JSONL part files.
        S3_UPLOADS_STAGING_BUCKET: Bucket for ≥50 MB browser uploads via
            presigned PUT. Backend copies into EFS, then deletes.
        S3_SNAPSHOTS_BUCKET: Bucket holding per-job working-tree snapshots
            that the worker downloads via minted credentials.
        DDB_TABLE: Single-table DynamoDB name (default ``eval-studio``).
        SCOPED_SESSION_ROLE_ARN: Role assumed by :class:`STSTokenVending` to
            mint per-request narrowed credentials (also called the
            token-vending, or "TVM", role).
        AWS_REGION: Default region for clients (default ``us-east-1``).
        COGNITO_REGION: Region of the Cognito user pool (default:
            ``AWS_REGION``). Used by :class:`CognitoBearerJwtAuth` to
            build the JWKS issuer URL.
        COGNITO_USER_POOL_ID: User pool id, validated against the ``iss``
            claim on every access token.
        COGNITO_SPA_CLIENT_ID: SPA Cognito user-pool client id; validated
            against the ``client_id`` claim on every access token.

    Returns:
        A :class:`app.platform.Platform` wired with AWS-backed adapters.

    Raises:
        RuntimeError: A required env var is missing.
    """
    # Defer boto3 import so dev-mode startups don't pay the boto3 cost.
    import boto3

    from .. import Platform
    from .auth_cognito_bearer import CognitoBearerJwtAuth
    from .dynamodb_metadata_store import DynamoDBMetadataStore
    from .s3_session_store import S3SessionStore
    from .sts_token_vending import STSTokenVending

    region = os.environ.get("AWS_REGION", "us-east-1")
    cognito_region = os.environ.get("COGNITO_REGION", region)
    # Cognito user pool + SPA client are required in cloud mode — auth
    # fails closed if either is missing.
    cognito_user_pool_id = _require_env("COGNITO_USER_POOL_ID")
    cognito_spa_client_id = _require_env("COGNITO_SPA_CLIENT_ID")

    # PROJECTS_DIR is required in cloud mode — it's the EFS mount root.
    # Bootstrap fails if EFS isn't mounted at the expected path.
    projects_dir = _require_env("PROJECTS_DIR")
    sessions_bucket = _require_env("S3_SESSIONS_BUCKET")
    uploads_staging_bucket = _require_env("S3_UPLOADS_STAGING_BUCKET")
    snapshots_bucket = _require_env("S3_SNAPSHOTS_BUCKET")
    table_name = os.environ.get("DDB_TABLE", "eval-studio")
    scoped_session_role_arn = _require_env("SCOPED_SESSION_ROLE_ARN")

    boto_session = boto3.session.Session(region_name=region)

    s3_client = boto_session.client(
        "s3", config=Config(max_pool_connections=LOAD_CONCURRENCY)
    )
    ddb_client = boto_session.client("dynamodb")
    sts_client = boto_session.client("sts")

    metadata_store = DynamoDBMetadataStore(
        ddb_client=ddb_client, table_name=table_name
    )
    session_store = S3SessionStore(s3_client=s3_client, bucket=sessions_bucket)
    token_vending = STSTokenVending(
        sts_client=sts_client,
        role_arn=scoped_session_role_arn,
        # Both the sessions bucket and snapshots bucket are named in the
        # per-request inline session policy. The worker reads its snapshot
        # from the snapshots bucket under the same tenant prefix that the
        # session policy uses (both computed via project_key_for_directory
        # from cwd=/tmp/workspace/{job_id}).
        sessions_bucket=sessions_bucket,
        snapshots_bucket=snapshots_bucket,
        table_name=table_name,
    )
    auth = CognitoBearerJwtAuth(
        metadata_store=metadata_store,
        region=cognito_region,
        user_pool_id=cognito_user_pool_id,
        client_id=cognito_spa_client_id,
    )

    # Project files live on EFS at PROJECTS_DIR and are accessed via real
    # Path APIs from ``app.services.project_paths.resolve_project_dir`` —
    # there is no project-store Platform field.
    return Platform(
        metadata_store=metadata_store,
        session_store=session_store,
        token_vending=token_vending,
        auth=auth,
    )


def _require_env(name: str) -> str:
    """Return ``os.environ[name]`` or raise a clear error.

    Args:
        name: Environment variable to read.

    Returns:
        The environment variable's string value.

    Raises:
        RuntimeError: ``name`` is unset or empty.
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Cloud mode requires {name} to be set. See "
            "app/platform/aws/__init__.py docstring for the full env list."
        )
    return value


__all__ = ["build_cloud_platform"]
