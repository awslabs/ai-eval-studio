# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""credential_process helper — mint worker STS credentials with session tags.

Invoked by botocore's ``ProcessProvider`` via the credential_process line in the
AWS config written by ``worker_credentials_config.py``. Assumes the
scoped-session role with ``uid`` + ``job`` + ``proj`` session tags (the worker
path) and emits a botocore Version-1 JSON credential document to stdout.

**Self-bootstrapping import path:** Python sets ``sys.path[0]`` to the
**script's own directory** for a script invocation — here ``app/scripts`` — not
to the package root, so ``app.*`` is NOT importable even though this file lives
inside the ``app`` package. The package root must therefore be inserted
explicitly on ``sys.path`` before any ``app.*`` import. The root is derived
from ``__file__`` rather than a constant so the helper is correct at any tree
prefix (the image's ``/app``, a dev checkout, a test copy). This is what lets
the config name the script by absolute path instead of relying on ``-m``.

**Recursion prevention:** The helper constructs a boto3 session to call STS,
which would normally resolve the same ``AWS_CONFIG_FILE`` and re-invoke this
script — infinite recursion. To prevent this, the helper deletes
``AWS_CONFIG_FILE`` from ``os.environ`` before creating its session, so the
session resolves the container task role (ECS credential endpoint, chain
position #11) instead of looping back to this script.

**Expiration field is mandatory:** botocore's ``ProcessProvider.load`` branches
on ``if creds_dict.get('expiry_time') is not None:`` to decide whether to
return a ``RefreshableCredentials`` (refreshable) or a plain ``Credentials``
(frozen). The helper emits the JSON key ``Expiration`` (ISO-8601), and
botocore maps it to the parsed-dict key ``expiry_time`` that the branch tests.
Omitting ``Expiration`` yields a frozen credential — silently reintroducing the
exact bug this guards against.

**Tag source:** Delegates to ``STSTokenVending.build_session_tags(owner_id,
project_id, job_id=job_id)`` so the tag shape has exactly ONE definition and
cannot drift from the coordinator's mint path. The worker and coordinator must
mint with identical tag shapes for the ABAC scoping to work.

**PackedPolicySize measurement:** Logs the STS ``PackedPolicySize`` field (a
percentage of the packed limit) from the assume_role response, so the headroom
of the worker's three session tags (uid, job, and proj, which authorizes the
PROJECT_SEQ# counter) is measured rather than assumed. Session tags pack far
smaller than an inline session policy, but role chaining still has a packed
limit.

CLI args:
    --owner-id: Authenticated owner sub (uid tag).
    --project-id: Project ID (proj tag; authorizes the project seq counter).
    --job-id: Job ID (job tag).
    --role-arn: ARN of the scoped-session role to assume.

Stdout:
    Version-1 JSON credential document with four keys: ``AccessKeyId``,
    ``SecretAccessKey``, ``SessionToken``, ``Expiration``. botocore parses this
    and returns a ``RefreshableCredentials`` provider.

Stderr:
    Error messages on failure. botocore surfaces these as
    ``CredentialRetrievalError``.

Exit:
    0 on success, non-zero on failure.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)


def main() -> None:
    """Parse args, assume role, emit Version-1 JSON credential document."""
    parser = argparse.ArgumentParser(
        description="Mint worker credentials via STS assume-role with session tags"
    )
    parser.add_argument("--owner-id", required=True, help="Owner sub (uid tag)")
    parser.add_argument("--project-id", required=True, help="Project ID")
    parser.add_argument("--job-id", required=True, help="Job ID (job tag)")
    parser.add_argument("--role-arn", required=True, help="Scoped-session role ARN")
    args = parser.parse_args()

    # Recursion prevention: delete AWS_CONFIG_FILE from our own environment
    # before creating any boto3 session. Without this, the session would
    # resolve the same credential_process and re-invoke this script — infinite
    # loop. The deletion forces the session to resolve the container task role
    # (ECS credential endpoint, chain position #11) instead.
    if "AWS_CONFIG_FILE" in os.environ:
        del os.environ["AWS_CONFIG_FILE"]

    try:
        # Bootstrap: insert the package root on sys.path so 'import app.*' works
        # when invoked as a script from any cwd. For script invocations, Python
        # sets sys.path[0] to the script's directory (app/scripts), not the
        # package root. Derive the root from __file__ (two parents up from this
        # script) and insert at position 0, ahead of any PYTHONPATH entries,
        # only if not already present (idempotent for nested calls).
        from pathlib import Path
        package_root = Path(__file__).resolve().parents[2]
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))

        # Import boto3 and STSTokenVending AFTER clearing AWS_CONFIG_FILE so
        # no module-level import can trigger a credential resolution that loops.
        import boto3

        from app.platform.aws.sts_token_vending import STSTokenVending

        # Build session tags via STSTokenVending so the tag shape is defined in
        # exactly ONE place and cannot drift from the coordinator's mint path.
        tags = STSTokenVending.build_session_tags(
            None,  # STSTokenVending signature takes a dummy instance
            args.owner_id,
            args.project_id,
            job_id=args.job_id,
        )

        # Assume the scoped-session role with session tags. DurationSeconds at
        # the 3600s ceiling (role-chaining cap). The session name is a
        # human-readable identifier for CloudTrail; it does not affect
        # authorization.
        session = boto3.Session()
        sts = session.client("sts")
        resp = sts.assume_role(
            RoleArn=args.role_arn,
            RoleSessionName=f"worker-{args.job_id}",
            DurationSeconds=3600,
            Tags=tags,
        )

        creds = resp["Credentials"]
        expiration = creds["Expiration"]

        # Log PackedPolicySize so the third tag's headroom is measured on the path
        # that changed. STS returns this as a PERCENTAGE of the allowed packed
        # size, 0-100 — not a character count. An earlier revision compared it
        # against 930 (the ~80% mark of the ~1163-char ceiling measured for role
        # chaining), which no valid response can exceed, so the warning could
        # never fire. Warn at 80% instead, matching STSTokenVending.
        packed_size = resp.get("PackedPolicySize", 0)
        packed_threshold_pct = 80
        log_msg = f"[worker-credentials] PackedPolicySize={packed_size}%"
        if packed_size >= packed_threshold_pct:
            log_msg += f" (at or above {packed_threshold_pct}% of the session-tag budget)"
        print(log_msg, file=sys.stderr)

        # Serialize Expiration as ISO-8601. botocore expects the string form;
        # boto3 may return a datetime object depending on the SDK version.
        if hasattr(expiration, "isoformat"):
            expiration_str = expiration.isoformat()
        else:
            expiration_str = str(expiration)

        # Emit Version-1 JSON credential document. The four keys are mandatory
        # for botocore's ProcessProvider. Expiration is what makes the
        # credential refreshable — its presence is the sole condition for
        # botocore returning a RefreshableCredentials instead of a frozen
        # Credentials (verified against the installed library).
        doc = {
            "Version": 1,
            "AccessKeyId": creds["AccessKeyId"],
            "SecretAccessKey": creds["SecretAccessKey"],
            "SessionToken": creds["SessionToken"],
            "Expiration": expiration_str,
        }

        # Write to stdout as a single JSON line. botocore parses this and
        # caches the provider for the process lifetime.
        print(json.dumps(doc), flush=True)

    except Exception as e:
        # On failure, write the reason to stderr and exit non-zero so botocore
        # surfaces a CredentialRetrievalError rather than a silent empty credential.
        print(f"mint_worker_credentials failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
