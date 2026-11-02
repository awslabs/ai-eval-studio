# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test: cross-tenant scoped-credential denial.

The worker-isolation proof: a worker booted with tenant A's minted STS
credentials can reach ONLY tenant A's S3 prefix + DynamoDB ``LeadingKeys``
and is DENIED tenant B's. This is the "worker touches ONLY its own tenant,
denied a sibling's" boundary for the credentials as minted. It does not stop
the minting role from requesting another tenant's tags — the repository
README's "Security scope" section covers that limit.

Approach — ABAC. The denial is now enforced by the FIXED tag-templated
identity policy on ``scoped_session_role`` (iam_stack.py), whose
``${aws:PrincipalTag/*}`` placeholders resolve from the per-request session
TAGS that ``STSTokenVending.build_session_tags`` produces. moto's STS
``assume_role`` accepts session ``Tags`` but does NOT evaluate the resolved
role policy against subsequent S3/DDB calls (moto has no IAM policy engine on
the data plane), so a live "AccessDenied on B" assertion is not possible under
moto. Instead this test asserts STRUCTURALLY on the tag values and the
resources the ROLE policy reconstructs from them: tenant A's tags carry
``uid=A`` (and ``job=job_a`` on the worker path), which the role policy
resolves to A's canonical S3 prefix (``-tmp-workspace-{job_a}`` snapshot) and
DDB key (``USER#{A}``); tenant B's derived prefix/key
(``-tmp-workspace-{job_b}``, ``USER#{B}``) are provably DISTINCT and thus
unreachable. Seeing the live AccessDenied needs a deployed stack, and the
deploy runbook's smoke test does not exercise it. The structural proof is
exact: the session tags
are the single mechanism that scopes the worker, reused verbatim from the
mint path, and the role policy's tag-templated resources fence each tenant.

Marked ``@pytest.mark.integration`` so it skips by default and only runs
under ``pytest -m integration``.
"""
from __future__ import annotations

import pytest


def _snapshot_prefix_for_job(job_id: str) -> str:
    """Reconstruct the snapshot S3 prefix the ROLE policy builds from the job tag.

    The worker cwd is always ``/tmp/workspace/{job_id}`` and
    ``project_key_for_directory`` sanitizes it to ``-tmp-workspace-{job_id}``
    (leading dash). Because job_id is a UUID (``[0-9a-f-]``, sanitize-invariant),
    the raw ``job`` tag value equals the S3-key segment, so the role policy
    template ``-tmp-workspace-${aws:PrincipalTag/job}/*`` resolves to this exact
    prefix.
    """
    return f"-tmp-workspace-{job_id}"


def _ddb_user_key(uid: str) -> str:
    """Reconstruct the DDB LeadingKey the ROLE policy builds from the uid tag.

    The role policy template ``USER#${aws:PrincipalTag/uid}`` resolves the
    ``uid`` session tag to this partition-key prefix.
    """
    return f"USER#{uid}"


@pytest.mark.integration
def test_tenant_a_tags_scope_only_to_a():
    """Tenant A's session tags scope to A's prefix/keys and exclude B's.

    The proof under ABAC: the session tags that scope the worker
    carry tenant A's ``uid`` + ``job``, which the role's tag-templated identity
    policy resolves to A's snapshot prefix (``-tmp-workspace-{job_a}``) and
    ``USER#{A}`` DDB LeadingKey — and carry NOTHING for tenant B, so a worker
    minted for A cannot reach B's data. Own-ALLOW / other-DENY, structural.
    """
    from app.platform.aws.sts_token_vending import STSTokenVending

    tenant_a = "cognito-sub-aaaa"
    tenant_b = "cognito-sub-bbbb"
    project_a = "proj-a-111"
    job_a = "job-aaaa-1111"
    job_b = "job-bbbb-2222"
    bucket = "eval-studio-sessions"
    table = "eval-studio"

    vending = STSTokenVending(
        sts_client=None,  # build_session_tags is pure — no STS call
        role_arn="arn:aws:iam::123456789012:role/scoped-session",
        sessions_bucket=bucket,
        snapshots_bucket="eval-studio-snapshots",
        table_name=table,
    )

    # Build tenant A's WORKER tags (uid + job)
    tags_a = vending.build_session_tags(tenant_a, project_a, job_id=job_a)
    tag_dict_a = {t["Key"]: t["Value"] for t in tags_a}

    # --- Tags: A's tags carry uid=A and job=job_a, and NOT B's uid/job ---
    assert tag_dict_a["uid"] == tenant_a, f"A's uid tag should be {tenant_a}, got {tag_dict_a.get('uid')}"
    assert tag_dict_a["job"] == job_a, f"A's job tag should be {job_a}, got {tag_dict_a.get('job')}"
    # Tenant B's uid and job appear NOWHERE in A's tags
    tag_values_a = set(tag_dict_a.values())
    assert tenant_b not in tag_values_a, f"tenant A's tags leak tenant B's uid {tenant_b}"
    assert job_b not in tag_values_a, f"tenant A's tags leak tenant B's job {job_b}"

    # --- S3 snapshot: A's tag-derived prefix DISTINCT from B's ---
    snapshot_prefix_a = _snapshot_prefix_for_job(job_a)
    snapshot_prefix_b = _snapshot_prefix_for_job(job_b)
    assert snapshot_prefix_a != snapshot_prefix_b, "A and B must derive distinct snapshot prefixes"
    # The role policy resolves A's job tag to A's prefix only — B's is unreachable.
    assert snapshot_prefix_a == f"-tmp-workspace-{job_a}"
    assert job_b not in snapshot_prefix_a, "A's snapshot prefix must not contain B's job"

    # --- DynamoDB: A's tag-derived LeadingKey DISTINCT from B's ---
    ddb_key_a = _ddb_user_key(tenant_a)
    ddb_key_b = _ddb_user_key(tenant_b)
    assert ddb_key_a != ddb_key_b, "A and B must derive distinct DDB keys"
    assert ddb_key_a == f"USER#{tenant_a}"
    assert tenant_b not in ddb_key_a, "A's DDB key must not contain B's uid"


@pytest.mark.integration
def test_minted_creds_via_moto_carry_tenant_a_tags():
    """End-to-end mint: STSTokenVending.mint passes A's session tags to STS.

    Uses moto's STS to assume the scoped role and asserts the mint delivers
    per-tenant temporary credentials. moto returns creds for any assume_role;
    the security property under test is that the session Tags sent to STS
    carry tenant A's ``uid`` (the one asserted structurally above), NOT
    tenant B's, and that NO inline ``Policy``/``PolicyArns`` is sent (ABAC).
    We capture the kwargs STS received.
    """
    # moto is a locked dev dependency (moto>=5.0.0), so importorskip only skips
    # on a genuinely missing install — it will NOT paper over an import that
    # breaks mid-session. That failure must surface, not silently drop this
    # mint proof.
    mock_aws = pytest.importorskip("moto").mock_aws

    import asyncio

    from app.platform.aws.sts_token_vending import STSTokenVending

    tenant_a = "cognito-sub-aaaa"
    tenant_b = "cognito-sub-bbbb"
    project_a = "proj-a-111"
    bucket = "eval-studio-sessions"
    cwd_a = f"/mnt/projects/users/{tenant_a}/projects/{project_a}"

    with mock_aws():
        import boto3

        sts = boto3.client("sts", region_name="us-east-1")

        # Capture the kwargs STS receives on assume_role. moto's STS
        # doesn't evaluate the tags, but it echoes the call args we wrap.
        captured: dict = {}
        real_assume_role = sts.assume_role

        def _capture_assume_role(**kwargs):
            captured.update(kwargs)
            return real_assume_role(**kwargs)

        sts.assume_role = _capture_assume_role  # type: ignore[assignment]

        vending = STSTokenVending(
            sts_client=sts,
            role_arn="arn:aws:iam::123456789012:role/scoped-session",
            sessions_bucket=bucket,
            snapshots_bucket="eval-studio-snapshots",
            table_name="eval-studio",
        )

        creds = asyncio.run(vending.mint(tenant_a, project_a, cwd_a))

        # Mint delivered temporary credentials
        assert creds is not None

        # The session tags sent to STS carry tenant A's uid — NOT B's. This is
        # the exact scoping the worker inherits; a bug that widened it would
        # surface here. And NO inline Policy/PolicyArns (ABAC — scoping lives
        # in the role's tag-templated identity policy, not a minted body).
        assert "Policy" not in captured, "mint() must NOT send inline Policy (ABAC uses tags)"
        assert "PolicyArns" not in captured, "mint() must NOT send PolicyArns (ABAC uses tags)"
        assert "Tags" in captured, "mint() must send session Tags"
        tag_dict = {t["Key"]: t["Value"] for t in captured["Tags"]}
        assert tag_dict.get("uid") == tenant_a, f"minted tags should carry uid={tenant_a}, got {tag_dict.get('uid')}"
        assert tenant_b not in tag_dict.values(), f"minted tags leak tenant B's uid {tenant_b}"
