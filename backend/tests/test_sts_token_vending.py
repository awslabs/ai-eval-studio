# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for STSTokenVending session tags structure.

These tests assert the RENDERED tag structure (uid+proj for coordinator,
uid+job+proj for worker, no inline Policy/PolicyArns on mint) without invoking
live AWS. They guard the ABAC tag structure; the IAM policy that consumes the
tags is not exercised here.
"""

import asyncio
from unittest.mock import MagicMock, patch

from app.platform.aws.sts_token_vending import STSTokenVending


def test_build_session_tags_coordinator_has_uid_and_proj():
    """Coordinator path (job_id=None) builds tags = [uid, proj].

    Regression guard: coordinator fences S3 on uid+proj-derived sessions
    prefix (-mnt-projects-users-{uid}-projects-{proj}/*), so both tags
    are required.
    """
    sts = MagicMock()
    tvm = STSTokenVending(
        sts_client=sts,
        role_arn="arn:aws:iam::123:role/scoped-session-role",
        sessions_bucket="my-sessions",
        snapshots_bucket="my-snapshots",
        table_name="eval-studio",
    )

    # Build coordinator tags (job_id=None)
    tags = tvm.build_session_tags("user-xyz", "proj-abc", job_id=None)

    # Assert exactly 2 tags
    assert len(tags) == 2, f"Coordinator should have 2 tags, got {len(tags)}"

    # Extract tag dict by Key
    tag_dict = {t["Key"]: t["Value"] for t in tags}

    # Assert uid and proj are present
    assert "uid" in tag_dict, "Coordinator tags missing 'uid'"
    assert "proj" in tag_dict, "Coordinator tags missing 'proj'"
    assert tag_dict["uid"] == "user-xyz", f"uid should be 'user-xyz', got {tag_dict['uid']}"
    assert tag_dict["proj"] == "proj-abc", f"proj should be 'proj-abc', got {tag_dict['proj']}"

    # Assert job is NOT present
    assert "job" not in tag_dict, "Coordinator tags should NOT have 'job'"


def test_build_session_tags_worker_has_uid_job_and_proj():
    """Worker path (job_id set) builds tags = [uid, job, proj].

    Regression guard: the worker fences S3 on the job-derived snapshot prefix
    (-tmp-workspace-{job}/*), and it also needs ``proj`` so the scoped-session
    role's ``PROJECT_SEQ#${aws:PrincipalTag/proj}`` LeadingKeys template resolves
    — without it the worker cannot allocate a durable project seq and its
    publishes fall back to a seq the client discards.

    The worker gaining ``proj`` does NOT widen its S3 reach: the sessions-bucket
    statements carry an explicit ``Null: {aws:PrincipalTag/job: "true"}`` gate,
    asserted on the infra side.
    """
    sts = MagicMock()
    tvm = STSTokenVending(
        sts_client=sts,
        role_arn="arn:aws:iam::123:role/scoped-session-role",
        sessions_bucket="my-sessions",
        snapshots_bucket="my-snapshots",
        table_name="eval-studio",
    )

    # Build worker tags (job_id set)
    tags = tvm.build_session_tags("user-xyz", "proj-abc", job_id="job-777")

    # Assert exactly 3 tags
    assert len(tags) == 3, f"Worker should have 3 tags, got {len(tags)}"

    # Extract tag dict
    tag_dict = {t["Key"]: t["Value"] for t in tags}

    # Assert uid, job, and proj are all present with the right values
    assert tag_dict == {
        "uid": "user-xyz",
        "job": "job-777",
        "proj": "proj-abc",
    }, f"Worker tags should be {{uid, job, proj}}, got {tag_dict}"


def test_mint_sends_tags_not_policy():
    """mint() calls assume_role with Tags kwarg, NO Policy or PolicyArns.

    Regression guard: ABAC migration replaces inline Policy with session Tags.
    The minted request must carry Tags and must NOT carry Policy/PolicyArns.
    """
    sts = MagicMock()
    tvm = STSTokenVending(
        sts_client=sts,
        role_arn="arn:aws:iam::123:role/scoped-session-role",
        sessions_bucket="my-sessions",
        snapshots_bucket="my-snapshots",
        table_name="eval-studio",
    )

    # Mock the STS response
    sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "AKIA...",
            "SecretAccessKey": "secret",
            "SessionToken": "token",
            "Expiration": "2026-07-16T12:00:00Z",
        },
        "PackedPolicySize": 31,  # Tags pack smaller than inline policy
    }

    # Call mint (coordinator path)
    creds = asyncio.run(tvm.mint("user-xyz", "proj-abc", "/some/cwd"))

    # Assert STS was called once
    assert sts.assume_role.call_count == 1

    # Extract kwargs
    call_kwargs = sts.assume_role.call_args.kwargs

    # Assert Tags is present
    assert "Tags" in call_kwargs, "assume_role should pass Tags kwarg"
    tags = call_kwargs["Tags"]
    assert isinstance(tags, list), f"Tags should be a list, got {type(tags)}"
    assert len(tags) == 2, f"Coordinator should send 2 tags, got {len(tags)}"

    # Assert Policy and PolicyArns are NOT present
    assert "Policy" not in call_kwargs, "assume_role should NOT pass Policy (ABAC uses tags)"
    assert "PolicyArns" not in call_kwargs, "assume_role should NOT pass PolicyArns (ABAC uses tags)"


def test_mint_coordinator_tags():
    """mint() sends uid+proj tags for coordinator path (job_id=None)."""
    sts = MagicMock()
    tvm = STSTokenVending(
        sts_client=sts,
        role_arn="arn:aws:iam::123:role/scoped-session-role",
        sessions_bucket="my-sessions",
        snapshots_bucket="my-snapshots",
        table_name="eval-studio",
    )

    sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "AKIA...",
            "SecretAccessKey": "secret",
            "SessionToken": "token",
            "Expiration": "2026-07-16T12:00:00Z",
        },
        "PackedPolicySize": 31,
    }

    # Call mint WITHOUT job_id (coordinator)
    creds = asyncio.run(tvm.mint("user-xyz", "proj-abc", "/some/cwd"))

    # Extract Tags
    call_kwargs = sts.assume_role.call_args.kwargs
    tags = call_kwargs["Tags"]
    tag_dict = {t["Key"]: t["Value"] for t in tags}

    # Assert coordinator tags
    assert tag_dict == {"uid": "user-xyz", "proj": "proj-abc"}, \
        f"Coordinator tags should be {{uid, proj}}, got {tag_dict}"


def test_mint_worker_tags():
    """mint() sends uid+job+proj tags for worker path (job_id set).

    ``proj`` is what makes the scoped-session role's
    ``PROJECT_SEQ#${aws:PrincipalTag/proj}`` LeadingKeys template resolve, so the
    worker can allocate a durable project seq rather than falling back.
    """
    sts = MagicMock()
    tvm = STSTokenVending(
        sts_client=sts,
        role_arn="arn:aws:iam::123:role/scoped-session-role",
        sessions_bucket="my-sessions",
        snapshots_bucket="my-snapshots",
        table_name="eval-studio",
    )

    sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "AKIA...",
            "SecretAccessKey": "secret",
            "SessionToken": "token",
            "Expiration": "2026-07-16T12:00:00Z",
        },
        "PackedPolicySize": 31,
    }

    # Call mint WITH job_id (worker)
    creds = asyncio.run(tvm.mint("user-xyz", "proj-abc", "/tmp/workspace/job-777", job_id="job-777"))

    # Extract Tags
    call_kwargs = sts.assume_role.call_args.kwargs
    tags = call_kwargs["Tags"]
    tag_dict = {t["Key"]: t["Value"] for t in tags}

    # Assert worker tags
    assert tag_dict == {
        "uid": "user-xyz",
        "job": "job-777",
        "proj": "proj-abc",
    }, f"Worker tags should be {{uid, job, proj}}, got {tag_dict}"


def test_mint_logs_packed_policy_size():
    """mint() logs PackedPolicySize INFO/WARN (load-bearing observability).

    Regression guard: even with tags, STS returns PackedPolicySize.
    Log it so operators can measure real headroom.
    """
    sts = MagicMock()
    tvm = STSTokenVending(
        sts_client=sts,
        role_arn="arn:aws:iam::123:role/scoped-session-role",
        sessions_bucket="my-sessions",
        snapshots_bucket="my-snapshots",
        table_name="eval-studio",
    )

    sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "AKIA...",
            "SecretAccessKey": "secret",
            "SessionToken": "token",
            "Expiration": "2026-07-16T12:00:00Z",
        },
        "PackedPolicySize": 85,  # Over 80 threshold
    }

    # Patch the module-level logger
    with patch("app.platform.aws.sts_token_vending.logger") as module_logger:
        creds = asyncio.run(tvm.mint("user-xyz", "proj-abc", "/some/cwd", job_id="job-777"))

        # Assert INFO log was called
        info_calls = [c for c in module_logger.info.call_args_list if "PackedPolicySize" in str(c)]
        assert len(info_calls) >= 1, "Should log PackedPolicySize at INFO level"

        # Assert WARN log was called (≥80)
        warn_calls = [c for c in module_logger.warning.call_args_list if "PackedPolicySize" in str(c)]
        assert len(warn_calls) >= 1, "Should log PackedPolicySize WARN when ≥80"


def test_mint_ttl_clamp():
    """mint() clamps TTL to [_MIN_TTL_SEC, _MAX_TTL_SEC] = [900, 3600].

    Regression guard: role-chaining ceiling is 3600s. Attempting >3600 makes
    assume_role FAIL. Clamp ensures we never exceed it.
    """
    sts = MagicMock()
    tvm = STSTokenVending(
        sts_client=sts,
        role_arn="arn:aws:iam::123:role/scoped-session-role",
        sessions_bucket="my-sessions",
        snapshots_bucket="my-snapshots",
        table_name="eval-studio",
    )

    sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "AKIA...",
            "SecretAccessKey": "secret",
            "SessionToken": "token",
            "Expiration": "2026-07-16T12:00:00Z",
        },
        "PackedPolicySize": 31,
    }

    # Request 5000 seconds (over 3600 cap)
    creds = asyncio.run(tvm.mint("user-xyz", "proj-abc", "/some/cwd", ttl_sec=5000))

    # Assert DurationSeconds was clamped to 3600
    call_kwargs = sts.assume_role.call_args.kwargs
    assert call_kwargs["DurationSeconds"] == 3600, \
        f"TTL should be clamped to 3600, got {call_kwargs['DurationSeconds']}"

    # Request 100 seconds (under 900 floor)
    sts.assume_role.reset_mock()
    creds = asyncio.run(tvm.mint("user-xyz", "proj-abc", "/some/cwd", ttl_sec=100))

    call_kwargs = sts.assume_role.call_args.kwargs
    assert call_kwargs["DurationSeconds"] == 900, \
        f"TTL should be clamped to 900, got {call_kwargs['DurationSeconds']}"
