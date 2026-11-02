# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for mint_worker_credentials — credential_process helper STS minting."""
import json
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import boto3
import pytest


def test_mint_emits_version_1_json_with_expiration():
    """mint_worker_credentials emits Version-1 JSON with Expiration field.

    Verifies the helper calls sts:AssumeRole with session tags from
    build_session_tags, and emits a Version-1 JSON document with the four
    required keys: AccessKeyId, SecretAccessKey, SessionToken, Expiration.
    The Expiration field is what makes the credential refreshable (botocore
    branches on its presence to decide RefreshableCredentials vs frozen
    Credentials).
    """
    from app.scripts import mint_worker_credentials

    fake_creds = {
        "AccessKeyId": "AKIA_MINTED",
        "SecretAccessKey": "secret_minted",
        "SessionToken": "token_minted",
        "Expiration": datetime.now(timezone.utc) + timedelta(hours=1),
    }

    with patch.object(boto3, "Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_sts = MagicMock()
        mock_session.client.return_value = mock_sts
        mock_sts.assume_role.return_value = {"Credentials": fake_creds}

        # Patch STSTokenVending.build_session_tags
        with patch("app.platform.aws.sts_token_vending.STSTokenVending.build_session_tags") as mock_tags:
            mock_tags.return_value = [
                {"Key": "uid", "Value": "test-owner"},
                {"Key": "job", "Value": "test-job"},
            ]

            # Patch sys.argv
            with patch("sys.argv", [
                "mint_worker_credentials.py",
                "--owner-id", "test-owner",
                "--project-id", "test-project",
                "--job-id", "test-job",
                "--role-arn", "arn:aws:iam::123456789012:role/test-role",
            ]):
                # Capture stdout
                import io
                import sys
                captured = io.StringIO()
                with patch.object(sys, "stdout", captured):
                    try:
                        mint_worker_credentials.main()
                    except SystemExit:
                        pass

                output = captured.getvalue()
                doc = json.loads(output)

                # Version-1 document with four keys
                assert doc["Version"] == 1
                assert doc["AccessKeyId"] == "AKIA_MINTED"
                assert doc["SecretAccessKey"] == "secret_minted"
                assert doc["SessionToken"] == "token_minted"
                assert "Expiration" in doc

                # Tags were built via STSTokenVending.build_session_tags
                mock_tags.assert_called_once_with(
                    None, "test-owner", "test-project", job_id="test-job"
                )

                # assume_role was called with those tags
                call_args = mock_sts.assume_role.call_args
                assert call_args[1]["RoleArn"] == "arn:aws:iam::123456789012:role/test-role"
                assert call_args[1]["Tags"] == [
                    {"Key": "uid", "Value": "test-owner"},
                    {"Key": "job", "Value": "test-job"},
                ]


def test_mint_emitted_tags_equal_build_session_tags():
    """Helper's emitted Tags equal STSTokenVending.build_session_tags output.

    Isolation assertion: the helper delegates to build_session_tags so the tag
    shape is defined in exactly ONE place and cannot drift from the
    coordinator's mint path. The worker and coordinator must mint with
    identical tag shapes for the ABAC scoping to work.
    """
    from app.platform.aws.sts_token_vending import STSTokenVending

    expected_tags = STSTokenVending.build_session_tags(
        None, "test-owner", "test-project", job_id="test-job"
    )

    # Now invoke the helper and capture the Tags it passes to assume_role
    from app.scripts import mint_worker_credentials

    fake_creds = {
        "AccessKeyId": "AKIA_MINTED",
        "SecretAccessKey": "secret_minted",
        "SessionToken": "token_minted",
        "Expiration": datetime.now(timezone.utc) + timedelta(hours=1),
    }

    with patch.object(boto3, "Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_sts = MagicMock()
        mock_session.client.return_value = mock_sts
        mock_sts.assume_role.return_value = {"Credentials": fake_creds}

        with patch("sys.argv", [
            "mint_worker_credentials.py",
            "--owner-id", "test-owner",
            "--project-id", "test-project",
            "--job-id", "test-job",
            "--role-arn", "arn:aws:iam::123456789012:role/test-role",
        ]):
            try:
                mint_worker_credentials.main()
            except SystemExit:
                pass

            call_args = mock_sts.assume_role.call_args
            actual_tags = call_args[1]["Tags"]

            # Tags passed to assume_role equal build_session_tags output
            assert actual_tags == expected_tags


def test_mint_expiration_field_required_for_refreshability():
    """Expiration field is always present — this is what makes the credential refreshable.

    botocore's ProcessProvider.load branches on `if creds_dict.get('expiry_time')
    is not None:` to decide whether to return a RefreshableCredentials
    (refreshable) or a plain Credentials (frozen). The helper emits the JSON
    key `Expiration`, and botocore maps it to the parsed-dict key `expiry_time`
    that the branch tests. Omitting it yields a frozen credential — silently
    reintroducing the exact bug this guards against.
    """
    from app.scripts import mint_worker_credentials

    fake_creds = {
        "AccessKeyId": "AKIA_MINTED",
        "SecretAccessKey": "secret_minted",
        "SessionToken": "token_minted",
        "Expiration": datetime.now(timezone.utc) + timedelta(hours=1),
    }

    with patch.object(boto3, "Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_sts = MagicMock()
        mock_session.client.return_value = mock_sts
        mock_sts.assume_role.return_value = {"Credentials": fake_creds}

        with patch("sys.argv", [
            "mint_worker_credentials.py",
            "--owner-id", "test-owner",
            "--project-id", "test-project",
            "--job-id", "test-job",
            "--role-arn", "arn:aws:iam::123456789012:role/test-role",
        ]):
            import io
            import sys
            captured = io.StringIO()
            with patch.object(sys, "stdout", captured):
                try:
                    mint_worker_credentials.main()
                except SystemExit:
                    pass

            output = captured.getvalue()
            doc = json.loads(output)

            # Expiration is present (this is the load-bearing field for
            # refreshability — pinned by the botocore branch on this)
            assert "Expiration" in doc
            assert doc["Expiration"]  # non-empty


def test_mint_recursion_prevention():
    """Helper clears AWS_CONFIG_FILE before creating boto3 session.

    Without this, the session would resolve the same credential_process and
    re-invoke this script — infinite loop. The deletion forces the session to
    resolve the container task role (ECS credential endpoint) instead.
    """
    from app.scripts import mint_worker_credentials
    import os

    # Simulate the environment the helper runs in (AWS_CONFIG_FILE pointing
    # at the config that invoked it)
    os.environ["AWS_CONFIG_FILE"] = "/tmp/fake_config"

    fake_creds = {
        "AccessKeyId": "AKIA_MINTED",
        "SecretAccessKey": "secret_minted",
        "SessionToken": "token_minted",
        "Expiration": datetime.now(timezone.utc) + timedelta(hours=1),
    }

    with patch.object(boto3, "Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_sts = MagicMock()
        mock_session.client.return_value = mock_sts
        mock_sts.assume_role.return_value = {"Credentials": fake_creds}

        with patch("sys.argv", [
            "mint_worker_credentials.py",
            "--owner-id", "test-owner",
            "--project-id", "test-project",
            "--job-id", "test-job",
            "--role-arn", "arn:aws:iam::123456789012:role/test-role",
        ]):
            try:
                mint_worker_credentials.main()
            except SystemExit:
                pass

            # AWS_CONFIG_FILE was deleted before boto3.Session was created
            assert "AWS_CONFIG_FILE" not in os.environ

    # Cleanup
    os.environ.pop("AWS_CONFIG_FILE", None)


def test_credential_refreshes_after_expiry():
    """A boto3 session resolving the generated config produces a provider whose frozen credential CHANGES after the emitted expiry passes.

    This is the deterministic test that pins the third-party dependency: if
    the credential stops refreshing while an expiry IS being emitted, the
    botocore `expiry_time` branch assumption has changed and the fix must move
    to an in-process RefreshableCredentials, not be patched around.

    Test structure: stand up a fake credential_process helper that emits
    near-future expiries with changing access keys, point a real boto3 session
    at it, assert the frozen credential differs across the expiry boundary with
    no static credential env keys present.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a fake credential_process helper that emits near-future expiries
        # with a changing access key (simulates a real refresh)
        helper_script = Path(tmpdir) / "fake_helper.py"
        helper_script.write_text("""
import json
import sys
from datetime import datetime, timedelta, timezone

# Emit a credential expiring 2 seconds from now
expiry = datetime.now(timezone.utc) + timedelta(seconds=2)

# Access key changes each invocation (append a counter to simulate refresh).
# The counter lives BESIDE this script, i.e. inside the test's own tmpdir. A
# fixed path in /tmp made the test fail wherever /tmp is not freely writable,
# and let two concurrent pytest processes share one counter — which would make
# the refresh assertion pass or fail on their interleaving.
import os
counter_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "counter.txt")
if os.path.exists(counter_file):
    with open(counter_file) as f:
        counter = int(f.read())
else:
    counter = 0
counter += 1
with open(counter_file, "w") as f:
    f.write(str(counter))

doc = {
    "Version": 1,
    "AccessKeyId": f"AKIA_TEST_{counter:04d}",
    "SecretAccessKey": "secret",
    "SessionToken": "token",
    "Expiration": expiry.isoformat(),
}
print(json.dumps(doc))
""", encoding="utf-8")

        # Write an AWS config pointing at the fake helper
        config_path = Path(tmpdir) / "aws_config"
        config_path.write_text(f"""[default]
credential_process = python3 {helper_script}
""", encoding="utf-8")

        # No prior counter to clear: it is created inside this tmpdir and dies
        # with it, so each run starts from zero by construction.
        import os

        # Remove static credential env keys (they would shadow the
        # credential_process provider)
        env_backup = {}
        for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]:
            if key in os.environ:
                env_backup[key] = os.environ.pop(key)

        # Point a real boto3 session at the fake config
        os.environ["AWS_CONFIG_FILE"] = str(config_path)

        try:
            session = boto3.Session()
            creds_provider = session.get_credentials()

            # First frozen credential
            frozen_1 = creds_provider.get_frozen_credentials()
            key_1 = frozen_1.access_key

            # Wait for expiry to pass (2s + 1s buffer)
            import time
            time.sleep(3)

            # Second frozen credential — should differ (the provider refreshed)
            frozen_2 = creds_provider.get_frozen_credentials()
            key_2 = frozen_2.access_key

            # Assertion: the access key changed (the credential refreshed)
            assert key_1 != key_2, (
                f"Credential did not refresh after expiry passed. "
                f"key_1={key_1}, key_2={key_2}. If the helper IS emitting an "
                f"expiry but the credential does not refresh, the botocore "
                f"expiry_time branch assumption has changed — the fix must move "
                f"to an in-process RefreshableCredentials, not be patched around."
            )
        finally:
            # Restore environment
            os.environ.pop("AWS_CONFIG_FILE", None)
            for key, value in env_backup.items():
                os.environ[key] = value
            # The counter file needs no cleanup — the TemporaryDirectory that
            # holds it is removed when this block exits.


def test_negative_control_no_expiration_does_not_refresh():
    """Negative control: a helper emitting NO expiration yields a non-refreshable credential whose key does NOT change.

    Pins the Expiration/expiry_time dependency: botocore returns a refreshable
    credential only when the emitted Expiration reaches its parsed-dict
    expiry_time key. If the refresh test ever fails while an expiry IS being
    emitted, this negative control confirms the mechanism assumption (the
    presence branch) is still valid.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a fake helper that emits NO Expiration field
        helper_script = Path(tmpdir) / "fake_helper_no_expiry.py"
        helper_script.write_text("""
import json

doc = {
    "Version": 1,
    "AccessKeyId": "AKIA_FROZEN",
    "SecretAccessKey": "secret",
    "SessionToken": "token",
}
print(json.dumps(doc))
""", encoding="utf-8")

        config_path = Path(tmpdir) / "aws_config"
        config_path.write_text(f"""[default]
credential_process = python3 {helper_script}
""", encoding="utf-8")

        import os
        env_backup = {}
        for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]:
            if key in os.environ:
                env_backup[key] = os.environ.pop(key)

        os.environ["AWS_CONFIG_FILE"] = str(config_path)

        try:
            session = boto3.Session()
            creds_provider = session.get_credentials()

            frozen_1 = creds_provider.get_frozen_credentials()
            key_1 = frozen_1.access_key

            # Wait (same duration as the positive test)
            import time
            time.sleep(3)

            frozen_2 = creds_provider.get_frozen_credentials()
            key_2 = frozen_2.access_key

            # Assertion: the access key did NOT change (no expiration → frozen)
            assert key_1 == key_2, (
                f"Credential refreshed even without an Expiration field. "
                f"This violates the botocore expiry_time branch assumption."
            )
        finally:
            os.environ.pop("AWS_CONFIG_FILE", None)
            for key, value in env_backup.items():
                os.environ[key] = value
