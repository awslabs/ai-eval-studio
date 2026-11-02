# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the scoped-session-store diagnostic helper in agent_runner.

These tests lock the diagnostic fields and the prefix_matches discriminator
(match + mismatch + fallback + secret-safety + STS-failure cases) by
patching the module logger (no caplog — global-logging-state flake rule).
"""

from unittest.mock import MagicMock, patch

import pytest

from app.services import agent_runner


def test_diagnostic_scoped_matching_prefix():
    """Scoped session + matching prefix logs all diagnostic fields + prefix_matches=True.

    Verifies the diagnostic captures: uid, proj, bucket, derived project_key,
    expected ABAC-template prefix, prefix_matches=True, and the assumed-role ARN.
    """
    # Arrange: owner_id and project_id chosen so project_key_for_directory
    # produces the same canonical form as the ABAC template
    owner_id = "user-abc123"
    project_id = "proj-xyz789"
    # Use a Path that canonicalizes to match the ABAC template
    from pathlib import Path
    project_dir = Path(f"/mnt/projects/users/{owner_id}/projects/{project_id}")
    bucket = "test-sessions-bucket"

    # Mock the scoped session's get_caller_identity call
    mock_sts_client = MagicMock()
    mock_sts_client.get_caller_identity.return_value = {
        "Arn": f"arn:aws:sts::123456789012:assumed-role/scoped-session-role/{owner_id}-{project_id}"
    }
    mock_scoped_session = MagicMock()
    mock_scoped_session.client.return_value = mock_sts_client

    # Patch the module logger
    with patch.object(agent_runner, "logger") as mock_logger:
        # Act: call the diagnostic helper
        agent_runner._log_scoped_session_store_diagnostic(
            owner_id=owner_id,
            project_id=project_id,
            project_dir=project_dir,
            bucket=bucket,
            scoped_session=mock_scoped_session,
        )

        # Assert: logger.info was called exactly once
        assert mock_logger.info.call_count == 1, "Expected exactly one INFO log"

        # Extract the logged message (format string + args interpolated)
        log_call = mock_logger.info.call_args
        format_str = log_call[0][0]
        format_args = log_call[0][1:]
        # Interpolate the format string with the args
        log_msg = format_str % format_args

        # Assert all required fields are present
        assert "uid=" in log_msg, "Missing uid field"
        assert owner_id in log_msg, f"uid value {owner_id} not in log"
        assert "proj=" in log_msg, "Missing proj field"
        assert project_id in log_msg, f"proj value {project_id} not in log"
        assert "bucket=" in log_msg, "Missing bucket field"
        assert bucket in log_msg, f"bucket {bucket} not in log"
        assert "project_key=" in log_msg, "Missing project_key field"
        assert "expected=" in log_msg, "Missing expected-template field"
        assert "prefix_matches=True" in log_msg, "Expected prefix_matches=True"
        assert "assumed-role" in log_msg, "Missing assumed-role ARN"
        assert "path=scoped" in log_msg or "scoped" in log_msg, "Expected scoped path indicator"


def test_diagnostic_scoped_mismatched_prefix():
    """Scoped session + mismatched prefix logs prefix_matches=False.

    When the project_dir's owner segment differs from owner_id, the SDK-derived
    project_key will not match the ABAC-template prefix.
    """
    # Arrange: intentional mismatch — owner_id != owner in path
    owner_id = "user-abc123"
    project_id = "proj-xyz789"
    # Path with a DIFFERENT owner segment to trigger mismatch
    from pathlib import Path
    project_dir = Path(f"/mnt/projects/users/different-owner/projects/{project_id}")
    bucket = "test-sessions-bucket"

    # Mock the scoped session's get_caller_identity call
    mock_sts_client = MagicMock()
    mock_sts_client.get_caller_identity.return_value = {
        "Arn": f"arn:aws:sts::123456789012:assumed-role/scoped-session-role/{owner_id}-{project_id}"
    }
    mock_scoped_session = MagicMock()
    mock_scoped_session.client.return_value = mock_sts_client

    # Patch the module logger
    with patch.object(agent_runner, "logger") as mock_logger:
        # Act
        agent_runner._log_scoped_session_store_diagnostic(
            owner_id=owner_id,
            project_id=project_id,
            project_dir=project_dir,
            bucket=bucket,
            scoped_session=mock_scoped_session,
        )

        # Assert: prefix_matches=False is logged
        assert mock_logger.info.call_count == 1, "Expected exactly one INFO log"
        log_call = mock_logger.info.call_args
        format_str = log_call[0][0]
        format_args = log_call[0][1:]
        log_msg = format_str % format_args
        assert "prefix_matches=False" in log_msg, "Expected prefix_matches=False for mismatched prefix"


def test_diagnostic_fallback_unminted():
    """Fallback (scoped_session=None) logs unminted path distinctly.

    When minting fails, the diagnostic should still log (without raising)
    and mark the unminted/fallback path so the operator can identify a
    total mint failure.
    """
    # Arrange: scoped_session=None (mint failure)
    owner_id = "user-abc123"
    project_id = "proj-xyz789"
    from pathlib import Path
    project_dir = Path(f"/mnt/projects/users/{owner_id}/projects/{project_id}")
    bucket = "test-sessions-bucket"

    # Patch the module logger
    with patch.object(agent_runner, "logger") as mock_logger:
        # Act: call with scoped_session=None
        agent_runner._log_scoped_session_store_diagnostic(
            owner_id=owner_id,
            project_id=project_id,
            project_dir=project_dir,
            bucket=bucket,
            scoped_session=None,
        )

        # Assert: logger.info called once, unminted/fallback path indicated
        assert mock_logger.info.call_count == 1, "Expected exactly one INFO log"
        log_call = mock_logger.info.call_args
        format_str = log_call[0][0]
        format_args = log_call[0][1:]
        log_msg = format_str % format_args
        assert "unminted" in log_msg or "fallback" in log_msg, "Expected unminted/fallback path indicator"
        assert owner_id in log_msg, "uid should still be logged"
        assert project_id in log_msg, "proj should still be logged"
        assert bucket in log_msg, "bucket should still be logged"


def test_diagnostic_secret_safety():
    """Diagnostic never logs AWS secret credential material.

    The helper is given no secret material and must log only non-secret
    values. This test asserts the logged message contains neither
    secret-key nor session-token markers.
    """
    # Arrange
    owner_id = "user-abc123"
    project_id = "proj-xyz789"
    from pathlib import Path
    project_dir = Path(f"/mnt/projects/users/{owner_id}/projects/{project_id}")
    bucket = "test-sessions-bucket"

    mock_sts_client = MagicMock()
    mock_sts_client.get_caller_identity.return_value = {
        "Arn": "arn:aws:sts::123456789012:assumed-role/scoped-session-role/test-session"
    }
    mock_scoped_session = MagicMock()
    mock_scoped_session.client.return_value = mock_sts_client

    # Patch the module logger
    with patch.object(agent_runner, "logger") as mock_logger:
        # Act
        agent_runner._log_scoped_session_store_diagnostic(
            owner_id=owner_id,
            project_id=project_id,
            project_dir=project_dir,
            bucket=bucket,
            scoped_session=mock_scoped_session,
        )

        # Assert: no secret markers in the logged message
        assert mock_logger.info.call_count == 1, "Expected exactly one INFO log"
        log_call = mock_logger.info.call_args
        format_str = log_call[0][0]
        format_args = log_call[0][1:]
        log_msg = format_str % format_args
        # Check for common secret field names (case-insensitive)
        log_lower = log_msg.lower()
        assert "secretaccesskey" not in log_lower.replace(" ", "").replace("_", ""), \
            "Diagnostic must NOT log secret_access_key"
        assert "sessiontoken" not in log_lower.replace(" ", "").replace("_", ""), \
            "Diagnostic must NOT log session_token"
        assert "accesskeyid" not in log_lower.replace(" ", "").replace("_", ""), \
            "Diagnostic must NOT log access_key_id"


def test_diagnostic_sts_failure_best_effort():
    """get_caller_identity failure still logs (ARN recorded as unavailable).

    The diagnostic is best-effort — an STS call failure should NOT raise;
    instead, the ARN should be recorded as unavailable and the log still emitted.
    """
    # Arrange: scoped session whose STS call raises
    owner_id = "user-abc123"
    project_id = "proj-xyz789"
    from pathlib import Path
    project_dir = Path(f"/mnt/projects/users/{owner_id}/projects/{project_id}")
    bucket = "test-sessions-bucket"

    mock_sts_client = MagicMock()
    mock_sts_client.get_caller_identity.side_effect = Exception("STS call failed")
    mock_scoped_session = MagicMock()
    mock_scoped_session.client.return_value = mock_sts_client

    # Patch the module logger
    with patch.object(agent_runner, "logger") as mock_logger:
        # Act: should NOT raise
        agent_runner._log_scoped_session_store_diagnostic(
            owner_id=owner_id,
            project_id=project_id,
            project_dir=project_dir,
            bucket=bucket,
            scoped_session=mock_scoped_session,
        )

        # Assert: logger.info called once, ARN marked as unavailable
        assert mock_logger.info.call_count == 1, "Expected exactly one INFO log"
        log_call = mock_logger.info.call_args
        format_str = log_call[0][0]
        format_args = log_call[0][1:]
        log_msg = format_str % format_args
        assert "unavailable" in log_msg, "Expected ARN recorded as unavailable on STS failure"
        assert "path=scoped" in log_msg or "scoped" in log_msg, "Expected scoped path indicator"
