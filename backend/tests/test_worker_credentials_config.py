# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker_credentials_config — AWS config writing and env mutations."""
import tempfile
from pathlib import Path

from app.services.worker_credentials_config import write_worker_credentials_config


def test_write_config_and_return_env_mutations():
    """write_worker_credentials_config writes AWS config and returns env mutations.

    Verifies the config file contains a credential_process line with explicit
    CLI arguments (owner, project, job, role ARN) and that the returned
    mutations set AWS_CONFIG_FILE plus removal sentinels for the three static
    credential keys.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        mutations = write_worker_credentials_config(
            owner_id="test-owner",
            project_id="test-project",
            job_id="test-job",
            role_arn="arn:aws:iam::123456789012:role/test-role",
            config_dir=tmpdir,
        )

        assert mutations is not None
        assert "AWS_CONFIG_FILE" in mutations
        config_path = Path(mutations["AWS_CONFIG_FILE"])
        assert config_path.exists()

        # Config contains credential_process with explicit arguments
        config_text = config_path.read_text()
        assert "[default]" in config_text
        assert "credential_process" in config_text
        assert "--owner-id test-owner" in config_text
        assert "--project-id test-project" in config_text
        assert "--job-id test-job" in config_text
        assert "--role-arn arn:aws:iam::123456789012:role/test-role" in config_text

        # Mutations include removal sentinels (empty strings) for static creds
        assert mutations["AWS_ACCESS_KEY_ID"] == ""
        assert mutations["AWS_SECRET_ACCESS_KEY"] == ""
        assert mutations["AWS_SESSION_TOKEN"] == ""


def test_apply_mutations_removes_static_creds():
    """Applying env mutations with empty-string values removes those keys.

    Verifies that the removal-sentinel pattern (empty string) is correctly
    interpreted by the caller as "delete this key", so the static credential
    triple is removed from the environment and cannot shadow the
    credential_process provider.
    """
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Pre-populate environment with static creds (simulating the RunTask
        # env before the worker boot path removes them)
        os.environ["AWS_ACCESS_KEY_ID"] = "AKIA_STATIC"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "secret_static"
        os.environ["AWS_SESSION_TOKEN"] = "token_static"

        mutations = write_worker_credentials_config(
            owner_id="test-owner",
            project_id="test-project",
            job_id="test-job",
            role_arn="arn:aws:iam::123456789012:role/test-role",
            config_dir=tmpdir,
        )

        # Apply mutations: set non-empty, delete empty
        for key, value in mutations.items():
            if value:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)

        # Static triple is now absent from environment
        assert "AWS_ACCESS_KEY_ID" not in os.environ
        assert "AWS_SECRET_ACCESS_KEY" not in os.environ
        assert "AWS_SESSION_TOKEN" not in os.environ
        # Config file path is set
        assert "AWS_CONFIG_FILE" in os.environ

        # Cleanup
        os.environ.pop("AWS_CONFIG_FILE", None)


def test_dev_mode_no_config_written():
    """When role_arn is falsy (dev mode), no config is written and None is returned.

    Dev mode signal: TokenVending.mint() returns None, so the caller passes
    role_arn=None. No config is written and the environment is left untouched
    (ambient credentials still serve).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        mutations = write_worker_credentials_config(
            owner_id="test-owner",
            project_id="test-project",
            job_id="test-job",
            role_arn=None,
            config_dir=tmpdir,
        )

        assert mutations is None
        # No aws_config file was written
        config_path = Path(tmpdir) / "aws_config"
        assert not config_path.exists()


def test_credential_process_uses_absolute_paths():
    """credential_process line contains absolute interpreter + absolute script path.

    Verifies the generated command is PATH- and cwd-independent: no bare
    'python' token, no '-m' flag, and both the interpreter and the helper script
    are absolute paths that exist.
    """
    import shlex

    with tempfile.TemporaryDirectory() as tmpdir:
        mutations = write_worker_credentials_config(
            owner_id="test-owner",
            project_id="test-project",
            job_id="test-job",
            role_arn="arn:aws:iam::123456789012:role/test-role",
            config_dir=tmpdir,
        )

        config_path = Path(mutations["AWS_CONFIG_FILE"])
        config_text = config_path.read_text()

        # Extract the credential_process line value
        lines = [l for l in config_text.splitlines() if l.startswith("credential_process")]
        assert len(lines) == 1, "exactly one credential_process line expected"
        _, value = lines[0].split("=", 1)
        value = value.strip()

        # Parse the command
        parts = shlex.split(value)

        # Element 0 must be an absolute path to an existing interpreter
        interpreter = Path(parts[0])
        assert interpreter.is_absolute(), f"interpreter not absolute: {parts[0]}"
        assert interpreter.exists(), f"interpreter does not exist: {parts[0]}"

        # Element 1 must be an absolute path ending in mint_worker_credentials.py
        script = Path(parts[1])
        assert script.is_absolute(), f"script not absolute: {parts[1]}"
        assert script.name == "mint_worker_credentials.py", f"wrong script: {parts[1]}"
        assert script.exists(), f"script does not exist: {parts[1]}"

        # No '-m' flag anywhere
        assert "-m" not in parts, "'-m' flag present — module resolution is cwd-dependent"

        # No bare 'python' anywhere
        assert "python" not in parts, "bare 'python' token present — PATH-dependent"


def test_credential_process_survives_space_in_path():
    """Generated command survives paths containing spaces via shlex round-trip.

    Verifies that shlex.join is used (not naive f-string interpolation), so a
    space-containing path in sys.executable or the helper __file__ is correctly
    quoted and survives botocore's config parser + shlex.split round-trip.
    """
    import shlex
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmpdir:
        # Fake sys.executable and helper path with spaces
        fake_interpreter = "/path with spaces/bin/python"
        fake_helper = "/another path with spaces/mint_worker_credentials.py"

        # Patch find_spec to return a spec with a space-containing origin
        class FakeSpec:
            origin = fake_helper

        with patch("sys.executable", fake_interpreter), \
             patch("importlib.util.find_spec", return_value=FakeSpec()):

            mutations = write_worker_credentials_config(
                owner_id="test-owner",
                project_id="test-project",
                job_id="test-job",
                role_arn="arn:aws:iam::123456789012:role/test-role",
                config_dir=tmpdir,
            )

            config_path = Path(mutations["AWS_CONFIG_FILE"])
            config_text = config_path.read_text()

            # Extract the credential_process line value
            lines = [l for l in config_text.splitlines() if l.startswith("credential_process")]
            assert len(lines) == 1
            _, value = lines[0].split("=", 1)
            value = value.strip()

            # Parse and verify round-trip — shlex.split should restore the exact paths
            parts = shlex.split(value)
            assert parts[0] == fake_interpreter, f"interpreter round-trip failed: {parts[0]}"
            assert parts[1] == fake_helper, f"helper round-trip failed: {parts[1]}"
