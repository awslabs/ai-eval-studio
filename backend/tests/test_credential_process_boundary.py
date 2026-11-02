# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real process/interpreter/cwd boundary test for credential_process.

This test file is the centerpiece — it exercises the REAL process boundary that
in-process tests cannot reach. The regression exists because the prior tests
invoked mint_worker_credentials.main() inside the pytest process, where 'app'
is importable because pytest's cwd is backend/. The defect lives entirely in
the transition to a **different interpreter** with a **different cwd**.

What is REAL vs faked:
- REAL: process boundary, interpreter, cwd, environment (os.environ.copy()),
  botocore config parsing, botocore's ProcessProvider
- FAKED: AWS (STS and container-credentials endpoint, over loopback HTTP)

Why not mark @pytest.mark.integration: These must run in the default suite.
pyproject.toml defines 'integration' as "hits real external services (AWS,
network)", and a loopback HTTPServer on 127.0.0.1:0 plus a local subprocess is
neither. Marking these would move the only guard against this regression class
out of the default run, silently retiring the protection.

Tradeoff: SigV4 signing and real IAM authorization are NOT exercised here.
Those are covered by infra/tests/test_worker_selfmint_iam.py (synth-level) and
by the operator's live checks. What this test uniquely proves is that the
import, argument parsing, STS call, and JSON emission path all execute inside a
foreign interpreter from a foreign cwd — which is exactly what broke.
"""
import http.server
import json
import os
import shlex
import subprocess
import tempfile
import threading
import venv
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import boto3
import botocore.credentials
import pytest

from app.services.worker_credentials_config import write_worker_credentials_config

# Byte-exact command value the older write_worker_credentials_config emitted,
# captured from a real run of that version. This is a recorded artifact, not a
# reconstruction — the generator no longer produces it, so it cannot be
# re-derived from the code under test.
_PREFIX_BUGGY_COMMAND = (
    "python -m app.scripts.mint_worker_credentials "
    "--owner-id OWNER --project-id PROJ --job-id JOB "
    "--role-arn arn:aws:iam::123456789012:role/ROLE"
)


class _FakeAWSHandler(http.server.BaseHTTPRequestHandler):
    """Serve container-credentials GET and AssumeRole POST over loopback HTTP."""

    def log_message(self, format, *args):
        """Silence request logging."""
        pass

    def do_GET(self):
        """Container-credentials endpoint — return ECS-style JSON."""
        expiration = (datetime.utcnow() + timedelta(hours=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        body = json.dumps({
            "AccessKeyId": "AKIA_CONTAINER",
            "SecretAccessKey": "secret_container",
            "Token": "token_container",
            "Expiration": expiration,
        })
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def do_POST(self):
        """STS AssumeRole endpoint — return XML response."""
        expiration = (datetime.utcnow() + timedelta(hours=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        xml_body = f"""<?xml version="1.0" encoding="UTF-8"?>
<AssumeRoleResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/">
    <AssumeRoleResult>
        <Credentials>
            <AccessKeyId>AKIAFAKEBOUNDARY</AccessKeyId>
            <SecretAccessKey>secret_boundary</SecretAccessKey>
            <SessionToken>token_boundary</SessionToken>
            <Expiration>{expiration}</Expiration>
        </Credentials>
        <AssumedRoleUser>
            <AssumedRoleId>AIDA:session</AssumedRoleId>
            <Arn>arn:aws:sts::123456789012:assumed-role/test/session</Arn>
        </AssumedRoleUser>
    </AssumeRoleResult>
    <ResponseMetadata>
        <RequestId>fake-request-id</RequestId>
    </ResponseMetadata>
</AssumeRoleResponse>
"""
        self.send_response(200)
        self.send_header("Content-Type", "text/xml")
        self.send_header("Content-Length", str(len(xml_body)))
        self.end_headers()
        self.wfile.write(xml_body.encode("utf-8"))


@pytest.fixture
def fake_aws_server():
    """Stand up a loopback HTTP server serving fake container-creds + STS."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _FakeAWSHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture
def bare_venv(tmp_path):
    """Create a bare venv that cannot import boto3 or app — faithful to harness."""
    venv_path = tmp_path / "harness_venv"
    venv.create(venv_path, symlinks=True, with_pip=False)
    return venv_path


def _container_like_env(fake_aws_server, isolated_home, **overrides) -> dict[str, str]:
    """Build a child environment that matches the worker container, not this laptop.

    ``os.environ.copy()`` alone is not faithful. The helper deliberately deletes
    ``AWS_CONFIG_FILE`` before building its boto3 session (to avoid recursing into
    itself), which makes botocore fall back to the DEFAULT config path,
    ``$HOME/.aws/config``. On a developer machine that file exists and usually
    carries a ``credential_process`` of its own, and ``AWS_PROFILE`` is exported --
    so botocore resolves the developer's profile instead of the fake container
    endpoint, and the helper fails with whatever that profile's helper says.

    A worker task has no ``~/.aws/config`` and no ``AWS_PROFILE``. Reproducing
    that is what makes the fake AWS server authoritative:

    - ``HOME`` points at an empty directory, so ``$HOME/.aws/config`` misses.
    - profile selectors are removed, so there is no named profile to resolve.
    - static credentials are removed, so the chain reaches the container provider.

    Args:
        fake_aws_server: Base URL of the loopback fake AWS server.
        isolated_home: Directory to use as ``HOME``; must not contain ``.aws``.
        **overrides: Extra environment entries applied last.

    Returns:
        The child environment mapping.
    """
    env = os.environ.copy()
    env["HOME"] = str(isolated_home)
    env["AWS_CONTAINER_CREDENTIALS_FULL_URI"] = f"{fake_aws_server}/creds"
    env["AWS_ENDPOINT_URL_STS"] = fake_aws_server
    env["AWS_DEFAULT_REGION"] = "us-east-1"
    for key in (
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    ):
        env.pop(key, None)
    env.update(overrides)
    return env


def test_raw_subprocess_boundary(fake_aws_server, bare_venv, tmp_path, monkeypatch):
    """Raw child invocation emits JSON with all five Version-1 keys.

    Verifies the helper executes successfully as a script by a different
    interpreter from a different cwd, with os.environ.copy()-derived env.
    """
    # Generate config via the real write_worker_credentials_config
    mutations = write_worker_credentials_config(
        owner_id="test-owner",
        project_id="test-project",
        job_id="test-job",
        role_arn="arn:aws:iam::123456789012:role/test-role",
        config_dir=tmp_path,
    )

    # Extract credential_process value
    config_path = Path(mutations["AWS_CONFIG_FILE"])
    config_text = config_path.read_text()
    lines = [l for l in config_text.splitlines() if l.startswith("credential_process")]
    _, value = lines[0].split("=", 1)
    cmd = shlex.split(value.strip())

    # Hostile environment: container-faithful AWS resolution plus a PATH holding
    # only the bare venv's bin, so the helper cannot lean on this repo's venv.
    isolated_home = tmp_path / "home"
    isolated_home.mkdir()
    child_env = _container_like_env(
        fake_aws_server, isolated_home, PATH=str(bare_venv / "bin")
    )

    # Override conftest's autouse _isolate_aws_credentials fixture
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

    # Run the child from a foreign cwd
    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        env=child_env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"Helper exited non-zero. stderr: {result.stderr}, stdout: {result.stdout}"
    )

    doc = json.loads(result.stdout)
    assert doc["Version"] == 1, f"Version key missing or wrong: {doc}"
    assert "AccessKeyId" in doc, f"AccessKeyId missing: {doc}"
    assert "SecretAccessKey" in doc, f"SecretAccessKey missing: {doc}"
    assert "SessionToken" in doc, f"SessionToken missing: {doc}"
    assert "Expiration" in doc, f"Expiration missing: {doc}"


def test_botocore_process_provider_returns_refreshable_credentials(
    fake_aws_server, bare_venv, tmp_path, monkeypatch
):
    """botocore's ProcessProvider returns RefreshableCredentials with correct key.

    Verifies the end-to-end contract: botocore parses the config, spawns the
    child, and returns a refreshable credential provider with method
    "custom-process". This proves the refreshability property through botocore
    instead of re-implementing its parsing.
    """
    # Generate config
    mutations = write_worker_credentials_config(
        owner_id="test-owner",
        project_id="test-project",
        job_id="test-job",
        role_arn="arn:aws:iam::123456789012:role/test-role",
        config_dir=tmp_path,
    )

    # Extract credential_process value to verify absoluteness
    config_path = Path(mutations["AWS_CONFIG_FILE"])
    config_text = config_path.read_text()
    lines = [l for l in config_text.splitlines() if l.startswith("credential_process")]
    _, value = lines[0].split("=", 1)
    cmd_parts = shlex.split(value.strip())

    # Assert the config pinned its own interpreter, not the bare venv's
    bare_venv_python = str(bare_venv / "bin" / "python")
    assert cmd_parts[0] != bare_venv_python, (
        f"Config did not pin its own interpreter — it would inherit the caller's. "
        f"Expected absolute interpreter, got: {cmd_parts[0]}"
    )

    # Set up test environment: config file pointing at the helper, hostile cwd
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_CONTAINER_CREDENTIALS_FULL_URI", raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config_path))
    monkeypatch.chdir(tmp_path)

    # Build environment for the subprocess that the ProcessProvider will spawn.
    # This is what makes the test hostile: the child inherits a PATH with only
    # the bare venv, and it needs AWS endpoints for its own STS call.
    #
    # HOME is an EMPTY directory, not the real one. The helper drops
    # AWS_CONFIG_FILE before building its session, so botocore falls back to
    # $HOME/.aws/config — passing the developer's HOME lets that file's own
    # credential_process answer instead of the fake container endpoint.
    isolated_home = tmp_path / "home"
    isolated_home.mkdir(exist_ok=True)
    subprocess_env = {
        "PATH": str(bare_venv / "bin"),
        "AWS_CONTAINER_CREDENTIALS_FULL_URI": f"{fake_aws_server}/creds",
        "AWS_ENDPOINT_URL_STS": fake_aws_server,
        "AWS_DEFAULT_REGION": "us-east-1",
        "HOME": str(isolated_home),
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    # Patch subprocess.Popen globally so botocore's ProcessProvider spawns
    # the child with our hostile environment. We cannot selectively patch only
    # the credential_process invocation because botocore calls Popen directly.
    original_popen_init = subprocess.Popen.__init__
    def patched_popen_init(self, args, *popen_args, **kwargs):
        # Inject subprocess_env for all Popen calls during this test. This is
        # safe because the test has no other subprocess spawns, and it's the
        # only way to control the environment that botocore's ProcessProvider
        # passes to the child.
        if "env" not in kwargs:
            kwargs["env"] = subprocess_env
        return original_popen_init(self, args, *popen_args, **kwargs)
    monkeypatch.setattr(subprocess.Popen, "__init__", patched_popen_init)

    # Get credentials via botocore — it will invoke ProcessProvider which
    # spawns our helper with the hostile subprocess_env
    session = boto3.Session()
    provider = session.get_credentials()

    assert isinstance(provider, botocore.credentials.RefreshableCredentials), (
        f"Expected RefreshableCredentials, got: {type(provider)}"
    )
    assert provider.method == "custom-process", f"method: {provider.method}"

    frozen = provider.get_frozen_credentials()
    assert frozen.access_key == "AKIAFAKEBOUNDARY", f"access_key: {frozen.access_key}"
    assert frozen.token == "token_boundary", f"token: {frozen.token}"


def test_negative_control_buggy_command_fails(
    fake_aws_server, bare_venv, tmp_path, monkeypatch
):
    """Negative control: the older 'python -m' command fails module resolution.

    This test pins the absolute-interpreter + absolute-script + importability
    contract. Its purpose is to fail if anyone reverts to a PATH- or
    cwd-dependent form. The command string is the older generator's captured
    output — the generator no longer produces it, so it cannot be re-derived
    from the code under test.
    """
    # Write a config with the byte-exact older command
    config_path = tmp_path / "buggy_aws_config"
    config_content = f"""[default]
credential_process = {_PREFIX_BUGGY_COMMAND}
"""
    config_path.write_text(config_content)

    # Build hostile environment: only the bare venv on PATH, no PYTHONPATH, and
    # an EMPTY HOME so a real ~/.aws/config cannot satisfy the child. Without
    # that this negative control can pass for the wrong reason — succeeding via
    # the developer's own profile rather than failing on the buggy command.
    isolated_home = tmp_path / "home"
    isolated_home.mkdir(exist_ok=True)
    child_env = {
        "PATH": str(bare_venv / "bin"),
        "AWS_CONTAINER_CREDENTIALS_FULL_URI": f"{fake_aws_server}/creds",
        "AWS_ENDPOINT_URL_STS": fake_aws_server,
        "AWS_DEFAULT_REGION": "us-east-1",
        "HOME": str(isolated_home),
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    # Override conftest
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config_path))
    monkeypatch.chdir(tmp_path)

    # Apply environment mutations to the current process — botocore's
    # ProcessProvider will inherit them when it spawns the child
    for key, value in child_env.items():
        monkeypatch.setenv(key, value)

    # Attempt to get credentials — must raise CredentialRetrievalError
    session = boto3.Session()
    with pytest.raises(botocore.exceptions.CredentialRetrievalError) as exc_info:
        session.get_credentials()

    # Message must name the module-resolution failure
    error_msg = str(exc_info.value)
    assert "app.scripts.mint_worker_credentials" in error_msg or "No module named" in error_msg, (
        f"Error does not name module-resolution failure: {error_msg}"
    )


def test_helper_runs_from_readonly_tree(fake_aws_server, tmp_path):
    """Helper remains executable after chmod -R a-w (read-only tree mitigation).

    Confirms the read-only-tree mitigation is unaffected by the absolute-script
    invocation. The helper must still import
    and emit JSON after its tree is made non-writable.
    """
    import shutil
    import stat

    # Copy the app package into tmp_path (skip __pycache__ and templates)
    app_src = Path(__file__).parent.parent / "app"
    app_copy = tmp_path / "app"
    shutil.copytree(
        app_src,
        app_copy,
        ignore=shutil.ignore_patterns("__pycache__", "templates"),
    )

    helper_copy = app_copy / "scripts" / "mint_worker_credentials.py"
    assert helper_copy.exists()

    # Make tree non-writable
    for root, dirs, files in os.walk(app_copy):
        for d in dirs:
            os.chmod(os.path.join(root, d), stat.S_IRUSR | stat.S_IXUSR)
        for f in files:
            os.chmod(os.path.join(root, f), stat.S_IRUSR)

    try:
        # Assert helper file has no write bits
        mode = helper_copy.stat().st_mode
        assert not (mode & stat.S_IWUSR), f"Helper still writable: {oct(mode)}"

        # Run the helper as a script with PYTHONDONTWRITEBYTECODE
        isolated_home = tmp_path / "home"
        isolated_home.mkdir(exist_ok=True)
        child_env = _container_like_env(
            fake_aws_server, isolated_home, PYTHONDONTWRITEBYTECODE="1"
        )

        import sys
        result = subprocess.run(
            [
                sys.executable,
                str(helper_copy),
                "--owner-id", "test-owner",
                "--project-id", "test-project",
                "--job-id", "test-job",
                "--role-arn", "arn:aws:iam::123456789012:role/test-role",
            ],
            cwd=tmp_path,
            env=child_env,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, (
            f"Helper failed in read-only tree. stderr: {result.stderr}"
        )
        doc = json.loads(result.stdout)
        assert doc["Version"] == 1
        assert "AccessKeyId" in doc

    finally:
        # Restore write permissions so tmp_path cleanup can succeed
        for root, dirs, files in os.walk(app_copy):
            for d in dirs:
                os.chmod(
                    os.path.join(root, d),
                    stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR,
                )
            for f in files:
                os.chmod(os.path.join(root, f), stat.S_IRUSR | stat.S_IWUSR)
