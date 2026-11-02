# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``dev-prep.sh`` must print its exports on a clean account.

The deploy runbook tells every adopter to run ``eval "$(dev-prep.sh)"``. Under
``set -euo pipefail`` two expected non-zero statuses used to abort the script
with nothing on stdout: ``describe-stacks`` on an auth stack that does not
exist yet, and ``tr`` dying of SIGPIPE when ``head`` stops reading
``/dev/urandom``. The caller's ``eval`` of an empty string succeeds, so the
failure only surfaced later, inside ``cdk deploy``, as an unset
``COGNITO_DOMAIN_PREFIX``.

The script runs here against stub ``aws``, ``cdk`` and ``podman`` commands on
``PATH``, so nothing reaches AWS.
"""

import re
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "dev-prep.sh"

# The stub ``aws`` answers only the calls dev-prep.sh makes. ``AUTH_STACK_URL``
# set means the auth stack is deployed with that hosted-domain URL; unset means
# it does not exist, which the real CLI reports with exit 254.
_AWS_STUB = """#!/bin/bash
case "$*" in
  *"sts get-caller-identity"*) echo 123456789012 ;;
  *"--stack-name ai-eval-studio-auth"*)
    if [[ -n "${AUTH_STACK_URL:-}" ]]; then echo "$AUTH_STACK_URL"; exit 0; fi
    echo "Stack with id ai-eval-studio-auth does not exist" >&2; exit 254 ;;
  *) echo "unexpected aws call: $*" >&2; exit 2 ;;
esac
"""

_CDK_STUB = "#!/bin/bash\nexit 0\n"

_PODMAN_STUB = """#!/bin/bash
case "$1" in
  info) exit 0 ;;
  machine) echo /tmp/podman.sock ;;
esac
"""

_EXPORT_RE = re.compile(r'^export COGNITO_DOMAIN_PREFIX="([^"]*)"$', re.M)


@pytest.fixture
def stub_path(tmp_path):
    """A PATH whose ``aws``, ``cdk`` and ``podman`` are the stubs above."""
    for name, body in (
        ("aws", _AWS_STUB),
        ("cdk", _CDK_STUB),
        ("podman", _PODMAN_STUB),
    ):
        stub = tmp_path / name
        stub.write_text(body)
        stub.chmod(0o755)
    return f"{tmp_path}:/usr/bin:/bin"


def _run(stub_path, **env):
    """Run dev-prep.sh with only the stubs and the given variables."""
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env={"PATH": stub_path, "AWS_REGION": "us-east-1", **env},
        capture_output=True,
        text=True,
        timeout=30,
    )


def _prefix(result):
    """The COGNITO_DOMAIN_PREFIX the script exported, or None."""
    match = _EXPORT_RE.search(result.stdout)
    return match.group(1) if match else None


def test_a_clean_account_gets_a_fresh_random_prefix(stub_path):
    """No auth stack: exit 0 and export a new ``ai-eval-studio-`` + 8 chars."""
    first = _run(stub_path)
    second = _run(stub_path)

    for result in (first, second):
        assert result.returncode == 0, result.stderr
        assert "Step 2: no deployment found" in result.stderr
        assert re.fullmatch(r"ai-eval-studio-[a-z0-9]{8}", _prefix(result) or "")
    assert _prefix(first) != _prefix(second)


def test_a_deployed_prefix_is_reused(stub_path):
    """An existing auth stack keeps its prefix, so the domain is not replaced."""
    result = _run(
        stub_path,
        AUTH_STACK_URL="https://ai-eval-studio-k3x9q2ab.auth.us-east-1.amazoncognito.com",
    )

    assert result.returncode == 0, result.stderr
    assert "(reused)" in result.stderr
    assert _prefix(result) == "ai-eval-studio-k3x9q2ab"


def test_an_explicit_prefix_wins(stub_path):
    """``COGNITO_DOMAIN_PREFIX`` in the environment overrides discovery."""
    result = _run(stub_path, COGNITO_DOMAIN_PREFIX="ai-eval-studio-override")

    assert result.returncode == 0, result.stderr
    assert _prefix(result) == "ai-eval-studio-override"


def test_a_real_failure_names_its_line(stub_path, tmp_path):
    """A genuine error still aborts, and says where instead of failing silently."""
    (tmp_path / "aws").write_text("#!/bin/bash\nexit 255\n")

    result = _run(stub_path)

    assert result.returncode != 0
    assert result.stdout == ""
    assert re.search(r"ERROR: dev-prep\.sh failed at line \d+", result.stderr)
