# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard on the SPA bundling options, which no synth test can reach.

``BucketDeployment`` bundles at CONSTRUCTION time, so importing
``frontend_stack`` at all shells out to podman and runs the real ``npm ci``.
That makes a ``Template.from_stack`` test on this stack impractical — which is
why the stack had no test, and why a broken bundling config shipped and was
only found by running ``cdk``. It does not fail in one stack either: every cdk
command synthesizes the whole app, so a bundling error here blocks deploys of
completely unrelated stacks.

So this reads the options out of the source instead. The invariant is the pair:
VOLUME_COPY and ``user="0:0"`` are only correct TOGETHER.

* VOLUME_COPY alone breaks. CDK's helper
  (``core/lib/private/asset-staging.js``, ``AssetBundlingVolumeCopy.run``)
  chowns ``/asset-input`` to the host uid:gid and only then runs ``docker cp
  <source>/. copyContainer:/asset-input``. ``docker cp`` without ``--archive``
  lands every entry root-owned, so a build running as the host uid can create
  files but cannot delete any copied one — and ``npm ci`` begins by removing
  ``node_modules``. Result: ``EACCES: permission denied, rmdir
  '/asset-input/node_modules/.bin'``.
* ``user="0:0"`` alone (with the default BIND_MOUNT) would be worse than the
  bug it fixes: root writing straight into the host's ``frontend/``.

Neither half looks load-bearing on its own, and dropping either is a one-line
edit, which is exactly the shape this file exists to catch.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_FRONTEND_STACK = Path(__file__).resolve().parent.parent / "cdk" / "frontend_stack.py"


@pytest.fixture(scope="module")
def bundling_options() -> dict[str, ast.expr]:
    """The ``bundling=`` dict passed to ``s3deploy.Source.asset``, keyed by name.

    Parsed rather than string-matched so comments and formatting cannot make
    the assertions pass or fail spuriously.
    """
    tree = ast.parse(_FRONTEND_STACK.read_text())

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "bundling" and isinstance(keyword.value, ast.Dict):
                return {
                    key.value: value
                    for key, value in zip(keyword.value.keys, keyword.value.values)
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }

    pytest.fail(
        "no call with a literal bundling={...} dict found in frontend_stack.py — "
        "if the SPA build moved elsewhere, move this guard with it"
    )


def test_bundling_uses_volume_copy(bundling_options):
    """BIND_MOUNT writes through to the host's frontend/ working tree.

    Under the default, ``npm ci`` ran in the real tree and replaced the
    developer's macOS ``node_modules`` with the container's linux-arm64
    bindings on every deploy, breaking ``npm run build`` and ``npm test``
    until someone reinstalled them by hand.
    """
    assert "bundling_file_access" in bundling_options, (
        "bundling_file_access is absent, so CDK's default BIND_MOUNT applies "
        "and the deploy writes into the host's frontend/"
    )
    assert "VOLUME_COPY" in ast.unparse(bundling_options["bundling_file_access"])


def test_volume_copy_builds_as_root(bundling_options):
    """The other half of the pair — see the module docstring for why."""
    assert "user" in bundling_options, (
        "VOLUME_COPY without user=\"0:0\" fails the whole synth on EACCES: the "
        "copied tree is root-owned but the build would run as the host uid"
    )
    user = bundling_options["user"]
    assert isinstance(user, ast.Constant) and user.value == "0:0", (
        f"expected user=\"0:0\" to match the ownership docker cp actually "
        f"produces, got {ast.unparse(user)}"
    )


def test_build_clears_host_state_before_installing(bundling_options):
    """The bundle must not depend on the developer's working state.

    ``node_modules`` and ``dist`` both arrive in the volume from the host and
    both are root-owned; removing them explicitly keeps the bundle reproducible
    and stops vite tripping over a stale ``dist`` it cannot empty.
    """
    command = ast.unparse(bundling_options["command"])
    assert "rm -rf node_modules dist" in command
    assert "npm ci" in command
    assert command.index("rm -rf") < command.index("npm ci"), (
        "the clean step has to run before npm ci, not after"
    )
