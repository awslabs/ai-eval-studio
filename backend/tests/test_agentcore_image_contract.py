# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentCore image contract tests (offline, no docker build).

Validates the Dockerfile.agentcore content via text assertions, without
running `docker build` (the suite needs no Docker and no network), PLUS asserts
that the heavy/native dependencies all have aarch64 manylinux wheels in the
committed uv.lock so the ARM64 build is wheel-only — no source builds forced.

The real ARM64 image build and a runtime readiness check need Docker and a
deploy; these offline tests gate only what can be validated without a build.
"""
import re
from pathlib import Path


def test_dockerfile_contract():
    """Offline Dockerfile text assertions for AgentCore ARM64 image.

    Asserts that backend/Dockerfile.agentcore contains:
    - Platform linux/arm64 (both FROM lines)
    - Non-root USER app created as uid/gid 1000
    - CLAUDE_CODE_USE_BEDROCK=1 in ENV
    - EXPOSE 8080
    - CMD binding 0.0.0.0:8080 and targeting app.agentcore_runtime:app

    Does NOT run the build.
    """
    dockerfile_path = Path(__file__).parent.parent / "Dockerfile.agentcore"
    assert dockerfile_path.exists(), "backend/Dockerfile.agentcore must exist"

    content = dockerfile_path.read_text()

    # Assert ARM64 platform on both FROM lines
    from_lines = [line for line in content.splitlines() if line.strip().startswith("FROM")]
    assert len(from_lines) >= 2, "Expected at least 2 FROM stages"
    for line in from_lines:
        # Either explicit --platform=linux/arm64 or arm64 in a base image name would work
        assert "--platform=linux/arm64" in line or "arm64" in line.lower(), (
            f"FROM line must specify ARM64 platform: {line}"
        )

    # Assert non-root user created as uid/gid 1000
    assert re.search(r"useradd.*-u\s+1000.*-g\s+(app|1000)", content), (
        "USER app must be created with uid 1000"
    )
    assert re.search(r"groupadd.*-g\s+1000\s+app", content), (
        "Group app must be created with gid 1000"
    )
    assert "USER app" in content, "Must switch to USER app"

    # Assert CLAUDE_CODE_USE_BEDROCK=1
    assert "CLAUDE_CODE_USE_BEDROCK=1" in content, (
        "ENV must set CLAUDE_CODE_USE_BEDROCK=1"
    )

    # Assert EXPOSE 8080
    assert "EXPOSE 8080" in content, "Must EXPOSE 8080 (AgentCore port)"

    # Assert CMD binds 0.0.0.0:8080 and targets app.agentcore_runtime:app
    cmd_pattern = r'CMD\s*\[.*"--host".*"0\.0\.0\.0".*"--port".*"8080".*\]'
    assert re.search(cmd_pattern, content, re.DOTALL), (
        "CMD must bind --host 0.0.0.0 --port 8080"
    )
    assert "app.agentcore_runtime:app" in content, (
        "CMD must target app.agentcore_runtime:app (NOT app.main:app)"
    )


def test_aarch64_wheels_available():
    """Assert aarch64 manylinux wheels exist in uv.lock for heavy dependencies.

    Offline proxy for "the ARM64 build won't force a slow/failing source compile."
    Asserts that claude_agent_sdk (at WHATEVER version pyproject pins — read,
    not hardcoded, so a version bump can't silently skip the check or fail it
    spuriously), pandas, numpy, and pydantic_core all publish a
    manylinux_*_aarch64 wheel.

    The ARM64 `docker buildx` installs from these wheels (uv sync --frozen).
    For claude_agent_sdk, we specifically assert a Linux ARM64 wheel
    (manylinux_*_aarch64) exists — not just the macosx_11_0_arm64 wheel (which
    wouldn't be pulled by a Linux ARM64 build). The real build happens during
    ``cdk deploy``.
    """
    lockfile_path = Path(__file__).parent.parent / "uv.lock"
    assert lockfile_path.exists(), "backend/uv.lock must exist"

    lockfile = lockfile_path.read_text()

    # Read the pinned SDK version from pyproject — never hardcode it here
    # (the old literal made every SDK bump fail this test for no real reason).
    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    pin = re.search(r'claude-agent-sdk==([\d.]+)', pyproject)
    assert pin, "pyproject.toml must pin claude-agent-sdk with =="
    sdk_version = re.escape(pin.group(1))

    # Assert the PINNED claude-agent-sdk has a manylinux_*_aarch64 wheel
    claude_sdk_pattern = rf"claude[_-]agent[_-]sdk-{sdk_version}-.*manylinux_\d+_\d+_aarch64\.whl"
    assert re.search(claude_sdk_pattern, lockfile), (
        f"claude-agent-sdk {pin.group(1)} must have a manylinux_*_aarch64 wheel in uv.lock "
        "(not just macosx_11_0_arm64 — the Linux ARM64 build needs the manylinux variant)"
    )

    # Assert pandas has an aarch64 wheel (any version in the lock)
    pandas_pattern = r"pandas-.*-.*aarch64.*\.whl"
    assert re.search(pandas_pattern, lockfile), (
        "pandas must have an aarch64 wheel in uv.lock"
    )

    # Assert numpy has an aarch64 wheel
    numpy_pattern = r"numpy-.*-.*aarch64.*\.whl"
    assert re.search(numpy_pattern, lockfile), (
        "numpy must have an aarch64 wheel in uv.lock"
    )

    # Assert pydantic_core has an aarch64 wheel
    pydantic_core_pattern = r"pydantic[_-]core-.*-.*aarch64.*\.whl"
    assert re.search(pydantic_core_pattern, lockfile), (
        "pydantic_core must have an aarch64 wheel in uv.lock"
    )
