# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that both ``_project_subprocess_env`` factories propagate ``MANTLE_*`` knobs.

Both factories use ``os.environ.copy()``, so any
``MANTLE_*`` var set in the parent env is automatically forwarded. These
tests assert that contract holds AND lock it in against accidental
future ``env.pop`` lines or env-clearing rewrites.

Two factories live at:

  * ``app.services.harness_runner._project_subprocess_env`` —
    PRODUCTION (sets ``DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=600``,
    keeps the full retry budget).
  * ``app.services.mcp_tools.harness._project_subprocess_env`` —
    VALIDATION (clamps ``BEDROCK_READ_TIMEOUT_S=30`` and
    ``BEDROCK_RETRY_MAX_ATTEMPTS=1`` so a hung-on-Converse model can't
    burn 20 minutes during validation smoke).

Do NOT conflate them. Each
test below exercises ONE factory; if a misguided "consolidate the
factories" change lands, both directions of leak get caught
simultaneously.
"""

from app.services.harness_runner import _project_subprocess_env as production_env
from app.services.mcp_tools.harness import _project_subprocess_env as validation_env


def test_production_factory_propagates_mantle_env(monkeypatch):
    """Production factory in harness_runner.py propagates MANTLE_* vars."""
    monkeypatch.setenv("MANTLE_MAX_CONNECTIONS", "42")
    monkeypatch.setenv("MANTLE_MAX_KEEPALIVE", "11")
    monkeypatch.setenv("MANTLE_BOOT_SMOKE", "1")
    env = production_env()
    assert env.get("MANTLE_MAX_CONNECTIONS") == "42"
    assert env.get("MANTLE_MAX_KEEPALIVE") == "11"
    assert env.get("MANTLE_BOOT_SMOKE") == "1"
    # Production-specific knob still set — no regression on the
    # 600s gather override that fixes the dropped-cases bug.
    assert env.get("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE") == "600"


def test_validation_factory_propagates_mantle_env(monkeypatch):
    """Validation factory in mcp_tools/harness.py propagates MANTLE_* vars."""
    monkeypatch.setenv("MANTLE_MAX_CONNECTIONS", "7")
    env = validation_env()
    assert env.get("MANTLE_MAX_CONNECTIONS") == "7"
    # Validation-specific clamps still set — non-invocable model
    # can't burn 20 min on Converse retries during smoke.
    assert env.get("BEDROCK_READ_TIMEOUT_S") == "30"
    assert env.get("BEDROCK_RETRY_MAX_ATTEMPTS") == "1"


def test_validation_factory_does_not_leak_into_production(monkeypatch):
    """Validation's BEDROCK_READ_TIMEOUT_S=30 must NOT appear in the production env."""
    monkeypatch.delenv("BEDROCK_READ_TIMEOUT_S", raising=False)
    env = production_env()
    # Production factory does NOT set BEDROCK_READ_TIMEOUT_S — it
    # inherits the parent value (None here). If a future commit
    # mistakenly adds the validation clamp here, this test fails.
    assert env.get("BEDROCK_READ_TIMEOUT_S") is None


def test_production_factory_does_not_leak_into_validation(monkeypatch):
    """Production's gather override must NOT appear in the validation env."""
    monkeypatch.delenv("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", raising=False)
    env = validation_env()
    assert env.get("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE") is None


def test_both_factories_strip_virtual_env(monkeypatch):
    """Both factories pop VIRTUAL_ENV so ``uv run`` activates the project's own venv cleanly."""
    monkeypatch.setenv("VIRTUAL_ENV", "/path/to/parent/venv")
    assert "VIRTUAL_ENV" not in production_env()
    assert "VIRTUAL_ENV" not in validation_env()


def test_both_factories_strip_uv_project_environment(monkeypatch):
    """Both factories pop UV_PROJECT_ENVIRONMENT so a child ``uv run`` in the harness
    project resolves the project's OWN environment, not the coordinator's pinned venv.

    Without this strip, a harness project's ``uv run``/``uv sync`` inherits the
    coordinator's ``UV_PROJECT_ENVIRONMENT`` (used in dev to relocate the backend
    venv) and re-syncs that shared venv against the harness project's own
    dependency set — silently mutating the backend environment out from under a
    running process. This test locks the strip in against a future env-copy
    rewrite that forgets it.
    """
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/opt/venvs/ai-eval-studio")
    assert "UV_PROJECT_ENVIRONMENT" not in production_env()
    assert "UV_PROJECT_ENVIRONMENT" not in validation_env()


def test_both_factories_propagate_mantle_discovery_regions(monkeypatch):
    """MANTLE_DISCOVERY_REGIONS propagates to both subprocess envs.

    ``MANTLE_DISCOVERY_REGIONS`` is the env knob the discovery orchestrator
    reads to resolve the per-region scan set. Both subprocess factories use
    ``os.environ.copy()``, so propagation is automatic — this test LOCKS THAT
    IN against a future ``env.pop`` line or env-clearing rewrite.

    The harness subprocess never imports ``bedrock_discovery`` (the
    discovery layer is coordinator-only), but the env knob must still reach
    the child process unchanged.
    """
    monkeypatch.setenv("MANTLE_DISCOVERY_REGIONS", "us-east-1,eu-central-1,ap-south-1")
    prod = production_env()
    val = validation_env()
    assert prod.get("MANTLE_DISCOVERY_REGIONS") == "us-east-1,eu-central-1,ap-south-1"
    assert val.get("MANTLE_DISCOVERY_REGIONS") == "us-east-1,eu-central-1,ap-south-1"

    # And the "all" sentinel propagates equally — there is
    # no special-case handling on the parent side, so the child sees
    # the same value the parent set.
    monkeypatch.setenv("MANTLE_DISCOVERY_REGIONS", "all")
    assert production_env().get("MANTLE_DISCOVERY_REGIONS") == "all"
    assert validation_env().get("MANTLE_DISCOVERY_REGIONS") == "all"
