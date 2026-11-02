# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression gate: no site may silently read optimization state from the wrong home.

Optimization state has two homes — an EFS ``state.json`` and a DynamoDB ``OPT``
item — and which is authoritative depends on the per-project ``arch_split`` flag.
Reading the wrong one does not raise; it returns *plausible* state, so the caller
proceeds against data that does not reflect reality. Eleven sites had simply
omitted the branch and always read EFS, including read-modify-write paths where a
bare EFS write left DynamoDB saying ``running`` while the UI announced
``cancelled``.

This test pins the EXACT set of direct ``OptimizationState.load`` callers. A new
direct caller must fail here and either route through
``optimization_state_repository`` or be added to the allowlist with a written
reason. A count-based assertion would let a new bad caller in whenever an old one
was removed, so the set is pinned by identity.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

# Repo-relative so this passes from any checkout path (a previous test in this
# suite hardcoded /workspace/... and could only pass inside that container —
# worse, it read a different checkout's file and stayed green while its own tree
# regressed).
BACKEND_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BACKEND_DIR / "app"

# ---------------------------------------------------------------------------
# The allowlist. Each entry is a module path mapped to why a DIRECT, sync EFS
# read is correct there — i.e. the caller wants the on-disk artifact, not the
# project's authoritative optimization status.
# ---------------------------------------------------------------------------
# Every module that reads AUTHORITATIVE optimization state now
# goes through the repository, so there is no pending-migration list any more. What
# remains below is the permanent allowlist — sites that deliberately read the
# on-disk artifact rather than the project's authoritative status.
#
# ``api/eval_runs.py`` was on this list for one reason: the bare-UUID run_id ->
# filename resolver correlated iteration rows against files on disk. Deleting that
# shim removed the route's only need to read state at all, so the entry went with
# it — and this test is what caught the leftover.

ALLOWED_DIRECT_LOAD_CALLERS: dict[str, str] = {
    "services/optimization_state_repository.py": (
        "The repository itself — it OWNS the arch_split branch and is the one "
        "module allowed to choose a home."
    ),
    "services/report.py": (
        "Report generation reads run FILES from the on-disk tree; state only "
        "supplies targets/winner for those files. Reports describe the disk."
    ),
    "services/jobs/eval_executor.py": (
        "_resolve_exec_winner correlates on-disk run files to pick a winner."
    ),
}


def _grep_load_callers() -> dict[str, list[int]]:
    """Return {module path relative to app/: [line numbers]} for direct load calls."""
    result = subprocess.run(
        ["grep", "-rn", r"OptimizationState\.load(", str(APP_DIR)],
        capture_output=True,
        text=True,
        check=False,
    )
    callers: dict[str, list[int]] = {}
    for line in result.stdout.splitlines():
        # Format: /abs/path/app/<rel>:<lineno>:<code>
        m = re.match(r"^(.*?):(\d+):(.*)$", line)
        if not m:
            continue
        abs_path, lineno, code = m.group(1), int(m.group(2)), m.group(3)
        stripped = code.strip()
        # Skip comments and docstring prose that merely mention the call.
        if stripped.startswith("#") or stripped.startswith("*"):
            continue
        rel = str(Path(abs_path).relative_to(APP_DIR))
        callers.setdefault(rel, []).append(lineno)
    return callers


def test_direct_load_callers_are_exactly_the_allowlist():
    """The set of modules calling OptimizationState.load directly is pinned.

    Fails on a NEW direct caller (route it through the repository, or allowlist it
    with a reason) and also on a REMOVED one (update the allowlist so it does not
    rot into a permanent lie).
    """
    actual = set(_grep_load_callers())
    allowed = set(ALLOWED_DIRECT_LOAD_CALLERS)

    unexpected = actual - allowed
    assert not unexpected, (
        "New direct OptimizationState.load caller(s): "
        f"{sorted(unexpected)}.\n"
        "Optimization state has TWO homes (EFS json / DynamoDB OPT item) and the "
        "authoritative one depends on the project's arch_split flag. A direct "
        "load() always reads EFS, which on the durable path is the WRONG HOME — "
        "and it fails SILENTLY, returning plausible state.\n"
        "Route the read through app.services.optimization_state_repository "
        "(read/read_for_display), or — if this caller genuinely wants the on-disk "
        "artifact rather than the project's authoritative status — add it to "
        "ALLOWED_DIRECT_LOAD_CALLERS in this file with a one-line reason."
    )

    stale = allowed - actual
    assert not stale, (
        f"Allowlisted module(s) no longer call OptimizationState.load: "
        f"{sorted(stale)}. Remove them from ALLOWED_DIRECT_LOAD_CALLERS so the "
        "allowlist keeps describing reality."
    )


def test_every_allowlist_entry_has_a_reason():
    """An allowlist without reasons is just a suppression list."""
    for module, reason in ALLOWED_DIRECT_LOAD_CALLERS.items():
        assert reason and len(reason) > 30, (
            f"Allowlist entry {module!r} needs a substantive reason explaining why "
            f"a direct EFS read is correct there; got {reason!r}"
        )


def test_coordinator_prompt_does_not_read_state():
    """The coordinator prompt builder must never read optimization state itself.

    It is sync and three of its four call paths are async request handlers, so any
    read inside it could only be the sync EFS read — the wrong home on the durable
    path, which made the agent report stale optimization status to the user. Its
    callers resolve the authoritative state and pass it in.

    Pinned separately from the allowlist because the risk here is a REGRESSION to a
    convenience default, not a new module appearing.
    """
    src = (APP_DIR / "services" / "coordinator_prompt.py").read_text()
    assert "OptimizationState.load" not in src, (
        "coordinator_prompt.py must not read optimization state. Accept it as a "
        "required argument and let the (async) caller resolve it via "
        "optimization_state_repository."
    )


def test_build_coordinator_prompt_requires_opt_state():
    """``opt_state`` must be a REQUIRED parameter, not an optional default.

    An optional parameter is what allowed the silent EFS fallback: a caller that
    forgot to pass state got a plausible-looking prompt built from the wrong home.
    Making it required turns that mistake into a TypeError at the call site.
    """
    import inspect

    from app.services.coordinator_prompt import build_coordinator_prompt

    sig = inspect.signature(build_coordinator_prompt)
    param = sig.parameters["opt_state"]
    assert param.default is inspect.Parameter.empty, (
        "build_coordinator_prompt's opt_state must have NO default. A default "
        "re-enables the silent 'nobody passed state, read EFS' path."
    )
