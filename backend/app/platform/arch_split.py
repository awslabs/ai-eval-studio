# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Durable-state path resolver (``resolve_arch_split``) and the cost kill switch.

This module provides the resolve-once resolver that determines whether an
operation uses the durable-state path (DynamoDB-authoritative jobs, event log,
OptimizationState) or the in-memory/EFS-JSON path. It is an environment check:
cloud mode is always durable, dev mode never is.

**Thread-safety / correctness guarantee:** Callers MUST resolve the flag exactly
ONCE at the start of each operation (request / turn / dispatch) and thread the
resolved boolean through the call stack. Never re-read the flag mid-operation.
This once-and-thread rule is what keeps one operation on a single path, never
split across both (prevents dual-writer corruption).

**Cost kill-switch:** `EVAL_STUDIO_COST_KILL_SWITCH` (env var)
halts new eval launches (dispatch checks it) AND authorizes the reconciler's
aggressive StopTask-all mode. This is an emergency brake on top of the
steady-state reconciler. The switch reads from the process environment (never from
a request/tenant), so no tenant can trip another tenant's switch. Flipping
the switch requires an operator env change + task restart. Resolve-once discipline:
the flag is a process-lifetime constant; never re-read os.environ mid-operation.
"""

import os
from typing import TYPE_CHECKING

from app.platform import get_platform
from app.platform.types import ProjectId

if TYPE_CHECKING:
    pass


def _resolve_cost_kill_switch() -> bool:
    """Resolve cost kill-switch from env (checked once at import time).

    Returns True iff the EVAL_STUDIO_COST_KILL_SWITCH env var is set to a truthy
    value ("1" or "true", case-insensitive). This is the operator-only override:
    the value comes from the process environment, never from a request/tenant, so
    no tenant can trip another tenant's kill switch.

    Environment-only by design: an operator flips it with an env change and a
    task restart, and nothing a request carries can reach it.

    Returns:
        True if the kill switch is active, False otherwise.
    """
    env_flag = os.environ.get("EVAL_STUDIO_COST_KILL_SWITCH", "").lower()
    return env_flag in ("1", "true")


# Cost kill-switch constant — resolved once at import time.
# Halts new eval launches (dispatch checks this) AND authorizes the
# reconciler's aggressive StopTask-all mode. Operator-scoped (not
# tenant-scoped) — no tenant can trip another tenant's switch. Flipping requires
# task restart.
COST_KILL_SWITCH_ENABLED = _resolve_cost_kill_switch()


def is_cost_kill_switch_active() -> bool:
    """Return whether the cost kill-switch is active (operator-scoped).

    This accessor takes zero arguments — the switch is process-scoped, not
    tenant-scoped. It returns the module constant resolved once at
    import time. Tests monkeypatch the module constant, call this accessor.

    Returns:
        True if the kill switch is active, False otherwise.
    """
    return COST_KILL_SWITCH_ENABLED


async def resolve_arch_split(project_id: ProjectId) -> bool:
    """Resolve whether an operation uses the durable-state path (once per operation).

    Args:
        project_id: Project identifier.

    Returns:
        ``True`` in cloud (the durable-state path: DDB-authoritative jobs,
        event log, OptimizationState), ``False`` in dev (in-process path —
        dev has no Fargate worker to route to).

    This is an ENVIRONMENT check, not a per-project choice.
    The per-project ``arch_split_enabled`` flag and the
    ``ARCH_SPLIT_KILL_SWITCH`` env var are dead: stored flag values are
    ignored, and the inline comment explains why.

    **Resolve once, thread the boolean:** Callers must resolve the flag ONCE at
    the operation entry point (request / turn / dispatch) and pass the resolved
    boolean as an explicit argument down the call chain.
    """
    # Cloud-only gate: the durable path is only available in cloud mode.
    # Dev cannot route to FargateTaskCompute, so force False outside cloud.
    mode = os.environ.get("EVAL_STUDIO_MODE", "dev").lower()
    if mode != "cloud":
        return False

    # Cloud IS the durable path — unconditionally.
    #
    # This resolver used to read a per-project flag here. That flag went wrong
    # two ways at once: the schema default wrote an EXPLICIT False into every
    # new project at creation (so the cloud default of True was dead code), and
    # a project on the in-process path ran its optimization loop inside the
    # microVM with its state in an EFS JSON that the coordinator's adoption
    # sweep cannot see (it point-reads the DDB store). Such a loop died with the
    # first microVM recycle and showed "running" forever — invisible to the
    # reconciler, adoption, and Stop alike.
    #
    # The target store model: EFS holds ONLY the project working tree (harness
    # code, data, prompts, run outputs — the downloadable and agent-cwd
    # surface); internal state lives in DynamoDB; agent transcripts/sessions
    # live in S3. A per-project branch between two state homes has no place in
    # that model, and neither does a kill switch that forces the fleet onto the
    # in-process path (that is not a rollback — it reproduces the failure above
    # on demand). Emergency rollback is redeploying the previous revision.
    return True


async def resolve_agentcore_relay(project_id: ProjectId) -> bool:  # noqa: ARG001 — id kept for call-site stability until the resolver is deleted
    """Resolve the user-turn path — an ENVIRONMENT check, not a per-project flag.

    Returns:
        ``True`` in cloud (the AgentCore relay is the sole user-turn path),
        ``False`` in dev (no AgentCore runtime to invoke — in-process SDK
        turns are dev's only path).

    Same reasoning that killed arch_split's per-project
    read: a per-project store-read here conflates "where does compute run"
    (an environment fact) with a per-project choice, and carries the same
    resolution bugs (requester vs owner, context-less background callers). The
    per-project ``agentcore_relay_enabled`` field and the
    ``AGENTCORE_RELAY_KILL_SWITCH`` env var are dead: stored values are
    ignored. Forcing cloud turns onto the in-process path would run them on
    the coordinator — a path no longer exercised in cloud and therefore an
    UNTESTED escape hatch, worse than the real rollback (redeploy the
    previous revision).

    **Resolve once, thread the boolean:** Callers must resolve ONCE at the
    operation entry point and pass the boolean down the call chain.
    """
    mode = os.environ.get("EVAL_STUDIO_MODE", "dev").lower()
    return mode == "cloud"
