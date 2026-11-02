# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report whether long-lived background work is in flight in this process.

The AgentCore runtime is a second execution home for the same coordinator turn
code. Anything the agent dispatches there — most consequentially the optimization
loop, created with ``asyncio.create_task`` in whichever process handled the
dispatch — outlives the turn that created it. AgentCore's readiness contract
decides session lifetime from the ``/ping`` answer: a session reporting
``Healthy`` is counted against the idle timer and terminated once the configured
idle window elapses, while ``HealthyBusy`` keeps it alive. A runtime that always
answered ``Healthy`` therefore had its own background work killed mid-flight,
stranding jobs non-terminal with no cleanup.

This module supplies the one predicate that answer depends on. It follows the
resolver shape already established by :mod:`app.platform.process_role` and
:mod:`app.platform.arch_split` — a module constant plus a zero-argument accessor
named for the thing it resolves — so the recognized set is inspectable and
testable without reaching into the probe.

It lives under ``app/services/`` rather than beside ``process_role.py`` because
the names it recognizes are service-layer task names: keeping the constant in the
same layer as the code that creates those tasks means a new named task and its
recognition live side by side.
"""
from __future__ import annotations

import asyncio

# Prefixes of the long-lived named tasks whose presence must hold a session open.
#
# This is deliberately an ALLOWLIST, not "any live task". The event loop always
# carries tasks that have nothing to do with our work — uvicorn's per-connection
# handlers, the lifespan task, the server loop itself — so a predicate that
# reported busy whenever any task was alive would report busy forever. That never
# idle-terminates, which means the session holds runtime quota for its entire
# max-lifetime ceiling on every invoke: a quota leak dressed as a fix.
#
# Derived from the ``name=`` arguments at the ``asyncio.create_task`` call sites
# under ``app/services/``. ``test_agentcore_ping_busy`` re-derives that set from
# source and fails if a new named task is added without a covering entry here, so
# this list cannot silently fall behind the code that creates the tasks.
_BACKGROUND_TASK_NAME_PREFIXES: tuple[str, ...] = (
    "optimization-",       # the optimization loop (longest-lived by far)
    "job-eval-",           # manual eval dispatch driver
    "job-harness-",        # harness build driver
    "eval-",               # the optimization orchestrator's child eval
    "agent-turn-",         # an in-process SDK turn
    "completion-turn-",    # the post-job completion turn
    "claim-release-",      # deferred release of a session claim
    "progress-poller-",    # durable job-progress bridge
    "turn-poller-",        # durable turn-event bridge
    "reconciler-loop",     # periodic ECS/DDB reconcile
    "ws-hub-keepalive",    # WebSocket keepalive pump
)


def has_background_work_in_flight() -> bool:
    """Return whether a recognized long-lived background task is still running.

    Scans the running loop's tasks for an unfinished task whose name starts with
    a prefix in :data:`_BACKGROUND_TASK_NAME_PREFIXES`, skipping the calling task
    so a probe never counts itself.

    This stays synchronous and allocation-light on purpose: it is called from the
    readiness probe, whose fast-answer guarantee is load-bearing. A probe that
    awaited I/O could hang, and a hanging readiness probe is what causes a
    restart loop — strictly worse than the idle termination this exists to avoid.

    Returns:
        ``True`` if at least one recognized background task is still running in
        this process; ``False`` when none is, or when there is no running loop.
    """
    try:
        tasks = asyncio.all_tasks()
    except RuntimeError:
        # No running loop (e.g. called from sync context) — nothing can be in
        # flight in a loop that isn't running.
        return False

    try:
        current = asyncio.current_task()
    except RuntimeError:
        current = None

    # No ``done()`` filter here: ``asyncio.all_tasks()`` already excludes finished
    # tasks (it returns only tasks where ``not t.done()``), so a completed task can
    # never appear in this scan and re-checking would be unreachable code. Finished
    # work therefore releases the session automatically.
    for task in tasks:
        if task is current:
            continue
        if task.get_name().startswith(_BACKGROUND_TASK_NAME_PREFIXES):
            return True
    return False
