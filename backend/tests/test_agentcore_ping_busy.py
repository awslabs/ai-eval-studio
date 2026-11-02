# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral proof that the AgentCore /ping busy signal controls session lifetime.

AgentCore counts a ``Healthy`` ping against the idle timer and terminates the
session once the idle window elapses; ``HealthyBusy`` holds it open. The runtime
spawns background work that outlives the turn that created it, so these tests pin
both directions: busy while recognized work runs, idle otherwise.

The idle direction matters as much as the busy one. A predicate that reported busy
unconditionally would never idle-terminate and would hold runtime quota for the
whole max-lifetime ceiling on every invoke, so the finished-task and
unrecognized-name cases are load-bearing, not incidental.

Also guards the two properties that keep the fix safe: the predicate stays
synchronous (a hanging readiness probe causes a restart loop, which is worse than
idle termination), and the recognized-name list cannot fall behind the code that
creates named tasks.
"""
import ast
import asyncio
import inspect
import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agentcore_runtime import ping
from app.services import background_work
from app.services.background_work import (
    _BACKGROUND_TASK_NAME_PREFIXES,
    has_background_work_in_flight,
)


@pytest.fixture
def client():
    """Test client for the standalone AgentCore runtime app."""
    from app.agentcore_runtime import app

    return TestClient(app)


async def _ping_while_task_named(task_name: str) -> dict[str, str]:
    """Run a parked task under ``task_name`` and return the /ping answer.

    Creates a real asyncio task that blocks on an Event so it is genuinely
    unfinished while the probe runs, then releases and awaits it so no task leaks
    into another test.

    Args:
        task_name: Name to give the parked background task.

    Returns:
        The dict returned by the ``/ping`` handler while that task was running.
    """
    release = asyncio.Event()
    task = asyncio.create_task(release.wait(), name=task_name)
    await asyncio.sleep(0)  # let the task start
    try:
        return await ping()
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_ping_idle_reports_healthy():
    """With no recognized background task running, /ping reports Healthy."""
    assert await ping() == {"status": "Healthy"}


@pytest.mark.asyncio
async def test_ping_reports_busy_for_optimization_loop():
    """A live optimization-loop task holds the session open."""
    assert await _ping_while_task_named("optimization-abc12345") == {
        "status": "HealthyBusy"
    }


@pytest.mark.asyncio
async def test_ping_reports_busy_for_agent_turn():
    """A live agent-turn task holds the session open."""
    assert await _ping_while_task_named("agent-turn-deadbeef") == {
        "status": "HealthyBusy"
    }


@pytest.mark.asyncio
async def test_ping_reports_idle_after_recognized_task_finishes():
    """Once recognized work finishes, the session goes back to idle.

    This is the release half of the contract: work that completed must stop
    holding the session, or a single optimization run would pin runtime quota for
    the whole max-lifetime ceiling.

    The mechanism is ``asyncio.all_tasks()`` itself, which returns only tasks
    where ``not t.done()`` (verified in CPython's ``asyncio.tasks.all_tasks``), so
    a finished task cannot appear in the scan at all. The test pins the resulting
    behaviour end to end — busy while running, idle once done — rather than the
    implementation detail.
    """
    release = asyncio.Event()
    task = asyncio.create_task(release.wait(), name="optimization-abc12345")
    await asyncio.sleep(0)
    assert await ping() == {"status": "HealthyBusy"}

    release.set()
    await task
    assert task.done()
    assert await ping() == {"status": "Healthy"}


@pytest.mark.asyncio
async def test_ping_reports_idle_for_unrecognized_task_name():
    """An unrecognized live task must NOT report busy.

    uvicorn's own connection and lifespan tasks are always present, so treating
    any live task as background work would report busy permanently.
    """
    assert await _ping_while_task_named("uvicorn-connection-7") == {
        "status": "Healthy"
    }


def test_ping_contract_via_client(client):
    """The HTTP contract still holds: 200 with a status field."""
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json()["status"] in ("Healthy", "HealthyBusy")


def test_no_await_in_predicate():
    """The predicate must stay synchronous and free of awaited I/O.

    The readiness probe's fast-answer guarantee is load-bearing: a slow or hanging
    probe causes a restart loop.
    """
    assert not asyncio.iscoroutinefunction(has_background_work_in_flight)

    # Parse rather than substring-match: prose in the docstring legitimately
    # mentions awaiting, and a grep would flag that instead of real control flow.
    source = inspect.getsource(has_background_work_in_flight)
    tree = ast.parse(textwrap.dedent(source))
    awaits = [n for n in ast.walk(tree) if isinstance(n, (ast.Await, ast.AsyncWith, ast.AsyncFor))]
    assert not awaits, (
        "has_background_work_in_flight must not await — it runs inside the "
        "readiness probe, where a hang causes a restart loop."
    )


def _named_create_task_prefixes() -> dict[str, str]:
    """Collect every literal ``name=`` prefix passed to create_task under services.

    Returns:
        Mapping of literal name prefix to the ``file:line`` that produced it.
        f-string names reduce to the literal text before the first placeholder.
    """
    # Walk ALL of app/, not just app/services/. The optimization loop is also
    # created in api/optimization.py (``start_optimization_rest`` and
    # ``resume_optimization``), and a
    # services-only walk covered those two by NAME COINCIDENCE alone — they happen
    # to use the same "optimization-" prefix a services site already registered. A
    # differently-named long-lived task created under app/api/ would have lost idle
    # protection with this guard still green.
    app_dir = Path(background_work.__file__).resolve().parent.parent
    found: dict[str, str] = {}

    for path in sorted(app_dir.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            attr = (
                func.attr
                if isinstance(func, ast.Attribute)
                else (func.id if isinstance(func, ast.Name) else "")
            )
            if attr != "create_task":
                continue
            for kw in node.keywords:
                if kw.arg != "name":
                    continue
                value = kw.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    literal = value.value
                elif isinstance(value, ast.JoinedStr):
                    literal = ""
                    for part in value.values:
                        if isinstance(part, ast.Constant) and isinstance(part.value, str):
                            literal += part.value
                        else:
                            break
                else:
                    continue
                if literal:
                    rel = path.relative_to(app_dir)
                    found[literal] = f"{rel}:{node.lineno}"
    return found


# Named tasks that deliberately do NOT hold a microVM session open, with the reason.
#
# The busy-allowlist is microVM-scoped: its only job is to stop AgentCore's idle
# timer from killing work the microVM is still doing. A task that can only ever
# exist in the coordinator process is irrelevant to that and must NOT be added —
# adding it would be harmless in effect but misleading about where the task runs.
#
# Each entry is asserted to be genuinely unreachable from the microVM by
# test_excluded_tasks_are_coordinator_only below, so this cannot become a dumping
# ground for anything inconvenient.
_COORDINATOR_ONLY_TASK_PREFIXES: dict[str, str] = {
    "relay-turn-": (
        "The coordinator's outbound INVOKE of the microVM. It exists only in the "
        "process making the call — the microVM is the callee, and its runtime app "
        "(agentcore_runtime.py) never mounts POST /chat."
    ),
    "ws-pump-": (
        "Per-browser-WebSocket outbound pump. The microVM holds zero WebSocket "
        "connections and its runtime app never mounts /ws."
    ),
    "provisional-bubble-": (
        "Fire-and-forget seed of the user's message at turn start, scheduled "
        "from the chat POST endpoint — coordinator-only by construction (the "
        "microVM never serves the chat API; its relay path writes the "
        "provisional bubble inline in agentcore_relay_client)."
    ),
    "transcript-backfill-": (
        "Detached transcript backfill for the whole-message flush race. Spawned "
        "only from the transcript append call site, which is gated on "
        "``not is_agentcore_runtime()`` (single-writer: inside the microVM "
        "the coordinator's relay drain owns the append) — so this task can only "
        "exist on the coordinator."
    ),
    "report-regenerate-": (
        "Manual report regeneration, started by the two regenerate endpoints in "
        "api/reports.py. The microVM's runtime app mounts no API routers, so only "
        "the coordinator serves those endpoints."
    ),
}


def test_every_named_background_task_is_covered():
    """Every named task created under app/ must be recognized OR explicitly excluded.

    Source-derived so the allowlist cannot silently fall behind: a new named
    background task added without a covering prefix fails here instead of losing
    idle protection in production.

    The walk covers ALL of app/, not just app/services/. A services-only walk (the
    original scope) covered the optimization loop created at api/optimization.py by
    NAME COINCIDENCE — it happens to reuse a prefix a services site had registered.
    Widening it immediately surfaced `relay-turn-` and `ws-pump-` as unclassified.
    """
    discovered = _named_create_task_prefixes()
    assert discovered, "AST walk found no named create_task calls — walker is broken"

    uncovered = {
        name: where
        for name, where in discovered.items()
        if not name.startswith(_BACKGROUND_TASK_NAME_PREFIXES)
        and name not in _COORDINATOR_ONLY_TASK_PREFIXES
    }
    assert not uncovered, (
        "Named background tasks neither covered by _BACKGROUND_TASK_NAME_PREFIXES "
        "nor listed in _COORDINATOR_ONLY_TASK_PREFIXES. If the microVM can run it, "
        "add a busy prefix (it would otherwise lose idle protection); if it is "
        f"coordinator-only, list it with a reason: {uncovered}"
    )


def test_excluded_tasks_are_coordinator_only():
    """Each exclusion must be justified AND unreachable from the microVM app.

    Keeps the exclusion list from becoming a way to silence the guard: the reason
    has to be substantive, and the creating module must not be reachable from
    agentcore_runtime's router set.
    """
    discovered = _named_create_task_prefixes()

    for prefix, reason in _COORDINATOR_ONLY_TASK_PREFIXES.items():
        assert len(reason) > 40, (
            f"exclusion {prefix!r} needs a substantive reason, got {reason!r}"
        )
        # If the prefix is no longer created anywhere, the exclusion is stale.
        assert any(name == prefix for name in discovered), (
            f"exclusion {prefix!r} no longer matches any create_task name — remove "
            "it so the list keeps describing reality"
        )

    # The microVM's app must not mount the routers that create these tasks.
    runtime_src = (
        Path(background_work.__file__).resolve().parent.parent
        / "agentcore_runtime.py"
    ).read_text()
    for router in ("chat.router", "ws.router", "include_router"):
        assert router not in runtime_src, (
            f"agentcore_runtime.py now references {router!r} — the "
            "coordinator-only exclusions above may no longer hold, because the "
            "microVM could reach those task-creating call sites."
        )


def test_never_terminating_loops_are_not_startable_in_the_microvm():
    """The inverse assertion: perpetual loops must be gated on process role.

    ``ws-hub-keepalive`` and ``reconciler-loop`` never finish by design. If either
    could start inside the microVM it would report busy forever, so the session
    would never idle-terminate and would hold runtime quota to its max-lifetime
    ceiling on every invoke — a quota leak wearing the costume of a fix.

    Asserted structurally (the starters are not reachable from the microVM app)
    rather than by running them.
    """
    app_dir = Path(background_work.__file__).resolve().parent.parent
    runtime_src = (app_dir / "agentcore_runtime.py").read_text()

    for starter in ("start_keepalive", "start_reconciler_loop"):
        assert starter not in runtime_src, (
            f"agentcore_runtime.py calls {starter!r}. That task never terminates, so "
            "the microVM would answer HealthyBusy forever and never idle-terminate — "
            "holding runtime quota to the max-lifetime ceiling on every invoke."
        )
