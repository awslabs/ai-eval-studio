# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-project asyncio locks (``services.project_locks``)."""

from __future__ import annotations

import asyncio

import pytest

from app.services import project_locks


@pytest.fixture(autouse=True)
def _reset_registry():
    """Clear the lock registry between tests so fixture state doesn't leak."""
    project_locks._reset()
    yield
    project_locks._reset()


async def test_same_project_returns_same_lock():
    """Repeat calls for the same project id return the identical Lock object."""
    lock_a = await project_locks.get_project_lock("proj-1")
    lock_b = await project_locks.get_project_lock("proj-1")
    assert lock_a is lock_b


async def test_different_projects_return_distinct_locks():
    """Different project ids get independent locks (no cross-project serialization)."""
    lock_a = await project_locks.get_project_lock("proj-a")
    lock_b = await project_locks.get_project_lock("proj-b")
    assert lock_a is not lock_b


async def test_lock_serializes_concurrent_sections():
    """Two coroutines holding the same project's lock run strictly sequentially."""
    events: list[str] = []

    async def critical(name: str) -> None:
        lock = await project_locks.get_project_lock("proj-x")
        async with lock:
            events.append(f"enter:{name}")
            # Yield so the other task gets a chance to interleave if the
            # lock wasn't actually held. If interleaving does happen this
            # test will record ``enter:a, enter:b`` instead of the
            # expected strictly-nested pattern.
            await asyncio.sleep(0)
            events.append(f"exit:{name}")

    await asyncio.gather(critical("a"), critical("b"))

    # Whichever coroutine won the lock first must fully exit before the
    # other enters. Nested order is what we're asserting.
    assert events in (
        ["enter:a", "exit:a", "enter:b", "exit:b"],
        ["enter:b", "exit:b", "enter:a", "exit:a"],
    )
