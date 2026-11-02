# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wait for background work by its effect, not by a guessed duration.

Several tests triggered work that runs in a background task and then reached
their assertion via ``await asyncio.sleep(0)`` a few times, or a short fixed
sleep. Both are bets on how quickly the task gets scheduled. Delaying task
startup by 50ms breaks ten such tests, which means their green depended on the
machine rather than on the behaviour.

The bet is only ever lost in one direction here — the work not having started
yet — so these tests fail loudly rather than certifying a broken feature. That
still costs real time: a suite that goes red for reasons unrelated to the change
in front of you teaches you to re-run it instead of read it, and the next genuine
failure gets re-run too.

:func:`until` polls the condition the test actually cares about, so it returns as
soon as the effect lands however long scheduling took, and fails with a useful
message if it never does. The timeout is a backstop against hanging, not the
thing correctness rests on.

Prefer joining the work directly — awaiting the task, or an
:class:`asyncio.Event` the code under test sets — when the test can reach it.
This is for the cases where the task is owned by the route or service and the
test only sees its effect.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable


async def until(
    predicate: Callable[[], object],
    *,
    timeout: float = 5.0,
    message: str = "condition never became true",
    interval: float = 0.005,
) -> None:
    """Yield to the event loop until ``predicate()`` is truthy.

    Args:
        predicate: Called repeatedly; any truthy return ends the wait. Must be
            cheap and side-effect free.
        timeout: Upper bound in seconds. Generous on purpose — it exists so a
            genuine failure reports instead of hanging the suite, and a slow
            machine must not be able to turn a passing test red.
        message: Assertion text when the timeout is reached.
        interval: Sleep between checks. Non-zero so a blocking worker thread
            gets real time to run; ``sleep(0)`` only yields to the event loop and
            would spin without letting a thread progress.

    Raises:
        AssertionError: If ``predicate`` is still falsy at the timeout.
    """
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return
        if time.monotonic() >= deadline:
            raise AssertionError(f"{message} (waited {timeout}s)")
        await asyncio.sleep(interval)


__all__ = ["until"]
