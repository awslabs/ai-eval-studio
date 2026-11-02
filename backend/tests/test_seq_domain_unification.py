# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server half: one seq domain per stream.

The backend has two seq counters — the in-memory ``make_envelope`` counter (per
process, starts at 0, dies with the process) and the durable ``allocate_*_seq``
counter (starts at 1, persists). ``ws-client`` keeps ONE cursor per stream, so if a
stream's frames come from both counters the client silently discards frames:

* a live frame whose in-memory seq is <= the cursor set by a durable frame is
  dropped as a duplicate; or
* a live frame whose in-memory seq jumps far ahead makes every later durable
  replay look stale.

Either way the drop is silent and includes terminal frames — the spinner hangs with
the backend having done everything right. The client-side cursor policy
(``ws-cursor.ts``) mitigates a domain switch; this is the cure: any stream with a
durable leg uses the durable seq for BOTH legs.

These tests assert the two legs AGREE, which is the property that matters, rather
than asserting a specific number.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.platform import get_platform
from app.services.ws_envelope import (
    make_envelope,
    make_envelope_with_seq,
    reset_counters,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# The constructor itself
# ---------------------------------------------------------------------------


async def test_make_envelope_with_seq_uses_the_given_seq():
    """The explicit constructor must not touch the in-memory counter."""
    reset_counters()
    env = make_envelope_with_seq("turn:t1", "result", {"a": 1}, seq=99)
    assert env == {"stream": "turn:t1", "seq": 99, "kind": "result", "data": {"a": 1}}
    # The in-memory counter for that stream is still untouched, so a later
    # make_envelope call starts from 0 rather than continuing from 99.
    assert make_envelope("turn:t1", "text_delta", {})["seq"] == 0


async def test_two_constructors_are_separate_functions_by_design():
    """``make_envelope`` must NOT accept a seq override.

    A ``seq=None`` default on one function is what lets a durable-stream caller
    forget and silently fall back to the in-memory counter — the exact bug. Two
    names force the choice to be visible at the call site.
    """
    import inspect

    sig = inspect.signature(make_envelope)
    assert "seq" not in sig.parameters, (
        "make_envelope must not take a seq parameter: an optional seq re-enables "
        "the silent fallback to the in-memory counter on durable streams."
    )


# ---------------------------------------------------------------------------
# Job events — live envelope, replay buffer, and durable record must agree
# ---------------------------------------------------------------------------


def _job(job_id: str, *, arch_split: bool):
    from app.services.jobs import Job

    return Job(
        job_id=job_id,
        project_id="proj-1",
        type="eval",
        config={"arch_split": arch_split, "user_id": "owner-1"},
        state="running",
    )


async def test_job_event_live_buffer_and_durable_share_one_seq():
    """On the durable path all three consumers of a job event carry the same seq.

    Against the old code the live/buffered envelope carried the in-memory seq (0, 1,
    2, ...) while only the durable record carried the DDB seq (1, 2, 3, ...) — two
    domains on one stream.
    """
    reset_counters()
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    job = _job(job_id, arch_split=True)

    appended: list[tuple[int, dict]] = []
    published: list[dict] = []

    store = MagicMock()
    store.allocate_job_seq = AsyncMock(side_effect=[11, 12, 13])

    async def _append(jid, seq, payload, ttl=None):
        appended.append((seq, payload))

    store.append_job_event = AsyncMock(side_effect=_append)
    platform = MagicMock()
    platform.metadata_store = store

    hub = MagicMock()

    async def _publish(owner, env):
        published.append(env)

    hub.publish_to_user = AsyncMock(side_effect=_publish)

    project = MagicMock()
    project.owner_id = "owner-1"
    pm = MagicMock()
    pm.get_project.return_value = project

    with (
        patch("app.platform.get_platform", return_value=platform),
        patch("app.services.ws_hub.get_ws_hub", return_value=hub),
        patch("app.main.get_project_manager", return_value=pm),
    ):
        for i in range(3):
            await job.append_event({"event": "progress", "case": i})
        # Drain the backgrounded durable appends.
        for _ in range(20):
            await asyncio.sleep(0)

    live_seqs = [e["seq"] for e in published]
    buffer_seqs = [e["seq"] for e in job.events]
    durable_seqs = [seq for seq, _ in appended]

    assert live_seqs == [11, 12, 13], (
        f"the live publish must carry the DURABLE seq, got {live_seqs}"
    )
    assert buffer_seqs == live_seqs, (
        "the replay buffer must hold the same seqs as the live publish — the buffer "
        f"is what replay_to_connection serves. live={live_seqs} buffer={buffer_seqs}"
    )
    assert durable_seqs == live_seqs, (
        f"the durable log must agree with the wire. live={live_seqs} "
        f"durable={durable_seqs}"
    )


async def test_job_event_durable_payload_is_the_published_envelope():
    """The durable record and the wire frame must be the same object shape.

    Not just the seq: stream/kind/data too. Two hand-built dicts drift (one gains a
    field, the other doesn't) and replay then delivers something the live path never
    sent.
    """
    reset_counters()
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    job = _job(job_id, arch_split=True)

    appended: list[dict] = []
    published: list[dict] = []

    store = MagicMock()
    store.allocate_job_seq = AsyncMock(return_value=5)

    async def _append(jid, seq, payload, ttl=None):
        appended.append(payload)

    store.append_job_event = AsyncMock(side_effect=_append)
    platform = MagicMock()
    platform.metadata_store = store

    hub = MagicMock()

    async def _publish(owner, env):
        published.append(env)

    hub.publish_to_user = AsyncMock(side_effect=_publish)
    project = MagicMock()
    project.owner_id = "owner-1"
    pm = MagicMock()
    pm.get_project.return_value = project

    with (
        patch("app.platform.get_platform", return_value=platform),
        patch("app.services.ws_hub.get_ws_hub", return_value=hub),
        patch("app.main.get_project_manager", return_value=pm),
    ):
        await job.append_event({"event": "scoring_start", "model": "m1"})
        for _ in range(20):
            await asyncio.sleep(0)

    assert appended and published
    assert appended[0] == published[0], (
        f"durable payload != published envelope:\n  durable={appended[0]}\n"
        f"  live={published[0]}"
    )


async def test_job_event_falls_back_to_in_memory_when_allocation_fails():
    """An allocation failure must not drop the event — and must publish it
    DEGRADED.

    The stream has a durable seq domain; a frame this process cannot number
    in that domain must not mint an in-memory seq into it (the client's
    single per-stream cursor would drop it as duplicate/stale). The degraded
    marker makes the client bypass its cursor and deliver unconditionally.
    """
    reset_counters()
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    job = _job(job_id, arch_split=True)

    published: list[dict] = []
    store = MagicMock()
    store.allocate_job_seq = AsyncMock(side_effect=RuntimeError("DDB down"))
    store.append_job_event = AsyncMock()
    platform = MagicMock()
    platform.metadata_store = store

    hub = MagicMock()

    async def _publish(owner, env):
        published.append(env)

    hub.publish_to_user = AsyncMock(side_effect=_publish)
    project = MagicMock()
    project.owner_id = "owner-1"
    pm = MagicMock()
    pm.get_project.return_value = project

    with (
        patch("app.platform.get_platform", return_value=platform),
        patch("app.services.ws_hub.get_ws_hub", return_value=hub),
        patch("app.main.get_project_manager", return_value=pm),
    ):
        await job.append_event({"event": "progress", "case": 1})
        for _ in range(20):
            await asyncio.sleep(0)

    assert len(published) == 1, "the event must still be published"
    assert len(job.events) == 1, "and still buffered for replay"
    # No durable append attempted, since there is no seq to append under.
    store.append_job_event.assert_not_awaited()
    # The fallback frame carries the degraded marker so the client
    # bypasses its replay cursor instead of comparing a foreign-domain seq.
    assert published[0].get("degraded") is True, (
        "allocation-failure publishes must be marked degraded"
    )
    assert job.events[0].get("degraded") is True, (
        "the buffered replay copy must carry the marker too"
    )


async def test_old_path_job_event_uses_the_in_memory_counter():
    """arch_split=False has no durable leg, so the in-memory counter is correct."""
    reset_counters()
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    job = _job(job_id, arch_split=False)

    published: list[dict] = []
    store = MagicMock()
    store.allocate_job_seq = AsyncMock(
        side_effect=AssertionError("old path must not allocate a durable seq")
    )
    platform = MagicMock()
    platform.metadata_store = store
    hub = MagicMock()

    async def _publish(owner, env):
        published.append(env)

    hub.publish_to_user = AsyncMock(side_effect=_publish)
    project = MagicMock()
    project.owner_id = "owner-1"
    pm = MagicMock()
    pm.get_project.return_value = project

    with (
        patch("app.platform.get_platform", return_value=platform),
        patch("app.services.ws_hub.get_ws_hub", return_value=hub),
        patch("app.main.get_project_manager", return_value=pm),
    ):
        await job.append_event({"event": "progress", "case": 1})
        await job.append_event({"event": "progress", "case": 2})

    assert [e["seq"] for e in published] == [0, 1]


# ---------------------------------------------------------------------------
# The terminal `done` frame — the most consequential frame on the stream
# ---------------------------------------------------------------------------


async def test_terminal_done_frame_carries_a_durable_seq():
    """The `done` frame must be in the same domain as the progress frames.

    This is the frame whose loss wedges the eval bar at "model 0 of N" with a live
    Stop button, so it is the one that must not land in a second seq domain.
    """
    reset_counters()
    from app.services.jobs import _publish_per_job_done

    job_id = f"job-{uuid.uuid4().hex[:8]}"
    job = _job(job_id, arch_split=True)

    published: list[dict] = []
    store = MagicMock()
    store.allocate_job_seq = AsyncMock(return_value=42)
    platform = MagicMock()
    platform.metadata_store = store

    hub = MagicMock()

    async def _publish(owner, env):
        published.append(env)

    hub.publish_to_user = AsyncMock(side_effect=_publish)
    project = MagicMock()
    project.owner_id = "owner-1"
    pm = MagicMock()
    pm.get_project.return_value = project

    with (
        patch("app.platform.get_platform", return_value=platform),
        patch("app.services.ws_hub.get_ws_hub", return_value=hub),
        patch("app.main.get_project_manager", return_value=pm),
    ):
        _publish_per_job_done(job, status="completed", output_file="r.json")
        for _ in range(20):
            await asyncio.sleep(0)

    assert len(published) == 1
    assert published[0]["seq"] == 42, (
        "the terminal done frame must carry the DURABLE seq on the arch_split path; "
        f"got {published[0]['seq']} (an in-memory counter value)"
    )
    assert published[0]["kind"] == "done"


async def test_terminal_done_frame_still_published_when_allocation_fails():
    """Never drop the terminal frame — a hung spinner is worse than a bad seq."""
    reset_counters()
    from app.services.jobs import _publish_per_job_done

    job = _job(f"job-{uuid.uuid4().hex[:8]}", arch_split=True)
    published: list[dict] = []
    store = MagicMock()
    store.allocate_job_seq = AsyncMock(side_effect=RuntimeError("DDB down"))
    platform = MagicMock()
    platform.metadata_store = store
    hub = MagicMock()

    async def _publish(owner, env):
        published.append(env)

    hub.publish_to_user = AsyncMock(side_effect=_publish)
    project = MagicMock()
    project.owner_id = "owner-1"
    pm = MagicMock()
    pm.get_project.return_value = project

    with (
        patch("app.platform.get_platform", return_value=platform),
        patch("app.services.ws_hub.get_ws_hub", return_value=hub),
        patch("app.main.get_project_manager", return_value=pm),
    ):
        _publish_per_job_done(job, status="failed")
        for _ in range(20):
            await asyncio.sleep(0)

    assert len(published) == 1, "the terminal frame must survive an allocation failure"
    assert published[0]["kind"] == "done"
    # Degraded, so the client renders it without consulting the
    # cursor — an in-memory seq minted into the durable domain used to be
    # droppable as duplicate/stale, wedging the eval bar on its most
    # consequential frame.
    assert published[0].get("degraded") is True


# ---------------------------------------------------------------------------
# Ephemeral streams keep the in-memory counter
# ---------------------------------------------------------------------------


async def test_user_stream_stays_on_in_memory_counter():
    """``user:`` has no durable log AND single publisher (keepalive only).

    Forcing a durable allocation would add a store round trip per keepalive
    for no benefit. ``user:`` keepalive envelopes are discarded before the
    cursor is consulted (at the top of ``dispatch`` in ``ws-client.ts``), so
    no multi-publisher collision is possible.
    """
    reset_counters()
    k = make_envelope("user:u1", "keepalive", {})
    assert k["seq"] == 0


async def test_project_stream_rejects_in_memory_counter():
    """``project:`` must use the durable allocator.

    Two producers (the coordinator hub and a microVM worker) publish on one
    ``project:`` stream within a single coordinator lifetime, so per-process
    in-memory counters collide rather than reset: both issue seq=0, the client
    cursor sits at 0 from the first, asks for > 0 on reconnect, and the second
    is filtered out as already seen.

    ``make_envelope`` now rejects project streams so the wrong thing is hard
    to do by accident.
    """
    import pytest

    reset_counters()
    with pytest.raises(ValueError, match="[Pp]roject.*durable allocator"):
        make_envelope("project:p1", "project_event", {"resource": "optimization"})
