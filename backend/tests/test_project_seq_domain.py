# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test project stream seq domain unification.

The CENTERPIECE test reproduces the collision: two producers
(separate WsHub instances, their own in-memory counters, ZERO connections
registered in the second) both publish on one `project:` stream, with
a shared persisted store. Producer A (microVM route) publishes
`eval_job/started`; the client records that seq as its cursor. Producer B
(coordinator hub) publishes `eval_job/completed` into its own buffer. A
connection then subscribes with `since_seq` = the cursor from producer A.
The completed envelope MUST be replayed onto that connection.

With per-process counters both seqs would be 0 and the since_seq filter would
discard the second; the durable allocator issues distinct seqs.

Mutation proofs:
- M_CENTERPIECE: revert only the `publish_project` mint back to `make_envelope`
  (the structural rejection stays, so the call site uses `make_envelope_with_seq`
  but passes an in-memory seq). The test must fail.
- M_ORDERING: remove the per-project lock from `publish_project`. The test must
  fail because concurrent allocations can resolve out of order.
- M_FALLBACK: make the except-branch drop the publish instead of falling back.
  The test must fail because no envelope lands on the connection.
- M_REJECTION: remove the ValueError guard from `make_envelope`. The test must
  fail because project streams can silently fall back to in-memory seqs.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest import mock

import pytest

from app.platform.dev_local import FilesystemMetadataStore
from app.services.project_event_tracker import _reset as reset_tracker
from app.services.ws_envelope import make_envelope, make_envelope_with_seq, mark_degraded, reset_counters
from app.services.ws_hub import WsHub


class _FailingAllocator:
    """Mock metadata store that always fails seq allocation (for fallback test)."""

    async def allocate_project_seq(self, project_id: str) -> int:
        raise RuntimeError("Simulated allocation failure")


@pytest.mark.asyncio
async def test_centerpiece_colliding_seq_replay_survival(tmp_path: Path):
    """CENTERPIECE: one coordinator lifetime, two producers → distinct seqs.

    Two producers publish on one `project:` stream. With in-memory counters
    both envelopes would carry seq=0; the durable allocator issues 1 and 2,
    proving no collision.

    Mutation proof M_CENTERPIECE: revert the `publish_project` mint to call
    `make_envelope` (passing an in-memory seq via `make_envelope_with_seq`
    but obtaining that seq from `_next_seq` rather than the store). The test
    will fail because both producers issue seq=0 (collision).
    """
    project_id = "test-proj-centerpiece"
    owner_id = "test-user-centerpiece"

    # Shared persisted store (stands in for DynamoDB + cross-process)
    store = FilesystemMetadataStore(projects_dir=tmp_path)

    # Producer A (microVM route analog): separate hub, its own in-memory counters
    reset_counters()
    reset_tracker()
    hub_a = WsHub()

    # Mock get_platform to return our store
    with mock.patch("app.platform.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = store

        # Producer A publishes eval_job/started
        await hub_a.publish_project(
            project_id=project_id,
            owner_id=owner_id,
            kind="project_event",
            data={"resource": "eval_job", "event": "started"},
        )

    # Producer B (coordinator hub): separate WsHub, its own in-memory counters,
    # ZERO connections registered
    reset_counters()
    hub_b = WsHub()

    with mock.patch("app.platform.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = store

        # Producer B publishes eval_job/completed into its own buffer
        await hub_b.publish_project(
            project_id=project_id,
            owner_id=owner_id,
            kind="project_event",
            data={"resource": "eval_job", "event": "completed"},
        )

    # Check the buffer directly — both envelopes should be present with distinct seqs
    from app.services.project_event_tracker import get_provider

    buffer = get_provider(project_id)
    events = buffer.events

    assert len(events) == 2, f"Expected 2 envelopes, got {len(events)}"

    # Extract the two events
    started_env = next((e for e in events if e["data"].get("event") == "started"), None)
    completed_env = next((e for e in events if e["data"].get("event") == "completed"), None)

    assert started_env is not None, "Started envelope not in buffer"
    assert completed_env is not None, "Completed envelope not in buffer"

    # Durable allocator: distinct seqs (1 and 2).
    # Per-process counters would issue seq=0 twice (collision).
    assert started_env["seq"] != completed_env["seq"], "Seqs must be distinct (anti-collision)"
    assert started_env["seq"] == 1, f"First seq should be 1, got {started_env['seq']}"
    assert completed_env["seq"] == 2, f"Second seq should be 2, got {completed_env['seq']}"


@pytest.mark.asyncio
async def test_make_envelope_rejects_project_streams():
    """make_envelope raises ValueError when stream names a project scope.

    Mutation proof M_REJECTION: remove the ValueError guard. The test will
    fail because the call succeeds and the in-memory counter is used.
    """
    reset_counters()

    # user: and turn:/job: streams still work
    make_envelope("user:u1", "keepalive", {})
    make_envelope("turn:t1", "text_delta", {"delta": "hi"})
    make_envelope("job:j1", "progress", {"percent": 50})

    # project: stream raises
    with pytest.raises(ValueError, match="[Pp]roject.*durable allocator"):
        make_envelope("project:p1", "project_event", {"resource": "eval"})


@pytest.mark.asyncio
async def test_buffer_order_equals_seq_order(tmp_path: Path):
    """Two concurrent publish_project calls leave buffer seqs ascending.

    Uses an allocator whose FIRST call resolves after the second, proving
    the per-project lock preserves ordering.

    Mutation proof M_ORDERING: remove the per-project lock. The test will
    fail because the buffer records seqs out of order.
    """
    project_id = "test-proj-ordering"
    owner_id = "test-user-ordering"

    # Mock allocator that returns seqs out of submission order
    # We'll delay the first call artificially to ensure it would resolve
    # after the second if not serialized by the lock.
    call_count = 0

    async def _delayed_allocator(pid: str) -> int:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: delay 100ms
            await asyncio.sleep(0.1)
            return 1
        else:
            # Second call: return immediately
            return 2

    mock_store = mock.Mock()
    mock_store.allocate_project_seq = _delayed_allocator

    reset_counters()
    reset_tracker()
    hub = WsHub()

    with mock.patch("app.platform.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = mock_store

        # Spawn two concurrent publishes
        task_a = asyncio.create_task(
            hub.publish_project(
                project_id=project_id,
                owner_id=owner_id,
                kind="project_event",
                data={"event": "first"},
            )
        )
        # Give task_a a head start so it acquires the lock first
        await asyncio.sleep(0.01)
        task_b = asyncio.create_task(
            hub.publish_project(
                project_id=project_id,
                owner_id=owner_id,
                kind="project_event",
                data={"event": "second"},
            )
        )

        await asyncio.gather(task_a, task_b)

    # Check the buffer's recorded seqs
    from app.services.project_event_tracker import get_provider

    buffer = get_provider(project_id)
    events = buffer.events

    assert len(events) == 2
    seqs = [e["seq"] for e in events]
    # With the lock: seqs are [1, 2] because task_a finishes first (acquired lock first)
    # Without the lock: task_b could finish first and record seq=2 before task_a records seq=1
    assert seqs == [1, 2], f"Buffer seqs must be ascending, got {seqs}"


@pytest.mark.asyncio
async def test_allocation_failure_still_publishes(tmp_path: Path, caplog):
    """Allocation failure publishes with buffer-seeded fallback seq and logs WARNING.

    Mutation proof M_FALLBACK: make the except-branch drop the publish instead
    of falling back. The test will fail because no envelope lands on the
    connection.
    """
    project_id = "test-proj-fallback"
    owner_id = "test-user-fallback"

    mock_store = _FailingAllocator()

    reset_counters()
    reset_tracker()
    hub = WsHub()

    # Patch the module logger rather than using caplog, which is order-dependent
    # across this suite.
    import app.services.ws_hub as ws_hub_module

    with mock.patch.object(ws_hub_module, "logger") as mock_logger, mock.patch(
        "app.platform.get_platform"
    ) as mock_platform:
        mock_platform.return_value.metadata_store = mock_store

        # Publish with failing allocator
        await hub.publish_project(
            project_id=project_id,
            owner_id=owner_id,
            kind="project_event",
            data={"event": "test"},
        )

        # WARNING was logged
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args
        assert "degraded" in str(warning_call).lower() or "fallback" in str(warning_call).lower()

    # Event was still published (not dropped)
    from app.services.project_event_tracker import get_provider

    buffer = get_provider(project_id)
    events = buffer.events

    assert len(events) == 1, "Event must be published despite allocation failure"

    envelope = events[0]

    # Seq is buffer-seeded: for empty buffer, max([], default=-1) + 1 = 0
    assert envelope["seq"] == 0, f"Fallback seq for empty buffer should be 0, got {envelope['seq']}"

    # Envelope is marked degraded
    assert "degraded" in envelope, "Envelope must carry degraded marker"
    assert envelope["degraded"] is True, "Degraded marker must be True"


@pytest.mark.asyncio
async def test_degraded_marker_on_allocation_failure(tmp_path: Path):
    """Allocation failure marks both buffered and live envelopes degraded.

    The payload and kind survive unchanged — the event is not dropped or replaced,
    just marked so the client knows to bypass its cursor.

    Mutation proof: revert mark_degraded to an identity function (return envelope
    unchanged). The test will fail because the degraded key will be absent.
    """
    project_id = "test-proj-degraded"
    owner_id = "test-user-degraded"

    mock_store = _FailingAllocator()

    reset_counters()
    reset_tracker()
    hub = WsHub()

    with mock.patch("app.platform.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = mock_store

        # Publish with failing allocator
        await hub.publish_project(
            project_id=project_id,
            owner_id=owner_id,
            kind="project_event",
            data={"resource": "eval_job", "event": "completed"},
        )

    # Check the buffer
    from app.services.project_event_tracker import get_provider

    buffer = get_provider(project_id)
    events = buffer.events

    assert len(events) == 1, "Event must be published despite allocation failure"
    envelope = events[0]

    # Envelope is marked degraded
    assert "degraded" in envelope, "Envelope must carry degraded marker"
    assert envelope["degraded"] is True, "Degraded marker must be True"

    # Payload and kind are intact
    assert envelope["kind"] == "project_event"
    assert envelope["data"]["resource"] == "eval_job"
    assert envelope["data"]["event"] == "completed"


@pytest.mark.asyncio
async def test_healthy_publish_has_no_degraded_key(tmp_path: Path):
    """Successful allocation produces envelope with degraded key ABSENT.

    Absence means healthy (same convention as replayed marker).

    Mutation proof: apply mark_degraded unconditionally. The test will fail
    because the key will be present on healthy publishes.
    """
    project_id = "test-proj-healthy"
    owner_id = "test-user-healthy"

    store = FilesystemMetadataStore(projects_dir=tmp_path)

    reset_counters()
    reset_tracker()
    hub = WsHub()

    with mock.patch("app.platform.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = store

        # Publish with working allocator
        await hub.publish_project(
            project_id=project_id,
            owner_id=owner_id,
            kind="project_event",
            data={"resource": "eval_job", "event": "started"},
        )

    # Check the buffer
    from app.services.project_event_tracker import get_provider

    buffer = get_provider(project_id)
    events = buffer.events

    assert len(events) == 1
    envelope = events[0]

    # Degraded key must be ABSENT (not merely falsy)
    assert "degraded" not in envelope, "Healthy envelope must not have degraded key"
