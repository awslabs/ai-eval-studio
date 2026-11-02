# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable outbox for microVM-originated project events.

Unroutable microVM events (published with no live turn to carry them to the
coordinator) used to reach NOBODY — the user saw silence (the "empty bubble"
class). They are now stashed durably in the
owner's partition and drained by the coordinator's reconciler: LATE (≤ one
sweep), never LOST.
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.platform.dev_local import FilesystemMetadataStore
from app.services.project_event_outbox import (
    drain_project_event_outboxes,
    outbox_sk,
    stash_unroutable_event,
)


@pytest.fixture
def store(tmp_path):
    return FilesystemMetadataStore(projects_dir=Path(tmp_path))


class TestOutboxStore:
    """Store contract, exercised on the dev impl (Protocol parity)."""

    async def test_put_list_delete_roundtrip(self, store):
        await store.put_outbox_event("u1", "p1", {"resource": "agent_turn", "type": "error"})
        await store.put_outbox_event("u1", "p1", {"resource": "eval_run", "type": "created"})
        await store.put_outbox_event("u1", "OTHER", {"resource": "x", "type": "y"})

        items = await store.list_outbox_events("u1", "p1")
        assert len(items) == 2
        # Oldest first (ns-timestamp SK ordering).
        assert items[0]["event"]["resource"] == "agent_turn"
        assert items[1]["event"]["resource"] == "eval_run"

        await store.delete_outbox_event("u1", items[0]["SK"])
        remaining = await store.list_outbox_events("u1", "p1")
        assert len(remaining) == 1
        assert remaining[0]["event"]["resource"] == "eval_run"

    async def test_delete_missing_is_noop(self, store):
        await store.delete_outbox_event("u1", "OUTBOX#p1#000")  # no raise

    def test_outbox_sk_is_timestamp_ordered(self):
        a = outbox_sk("p1")
        b = outbox_sk("p1")
        assert a.startswith("OUTBOX#p1#")
        assert a < b


class TestStash:
    async def test_stash_writes_through_platform_store(self, monkeypatch):
        mock_store = Mock()
        mock_store.put_outbox_event = AsyncMock()
        platform = Mock()
        platform.metadata_store = mock_store
        monkeypatch.setattr("app.platform.get_platform", lambda: platform)

        await stash_unroutable_event("u1", "p1", {"resource": "r", "type": "t"})

        mock_store.put_outbox_event.assert_awaited_once_with(
            "u1", "p1", {"resource": "r", "type": "t"}
        )

    async def test_stash_failure_is_swallowed(self, monkeypatch):
        """The stash must never make the publish path less reliable than the
        pre-outbox lost-event behavior."""
        mock_store = Mock()
        mock_store.put_outbox_event = AsyncMock(side_effect=RuntimeError("ddb down"))
        platform = Mock()
        platform.metadata_store = mock_store
        monkeypatch.setattr("app.platform.get_platform", lambda: platform)

        await stash_unroutable_event("u1", "p1", {"resource": "r", "type": "t"})  # no raise


class TestDrain:
    def _wire(self, monkeypatch, store, project_dirs, hub=None):
        platform = Mock()
        platform.metadata_store = store
        monkeypatch.setattr(
            "app.platform.get_platform", lambda: platform
        )
        monkeypatch.setattr(
            "app.main._iter_project_dirs", lambda: iter(project_dirs)
        )
        monkeypatch.setattr(
            "app.platform.process_role.is_agentcore_runtime", lambda: False
        )
        hub = hub or Mock()
        if not isinstance(hub.publish_project, AsyncMock):
            hub.publish_project = AsyncMock()
        monkeypatch.setattr(
            "app.services.ws_hub.get_ws_hub", lambda: hub
        )
        return hub

    async def test_drain_publishes_and_deletes(self, store, tmp_path, monkeypatch):
        project_dir = tmp_path / "users" / "u1" / "projects" / "p1"
        project_dir.mkdir(parents=True)
        await store.put_outbox_event("u1", "p1", {"resource": "agent_turn", "type": "error"})

        hub = self._wire(monkeypatch, store, [project_dir])
        drained = await drain_project_event_outboxes()

        assert drained == 1
        hub.publish_project.assert_awaited_once_with(
            "p1", "u1", {"resource": "agent_turn", "type": "error"}
        )
        assert await store.list_outbox_events("u1", "p1") == []

    async def test_drain_empty_outbox_is_quiet(self, store, tmp_path, monkeypatch):
        project_dir = tmp_path / "users" / "u1" / "projects" / "p-empty"
        project_dir.mkdir(parents=True)

        hub = self._wire(monkeypatch, store, [project_dir])
        drained = await drain_project_event_outboxes()

        assert drained == 0
        hub.publish_project.assert_not_awaited()

    async def test_publish_failure_retries_next_pass(self, store, tmp_path, monkeypatch):
        """Publish-then-delete: a failed publish leaves the item for the next
        sweep (at-least-once)."""
        project_dir = tmp_path / "users" / "u1" / "projects" / "p1"
        project_dir.mkdir(parents=True)
        await store.put_outbox_event("u1", "p1", {"resource": "r", "type": "t"})

        hub = Mock()
        hub.publish_project = AsyncMock(side_effect=RuntimeError("hub down"))
        self._wire(monkeypatch, store, [project_dir], hub=hub)

        drained = await drain_project_event_outboxes()

        assert drained == 0
        assert len(await store.list_outbox_events("u1", "p1")) == 1

    async def test_microvm_process_never_drains(self, store, monkeypatch):
        monkeypatch.setattr(
            "app.platform.process_role.is_agentcore_runtime", lambda: True
        )
        assert await drain_project_event_outboxes() == 0


class TestSortKeyUniqueness:
    """The key must be unique AND ordered — it used to be neither, reliably.

    ``time.time_ns()`` reports nanoseconds but resolves to microseconds here, so
    two events stashed in one tick minted the SAME key (measured 16198/20000 =
    81% for back-to-back calls) and the DynamoDB write silently overwrote the
    earlier event. The existing ordering test only caught it warm inside a large
    suite — it passes in isolation because a cold process spaces the two calls
    out — so it read as a flake and hid a real defect.
    """

    def test_back_to_back_keys_are_distinct(self):
        """The assertion that used to fail ~81% of the time when warm."""
        a = outbox_sk("p1")
        b = outbox_sk("p1")
        assert a != b, "two events stashed in one clock tick collide"
        assert a < b, "and the later one must still sort later"

    def test_a_burst_under_a_FROZEN_clock_is_distinct_and_ordered(self):
        """Bursts are the common path, and this pins the mechanism, not the timing.

        A job-completed plus a report dispatch, or several progress events from
        one handler, all land inside the same microsecond. Measuring that with a
        real clock is not a test: a loop slow enough to cross a microsecond per
        iteration passes even with the bug (verified — the un-frozen version of
        this test stayed green when the fix was reverted). Freezing the clock
        removes the only thing that made it look fine.
        """
        with patch("app.services.project_event_outbox.time") as fake_time:
            fake_time.time_ns.return_value = 1_788_566_113_870_457_000
            keys = [outbox_sk("p1") for _ in range(20_000)]
        assert len(set(keys)) == len(keys), (
            "a clock that does not advance must still yield unique keys"
        )
        assert keys == sorted(keys), "generation order must be sort order"

    def test_keys_stay_ordered_across_projects_in_one_partition(self):
        """Different projects share the owner's partition, and the counter is
        global — so a p2 event minted between two p1 events must not make the p1
        pair collide or invert. Frozen clock, for the same reason as above."""
        with patch("app.services.project_event_outbox.time") as fake_time:
            fake_time.time_ns.return_value = 1_788_566_113_870_457_000
            keys = (outbox_sk("p1"), outbox_sk("p2"), outbox_sk("p1"))
        p1 = [k for k in keys if k.startswith("OUTBOX#p1#")]
        assert len(p1) == 2
        assert p1[0] != p1[1]
        assert p1 == sorted(p1)

    def test_the_key_shape_is_unchanged(self):
        """Ordering against keys ALREADY in the table depends on the shape.

        The fix bumps the timestamp rather than appending a counter field
        precisely so this stays true — a wider or differently-delimited key
        would sort unpredictably against existing rows during a deploy.
        """
        sk = outbox_sk("p1")
        prefix, _, ns = sk.rpartition("#")
        assert prefix == "OUTBOX#p1"
        assert len(ns) == 20 and ns.isdigit()
