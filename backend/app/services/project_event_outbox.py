# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable outbox for microVM-originated project events.

Closes the microVM transport gap. The AgentCore microVM holds no browser
WebSocket connections; its project events reach the user only when a live
turn's SSE stream can carry them to the coordinator
(``project_event_frame.forward_project_event_to_coordinator``). When no live
turn exists — an error published after a turn died, a background flush racing
turn teardown — the event would otherwise reach NOBODY, and the user would see
silence.

Same pattern as the optimization-adoption seam: the durable record IS the
queue. Unroutable events are written to a DynamoDB outbox under the OWNER's
partition (``PK=USER#{owner}``, ``SK=OUTBOX#{project}#{ns-timestamp}`` — fits
the microVM's LeadingKeys-scoped write policy, no IAM change), and the
coordinator's reconciler drains them on its cadence: publish through the real
ws_hub (durable seq mint) then delete. Delivery is at-least-once and up to a
sweep late (~60s) — correct for the toast/refetch class these events serve;
an event older than the item TTL (1 hour) is stale by definition and expires
unread.

Ordering note: drained events publish AFTER any live events the coordinator
minted in the meantime — seqs are allocation-ordered, not occurrence-ordered.
Acceptable: consumers treat project events as refetch triggers, not as a
transaction log.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Bound each drain pass per project so one flooded outbox cannot starve the
# rest of the reconciler pass. Anything beyond the bound is picked up next pass.
DRAIN_BATCH_LIMIT = 50


async def stash_unroutable_event(
    owner_id: str, project_id: str, event: dict[str, Any]
) -> None:
    """Write an unroutable microVM event to the durable outbox.

    Best-effort by contract: a failed stash logs at WARNING and the event is
    lost exactly as it was before the outbox existed — it must never make the publish
    path less reliable than it already was.
    """
    from app.platform import get_platform

    try:
        await get_platform().metadata_store.put_outbox_event(
            owner_id, project_id, event
        )
        logger.info(
            "[event-outbox] Stashed unroutable event (project=%s resource=%s "
            "type=%s) — coordinator drains it within a reconciler pass.",
            project_id[:12], event.get("resource"), event.get("type"),
        )
    except Exception:  # noqa: BLE001 — the stash must never break the publish
        logger.warning(
            "[event-outbox] Failed to stash unroutable event (project=%s "
            "resource=%s type=%s) — event is lost.",
            project_id[:12], event.get("resource"), event.get("type"),
            exc_info=True,
        )


async def drain_project_event_outboxes() -> int:
    """Coordinator-side sweep: publish and delete every stashed event.

    Enumerates project working trees (the same scan-free index the adoption
    sweep uses — the OPT/OUTBOX items carry no GSI keys and the coordinator
    role deliberately has no table Scan), point-queries each project's
    outbox, publishes through the real ws_hub (durable seq), and deletes on
    success. Publish-then-delete: a crash in between re-delivers next pass —
    at-least-once, and these events are refetch triggers so duplicates are
    harmless.

    Returns:
        Number of events drained this pass.
    """
    from app.main import _iter_project_dirs  # lazy: app.main imports many services
    from app.platform import get_platform
    from app.platform.auth import get_local_dev_user_id
    from app.platform.process_role import is_agentcore_runtime
    from app.services.ws_hub import get_ws_hub

    if is_agentcore_runtime():
        return 0

    store = get_platform().metadata_store
    hub = get_ws_hub()
    drained = 0

    for project_dir in _iter_project_dirs():
        project_id = project_dir.name
        parts = project_dir.parts
        try:
            owner_id = parts[parts.index("users") + 1]
        except ValueError:
            owner_id = get_local_dev_user_id()

        try:
            items = await store.list_outbox_events(
                owner_id, project_id, limit=DRAIN_BATCH_LIMIT
            )
        except Exception:  # noqa: BLE001 — one unreadable outbox must not kill the sweep
            logger.exception(
                "[event-outbox] Could not list outbox for project=%s",
                project_id[:12],
            )
            continue

        for item in items:
            event = item.get("event")
            sk = item.get("SK")
            if not isinstance(event, dict) or not sk:
                # Malformed item: delete rather than re-log it every pass.
                logger.warning(
                    "[event-outbox] Dropping malformed outbox item project=%s sk=%r",
                    project_id[:12], sk,
                )
                if sk:
                    try:
                        await store.delete_outbox_event(owner_id, sk)
                    except Exception:  # noqa: BLE001
                        # The warning above already names the item. A failed
                        # delete only means the next pass sees it again, so
                        # raising here would abort the drain over an item that
                        # was already unusable.
                        logger.debug(
                            "[event-outbox] Delete of malformed item failed "
                            "project=%s sk=%r",
                            project_id[:12], sk, exc_info=True,
                        )
                continue
            try:
                await hub.publish_project(project_id, owner_id, event)
                await store.delete_outbox_event(owner_id, sk)
                drained += 1
            except Exception:  # noqa: BLE001 — retry next pass
                logger.exception(
                    "[event-outbox] Publish/delete failed for project=%s sk=%s "
                    "— retrying next pass", project_id[:12], sk,
                )
                break  # preserve per-project ordering; retry from here next pass

    if drained:
        logger.info("[event-outbox] Drained %d stashed event(s)", drained)
    return drained


#: Guards ``_last_sk_ns`` so two threads cannot read the same value and mint the
#: same key. The publish path runs from asyncio tasks and from ``to_thread``
#: workers, so a plain read-modify-write is not enough.
_sk_lock = threading.Lock()
_last_sk_ns = 0


def outbox_sk(project_id: str) -> str:
    """Strictly-increasing, timestamp-ordered sort key for a new outbox item.

    ``time.time_ns()`` reports NANOSECONDS but resolves to MICROSECONDS on this
    platform, so two events stashed in the same tick used to produce the SAME
    key — measured at 16198/20000 (81%) for back-to-back calls. The DynamoDB
    write is keyed on it, so the second event silently overwrote the first: the
    earlier event gone, no error, no log, on the common path (bursty publishes
    are the norm — a job-completed plus a report dispatch, or several progress
    events from one handler).

    This keeps the key SHAPE — a bare 20-digit ns count — so ordering against
    keys already in the table is unchanged and there is no mixed-format window.
    A clock reading that does not advance past the last one issued is bumped by
    one ns instead. The key can therefore run a few ns ahead of the real clock,
    which is harmless: nothing parses a timestamp back out of it, and a few ns
    of skew inside one microsecond is not observable.

    This makes keys unique WITHIN a process. Across processes — two coordinator
    tasks overlap by design during a rolling deploy — uniqueness is enforced at
    the write, by ``put_outbox_event``'s conditional put.
    """
    global _last_sk_ns
    with _sk_lock:
        ns = max(time.time_ns(), _last_sk_ns + 1)
        _last_sk_ns = ns
    return f"OUTBOX#{project_id}#{ns:020d}"
