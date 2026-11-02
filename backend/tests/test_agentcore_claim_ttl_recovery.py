# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""An expired AgentCore session claim must not wedge its (user, project) pair.

The claim's TTL was treated as the backstop bounding a leaked claim ("2h TTL
backstop"). It is not one. Three independent gaps made an expired claim block
indefinitely:

1. The claim's conditional write accepted only ``status = idle OR
   attribute_not_exists(status)``, so an item still marked ``running`` blocked
   every subsequent claim REGARDLESS of whether its ttl had passed.
2. DynamoDB TTL deletion is a best-effort background process — AWS documents
   expired items as "typically" deleted within a few DAYS, not at expiry. So the
   item that would clear the block can linger for days.
3. The AgentCore claim reconciler ran at STARTUP only, so a long-lived
   coordinator never revisited the leak.

Fixes: the claim condition also accepts an expired ttl, and a periodic sweep
clears expired claims on the eval_reconciler cadence. These tests pin the
behavioral effect (a fresh claim WINS), not the presence of the code.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio


@pytest.fixture
def dev_store(projects_dir: Path, monkeypatch):
    """Dev platform backed by the REAL FilesystemMetadataStore.

    Claims go through the actual first-wins path so "held" vs "free" is a genuine
    store answer rather than a mocked return.
    """
    from app.platform import Platform, set_platform
    from app.platform.dev_local import (
        FilesystemMetadataStore,
        LocalAuth,
        NoOpTokenVending,
        _build_dev_session_store,
    )

    set_platform(None)
    metadata_store = FilesystemMetadataStore(projects_dir=projects_dir)
    auth = LocalAuth(metadata_store=metadata_store)

    async def _noop_require_access(self, user_id, project_id, action):
        return None

    monkeypatch.setattr(LocalAuth, "require_access", _noop_require_access)
    set_platform(
        Platform(
            metadata_store=metadata_store,
            session_store=_build_dev_session_store(),
            auth=auth,
            token_vending=NoOpTokenVending(),
        )
    )
    yield metadata_store
    set_platform(None)


async def test_expired_claim_is_claimable(dev_store):
    """An expired claim must be re-claimable without waiting for TTL deletion.

    Against the buggy condition the second claim returns False because the item
    still says ``running`` — the pair stays wedged until DynamoDB physically
    deletes the item, which lags expiry by up to days.
    """
    # Claim with a ttl that has ALREADY passed (simulating a leaked claim whose
    # expiry has come and gone while the item itself has not been deleted).
    won = await dev_store.claim_agentcore_session_turn(
        "userA", "projA", ttl=int(time.time()) - 60
    )
    assert won is True, "First claim must win"

    # A fresh claim must now succeed — the previous one is expired.
    reclaim = await dev_store.claim_agentcore_session_turn(
        "userA", "projA", ttl=int(time.time()) + 7200
    )
    assert reclaim is True, (
        "An EXPIRED claim must be claimable. Otherwise a leaked claim wedges the "
        "(user, project) pair until DynamoDB's best-effort TTL deletion runs — "
        "AWS documents that as 'typically within a few days', not at expiry."
    )


async def test_unexpired_claim_still_blocks(dev_store):
    """Counterpart: a LIVE claim must still block, or serialization is gone.

    Keeps the expiry arm from degrading the claim into a no-op — the claim is the
    primary one-turn-at-a-time serializer.
    """
    won = await dev_store.claim_agentcore_session_turn(
        "userB", "projB", ttl=int(time.time()) + 7200
    )
    assert won is True

    second = await dev_store.claim_agentcore_session_turn(
        "userB", "projB", ttl=int(time.time()) + 7200
    )
    assert second is False, (
        "A live (unexpired) claim MUST still block a concurrent claim — this is "
        "the one-turn-at-a-time serializer."
    )


async def test_periodic_sweep_clears_expired_claim_only(dev_store):
    """The periodic sweep clears expired claims and leaves live ones alone.

    The startup reconciler clears every running claim (justified by
    restart-implies-orphaned). That invariant does NOT hold mid-run, so the
    periodic sweep must only touch claims whose ttl has passed.
    """
    from app.services.agentcore_session_reconciler import (
        reconcile_expired_agentcore_claims,
    )

    # One expired claim, one live claim.
    assert await dev_store.claim_agentcore_session_turn(
        "userExpired", "projX", ttl=int(time.time()) - 120
    )
    assert await dev_store.claim_agentcore_session_turn(
        "userLive", "projY", ttl=int(time.time()) + 7200
    )

    result = await reconcile_expired_agentcore_claims()

    assert result["claims_cleared"] == 1, (
        f"Exactly the expired claim should be cleared, got {result}"
    )

    # The live claim must still be held — a fresh claim on it fails.
    assert await dev_store.claim_agentcore_session_turn(
        "userLive", "projY", ttl=int(time.time()) + 7200
    ) is False, "The periodic sweep must not clear a LIVE claim (active turn)."


async def test_periodic_sweep_is_wired_into_reconcile_pass(dev_store, monkeypatch):
    """reconcile_once must actually invoke the expired-claim sweep.

    A sweep that exists but is never called is inert — this is the wiring
    assertion, not an existence check.
    """
    from app.services import eval_reconciler

    calls: list[str] = []

    async def _spy_sweep():
        calls.append("swept")
        return {"claims_cleared": 0}

    monkeypatch.setattr(
        "app.services.agentcore_session_reconciler."
        "reconcile_expired_agentcore_claims",
        _spy_sweep,
    )

    await eval_reconciler.reconcile_once()

    assert calls == ["swept"], (
        "reconcile_once must call the expired-claim sweep — otherwise expired "
        "claims are only recovered at startup, which a long-lived coordinator "
        "never reaches."
    )


async def test_sweep_failure_does_not_break_reconcile_pass(dev_store, monkeypatch):
    """A failing sweep must not kill the reconcile pass or its eval-side results.

    The sweep is additive; the eval-side reconciliation it shares a pass with is
    load-bearing and must survive an unrelated failure.
    """
    from app.services import eval_reconciler

    async def _boom():
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(
        "app.services.agentcore_session_reconciler."
        "reconcile_expired_agentcore_claims",
        _boom,
    )

    result = await eval_reconciler.reconcile_once()

    assert result is not None, "reconcile_once must still return its result"
    assert "orphans_stopped" in result, (
        "The eval-side reconcile result must be intact despite the sweep failure"
    )
