# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the arch_split resolver — an ENVIRONMENT check, not a per-project flag.

Cloud resolves True unconditionally; dev resolves False (no Fargate worker).
The per-project ``arch_split_enabled`` flag and the ``ARCH_SPLIT_KILL_SWITCH``
env var are DEAD and must stay dead.

Why: the schema default used to write an EXPLICIT ``arch_split_enabled: False``
into every new project at creation, so the resolver's cloud-default-True was
dead code. Every fresh project's chat-dispatched optimization loop then ran
INSIDE the microVM with its state in an EFS JSON invisible to the coordinator's
adoption sweep (which point-reads DDB). The loop died with the first microVM
recycle and showed "running — Analyzing failures" forever: unstoppable,
unadoptable, unowned.

The rule these tests pin: one logical schema with two physical bindings — no
state family may branch between homes on a per-project value.
"""

import pytest

from app.platform.arch_split import resolve_arch_split


class TestArchSplitIsAnEnvironmentCheck:
    """Cloud = durable path, dev = in-process path. Nothing else matters."""

    @pytest.mark.asyncio
    async def test_dev_resolves_false(self, monkeypatch):
        """Dev has no Fargate worker to route to — always the in-process path."""
        monkeypatch.setenv("EVAL_STUDIO_MODE", "dev")
        assert await resolve_arch_split("proj-any") is False

    @pytest.mark.asyncio
    async def test_unset_mode_resolves_false(self, monkeypatch):
        """Absent EVAL_STUDIO_MODE means dev."""
        monkeypatch.delenv("EVAL_STUDIO_MODE", raising=False)
        assert await resolve_arch_split("proj-any") is False

    @pytest.mark.asyncio
    async def test_cloud_resolves_true(self, monkeypatch):
        monkeypatch.setenv("EVAL_STUDIO_MODE", "cloud")
        assert await resolve_arch_split("proj-any") is True

    @pytest.mark.asyncio
    async def test_cloud_ignores_stored_explicit_false(self, monkeypatch):
        """THE regression pin for the explicit-False schema default.

        Every older project record carries an explicit ``False`` written by
        the old schema default. If the resolver ever reads it again, every
        one of those projects walks back onto the unowned legacy path. The
        resolver must resolve True in cloud WITHOUT consulting any store —
        asserted structurally below by the no-read spy."""
        monkeypatch.setenv("EVAL_STUDIO_MODE", "cloud")
        # No store is wired at all; a resolver that tries to read explodes.
        assert await resolve_arch_split("proj-with-stored-false") is True

    @pytest.mark.asyncio
    async def test_cloud_ignores_dead_kill_switch(self, monkeypatch):
        """ARCH_SPLIT_KILL_SWITCH is dead: forcing the fleet onto a path with
        no owner is not a rollback — it is that outage on demand.
        Emergency rollback is redeploying the previous revision."""
        monkeypatch.setenv("EVAL_STUDIO_MODE", "cloud")
        monkeypatch.setenv("ARCH_SPLIT_KILL_SWITCH", "1")
        assert await resolve_arch_split("proj-any") is True

    @pytest.mark.asyncio
    async def test_cloud_performs_no_store_read(self, monkeypatch):
        """The resolver must not touch the platform/store at all.

        Inverts the old read-once spy: the correct number of project reads
        is now ZERO. Any read reintroduces per-project routing (and the
        owner resolution bugs that came with it)."""
        monkeypatch.setenv("EVAL_STUDIO_MODE", "cloud")

        def _explode():
            raise AssertionError("resolve_arch_split must not read the platform")

        monkeypatch.setattr("app.platform.arch_split.get_platform", _explode)
        assert await resolve_arch_split("proj-any") is True

    @pytest.mark.asyncio
    async def test_cloud_true_even_for_missing_project(self, monkeypatch):
        """No project lookup happens, so a nonexistent id resolves True too.
        Authorization is enforced at API/MCP entry points, not here — this
        is a store-routing fact, not an access decision."""
        monkeypatch.setenv("EVAL_STUDIO_MODE", "cloud")
        assert await resolve_arch_split("proj-does-not-exist") is True
