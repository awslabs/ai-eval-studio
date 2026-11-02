# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the optimization-state repository seam.

The repository's whole reason to exist is that four things must be right on every
state access — home selection, version threading, owner partition, Decimal
coercion — and ~38 hand-rolled call sites each got an independent chance to be
wrong. These tests pin the four.

**Version chaining is the property most worth testing.** DynamoDB writes here are
conditional on ``expected_version``, so consecutive writes must chain the version
the PREVIOUS write returned. Guessing (``version + 1``) or reusing a stale value
loses the write, and the store raises ``ConcurrentUpdateError`` only if something
else wrote — otherwise the loss is silent. When this suite was first run against a
deliberately broken chain (returning the stale version from
``record_iteration``), 172 optimization tests still passed, so nothing covered it.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import optimization_state_repository as opt_repo
from app.services.optimization_state import OptimizationState

pytestmark = pytest.mark.asyncio


def _state_dict(version: int = 1, status: str = "running") -> dict:
    return {
        "version": version,
        "status": status,
        "iterations": [],
        "target_metric_scores": {"ExactMatch": 0.85},
        "max_iterations": 5,
        "plateau_window": 3,
        "max_iter_cap": 5,
    }


def _fresh_state(tmp_path: Path) -> OptimizationState:
    return OptimizationState(
        project_dir=tmp_path,
        target_metric_scores={"ExactMatch": 0.85},
        max_iterations=5,
        plateau_window=3,
        iterations=[],
        status="running",
    )


# ---------------------------------------------------------------------------
# Home selection
# ---------------------------------------------------------------------------


async def test_read_uses_durable_store_on_arch_split(tmp_path: Path):
    """arch_split=True must read the store, never EFS."""
    store = MagicMock()
    store.get_optimization_state = AsyncMock(return_value=_state_dict(version=7))
    platform = MagicMock()
    platform.metadata_store = store

    with (
        patch("app.platform.get_platform", return_value=platform),
        patch.object(opt_repo, "resolve_owner", AsyncMock(return_value="owner-1")),
        patch.object(
            OptimizationState, "load",
            staticmethod(lambda _d: pytest.fail("EFS must not be read on arch_split")),
        ),
    ):
        handle = await opt_repo.read("proj-1", tmp_path, arch_split=True)

    store.get_optimization_state.assert_awaited_once_with("owner-1", "proj-1")
    assert handle.version == 7
    assert handle.arch_split is True
    assert handle.owner_id == "owner-1"


async def test_read_uses_efs_on_old_path(tmp_path: Path):
    """arch_split=False must read EFS and never touch the store."""
    store = MagicMock()
    store.get_optimization_state = AsyncMock(
        side_effect=AssertionError("store must not be read on the old path")
    )
    platform = MagicMock()
    platform.metadata_store = store
    state = _fresh_state(tmp_path)

    with (
        patch("app.platform.get_platform", return_value=platform),
        patch.object(OptimizationState, "load", staticmethod(lambda _d: state)),
    ):
        handle = await opt_repo.read("proj-1", tmp_path, arch_split=False)

    assert handle.state is state
    # No version on the EFS path — there is nothing to condition a write on.
    assert handle.version is None
    assert handle.owner_id is None


async def test_read_keys_durable_read_on_owner_not_requester(tmp_path: Path):
    """The durable read must use the OWNER partition.

    Keying it on the requester is the bug class: any caller without request
    context reads an empty partition and gets "no state" for a live run.
    """
    store = MagicMock()
    store.get_optimization_state = AsyncMock(return_value=_state_dict())
    platform = MagicMock()
    platform.metadata_store = store

    with (
        patch("app.platform.get_platform", return_value=platform),
        patch(
            "app.services.agent_turn._resolve_owner_id_for_project",
            return_value="the-owner",
        ),
    ):
        await opt_repo.read("proj-1", tmp_path, arch_split=True)

    args = store.get_optimization_state.await_args.args
    assert args[0] == "the-owner", f"durable read must key on the owner, got {args[0]!r}"


async def test_missing_durable_record_yields_version_none(tmp_path: Path):
    """No record means version None, so a following write uses the first-write guard.

    Returning 0 here would assert a version that was never observed, and the
    store's condition (attribute_not_exists(version) OR version = :expected) would
    then fail on a genuine first write.
    """
    store = MagicMock()
    store.get_optimization_state = AsyncMock(return_value=None)
    platform = MagicMock()
    platform.metadata_store = store

    with (
        patch("app.platform.get_platform", return_value=platform),
        patch.object(opt_repo, "resolve_owner", AsyncMock(return_value="owner-1")),
    ):
        handle = await opt_repo.read("proj-1", tmp_path, arch_split=True)

    assert handle.state is None
    assert handle.version is None
    assert handle.exists is False


# ---------------------------------------------------------------------------
# Version threading — the property nothing covered before
# ---------------------------------------------------------------------------


async def test_write_threads_the_read_version(tmp_path: Path):
    """The version observed at read time is the conditional-write guard."""
    state = _fresh_state(tmp_path)
    saved: list[dict] = []

    def _capture(self, **kwargs):
        saved.append(kwargs)
        return (kwargs.get("expected_version") or 0) + 1

    handle = opt_repo.OptimizationStateHandle(
        state=state, version=11, arch_split=True, owner_id="owner-1",
        project_id="proj-1", project_dir=tmp_path,
    )
    with patch.object(OptimizationState, "save", _capture):
        opt_repo.write(handle)

    assert saved[0]["expected_version"] == 11
    assert saved[0]["arch_split"] is True
    assert saved[0]["user_id"] == "owner-1"


async def test_consecutive_writes_chain_the_returned_version(tmp_path: Path):
    """Write N+1 must use the version write N RETURNED, not the original read's.

    This is the exact historical failure: a branch that reused the read version for
    a second write lost it. Against a broken chain (returning the stale version)
    the second write here asserts 11 instead of 12.
    """
    state = _fresh_state(tmp_path)
    versions_seen: list[int | None] = []

    def _capture(self, **kwargs):
        v = kwargs.get("expected_version")
        versions_seen.append(v)
        return (v or 0) + 1

    handle = opt_repo.OptimizationStateHandle(
        state=state, version=11, arch_split=True, owner_id="owner-1",
        project_id="proj-1", project_dir=tmp_path,
    )
    with patch.object(OptimizationState, "save", _capture):
        handle = opt_repo.write(handle)
        handle = opt_repo.write(handle)
        handle = opt_repo.write(handle)

    assert versions_seen == [11, 12, 13], (
        "Consecutive writes must chain the store-returned version. Seeing a "
        f"repeated value means the chain is broken: {versions_seen}"
    )


async def test_record_iteration_advances_the_version(tmp_path: Path):
    """record_iteration participates in the chain (it writes)."""
    state = _fresh_state(tmp_path)
    seen: list[int | None] = []

    def _capture_record(self, **kwargs):
        seen.append(kwargs.get("expected_version"))
        return (kwargs.get("expected_version") or 0) + 1

    handle = opt_repo.OptimizationStateHandle(
        state=state, version=4, arch_split=True, owner_id="owner-1",
        project_id="proj-1", project_dir=tmp_path,
    )
    with patch.object(OptimizationState, "record_iteration", _capture_record):
        handle = opt_repo.record_iteration(
            handle, prompt_version="v1", run_id="r.json",
            metrics_summary={}, total_cases=10, passed=8,
        )

    assert seen == [4]
    assert handle.version == 5, (
        f"record_iteration must return the ADVANCED version, got {handle.version}"
    )


async def test_check_convergence_chains_after_record_iteration(tmp_path: Path):
    """The record -> convergence sequence must chain versions across both writes.

    Both write, and they run back to back in the executor. This is where two
    separate hand-threaded locals used to drift.
    """
    state = _fresh_state(tmp_path)
    record_seen: list[int | None] = []
    converge_seen: list[int | None] = []

    def _capture_record(self, **kwargs):
        record_seen.append(kwargs.get("expected_version"))
        return (kwargs.get("expected_version") or 0) + 1

    def _capture_converge(self, **kwargs):
        converge_seen.append(kwargs.get("expected_version"))
        return None, (kwargs.get("expected_version") or 0) + 1

    handle = opt_repo.OptimizationStateHandle(
        state=state, version=20, arch_split=True, owner_id="owner-1",
        project_id="proj-1", project_dir=tmp_path,
    )
    with (
        patch.object(OptimizationState, "record_iteration", _capture_record),
        patch.object(OptimizationState, "check_convergence", _capture_converge),
    ):
        handle = opt_repo.record_iteration(
            handle, prompt_version="v1", run_id="r.json",
            metrics_summary={}, total_cases=10, passed=8,
        )
        _convergence, handle = opt_repo.check_convergence(handle)

    assert record_seen == [20]
    assert converge_seen == [21], (
        "check_convergence must receive the version record_iteration produced, "
        f"not the original read's: {converge_seen}"
    )
    assert handle.version == 22


async def test_write_on_old_path_ignores_version(tmp_path: Path):
    """The EFS path must not pass store kwargs to save()."""
    state = _fresh_state(tmp_path)
    saved: list[dict] = []

    def _capture(self, **kwargs):
        saved.append(kwargs)
        return None

    handle = opt_repo.OptimizationStateHandle(
        state=state, version=None, arch_split=False, owner_id=None,
        project_id="proj-1", project_dir=tmp_path,
    )
    with patch.object(OptimizationState, "save", _capture):
        result = opt_repo.write(handle)

    assert saved == [{}], f"old path must call a bare save(), got {saved}"
    assert result.version is None


async def test_write_without_state_raises(tmp_path: Path):
    """Writing an empty handle is a programming error, not a silent no-op."""
    handle = opt_repo.OptimizationStateHandle(
        state=None, version=None, arch_split=True, owner_id="o",
        project_id="p", project_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="no state"):
        opt_repo.write(handle)


# ---------------------------------------------------------------------------
# Decimal coercion
# ---------------------------------------------------------------------------


async def test_durable_read_coerces_decimals(tmp_path: Path):
    """Decimals from the store must not reach the convergence math.

    The store returns Decimal for fractional numbers to protect cost_usd precision.
    The convergence math mixes those with float literals, and Decimal - float
    raises TypeError, so the read path normalizes.
    """
    data = _state_dict()
    data["target_metric_scores"] = {"ExactMatch": Decimal("0.85")}
    data["iterations"] = [
        {"iteration": 1, "prompt_version": "v1", "passed": 8, "total_cases": 10,
         "metrics_summary": {"ExactMatch": Decimal("0.8")}}
    ]
    store = MagicMock()
    store.get_optimization_state = AsyncMock(return_value=data)
    platform = MagicMock()
    platform.metadata_store = store

    with (
        patch("app.platform.get_platform", return_value=platform),
        patch.object(opt_repo, "resolve_owner", AsyncMock(return_value="owner-1")),
    ):
        handle = await opt_repo.read("proj-1", tmp_path, arch_split=True)

    assert isinstance(handle.state.target_metric_scores["ExactMatch"], float)
    metric = handle.state.iterations[0]["metrics_summary"]["ExactMatch"]
    assert isinstance(metric, float), f"Decimal leaked into iterations: {type(metric)}"
    # And the arithmetic that used to raise now works.
    assert handle.state.target_metric_scores["ExactMatch"] - 0.1 == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Handle predicates + display read
# ---------------------------------------------------------------------------


async def test_is_live_covers_running_and_stopping(tmp_path: Path):
    """is_live() is the one definition of "a run is in flight".

    Open-coded as `status in ("running","stopping")` at a dozen sites, one of which
    checked only "running" and missed stopping loops.
    """
    for status, expected in [
        ("running", True), ("stopping", True),
        ("cancelled", False), ("converged", False), ("failed", False),
        ("plateau", False), ("starting", False),
    ]:
        state = _fresh_state(tmp_path)
        state.status = status
        handle = opt_repo.OptimizationStateHandle(
            state=state, version=1, arch_split=True, owner_id="o",
            project_id="p", project_dir=tmp_path,
        )
        assert handle.is_live() is expected, f"{status} -> expected {expected}"

    empty = opt_repo.OptimizationStateHandle(
        state=None, version=None, arch_split=True, owner_id="o",
        project_id="p", project_dir=tmp_path,
    )
    assert empty.is_live() is False


async def test_read_for_display_degrades_to_none_not_efs(tmp_path: Path):
    """A store failure must report "unknown", never fall back to the other home.

    Falling back to EFS on the durable path returns state from a
    non-authoritative source, which reads as success and silently shows stale data.
    """
    store = MagicMock()
    store.get_optimization_state = AsyncMock(side_effect=RuntimeError("DDB down"))
    platform = MagicMock()
    platform.metadata_store = store

    with (
        patch("app.platform.get_platform", return_value=platform),
        patch.object(opt_repo, "resolve_owner", AsyncMock(return_value="owner-1")),
        patch.object(
            OptimizationState, "load",
            staticmethod(lambda _d: pytest.fail("must NOT fall back to EFS")),
        ),
    ):
        state = await opt_repo.read_for_display("proj-1", tmp_path, arch_split=True)

    assert state is None
