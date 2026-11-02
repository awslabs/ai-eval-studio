# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MantleDescriptor + descriptor I/O.

Covers the TypedDict shape, the read/write roundtrip, atomic-write
behavior, and the lock-serialized async wrapper. The lock test asserts
that two concurrent writes for the SAME project are serialized
(no overlapping critical sections) and that two concurrent writes for
DIFFERENT projects are NOT blocked by the lock.
"""

import asyncio
import json
import time
from pathlib import Path

import pytest

from app.services.model_descriptors import (
    DESCRIPTOR_FILE_RELATIVE,
    MantleDescriptor,
    read_descriptors,
    write_descriptors,
)
from app.services.model_descriptors_async import persist_descriptors_locked
from app.services import project_locks


@pytest.fixture(autouse=True)
def _reset_locks():
    """Clear the lock registry between tests so per-test state is isolated."""
    project_locks._reset()
    yield
    project_locks._reset()


# --------------------------------------------------------------------------- #
# TypedDict shape                                                             #
# --------------------------------------------------------------------------- #


def test_descriptor_typeddict_required_fields():
    """The three required fields round-trip through JSON cleanly."""
    d: MantleDescriptor = {
        "endpoint": "mantle",
        "api": "responses",
        "region": "us-east-1",
    }
    serialized = json.dumps(d)
    parsed = json.loads(serialized)
    assert parsed == {
        "endpoint": "mantle",
        "api": "responses",
        "region": "us-east-1",
    }


def test_descriptor_typeddict_optional_fields():
    """All five fields (3 required + 2 optional) round-trip through JSON cleanly."""
    d: MantleDescriptor = {
        "endpoint": "mantle",
        "api": "responses",
        "region": "us-west-2",
        "supports_temperature": False,
        "supports_system_prompt": True,
    }
    parsed = json.loads(json.dumps(d))
    assert parsed["supports_temperature"] is False
    assert parsed["supports_system_prompt"] is True


def test_descriptor_file_relative_path_is_src_subdir():
    """DESCRIPTOR_FILE_RELATIVE is exactly ``src/model_descriptors.json``."""
    assert DESCRIPTOR_FILE_RELATIVE == Path("src") / "model_descriptors.json"


# --------------------------------------------------------------------------- #
# read_descriptors / write_descriptors                                        #
# --------------------------------------------------------------------------- #


def test_read_descriptors_returns_empty_when_file_absent(tmp_path):
    """No descriptor file → empty dict (project predates mantle support)."""
    assert read_descriptors(tmp_path) == {}


def test_write_then_read_roundtrip(tmp_path):
    """Write a descriptor map; read it back; equal."""
    descriptors: dict[str, MantleDescriptor] = {
        "openai.gpt-5.5": {
            "endpoint": "mantle",
            "api": "responses",
            "region": "us-east-1",
        }
    }
    out = write_descriptors(tmp_path, descriptors)
    assert out == tmp_path / "src" / "model_descriptors.json"
    assert read_descriptors(tmp_path) == descriptors


def test_write_creates_src_dir_if_absent(tmp_path):
    """``src/`` is created on first write — same precedent as bedrock_pricing.write_pricing_file."""
    assert not (tmp_path / "src").exists()
    write_descriptors(tmp_path, {"k": {"endpoint": "mantle", "api": "responses", "region": "us-east-1"}})
    assert (tmp_path / "src").exists()


def test_write_is_atomic_via_tmp_rename(tmp_path):
    """No ``.tmp`` file remains after a successful write."""
    write_descriptors(
        tmp_path,
        {"k": {"endpoint": "mantle", "api": "responses", "region": "us-east-1"}},
    )
    leftovers = list((tmp_path / "src").glob("*.tmp"))
    assert leftovers == []


def test_write_overwrites_prior_descriptors(tmp_path):
    """The second write fully replaces the first — no merge (caller's job)."""
    write_descriptors(
        tmp_path,
        {"a": {"endpoint": "mantle", "api": "responses", "region": "us-east-1"}},
    )
    write_descriptors(
        tmp_path,
        {"b": {"endpoint": "mantle", "api": "responses", "region": "us-west-2"}},
    )
    result = read_descriptors(tmp_path)
    assert "a" not in result
    assert "b" in result


def test_read_descriptors_returns_dict_for_well_formed_json(tmp_path):
    """Hand-crafted descriptor file is read back as a dict with the same keys."""
    src = tmp_path / "src"
    src.mkdir()
    payload = {
        "openai.gpt-5.5": {
            "endpoint": "mantle",
            "api": "responses",
            "region": "us-east-1",
            "supports_temperature": True,
        }
    }
    (src / "model_descriptors.json").write_text(json.dumps(payload), encoding="utf-8")
    assert read_descriptors(tmp_path) == payload


# --------------------------------------------------------------------------- #
# persist_descriptors_locked                                                  #
# --------------------------------------------------------------------------- #


async def test_persist_descriptors_locked_uses_project_lock(monkeypatch, tmp_path):
    """Concurrent writes for the SAME project_id are serialized.

    Asserted by recording entry/exit timestamps on every write call and
    verifying that NO two intervals overlap. If the lock is bypassed,
    threads from ``asyncio.to_thread`` would interleave and the test
    catches the violation.
    """
    intervals: list[tuple[float, float]] = []
    real_write = write_descriptors

    def instrumented(pd, ds):
        entry = time.perf_counter()
        # Hold long enough to make non-serialized interleavings visible.
        time.sleep(0.02)
        out = real_write(pd, ds)
        exit_ = time.perf_counter()
        intervals.append((entry, exit_))
        return out

    # Patch the symbol the async wrapper resolves at call time
    # (it does ``from app.services.model_descriptors import write_descriptors``
    # at import — patch at that namespace).
    monkeypatch.setattr(
        "app.services.model_descriptors_async.write_descriptors",
        instrumented,
    )

    descriptors_each = [
        {f"k{i}": {"endpoint": "mantle", "api": "responses", "region": "us-east-1"}}
        for i in range(5)
    ]
    await asyncio.gather(
        *[
            persist_descriptors_locked(tmp_path, descriptors_each[i], "proj-X")
            for i in range(5)
        ]
    )

    assert len(intervals) == 5, "expected 5 instrumented writes"
    intervals.sort()
    # No two intervals overlap — interval[i].exit <= interval[i+1].entry.
    for prev, nxt in zip(intervals, intervals[1:]):
        assert prev[1] <= nxt[0] + 1e-6, (
            f"writes overlapped: {prev} vs {nxt} — lock not enforced"
        )


async def test_persist_descriptors_locked_does_not_serialize_different_projects(
    tmp_path,
):
    """Different project_ids → independent locks; no cross-project blocking."""
    p1 = tmp_path / "p1"
    p2 = tmp_path / "p2"
    p1.mkdir()
    p2.mkdir()

    started = asyncio.Event()
    can_finish = asyncio.Event()

    async def slow_first():
        # Acquire proj-A's lock and hold it briefly to confirm proj-B
        # can still proceed.
        lock = await project_locks.get_project_lock("proj-A")
        async with lock:
            started.set()
            await can_finish.wait()
            await asyncio.to_thread(
                write_descriptors,
                p1,
                {"a": {"endpoint": "mantle", "api": "responses", "region": "us-east-1"}},
            )

    async def fast_second():
        await started.wait()
        # If the lock registry erroneously serialized across projects,
        # this call would block indefinitely (slow_first holds proj-A
        # and is awaiting can_finish). Use a short asyncio.wait_for to
        # surface the bug as TimeoutError.
        await asyncio.wait_for(
            persist_descriptors_locked(
                p2,
                {"b": {"endpoint": "mantle", "api": "responses", "region": "us-east-1"}},
                "proj-B",
            ),
            timeout=2.0,
        )
        can_finish.set()

    await asyncio.gather(slow_first(), fast_second())
    # Both writes landed.
    assert read_descriptors(p1) == {
        "a": {"endpoint": "mantle", "api": "responses", "region": "us-east-1"}
    }
    assert read_descriptors(p2) == {
        "b": {"endpoint": "mantle", "api": "responses", "region": "us-east-1"}
    }


async def test_persist_descriptors_locked_returns_path(tmp_path):
    """The async wrapper returns the same Path that ``write_descriptors`` returns."""
    out = await persist_descriptors_locked(
        tmp_path,
        {"k": {"endpoint": "mantle", "api": "responses", "region": "us-east-1"}},
        "proj-X",
    )
    assert out == tmp_path / "src" / "model_descriptors.json"
    assert out.exists()
