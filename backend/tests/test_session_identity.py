# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session identity: resume vs adopt-and-create, derived from store evidence.

Start-of-turn persistence mints a session id for a brand-new project before
the first turn runs. Passing every persisted id as ``resume`` breaks that
case: the CLI hard-fails resuming a session that was never created ("No
conversation found with session ID", exit 1), so a new project's first cloud
chat turn fails and the project stays wedged on the ghost id.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.agent_runner import _apply_session_identity


def _store(contents: list | None, raise_probe: bool = False):
    def list_objects_v2(Bucket, Prefix, MaxKeys):
        if raise_probe:
            raise RuntimeError("S3 blip")
        return {"Contents": contents} if contents else {}

    return SimpleNamespace(
        s3=SimpleNamespace(list_objects_v2=list_objects_v2),
        bucket="test-bucket",
    )


@pytest.mark.asyncio
async def test_existing_session_resumes():
    opts = {"session_store": _store([{"Key": "k/0001.jsonl"}])}
    await _apply_session_identity(opts, "sess-1", Path("/tmp/p"))
    assert opts.get("resume") == "sess-1"
    assert "session_id" not in opts


@pytest.mark.asyncio
async def test_ghost_session_is_adopted_not_resumed():
    """THE regression discriminator: an id with no stored object must become
    a create-with-id (--session-id), never a resume. Setting resume here
    makes the CLI exit 1 on every turn of a new project."""
    opts = {"session_store": _store(None)}
    await _apply_session_identity(opts, "ghost-1", Path("/tmp/p"))
    assert opts.get("session_id") == "ghost-1"
    assert "resume" not in opts


@pytest.mark.asyncio
async def test_probe_failure_fails_open_to_resume():
    """An S3 blip must not change behavior: fall back to resume."""
    opts = {"session_store": _store(None, raise_probe=True)}
    await _apply_session_identity(opts, "sess-2", Path("/tmp/p"))
    assert opts.get("resume") == "sess-2"
    assert "session_id" not in opts


@pytest.mark.asyncio
async def test_dev_store_none_resumes():
    """Dev (SDK default on-disk store): ids are persisted only after a
    successful turn there, so resume semantics were never wrong."""
    opts = {}
    await _apply_session_identity(opts, "sess-3", Path("/tmp/p"))
    assert opts.get("resume") == "sess-3"


@pytest.mark.asyncio
async def test_dashless_ghost_adopted_in_canonical_dashed_form():
    """Ids minted with ``uuid4().hex`` are dashless, and the CLI rejects a
    dashless ``--session-id`` with exit 1 ("Invalid session ID. Must be a
    valid UUID."). A persisted dashless ghost must be adopted in the
    canonical dashed form."""
    opts = {"session_store": _store(None)}
    await _apply_session_identity(
        opts, "721310b32d2e49898869b93020ce349c", Path("/tmp/p")
    )
    assert opts.get("session_id") == "721310b3-2d2e-4989-8869-b93020ce349c"
    assert "resume" not in opts


@pytest.mark.asyncio
async def test_dashed_real_session_resumes_unchanged():
    """A canonical dashed id (every real session — the CLI reports dashed)
    resumes exactly as persisted."""
    sid = "3f2b8c1a-9d4e-4f6a-8b2c-1d5e7f9a0b3c"
    opts = {"session_store": _store([{"Key": "k/0001.jsonl"}])}
    await _apply_session_identity(opts, sid, Path("/tmp/p"))
    assert opts.get("resume") == sid


@pytest.mark.asyncio
async def test_probe_prefix_uses_normalized_id():
    """The existence probe must ask about the SAME id the CLI will be given —
    the normalized dashed form — or probe and adoption would disagree on
    the key shape."""
    seen: dict = {}

    def list_objects_v2(Bucket, Prefix, MaxKeys):
        seen["prefix"] = Prefix
        return {}

    store = SimpleNamespace(
        s3=SimpleNamespace(list_objects_v2=list_objects_v2), bucket="b"
    )
    opts = {"session_store": store}
    await _apply_session_identity(
        opts, "721310b32d2e49898869b93020ce349c", Path("/tmp/p")
    )
    assert "721310b3-2d2e-4989-8869-b93020ce349c/" in seen["prefix"]
    assert "721310b32d2e49898869b93020ce349c/" not in seen["prefix"]


def test_canonical_session_id_forms():
    from app.services.agent_runner import _canonical_session_id

    # Dashless hex normalizes to dashed
    assert (
        _canonical_session_id("721310b32d2e49898869b93020ce349c")
        == "721310b3-2d2e-4989-8869-b93020ce349c"
    )
    # Already-canonical id is unchanged
    sid = "3f2b8c1a-9d4e-4f6a-8b2c-1d5e7f9a0b3c"
    assert _canonical_session_id(sid) == sid
    # Non-UUID passes through untouched (CLI is the authority on rejecting)
    assert _canonical_session_id("sess-1") == "sess-1"


@pytest.mark.asyncio
async def test_await_project_dir_returns_immediately_when_present(tmp_path):
    """Visible dir: zero retries, zero warnings — the hot path is silent."""
    from unittest import mock

    import app.services.agent_runner as ar

    with mock.patch.object(ar.logger, "warning") as warn:
        await ar._await_project_dir(tmp_path, attempts=4, delay_s=5.0)
    warn.assert_not_called()


@pytest.mark.asyncio
async def test_await_project_dir_retries_then_gives_up_quietly(tmp_path):
    """Missing dir: exactly ``attempts`` warned retries, no exception — the
    SDK's own cwd check owns the user-visible error."""
    from unittest import mock

    import app.services.agent_runner as ar

    with mock.patch.object(ar.logger, "warning") as warn:
        await ar._await_project_dir(
            tmp_path / "never-created", attempts=3, delay_s=0.01
        )
    assert warn.call_count == 3


@pytest.mark.asyncio
async def test_await_project_dir_sees_late_arrival(tmp_path):
    """Dir appearing between attempts ends the wait early."""
    from unittest import mock

    import app.services.agent_runner as ar

    target = tmp_path / "late"

    def _warn_then_create(*args, **kwargs):
        # First retry warning fires after the first failed check; create the
        # dir now so the SECOND check succeeds and no further warning fires.
        if not target.exists():
            target.mkdir()

    with mock.patch.object(ar.logger, "warning", side_effect=_warn_then_create) as warn:
        await ar._await_project_dir(target, attempts=5, delay_s=0.01)
    assert warn.call_count == 1
