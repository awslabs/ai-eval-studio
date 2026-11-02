# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup audit for missing ``PROJECT_INDEX#{id}/META`` items.

Both feature-path flags resolve through ``find_project_by_id``, which is a
TWO-STEP read: the index item gives the owner, then the owner's partition gives
the record. If step 1 is missing, the lookup returns ``None`` and both flags
resolve to the OLD in-process path — silently, with no error anywhere. A project
in that state runs on a different architecture than every other project.

The index item is written by ``put_project``, so only a project never re-saved
since the sharing feature landed can lack it. That is expected to be zero
projects, but this exists precisely so "expected to be zero" is verified by a
machine rather than spot-checked by hand.

**These tests require the CLOUD store.** The dev ``FilesystemMetadataStore``
resolves ``find_project_by_id`` by reading config.json for the id and has no index
concept at all, so it cannot represent the failure — a dev-backed test would pass
against the bug.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

import app.main as main_module
from app.main import _audit_project_index_items
from app.platform import Platform, set_platform
from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore
from app.platform.dev_local import (
    LocalAuth,
    NoOpTokenVending,
    _build_dev_session_store,
)

pytestmark = pytest.mark.asyncio


def _create_table(ddb, name: str = "test-metadata") -> str:
    ddb.create_table(
        TableName=name,
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
            {"AttributeName": "GSI1PK", "AttributeType": "S"},
            {"AttributeName": "GSI1SK", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "GSI1",
                "KeySchema": [
                    {"AttributeName": "GSI1PK", "KeyType": "HASH"},
                    {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    return name


def _make_cloud_tree(projects_dir: Path, owner: str, project_id: str) -> Path:
    """Scaffold the EFS-side cloud tree the sweep enumerates."""
    d = projects_dir / "users" / owner / "projects" / project_id
    d.mkdir(parents=True)
    (d / "config.json").write_text(
        json.dumps(
            {
                "id": project_id,
                "name": "test",
                "description": "",
                "created_at": "2026-01-01T00:00:00Z",
                "owner_id": owner,
            }
        )
    )
    return d


async def _delete_index_item(store: DynamoDBMetadataStore, project_id: str) -> None:
    """Remove only the index item, simulating a project that predates it."""
    store.client.delete_item(
        TableName=store.table,
        Key={
            "PK": {"S": f"PROJECT_INDEX#{project_id}"},
            "SK": {"S": "META"},
        },
    )


async def test_missing_index_is_named_and_backfilled(tmp_path, monkeypatch):
    """A project missing its index item is named in the log AND repaired.

    Against the un-audited state this project resolves arch_split=False forever
    while its siblings resolve True, with nothing surfacing the split.
    """
    with mock_aws():
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        table = _create_table(ddb)
        store = DynamoDBMetadataStore(ddb_client=ddb, table_name=table)
        set_platform(
            Platform(
                metadata_store=store,
                session_store=_build_dev_session_store(),
                auth=LocalAuth(metadata_store=store),
                token_vending=NoOpTokenVending(),
            )
        )
        try:
            root = tmp_path / "projects"
            root.mkdir()
            monkeypatch.setattr(main_module, "PROJECTS_DIR", root)
            _make_cloud_tree(root, "owner-1", "proj-orphan")

            await store.put_project(
                {"id": "proj-orphan", "name": "test", "owner_id": "owner-1"}
            )
            # Simulate the pre-sharing state: canonical record present, index gone.
            await _delete_index_item(store, "proj-orphan")
            assert await store.find_project_by_id("proj-orphan") is None, (
                "precondition: the lookup the flag resolvers use must be broken"
            )

            await _audit_project_index_items()

            # ACCEPTANCE is the effect, not the log: the resolver lookup works.
            found = await store.find_project_by_id("proj-orphan")
            assert found is not None, (
                "The index item must be backfilled so find_project_by_id — and "
                "therefore both flag resolvers — resolve this project again."
            )
            assert found["owner_id"] == "owner-1"
        finally:
            set_platform(None)


async def test_healthy_project_is_not_rewritten(tmp_path, monkeypatch):
    """A project with a valid index item must not be re-put.

    Guards against the sweep burning a write per project on every boot, and
    against it clobbering a record it only meant to inspect.
    """
    with mock_aws():
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        table = _create_table(ddb)
        store = DynamoDBMetadataStore(ddb_client=ddb, table_name=table)
        set_platform(
            Platform(
                metadata_store=store,
                session_store=_build_dev_session_store(),
                auth=LocalAuth(metadata_store=store),
                token_vending=NoOpTokenVending(),
            )
        )
        try:
            root = tmp_path / "projects"
            root.mkdir()
            monkeypatch.setattr(main_module, "PROJECTS_DIR", root)
            _make_cloud_tree(root, "owner-1", "proj-healthy")
            await store.put_project(
                {"id": "proj-healthy", "name": "test", "owner_id": "owner-1"}
            )

            writes: list[dict] = []
            real_put = store.put_project

            async def _spy_put(project):
                writes.append(project)
                return await real_put(project)

            with patch.object(store, "put_project", side_effect=_spy_put):
                await _audit_project_index_items()

            assert writes == [], f"A healthy project must not be re-put: {writes}"
        finally:
            set_platform(None)


async def test_efs_tree_without_durable_record_is_left_alone(tmp_path, monkeypatch):
    """An EFS tree with no durable record is reported but NOT fabricated.

    A half-deleted project has a tree and no record. The sweep must not invent a
    project row from a directory name — that would resurrect deleted projects on
    every boot.
    """
    with mock_aws():
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        table = _create_table(ddb)
        store = DynamoDBMetadataStore(ddb_client=ddb, table_name=table)
        set_platform(
            Platform(
                metadata_store=store,
                session_store=_build_dev_session_store(),
                auth=LocalAuth(metadata_store=store),
                token_vending=NoOpTokenVending(),
            )
        )
        try:
            root = tmp_path / "projects"
            root.mkdir()
            monkeypatch.setattr(main_module, "PROJECTS_DIR", root)
            _make_cloud_tree(root, "owner-1", "proj-ghost")
            # Deliberately NO put_project — tree exists, table does not know it.

            writes: list[dict] = []
            real_put = store.put_project

            async def _spy_put(project):
                writes.append(project)
                return await real_put(project)

            with patch.object(store, "put_project", side_effect=_spy_put):
                await _audit_project_index_items()

            assert writes == [], (
                f"Must not fabricate a project record from a directory: {writes}"
            )
            assert await store.find_project_by_id("proj-ghost") is None
        finally:
            set_platform(None)


async def test_dev_layout_projects_skipped(tmp_path, monkeypatch):
    """Flat dev-layout trees have no durable index to audit."""
    with mock_aws():
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        table = _create_table(ddb)
        store = DynamoDBMetadataStore(ddb_client=ddb, table_name=table)
        set_platform(
            Platform(
                metadata_store=store,
                session_store=_build_dev_session_store(),
                auth=LocalAuth(metadata_store=store),
                token_vending=NoOpTokenVending(),
            )
        )
        try:
            root = tmp_path / "projects"
            root.mkdir()
            monkeypatch.setattr(main_module, "PROJECTS_DIR", root)
            flat = root / "proj-flat"
            flat.mkdir()
            (flat / "config.json").write_text(
                json.dumps({"id": "proj-flat", "name": "t", "owner_id": "local-dev"})
            )

            reads: list[str] = []
            real_find = store.find_project_by_id

            async def _spy_find(pid):
                reads.append(pid)
                return await real_find(pid)

            with patch.object(store, "find_project_by_id", side_effect=_spy_find):
                await _audit_project_index_items()

            assert reads == [], f"Dev-layout trees must be skipped: {reads}"
        finally:
            set_platform(None)


async def test_one_project_failure_does_not_abort_the_sweep(tmp_path, monkeypatch):
    """A read failure on one project must not stop the others being repaired."""
    with mock_aws():
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        table = _create_table(ddb)
        store = DynamoDBMetadataStore(ddb_client=ddb, table_name=table)
        set_platform(
            Platform(
                metadata_store=store,
                session_store=_build_dev_session_store(),
                auth=LocalAuth(metadata_store=store),
                token_vending=NoOpTokenVending(),
            )
        )
        try:
            root = tmp_path / "projects"
            root.mkdir()
            monkeypatch.setattr(main_module, "PROJECTS_DIR", root)
            # "proj-a" sorts before "proj-b" so the failure is hit first.
            _make_cloud_tree(root, "owner-1", "proj-a")
            _make_cloud_tree(root, "owner-1", "proj-b")
            for pid in ("proj-a", "proj-b"):
                await store.put_project(
                    {"id": pid, "name": "t", "owner_id": "owner-1"}
                )
                await _delete_index_item(store, pid)

            real_find = store.find_project_by_id

            async def _flaky_find(pid):
                if pid == "proj-a":
                    raise RuntimeError("DDB unavailable")
                return await real_find(pid)

            with patch.object(store, "find_project_by_id", side_effect=_flaky_find):
                # Must not raise.
                await _audit_project_index_items()

            # proj-b was still repaired despite proj-a's failure.
            assert await store.find_project_by_id("proj-b") is not None
        finally:
            set_platform(None)
