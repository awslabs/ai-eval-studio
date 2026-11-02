# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A project's identity comes from where its config.json is, never from its content.

``config.json`` sits in the project tree, which the project's agent can write.
Every save derives its destination from the config's ``id`` and ``owner_id``, so
when those came from the file, tenant A could edit its own ``config.json`` to name
tenant B's project and owner, and the coordinator's next save of A's project
(a rename, or the session save after every turn) overwrote B's config on disk and
B's DynamoDB record.

Each test poisons A's file and asserts the EFFECT on B: B's config file, B's
record and B's index item are byte-for-byte or field-for-field unchanged. A
response code alone would pass against the bug.
"""
from __future__ import annotations

import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

import app.main as main_module
from app.models.schemas import ProjectUpdate
from app.platform import Platform, set_platform
from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore
from app.platform.dev_local import LocalAuth, NoOpTokenVending, _build_dev_session_store
from app.platform.metadata_store import ProjectOwnerConflictError
from app.services.project_manager import ProjectManager

OWNER_A = "a1b2c3d4-e5f6-4a7b-8c9d-aaaabbbbcccc"
OWNER_B = "b1c2d3e4-f5a6-4b7c-8d9e-bbbbccccdddd"
PROJECT_A = "c1d2e3f4-a5b6-4c7d-8e9f-ccccddddeeee"
PROJECT_B = "d1e2f3a4-b5c6-4d7e-8f9a-ddddeeeeffff"


def _create_table(ddb) -> str:
    ddb.create_table(
        TableName="test-metadata",
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    return "test-metadata"


def _write_config(root: Path, owner: str, project_id: str, **content) -> Path:
    path = root / "users" / owner / "projects" / project_id / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "id": project_id,
        "name": f"project {project_id[:4]}",
        "description": "",
        "created_at": "2026-01-01T00:00:00Z",
        "owner_id": owner,
    }
    body.update(content)
    path.write_text(json.dumps(body, indent=2))
    return path


def _poisons() -> list[dict]:
    """Ways A's own config.json can name B: each field alone and both together."""
    return [
        {"id": PROJECT_B, "owner_id": OWNER_B},
        {"id": PROJECT_B},
        {"owner_id": OWNER_B},
    ]


@pytest.fixture
def cloud_tree(tmp_path, monkeypatch):
    """A cloud-mode tree holding A's and B's projects, with the app pointed at it."""
    monkeypatch.setenv("EVAL_STUDIO_MODE", "cloud")
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setattr(main_module, "PROJECTS_DIR", root)
    monkeypatch.setattr(main_module, "_project_manager", None)
    _write_config(root, OWNER_A, PROJECT_A)
    b_path = _write_config(root, OWNER_B, PROJECT_B, name="B's project")
    return root, b_path


@pytest.mark.parametrize("poison", _poisons())
def test_load_takes_identity_from_the_path(cloud_tree, poison):
    root, _ = cloud_tree
    _write_config(root, OWNER_A, PROJECT_A, **poison)

    config = ProjectManager(root).get_project(PROJECT_A)

    assert config is not None
    assert (config.id, config.owner_id) == (PROJECT_A, OWNER_A)


@pytest.mark.parametrize("poison", _poisons())
def test_saving_a_poisoned_config_writes_only_its_own_project(cloud_tree, poison):
    root, b_path = cloud_tree
    b_before = b_path.read_bytes()
    _write_config(root, OWNER_A, PROJECT_A, **poison)
    pm = ProjectManager(root)

    config = pm.get_project(PROJECT_A)
    config.name = "renamed"
    pm.save_config(config)

    assert b_path.read_bytes() == b_before
    a_saved = json.loads((root / "users" / OWNER_A / "projects" / PROJECT_A / "config.json").read_text())
    assert (a_saved["id"], a_saved["owner_id"], a_saved["name"]) == (PROJECT_A, OWNER_A, "renamed")


@pytest.mark.parametrize("poison", _poisons())
async def test_session_save_after_a_turn_leaves_b_untouched(cloud_tree, poison):
    """The save after every turn needs no API call, so it is its own path."""
    from app.services.agent_runner import _persist_session

    root, b_path = cloud_tree
    b_before = b_path.read_bytes()
    _write_config(root, OWNER_A, PROJECT_A, **poison)

    await _persist_session(PROJECT_A, "session-after-turn")

    assert b_path.read_bytes() == b_before
    a_saved = json.loads((root / "users" / OWNER_A / "projects" / PROJECT_A / "config.json").read_text())
    assert (a_saved["id"], a_saved["owner_id"]) == (PROJECT_A, OWNER_A)
    assert a_saved["session_id"] == "session-after-turn"


def test_listing_reports_the_path_identity(cloud_tree):
    root, _ = cloud_tree
    _write_config(root, OWNER_A, PROJECT_A, id=PROJECT_B, owner_id=OWNER_B)
    pm = ProjectManager(root)

    b_listing = {p.id for p in pm.list_projects(owner_id=OWNER_B)}
    a_listing = {p.id for p in pm.list_projects(owner_id=OWNER_A)}

    assert b_listing == {PROJECT_B}
    assert a_listing == {PROJECT_A}


@pytest.fixture
def ddb_store():
    with mock_aws():
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        yield DynamoDBMetadataStore(ddb_client=ddb, table_name=_create_table(ddb))


def _record(store: DynamoDBMetadataStore, owner: str, project_id: str) -> dict | None:
    return store.client.get_item(
        TableName=store.table,
        Key={"PK": {"S": f"USER#{owner}"}, "SK": {"S": f"PROJECT#{project_id}"}},
    ).get("Item")


def _index(store: DynamoDBMetadataStore, project_id: str) -> dict | None:
    return store.client.get_item(
        TableName=store.table,
        Key={"PK": {"S": f"PROJECT_INDEX#{project_id}"}, "SK": {"S": "META"}},
    ).get("Item")


async def test_put_project_cannot_move_an_id_to_another_owner(ddb_store):
    await ddb_store.put_project({"id": PROJECT_B, "name": "B's project", "owner_id": OWNER_B})
    index_before = _index(ddb_store, PROJECT_B)

    with pytest.raises(ProjectOwnerConflictError):
        await ddb_store.put_project({"id": PROJECT_B, "name": "taken", "owner_id": OWNER_A})

    assert _index(ddb_store, PROJECT_B) == index_before
    assert _record(ddb_store, OWNER_A, PROJECT_B) is None
    assert (await ddb_store.find_project_by_id(PROJECT_B))["owner_id"] == OWNER_B


async def test_put_project_still_creates_and_updates_its_own(ddb_store):
    await ddb_store.put_project({"id": PROJECT_A, "name": "first", "owner_id": OWNER_A})
    await ddb_store.put_project({"id": PROJECT_A, "name": "second", "owner_id": OWNER_A})

    found = await ddb_store.find_project_by_id(PROJECT_A)
    assert (found["owner_id"], found["name"]) == (OWNER_A, "second")


@pytest.fixture
async def cloud_app(cloud_tree, ddb_store):
    """The real PATCH handler over the cloud tree and the DynamoDB store."""
    set_platform(
        Platform(
            metadata_store=ddb_store,
            session_store=_build_dev_session_store(),
            auth=LocalAuth(metadata_store=ddb_store),
            token_vending=NoOpTokenVending(),
        )
    )
    await ddb_store.put_project({"id": PROJECT_A, "name": "A", "owner_id": OWNER_A})
    await ddb_store.put_project({"id": PROJECT_B, "name": "B's project", "owner_id": OWNER_B})
    try:
        yield cloud_tree
    finally:
        set_platform(None)


@pytest.mark.parametrize("poison", _poisons())
async def test_patch_of_a_poisoned_project_leaves_b_unchanged(cloud_app, ddb_store, poison):
    from app.api.projects import update_project

    root, b_path = cloud_app
    b_file = b_path.read_bytes()
    b_record = _record(ddb_store, OWNER_B, PROJECT_B)
    b_index = _index(ddb_store, PROJECT_B)
    _write_config(root, OWNER_A, PROJECT_A, **poison)

    result = await update_project(
        PROJECT_A, ProjectUpdate(name="renamed"), _checked=PROJECT_A, _user=OWNER_A
    )

    assert (result.id, result.owner_id, result.name) == (PROJECT_A, OWNER_A, "renamed")
    assert b_path.read_bytes() == b_file
    assert _record(ddb_store, OWNER_B, PROJECT_B) == b_record
    assert _index(ddb_store, PROJECT_B) == b_index
    assert _record(ddb_store, OWNER_A, PROJECT_B) is None


async def test_fields_the_agent_adds_to_the_file_reach_no_record(cloud_app, ddb_store):
    """Anything an agent writes into its own project's config.json beyond the
    settings must not reach the record that authorizes access."""
    from app.api.projects import update_project

    root, _ = cloud_app
    _write_config(root, OWNER_A, PROJECT_A, viewer_ids=[OWNER_B])

    await update_project(
        PROJECT_A, ProjectUpdate(name="renamed"), _checked=PROJECT_A, _user=OWNER_A
    )

    record = _record(ddb_store, OWNER_A, PROJECT_A)
    assert record["name"]["S"] == "renamed"
    assert "viewer_ids" not in record


async def test_patch_never_writes_a_whole_project_record(cloud_app, ddb_store, monkeypatch):
    """A whole-record put carries every field of the caller's copy, ACL
    included, so a settings edit must go through the settings-only update."""
    from app.api.projects import update_project

    async def refuse_put(project):
        raise AssertionError("PATCH must not replace the project record")

    monkeypatch.setattr(ddb_store, "put_project", refuse_put)

    result = await update_project(
        PROJECT_A, ProjectUpdate(name="renamed"), _checked=PROJECT_A, _user=OWNER_A
    )

    assert result.name == "renamed"
    assert _record(ddb_store, OWNER_A, PROJECT_A)["name"]["S"] == "renamed"


async def test_settings_update_refuses_identity_and_access_fields(ddb_store):
    await ddb_store.put_project({"id": PROJECT_A, "name": "A", "owner_id": OWNER_A})

    for field in ("owner_id", "deleting", "id"):
        with pytest.raises(ValueError):
            await ddb_store.update_project_settings(OWNER_A, PROJECT_A, {field: "x"})


async def test_patch_refuses_a_copy_outside_the_indexed_owners_tree(cloud_app, ddb_store):
    """A planted ``users/A/projects/<B's id>/`` must not be saved as B's project."""
    from fastapi import HTTPException

    from app.api.projects import update_project

    root, b_path = cloud_app
    b_path.unlink()
    b_path.parent.rmdir()
    _write_config(root, OWNER_A, PROJECT_B)
    b_index = _index(ddb_store, PROJECT_B)

    with pytest.raises(HTTPException) as refused:
        await update_project(
            PROJECT_B, ProjectUpdate(name="taken"), _checked=PROJECT_B, _user=OWNER_A
        )

    assert refused.value.status_code == 409
    assert _index(ddb_store, PROJECT_B) == b_index
    planted = json.loads((root / "users" / OWNER_A / "projects" / PROJECT_B / "config.json").read_text())
    assert planted["name"] != "taken"
