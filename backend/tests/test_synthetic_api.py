# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the synthetic data review REST endpoints (app/api/synthetic.py)."""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
def app(tmp_path):
    """Create a FastAPI app with the projects dir pointed at a tmp dir."""
    import app.main as main_module

    original = main_module.PROJECTS_DIR
    main_module.PROJECTS_DIR = tmp_path / "projects"
    main_module.PROJECTS_DIR.mkdir()
    main_module._project_manager = None
    application = create_app()
    yield application
    main_module._project_manager = None
    main_module.PROJECTS_DIR = original


@pytest.fixture
async def client(app):
    """Async HTTP client wired to the test app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _seed_batch(
    projects_dir,
    project_id: str,
    batch_id: str,
    *,
    strategy: str = "paraphrase",
    timestamp: str | None = None,
    columns: list[str] | None = None,
    rows: list[dict] | None = None,
    statuses: list[str] | None = None,
    seed_source: str | None = None,
    seed_run_id: str | None = None,
):
    """Write a fake synthetic batch (manifest + CSV) into a project.

    Keeps the test bodies terse; mirrors the shape produced by
    ``synthesizer.write_batch`` without actually running DeepEval.
    """
    synthetic_dir = projects_dir / project_id / "data" / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    columns = columns or ["question", "answer"]
    rows = rows or [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2"},
        {"question": "q3", "answer": "a3"},
    ]
    statuses = statuses or ["pending"] * len(rows)
    assert len(rows) == len(statuses)

    csv_name = f"{batch_id}.csv"
    csv_path = synthetic_dir / csv_name
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(row.get(c, "") for c in columns) + "\n")

    manifest = {
        "batch_id": batch_id,
        "timestamp": timestamp or batch_id.split("_", 1)[0],
        "strategy": strategy,
        "seed_source": seed_source,
        "seed_run_id": seed_run_id,
        "total_cases": len(rows),
        "field_mappings": {"input": columns[0], "expected_output": columns[1]},
        "csv_file": csv_name,
        "cases": [
            {"index": i, "status": statuses[i]} for i in range(len(rows))
        ],
    }
    (synthetic_dir / f"{batch_id}.manifest.json").write_text(json.dumps(manifest))
    return manifest


class TestSyntheticBatchesAPI:
    async def test_list_empty_when_no_synthetic_dir(self, client):
        resp = await client.post("/api/projects", json={"name": "Synth Empty"})
        project_id = resp.json()["id"]

        list_resp = await client.get(f"/api/projects/{project_id}/synthetic/batches")
        assert list_resp.status_code == 200
        assert list_resp.json() == []

    async def test_list_sorts_by_timestamp_desc(self, client):
        import app.main as main_module

        resp = await client.post("/api/projects", json={"name": "Synth List"})
        project_id = resp.json()["id"]

        _seed_batch(
            main_module.PROJECTS_DIR,
            project_id,
            "2026-04-01T10-00-00_paraphrase",
        )
        _seed_batch(
            main_module.PROJECTS_DIR,
            project_id,
            "2026-04-10T12-00-00_from_scratch",
            strategy="from_scratch",
        )

        list_resp = await client.get(f"/api/projects/{project_id}/synthetic/batches")
        assert list_resp.status_code == 200
        batches = list_resp.json()
        assert len(batches) == 2
        # Newest first
        assert batches[0]["batch_id"] == "2026-04-10T12-00-00_from_scratch"
        assert batches[0]["strategy"] == "from_scratch"
        assert batches[1]["batch_id"] == "2026-04-01T10-00-00_paraphrase"
        assert batches[0]["pending"] == 3
        assert batches[0]["approved"] == 0
        assert batches[0]["rejected"] == 0
        assert batches[0]["total_cases"] == 3
        assert batches[0]["csv_file"].endswith(".csv")

    async def test_get_single_batch_returns_rows(self, client):
        import app.main as main_module

        resp = await client.post("/api/projects", json={"name": "Synth Detail"})
        project_id = resp.json()["id"]

        batch_id = "2026-04-22T14-05-31_paraphrase"
        _seed_batch(
            main_module.PROJECTS_DIR,
            project_id,
            batch_id,
            rows=[
                {"question": "what is 1+1", "answer": "2"},
                {"question": "sky color", "answer": "blue"},
            ],
            statuses=["pending", "approved"],
        )

        resp = await client.get(
            f"/api/projects/{project_id}/synthetic/batches/{batch_id}"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["batch_id"] == batch_id
        assert data["strategy"] == "paraphrase"
        assert data["field_mappings"] == {
            "input": "question",
            "expected_output": "answer",
        }
        assert len(data["rows"]) == 2
        assert data["rows"][0] == {
            "index": 0,
            "status": "pending",
            "values": {"question": "what is 1+1", "answer": "2"},
        }
        assert data["rows"][1]["status"] == "approved"

    async def test_get_unknown_batch_returns_404(self, client):
        resp = await client.post("/api/projects", json={"name": "Synth 404"})
        project_id = resp.json()["id"]

        resp = await client.get(
            f"/api/projects/{project_id}/synthetic/batches/does-not-exist"
        )
        assert resp.status_code == 404

    async def test_approve_by_indices(self, client):
        import app.main as main_module

        resp = await client.post("/api/projects", json={"name": "Synth Approve"})
        project_id = resp.json()["id"]

        batch_id = "2026-04-22T14-05-31_paraphrase"
        _seed_batch(main_module.PROJECTS_DIR, project_id, batch_id)

        resp = await client.post(
            f"/api/projects/{project_id}/synthetic/batches/{batch_id}/approve",
            json={"indices": [0, 2]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["approved"] == 2
        assert data["pending"] == 1
        assert data["rejected"] == 0
        statuses = {c["index"]: c["status"] for c in data["cases"]}
        assert statuses == {0: "approved", 1: "pending", 2: "approved"}

    async def test_approve_all_flips_only_pending(self, client):
        import app.main as main_module

        resp = await client.post("/api/projects", json={"name": "Synth ApproveAll"})
        project_id = resp.json()["id"]

        batch_id = "2026-04-22T14-05-31_paraphrase"
        _seed_batch(
            main_module.PROJECTS_DIR,
            project_id,
            batch_id,
            statuses=["pending", "rejected", "pending"],
        )

        resp = await client.post(
            f"/api/projects/{project_id}/synthetic/batches/{batch_id}/approve",
            json={"approve_all": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        # The previously-rejected case must NOT be silently re-approved.
        statuses = {c["index"]: c["status"] for c in data["cases"]}
        assert statuses == {0: "approved", 1: "rejected", 2: "approved"}
        assert data["approved"] == 2
        assert data["rejected"] == 1
        assert data["pending"] == 0

    async def test_approve_empty_payload_returns_400(self, client):
        import app.main as main_module

        resp = await client.post("/api/projects", json={"name": "Synth Approve 400"})
        project_id = resp.json()["id"]

        batch_id = "2026-04-22T14-05-31_paraphrase"
        _seed_batch(main_module.PROJECTS_DIR, project_id, batch_id)

        resp = await client.post(
            f"/api/projects/{project_id}/synthetic/batches/{batch_id}/approve",
            json={"indices": [], "approve_all": False},
        )
        assert resp.status_code == 400

    async def test_reject_by_indices(self, client):
        import app.main as main_module

        resp = await client.post("/api/projects", json={"name": "Synth Reject"})
        project_id = resp.json()["id"]

        batch_id = "2026-04-22T14-05-31_paraphrase"
        _seed_batch(main_module.PROJECTS_DIR, project_id, batch_id)

        resp = await client.post(
            f"/api/projects/{project_id}/synthetic/batches/{batch_id}/reject",
            json={"indices": [1]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["rejected"] == 1
        assert data["pending"] == 2
        assert data["approved"] == 0
        statuses = {c["index"]: c["status"] for c in data["cases"]}
        assert statuses[1] == "rejected"

    async def test_reject_empty_payload_returns_400(self, client):
        import app.main as main_module

        resp = await client.post("/api/projects", json={"name": "Synth Reject 400"})
        project_id = resp.json()["id"]

        batch_id = "2026-04-22T14-05-31_paraphrase"
        _seed_batch(main_module.PROJECTS_DIR, project_id, batch_id)

        resp = await client.post(
            f"/api/projects/{project_id}/synthetic/batches/{batch_id}/reject",
            json={"indices": []},
        )
        assert resp.status_code == 400
