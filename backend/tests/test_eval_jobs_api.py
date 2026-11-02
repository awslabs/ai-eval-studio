# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for eval job lifecycle endpoints (backend/app/api/eval_jobs.py)."""

import pytest
from httpx import ASGITransport, AsyncClient
from app.main import create_app


@pytest.fixture
def app(tmp_path):
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
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestEvalJobsAPI:
    async def test_get_eval_job_not_found(self, client):
        """Get a nonexistent eval job returns 404."""
        create_resp = await client.post("/api/projects", json={"name": "Job Test"})
        project_id = create_resp.json()["id"]

        resp = await client.get(f"/api/projects/{project_id}/eval/jobs/nonexistent-job-id")
        assert resp.status_code == 404

