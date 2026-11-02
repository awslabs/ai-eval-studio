# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the report endpoints (backend/app/api/reports.py).

The API now serves JSON, not HTML. The frontend renders the React
components from this JSON. PDF export is client-side; no backend
involvement, no test for it here.
"""

import json

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


#: config.py states the platform cannot read: not valid Python, and a field
#: whose value has the wrong shape. The agent writes this file, so either can
#: happen; neither may take a screen down.
BROKEN_CONFIGS = {
    "syntax-error": "class EvalConfig(:\n",
    "models-not-strings": "models = [1, 2]\ndata_file = 'cases.csv'\n",
}


def _seed_run(project_id: str):
    """Write a minimal eval run + prompt into a created project."""
    import app.main as main_module
    proj = main_module.PROJECTS_DIR / project_id
    output_dir = proj / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = proj / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "baseline.md").write_text("You are a helpful assistant.")
    run = [{
        "source": "manual",
        "model_id": "us.anthropic.claude-sonnet-4-6",
        "test_run_id": "sonnet",
        "timestamp": "2026-06-05T10:00:00Z",
        "hyperparameters": {
            "model": "us.anthropic.claude-sonnet-4-6",
            "endpoint": "runtime",
            "region": "us-east-1",
            "prompt_version": "baseline",
            "temperature": 0.0,
            "max_tokens": 2048,
        },
        "results": [{
            "name": "tc0", "success": True, "input": "hi", "actual_output": "yo",
            "expected_output": "yo", "context": None, "retrieval_context": None,
            "additional_metadata": {
                "id": 0, "latency_ms": 100, "generation_cost": 0.001,
                "input_tokens": 100, "output_tokens": 50,
                "cache_read_input_tokens": 0, "cache_write_input_tokens": 0,
            },
            "metrics_data": [{"name": "Acc", "score": 1.0, "threshold": 0.5,
                              "success": True, "reason": "r", "error": None}],
        }],
        "generation_cost_usd": 0.001,
        "judge_cost_usd": 0.0005,
        "data_quality": {
            "reliability": "ok", "errored_count": 0, "dropped_count": 0,
            "null_count": 0, "unrecovered_count": 0, "total_cases": 1,
        },
    }]
    (output_dir / "eval_results_baseline_20260605.json").write_text(json.dumps(run))


class TestRunReportEndpoint:
    async def test_returns_json_envelope(self, client):
        create = await client.post("/api/projects", json={"name": "Report Proj"})
        project_id = create.json()["id"]
        _seed_run(project_id)

        resp = await client.post(f"/api/projects/{project_id}/report", json={})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        body = resp.json()
        assert body["scope"] == "run"
        assert body["schema_version"] >= 1
        # Facts surface the right shape.
        assert body["facts"]["project_name"] == "Report Proj"
        assert body["facts"]["run_filename"] == "eval_results_baseline_20260605.json"
        # Narrative absent (no agent run yet) — the frontend renders the lean fallback.
        assert body["narrative"] is None

    async def test_explicit_run_in_body(self, client):
        create = await client.post("/api/projects", json={"name": "Report Proj"})
        project_id = create.json()["id"]
        _seed_run(project_id)
        resp = await client.post(
            f"/api/projects/{project_id}/report",
            json={"run": "eval_results_baseline_20260605.json"},
        )
        assert resp.status_code == 200
        assert resp.json()["facts"]["run_filename"] == "eval_results_baseline_20260605.json"

    async def test_404_when_no_runs(self, client):
        create = await client.post("/api/projects", json={"name": "Empty"})
        project_id = create.json()["id"]
        resp = await client.post(f"/api/projects/{project_id}/report", json={})
        assert resp.status_code == 404

    async def test_404_when_run_missing(self, client):
        create = await client.post("/api/projects", json={"name": "Has Runs"})
        project_id = create.json()["id"]
        _seed_run(project_id)
        resp = await client.post(
            f"/api/projects/{project_id}/report",
            json={"run": "nope.json"},
        )
        assert resp.status_code == 404

    async def test_404_for_unknown_project(self, client):
        resp = await client.post("/api/projects/does-not-exist/report", json={})
        assert resp.status_code == 404


class TestExecutiveReportEndpoint:
    async def test_returns_json_envelope(self, client):
        create = await client.post("/api/projects", json={"name": "Exec Proj"})
        project_id = create.json()["id"]
        _seed_run(project_id)
        resp = await client.post(f"/api/projects/{project_id}/report/executive")
        assert resp.status_code == 200
        body = resp.json()
        assert body["scope"] == "executive"
        # Exec facts inline the winner run + journey.
        assert "winner_run" in body["facts"]
        assert "journey" in body["facts"]

    @pytest.mark.parametrize("source", list(BROKEN_CONFIGS.values()), ids=list(BROKEN_CONFIGS))
    async def test_builds_when_config_py_is_unreadable(self, client, source):
        """The winner is ranked against config.py targets; a broken file means no ranking, not a 500."""
        import app.main as main_module

        project_id = (await client.post("/api/projects", json={"name": "Exec Broken"})).json()["id"]
        _seed_run(project_id)
        src = main_module.PROJECTS_DIR / project_id / "src"
        src.mkdir(parents=True, exist_ok=True)
        (src / "config.py").write_text(source)

        resp = await client.post(f"/api/projects/{project_id}/report/executive")
        assert resp.status_code == 200, resp.text

    async def test_404_when_no_runs(self, client):
        create = await client.post("/api/projects", json={"name": "Empty Exec"})
        project_id = create.json()["id"]
        resp = await client.post(f"/api/projects/{project_id}/report/executive")
        assert resp.status_code == 404

    async def test_cached_narrative_kept_on_same_version_rerun(self, client):
        # Mid-optimization, the same prompt version often keeps winning across
        # iterations. Each new run rolls the trophy filename forward — the
        # narrative is about the VERSION, not the specific run, so dropping
        # it on every iteration produced a barebones flicker. Pin the new
        # contract: same winner_version → cached narrative survives.
        import app.main as main_module
        from app.services.report import build_exec_facts, write_report_json

        create = await client.post("/api/projects", json={"name": "Same Ver"})
        project_id = create.json()["id"]
        _seed_run(project_id)

        proj = main_module.PROJECTS_DIR / project_id
        # Seed a sidecar stamped with the v=baseline winner the seed run
        # produces, plus a richer-but-stale prose body to detect drops.
        facts = build_exec_facts(proj)
        write_report_json(
            proj, "executive", facts,
            narrative={"intro": "Cached prose for baseline."},
        )

        # Now drop in a SECOND run of the same baseline version with a
        # later timestamp — the trophy run filename changes, but the
        # winning version does not.
        run2 = json.loads(
            (proj / "output" / "eval_results_baseline_20260605.json").read_text()
        )
        run2[0]["timestamp"] = "2026-06-06T10:00:00Z"
        (proj / "output" / "eval_results_baseline_20260606.json").write_text(json.dumps(run2))

        resp = await client.post(f"/api/projects/{project_id}/report/executive")
        assert resp.status_code == 200
        body = resp.json()
        assert body["narrative"] == {"intro": "Cached prose for baseline."}
        # Same version, different run file → narrative kept but flagged stale
        # so the UI can disable Download until a fresh narration lands.
        assert body["stale"] is True

    async def test_stale_false_when_winner_run_matches(self, client):
        # No new run since the cached narrative was written → stale is
        # False. The cached prose is fully current.
        import app.main as main_module
        from app.services.report import build_exec_facts, write_report_json

        create = await client.post("/api/projects", json={"name": "Fresh"})
        project_id = create.json()["id"]
        _seed_run(project_id)
        proj = main_module.PROJECTS_DIR / project_id
        facts = build_exec_facts(proj)
        write_report_json(
            proj, "executive", facts,
            narrative={"intro": "Cached prose for baseline."},
        )

        resp = await client.post(f"/api/projects/{project_id}/report/executive")
        body = resp.json()
        assert body["narrative"] == {"intro": "Cached prose for baseline."}
        assert body["stale"] is False

    async def test_stale_false_when_narrative_absent(self, client):
        # No sidecar yet → narrative is None → stale is False (a missing
        # narrative is honest, not stale; the lean fallback renders).
        create = await client.post("/api/projects", json={"name": "No Sidecar"})
        project_id = create.json()["id"]
        _seed_run(project_id)
        resp = await client.post(f"/api/projects/{project_id}/report/executive")
        body = resp.json()
        assert body["narrative"] is None
        assert body["stale"] is False

    async def test_cached_narrative_dropped_on_version_change(self, client):
        # Counter-test: a real version change MUST still drop the cached
        # prose so we never show a stale recommendation.
        import app.main as main_module
        from app.services.report import build_exec_facts, write_report_json

        create = await client.post("/api/projects", json={"name": "New Ver"})
        project_id = create.json()["id"]
        _seed_run(project_id)
        proj = main_module.PROJECTS_DIR / project_id
        facts = build_exec_facts(proj)
        write_report_json(
            proj, "executive", facts,
            narrative={"intro": "Cached prose for baseline."},
        )

        # New version v2 wins outright (higher pass rate, later timestamp).
        (proj / "prompts" / "v2.md").write_text("Better prompt.")
        run_v2 = json.loads(
            (proj / "output" / "eval_results_baseline_20260605.json").read_text()
        )
        run_v2[0]["hyperparameters"]["prompt_version"] = "v2"
        run_v2[0]["timestamp"] = "2026-06-06T10:00:00Z"
        (proj / "output" / "eval_results_v2_20260606.json").write_text(json.dumps(run_v2))

        resp = await client.post(f"/api/projects/{project_id}/report/executive")
        assert resp.status_code == 200
        body = resp.json()
        assert body["narrative"] is None  # Cached prose was about baseline.
        assert body["facts"]["winner_version"] == "v2"


class TestRegenerateEndpoints:
    async def test_run_regenerate_400_without_run(self, client):
        create = await client.post("/api/projects", json={"name": "Regen Run"})
        project_id = create.json()["id"]
        _seed_run(project_id)
        resp = await client.post(
            f"/api/projects/{project_id}/report/regenerate", json={},
        )
        assert resp.status_code == 400

    @pytest.mark.parametrize(
        ("path", "body", "generator", "expected_args"),
        [
            (
                "report/regenerate",
                {"run": "eval_results_baseline_20260605.json"},
                "generate_run_report",
                ("eval_results_baseline_20260605.json",),
            ),
            ("report/executive/regenerate", None, "generate_exec_report", ()),
        ],
    )
    async def test_regenerate_returns_before_generation_finishes(
        self, client, path, body, generator, expected_args,
    ):
        # A narration outlasts CloudFront's 30-second origin read timeout,
        # so an endpoint that waited for it reached the browser as a 504
        # even though the report was written.
        import asyncio
        from unittest.mock import patch

        from app.api import reports as reports_api

        create = await client.post("/api/projects", json={"name": "Regen Async"})
        project_id = create.json()["id"]
        _seed_run(project_id)

        started = asyncio.Event()
        release = asyncio.Event()
        calls: list[tuple] = []

        async def slow_generate(project_dir, pid, *args):
            calls.append((pid, *args))
            started.set()
            await release.wait()
            return True

        with patch(f"app.services.report_generation.{generator}", slow_generate):
            # Generation is held open, so an endpoint that waited for it
            # would time out here rather than return.
            resp = await asyncio.wait_for(
                client.post(
                    f"/api/projects/{project_id}/{path}",
                    **({"json": body} if body is not None else {}),
                ),
                timeout=5,
            )
            assert resp.status_code == 202
            assert resp.json() == {"ok": True}

            await asyncio.wait_for(started.wait(), timeout=5)
            (task,) = reports_api._regenerate_tasks
            assert not task.done()

            release.set()
            assert await asyncio.wait_for(task, timeout=5) is True

        # The caller is the owner the narrator acts for.
        from app.platform.auth import DEFAULT_LOCAL_DEV_USER_ID
        assert calls == [(project_id, *expected_args, DEFAULT_LOCAL_DEV_USER_ID)]
        # Done-callbacks run on the next loop pass.
        await asyncio.sleep(0)
        assert reports_api._regenerate_tasks == set()


class TestReportAuthz:
    def test_route_requires_project_read(self, app):
        from app.platform.api_authz import require_project_read
        from tests.conftest import iter_api_routes

        for route in iter_api_routes(app):
            if route.path == "/api/projects/{project_id}/report":
                calls = {d.call for d in route.dependant.dependencies}
                assert require_project_read in calls
                return
        raise AssertionError("No /api/projects/{project_id}/report route registered")

    def test_executive_route_requires_project_read(self, app):
        from app.platform.api_authz import require_project_read
        from tests.conftest import iter_api_routes

        for route in iter_api_routes(app):
            if route.path == "/api/projects/{project_id}/report/executive":
                calls = {d.call for d in route.dependant.dependencies}
                assert require_project_read in calls
                return
        raise AssertionError("No executive report route registered")

    def test_regenerate_routes_require_project_write(self, app):
        from app.platform.api_authz import require_project_write
        from tests.conftest import iter_api_routes

        targets = {
            "/api/projects/{project_id}/report/regenerate",
            "/api/projects/{project_id}/report/executive/regenerate",
        }
        found: set[str] = set()
        for route in iter_api_routes(app):
            if route.path in targets:
                calls = {d.call for d in route.dependant.dependencies}
                assert require_project_write in calls, route.path
                found.add(route.path)
        assert found == targets, f"missing routes: {targets - found}"
