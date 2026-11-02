# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for eval run history and dataset endpoints (backend/app/api/eval_runs.py)."""

import pytest
import json
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


#: config.py states the platform cannot read: not valid Python, and a field
#: whose value has the wrong shape. The agent writes this file, so either can
#: happen; neither may take a screen down.
BROKEN_CONFIGS = {
    "syntax-error": "class EvalConfig(:\n",
    "models-not-strings": "models = [1, 2]\ndata_file = 'cases.csv'\n",
}


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestEvalRunsAPI:
    async def test_list_runs_empty(self, client):
        create_resp = await client.post("/api/projects", json={"name": "No Runs"})
        project_id = create_resp.json()["id"]

        resp = await client.get(f"/api/projects/{project_id}/eval/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_runs_with_data(self, client, tmp_path):
        create_resp = await client.post("/api/projects", json={"name": "Has Runs"})
        project_id = create_resp.json()["id"]

        # Manually write a result file in DeepEval-native format (array of model runs)
        import app.main as main_module
        output_dir = main_module.PROJECTS_DIR / project_id / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        result = [{
            "source": "manual",
            "test_run_id": "test-model",
            "timestamp": "2026-01-01T00:00:00",
            "hyperparameters": {
                "model": "test",
                "endpoint": "runtime",
                "region": "us-east-1",
                "prompt_version": "baseline",
            },
            "results": [{"name": "tc0", "success": True, "input": "hello", "actual_output": "world", "expected_output": "world", "metrics_data": []}],
        }]
        (output_dir / "eval_results_2026-01-01_00-00-00.json").write_text(json.dumps(result))

        resp = await client.get(f"/api/projects/{project_id}/eval/runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert len(runs) == 1
        assert "model_runs" in runs[0]
        assert runs[0]["model_runs"][0]["test_run_id"] == "test-model"

    async def test_list_prompts_empty(self, client):
        """List prompts for a project with no prompt files returns empty list."""
        create_resp = await client.post("/api/projects", json={"name": "No Prompts"})
        project_id = create_resp.json()["id"]

        resp = await client.get(f"/api/projects/{project_id}/prompts")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_prompts_with_files(self, client):
        """List prompts returns sorted filenames when prompt files exist."""
        create_resp = await client.post("/api/projects", json={"name": "Has Prompts"})
        project_id = create_resp.json()["id"]

        # Manually create prompt files in the project's prompts directory
        import app.main as main_module
        prompts_dir = main_module.PROJECTS_DIR / project_id / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        (prompts_dir / "baseline.md").write_text("You are a helpful assistant.")
        (prompts_dir / "v1.md").write_text("You are an expert assistant.")
        (prompts_dir / "v2.md").write_text("You are a precise assistant.")

        resp = await client.get(f"/api/projects/{project_id}/prompts")
        assert resp.status_code == 200
        prompts = resp.json()
        assert prompts == ["v2.md", "v1.md", "baseline.md"]

    async def test_delete_run_removes_file(self, client):
        """``DELETE /eval/runs/{file}`` removes the file and returns 204."""
        create_resp = await client.post("/api/projects", json={"name": "Has Runs"})
        project_id = create_resp.json()["id"]

        import app.main as main_module
        output_dir = main_module.PROJECTS_DIR / project_id / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        result = [{
            "source": "manual",
            "test_run_id": "m",
            "timestamp": "2026-01-01T00:00:00",
            "hyperparameters": {
                "model": "m",
                "endpoint": "runtime",
                "region": "us-east-1",
                "prompt_version": "baseline",
            },
            "results": [],
        }]
        run_path = output_dir / "eval_results_2026-01-01_00-00-00.json"
        run_path.write_text(json.dumps(result))

        resp = await client.delete(
            f"/api/projects/{project_id}/eval/runs/eval_results_2026-01-01_00-00-00.json"
        )
        assert resp.status_code == 204
        assert not run_path.exists()

    async def test_delete_run_publishes_project_event(self, client):
        """A successful delete must publish ``eval_run/deleted``.

        The run list is a pull-only REST resource, so without this event the
        SPA keeps rendering a run whose file is already gone until unrelated
        navigation happens to refetch.
        """
        from unittest.mock import patch

        create_resp = await client.post("/api/projects", json={"name": "Publish"})
        project_id = create_resp.json()["id"]

        import app.main as main_module
        output_dir = main_module.PROJECTS_DIR / project_id / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "eval_results_pub.json").write_text("[]")

        with patch("app.services.jobs._publish_project_event") as publish:
            resp = await client.delete(
                f"/api/projects/{project_id}/eval/runs/eval_results_pub.json"
            )

        assert resp.status_code == 204
        publish.assert_called_once()
        published_project_id, event = publish.call_args.args
        assert published_project_id == project_id
        assert event["resource"] == "eval_run"
        assert event["type"] == "deleted"
        assert event["run_filename"] == "eval_results_pub.json"

    async def test_delete_run_404_publishes_nothing(self, client):
        """A failed delete must NOT publish — the list is unchanged."""
        from unittest.mock import patch

        create_resp = await client.post("/api/projects", json={"name": "NoPublish"})
        project_id = create_resp.json()["id"]

        with patch("app.services.jobs._publish_project_event") as publish:
            resp = await client.delete(
                f"/api/projects/{project_id}/eval/runs/nonexistent.json"
            )

        assert resp.status_code == 404
        publish.assert_not_called()

    async def test_delete_run_404_when_missing(self, client):
        create_resp = await client.post("/api/projects", json={"name": "Empty"})
        project_id = create_resp.json()["id"]
        resp = await client.delete(
            f"/api/projects/{project_id}/eval/runs/nonexistent.json"
        )
        assert resp.status_code == 404

    async def test_delete_run_rejects_path_traversal(self, client):
        create_resp = await client.post("/api/projects", json={"name": "Trav"})
        project_id = create_resp.json()["id"]
        resp = await client.delete(
            f"/api/projects/{project_id}/eval/runs/..%2Fconfig.json"
        )
        # FastAPI's default URL parsing treats ``%2F`` as ``/`` only when
        # explicitly enabled — so the route either 404s (no match) or
        # 400s (validator rejects). Either is acceptable; the
        # invariant is "the parent dir is never touched".
        assert resp.status_code in (400, 404)

    async def test_get_run_rejects_symlink_to_outside_output(self, client, tmp_path):
        """A symlink in output/ pointing outside the project tree must be
        rejected by GET — without ``safe_path``'s symlink resolution,
        ``GET /eval/runs/{name}`` becomes an arbitrary-read primitive
        scoped to project_read."""
        import os
        create_resp = await client.post("/api/projects", json={"name": "Sym"})
        project_id = create_resp.json()["id"]

        import app.main as main_module
        output_dir = main_module.PROJECTS_DIR / project_id / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        secret = tmp_path / "secret.txt"
        secret.write_text("nope")
        os.symlink(secret, output_dir / "eval_results_pwn.json")

        resp = await client.get(
            f"/api/projects/{project_id}/eval/runs/eval_results_pwn.json"
        )
        # safe_path's resolved containment check catches the symlink
        # target → 400 Invalid run filename.
        assert resp.status_code == 400, resp.text

    async def test_delete_run_500_does_not_leak_efs_path(self, client, monkeypatch):
        """OSError on unlink must NOT leak the absolute filesystem path
        (per-tenant EFS layout) to the HTTP response — log it server-side
        only."""
        from pathlib import Path
        create_resp = await client.post("/api/projects", json={"name": "Leak"})
        project_id = create_resp.json()["id"]

        import app.main as main_module
        output_dir = main_module.PROJECTS_DIR / project_id / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "eval_results_leak.json"
        path.write_text("[]")

        original_unlink = Path.unlink
        def fail_unlink(self, *args, **kw):
            raise PermissionError(f"[Errno 13] Permission denied: {self}")
        monkeypatch.setattr(Path, "unlink", fail_unlink)

        resp = await client.delete(
            f"/api/projects/{project_id}/eval/runs/eval_results_leak.json"
        )
        assert resp.status_code == 500
        # The detail must NOT contain the absolute path or owner_sub.
        assert "Errno" not in resp.text
        assert str(output_dir) not in resp.text
        # restore — pytest fixture teardown also handles it but be safe.
        monkeypatch.setattr(Path, "unlink", original_unlink)

    async def test_get_run_compact_projection(self, client):
        """Compact endpoint returns slim per-case projections without full text."""
        create_resp = await client.post("/api/projects", json={"name": "Compact Test"})
        project_id = create_resp.json()["id"]

        import app.main as main_module
        output_dir = main_module.PROJECTS_DIR / project_id / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Seed a run with one case carrying additional_metadata.id, latency,
        # tokens, and a metrics_data entry with a long reason + actual_output
        result = [{
            "source": "manual",
            "test_run_id": "test-model",
            "timestamp": "2026-01-01T00:00:00",
            "hyperparameters": {
                "model": "test",
                "endpoint": "runtime",
                "region": "us-east-1",
                "prompt_version": "baseline",
            },
            "results": [{
                "name": "tc0",
                "success": True,
                "input": "hello world input text",
                "actual_output": "world output text response",
                "expected_output": "world",
                "context": None,
                "retrieval_context": None,
                "additional_metadata": {
                    "id": "case-123",
                    "latency_ms": 500,
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "cache_read_input_tokens": 5,
                    "cache_write_input_tokens": 3,
                },
                "metrics_data": [{
                    "name": "Exact",
                    "success": False,
                    "score": 0.0,
                    "reason": "This is a very long reason explaining why the metric failed in detail.",
                    "threshold": 1.0,
                }],
            }],
        }]
        (output_dir / "eval_results_2026-01-01_00-00-00.json").write_text(json.dumps(result))

        resp = await client.get(
            f"/api/projects/{project_id}/eval/runs/eval_results_2026-01-01_00-00-00.json/compact"
        )
        assert resp.status_code == 200
        compact = resp.json()
        assert len(compact) == 1
        assert len(compact[0]["results"]) == 1
        case = compact[0]["results"][0]
        # Compact case has output_preview/output_hash/promoted latency_ms
        assert "output_preview" in case
        assert case["output_preview"] == "world output text response"
        assert "output_hash" in case
        assert case["latency_ms"] == 500
        assert case["input_tokens"] == 10
        assert case["case_id"] == "case-123"
        # metrics has name/success/score but NOT reason
        assert len(case["metrics"]) == 1
        assert case["metrics"][0]["name"] == "Exact"
        assert case["metrics"][0]["success"] is False
        assert case["metrics"][0]["score"] == 0.0
        assert "reason" not in case["metrics"][0]
        # Full text fields are absent
        assert "actual_output" not in case
        assert "input" not in case
        assert "expected_output" not in case

    async def test_get_case_detail_happy_path(self, client):
        """Case detail endpoint returns full input/expected + per-model actual_output + metrics.reason."""
        create_resp = await client.post("/api/projects", json={"name": "Detail Test"})
        project_id = create_resp.json()["id"]

        import app.main as main_module
        output_dir = main_module.PROJECTS_DIR / project_id / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = [{
            "source": "manual",
            "test_run_id": "model-a",
            "timestamp": "2026-01-01T00:00:00",
            "hyperparameters": {
                "model": "model-a",
                "endpoint": "runtime",
                "region": "us-east-1",
                "prompt_version": "baseline",
            },
            "results": [{
                "name": "tc0",
                "success": True,
                "input": "full input text",
                "actual_output": "output from model a",
                "expected_output": "expected output text",
                "additional_metadata": {
                    "id": "case-456",
                    "latency_ms": 300,
                    "input_tokens": 8,
                    "output_tokens": 12,
                },
                "metrics_data": [{
                    "name": "Exact",
                    "success": True,
                    "score": 1.0,
                    "reason": "Full metric reason for model a",
                }],
            }],
        }]
        (output_dir / "eval_results_2026-01-01_00-00-00.json").write_text(json.dumps(result))

        resp = await client.get(
            f"/api/projects/{project_id}/eval/runs/eval_results_2026-01-01_00-00-00.json/cases/case-456"
        )
        assert resp.status_code == 200
        detail = resp.json()
        assert detail["input"] == "full input text"
        assert detail["expected_output"] == "expected output text"
        assert "model-a" in detail["models"]
        model_a = detail["models"]["model-a"]
        assert model_a["actual_output"] == "output from model a"
        assert model_a["latency_ms"] == 300
        assert model_a["input_tokens"] == 8
        assert len(model_a["metrics"]) == 1
        assert model_a["metrics"][0]["reason"] == "Full metric reason for model a"

    async def test_get_case_detail_404_for_unknown_id(self, client):
        """Case detail endpoint returns 404 when case_id does not match any case."""
        create_resp = await client.post("/api/projects", json={"name": "Not Found"})
        project_id = create_resp.json()["id"]

        import app.main as main_module
        output_dir = main_module.PROJECTS_DIR / project_id / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = [{
            "source": "manual",
            "test_run_id": "m",
            "timestamp": "2026-01-01T00:00:00",
            "hyperparameters": {
                "model": "m",
                "endpoint": "runtime",
                "region": "us-east-1",
                "prompt_version": "baseline",
            },
            "results": [{
                "name": "tc0",
                "success": True,
                "input": "x",
                "actual_output": "y",
                "expected_output": "y",
                "additional_metadata": {"id": "known"},
                "metrics_data": [],
            }],
        }]
        (output_dir / "eval_results_2026-01-01_00-00-00.json").write_text(json.dumps(result))

        resp = await client.get(
            f"/api/projects/{project_id}/eval/runs/eval_results_2026-01-01_00-00-00.json/cases/unknown"
        )
        assert resp.status_code == 404

    async def test_get_compact_rejects_symlink_to_outside_output(self, client, tmp_path):
        """Compact endpoint must reject symlinks pointing outside output/ (same defense as get_eval_run)."""
        import os
        create_resp = await client.post("/api/projects", json={"name": "Sym"})
        project_id = create_resp.json()["id"]

        import app.main as main_module
        output_dir = main_module.PROJECTS_DIR / project_id / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        secret = tmp_path / "secret.txt"
        secret.write_text("nope")
        os.symlink(secret, output_dir / "eval_results_pwn.json")

        resp = await client.get(
            f"/api/projects/{project_id}/eval/runs/eval_results_pwn.json/compact"
        )
        assert resp.status_code == 400, resp.text


class TestEvalRunsAuthz:
    """The project-scoped read endpoints must enforce ``require_project_read``.

    These endpoints expose eval data and prompts; they previously imported
    the read gate but never applied it, leaving them ungated at the route
    level. Pin the dependency so a future edit can't silently drop it.
    Introspecting the route's dependencies proves the gate is wired
    without coupling to dev-mode auth internals.
    """

    def _dependency_calls(self, app, path: str, method: str = "GET"):
        from tests.conftest import iter_api_routes

        for route in iter_api_routes(app):
            if route.path == path and method in route.methods:
                return {d.call for d in route.dependant.dependencies}
        raise AssertionError(f"No {method} route for {path}")

    def test_read_endpoints_require_project_read(self, app):
        from app.platform.api_authz import require_project_read

        gated = [
            "/api/projects/{project_id}/prompts",
            "/api/projects/{project_id}/eval/runs",
            "/api/projects/{project_id}/eval/runs/{run_filename}",
            "/api/projects/{project_id}/eval/runs/{run_filename}/compact",
            "/api/projects/{project_id}/eval/runs/{run_filename}/cases/{case_id}",
            "/api/projects/{project_id}/prompts/{version}",
        ]
        for path in gated:
            calls = self._dependency_calls(app, path)
            assert require_project_read in calls, (
                f"{path} must depend on require_project_read"
            )

    def test_delete_run_requires_project_write(self, app):
        """The DELETE endpoint must enforce write authz, not read."""
        from app.platform.api_authz import require_project_write

        calls = self._dependency_calls(
            app,
            "/api/projects/{project_id}/eval/runs/{run_filename}",
            method="DELETE",
        )
        assert require_project_write in calls


class TestWriteThroughSidecarAPI:
    """Tests for write-through sidecar caching via API endpoints."""

    async def test_compact_endpoint_creates_sidecar_and_is_stable(self, client):
        """GET /compact twice creates sidecar on first call; both responses equal."""
        # Create project and write a run file
        create_resp = await client.post("/api/projects", json={"name": "Sidecar Test"})
        project_id = create_resp.json()["id"]

        import app.main as main_module
        output_dir = main_module.PROJECTS_DIR / project_id / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        result = [{
            "test_run_id": "model-a",
            "timestamp": "2026-06-26T00:00:00",
            "hyperparameters": {"model": "test", "endpoint": "runtime", "region": "us-east-1"},
            "results": [{
                "name": "case1",
                "success": True,
                "input": "What is 2+2?",
                "actual_output": "4",
                "expected_output": "4",
                "additional_metadata": {"id": "tc1", "latency_ms": 100},
                "metrics_data": [{"name": "ExactMatch", "success": True, "score": 1.0}],
            }],
        }]
        run_file = output_dir / "eval_results_test.json"
        run_file.write_text(json.dumps(result))

        # First GET creates sidecar
        resp1 = await client.get(f"/api/projects/{project_id}/eval/runs/eval_results_test.json/compact")
        assert resp1.status_code == 200
        data1 = resp1.json()

        # Verify sidecar exists on disk
        from app.services.eval_runner import _derived_dir
        derived = _derived_dir(run_file)
        compact_path = derived / "compact.json"
        assert compact_path.exists()

        # Second GET reads sidecar
        resp2 = await client.get(f"/api/projects/{project_id}/eval/runs/eval_results_test.json/compact")
        assert resp2.status_code == 200
        data2 = resp2.json()

        # Both responses must be equal
        assert data1 == data2

    async def test_case_detail_endpoint_404_unchanged_with_sidecars(self, client):
        """GET /cases/{unknown} still 404s; GET /cases/{known} 200 and byte-identical."""
        create_resp = await client.post("/api/projects", json={"name": "Case Test"})
        project_id = create_resp.json()["id"]

        import app.main as main_module
        output_dir = main_module.PROJECTS_DIR / project_id / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        result = [{
            "test_run_id": "model-a",
            "timestamp": "2026-06-26T00:00:00",
            "hyperparameters": {"model": "test", "endpoint": "runtime", "region": "us-east-1"},
            "results": [{
                "name": "case1",
                "success": True,
                "input": "What is 2+2?",
                "actual_output": "4",
                "expected_output": "4",
                "additional_metadata": {"id": "tc1", "latency_ms": 100},
                "metrics_data": [{"name": "ExactMatch", "success": True, "score": 1.0, "reason": "Match"}],
            }],
        }]
        run_file = output_dir / "eval_results_test.json"
        run_file.write_text(json.dumps(result))

        # Unknown case_id still 404s
        resp_unknown = await client.get(
            f"/api/projects/{project_id}/eval/runs/eval_results_test.json/cases/unknown"
        )
        assert resp_unknown.status_code == 404

        # Known case_id returns 200
        resp_known = await client.get(
            f"/api/projects/{project_id}/eval/runs/eval_results_test.json/cases/tc1"
        )
        assert resp_known.status_code == 200
        data = resp_known.json()
        assert data["input"] == "What is 2+2?"
        assert data["models"]["test"]["actual_output"] == "4"

    async def test_delete_removes_sidecar_dir(self, client):
        """DELETE removes both run file and output/_derived/<stem>/ tree."""
        create_resp = await client.post("/api/projects", json={"name": "Delete Test"})
        project_id = create_resp.json()["id"]

        import app.main as main_module
        output_dir = main_module.PROJECTS_DIR / project_id / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        result = [{
            "test_run_id": "model-a",
            "timestamp": "2026-06-26T00:00:00",
            "hyperparameters": {"model": "test", "endpoint": "runtime", "region": "us-east-1"},
            "results": [{
                "name": "case1",
                "success": True,
                "input": "What is 2+2?",
                "actual_output": "4",
                "expected_output": "4",
                "additional_metadata": {"id": "tc1"},
                "metrics_data": [],
            }],
        }]
        run_file = output_dir / "eval_results_test.json"
        run_file.write_text(json.dumps(result))

        # Populate sidecars
        await client.get(f"/api/projects/{project_id}/eval/runs/eval_results_test.json/compact")
        await client.get(f"/api/projects/{project_id}/eval/runs/eval_results_test.json/cases/tc1")

        # Verify sidecars exist
        from app.services.eval_runner import _derived_dir
        derived = _derived_dir(run_file)
        assert derived.exists()
        assert (derived / "compact.json").exists()
        assert (derived / "cases").exists()

        # DELETE the run
        resp = await client.delete(f"/api/projects/{project_id}/eval/runs/eval_results_test.json")
        assert resp.status_code == 204

        # Verify both run file and sidecar dir are gone
        assert not run_file.exists()
        assert not derived.exists()


# --- Run-file resolution ---


class TestRunFileResolution:
    """A run is identified by its FILENAME, and nothing else resolves.

    ``_resolve_run_file_or_404`` used to fall back to treating an unresolvable
    name as a bare eval-job UUID, correlating it against optimization state and
    returning the lexicographically newest file at that prompt version. It existed
    to read iterations recorded before run files were materialized, and it could
    answer a request for one iteration's results with a DIFFERENT run of the same
    version. Deleted along with the two tests that pinned it.
    """

    async def test_bare_uuid_404s_even_when_a_same_version_file_exists(self, client):
        """The replacement for "picks the newest matching file": it picks none.

        This is the shape the deleted fallback was built for — a bare identifier
        plus a real run file at that prompt version — and the correct answer is
        404, not a guess.
        """
        create_resp = await client.post("/api/projects", json={"name": "Bare UUID"})
        project_id = create_resp.json()["id"]

        import app.main as main_module
        project_dir = main_module.PROJECTS_DIR / project_id
        (project_dir / "output").mkdir(parents=True, exist_ok=True)
        (project_dir / "output" / "eval_results_v1_20260101_000000.json").write_text(
            json.dumps([{"test_run_id": "m1", "results": []}])
        )
        resp = await client.get(f"/api/projects/{project_id}/eval/runs/bare-uuid-123")
        assert resp.status_code == 404

    async def test_legacy_uuid_no_match_still_404(self, client):
        """Bare UUID with no iteration/file match still returns 404."""
        create_resp = await client.post("/api/projects", json={"name": "No Match"})
        project_id = create_resp.json()["id"]

        import app.main as main_module
        project_dir = main_module.PROJECTS_DIR / project_id

        # Write optimization state with NO matching iteration
        opt_state_dir = project_dir / "output" / "optimization"
        opt_state_dir.mkdir(parents=True, exist_ok=True)
        state_data = {
            "target_metric_scores": {"correctness": 0.9},
            "max_iterations": 10,
            "plateau_window": 3,
            "status": "running",
            "iterations": [],  # Empty
        }
        (opt_state_dir / "state.json").write_text(json.dumps(state_data))

        # Try to GET with a bare UUID that has no iteration
        resp = await client.get(f"/api/projects/{project_id}/eval/runs/no-such-uuid")
        assert resp.status_code == 404

    async def test_run_resolver_rejects_traversal(self, client):
        """Path traversal attempts still return 400 (or 404), no escape."""
        create_resp = await client.post("/api/projects", json={"name": "Traversal Test"})
        project_id = create_resp.json()["id"]

        # Try various traversal patterns
        traversal_attempts = [
            "../../etc/passwd",
            "../../../etc/hosts",
            "..%2f..%2fetc%2fpasswd",
            "output/../../../etc/passwd",
        ]

        for attempt in traversal_attempts:
            resp = await client.get(f"/api/projects/{project_id}/eval/runs/{attempt}")
            assert resp.status_code in (400, 404), f"Traversal '{attempt}' did not get rejected"

    async def test_direct_filename_still_resolves(self, client):
        """Direct eval_results_*.json filename still resolves (fast path unchanged)."""
        create_resp = await client.post("/api/projects", json={"name": "Direct Filename"})
        project_id = create_resp.json()["id"]

        import app.main as main_module
        project_dir = main_module.PROJECTS_DIR / project_id

        # Write the real eval_results file
        output_dir = project_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        result_filename = "eval_results_v2_20260718_120000.json"
        result_data = [
            {
                "source": "manual",
                "test_run_id": "test-model",
                "timestamp": "2026-07-18T12:00:00Z",
                "hyperparameters": {"model": "test-model", "prompt_version": "v2"},
                "results": [{"input": "fast path", "actual_output": "works"}],
            }
        ]
        (output_dir / result_filename).write_text(json.dumps(result_data))

        # GET with the direct filename should resolve via fast path
        resp = await client.get(f"/api/projects/{project_id}/eval/runs/{result_filename}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["test_run_id"] == "test-model"

    async def test_api_responses_carry_non_storable_cache_directive(self, client):
        """API responses declare themselves non-storable so the browser HTTP
        cache cannot serve a stale list body after SPA navigation.

        The origin's Cache-Control directive is the durable half of the
        transport contract — the SPA's request-side enforcement is primary,
        but this header makes the guarantee independent of any one client.
        """
        create_resp = await client.post("/api/projects", json={"name": "Cache Test"})
        project_id = create_resp.json()["id"]

        # Any 200 API response — using listEvalRuns as the triggering symptom
        resp = await client.get(f"/api/projects/{project_id}/eval/runs")
        assert resp.status_code == 200
        cache_control = resp.headers.get("Cache-Control", "")
        assert "no-store" in cache_control.lower()


class TestRunDeletedDuringRead:
    """A run deleted between resolve and read is a 404, not a 500.

    ``_resolve_run_file_or_404`` checks existence and then returns; the read
    happens afterwards. A concurrent delete lands in that window, so the reader
    hits a missing file. That escaped as an unhandled ``FileNotFoundError`` and
    a 500 — describing a server fault where the correct answer is "this run is
    gone". The three read routes (full, compact, case detail) all share the
    window, so all three are pinned here.

    The delete is simulated by unlinking inside a patched reader, which is the
    only way to land precisely in the window from a test.
    """

    @staticmethod
    def _seed_run(project_id: str, filename: str = "eval_results_baseline_2026-01-01_00-00-00.json"):
        """Write one minimal run file into the project's output dir."""
        import app.main as main_module
        output_dir = main_module.PROJECTS_DIR / project_id / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        run_path = output_dir / filename
        run_path.write_text(json.dumps([{
            "source": "manual",
            "test_run_id": "run-1",
            "timestamp": "2026-01-01T00:00:00",
            "hyperparameters": {
                "model": "m",
                "endpoint": "runtime",
                "region": "us-east-1",
                "prompt_version": "baseline",
            },
            "results": [{
                "name": "tc0",
                "success": True,
                "input": "x",
                "actual_output": "y",
                "expected_output": "y",
                "additional_metadata": {"id": "known"},
                "metrics_data": [],
            }],
        }]))
        return run_path

    @pytest.fixture
    def delete_at_read(self, monkeypatch):
        """Patch the shared reader to unlink the file, then read it as normal.

        Reproduces the race exactly: the route's existence pre-check has already
        passed by the time this runs, so the read sees a file that no longer
        exists and the real production error path executes.
        """
        import app.services.eval_runner as runner_module

        original = runner_module._read_run_json

        def _unlink_then_read(results_path):
            results_path.unlink(missing_ok=True)
            return original(results_path)

        monkeypatch.setattr(runner_module, "_read_run_json", _unlink_then_read)

    async def test_get_run_404_when_deleted_during_read(self, client, delete_at_read):
        create_resp = await client.post("/api/projects", json={"name": "Race Full"})
        project_id = create_resp.json()["id"]
        run_path = self._seed_run(project_id)

        resp = await client.get(f"/api/projects/{project_id}/eval/runs/{run_path.name}")

        assert resp.status_code == 404, f"expected 404, got {resp.status_code}"
        assert resp.json()["detail"] == f"Run not found: {run_path.name}"

    async def test_get_compact_404_when_deleted_during_read(self, client, delete_at_read):
        create_resp = await client.post("/api/projects", json={"name": "Race Compact"})
        project_id = create_resp.json()["id"]
        run_path = self._seed_run(project_id)

        resp = await client.get(
            f"/api/projects/{project_id}/eval/runs/{run_path.name}/compact"
        )

        assert resp.status_code == 404
        assert resp.json()["detail"] == f"Run not found: {run_path.name}"

    async def test_get_case_detail_404_when_deleted_during_read(self, client, delete_at_read):
        create_resp = await client.post("/api/projects", json={"name": "Race Case"})
        project_id = create_resp.json()["id"]
        run_path = self._seed_run(project_id)

        resp = await client.get(
            f"/api/projects/{project_id}/eval/runs/{run_path.name}/cases/known"
        )

        assert resp.status_code == 404
        assert resp.json()["detail"] == f"Run not found: {run_path.name}"

    async def test_corrupt_run_file_is_not_mapped_to_404(self, client):
        """A corrupt file stays a 500 — only absence means "not found".

        The guard is deliberately narrow. Funneling every IO failure into a 404
        would report unreadable-but-present data as a missing run, hiding real
        faults behind an error the client treats as routine.
        """
        create_resp = await client.post("/api/projects", json={"name": "Corrupt"})
        project_id = create_resp.json()["id"]
        run_path = self._seed_run(project_id)
        run_path.write_text("{ this is not valid json")

        with pytest.raises(json.JSONDecodeError):
            await client.get(f"/api/projects/{project_id}/eval/runs/{run_path.name}")

    async def test_unreadable_run_file_is_not_mapped_to_404(self, client):
        """A permission error stays a 500 for the same reason as a corrupt file."""
        import os
        create_resp = await client.post("/api/projects", json={"name": "Unreadable"})
        project_id = create_resp.json()["id"]
        run_path = self._seed_run(project_id)
        os.chmod(run_path, 0o000)
        if os.access(run_path, os.R_OK):
            pytest.skip("running as a user that bypasses file permissions (e.g. root)")
        try:
            with pytest.raises(PermissionError):
                await client.get(f"/api/projects/{project_id}/eval/runs/{run_path.name}")
        finally:
            os.chmod(run_path, 0o644)


@pytest.mark.parametrize("source", list(BROKEN_CONFIGS.values()), ids=list(BROKEN_CONFIGS))
async def test_eval_runs_list_survives_an_unreadable_config(client, source):
    """The run list ranks runs against config.py targets; a broken file means no ranking, not a 500."""
    import app.main as main_module

    project_id = (await client.post("/api/projects", json={"name": "Broken"})).json()["id"]
    src = main_module.PROJECTS_DIR / project_id / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "config.py").write_text(source)

    resp = await client.get(f"/api/projects/{project_id}/eval/runs")
    assert resp.status_code == 200, resp.text
