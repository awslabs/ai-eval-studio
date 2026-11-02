# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the files API: harness download + disabled source editing.

Covers the ``GET /harness/download`` zip stream (inclusion of runnable
content, exclusion of regenerable cruft) and the ``PUT /files/...`` guard
that rejects direct source edits with 403.
"""
from __future__ import annotations

import io
import zipfile

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
def app(tmp_path):
    import app.main as main_module

    original = main_module.PROJECTS_DIR
    original_pm = main_module._project_manager
    main_module.PROJECTS_DIR = tmp_path / "projects"
    main_module.PROJECTS_DIR.mkdir()
    main_module._project_manager = None
    application = create_app()
    yield application
    main_module.PROJECTS_DIR = original
    main_module._project_manager = original_pm


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _project_dir(app, project_id: str):
    import app.main as main_module

    return main_module.PROJECTS_DIR / project_id


async def _create_project(client) -> str:
    resp = await client.post("/api/projects", json={"name": "Files Test"})
    return resp.json()["id"]


async def test_download_harness_includes_runnable_files_and_excludes_cruft(
    client, app
):
    """The zip carries runnable content and drops regenerable cruft."""
    project_id = await _create_project(client)
    proj = _project_dir(app, project_id)

    # Runnable content that MUST be in the zip.
    (proj / "run_eval.py").write_text("print('hi')\n")
    (proj / "src").mkdir(exist_ok=True)
    (proj / "src" / "config.py").write_text("DATA = 1\n")
    (proj / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (proj / "data" / "raw" / "cases.csv").write_text("a,b\n1,2\n")

    # Cruft that MUST be excluded.
    (proj / ".venv").mkdir(exist_ok=True)
    (proj / ".venv" / "marker").write_text("x")
    (proj / "src" / "__pycache__").mkdir(parents=True, exist_ok=True)
    (proj / "src" / "__pycache__" / "config.cpython-311.pyc").write_text("x")
    (proj / "src" / "stale.pyc").write_text("x")
    # App-internal state that MUST NOT ship: chat transcript + deepeval cache.
    (proj / "chat").mkdir(exist_ok=True)
    (proj / "chat" / "transcript.jsonl").write_text('{"role":"user"}\n')
    (proj / ".deepeval").mkdir(exist_ok=True)
    (proj / ".deepeval" / ".deepeval-cache.json").write_text("{}")

    resp = await client.get(f"/api/projects/{project_id}/harness/download")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert "attachment" in resp.headers["content-disposition"]

    names = set(zipfile.ZipFile(io.BytesIO(resp.content)).namelist())
    assert "run_eval.py" in names
    assert "src/config.py" in names
    assert "data/raw/cases.csv" in names
    # Cruft excluded.
    assert not any(n.startswith(".venv/") for n in names)
    assert not any("__pycache__" in n for n in names)
    assert not any(n.endswith(".pyc") for n in names)
    # Internal state excluded: a downloaded harness must never carry the
    # user's chat transcript or the deepeval result cache.
    assert not any(n.startswith("chat/") for n in names)
    assert not any(n.startswith(".deepeval/") for n in names)


async def test_file_tree_hides_chat_dir(client, app):
    """The file browser must not surface the app-internal chat/ folder."""
    project_id = await _create_project(client)
    proj = _project_dir(app, project_id)
    (proj / "chat").mkdir(exist_ok=True)
    (proj / "chat" / "transcript.jsonl").write_text('{"role":"user"}\n')
    (proj / "docs").mkdir(exist_ok=True)
    (proj / "docs" / "guide.md").write_text("# hi\n")

    resp = await client.get(f"/api/projects/{project_id}/files")
    assert resp.status_code == 200
    top_level = {node["name"] for node in resp.json()["tree"]}
    assert "chat" not in top_level
    assert "docs" in top_level


async def test_download_harness_404_for_unknown_project(client):
    """Unknown project id yields 404, not an empty zip."""
    resp = await client.get(
        "/api/projects/00000000-0000-0000-0000-000000000000/harness/download"
    )
    assert resp.status_code == 404


async def test_save_file_is_disabled(client, app):
    """Direct source edits are rejected with 403 and the chat hint."""
    project_id = await _create_project(client)
    (_project_dir(app, project_id) / "src").mkdir(exist_ok=True)

    resp = await client.put(
        f"/api/projects/{project_id}/files/src/config.py",
        json={"content": "HACKED = True\n"},
    )
    assert resp.status_code == 403
    assert "ask in chat" in resp.json()["detail"].lower()

    # The file on disk must be untouched (never created by the write).
    assert not (_project_dir(app, project_id) / "src" / "config.py").exists()


async def test_harness_config_not_configured_is_200_null(client, app):
    """A project with no src/config.py returns 200 + JSON null, NOT 404.

    Not-configured-yet is a normal state for a new project. The SPA fetches
    this endpoint on every page load, and a 404 put a browser network-error
    console line on every refresh of every fresh project (browsers log
    non-2xx fetches unconditionally — unsuppressible from JS)."""
    project_id = await _create_project(client)

    resp = await client.get(f"/api/projects/{project_id}/harness/config")

    assert resp.status_code == 200
    assert resp.json() is None


async def test_harness_config_parses_when_present(client, app):
    """With src/config.py present the endpoint returns the parsed config."""
    project_id = await _create_project(client)
    proj = _project_dir(app, project_id)
    (proj / "src").mkdir(exist_ok=True)
    (proj / "src" / "config.py").write_text(
        'models = ["m1"]\n'
        'data_file = "cases.csv"\n'
    )

    resp = await client.get(f"/api/projects/{project_id}/harness/config")

    assert resp.status_code == 200
    body = resp.json()
    assert body is not None
    assert body["models"] == ["m1"]


#: A config.py shaped like the ones the harness builder writes: a dataclass with
#: ``field(default_factory=...)`` values, metrics named only as gating targets,
#: and NO ``metrics`` field — a harness defines its metrics as code in
#: src/metrics/metrics.py. A synthetic ``metrics = [...]`` line once let a
#: readiness rule pass its test while failing every real harness.
_BUILDER_SHAPED_CONFIG = '''\
from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    models: list[str] = field(default_factory=lambda: {models})
    data_file: str = {data_file}
    prompt_version: str = "baseline"
    field_mappings: dict[str, str] = field(default_factory=lambda: {{
        "input": "question",
        "expected_output": "answer",
    }})
    target_metric_scores: dict[str, float] = field(default_factory=lambda: {{
        "Answer Correctness": 0.9,
    }})
    max_iterations: int = 10
    plateau_window: int = 3
'''


def _write_builder_config(proj, models='["m1"]', data_file='"cases.csv"'):
    (proj / "src").mkdir(exist_ok=True)
    (proj / "src" / "config.py").write_text(
        _BUILDER_SHAPED_CONFIG.format(models=models, data_file=data_file)
    )


async def test_harness_config_ready_requires_runnable_harness(client, app):
    """``ready`` means dispatchable, not merely parsed.

    config.py lands with the scaffold minutes before the baseline prompt does;
    the SPA's Start Optimization and Start New buttons gate on this field, and
    gating on parse success alone lit the button mid-build (measured live).
    """
    project_id = await _create_project(client)
    proj = _project_dir(app, project_id)
    _write_builder_config(proj)
    resp = await client.get(f"/api/projects/{project_id}/harness/config")
    assert resp.status_code == 200
    assert resp.json()["ready"] is False, "prompt file missing → not ready"

    # Baseline prompt lands → ready, with no ``metrics`` field anywhere.
    (proj / "prompts").mkdir(exist_ok=True)
    (proj / "prompts" / "baseline.md").write_text("# system prompt")
    resp = await client.get(f"/api/projects/{project_id}/harness/config")
    body = resp.json()
    assert "metrics" not in body, "metrics are code in src/metrics/metrics.py"
    assert body["ready"] is True, (
        "a builder-shaped harness with its prompt in place must be ready; requiring "
        "a config.py metrics field hid Start New after every converged loop"
    )


@pytest.mark.parametrize(("models", "data_file"), [("[]", '"cases.csv"'), ('["m1"]', '""')])
async def test_harness_config_not_ready_without_models_or_data(client, app, models, data_file):
    project_id = await _create_project(client)
    proj = _project_dir(app, project_id)
    _write_builder_config(proj, models=models, data_file=data_file)
    (proj / "prompts").mkdir(exist_ok=True)
    (proj / "prompts" / "baseline.md").write_text("# system prompt")
    resp = await client.get(f"/api/projects/{project_id}/harness/config")
    assert resp.json()["ready"] is False


async def test_harness_config_ignores_a_metric_list_in_config_py(client, app):
    """The config.py from a live failure, with ``metrics`` as a list of names.

    The platform used to read that field, rejected the shape and answered 500.
    It no longer reads it, so the file loads like any other.
    """
    project_id = await _create_project(client)
    proj = _project_dir(app, project_id)
    _write_builder_config(proj)
    with open(proj / "src" / "config.py", "a") as f:
        f.write('    metrics: list[str] = field(default_factory=lambda: ["Decision Correctness"])\n')

    resp = await client.get(f"/api/projects/{project_id}/harness/config")

    assert resp.status_code == 200
    assert resp.json()["models"] == ["m1"]


async def test_harness_config_with_a_wrongly_shaped_value_is_400(client, app):
    """A field whose value has the wrong shape is reported, not a 500."""
    project_id = await _create_project(client)
    proj = _project_dir(app, project_id)
    (proj / "src").mkdir(exist_ok=True)
    (proj / "src" / "config.py").write_text("models = [1, 2]\n")

    resp = await client.get(f"/api/projects/{project_id}/harness/config")

    assert resp.status_code == 400, resp.text
    assert "models" in resp.json()["detail"]


async def test_harness_config_syntax_error_is_400(client, app):
    """A config.py that doesn't parse is a real error — stays 400."""
    project_id = await _create_project(client)
    proj = _project_dir(app, project_id)
    (proj / "src").mkdir(exist_ok=True)
    (proj / "src" / "config.py").write_text("MODELS = [unterminated\n")

    resp = await client.get(f"/api/projects/{project_id}/harness/config")

    assert resp.status_code == 400
