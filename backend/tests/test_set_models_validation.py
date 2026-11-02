# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``set_models`` MCP tool.

``set_models`` is the load-bearing server-side validation gate that
replaces the agent prompt's "Write src/config.py directly" instruction.
It validates each id against the runtime catalog OR the fully-resolved
mantle descriptor map, and rewrites the project's ``src/config.py``
``models`` and ``judge_model_id`` fields ONLY — preserving every other
EvalConfig field byte-for-byte.

Persistence uses the same atomic ``.tmp`` + ``os.replace`` pattern as
the descriptor writer, under the per-project lock from
``services.project_locks.get_project_lock``.
``ast.unparse`` is deterministic, so running ``set_models`` twice with
the same args produces byte-identical output.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from app.services.bedrock_discovery import DiscoveryResult
from app.services.harness_config import load_harness_config, parse_harness_config
from app.services.mcp_tools.harness import (
    _rewrite_config_models_and_judge,
    set_models,
)


# A minimal EvalConfig that exercises field(default_factory=lambda: [...])
# AND simple-literal annotated assignments — both forms `_extract_literal`
# in ``services.harness_config`` understands. The set_models rewriter must
# not perturb any of the other fields.
_MINIMAL_CONFIG_PY = '''"""Test EvalConfig."""

from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    """Test config."""

    models: list[str] = field(default_factory=lambda: ["legacy.id-1"])
    prompt_version: str = "v3"
    data_file: str = "x.csv"
    judge_model_id: str = "legacy.judge"
    temperature: float = 0.7
    max_tokens: int = 2048
'''


def _scaffold_minimal_config(project_dir: Path) -> Path:
    """Drop a minimal src/config.py into ``project_dir`` and return its path."""
    src = project_dir / "src"
    src.mkdir(parents=True, exist_ok=True)
    cfg = src / "config.py"
    cfg.write_text(_MINIMAL_CONFIG_PY, encoding="utf-8")
    return cfg


def _stub_discover(
    runtime_ids: list[str],
    descriptors: dict | None = None,
):
    """Return a coroutine that fakes ``discover_models``.

    Catalog carries one runtime entry per id in ``runtime_ids`` (so the
    server-side validator counts them as known). Mantle descriptor entries
    come from ``descriptors``; the catalog also carries a mantle entry per
    descriptor key so listing-style lookups succeed.
    """
    descriptors = dict(descriptors or {})

    catalog = [
        {
            "model_id": mid,
            "model_name": mid,
            "provider": "test",
            "endpoint": "runtime",
            "api": "converse",
            "region": None,
        }
        for mid in runtime_ids
    ]
    for mid, desc in descriptors.items():
        catalog.append(
            {
                "model_id": mid,
                "model_name": mid,
                "provider": mid.split(".", 1)[0],
                "endpoint": "mantle",
                "api": desc["api"],
                "region": desc["region"],
            }
        )

    async def _fake_discover(project_id: str, project_dir: Path) -> DiscoveryResult:
        return DiscoveryResult(catalog=catalog, descriptors=descriptors)

    return _fake_discover


@pytest.fixture
def project_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test project root with ``resolve_project_dir`` redirected here."""
    pd = tmp_path / "p1"
    pd.mkdir()

    def _fake_resolve(pid: str, base_dir: Path | None = None) -> Path:  # noqa: ARG001
        return pd

    monkeypatch.setattr(
        "app.services.mcp_tools.harness.resolve_project_dir",
        _fake_resolve,
        raising=False,
    )
    return pd


@pytest.fixture(autouse=True)
def _reset_project_locks():
    """Clear the project-lock registry between tests."""
    from app.services import project_locks

    project_locks._reset()
    yield
    project_locks._reset()


# --------------------------------------------------------------------------- #
# Pure helper tests — the AST rewrite is testable without project plumbing.   #
# --------------------------------------------------------------------------- #


def test_rewrite_helper_replaces_models_and_judge_only() -> None:
    """The pure helper changes ONLY the two target fields."""
    out = _rewrite_config_models_and_judge(
        _MINIMAL_CONFIG_PY,
        ["new.target-a", "new.target-b"],
        "new.judge",
    )
    parsed = parse_harness_config(out)
    assert parsed.models == ["new.target-a", "new.target-b"]
    assert parsed.judge_model_id == "new.judge"
    # Other fields preserved.
    assert parsed.prompt_version == "v3"
    assert parsed.data_file == "x.csv"


def test_rewrite_helper_is_idempotent() -> None:
    """Same args twice → byte-identical output."""
    out1 = _rewrite_config_models_and_judge(
        _MINIMAL_CONFIG_PY, ["x.a"], "x.judge"
    )
    out2 = _rewrite_config_models_and_judge(
        out1, ["x.a"], "x.judge"
    )
    assert out1 == out2


def test_rewrite_helper_raises_when_class_missing() -> None:
    """Source without an EvalConfig class → ValueError (caller wraps as mcp_error)."""
    src = "x = 1\n"
    with pytest.raises(ValueError, match="EvalConfig"):
        _rewrite_config_models_and_judge(src, ["a"], "judge")


# --------------------------------------------------------------------------- #
# Tool-handler behavioral tests                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_known_runtime_ids_succeed_writes_config(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path
) -> None:
    """Two runtime ids + runtime judge → success; src/config.py reflects them."""
    cfg = _scaffold_minimal_config(project_dir)
    monkeypatch.setattr(
        "app.services.bedrock_discovery.discover_models",
        _stub_discover(["rt.alpha", "rt.beta", "rt.judge"]),
        raising=False,
    )

    result = await set_models.handler(
        {
            "project_id": "p1",
            "model_ids": ["rt.alpha", "rt.beta"],
            "judge_model_id": "rt.judge",
        }
    )
    assert "is_error" not in result, result["content"][0]["text"]

    parsed = parse_harness_config(cfg.read_text(encoding="utf-8"))
    assert parsed.models == ["rt.alpha", "rt.beta"]
    assert parsed.judge_model_id == "rt.judge"
    # Sibling fields preserved.
    assert parsed.prompt_version == "v3"
    assert parsed.data_file == "x.csv"


@pytest.mark.asyncio
async def test_known_mantle_descriptor_id_succeeds(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path
) -> None:
    """Mantle id with a fully-resolved descriptor passes validation."""
    _scaffold_minimal_config(project_dir)
    monkeypatch.setattr(
        "app.services.bedrock_discovery.discover_models",
        _stub_discover(
            ["rt.judge"],
            descriptors={
                "openai.gpt-5.5": {
                    "endpoint": "mantle",
                    "api": "responses",
                    "region": "us-east-1",
                }
            },
        ),
        raising=False,
    )

    result = await set_models.handler(
        {
            "project_id": "p1",
            "model_ids": ["openai.gpt-5.5"],
            "judge_model_id": "rt.judge",
        }
    )
    assert "is_error" not in result, result["content"][0]["text"]
    parsed = load_harness_config(project_dir)
    assert parsed is not None
    assert parsed.models == ["openai.gpt-5.5"]


@pytest.mark.asyncio
async def test_unknown_id_returns_error(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path
) -> None:
    """Id not in runtime catalog and not in descriptor map → mcp_error."""
    _scaffold_minimal_config(project_dir)
    monkeypatch.setattr(
        "app.services.bedrock_discovery.discover_models",
        _stub_discover(["rt.alpha"]),
        raising=False,
    )

    result = await set_models.handler(
        {
            "project_id": "p1",
            "model_ids": ["rt.alpha", "garbage.id"],
            "judge_model_id": "rt.alpha",
        }
    )
    assert result.get("is_error") is True
    assert "garbage.id" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_multiple_bad_ids_listed(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path
) -> None:
    """All three bad ids appear in the error message (developer ergonomics)."""
    _scaffold_minimal_config(project_dir)
    monkeypatch.setattr(
        "app.services.bedrock_discovery.discover_models",
        _stub_discover(["rt.alpha"]),
        raising=False,
    )

    result = await set_models.handler(
        {
            "project_id": "p1",
            "model_ids": ["bad.a", "bad.b"],
            "judge_model_id": "bad.c",
        }
    )
    assert result.get("is_error") is True
    msg = result["content"][0]["text"]
    for bad in ("bad.a", "bad.b", "bad.c"):
        assert bad in msg


@pytest.mark.asyncio
async def test_empty_model_ids_rejected(project_dir: Path) -> None:
    """Empty model_ids list → mcp_error (no discover call needed)."""
    _scaffold_minimal_config(project_dir)
    result = await set_models.handler(
        {"project_id": "p1", "model_ids": [], "judge_model_id": "any.judge"}
    )
    assert result.get("is_error") is True
    assert "model_ids" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_empty_judge_rejected(project_dir: Path) -> None:
    """Empty judge id → mcp_error."""
    _scaffold_minimal_config(project_dir)
    result = await set_models.handler(
        {"project_id": "p1", "model_ids": ["a"], "judge_model_id": ""}
    )
    assert result.get("is_error") is True
    assert "judge_model_id" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_missing_project_id_rejected() -> None:
    """No project_id → mcp_error."""
    result = await set_models.handler(
        {"model_ids": ["a"], "judge_model_id": "j"}
    )
    assert result.get("is_error") is True


@pytest.mark.asyncio
async def test_missing_config_py_rejected(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path
) -> None:
    """No src/config.py (scaffold not run) → mcp_error pointing at scaffold_harness."""
    monkeypatch.setattr(
        "app.services.bedrock_discovery.discover_models",
        _stub_discover(["rt.alpha"]),
        raising=False,
    )
    result = await set_models.handler(
        {
            "project_id": "p1",
            "model_ids": ["rt.alpha"],
            "judge_model_id": "rt.alpha",
        }
    )
    assert result.get("is_error") is True
    assert "scaffold_harness" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_idempotent(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path
) -> None:
    """Running set_models twice with the same args yields byte-identical config.py."""
    cfg = _scaffold_minimal_config(project_dir)
    monkeypatch.setattr(
        "app.services.bedrock_discovery.discover_models",
        _stub_discover(["rt.alpha", "rt.beta", "rt.judge"]),
        raising=False,
    )

    args = {
        "project_id": "p1",
        "model_ids": ["rt.alpha", "rt.beta"],
        "judge_model_id": "rt.judge",
    }
    await set_models.handler(args)
    first = cfg.read_bytes()
    await set_models.handler(args)
    second = cfg.read_bytes()
    assert first == second


@pytest.mark.asyncio
async def test_concurrent_set_models_serialized(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path
) -> None:
    """Two concurrent set_models tasks for one project produce a consistent file."""
    cfg = _scaffold_minimal_config(project_dir)
    monkeypatch.setattr(
        "app.services.bedrock_discovery.discover_models",
        _stub_discover(["rt.alpha", "rt.beta", "rt.judge"]),
        raising=False,
    )

    args = {
        "project_id": "p1",
        "model_ids": ["rt.alpha", "rt.beta"],
        "judge_model_id": "rt.judge",
    }
    results = await asyncio.gather(
        set_models.handler(args),
        set_models.handler(args),
    )
    assert all("is_error" not in r for r in results), results

    parsed = parse_harness_config(cfg.read_text(encoding="utf-8"))
    assert parsed.models == ["rt.alpha", "rt.beta"]
    assert parsed.judge_model_id == "rt.judge"


@pytest.mark.asyncio
async def test_preserves_other_eval_config_fields(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path
) -> None:
    """Sibling EvalConfig fields (data_file, prompt_version, temperature) untouched."""
    cfg = _scaffold_minimal_config(project_dir)
    before = parse_harness_config(cfg.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        "app.services.bedrock_discovery.discover_models",
        _stub_discover(["rt.alpha", "rt.judge"]),
        raising=False,
    )

    await set_models.handler(
        {
            "project_id": "p1",
            "model_ids": ["rt.alpha"],
            "judge_model_id": "rt.judge",
        }
    )
    after = parse_harness_config(cfg.read_text(encoding="utf-8"))
    assert after.data_file == before.data_file
    assert after.prompt_version == before.prompt_version


@pytest.mark.asyncio
async def test_post_write_parseability_via_load_harness_config(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path
) -> None:
    """End-to-end — after set_models, load_harness_config returns the new ids."""
    _scaffold_minimal_config(project_dir)
    monkeypatch.setattr(
        "app.services.bedrock_discovery.discover_models",
        _stub_discover(["rt.alpha", "rt.beta", "rt.judge"]),
        raising=False,
    )

    await set_models.handler(
        {
            "project_id": "p1",
            "model_ids": ["rt.alpha", "rt.beta"],
            "judge_model_id": "rt.judge",
        }
    )
    parsed = load_harness_config(project_dir)
    assert parsed is not None
    assert parsed.models == ["rt.alpha", "rt.beta"]
    assert parsed.judge_model_id == "rt.judge"


@pytest.mark.asyncio
async def test_model_ids_must_be_list(project_dir: Path) -> None:
    """model_ids must be a list — pass a string and we reject it."""
    _scaffold_minimal_config(project_dir)
    result = await set_models.handler(
        {
            "project_id": "p1",
            "model_ids": "not a list",  # type: ignore[arg-type]
            "judge_model_id": "j",
        }
    )
    assert result.get("is_error") is True
    assert "model_ids" in result["content"][0]["text"]


# --------------------------------------------------------------------------- #
# Running-loop fence.                                                          #
#                                                                              #
# Both convergence rules compare models ACROSS iterations, so a mid-run lineup  #
# change makes them lie: a model can be named the winner on iterations it never #
# ran, and a model added late has no history to flatline, silently blocking the #
# plateau verdict. The workload fingerprint detects the change afterwards, but  #
# cannot reconstruct the comparison the run was making.                         #
# --------------------------------------------------------------------------- #


def _stub_loop_status(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path, status: str
) -> None:
    """Make ``opt_repo.read`` report a real loop in ``status`` for any project.

    Builds a genuine ``OptimizationState`` inside a genuine handle rather than a
    mock, so the fence is exercised through the real ``is_live()`` predicate.
    Faking ``status`` directly would let the test pass while production used a
    different definition of "running" — the exact bug ``is_live`` was extracted
    to prevent.
    """
    from app.services import optimization_state_repository as opt_repo
    from app.services.optimization_state import OptimizationState

    state = OptimizationState(
        project_dir=project_dir,
        target_metric_scores={},
        max_iterations=3,
        plateau_window=2,
        iterations=[],
        status=status,
    )

    async def _fake_read(*_args: Any, **_kwargs: Any) -> Any:
        return opt_repo.OptimizationStateHandle(
            state=state,
            version=None,
            arch_split=False,
            owner_id=None,
            project_id="p1",
            project_dir=project_dir,
        )

    monkeypatch.setattr(opt_repo, "read", _fake_read, raising=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["running", "stopping"])
async def test_set_models_is_refused_while_a_loop_is_live(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path, status: str
) -> None:
    """Both live statuses, because checking only ``running`` misses a stopping loop."""
    cfg = _scaffold_minimal_config(project_dir)
    before = cfg.read_bytes()
    _stub_loop_status(monkeypatch, project_dir, status)

    result = await set_models.handler(
        {
            "project_id": "p1",
            "model_ids": ["rt.alpha"],
            "judge_model_id": "rt.judge",
        }
    )
    assert result.get("is_error") is True
    text = result["content"][0]["text"]
    assert "Optimization loop is running" in text
    assert "Cancel the loop" in text, "the agent needs to be told the way out"
    assert cfg.read_bytes() == before, "a refused set_models must not touch config.py"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["converged", "plateau", "max_iterations", "cancelled"])
async def test_a_finished_loop_does_not_fence(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path, status: str
) -> None:
    """The fence is about a run in flight, not about having ever run one."""
    cfg = _scaffold_minimal_config(project_dir)
    _stub_loop_status(monkeypatch, project_dir, status)
    monkeypatch.setattr(
        "app.services.bedrock_discovery.discover_models",
        _stub_discover(["rt.alpha", "rt.judge"]),
        raising=False,
    )

    result = await set_models.handler(
        {
            "project_id": "p1",
            "model_ids": ["rt.alpha"],
            "judge_model_id": "rt.judge",
        }
    )
    assert "is_error" not in result, result["content"][0]["text"]
    assert parse_harness_config(cfg.read_text(encoding="utf-8")).models == ["rt.alpha"]


@pytest.mark.asyncio
async def test_the_fence_trips_before_discovery_is_called(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path
) -> None:
    """Discovery hits the network and is the expensive part of this tool."""
    _scaffold_minimal_config(project_dir)
    _stub_loop_status(monkeypatch, project_dir, "running")

    called = False

    async def _explode(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("discovery ran despite the loop fence")

    monkeypatch.setattr(
        "app.services.bedrock_discovery.discover_models", _explode, raising=False
    )

    result = await set_models.handler(
        {
            "project_id": "p1",
            "model_ids": ["rt.alpha"],
            "judge_model_id": "rt.judge",
        }
    )
    assert result.get("is_error") is True
    assert not called


def test_the_tool_description_warns_the_agent_about_the_fence() -> None:
    """The agent reads the description, not this file — an undocumented refusal
    just looks like a broken tool to it.

    Asserts the description states a REFUSAL and names the way out, not merely
    that the words "optimization loop" appear somewhere: a description saying
    set_models is *allowed* during a loop contains those words too, and would
    contradict the code while this test stayed green.
    """
    description = set_models.description.lower()
    assert "optimization loop" in description
    assert "refused" in description, (
        "the description must tell the agent the call is refused during a loop, "
        f"not just mention loops: {set_models.description!r}"
    )
    assert "cancel" in description, "the agent needs to be told the way out"
