# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``list_bedrock_models`` MCP tool.

The @tool body delegates to
``services.bedrock_discovery.discover_models``, so every entry carries
``endpoint``/``api``/``region`` from the merged runtime + mantle catalog,
and ``project_id`` is a required arg.

Pricing enrichment attaches LiteLLM per-1M prices where they exist and the
literal ``"n/a"`` for an in-allowlist mantle id LiteLLM has no data for.
The cheapest-first sort ranks priced entries first, then unknown-priced
ones, then ``"n/a"``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.services.bedrock_discovery import DiscoveryResult
from app.services.mcp_tools.models import list_bedrock_models


def _entry(
    *,
    model_id: str,
    model_name: str | None = None,
    provider: str = "test",
    endpoint: str = "runtime",
    api: str | None = "converse",
    region: str | None = None,
) -> dict[str, Any]:
    """Build a CatalogEntry-shaped dict for stub returns."""
    return {
        "model_id": model_id,
        "model_name": model_name or model_id,
        "provider": provider,
        "endpoint": endpoint,
        "api": api,
        "region": region,
    }


def _stub_discover(
    catalog: list[dict[str, Any]],
    descriptors: dict | None = None,
    excluded: dict[str, str] | None = None,
):
    """Return an awaitable that mimics the ``discover_models`` signature."""
    async def _fake_discover(project_id: str, project_dir: Path) -> DiscoveryResult:
        return DiscoveryResult(
            catalog=list(catalog),
            descriptors=dict(descriptors or {}),
            excluded=dict(excluded or {}),
        )

    return _fake_discover


def _patch_discover(monkeypatch: pytest.MonkeyPatch, fake) -> None:
    """Patch ``discover_models`` at the source module since the MCP tool imports it lazily."""
    monkeypatch.setattr(
        "app.services.bedrock_discovery.discover_models", fake, raising=False
    )


def _decode_json_response(result: dict[str, Any]) -> Any:
    """Pull the JSON list out of an ``mcp_json`` response body."""
    assert "content" in result
    text = result["content"][0]["text"]
    return json.loads(text)


@pytest.fixture
def project_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A clean per-test project root, with ``resolve_project_dir`` redirected to it."""
    pd = tmp_path / "p1"
    pd.mkdir()

    def _fake_resolve(pid: str, base_dir: Path | None = None) -> Path:  # noqa: ARG001
        return pd

    monkeypatch.setattr(
        "app.services.mcp_tools.models.resolve_project_dir",
        _fake_resolve,
        raising=False,
    )
    return pd


@pytest.fixture(autouse=True)
def _stub_pricing(monkeypatch: pytest.MonkeyPatch):
    """Default: pricing fetch returns nothing — the per-test stub overrides as needed."""
    async def _no_pricing(_ids: list[str]) -> dict[str, Any]:
        return {"models": {}}

    monkeypatch.setattr(
        "app.services.mcp_tools.models.fetch_model_pricing",
        _no_pricing,
    )


@pytest.mark.asyncio
async def test_returns_endpoint_api_region_per_entry(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path
) -> None:
    """Every returned entry carries the 6 keys; runtime + mantle tagged distinctly."""
    catalog = [
        _entry(
            model_id="us.anthropic.claude-haiku-4-5",
            model_name="Claude Haiku 4.5",
            provider="anthropic",
            endpoint="runtime",
            api="converse",
            region=None,
        ),
        _entry(
            model_id="us.amazon.nova-2-lite-v1:0",
            model_name="Nova 2 Lite",
            provider="amazon",
            endpoint="runtime",
            api="converse",
            region=None,
        ),
        _entry(
            model_id="openai.gpt-5.5",
            model_name="openai.gpt-5.5",
            provider="openai",
            endpoint="mantle",
            api="responses",
            region="us-east-1",
        ),
        _entry(
            model_id="xai.grok-4",
            model_name="xai.grok-4",
            provider="xai",
            endpoint="mantle",
            api="responses",
            region="us-west-2",
        ),
    ]
    monkeypatch.setattr(
        "app.services.bedrock_discovery.discover_models",
        _stub_discover(catalog),
        raising=False,
    )

    result = await list_bedrock_models.handler({"project_id": "p1"})
    assert "is_error" not in result, result

    items = _decode_json_response(result)
    assert len(items) == 4
    for item in items:
        for key in ("model_id", "model_name", "provider", "endpoint", "api", "region"):
            assert key in item, f"missing {key} in {item}"

    runtime_items = [i for i in items if i["endpoint"] == "runtime"]
    assert all(i["api"] == "converse" for i in runtime_items)
    assert all(i["region"] is None for i in runtime_items)

    mantle_items = [i for i in items if i["endpoint"] == "mantle"]
    assert {i["model_id"] for i in mantle_items} == {"openai.gpt-5.5", "xai.grok-4"}
    assert all(isinstance(i["region"], str) for i in mantle_items)
    assert all(
        i["api"] in {"responses", "chat_completions", "anthropic_messages"}
        for i in mantle_items
    )


@pytest.mark.asyncio
async def test_missing_project_id_returns_error() -> None:
    """No ``project_id`` → mcp_error by contract (no silent fallback)."""
    result = await list_bedrock_models.handler({})
    assert result.get("is_error") is True
    assert "project_id is required" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_empty_project_id_returns_error() -> None:
    """Empty string is treated like missing — no silent fallback."""
    result = await list_bedrock_models.handler({"project_id": ""})
    assert result.get("is_error") is True
    assert "project_id is required" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_provider_filter_works_across_runtime_and_mantle(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path
) -> None:
    """``provider`` filter is a case-insensitive substring match on the provider field."""
    catalog = [
        _entry(model_id="us.anthropic.claude-haiku-4-5", provider="anthropic"),
        _entry(
            model_id="openai.gpt-5.5",
            provider="openai",
            endpoint="mantle",
            api="responses",
            region="us-east-1",
        ),
        _entry(
            model_id="xai.grok-4",
            provider="xai",
            endpoint="mantle",
            api="responses",
            region="us-east-1",
        ),
    ]
    monkeypatch.setattr(
        "app.services.bedrock_discovery.discover_models",
        _stub_discover(catalog),
        raising=False,
    )

    result = await list_bedrock_models.handler(
        {"project_id": "p1", "provider": "OpenAI"}
    )
    items = _decode_json_response(result)
    assert {i["model_id"] for i in items} == {"openai.gpt-5.5"}


@pytest.mark.asyncio
async def test_mantle_ids_without_pricing_sink_to_end(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path
) -> None:
    """Mantle ids without LiteLLM pricing fall to the end ('unknown→inf' semantics)."""
    catalog = [
        _entry(
            model_id="openai.gpt-5.5",
            provider="openai",
            endpoint="mantle",
            api="responses",
            region="us-east-1",
        ),
        _entry(model_id="us.amazon.nova-2-lite-v1:0", provider="amazon"),
        _entry(model_id="us.anthropic.claude-haiku-4-5", provider="anthropic"),
    ]
    monkeypatch.setattr(
        "app.services.bedrock_discovery.discover_models",
        _stub_discover(catalog),
        raising=False,
    )

    async def _priced(ids: list[str]) -> dict[str, Any]:
        return {
            "models": {
                "us.amazon.nova-2-lite-v1:0": {"input": 0.06, "output": 0.24},
                "us.anthropic.claude-haiku-4-5": {"input": 0.8, "output": 4.0},
            }
        }

    monkeypatch.setattr(
        "app.services.mcp_tools.models.fetch_model_pricing", _priced
    )

    result = await list_bedrock_models.handler({"project_id": "p1"})
    items = _decode_json_response(result)
    ids_in_order = [i["model_id"] for i in items]
    # Cheapest priced first, unknown-priced sinks to the end.
    assert ids_in_order[0] == "us.amazon.nova-2-lite-v1:0"
    assert ids_in_order[1] == "us.anthropic.claude-haiku-4-5"
    assert ids_in_order[-1] == "openai.gpt-5.5"


@pytest.mark.asyncio
async def test_discover_models_failure_returns_error_not_raise(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path
) -> None:
    """``discover_models`` raising → mcp_error so the agent loop survives."""
    async def _bad(_pid: str, _pd: Path) -> DiscoveryResult:
        raise RuntimeError("AWS creds missing")

    monkeypatch.setattr(
        "app.services.bedrock_discovery.discover_models", _bad, raising=False
    )

    result = await list_bedrock_models.handler({"project_id": "p1"})
    assert result.get("is_error") is True
    text = result["content"][0]["text"]
    assert "AWS creds missing" in text


@pytest.mark.asyncio
async def test_pricing_enrichment_attached(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path
) -> None:
    """Pricing fields attach to entries that match LiteLLM (runtime path)."""
    catalog = [_entry(model_id="us.anthropic.claude-haiku-4-5", provider="anthropic")]
    monkeypatch.setattr(
        "app.services.bedrock_discovery.discover_models",
        _stub_discover(catalog),
        raising=False,
    )

    async def _priced(ids: list[str]) -> dict[str, Any]:
        return {
            "models": {
                "us.anthropic.claude-haiku-4-5": {"input": 0.8, "output": 4.0}
            }
        }

    monkeypatch.setattr(
        "app.services.mcp_tools.models.fetch_model_pricing", _priced
    )

    result = await list_bedrock_models.handler({"project_id": "p1"})
    items = _decode_json_response(result)
    assert items[0]["input_price_per_1m"] == 0.8
    assert items[0]["output_price_per_1m"] == 4.0


@pytest.mark.asyncio
async def test_invalid_project_id_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ProjectTraversalError`` is converted to ``mcp_error`` (not raised out)."""
    from app.services.project_paths import ProjectTraversalError

    def _raise(_pid: str, base_dir: Path | None = None) -> Path:  # noqa: ARG001
        raise ProjectTraversalError("Invalid project_id: ../etc")

    monkeypatch.setattr(
        "app.services.mcp_tools.models.resolve_project_dir", _raise, raising=False
    )

    result = await list_bedrock_models.handler({"project_id": "../etc"})
    assert result.get("is_error") is True


@pytest.mark.asyncio
async def test_mantle_na_priced_renders_literal_na_string(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path
) -> None:
    """The NA sentinel from the matcher converts to the literal wire string ``"n/a"``.

    When an in-allowlist mantle id has no LiteLLM data,
    :func:`_match_model_ids` puts :data:`bedrock_pricing.NA` in the result.
    :func:`_enrich_with_pricing` converts that to the wire string
    ``"n/a"`` at the MCP boundary so the catalog entry literally carries
    ``input_price_per_1m="n/a"`` / ``output_price_per_1m="n/a"``.
    """
    from app.services.bedrock_pricing import NA

    catalog = [
        _entry(
            model_id="qwen.qwen-2.5-72b",
            provider="qwen",
            endpoint="mantle",
            api="chat_completions",
            region="us-west-2",
        )
    ]
    monkeypatch.setattr(
        "app.services.bedrock_discovery.discover_models",
        _stub_discover(catalog),
        raising=False,
    )

    async def _priced(_ids: list[str]) -> dict[str, Any]:
        # Matcher returned NA — the enrichment layer must convert.
        return {"models": {"qwen.qwen-2.5-72b": NA}}

    monkeypatch.setattr(
        "app.services.mcp_tools.models.fetch_model_pricing", _priced
    )

    result = await list_bedrock_models.handler({"project_id": "p1"})
    items = _decode_json_response(result)
    assert len(items) == 1
    entry = items[0]
    assert entry["input_price_per_1m"] == "n/a"
    assert entry["output_price_per_1m"] == "n/a"


@pytest.mark.asyncio
async def test_mantle_na_sinks_below_unknown(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path
) -> None:
    """n/a-priced mantle ids sort BELOW unknown-priced runtime ids.

    Sort contract: priced → unknown → n/a (ascending strength of
    "do not auto-pick"). Manual selection unaffected because ``set_models``
    validates against descriptors, not pricing.
    """
    from app.services.bedrock_pricing import NA

    catalog = [
        _entry(
            model_id="qwen.qwen-2.5-72b",
            provider="qwen",
            endpoint="mantle",
            api="chat_completions",
            region="us-west-2",
        ),
        _entry(model_id="us.amazon.nova-2-lite-v1:0", provider="amazon"),
        _entry(model_id="us.anthropic.claude-haiku-4-5", provider="anthropic"),
        # An "unknown" runtime id — no LiteLLM entry, but not in the mantle
        # allowlist either, so it stays None / unknown.
        _entry(model_id="us.meta.llama-99", provider="meta"),
    ]
    monkeypatch.setattr(
        "app.services.bedrock_discovery.discover_models",
        _stub_discover(catalog),
        raising=False,
    )

    async def _priced(_ids: list[str]) -> dict[str, Any]:
        return {
            "models": {
                "us.amazon.nova-2-lite-v1:0": {"input": 0.06, "output": 0.24},
                "us.anthropic.claude-haiku-4-5": {"input": 0.8, "output": 4.0},
                "qwen.qwen-2.5-72b": NA,
                # us.meta.llama-99 is unmapped → key absent → "unknown"
            }
        }

    monkeypatch.setattr(
        "app.services.mcp_tools.models.fetch_model_pricing", _priced
    )

    result = await list_bedrock_models.handler({"project_id": "p1"})
    items = _decode_json_response(result)
    ids_in_order = [i["model_id"] for i in items]
    # Priced ascending: nova-lite, haiku.
    assert ids_in_order[0] == "us.amazon.nova-2-lite-v1:0"
    assert ids_in_order[1] == "us.anthropic.claude-haiku-4-5"
    # Unknown sits ABOVE n/a (strongest "do not auto-pick" signal).
    assert ids_in_order[2] == "us.meta.llama-99"
    assert ids_in_order[3] == "qwen.qwen-2.5-72b"


@pytest.mark.asyncio
async def test_the_catalog_the_agent_receives_is_normalized(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wiring, not the normalizer: is the tool's OWN output collapsed?

    ``model_catalog``'s unit tests all passed with the normalization call removed
    from this handler — a well-tested core with nothing calling it. This asserts
    the response the agent actually reads.

    Fixture: one model under three ids at two price points, plus a newer and
    cheaper sibling that supersedes it.
    """
    catalog = [
        _entry(model_id="anthropic.claude-sonnet-4-6", provider="Anthropic"),
        _entry(model_id="us.anthropic.claude-sonnet-4-6", provider="Anthropic"),
        _entry(model_id="global.anthropic.claude-sonnet-4-6", provider="Anthropic"),
        _entry(model_id="anthropic.claude-sonnet-5", provider="Anthropic"),
        _entry(model_id="us.anthropic.claude-sonnet-5", provider="Anthropic"),
    ]
    _patch_discover(monkeypatch, _stub_discover(catalog))

    async def _priced(_ids: list[str]) -> dict[str, Any]:
        return {"models": {
            "anthropic.claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
            "us.anthropic.claude-sonnet-4-6": {"input": 3.30, "output": 16.50},
            "global.anthropic.claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
            "anthropic.claude-sonnet-5": {"input": 2.00, "output": 10.00},
            "us.anthropic.claude-sonnet-5": {"input": 2.20, "output": 11.00},
        }}

    monkeypatch.setattr(
        "app.services.mcp_tools.models.fetch_model_pricing", _priced, raising=False
    )

    payload = _decode_json_response(
        await list_bedrock_models.handler({"project_id": "p1"})
    )

    # Five ids in, two models out — nothing to compare across pricing bases.
    assert [e["model_id"] for e in payload] == [
        "anthropic.claude-sonnet-5",       # cheapest first, still
        "anthropic.claude-sonnet-4-6",
    ]
    # Each surviving row carries ITS OWN price, and the alternates are visible.
    assert payload[0]["input_price_per_1m"] == 2.00
    assert payload[1]["input_price_per_1m"] == 3.00
    assert {v["model_id"] for v in payload[1]["variants"]} == {
        "anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-4-6",
        "global.anthropic.claude-sonnet-4-6",
    }
    # And the supersession is stated on the row the agent reads.
    assert payload[1]["superseded_by"] == "anthropic.claude-sonnet-5"
    assert "superseded_by" not in payload[0]


# --------------------------------------------------------------------------- #
# Response shape: the array, and the wrapped form when models were excluded    #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_no_exclusions_returns_a_bare_array(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path
) -> None:
    """The common path is unchanged — a plain JSON array of entries.

    Pinned because this is the shape the coordinator reads to pick the judge,
    which is most of a run's spend. A silent wrapper appearing here would change
    what the agent parses on every single call.
    """
    catalog = [
        _entry(
            model_id="global.anthropic.claude-sonnet-5",
            provider="anthropic",
            endpoint="runtime",
            api="converse",
            region=None,
        ),
    ]
    _patch_discover(monkeypatch, _stub_discover(catalog))

    result = await list_bedrock_models.handler({"project_id": "p1"})

    assert "is_error" not in result, result
    payload = _decode_json_response(result)
    assert isinstance(payload, list)
    assert payload[0]["model_id"] == "global.anthropic.claude-sonnet-5"


@pytest.mark.asyncio
async def test_exclusions_are_reported_in_the_response(
    monkeypatch: pytest.MonkeyPatch, project_dir: Path
) -> None:
    """A model we could not route must reach a reader, not only CloudWatch.

    Under runtime-first an exclusion is always a mantle-ONLY model, so it is a
    routing gap on our side rather than the model being unavailable to the
    account — and a dropped model has NO symptom, it is simply absent from the
    list. The excluded id must NOT appear among the selectable models.
    """
    catalog = [
        _entry(
            model_id="global.anthropic.claude-sonnet-5",
            provider="anthropic",
            endpoint="runtime",
            api="converse",
            region=None,
        ),
    ]
    _patch_discover(
        monkeypatch,
        _stub_discover(
            catalog,
            excluded={"qwen.qwen3-coder": "every mantle API rejected this id"},
        ),
    )

    result = await list_bedrock_models.handler({"project_id": "p1"})

    assert "is_error" not in result, result
    payload = _decode_json_response(result)
    assert isinstance(payload, dict), "exclusions must be reported, not dropped"
    assert [m["model_id"] for m in payload["models"]] == [
        "global.anthropic.claude-sonnet-5"
    ]
    assert payload["excluded_models"] == [
        {
            "model_id": "qwen.qwen3-coder",
            "reason": "every mantle API rejected this id",
        }
    ]
    # The excluded id must never look selectable.
    assert "qwen.qwen3-coder" not in {m["model_id"] for m in payload["models"]}


def test_the_tool_description_documents_both_response_shapes() -> None:
    """A response that is sometimes an array and sometimes an object is only
    acceptable if the agent is TOLD that. The description is the contract."""
    description = list_bedrock_models.description
    assert "ARRAY" in description
    assert "excluded_models" in description
    # And the precedence direction, so a mantle row is not read as second-class.
    assert "runtime-first" in description.lower()
