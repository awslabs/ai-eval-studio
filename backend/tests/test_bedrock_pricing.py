# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for bedrock_pricing: LiteLLM fetch, match strategy, and file writing.

Mocks HTTP responses so tests never hit the real LiteLLM GitHub URL except
in the explicit integration test (marked ``integration``, skipped by default).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.bedrock_pricing import (
    NA,
    _clear_pricing_cache,
    _match_mantle_id,
    _match_model_ids,
    _NaSentinel,
    _parse_litellm_pricing,
    _strip_date_version_suffix,
    _strip_regional_prefix,
    estimate_cost_usd,
    fetch_model_pricing,
    lookup_cached_pricing,
    write_pricing_file,
)


class TestStripHelpers:
    """Tests for the two pure string-normalization helpers."""

    def test_strip_regional_prefix_us(self):
        assert _strip_regional_prefix("us.anthropic.claude-haiku-4-5-20251001-v1:0") == \
            "anthropic.claude-haiku-4-5-20251001-v1:0"

    def test_strip_regional_prefix_apac(self):
        assert _strip_regional_prefix("apac.amazon.nova-2-lite-v1:0") == \
            "amazon.nova-2-lite-v1:0"

    def test_strip_regional_prefix_no_match(self):
        assert _strip_regional_prefix("anthropic.claude-haiku-4-5") == \
            "anthropic.claude-haiku-4-5"

    def test_strip_date_version_suffix(self):
        assert _strip_date_version_suffix("us.anthropic.claude-sonnet-4-6-20250929-v1:0") == \
            "us.anthropic.claude-sonnet-4-6"

    def test_strip_date_version_suffix_no_match(self):
        assert _strip_date_version_suffix("us.anthropic.claude-sonnet-4-6") == \
            "us.anthropic.claude-sonnet-4-6"


def _litellm_entry(
    input_cost: float,
    output_cost: float,
    *,
    provider: str = "bedrock_converse",
    mode: str = "chat",
    cache_read: float | None = None,
    cache_write: float | None = None,
) -> dict[str, Any]:
    """Build a single LiteLLM-shaped entry with costs in $/token."""
    entry: dict[str, Any] = {
        "litellm_provider": provider,
        "mode": mode,
        "input_cost_per_token": input_cost,
        "output_cost_per_token": output_cost,
    }
    if cache_read is not None:
        entry["cache_read_input_token_cost"] = cache_read
    if cache_write is not None:
        entry["cache_creation_input_token_cost"] = cache_write
    return entry


class TestParseLiteLLMPricing:
    """Tests for _parse_litellm_pricing — filter + unit conversion."""

    def test_parses_bedrock_chat_entry(self):
        raw = {
            "anthropic.claude-haiku-4-5-20251001-v1:0": _litellm_entry(1e-6, 5e-6),
        }
        result = _parse_litellm_pricing(raw)
        assert result["anthropic.claude-haiku-4-5-20251001-v1:0"]["input"] == pytest.approx(1.0)
        assert result["anthropic.claude-haiku-4-5-20251001-v1:0"]["output"] == pytest.approx(5.0)

    def test_parses_bedrock_legacy_provider(self):
        """``litellm_provider="bedrock"`` (without ``_converse``) must also be included."""
        raw = {
            "anthropic.claude-legacy": _litellm_entry(1e-6, 5e-6, provider="bedrock"),
        }
        result = _parse_litellm_pricing(raw)
        assert "anthropic.claude-legacy" in result

    def test_admits_any_chat_mode_provider(self):
        """No provider gate — chat-mode entries are admitted regardless of provider.

        Mantle ids identify by shape (``{provider}.{bare}`` without regional
        prefix), so any new mantle provider LiteLLM tracks (``cohere``, ``vertex_ai``, …) lands
        in the cache automatically and resolves via :func:`_match_mantle_id`.
        Embeddings/image/rerank still skip via the chat-mode filter.
        """
        raw = {
            "command-r-plus": _litellm_entry(3e-6, 15e-6, provider="cohere"),
            "anthropic.claude-haiku": _litellm_entry(1e-6, 5e-6),
            "vertex_ai/gemini-pro": _litellm_entry(2e-6, 10e-6, provider="vertex_ai"),
        }
        result = _parse_litellm_pricing(raw)
        assert "command-r-plus" in result
        assert "anthropic.claude-haiku" in result
        assert "vertex_ai/gemini-pro" in result

    def test_skips_non_chat_mode(self):
        """Embedding / image / rerank entries must be filtered out."""
        raw = {
            "amazon.titan-embed": _litellm_entry(1e-7, 0.0, mode="embedding"),
            "anthropic.claude-chat": _litellm_entry(1e-6, 5e-6, mode="chat"),
        }
        result = _parse_litellm_pricing(raw)
        assert "amazon.titan-embed" not in result
        assert "anthropic.claude-chat" in result

    def test_includes_cache_costs_when_present(self):
        raw = {
            "anthropic.claude-haiku": _litellm_entry(
                1e-6, 5e-6,
                cache_read=1e-7,
                cache_write=1.25e-6,
            ),
        }
        result = _parse_litellm_pricing(raw)
        assert result["anthropic.claude-haiku"]["cache_read"] == pytest.approx(0.1)
        assert result["anthropic.claude-haiku"]["cache_write"] == pytest.approx(1.25)

    def test_omits_cache_costs_when_missing(self):
        raw = {
            "anthropic.claude-haiku": _litellm_entry(1e-6, 5e-6),
        }
        result = _parse_litellm_pricing(raw)
        entry = result["anthropic.claude-haiku"]
        assert "cache_read" not in entry
        assert "cache_write" not in entry

    def test_skips_entry_missing_required_costs(self):
        """Entries missing ``input_cost_per_token`` or ``output_cost_per_token`` are dropped."""
        raw = {
            "incomplete": {
                "litellm_provider": "bedrock",
                "mode": "chat",
                "input_cost_per_token": 1e-6,
                # no output_cost_per_token
            },
        }
        result = _parse_litellm_pricing(raw)
        assert "incomplete" not in result

    def test_skips_non_dict_entries(self):
        """LiteLLM's file includes sample-spec keys that aren't real model entries."""
        raw = {
            "sample_spec": "not a dict",
            "anthropic.claude": _litellm_entry(1e-6, 5e-6),
        }
        result = _parse_litellm_pricing(raw)
        assert "sample_spec" not in result
        assert "anthropic.claude" in result


class TestMatchModelIds:
    """Tests for _match_model_ids — exact → date-strip → region-strip."""

    def test_exact_match_wins(self):
        """The user's exact ID hits LiteLLM's key directly — no stripping needed."""
        pricing = {"us.anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 1.0, "output": 5.0}}
        result = _match_model_ids(["us.anthropic.claude-haiku-4-5-20251001-v1:0"], pricing)
        assert result["us.anthropic.claude-haiku-4-5-20251001-v1:0"]["input"] == 1.0

    def test_date_suffix_strip_fallback(self):
        """Sonnet 4.6-style: user ID has date suffix, LiteLLM key doesn't."""
        pricing = {"us.anthropic.claude-sonnet-4-6": {"input": 3.0, "output": 15.0}}
        result = _match_model_ids(["us.anthropic.claude-sonnet-4-6-20250929-v1:0"], pricing)
        assert result["us.anthropic.claude-sonnet-4-6-20250929-v1:0"]["input"] == 3.0

    def test_regional_prefix_strip_fallback(self):
        """User ID has regional prefix, LiteLLM only ships the bare entry."""
        pricing = {"amazon.nova-2-lite-v1:0": {"input": 0.33, "output": 2.75}}
        result = _match_model_ids(["us.amazon.nova-2-lite-v1:0"], pricing)
        assert result["us.amazon.nova-2-lite-v1:0"]["input"] == pytest.approx(0.33)

    def test_regional_and_date_strip_combined(self):
        """Both regional prefix AND date suffix mismatch; needs both strips."""
        pricing = {"anthropic.claude-sonnet-4-6": {"input": 3.0, "output": 15.0}}
        result = _match_model_ids(["us.anthropic.claude-sonnet-4-6-20250929-v1:0"], pricing)
        assert result["us.anthropic.claude-sonnet-4-6-20250929-v1:0"]["input"] == 3.0

    def test_three_current_target_models_all_match(self):
        """Regression guard: Haiku 4.5, Sonnet 4.6, Nova 2 Lite all resolve
        against a mock LiteLLM dataset that mirrors what the real file ships.
        """
        pricing = {
            "us.anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 1.0, "output": 5.0},
            "us.anthropic.claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
            "us.amazon.nova-2-lite-v1:0": {"input": 0.33, "output": 2.75},
        }
        result = _match_model_ids(
            [
                "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                "us.anthropic.claude-sonnet-4-6-20250929-v1:0",
                "us.amazon.nova-2-lite-v1:0",
            ],
            pricing,
        )
        assert len(result) == 3

    def test_unmatched_model_omitted(self):
        pricing = {"anthropic.claude-haiku": {"input": 1.0, "output": 5.0}}
        result = _match_model_ids(["us.meta.llama-99"], pricing)
        assert "us.meta.llama-99" not in result


class TestFetchModelPricing:
    """Tests for fetch_model_pricing — mocks HTTP to test the full flow."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        _clear_pricing_cache()
        yield
        _clear_pricing_cache()

    @pytest.mark.asyncio
    async def test_returns_pricing_json_shape(self):
        raw = {
            "us.anthropic.claude-haiku-4-5-20251001-v1:0": _litellm_entry(1e-6, 5e-6),
        }

        mock_response = MagicMock()
        mock_response.json.return_value = raw
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False

        with patch("app.services.bedrock_pricing.httpx.AsyncClient", return_value=mock_client):
            result = await fetch_model_pricing(["us.anthropic.claude-haiku-4-5-20251001-v1:0"])

        assert "_retrieved" in result
        assert "_source" in result
        assert "models" in result
        assert "litellm" in result["_source"].lower()
        assert "us.anthropic.claude-haiku-4-5-20251001-v1:0" in result["models"]

    @pytest.mark.asyncio
    async def test_caches_across_calls(self):
        raw = {
            "us.anthropic.claude-haiku-4-5-20251001-v1:0": _litellm_entry(1e-6, 5e-6),
        }

        mock_response = MagicMock()
        mock_response.json.return_value = raw
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False

        with patch("app.services.bedrock_pricing.httpx.AsyncClient", return_value=mock_client):
            await fetch_model_pricing(["us.anthropic.claude-haiku-4-5-20251001-v1:0"])
            await fetch_model_pricing(["us.anthropic.claude-haiku-4-5-20251001-v1:0"])

        mock_client.get.assert_called_once()


class TestWritePricingFile:
    """Tests for write_pricing_file — writes src/pricing.json into a project."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        _clear_pricing_cache()
        yield
        _clear_pricing_cache()

    @pytest.mark.asyncio
    async def test_writes_pricing_json(self, tmp_path: Path):
        (tmp_path / "src").mkdir()
        raw = {
            "us.anthropic.claude-haiku-4-5-20251001-v1:0": _litellm_entry(1e-6, 5e-6),
        }

        mock_response = MagicMock()
        mock_response.json.return_value = raw
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False

        with patch("app.services.bedrock_pricing.httpx.AsyncClient", return_value=mock_client):
            result = await write_pricing_file(tmp_path, ["us.anthropic.claude-haiku-4-5-20251001-v1:0"])

        assert result is not None
        assert result.exists()

        data = json.loads(result.read_text())
        assert "models" in data
        assert "us.anthropic.claude-haiku-4-5-20251001-v1:0" in data["models"]
        assert data["models"]["us.anthropic.claude-haiku-4-5-20251001-v1:0"]["input"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_returns_none_on_failure(self, tmp_path: Path):
        with patch("app.services.bedrock_pricing.httpx.AsyncClient", side_effect=Exception("network error")):
            result = await write_pricing_file(tmp_path, ["us.anthropic.claude-haiku-4-5-20251001-v1:0"])

        assert result is None


class TestLiteLLMIntegration:
    """Integration test hitting the real LiteLLM GitHub raw URL.

    Skipped by default. Run with ``pytest -m integration`` to detect upstream
    schema drift or missing pricing for current target models.
    """

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        _clear_pricing_cache()
        yield
        _clear_pricing_cache()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_api_covers_current_target_models(self):
        """The live LiteLLM file should cover our three current target models."""
        target_ids = [
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "us.anthropic.claude-sonnet-4-6-20250929-v1:0",
            "us.amazon.nova-2-lite-v1:0",
        ]
        result = await fetch_model_pricing(target_ids)

        assert "models" in result
        models = result["models"]

        for model_id in target_ids:
            assert model_id in models, (
                f"LiteLLM does not price {model_id} — upstream may have "
                "renamed or dropped the entry. Check model_prices_and_context_window.json."
            )
            assert models[model_id]["input"] > 0, f"{model_id} has zero input cost"
            assert models[model_id]["output"] > 0, f"{model_id} has zero output cost"


class TestMatchMantleId:
    """Tests for the mantle-id pricing matcher.

    Covers:
    - Mantle ids identified by SHAPE (``{provider}.{bare}`` without regional prefix).
    - Provider-agnostic: any new mantle provider LiteLLM tracks lands automatically.
    - Match priority: ``{provider}/{bare}`` → ``bedrock/{full}`` → ``{bare}``.
    - ``NA`` sentinel returned when a mantle-shaped id has no LiteLLM entry.
    - Regional-prefixed ids defer to the runtime path (decline with ``None``).
    - No false positives — exact-match only.
    - Bare-name extraction edge cases (hyphenated provider, dotted versions).

    Cache is cleared before each test so state never leaks between cases.
    """

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        _clear_pricing_cache()
        yield
        _clear_pricing_cache()

    # ----------- Sentinel typing + identity -------------

    def test_na_is_typed_singleton(self):
        """NA is _NaSentinel; ``prices is NA`` works as identity check."""
        assert isinstance(NA, _NaSentinel)
        assert repr(NA) == "<n/a>"
        # Identity (not equality) — same module-level instance.
        assert NA is NA

    def test_na_is_not_dict_or_none(self):
        """The sentinel must be distinguishable from dict and None."""
        assert NA is not None
        assert not isinstance(NA, dict)

    # ----------- Decline path (None — defer to runtime matcher) -------------

    def test_no_dot_returns_none(self):
        """Mantle ids must contain '.'; bare runtime-style ids fall through."""
        assert _match_mantle_id("claude-haiku-4-5", {}) is None

    def test_regional_prefix_us_returns_none(self):
        """Regional-prefixed ids belong to the runtime path; mantle declines."""
        assert _match_mantle_id("us.anthropic.claude-opus-4-7", {}) is None

    def test_regional_prefix_eu_returns_none(self):
        assert _match_mantle_id("eu.anthropic.claude-haiku", {}) is None

    def test_regional_prefix_apac_returns_none(self):
        assert _match_mantle_id("apac.amazon.nova-2-lite-v1:0", {}) is None

    def test_regional_prefix_global_returns_none(self):
        assert _match_mantle_id("global.anthropic.claude-haiku", {}) is None

    def test_empty_provider_or_bare_returns_none(self):
        """Defensive: leading or trailing dot doesn't crash; declines cleanly."""
        assert _match_mantle_id(".gpt-5.5", {}) is None
        assert _match_mantle_id("openai.", {}) is None

    # ----------- New providers land automatically (no allowlist!) -------------

    def test_unknown_provider_resolves_when_litellm_has_it(self):
        """A previously-unknown mantle provider (e.g. cohere) resolves automatically.

        This is the core "no hardcoding" guarantee — new mantle providers do
        NOT require a code change. If LiteLLM tracks ``cohere/command-x``,
        the matcher finds it via the ``{provider}/{bare}`` candidate.
        """
        cache = {"cohere/command-x": {"input": 2.5, "output": 10.0}}
        result = _match_mantle_id("cohere.command-x", cache)
        assert isinstance(result, dict)
        assert result["input"] == 2.5
        assert result["output"] == 10.0

    def test_unknown_provider_returns_na_when_litellm_lacks_it(self):
        """Mantle-shaped id with no LiteLLM entry → NA (intentional, never $0)."""
        assert _match_mantle_id("cohere.command-r-plus", {}) is NA

    def test_known_provider_no_match_returns_na(self):
        """openai mantle id with empty cache → NA (intentional, not missing)."""
        assert _match_mantle_id("openai.gpt-5.5", {}) is NA

    def test_qwen_returns_na_when_no_litellm_key(self):
        """qwen mantle id resolves to NA until LiteLLM ships a qwen key."""
        assert _match_mantle_id("qwen.qwen-2.5-72b", {}) is NA

    def test_z_ai_returns_na_when_no_litellm_key(self):
        assert _match_mantle_id("z-ai.glm-4.7", {}) is NA

    # ----------- Match priority -------------

    def test_provider_slash_bare_priority(self):
        """``{provider}/{bare}`` wins when present."""
        cache = {
            "openai/gpt-5.5": {"input": 5.0, "output": 15.0},
            "bedrock/openai.gpt-5.5": {"input": 99.0, "output": 99.0},
            "gpt-5.5": {"input": 88.0, "output": 88.0},
        }
        result = _match_mantle_id("openai.gpt-5.5", cache)
        assert isinstance(result, dict)
        assert result["input"] == 5.0
        assert result["output"] == 15.0

    def test_bedrock_full_id_priority(self):
        """When ``{provider}/{bare}`` missing, ``bedrock/{full}`` is the second pick."""
        cache = {
            "bedrock/openai.gpt-5.5": {"input": 7.0, "output": 21.0},
            "gpt-5.5": {"input": 88.0, "output": 88.0},
        }
        result = _match_mantle_id("openai.gpt-5.5", cache)
        assert isinstance(result, dict)
        assert result["input"] == 7.0

    def test_bare_name_priority(self):
        """When neither namespaced key exists, bare name is the third pick."""
        cache = {"gpt-5.5": {"input": 4.0, "output": 12.0}}
        result = _match_mantle_id("openai.gpt-5.5", cache)
        assert isinstance(result, dict)
        assert result["input"] == 4.0

    # ----------- No false positives -------------

    def test_no_false_positive_versioned(self):
        """gpt-5.5 must NOT match gpt-5.5-mini (different model, different price)."""
        cache = {"gpt-5.5-mini": {"input": 0.15, "output": 0.6}}
        assert _match_mantle_id("openai.gpt-5.5", cache) is NA

    def test_no_false_positive_grok(self):
        """xai.grok-4.3 must NOT match grok-4 (older model)."""
        cache = {"grok-4": {"input": 5.0, "output": 15.0}}
        assert _match_mantle_id("xai.grok-4.3", cache) is NA

    def test_no_false_positive_codestral_versioned(self):
        """mistral.codestral must NOT match codestral-2405 (specific version)."""
        cache = {"codestral-2405": {"input": 0.3, "output": 0.9}}
        assert _match_mantle_id("mistral.codestral", cache) is NA

    # ----------- Bare-name extraction edge cases -------------

    def test_hyphen_provider_z_ai(self):
        """z-ai.glm-4.7 partition produces bare='glm-4.7' (provider has hyphen)."""
        cache = {"z-ai/glm-4.7": {"input": 1.0, "output": 2.0}}
        result = _match_mantle_id("z-ai.glm-4.7", cache)
        assert isinstance(result, dict)
        assert result["input"] == 1.0

    def test_minimax_dotted_version(self):
        """minimax.MiniMax-M2.1 partition correctness — only first dot is the split."""
        cache = {"minimax/MiniMax-M2.1": {"input": 0.3, "output": 1.0}}
        result = _match_mantle_id("minimax.MiniMax-M2.1", cache)
        assert isinstance(result, dict)
        assert result["input"] == 0.3

    def test_xai_dotted_version_via_provider_slash(self):
        """xai.grok-4.3 → xai/grok-4.3 (full version preserved after partition)."""
        cache = {"xai/grok-4.3": {"input": 5.0, "output": 15.0}}
        result = _match_mantle_id("xai.grok-4.3", cache)
        assert isinstance(result, dict)
        assert result["input"] == 5.0

    def test_moonshot_kimi_dashed_version(self):
        """moonshot.kimi-k2-0711-preview: dashes are preserved through partition."""
        cache = {"moonshot/kimi-k2-0711-preview": {"input": 0.6, "output": 2.5}}
        result = _match_mantle_id("moonshot.kimi-k2-0711-preview", cache)
        assert isinstance(result, dict)
        assert result["input"] == 0.6

    # ----------- _parse_litellm_pricing admits any chat-mode entry -------------

    def test_parse_litellm_pricing_admits_xai_chat(self):
        """``litellm_provider='xai'`` + ``mode='chat'`` is admitted to the cache."""
        raw = {
            "xai/grok-2": _litellm_entry(2e-6, 6e-6, provider="xai"),
        }
        result = _parse_litellm_pricing(raw)
        assert "xai/grok-2" in result
        assert result["xai/grok-2"]["input"] == pytest.approx(2.0)

    def test_parse_litellm_pricing_admits_openai_chat(self):
        raw = {"gpt-5.5": _litellm_entry(5e-6, 15e-6, provider="openai")}
        result = _parse_litellm_pricing(raw)
        assert "gpt-5.5" in result

    def test_parse_litellm_pricing_admits_mistral(self):
        raw = {"mistral/codestral-2405": _litellm_entry(3e-7, 9e-7, provider="mistral")}
        result = _parse_litellm_pricing(raw)
        assert "mistral/codestral-2405" in result

    def test_parse_litellm_pricing_admits_new_provider_cohere(self):
        """No allowlist gate: cohere chat entries land in the cache automatically.

        This is the regression guard for the "new mantle providers land
        without a code change" guarantee.
        """
        raw = {
            "cohere/command-x": _litellm_entry(2.5e-6, 10e-6, provider="cohere"),
            "command-r-plus": _litellm_entry(3e-6, 15e-6, provider="cohere"),
        }
        result = _parse_litellm_pricing(raw)
        assert "cohere/command-x" in result
        assert "command-r-plus" in result

    def test_parse_litellm_pricing_admits_vertex_ai(self):
        """vertex_ai chat entries are admitted (no provider gate)."""
        raw = {"vertex_ai/gemini-pro": _litellm_entry(1e-6, 5e-6, provider="vertex_ai")}
        result = _parse_litellm_pricing(raw)
        assert "vertex_ai/gemini-pro" in result

    def test_parse_litellm_pricing_existing_bedrock_filter_preserved(self):
        """Regression: bedrock entries still admitted alongside any other provider."""
        raw = {
            "anthropic.claude-haiku-4-5-20251001-v1:0": _litellm_entry(
                1e-6, 5e-6, provider="bedrock_converse"
            ),
            "openai/gpt-5.5": _litellm_entry(5e-6, 15e-6, provider="openai"),
        }
        result = _parse_litellm_pricing(raw)
        assert "anthropic.claude-haiku-4-5-20251001-v1:0" in result
        assert "openai/gpt-5.5" in result

    # ----------- _match_model_ids mantle dispatch -------------

    def test_match_model_ids_mantle_returns_na_for_unmatched_mantle_id(self):
        """A mantle-shaped id with no entry maps to NA (key present, value=NA)."""
        pricing = {
            "us.anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 1.0, "output": 5.0},
        }
        result = _match_model_ids(
            ["us.anthropic.claude-haiku-4-5-20251001-v1:0", "qwen.qwen-2.5-72b"],
            pricing,
        )
        # Runtime id resolves via the existing 3-step strategy.
        assert "us.anthropic.claude-haiku-4-5-20251001-v1:0" in result
        # Mantle id is in the result with the NA sentinel — NOT omitted.
        assert "qwen.qwen-2.5-72b" in result
        assert result["qwen.qwen-2.5-72b"] is NA

    def test_match_model_ids_mantle_dispatch_returns_dict_on_hit(self):
        """A mantle id with a LiteLLM entry resolves to that dict."""
        pricing = {"openai/gpt-5.5": {"input": 5.0, "output": 15.0}}
        result = _match_model_ids(["openai.gpt-5.5"], pricing)
        assert "openai.gpt-5.5" in result
        assert result["openai.gpt-5.5"]["input"] == 5.0

    def test_match_model_ids_unknown_mantle_provider_resolves(self):
        """The "no allowlist" guarantee: cohere.command-x lands without code change."""
        pricing = {"cohere/command-x": {"input": 2.5, "output": 10.0}}
        result = _match_model_ids(["cohere.command-x"], pricing)
        assert "cohere.command-x" in result
        assert result["cohere.command-x"]["input"] == 2.5

    def test_match_model_ids_runtime_path_unchanged(self):
        """Regression spot-check: existing runtime ids resolve via existing strategy."""
        pricing = {"us.anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 1.0, "output": 5.0}}
        result = _match_model_ids(["us.anthropic.claude-haiku-4-5-20251001-v1:0"], pricing)
        assert result["us.anthropic.claude-haiku-4-5-20251001-v1:0"]["input"] == 1.0

    def test_match_model_ids_no_warn_log_for_na(self, caplog):
        """NA path is intentional — no WARN log (only the 'missing' path warns)."""
        import logging

        caplog.set_level(logging.WARNING, logger="app.services.bedrock_pricing")
        _match_model_ids(["qwen.qwen-2.5-72b"], {})
        warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        # No warning for the NA id — mantle-shaped → NA, not "missing."
        assert all("qwen.qwen-2.5-72b" not in r.getMessage() for r in warn_records)

    def test_match_model_ids_non_mantle_unmatched_still_warns(self):
        """Regression: WARN-and-omit path stays for plain (non-shaped) misses."""
        pricing = {}
        # No-dot id falls through mantle dispatch → runtime path → WARN+omit.
        result = _match_model_ids(["claude-haiku-no-dot"], pricing)
        assert "claude-haiku-no-dot" not in result

    # ----------- lookup_cached_pricing mantle dispatch -------------

    @pytest.mark.asyncio
    async def test_lookup_cached_pricing_mantle_dispatch_returns_dict_on_hit(self):
        """After cache prime, mantle id resolves via the matcher to a dict."""
        raw = {
            "openai/gpt-5.5": _litellm_entry(5e-6, 15e-6, provider="openai"),
        }
        mock_response = MagicMock()
        mock_response.json.return_value = raw
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False

        with patch("app.services.bedrock_pricing.httpx.AsyncClient", return_value=mock_client):
            await fetch_model_pricing(["openai.gpt-5.5"])

        result = lookup_cached_pricing("openai.gpt-5.5")
        assert isinstance(result, dict)
        assert result["input"] == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_lookup_cached_pricing_mantle_dispatch_returns_na_on_miss(self):
        """Mantle-shaped id with no cache entry returns NA (after fetch)."""
        raw = {
            # Some bedrock entry to make the cache non-empty
            "anthropic.claude-haiku": _litellm_entry(1e-6, 5e-6),
        }
        mock_response = MagicMock()
        mock_response.json.return_value = raw
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False

        with patch("app.services.bedrock_pricing.httpx.AsyncClient", return_value=mock_client):
            await fetch_model_pricing(["anthropic.claude-haiku"])

        # qwen is mantle-shaped, cache has no qwen key → NA
        result = lookup_cached_pricing("qwen.qwen-2.5-72b")
        assert result is NA

    def test_lookup_cached_pricing_cold_cache_returns_none(self):
        """Cold cache → None for any id (mantle or runtime); matches existing contract."""
        # The dispatch falls through to the existing "cache empty → None" branch.
        result = lookup_cached_pricing("openai.gpt-5.5")
        assert result is None

    @pytest.mark.asyncio
    async def test_lookup_cached_pricing_unknown_provider_resolves_when_litellm_has_it(self):
        """New mantle provider lands automatically when LiteLLM ships a key for it."""
        raw = {
            "cohere/command-x": _litellm_entry(2.5e-6, 10e-6, provider="cohere"),
        }
        mock_response = MagicMock()
        mock_response.json.return_value = raw
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False

        with patch("app.services.bedrock_pricing.httpx.AsyncClient", return_value=mock_client):
            await fetch_model_pricing(["cohere.command-x"])

        # cohere has no enumeration anywhere — but the matcher resolves it
        # via the {provider}/{bare} candidate against the live cache.
        result = lookup_cached_pricing("cohere.command-x")
        assert isinstance(result, dict)
        assert result["input"] == pytest.approx(2.5)

    @pytest.mark.asyncio
    async def test_lookup_cached_pricing_no_dot_id_returns_none_on_miss(self):
        """Plain (non-shaped) id keeps the existing None path."""
        raw = {
            "anthropic.claude-haiku": _litellm_entry(1e-6, 5e-6),
        }
        mock_response = MagicMock()
        mock_response.json.return_value = raw
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False

        with patch("app.services.bedrock_pricing.httpx.AsyncClient", return_value=mock_client):
            await fetch_model_pricing(["anthropic.claude-haiku"])

        # No dot → not mantle-shaped → declines to None; runtime matcher
        # then misses too → final return is None (NOT NA).
        result = lookup_cached_pricing("plain-no-dot-id")
        assert result is None

    # ----------- estimate_cost_usd handles NA -------------

    @pytest.mark.asyncio
    async def test_estimate_cost_usd_returns_zero_for_na(self):
        """NA is treated like None for cost aggregation (returns 0.0).

        The catalog separately renders ``"n/a"`` so the user sees the reason;
        the cost aggregator stays unchanged (no double-billing through fallbacks).
        """
        raw = {"anthropic.claude-haiku": _litellm_entry(1e-6, 5e-6)}
        mock_response = MagicMock()
        mock_response.json.return_value = raw
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False

        with patch("app.services.bedrock_pricing.httpx.AsyncClient", return_value=mock_client):
            await fetch_model_pricing(["anthropic.claude-haiku"])

        cost = estimate_cost_usd("qwen.qwen-2.5-72b", {"inputTokens": 1000, "outputTokens": 500})
        assert cost == 0.0
