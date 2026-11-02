# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bedrock model pricing fetched from LiteLLM's community price list.

Fetches ``model_prices_and_context_window.json`` from LiteLLM's GitHub repo,
filters to Bedrock-hosted chat models, converts per-token costs to per-1M-tokens
for our on-disk schema, and writes a ``src/pricing.json`` file into eval
harness projects so the runner can compute accurate generation costs.

LiteLLM was chosen over AWS's Bulk Pricing API because AWS frequently lags
months behind on publishing prices for newly-released Bedrock models (Haiku 4.5,
Sonnet 4.6, Nova 2 Lite all missing as of this writing). The community list
tracks releases within days and exposes the same information in a simpler shape.

Source schema (entries live under a top-level model-ID key):
- ``litellm_provider``: ``"bedrock"`` or ``"bedrock_converse"`` for Bedrock entries
- ``mode``: ``"chat"`` / ``"embedding"`` / ``"image_generation"`` / ``"rerank"``;
  we keep only ``"chat"``
- ``input_cost_per_token``, ``output_cost_per_token``: dollars per token (scalar)
- ``cache_read_input_token_cost``, ``cache_creation_input_token_cost``: optional
  caching costs, same unit
- Keys include both bare (``anthropic.claude-haiku-4-5-20251001-v1:0``) and
  regional inference-profile (``us.anthropic.claude-haiku-4-5-20251001-v1:0``,
  ``apac.*``, ``eu.*``, ``au.*``, ``global.*``) variants.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

# Cache TTL: community list is updated frequently but we don't need to
# re-fetch within a single session. 24h keeps us fresh without hammering GitHub.
_CACHE_TTL_SECONDS = 24 * 60 * 60

# Populated dict shape: {model_id_key: {"input": float, "output": float,
# optionally "cache_read": float, "cache_write": float}} in per-1M-tokens USD.
_pricing_cache: dict[str, dict[str, float]] | None = None
_pricing_cache_timestamp: float = 0.0

# Regional inference-profile prefixes on Bedrock. Stripping these lets a
# caller-supplied ``us.anthropic.claude-sonnet-4-6-...`` fall back to the
# base ``anthropic.claude-sonnet-4-6`` entry when LiteLLM only ships one.
_REGIONAL_PREFIXES = ("us.", "eu.", "apac.", "au.", "global.")

# Strips LiteLLM's optional date + version suffix, e.g.
# ``claude-haiku-4-5-20251001-v1:0`` → ``claude-haiku-4-5``. Used as a
# last-resort match step.
_DATE_VERSION_SUFFIX = re.compile(r"-\d{8,}-v\d+:\d+$")


# NOTE: a static provider allowlist used to live here (anthropic, openai,
# xai, qwen, z-ai, google, …). It was removed because the mantle endpoint
# keeps adding providers (cohere, etc.), and any enumeration goes stale the
# moment a new one ships. The matcher identifies a mantle id by its
# ``{provider}.{bare}`` shape alone (without a regional prefix — those
# belong to the runtime path) and consults LiteLLM via the same exact-match
# candidates. New providers land automatically: if LiteLLM has the key,
# real pricing returns; if not, the typed ``NA`` sentinel surfaces (never a
# silent ``$0``).


class _NaSentinel:
    """Pricing intentionally unavailable — distinct from ``None``.

    Returned by :func:`_match_mantle_id` and :func:`lookup_cached_pricing`
    when a mantle-shaped id has no LiteLLM entry. Callers use
    identity (``prices is NA``) to distinguish "intentionally unavailable"
    (don't auto-pick, render literal ``"n/a"``) from "unknown / cache cold"
    (``None`` — runtime path semantics).

    A typed singleton (rather than a string ``"n/a"``) keeps the
    ``dict[str, float]`` typing of pricing-dict callers honest — code that
    does ``prices["input"]`` would crash on a string but is forced to check
    ``is NA`` first by the type hint.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<n/a>"


# Module-level singleton. Compare via identity: ``prices is NA``.
NA: _NaSentinel = _NaSentinel()


def _strip_regional_prefix(model_id: str) -> str:
    """Return ``model_id`` with a regional inference-profile prefix removed, if any."""
    for prefix in _REGIONAL_PREFIXES:
        if model_id.startswith(prefix):
            return model_id[len(prefix):]
    return model_id


def _strip_date_version_suffix(model_id: str) -> str:
    """Return ``model_id`` with a trailing ``-YYYYMMDD-v1:0`` suffix removed, if any."""
    return _DATE_VERSION_SUFFIX.sub("", model_id)


def _parse_litellm_pricing(data: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Filter LiteLLM's raw JSON to chat-mode pricing entries.

    Single cache, single TTL — admits any chat-mode entry regardless of
    ``litellm_provider``. Two reasons:

    1. **Bedrock runtime path**: legacy ``bedrock`` and ``bedrock_converse``
       providers used by :func:`_match_model_ids`'s 3-step strategy
       (regression-protected).
    2. **Bedrock-mantle path**: the mantle endpoint keeps adding providers
       (``openai``, ``xai``, ``anthropic``, ``mistral``, ``cohere``, …).
       Enumerating them goes stale; instead, :func:`_match_mantle_id`
       consults whatever is in the cache via exact-match candidates. If
       LiteLLM has the key, real pricing returns; if not, ``NA``.

    The chat-mode filter still excludes embeddings / image / rerank entries
    (those aren't valid eval target/judge models). Non-chat entries are
    skipped silently.

    Args:
        data: Parsed JSON from ``model_prices_and_context_window.json``.

    Returns:
        Dict keyed by the LiteLLM model-ID string (preserved verbatim so matches
        can be exact). Each value has ``input`` and ``output`` (required) plus
        optional ``cache_read`` and ``cache_write`` in USD per 1M tokens.
    """
    result: dict[str, dict[str, float]] = {}

    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue

        provider = entry.get("litellm_provider", "")
        if not isinstance(provider, str):
            continue
        # No provider gate — admit any chat-mode entry. Mantle id matching
        # is done by id shape (``{provider}.{bare}``), not by enumerating
        # known providers. New mantle providers land automatically.

        # Drop non-chat Bedrock entries — embeddings, image generation, rerank
        # aren't valid target/judge models for text evals.
        if entry.get("mode") != "chat":
            continue

        input_cost = entry.get("input_cost_per_token")
        output_cost = entry.get("output_cost_per_token")
        if input_cost is None or output_cost is None:
            continue

        try:
            input_per_1m = float(input_cost) * 1_000_000
            output_per_1m = float(output_cost) * 1_000_000
        except (ValueError, TypeError):
            continue

        prices: dict[str, float] = {
            "input": input_per_1m,
            "output": output_per_1m,
        }

        cache_read_cost = entry.get("cache_read_input_token_cost")
        if cache_read_cost is not None:
            try:
                prices["cache_read"] = float(cache_read_cost) * 1_000_000
            except (ValueError, TypeError):
                pass

        cache_write_cost = entry.get("cache_creation_input_token_cost")
        if cache_write_cost is not None:
            try:
                prices["cache_write"] = float(cache_write_cost) * 1_000_000
            except (ValueError, TypeError):
                pass

        result[key] = prices

    return result


def _match_mantle_id(
    mantle_id: str,
    pricing: dict[str, dict[str, float]],
) -> dict[str, float] | _NaSentinel | None:
    """Match a mantle-tier id to a LiteLLM pricing entry via exact-match only.

    Mantle ids have the shape ``{provider}.{bare-name}`` (e.g.
    ``openai.gpt-5.5``, ``xai.grok-4.3``, ``cohere.command-x``,
    ``z-ai.glm-4.7``). The matcher is **provider-agnostic** — any mantle id
    LiteLLM tracks resolves automatically without code change. Mantle-ness
    is identified by id shape, not by enumeration.

    Mantle ids must NOT carry a regional prefix (``us.``, ``eu.``, ``apac.``,
    ``au.``, ``global.``) — those are runtime inference-profile ids handled
    by :func:`_match_model_ids`'s 3-step strategy. Returning ``None`` for
    prefixed ids lets that strategy take its turn.

    Match priority (first hit wins, ALL exact-match — never substring):

    1. ``{provider}/{bare}`` (e.g. ``openai/gpt-5.5``) — LiteLLM's primary
       shape for native-provider entries.
    2. ``bedrock/{full_id}`` (e.g. ``bedrock/openai.gpt-5.5``) — covers
       hypothetical future LiteLLM mirrors of mantle entries under bedrock.
    3. ``{bare}`` alone (e.g. ``gpt-5.5``) — LiteLLM's bare-key shape for
       providers that omit the ``provider/`` prefix.

    Substring/regex/startswith are explicitly rejected
    (false positives across versioned models would silently corrupt cost).

    Returns:
        - ``dict`` (copy of the matched entry) on hit.
        - :data:`NA` when the id has the mantle shape but NO candidate key
          matched. Documented "intentionally unavailable" signal
          — never silent ``$0``.
        - ``None`` when the id is not a mantle id (no dot, or carries a
          regional prefix). Runtime matcher takes over.

    Args:
        mantle_id: Caller-supplied model id; may or may not be a mantle id.
        pricing: The parsed LiteLLM cache (keys preserved verbatim).
    """
    if "." not in mantle_id:
        return None

    # Regional-prefixed ids belong to the runtime path. Defer.
    for prefix in _REGIONAL_PREFIXES:
        if mantle_id.startswith(prefix):
            return None

    # Only first dot is the split — preserves dotted versions in the bare
    # name (e.g. ``openai.gpt-5.5`` → bare ``gpt-5.5``; ``z-ai.glm-4.7`` →
    # bare ``glm-4.7``). LiteLLM provider keys do not contain dots, so the
    # partition is unambiguous.
    provider, _, bare = mantle_id.partition(".")
    if not provider or not bare:
        return None

    # Exact-match candidates only — substring/regex/startswith would silently
    # collide across versioned models.
    candidates = (f"{provider}/{bare}", f"bedrock/{mantle_id}", bare)
    for key in candidates:
        if key in pricing:
            return dict(pricing[key])

    return NA


def _match_model_ids(
    model_ids: list[str],
    pricing: dict[str, dict[str, float]],
) -> dict[str, dict[str, float] | _NaSentinel]:
    """Match caller model IDs to LiteLLM pricing entries.

    Runs the 3-step runtime strategy FIRST and hands an id to
    :func:`_match_mantle_id` only when that strategy misses. Many runtime ids
    are mantle-shaped (``amazon.nova-micro-v1:0``), so trying the mantle
    matcher first would mis-tag them as :data:`NA`; mantle ids have no
    runtime cache entry, so they reach the matcher cleanly.

    Match strategy for runtime ids (first hit wins):

    1. Exact key match. Haiku 4.5 and Nova 2 Lite hit here because LiteLLM
       tracks the regional-prefixed inference-profile IDs directly.
    2. Strip ``-YYYYMMDD-v1:0`` date+version suffix and retry. Catches Sonnet
       4.6 where LiteLLM ships ``us.anthropic.claude-sonnet-4-6`` without the
       date-dated variant.
    3. Strip regional prefix (``us.``, ``eu.``, ``apac.``, ``au.``, ``global.``)
       and retry steps 1 & 2. Final fallback for regional IDs without dedicated
       entries.

    Unmatched runtime IDs log a warning and are omitted from the returned dict
    — loud rather than silent, so missing coverage is visible. Mantle-shaped
    ids whose candidates miss LiteLLM are kept with the :data:`NA` sentinel
    (intentional, not missing) and skip the WARN — auto-pick uses the
    sentinel to skip them and the catalog renders ``"n/a"`` literally.

    Args:
        model_ids: Full Bedrock model IDs, mantle ids, or inference-profile IDs.
        pricing: Parsed LiteLLM pricing keyed by LiteLLM model ID.

    Returns:
        Dict keyed by the caller's original model ID. Values are the per-1M
        USD pricing dict on hit, or :data:`NA` for mantle-shaped ids
        with no LiteLLM data.
    """
    result: dict[str, dict[str, float] | _NaSentinel] = {}

    for model_id in model_ids:
        # Runtime path FIRST. Many bedrock-runtime ids look mantle-shaped
        # (``amazon.nova-micro-v1:0``, ``anthropic.claude-haiku``) — let the
        # 3-step strategy resolve them via direct cache hits before the
        # mantle matcher sees them. Mantle ids have no bedrock cache entry,
        # so they fall through cleanly to the matcher below.
        hit: dict[str, float] | None = None

        # 1. Exact
        if model_id in pricing:
            hit = pricing[model_id]

        # 2. Strip date+version suffix, try again
        if hit is None:
            stripped = _strip_date_version_suffix(model_id)
            if stripped != model_id and stripped in pricing:
                hit = pricing[stripped]

        # 3. Strip regional prefix, then retry (1) and (2)
        if hit is None:
            no_region = _strip_regional_prefix(model_id)
            if no_region != model_id:
                if no_region in pricing:
                    hit = pricing[no_region]
                else:
                    stripped = _strip_date_version_suffix(no_region)
                    if stripped != no_region and stripped in pricing:
                        hit = pricing[stripped]

        if hit is not None:
            result[model_id] = dict(hit)
            continue

        # Runtime path missed → try mantle dispatch. ``None`` means "not a
        # mantle-shaped id" → WARN+omit. ``NA`` is the
        # documented "intentionally unavailable" path. ``dict`` is a hit.
        # New mantle providers land here automatically — no enumeration.
        mantle_hit = _match_mantle_id(model_id, pricing)
        if isinstance(mantle_hit, dict):
            result[model_id] = mantle_hit
            continue
        if mantle_hit is NA:
            # Mantle-shaped id with no LiteLLM coverage. Catalog renders
            # ``"n/a"`` literally (never a silent ``$0``).
            result[model_id] = NA
            continue

        logger.warning("pricing missing for model %s", model_id)

    return result


async def fetch_model_pricing(
    model_ids: list[str],
) -> dict[str, Any]:
    """Fetch pricing for the given Bedrock model IDs from the LiteLLM price list.

    Returns a dict suitable for writing as ``src/pricing.json``. Uses an
    in-process cache so repeated calls don't re-fetch.

    Args:
        model_ids: List of full Bedrock model IDs to look up. LiteLLM covers
            every region in one file, so no region argument is needed.

    Returns:
        Dict with ``_retrieved``, ``_source``, ``_note``, and ``models`` keys
        matching the pricing.json schema (USD per 1M tokens).
    """
    global _pricing_cache, _pricing_cache_timestamp

    now = time.time()
    cache_stale = (now - _pricing_cache_timestamp) > _CACHE_TTL_SECONDS
    if _pricing_cache is None or cache_stale:
        logger.info("Fetching Bedrock pricing from %s", LITELLM_PRICING_URL)

        fetch_start = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(LITELLM_PRICING_URL)
            resp.raise_for_status()
            raw = resp.json()
        fetch_duration = time.monotonic() - fetch_start

        _pricing_cache = _parse_litellm_pricing(raw)
        _pricing_cache_timestamp = now
        logger.info(
            "Parsed pricing for %d Bedrock chat models (fetch took %.1fs)",
            len(_pricing_cache),
            fetch_duration,
        )

    matched = _match_model_ids(model_ids, _pricing_cache)

    return {
        "_retrieved": datetime.now(timezone.utc).isoformat(),
        "_source": LITELLM_PRICING_URL,
        "_note": "Prices in USD per 1M tokens (source: LiteLLM). Edit as needed.",
        "models": matched,
    }


async def write_pricing_file(
    project_dir: Path,
    model_ids: list[str],
) -> Path | None:
    """Fetch pricing and write ``src/pricing.json`` into a harness project.

    Non-fatal: returns None if the fetch fails. There are NO
    fallback prices in the harness — models missing from pricing.json
    report their cost as null (with a one-time stderr warning), so a
    failed write means unknown costs, never invented ones.

    Args:
        project_dir: Root of the eval harness project.
        model_ids: All model IDs to include (target + judge).

    Returns:
        Path to the written file, or None on failure.
    """
    try:
        pricing = await fetch_model_pricing(model_ids)
        output_path = project_dir / "src" / "pricing.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(pricing, indent=2), encoding="utf-8")
        logger.info("Wrote pricing for %d models to %s", len(pricing["models"]), output_path)
        return output_path
    except Exception:
        logger.warning(
            "Failed to fetch/write pricing — affected models will report "
            "cost as unknown (null)", exc_info=True,
        )
        return None


def lookup_cached_pricing(model_id: str) -> dict[str, float] | _NaSentinel | None:
    """Sync lookup against the in-process pricing cache.

    Returns the per-1M USD pricing dict for ``model_id`` if the cache has
    been populated (via an earlier ``fetch_model_pricing`` / ``refresh_pricing``
    call). Mirrors :func:`_match_model_ids`: the runtime 3-step strategy
    (exact → date-version-stripped → regional-prefix-stripped) runs first,
    and :func:`_match_mantle_id` only on a miss.

    Returns:
        - ``dict`` with ``input`` / ``output`` (per 1M tokens, USD) plus
          optional ``cache_read`` / ``cache_write`` on hit.
        - :data:`NA` when ``model_id`` is a mantle-shaped id with no
          LiteLLM entry in the cache (intentional unavailability —
          :func:`estimate_cost_usd` returns 0.0 just like the ``None`` path,
          but downstream catalog rendering shows ``"n/a"`` literally).
        - ``None`` when the cache is empty or the id is genuinely unknown.

    Args:
        model_id: Full Bedrock model ID, mantle id, or inference-profile ID.
    """
    if _pricing_cache is None:
        return None

    # Runtime match FIRST. Bedrock-runtime ids like ``amazon.nova-micro-v1:0``
    # look mantle-shaped but resolve via direct cache hits below; trying the
    # mantle matcher first would mis-tag them as NA. Mantle ids have no
    # bedrock cache entry so they fall through to the matcher cleanly.
    if model_id in _pricing_cache:
        return dict(_pricing_cache[model_id])

    stripped = _strip_date_version_suffix(model_id)
    if stripped != model_id and stripped in _pricing_cache:
        return dict(_pricing_cache[stripped])

    no_region = _strip_regional_prefix(model_id)
    if no_region != model_id:
        if no_region in _pricing_cache:
            return dict(_pricing_cache[no_region])
        stripped_no_region = _strip_date_version_suffix(no_region)
        if stripped_no_region != no_region and stripped_no_region in _pricing_cache:
            return dict(_pricing_cache[stripped_no_region])

    # Runtime path missed → mantle dispatch. ``None`` = not mantle-shaped
    # → return None (genuinely unknown). ``NA`` = mantle id with no LiteLLM
    # coverage. ``dict`` = mantle hit.
    mantle_hit = _match_mantle_id(model_id, _pricing_cache)
    if isinstance(mantle_hit, dict):
        return mantle_hit
    if mantle_hit is NA:
        return NA

    return None


def estimate_cost_usd(model_id: str, usage: dict[str, Any]) -> float:
    """Estimate USD cost from token usage and model ID.

    Resolves pricing via :func:`lookup_cached_pricing` (returns 0.0 when
    the cache is cold or the model is unrecognized — cost tracking is
    best-effort outside the harness, which has its own ``pricing.json``).

    Cache accounting matches the harness ``_estimate_cost``: Bedrock's
    Converse ``TokenUsage`` reports ``cacheReadInputTokens`` and
    ``cacheWriteInputTokens`` as **separate** counts; ``inputTokens``
    is the non-cached portion. Cache terms are **added on top**.
    A model whose price entry carries no ``cache_read`` / ``cache_write``
    rate is billed at the input rate for those tokens — the closest
    available number, and identical to plain input pricing on a run that
    reported no cache tokens at all.

    Args:
        model_id: Full Bedrock model identifier.
        usage: Dict with ``inputTokens``/``input_tokens``,
            ``outputTokens``/``output_tokens``, and optional
            ``cacheReadInputTokens`` / ``cacheWriteInputTokens`` keys
            (snake_case variants accepted on read).

    Returns:
        Estimated cost in USD. Zero when pricing isn't available.
    """
    prices = lookup_cached_pricing(model_id)
    if prices is None:
        return 0.0
    # NA (a mantle id with no LiteLLM data) is treated like None for
    # cost aggregation — returns 0.0 so we don't double-bill through the
    # input-rate fallbacks below. The catalog renders ``"n/a"`` separately
    # so the user sees the reason.
    if prices is NA:
        return 0.0

    input_tokens = usage.get("inputTokens") or usage.get("input_tokens") or 0
    output_tokens = usage.get("outputTokens") or usage.get("output_tokens") or 0
    cache_read = (
        usage.get("cacheReadInputTokens")
        or usage.get("cache_read_input_tokens")
        or 0
    )
    cache_write = (
        usage.get("cacheWriteInputTokens")
        or usage.get("cache_write_input_tokens")
        or 0
    )

    input_price = prices.get("input", 0.0)
    cache_read_price = prices.get("cache_read", input_price)
    cache_write_price = prices.get("cache_write", input_price)

    # Pricing is per 1M tokens; tokens / 1_000_000 * price_per_1M = USD.
    return (
        input_tokens / 1_000_000 * input_price
        + cache_read / 1_000_000 * cache_read_price
        + cache_write / 1_000_000 * cache_write_price
        + output_tokens / 1_000_000 * prices.get("output", 0.0)
    )


def _clear_pricing_cache() -> None:
    """Clear the in-process pricing cache. Test-only — prefixed with ``_``.

    Resets both the cached data and the timestamp so the next fetch is forced.
    """
    global _pricing_cache, _pricing_cache_timestamp
    _pricing_cache = None
    _pricing_cache_timestamp = 0.0
