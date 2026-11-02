# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP tool ``list_bedrock_models``: merged runtime + mantle catalog discovery.

Delegates to :func:`app.services.bedrock_discovery.discover_models` for the
authoritative merged catalog (runtime via Converse + per-region mantle via
``bedrock-mantle.{region}.api.aws``), then applies LiteLLM-backed pricing
enrichment (:mod:`app.services.bedrock_pricing`) and the cheapest-first sort
the optimizer's "lift the cheapest viable model" strategy depends on.

Each returned entry carries six routing fields — ``model_id``,
``model_name``, ``provider``, ``endpoint``, ``api``, ``region`` —
plus ``input_price_per_1m`` / ``output_price_per_1m`` when LiteLLM
covers the id. Mantle ids are matched by
:func:`app.services.bedrock_pricing._match_mantle_id`; an id LiteLLM
doesn't carry sinks to the end of the list via the "unknown→inf" sort
key rather than being priced at a guess.

The discovery layer also persists per-project mantle descriptors as a
side-effect of listing, which is why ``project_id`` is required:
descriptors are project-scoped and there is no cross-project fallback.
"""

from __future__ import annotations

import logging
from typing import Any

from claude_agent_sdk import ToolAnnotations, tool

from app.services.bedrock_pricing import NA, _NaSentinel, fetch_model_pricing
from app.services.mcp_response import mcp_error, mcp_json
from app.services.project_paths import ProjectTraversalError, resolve_project_dir

logger = logging.getLogger(__name__)


async def _enrich_with_pricing(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach per-1M-token pricing to each model entry (mutates in place).

    Fetches pricing from the LiteLLM price list (cached in-process).
    If the fetch fails, returns the original list unchanged.

    Args:
        models: List of model dicts with at least a ``model_id`` key. Each
            dict is mutated in place.

    Returns:
        The same list reference, now with ``input_price_per_1m`` and
        ``output_price_per_1m`` fields added where pricing data is
        available.
    """
    try:
        model_ids = [m["model_id"] for m in models]
        pricing = await fetch_model_pricing(model_ids)
        price_map = pricing.get("models", {})

        for model in models:
            prices = price_map.get(model["model_id"])
            # NA is the sentinel for "mantle id with no LiteLLM data" —
            # convert it to the literal wire string ``"n/a"`` here, so the
            # agent reading this tool's response sees an explicit "no price"
            # rather than a missing field. The conversion happens at exactly
            # this MCP boundary; backend-internal callers see the typed
            # sentinel.
            if isinstance(prices, _NaSentinel):
                model["input_price_per_1m"] = "n/a"
                model["output_price_per_1m"] = "n/a"
            elif prices:
                model["input_price_per_1m"] = prices.get("input")
                model["output_price_per_1m"] = prices.get("output")
    except Exception:
        logger.warning("Failed to enrich models with pricing", exc_info=True)

    return models


def _sort_by_price(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort models cheapest→most expensive by per-1M token price (in place).

    Deterministic code sort — NOT an LLM judgment — so the order is stable
    and reproducible. This is the canonical cheapest-first ordering the
    optimizer's "lift the cheapest model within reach" strategy reads: the
    coordinator copies IDs from this already-sorted list, the harness
    writes ``config.models`` in this order, and the optimizer receives
    per-model pass rates in this order.

    Ordering (priced → unknown → n/a):
    * Ascending by ``input_price_per_1m``, tie-broken by
      ``output_price_per_1m`` (input dominates eval cost — prompt tokens
      are sent for every case; output is usually a short answer).
    * Models with unknown/missing price (``None``, missing key — the
      runtime "cache cold or no LiteLLM coverage" signal) sort BELOW
      priced models, with their incoming relative order preserved
      (Python's sort is stable).
    * Models with explicit ``"n/a"`` (the wire string set by
      :func:`_enrich_with_pricing` from the :data:`bedrock_pricing.NA`
      sentinel — a mantle id with no LiteLLM data) sink BELOW
      unknown. ``"n/a"`` is the strongest "do not auto-pick" signal.
      Manual selection is unaffected — :func:`set_models` validates
      against descriptors, not pricing.

    Conscious trade-off: this replaces the previous ``(provider,
    model_name)`` grouping. Price-ascending is more useful for the
    coordinator's Phase 2 (Configuration) cost/quality table and is required for the
    cheapest-first strategy; the provider grouping is given up
    deliberately. Judge-model selection is unaffected — the coordinator
    picks the judge separately and is not required to take the cheapest
    target as judge.

    Args:
        models: Model dicts, each optionally carrying
            ``input_price_per_1m`` / ``output_price_per_1m``.

    Returns:
        The same list reference, sorted in place.
    """
    # 3-tuple: (is_na, input_price_or_inf, output_price_or_inf).
    # ``is_na=1`` for entries whose input OR output is the literal string
    # ``"n/a"`` — these sink BELOW unknown-priced ones. Numeric prices
    # compare directly; non-numeric (None, missing, ``"n/a"``) get
    # ``float("inf")`` so the tuple is well-ordered. Stable sort means
    # equally-priced and equally-n/a entries preserve their incoming order.
    def _key(m: dict[str, Any]) -> tuple[int, float, float]:
        inp = m.get("input_price_per_1m")
        out = m.get("output_price_per_1m")
        is_na = 1 if (inp == "n/a" or out == "n/a") else 0
        return (
            is_na,
            inp if isinstance(inp, (int, float)) else float("inf"),
            out if isinstance(out, (int, float)) else float("inf"),
        )

    models.sort(key=_key)
    return models


# ``discover_models`` is imported lazily inside the handler to avoid an
# early import cycle: ``mcp_tools/models.py`` is reached very early via
# ``mcp_tools/__init__.py`` → ``agent_runner.py``, while
# ``bedrock_discovery.py`` pulls in ``BedrockMantleModel`` (which imports
# DeepEval). Pushing the import inside the handler keeps the
# DeepEval cold-start cost off the agent_runner critical path.


@tool(
    "list_bedrock_models",
    "List available models — runtime (Converse) AND mantle (bedrock-mantle.*.api.aws) — "
    "merged into a single catalog with endpoint/api/region per id. "
    "ONE ROW PER MODEL: cross-Region variants of the same model (bare, us., "
    "global.) are collapsed into the cheapest one, and the others appear under "
    "``variants`` with their OWN prices and ``routing_scope``. Quote a price only "
    "from the row or variant whose model_id you are naming — geographic (us.) "
    "routing costs ~10% more than base/global, so the numbers are NOT "
    "interchangeable. If a tenant needs geographic data residency, pick the us. "
    "variant explicitly and say that it costs more. "
    "A row carrying ``superseded_by`` has a newer generation of the same model "
    "line available at no higher price: do NOT propose that row — as target or "
    "as judge — without stating why the newer one is unsuitable. Absence of the "
    "field is not a guarantee of currency (a generation that is not a trailing "
    "number, like nova-2-lite, is never compared). "
    "RESPONSE SHAPE: normally a JSON ARRAY of model entries. If any model had to "
    "be excluded, the response is instead an OBJECT with ``models`` (the same "
    "array) plus ``excluded_models`` — models this account has that we could not "
    "determine how to call. An exclusion is a gap on our side, not the model "
    "being unavailable, so mention any that look relevant to the task. Never "
    "pass an excluded id to set_models. "
    "Entries with ``endpoint: \"mantle\"`` are models RUNTIME CANNOT SERVE — the "
    "catalog is runtime-first, so a mantle row is the only way to reach that "
    "model, not a second-class alternative. "
    "Sorted CHEAPEST-FIRST by per-1M-token price (input ascending, then output; "
    "unknown-price models last). Use this to discover valid model IDs before "
    "running evaluations and BEFORE calling set_models. ALWAYS call this "
    "instead of guessing model IDs. When you select target models, preserve "
    "this cheapest-first order so the optimizer can lift the cheapest viable model. "
    "Optionally filter by provider (e.g., 'Anthropic', 'Amazon', 'OpenAI', 'XAI').",
    {
        "type": "object",
        "properties": {
            "project_id": {
                "type": "string",
                "description": (
                    "Project UUID. REQUIRED — discovery persists per-project "
                    "mantle descriptors as a side-effect of listing."
                ),
            },
            "provider": {
                "type": "string",
                "description": (
                    "Optional provider filter (e.g., 'Anthropic', 'Amazon', "
                    "'OpenAI', 'XAI'). Case-insensitive substring match against "
                    "the provider field of each entry."
                ),
            },
        },
        "required": ["project_id"],
    },
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_bedrock_models(args: dict[str, Any]) -> dict[str, Any]:
    """List the merged runtime + mantle catalog with pricing + cheapest-first sort.

    Calls :func:`app.services.bedrock_discovery.discover_models` to get the
    authoritative merged catalog (each entry already tagged with
    ``endpoint``/``api``/``region``), applies the optional provider filter,
    then enriches with LiteLLM pricing and sorts cheapest-first. Mantle
    descriptors are persisted under the project lock as a side-effect of
    discovery.

    Args:
        args: Dict with required ``project_id`` and optional ``provider``
            substring filter (case-insensitive).

    Returns:
        ``mcp_json`` listing of model entries on success;
        ``mcp_error`` on validation failure or discovery error so the
        agent loop survives instead of dying mid-turn.
    """
    project_id = args.get("project_id") if isinstance(args, dict) else None
    if not isinstance(project_id, str) or not project_id:
        return mcp_error(
            "project_id is required for list_bedrock_models — "
            "descriptors are persisted per project."
        )

    try:
        project_dir = resolve_project_dir(project_id)
    except ProjectTraversalError as exc:
        return mcp_error(f"Invalid project_id: {exc}")

    # Lazy import — see module docstring for why this lives inside the
    # handler instead of at import time.
    from app.services.bedrock_discovery import discover_models

    try:
        result = await discover_models(project_id, project_dir)
    except Exception as exc:  # noqa: BLE001 — surface as mcp_error so agent survives
        logger.exception("[list_bedrock_models] discovery failed")
        return mcp_error(f"Discovery failed: {exc}")

    # Project the CatalogEntry TypedDict into the plain dict shape the
    # MCP response uses. Copy each entry so subsequent in-place pricing
    # enrichment doesn't mutate the discovery layer's TypedDicts.
    entries: list[dict[str, Any]] = [dict(e) for e in result.catalog]

    # Apply provider filter BEFORE pricing so we don't fetch prices for
    # entries we'll just drop. Case-insensitive substring match against
    # the entry's ``provider`` field (already populated by discovery for
    # both runtime and mantle entries).
    provider_filter = args.get("provider") if isinstance(args, dict) else None
    if isinstance(provider_filter, str) and provider_filter:
        needle = provider_filter.lower()
        entries = [
            e for e in entries
            if needle in str(e.get("provider", "")).lower()
        ]

    enriched = await _enrich_with_pricing(entries)
    # Normalize BEFORE the sort, because the sort is what the two measured
    # defects came out of. Cross-Region variants of one model carry two
    # different pricing bases (geographic ``us.`` sits 10% above bare/global),
    # so cheapest-first was comparing +10% prices against base prices and the
    # agent quoted one variant's price while naming another's id. Collapsing
    # leaves one row per model, whose price belongs to its own id, with the
    # alternates listed so choosing geographic routing for data residency stays
    # a visible decision. Supersession is annotated after the collapse so a
    # model is compared once, on one basis.
    from app.services.model_catalog import annotate_supersession, collapse_variants

    normalized = annotate_supersession(collapse_variants(enriched))
    sorted_models = _sort_by_price(normalized)

    # A model the probe could not route is dropped from the catalog, and a
    # dropped model has NO symptom — it is simply absent from a list. Under
    # runtime-first an exclusion can only ever be a mantle-ONLY model, so each
    # one is a routing gap on our side rather than the model being unavailable
    # to this account. Reported here because this response is the only place a
    # reader looks; the WARNING it also writes goes to CloudWatch, which nobody
    # reads while choosing models.
    if result.excluded:
        return mcp_json({
            "models": sorted_models,
            "excluded_models": [
                {"model_id": mid, "reason": reason}
                for mid, reason in sorted(result.excluded.items())
            ],
            "excluded_note": (
                "These models are available to this account but we could not "
                "determine how to call them, so they are NOT in the list above. "
                "That is a gap on our side. Mention them to the user if one "
                "looks relevant to the task."
            ),
        })
    return mcp_json(sorted_models)
