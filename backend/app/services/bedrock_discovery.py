# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single discovery orchestrator for the runtime + mantle model catalog.

Runtime listing + per-region mantle listing in parallel, RUNTIME-first
precedence, capability probe, and descriptor persistence under the project
lock. Mantle is retained for models runtime cannot serve — see
:func:`_merge_runtime_first` for the precedence rule and why it reversed.

Knowledge MCP is intentionally NOT called from in-process platform code:
the capability question is answered vacuously by the authoritative
``400 "does not support"`` probe response. The agent tool loop in
``services/agent_runner.py`` is the only call site that reaches MCP
tools; this module never imports from that surface.

Public surface:

  * ``discover_models(project_id, project_dir) -> DiscoveryResult`` —
    the orchestrator. Async — awaitable from FastAPI handlers.
  * ``DiscoveryResult`` — frozen dataclass with ``catalog`` (list of
    ``CatalogEntry``) and ``descriptors`` (map id → ``MantleDescriptor``).
  * ``MantleDiscoveryError`` — base named error. Its "every candidate API
    definitively rejected the id" branch is the LIVE exclusion path.
  * ``MantleMapperUnsupported`` — raised only if every probe attempt hit
    ``NotImplementedError``, which no shipped mapper does; see its docstring.

Internal helpers exposed for test introspection:

  * ``_resolve_discovery_regions`` — env-driven region list.
  * ``_list_runtime_catalog`` — sync boto3 path (mirrors
    ``mcp_tools/models.py``'s inference-profiles → foundation-models flow).
  * ``_list_mantle_in_region`` — async per-region SigV4 GET wrapper.
  * ``_merge_runtime_first`` — pure dedup/precedence function.
  * ``_probe_order_for`` — heuristic-ordered API sweep.
  * ``_probe_capability`` — async sweep + adapter probe.
  * ``_get_shared_async_mantle_client`` — module-level singleton
    ``httpx.AsyncClient`` for the listing path.

Design decisions:

  * Runtime entries set ``endpoint="runtime"``, ``api="converse"``,
    ``region=None``. Mantle entries set ``endpoint="mantle"``,
    ``api=<resolved>``, ``region=<listing region>``. Descriptor map only
    carries mantle entries.
  * Probe heuristic by id-prefix; the clean ``400`` advances the sweep.
    ``NotImplementedError`` from a stub mapper continues, and if every attempt
    were a stub the sweep would raise ``MantleMapperUnsupported`` — a guard for
    a future stub, not a live path (all three mappers are implemented).
  * Throwaway ``BedrockMantleModel`` per probe attempt — no
    accumulated_usage pollution.
  * Per-attempt 20s ``asyncio.wait_for`` budget.
  * NO in-process Knowledge MCP.
  * ``MANTLE_DISCOVERY_REGIONS`` env semantics — unset →
    fallback list; ``"all"`` → ``account.list_regions``; CSV → as-is.
  * Discover_models does NOT acquire ``get_project_lock``
    directly; the persistence wrapper does.
  * Existing descriptors are sticky (no re-probe).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import time
from pathlib import Path
from typing import TypedDict

import boto3
import certifi
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from app.services.aws_retry import _mantle_retry
from app.services.bedrock_mantle_mappers import (
    MantleAPIRejected,
    MantleContentSafetyBlocked,
    MantleReasoningTruncated,
    MantleSchemaDrift,
    MantleTemperatureUnsupported,
)
from app.services.bedrock_mantle_model import (
    BedrockMantleModel,
    MantleSignatureMismatch,
)
from app.services.model_descriptors import (
    MantleAPIName,
    MantleDescriptor,
    MantleEndpointKind,
    MantleProbeStatus,
    read_descriptors,
)
from app.services.model_descriptors_async import persist_descriptors_locked


logger = logging.getLogger(__name__)


# Per-attempt timeout for the probe call. Three APIs × 20s = 60s worst
# case per id. Tests can monkeypatch this constant for speed.
_PROBE_TIMEOUT_S: float = 20.0


# Probe ``max_output_tokens``. Live behavior verified 2026-06-19:
#   * 1 — bedrock-mantle responses 400's with a min-value validation
#         BEFORE the wrong-api validation_error can surface, so the
#         capability sweep is broken for chat/anthropic models.
#   * 16 — API minimum, but reasoning models (gpt-5.5, grok-4.3) burn the
#          entire budget on hidden reasoning tokens and truncate before
#          emitting visible output → MantleReasoningTruncated.
#   * 64 — verified working: reasoning probe completes with a token-cap
#          truncation that's safely treated as a successful api match.
_PROBE_MAX_OUTPUT_TOKENS: int = 64


# Per-region listing client connect/read timeouts. The probe path uses
# its own ``asyncio.wait_for`` budget, so this only governs the GETs.
_LISTING_CONNECT_TIMEOUT_S: float = 10.0
_LISTING_READ_TIMEOUT_S: float = 30.0


# Module-level singleton ``httpx.AsyncClient`` for the per-region GETs.
# Lazy: built on first call so the constructor binds to the active event
# loop. The lock itself is also lazy because ``asyncio.Lock()`` at
# import time binds to no loop and trips RuntimeError on async use.
_shared_async_client: httpx.AsyncClient | None = None
_shared_async_lock: asyncio.Lock | None = None


# Default region list when MANTLE_DISCOVERY_REGIONS is unset/empty AND
# also the account-listing fallback when ``account.list_regions`` fails.
# Single source so the only ``"us-east-1"`` LITERAL in this file lives
# inside this constant — every other reference uses the canonical 3-region
# default through this name.
_DEFAULT_DISCOVERY_REGIONS: tuple[str, ...] = (
    "us-east-1",
    "us-west-2",
    "eu-west-1",
)


# The region passed to ``boto3.client(...)`` for the ``bedrock`` and
# ``account`` services. Both APIs are global, so any enabled region
# works; the account API is only available in a handful of opt-in
# regions and us-east-1 is one of them. Defined as a constant so the
# us-east-1 literal does not appear in code paths.
_BOTO3_GLOBAL_CLIENT_REGION: str = _DEFAULT_DISCOVERY_REGIONS[0]


# --------------------------------------------------------------------------- #
# Catalog + result types                                                       #
# --------------------------------------------------------------------------- #


class CatalogEntry(TypedDict):
    """A single discovered model entry in the merged catalog.

    Runtime entries: ``endpoint="runtime"``, ``api="converse"``,
    ``region=None``. Mantle entries: ``endpoint="mantle"``, ``api`` is
    None until probe-resolved (final entries always carry the resolved
    API), ``region`` is the listing region.
    """

    model_id: str
    model_name: str
    provider: str
    endpoint: MantleEndpointKind
    api: MantleAPIName | None
    region: str | None


@dataclasses.dataclass(frozen=True)
class DiscoveryResult:
    """Outcome of a single ``discover_models`` invocation.

    ``catalog`` is the merged runtime + mantle entry list (consumed by
    the ``list_bedrock_models`` tool). ``descriptors``
    maps the mantle ids to their persisted ``MantleDescriptor`` —
    runtime entries do NOT appear in this map.

    ``excluded`` names the mantle ids the probe could not resolve, each with the
    reason. Under runtime-first these are always mantle-ONLY models, so an
    exclusion is a COVERAGE HOLE ON OUR SIDE — the model exists and the tenant
    is entitled to it, we could not establish which API it speaks. That is worth
    telling someone: a dropped model has no symptom, it is simply absent from a
    list, so a WARNING in CloudWatch is read by nobody and the agent silently
    plans around a model it should have been offered. ``list_bedrock_models``
    puts these in its response so the agent relays them.
    """

    catalog: list[CatalogEntry]
    descriptors: dict[str, MantleDescriptor]
    excluded: dict[str, str] = dataclasses.field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Named errors                                                                 #
# --------------------------------------------------------------------------- #


class MantleDiscoveryError(Exception):
    """Base class for named discovery failures.

    Subclasses cover the two actionable cases (mapper unsupported,
    probe sweep exhausted). Generic discovery failures with no
    actionable subclass surface as this base.
    """


class MantleMapperUnsupported(MantleDiscoveryError):
    """Every probe attempt hit ``NotImplementedError``.

    **CURRENTLY UNREACHABLE IN PRODUCTION, and kept deliberately.** All three
    mappers are implemented (``_build_responses_request``,
    ``_build_chat_completions_request``, ``_build_anthropic_messages_request``
    in ``bedrock_mantle_mappers``) and ``BedrockMantleModel._BUILDERS`` maps
    every ``MantleAPIName`` to one of them, so nothing raises
    ``NotImplementedError`` on the probe path. An earlier version of this
    docstring claimed only ``responses`` was live and the other two were stubs —
    that was already false and it misrepresented mantle coverage, which is worth
    more than the exception itself.

    It stays because ``_BUILDERS`` is a registry: adding a fourth API as a stub
    is a normal way to extend it, and this is the branch that then drops an id
    that speaks none of the implemented APIs instead of stalling discovery. The
    tests reach it by monkeypatching ``_BUILDERS``, which is exactly that future
    shape.

    The exclusion path that DOES fire live is the base
    :class:`MantleDiscoveryError` "every candidate API returned the definitive
    wrong-API validation_error" branch. Under runtime-first an exclusion can
    only ever remove a mantle-ONLY id — see :func:`_merge_runtime_first`.
    """

    def __init__(self, model_id: str):
        super().__init__(
            f"{model_id}: every mantle mapper raised NotImplementedError"
        )
        self.model_id = model_id


# --------------------------------------------------------------------------- #
# Shared async client                                                          #
# --------------------------------------------------------------------------- #


async def _get_shared_async_mantle_client() -> httpx.AsyncClient:
    """Return the module-level singleton ``httpx.AsyncClient`` for listing GETs.

    Mirrors ``bedrock_mantle_model._get_shared_mantle_client`` env knobs
    (``MANTLE_MAX_CONNECTIONS`` / ``MANTLE_MAX_KEEPALIVE``) and the
    ``verify=certifi.where()`` rule for ``*.api.aws``.
    Connect timeout 10s, read timeout 30s — listing GETs return small
    JSON quickly; the probe path uses its own ``asyncio.wait_for``.

    Async-safe via lazy ``asyncio.Lock`` + double-checked construction.
    Using ``asyncio.Lock()`` at module import binds to no loop and
    raises RuntimeError on ``async with`` from a real loop later, so
    the lock itself is also lazy.
    """
    global _shared_async_client, _shared_async_lock
    if _shared_async_client is not None:
        return _shared_async_client
    if _shared_async_lock is None:
        _shared_async_lock = asyncio.Lock()
    async with _shared_async_lock:
        if _shared_async_client is None:
            max_connections = int(os.environ.get("MANTLE_MAX_CONNECTIONS", "20"))
            max_keepalive = int(os.environ.get("MANTLE_MAX_KEEPALIVE", "10"))
            _shared_async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=_LISTING_CONNECT_TIMEOUT_S,
                    read=_LISTING_READ_TIMEOUT_S,
                    write=30.0,
                    pool=30.0,
                ),
                limits=httpx.Limits(
                    max_connections=max_connections,
                    max_keepalive_connections=max_keepalive,
                ),
                verify=certifi.where(),
                http2=False,
            )
    return _shared_async_client


# --------------------------------------------------------------------------- #
# Region resolver                                                              #
# --------------------------------------------------------------------------- #


def _list_account_regions() -> list[str]:
    """Return enabled-region names via ``boto3.client("account").list_regions``.

    Falls back to the 3-region default on any boto3 error. Surfacing a
    real-world failure as the fallback list (rather than as an
    exception) is intentional: an account.list_regions failure must
    NEVER block discovery — at worst we miss a region the operator
    explicitly opted in to, and that surfaces as "no models in that
    region" downstream.
    """
    try:
        client = boto3.client("account", region_name=_BOTO3_GLOBAL_CLIENT_REGION)
        resp = client.list_regions(
            RegionOptStatusContains=["ENABLED", "ENABLED_BY_DEFAULT"]
        )
        return [r["RegionName"] for r in resp.get("Regions", [])]
    except Exception as exc:  # noqa: BLE001 — fallback contract
        logger.warning(
            "[discovery] account.list_regions failed: %s — falling back to 3-region default",
            exc,
        )
        return list(_DEFAULT_DISCOVERY_REGIONS)


def _resolve_discovery_regions() -> list[str]:
    """Resolve the per-region listing scan set from ``MANTLE_DISCOVERY_REGIONS``.

    Semantics:
      * unset / empty → fallback list ``_DEFAULT_DISCOVERY_REGIONS`` (the
        only canonical region tuple in this module).
      * ``"all"`` (case-insensitive) → ``_list_account_regions()``.
      * CSV → split on ``,``, strip whitespace, drop empties; preserve
        order.
    """
    raw = os.environ.get("MANTLE_DISCOVERY_REGIONS", "").strip()
    if not raw:
        return list(_DEFAULT_DISCOVERY_REGIONS)
    if raw.lower() == "all":
        return _list_account_regions()
    return [r.strip() for r in raw.split(",") if r.strip()]


# --------------------------------------------------------------------------- #
# Runtime listing                                                              #
# --------------------------------------------------------------------------- #


def _list_runtime_catalog_in_region(region: str) -> list[CatalogEntry]:
    """Return the runtime (Converse) catalog for ONE region.

    Sync — boto3 calls are blocking. Inference profiles take precedence;
    foundation models are the fallback when the profile listing returns
    nothing usable. Each entry is tagged with the listing region so the
    multi-region merger can dedup and report origin.

    The fallback is filtered to ON_DEMAND models: a bare foundation-model id
    for an inference-profile-only model is not invocable, so listing it would
    advertise a model id that fails at invoke time. See the inline note.
    """
    client = boto3.client("bedrock", region_name=region)

    profile_models: list[CatalogEntry] = []
    try:
        resp = client.list_inference_profiles(maxResults=1000)
        for p in resp.get("inferenceProfileSummaries", []):
            if p.get("type") != "SYSTEM_DEFINED":
                continue
            profile_id = p.get("inferenceProfileId", "")
            name = p.get("inferenceProfileName", "")
            provider = profile_id.split(".")[1] if "." in profile_id else ""
            profile_models.append(
                CatalogEntry(
                    model_id=profile_id,
                    model_name=name,
                    provider=provider,
                    endpoint="runtime",
                    api="converse",
                    region=region,
                )
            )
    except Exception as exc:  # noqa: BLE001 — fall through to foundation models
        logger.warning(
            "[discovery] list_inference_profiles region=%s failed: %s — "
            "falling back to list_foundation_models",
            region,
            exc,
        )

    if profile_models:
        return profile_models

    try:
        resp = client.list_foundation_models()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[discovery] list_foundation_models region=%s failed: %s — "
            "region contributes no runtime entries",
            region,
            exc,
        )
        return []

    models: list[CatalogEntry] = []
    not_on_demand = 0
    for m in resp.get("modelSummaries", []):
        output_modalities = m.get("outputModalities", [])
        if "TEXT" not in output_modalities:
            continue
        # A bare foundation-model id is only invocable when the model supports
        # ON_DEMAND throughput. Verified live 2026-09-06 against us-east-1:
        # many TEXT models from ``list_foundation_models`` are NOT on-demand
        # (``["INFERENCE_PROFILE"]``, ``["PROVISIONED"]``, or empty), and
        # Converse on those bare ids fails with
        # ``ValidationException: Invocation of model ID <id> with on-demand
        # throughput isn't supported`` — including every current Claude model.
        # Those ids must be reached through their inference profile
        # (``us.``/``global.``) instead, which the primary listing above
        # already returns. Emitting them here put ids in the catalog that the
        # agent could select and set_models would accept, failing only later
        # at invoke time.
        inference_types = m.get("inferenceTypesSupported")
        if inference_types is not None and "ON_DEMAND" not in inference_types:
            not_on_demand += 1
            continue
        models.append(
            CatalogEntry(
                model_id=m["modelId"],
                model_name=m.get("modelName", ""),
                provider=m.get("providerName", ""),
                endpoint="runtime",
                api="converse",
                region=region,
            )
        )
    if not_on_demand:
        logger.info(
            "[discovery] region=%s foundation-model fallback dropped %d "
            "inference-profile-only/provisioned-only model(s) — their bare ids "
            "are not invocable on demand",
            region,
            not_on_demand,
        )
    return models


def _list_runtime_catalog() -> list[CatalogEntry]:
    """Return the runtime (Converse) catalog across all discovery regions.

    Bedrock runtime is per-region: a foundation model available only in
    a non-default region was previously missed because the listing was
    pinned to ``_BOTO3_GLOBAL_CLIENT_REGION`` (us-east-1). This iterates
    the same ``_resolve_discovery_regions()`` set that the mantle path
    uses and dedups by ``model_id``.

    Dedup contract: cross-region inference profiles (``us.*``,
    ``global.*``) collapse into a single entry; the first region's
    tagging wins (subsequent duplicates are dropped). Region-specific
    foundation models (no regional prefix) are picked up wherever they
    surface and tagged with that region. Boto3 calls are blocking so
    the orchestrator wraps this in ``asyncio.to_thread``.
    """
    regions = _resolve_discovery_regions()
    seen: set[str] = set()
    merged: list[CatalogEntry] = []
    for region in regions:
        for entry in _list_runtime_catalog_in_region(region):
            mid = entry["model_id"]
            if mid in seen:
                continue
            seen.add(mid)
            merged.append(entry)
    return merged


# --------------------------------------------------------------------------- #
# Per-region mantle listing                                                    #
# --------------------------------------------------------------------------- #


async def _list_mantle_in_region(region: str) -> list[CatalogEntry]:
    """Return the mantle catalog for one region via SigV4 ``GET /v1/models``.

    Async — uses the module-level shared ``httpx.AsyncClient``. SigV4
    signing is sync (botocore's signer doesn't have an async API); the
    request is signed in-line and the GET is awaited.

    Contract: a single bad region must NOT abort the gather. Any
    httpx/HTTPX-status error is caught here, logged at WARNING, and
    converted to an empty list.

    The live response is OpenAI-compatible — shape
    ``{"object": "list", "data": [{"id": "...", "object": "model", ...}, ...]}``.
    An earlier design described the key as ``"models"``; verified
    live (2026-06-19) it's ``"data"``. We extract the id list and tag
    every entry with the listing region.
    """
    url = f"https://bedrock-mantle.{region}.api.aws/v1/models"
    try:
        session = boto3.Session()
        creds_provider = session.get_credentials()
        if creds_provider is None:
            logger.warning(
                "[discovery] mantle list region=%s: no AWS credentials available — region contributes no entries",
                region,
            )
            return []
        creds = creds_provider.get_frozen_credentials()
        req = AWSRequest(method="GET", url=url, data=b"")
        SigV4Auth(creds, "bedrock-mantle", region).add_auth(req)

        client = await _get_shared_async_mantle_client()
        response = await client.get(url, headers=dict(req.headers))
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, httpx.HTTPStatusError) as exc:
        logger.warning(
            "[discovery] mantle list region=%s failed: %s — region contributes no entries",
            region,
            exc,
        )
        return []
    except Exception as exc:  # noqa: BLE001 — same contract as the httpx branch
        logger.warning(
            "[discovery] mantle list region=%s unexpected error: %s — region contributes no entries",
            region,
            exc,
        )
        return []

    entries: list[CatalogEntry] = []
    for m in body.get("data", []) if isinstance(body, dict) else []:
        if not isinstance(m, dict):
            continue
        model_id = m.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        provider = model_id.split(".", 1)[0]
        entries.append(
            CatalogEntry(
                model_id=model_id,
                model_name=model_id,
                provider=provider,
                endpoint="mantle",
                api=None,
                region=region,
            )
        )
    return entries


# --------------------------------------------------------------------------- #
# Runtime-first precedence                                                     #
# --------------------------------------------------------------------------- #


def _merge_runtime_first(
    runtime: list[CatalogEntry],
    per_region: list[list[CatalogEntry]],
) -> tuple[list[CatalogEntry], list[CatalogEntry]]:
    """Dedup runtime + per-region mantle lists; **runtime** wins on collision.

    Runtime is the default path. This reverses an earlier mantle-first
    precedence, and mantle is NOT abandoned — some models are available only
    there, so a mantle entry with no runtime collision keeps its catalog entry
    and its descriptor.

    WHY RUNTIME-FIRST IS SAFE. Mantle-first existed because of two failure
    modes, and both are gone:

      1. The reasoning-model temperature gate is fixed.
         ``bedrock_runtime_model.py`` is now response-driven: it attempts with
         ``temperature``, catches a ``ValidationException`` that names
         temperature, drops the param, retries once, and latches
         ``_supports_temperature_flag`` under a lock. There is no static model
         list, and ``is_temperature_capability_reject`` separates a capability
         rejection from an out-of-range value so a genuine caller-config bug
         still propagates. That gate caused the silent null scores and the
         retry storm.
      2. The ``config.py`` / ``model_descriptors.json`` id-form mismatch stops
         being a bug. Under mantle-first, resolving a runtime-form id against a
         mantle-keyed descriptor map was an accidental fallthrough; under
         runtime-first, routing to runtime IS the intent.

    Same-model detection is unchanged and still correct: the structural
    ``short_model_id`` rule (``app.services.model_id_short``) strips leading
    dot-segments until the first segment containing a digit, so runtime
    ``global.anthropic.claude-opus-4-8`` and mantle
    ``anthropic.claude-opus-4-8`` both reduce to ``claude-opus-4-8``. No
    hardcoded provider or region list.

    NON-COLLIDING ENTRIES ARE UNTOUCHED IN BOTH DIRECTIONS. That is the whole
    mechanism preserving mantle-only coverage, and it is also what makes an
    exclusion safe: a mantle id can only be excluded downstream if it had no
    runtime counterpart, so excluding it can never remove a model runtime can
    serve. Under mantle-first that was not true — the runtime entry was dropped
    here, BEFORE the probe ran, so a later exclusion removed the model from the
    catalog entirely and a dropped model has no symptom, it is simply absent.

    Returns ``(merged_catalog, mantle_to_resolve)``:
      * ``merged_catalog``: every runtime entry first (incoming order), then
        mantle entries whose ``short_model_id`` is NOT already covered by a
        runtime entry.
      * ``mantle_to_resolve``: exactly the SURVIVING mantle entries. Probing a
        mantle id that runtime already serves would be wasted HTTP and would
        write a descriptor for an id no ``config.py`` can now contain.

    Within mantle entries the first occurrence wins, so region order in
    ``per_region`` is preserved end-to-end.

    Pure function: no I/O, no logging.
    """
    from app.services.model_id_short import short_model_id

    # First pass: the short ids runtime already covers.
    runtime_short_keys = {short_model_id(e["model_id"]) for e in runtime}

    # Second pass: keep mantle entries runtime does NOT already cover
    # (first-region-wins within mantle).
    surviving_mantle: list[CatalogEntry] = []
    seen_mantle_ids: set[str] = set()
    for region_entries in per_region:
        for entry in region_entries:
            mid = entry["model_id"]
            if mid in seen_mantle_ids:
                continue  # first-region-wins for mantle dedup
            seen_mantle_ids.add(mid)
            if short_model_id(mid) in runtime_short_keys:
                continue  # runtime equivalent wins
            surviving_mantle.append(entry)

    # Runtime first, then mantle-only: the catalog reads runtime-as-default,
    # and a mantle row now means "runtime cannot serve this model".
    merged = list(runtime) + surviving_mantle
    return merged, list(surviving_mantle)




# --------------------------------------------------------------------------- #
# Capability probe                                                             #
# --------------------------------------------------------------------------- #


def _probe_order_for(model_id: str) -> tuple[MantleAPIName, ...]:
    """Order the API sweep by id-prefix heuristic.

    The heuristic only ORDERS the sweep — every order ends up walking
    the same three APIs. The clean ``400 "does not support"`` from the
    wrong API is the authoritative non-match signal.
    """
    if model_id.startswith("openai.") or model_id.startswith("xai."):
        return ("responses", "chat_completions", "anthropic_messages")
    if model_id.startswith("anthropic."):
        return ("anthropic_messages", "responses", "chat_completions")
    return ("chat_completions", "responses", "anthropic_messages")


async def _probe_capability(
    model_id: str, region: str
) -> tuple[MantleAPIName, bool, MantleProbeStatus]:
    """Probe the mantle endpoint and resolve ``model_id`` permissively.

    Returns ``(api, supports_temperature, probe_status)``. PERMISSIVE
    discovery: the probe RESOLVES api/region/flags and KEEPS the model
    unless every candidate API definitively rejects it. The real gate
    is the dispatch preflight + the eval, not discovery.

    ``probe_status`` values:

      * ``"ok"`` — clean 2xx / temperature self-corrected / reasoning
        truncated / null-or-empty content / content-safety block. The
        api/region is resolved and the model invokes today.
      * ``"access_not_enabled"`` — 401/403 entitlement: the model is
        reachable but the account cannot invoke it. Descriptor kept
        with the resolved api so a future entitlement grant works.
      * ``"unverified"`` — transient (timeout / 5xx / network / schema
        drift). The api/region is best-effort; the next call (preflight
        or eval) will validate.

    Per-API outcome → action:

      * 2xx success — resolve, return ``"ok"``.
      * ``MantleReasoningTruncated`` — resolve, return ``"ok"`` (api
        confirmed; cap fired, which is normal for reasoning models).
      * ``MantleTemperatureUnsupported`` (escapes the adapter's temperature
        self-correction) — resolve with ``supports_temperature=False``.
      * ``MantleContentSafetyBlocked`` — resolve, return ``"ok"`` (the
        request reached the model; the model declined on policy).
      * ``MantleAPIRejected`` (definitive wrong-API
        ``validation_error``) — advance to next api.
      * ``MantleSignatureMismatch`` (403) — record an
        ``"access_not_enabled"`` candidate; advance.
      * ``NotImplementedError`` (stub mapper) — advance.
      * ``asyncio.TimeoutError`` / ``MantleSchemaDrift`` / generic
        exception — record an ``"unverified"`` candidate; advance.

    Resolution after the sweep:
      1. If any api resolved cleanly during the walk, the function
         already returned.
      2. If every api was a stub → ``MantleMapperUnsupported`` (the
         caller excludes the model — genuinely no support).
      3. If every api returned the definitive wrong-API signal →
         ``MantleDiscoveryError`` (the caller excludes the model —
         the live API surface confirms it doesn't speak any of the
         three).
      4. Otherwise, return the FIRST recorded entitlement candidate
         (preferred) or transient candidate, with the corresponding
         ``probe_status``. The descriptor is kept; the user sees the
         model.
    """
    order = _probe_order_for(model_id)
    all_unimplemented = True
    any_definitive_wrong_api = False

    # Recorded best-effort candidates for the after-sweep resolution.
    # Preference order at sweep end: entitlement > transient. Within
    # each kind the FIRST recorded candidate wins (preserves probe
    # heuristic order).
    entitlement_candidate: tuple[MantleAPIName, bool] | None = None
    unverified_candidate: tuple[MantleAPIName, bool] | None = None

    raw_generate = BedrockMantleModel.generate_with_system.__wrapped__

    for api in order:
        stub_descriptor: MantleDescriptor = {
            "endpoint": "mantle",
            "api": api,
            "region": region,
        }
        try:
            probe = BedrockMantleModel(
                stub_descriptor,
                model_id,
                max_output_tokens=_PROBE_MAX_OUTPUT_TOKENS,
            )
        except NotImplementedError as exc:
            logger.info(
                "[discovery] probe %s region=%s api=%s mapper_unimplemented (constructor): %s",
                model_id,
                region,
                api,
                exc,
            )
            continue
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[discovery] probe %s region=%s api=%s constructor failed: %s",
                model_id,
                region,
                api,
                exc,
            )
            all_unimplemented = False
            if unverified_candidate is None:
                unverified_candidate = (api, True)
            continue

        try:
            await asyncio.wait_for(
                asyncio.to_thread(raw_generate, probe, None, "Say OK."),
                timeout=_PROBE_TIMEOUT_S,
            )
            supports_temperature = bool(
                probe._descriptor.get("supports_temperature", True)
            )
            logger.info(
                "[discovery] probe %s region=%s api=%s ok "
                "(supports_temperature=%s)",
                model_id,
                region,
                api,
                supports_temperature,
            )
            return api, supports_temperature, "ok"
        except MantleReasoningTruncated as exc:
            # Truncation = api confirmed; cap fired. Permissive: keep.
            supports_temperature = bool(
                probe._descriptor.get("supports_temperature", True)
            )
            logger.info(
                "[discovery] probe %s region=%s api=%s ok "
                "(truncated at %d tokens — api confirmed; "
                "supports_temperature=%s)",
                model_id,
                region,
                api,
                _PROBE_MAX_OUTPUT_TOKENS,
                supports_temperature,
            )
            return api, supports_temperature, "ok"
        except MantleTemperatureUnsupported as exc:
            # Defensive: adapter normally swallows + retries internally.
            # If it ever escapes here, the api IS right; drop temperature.
            logger.info(
                "[discovery] probe %s region=%s api=%s rejected temperature "
                "(escaped adapter self-correction — keep with supports_temperature=False): %s",
                model_id,
                region,
                api,
                exc,
            )
            return api, False, "ok"
        except MantleContentSafetyBlocked as exc:
            # Permissive: the request reached the model and the model
            # declined on policy grounds. The api routing is correct;
            # the descriptor is keepable. Read the descriptor flag in
            # case temperature self-correction fired earlier in the same probe.
            supports_temperature = bool(
                probe._descriptor.get("supports_temperature", True)
            )
            logger.info(
                "[discovery] probe %s region=%s api=%s content-safety blocked "
                "(api reachable — keep): %s",
                model_id,
                region,
                api,
                exc,
            )
            return api, supports_temperature, "ok"
        except MantleAPIRejected as exc:
            # Definitive wrong-API — try the next api.
            logger.info(
                "[discovery] probe %s region=%s api=%s rejected (validation_error): %s",
                model_id,
                region,
                api,
                exc,
            )
            all_unimplemented = False
            any_definitive_wrong_api = True
            continue
        except MantleSignatureMismatch as exc:
            # 403 — entitlement / region drift. Permissive: keep.
            logger.info(
                "[discovery] probe %s region=%s api=%s entitlement-blocked "
                "(SignatureMismatch / not enabled — keep): %s",
                model_id,
                region,
                api,
                exc,
            )
            all_unimplemented = False
            if entitlement_candidate is None:
                entitlement_candidate = (api, True)
            continue
        except NotImplementedError as exc:
            logger.info(
                "[discovery] probe %s region=%s api=%s mapper_unimplemented: %s",
                model_id,
                region,
                api,
                exc,
            )
            continue
        except asyncio.TimeoutError as exc:
            logger.warning(
                "[discovery] probe %s region=%s api=%s timed out after %.1fs — keep as unverified: %s",
                model_id,
                region,
                api,
                _PROBE_TIMEOUT_S,
                exc,
            )
            all_unimplemented = False
            if unverified_candidate is None:
                unverified_candidate = (api, True)
            continue
        except MantleSchemaDrift as exc:
            # Schema drift in the parser is "reachable but I can't
            # parse this response shape" — permissive: keep as
            # unverified so the next eval validates.
            logger.warning(
                "[discovery] probe %s region=%s api=%s schema drift — keep as unverified: %s",
                model_id,
                region,
                api,
                exc,
            )
            all_unimplemented = False
            if unverified_candidate is None:
                supports_temperature = bool(
                    probe._descriptor.get("supports_temperature", True)
                )
                unverified_candidate = (api, supports_temperature)
            continue
        except Exception as exc:  # noqa: BLE001 — broad on purpose; permissive policy
            logger.warning(
                "[discovery] probe %s region=%s api=%s failed (%s) — keep as unverified: %s",
                model_id,
                region,
                api,
                type(exc).__name__,
                exc,
            )
            all_unimplemented = False
            if unverified_candidate is None:
                unverified_candidate = (api, True)
            continue

    # Sweep ended without a clean resolution. Apply the permissive
    # exclusion rules.
    if all_unimplemented:
        # No mapper supported the id at all.
        raise MantleMapperUnsupported(model_id)
    if any_definitive_wrong_api and (
        entitlement_candidate is None and unverified_candidate is None
    ):
        # Every API said "I don't speak this id" with the definitive
        # validation_error — exclude.
        raise MantleDiscoveryError(
            f"{model_id}: every candidate API returned the definitive "
            f"wrong-API validation_error in region {region}"
        )
    # Otherwise: keep with the best-effort candidate. Prefer
    # entitlement-blocked over transient.
    if entitlement_candidate is not None:
        api, supports_temperature = entitlement_candidate
        logger.info(
            "[discovery] probe %s region=%s resolved via entitlement candidate api=%s "
            "(probe_status=access_not_enabled)",
            model_id,
            region,
            api,
        )
        return api, supports_temperature, "access_not_enabled"
    assert unverified_candidate is not None  # one of the buckets must be filled
    api, supports_temperature = unverified_candidate
    logger.info(
        "[discovery] probe %s region=%s resolved via transient candidate api=%s "
        "(probe_status=unverified)",
        model_id,
        region,
        api,
    )
    return api, supports_temperature, "unverified"


# --------------------------------------------------------------------------- #
# Orchestrator                                                                 #
# --------------------------------------------------------------------------- #


async def discover_models(project_id: str, project_dir: Path) -> DiscoveryResult:
    """Discover the merged runtime + mantle catalog and persist mantle descriptors.

    Steps:

      1. Load existing descriptors. Existing entries are STICKY; they are
         NOT re-probed here. Re-resolution on stale-400/404 is left to
         preflight.
      2. List the runtime catalog via ``_list_runtime_catalog`` in a
         thread (boto3 is sync).
      3. Resolve the mantle scan regions.
      4. Per-region mantle listings via ``asyncio.gather`` — each
         helper catches its own errors so one bad region cannot abort
         the gather.
      5. Merge with RUNTIME-first precedence (:func:`_merge_runtime_first`).
      6. For each new SURVIVING mantle id — i.e. mantle-only models; runtime
         collisions were already resolved in favour of runtime and are not
         probed — probe its api. Update the catalog entry's ``api`` field and
         write a descriptor; on an unresolvable probe exclude the id from BOTH
         catalog and descriptors, and record it in ``result.excluded`` so the
         coverage hole reaches a reader.
      7. For mantle entries already present in ``existing``, patch
         their catalog ``api``/``region`` from the existing
         descriptor (so callers see a fully-tagged catalog).
      8. Persist the final descriptor map under the project lock.

    Args:
        project_id: Project UUID; scopes the per-project async lock.
        project_dir: Project root; descriptor file lives at
            ``{project_dir}/src/model_descriptors.json``.

    Returns:
        ``DiscoveryResult(catalog, descriptors, excluded)``.
    """
    existing = read_descriptors(project_dir)

    runtime = await asyncio.to_thread(_list_runtime_catalog)
    regions = _resolve_discovery_regions()

    mantle_lists = await asyncio.gather(
        *[_list_mantle_in_region(r) for r in regions]
    )

    merged_catalog, mantle_to_resolve = _merge_runtime_first(runtime, mantle_lists)
    descriptors: dict[str, MantleDescriptor] = dict(existing)

    # id → reason. Every id in here is mantle-ONLY (runtime collisions were
    # resolved in the merge and are never probed), so each one is a model the
    # tenant has and we could not route.
    excluded: dict[str, str] = {}

    for entry in mantle_to_resolve:
        mid = entry["model_id"]
        if mid in existing:
            # Sticky: patch the catalog entry from the existing
            # descriptor so the consumer sees the resolved api/region.
            stored = existing[mid]
            entry["api"] = stored["api"]
            entry["region"] = stored["region"]
            continue
        try:
            resolved, supports_temperature, probe_status = await _probe_capability(
                mid, entry["region"]
            )
        except MantleMapperUnsupported:
            logger.warning(
                "[discovery] excluding %s — every mapper raised NotImplementedError",
                mid,
            )
            excluded[mid] = (
                "no implemented mantle API mapper accepted this model"
            )
            continue
        except MantleDiscoveryError as exc:
            # Permissive policy: this is now reserved for "every
            # candidate API returned the definitive wrong-API
            # validation_error". Anything else (transient, schema drift,
            # entitlement) is kept inside _probe_capability with a flag.
            logger.warning(
                "[discovery] excluding %s — every API definitively rejected: %s",
                mid,
                exc,
            )
            excluded[mid] = (
                "every mantle API definitively rejected this model id, so we "
                "could not determine how to call it"
            )
            continue

        descriptor: MantleDescriptor = {
            "endpoint": "mantle",
            "api": resolved,
            "region": entry["region"],
        }
        # Only set the flag when False — keep the descriptor lean for
        # supports-temperature models (the adapter defaults to True
        # when the key is absent).
        if not supports_temperature:
            descriptor["supports_temperature"] = False
        # Only persist non-default probe_status. "ok" is the cheap
        # default and stays absent (mirrors supports_temperature).
        if probe_status != "ok":
            descriptor["probe_status"] = probe_status
        descriptors[mid] = descriptor
        entry["api"] = resolved

    if excluded:
        merged_catalog = [e for e in merged_catalog if e["model_id"] not in excluded]

    await persist_descriptors_locked(project_dir, descriptors, project_id)

    return DiscoveryResult(
        catalog=merged_catalog, descriptors=descriptors, excluded=excluded
    )
