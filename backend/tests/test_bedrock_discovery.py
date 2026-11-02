# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Bedrock model discovery orchestrator.

This file pins the discovery behaviors. Each one has at least one named
test below; the mapping is:

  * Module exports + orchestrator return shape
    → test_discover_models_returns_typed_catalog,
      test_discover_models_module_exports
  * Runtime listing tagging
    → test_runtime_listing_extracts_inference_profiles
  * Per-region SigV4 GET, parallelized
    → test_mantle_listing_per_region_parallel,
      test_mantle_listing_one_region_failure_does_not_abort_gather
  * MANTLE_DISCOVERY_REGIONS env semantics
    → test_mantle_discovery_regions_env_unset,
      test_mantle_discovery_regions_env_csv,
      test_mantle_discovery_regions_env_all
  * Runtime-first precedence
    → test_runtime_first_precedence_drops_mantle_dup
  * Capability-probe sweep
    → test_probe_heuristic_orders_sweep,
      test_probe_succeeds_first_attempt,
      test_probe_falls_through_400_to_next_api,
      test_probe_excludes_id_when_all_mappers_unimplemented
  * Descriptor persistence under project lock
    → test_persist_under_lock,
      test_concurrent_discover_serialized
  * Existing descriptors are sticky
    → test_existing_descriptor_not_reprobed
  * Catalog shape consumed by the model-listing tool
    → test_discover_models_returns_typed_catalog

Plus probe usage isolation and probe timeout:
  → test_probe_does_not_pollute_accumulated_usage,
    test_probe_timeout_via_wait_for

All tests use pytest-httpx for the mantle endpoint family and
botocore.stub.Stubber for the boto3 ``bedrock``/``account`` paths so no
real network or AWS service is touched.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import boto3
import httpx
import pytest
from botocore.stub import Stubber

from app.services import project_locks


FIXTURES = Path(__file__).parent / "fixtures" / "mantle_payloads"


# --------------------------------------------------------------------------- #
# Common fixtures                                                              #
# --------------------------------------------------------------------------- #


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _reset_locks():
    """Clear the project-lock registry between tests so per-test state is isolated."""
    project_locks._reset()
    yield
    project_locks._reset()


@pytest.fixture(autouse=True)
def _reset_shared_async_client():
    """Reset the discovery module's shared httpx.AsyncClient between tests."""
    # Imported lazily so a missing module fails the test that needs it,
    # not every test in the file.
    from app.services import bedrock_discovery

    bedrock_discovery._shared_async_client = None
    bedrock_discovery._shared_async_lock = None
    yield
    bedrock_discovery._shared_async_client = None
    bedrock_discovery._shared_async_lock = None


@pytest.fixture(autouse=True)
def _reset_mantle_singleton():
    """Reset the sync mantle adapter's singleton client between tests."""
    from app.services.bedrock_mantle_model import _close_shared_mantle_client

    _close_shared_mantle_client()
    yield
    _close_shared_mantle_client()


@pytest.fixture(autouse=True)
def _aws_creds(monkeypatch):
    """Static AWS credentials so SigV4 signing and boto3.Session.get_credentials succeed."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATESTACCESSKEY")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    """Shrink retry timing so the per-region listing's _mantle_retry doesn't sleep real seconds."""
    monkeypatch.setenv("BEDROCK_RETRY_INITIAL_S", "0.001")
    monkeypatch.setenv("BEDROCK_RETRY_CAP_S", "0.002")


# --------------------------------------------------------------------------- #
# Module exports                                                               #
# --------------------------------------------------------------------------- #


def test_discover_models_module_exports():
    """The module must export the symbols the listing tool consumes."""
    from app.services import bedrock_discovery as bd

    assert callable(bd.discover_models)
    assert hasattr(bd, "DiscoveryResult")
    assert hasattr(bd, "MantleDiscoveryError")
    assert hasattr(bd, "MantleMapperUnsupported")
    # Internal helpers exposed for unit tests below
    assert callable(bd._resolve_discovery_regions)
    assert callable(bd._list_runtime_catalog)
    assert callable(bd._list_mantle_in_region)
    assert callable(bd._merge_runtime_first)
    assert callable(bd._probe_order_for)
    assert callable(bd._probe_capability)
    assert callable(bd._get_shared_async_mantle_client)


# --------------------------------------------------------------------------- #
# MANTLE_DISCOVERY_REGIONS env semantics                                       #
# --------------------------------------------------------------------------- #


def test_mantle_discovery_regions_env_unset(monkeypatch):
    """Unset → fallback list."""
    from app.services.bedrock_discovery import _resolve_discovery_regions

    monkeypatch.delenv("MANTLE_DISCOVERY_REGIONS", raising=False)
    assert _resolve_discovery_regions() == ["us-east-1", "us-west-2", "eu-west-1"]


def test_mantle_discovery_regions_env_empty(monkeypatch):
    """Empty string → same fallback as unset."""
    from app.services.bedrock_discovery import _resolve_discovery_regions

    monkeypatch.setenv("MANTLE_DISCOVERY_REGIONS", "")
    assert _resolve_discovery_regions() == ["us-east-1", "us-west-2", "eu-west-1"]


def test_mantle_discovery_regions_env_csv(monkeypatch):
    """CSV → split, strip, drop empties, preserve order."""
    from app.services.bedrock_discovery import _resolve_discovery_regions

    monkeypatch.setenv(
        "MANTLE_DISCOVERY_REGIONS", " us-east-1, eu-central-1 ,, ap-south-1 "
    )
    assert _resolve_discovery_regions() == [
        "us-east-1",
        "eu-central-1",
        "ap-south-1",
    ]


def test_mantle_discovery_regions_env_all(monkeypatch):
    """``all`` (case-insensitive) → boto3.client('account').list_regions()."""
    from app.services import bedrock_discovery

    monkeypatch.setenv("MANTLE_DISCOVERY_REGIONS", "ALL")

    fake_response = {
        "Regions": [
            {"RegionName": "us-east-1"},
            {"RegionName": "us-west-2"},
            {"RegionName": "eu-west-1"},
            {"RegionName": "ap-southeast-2"},
            {"RegionName": "ca-central-1"},
        ]
    }

    def _fake_list_account_regions() -> list[str]:
        return [r["RegionName"] for r in fake_response["Regions"]]

    monkeypatch.setattr(
        bedrock_discovery,
        "_list_account_regions",
        _fake_list_account_regions,
    )
    regions = bedrock_discovery._resolve_discovery_regions()
    assert regions == [
        "us-east-1",
        "us-west-2",
        "eu-west-1",
        "ap-southeast-2",
        "ca-central-1",
    ]


# --------------------------------------------------------------------------- #
# probe heuristic ordering (pure)                                              #
# --------------------------------------------------------------------------- #


def test_probe_heuristic_orders_sweep():
    """ID-prefix heuristic orders the probe sweep."""
    from app.services.bedrock_discovery import _probe_order_for

    assert _probe_order_for("openai.gpt-5.5")[0] == "responses"
    assert _probe_order_for("xai.grok-4")[0] == "responses"
    assert _probe_order_for("anthropic.claude-3-5-sonnet")[0] == "anthropic_messages"
    assert _probe_order_for("mistral.large-2")[0] == "chat_completions"
    assert _probe_order_for("qwen.qwen3-coder")[0] == "chat_completions"
    # Heuristic only orders the sweep — every order ends up walking the
    # same three APIs.
    assert set(_probe_order_for("openai.gpt-5.5")) == {
        "responses",
        "chat_completions",
        "anthropic_messages",
    }


# --------------------------------------------------------------------------- #
# runtime listing extracts inference profiles                                  #
# --------------------------------------------------------------------------- #


def test_runtime_listing_extracts_inference_profiles(monkeypatch):
    """list_inference_profiles → entries tagged endpoint=runtime, api=converse, region=<scanned>."""
    from app.services import bedrock_discovery

    # Restrict the multi-region scan to one region so this test pins a
    # single boto3 call rather than fanning out across the fallback list.
    monkeypatch.setattr(
        bedrock_discovery, "_resolve_discovery_regions", lambda: ["us-east-1"]
    )

    fake_client = boto3.client("bedrock", region_name="us-east-1")
    stubber = Stubber(fake_client)
    stubber.add_response(
        "list_inference_profiles",
        {
            "inferenceProfileSummaries": [
                {
                    "inferenceProfileId": "us.anthropic.claude-3-7",
                    "inferenceProfileName": "Claude 3.7",
                    "inferenceProfileArn": (
                        "arn:aws:bedrock:us-east-1:0:inference-profile/"
                        "us.anthropic.claude-3-7"
                    ),
                    "models": [
                        {
                            "modelArn": (
                                "arn:aws:bedrock:us-east-1::foundation-model/"
                                "anthropic.claude-3-7"
                            )
                        }
                    ],
                    "status": "ACTIVE",
                    "type": "SYSTEM_DEFINED",
                },
                {
                    "inferenceProfileId": "us.amazon.nova-pro-v1:0",
                    "inferenceProfileName": "Nova Pro",
                    "inferenceProfileArn": (
                        "arn:aws:bedrock:us-east-1:0:inference-profile/"
                        "us.amazon.nova-pro-v1:0"
                    ),
                    "models": [
                        {
                            "modelArn": (
                                "arn:aws:bedrock:us-east-1::foundation-model/"
                                "amazon.nova-pro-v1:0"
                            )
                        }
                    ],
                    "status": "ACTIVE",
                    "type": "SYSTEM_DEFINED",
                },
            ]
        },
        {"maxResults": 1000},
    )
    stubber.activate()
    monkeypatch.setattr(boto3, "client", lambda *a, **k: fake_client)

    catalog = bedrock_discovery._list_runtime_catalog()

    assert len(catalog) == 2
    for entry in catalog:
        assert entry["endpoint"] == "runtime"
        assert entry["api"] == "converse"
        assert entry["region"] == "us-east-1"
    ids = {e["model_id"] for e in catalog}
    assert ids == {"us.anthropic.claude-3-7", "us.amazon.nova-pro-v1:0"}


def test_foundation_model_fallback_drops_ids_that_cannot_be_invoked(monkeypatch):
    """The fallback must not advertise bare ids that fail at invoke time.

    In us-east-1, ``list_foundation_models`` returns many TEXT models that are
    NOT on-demand (60 of 122 in one live listing), and Converse against those
    bare ids fails with ``ValidationException: Invocation of model ID <id> with
    on-demand throughput isn't supported`` — that includes every current Claude
    model. The real ids below are taken from that listing; the two ``some.*``
    ids are synthetic edge cases. Only the on-demand ones are reachable by bare
    id; the rest must be reached through their inference profile, which the
    primary listing already returns.
    """
    from app.services import bedrock_discovery

    monkeypatch.setattr(
        bedrock_discovery, "_resolve_discovery_regions", lambda: ["us-east-1"]
    )

    fake_client = boto3.client("bedrock", region_name="us-east-1")
    stubber = Stubber(fake_client)
    # Empty profile listing → the foundation-model fallback path is taken.
    stubber.add_response(
        "list_inference_profiles",
        {"inferenceProfileSummaries": []},
        {"maxResults": 1000},
    )
    stubber.add_response(
        "list_foundation_models",
        {
            "modelSummaries": [
                {
                    "modelId": "anthropic.claude-sonnet-5",
                    "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-5",
                    "modelName": "Claude Sonnet 5",
                    "providerName": "Anthropic",
                    "outputModalities": ["TEXT"],
                    "inferenceTypesSupported": ["INFERENCE_PROFILE"],
                },
                {
                    "modelId": "amazon.nova-2-lite-v1:0",
                    "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-2-lite-v1:0",
                    "modelName": "Nova 2 Lite",
                    "providerName": "Amazon",
                    "outputModalities": ["TEXT"],
                    "inferenceTypesSupported": ["INFERENCE_PROFILE"],
                },
                {
                    "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
                    "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
                    "modelName": "Claude 3 Haiku",
                    "providerName": "Anthropic",
                    "outputModalities": ["TEXT"],
                    "inferenceTypesSupported": ["ON_DEMAND"],
                },
                {
                    "modelId": "meta.llama3-70b-instruct-v1:0",
                    "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/meta.llama3-70b-instruct-v1:0",
                    "modelName": "Llama 3 70B",
                    "providerName": "Meta",
                    "outputModalities": ["TEXT"],
                    "inferenceTypesSupported": ["ON_DEMAND", "PROVISIONED"],
                },
                {
                    "modelId": "some.provisioned-only-v1:0",
                    "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/some.provisioned-only-v1:0",
                    "modelName": "Provisioned Only",
                    "providerName": "Other",
                    "outputModalities": ["TEXT"],
                    "inferenceTypesSupported": ["PROVISIONED"],
                },
                {
                    "modelId": "some.empty-inference-types-v1:0",
                    "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/some.empty-inference-types-v1:0",
                    "modelName": "Empty Types",
                    "providerName": "Other",
                    "outputModalities": ["TEXT"],
                    "inferenceTypesSupported": [],
                },
            ]
        },
        {},
    )
    stubber.activate()
    monkeypatch.setattr(boto3, "client", lambda *a, **k: fake_client)

    catalog = bedrock_discovery._list_runtime_catalog()

    assert {e["model_id"] for e in catalog} == {
        "anthropic.claude-3-haiku-20240307-v1:0",
        "meta.llama3-70b-instruct-v1:0",
    }


def test_foundation_model_fallback_keeps_ids_with_no_inference_type_field(monkeypatch):
    """An ABSENT field is unknowable, so it is not treated as a verdict.

    Dropping on a missing key would empty the fallback entirely if the API shape
    ever changed, turning an unknown into a silent no-catalog. Present-but-
    lacking-ON_DEMAND is the only signal strong enough to drop on.
    """
    from app.services import bedrock_discovery

    monkeypatch.setattr(
        bedrock_discovery, "_resolve_discovery_regions", lambda: ["us-east-1"]
    )

    fake_client = boto3.client("bedrock", region_name="us-east-1")
    stubber = Stubber(fake_client)
    stubber.add_response(
        "list_inference_profiles",
        {"inferenceProfileSummaries": []},
        {"maxResults": 1000},
    )
    stubber.add_response(
        "list_foundation_models",
        {
            "modelSummaries": [
                {
                    "modelId": "vendor.unknown-shape-v1:0",
                    "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/vendor.unknown-shape-v1:0",
                    "modelName": "Unknown Shape",
                    "providerName": "Vendor",
                    "outputModalities": ["TEXT"],
                },
            ]
        },
        {},
    )
    stubber.activate()
    monkeypatch.setattr(boto3, "client", lambda *a, **k: fake_client)

    catalog = bedrock_discovery._list_runtime_catalog()

    assert [e["model_id"] for e in catalog] == ["vendor.unknown-shape-v1:0"]


def test_runtime_listing_dedups_across_regions(monkeypatch):
    """Same model id surfaces in two regions → kept once with first-region tagging."""
    from app.services import bedrock_discovery

    monkeypatch.setattr(
        bedrock_discovery, "_resolve_discovery_regions",
        lambda: ["us-east-1", "us-west-2"],
    )

    def _fake_list_in_region(region):
        # Cross-region inference profile id collapses; a region-only
        # foundation model surfaces from us-west-2 only.
        if region == "us-east-1":
            return [
                {
                    "model_id": "us.anthropic.claude-3-7",
                    "model_name": "Claude 3.7",
                    "provider": "anthropic",
                    "endpoint": "runtime",
                    "api": "converse",
                    "region": "us-east-1",
                },
            ]
        if region == "us-west-2":
            return [
                {
                    "model_id": "us.anthropic.claude-3-7",
                    "model_name": "Claude 3.7",
                    "provider": "anthropic",
                    "endpoint": "runtime",
                    "api": "converse",
                    "region": "us-west-2",
                },
                {
                    "model_id": "amazon.titan-text-west-only",
                    "model_name": "Titan West",
                    "provider": "amazon",
                    "endpoint": "runtime",
                    "api": "converse",
                    "region": "us-west-2",
                },
            ]
        return []

    monkeypatch.setattr(
        bedrock_discovery, "_list_runtime_catalog_in_region", _fake_list_in_region
    )

    catalog = bedrock_discovery._list_runtime_catalog()
    by_id = {e["model_id"]: e for e in catalog}
    # Cross-region model dedups with first region (us-east-1) winning.
    assert by_id["us.anthropic.claude-3-7"]["region"] == "us-east-1"
    # Region-specific model picked up from its actual region.
    assert by_id["amazon.titan-text-west-only"]["region"] == "us-west-2"
    assert len(catalog) == 2


# --------------------------------------------------------------------------- #
# per-region mantle listing parallel                                           #
# --------------------------------------------------------------------------- #


async def test_mantle_listing_per_region_parallel(httpx_mock):
    """Three regions, three GETs — each returns its own id set."""
    from app.services import bedrock_discovery

    httpx_mock.add_response(
        method="GET",
        url="https://bedrock-mantle.us-east-1.api.aws/v1/models",
        json={"object": "list", "data": [{"id": "openai.gpt-5.5"}, {"id": "anthropic.claude-3-7"}]},
    )
    httpx_mock.add_response(
        method="GET",
        url="https://bedrock-mantle.us-west-2.api.aws/v1/models",
        json={"object": "list", "data": [{"id": "xai.grok-4"}]},
    )
    httpx_mock.add_response(
        method="GET",
        url="https://bedrock-mantle.eu-west-1.api.aws/v1/models",
        json={"object": "list", "data": [{"id": "mistral.large-2"}]},
    )

    results = await asyncio.gather(
        bedrock_discovery._list_mantle_in_region("us-east-1"),
        bedrock_discovery._list_mantle_in_region("us-west-2"),
        bedrock_discovery._list_mantle_in_region("eu-west-1"),
    )

    east, west, eu = results
    assert {e["model_id"] for e in east} == {"openai.gpt-5.5", "anthropic.claude-3-7"}
    assert {e["model_id"] for e in west} == {"xai.grok-4"}
    assert {e["model_id"] for e in eu} == {"mistral.large-2"}
    for batch in (east, west, eu):
        for entry in batch:
            assert entry["endpoint"] == "mantle"
            assert entry["api"] is None  # resolved later by probe
            assert entry["region"] in ("us-east-1", "us-west-2", "eu-west-1")


async def test_mantle_listing_one_region_failure_does_not_abort_gather(httpx_mock):
    """One region's httpx.ConnectError must not abort the gather."""
    from app.services import bedrock_discovery

    httpx_mock.add_response(
        method="GET",
        url="https://bedrock-mantle.us-east-1.api.aws/v1/models",
        json={"object": "list", "data": [{"id": "openai.gpt-5.5"}]},
    )
    httpx_mock.add_exception(
        httpx.ConnectError("simulated connect refused"),
        url="https://bedrock-mantle.us-west-2.api.aws/v1/models",
    )
    httpx_mock.add_response(
        method="GET",
        url="https://bedrock-mantle.eu-west-1.api.aws/v1/models",
        json={"object": "list", "data": [{"id": "mistral.large-2"}]},
    )

    results = await asyncio.gather(
        bedrock_discovery._list_mantle_in_region("us-east-1"),
        bedrock_discovery._list_mantle_in_region("us-west-2"),
        bedrock_discovery._list_mantle_in_region("eu-west-1"),
    )
    east, west, eu = results
    assert {e["model_id"] for e in east} == {"openai.gpt-5.5"}
    assert west == []  # the failed region contributed nothing
    assert {e["model_id"] for e in eu} == {"mistral.large-2"}


# --------------------------------------------------------------------------- #
# runtime-first precedence                                                     #
# --------------------------------------------------------------------------- #


def test_runtime_first_precedence_drops_mantle_dup():
    """When the same model exists on both endpoints, the runtime id wins.

    Runtime is the forward path. Implemented structurally — runtime
    ``global.anthropic.claude-opus-4-8`` and mantle
    ``anthropic.claude-opus-4-8`` reduce to the same ``short_model_id``
    (``claude-opus-4-8``) and are recognized as the same model. The mantle entry
    is dropped so config.py and the descriptor map agree on a canonical id: no
    descriptor is written for the dropped id, and ``_make_model``'s
    exact-string lookup therefore misses and routes to Converse, which is now
    the intent rather than an accidental fallthrough.
    """
    from app.services.bedrock_discovery import _merge_runtime_first as _merge

    runtime = [
        {
            "model_id": "global.anthropic.claude-opus-4-8",
            "model_name": "Claude Opus 4.8 (global)",
            "provider": "anthropic",
            "endpoint": "runtime",
            "api": "converse",
            "region": None,
        },
        {
            "model_id": "us.anthropic.claude-haiku-4-5-v1:0",
            "model_name": "Claude Haiku 4.5 (us)",
            "provider": "anthropic",
            "endpoint": "runtime",
            "api": "converse",
            "region": None,
        },
    ]
    mantle_lists = [
        [
            {
                "model_id": "anthropic.claude-opus-4-8",
                "model_name": "anthropic.claude-opus-4-8",
                "provider": "anthropic",
                "endpoint": "mantle",
                "api": None,
                "region": "us-east-1",
            },
            {
                "model_id": "openai.gpt-5.5",
                "model_name": "openai.gpt-5.5",
                "provider": "openai",
                "endpoint": "mantle",
                "api": None,
                "region": "us-east-1",
            },
        ]
    ]
    merged, to_resolve = _merge(runtime, mantle_lists)
    by_id = {e["model_id"]: e for e in merged}
    # Same-model dup: the runtime id wins.
    assert "global.anthropic.claude-opus-4-8" in by_id
    assert by_id["global.anthropic.claude-opus-4-8"]["endpoint"] == "runtime"
    assert "anthropic.claude-opus-4-8" not in by_id
    # MANTLE-ONLY id survives, with its endpoint intact — mantle is retained for
    # coverage runtime does not have, which is the point of the whole merge.
    assert by_id["openai.gpt-5.5"]["endpoint"] == "mantle"
    # Runtime-only id (no mantle equivalent) survives unchanged.
    assert by_id["us.anthropic.claude-haiku-4-5-v1:0"]["endpoint"] == "runtime"
    # to_resolve carries ONLY the surviving mantle entries. Probing a mantle id
    # runtime already serves is wasted HTTP and would write a descriptor for an
    # id no config.py can now contain.
    assert {e["model_id"] for e in to_resolve} == {"openai.gpt-5.5"}


async def test_a_mantle_only_model_keeps_its_entry_and_gets_a_descriptor(
    monkeypatch, httpx_mock, tmp_path
):
    """Mantle is NOT abandoned: end-to-end proof through the orchestrator.

    The precedence flip is only safe if a model that exists ONLY on mantle still
    reaches the catalog with a resolved api and a persisted descriptor — that
    descriptor is what makes ``_make_model`` route it to ``BedrockMantleModel``
    instead of Converse, which cannot serve it.
    """
    from app.services import bedrock_discovery

    monkeypatch.setattr(
        bedrock_discovery,
        "_list_runtime_catalog",
        lambda: [
            {
                "model_id": "us.anthropic.claude-haiku-4-5-v1:0",
                "model_name": "Claude Haiku 4.5",
                "provider": "anthropic",
                "endpoint": "runtime",
                "api": "converse",
                "region": None,
            }
        ],
    )
    monkeypatch.setattr(
        bedrock_discovery, "_resolve_discovery_regions", lambda: ["us-east-1"]
    )
    httpx_mock.add_response(
        method="GET",
        url="https://bedrock-mantle.us-east-1.api.aws/v1/models",
        json={"object": "list", "data": [{"id": "openai.gpt-5.5"}]},
    )
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_load("responses_200_minimal.json"),
    )

    result = await (bedrock_discovery.discover_models("proj-mantle-only", tmp_path))

    by_id = {e["model_id"]: e for e in result.catalog}
    assert "openai.gpt-5.5" in by_id, "a mantle-only model was dropped"
    assert by_id["openai.gpt-5.5"]["endpoint"] == "mantle"
    assert by_id["openai.gpt-5.5"]["api"] == "responses"
    # The descriptor is what routes it away from Converse.
    assert result.descriptors["openai.gpt-5.5"]["endpoint"] == "mantle"
    assert result.descriptors["openai.gpt-5.5"]["api"] == "responses"
    # The runtime-only model is untouched and carries no descriptor.
    assert by_id["us.anthropic.claude-haiku-4-5-v1:0"]["endpoint"] == "runtime"
    assert "us.anthropic.claude-haiku-4-5-v1:0" not in result.descriptors


async def test_an_excluded_mantle_id_cannot_remove_a_model_runtime_can_serve(
    monkeypatch, httpx_mock, tmp_path
):
    """THE invariant the precedence seam never had, in either direction.

    Under mantle-first this was violated by construction: the merge dropped the
    runtime entry as soon as a mantle listing existed — BEFORE the probe ran —
    so when the probe then excluded the mantle id, the runtime counterpart was
    already gone and the model vanished from the catalog entirely. A dropped
    model produces no symptom; it is just absent from a list, which is why this
    went unexamined.

    Runtime-first eliminates it structurally (the mantle dup is dropped, so the
    excluded id is never one runtime covers), but nothing pinned it. This does:
    a mantle id whose probe fails outright must leave the runtime entry for the
    SAME model standing.
    """
    from app.services import bedrock_discovery

    monkeypatch.setattr(
        bedrock_discovery,
        "_list_runtime_catalog",
        lambda: [
            {
                "model_id": "global.anthropic.claude-opus-4-8",
                "model_name": "Claude Opus 4.8",
                "provider": "anthropic",
                "endpoint": "runtime",
                "api": "converse",
                "region": None,
            }
        ],
    )
    monkeypatch.setattr(
        bedrock_discovery, "_resolve_discovery_regions", lambda: ["us-east-1"]
    )
    # Mantle lists the SAME model under its bare id.
    httpx_mock.add_response(
        method="GET",
        url="https://bedrock-mantle.us-east-1.api.aws/v1/models",
        json={"object": "list", "data": [{"id": "anthropic.claude-opus-4-8"}]},
    )
    # Make any probe attempt fail hard, so an exclusion WOULD happen if the
    # mantle id were ever probed.
    def _boom(*a, **k):
        raise NotImplementedError("stub")

    from app.services.bedrock_mantle_model import BedrockMantleModel

    for api in ("responses", "chat_completions", "anthropic_messages"):
        monkeypatch.setitem(BedrockMantleModel._BUILDERS, api, _boom)

    result = await (bedrock_discovery.discover_models("proj-inv", tmp_path))

    ids = {e["model_id"] for e in result.catalog}
    assert "global.anthropic.claude-opus-4-8" in ids, (
        "the runtime entry for a model was removed because its mantle "
        "counterpart could not be resolved — the model is now invisible"
    )
    assert "anthropic.claude-opus-4-8" not in ids
    # And it was never probed at all, so there is nothing to exclude.
    assert result.excluded == {}


async def test_an_excluded_mantle_only_id_is_reported_not_just_logged(
    monkeypatch, httpx_mock, tmp_path
):
    """A mantle-ONLY exclusion is total, so it has to reach a reader.

    This is our coverage hole rather than the model being unavailable, and a
    WARNING in CloudWatch is read by nobody while choosing models.
    """
    from app.services import bedrock_discovery
    from app.services.bedrock_mantle_model import BedrockMantleModel

    monkeypatch.setattr(bedrock_discovery, "_list_runtime_catalog", lambda: [])
    monkeypatch.setattr(
        bedrock_discovery, "_resolve_discovery_regions", lambda: ["us-east-1"]
    )
    httpx_mock.add_response(
        method="GET",
        url="https://bedrock-mantle.us-east-1.api.aws/v1/models",
        json={"object": "list", "data": [{"id": "qwen.qwen3-coder"}]},
    )

    def _boom(*a, **k):
        raise NotImplementedError("stub")

    for api in ("responses", "chat_completions", "anthropic_messages"):
        monkeypatch.setitem(BedrockMantleModel._BUILDERS, api, _boom)

    result = await (bedrock_discovery.discover_models("proj-excl", tmp_path))

    assert "qwen.qwen3-coder" not in {e["model_id"] for e in result.catalog}
    assert "qwen.qwen3-coder" in result.excluded
    assert result.excluded["qwen.qwen3-coder"]


def test_merge_dedups_mantle_across_regions_keeping_first():
    """An id appearing in two mantle regions gets one entry from the first region."""
    from app.services.bedrock_discovery import _merge_runtime_first as _merge

    runtime: list = []
    mantle_lists = [
        [
            {
                "model_id": "openai.gpt-5.5",
                "model_name": "openai.gpt-5.5",
                "provider": "openai",
                "endpoint": "mantle",
                "api": None,
                "region": "us-east-1",
            }
        ],
        [
            {
                "model_id": "openai.gpt-5.5",
                "model_name": "openai.gpt-5.5",
                "provider": "openai",
                "endpoint": "mantle",
                "api": None,
                "region": "us-west-2",
            }
        ],
    ]
    merged, to_resolve = _merge(runtime, mantle_lists)
    assert len(merged) == 1
    assert merged[0]["region"] == "us-east-1"
    assert len(to_resolve) == 1
    assert to_resolve[0]["region"] == "us-east-1"


# --------------------------------------------------------------------------- #
# probe success / fallthrough                                                  #
# --------------------------------------------------------------------------- #


async def test_probe_succeeds_first_attempt(httpx_mock):
    """Captured 200 fixture → probe returns the first-attempted API."""
    from app.services import bedrock_discovery

    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_load("responses_200_minimal.json"),
    )

    api, supports_temperature, probe_status = await bedrock_discovery._probe_capability(
        "openai.gpt-5.5", "us-east-1"
    )
    assert api == "responses"
    # Default-true when the model didn't reject temperature.
    assert supports_temperature is True
    assert probe_status == "ok"
    # Only one network call — the probe bypassed the retry loop.
    assert len(httpx_mock.get_requests()) == 1


async def test_probe_falls_through_400_to_next_api(monkeypatch, httpx_mock):
    """First-API 400 validation_error → sweep continues, success on second.

    With the mapper live, the chat_completions mapper has its own captured
    payload. The probe sweep wires up cleanly: first attempt hits
    /openai/v1/responses → 400 validation_error → MantleAPIRejected →
    fallthrough. Second attempt: /v1/chat/completions → 200 → success.
    """
    from app.services import bedrock_discovery

    # openai.* heuristic order = (responses, chat_completions,
    # anthropic_messages).
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        status_code=400,
        json=_load("responses_400_wrong_api.json"),
    )
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions",
        json=_load("chat_completions_200_minimal.json"),
    )

    api, supports_temperature, probe_status = await bedrock_discovery._probe_capability(
        "openai.gpt-5.5", "us-east-1"
    )
    assert api == "chat_completions"
    assert supports_temperature is True
    assert probe_status == "ok"
    requests = httpx_mock.get_requests()
    assert len(requests) == 2


async def test_probe_temperature_rejection_keeps_responses_caches_flag(
    httpx_mock,
):
    """Temperature self-correction at probe time — reasoning model returns (responses, False).

    Without this, GPT-5.5 / Grok 4.3 / similar reasoning models hit the
    400 unsupported_parameter and the (now-fixed) old code mistook it for
    "wrong API", falling through to chat_completions/anthropic_messages —
    which answer for a model that does not speak them, so the sweep ended with
    no clean resolution and the id was excluded from the catalog. (An earlier
    version of this docstring said those two mappers raise
    ``NotImplementedError``; they do not, and never did in shipped code — all
    three are implemented. See ``MantleMapperUnsupported``.) This test pins the
    fix: the probe stays on responses and flips supports_temperature=False.
    """
    from app.services import bedrock_discovery

    # First request (with temperature) → 400 unsupported_parameter.
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        status_code=400,
        json=_load("responses_400_unsupported_temperature.json"),
    )
    # Second request (without temperature, after adapter self-correction)
    # → 200 success.
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_load("responses_200_minimal.json"),
    )

    api, supports_temperature, probe_status = await bedrock_discovery._probe_capability(
        "openai.gpt-5.5", "us-east-1"
    )
    assert api == "responses", (
        "probe must STAY on responses — the temperature rejection means "
        "'right API, drop the flag', not 'wrong API'"
    )
    assert supports_temperature is False
    assert probe_status == "ok"
    # Two network calls — initial 400 + retry without temperature.
    assert len(httpx_mock.get_requests()) == 2


async def test_discover_persists_supports_temperature_false(
    monkeypatch, httpx_mock, tmp_path
):
    """End-to-end: a reasoning model produces a descriptor with supports_temperature=False on disk."""
    from app.services import bedrock_discovery

    monkeypatch.setattr(
        bedrock_discovery, "_list_runtime_catalog", lambda: []
    )
    monkeypatch.setattr(
        bedrock_discovery, "_resolve_discovery_regions", lambda: ["us-east-1"]
    )
    httpx_mock.add_response(
        method="GET",
        url="https://bedrock-mantle.us-east-1.api.aws/v1/models",
        json={"object": "list", "data": [{"id": "openai.gpt-5.5"}]},
    )
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        status_code=400,
        json=_load("responses_400_unsupported_temperature.json"),
    )
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_load("responses_200_minimal.json"),
    )

    result = await bedrock_discovery.discover_models("proj-temp", tmp_path)
    descriptor = result.descriptors["openai.gpt-5.5"]
    assert descriptor["api"] == "responses"
    assert descriptor["region"] == "us-east-1"
    assert descriptor["supports_temperature"] is False, (
        "Reasoning model descriptor MUST persist supports_temperature=False "
        "so the harness adapter skips temperature on first call (no rejection round-trip)"
    )

    # Catalog entry includes the model — NOT excluded.
    by_id = {e["model_id"]: e for e in result.catalog}
    assert "openai.gpt-5.5" in by_id


async def test_probe_excludes_id_when_all_mappers_unimplemented(
    monkeypatch, httpx_mock
):
    """Every mapper → NotImplementedError → MantleMapperUnsupported, no descriptor written."""
    from app.services import bedrock_discovery
    from app.services.bedrock_mantle_model import BedrockMantleModel

    # Force every mapper to raise NotImplementedError. The
    # adapter ships a real responses mapper (200-passing) so we need to
    # patch the dispatch table itself.
    def _stub_builder(*args, **kwargs):
        raise NotImplementedError("stub")

    monkeypatch.setitem(BedrockMantleModel._BUILDERS, "responses", _stub_builder)
    monkeypatch.setitem(BedrockMantleModel._BUILDERS, "chat_completions", _stub_builder)
    monkeypatch.setitem(
        BedrockMantleModel._BUILDERS, "anthropic_messages", _stub_builder
    )

    with pytest.raises(bedrock_discovery.MantleMapperUnsupported):
        await bedrock_discovery._probe_capability(
            "qwen.qwen3-coder", "us-east-1"
        )


async def test_probe_does_not_pollute_accumulated_usage(httpx_mock):
    """Probes use throwaway adapters; production adapter usage stays at 0."""
    from app.services import bedrock_discovery
    from app.services.bedrock_mantle_model import BedrockMantleModel

    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_load("responses_200_minimal.json"),
        is_reusable=True,
    )

    for _ in range(3):
        await bedrock_discovery._probe_capability(
            "openai.gpt-5.5", "us-east-1"
        )

    production = BedrockMantleModel(
        {"endpoint": "mantle", "api": "responses", "region": "us-east-1"},
        "openai.gpt-5.5",
    )
    assert production._usage_input_tokens == 0
    assert production._usage_output_tokens == 0
    assert production.accumulated_usage == {"inputTokens": 0, "outputTokens": 0}


async def test_probe_timeout_via_wait_for(monkeypatch):
    """A probe attempt that exceeds 20s is cancelled and the sweep continues."""
    from app.services import bedrock_discovery
    from app.services.bedrock_mantle_model import BedrockMantleModel

    # Patch the wait_for budget down so the test runs quickly while
    # still exercising the asyncio.wait_for path.
    monkeypatch.setattr(bedrock_discovery, "_PROBE_TIMEOUT_S", 0.05)

    # First attempt: hangs. Second attempt: returns successfully via a
    # stubbed builder. We patch the dispatch on BedrockMantleModel so
    # the first attempt sleeps and the second resolves.
    real_responses_builder = BedrockMantleModel._BUILDERS["responses"]

    def _hang_then_succeed(*args, **kwargs):
        # Sleep longer than the patched 50ms timeout so wait_for fires.
        time.sleep(0.5)
        return real_responses_builder(*args, **kwargs)

    monkeypatch.setitem(
        BedrockMantleModel._BUILDERS, "responses", _hang_then_succeed
    )
    # Force a quick path on the second attempt by raising
    # NotImplementedError, which the sweep treats as "try the next".
    monkeypatch.setitem(
        BedrockMantleModel._BUILDERS,
        "chat_completions",
        lambda *a, **k: (_ for _ in ()).throw(NotImplementedError("stub")),
    )
    monkeypatch.setitem(
        BedrockMantleModel._BUILDERS,
        "anthropic_messages",
        lambda *a, **k: (_ for _ in ()).throw(NotImplementedError("stub")),
    )

    start = time.perf_counter()
    # Permissive policy: a timeout is "unverified" not "exclude". The
    # other two attempts are NotImplementedError stubs which advance
    # without flagging unverified, so the timeout's recorded candidate
    # is what resolves the sweep. The wait_for budget enforcement is
    # the actual thing under test (elapsed << 5s).
    api, supports_temperature, probe_status = await bedrock_discovery._probe_capability(
        "openai.gpt-5.5", "us-east-1"
    )
    elapsed = time.perf_counter() - start
    assert api == "responses"
    assert probe_status == "unverified"
    # Each attempt is bounded at 50ms; three attempts × 50ms = 150ms,
    # plus thread-pool overhead. Allow 5s slack for slow CI.
    assert elapsed < 5.0, f"probe took {elapsed:.2f}s — wait_for budget not honored"


# --------------------------------------------------------------------------- #
# descriptor persistence under project lock                                    #
# --------------------------------------------------------------------------- #


async def test_persist_under_lock(monkeypatch, httpx_mock, tmp_path):
    """discover_models calls persist_descriptors_locked exactly once with the final descriptors."""
    from app.services import bedrock_discovery

    # Stub runtime listing → empty (so the runtime branch doesn't
    # ClientError on missing creds inside boto3.client('bedrock')).
    monkeypatch.setattr(
        bedrock_discovery, "_list_runtime_catalog", lambda: []
    )
    # Restrict regions for speed.
    monkeypatch.setattr(
        bedrock_discovery, "_resolve_discovery_regions", lambda: ["us-east-1"]
    )
    # Mantle listing returns one id; probe returns 'responses'.
    httpx_mock.add_response(
        method="GET",
        url="https://bedrock-mantle.us-east-1.api.aws/v1/models",
        json={"object": "list", "data": [{"id": "openai.gpt-5.5"}]},
    )
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_load("responses_200_minimal.json"),
    )

    persist_calls: list[tuple[Path, dict, str]] = []

    async def _spy_persist(project_dir, descriptors, project_id):
        persist_calls.append((project_dir, dict(descriptors), project_id))
        from app.services.model_descriptors import write_descriptors

        return write_descriptors(project_dir, descriptors)

    monkeypatch.setattr(
        bedrock_discovery, "persist_descriptors_locked", _spy_persist
    )

    result = await bedrock_discovery.discover_models("proj-A", tmp_path)

    assert len(persist_calls) == 1
    pd, descriptors, pid = persist_calls[0]
    assert pid == "proj-A"
    assert pd == tmp_path
    assert "openai.gpt-5.5" in descriptors
    assert descriptors["openai.gpt-5.5"] == {
        "endpoint": "mantle",
        "api": "responses",
        "region": "us-east-1",
    }
    # Result mirrors what was persisted.
    assert "openai.gpt-5.5" in result.descriptors


async def test_concurrent_discover_serialized(monkeypatch, httpx_mock, tmp_path):
    """Two concurrent discover_models calls for the SAME project_id serialize through the lock."""
    from app.services import bedrock_discovery
    from app.services import model_descriptors_async

    monkeypatch.setattr(
        bedrock_discovery, "_list_runtime_catalog", lambda: []
    )
    monkeypatch.setattr(
        bedrock_discovery, "_resolve_discovery_regions", lambda: ["us-east-1"]
    )
    # Each task hits the GET twice (one per task) — make the response reusable.
    httpx_mock.add_response(
        method="GET",
        url="https://bedrock-mantle.us-east-1.api.aws/v1/models",
        json={"object": "list", "data": [{"id": "openai.gpt-5.5"}]},
        is_reusable=True,
    )
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_load("responses_200_minimal.json"),
        is_reusable=True,
    )

    # Instrument write_descriptors to record entry/exit timestamps.
    from app.services.model_descriptors import write_descriptors as real_write

    intervals: list[tuple[float, float]] = []

    def instrumented(pd, ds):
        entry = time.perf_counter()
        time.sleep(0.02)
        out = real_write(pd, ds)
        intervals.append((entry, time.perf_counter()))
        return out

    monkeypatch.setattr(
        "app.services.model_descriptors_async.write_descriptors",
        instrumented,
    )

    await asyncio.gather(
        bedrock_discovery.discover_models("proj-X", tmp_path),
        bedrock_discovery.discover_models("proj-X", tmp_path),
    )

    assert len(intervals) == 2
    intervals.sort()
    prev, nxt = intervals[0], intervals[1]
    assert prev[1] <= nxt[0] + 1e-6, (
        f"writes overlapped: {prev} vs {nxt} — lock not enforced"
    )


# --------------------------------------------------------------------------- #
# existing descriptors are sticky                                              #
# --------------------------------------------------------------------------- #


async def test_existing_descriptor_not_reprobed(monkeypatch, httpx_mock, tmp_path):
    """An id with an existing descriptor is NOT re-probed."""
    from app.services import bedrock_discovery
    from app.services.model_descriptors import write_descriptors

    monkeypatch.setattr(
        bedrock_discovery, "_list_runtime_catalog", lambda: []
    )
    monkeypatch.setattr(
        bedrock_discovery, "_resolve_discovery_regions", lambda: ["us-east-1"]
    )
    write_descriptors(
        tmp_path,
        {
            "openai.gpt-5.5": {
                "endpoint": "mantle",
                "api": "responses",
                "region": "us-east-1",
            }
        },
    )

    httpx_mock.add_response(
        method="GET",
        url="https://bedrock-mantle.us-east-1.api.aws/v1/models",
        json={"object": "list", "data": [{"id": "openai.gpt-5.5"}]},
    )

    probe_calls: list[tuple[str, str]] = []

    async def _spy_probe(model_id, region):
        probe_calls.append((model_id, region))
        return "responses", True, "ok"

    monkeypatch.setattr(
        bedrock_discovery, "_probe_capability", _spy_probe
    )

    result = await bedrock_discovery.discover_models("proj-Y", tmp_path)

    assert probe_calls == []
    assert "openai.gpt-5.5" in result.descriptors
    # Catalog entry is patched from the existing descriptor.
    by_id = {e["model_id"]: e for e in result.catalog}
    assert by_id["openai.gpt-5.5"]["api"] == "responses"
    assert by_id["openai.gpt-5.5"]["region"] == "us-east-1"


# --------------------------------------------------------------------------- #
# catalog shape                                                                #
# --------------------------------------------------------------------------- #


async def test_discover_models_returns_typed_catalog(monkeypatch, httpx_mock, tmp_path):
    """Final catalog: every entry has endpoint/api/region; runtime entries are tagged correctly."""
    from app.services import bedrock_discovery

    runtime_entries = [
        {
            "model_id": "us.anthropic.claude-3-7",
            "model_name": "Claude 3.7",
            "provider": "anthropic",
            "endpoint": "runtime",
            "api": "converse",
            "region": None,
        }
    ]
    monkeypatch.setattr(
        bedrock_discovery, "_list_runtime_catalog", lambda: list(runtime_entries)
    )
    monkeypatch.setattr(
        bedrock_discovery, "_resolve_discovery_regions", lambda: ["us-east-1"]
    )

    httpx_mock.add_response(
        method="GET",
        url="https://bedrock-mantle.us-east-1.api.aws/v1/models",
        json={"object": "list", "data": [{"id": "openai.gpt-5.5"}]},
    )
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_load("responses_200_minimal.json"),
    )

    result = await bedrock_discovery.discover_models("proj-Z", tmp_path)

    by_id = {e["model_id"]: e for e in result.catalog}
    runtime_entry = by_id["us.anthropic.claude-3-7"]
    assert runtime_entry["endpoint"] == "runtime"
    assert runtime_entry["api"] == "converse"
    assert runtime_entry["region"] is None

    mantle_entry = by_id["openai.gpt-5.5"]
    assert mantle_entry["endpoint"] == "mantle"
    assert mantle_entry["api"] == "responses"
    assert mantle_entry["region"] == "us-east-1"

    # Descriptor map carries ONLY mantle entries.
    assert "openai.gpt-5.5" in result.descriptors
    assert "us.anthropic.claude-3-7" not in result.descriptors


# --------------------------------------------------------------------------- #
# Final-catalog: mantle dropped after MantleMapperUnsupported                  #
# --------------------------------------------------------------------------- #


async def test_discover_excludes_id_when_probe_unsupported(
    monkeypatch, httpx_mock, tmp_path
):
    """All mappers raise NotImplementedError → id is excluded from catalog AND descriptors."""
    from app.services import bedrock_discovery
    from app.services.bedrock_mantle_model import BedrockMantleModel

    monkeypatch.setattr(
        bedrock_discovery, "_list_runtime_catalog", lambda: []
    )
    monkeypatch.setattr(
        bedrock_discovery, "_resolve_discovery_regions", lambda: ["us-east-1"]
    )
    httpx_mock.add_response(
        method="GET",
        url="https://bedrock-mantle.us-east-1.api.aws/v1/models",
        json={"object": "list", "data": [{"id": "qwen.qwen3-coder"}]},
    )

    def _stub(*a, **k):
        raise NotImplementedError("stub")

    monkeypatch.setitem(BedrockMantleModel._BUILDERS, "responses", _stub)
    monkeypatch.setitem(BedrockMantleModel._BUILDERS, "chat_completions", _stub)
    monkeypatch.setitem(BedrockMantleModel._BUILDERS, "anthropic_messages", _stub)

    result = await bedrock_discovery.discover_models("proj-Q", tmp_path)
    by_id = {e["model_id"]: e for e in result.catalog}
    assert "qwen.qwen3-coder" not in by_id
    assert "qwen.qwen3-coder" not in result.descriptors


# --------------------------------------------------------------------------- #
# Permissive discovery policy — one test per probe outcome                    #
# --------------------------------------------------------------------------- #
#
# Outcomes: 1 clean 2xx → keep (ok); 2 temperature 400 → drop the flag, keep
# (ok); 3 reasoning truncated → keep (ok); 4 null content → keep (ok);
# 5 content-safety 400 → keep (ok); 6 wrong-API validation_error → try the next
# API; 7 403 entitlement block → keep (access_not_enabled). A model is excluded
# only when EVERY API returns the definitive wrong-API validation_error.


async def test_probe_keeps_on_clean_2xx(httpx_mock):
    """Outcome 1 — clean 2xx resolves the api with probe_status=ok."""
    from app.services import bedrock_discovery

    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_load("responses_200_minimal.json"),
    )

    api, supports_temperature, probe_status = await bedrock_discovery._probe_capability(
        "openai.gpt-5.5", "us-east-1"
    )
    assert api == "responses"
    assert supports_temperature is True
    assert probe_status == "ok"


async def test_probe_keeps_on_temperature_4xx_anthropic_message_shape(
    httpx_mock,
):
    """Outcome 2 — anthropic 'temperature is deprecated' triggers temperature self-correction.

    Regression for Claude Opus 4.7/4.8: the anthropic API rejects
    temperature with a message-only 400 (no ``unsupported_parameter``
    code). The generalized trigger detects it via the lowercase message
    text containing ``"temperature"``.
    """
    from app.services import bedrock_discovery

    # First probe attempt for anthropic.* uses the anthropic_messages
    # API per heuristic order; first request includes temperature, gets
    # the 400; adapter retries without temperature; second request
    # returns 200.
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages",
        status_code=400,
        json=_load("anthropic_messages_400_unsupported_temperature.json"),
    )
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages",
        json=_load("anthropic_messages_200_minimal.json"),
    )

    api, supports_temperature, probe_status = await bedrock_discovery._probe_capability(
        "anthropic.claude-opus-4-8", "us-east-1"
    )
    assert api == "anthropic_messages", (
        "probe must STAY on anthropic_messages — the temperature rejection means "
        "'right API, drop the flag', not 'wrong API'"
    )
    assert supports_temperature is False
    assert probe_status == "ok"
    assert len(httpx_mock.get_requests()) == 2


async def test_probe_keeps_on_reasoning_truncated(httpx_mock):
    """Outcome 3 — MantleReasoningTruncated is api-confirmed, keep with ok."""
    from app.services import bedrock_discovery

    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_load("responses_incomplete_truncation.json"),
    )

    api, supports_temperature, probe_status = await bedrock_discovery._probe_capability(
        "openai.gpt-5.5", "us-east-1"
    )
    assert api == "responses"
    assert supports_temperature is True
    assert probe_status == "ok"


async def test_probe_keeps_on_null_content(httpx_mock):
    """Outcome 4 — chat_completions content:null is reachable, keep with ok.

    Regression for MiniMax M2 / GLM 4.7 / Kimi K2.5: reasoning models on
    chat_completions surfaces emit ``content: null`` while burning the
    cap on hidden reasoning. The parser MUST treat null as empty text;
    the probe MUST resolve and keep.
    """
    from app.services import bedrock_discovery

    # mistral.* heuristic order = (chat_completions, responses,
    # anthropic_messages). Use a mistral-prefixed id so the first attempt
    # hits chat_completions directly.
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions",
        json=_load("chat_completions_200_null_content.json"),
    )

    api, supports_temperature, probe_status = await bedrock_discovery._probe_capability(
        "minimax.m2", "us-east-1"
    )
    assert api == "chat_completions"
    assert supports_temperature is True
    assert probe_status == "ok"


async def test_probe_keeps_on_content_safety_block(httpx_mock):
    """Outcome 5 — a 400 content_filter is reachable, keep with ok.

    Regression for Grok 4.3: a bland-probe payload that gets safety-
    blocked is a model that responded — the api routing is correct.
    """
    from app.services import bedrock_discovery

    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        status_code=400,
        json=_load("responses_400_content_filter.json"),
    )

    api, supports_temperature, probe_status = await bedrock_discovery._probe_capability(
        "xai.grok-4.3", "us-east-1"
    )
    assert api == "responses"
    assert supports_temperature is True
    assert probe_status == "ok"


async def test_probe_advances_on_validation_error_wrong_api(httpx_mock):
    """Outcome 6 — definitive wrong-API validation_error advances the sweep."""
    from app.services import bedrock_discovery

    # openai.* heuristic = (responses, chat_completions, anthropic).
    # First attempt 400 wrong-api; second attempt 200.
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        status_code=400,
        json=_load("responses_400_wrong_api.json"),
    )
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions",
        json=_load("chat_completions_200_minimal.json"),
    )

    api, supports_temperature, probe_status = await bedrock_discovery._probe_capability(
        "openai.gpt-5.5", "us-east-1"
    )
    assert api == "chat_completions"
    assert probe_status == "ok"


async def test_probe_keeps_on_access_not_enabled_403(httpx_mock):
    """Outcome 7 — 403 SignatureMismatch maps to access_not_enabled, keep."""
    from app.services import bedrock_discovery

    # All three APIs return 403. Heuristic order = (responses, chat,
    # anthropic). Each 403 on the same descriptor recycles the URL.
    for path in (
        "/openai/v1/responses",
        "/v1/chat/completions",
        "/anthropic/v1/messages",
    ):
        httpx_mock.add_response(
            method="POST",
            url=f"https://bedrock-mantle.us-east-1.api.aws{path}",
            status_code=403,
            text="SignatureDoesNotMatch (account not enabled)",
        )

    api, supports_temperature, probe_status = await bedrock_discovery._probe_capability(
        "openai.gpt-5.5", "us-east-1"
    )
    # First-attempted api wins as the entitlement candidate.
    assert api == "responses"
    assert supports_temperature is True
    assert probe_status == "access_not_enabled"


async def test_probe_excludes_only_when_all_apis_wrong_validation_error(
    httpx_mock, monkeypatch
):
    """Exclusion fires ONLY when every candidate API returns validation_error."""
    from app.services import bedrock_discovery

    # Every attempt returns a definitive wrong-API 400.
    for path in (
        "/openai/v1/responses",
        "/v1/chat/completions",
        "/anthropic/v1/messages",
    ):
        httpx_mock.add_response(
            method="POST",
            url=f"https://bedrock-mantle.us-east-1.api.aws{path}",
            status_code=400,
            json=_load("responses_400_wrong_api.json"),
        )

    with pytest.raises(bedrock_discovery.MantleDiscoveryError):
        await bedrock_discovery._probe_capability(
            "openai.gpt-5.5", "us-east-1"
        )


async def test_probe_keeps_when_some_apis_wrong_and_one_transient(
    monkeypatch, httpx_mock
):
    """Mixed outcomes: 2 wrong-API + 1 transient → KEEP (permissive)."""
    from app.services import bedrock_discovery

    # Patch the timeout to be tiny so the unverified attempt fires
    # quickly. Heuristic for openai.* = (responses, chat, anthropic).
    monkeypatch.setattr(bedrock_discovery, "_PROBE_TIMEOUT_S", 0.05)

    # responses: definitive wrong-API.
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        status_code=400,
        json=_load("responses_400_wrong_api.json"),
    )
    # chat_completions: timeout via slow request handler.
    httpx_mock.add_callback(
        lambda request: time.sleep(0.5)
        or httpx.Response(status_code=200, json={"choices": [{"message": {"content": "x"}}], "usage": {}}),
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions",
    )
    # anthropic: also definitive wrong-API.
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages",
        status_code=400,
        json=_load("responses_400_wrong_api.json"),
    )

    api, supports_temperature, probe_status = await bedrock_discovery._probe_capability(
        "openai.gpt-5.5", "us-east-1"
    )
    # The transient (chat_completions) is the surviving candidate.
    assert api == "chat_completions"
    assert probe_status == "unverified"


async def test_discover_persists_probe_status_unverified(
    monkeypatch, tmp_path
):
    """When the probe resolves transiently, descriptor on disk has probe_status='unverified'."""
    from app.services import bedrock_discovery

    monkeypatch.setattr(
        bedrock_discovery, "_list_runtime_catalog", lambda: []
    )
    monkeypatch.setattr(
        bedrock_discovery, "_resolve_discovery_regions", lambda: ["us-east-1"]
    )
    # Stub the listing to return one mantle id without going through
    # httpx — we only want to test the persistence of probe_status.
    monkeypatch.setattr(
        bedrock_discovery,
        "_list_mantle_in_region",
        lambda region: __make_listing_for_region(region, ["openai.gpt-5.5"]),
    )

    async def _spy_probe(model_id, region):
        return "responses", True, "unverified"

    monkeypatch.setattr(bedrock_discovery, "_probe_capability", _spy_probe)

    result = await bedrock_discovery.discover_models("proj-unv", tmp_path)
    descriptor = result.descriptors["openai.gpt-5.5"]
    assert descriptor["probe_status"] == "unverified"
    # supports_temperature stays absent (default-true).
    assert "supports_temperature" not in descriptor


async def test_discover_persists_probe_status_access_not_enabled(
    monkeypatch, tmp_path
):
    """access_not_enabled probe status persists on disk."""
    from app.services import bedrock_discovery

    monkeypatch.setattr(
        bedrock_discovery, "_list_runtime_catalog", lambda: []
    )
    monkeypatch.setattr(
        bedrock_discovery, "_resolve_discovery_regions", lambda: ["us-east-1"]
    )
    monkeypatch.setattr(
        bedrock_discovery,
        "_list_mantle_in_region",
        lambda region: __make_listing_for_region(region, ["xai.grok-4.3"]),
    )

    async def _spy_probe(model_id, region):
        return "responses", True, "access_not_enabled"

    monkeypatch.setattr(bedrock_discovery, "_probe_capability", _spy_probe)

    result = await bedrock_discovery.discover_models("proj-ent", tmp_path)
    descriptor = result.descriptors["xai.grok-4.3"]
    assert descriptor["probe_status"] == "access_not_enabled"


async def test_discover_persists_probe_status_ok_omitted(monkeypatch, tmp_path):
    """probe_status='ok' is the default and stays absent from the descriptor."""
    from app.services import bedrock_discovery

    monkeypatch.setattr(
        bedrock_discovery, "_list_runtime_catalog", lambda: []
    )
    monkeypatch.setattr(
        bedrock_discovery, "_resolve_discovery_regions", lambda: ["us-east-1"]
    )
    monkeypatch.setattr(
        bedrock_discovery,
        "_list_mantle_in_region",
        lambda region: __make_listing_for_region(region, ["openai.gpt-5.5"]),
    )

    async def _spy_probe(model_id, region):
        return "responses", True, "ok"

    monkeypatch.setattr(bedrock_discovery, "_probe_capability", _spy_probe)

    result = await bedrock_discovery.discover_models("proj-ok", tmp_path)
    descriptor = result.descriptors["openai.gpt-5.5"]
    assert "probe_status" not in descriptor


async def test_discover_keeps_grok_minimax_opus_under_real_failure_shapes(
    monkeypatch, httpx_mock, tmp_path
):
    """Live-sweep regression: three ids, three failure shapes — all three survive.

    The three ids the OLD exclusion gate dropped:
      * claude-opus-4-8 — anthropic 'temperature is deprecated' 400
      * minimax.m2 — chat_completions content:null reasoning response
      * xai.grok-4.3 — content-safety 400 on the bland probe payload
    Permissive policy: all three KEEP, with descriptors written.
    """
    from app.services import bedrock_discovery

    monkeypatch.setattr(
        bedrock_discovery, "_list_runtime_catalog", lambda: []
    )
    monkeypatch.setattr(
        bedrock_discovery, "_resolve_discovery_regions", lambda: ["us-east-1"]
    )
    httpx_mock.add_response(
        method="GET",
        url="https://bedrock-mantle.us-east-1.api.aws/v1/models",
        json={
            "object": "list",
            "data": [
                {"id": "anthropic.claude-opus-4-8"},
                {"id": "minimax.m2"},
                {"id": "xai.grok-4.3"},
            ],
        },
    )
    # Claude Opus: anthropic_messages → 400 'temperature is deprecated' →
    # adapter temperature self-correction → 200.
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages",
        status_code=400,
        json=_load("anthropic_messages_400_unsupported_temperature.json"),
    )
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages",
        json=_load("anthropic_messages_200_minimal.json"),
    )
    # MiniMax M2: chat_completions → 200 content:null → parser tolerates.
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions",
        json=_load("chat_completions_200_null_content.json"),
    )
    # Grok 4.3: responses → 400 content_filter → safety block, keep.
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        status_code=400,
        json=_load("responses_400_content_filter.json"),
    )

    result = await bedrock_discovery.discover_models("proj-perm", tmp_path)
    by_id = {e["model_id"]: e for e in result.catalog}

    # All three SURVIVE the sweep — that's the regression assertion.
    assert "anthropic.claude-opus-4-8" in by_id
    assert "minimax.m2" in by_id
    assert "xai.grok-4.3" in by_id

    # Descriptors are persisted with the resolved api/region.
    opus = result.descriptors["anthropic.claude-opus-4-8"]
    assert opus["api"] == "anthropic_messages"
    assert opus.get("supports_temperature") is False
    assert opus.get("probe_status", "ok") == "ok"

    minimax = result.descriptors["minimax.m2"]
    assert minimax["api"] == "chat_completions"
    assert minimax.get("probe_status", "ok") == "ok"

    grok = result.descriptors["xai.grok-4.3"]
    assert grok["api"] == "responses"
    assert grok.get("probe_status", "ok") == "ok"


# Helper used by the spy-probe tests above.
def __make_listing_for_region(region: str, ids: list[str]):
    """Return a small awaitable that mimics _list_mantle_in_region for tests."""

    async def _inner():
        return [
            {
                "model_id": mid,
                "model_name": mid,
                "provider": mid.split(".", 1)[0],
                "endpoint": "mantle",
                "api": None,
                "region": region,
            }
            for mid in ids
        ]

    return _inner()
