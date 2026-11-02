# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# backend/tests/test_eval_runner.py
"""Tests for loading and listing eval run files, and for cost estimation."""

import pytest
import json
import time
from pathlib import Path
from unittest.mock import patch
from app.services.eval_runner import EvalRunner
from app.services.bedrock_pricing import (
    _clear_pricing_cache,
    estimate_cost_usd,
)
import app.services.bedrock_pricing as bedrock_pricing


@pytest.fixture
def project_dir(tmp_path):
    """Create a minimal project directory with test data and a prompt."""
    project = tmp_path / "project"
    (project / "data" / "raw").mkdir(parents=True)
    (project / "prompts").mkdir(parents=True)
    (project / "output").mkdir(parents=True)

    # Sample data file
    csv = project / "data" / "raw" / "test.csv"
    csv.write_text("input,expected\nWhat is 2+2?,4\nCapital of France?,Paris\n")

    # Sample prompt
    prompt = project / "prompts" / "baseline.md"
    prompt.write_text("Answer the question concisely.")

    return project


_RUN_SEQ = iter(range(1_000_000))


def _write_run(project_dir: Path, results_data: list | dict) -> Path:
    """Write a run file the way the harness does: ``output/eval_results_*.json``."""
    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"eval_results_2026-01-01_00-00-00-{next(_RUN_SEQ):06d}.json"
    path.write_text(json.dumps(results_data, indent=2, default=str))
    return path


class TestEvalRunner:
    def test_load_results(self, project_dir):
        runner = EvalRunner()
        results = [{
            "test_run_id": "test-model",
            "timestamp": "2026-01-01T00:00:00",
            "hyperparameters": {"model": "test", "prompt_version": "baseline"},
            "results": [{"name": "test_case_0", "success": True, "input": "q", "actual_output": "a", "expected_output": "a", "metrics_data": []}],
        }]
        path = _write_run(project_dir, results)
        assert path.exists()

        loaded = runner.load_results(path)
        assert isinstance(loaded, list)
        assert loaded[0]["test_run_id"] == "test-model"

    def test_list_runs(self, project_dir):
        runner = EvalRunner()
        run1 = [{
            "source": "manual",
            "test_run_id": "model-a",
            "timestamp": "2026-01-01T00:00:00",
            "hyperparameters": {
                "model": "a",
                "endpoint": "runtime",
                "region": "us-east-1",
                "prompt_version": "baseline",
            },
            "results": [{"name": "tc0", "success": True, "input": "q", "actual_output": "a", "expected_output": "a", "metrics_data": []}],
        }]
        run2 = [{
            "source": "manual",
            "test_run_id": "model-b",
            "timestamp": "2026-01-02T00:00:00",
            "hyperparameters": {
                "model": "b",
                "endpoint": "runtime",
                "region": "us-east-1",
                "prompt_version": "v1",
            },
            "results": [{"name": "tc0", "success": False, "input": "q", "actual_output": "b", "expected_output": "a", "metrics_data": []}],
        }]
        _write_run(project_dir, run1)
        time.sleep(1.1)
        _write_run(project_dir, run2)

        runs = runner.list_runs(project_dir)
        assert len(runs) == 2
        assert "model_runs" in runs[0]

    def test_list_runs_skips_files_with_partial_multi_model_hp(self, project_dir):
        """A multi-model run where only model[0] has the required fields
        and model[1] is genuinely malformed (missing ``model``) must
        skip-and-warn at file granularity, not silently disappear via
        the outer except. Note: missing endpoint/region is NOT
        malformed — those are optional (in-process/pre-convention runs
        omit them) — so the skip must be driven by the truly-required
        ``model`` field, not routing metadata."""
        partial = [
            {
                "source": "manual",
                "test_run_id": "model-a",
                "timestamp": "2026-06-19T18:00:00",
                "hyperparameters": {
                    "model": "a", "endpoint": "runtime",
                    "region": "us-east-1", "prompt_version": "v1",
                },
                "results": [],
            },
            {
                "source": "manual",
                "test_run_id": "model-b",
                "timestamp": "2026-06-19T18:00:01",
                # Missing ``model`` — the only truly-required hp field —
                # should trigger a file-granularity skip.
                "hyperparameters": {"prompt_version": "v1"},
                "results": [],
            },
        ]
        import json as _json
        out = project_dir / "output"
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval_results_partial.json").write_text(_json.dumps(partial))

        runner = EvalRunner()
        runs = runner.list_runs(project_dir)
        assert runs == [], (
            "partial multi-model file must be skipped, not partially returned"
        )

    def test_list_runs_treats_missing_source_as_manual(self, project_dir):
        """Pre-source-tagging legacy files lack ``source``; they should
        not silently vanish via KeyError — they should display as
        ``source="manual"``."""
        legacy = [{
            "test_run_id": "old-run",
            "timestamp": "2026-01-01T00:00:00",
            "hyperparameters": {
                "model": "a", "endpoint": "runtime",
                "region": "us-east-1", "prompt_version": "baseline",
            },
            "results": [],
        }]
        import json as _json
        out = project_dir / "output"
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval_results_no_source.json").write_text(_json.dumps(legacy))

        runner = EvalRunner()
        runs = runner.list_runs(project_dir)
        assert len(runs) == 1
        assert runs[0]["source"] == "manual"

    def test_list_runs_lists_files_without_endpoint_region(self, project_dir):
        """Run files whose hyperparameters carry ``model`` but OMIT
        ``endpoint`` / ``region`` MUST still be listed — in-process and
        pre-convention projects write ``{model, temperature, max_tokens,
        prompt_version}`` with no routing metadata, and requiring
        endpoint/region skipped every such run so the whole
        Evaluations/Reports view went empty for real user projects
        (regression from the strict-schema change). The summary builder
        defaults endpoint→"runtime", region→"" rather than KeyError-and-
        skip."""
        legacy = [{
            "source": "manual",
            "test_run_id": "old-run",
            "timestamp": "2026-01-01T00:00:00",
            "hyperparameters": {
                "model": "a",
                "temperature": 0.0,
                "max_tokens": 512,
                "prompt_version": "baseline",
            },
            "results": [],
        }]
        # Write directly: ``save_results`` writes the strict schema, so
        # we can't round-trip a routing-metadata-less file through it.
        import json as _json
        out = project_dir / "output"
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval_results_old.json").write_text(_json.dumps(legacy))

        runner = EvalRunner()
        runs = runner.list_runs(project_dir)
        assert len(runs) == 1, "run without endpoint/region must be listed, not skipped"
        mr = runs[0]["model_runs"][0]
        assert mr["model"] == "a"
        assert mr["endpoint"] == "runtime"  # defaulted
        assert mr["region"] == ""            # defaulted

    def test_list_runs_surfaces_endpoint_and_region(self, project_dir):
        """``hyperparameters.endpoint`` + ``hyperparameters.region`` (set
        by the template runner from the descriptor) flow through to
        each ``model_runs`` entry so the SPA can render the badge
        without parsing the model id.
        """
        runner = EvalRunner()
        run = [{
            "source": "manual",
            "test_run_id": "claude-opus-4-8",
            "timestamp": "2026-06-19T18:00:00",
            "hyperparameters": {
                "model": "anthropic.claude-opus-4-8",
                "endpoint": "mantle",
                "region": "us-east-1",
                "prompt_version": "baseline",
            },
            "results": [{"name": "tc0", "success": True, "input": "q", "actual_output": "a", "expected_output": "a", "metrics_data": []}],
        }]
        _write_run(project_dir, run)
        runs = runner.list_runs(project_dir)
        assert len(runs) == 1
        mr = runs[0]["model_runs"][0]
        assert mr["endpoint"] == "mantle"
        assert mr["region"] == "us-east-1"

    def test_list_runs_excludes_validation_smoke(self, project_dir):
        """``source="validation"`` runs (1-case smokes from
        ``validate_harness``) must NEVER appear in ``list_runs`` — they'd
        skew the chart next to real 36-case evals.
        """
        runner = EvalRunner()
        real = [{
            "source": "manual",
            "test_run_id": "model-a",
            "timestamp": "2026-06-19T18:00:00",
            "hyperparameters": {
                "model": "a",
                "endpoint": "runtime",
                "region": "us-east-1",
                "prompt_version": "baseline",
            },
            "results": [{"name": "tc0", "success": True, "input": "q", "actual_output": "a", "expected_output": "a", "metrics_data": []}],
        }]
        smoke = [{
            "source": "validation",
            "test_run_id": "model-a",
            "timestamp": "2026-06-19T18:00:01",
            "hyperparameters": {
                "model": "a",
                "endpoint": "runtime",
                "region": "us-east-1",
                "prompt_version": "baseline",
            },
            "results": [{"name": "smoke", "success": True, "input": "q", "actual_output": "a", "expected_output": "a", "metrics_data": []}],
        }]
        _write_run(project_dir, real)
        time.sleep(1.1)
        _write_run(project_dir, smoke)

        runs = runner.list_runs(project_dir)
        assert len(runs) == 1, "validation smoke must be filtered out"
        assert runs[0]["source"] == "manual"


class TestMultiModelRunner:
    def test_short_model_name(self):
        """Structural short-name derivation: strip leading dot-segments
        until the first segment with a digit, then strip trailing version
        + date tags. Remainder is verbatim — dotted versions
        (``grok-4.3``, ``gpt-5.4``) keep their dots, and ``claude-`` is
        NO LONGER stripped (it disambiguates Anthropic models from other
        ``opus-*`` / ``sonnet-*`` lines a vendor might ship).
        """
        runner = EvalRunner()
        # Anthropic — verbatim (claude- stays).
        assert runner.short_model_name("us.anthropic.claude-sonnet-4-6") == "claude-sonnet-4-6"
        assert runner.short_model_name("anthropic.claude-sonnet-4-6") == "claude-sonnet-4-6"
        assert runner.short_model_name("global.anthropic.claude-opus-4-8") == "claude-opus-4-8"
        # Multi-region prefixes (us/eu/apac/global/au/us-gov) all strip via
        # the structural rule, no hardcoded region list.
        assert runner.short_model_name("eu.anthropic.claude-sonnet-4-6") == "claude-sonnet-4-6"
        assert runner.short_model_name("apac.amazon.nova-lite-v1:0") == "nova-lite"
        # Date-stamped versions strip the date.
        assert runner.short_model_name("us.anthropic.claude-haiku-4-5-20251001-v1:0") == "claude-haiku-4-5"
        # Amazon Nova — version tag strips.
        assert runner.short_model_name("us.amazon.nova-lite-v1:0") == "nova-lite"
        # Dotted versions — a past regression mangled these to "3", "4", "5".
        assert runner.short_model_name("xai.grok-4.3") == "grok-4.3"
        assert runner.short_model_name("openai.gpt-5.4") == "gpt-5.4"
        assert runner.short_model_name("openai.gpt-5.5") == "gpt-5.5"
        assert runner.short_model_name("deepseek.v3.2") == "v3.2"
        # New-provider passthrough — the rule has no provider list to update.
        assert runner.short_model_name("google.gemma-3-27b-it") == "gemma-3-27b-it"
        assert runner.short_model_name("mistral.mistral-large-3-v1:0") == "mistral-large-3"
        # Single-segment / dotless ids fall through unchanged.
        assert runner.short_model_name("custom-model") == "custom-model"


@pytest.fixture
def primed_pricing_cache(monkeypatch):
    """Seed the bedrock_pricing in-process cache so estimate_cost_usd works
    without hitting the LiteLLM JSON over the network.

    Prices are in USD per 1M tokens (the on-disk schema). Covers the
    inference-profile-prefixed IDs the existing tests use plus a few bare
    ones so the prefix-strip fallback path is exercised too.
    """
    primed = {
        "us.anthropic.claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
        "us.anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 0.8, "output": 4.0},
        "us.anthropic.claude-opus-4-0": {"input": 15.0, "output": 75.0},
        "us.amazon.nova-lite-v1:0": {"input": 0.06, "output": 0.24},
        "amazon.nova-micro-v1:0": {"input": 0.035, "output": 0.14},
        "us.amazon.nova-pro-v1:0": {"input": 0.8, "output": 3.2},
    }
    monkeypatch.setattr(bedrock_pricing, "_pricing_cache", primed)
    monkeypatch.setattr(bedrock_pricing, "_pricing_cache_timestamp", 1.0)
    yield
    _clear_pricing_cache()


class TestEstimateCost:
    """Tests for estimate_cost_usd with a primed in-process pricing cache."""

    def test_claude_sonnet_cost(self, primed_pricing_cache):
        """Per-1M pricing yields correct USD for a Claude Sonnet call."""
        usage = {"inputTokens": 1000, "outputTokens": 500}
        cost = estimate_cost_usd("us.anthropic.claude-sonnet-4-6", usage)
        # 1000/1M * 3.00 + 500/1M * 15.00 = 0.003 + 0.0075 = 0.0105
        assert abs(cost - 0.0105) < 1e-9

    def test_claude_haiku_cost(self, primed_pricing_cache):
        usage = {"inputTokens": 2000, "outputTokens": 1000}
        cost = estimate_cost_usd(
            "us.anthropic.claude-haiku-4-5-20251001-v1:0", usage
        )
        # 2000/1M * 0.80 + 1000/1M * 4.00 = 0.0016 + 0.004 = 0.0056
        assert abs(cost - 0.0056) < 1e-9

    def test_claude_opus_cost(self, primed_pricing_cache):
        usage = {"input_tokens": 500, "output_tokens": 200}
        cost = estimate_cost_usd("us.anthropic.claude-opus-4-0", usage)
        # 500/1M * 15.00 + 200/1M * 75.00 = 0.0075 + 0.015 = 0.0225
        assert abs(cost - 0.0225) < 1e-9

    def test_nova_lite_cost(self, primed_pricing_cache):
        usage = {"inputTokens": 10000, "outputTokens": 5000}
        cost = estimate_cost_usd("us.amazon.nova-lite-v1:0", usage)
        # 10000/1M * 0.06 + 5000/1M * 0.24 = 0.0006 + 0.0012 = 0.0018
        assert abs(cost - 0.0018) < 1e-9

    def test_nova_micro_cost(self, primed_pricing_cache):
        usage = {"inputTokens": 1000, "outputTokens": 1000}
        cost = estimate_cost_usd("amazon.nova-micro-v1:0", usage)
        # 1000/1M * 0.035 + 1000/1M * 0.14 = 0.000035 + 0.00014 = 0.000175
        assert abs(cost - 0.000175) < 1e-9

    def test_nova_pro_cost(self, primed_pricing_cache):
        usage = {"inputTokens": 1000, "outputTokens": 1000}
        cost = estimate_cost_usd("us.amazon.nova-pro-v1:0", usage)
        # 1000/1M * 0.80 + 1000/1M * 3.20 = 0.0008 + 0.0032 = 0.004
        assert abs(cost - 0.004) < 1e-9

    def test_unknown_model_returns_zero(self, primed_pricing_cache):
        """No entry in the cache ⇒ cost unknown ⇒ 0.0 (best-effort)."""
        usage = {"inputTokens": 1000, "outputTokens": 500}
        cost = estimate_cost_usd("some-unknown-model-xyz", usage)
        assert cost == 0.0

    def test_empty_cache_returns_zero(self):
        """Cold cache (cache never populated) returns 0.0 without raising."""
        _clear_pricing_cache()
        cost = estimate_cost_usd(
            "us.anthropic.claude-sonnet-4-6",
            {"inputTokens": 1000, "outputTokens": 500},
        )
        assert cost == 0.0

    def test_empty_usage_returns_zero(self, primed_pricing_cache):
        """Known model but no token counts ⇒ 0.0 cost."""
        cost = estimate_cost_usd("us.anthropic.claude-sonnet-4-6", {})
        assert cost == 0.0

    def test_alternative_token_keys(self, primed_pricing_cache):
        """Accepts both inputTokens/outputTokens and input_tokens/output_tokens."""
        usage_camel = {"inputTokens": 1000, "outputTokens": 1000}
        usage_snake = {"input_tokens": 1000, "output_tokens": 1000}
        cost_camel = estimate_cost_usd(
            "us.anthropic.claude-sonnet-4-6", usage_camel
        )
        cost_snake = estimate_cost_usd(
            "us.anthropic.claude-sonnet-4-6", usage_snake
        )
        assert abs(cost_camel - cost_snake) < 1e-9

    def test_no_cache_unchanged_from_prior_formula(self, primed_pricing_cache):
        """Regression guard: zero cache tokens MUST yield the same dollar
        figure as the pre-cache-aware formula. Verifies the additive
        cache terms don't perturb existing no-cache evals."""
        usage = {
            "inputTokens": 1000,
            "outputTokens": 500,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokens": 0,
        }
        cost = estimate_cost_usd("us.anthropic.claude-sonnet-4-6", usage)
        # Same expectation as ``test_claude_sonnet_cost`` — adding zero
        # cache tokens at any rate must not change the total.
        assert abs(cost - 0.0105) < 1e-9


class TestEstimateCostCacheAware:
    """Cost formula must add cache-token charges on top of inputTokens.

    Bedrock's Converse ``TokenUsage`` reports ``inputTokens``,
    ``cacheReadInputTokens``, and ``cacheWriteInputTokens`` as separate
    counts; ``inputTokens`` is the standard-rate portion. So the
    formula adds cache terms — does NOT subtract them from
    ``inputTokens``. These tests pin exact dollar figures so a future
    refactor that flips back to subtraction (or skips cache entirely)
    fails loudly."""

    @pytest.fixture
    def cache_priced(self, monkeypatch):
        """Pricing fixture with cache rates so the additive path is testable."""
        primed = {
            # Sonnet 4.6: input 3.00, output 15.00, cache_read 0.30, cache_write 3.75
            # (mirrors the LiteLLM shape: cache_read = input/10, cache_write = input × 1.25).
            "us.anthropic.claude-sonnet-4-6": {
                "input": 3.00,
                "output": 15.00,
                "cache_read": 0.30,
                "cache_write": 3.75,
            },
            # Haiku without cache rates → falls back to input rate.
            "us.anthropic.claude-haiku-4-5": {
                "input": 0.80,
                "output": 4.00,
            },
        }
        monkeypatch.setattr(bedrock_pricing, "_pricing_cache", primed)
        monkeypatch.setattr(bedrock_pricing, "_pricing_cache_timestamp", 1.0)
        yield
        _clear_pricing_cache()

    def test_additive_cache_read_billed_at_cache_rate(self, cache_priced):
        """Cache-read tokens billed at cache_read price, ADDED on top of
        inputTokens at the standard rate. If the formula subtracted
        cache from input the dollar figure would be wrong by exactly
        the cache_read tokens × (input_rate − cache_read_rate)."""
        usage = {
            "inputTokens": 1000,
            "outputTokens": 500,
            "cacheReadInputTokens": 2000,
            "cacheWriteInputTokens": 0,
        }
        cost = estimate_cost_usd("us.anthropic.claude-sonnet-4-6", usage)
        # 1000 × 3.00 + 2000 × 0.30 + 0 × 3.75 + 500 × 15.00 = 3000 + 600 + 7500 = 11100
        # 11100 / 1_000_000 = 0.0111
        expected = 0.0111
        assert abs(cost - expected) < 1e-9, (
            f"expected additive cost ${expected}, got ${cost}. "
            f"If this fails by exactly cache_tokens × (input_rate - cache_rate), "
            f"the formula was flipped to subtractive."
        )

    def test_additive_cache_write_billed_at_cache_rate(self, cache_priced):
        """Cache-write tokens billed at cache_write price, additive."""
        usage = {
            "inputTokens": 1000,
            "outputTokens": 500,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokens": 1000,
        }
        cost = estimate_cost_usd("us.anthropic.claude-sonnet-4-6", usage)
        # 1000 × 3.00 + 0 × 0.30 + 1000 × 3.75 + 500 × 15.00 = 3000 + 3750 + 7500 = 14250
        # 14250 / 1_000_000 = 0.01425
        assert abs(cost - 0.01425) < 1e-9

    def test_additive_both_cache_dimensions(self, cache_priced):
        """Both cache_read and cache_write billed independently and added on top."""
        usage = {
            "inputTokens": 1000,
            "outputTokens": 500,
            "cacheReadInputTokens": 2000,
            "cacheWriteInputTokens": 1000,
        }
        cost = estimate_cost_usd("us.anthropic.claude-sonnet-4-6", usage)
        # 1000 × 3.00 + 2000 × 0.30 + 1000 × 3.75 + 500 × 15.00
        # = 3000 + 600 + 3750 + 7500 = 14850 → 0.01485
        assert abs(cost - 0.01485) < 1e-9

    def test_missing_cache_rate_falls_back_to_input_rate(self, cache_priced):
        """Models without cache_read/cache_write in pricing bill cache
        tokens at the input rate. Keeps the formula safe: an
        over-estimate, never an under-estimate."""
        usage = {
            "inputTokens": 1000,
            "outputTokens": 500,
            "cacheReadInputTokens": 2000,
            "cacheWriteInputTokens": 1000,
        }
        cost = estimate_cost_usd("us.anthropic.claude-haiku-4-5", usage)
        # All cache tokens billed at input rate (0.80):
        # (1000 + 2000 + 1000) × 0.80 + 500 × 4.00 = 3200 + 2000 = 5200 → 0.0052
        assert abs(cost - 0.0052) < 1e-9

    def test_zero_tokens(self, primed_pricing_cache):
        """Zero tokens returns 0.0 cost."""
        usage = {"inputTokens": 0, "outputTokens": 0}
        cost = estimate_cost_usd("us.anthropic.claude-sonnet-4-6", usage)
        assert cost == 0.0


class TestWriteThroughSidecars:
    """Tests for write-through sidecar caching in load_compact and load_case_detail."""

    def test_load_compact_write_through_creates_sidecar(self, project_dir):
        """First load_compact creates output/_derived/<stem>/compact.json."""
        runner = EvalRunner()
        results = [{
            "test_run_id": "model-a",
            "timestamp": "2026-06-26T00:00:00",
            "hyperparameters": {"model": "test-model", "endpoint": "runtime", "region": "us-east-1"},
            "results": [{
                "name": "case1",
                "success": True,
                "input": "What is 2+2?",
                "actual_output": "4",
                "expected_output": "4",
                "additional_metadata": {"id": "tc1", "latency_ms": 100, "input_tokens": 10, "output_tokens": 5},
                "metrics_data": [{"name": "ExactMatch", "success": True, "score": 1.0}],
            }],
        }]
        path = _write_run(project_dir, results)

        # First load creates sidecar
        compact = runner.load_compact(path)
        assert len(compact) == 1

        # Verify sidecar exists
        from app.services.eval_runner import _derived_dir
        derived = _derived_dir(path)
        compact_path = derived / "compact.json"
        assert compact_path.exists()

        # Verify sidecar content equals returned value
        sidecar_content = json.loads(compact_path.read_text())
        assert sidecar_content == compact

    def test_load_compact_byte_identical_to_projection(self, project_dir):
        """Sidecar content is byte-identical to the in-memory projection."""
        runner = EvalRunner()
        results = [{
            "test_run_id": "model-a",
            "timestamp": "2026-06-26T00:00:00",
            "hyperparameters": {"model": "test-model", "endpoint": "runtime", "region": "us-east-1"},
            "results": [{
                "name": "case1",
                "success": True,
                "input": "What is 2+2?",
                "actual_output": "4",
                "expected_output": "4",
                "additional_metadata": {"id": "tc1", "latency_ms": 100, "input_tokens": 10, "output_tokens": 5},
                "metrics_data": [{"name": "ExactMatch", "success": True, "score": 1.0}],
            }],
        }]
        path = _write_run(project_dir, results)

        # First call
        first = runner.load_compact(path)

        # Delete sidecar dir and call again to force recompute
        from app.services.eval_runner import _derived_dir
        import shutil
        shutil.rmtree(_derived_dir(path), ignore_errors=True)
        second = runner.load_compact(path)

        # Both returns must be equal (byte-identical projection)
        assert first == second

    def test_load_compact_reads_sidecar_without_reparse(self, project_dir):
        """Second load reads sidecar without re-parsing the run file."""
        runner = EvalRunner()
        results = [{
            "test_run_id": "model-a",
            "timestamp": "2026-06-26T00:00:00",
            "hyperparameters": {"model": "test-model", "endpoint": "runtime", "region": "us-east-1"},
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
        path = _write_run(project_dir, results)

        # First call creates sidecar
        first = runner.load_compact(path)

        # Overwrite run file with garbage (but keep sidecar fresh)
        from app.services.eval_runner import _derived_dir
        import os
        derived = _derived_dir(path)
        compact_path = derived / "compact.json"

        path.write_text("{invalid json}")
        # Touch sidecar to be newer than run file
        os.utime(compact_path, (time.time() + 10, time.time() + 10))

        # Second call should read from sidecar and succeed (no parse of garbage run file)
        second = runner.load_compact(path)
        assert second == first

    def test_load_compact_mtime_guard_recomputes(self, project_dir):
        """Run file newer than sidecar triggers recompute."""
        runner = EvalRunner()
        results = [{
            "test_run_id": "model-a",
            "timestamp": "2026-06-26T00:00:00",
            "hyperparameters": {"model": "test-model", "endpoint": "runtime", "region": "us-east-1"},
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
        path = _write_run(project_dir, results)

        # First call creates sidecar
        first = runner.load_compact(path)

        # Wait a bit then modify run file content
        time.sleep(0.1)
        results[0]["results"][0]["actual_output"] = "changed"
        path.write_text(json.dumps(results))

        # Second call should recompute (run file newer)
        second = runner.load_compact(path)
        assert second != first
        assert second[0]["results"][0]["output_preview"] == "changed"

    def test_load_case_detail_write_through_and_sha1_name(self, project_dir):
        """First load writes cases/<sha1(case_id)>.json for each real case."""
        runner = EvalRunner()
        results = [{
            "test_run_id": "model-a",
            "timestamp": "2026-06-26T00:00:00",
            "hyperparameters": {"model": "test-model", "endpoint": "runtime", "region": "us-east-1"},
            "results": [
                {
                    "name": "case1",
                    "success": True,
                    "input": "What is 2+2?",
                    "actual_output": "4",
                    "expected_output": "4",
                    "additional_metadata": {"id": "tc1", "latency_ms": 100},
                    "metrics_data": [{"name": "ExactMatch", "success": True, "score": 1.0, "reason": "Match"}],
                },
                {
                    "name": "case2",
                    "success": False,
                    "input": "Capital of France?",
                    "actual_output": "London",
                    "expected_output": "Paris",
                    "additional_metadata": {"id": "tc2", "latency_ms": 150},
                    "metrics_data": [{"name": "ExactMatch", "success": False, "score": 0.0, "reason": "Mismatch"}],
                },
            ],
        }]
        path = _write_run(project_dir, results)

        # First call for tc1 creates sidecars for ALL cases
        detail = runner.load_case_detail(path, "tc1")
        assert detail["input"] == "What is 2+2?"
        assert detail["models"]["test-model"]["actual_output"] == "4"

        # Verify sidecar exists with sha1 name
        from app.services.eval_runner import _derived_dir, _case_sidecar_name
        import hashlib
        derived = _derived_dir(path)
        cases_dir = derived / "cases"
        case1_path = cases_dir / _case_sidecar_name("tc1")
        case2_path = cases_dir / _case_sidecar_name("tc2")

        assert case1_path.exists()
        assert case2_path.exists()

        # Verify sha1 mapping
        expected_sha1 = hashlib.sha1(
            "tc1".encode("utf-8"), usedforsecurity=False
        ).hexdigest() + ".json"
        assert case1_path.name == expected_sha1

        # Verify sidecar content is byte-identical to returned detail
        sidecar_content = json.loads(case1_path.read_text())
        assert sidecar_content == detail

    def test_load_case_detail_unknown_id_no_sidecar(self, project_dir):
        """Unknown case_id returns empty-models sentinel and writes no sidecar."""
        runner = EvalRunner()
        results = [{
            "test_run_id": "model-a",
            "timestamp": "2026-06-26T00:00:00",
            "hyperparameters": {"model": "test-model", "endpoint": "runtime", "region": "us-east-1"},
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
        path = _write_run(project_dir, results)

        # Request unknown case
        detail = runner.load_case_detail(path, "unknown")
        assert detail == {"input": "", "expected_output": "", "models": {}}

        # Verify no sidecar created for unknown id
        from app.services.eval_runner import _derived_dir, _case_sidecar_name
        derived = _derived_dir(path)
        cases_dir = derived / "cases"
        unknown_path = cases_dir / _case_sidecar_name("unknown")
        assert not unknown_path.exists()

    def test_load_case_detail_reads_single_sidecar(self, project_dir):
        """Second call for a different case reads only its small sidecar."""
        runner = EvalRunner()
        results = [{
            "test_run_id": "model-a",
            "timestamp": "2026-06-26T00:00:00",
            "hyperparameters": {"model": "test-model", "endpoint": "runtime", "region": "us-east-1"},
            "results": [
                {"name": "case1", "success": True, "input": "q1", "actual_output": "a1", "expected_output": "a1",
                 "additional_metadata": {"id": "tc1"}, "metrics_data": []},
                {"name": "case2", "success": True, "input": "q2", "actual_output": "a2", "expected_output": "a2",
                 "additional_metadata": {"id": "tc2"}, "metrics_data": []},
            ],
        }]
        path = _write_run(project_dir, results)

        # First call for tc1 writes all sidecars
        first = runner.load_case_detail(path, "tc1")

        # Overwrite run file with garbage (but keep sidecars fresh)
        from app.services.eval_runner import _derived_dir, _case_sidecar_name
        import os
        derived = _derived_dir(path)
        cases_dir = derived / "cases"
        tc2_path = cases_dir / _case_sidecar_name("tc2")

        path.write_text("{invalid json}")
        # Touch tc2 sidecar to be newer
        os.utime(tc2_path, (time.time() + 10, time.time() + 10))

        # Second call for tc2 should read its sidecar (no parse of garbage run file)
        second = runner.load_case_detail(path, "tc2")
        assert second["input"] == "q2"

    def test_list_runs_ignores_derived_dir(self, project_dir):
        """list_runs lists ONLY real runs; _derived/ is invisible to the glob."""
        runner = EvalRunner()
        results = [{
            "test_run_id": "model-a",
            "timestamp": "2026-06-26T00:00:00",
            "hyperparameters": {"model": "test-model", "endpoint": "runtime", "region": "us-east-1"},
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
        path = _write_run(project_dir, results)

        # Create sidecar
        runner.load_compact(path)

        # Verify _derived/ exists
        from app.services.eval_runner import _derived_dir
        derived = _derived_dir(path)
        assert derived.exists()

        # list_runs should return exactly 1 run (not count _derived/)
        runs = runner.list_runs(project_dir)
        assert len(runs) == 1
        assert runs[0]["_file"] == path.name
