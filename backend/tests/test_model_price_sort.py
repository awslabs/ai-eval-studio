# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cheapest-first target-model ordering.

``list_bedrock_models`` sorts its output by per-1M token price so the
coordinator presents — and writes to ``config.models`` — the canonical
cheapest-first order the optimizer's "lift the cheapest model within
reach" strategy depends on. The sort is deterministic code, not an LLM
judgment. These tests pin the ordering rules.
"""

from app.services.mcp_tools.models import _sort_by_price


def _m(name, inp=None, out=None):
    """Build a model dict, optionally with input/output per-1M prices."""
    d = {"model_id": name, "model_name": name, "provider": "p"}
    if inp is not None:
        d["input_price_per_1m"] = inp
    if out is not None:
        d["output_price_per_1m"] = out
    return d


def test_sorts_ascending_by_input_price():
    models = [_m("sonnet", 3.0, 15.0), _m("nova-lite", 0.06, 0.24), _m("haiku", 0.8, 4.0)]
    ordered = [m["model_id"] for m in _sort_by_price(models)]
    assert ordered == ["nova-lite", "haiku", "sonnet"]


def test_tie_break_on_output_price():
    # Same input price; cheaper output wins.
    models = [_m("a", 1.0, 9.0), _m("b", 1.0, 3.0)]
    ordered = [m["model_id"] for m in _sort_by_price(models)]
    assert ordered == ["b", "a"]


def test_unknown_price_sorts_last_preserving_order():
    # Two priced, two unpriced; unpriced sink to the end in their
    # original relative order (stable sort).
    models = [_m("unpriced1"), _m("cheap", 0.5, 1.0), _m("unpriced2"), _m("pricey", 5.0, 10.0)]
    ordered = [m["model_id"] for m in _sort_by_price(models)]
    assert ordered == ["cheap", "pricey", "unpriced1", "unpriced2"]


def test_all_unknown_keeps_incoming_order():
    models = [_m("x"), _m("y"), _m("z")]
    ordered = [m["model_id"] for m in _sort_by_price(models)]
    assert ordered == ["x", "y", "z"]


def test_partial_price_missing_output_still_orders_by_input():
    # A model with input price but no output price still ranks by input,
    # ahead of an unpriced model.
    models = [_m("noprice"), _m("inp_only", 0.2), _m("full", 0.1, 0.5)]
    ordered = [m["model_id"] for m in _sort_by_price(models)]
    assert ordered == ["full", "inp_only", "noprice"]


def test_returns_same_reference_sorted_in_place():
    models = [_m("b", 2.0), _m("a", 1.0)]
    result = _sort_by_price(models)
    assert result is models  # in-place
    assert result[0]["model_id"] == "a"


def test_na_sinks_below_unknown():
    """N/a-priced entries sort BELOW unknown-priced entries.

    priced → unknown → n/a is the contract. The literal string
    ``"n/a"`` (wire format from :func:`_enrich_with_pricing`) is the strongest
    "do not auto-pick" signal — strictly worse than missing data.
    """
    # Input order: priced-mid, n/a, unknown, priced-cheap.
    # Expected output: priced-cheap, priced-mid, unknown, n/a.
    models = [
        {"model_id": "a", "input_price_per_1m": 5.0, "output_price_per_1m": 15.0},
        {
            "model_id": "b",
            "input_price_per_1m": "n/a",
            "output_price_per_1m": "n/a",
        },
        {"model_id": "c"},
        {"model_id": "d", "input_price_per_1m": 1.0, "output_price_per_1m": 3.0},
    ]
    ordered = [m["model_id"] for m in _sort_by_price(models)]
    assert ordered == ["d", "a", "c", "b"]


def test_na_only_one_field_still_sinks():
    """A half-formed n/a (only one of input/output) still sinks to the bottom.

    Defensive: the matcher always sets BOTH fields together, but the sort key
    treats either field as ``"n/a"`` as the n/a signal — covers any future
    enrichment path that might mark only one side.
    """
    # Only output is "n/a" — still sinks below unknown.
    models = [
        {"model_id": "priced", "input_price_per_1m": 1.0, "output_price_per_1m": 2.0},
        {"model_id": "unknown"},
        {"model_id": "half_na", "input_price_per_1m": 0.5, "output_price_per_1m": "n/a"},
    ]
    ordered = [m["model_id"] for m in _sort_by_price(models)]
    assert ordered == ["priced", "unknown", "half_na"]

    # Inverse: only input is "n/a" — also sinks below unknown.
    models2 = [
        {"model_id": "priced", "input_price_per_1m": 1.0, "output_price_per_1m": 2.0},
        {"model_id": "unknown"},
        {"model_id": "half_na2", "input_price_per_1m": "n/a", "output_price_per_1m": 0.5},
    ]
    ordered2 = [m["model_id"] for m in _sort_by_price(models2)]
    assert ordered2 == ["priced", "unknown", "half_na2"]
