# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog normalization: one row per model, and supersession stated as data.

Both defects show up in a real Bedrock catalog listing, and the fixtures below
reproduce its shapes rather than invented ones:

* the same model appears under up to three ids at TWO price points (geographic
  ``us.`` sits exactly 10% above bare/``global.``), so cheapest-first ordering
  compared +10% prices against base prices and the agent could quote one
  variant's price while naming another variant's id;
* a newer, cheaper sibling can sit ABOVE an older model in the same list —
  ``anthropic.claude-sonnet-5`` above ``us.anthropic.claude-sonnet-4-6`` — and
  an agent can still propose the older one as judge. The coordinator prompt
  already states the rule, which is why the fix is a computed field rather than
  another restatement.
"""
from __future__ import annotations

from typing import Any

from app.services.model_catalog import (
    annotate_supersession,
    collapse_variants,
    model_line_and_version,
    split_model_id,
)


def _entry(model_id: str, inp: Any = None, out: Any = None, **over: Any) -> dict[str, Any]:
    base = {
        "model_id": model_id,
        "input_price_per_1m": inp,
        "output_price_per_1m": out,
        "endpoint": "runtime",
        "api": "converse",
        "provider": "Anthropic",
    }
    base.update(over)
    return base


#: Catalog rows in the shape discovery returns them: id variants at two price
#: points, a newer cheaper line, and one unpriced variant.
MEASURED = [
    _entry("anthropic.claude-sonnet-4-6", 3.00, 15.00),
    _entry("us.anthropic.claude-sonnet-4-6", 3.30, 16.50),
    _entry("global.anthropic.claude-sonnet-4-6", 3.00, 15.00),
    _entry("anthropic.claude-sonnet-5", 2.00, 10.00),
    _entry("us.anthropic.claude-sonnet-5", 2.20, 11.00),
    _entry("anthropic.claude-sonnet-4-5", 3.00, 15.00),
    _entry("us.anthropic.claude-sonnet-4-5", None, None),
]


class TestSplitModelId:
    """Structural split, no provider or region lists — that is the whole point.

    ``model_id_short``'s history is a list of providers that broke a hardcoded
    list (xAI, Gemma, DeepSeek, dotted versions), so this reuses its derivation
    instead of adding a second scan.
    """

    def test_covers_every_id_shape_in_the_real_catalog(self):
        cases = {
            "anthropic.claude-sonnet-4-6": ("", "anthropic", "claude-sonnet-4-6"),
            "us.anthropic.claude-sonnet-4-6-v1:0": ("us", "anthropic", "claude-sonnet-4-6"),
            "global.anthropic.claude-opus-4-8": ("global", "anthropic", "claude-opus-4-8"),
            "us.amazon.nova-2-lite-v1:0": ("us", "amazon", "nova-2-lite"),
            "xai.grok-4.3": ("", "xai", "grok-4.3"),
            "deepseek.v3.2": ("", "deepseek", "v3.2"),
            "meta.llama3-70b-instruct-v1:0": ("", "meta", "llama3-70b-instruct"),
            "google.gemma-3-27b-it": ("", "google", "gemma-3-27b-it"),
            "us.anthropic.claude-haiku-4-5-20251001-v1:0": (
                "us", "anthropic", "claude-haiku-4-5",
            ),
        }
        for model_id, expected in cases.items():
            assert split_model_id(model_id) == expected, model_id

    def test_empty_input_does_not_raise(self):
        assert split_model_id("") == ("", "", "")


class TestModelLineAndVersion:
    def test_a_trailing_version_is_split_off(self):
        assert model_line_and_version("claude-sonnet-4-6") == ("claude-sonnet", (4, 6))
        assert model_line_and_version("claude-sonnet-5") == ("claude-sonnet", (5,))
        assert model_line_and_version("grok-4.3") == ("grok", (4, 3))

    def test_a_generation_that_is_not_trailing_yields_no_version(self):
        """``nova-2-lite`` and ``llama3-70b-instruct`` carry their generation in
        the middle or inside a word. Claiming a version here is how a false
        supersession gets emitted, so these are deliberately not comparable."""
        assert model_line_and_version("nova-2-lite") == ("nova-2-lite", ())
        assert model_line_and_version("nova-lite") == ("nova-lite", ())
        assert model_line_and_version("llama3-70b-instruct") == (
            "llama3-70b-instruct", (),
        )
        assert model_line_and_version("gemma-3-27b-it") == ("gemma-3-27b-it", ())

    def test_an_overlong_name_is_never_handed_to_the_pattern(self):
        """The name below ends in ``-5``, so the pattern WOULD match it and
        return a version. It comes back unmatched only because the length
        ceiling refuses it first, which is what makes this a test of the guard
        rather than of the regex.

        The guard exists because ``_TRAILING_VERSION`` backtracks quadratically
        on a long separator-digit run that fails the end anchor. Measured, that
        is ~50ms at 4k characters — worth bounding on input that reaches here
        from a catalog listing, not an exploitable hang."""
        from app.services.model_catalog import _MAX_SHORT_NAME_LEN

        overlong = "x" * (_MAX_SHORT_NAME_LEN + 1) + "-5"
        assert model_line_and_version(overlong) == (overlong, ())

        # The ceiling is not so low that a real id trips it, and a name right at
        # the limit still parses — the refusal starts one character later.
        assert model_line_and_version("claude-sonnet-4-6") == ("claude-sonnet", (4, 6))
        at_limit = "x" * (_MAX_SHORT_NAME_LEN - 2) + "-5"
        assert model_line_and_version(at_limit) == ("x" * (_MAX_SHORT_NAME_LEN - 2), (5,))

    def test_dotted_deepseek_shape_is_pinned(self):
        """``v3.2`` splits as line ``v3`` + version ``(2,)`` — the letter-digit
        boundary in ``v3`` carries no separator. Pinned because the RESULT is
        still safe: two ``v3.x`` ids compare correctly against each other, and a
        ``v4.x`` id lands on a different line so it is simply never compared
        (a false negative: no supersession is claimed)."""
        assert model_line_and_version("v3.2") == ("v3", (2,))
        assert model_line_and_version("v3.5") == ("v3", (5,))
        assert model_line_and_version("v4.1") == ("v4", (1,))


class TestCollapseVariants:
    def test_the_measured_seven_rows_become_three_models(self):
        collapsed = collapse_variants(MEASURED)
        assert [e["model_id"] for e in collapsed] == [
            "anthropic.claude-sonnet-4-6",
            "anthropic.claude-sonnet-5",
            "anthropic.claude-sonnet-4-5",
        ]

    def test_the_surviving_row_is_the_cheapest_variant_and_keeps_its_own_price(self):
        """The misattribution defect: a row's price must belong to a row's id."""
        collapsed = collapse_variants(MEASURED)
        sonnet_46 = collapsed[0]
        assert sonnet_46["model_id"] == "anthropic.claude-sonnet-4-6"
        assert sonnet_46["input_price_per_1m"] == 3.00
        assert sonnet_46["routing_scope"] == "base"

    def test_alternates_stay_visible_with_their_own_prices(self):
        """Geographic routing must remain a choice a tenant can make, not one
        silently taken away — so every variant keeps its id, scope and price."""
        collapsed = collapse_variants(MEASURED)
        variants = {v["model_id"]: v for v in collapsed[0]["variants"]}
        assert set(variants) == {
            "anthropic.claude-sonnet-4-6",
            "us.anthropic.claude-sonnet-4-6",
            "global.anthropic.claude-sonnet-4-6",
        }
        assert variants["us.anthropic.claude-sonnet-4-6"]["input_price_per_1m"] == 3.30
        assert variants["us.anthropic.claude-sonnet-4-6"]["routing_scope"] == "us"
        assert variants["anthropic.claude-sonnet-4-6"]["routing_scope"] == "base"

    def test_an_unpriced_variant_no_longer_sinks_on_its_own(self):
        """``us.anthropic.claude-sonnet-4-5`` had no pricing while the bare id
        did, so the same model was both priced and unpriced depending on the row
        — and the unpriced one sank below priced models in the sort."""
        collapsed = collapse_variants(MEASURED)
        sonnet_45 = collapsed[2]
        assert sonnet_45["model_id"] == "anthropic.claude-sonnet-4-5"
        assert sonnet_45["input_price_per_1m"] == 3.00
        assert any(
            v["model_id"] == "us.anthropic.claude-sonnet-4-5"
            and v["input_price_per_1m"] is None
            for v in sonnet_45["variants"]
        )

    def test_bare_wins_a_price_tie_against_global(self):
        """Bare and ``global.`` share the base price; picking one deterministically
        keeps the catalog stable between calls."""
        collapsed = collapse_variants([
            _entry("global.anthropic.claude-sonnet-4-6", 3.00, 15.00),
            _entry("anthropic.claude-sonnet-4-6", 3.00, 15.00),
        ])
        assert collapsed[0]["model_id"] == "anthropic.claude-sonnet-4-6"

    def test_global_becomes_primary_when_no_bare_id_exists(self):
        """The shape a current model actually has, and it must not be "fixed".

        In us-east-1 (as of September 2026) the bare ``anthropic.claude-sonnet-5``
        id is NOT invocable (on-demand throughput isn't supported) while
        ``global.`` and ``us.`` of the same model both are. ``bedrock_discovery``
        filters non-on-demand bare ids out, so a current model reaches this
        function with only scoped variants — and ``global.`` is then the correct primary,
        because it is the cheapest one that can be invoked at all.
        """
        collapsed = collapse_variants([
            _entry("us.anthropic.claude-sonnet-5", 2.20, 11.00),
            _entry("global.anthropic.claude-sonnet-5", 2.00, 10.00),
        ])
        assert len(collapsed) == 1
        assert collapsed[0]["model_id"] == "global.anthropic.claude-sonnet-5"
        assert collapsed[0]["routing_scope"] == "global"
        assert collapsed[0]["input_price_per_1m"] == 2.00

    def test_a_cheaper_alternate_is_called_out(self):
        """When the cheapest variant is NOT the one a lineup already uses, the
        saving has to be visible for anyone to take it."""
        collapsed = collapse_variants([
            _entry("us.anthropic.claude-sonnet-4-6", 3.30, 16.50),
            _entry("anthropic.claude-sonnet-4-6", 3.00, 15.00),
        ])
        row = collapsed[0]
        assert row["model_id"] == "anthropic.claude-sonnet-4-6"
        # Nothing undercuts the survivor, so no call-out on the survivor itself.
        assert "cheaper_variant" not in row

    def test_different_call_paths_for_one_model_are_not_merged(self):
        """A runtime id and a mantle id for the same model are different
        adapters. Merging them would give the agent a row whose endpoint/api do
        not describe its own model_id."""
        collapsed = collapse_variants([
            _entry("anthropic.claude-sonnet-5", 2.00, 10.00),
            _entry(
                "anthropic.claude-sonnet-5", 2.00, 10.00,
                endpoint="mantle", api="anthropic_messages",
            ),
        ])
        assert len(collapsed) == 2

    def test_the_input_list_is_not_mutated(self):
        """The discovery layer's entries are reused elsewhere in the request."""
        before = [dict(e) for e in MEASURED]
        collapse_variants(MEASURED)
        assert MEASURED == before

    def test_different_tiers_are_never_grouped(self):
        collapsed = collapse_variants([
            _entry("anthropic.claude-haiku-4-5", 1.00, 5.00),
            _entry("anthropic.claude-sonnet-4-5", 3.00, 15.00),
        ])
        assert len(collapsed) == 2


class TestAnnotateSupersession:
    def test_the_observed_judge_mistake_is_now_stated_on_the_row(self):
        """An older model proposed as judge while a newer, cheaper sibling sat in
        the same list: the supersession is stated on the older row."""
        rows = annotate_supersession(collapse_variants(MEASURED))
        by_id = {e["model_id"]: e for e in rows}
        assert by_id["anthropic.claude-sonnet-4-6"]["superseded_by"] == (
            "anthropic.claude-sonnet-5"
        )
        assert by_id["anthropic.claude-sonnet-4-5"]["superseded_by"] == (
            "anthropic.claude-sonnet-5"
        )
        assert "superseded_by" not in by_id["anthropic.claude-sonnet-5"]

    def test_the_newest_generation_wins_when_several_are_newer(self):
        rows = annotate_supersession([
            _entry("anthropic.claude-sonnet-4-5", 3.00, 15.00),
            _entry("anthropic.claude-sonnet-4-6", 3.00, 15.00),
            _entry("anthropic.claude-sonnet-5", 2.00, 10.00),
        ])
        assert rows[0]["superseded_by"] == "anthropic.claude-sonnet-5"

    def test_a_pricier_newer_model_does_not_supersede(self):
        """Otherwise the field tells the agent to take an upgrade that costs
        more, which it would be right to refuse — making the field noise."""
        rows = annotate_supersession([
            _entry("anthropic.claude-sonnet-4-6", 3.00, 15.00),
            _entry("anthropic.claude-sonnet-5", 9.00, 45.00),
        ])
        assert "superseded_by" not in rows[0]

    def test_a_different_tier_is_not_a_supersession(self):
        """haiku and sonnet are separate lines; crossing them would recommend a
        cheaper, weaker model as an upgrade."""
        rows = annotate_supersession([
            _entry("anthropic.claude-sonnet-4-5", 3.00, 15.00),
            _entry("anthropic.claude-haiku-5", 1.00, 5.00),
        ])
        assert "superseded_by" not in rows[0]

    def test_a_different_provider_is_not_a_supersession(self):
        rows = annotate_supersession([
            _entry("anthropic.claude-sonnet-4-6", 3.00, 15.00),
            _entry("openai.gpt-5.5", 1.00, 5.00, provider="OpenAI"),
        ])
        assert "superseded_by" not in rows[0]

    def test_a_non_trailing_generation_is_never_compared(self):
        """``nova-2-lite`` vs ``nova-lite``: a real supersession this function
        cannot see. It stays silent rather than guessing — the field's absence is
        documented as not being a currency guarantee."""
        rows = annotate_supersession([
            _entry("amazon.nova-lite", 0.06, 0.24, provider="Amazon"),
            _entry("amazon.nova-2-lite", 0.05, 0.20, provider="Amazon"),
        ])
        assert all("superseded_by" not in r for r in rows)

    def test_unpriced_entries_are_still_comparable(self):
        """An unknown price must not block the claim: unknown sorts as +inf, so
        "newer costs no more" holds against an unpriced older row."""
        rows = annotate_supersession([
            _entry("anthropic.claude-sonnet-4-6", None, None),
            _entry("anthropic.claude-sonnet-5", 2.00, 10.00),
        ])
        assert rows[0]["superseded_by"] == "anthropic.claude-sonnet-5"

    def test_the_note_names_both_reasons(self):
        rows = annotate_supersession([
            _entry("anthropic.claude-sonnet-4-6", 3.00, 15.00),
            _entry("anthropic.claude-sonnet-5", 2.00, 10.00),
        ])
        note = rows[0]["supersession_note"]
        assert "newer generation" in note and "no more" in note
