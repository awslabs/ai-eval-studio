# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog normalization: one row per model, and supersession stated as data.

Two defects, one pass, because both come from the catalog listing raw ids and
leaving the reasoning to the agent.

**Mixed pricing bases.** The same model appears under up to three ids at two
price points — with Bedrock list prices as of 2026-09-02,
``anthropic.claude-sonnet-4-6`` at $3.00/$15.00,
``us.anthropic.claude-sonnet-4-6`` at $3.30/$16.50, ``global.…-4-6`` back at
$3.00/$15.00. Geographic cross-Region (``us.``) sits exactly 10% above the bare
and ``global.`` price per AWS's global-cross-region-inference docs. A lineup can
span both bases, so cheapest-first ordering and per-run cost projections would
compare +10% prices against base prices, and an agent could quote one variant's
price while naming another variant's id.
Collapsing the variants of one model into ONE row removes the comparison: a row's
price belongs to the id in that row, and every alternate is listed with its own
price and routing scope so choosing geographic routing for data residency stays
a visible choice rather than a silent one.

**Supersession left to recall.** An agent choosing from the cheapest-first list
can propose ``us.anthropic.claude-sonnet-4-6`` as judge while
``anthropic.claude-sonnet-5`` sits ABOVE it, newer AND cheaper. The data is
present and favourable, and the rule is already written in the coordinator
prompt — restating it again is not the fix. So the row STATES it:
``superseded_by`` is computed here, the way the price sort is computed here, and
the prompt only has to forbid proposing a row that carries the field without a
written reason. A check on data, not on recall.

WHERE THE DUPLICATES COME FROM, since it is not where you would guess.
``bedrock_discovery._merge_runtime_first`` already dedups MANTLE against RUNTIME
by short id (runtime wins), and ``_list_runtime_catalog``
dedups runtime entries by exact id across regions. What neither does is dedup two
RUNTIME entries against each other: ``list_inference_profiles`` returns every
SYSTEM_DEFINED profile, so ``us.``, ``global.`` and ``eu.`` variants of one model
all survive as separate rows — which is exactly the mixed-basis comparison.
Grouping here therefore includes ``endpoint`` and ``api``: a runtime id and a
mantle id are different adapters, and merging them would reintroduce that mixed basis and hand
the agent a row whose routing fields do not describe its own ``model_id``.

DERIVATION DISCIPLINE. Every id is split with :func:`short_model_id` — the
existing parity-tested derivation — rather than a second scan with its own
provider and region lists. That module's whole history is about hardcoded lists
breaking on each new provider (xAI, Gemma, DeepSeek, dotted versions), and this
module would have reproduced it.

CLAIMS ONLY WHAT IT CAN DERIVE. Supersession needs a comparable version, and only
a TRAILING numeric version is unambiguous: ``claude-sonnet-4-6`` → 4.6 and
``claude-sonnet-5`` → 5 compare cleanly, while ``nova-2-lite`` carries its
generation in the middle and ``llama3-70b-instruct`` inside a word. Those get NO
claim. A false negative leaves today's behaviour; a false positive would tell the
agent to skip a model it should have considered, so absence of the field is
explicitly not evidence that a model is current.
"""
from __future__ import annotations

import re
from typing import Any

from app.services.model_id_short import short_model_id

#: A trailing version made of numeric segments: ``-4-6``, ``-5``, ``-4.3``.
#: Anchored at the end so a generation embedded mid-name (``nova-2-lite``) is
#: deliberately NOT matched — see the module docstring.
#:
#: Its backtracking is bounded by ``_MAX_SHORT_NAME_LEN`` below: the one caller
#: refuses a longer name before this pattern ever sees it.
# nosemgrep: regex_dos
_TRAILING_VERSION = re.compile(r"[-.](\d+(?:[-.]\d+)*)$")

#: Longest model name this module will pattern-match. The repeated group in
#: ``_TRAILING_VERSION`` backtracks quadratically against a long run of
#: separator-digit pairs that fails the end anchor, so the input gets a ceiling
#: rather than the pattern getting cleverer. Real Bedrock short names run well
#: under 100 characters; anything past this is not a model id, and the fail-safe
#: for an unparseable name is already "claim no version".
_MAX_SHORT_NAME_LEN = 256


def split_model_id(model_id: str) -> tuple[str, str, str]:
    """Split a Bedrock id into ``(routing_scope, provider, short_name)``.

    Structural, with no provider or region lists: a Bedrock id is
    ``[<scope>.]<provider>.<model-name-with-version>``, and
    :func:`short_model_id` already locates where the model name starts. Whatever
    precedes it is one segment (provider only) or two (scope + provider).

    Args:
        model_id: Full Bedrock model identifier.

    Returns:
        ``(scope, provider, short_name)``. ``scope`` is ``""`` for a bare id —
        which is the base-priced variant — and e.g. ``"us"``, ``"eu"``,
        ``"global"`` otherwise. ``provider`` is ``""`` when the id carries none.
    """
    short = short_model_id(model_id)
    if not model_id or not short:
        return "", "", short
    try:
        leading = model_id[: model_id.index(short)]
    except ValueError:  # short is not a substring — nothing to split on
        return "", "", short
    segments = [seg for seg in leading.split(".") if seg]
    if len(segments) >= 2:
        return segments[0], segments[1], short
    if len(segments) == 1:
        return "", segments[0], short
    return "", "", short


def model_line_and_version(short_name: str) -> tuple[str, tuple[int, ...]]:
    """Split a short model name into its family line and trailing version.

    Args:
        short_name: Output of :func:`short_model_id`, e.g. ``claude-sonnet-4-6``.

    Returns:
        ``(line, version)``. ``version`` is empty when the name carries no
        trailing numeric segment, and the line is then the whole name — which is
        what stops ``nova-2-lite`` being compared against ``nova-lite`` on a
        generation this function cannot see. A name longer than
        ``_MAX_SHORT_NAME_LEN`` is returned unmatched for the same reason,
        without being handed to the pattern.
    """
    if len(short_name or "") > _MAX_SHORT_NAME_LEN:
        return short_name or "", ()
    match = _TRAILING_VERSION.search(short_name or "")
    if not match:
        return short_name or "", ()
    digits = tuple(int(part) for part in re.split(r"[-.]", match.group(1)))
    return short_name[: match.start()], digits


def _price(entry: dict[str, Any], key: str) -> float:
    """Numeric price for sorting, with unknown and ``"n/a"`` as +inf."""
    value = entry.get(key)
    return float(value) if isinstance(value, (int, float)) else float("inf")


def _variant_record(entry: dict[str, Any]) -> dict[str, Any]:
    """One alternate id, with the price that belongs to IT."""
    scope, _provider, _short = split_model_id(str(entry.get("model_id", "")))
    return {
        "model_id": entry.get("model_id"),
        "routing_scope": scope or "base",
        "input_price_per_1m": entry.get("input_price_per_1m"),
        "output_price_per_1m": entry.get("output_price_per_1m"),
    }


def collapse_variants(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse cross-Region variants of one model into a single row.

    The surviving row is the CHEAPEST priced variant, with the bare id winning a
    tie — bare and ``global.`` share the base price, and preferring the cheaper
    one is the ~10% saving on the dominant cost line. It is not applied silently:
    every alternate stays on the row under ``variants`` with its own price and
    routing scope, so a tenant that needs geographic routing for data residency
    can see and pick it.

    A BARE ID IS OFTEN ABSENT, and that is correct rather than a gap to close.
    As observed in us-east-1 on 2026-09-06 (Bedrock availability changes over
    time): ``anthropic.claude-sonnet-5`` is NOT
    invocable (``ValidationException``: on-demand throughput isn't supported)
    while ``global.``/``us.`` of the same model both are, and 60 of 122 listed
    TEXT models are inference-profile-only or provisioned-only in the same way.
    ``bedrock_discovery`` therefore filters non-on-demand bare ids out before
    they reach here, which is why ``global.<provider>.<model>`` legitimately
    becomes the primary row for current models — it is the cheapest variant that
    can actually be invoked. Do not "fix" the tie-break to prefer bare harder:
    the tie-break only ever sees a bare id that discovery proved reachable.

    Grouping is by ``(provider, short_name)`` — the same derivation the display
    name uses — so ``us.anthropic.claude-sonnet-4-6-v1:0`` and
    ``anthropic.claude-sonnet-4-6`` are one model, while ``claude-sonnet-5`` and
    ``claude-haiku-4-5`` stay separate.

    Entries that share a group but differ in ``endpoint``/``api`` (a runtime id
    beside a mantle id for the same model) are NOT merged: they are different
    call paths with different adapters, and merging them would hand the agent a
    row whose routing fields do not describe its own ``model_id``.

    Args:
        entries: Catalog entries, already pricing-enriched.

    Returns:
        A NEW list, incoming order preserved by first appearance. Each collapsed
        row gains ``routing_scope``, ``variants`` (2+ entries only), and
        ``cheaper_variant`` when an alternate undercuts the survivor.
    """
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    order: list[tuple[str, str, str, str]] = []
    for entry in entries:
        _scope, provider, short = split_model_id(str(entry.get("model_id", "")))
        key = (
            provider,
            short,
            str(entry.get("endpoint", "")),
            str(entry.get("api", "")),
        )
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(entry)

    collapsed: list[dict[str, Any]] = []
    for key in order:
        members = groups[key]
        # Cheapest first, bare id winning a tie (scope "" sorts before any).
        ranked = sorted(
            members,
            key=lambda e: (
                _price(e, "input_price_per_1m"),
                _price(e, "output_price_per_1m"),
                split_model_id(str(e.get("model_id", "")))[0] != "",
                str(e.get("model_id", "")),
            ),
        )
        primary = dict(ranked[0])
        scope, _provider, _short = split_model_id(str(primary.get("model_id", "")))
        primary["routing_scope"] = scope or "base"
        if len(ranked) > 1:
            primary["variants"] = [_variant_record(m) for m in ranked]
            cheaper = next(
                (
                    m for m in ranked[1:]
                    if _price(m, "input_price_per_1m")
                    < _price(primary, "input_price_per_1m")
                ),
                None,
            )
            if cheaper is not None:
                primary["cheaper_variant"] = _variant_record(cheaper)
        collapsed.append(primary)
    return collapsed


def annotate_supersession(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark every entry that a newer, no-more-expensive sibling supersedes.

    Same provider, same family line, strictly higher trailing version, and a
    price that is not higher. All four conditions are required: the version stops
    a tier comparison (haiku vs sonnet), and the price condition stops the field
    pointing at an upgrade that costs more, which the agent would be right to
    refuse and which would make the field noise.

    Mutates in place and returns the same list.

    Args:
        entries: Catalog entries, pricing-enriched. Run AFTER
            :func:`collapse_variants` so a model is compared once, on one
            pricing basis, rather than once per cross-Region variant.

    Returns:
        The same list. Each superseded entry gains ``superseded_by`` — the newer
        id — and ``supersession_note`` explaining the comparison. An entry
        WITHOUT the field is not thereby current: a name whose generation is not
        a trailing number (``nova-2-lite``, ``llama3-70b-instruct``) is never
        compared at all.
    """
    parsed = []
    for entry in entries:
        _scope, provider, short = split_model_id(str(entry.get("model_id", "")))
        line, version = model_line_and_version(short)
        parsed.append((entry, provider, line, version))

    for entry, provider, line, version in parsed:
        if not version:
            continue
        best: tuple[tuple[int, ...], dict[str, Any]] | None = None
        for other, o_provider, o_line, o_version in parsed:
            if other is entry or not o_version:
                continue
            if (o_provider, o_line) != (provider, line):
                continue
            if o_version <= version:
                continue
            if _price(other, "input_price_per_1m") > _price(entry, "input_price_per_1m"):
                continue
            if best is None or o_version > best[0]:
                best = (o_version, other)
        if best is not None:
            newer = best[1]
            entry["superseded_by"] = newer.get("model_id")
            entry["supersession_note"] = (
                f"{newer.get('model_id')} is a newer generation of the same "
                "model line and costs no more per input token."
            )
    return entries
