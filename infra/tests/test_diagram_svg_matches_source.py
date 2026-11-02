# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The exported diagram must match its ``.drawio`` source.

The export is what people actually look at — it is what README embeds and what a
reviewer opens. Editing the source and forgetting to re-export leaves the
readable artifact stating the old thing. That is not hypothetical twice over:
the ALB box kept claiming "Reachable only from CloudFront" after the source had
been corrected, and later the legend kept describing an unfixed cross-tenant gap
after that sentence had been removed from the source. A diagram is the one
channel a prose review never inspects.

Gating it looks impossible at first: an export is a rendered picture, and
re-rendering it needs draw.io. But draw.io's "Include a copy of my diagram"
export embeds the whole source in the file, so the export carries the source it
was produced from. Comparing that against the on-disk ``.drawio`` detects drift
exactly, with no rendering and no reliance on file mtimes (which git does not
preserve, so a fresh clone would make any timestamp-based check meaningless).

SVG rather than PNG: it is vector, so there is no export-resolution question at
all, it is a fraction of the size, and both GitHub and GitLab render it inline in
markdown. It also embeds its source as a plain XML attribute rather than a binary
PNG ``tEXt`` chunk, so this check needs no struct parsing.

Compares LABELS rather than raw XML: draw.io rewrites geometry and attribute
order on save, which would make a byte comparison fail constantly for no reason.
Labels are the text that can be true or false, which is what staleness is about.
"""

from __future__ import annotations

import base64
import html
import re
import urllib.parse
import zlib
from pathlib import Path

import pytest

_DIAGRAMS = Path(__file__).resolve().parent.parent.parent / "docs" / "diagrams"

#: draw.io stores the round-trip copy on the root element as ``content="..."``.
_CONTENT_ATTR = re.compile(r'\scontent="([^"]*)"')


def _embedded_source(svg: Path) -> str:
    """The ``mxfile`` XML draw.io embedded in an SVG export.

    Args:
        svg: Path to a draw.io SVG export.

    Returns:
        The decoded diagram XML.

    Raises:
        AssertionError: if the SVG carries no embedded source, which means it was
            exported without "Include a copy of my diagram" and drift can no
            longer be detected at all.
    """
    text = svg.read_text(encoding="utf-8", errors="replace")
    match = _CONTENT_ATTR.search(text)
    assert match, (
        f"{svg.name} has no embedded diagram. Re-export from draw.io with "
        "'Include a copy of my diagram' enabled — without it, the export and its "
        "source can drift silently and nothing can tell."
    )
    # draw.io HTML-escapes the attribute; some versions also URL-encode it.
    raw = html.unescape(match.group(1))
    if raw.lstrip().startswith("%"):
        raw = urllib.parse.unquote(raw)

    # The <diagram> body is usually draw.io's COMPRESSED form: base64 of a raw
    # deflate stream of the URL-encoded mxGraphModel. Reading the attribute alone
    # therefore yields an mxfile with no mxCell in it at all, and a label
    # comparison against it silently finds nothing — it reports every label as
    # missing rather than admitting it could not read them.
    body = re.search(r"<diagram[^>]*>([^<]+)</diagram>", raw)
    if body and "<mxCell" not in raw:
        try:
            inflated = zlib.decompress(
                base64.b64decode(body.group(1)), -zlib.MAX_WBITS
            ).decode("utf-8")
        except (ValueError, zlib.error) as exc:  # noqa: PERF203
            raise AssertionError(
                f"{svg.name} carries a <diagram> payload that is neither plain "
                f"XML nor draw.io's base64+deflate form: {exc}"
            ) from exc
        raw = urllib.parse.unquote(inflated)

    assert "<mxCell" in raw, (
        f"{svg.name}'s embedded copy contains no mxCell — it cannot be compared "
        "to the source. Re-export with 'Include a copy of my diagram'."
    )
    return raw


def _labels(xml: str) -> set[str]:
    """Every visible label in a diagram, which is where factual claims live."""
    return set(re.findall(r'value="([^"]*)"', xml))


def _diagram_pairs() -> list[tuple[Path, Path]]:
    return [
        (source, source.with_suffix(".svg"))
        for source in sorted(_DIAGRAMS.glob("*.drawio"))
    ]


@pytest.mark.parametrize(
    "source,svg", _diagram_pairs(), ids=lambda p: getattr(p, "stem", str(p))
)
def test_the_export_carries_the_current_source(source: Path, svg: Path):
    assert svg.exists(), (
        f"{source.name} has no exported {svg.name}. Export it from draw.io:\n"
        "  File > Export as > SVG...\n"
        "  tick 'Include a copy of my diagram'\n"
        f"  save as {svg.name} beside the source\n"
        "Without the export the readable artifact is missing entirely."
    )

    exported = _labels(_embedded_source(svg))
    current = _labels(source.read_text())

    stale = current - exported
    orphaned = exported - current
    assert not stale and not orphaned, (
        f"{svg.name} is out of sync with {source.name} — re-export it from "
        f"draw.io with 'Include a copy of my diagram' enabled.\n"
        f"  in source but not in the export: {sorted(stale)}\n"
        f"  in the export but not in source: {sorted(orphaned)}"
    )


@pytest.mark.parametrize(
    "source,svg", _diagram_pairs(), ids=lambda p: getattr(p, "stem", str(p))
)
def test_the_export_uses_svg_text_not_foreignobject(source: Path, svg: Path):
    """Labels must be real SVG ``<text>``, never ``<foreignObject>``.

    draw.io renders an ``html=1`` label by wrapping XHTML in a
    ``<foreignObject>``, inside a ``<switch>`` that also holds a plain ``<text>``
    fallback. Browsers advertise the SVG 1.1 Extensibility feature the switch
    tests for, so they take the foreignObject branch — and an SVG referenced by
    an ``<img>``, which is how GitHub, GitLab and every markdown renderer embed
    one, does not render HTML inside a foreignObject at all. The labels come out
    blank: boxes with no text. Opening the file directly is not image mode, so
    it looks correct locally, which is how it would ship broken.

    The fallback is not a fix by itself — draw.io truncates it to the first line
    plus an ellipsis on any multi-line label, and half of these are multi-line.
    ``scripts/flatten-diagram-labels.py`` rebuilds the labels as ``<text>`` from
    the foreignObject's own content, reusing the coordinates draw.io computed.
    Run it after every re-export. It also keeps the text greppable, which is the
    reason this diagram is SVG at all.
    """
    if not svg.exists():
        pytest.skip("no export yet — the sibling test reports that")

    text = svg.read_text(encoding="utf-8", errors="replace")
    assert "foreignObject" not in text, (
        f"{svg.name} renders labels as <foreignObject>. In an <img>-embedded "
        "SVG that element renders nothing, so every HTML-formatted label will "
        "be BLANK in the README.\n"
        "Fix: ./scripts/flatten-diagram-labels.py"
    )
    assert "<text" in text, (
        f"{svg.name} contains no <text> element at all — the labels did not "
        "survive the export in any form."
    )


@pytest.mark.parametrize(
    "source,svg", _diagram_pairs(), ids=lambda p: getattr(p, "stem", str(p))
)
def test_every_source_word_is_visible_in_the_export(source: Path, svg: Path):
    """The RENDERED text must carry every word, not just the embedded copy.

    The sibling checks cannot see this. One compares the embedded copy against
    the source — and the embedded copy is a verbatim round-trip, so it agrees
    even when the picture shows nothing. The other only rejects
    ``<foreignObject>``. Keeping draw.io's ``<text>`` fallback and dropping the
    foreignObject satisfies both while losing every line after the first, since
    that fallback is truncated to "First line...".

    So this reads the text nodes, which is what a viewer sees. Attributes are
    stripped along with their tags, which is what excludes ``content=`` — were
    it included the assertion would pass on any export at all.

    Word sets, not strings: the export wraps a long line to the label's box
    width, so line breaks legitimately differ from the source.
    """
    if not svg.exists():
        pytest.skip("no export yet — the sibling test reports that")

    def words(fragment: str) -> set[str]:
        plain = re.sub(r"<[^>]+>", " ", html.unescape(fragment))
        return {w for w in re.split(r"[\s ]+", html.unescape(plain)) if w}

    expected: set[str] = set()
    for value in _labels(source.read_text()):
        expected |= words(value)
    rendered = words(svg.read_text(encoding="utf-8", errors="replace"))

    missing = expected - rendered
    assert not missing, (
        f"{svg.name} does not render {len(missing)} word(s) that "
        f"{source.name} defines — the picture says less than the source:\n"
        f"  {sorted(missing)}\n"
        "Re-export, then run ./scripts/flatten-diagram-labels.py"
    )


def test_no_stale_png_export_remains():
    """A leftover PNG would still be a readable artifact nobody re-exports.

    The switch to SVG only helps if the PNG is gone: a stale PNG beside a current
    SVG is exactly the drift this file exists to prevent, and the README could
    silently keep pointing at it.
    """
    strays = sorted(p.name for p in _DIAGRAMS.glob("*.png"))
    assert not strays, (
        f"PNG export(s) still present in docs/diagrams: {strays}. The diagrams "
        "are SVG now — delete these so there is one readable artifact, not two "
        "that can disagree."
    )
