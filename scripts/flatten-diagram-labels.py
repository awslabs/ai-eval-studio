#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rewrite a draw.io SVG export's HTML labels as real SVG text.

draw.io emits every ``html=1`` label twice: an ``<foreignObject>`` holding XHTML,
and a ``<text>`` fallback, wrapped in a ``<switch>``. Browsers advertise the
SVG 1.1 Extensibility feature the switch tests for, so they pick the
``foreignObject`` branch -- and an SVG referenced by an ``<img>`` (which is how
GitHub, GitLab and every markdown renderer embed it) does not render HTML
inside a ``foreignObject`` at all. The labels come out blank. It looks correct
in a local browser, because opening the file directly is not image mode.

The ``<text>`` fallback is not a fix on its own: draw.io truncates it to the
first line plus an ellipsis for any multi-line label. Of the labels in the
architecture diagram, half are multi-line, so keeping only the fallback would
silently drop most of the text.

So this rewrites each switch into ``<text>`` elements built from the
foreignObject's own content -- one per visual line, with ``<tspan>`` runs where
a line mixes weights. Positions come from the fallback ``<text>`` draw.io
already computed: x, text-anchor, fill, font-family and font-size are reused
verbatim, and only the per-line vertical offset is added. That was checked
against every label in the diagram, reproducing draw.io's own x/anchor/baseline
exactly, so the geometry here is not an independent guess at its layout.

HTML wraps a long line to the label's box width; SVG text does not. Lines are
therefore wrapped here, which needs glyph advances -- embedded below rather
than measured, so the result does not depend on which fonts a machine has.
Helvetica and Arial share advance widths by design; these came from Arial and
match the Helvetica metrics (space 278, A 667, bold A 722).

Idempotent: a file with no foreignObject is left untouched. Run it after every
re-export::

    ./scripts/flatten-diagram-labels.py            # rewrite in place
    ./scripts/flatten-diagram-labels.py --check     # exit 1 if any file needs it

``infra/tests/test_diagram_svg_matches_source.py`` fails on an export that
still carries a foreignObject, so a re-export cannot ship unflattened.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

#: Glyph advances per 1000 em, keyed by character.
_WIDTHS = {
    "regular": {
        ' ': 278, '!': 278, '"': 355, '#': 556, '$': 556, '%': 889, '&': 667,
        "'": 191, '(': 333, ')': 333, '*': 389, '+': 584, ',': 278, '-': 333,
        '.': 278, '/': 278, '0': 556, '1': 556, '2': 556, '3': 556, '4': 556,
        '5': 556, '6': 556, '7': 556, '8': 556, '9': 556, ':': 278, ';': 278,
        '<': 584, '=': 584, '>': 584, '?': 556, '@': 1015, 'A': 667, 'B': 667,
        'C': 722, 'D': 722, 'E': 667, 'F': 611, 'G': 778, 'H': 722, 'I': 278,
        'J': 500, 'K': 667, 'L': 556, 'M': 833, 'N': 722, 'O': 778, 'P': 667,
        'Q': 778, 'R': 722, 'S': 667, 'T': 611, 'U': 722, 'V': 667, 'W': 944,
        'X': 667, 'Y': 667, 'Z': 611, '[': 278, '\\': 278, ']': 278, '^': 469,
        '_': 556, '`': 333, 'a': 556, 'b': 556, 'c': 500, 'd': 556, 'e': 556,
        'f': 278, 'g': 556, 'h': 556, 'i': 222, 'j': 222, 'k': 500, 'l': 222,
        'm': 833, 'n': 556, 'o': 556, 'p': 556, 'q': 556, 'r': 333, 's': 500,
        't': 278, 'u': 556, 'v': 500, 'w': 722, 'x': 500, 'y': 500, 'z': 500,
        '{': 334, '|': 260, '}': 334, '~': 584, '\xa0': 278, '·': 333, '—':
        1000, '→': 1000, '≥': 549
    },
    "bold": {
        ' ': 278, '!': 333, '"': 474, '#': 556, '$': 556, '%': 889, '&': 722,
        "'": 238, '(': 333, ')': 333, '*': 389, '+': 584, ',': 278, '-': 333,
        '.': 278, '/': 278, '0': 556, '1': 556, '2': 556, '3': 556, '4': 556,
        '5': 556, '6': 556, '7': 556, '8': 556, '9': 556, ':': 333, ';': 333,
        '<': 584, '=': 584, '>': 584, '?': 611, '@': 975, 'A': 722, 'B': 722,
        'C': 722, 'D': 722, 'E': 667, 'F': 611, 'G': 778, 'H': 722, 'I': 278,
        'J': 556, 'K': 722, 'L': 611, 'M': 833, 'N': 722, 'O': 778, 'P': 667,
        'Q': 778, 'R': 722, 'S': 667, 'T': 611, 'U': 722, 'V': 667, 'W': 944,
        'X': 667, 'Y': 667, 'Z': 611, '[': 333, '\\': 278, ']': 333, '^': 584,
        '_': 556, '`': 333, 'a': 556, 'b': 611, 'c': 556, 'd': 611, 'e': 556,
        'f': 333, 'g': 611, 'h': 611, 'i': 278, 'j': 278, 'k': 556, 'l': 278,
        'm': 889, 'n': 611, 'o': 611, 'p': 611, 'q': 611, 'r': 389, 's': 556,
        't': 333, 'u': 611, 'v': 556, 'w': 778, 'x': 556, 'y': 556, 'z': 500,
        '{': 389, '|': 280, '}': 389, '~': 584, '\xa0': 278, '·': 333, '—':
        1000, '→': 1000, '≥': 549
    },
}

#: Advance assumed for a character absent from the tables. Digit width, which
#: is mid-range for Helvetica, so a wrap estimate degrades rather than breaks.
_FALLBACK_ADVANCE = 556

#: draw.io writes this on every label it styles ``html=1``.
_LINE_HEIGHT = 1.2

#: Fraction of the font size between a centred box's middle and the baseline,
#: and between its top and the first baseline. Both were derived from the
#: fallback ``<text>`` draw.io emits and verified against every label.
_CENTRE_BASELINE = 0.3
_TOP_BASELINE = 1.0

_SWITCH = re.compile(r"<switch>(.*?)</switch>", re.S)
_FOREIGN = re.compile(r"<foreignObject.*?</foreignObject>", re.S)
_OUTER_STYLE = re.compile(r'<div xmlns="http://www\.w3\.org/1999/xhtml" style="([^"]*)"')
_INNER = re.compile(r'<div style="display: inline-block;([^"]*)">(.*?)</div></div></div>', re.S)
_TEXT_TAG = re.compile(r"<text ([^>]*)>")
_BREAK = re.compile(r"<br\s*/?>")
_TAG = re.compile(r"<[^>]+>")
_BOLD_RUN = re.compile(r"<b\s*>(.*?)</b\s*>", re.S)


class LabelError(Exception):
    """A label whose shape this rewriter does not understand.

    Raised rather than guessed at: a silently mispositioned label is the exact
    failure this script exists to remove.
    """


def _advance(text: str, bold: bool, size: float) -> float:
    """Width of ``text`` in px at ``size`` px."""
    table = _WIDTHS["bold" if bold else "regular"]
    return sum(table.get(ch, _FALLBACK_ADVANCE) for ch in text) * size / 1000.0


def _runs(line: str) -> list[tuple[bool, str]]:
    """Split one source line into ``(bold, text)`` runs, in order.

    Collapses runs of ASCII whitespace the way an HTML renderer does. U+00A0 is
    left alone -- draw.io uses it for the gap after a numbered-step marker, and
    collapsing it would close that gap.
    """
    out: list[tuple[bool, str]] = []
    pos = 0
    for match in _BOLD_RUN.finditer(line):
        if match.start() > pos:
            out.append((False, line[pos:match.start()]))
        out.append((True, match.group(1)))
        pos = match.end()
    if pos < len(line):
        out.append((False, line[pos:]))

    cleaned: list[tuple[bool, str]] = []
    for bold, raw in out:
        text = html.unescape(_TAG.sub("", raw))
        text = re.sub(r"[ \t\r\n]+", " ", text)
        if text:
            cleaned.append((bold, text))
    return _coalesce(cleaned)


def _coalesce(runs: list[tuple[bool, str]]) -> list[tuple[bool, str]]:
    """Merge neighbouring runs of the same weight.

    Wrapping splits a run into one token per word, and rendering each of those
    as its own ``<tspan>`` produces markup that says nothing the merged form
    does not. Merging keeps the emitted line greppable as one string.
    """
    merged: list[tuple[bool, str]] = []
    for bold, text in runs:
        if merged and merged[-1][0] == bold:
            merged[-1] = (bold, merged[-1][1] + text)
        else:
            merged.append((bold, text))
    return merged


def _wrap(runs: list[tuple[bool, str]], width: float, size: float) -> list[list[tuple[bool, str]]]:
    """Greedy word-wrap ``runs`` to ``width`` px, preserving run boundaries.

    Greedy because that is what a browser does for this content: no hyphenation
    and ``overflow-wrap: normal``, so the break points are the spaces.
    """
    if width <= 2:                       # draw.io's "unconstrained" 1px box
        return [runs]

    tokens: list[tuple[bool, str]] = []  # words and the spaces between them
    for bold, text in runs:
        for piece in re.split(r"( )", text):
            if piece:
                tokens.append((bold, piece))

    lines: list[list[tuple[bool, str]]] = []
    current: list[tuple[bool, str]] = []
    used = 0.0
    for bold, piece in tokens:
        advance = _advance(piece, bold, size)
        if piece == " ":
            if current:
                current.append((bold, piece))
                used += advance
            continue
        if current and used + advance > width:
            while current and current[-1][1] == " ":
                current.pop()
            lines.append(current)
            current, used = [(bold, piece)], advance
        else:
            current.append((bold, piece))
            used += advance
    if current:
        lines.append(current)
    return [_coalesce(line) for line in lines] or [[]]


def _escape(text: str) -> str:
    """XML-escape label text, keeping U+00A0 explicit so it survives editors."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\u00a0", "&#160;")
    )


def _render_line(line: list[tuple[bool, str]], attrs: str, y: float, base_bold: bool) -> str:
    """One visual line as a ``<text>`` element."""
    if len(line) == 1 and line[0][0] == base_bold:
        content = _escape(line[0][1])
    else:
        content = "".join(
            _escape(text)
            if bold == base_bold
            else f'<tspan font-weight="{"bold" if bold else "normal"}">{_escape(text)}</tspan>'
            for bold, text in line
        )
    return f'<text {attrs} y="{y:g}">{content}</text>'


def _rewrite_switch(block: str) -> str:
    """Replace one ``<switch>`` with the ``<text>`` elements it should have been."""
    foreign = _FOREIGN.search(block)
    if not foreign:
        return f"<switch>{block}</switch>"

    outer = _OUTER_STYLE.search(foreign.group(0))
    inner = _INNER.search(foreign.group(0))
    if not outer or not inner:
        raise LabelError("foreignObject does not have draw.io's usual two-div shape")

    def field(pattern: str, source: str, name: str) -> str:
        match = re.search(pattern, source)
        if not match:
            raise LabelError(f"label is missing {name}")
        return match.group(1)

    align = field(r"align-items: unsafe (\S+);", outer.group(1), "align-items")
    box_width = float(field(r"width: ([\d.]+)px;", outer.group(1), "width"))
    padding = float(field(r"padding-top: (-?[\d.]+)px;", outer.group(1), "padding-top"))
    size = float(field(r"font-size: ([\d.]+)px;", inner.group(1), "font-size"))
    wrapping = re.search(r"white-space: (\w+);", inner.group(1))
    base_bold = "font-weight: bold" in inner.group(1)

    tail = block[foreign.end():]
    text_tag = _TEXT_TAG.search(tail)
    if not text_tag:
        raise LabelError("switch has no <text> fallback to take coordinates from")
    attrs = text_tag.group(1)
    baseline = float(field(r'y="(-?[\d.]+)"', attrs, "y"))
    attrs = re.sub(r'\s*y="-?[\d.]+"', "", attrs).strip()

    source_lines = _BREAK.split(inner.group(2))
    visual: list[list[tuple[bool, str]]] = []
    for line in source_lines:
        runs = _runs(line)
        if not runs:
            visual.append([])                      # a deliberate blank line
        elif wrapping and wrapping.group(1) == "normal":
            visual.extend(_wrap(runs, box_width, size))
        else:
            visual.append(runs)

    step = _LINE_HEIGHT * size
    if align == "flex-start":
        first = baseline                           # draw.io already put it at the top
    elif align == "center":
        first = baseline - (len(visual) - 1) * step / 2
    else:
        raise LabelError(f"unsupported align-items: {align!r}")

    rendered = [
        _render_line(line, attrs, first + index * step, base_bold)
        for index, line in enumerate(visual)
        if line
    ]
    return "".join(rendered)


def flatten(svg: str) -> tuple[str, int]:
    """Rewrite every HTML label in ``svg``.

    Returns:
        The rewritten document and the number of labels changed.

    Raises:
        LabelError: if a label's text would not survive the rewrite, so a
            silent loss of words is impossible.
    """
    changed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        if "<foreignObject" not in match.group(1):
            return match.group(0)
        changed += 1
        return _rewrite_switch(match.group(1))

    out = _SWITCH.sub(replace, svg)

    if changed:
        if "foreignObject" in out:
            raise LabelError("a foreignObject survived the rewrite")
        lost = _words(_authoritative_text(svg)) - _words(out)
        if lost:
            raise LabelError(f"words lost in the rewrite: {sorted(lost)}")
    return out, changed


def _authoritative_text(svg: str) -> str:
    """The input reduced to the text the output has to reproduce.

    Each rewritten switch collapses to its foreignObject alone. Its sibling
    ``<text>`` fallback has to be dropped: draw.io truncates that to "First
    line..." on a multi-line label, and the trailing ellipsis is a word the
    output correctly does not contain, so keeping it would fail every time.
    """
    def reduce(match: re.Match[str]) -> str:
        foreign = _FOREIGN.search(match.group(1))
        return foreign.group(0) if foreign else match.group(0)

    return _SWITCH.sub(reduce, svg)


def _words(fragment: str) -> set[str]:
    """Every word in a fragment, for the no-text-lost assertion."""
    text = html.unescape(_TAG.sub(" ", fragment))
    return {w for w in re.split(r"[\s\u00a0]+", text) if w}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path,
                        help="SVG files (default: every SVG under docs/diagrams)")
    parser.add_argument("--check", action="store_true",
                        help="report files that need rewriting; change nothing")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    paths = args.paths or sorted((root / "docs" / "diagrams").glob("*.svg"))
    if not paths:
        print("no SVG files found", file=sys.stderr)
        return 1

    pending = 0
    for path in paths:
        original = path.read_text(encoding="utf-8")
        try:
            rewritten, changed = flatten(original)
        except LabelError as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            return 1
        if not changed:
            print(f"{path.name}: already SVG text, unchanged")
            continue
        pending += 1
        if args.check:
            print(f"{path.name}: {changed} HTML label(s) need flattening", file=sys.stderr)
            continue
        path.write_text(rewritten, encoding="utf-8")
        print(f"{path.name}: rewrote {changed} HTML label(s) as SVG text")

    if args.check and pending:
        print("run ./scripts/flatten-diagram-labels.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
