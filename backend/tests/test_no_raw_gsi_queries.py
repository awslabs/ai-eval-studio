# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enforce the GSI chokepoint rule.

All DynamoDB GSI queries must go through ``app/platform/metadata_store.py``
(Protocol) and ``app/platform/aws/dynamodb_metadata_store.py`` (cloud
implementation). This test fails if any other file makes a raw
``IndexName=...`` query, which would bypass the in-code authorization
check that enforces ``caller_user_id == target_user_id``.

IAM ``dynamodb:LeadingKeys`` only scopes the base table's partition key.
GSI partition keys are *not* scoped by the session policy — application
code is the only enforcement point. Adding a new GSI query outside the
allowlisted files would silently widen the trust boundary, so this test
fails the build on any such addition.

The test inspects file contents (not just paths) and skips matches that
appear in comments or docstrings, so prose explanations of the rule
don't accidentally fail the test.
"""
from __future__ import annotations

import re
from pathlib import Path

# The two files allowed to issue ``IndexName=...`` queries. Any other
# match must be rejected. Paths are relative to the backend root so the
# test is portable across checkout locations.
_ALLOWLISTED_FILES = frozenset({
    "app/platform/metadata_store.py",
    "app/platform/aws/dynamodb_metadata_store.py",
})

# Match ``IndexName=`` and ``IndexName =`` (with optional whitespace).
_GSI_PATTERN = re.compile(r"IndexName\s*=")

# Backend app root used for the recursive scan.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_ROOT = _BACKEND_ROOT / "app"


def _strip_docstrings_and_comments(content: str) -> list[tuple[int, str]]:
    """Return ``(lineno, line)`` pairs with comments + docstrings removed.

    The scanner uses a simple state machine to track triple-quoted
    string literals (``\"\"\"`` and ``'''``) so multi-line docstrings
    that mention ``IndexName=`` in prose don't trip the test. Single-line
    ``#`` comments are also dropped from each retained line.

    The state machine doesn't try to parse Python perfectly — it ignores
    f-strings, escapes, and other corner cases — but it's sufficient
    for the GSI-query check, where any real call site is a function
    argument, not a string literal.
    """
    cleaned: list[tuple[int, str]] = []
    in_triple = False
    triple_char = ""
    for lineno, raw in enumerate(content.splitlines(), start=1):
        line = raw
        out_chars: list[str] = []
        i = 0
        while i < len(line):
            if in_triple:
                # Look for the closing triple-quote of the active style.
                end = line.find(triple_char * 3, i)
                if end == -1:
                    i = len(line)
                else:
                    i = end + 3
                    in_triple = False
                    triple_char = ""
                continue
            # Outside a triple-quoted block: comments terminate the line.
            if line[i] == "#":
                break
            # Detect a triple-quote opening.
            if line[i:i + 3] in ('"""', "'''"):
                triple_char = line[i]
                # Check for an inline close on the same line.
                close = line.find(triple_char * 3, i + 3)
                if close == -1:
                    in_triple = True
                    i = len(line)
                else:
                    i = close + 3
                continue
            out_chars.append(line[i])
            i += 1
        cleaned_line = "".join(out_chars)
        cleaned.append((lineno, cleaned_line))
    return cleaned


def _scan_for_violations() -> list[tuple[Path, int, str]]:
    """Walk ``backend/app`` and collect raw GSI-query offenders.

    Returns:
        List of ``(path, line_number, line_text)`` for every match
        outside :data:`_ALLOWLISTED_FILES` after stripping comments and
        docstrings.
    """
    violations: list[tuple[Path, int, str]] = []
    for path in sorted(_APP_ROOT.rglob("*.py")):
        rel = path.relative_to(_BACKEND_ROOT).as_posix()
        if rel in _ALLOWLISTED_FILES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for lineno, line in _strip_docstrings_and_comments(content):
            if _GSI_PATTERN.search(line):
                violations.append((path, lineno, line.rstrip()))
    return violations


def test_no_raw_gsi_queries_outside_chokepoint() -> None:
    """Fail loudly if a non-allowlisted file calls ``IndexName=`` directly."""
    violations = _scan_for_violations()
    if violations:
        formatted = "\n".join(
            f"  {p.relative_to(_BACKEND_ROOT)}:{ln}: {text}"
            for p, ln, text in violations
        )
        raise AssertionError(
            "Raw GSI query (IndexName=...) found outside the chokepoint. "
            "All GSI queries must route through "
            "app/platform/metadata_store.py and "
            "app/platform/aws/dynamodb_metadata_store.py. Offenders:\n"
            + formatted
        )
