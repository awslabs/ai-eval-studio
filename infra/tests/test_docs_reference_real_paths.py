# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every path a doc points at must exist.

Defects of this shape recur, so they get a gate rather than another
correction: the deploy runbook's ``cd infra``, the diagram's "Reachable only
from CloudFront", and ``naming-conventions.md`` prescribing
``tests/mcp_tools/test_data.py`` when the repo has always used the flat
``tests/test_mcp_tools_data.py``. Different files, one failure mode — prose that
reads fine and is false, which a human review pass does not catch because
nothing about it looks wrong.

Scope is deliberately narrow, because a noisy gate gets deleted:

* Only references containing a ``/`` are checked. A bare `` `cdk.json` `` is
  prose naming a file, not a claim about where it lives; treating those as paths
  produced ~90 false positives on the first attempt.
* A reference resolves if it is the SUFFIX of any tracked path, because docs
  reasonably write ``lib/themes.ts`` for ``frontend/src/lib/themes.ts``.
  Enumerating allowed roots instead was a losing game — see ``tracked_paths``.
* Deliberately-historical references are allowlisted individually. Those are
  real — ``naming-conventions.md`` explains renames by naming the pre-refactor
  file — so the escape hatch has to exist, but each entry needs a reason.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_DOCS = sorted(
    [*(_REPO_ROOT / "docs").rglob("*.md"), _REPO_ROOT / "README.md",
     _REPO_ROOT / "infra" / "README.md"]
)

@pytest.fixture(scope="module")
def tracked_paths() -> frozenset[str]:
    """Every git-tracked path, for suffix matching.

    Suffix matching rather than a list of allowed roots: docs legitimately write
    partial paths (``compare/SummaryTable.tsx`` for
    ``frontend/src/components/eval/compare/SummaryTable.tsx``), and enumerating
    roots to cover those turned into a losing game. Permissive about WHERE a
    file is, strict about whether it exists — which is the property that
    matters. A reference like ``tests/mcp_tools/test_data.py`` is no tracked
    path's suffix, so it is still caught.

    ``git ls-files`` also gets the ignore rules for free, so venvs, node_modules
    and cdk.out never enter the set.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # No git, or not a repository — e.g. a ZIP download.
        pytest.skip("needs a git index to resolve path references")
    tracked = frozenset(out.stdout.splitlines())
    if not tracked:
        # An empty index (a tree with .git but nothing committed) exits 0, so
        # check=True does not fire. Without this the suffix-match half of the
        # gate silently vanishes and every reference looks broken.
        pytest.skip("git index is empty — cannot resolve path references")
    return tracked

#: References that intentionally name something absent. Each needs a reason —
#: an entry without one is indistinguishable from an unfixed staleness bug.
_ALLOWED_ABSENT = {
    # naming-conventions.md teaches by contrast, citing files as they were
    # BEFORE a rename. The whole point is that these no longer exist.
    "services/eval_tools.py": "pre-refactor example: split into mcp_tools/",
    "services/mcp_server.py": "pre-refactor example: renamed mcp_server_cache.py",
    # Gitignored artifacts. They are correct to document and correct to be
    # absent from a fresh clone, so requiring them on disk would make this gate
    # unpassable for anyone who has not yet run the thing that creates them.
    ".dev-logs/backend.log": "created by scripts/start.sh",
    ".dev-logs/frontend.log": "created by scripts/start.sh",
    "backend/.venv": "created by the documented setup",
    "infra/.venv": "created by the documented setup",
    "infra/cdk/cdk.context.json": "written by dev-postdeploy.sh after a deploy",
}

#: Things that match a path shape but are not paths.
_NOT_PATHS = re.compile(
    r"""^(
        \d+\.\d+\.\d+\.\d+(/\d+)?      # a CIDR, e.g. 0.0.0.0/0
        | .*\.\.\.                      # an elision, e.g. ./infra/...
        | (bg|text|border|data)-.*      # tailwind / data-attribute tokens
        | local/.*                      # eslint rule ids
        | https?:.*
    )$""",
    re.VERBOSE,
)


def _candidates(text: str) -> set[str]:
    """Path-like references in a markdown document."""
    found = set(re.findall(r"`([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)`", text))
    found |= set(re.findall(r"\]\((?!http)([^)#\s]+)", text))
    return {c.strip().rstrip("/") for c in found}


def _resolves(doc: Path, ref: str, tracked: frozenset[str]) -> bool:
    if (doc.parent / ref).exists() or (_REPO_ROOT / ref).exists():
        return True
    suffix = "/" + ref
    return any(path.endswith(suffix) for path in tracked)


@pytest.mark.parametrize("doc", _DOCS, ids=lambda p: p.name)
def test_every_path_like_reference_resolves(doc, tracked_paths):
    broken = sorted(
        ref
        for ref in _candidates(doc.read_text())
        if ref
        and "*" not in ref
        and not _NOT_PATHS.match(ref)
        and ref not in _ALLOWED_ABSENT
        and not _resolves(doc, ref, tracked_paths)
    )

    assert not broken, (
        f"{doc.relative_to(_REPO_ROOT)} points at paths that do not exist: "
        f"{broken}. Either fix the reference, or if it names something "
        f"deliberately absent (a pre-refactor file), add it to _ALLOWED_ABSENT "
        f"with the reason."
    )


def test_the_allowlist_does_not_hide_paths_that_now_exist(tracked_paths):
    """A stale allowlist is its own trap.

    If an allowlisted path comes back, the entry silently stops meaning what it
    says and would mask a later real break at that path.
    """
    resurrected = {
        ref: reason
        for ref, reason in _ALLOWED_ABSENT.items()
        if any(path.endswith("/" + ref) for path in tracked_paths)
    }
    assert not resurrected, (
        f"these are allowlisted as absent but now exist: {resurrected}. Drop "
        "them from _ALLOWED_ABSENT."
    )


def test_the_allowlist_entries_carry_a_reason():
    for ref, reason in _ALLOWED_ABSENT.items():
        assert reason and len(reason) > 10, (
            f"_ALLOWED_ABSENT[{ref!r}] needs a real reason, not {reason!r}"
        )
