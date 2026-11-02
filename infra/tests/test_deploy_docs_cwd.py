# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard on the working directory the deploy docs tell you to use.

`docs/runbooks/deploy.md` said::

    cd infra
    cdk deploy --all --require-approval never

`cdk.json` is at `infra/cdk/cdk.json`, so from `infra/` every cdk command dies
with ``--app is required either in command-line, in cdk.json or in
~/.cdk.json``. That text had never been executed as written.

This is the failure mode a staleness review cannot catch by reading: the prose
is coherent, the commands are plausible, and only running them reveals the
error. So gate it instead. The check is deliberately structural — it does not
know the right answer, it derives it by locating `cdk.json` — so moving the CDK
app makes the test demand the docs move with it rather than going quietly
stale.

Scope: the directory claim only. Nothing here proves a deploy works.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Docs whose bash blocks a reader copies verbatim. ``teardown.md`` is here
#: because it had the same defect one file over — a bare ``cdk destroy --all``
#: with no directory at all — and was not covered when deploy.md was fixed.
_DOCS = (
    "docs/runbooks/deploy.md",
    "docs/runbooks/teardown.md",
    "infra/README.md",
)

#: A line invoking the cdk CLI, optionally through npx, optionally inside a
#: subshell such as ``(cd infra/cdk && cdk deploy ...)``.
_CDK_INVOCATION = re.compile(r"(?:^|\(|&&\s*)(?:npx\s+)?cdk\s+\S", re.MULTILINE)

#: ``cd <path>`` anywhere a reader would act on it, subshells included.
_CD = re.compile(r"(?:^|\(|&&\s*)cd\s+(\S+)", re.MULTILINE)


@pytest.fixture(scope="module")
def cdk_app_dir() -> Path:
    """The directory holding the CDK app's ``cdk.json``, found not assumed."""
    matches = sorted(_REPO_ROOT.glob("infra/**/cdk.json"))
    matches = [m for m in matches if ".venv" not in m.parts and "cdk.out" not in m.parts]
    assert len(matches) == 1, (
        f"expected exactly one CDK app config under infra/, found {matches}. "
        "With more than one, 'the directory to run cdk from' is ambiguous and "
        "this guard cannot speak for the docs."
    )
    return matches[0].parent


def _bash_blocks(doc: Path) -> list[str]:
    """Fenced ```bash blocks in ``doc``, in order."""
    return re.findall(r"```bash\n(.*?)```", doc.read_text(), re.DOTALL)


@pytest.mark.parametrize("doc_path", _DOCS)
def test_no_block_runs_cdk_after_cd_to_the_wrong_directory(doc_path, cdk_app_dir):
    """The exact bug: a ``cd`` into a directory with no ``cdk.json``, then cdk.

    Only blocks that do both are checked. A block that runs cdk without any
    ``cd`` is inheriting a directory from earlier prose, which is a
    readability question rather than a command that cannot work.
    """
    expected = cdk_app_dir.relative_to(_REPO_ROOT).as_posix()  # "infra/cdk"
    doc = _REPO_ROOT / doc_path

    for index, block in enumerate(_bash_blocks(doc)):
        if not _CDK_INVOCATION.search(block):
            continue
        targets = _CD.findall(block)
        if not targets:
            continue

        # Only ``cd``s that could still be in effect matter. A trailing
        # ``cd ../..`` handing off to a repo-root command is fine.
        assert any(t.rstrip("/").endswith(expected) for t in targets), (
            f"{doc_path} block #{index + 1} runs cdk after cd'ing to "
            f"{targets!r}, none of which is {expected!r} (where cdk.json is). "
            f"A reader following this literally gets '--app is required'.\n"
            f"----\n{block.strip()}\n----"
        )


@pytest.mark.parametrize("doc_path", _DOCS)
def test_docs_name_the_cdk_app_directory_somewhere(doc_path, cdk_app_dir):
    """A reader must be able to find the directory without guessing.

    Cheap, but it is the property that was missing: the old text mentioned
    ``infra`` and never ``infra/cdk``.
    """
    expected = cdk_app_dir.relative_to(_REPO_ROOT).as_posix()
    assert expected in (_REPO_ROOT / doc_path).read_text(), (
        f"{doc_path} never mentions {expected}, so it does not tell a reader "
        "where the CDK app is"
    )


def _canonical_venv() -> str:
    """The venv path ``scripts/test-infra.sh`` actually runs from.

    Derived rather than hardcoded so the runner stays the single source of
    truth for which environment is real.
    """
    runner = (_REPO_ROOT / "scripts" / "test-infra.sh").read_text()
    match = re.search(r"(infra/\.venv)/lib/python", runner)
    assert match, "scripts/test-infra.sh no longer names a venv path to align with"
    return match.group(1)


@pytest.mark.parametrize("doc_path", _DOCS)
def test_the_docs_name_the_venv_the_test_runner_uses(doc_path):
    """``cdk.json`` runs ``python app.py``, so which venv is active is
    load-bearing and cannot be left implicit."""
    venv = _canonical_venv()
    assert venv in (_REPO_ROOT / doc_path).read_text(), (
        f"{doc_path} does not name {venv}, the venv scripts/test-infra.sh uses"
    )


def test_only_one_virtualenv_exists_under_infra():
    """Exactly one, and it is the canonical one.

    There used to be three. ``infra/cdk/.venv`` looked authoritative for
    sitting beside ``cdk.json`` while nothing read it, and ``infra/.venv-test``
    was a dead container artifact whose interpreter no longer resolved. Each
    extra one is another plausible-looking answer to "which python runs cdk",
    and picking wrong fails in a way that looks like a CDK problem.

    Checks for real venv roots (``pyvenv.cfg``) rather than a fixed list of
    paths, so a fourth one somewhere new is caught too.
    """
    canonical = _canonical_venv()
    candidates = sorted(
        p.parent.relative_to(_REPO_ROOT).as_posix()
        # Depth-limited on purpose: recursing into a 500 MB venv is slow and a
        # venv nested inside a venv is not a case worth chasing.
        for pattern in ("infra/*/pyvenv.cfg", "infra/*/*/pyvenv.cfg")
        for p in _REPO_ROOT.glob(pattern)
    )
    assert candidates == [canonical], (
        f"expected exactly one virtualenv under infra/ ({canonical}), found "
        f"{candidates}. Delete the extras — scripts/test-infra.sh and the docs "
        f"both point at {canonical}."
    )
