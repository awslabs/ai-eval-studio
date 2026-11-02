# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""File-serving paths must refuse symlinks rather than follow them.

``Path.is_file()`` and ``Path.is_dir()`` both follow links, and
``ZipFile.write()`` opens the link's TARGET. So a single symlink planted inside a
project tree turned two read paths into disclosure primitives:

- the harness download packaged the target's BYTES into a zip the user gets
- the file tree walked a directory link and reported every name and size under it

Neither needed an authentication bypass — the caller is a legitimate owner of
their own project. What makes it reachable is that project trees are writable by
things that are not the REST API: the coordinator agent has unjailed Bash with
the project as its cwd, and the eval worker runs agent-authored harness code. On
the shared EFS mount every tenant's tree is a sibling, all squashed to one uid.
"""
import zipfile
from pathlib import Path

import pytest

from app.api.files import _build_file_tree, _build_harness_zip

SECRET = "tenant-b-private-data\n"


@pytest.fixture
def project_with_symlinks(tmp_path: Path) -> Path:
    """A project tree containing a file link and a directory link that escape."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text(SECRET)
    (outside / "nested").mkdir()
    (outside / "nested" / "also-secret.txt").write_text(SECRET)

    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "run_eval.py").write_text("# real harness file\n")
    (project / "src" / "leak.txt").symlink_to(outside / "secret.txt")
    (project / "escape").symlink_to(outside, target_is_directory=True)
    return project


def test_harness_zip_does_not_package_symlink_targets(project_with_symlinks: Path):
    """The reproduced defect: link target bytes ended up inside the archive."""
    buffer = _build_harness_zip(project_with_symlinks)

    with zipfile.ZipFile(buffer) as archive:
        names = archive.namelist()
        contents = [archive.read(n) for n in names]

    assert "run_eval.py" in names, "real files must still be archived"
    assert "src/leak.txt" not in names, "a symlink must not be archived"
    assert all(SECRET.encode() not in blob for blob in contents), (
        f"the link target's bytes reached the archive; members: {names}"
    )


def test_file_tree_does_not_walk_symlinked_directories(project_with_symlinks: Path):
    """Directory links leaked names and sizes of whatever they pointed at."""
    nodes = _build_file_tree(project_with_symlinks, project_with_symlinks)

    def flatten(ns):
        for n in ns:
            yield n
            yield from flatten(n.children or [])

    paths = {n.path for n in flatten(nodes)}
    assert "run_eval.py" in paths, "real files must still be listed"
    assert not any("escape" in p for p in paths), (
        f"a directory symlink was walked; listed: {sorted(paths)}"
    )
    assert not any("secret" in p for p in paths)


def test_harness_zip_refuses_a_tree_over_the_size_ceiling(
    tmp_path: Path, monkeypatch
):
    """The archive is built in memory, so an oversize tree must be refused.

    ``data/`` is deliberately included in the download, and uploads are accepted
    far past what the coordinator can hold, so without a ceiling one request
    could exhaust it.
    """
    from fastapi import HTTPException

    from app.api import files as files_mod

    project = tmp_path / "project"
    (project / "data" / "raw").mkdir(parents=True)
    (project / "data" / "raw" / "big.csv").write_bytes(b"x" * 4096)

    monkeypatch.setattr(files_mod, "HARNESS_ZIP_MAX_BYTES", 1024)

    with pytest.raises(HTTPException) as exc:
        files_mod._build_harness_zip(project)
    assert exc.value.status_code == 413
