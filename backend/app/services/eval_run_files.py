# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which files under a project's ``output/`` directory are eval runs.

``output/`` holds more than eval results. Two other JSON files live beside the
runs: the metric-audit sidecar written next to each run file, and the
active-eval marker recording the running subprocess PID. Reports live one level
down in ``output/reports/``.

Several call sites independently globbed ``output/*.json`` and treated whatever
came back as an eval run. Each one then mis-read a sidecar as results, and the
symptoms looked unrelated — a project list showing no models, a synthesis tool
refusing to find failures, an agent prompt listing sidecars as runs. Naming the
exclusions once means the next sidecar has a single place to be registered
instead of three places to be forgotten.

The filter is a denylist rather than an ``eval_results_*`` allowlist on purpose:
the platform names its runs ``eval_results_*.json`` but a generated harness may
write ``results.json`` or another name of its own, and those are genuine runs.
An allowlist would silently drop them.
"""
from __future__ import annotations

from pathlib import Path

from app.services.active_eval_state import ACTIVE_EVAL_FILENAME
from app.services.metric_disagreement import SIDECAR_SUFFIX

#: Suffixes that mark a JSON file in ``output/`` as something other than a run.
#: ``.report.json`` is listed defensively — reports are written to
#: ``output/reports/`` today, which a non-recursive glob never reaches, so this
#: only matters if a caller ever recurses.
_NON_RUN_SUFFIXES = (SIDECAR_SUFFIX, ".report.json")

#: Exact filenames in ``output/`` that are not runs.
_NON_RUN_NAMES = frozenset({ACTIVE_EVAL_FILENAME})


def is_eval_run_file(path: Path) -> bool:
    """Whether ``path`` names an eval-run results file rather than a sidecar.

    Args:
        path: A path under a project's ``output/`` directory. Only the file name
            is inspected; the file need not exist.

    Returns:
        True unless the name matches a known non-run file.
    """
    name = path.name
    if name in _NON_RUN_NAMES:
        return False
    return not name.endswith(_NON_RUN_SUFFIXES)


def eval_run_files(output_dir: Path) -> list[Path]:
    """Every eval-run results file in ``output_dir``, sorted by file name.

    Name order is ascending, so for the platform's timestamped
    ``eval_results_<label>_<stamp>.json`` names the last element is the most
    recent run. Callers that need modification order should re-sort on
    ``st_mtime`` rather than relying on this.

    Args:
        output_dir: A project's ``output/`` directory. A missing directory is
            not an error.

    Returns:
        Matching files, or an empty list when the directory is absent or holds
        no runs.
    """
    if not output_dir.exists():
        return []
    return sorted(
        (p for p in output_dir.glob("*.json") if p.is_file() and is_eval_run_file(p)),
        key=lambda p: p.name,
    )


__all__ = ["eval_run_files", "is_eval_run_file"]
