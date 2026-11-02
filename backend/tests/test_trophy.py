# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trophy scoring — the single implementation and its golden vector.

The golden file is the cross-language contract: vitest runs the SAME fixture
against the frontend's ordering comparator. A scoring change that edits the
fixture forces both suites to move together; one that doesn't goes red here.
"""

import json

import pytest
from pathlib import Path

from app.services.trophy import (
    model_meets_targets,
    resolve_trophy_targets,
    trophy_score_loop,
    trophy_score_run,
    trophy_sort_key,
)

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "trophy_golden.json").read_text()
)


def test_golden_vector_scores():
    targets = GOLDEN["targets"]
    for cand in GOLDEN["candidates"]:
        fn = trophy_score_loop if cand["kind"] == "loop" else trophy_score_run
        got = fn(cand["row"], targets)
        assert got == cand["expected_score"], cand["name"]


def test_golden_vector_winner_by_ordering():
    """The ordering the frontend replicates: lexicographic tuple, later wins."""
    targets = GOLDEN["targets"]
    best_idx = -1
    best_key = None
    for i, cand in enumerate(GOLDEN["candidates"]):
        fn = trophy_score_loop if cand["kind"] == "loop" else trophy_score_run
        key = trophy_sort_key(fn(cand["row"], targets))
        if best_key is None or key >= best_key:
            best_key = key
            best_idx = i
    assert best_idx == GOLDEN["expected_winner_index"]


def test_suffix_tolerance_is_symmetric():
    targets = {"Answer Correctness": 0.9}
    assert model_meets_targets({"Answer Correctness [GEval]": 0.95}, targets)
    assert model_meets_targets({"Answer Correctness": 0.95}, targets)
    assert not model_meets_targets({"Answer Correctness [GEval]": 0.85}, targets)
    assert not model_meets_targets({"Other Metric": 1.0}, targets)


def test_no_targets_means_unrankable_not_default():
    """Empty targets must not fabricate a bar (no 0.85 ghosts): with nothing to
    clear, every model vacuously 'meets' the target set, making at_target
    degenerate — which is why callers with no targets skip trophy ranking
    entirely rather than rank on this.

    The model run carries per-metric data on purpose. With an EMPTY per_metric
    the score would be 0 for a different reason (the check cannot run at all),
    and the test would pass without ever exercising empty targets."""
    score = trophy_score_run(
        {"model_runs": [{"total": 10, "passed": 1, "per_metric": {"Any Metric": 0.1}}]},
        {},
    )
    assert score["at_target"] == 1  # degenerate by design; callers gate on targets


def test_missing_per_metric_scores_zero_even_with_targets_configured():
    """The deleted loosest-target substitution, stated as its own case."""
    score = trophy_score_run(
        {"model_runs": [{"total": 10, "passed": 10, "per_metric": {}}]},
        {"Strict": 0.9, "Loose": 0.1},
    )
    assert score["at_target"] == 0, (
        "a 100% rate cleared the loosest target and claimed compliance with a "
        "metric set it was never measured against"
    )
    assert score["best_rate"] == 1.0, "the rate still populates the tie-breakers"


def test_pick_trophy_run_ranks_against_config_not_state_snapshot(monkeypatch, tmp_path):
    """The report ranks against the harness config's targets, NOT
    OptimizationState's dispatch-time snapshot — the config owns the
    criteria. A user who edits targets after dispatch must get report and
    panel crowning the SAME run. State here carries targets that would crown run A; config
    carries targets that crown run B — config must win."""
    from types import SimpleNamespace

    from app.services.report import _pick_trophy_run

    run_a = {  # perfect on "Metric X" only
        "_file": "a.json", "timestamp": "2026-01-01T00:00:00Z",
        "model_runs": [{"total": 10, "passed": 10, "per_metric": {"Metric X": 1.0, "Metric Y": 0.5}}],
    }
    run_b = {  # perfect on "Metric Y" only
        "_file": "b.json", "timestamp": "2026-01-02T00:00:00Z",
        "model_runs": [{"total": 10, "passed": 9, "per_metric": {"Metric X": 0.5, "Metric Y": 1.0}}],
    }
    runner = SimpleNamespace(list_runs=lambda d: [run_a, run_b])
    state = SimpleNamespace(target_metric_scores={"Metric X": 1.0}, iterations=[])
    monkeypatch.setattr(
        "app.services.harness_config.load_harness_config",
        lambda d: SimpleNamespace(target_metric_scores={"Metric Y": 1.0}),
    )

    winner = _pick_trophy_run(tmp_path, runner, state)

    assert winner == "b.json", (
        "config targets (Metric Y) must decide the trophy; the state snapshot "
        "(Metric X) would have crowned a.json"
    )


@pytest.mark.parametrize(
    "source",
    ["class EvalConfig(:\n", "models = [1, 2]\ntarget_metric_scores = {'A': 0.9}\n"],
    ids=["syntax-error", "models-not-strings"],
)
def test_an_unreadable_config_means_no_targets(tmp_path, source):
    """Run listings and the executive report resolve targets; a broken config.py must not fail them."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "config.py").write_text(source)
    assert resolve_trophy_targets(tmp_path) == {}
