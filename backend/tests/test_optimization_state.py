# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for OptimizationState tracking iterative eval-optimize loops."""

import pytest
import json
from pathlib import Path
from app.services.optimization_state import OptimizationState


class TestOptimizationState:
    """Test suite for OptimizationState class."""

    def test_create_new_state(self, tmp_path):
        """Create OptimizationState with targets, verify defaults."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        target_scores = {"ExactMatch": 0.9, "Precision": 0.85}
        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores=target_scores,
            max_iterations=15,
            plateau_window=4
        )

        assert state.project_dir == project_dir
        assert state.target_metric_scores == target_scores
        assert state.max_iterations == 15
        assert state.plateau_window == 4
        assert state.iterations == []
        assert state.status == "running"
        assert state.current_iteration == 0

    def test_record_iteration(self, tmp_path):
        """Record one iteration, verify state updated."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.9}
        )

        metrics_summary = {"ExactMatch": 0.75, "Precision": 0.80}
        state.record_iteration(
            prompt_version="v1",
            run_id="run-abc-123",
            metrics_summary=metrics_summary,
            total_cases=10,
            passed=7
        )

        assert state.current_iteration == 1
        assert len(state.iterations) == 1

        iteration = state.iterations[0]
        assert iteration["iteration"] == 1
        assert iteration["prompt_version"] == "v1"
        assert iteration["run_id"] == "run-abc-123"
        assert iteration["metrics_summary"] == metrics_summary
        assert iteration["total_cases"] == 10
        assert iteration["passed"] == 7
        assert iteration["pass_rate"] == 0.7
        assert "timestamp" in iteration

    def test_convergence_target_reached(self, tmp_path):
        """Metrics meet targets, returns 'target_reached'."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.9, "Precision": 0.85}
        )

        # Record iteration that meets targets — with per-model data, the
        # writer contract every producer satisfies now (the aggregate
        # convergence fallback is deleted; see test_prompt_dedup_guards for
        # the withheld-verdict pin on rows without it).
        state.record_iteration(
            prompt_version="v1",
            run_id="run-1",
            metrics_summary={"ExactMatch": 0.92, "Precision": 0.87},
            total_cases=10,
            passed=9,
            per_model_pass_rates={"m1": 0.9},
            per_model_metrics={"m1": {"ExactMatch": 0.92, "Precision": 0.87}},
        )

        convergence, _version = state.check_convergence()
        assert convergence == "target_reached"

    def test_convergence_max_iterations(self, tmp_path):
        """Hit max iterations, returns 'max_iterations'."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.9},
            max_iterations=3
        )

        # Record 3 iterations that don't meet targets
        for i in range(3):
            state.record_iteration(
                prompt_version=f"v{i+1}",
                run_id=f"run-{i+1}",
                metrics_summary={"ExactMatch": 0.7 + i * 0.05},
                total_cases=10,
                passed=7 + i
            )

        convergence, _version = state.check_convergence()
        assert convergence == "max_iterations"

    def test_plateau_detection(self, tmp_path):
        """3 iterations within 2% spread, returns 'plateau'."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.95},  # High target not met
            max_iterations=10,
            plateau_window=3
        )

        # Record 3 iterations within 2% spread — per-model series, the
        # evidence plateau demands (aggregate-only windows withhold).
        for v, (rate, passed) in enumerate(
            [(0.75, 75), (0.76, 76), (0.755, 75)], start=1
        ):
            state.record_iteration(
                prompt_version=f"v{v}",
                run_id=f"run-{v}",
                metrics_summary={"ExactMatch": rate},
                total_cases=100,
                passed=passed,
                per_model_pass_rates={"m1": rate},
                per_model_metrics={"m1": {"ExactMatch": rate}},
            )

        convergence, _version = state.check_convergence()
        assert convergence == "plateau"

    def test_convergence_over_decimal_sourced_state_no_typeerror(self, tmp_path):
        """
        _from_dict must normalize Decimal→float so convergence arithmetic
        (max(series) - min(series)) does not raise TypeError when mixing
        Decimal with float fallback values.

        This test simulates the DDB read path: get_optimization_state →
        _from_dynamo yields Decimal for fractional numbers → _from_dict
        must normalize to float before __init__ so convergence math is
        pure float arithmetic.
        """
        from decimal import Decimal

        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Build a data dict as get_optimization_state/_from_dynamo would surface it,
        # with a plateau window whose per_model_pass_rates mix Decimal in one iteration
        # and omit the model in another (so _aggregate fallback returns float).
        data = {
            "target_metric_scores": {"Correctness": Decimal("0.95")},
            "max_iterations": 10,
            "plateau_window": 2,
            "status": "running",
            "iterations": [
                {
                    "prompt_version": "v1",
                    "run_id": "run-1",
                    "metrics_summary": {"Correctness": Decimal("0.85")},
                    "per_model_pass_rates": {"model-a": Decimal("0.30")},
                    "total_cases": 100,
                    "passed": 85,
                },
                {
                    "prompt_version": "v2",
                    "run_id": "run-2",
                    "metrics_summary": {"Correctness": Decimal("0.86")},
                    # per_model_pass_rates omits "model-a" so _aggregate returns float
                    "per_model_pass_rates": {},
                    "total_cases": 100,
                    "passed": 86,
                },
            ],
        }

        # Uncoerced, series = [Decimal("0.30"), <float from _aggregate>] and
        # max(series) - min(series) is Decimal - float → TypeError. _from_dict
        # coerces Decimal→float, so max - min is float arithmetic.
        state = OptimizationState._from_dict(project_dir, data)

        # check_convergence must return a tuple without raising TypeError.
        # The return value is (None, None) when still running, or (reason, version) on convergence.
        # We're just verifying no TypeError is raised during the convergence arithmetic.
        result = state.check_convergence()
        assert isinstance(result, tuple), "check_convergence should return (reason, version) tuple"
        reason, version = result
        # None means still running (not converged yet), which is expected for this test data
        assert reason in {None, "plateau", "max_iterations", "target_reached"}, f"Got unexpected reason: {reason}"

        # Also assert that numeric fields are native float, not Decimal
        assert type(state.target_metric_scores["Correctness"]) is float, (
            "target_metric_scores should be float after _from_dict normalization"
        )
        assert state.target_metric_scores["Correctness"] == 0.95

        # Verify nested per_model_pass_rates values are also float
        first_iter = state.iterations[0]
        if "model-a" in first_iter.get("per_model_pass_rates", {}):
            model_a_rate = first_iter["per_model_pass_rates"]["model-a"]
            assert type(model_a_rate) is float, (
                "per_model_pass_rates values should be float after normalization"
            )
            assert model_a_rate == 0.30

    def test_save_and_load(self, tmp_path):
        """Save state, load from disk, verify round-trip."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create and populate state
        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.9, "Precision": 0.85},
            max_iterations=15,
            plateau_window=4
        )

        state.record_iteration(
            prompt_version="v1",
            run_id="run-1",
            metrics_summary={"ExactMatch": 0.75, "Precision": 0.80},
            total_cases=10,
            passed=7
        )
        state.record_iteration(
            prompt_version="v2",
            run_id="run-2",
            metrics_summary={"ExactMatch": 0.82, "Precision": 0.83},
            total_cases=10,
            passed=8
        )

        # Save
        state.save()

        # Load from disk
        loaded_state = OptimizationState.load(project_dir)

        assert loaded_state is not None
        assert loaded_state.project_dir == project_dir
        assert loaded_state.target_metric_scores == {"ExactMatch": 0.9, "Precision": 0.85}
        assert loaded_state.max_iterations == 15
        assert loaded_state.plateau_window == 4
        assert loaded_state.status == "running"
        assert len(loaded_state.iterations) == 2
        assert loaded_state.current_iteration == 2
        assert loaded_state.iterations[0]["prompt_version"] == "v1"
        assert loaded_state.iterations[1]["prompt_version"] == "v2"

    def test_load_missing_returns_none(self, tmp_path):
        """Load from empty dir returns None."""
        project_dir = tmp_path / "empty_project"
        project_dir.mkdir()

        loaded_state = OptimizationState.load(project_dir)
        assert loaded_state is None

    def test_best_iteration(self, tmp_path):
        """Returns iteration with highest pass rate."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.95}
        )

        # Per-model rates are what every current writer records, and what
        # best_iteration ranks on — "highest single-model pass rate", not the
        # cross-model aggregate.
        for version, run, rate, passed in (
            ("v1", "run-1", 0.75, 7),
            ("v2", "run-2", 0.90, 9),
            ("v3", "run-3", 0.80, 8),
        ):
            state.record_iteration(
                prompt_version=version,
                run_id=run,
                metrics_summary={"ExactMatch": rate},
                total_cases=10,
                passed=passed,
                per_model_pass_rates={"model-a": rate},
            )

        best = state.best_iteration()
        assert best is not None
        assert best["prompt_version"] == "v2"
        assert best["pass_rate"] == 0.9
        assert best["run_id"] == "run-2"

    def test_best_iteration_is_none_without_per_model_data(self, tmp_path):
        """Rows that cannot be ranked must not produce an arbitrary winner.

        ``best_iteration`` used to substitute a row's aggregate rate when it had
        no per-model data, as display-only tolerance for rows predating per-model
        metrics. That compared one model's best score against a cross-model
        average — different quantities — so such a row could win a comparison it
        should have lost. Now they rank last, and an all-unrankable history
        returns None rather than whichever row ``max`` happened to see first.
        """
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.95},
        )
        state.record_iteration(
            prompt_version="v1", run_id="run-1",
            metrics_summary={"ExactMatch": 0.75}, total_cases=10, passed=7,
        )

        assert state.best_iteration() is None

    def test_best_iteration_empty_state(self, tmp_path):
        """best_iteration returns None when no iterations recorded."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.9}
        )

        best = state.best_iteration()
        assert best is None

    def test_no_convergence_initially(self, tmp_path):
        """check_convergence returns None when conditions not met."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.95},
            max_iterations=10,
            plateau_window=3
        )

        # Record just 1 iteration
        state.record_iteration(
            prompt_version="v1",
            run_id="run-1",
            metrics_summary={"ExactMatch": 0.75},
            total_cases=10,
            passed=7
        )

        convergence, _version = state.check_convergence()
        assert convergence is None

    def test_plateau_detection_insufficient_iterations(self, tmp_path):
        """Plateau not detected if fewer iterations than plateau_window."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.95},
            max_iterations=10,
            plateau_window=3
        )

        # Record only 2 iterations
        state.record_iteration(
            prompt_version="v1",
            run_id="run-1",
            metrics_summary={"ExactMatch": 0.75},
            total_cases=10,
            passed=7
        )
        state.record_iteration(
            prompt_version="v2",
            run_id="run-2",
            metrics_summary={"ExactMatch": 0.76},
            total_cases=10,
            passed=7
        )

        convergence, _version = state.check_convergence()
        assert convergence is None

    def test_record_iteration_persists_to_disk(self, tmp_path):
        """record_iteration calls save(), so state file exists after the call."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.9}
        )

        # Delete the state file to confirm record_iteration recreates it
        from app.services.optimization_state import OPTIMIZATION_STATE_FILE
        state_file = project_dir / OPTIMIZATION_STATE_FILE
        assert state_file.exists()  # create() saves it
        state_file.unlink()
        assert not state_file.exists()

        # record_iteration should persist
        state.record_iteration(
            prompt_version="v1",
            run_id="run-persist",
            metrics_summary={"ExactMatch": 0.75},
            total_cases=10,
            passed=7,
        )

        assert state_file.exists()

        # Verify the persisted data includes the iteration
        loaded = OptimizationState.load(project_dir)
        assert loaded is not None
        assert len(loaded.iterations) == 1
        assert loaded.iterations[0]["prompt_version"] == "v1"
        assert loaded.iterations[0]["run_id"] == "run-persist"

    def test_plateau_detection_above_threshold(self, tmp_path):
        """Plateau not detected if improvement exceeds 2% threshold."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.95},
            max_iterations=10,
            plateau_window=3
        )

        # Record 3 iterations with pass rates spread > 2% (0.70, 0.75, 0.80)
        state.record_iteration(
            prompt_version="v1",
            run_id="run-1",
            metrics_summary={"ExactMatch": 0.70},
            total_cases=10,
            passed=7
        )
        state.record_iteration(
            prompt_version="v2",
            run_id="run-2",
            metrics_summary={"ExactMatch": 0.75},
            total_cases=10,
            passed=7
        )
        state.record_iteration(
            prompt_version="v3",
            run_id="run-3",
            metrics_summary={"ExactMatch": 0.80},
            total_cases=10,
            passed=8
        )

        convergence, _version = state.check_convergence()
        assert convergence is None


class TestConcurrentUpdate:
    """Proof: OptimizationState no-lost-update under simulated concurrency.

    Tests the shipped conditional-write mechanism (put_optimization_state with
    expected_version) prevents two writers with the same stale version from both
    succeeding — one is rejected, no lost update.

    Test-only; the mechanism already exists.
    """

    @pytest.mark.asyncio
    async def test_concurrent_updates(self):
        """Stale expected_version is rejected — no lost update.

        Simulate two writers racing on the same optimization state:
        1. Seed an optimization state at version N.
        2. Reader A and Reader B both read version N.
        3. Writer A writes with expected_version=N → succeeds, returns N+1.
        4. Writer B writes with expected_version=N (same stale N) → raises
           ConcurrentUpdateError (the conditional-write mismatch).
        5. Assert the stored version is N+1 (writer A's), not N+2, and
           writer A's payload persisted.

        This proves both cannot succeed with the same expected_version; the
        second is rejected → no lost update.
        """
        from moto import mock_aws
        import boto3
        from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore
        from app.platform.metadata_store import ConcurrentUpdateError

        with mock_aws():
            # Setup DynamoDB table
            dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
            table_name = "test-table"
            dynamodb_client.create_table(
                TableName=table_name,
                AttributeDefinitions=[
                    {"AttributeName": "PK", "AttributeType": "S"},
                    {"AttributeName": "SK", "AttributeType": "S"},
                ],
                KeySchema=[
                    {"AttributeName": "PK", "KeyType": "HASH"},
                    {"AttributeName": "SK", "KeyType": "RANGE"},
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

            user_id = "owner-1"
            project_id = "project-1"

            # 1. Seed an optimization state at version 1 (first write, no expected_version)
            initial_state = {
                "target_metric_scores": {"ExactMatch": 0.9},
                "max_iterations": 10,
                "plateau_window": 3,
                "plateau_threshold": 0.02,
                "iterations": [],
                "status": "running",
                "max_iter_cap": 10,
                "workload": None,
                "pending_resume_drift": None,
                "winning_model_id": None,
                "current_prompt_version": None,
                "error": None,
            }
            version_1 = await store.put_optimization_state(
                user_id, project_id, initial_state, expected_version=None
            )
            assert version_1 == 1, "First write should return version 1"

            # 2. Both Reader A and Reader B read version 1 (shared by both simulated writers)
            read_state = await store.get_optimization_state(user_id, project_id)
            assert read_state is not None
            assert read_state["version"] == 1

            # 3. Writer A: update with expected_version=1 → succeeds
            writer_a_state = {**initial_state, "status": "writer_a_modified"}
            version_2 = await store.put_optimization_state(
                user_id, project_id, writer_a_state, expected_version=1
            )
            assert version_2 == 2, "Writer A should succeed and return version 2"

            # 4. Writer B: update with expected_version=1 (same stale version) → raises
            writer_b_state = {**initial_state, "status": "writer_b_modified"}
            with pytest.raises(ConcurrentUpdateError):
                await store.put_optimization_state(
                    user_id, project_id, writer_b_state, expected_version=1
                )

            # 5. Assert the stored version is 2 (writer A's), not 3, and writer A's payload persisted
            final_state = await store.get_optimization_state(user_id, project_id)
            assert final_state is not None
            assert final_state["version"] == 2, "Final version should be 2 (only writer A succeeded)"
            assert final_state["status"] == "writer_a_modified", "Writer A's payload should persist, not writer B's"


class TestLegacyStateForwardCompat:
    """A state.json written by an earlier version with the now-removed
    ``use_prompt_architect`` field must still load cleanly. Pre-release
    we don't ship migration shims, but cloud already has live state
    files in flight; an unknown key on disk should not crash the
    loader."""

    def test_legacy_state_with_removed_key_loads(self, tmp_path):
        project_dir = tmp_path / "p"
        (project_dir / "output" / "optimization").mkdir(parents=True)
        legacy = {
            "target_metric_scores": {"ExactMatch": 0.9},
            "max_iterations": 10,
            "plateau_window": 3,
            "plateau_threshold": 0.02,
            "iterations": [],
            "status": "running",
            "max_iter_cap": 10,
            "workload": None,
            "pending_resume_drift": None,
            "winning_model_id": None,
            "current_prompt_version": None,
            "error": None,
            # The old key — silently ignored by ``_from_dict`` because
            # the dict-driven kwargs only feed declared params.
            "use_prompt_architect": True,
        }
        (project_dir / "output" / "optimization" / "state.json").write_text(
            json.dumps(legacy)
        )
        loaded = OptimizationState.load(project_dir)
        assert loaded is not None
        assert loaded.status == "running"
        assert not hasattr(loaded, "use_prompt_architect")


class TestPhaseField:
    """Loop phase on the durable record — the honest status-bar signal.

    Written only by the loop orchestrator ("evaluating" at iteration top,
    "optimizing_prompt" before the prompt-optimizer turn). Records that
    predate the field deserialize to None and the bar shows a generic
    label instead of inventing a phase."""

    _BASE = {
        "target_metric_scores": {"ExactMatch": 0.9},
        "max_iterations": 10,
        "plateau_window": 3,
        "iterations": [],
        "status": "running",
    }

    def test_phase_roundtrips_through_save_and_load(self, tmp_path):
        import json as _json

        from app.services.optimization_state import (
            OPTIMIZATION_STATE_FILE,
            OptimizationState,
        )

        instance = OptimizationState._from_dict(
            tmp_path, {**self._BASE, "phase": "optimizing_prompt"}
        )
        assert instance.phase == "optimizing_prompt"

        # Serialize through the real save path (old/EFS branch writes json).
        instance.save(arch_split=False)
        written = _json.loads((tmp_path / OPTIMIZATION_STATE_FILE).read_text())
        assert written["phase"] == "optimizing_prompt"

    def test_absent_phase_deserializes_to_none(self, tmp_path):
        """Pre-field records must load with phase=None, not KeyError."""
        from app.services.optimization_state import OptimizationState

        instance = OptimizationState._from_dict(tmp_path, dict(self._BASE))
        assert instance.phase is None
