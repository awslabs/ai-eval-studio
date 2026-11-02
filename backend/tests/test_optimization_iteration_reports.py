# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-iteration report dispatch hook.

Optimization iterations call ``run_eval_via_harness`` directly (via
``dispatch_eval_job``) instead of going through the standalone-eval
executor, so they bypass the executor's ``_dispatch_run_report`` /
``_dispatch_exec_if_winner_changed`` call. ``_dispatch_iteration_reports``
adds those calls back from the optimization side; without them, every
per-run report and the live exec narrative would stay barebones until
convergence.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from .async_wait import until


class TestDispatchIterationReports:
    async def test_dispatches_both_run_and_exec_narrations(self, tmp_path):
        from app.services.jobs.optimization_executor import (
            _dispatch_iteration_reports,
        )
        from app.services.jobs import eval_executor

        run_calls: list[tuple] = []
        exec_calls: list[tuple] = []

        async def fake_run(pd, pid, fname, owner):
            run_calls.append((str(pd), pid, fname, owner))

        async def fake_exec(pd, pid, owner):
            exec_calls.append((str(pd), pid, owner))

        with patch.object(eval_executor, "_dispatch_run_report", side_effect=fake_run), \
             patch.object(eval_executor, "_dispatch_exec_if_winner_changed", side_effect=fake_exec):
            await _dispatch_iteration_reports(
                tmp_path, "proj-id",
                "/abs/path/output/eval_results_v3_20260616.json",
                "owner-id",
            )
            # The helper schedules both dispatches fire-and-forget, so wait for
            # the calls themselves to land rather than for a fixed number of loop
            # yields — two yields only happens to be enough when the tasks start
            # immediately.
            await until(
                lambda: run_calls and exec_calls,
                message="the fire-and-forget report dispatches never ran",
            )

        # Run dispatch gets the basename, not the absolute path.
        assert run_calls == [
            (str(tmp_path), "proj-id",
             "eval_results_v3_20260616.json", "owner-id"),
        ]
        assert exec_calls == [(str(tmp_path), "proj-id", "owner-id")]

    async def test_dispatches_the_metric_audit_too(self, tmp_path):
        """The audit must fire from the OPTIMIZATION path, not just standalone.

        Optimization iterations bypass ``run_eval_job``, so wiring the audit only
        into the executor would leave every optimization run unaudited — and that
        is the half where a mis-scoring metric feeds each rewrite decision rather
        than merely misleading a reader.
        """
        from app.services.jobs.optimization_executor import (
            _dispatch_iteration_reports,
        )
        from app.services.jobs import eval_executor

        audit_calls: list[tuple] = []

        async def fake_audit(pd, pid, fname):
            audit_calls.append((str(pd), pid, fname))

        async def noop(*_a, **_k):
            return None

        with patch.object(eval_executor, "_dispatch_run_report", side_effect=noop), \
             patch.object(eval_executor, "_dispatch_exec_if_winner_changed", side_effect=noop), \
             patch.object(eval_executor, "_dispatch_metric_audit", side_effect=fake_audit):
            await _dispatch_iteration_reports(
                tmp_path, "proj-id",
                "/abs/path/output/eval_results_v3_20260616.json",
                "owner-id",
            )
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        assert audit_calls == [
            (str(tmp_path), "proj-id", "eval_results_v3_20260616.json"),
        ], "the metric audit was not dispatched from the optimization path"

    async def test_empty_run_filename_dispatches_nothing(self, tmp_path):
        # Resume / no-output edge: the iteration completed but the run
        # filename slot is empty (e.g. an iteration that short-circuited
        # before writing). Don't fire a narrator on a missing run file.
        from app.services.jobs.optimization_executor import (
            _dispatch_iteration_reports,
        )
        from app.services.jobs import eval_executor

        run_calls = 0
        exec_calls = 0

        async def fake_run(*_a, **_kw):
            nonlocal run_calls
            run_calls += 1

        async def fake_exec(*_a, **_kw):
            nonlocal exec_calls
            exec_calls += 1

        with patch.object(eval_executor, "_dispatch_run_report", side_effect=fake_run), \
             patch.object(eval_executor, "_dispatch_exec_if_winner_changed", side_effect=fake_exec):
            await _dispatch_iteration_reports(tmp_path, "proj-id", "", "owner-id")
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        assert run_calls == 0
        assert exec_calls == 0

    async def test_run_dispatch_swallows_dispatcher_failure(self, tmp_path):
        # Best-effort contract: even if a dispatcher raises, the helper must not
        # propagate — an exception here would mark the iteration job failed for
        # purely cosmetic reasons.
        #
        # The metric audit is the one that is AWAITED rather than fired, so it is
        # the only dispatcher whose failure is on this helper's own code path.
        # That is exactly why it is included here: while it was a create_task a
        # raising audit could only log, and awaiting it moved the guarantee from
        # the callee into this function.
        from app.services.jobs.optimization_executor import (
            _dispatch_iteration_reports,
        )
        from app.services.jobs import eval_executor

        async def boom(*_a, **_kw):
            raise RuntimeError("dispatcher exploded")

        with patch.object(eval_executor, "_dispatch_run_report", side_effect=boom), \
             patch.object(eval_executor, "_dispatch_exec_if_winner_changed", side_effect=boom), \
             patch.object(eval_executor, "_dispatch_metric_audit", side_effect=boom):
            await _dispatch_iteration_reports(
                tmp_path, "proj-id", "eval.json", "owner-id",
            )
            # Drain the loop so the fire-and-forget narration tasks complete.
            # They log their exceptions rather than propagating.
            await asyncio.sleep(0)
            await asyncio.sleep(0)


class TestExecDispatchWinnerGating:
    """Pins the safety property the iteration hook relies on: the exec
    dispatcher is winner-gated, so calling it on every iteration is
    cheap when the trophy doesn't move."""

    async def test_no_call_to_generate_exec_when_winner_unchanged(self, tmp_path):
        # Build a project with one run, write the exec sidecar stamped
        # with that same run, and ensure ``_dispatch_exec_if_winner_changed``
        # does NOT fire ``generate_exec_report``.
        from app.services.jobs.eval_executor import _dispatch_exec_if_winner_changed
        from app.services.jobs import eval_executor

        # Scaffold project on disk.
        import json
        (tmp_path / "output").mkdir()
        (tmp_path / "prompts").mkdir()
        (tmp_path / "prompts" / "v1.md").write_text("v1")
        (tmp_path / "config.json").write_text(json.dumps({
            "id": "p", "name": "P", "description": "",
            "created_at": "2026-06-01T00:00:00Z", "agent_model": "opus",
        }))
        run_payload = [{
            "source": "optimization",
            "model_id": "m", "test_run_id": "m",
            "timestamp": "2026-06-05T10:00:00Z",
            "hyperparameters": {"model": "m", "endpoint": "runtime",
                                "region": "us-east-1", "prompt_version": "v1",
                                "temperature": 0.0, "max_tokens": 2048},
            "results": [{"name": "tc0", "success": True,
                         "input": "i", "actual_output": "a",
                         "expected_output": "a", "context": None,
                         "retrieval_context": None,
                         "additional_metadata": {
                             "id": 0, "latency_ms": 10, "generation_cost": 0,
                             "input_tokens": 1, "output_tokens": 1,
                             "cache_read_input_tokens": 0,
                             "cache_write_input_tokens": 0,
                         },
                         "metrics_data": [{"name": "Acc", "score": 1.0,
                                           "threshold": 0.5, "success": True,
                                           "reason": "r", "error": None}]}],
            "generation_cost_usd": 0, "judge_cost_usd": 0,
            "data_quality": {"reliability": "ok", "errored_count": 0,
                              "dropped_count": 0, "null_count": 0,
                              "unrecovered_count": 0, "total_cases": 1},
        }]
        (tmp_path / "output" / "eval_results_v1_x.json").write_text(json.dumps(run_payload))

        # Optimization state with targets so the trophy ranker is in play.
        sd = tmp_path / "output" / "optimization"
        sd.mkdir(parents=True)
        (sd / "state.json").write_text(json.dumps({
            "target_metric_scores": {"Acc": 0.5},
            "max_iterations": 10, "plateau_window": 3, "status": "running",
            "iterations": [{
                "iteration": 1, "prompt_version": "v1",
                "run_id": "eval_results_v1_x.json",
                "metrics_summary": {"Acc": 1.0},
                "total_cases": 1, "passed": 1, "pass_rate": 1.0,
                "timestamp": "2026-06-05T10:00:00Z",
                "per_model_pass_rates": {"m": 1.0},
                "per_model_metrics": {"m": {"Acc": 1.0}},
            }],
        }))

        # Pre-write an exec sidecar stamped with the same winning file.
        from app.services.report import build_exec_facts, write_report_json
        facts = build_exec_facts(tmp_path)
        write_report_json(tmp_path, "executive", facts,
                          narrative={"intro": "current"})

        gen_called = False

        async def fake_gen(*_a, **_kw):
            nonlocal gen_called
            gen_called = True
            return True

        with patch("app.services.report_generation.generate_exec_report",
                   side_effect=fake_gen):
            await _dispatch_exec_if_winner_changed(
                tmp_path, "p", "owner-id",
            )

        assert gen_called is False, (
            "winner-gated dispatch should be a no-op when the trophy run "
            "matches the sidecar's stamped winner"
        )
        del eval_executor  # keep import explicit for readers
