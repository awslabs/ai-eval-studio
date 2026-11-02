# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the exec-narration serializer in ``report_generation.py``.

The lock + coalescer + snapshot guarantees:

1. Two overlapping ``generate_exec_report`` calls don't both run the
   narrator — the second flips a "rerun pending" flag and bails.
2. Whatever facts the in-flight narration was reasoning about is what
   ``save_report`` writes — even if a winner change lands mid-narration.
   No prose-A/stamp-B sidecar.
3. After the in-flight cycle finishes, exactly one rerun fires and
   rebuilds facts → latest winner.

We patch ``_run_narrator`` so we can drive the timing precisely without
spinning up a real Claude SDK session.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest


@pytest.fixture
def project_dir(tmp_path: Path, monkeypatch) -> Path:
    """Project on disk with a single baseline run + a stub project_manager."""
    pd = tmp_path / "proj"
    (pd / "output").mkdir(parents=True)
    (pd / "prompts").mkdir()
    (pd / "prompts" / "baseline.md").write_text("baseline prompt")
    (pd / "config.json").write_text(json.dumps({
        "id": "test-proj-id",
        "name": "Serial Test",
        "description": "",
        "created_at": "2026-06-01T00:00:00Z",
        "agent_model": "opus",
    }))
    _write_run(pd, "baseline", "eval_results_baseline_v1.json", success=True)
    return pd


def _write_run(pd: Path, version: str, filename: str, *, success: bool) -> None:
    payload = [{
        "source": "manual",
        "model_id": "us.anthropic.claude-sonnet-4-6",
        "test_run_id": "sonnet",
        "timestamp": "2026-06-05T10:00:00Z",
        "hyperparameters": {
            "model": "us.anthropic.claude-sonnet-4-6",
            "endpoint": "runtime",
            "region": "us-east-1",
            "prompt_version": version,
            "temperature": 0.0,
            "max_tokens": 2048,
        },
        "results": [{
            "name": "tc0",
            "success": success,
            "input": "hi",
            "actual_output": "yo",
            "expected_output": "yo",
            "context": None,
            "retrieval_context": None,
            "additional_metadata": {
                "id": 0,
                "latency_ms": 100,
                "generation_cost": 0.001,
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 0,
                "cache_write_input_tokens": 0,
            },
            "metrics_data": [{
                "name": "Acc",
                "score": 1.0 if success else 0.0,
                "threshold": 0.5,
                "success": success,
                "reason": "r",
                "error": None,
            }],
        }],
        "generation_cost_usd": 0.001,
        "judge_cost_usd": 0.0005,
        "data_quality": {
            "reliability": "ok",
            "errored_count": 0,
            "dropped_count": 0,
            "null_count": 0,
            "unrecovered_count": 0,
            "total_cases": 1,
        },
    }]
    (pd / "output" / filename).write_text(json.dumps(payload))


@pytest.fixture(autouse=True)
def reset_narration_state():
    """Clear module-private state so each test starts clean."""
    from app.services.report_generation import _reset_exec_narration_state
    _reset_exec_narration_state()
    yield
    _reset_exec_narration_state()


class TestExecNarrationSerializer:
    async def test_overlapping_calls_coalesce_to_one_rerun(self, project_dir):
        """Two concurrent calls produce two cycles total: the in-flight
        one plus exactly one coalesced rerun (not two separate runs)."""
        from app.services import report_generation as rg

        cycle_count = 0
        gate = asyncio.Event()

        async def fake_narrator(_pd, _project_id, _task, _owner):
            nonlocal cycle_count
            cycle_count += 1
            # Hold the first call open until we've fired the second one.
            if cycle_count == 1:
                await gate.wait()
            return True

        with patch.object(rg, "_run_narrator", side_effect=fake_narrator):
            # Kick off the first call (will block at gate.wait()).
            t1 = asyncio.create_task(rg.generate_exec_report(
                project_dir, "test-proj-id",
            ))
            # Yield so t1 acquires the lock and enters the narrator stub.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            # While t1 is mid-narration, fire two more calls. They should
            # both flip "rerun pending" (idempotent) and return False
            # immediately without touching the narrator.
            r2 = await rg.generate_exec_report(project_dir, "test-proj-id")
            r3 = await rg.generate_exec_report(project_dir, "test-proj-id")
            assert r2 is False
            assert r3 is False
            # Now release the first call. The lock-holder should see the
            # pending flag and run exactly ONE more cycle.
            gate.set()
            await t1
        assert cycle_count == 2, (
            f"expected 2 narration cycles (initial + 1 coalesced rerun), "
            f"got {cycle_count}"
        )

    async def test_save_report_uses_in_flight_facts_snapshot(self, project_dir):
        """The sidecar's stamped facts must come from the same build the
        narrator was reasoning about — not a fresh rebuild that could
        capture a winner change that landed mid-narration. Pin the
        contract: prose-and-stamp come from one snapshot, never two."""
        from app.services import report_generation as rg
        from app.services.mcp_tools.reports import save_report as save_report_tool

        save_report = save_report_tool.handler

        async def fake_narrator(pd, project_id, task, owner):
            # Simulate a winner change landing during narration: a NEW
            # run of a NEW version (v2) is dropped on disk while the
            # narrator is mid-thinking. The stash must still hold the
            # baseline (v1) facts so save_report stamps v1, matching the
            # prose the agent wrote.
            _write_run(pd, "v2", "eval_results_v2_late.json", success=True)
            (pd / "prompts" / "v2.md").write_text("better")
            # Simulate the agent calling save_report with its narrative.
            result = await save_report({
                "project_id": project_id,
                "report_type": "executive",
                "narrative": {"intro": "About baseline (v1)."},
            })
            assert not result.get("is_error"), result
            return True

        with patch.object(
            rg, "_run_narrator", side_effect=fake_narrator,
        ), patch(
            "app.services.mcp_tools.reports.resolve_project_dir",
            return_value=project_dir,
        ):
            # Single call, no overlap — exercises the snapshot path
            # purely. The mid-narration winner flip would, without the
            # snapshot, cause save_report to rebuild facts and stamp v2
            # while the prose was about v1.
            await rg.generate_exec_report(project_dir, "test-proj-id")

        sidecar = json.loads(
            (project_dir / "output" / "reports" / "executive.report.json")
            .read_text()
        )
        # Stamped facts and prose both refer to baseline (v1). Without
        # the snapshot, ``winner_version`` would be "v2" while the
        # narrative talks about baseline.
        assert sidecar["facts"]["winner_version"] == "baseline"
        assert sidecar["narrative"] == {"intro": "About baseline (v1)."}

    async def test_lock_acquisition_failure_is_swallowed(self, project_dir):
        """Best-effort contract: any failure short-circuits to False —
        never propagates and never blocks the eval-completion path."""
        from app.services import report_generation as rg

        async def boom(_pid):
            raise RuntimeError("registry exploded")

        with patch.object(rg, "_get_exec_narration_lock", side_effect=boom):
            result = await rg.generate_exec_report(project_dir, "x")
        assert result is False

    async def test_pending_flag_cleared_at_cycle_start(self, project_dir):
        """A pending flag set BEFORE the cycle starts (e.g. left over
        from a prior run that never ran its rerun) must not cause an
        extra cycle on the next call. Reset semantics, not accumulating."""
        from app.services import report_generation as rg

        cycle_count = 0

        async def fake_narrator(*_args, **_kw):
            nonlocal cycle_count
            cycle_count += 1
            return True

        # Pre-set the flag — simulate stale state.
        rg._exec_narration_pending["test-proj-id"] = True

        with patch.object(rg, "_run_narrator", side_effect=fake_narrator):
            await rg.generate_exec_report(project_dir, "test-proj-id")

        assert cycle_count == 1
        assert "test-proj-id" not in rg._exec_narration_pending


class TestNarrationPublishesEvent:
    """A successful narration emits a ``report/updated`` event so the
    open Reports tab refetches without polling. A failed narration emits
    nothing — no false-positive refresh of unchanged prose."""

    async def test_run_narration_publishes_event_with_run_filename(self, project_dir):
        from app.services import report_generation as rg

        published: list[tuple[str, dict]] = []

        async def fake_narrator(*_args, **_kw):
            return True

        def fake_publish(project_id, event):
            published.append((project_id, event))

        with patch.object(rg, "_run_narrator", side_effect=fake_narrator), \
             patch("app.services.jobs._publish_project_event", side_effect=fake_publish):
            await rg.generate_run_report(
                project_dir, "test-proj-id",
                "eval_results_baseline_v1.json",
            )

        assert published == [(
            "test-proj-id",
            {
                "resource": "report",
                "type": "updated",
                "scope": "run",
                "run_filename": "eval_results_baseline_v1.json",
            },
        )]

    async def test_exec_narration_publishes_event_with_scope(self, project_dir):
        from app.services import report_generation as rg

        published: list[tuple[str, dict]] = []

        async def fake_narrator(*_args, **_kw):
            return True

        def fake_publish(project_id, event):
            published.append((project_id, event))

        with patch.object(rg, "_run_narrator", side_effect=fake_narrator), \
             patch("app.services.jobs._publish_project_event", side_effect=fake_publish):
            await rg.generate_exec_report(project_dir, "test-proj-id")

        assert published == [(
            "test-proj-id",
            {"resource": "report", "type": "updated", "scope": "executive"},
        )]

    async def test_failed_narration_publishes_failed(self, project_dir):
        # When the narrator returns False (creds, network blip, agent
        # crash), the sidecar didn't change — an "updated" event would
        # falsely refresh the UI to the SAME content. "failed" ends the
        # tab's pending regenerate instead, which would otherwise spin
        # forever now that the endpoint no longer waits.
        from app.services import report_generation as rg

        published: list[tuple[str, dict]] = []

        async def fake_narrator(*_args, **_kw):
            return False

        def fake_publish(project_id, event):
            published.append((project_id, event))

        with patch.object(rg, "_run_narrator", side_effect=fake_narrator), \
             patch("app.services.jobs._publish_project_event", side_effect=fake_publish):
            await rg.generate_run_report(
                project_dir, "test-proj-id",
                "eval_results_baseline_v1.json",
            )
            await rg.generate_exec_report(project_dir, "test-proj-id")

        assert published == [
            ("test-proj-id", {
                "resource": "report",
                "type": "failed",
                "scope": "run",
                "run_filename": "eval_results_baseline_v1.json",
            }),
            ("test-proj-id", {
                "resource": "report", "type": "failed", "scope": "executive",
            }),
        ]

    async def test_facts_build_failure_publishes_failed(self, project_dir):
        # The narrator never runs when facts cannot be built, and that
        # exit must end a pending regenerate too.
        from app.services import report_generation as rg

        published: list[dict] = []

        def fake_publish(_project_id, event):
            published.append(event)

        with patch.object(rg, "build_run_facts", side_effect=ValueError("bad run")), \
             patch.object(rg, "build_exec_facts", side_effect=ValueError("bad facts")), \
             patch("app.services.jobs._publish_project_event", side_effect=fake_publish):
            await rg.generate_run_report(project_dir, "test-proj-id", "missing.json")
            await rg.generate_exec_report(project_dir, "test-proj-id")

        assert [(e["type"], e["scope"]) for e in published] == [
            ("failed", "run"), ("failed", "executive"),
        ]

    async def test_coalesced_exec_call_publishes_only_the_cycles_outcomes(
        self, project_dir,
    ):
        # A call that joins an in-flight narration publishes nothing of its
        # own; the in-flight cycle and the one rerun it owes each publish.
        from app.services import report_generation as rg

        published: list[dict] = []
        release = asyncio.Event()

        async def fake_narrator(*_args, **_kw):
            await release.wait()
            return True

        def fake_publish(_project_id, event):
            published.append(event)

        with patch.object(rg, "_run_narrator", side_effect=fake_narrator), \
             patch("app.services.jobs._publish_project_event", side_effect=fake_publish):
            first = asyncio.create_task(
                rg.generate_exec_report(project_dir, "test-proj-id"),
            )
            await asyncio.sleep(0)
            await rg.generate_exec_report(project_dir, "test-proj-id")
            assert published == []
            release.set()
            await first

        assert published == [
            {"resource": "report", "type": "updated", "scope": "executive"},
            {"resource": "report", "type": "updated", "scope": "executive"},
        ]
