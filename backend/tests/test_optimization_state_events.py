# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract test: every OptimizationState transition publishes a project event.

The SPA's ``OptimizationStateProvider`` subscribes to two envelope
shapes on the project bus to refetch state without waiting for the
``/activity`` backstop poll: ``{resource:"optimization", type:"terminal"}``
and ``{resource:"optimization", type:"running"}``. Every backend code path
that mutates ``OptimizationState.status`` must publish one of those
envelopes — otherwise the bar lags until the next poll after a click.

The publish was first wired one transition at a time — convergence first,
then the loop's cancel path separately. This test locks the contract
so a future contributor adding a new terminal status can't ship a
backend mutation without also wiring the publish.

Two layers of check:

1. **Static envelope-shape audit** (TestEnvelopeShapeContract). Greps
   every optimization-resource ``_publish_project_event`` call across
   the API + dispatch modules and asserts each has the keys the SPA
   filter needs and a type the SPA subscribes to. Catches ~90% of
   "forgot a key" or "wrong type string" regressions.
2. **Dynamic loop-cancel test** (TestLoopCancelledPublishesTerminal).
   Runs the actual optimization loop coroutine just long enough to
   enter its try-block, cancels it, and asserts the
   ``except asyncio.CancelledError`` branch published. The cancel
   path is the one most easily missed.

Tests for the REST routes (start/cancel/resume) are NOT here — the
routes have heavy guard logic (harness load, workload-drift check,
etc.) that's painful to mock honestly. The static check covers
"the publish call exists in the right shape"; integration coverage
for the routes lives in test_optimization_api.py.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch, MagicMock

import pytest


def _capture_published_events():
    """Return ``(captured_list, capture_fn)`` — caller substitutes
    ``capture_fn`` for ``_publish_project_event`` at the call site
    being exercised, then asserts on ``captured_list``."""
    captured: list[tuple[str, dict]] = []

    def _capture(project_id: str, event: dict) -> None:
        captured.append((project_id, event))

    return captured, _capture


class TestEnvelopeShapeContract:
    """Static audit of every optimization-resource publish in the codebase.

    Pins the contract so a future contributor can't accidentally publish
    a malformed envelope that the SPA's useProjectEvent filter would
    silently drop.
    """

    SOURCES = [
        "app/api/optimization.py",
        "app/api/eval_jobs.py",
        "app/services/mcp_tools/dispatch.py",
        "app/services/mcp_tools/eval.py",
        # The optimization loop driver moved here from dispatch.py; its
        # terminal/error publishes are the ones the SPA's bar depends on, so the
        # audit must follow them.
        "app/services/optimization_loop_orchestrator.py",
    ]
    # Types the SPA's OptimizationStateProvider + useBackendEvents
    # toast layer subscribe to. A publish using any other ``type``
    # value silently lands in no consumer's hands.
    ALLOWED_TYPES = {"terminal", "running", "error"}

    def _find_optimization_publishes(self):
        """Yield (relative_path, full_call_text) for every
        ``_publish_project_event`` call whose dict literal sets
        ``resource: "optimization"``."""
        import re
        from pathlib import Path

        backend_root = Path(__file__).resolve().parent.parent
        # Match _publish_project_event(<id>, {<dict>}) where the dict
        # contains ``"resource": "optimization"`` somewhere. The
        # outer dict capture stops at the first balanced ``}``.
        pattern = re.compile(
            r'_publish_project_event\([^,]+,\s*\{(?P<body>(?:[^{}]|\{[^{}]*\})*)\}',
            re.DOTALL,
        )

        for rel in self.SOURCES:
            full = backend_root / rel
            if not full.exists():
                continue
            text = full.read_text()
            for match in pattern.finditer(text):
                body = match.group("body")
                if '"resource": "optimization"' in body:
                    yield rel, match.group(0)

    def test_every_optimization_publish_has_required_keys(self):
        sites = list(self._find_optimization_publishes())
        assert sites, (
            "No optimization _publish_project_event calls found — "
            "either the test's regex broke, or the publish wiring "
            "was deleted."
        )
        for rel, block in sites:
            for required in ("resource", "type"):
                assert f'"{required}":' in block, (
                    f"{rel}: optimization publish missing required "
                    f"key '{required}'\n  in: {block[:300]}"
                )

    def test_every_optimization_publish_uses_subscribed_type(self):
        for rel, block in self._find_optimization_publishes():
            type_match = next(
                (t for t in self.ALLOWED_TYPES if f'"type": "{t}"' in block),
                None,
            )
            assert type_match, (
                f"{rel}: optimization publish has no type the SPA "
                f"subscribes to (expected one of {self.ALLOWED_TYPES}). "
                "If you're adding a new envelope shape, add it to the "
                "SPA's useProjectEvent subscriptions in "
                "OptimizationStateProvider.tsx FIRST.\n"
                f"  in: {block[:300]}"
            )


class TestLoopCancelledPublishesTerminal:
    """Loop's ``except asyncio.CancelledError`` branch publishes terminal.

    External cancel (via the cancel REST endpoint or kill_harness)
    raises CancelledError inside the loop. The branch logs and exits;
    without an explicit publish here, the SPA only notices the loop is
    gone on its next backstop poll. Regression-guard: this path was left
    out when the publish was first added for convergence and was wired
    separately afterwards.
    """

    @pytest.mark.asyncio
    async def test_cancellation_publishes_terminal(self, tmp_path, monkeypatch):
        from app.services import optimization_loop_orchestrator as loop_mod
        from app.services.optimization_state import OptimizationState

        captured, capture = _capture_published_events()
        # The loop body imports _publish_project_event at module
        # load time (not function-local), so patching the binding
        # in the loop orchestrator module is what catches it.
        monkeypatch.setattr(
            "app.services.optimization_loop_orchestrator._publish_project_event",
            capture,
        )

        # Seed a state file so OptimizationState.load() returns
        # something the loop can read; the loop's first action is to
        # check disk_state.status. We want it to enter the try-block
        # and reach the await point so we can cancel it.
        from app.services.optimization_state import OptimizationState as _OS
        state = _OS.create(
            project_dir=tmp_path,
            target_metric_scores={"ExactMatch": 0.85},
            max_iterations=1,
            plateau_window=3,
        )
        state.status = "running"
        state.save()

        # The loop driver lives in services/optimization_loop_orchestrator.
        from app.services.optimization_loop_orchestrator import run_optimization_loop

        async def _driver() -> None:
            await run_optimization_loop({
                "project_id": "p-1",
                "project_dir": str(tmp_path),
                "model_ids": ["m1"],
                "data_file": "",
                "field_mappings": {"input": "q", "expected_output": "a"},
                "judge_model_id": None,
                "target_metric_scores": {"ExactMatch": 0.85},
                "max_iterations": 1,
                "plateau_window": 3,
                "max_test_cases": None,
                "data_file_override": None,
                "prompt_version": "baseline",
            })

        # Mock the heavy iteration-creation path so the loop can
        # enter its try-block without trying to dispatch a real job.
        # The first await inside the loop is on JobTracker / the
        # iteration runner — we'll inject a CancelledError at that
        # point.
        with patch.object(loop_mod, "get_job_tracker") as mock_tracker, \
             patch("app.services.jobs.optimization_executor.run_optimization_iteration_job",
                   new=MagicMock(side_effect=asyncio.CancelledError())):
            mock_tracker.return_value.create_job = MagicMock(return_value=MagicMock(job_id="j-1"))
            try:
                await _driver()
            except asyncio.CancelledError:
                pass

        events = [e for (_, e) in captured if e.get("resource") == "optimization"]
        terminal_cancelled = [
            e for e in events
            if e.get("type") == "terminal" and e.get("reason") == "cancelled"
        ]
        assert terminal_cancelled, (
            "loop CancelledError branch must publish "
            "{resource:'optimization', type:'terminal', reason:'cancelled'}; "
            f"actual: {events}"
        )
