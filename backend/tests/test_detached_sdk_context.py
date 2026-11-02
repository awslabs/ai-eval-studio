# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that detached SDK sessions carry an authz context for their MCP tools.

Two places spawn their own Agent-SDK session from server-side code, outside any
HTTP request and outside ``run_agent_turn``: the report narrator
(``report_generation._run_narrator``) and the prompt optimizer
(``optimization_loop_orchestrator._run_prompt_optimizer``). Neither used to set
``request_context``, and ``mcp_authz.authorize_tool_call`` fails CLOSED — with no
context it returns ``mcp_error("not authenticated")`` and never invokes the
handler. So every MCP tool those sessions hold was refused: the narrator's ONLY
tool (``save_report``), and three of the optimizer's four (``analyze_failures``,
``get_eval_results``, ``save_prompt`` — only ``Read`` is an ungated built-in).

Why these live outside ``test_report_generation.py``: every test there patches
``_run_narrator`` out wholesale to drive the coalescer's timing, so nothing ever
exercised the function's own body. That is precisely the gap that let this ship.

These tests therefore drive the REAL functions with a fake SDK client and assert
the observable effect — what ``request_context`` holds at the moment the client is
constructed and queried, which is what the in-process MCP tool handlers read.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.platform.types import (
    RequestContextValue,
    attribute_detached_context,
    request_context,
)
from app.services import report_generation as rg

OWNER = "owner-sub-1234567890"
PROJECT = "proj-abcdef12-3456-7890-abcd-ef1234567890"


class _FakeClient:
    """Stands in for ``ClaudeSDKClient``, recording the ambient context.

    Records at construction, ``__aenter__`` and ``query`` because the real
    client spawns its transport tasks during ``__aenter__`` — those tasks
    inherit the ContextVar as it stands then, so a context set later would be
    invisible to the MCP handlers.
    """

    seen: list[RequestContextValue | None] = []

    opts_seen: list[Any] = []

    def __init__(self, options: Any = None) -> None:
        type(self).seen.append(request_context.get())
        type(self).opts_seen.append(options)

    async def __aenter__(self) -> _FakeClient:
        type(self).seen.append(request_context.get())
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    async def query(self, _message: str) -> None:
        type(self).seen.append(request_context.get())

    async def receive_response(self):
        type(self).seen.append(request_context.get())
        return
        yield  # pragma: no cover — makes this an async generator


@pytest.fixture
def narrator_env(tmp_path: Path, monkeypatch) -> Path:
    """Project dir plus stubs for every collaborator ``_run_narrator`` imports."""
    pd = tmp_path / "proj"
    (pd / "output").mkdir(parents=True)
    (pd / "config.json").write_text(json.dumps({
        "id": PROJECT,
        "name": "Ctx Test",
        "description": "",
        "created_at": "2026-06-01T00:00:00Z",
        "agent_model": "opus",
    }))

    _FakeClient.seen = []
    _FakeClient.opts_seen = []
    monkeypatch.setattr("claude_agent_sdk.ClaudeSDKClient", _FakeClient)
    monkeypatch.setattr(
        "app.services.mcp_server_cache.get_eval_mcp_server", lambda: {}
    )
    monkeypatch.setattr(
        "app.services.project_manager._is_cloud_mode", lambda: False
    )
    monkeypatch.setattr(
        "app.services.sdk_subprocess_env.sdk_subprocess_env", lambda: {}
    )

    class _Project:
        agent_model = "opus"

    class _PM:
        def get_project(self, _pid: str) -> _Project:
            return _Project()

    monkeypatch.setattr("app.main.get_project_manager", lambda: _PM())

    class _Platform:
        session_store = None

    monkeypatch.setattr("app.platform.get_platform", lambda: _Platform())
    return pd


@pytest.mark.asyncio
async def test_detached_call_attributes_context_to_project_owner(narrator_env):
    """With no ambient context, the owner is attributed before the client runs.

    This is the post-eval dispatch path. Without this, save_report is refused
    and the narrator cannot deliver anything.
    """
    assert request_context.get() is None, "test must start with no context"

    await rg._run_narrator(narrator_env, PROJECT, "narrate please", OWNER)

    assert _FakeClient.seen, "the fake client was never constructed"
    assert all(ctx is not None for ctx in _FakeClient.seen), (
        "request_context was missing at some point during the SDK session — "
        "save_report would be refused by mcp_authz there"
    )
    for ctx in _FakeClient.seen:
        assert ctx.user_id == OWNER
        assert ctx.project_id == PROJECT


@pytest.mark.asyncio
async def test_existing_context_is_not_replaced(narrator_env):
    """The API path already carries the caller's context; leave it alone."""
    caller = RequestContextValue(
        user_id="real-http-caller",
        request_id="http-req-1",
        project_id=PROJECT,
    )
    token = request_context.set(caller)
    try:
        await rg._run_narrator(narrator_env, PROJECT, "narrate please", OWNER)
    finally:
        request_context.reset(token)

    assert _FakeClient.seen
    for ctx in _FakeClient.seen:
        assert ctx is caller, (
            "an owner-attributed context replaced the real caller's — the API "
            "path must not be re-attributed"
        )


@pytest.mark.asyncio
async def test_detached_context_does_not_leak_past_return(narrator_env):
    """The attribution is reset, so it can't bleed into a shared caller task."""
    assert request_context.get() is None

    await rg._run_narrator(narrator_env, PROJECT, "narrate please", OWNER)

    assert request_context.get() is None, (
        "owner attribution outlived the narration — an inline caller would "
        "inherit the owner identity for every later operation"
    )


@pytest.mark.asyncio
async def test_no_owner_leaves_context_unset(narrator_env):
    """No owner to attribute to ⇒ no context invented, and no crash."""
    assert request_context.get() is None

    result = await rg._run_narrator(narrator_env, PROJECT, "narrate please", None)

    assert result is False, "no save_report succeeded, so the result is False"
    assert request_context.get() is None


class TestAttributeDetachedContext:
    """Unit tests for the shared helper both detached sites use."""

    def test_sets_when_absent(self):
        assert request_context.get() is None
        token = attribute_detached_context(
            owner_id=OWNER, project_id=PROJECT, request_id="detached-x",
        )
        try:
            assert token is not None
            ctx = request_context.get()
            assert ctx is not None
            assert ctx.user_id == OWNER
            assert ctx.project_id == PROJECT
            assert ctx.request_id == "detached-x"
        finally:
            request_context.reset(token)
        assert request_context.get() is None

    def test_context_already_bound_is_returned_untouched(self):
        caller = RequestContextValue(
            user_id="real-caller", request_id="http-1", project_id=PROJECT,
        )
        token = request_context.set(caller)
        try:
            assert attribute_detached_context(
                owner_id=OWNER, project_id=PROJECT, request_id="detached-x",
            ) is None
            assert request_context.get() is caller
        finally:
            request_context.reset(token)

    def test_existing_context_without_a_project_is_bound(self):
        """The shape an HTTP route hands these sites.

        ``auth_middleware`` sets user_id and request_id before routing and
        leaves project_id None. Both detached sites are reachable straight from
        a route — regenerating a report narrative, starting an optimization
        loop — so guarding on presence alone leaves the SDK session unbound and
        ``mcp_authz`` denies its ``save_report`` / ``save_prompt`` call.
        """
        caller = RequestContextValue(user_id="real-caller", request_id="http-1")
        outer = request_context.set(caller)
        try:
            token = attribute_detached_context(
                owner_id=OWNER, project_id=PROJECT, request_id="detached-x",
            )
            assert token is not None, (
                "a project-less context must be bound, not accepted as-is"
            )
            try:
                ctx = request_context.get()
                assert ctx is not None
                assert ctx.project_id == PROJECT
                assert ctx.user_id == "real-caller", (
                    "binding must preserve the real caller, never swap in the "
                    "owner — the two mean different things in an audit log"
                )
                assert ctx.request_id == "http-1"
            finally:
                request_context.reset(token)
            assert request_context.get() is caller
        finally:
            request_context.reset(outer)

    def test_binding_does_not_require_an_owner(self):
        """Binding a project is independent of owner attribution.

        ``owner_id`` decides who unattributed work acts as. It has no bearing
        on which project an already-identified caller is working in.
        """
        caller = RequestContextValue(user_id="real-caller", request_id="http-1")
        outer = request_context.set(caller)
        try:
            token = attribute_detached_context(
                owner_id=None, project_id=PROJECT, request_id="detached-x",
            )
            assert token is not None
            try:
                assert request_context.get().project_id == PROJECT
                assert request_context.get().user_id == "real-caller"
            finally:
                request_context.reset(token)
        finally:
            request_context.reset(outer)

    def test_no_project_leaves_an_existing_context_alone(self):
        """A ``None`` project must never blank a binding that already exists."""
        caller = RequestContextValue(
            user_id="real-caller", request_id="http-1", project_id=PROJECT,
        )
        outer = request_context.set(caller)
        try:
            assert attribute_detached_context(
                owner_id=OWNER, project_id=None, request_id="detached-x",
            ) is None
            assert request_context.get() is caller
        finally:
            request_context.reset(outer)

    def test_returns_none_without_an_owner(self):
        """Never invent an identity — no owner means no context."""
        assert attribute_detached_context(
            owner_id=None, project_id=PROJECT, request_id="detached-x",
        ) is None
        assert request_context.get() is None


@pytest.mark.asyncio
async def test_detached_optimizer_attributes_context_to_owner(narrator_env):
    """The prompt optimizer's MCP tools need a context just as the narrator's do.

    Without this, analyze_failures / get_eval_results / save_prompt are all
    refused and the loop cannot write a next prompt version at all.
    """
    from app.services import optimization_loop_orchestrator as olo

    (narrator_env / "prompts").mkdir(exist_ok=True)
    (narrator_env / "prompts" / "baseline.md").write_text("baseline prompt")
    assert request_context.get() is None

    await olo._run_prompt_optimizer(
        project_id=PROJECT,
        project_dir=narrator_env,
        iteration=1,
        prompt_version="baseline",
        job_result={
            "passed_cases": 10,
            "total_cases": 20,
            "metrics_summary": {},
            "run_output_file": "eval_results_baseline_x.json",
            "per_model_pass_rates": {},
        },
        owner_id=OWNER,
    )

    assert _FakeClient.seen, "the fake client was never constructed"
    assert all(ctx is not None for ctx in _FakeClient.seen), (
        "request_context was missing during the optimizer's SDK session — its "
        "MCP tools would be refused by mcp_authz"
    )
    for ctx in _FakeClient.seen:
        assert ctx.user_id == OWNER
    assert request_context.get() is None, "attribution leaked past the call"


class TestNarratorFailedCaseSampling:
    """The narrator's copy of the facts pack is sampled; the UI's is not."""

    def _facts(self, n: int) -> dict:
        return {
            "failed_cases": [{"case_id": i, "reason": f"r{i}"} for i in range(n)],
            "total_failures": n,
        }

    def test_small_pack_is_passed_through_unchanged(self):
        facts = self._facts(10)
        assert rg._sample_failed_cases_for_narrator(facts) is facts

    def test_large_pack_is_sampled_and_labelled_with_the_true_total(self):
        facts = self._facts(1184)
        out = rg._sample_failed_cases_for_narrator(facts)

        assert len(out["failed_cases"]) == rg._NARRATOR_MAX_FAILED_CASES
        assert out["failed_cases_sampling"]["total"] == 1184, (
            "the narrator is told to quote numbers verbatim, so an unlabelled "
            "truncation invites it to state a wrong failure count as fact"
        )
        assert out["failed_cases_sampling"]["shown"] == len(out["failed_cases"])

    def test_caller_facts_are_not_mutated(self):
        """The same dict is persisted for the UI — it must keep every row."""
        facts = self._facts(1184)
        rg._sample_failed_cases_for_narrator(facts)
        assert len(facts["failed_cases"]) == 1184

    def test_sample_spans_the_list_rather_than_taking_the_head(self):
        """A head slice would show only the first model's failures."""
        facts = self._facts(1184)
        ids = [r["case_id"] for r in
               rg._sample_failed_cases_for_narrator(facts)["failed_cases"]]
        assert ids[0] == 0
        assert ids[-1] > 1000, "the tail of the run must be represented"
        assert len(set(ids)) == len(ids), "no duplicated rows"


class TestOptimizerToolRestriction:
    """The optimizer must not be able to write prompt files directly.

    While ``save_prompt`` was refused for lack of an authz context, the optimizer
    wrote ``prompts/<version>.md`` with Write and the loop reported success from its
    filesystem fallback — bypassing every guard in that tool (dedup,
    degenerate-state, empty-content, version-name validation).
    """

    @pytest.mark.asyncio
    async def test_write_edit_bash_are_removed_not_merely_unapproved(self, narrator_env):
        """``allowed_tools`` is auto-approve, NOT a gate — the removal list is.

        This is the trap: the session already listed only Read + eval MCP tools in
        ``allowed_tools``, which removed nothing, and ``bypassPermissions`` approved
        everything anyway. A future "fix" that only tightens ``allowed_tools`` would
        change nothing, so this asserts the REMOVAL list specifically.
        """
        from app.services import optimization_loop_orchestrator as olo

        (narrator_env / "prompts").mkdir(exist_ok=True)
        (narrator_env / "prompts" / "baseline.md").write_text("baseline prompt")

        await olo._run_prompt_optimizer(
            project_id=PROJECT,
            project_dir=narrator_env,
            iteration=1,
            prompt_version="baseline",
            job_result={
                "passed_cases": 10, "total_cases": 20, "metrics_summary": {},
                "run_output_file": "eval_results_baseline_x.json",
                "per_model_pass_rates": {},
            },
            owner_id=OWNER,
        )

        assert _FakeClient.opts_seen, "the optimizer client was never constructed"
        opts = _FakeClient.opts_seen[-1]
        removed = set(getattr(opts, "disallowed_tools", None) or [])
        for tool in ("Write", "Edit", "Bash"):
            assert tool in removed, (
                f"{tool} must be REMOVED from the optimizer session; "
                "allowed_tools does not gate it"
            )

    @pytest.mark.asyncio
    async def test_optimizer_keeps_read_and_its_mcp_tools(self, narrator_env):
        """Over-restricting stalls the loop, so the tools it needs must survive."""
        from app.services import optimization_loop_orchestrator as olo

        (narrator_env / "prompts").mkdir(exist_ok=True)
        (narrator_env / "prompts" / "baseline.md").write_text("baseline prompt")

        await olo._run_prompt_optimizer(
            project_id=PROJECT, project_dir=narrator_env, iteration=1,
            prompt_version="baseline",
            job_result={"passed_cases": 1, "total_cases": 2, "metrics_summary": {},
                        "run_output_file": "r.json", "per_model_pass_rates": {}},
            owner_id=OWNER,
        )
        opts = _FakeClient.opts_seen[-1]
        removed = set(getattr(opts, "disallowed_tools", None) or [])
        assert "Read" not in removed
        assert not any(t.startswith("mcp__eval__") for t in removed), (
            "removing its own MCP tools would stall the loop"
        )


class TestOptimizerOutcomeReporting:
    """What the loop concludes after the optimizer session ends.

    Lives here because this is the only fixture that drives the real
    ``_run_prompt_optimizer`` (a fake SDK client plus a project tree); the
    outcome branches had no coverage at all.

    The verdict is the filesystem — did a new prompt version appear — and the
    hook state is diagnostic. That split matters now that ``Write``/``Edit``
    are removed from the session: ``save_prompt`` is the only way a version
    file can appear, so a new file with no hook success means the HOOK is
    broken. The old code logged that case as "detected via filesystem" at INFO,
    wording from when the agent really could write around the tool.
    """

    @pytest.mark.asyncio
    async def test_no_new_prompt_returns_false_and_reports_hook_state(
        self, narrator_env, monkeypatch,
    ):
        from unittest.mock import MagicMock

        from app.services import optimization_loop_orchestrator as olo

        (narrator_env / "prompts").mkdir(exist_ok=True)
        (narrator_env / "prompts" / "baseline.md").write_text("baseline prompt")

        fake_logger = MagicMock()
        monkeypatch.setattr(olo, "logger", fake_logger)

        produced = await olo._run_prompt_optimizer(
            project_id=PROJECT, project_dir=narrator_env, iteration=3,
            prompt_version="baseline",
            job_result={"passed_cases": 1, "total_cases": 2, "metrics_summary": {},
                        "run_output_file": "r.json", "per_model_pass_rates": {}},
            owner_id=OWNER,
        )

        assert produced is False, (
            "no new prompt version on disk means the optimizer produced nothing"
        )
        warnings = " ".join(str(c) for c in fake_logger.warning.call_args_list)
        assert "without producing a new prompt" in warnings
        # Both hook flags are reported: "attempted but failed" and "never
        # called at all" are different defects and the log must tell them apart.
        assert "attempted=%s" in warnings and "succeeded=%s" in warnings

    @pytest.mark.asyncio
    async def test_new_prompt_without_hook_success_is_flagged_as_hook_defect(
        self, narrator_env, monkeypatch,
    ):
        from unittest.mock import MagicMock

        from app.services import optimization_loop_orchestrator as olo

        prompts = narrator_env / "prompts"
        prompts.mkdir(exist_ok=True)
        (prompts / "baseline.md").write_text("baseline prompt")
        # A version newer than the one passed in — as though save_prompt wrote it
        # while the PostToolUse hook never reported success.
        (prompts / "v1.md").write_text("optimized prompt")

        fake_logger = MagicMock()
        monkeypatch.setattr(olo, "logger", fake_logger)

        produced = await olo._run_prompt_optimizer(
            project_id=PROJECT, project_dir=narrator_env, iteration=3,
            prompt_version="baseline",
            job_result={"passed_cases": 1, "total_cases": 2, "metrics_summary": {},
                        "run_output_file": "r.json", "per_model_pass_rates": {}},
            owner_id=OWNER,
        )

        assert produced is True, "a new prompt version exists — the loop continues"
        warnings = " ".join(str(c) for c in fake_logger.warning.call_args_list)
        assert "hook" in warnings.lower(), (
            "a prompt file with no hook success must read as a hook defect, not "
            "as a routine filesystem detection"
        )
        infos = " ".join(str(c) for c in fake_logger.info.call_args_list)
        assert "Optimizer saved new prompt" not in infos, (
            "the clean-success line must not fire when the hook reported nothing"
        )
