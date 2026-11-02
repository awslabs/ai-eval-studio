# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the model-invocability preflight on dispatch.

The preflight catches models added to a project's lineup AFTER
``validate_harness`` last ran (e.g. edited directly into ``config.py``)
and refuses to dispatch with a clear error rather than letting the
real eval stall for 20 minutes on a hung Converse call. It runs on a
dedicated boto3 client with tight timeouts so a single dead model
costs ~15s, not the production 300s × 4 retries.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.jobs import get_job_tracker
# _reap_loop_child_eval moved with the loop driver to
# services/optimization_loop_orchestrator.
from app.services.optimization_loop_orchestrator import _reap_loop_child_eval
from app.services.mcp_tools.dispatch import (
    _preflight_models_invocable,
    dispatch_eval as _dispatch_eval_tool,
    dispatch_optimization as _dispatch_opt_tool,
)

_dispatch_eval = _dispatch_eval_tool.handler
_dispatch_optimization = _dispatch_opt_tool.handler


class TestReapLoopChildEval:
    """When the optimization loop exits via cancel/exception, it
    must reap its in-flight child eval so the tracker never holds an
    active eval that no live loop owns — the invariant the rebuilt
    /activity endpoint depends on."""

    def test_reaps_active_optimization_child_eval(self):
        tracker = get_job_tracker()
        tracker._reset()
        job = tracker.create_job(
            "proj-reap", "eval",
            config={"model_ids": ["m"], "prompt_version": "v1",
                    "source": "optimization"},
        )
        tracker.update_progress(job.job_id, message="running", models_total=1)
        assert tracker.get_active_job_by_type("proj-reap", "eval") is not None

        _reap_loop_child_eval("proj-reap")

        # The child eval is now terminal (cancelled), not active.
        assert tracker.get_active_job_by_type("proj-reap", "eval") is None
        assert tracker.get_job(job.job_id).state == "cancelled"
        tracker._reset()

    def test_leaves_standalone_manual_eval_untouched(self):
        # A manual eval is not the loop's child — reaping must not touch
        # it even if a (now-dead) loop is exiting on the same project.
        tracker = get_job_tracker()
        tracker._reset()
        job = tracker.create_job(
            "proj-reap", "eval",
            config={"model_ids": ["m"], "prompt_version": "v1",
                    "source": "manual"},
        )
        tracker.update_progress(job.job_id, message="running", models_total=1)

        _reap_loop_child_eval("proj-reap")

        assert tracker.get_active_job_by_type("proj-reap", "eval") is not None
        assert tracker.get_job(job.job_id).state == "running"
        tracker._reset()

    def test_noop_when_no_active_eval(self):
        tracker = get_job_tracker()
        tracker._reset()
        # No jobs at all — must not raise.
        _reap_loop_child_eval("proj-reap")
        assert tracker.get_active_job_by_type("proj-reap", "eval") is None


@pytest.fixture
def project_setup(tmp_path):
    """Minimal project that gets past the data-file / prompt checks."""
    project_id = "preflight-test-id"
    pd = tmp_path / project_id
    (pd / "data" / "raw").mkdir(parents=True)
    (pd / "prompts").mkdir(parents=True)
    (pd / "data" / "raw" / "data.csv").write_text("input,expected\nq,a\n")
    (pd / "prompts" / "baseline.md").write_text("hi")
    (pd / "run_eval.py").write_text("# stub")
    get_job_tracker()._reset()
    return {"project_id": project_id, "project_dir": pd, "tmp_path": tmp_path}


class TestPreflightHelper:
    """Direct unit tests of ``_preflight_models_invocable``."""

    async def test_all_good_returns_empty(self):
        # Every probe returns a Converse-shaped response with text — no
        # model is reported as bad.
        ok_resp = {"output": {"message": {"content": [{"text": "OK"}]}}}
        client = MagicMock()
        client.converse.return_value = ok_resp
        with patch("boto3.client", return_value=client):
            bad = await _preflight_models_invocable(["good-1", "good-2"])
        assert bad == []
        assert client.converse.call_count == 2

    async def test_read_timeout_marks_model_bad(self):
        from botocore.exceptions import ReadTimeoutError
        client = MagicMock()

        def converse(modelId, **_kw):
            if modelId == "hangs":
                raise ReadTimeoutError(endpoint_url="https://x")
            return {"output": {"message": {"content": [{"text": "OK"}]}}}

        client.converse.side_effect = converse
        with patch("boto3.client", return_value=client):
            bad = await _preflight_models_invocable(["good", "hangs"])

        assert len(bad) == 1
        assert "hangs" in bad[0]
        assert "ReadTimeoutError" in bad[0]

    async def test_validation_exception_marks_model_bad(self):
        from botocore.exceptions import ClientError
        client = MagicMock()

        def converse(modelId, **_kw):
            if modelId == "invalid-id":
                raise ClientError(
                    {"Error": {"Code": "ValidationException",
                               "Message": "model not found"}},
                    "Converse",
                )
            return {"output": {"message": {"content": [{"text": "OK"}]}}}

        client.converse.side_effect = converse
        with patch("boto3.client", return_value=client):
            bad = await _preflight_models_invocable(["good", "invalid-id"])

        assert len(bad) == 1
        assert "invalid-id" in bad[0]
        assert "ValidationException" in bad[0]

    async def test_empty_output_marks_model_bad(self):
        # Bedrock returned a 200 but the model produced no text — treat
        # it as unusable rather than letting it through to score nulls.
        client = MagicMock()
        client.converse.return_value = {
            "output": {"message": {"content": [{"text": ""}]}}
        }
        with patch("boto3.client", return_value=client):
            bad = await _preflight_models_invocable(["mute"])
        assert len(bad) == 1
        assert "mute" in bad[0]
        assert "empty output" in bad[0]

    async def test_uses_short_timeout_client(self):
        # Pin the contract: this MUST NOT use the production
        # 300s×4-retries client. A regression here re-introduces the
        # 20-min stall.
        from botocore.config import Config
        ok_resp = {"output": {"message": {"content": [{"text": "OK"}]}}}
        client = MagicMock()
        client.converse.return_value = ok_resp
        with patch("boto3.client", return_value=client) as mock_factory:
            await _preflight_models_invocable(["m"])

        assert mock_factory.call_count == 1
        kwargs = mock_factory.call_args.kwargs
        cfg = kwargs.get("config")
        assert isinstance(cfg, Config)
        assert cfg.read_timeout == 15
        assert cfg.connect_timeout == 5
        # ``retries`` lands as a dict; the key the helper passed was
        # ``max_attempts``: 1 (single try, no exponential backoff).
        assert cfg.retries["max_attempts"] == 1

    async def test_empty_input_returns_empty(self):
        # Defensive: don't construct a client / spawn tasks for an
        # empty list.
        with patch("boto3.client") as mock_factory:
            bad = await _preflight_models_invocable([])
        assert bad == []
        assert mock_factory.call_count == 0

    async def test_builds_client_with_explicit_region(self, monkeypatch):
        """Without an explicit region, the probe client cannot be
        constructed at all in an environment that carries this app's
        region variable but no AWS config file, so dispatch dies
        before any model is checked."""
        from botocore.exceptions import NoRegionError

        captured_region = None

        def factory(*args, **kwargs):
            # Reject calls without a truthy region_name, matching
            # botocore's own ambient-region behavior: when neither
            # AWS_DEFAULT_REGION nor a config file is present, client
            # construction raises NoRegionError before any request.
            nonlocal captured_region
            region = kwargs.get("region_name")
            if not region:
                raise NoRegionError()
            captured_region = region
            # Return a stub that answers the Converse shape.
            client = MagicMock()
            client.converse.return_value = {
                "output": {"message": {"content": [{"text": "OK"}]}}
            }
            return client

        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        with patch("boto3.client", factory):
            bad = await _preflight_models_invocable(["some-runtime-id"])

        assert bad == [], f"Expected no failures but got: {bad}"
        assert captured_region == "eu-west-1", (
            f"Expected region_name='eu-west-1' but got {captured_region!r}"
        )

    async def test_region_defaults_when_env_unset(self, monkeypatch):
        """With AWS_REGION unset, the probe client still gets an explicit
        region — the us-east-1 default — instead of relying on botocore's
        ambient resolution, which raises NoRegionError without a config file."""
        from botocore.exceptions import NoRegionError

        captured_region = None

        def factory(*args, **kwargs):
            nonlocal captured_region
            region = kwargs.get("region_name")
            if not region:
                raise NoRegionError()
            captured_region = region
            client = MagicMock()
            client.converse.return_value = {
                "output": {"message": {"content": [{"text": "OK"}]}}
            }
            return client

        monkeypatch.delenv("AWS_REGION", raising=False)
        with patch("boto3.client", factory):
            bad = await _preflight_models_invocable(["some-runtime-id"])

        assert bad == [], f"Expected no failures but got: {bad}"
        assert captured_region == "us-east-1", (
            f"Expected region_name='us-east-1' but got {captured_region!r}"
        )


class TestDispatchEvalGate:
    """``dispatch_eval`` refuses to create a job when the preflight fails."""

    async def test_refuses_with_named_bad_models(self, project_setup):
        # The job is now created BEFORE the preflight probe (so /activity
        # surfaces the eval the instant dispatch begins, killing the
        # startup blackout). A failed preflight must therefore FAIL the
        # job, not orphan it active — otherwise the eval bar would report
        # a phantom "running" eval forever.
        tracker = get_job_tracker()
        with patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve, \
             patch(
                 "app.services.mcp_tools.dispatch._preflight_models_invocable",
                 new_callable=AsyncMock,
                 return_value=["zai.glm-5 (ReadTimeoutError)"],
             ):
            mock_resolve.return_value = project_setup["project_dir"]
            result = await _dispatch_eval({
                "project_id": project_setup["project_id"],
                "data_file": "data.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["zai.glm-5", "us.anthropic.claude-haiku-4-5-20251001-v1:0"],
            })

        assert result.get("is_error") is True
        text = result["content"][0]["text"]
        assert "zai.glm-5" in text
        assert "ReadTimeoutError" in text
        # A job WAS created (pre-preflight) but it must be terminal-failed,
        # never left active. No active eval can survive a bad preflight.
        eval_jobs = [j for j in tracker._jobs.values() if j.type == "eval"]
        assert len(eval_jobs) == 1
        job = eval_jobs[0]
        assert job.state == "failed"
        assert job.is_active is False
        assert "Models not invocable" in job.error.message
        assert tracker.get_active_job_by_type(
            project_setup["project_id"], "eval"
        ) is None

    async def test_proceeds_when_all_models_invocable(self, project_setup):
        # Patch the background-task spawn — the goal of this test is to
        # confirm the gate doesn't refuse when preflight is clean, not
        # to drive a real eval. The real eval-executor + completion-turn
        # require auth context + project manager that the dispatch test
        # fixture doesn't set up; leaving the task to actually run leaks
        # a coroutine that hangs subsequent tests in the file.
        with patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve, \
             patch(
                 "app.services.mcp_tools.dispatch._preflight_models_invocable",
                 new_callable=AsyncMock, return_value=[],
             ), patch(
                 "app.services.mcp_tools.dispatch._run_eval_with_completion",
                 new_callable=AsyncMock,
             ):
            mock_resolve.return_value = project_setup["project_dir"]
            result = await _dispatch_eval({
                "project_id": project_setup["project_id"],
                "data_file": "data.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["us.anthropic.claude-haiku-4-5-20251001-v1:0"],
            })

        # The job was created — happy path.
        assert "is_error" not in result
        # The job lands as JSON via mcp_json.
        body = json.loads(result["content"][0]["text"])
        assert "job_id" in body

    async def test_preflight_includes_judge_model(self, project_setup):
        # Judge has its own Bedrock cost path and needs its own probe;
        # a broken judge would silently null every metric score. Patch
        # the background-task spawn so the test doesn't leak a coroutine
        # into other tests via the (real) eval executor + completion-turn
        # plumbing.
        preflight_mock = AsyncMock(return_value=[])
        with patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve, \
             patch(
                 "app.services.mcp_tools.dispatch._preflight_models_invocable",
                 preflight_mock,
             ), patch(
                 "app.services.mcp_tools.dispatch._run_eval_with_completion",
                 new_callable=AsyncMock,
             ):
            mock_resolve.return_value = project_setup["project_dir"]
            await _dispatch_eval({
                "project_id": project_setup["project_id"],
                "data_file": "data.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["target-a", "target-b"],
                "judge_model_id": "judge-x",
            })

        preflight_mock.assert_awaited_once()
        ids = preflight_mock.await_args.args[0]
        assert "target-a" in ids
        assert "target-b" in ids
        assert "judge-x" in ids

    async def test_preflight_raise_fails_the_job_and_frees_the_fence(self, project_setup):
        """An orphaned active job answers the concurrent-eval fence forever,
        so the coordinator's retries are all rejected and dev-mode recovery
        needs a process restart."""
        tracker = get_job_tracker()

        # First call: preflight raises, job should be failed and fence freed
        with patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve, \
             patch(
                 "app.services.mcp_tools.dispatch._preflight_models_invocable",
                 new_callable=AsyncMock,
                 side_effect=Exception("Preflight boom"),
             ):
            mock_resolve.return_value = project_setup["project_dir"]
            result = await _dispatch_eval({
                "project_id": project_setup["project_id"],
                "data_file": "data.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["test-model"],
            })

        # Result is an error envelope with the exception text
        assert result.get("is_error") is True, "Expected error envelope"
        text = result["content"][0]["text"]
        assert "Preflight boom" in text, f"Expected exception text in error, got: {text}"

        # Job must be failed, not active
        eval_jobs = [j for j in tracker._jobs.values() if j.type == "eval"]
        assert len(eval_jobs) == 1, f"Expected 1 eval job, got {len(eval_jobs)}"
        job = eval_jobs[0]
        assert job.state == "failed", f"Expected job state 'failed', got {job.state!r}"
        assert job.is_active is False, "Expected job.is_active to be False"
        assert job.error is not None, "Expected job.error to be set"

        # Fence must be freed
        assert tracker.get_active_job_by_type(
            project_setup["project_id"], "eval"
        ) is None, "Expected no active eval job after failure"

        # Second call: clean preflight should succeed, not be rejected by fence
        with patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve, \
             patch(
                 "app.services.mcp_tools.dispatch._preflight_models_invocable",
                 new_callable=AsyncMock,
                 return_value=[],
             ), patch(
                 "app.services.mcp_tools.dispatch._run_eval_with_completion",
                 new_callable=AsyncMock,
             ):
            mock_resolve.return_value = project_setup["project_dir"]
            result2 = await _dispatch_eval({
                "project_id": project_setup["project_id"],
                "data_file": "data.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["test-model"],
            })

        # Second dispatch succeeded
        assert "is_error" not in result2, "Expected second dispatch to succeed"
        body = json.loads(result2["content"][0]["text"])
        assert "job_id" in body, "Expected job_id in second dispatch response"

    async def test_exception_before_job_creation_returns_clean_error(self, project_setup):
        """An exception before the job exists returns an error envelope and
        leaves no job behind."""
        tracker = get_job_tracker()

        # Patch resolve_project_dir to raise before job creation
        with patch(
            "app.services.mcp_tools.dispatch.resolve_project_dir",
            side_effect=Exception("Pre-job boom"),
        ):
            result = await _dispatch_eval({
                "project_id": project_setup["project_id"],
                "data_file": "data.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["test-model"],
            })

        # Must return an error envelope, not raise
        assert result.get("is_error") is True, "Expected error envelope"
        text = result["content"][0]["text"]
        assert "Pre-job boom" in text, f"Expected exception text in error, got: {text}"

        # No eval job should exist
        eval_jobs = [j for j in tracker._jobs.values() if j.type == "eval"]
        assert len(eval_jobs) == 0, (
            f"Expected no eval jobs when exception before job creation, got {len(eval_jobs)}"
        )

    async def test_successful_dispatch_leaves_job_active(self, project_setup):
        """A successful dispatch reports "dispatched" and leaves exactly one
        eval job active with no error; the run itself is patched out."""
        tracker = get_job_tracker()

        with patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve, \
             patch(
                 "app.services.mcp_tools.dispatch._preflight_models_invocable",
                 new_callable=AsyncMock,
                 return_value=[],
             ), patch(
                 "app.services.mcp_tools.dispatch._run_eval_with_completion",
                 new_callable=AsyncMock,
             ):
            mock_resolve.return_value = project_setup["project_dir"]
            result = await _dispatch_eval({
                "project_id": project_setup["project_id"],
                "data_file": "data.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["test-model"],
            })

        # Success response
        assert "is_error" not in result, "Expected successful dispatch"
        body = json.loads(result["content"][0]["text"])
        assert body["status"] == "dispatched", (
            f"Expected status 'dispatched', got {body.get('status')!r}"
        )

        # Job must be active with no error
        eval_jobs = [j for j in tracker._jobs.values() if j.type == "eval"]
        assert len(eval_jobs) == 1, f"Expected 1 eval job, got {len(eval_jobs)}"
        job = eval_jobs[0]
        assert job.is_active is True, "Expected job.is_active to be True"
        assert job.error is None, f"Expected no error on success, got {job.error!r}"


class TestDispatchOptimizationGate:
    """``dispatch_optimization`` runs the same preflight before creating
    optimization state, so a broken model in the lineup doesn't leave a
    half-converged ``state.json`` behind."""

    async def test_refuses_with_named_bad_models(self, project_setup):
        # Stub a harness config so we get past the harness checks.
        from app.models.schemas import HarnessConfig
        harness = HarnessConfig(
            models=["zai.glm-5", "good-model"],
            judge_model_id="judge-x",
            metrics=[{"name": "ExactMatch", "type": "exact_match"}],
            field_mappings={"input": "input", "expected_output": "expected"},
            data_file="data.csv",
            target_metric_scores={"ExactMatch": 0.85},
            max_iterations=10,
            plateau_window=3,
        )

        with patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve, \
             patch(
                 "app.services.mcp_tools.dispatch.load_harness_config",
                 return_value=harness,
             ), patch(
                 "app.services.mcp_tools.dispatch._preflight_models_invocable",
                 new_callable=AsyncMock,
                 return_value=["zai.glm-5 (ReadTimeoutError)"],
             ):
            mock_resolve.return_value = project_setup["project_dir"]
            result = await _dispatch_optimization({
                "project_id": project_setup["project_id"],
            })

        assert result.get("is_error") is True
        text = result["content"][0]["text"]
        assert "zai.glm-5" in text
        assert "ReadTimeoutError" in text
        # State must NOT have been created.
        state_file = project_setup["project_dir"] / "output" / "optimization" / "state.json"
        assert not state_file.exists()

    async def test_preflight_includes_judge_in_optimization(self, project_setup):
        from app.models.schemas import HarnessConfig
        harness = HarnessConfig(
            models=["m1", "m2"],
            judge_model_id="judge-x",
            metrics=[{"name": "ExactMatch", "type": "exact_match"}],
            field_mappings={"input": "input", "expected_output": "expected"},
            data_file="data.csv",
            target_metric_scores={"ExactMatch": 0.85},
            max_iterations=10,
            plateau_window=3,
        )
        preflight_mock = AsyncMock(return_value=[])
        # Patch the orchestrator's loop spawn so we don't actually start
        # iterations — we only care that the preflight gate ran with the
        # right ids and didn't refuse.
        with patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve, \
             patch(
                 "app.services.mcp_tools.dispatch.load_harness_config",
                 return_value=harness,
             ), patch(
                 "app.services.mcp_tools.dispatch._preflight_models_invocable",
                 preflight_mock,
             ), patch(
                 "app.services.optimization_loop_orchestrator.run_optimization_loop",
                 new_callable=AsyncMock,
             ):
            mock_resolve.return_value = project_setup["project_dir"]
            await _dispatch_optimization({
                "project_id": project_setup["project_id"],
            })

        preflight_mock.assert_awaited_once()
        ids = preflight_mock.await_args.args[0]
        assert "m1" in ids
        assert "m2" in ids
        assert "judge-x" in ids


class TestPreflightMantleBranch:
    """Mantle-aware branch of ``_preflight_models_invocable``.

    Pinned behaviors:
      * Mantle ids in ``descriptors`` route to ``BedrockMantleModel.preflight``.
      * Runtime ids stay on the existing boto3 Converse path.
      * 404 ``ModelRemovedFromMantle`` AND 400 ``MantleAPIRejected``
        invalidate the on-disk descriptor.
      * Timeout AND ``MantleNotInvocable`` (transient/auth) DO NOT
        invalidate.
      * The augmented user-facing reason includes ``descriptor
        invalidated — re-run list_bedrock_models``.
      * Idempotent — second invocation routes the now-deleted id to
        the runtime path; descriptor file unchanged on disk.
      * ``descriptors=None`` (existing test pattern) falls back to all
        runtime — backward-compat regression gate.
      * ``dispatch_eval`` and ``dispatch_optimization`` thread
        descriptors+project_id+project_dir through.
    """

    def _make_descriptor_file(self, project_dir: Path, descriptors: dict) -> Path:
        from app.services.model_descriptors import write_descriptors
        return write_descriptors(project_dir, descriptors)

    async def test_mantle_id_success_returns_no_failure(
        self, project_setup, monkeypatch
    ):
        from app.services.bedrock_mantle_model import BedrockMantleModel

        pd = project_setup["project_dir"]
        descriptors = {
            "openai.gpt-5.5": {
                "endpoint": "mantle",
                "api": "responses",
                "region": "us-east-1",
            }
        }
        self._make_descriptor_file(pd, descriptors)

        # Stub preflight to return None (success).
        monkeypatch.setattr(
            BedrockMantleModel, "preflight", lambda self: None
        )

        bad = await _preflight_models_invocable(
            ["openai.gpt-5.5"],
            descriptors=descriptors,
            project_id=project_setup["project_id"],
            project_dir=pd,
        )
        assert bad == []
        # Descriptor file still has the id — no invalidation on success.
        from app.services.model_descriptors import read_descriptors
        assert "openai.gpt-5.5" in read_descriptors(pd)

    async def test_mantle_404_invalidates_descriptor(
        self, project_setup, monkeypatch
    ):
        from app.services.bedrock_mantle_model import (
            BedrockMantleModel,
            ModelRemovedFromMantle,
        )

        pd = project_setup["project_dir"]
        descriptors = {
            "openai.gpt-5.4": {
                "endpoint": "mantle",
                "api": "responses",
                "region": "us-east-1",
            }
        }
        self._make_descriptor_file(pd, descriptors)

        def boom(self):
            raise ModelRemovedFromMantle(self.model_id)

        monkeypatch.setattr(BedrockMantleModel, "preflight", boom)

        bad = await _preflight_models_invocable(
            ["openai.gpt-5.4"],
            descriptors=descriptors,
            project_id=project_setup["project_id"],
            project_dir=pd,
        )
        assert len(bad) == 1
        assert "openai.gpt-5.4" in bad[0]
        assert "ModelRemovedFromMantle" in bad[0]
        assert "descriptor invalidated" in bad[0]
        # On-disk descriptor file no longer has the id.
        from app.services.model_descriptors import read_descriptors
        assert "openai.gpt-5.4" not in read_descriptors(pd)

    async def test_mantle_400_does_not_support_invalidates_descriptor(
        self, project_setup, monkeypatch
    ):
        from app.services.bedrock_mantle_mappers import MantleAPIRejected
        from app.services.bedrock_mantle_model import BedrockMantleModel

        pd = project_setup["project_dir"]
        descriptors = {
            "openai.gpt-5.5": {
                "endpoint": "mantle",
                "api": "chat_completions",
                "region": "us-east-1",
            }
        }
        self._make_descriptor_file(pd, descriptors)

        def boom(self):
            raise MantleAPIRejected(
                f"{self.model_id}: does not support /chat/completions"
            )

        monkeypatch.setattr(BedrockMantleModel, "preflight", boom)

        bad = await _preflight_models_invocable(
            ["openai.gpt-5.5"],
            descriptors=descriptors,
            project_id=project_setup["project_id"],
            project_dir=pd,
        )
        assert len(bad) == 1
        assert "openai.gpt-5.5" in bad[0]
        assert "MantleAPIRejected" in bad[0]
        assert "descriptor invalidated" in bad[0]
        from app.services.model_descriptors import read_descriptors
        assert "openai.gpt-5.5" not in read_descriptors(pd)

    async def test_mantle_timeout_does_not_invalidate(
        self, project_setup, monkeypatch
    ):
        import time
        from app.services.bedrock_mantle_model import BedrockMantleModel

        pd = project_setup["project_dir"]
        descriptors = {
            "openai.gpt-5.5": {
                "endpoint": "mantle",
                "api": "responses",
                "region": "us-east-1",
            }
        }
        self._make_descriptor_file(pd, descriptors)

        # Patch the wait_for ceiling to a tiny value, then make
        # preflight sleep past it so asyncio.TimeoutError fires.
        import app.services.mcp_tools.dispatch as dispatch_mod
        original_wait_for = dispatch_mod.asyncio.wait_for

        async def fast_wait_for(coro, timeout):
            return await original_wait_for(coro, timeout=0.01)

        monkeypatch.setattr(dispatch_mod.asyncio, "wait_for", fast_wait_for)

        def slow_preflight(self):
            time.sleep(0.5)

        monkeypatch.setattr(BedrockMantleModel, "preflight", slow_preflight)

        bad = await _preflight_models_invocable(
            ["openai.gpt-5.5"],
            descriptors=descriptors,
            project_id=project_setup["project_id"],
            project_dir=pd,
        )
        assert len(bad) == 1
        assert "openai.gpt-5.5" in bad[0]
        assert "TimeoutError" in bad[0]
        # Descriptor file untouched — transient.
        assert "descriptor invalidated" not in bad[0]
        from app.services.model_descriptors import read_descriptors
        assert "openai.gpt-5.5" in read_descriptors(pd)

    async def test_mantle_not_invocable_signature_mismatch_does_not_invalidate(
        self, project_setup, monkeypatch
    ):
        from app.services.bedrock_mantle_model import (
            BedrockMantleModel,
            MantleNotInvocable,
        )

        pd = project_setup["project_dir"]
        descriptors = {
            "openai.gpt-5.5": {
                "endpoint": "mantle",
                "api": "responses",
                "region": "us-east-1",
            }
        }
        self._make_descriptor_file(pd, descriptors)

        def boom(self):
            raise MantleNotInvocable(
                self.model_id, "MantleSignatureMismatch", "403 body"
            )

        monkeypatch.setattr(BedrockMantleModel, "preflight", boom)

        bad = await _preflight_models_invocable(
            ["openai.gpt-5.5"],
            descriptors=descriptors,
            project_id=project_setup["project_id"],
            project_dir=pd,
        )
        assert len(bad) == 1
        assert "openai.gpt-5.5" in bad[0]
        # Transient — descriptor stays.
        assert "descriptor invalidated" not in bad[0]
        from app.services.model_descriptors import read_descriptors
        assert "openai.gpt-5.5" in read_descriptors(pd)

    async def test_mixed_runtime_and_mantle_batch(
        self, project_setup, monkeypatch
    ):
        from app.services.bedrock_mantle_model import (
            BedrockMantleModel,
            ModelRemovedFromMantle,
        )

        pd = project_setup["project_dir"]
        descriptors = {
            "openai.gpt-5.5": {
                "endpoint": "mantle",
                "api": "responses",
                "region": "us-east-1",
            },
            "openai.gpt-5.4": {
                "endpoint": "mantle",
                "api": "responses",
                "region": "us-east-1",
            },
        }
        self._make_descriptor_file(pd, descriptors)

        # 2 runtime ids: one good, one bad. 2 mantle ids: one good, one removed.
        ok_resp = {"output": {"message": {"content": [{"text": "OK"}]}}}
        from botocore.exceptions import ClientError
        client = MagicMock()

        def converse(modelId, **_kw):
            if modelId == "bad-runtime":
                raise ClientError(
                    {"Error": {"Code": "ValidationException",
                               "Message": "bad"}},
                    "Converse",
                )
            return ok_resp

        client.converse.side_effect = converse

        def mantle_preflight(self):
            if self.model_id == "openai.gpt-5.4":
                raise ModelRemovedFromMantle(self.model_id)
            return None

        monkeypatch.setattr(BedrockMantleModel, "preflight", mantle_preflight)

        with patch("boto3.client", return_value=client):
            bad = await _preflight_models_invocable(
                [
                    "good-runtime",
                    "bad-runtime",
                    "openai.gpt-5.5",
                    "openai.gpt-5.4",
                ],
                descriptors=descriptors,
                project_id=project_setup["project_id"],
                project_dir=pd,
            )

        # 2 failures: bad-runtime and openai.gpt-5.4.
        assert len(bad) == 2
        # Order: runtime failures first (existing contract preserved),
        # then mantle failures.
        assert "bad-runtime" in bad[0]
        assert "openai.gpt-5.4" in bad[1]
        assert "descriptor invalidated" in bad[1]
        # Descriptor file: only the good mantle id remains.
        from app.services.model_descriptors import read_descriptors
        on_disk = read_descriptors(pd)
        assert "openai.gpt-5.5" in on_disk
        assert "openai.gpt-5.4" not in on_disk

    async def test_idempotent_double_invalidate(
        self, project_setup, monkeypatch
    ):
        from app.services.bedrock_mantle_model import (
            BedrockMantleModel,
            ModelRemovedFromMantle,
        )
        from app.services.model_descriptors import read_descriptors

        pd = project_setup["project_dir"]
        descriptors = {
            "openai.gpt-5.4": {
                "endpoint": "mantle",
                "api": "responses",
                "region": "us-east-1",
            },
        }
        self._make_descriptor_file(pd, descriptors)

        def boom(self):
            raise ModelRemovedFromMantle(self.model_id)

        monkeypatch.setattr(BedrockMantleModel, "preflight", boom)

        # First invocation invalidates.
        bad1 = await _preflight_models_invocable(
            ["openai.gpt-5.4"],
            descriptors=descriptors,
            project_id=project_setup["project_id"],
            project_dir=pd,
        )
        assert len(bad1) == 1
        assert "descriptor invalidated" in bad1[0]
        descriptor_path = pd / "src" / "model_descriptors.json"
        size_after_first = descriptor_path.stat().st_size
        bytes_after_first = descriptor_path.read_bytes()

        # Second invocation: pass the ORIGINAL descriptors map so
        # routing still tries the mantle path; on-disk persistence
        # uses the read-fresh value (which is already empty), so the
        # delete loop finds nothing to delete and skips the write.
        # The id falls into the augmented "invalidated" failure list
        # again, but the file does NOT change.
        bad2 = await _preflight_models_invocable(
            ["openai.gpt-5.4"],
            descriptors=descriptors,
            project_id=project_setup["project_id"],
            project_dir=pd,
        )
        # Either way, descriptor file is unchanged.
        assert descriptor_path.stat().st_size == size_after_first
        assert descriptor_path.read_bytes() == bytes_after_first
        # And the id is still gone from disk.
        assert "openai.gpt-5.4" not in read_descriptors(pd)

    async def test_descriptors_none_falls_back_to_all_runtime(self):
        # Existing test pattern — positional call without descriptors
        # routes everything to the runtime probe path.
        ok_resp = {"output": {"message": {"content": [{"text": "OK"}]}}}
        client = MagicMock()
        client.converse.return_value = ok_resp
        with patch("boto3.client", return_value=client):
            bad = await _preflight_models_invocable(["good-1", "good-2"])
        assert bad == []
        # Confirms runtime path was taken (boto3.client called).
        assert client.converse.call_count == 2

    async def test_dispatch_eval_threads_descriptors_through(
        self, project_setup
    ):
        # Write a descriptor file so dispatch_eval picks it up.
        from app.services.model_descriptors import write_descriptors
        write_descriptors(
            project_setup["project_dir"],
            {
                "openai.gpt-5.5": {
                    "endpoint": "mantle",
                    "api": "responses",
                    "region": "us-east-1",
                }
            },
        )

        preflight_mock = AsyncMock(return_value=[])
        with patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve, \
             patch(
                 "app.services.mcp_tools.dispatch._preflight_models_invocable",
                 preflight_mock,
             ), patch(
                 "app.services.mcp_tools.dispatch._run_eval_with_completion",
                 new_callable=AsyncMock,
             ):
            mock_resolve.return_value = project_setup["project_dir"]
            await _dispatch_eval({
                "project_id": project_setup["project_id"],
                "data_file": "data.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["openai.gpt-5.5"],
            })

        preflight_mock.assert_awaited_once()
        kwargs = preflight_mock.await_args.kwargs
        assert "openai.gpt-5.5" in kwargs["descriptors"]
        assert kwargs["project_id"] == project_setup["project_id"]
        assert kwargs["project_dir"] == project_setup["project_dir"]

    async def test_dispatch_optimization_threads_descriptors_through(
        self, project_setup
    ):
        from app.services.model_descriptors import write_descriptors
        write_descriptors(
            project_setup["project_dir"],
            {
                "openai.gpt-5.5": {
                    "endpoint": "mantle",
                    "api": "responses",
                    "region": "us-east-1",
                }
            },
        )

        from app.models.schemas import HarnessConfig
        harness = HarnessConfig(
            models=["openai.gpt-5.5"],
            judge_model_id="judge-x",
            metrics=[{"name": "ExactMatch", "type": "exact_match"}],
            field_mappings={"input": "input", "expected_output": "expected"},
            data_file="data.csv",
            target_metric_scores={"ExactMatch": 0.85},
            max_iterations=10,
            plateau_window=3,
        )
        preflight_mock = AsyncMock(return_value=[])
        with patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve, \
             patch(
                 "app.services.mcp_tools.dispatch.load_harness_config",
                 return_value=harness,
             ), patch(
                 "app.services.mcp_tools.dispatch._preflight_models_invocable",
                 preflight_mock,
             ), patch(
                 "app.services.optimization_loop_orchestrator.run_optimization_loop",
                 new_callable=AsyncMock,
             ):
            mock_resolve.return_value = project_setup["project_dir"]
            await _dispatch_optimization({
                "project_id": project_setup["project_id"],
            })

        preflight_mock.assert_awaited_once()
        kwargs = preflight_mock.await_args.kwargs
        assert "openai.gpt-5.5" in kwargs["descriptors"]
        assert kwargs["project_id"] == project_setup["project_id"]
        assert kwargs["project_dir"] == project_setup["project_dir"]
