# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two report-path defects: narrator facts-pack sampling and report durability.

1. ``_sample_failed_cases_for_narrator`` bounded the narrator's facts pack on the
   RUN path and silently did nothing on the EXECUTIVE path. ``RunFacts`` carries
   ``failed_cases`` at the top level; ``ExecFacts`` has none there and nests a
   whole ``RunFacts`` under ``winner_run``, so ``facts_dict.get("failed_cases")``
   was ``None``, the guard returned early, and the narrator got every row. With
   enough failures the executive narrator then dies with ``JSON message exceeded
   maximum buffer size of 1048576 bytes`` and recovers only once the failure
   count drops on its own.

2. Reports were written to EFS and nowhere else. The run file goes to S3 and the
   transcript goes to S3; the report — the artifact a user reads — had exactly
   one copy and no out-of-band way to fetch it.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.report_generation import (
    _NARRATOR_MAX_FAILED_CASES,
    _sample_failed_cases_for_narrator,
)
from app.services.report_snapshot import (
    delete_project_reports_from_s3,
    report_key_for_project,
    upload_report_to_s3,
)


#: Number of models the synthetic rows are spread across.
_MODEL_COUNT = 3


def _rows(n: int) -> list[dict]:
    """``n`` failed-case rows GROUPED BY MODEL, as production builds them.

    Grouping matters: ``_build_failed_cases`` walks model runs in order, so all
    of model 0's failures precede model 1's. Interleaving the models here would
    make a head slice indistinguishable from an even stride and the
    ``spans_models`` assertion below would pass for either.
    """
    per_model = max(1, n // _MODEL_COUNT)
    return [
        {
            "case_id": i,
            "model_id": f"m{min(i // per_model, _MODEL_COUNT - 1)}",
            "metric": "Answer Correctness",
            "input": "show urgent orders",
            "expected": "priority = 'Urgent'",
            "actual": "priority = 'High'",
            "reason": "x" * 400,
        }
        for i in range(n)
    ]


_OVER = _NARRATOR_MAX_FAILED_CASES * 4


class TestRunShapedFacts:
    """The path that always worked — kept so the fix can't regress it."""

    def test_an_oversized_run_pack_is_sampled(self):
        out = _sample_failed_cases_for_narrator({"failed_cases": _rows(_OVER)})
        assert len(out["failed_cases"]) == _NARRATOR_MAX_FAILED_CASES
        assert out["failed_cases_sampling"]["total"] == _OVER

    def test_a_small_run_pack_is_returned_untouched(self):
        facts = {"failed_cases": _rows(3)}
        assert _sample_failed_cases_for_narrator(facts) is facts


class TestExecShapedFacts:
    """THE REGRESSION. Exec facts nest their rows under ``winner_run``."""

    @staticmethod
    def _exec(n: int) -> dict:
        # No top-level failed_cases — this is what ExecFacts actually serializes
        # to, and the reason the old guard returned early.
        return {
            "winner_version": "v4",
            "journey": [{"version": "v1"}],
            "winner_run": {"run_filename": "r.json", "failed_cases": _rows(n)},
        }

    def test_nested_rows_are_sampled(self):
        """Unsampled, every nested row would reach the narrator and blow the frame limit."""
        out = _sample_failed_cases_for_narrator(self._exec(_OVER))
        assert len(out["winner_run"]["failed_cases"]) == _NARRATOR_MAX_FAILED_CASES

    def test_the_nested_sample_is_labelled_with_the_true_total(self):
        """The narrator is told to quote counts verbatim; an unlabelled
        truncation invites it to state a wrong failure count as fact."""
        out = _sample_failed_cases_for_narrator(self._exec(_OVER))
        sampling = out["winner_run"]["failed_cases_sampling"]
        assert sampling["shown"] == _NARRATOR_MAX_FAILED_CASES
        assert sampling["total"] == _OVER

    def test_the_callers_dict_is_not_mutated(self):
        """The same facts object is persisted for the UI, which keeps every row."""
        facts = self._exec(_OVER)
        _sample_failed_cases_for_narrator(facts)
        assert len(facts["winner_run"]["failed_cases"]) == _OVER

    def test_a_small_exec_pack_is_returned_untouched(self):
        facts = self._exec(3)
        assert _sample_failed_cases_for_narrator(facts) is facts

    def test_sampling_spans_models_rather_than_taking_a_head_slice(self):
        """Rows are grouped by model, so a head slice would describe one model
        while claiming to describe the run."""
        out = _sample_failed_cases_for_narrator(self._exec(_OVER))
        models = {r["model_id"] for r in out["winner_run"]["failed_cases"]}
        assert models == {f"m{i}" for i in range(_MODEL_COUNT)}, (
            f"sample covers only {sorted(models)} — a head slice would report "
            "one model's failure patterns as the whole run's"
        )

    def test_the_sampled_exec_pack_fits_under_the_sdk_frame_ceiling(self):
        """Ties the fix to the failure it exists for, end to end."""
        import json

        from app.services.sdk_subprocess_env import SDK_MAX_STDOUT_FRAME_BYTES

        out = _sample_failed_cases_for_narrator(self._exec(_OVER))
        assert len(json.dumps(out, indent=2).encode()) < SDK_MAX_STDOUT_FRAME_BYTES


class TestReportKey:
    """Keys must stay out of the session key space."""

    def test_the_key_is_project_scoped_and_uses_the_on_disk_filename(self):
        key = report_key_for_project("proj-1", "executive.report.json")
        assert key == "reports/proj-1/executive.report.json"

    def test_the_key_avoids_the_session_store_key_space(self):
        """``S3SessionStore.list_sessions`` paginates ``{project_key}/`` and
        treats each child as a session, so a report nested there would surface
        as a phantom chat session."""
        key = report_key_for_project("proj-1", "executive.report.json")
        assert not key.startswith("-mnt-projects")
        assert key.startswith("reports/")


@pytest.fixture
def report_file(tmp_path: Path) -> Path:
    p = tmp_path / "executive.report.json"
    p.write_text('{"scope": "executive"}', encoding="utf-8")
    return p


def _platform(store):
    return patch("app.platform.get_platform", return_value=MagicMock(session_store=store))


class TestUpload:
    @pytest.mark.asyncio
    async def test_dev_mode_is_a_no_op(self, report_file: Path):
        """No session store means no S3; reports live under the tree."""
        with _platform(None):
            assert await upload_report_to_s3("proj-1", report_file) is None

    @pytest.mark.asyncio
    async def test_the_report_is_put_at_the_project_scoped_key(self, report_file: Path):
        store = MagicMock(bucket="sessions-bucket")
        with _platform(store):
            key = await upload_report_to_s3("proj-1", report_file)
        assert key == "reports/proj-1/executive.report.json"
        kwargs = store.s3.put_object.call_args.kwargs
        assert kwargs["Bucket"] == "sessions-bucket"
        assert kwargs["Key"] == "reports/proj-1/executive.report.json"

    @pytest.mark.asyncio
    async def test_an_upload_failure_does_not_fail_the_report(self, report_file: Path):
        """The EFS copy is already committed — losing the backup must not lose
        the report the user asked for."""
        store = MagicMock(bucket="b")
        store.s3.put_object.side_effect = RuntimeError("AccessDenied")
        with _platform(store):
            assert await upload_report_to_s3("proj-1", report_file) is None

    @pytest.mark.asyncio
    async def test_an_upload_failure_is_logged_not_swallowed(self, report_file: Path):
        """A durable copy that quietly stopped happening is the bug this module
        was written to fix."""
        store = MagicMock(bucket="b")
        store.s3.put_object.side_effect = RuntimeError("AccessDenied")
        with _platform(store), patch(
            "app.services.report_snapshot.logger"
        ) as log:
            await upload_report_to_s3("proj-1", report_file)
        assert log.warning.called


class TestDeleteCascade:
    @pytest.mark.asyncio
    async def test_dev_mode_is_a_no_op(self):
        with _platform(None):
            assert await delete_project_reports_from_s3("proj-1") == 0

    @pytest.mark.asyncio
    async def test_every_report_object_for_the_project_is_deleted(self):
        store = MagicMock(bucket="b")
        store.s3.get_paginator.return_value.paginate.return_value = [
            {"Contents": [{"Key": "reports/proj-1/a.report.json"},
                          {"Key": "reports/proj-1/b.report.json"}]},
        ]
        with _platform(store):
            assert await delete_project_reports_from_s3("proj-1") == 2
        kwargs = store.s3.delete_objects.call_args.kwargs
        assert kwargs["Bucket"] == "b"
        assert len(kwargs["Delete"]["Objects"]) == 2

    @pytest.mark.asyncio
    async def test_an_empty_prefix_deletes_nothing(self):
        store = MagicMock(bucket="b")
        store.s3.get_paginator.return_value.paginate.return_value = [{}]
        with _platform(store):
            assert await delete_project_reports_from_s3("proj-1") == 0
        assert not store.s3.delete_objects.called

    @pytest.mark.asyncio
    async def test_a_delete_failure_raises_so_the_delete_is_retried(self):
        """``_complete_project_deletion`` keeps the ``deleting`` marker and retries.
        Swallowing this would orphan tenant data in S3 forever — the opposite
        policy from upload, deliberately."""
        store = MagicMock(bucket="b")
        store.s3.get_paginator.side_effect = RuntimeError("throttled")
        with _platform(store), pytest.raises(RuntimeError):
            await delete_project_reports_from_s3("proj-1")


@pytest.mark.asyncio
async def test_the_delete_cascade_actually_runs_during_project_purge():
    """A durable copy with no cascade is a tenant-data leak on delete.

    Invokes the real ``_complete_project_deletion`` rather than grepping for the
    symbol: an earlier version of this test searched the source text and stayed
    green when the CALL was removed but the import left behind.
    """
    from app.api import projects as projects_api

    with patch(
        "app.services.report_snapshot.delete_project_reports_from_s3"
    ) as cascade, patch.object(
        projects_api, "_delete_session_transcript"
    ), patch.object(
        projects_api, "_get_pm", return_value=MagicMock()
    ), patch(
        "app.platform.get_platform", return_value=MagicMock()
    ):
        await projects_api._complete_project_deletion(
            "proj-1", owner_id=None, session_id=None
        )

    cascade.assert_awaited_once_with("proj-1")


def test_both_save_report_scopes_take_a_durable_copy():
    """The run scope and the executive scope each write a sidecar, so each needs
    an upload. Counting them together means adding a third scope without a
    durable copy fails here rather than in cloud six weeks later.
    """
    import ast

    source = (
        Path(__file__).resolve().parent.parent
        / "app" / "services" / "mcp_tools" / "reports.py"
    ).read_text()
    tree = ast.parse(source)

    def _calls(name: str) -> int:
        return sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        )

    writes = _calls("write_report_json")
    uploads = _calls("upload_report_to_s3")
    assert writes >= 2, "expected a sidecar write per scope; did reports.py change shape?"
    assert uploads == writes, (
        f"reports.py writes {writes} sidecars but takes {uploads} durable "
        "copies — a scope is writing to EFS only"
    )
