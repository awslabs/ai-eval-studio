# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the reverse-sync leg (worker output/ → S3 → coordinator EFS).

Behavioral, moto-S3 round-trips (mirroring tests/test_event_log.py's mock_aws
pattern). The reverse leg exists because the detached worker's EFS mount is
read-only, so a completed eval's results travel worker→S3→coordinator→EFS.

Trips each guard:
* upload_run_output_to_s3 puts eval_results_*.json under output_key_for_job
  (get_object round-trips the bytes back).
* download_run_output_to_efs writes the object to a dest output dir + returns it.
* download raises RunOutputNotAvailable when the key never appears (ceiling).
* download rejects a traversal run_filename via safe_basename_path.
* load_materialized_results normalizes a dict→[dict] and raises on empty/non-list.
"""
from __future__ import annotations

import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from app.services.run_output_snapshot import (
    RunOutputNotAvailable,
    download_run_output_to_efs,
    load_materialized_results,
    upload_run_output_to_s3,
)
from app.services.snapshot_key import output_key_for_job

_BUCKET = "test-snapshots-bucket"
_JOB = "job-abc"


@pytest.mark.asyncio
async def test_upload_run_output_puts_under_output_key(tmp_path: Path):
    """upload_run_output_to_s3 puts the results file under output_key_for_job."""
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=_BUCKET)

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        results = [{"model_id": "m1", "pass_rate": 0.5}]
        (output_dir / "eval_results_v1_20260101.json").write_text(json.dumps(results))
        # A non-matching file must be ignored.
        (output_dir / "notes.txt").write_text("ignore me")

        uploaded = await upload_run_output_to_s3(output_dir, s3, _BUCKET, _JOB)

        assert uploaded == ["eval_results_v1_20260101.json"]
        # The object round-trips at the expected key.
        key = output_key_for_job(_JOB, "eval_results_v1_20260101.json")
        body = s3.get_object(Bucket=_BUCKET, Key=key)["Body"].read()
        assert json.loads(body) == results


@pytest.mark.asyncio
async def test_upload_run_output_empty_when_no_results_file(tmp_path: Path):
    """No eval_results_*.json in output_dir → nothing uploaded."""
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=_BUCKET)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        uploaded = await upload_run_output_to_s3(output_dir, s3, _BUCKET, _JOB)
        assert uploaded == []


@pytest.mark.asyncio
async def test_download_run_output_writes_efs_and_returns_path(tmp_path: Path):
    """download_run_output_to_efs writes the object to dest and returns the path."""
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=_BUCKET)
        results = [{"model_id": "m1", "pass_rate": 0.9}]
        run_filename = "eval_results_v2_20260101.json"
        s3.put_object(
            Bucket=_BUCKET,
            Key=output_key_for_job(_JOB, run_filename),
            Body=json.dumps(results).encode(),
        )

        dest = tmp_path / "coord" / "output"
        path = await download_run_output_to_efs(
            s3, _BUCKET, _JOB, run_filename, dest, max_attempts=2, delay_s=0
        )

        assert path == dest / run_filename
        assert json.loads(path.read_text()) == results


@pytest.mark.asyncio
async def test_download_run_output_ceiling_raises(tmp_path: Path):
    """Key never appears within max_attempts → RunOutputNotAvailable."""
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=_BUCKET)
        dest = tmp_path / "output"

        with pytest.raises(RunOutputNotAvailable):
            await download_run_output_to_efs(
                s3, _BUCKET, _JOB, "eval_results_missing.json", dest,
                max_attempts=2, delay_s=0,
            )


@pytest.mark.asyncio
async def test_download_run_output_rejects_traversal(tmp_path: Path):
    """A traversal run_filename is rejected via safe_basename_path (no write)."""
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=_BUCKET)
        dest = tmp_path / "output"

        with pytest.raises(RunOutputNotAvailable):
            await download_run_output_to_efs(
                s3, _BUCKET, _JOB, "../../etc/passwd", dest,
                max_attempts=2, delay_s=0,
            )
        # Nothing was written outside output/.
        assert not (tmp_path / "etc").exists()


def test_load_materialized_results_normalizes_dict(tmp_path: Path):
    """A dict-shaped file is normalized to [dict] (mirrors _load_harness_results)."""
    path = tmp_path / "eval_results_v1.json"
    path.write_text(json.dumps({"model_id": "m1", "pass_rate": 0.5}))

    results = load_materialized_results(path)
    assert results == [{"model_id": "m1", "pass_rate": 0.5}]


def test_load_materialized_results_passes_list(tmp_path: Path):
    """A list-shaped file passes through unchanged."""
    payload = [{"model_id": "m1"}, {"model_id": "m2"}]
    path = tmp_path / "eval_results_v1.json"
    path.write_text(json.dumps(payload))

    assert load_materialized_results(path) == payload


def test_load_materialized_results_raises_on_empty(tmp_path: Path):
    """An empty list raises RunOutputNotAvailable (consistent with ceiling-fail)."""
    path = tmp_path / "eval_results_v1.json"
    path.write_text(json.dumps([]))

    with pytest.raises(RunOutputNotAvailable):
        load_materialized_results(path)
