# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshot S3 key derivation helper.

This module is the single source of job S3-key computation that launch,
worker, and coordinator MUST agree on. It covers BOTH directions of the
per-job transfer:

- the snapshot IN key (``snapshot_key_for_job``): the coordinator uploads
  the working-tree snapshot; the worker downloads it. Divergence between
  launch and worker keys 404s every eval.
- the output OUT key (``output_key_for_job``): a completed detached worker
  uploads its ``output/`` eval-results file; the coordinator downloads it
  to EFS at the shared detached-terminal point (the reverse-sync leg — the
  worker EFS mount is read-only). Both keys live under the same job-derived
  ``project_key_for_directory("/tmp/workspace/{job_id}")`` prefix.
"""

from claude_agent_sdk import project_key_for_directory


def snapshot_key_for_job(job_id: str) -> str:
    """Compute the S3 key for a job's working-tree snapshot.

    Args:
        job_id: The job ID (used for both workspace cwd and snapshot filename).

    Returns:
        The full S3 key path: "{canonical_prefix}/snapshots/{job_id}.tar.gz"

    The launch side (coordinator) uploads the snapshot to this key; the worker
    downloads from this key. Both compute the key from the same job_id, using
    the same workspace path (/tmp/workspace/{job_id}) so the tenant prefix is
    identical. Any drift between launch and worker keys breaks every eval.
    """
    # Worker workspace cwd is always /tmp/workspace/{job_id} (standard across launch + worker)
    worker_project_cwd = f"/tmp/workspace/{job_id}"
    canonical_prefix = project_key_for_directory(worker_project_cwd)
    return f"{canonical_prefix}/snapshots/{job_id}.tar.gz"


def output_key_for_job(job_id: str, run_filename: str) -> str:
    """Compute the S3 key for a completed job's uploaded ``output/`` result.

    Args:
        job_id: The job ID (its workspace cwd is ``/tmp/workspace/{job_id}``).
        run_filename: The eval-results filename (e.g. ``eval_results_v1.json``),
            NOT a path — the worker emits the bare filename in its terminal
            ``results`` event.

    Returns:
        The full S3 key: ``{canonical_prefix}/output/{run_filename}``.

    The worker (read-only EFS mount) uploads its completed results here; the
    coordinator downloads from this key and writes it to the tenant EFS tree.
    The canonical prefix is derived from the SAME workspace path as
    ``snapshot_key_for_job`` (``project_key_for_directory("/tmp/workspace/{job_id}")``
    → ``-tmp-workspace-{job_id}``), so the output key falls under the worker's
    existing ``-tmp-workspace-{job_id}/*`` ABAC S3 grant.
    """
    worker_project_cwd = f"/tmp/workspace/{job_id}"
    canonical_prefix = project_key_for_directory(worker_project_cwd)
    return f"{canonical_prefix}/output/{run_filename}"


__all__ = ["snapshot_key_for_job", "output_key_for_job"]
