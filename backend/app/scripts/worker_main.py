# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker CLI entrypoint — invoked by Fargate RunTask container command override.

Reads ``JOB_ID`` / ``OWNER_ID`` / ``PROJECT_ID`` from ``os.environ``, calls
``asyncio.run(run_worker(job_id, owner_id, project_id))``, and ``sys.exit(rc)``.

Invoked by Fargate RunTask container command override:
``python -m app.scripts.worker_main``.
"""
import asyncio
import logging
import os
import sys

logger = logging.getLogger(__name__)


def main() -> None:
    """Worker CLI entrypoint."""
    # Read env vars
    job_id = os.environ.get("JOB_ID")
    owner_id = os.environ.get("OWNER_ID")
    project_id = os.environ.get("PROJECT_ID")
    role_arn = os.environ.get("SCOPED_SESSION_ROLE_ARN")

    if not job_id or not owner_id or not project_id:
        logger.error(
            "Missing required env vars: JOB_ID=%s, OWNER_ID=%s, PROJECT_ID=%s",
            job_id,
            owner_id,
            project_id,
        )
        sys.exit(1)

    # Write AWS config for credential_process chain BEFORE any AWS client is
    # built. This makes the refreshable credential provider available to the
    # parent process AND every harness child subprocess (which inherits
    # AWS_CONFIG_FILE and re-resolves the chain itself).
    from app.services.worker_credentials_config import write_worker_credentials_config

    config_dir = f"/tmp/workspace/{job_id}"
    mutations = write_worker_credentials_config(
        owner_id=owner_id,
        project_id=project_id,
        job_id=job_id,
        role_arn=role_arn,
        config_dir=config_dir,
    )

    # Apply mutations to os.environ so the parent process AND every subprocess
    # it later spawns inherit the refreshable chain. When a key maps to an
    # empty string, delete it from the environment (the removal sentinel).
    if mutations:
        for key, value in mutations.items():
            if value:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)

    # Run worker
    from app.services.worker_entrypoint import run_worker

    exit_code = asyncio.run(run_worker(job_id, owner_id, project_id))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
