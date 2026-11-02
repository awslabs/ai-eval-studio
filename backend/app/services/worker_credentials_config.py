# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write AWS config for worker credential_process chain and return env mutations.

The worker must mint its own refreshable credentials through a
``credential_process`` so eval runs exceeding 1 hour stay authorized instead of
expiring mid-run. This module writes the AWS config file (under an ephemeral
/tmp workspace, not the read-only EFS) and returns the environment mutations
that make botocore resolve the refreshable chain:

1. Set ``AWS_CONFIG_FILE`` pointing at the written config.
2. REMOVE the three static credential keys (``AWS_ACCESS_KEY_ID`` /
   ``AWS_SECRET_ACCESS_KEY`` / ``AWS_SESSION_TOKEN``), which would otherwise
   shadow the credential-process provider (botocore resolves env credentials at
   chain position #1, credential_process at position #7).

**Why the static keys must be removed:** botocore's credential resolver chain
positions the ``Environment`` provider (#1) ahead of the ``ProcessProvider``
(#7). Leaving the static triple in the environment would shadow the refreshable
chain and re-freeze the credential at its initial values — the exact bug this
design fixes.

**Why the file form is used rather than an in-process provider:** The harness
subprocess runs as ``os.environ.copy()`` which inherits ``AWS_CONFIG_FILE`` and
re-resolves the chain itself. An in-process ``RefreshableCredentials`` built in
the worker parent would not cross the subprocess boundary.

**Subprocess invocation contract:** botocore's ``ProcessProvider`` spawns the
``credential_process`` command line as a subprocess under the **caller's
environment**. The caller is not necessarily the worker parent — it is
whichever process resolves the credential chain, which in practice is the
harness subprocess. That harness subprocess has a **different cwd** (a project
working tree), a **different PATH** (a project-local venv at
``/tmp/workspace/{job}/.venv/bin``), and an interpreter that cannot import
``app``. So a bare ``python`` resolves to the wrong interpreter and a ``-m``
module path resolves against the wrong ``sys.path`` — both fail credential
retrieval outright. Therefore **both** components are absolute: the interpreter
from ``sys.executable`` and the script from
``importlib.util.find_spec(...).origin``, each captured in the worker parent
where they are correct. Neither is a hardcoded deployment path.
``shlex.join`` is required because botocore re-parses the stored value with
``RawConfigParser`` then ``shlex.split``; naive interpolation breaks on any
path containing a space.

**Third-party dependency:** This rests on botocore returning a refreshable
credential when the credential_process helper emits an ``Expiration`` key. That
key is mapped to the parsed-dict key ``expiry_time``, and botocore's
``ProcessProvider.load`` branches on ``if creds_dict.get('expiry_time') is not
None: ... RefreshableCredentials.create_from_metadata(...)`` (verified against
the installed library). If the refresh test ever fails while an expiry IS being
emitted, that branch assumption has changed — the fix must then move to an
in-process ``RefreshableCredentials``, not be patched around with workarounds.

Exports:
    write_worker_credentials_config(*, owner_id, project_id, job_id, role_arn,
        config_dir) -> dict[str, str] | None: Write config and return env
        mutations, or None in dev mode (role_arn falsy).
"""
from __future__ import annotations

import importlib.util
import logging
import os
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..platform.types import JobId, ProjectId, UserId

logger = logging.getLogger(__name__)


def write_worker_credentials_config(
    *,
    owner_id: "UserId",
    project_id: "ProjectId",
    job_id: "JobId",
    role_arn: str | None,
    config_dir: str | Path,
) -> dict[str, str] | None:
    """Write AWS config with credential_process and return env mutations.

    Returns a dict of environment mutations to apply:
    - ``AWS_CONFIG_FILE``: path to the written config file
    - ``AWS_ACCESS_KEY_ID``: removal sentinel (empty string)
    - ``AWS_SECRET_ACCESS_KEY``: removal sentinel (empty string)
    - ``AWS_SESSION_TOKEN``: removal sentinel (empty string)

    The caller applies these mutations to ``os.environ`` so the parent process
    AND every subprocess it later spawns inherit the refreshable chain.

    Dev mode: when ``role_arn`` is falsy (the signal from
    ``TokenVending.mint() -> None``), returns None and writes no config. The
    environment is left untouched and ambient credentials still serve.

    Args:
        owner_id: Authenticated owner sub (uid tag for mint helper).
        project_id: Project ID (unused in worker path, but threaded for symmetry).
        job_id: Job ID (job tag for mint helper).
        role_arn: ARN of the scoped-session role to assume. When falsy, dev-mode
            signal — no config is written and None is returned.
        config_dir: Writable ephemeral directory for the config file (e.g.
            ``/tmp/workspace/{job_id}``). The worker's EFS mount is read-only,
            so this must be a writable /tmp area.

    Returns:
        Dict of environment mutations (AWS_CONFIG_FILE + three removal sentinels),
        or None in dev mode (role_arn falsy).
    """
    # Dev mode: no role ARN → no minted creds → no config needed
    if not role_arn:
        return None

    config_path = Path(config_dir) / "aws_config"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Derive both absolute paths in the worker parent where 'app' is importable
    # and sys.executable points at the correct interpreter. The harness
    # subprocess has a different cwd and PATH, so these must be pinned here.
    interpreter = sys.executable

    spec = importlib.util.find_spec("app.scripts.mint_worker_credentials")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Cannot locate app.scripts.mint_worker_credentials — "
            "the helper is missing or the package is not importable"
        )
    helper_script = spec.origin

    # Build the credential_process command with shlex.join so paths containing
    # spaces survive botocore's RawConfigParser + shlex.split round-trip. The
    # helper receives explicit CLI arguments (owner, project, job, role ARN)
    # rather than environment inheritance, so the call is self-contained and
    # auditable.
    cmd_parts = [
        interpreter,
        helper_script,
        "--owner-id", owner_id,
        "--project-id", project_id,
        "--job-id", job_id,
        "--role-arn", role_arn,
    ]
    credential_process_line = shlex.join(cmd_parts)

    config_content = f"""[default]
credential_process = {credential_process_line}
"""

    config_path.write_text(config_content, encoding="utf-8")

    logger.info(
        "Wrote worker credentials config",
        extra={"config_path": str(config_path), "job_id": job_id},
    )

    # Return env mutations: point AWS_CONFIG_FILE at the written file and
    # remove the three static credential keys (empty string is the removal
    # sentinel; the caller deletes or pops keys with empty values).
    return {
        "AWS_CONFIG_FILE": str(config_path),
        "AWS_ACCESS_KEY_ID": "",
        "AWS_SECRET_ACCESS_KEY": "",
        "AWS_SESSION_TOKEN": "",
    }


__all__ = ["write_worker_credentials_config"]
