# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-project Mantle model descriptor type and persistence helpers.

Each project carries a ``src/model_descriptors.json`` file mapping
``model_id`` → ``MantleDescriptor`` with the model's invocation routing
(``endpoint``, ``api``, ``region``), two optional capability flags
(``supports_temperature``, ``supports_system_prompt``) and an optional
discovery hint (``probe_status``). :class:`MantleDescriptor` explains why
the shape is kept small.

Reads are unsynchronized — the file is small and atomic-replaced on
write, so a torn read can't happen. Writes are synchronous + atomic
(``.tmp`` + ``os.replace``); the service-side caller is expected to wrap
the write in ``services.project_locks.get_project_lock(project_id)``.
The lock-acquiring async wrapper lives in
``services.model_descriptors_async`` (NOT mirrored to the harness
template), which keeps this module free of platform imports so it can be
duplicated byte for byte between ``backend/app/services/`` and
``backend/app/templates/harness/src/models/``;
``tests/test_templates_parity.py`` enforces the match.

Mirrors the ``bedrock_pricing.write_pricing_file`` precedent:
project-scoped JSON in ``projects/{guid}/src/``, written at
harness-build time (and at discovery time), read by the runner at
eval time. Stdlib + ``typing`` only — no ``app.`` imports anywhere.
"""

import json
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

# Re-exported here so the descriptor file's TypedDict can reference the
# canonical Literal aliases without depending on the mapper module.
MantleEndpointKind = Literal["runtime", "mantle"]
MantleAPIName = Literal[
    "converse", "responses", "chat_completions", "anthropic_messages"
]
# Permissive-discovery probe status. ``"ok"`` is the cheap default and
# is OMITTED from the descriptor (the absence-of-flag is the success
# case, keeping the file lean). ``"unverified"`` covers transient probe
# failures (timeout, schema drift, 5xx, network) — the api/region was
# resolved on best-effort grounds. ``"access_not_enabled"`` covers
# 401/403 entitlement blocks: the model is reachable but the account
# can't currently invoke it; the user sees the model with a hint.
MantleProbeStatus = Literal["ok", "unverified", "access_not_enabled"]


class MantleDescriptor(TypedDict):
    """Per-model invocation descriptor.

    The shape is intentionally small: three required routing fields, two
    optional capability flags and one optional discovery hint. The
    harness runner, the discovery layer and eval dispatch all read this
    file, so a new field (for example a tool-use capability flag) should
    be a deliberate addition made in both copies of this module, not an
    ad-hoc extra key.

    Required:
      * ``endpoint``: which adapter family — ``"runtime"`` for the
        ``bedrock-runtime`` Converse path, ``"mantle"`` for the
        ``bedrock-mantle.{region}.api.aws`` path.
      * ``api``: which mapper to dispatch on the mantle path
        (``responses``, ``chat_completions``, ``anthropic_messages``);
        ``converse`` is the value paired with the ``runtime`` endpoint.
      * ``region``: the SINGLE source of truth for SigV4 signing region
        AND the URL host's region segment (never accept
        region as a separate kwarg in the adapter).

    Optional:
      * ``supports_temperature``: defaults True; set False after the
        temperature self-correction caches a reasoning model's rejection
        of the parameter.
      * ``supports_system_prompt``: defaults True; observed True for
        all three mantle APIs.
      * ``probe_status``: permissive-discovery hint. Absent = clean
        probe (the default). ``"unverified"`` = api/region resolved on
        best-effort after a transient failure. ``"access_not_enabled"``
        = model reachable but account entitlement blocks invocation.
        UI surfaces the flag so the user knows why a kept model might
        not work yet.
    """

    endpoint: MantleEndpointKind
    api: MantleAPIName
    region: str
    supports_temperature: NotRequired[bool]
    supports_system_prompt: NotRequired[bool]
    probe_status: NotRequired[MantleProbeStatus]


# Path of the descriptor file relative to the project root. Read by the
# adapter at instantiation; written by the coordinator (harness-build
# time) and by the discovery layer (after probing a model's API).
DESCRIPTOR_FILE_RELATIVE = Path("src") / "model_descriptors.json"


def read_descriptors(project_dir: Path) -> dict[str, MantleDescriptor]:
    """Read the descriptor map from ``{project_dir}/src/model_descriptors.json``.

    Returns an empty dict if the file is absent (project predates
    mantle support or has only runtime models). On a JSON parse error,
    propagates ``json.JSONDecodeError`` — malformed file blocks the
    coordinator turn cleanly via the caller's error path. Do not
    auto-repair: a malformed file is surfaced to the caller, never
    silently rewritten or migrated to a guessed shape.

    Args:
        project_dir: Project root directory.

    Returns:
        Dict mapping ``model_id`` → ``MantleDescriptor``; empty if no
        descriptor file exists.
    """
    path = project_dir / DESCRIPTOR_FILE_RELATIVE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_descriptors(
    project_dir: Path,
    descriptors: dict[str, MantleDescriptor],
) -> Path:
    """Write the descriptor map atomically to ``{project_dir}/src/model_descriptors.json``.

    Synchronous; DOES NOT acquire a project lock. Service-side callers
    MUST wrap this in ``services.project_locks.get_project_lock(project_id)``
    — see ``services.model_descriptors_async.persist_descriptors_locked``.
    Template-side (harness subprocess) callers don't write descriptors
    at eval time — descriptors are read-only in the harness — so the
    lockless contract is fine for the mirror.

    Atomic-write pattern: write to ``.tmp``, then ``os.replace`` to the
    final path. Same shape as ``bedrock_pricing.write_pricing_file`` but
    one step further; this file is the runner's input, and a partial
    read corrupts an eval, so the rename is mandatory.

    Args:
        project_dir: Project root directory; ``src/`` is created if
            absent.
        descriptors: Dict mapping ``model_id`` → ``MantleDescriptor``.

    Returns:
        Absolute path to the written file.
    """
    path = project_dir / DESCRIPTOR_FILE_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(descriptors, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path
