# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async wrapper around ``model_descriptors.write_descriptors``.

Acquires a per-project lock via ``services.project_locks.get_project_lock``
and runs the synchronous write in a thread. Lives outside
``model_descriptors.py`` so the latter stays byte-identical with
``templates/harness/src/models/model_descriptors.py`` (the harness
template has no ``app.`` namespace and no project_locks module).

EVERY descriptor write from the coordinator
side must go through this function (or call ``write_descriptors``
inside a ``get_project_lock(project_id)`` context manager directly).
The writers are model discovery and the dispatch preflight, which
removes a descriptor that has gone stale.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.services.model_descriptors import MantleDescriptor, write_descriptors
from app.services.project_locks import get_project_lock


async def persist_descriptors_locked(
    project_dir: Path,
    descriptors: dict[str, MantleDescriptor],
    project_id: str,
) -> Path:
    """Acquire the project's async lock and write the descriptor file.

    Args:
        project_dir: Project root directory.
        descriptors: Map of ``model_id`` → ``MantleDescriptor``.
        project_id: The project's UUID — scopes the lock; concurrent
            calls for DIFFERENT projects do not block each other (per
            ``project_locks`` contract).

    Returns:
        Absolute path to the written file (forwarded from
        ``write_descriptors``).
    """
    lock = await get_project_lock(project_id)
    async with lock:
        return await asyncio.to_thread(write_descriptors, project_dir, descriptors)
