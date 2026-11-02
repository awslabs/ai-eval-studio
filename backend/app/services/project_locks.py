# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-project asyncio locks for serializing read-then-write critical sections.

Several paths race when concurrent tasks touch the same project's files:

* ``save_prompt`` — two optimizer turns landing the same dedupe check +
  atomic-rename sequence would each happily write their temp file and the
  second ``os.replace`` would clobber the first.
* the durable chat transcript, ``set_models``'s ``src/config.py`` rewrite,
  report sidecars, and the model-descriptor file — each a read-modify-write
  on one file.

``OptimizationState.save`` is not serialized here: it has its own per-file
``threading.Lock``, and a version-conditional write on the durable path.

All of these are small critical sections (read-modify-write on a single file
in one Python process), so a per-project ``asyncio.Lock`` acquired inside
the operation is enough. We deliberately scope one lock per project id
to avoid serializing unrelated projects.

Scope it to per-project file critical sections like these; other locking
needs should use their own explicit lock rather than broadening this module.
"""

from __future__ import annotations

import asyncio
from typing import Dict

_locks: Dict[str, asyncio.Lock] = {}
_registry_lock = asyncio.Lock()


async def get_project_lock(project_id: str) -> asyncio.Lock:
    """Return the shared async lock for ``project_id``, creating it on first use.

    Locks are created lazily and kept for the process lifetime. A single
    registry mutex gates the dict mutation so two first-time callers
    never race to create two different Lock objects.

    Args:
        project_id: Project UUID the critical section belongs to.

    Returns:
        The ``asyncio.Lock`` scoped to this project id.
    """
    async with _registry_lock:
        lock = _locks.get(project_id)
        if lock is None:
            lock = asyncio.Lock()
            _locks[project_id] = lock
        return lock


def _reset() -> None:
    """Clear the lock registry. Test-only — prefixed with ``_``."""
    _locks.clear()
