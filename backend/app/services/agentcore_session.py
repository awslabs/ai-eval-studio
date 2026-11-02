# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentCore session mapping: runtimeSessionId <-> app session_id.

In-memory session mapping used by the AgentCore serving adapter
(``agentcore_runtime.py``) inside the microVM:
- Maps AgentCore's runtimeSessionId to the app's session_id for turn resume
- Falls back to the project's persisted session_id, so a turn resumes from
  the S3 transcript
- Thread-safe registry with bounded size (no unbounded growth)

The durable per-(user, project) registry and the one-turn-at-a-time claim live
in ``agentcore_session_registry.py``; the relay control plane uses that, not
this map.
"""
import logging
import threading
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum entries in the in-memory mapping. This path is single-session,
# not a durable store. Once AgentCore StopRuntimeSession fires, the mapping
# is no longer needed. Cap at 100 to bound memory.
_MAX_REGISTRY_SIZE = 100

# In-memory registry: (project_id, runtimeSessionId) -> app session_id.
# The key is COMPOSITE and project-scoped on purpose: runtimeSessionId is
# client-influenced (AgentCore injects it, but the /invocations payload can
# also carry it), so a bare runtimeSessionId key would let a caller resolve a
# session recorded under a DIFFERENT tenant's project. Scoping the key by
# project_id means a mapping recorded for project A can never be resolved for
# project B — cross-tenant transcript resume is structurally impossible here.
# OrderedDict for efficient LRU eviction (oldest keys evicted first).
_session_registry: OrderedDict[tuple[str, str], str] = OrderedDict()
_registry_lock = threading.Lock()


def record_session_mapping(
    runtime_session_id: str,
    app_session_id: str,
    project_id: str,
) -> None:
    """Record a mapping from (project_id, runtimeSessionId) to app session_id.

    Thread-safe. Evicts oldest entries when registry exceeds _MAX_REGISTRY_SIZE.

    Args:
        runtime_session_id: AgentCore's runtime session identifier
        app_session_id: The app's session_id for SDK resume
        project_id: Project UUID — scopes the key so a runtimeSessionId
            recorded for one tenant cannot be resolved for another
    """
    key = (project_id, runtime_session_id)
    with _registry_lock:
        # Remove if exists (to refresh position in OrderedDict)
        _session_registry.pop(key, None)

        # Add/update mapping
        _session_registry[key] = app_session_id

        # Evict oldest if we exceed cap
        while len(_session_registry) > _MAX_REGISTRY_SIZE:
            oldest_key = next(iter(_session_registry))
            evicted_value = _session_registry.pop(oldest_key)
            logger.debug(
                "Evicted oldest session mapping: %s -> %s (registry at %d entries)",
                oldest_key[1][:12],
                evicted_value[:12],
                len(_session_registry),
            )


def resume_session_id_for(
    runtime_session_id: str,
    project_id: str,
) -> Optional[str]:
    """Resolve the app session_id to resume for the given runtimeSessionId + project.

    Returns the app session_id from the mapping if present, otherwise returns
    the project's persisted session_id (if any). When AgentCore resumes a
    runtime session, the returned id is passed to ``start_agent_turn`` as
    ``session_id``, so the turn resumes from the S3 transcript.

    Thread-safe.

    Args:
        runtime_session_id: AgentCore's runtime session identifier
        project_id: Project UUID

    Returns:
        The app session_id to pass to ``start_agent_turn`` as ``session_id``,
        or None for a fresh session.
    """
    # First check the explicit mapping, scoped to this project so a
    # client-influenced runtimeSessionId cannot resolve another tenant's
    # session.
    with _registry_lock:
        mapped_session = _session_registry.get((project_id, runtime_session_id))

    if mapped_session is not None:
        return mapped_session

    # Not in mapping - check if project has a persisted session_id
    # (returning it means the turn resumes from the S3 transcript)
    try:
        from app.main import get_project_manager
        project = get_project_manager().get_project(project_id)
        if project is not None and project.session_id:
            logger.debug(
                "Resolved runtime_session %s -> persisted session %s for project %s",
                runtime_session_id[:12],
                project.session_id[:12],
                project_id[:8],
            )
            return project.session_id
    except Exception as e:
        logger.warning(
            "Failed to resolve persisted session_id for project %s: %s",
            project_id[:8],
            e,
        )

    # No mapping and no persisted session - fresh session
    return None


def _get_registry_size() -> int:
    """Return current registry size (for test assertions)."""
    with _registry_lock:
        return len(_session_registry)
