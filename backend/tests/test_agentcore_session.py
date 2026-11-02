# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AgentCore session mapping (runtimeSessionId <-> app session_id).

Verifies the minimal mapping needed for AgentCore session resume:
- Round-trip mapping record/resolve
- Transcript resume via project's persisted session_id
- Thread-safety under concurrent access
- Bounded registry (no unbounded growth)
"""
import threading
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from app.services.agentcore_session import (
    record_session_mapping,
    resume_session_id_for,
)


class TestSessionMapping:
    """Test the runtimeSessionId <-> app session_id mapping."""

    def test_record_and_resolve_round_trip(self):
        """Recording a mapping then resolving it (same project) returns the recorded session_id."""
        runtime_session_id = "rt-test-001"
        app_session_id = "app-session-123"
        project_id = "proj-456"

        record_session_mapping(runtime_session_id, app_session_id, project_id)
        resolved = resume_session_id_for(runtime_session_id, project_id)

        assert resolved == app_session_id

    def test_mapping_is_project_scoped_no_cross_tenant_resume(self):
        """A runtimeSessionId recorded for one project must NOT resolve for another.

        This is the regression guard: a client-influenced
        runtimeSessionId recorded under tenant A's project must not let a
        request scoped to tenant B's project resume tenant A's transcript.
        """
        runtime_session_id = "rt-shared-id"
        tenant_a_session = "tenant-a-secret-session"
        project_a = "proj-tenant-a"
        project_b = "proj-tenant-b"

        # Tenant A records a mapping under their project.
        record_session_mapping(runtime_session_id, tenant_a_session, project_a)

        # Tenant B supplies the SAME runtimeSessionId but is scoped to their
        # own project, which has no persisted session. The mapping must miss,
        # and the persisted-session fallback must return None — never tenant
        # A's session.
        with patch("app.main.get_project_manager") as mock_pm:
            mock_project_b = Mock()
            mock_project_b.session_id = None
            mock_pm.return_value.get_project.return_value = mock_project_b

            resolved_for_b = resume_session_id_for(runtime_session_id, project_b)

        assert resolved_for_b != tenant_a_session
        assert resolved_for_b is None

        # And tenant A still resolves their own mapping.
        assert resume_session_id_for(runtime_session_id, project_a) == tenant_a_session

    def test_resolve_unknown_runtime_session_with_no_persisted_session(self):
        """Resolving an unknown runtimeSessionId for a project with no session_id returns None."""
        runtime_session_id = "rt-unknown"
        project_id = "proj-no-session"

        # Mock project manager to return a project with no session_id
        with patch("app.main.get_project_manager") as mock_pm:
            mock_project = Mock()
            mock_project.session_id = None
            mock_pm.return_value.get_project.return_value = mock_project

            resolved = resume_session_id_for(runtime_session_id, project_id)

        assert resolved is None

    def test_resolve_unknown_runtime_session_with_persisted_session(self):
        """Resolving an unknown runtimeSessionId for a project WITH a persisted session_id returns that session_id."""
        runtime_session_id = "rt-unknown-2"
        project_id = "proj-with-session"
        persisted_session_id = "persisted-abc-789"

        # Mock project manager to return a project WITH a session_id
        with patch("app.main.get_project_manager") as mock_pm:
            mock_project = Mock()
            mock_project.session_id = persisted_session_id
            mock_pm.return_value.get_project.return_value = mock_project

            resolved = resume_session_id_for(runtime_session_id, project_id)

        # Should return the persisted session_id, proving resume will work
        assert resolved == persisted_session_id

    def test_repeated_resolve_is_stable(self):
        """Calling resolve multiple times for the same runtimeSessionId returns consistent results."""
        runtime_session_id = "rt-stable"
        app_session_id = "app-stable-999"
        project_id = "proj-stable"

        record_session_mapping(runtime_session_id, app_session_id, project_id)

        # Call resolve multiple times
        result1 = resume_session_id_for(runtime_session_id, project_id)
        result2 = resume_session_id_for(runtime_session_id, project_id)
        result3 = resume_session_id_for(runtime_session_id, project_id)

        assert result1 == app_session_id
        assert result2 == app_session_id
        assert result3 == app_session_id

    def test_thread_safe_concurrent_record_resolve(self):
        """Recording and resolving from multiple threads concurrently never raises."""
        project_id = "proj-concurrent"
        errors = []

        def worker(idx: int):
            try:
                rt_id = f"rt-thread-{idx}"
                app_id = f"app-thread-{idx}"
                record_session_mapping(rt_id, app_id, project_id)
                resolved = resume_session_id_for(rt_id, project_id)
                # Basic sanity: either we get our own mapping or None (if project has no session)
                assert resolved in (app_id, None)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread-safety violations: {errors}"

    def test_registry_stays_bounded(self):
        """Registry evicts oldest entries to prevent unbounded growth."""
        # Record many mappings (more than any reasonable cap)
        for i in range(200):
            record_session_mapping(f"rt-{i}", f"app-{i}", f"proj-{i}")

        # Check that the registry doesn't grow unbounded
        # (The implementation will define a cap; this test verifies eviction occurs)
        # We'll verify by checking internal state size in the implementation
        from app.services.agentcore_session import _get_registry_size

        size = _get_registry_size()
        assert size <= 100, f"Registry grew unbounded: {size} entries"
