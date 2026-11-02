# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the PreToolUse project-dir jail.

The jail rejects Write/Edit/NotebookEdit whose resolved target falls outside
``project_dir``. It is defence-in-depth on top of the SDK's ``cwd`` setting
because ``permission_mode="bypassPermissions"`` + a sandbox-disabled Fargate task
leave the OS user as the only filesystem boundary.

Bash is deliberately NOT jailed — see ``test_bash_is_not_jailed`` below and the
rationale on the hook itself: a static scan of a command line cannot contain
shell escapes, so jailing it would assert a guarantee it does not deliver. This
docstring used to list Bash among the jailed tools, contradicting the test two
screens down.

Logic is exercised through ``_pre_tool_use_jail_decision`` — the pure
helper the closure-based hook delegates to. That keeps the tests
fast and side-effect-free, and pins the contract regardless of
whether the SDK shape evolves around the closure.
"""

from __future__ import annotations

import os
import pytest

from app.services.agent_runner import _pre_tool_use_jail_decision


@pytest.fixture
def project_dir(tmp_path):
    p = tmp_path / "projects" / "abc123"
    p.mkdir(parents=True)
    return p


def test_write_inside_project_allowed(project_dir):
    decision = _pre_tool_use_jail_decision(
        "Write",
        {"file_path": str(project_dir / "prompts" / "v1.md")},
        project_dir,
    )
    assert decision is None


def test_write_outside_project_denied(project_dir):
    decision = _pre_tool_use_jail_decision(
        "Write",
        {"file_path": "/etc/hosts"},
        project_dir,
    )
    assert decision is not None
    out = decision["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    assert "outside" in out["permissionDecisionReason"]


def test_edit_outside_project_denied(project_dir):
    decision = _pre_tool_use_jail_decision(
        "Edit",
        {"file_path": "/workspace/backend/app/services/agent_runner.py"},
        project_dir,
    )
    assert decision is not None
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_relative_path_anchored_to_tool_cwd_inside_project(project_dir):
    """When the SDK reports tool_cwd=project_dir (the production case),
    a relative ``file_path`` must resolve inside the tree and be allowed.
    Without ``tool_cwd``, ``realpath`` on a relative path uses the host
    process cwd (the FastAPI backend) which is NOT project_dir — every
    legitimate relative write would be denied.
    """
    decision = _pre_tool_use_jail_decision(
        "Write",
        {"file_path": "prompts/v1.md"},
        project_dir,
        tool_cwd=str(project_dir),
    )
    assert decision is None, (
        "relative path under tool_cwd=project_dir must NOT be denied"
    )


def test_relative_path_anchored_to_tool_cwd_escaping_via_dotdot(project_dir):
    """``../escape.txt`` from tool_cwd=project_dir resolves above the
    project root and must be denied — the jail's containment check has
    to follow ``..`` segments structurally."""
    decision = _pre_tool_use_jail_decision(
        "Write",
        {"file_path": "../escape.txt"},
        project_dir,
        tool_cwd=str(project_dir),
    )
    assert decision is not None
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_relative_path_falls_back_to_project_dir_when_no_tool_cwd(project_dir):
    """The fallback path (no tool_cwd supplied) must still resolve
    relative paths against project_dir — never against the host process
    cwd. This matches production where ClaudeAgentOptions.cwd = project_dir."""
    decision = _pre_tool_use_jail_decision(
        "Write",
        {"file_path": "prompts/v1.md"},
        project_dir,
    )
    assert decision is None


def test_bash_is_not_jailed(project_dir):
    """Bash is intentionally NOT in _JAILED_TOOLS — a static command-line
    scan can't represent every shell escape (``~``, ``..``, ``$VAR``,
    ``$(...)``, glued redirects ``>>/etc/x``, brace expansion). Static
    enforcement produced false denies on legitimate commands AND false
    allows on trivial bypasses, so Bash relies on the SDK's cwd
    constraint plus the trusted-dev-loop guarantee."""
    decision = _pre_tool_use_jail_decision(
        "Bash",
        {"command": "cat /etc/passwd"},
        project_dir,
    )
    assert decision is None


def test_read_tool_not_jailed(project_dir):
    """The jail only fires on Write/Edit/NotebookEdit. Read, LS, Glob, Grep,
    Bash, MCP tools etc. pass through untouched — MCP tools enforce project_id
    authz themselves; Read is intentionally unrestricted in dev so the agent can
    inspect repo source for debugging."""
    decision = _pre_tool_use_jail_decision(
        "Read",
        {"file_path": "/etc/passwd"},
        project_dir,
    )
    assert decision is None


def test_notebook_edit_outside_project_denied(project_dir):
    decision = _pre_tool_use_jail_decision(
        "NotebookEdit",
        {"notebook_path": "/tmp/foo.ipynb"},
        project_dir,
    )
    assert decision is not None
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_symlink_target_resolves(project_dir, tmp_path):
    """Symlinks pointing OUT of project_dir are caught — ``realpath``
    chases the link before the prefix check, so the jail can't be
    bypassed by symlinking ``data/raw/X`` to ``/etc/passwd`` inside
    project_dir."""
    outside = tmp_path / "secrets.txt"
    outside.write_text("nope")
    link_dir = project_dir / "data" / "raw"
    link_dir.mkdir(parents=True, exist_ok=True)
    os.symlink(outside, link_dir / "evil")
    link = link_dir
    decision = _pre_tool_use_jail_decision(
        "Write",
        {"file_path": str(link / "evil")},
        project_dir,
    )
    assert decision is not None
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"
