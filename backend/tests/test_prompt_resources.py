# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for packaged prompt resource resolution.

The defect: optimization_loop_orchestrator._run_prompt_optimizer derived its own
prompts_dir via Path(__file__).resolve().parent.parent.parent / "prompts". When the
function was moved from app/services/mcp_tools/dispatch.py (depth 3) to
app/services/optimization_loop_orchestrator.py (depth 2), the same arithmetic landed
on backend/prompts instead of backend/app/prompts, causing FileNotFoundError at
runtime. That derivation was one of seven — six correct, one drifted.

The failure mode test_coordinator_prompt.py could not see: it asserted prompt-optimizer.md
exists by resolving through app.services.coordinator_prompt.PROMPTS_DIR, a sibling
module whose arithmetic was correct. It proved the FILE exists and never exercised the
orchestrator's own resolution. Wrong unit tested.

This file asserts the resolution used by the actual failing call site, covering four
surfaces: the orchestrator's call-site effect (did the real prompt content reach the
SDK option?), anchor identity (is the orchestrator using the shared anchor?), drift
guard (no second hand-rolled derivation exists), and boot-check behavior (missing
resources fail loudly at startup).
"""

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import re

import pytest

# Expected first line of prompt-optimizer.md, verbatim. This is a module constant
# rather than a re-read so that, if the shared anchor ever pointed one directory
# up, the failure surfaces in the assertion instead of as a muddy error in the
# test's own setup.
_OPTIMIZER_PROMPT_FIRST_LINE = "You are a prompt optimization specialist for AI Eval Studio."


class TestCallSiteEffect:
    """Verify the orchestrator reads the real prompt content through the shared anchor."""

    async def test_orchestrator_reads_prompt_optimizer_through_anchor(self, tmp_path: Path) -> None:
        """Assert the real prompt content reached ClaudeAgentOptions.

        This is the centrepiece test. It exercises _run_prompt_optimizer with the Agent SDK
        stubbed and asserts that the real content of prompt-optimizer.md reached the
        ClaudeAgentOptions(system_prompt=...) kwarg.

        The function has a broad except Exception that swallows read failures. When the
        read raises (e.g., the file is not found), ClaudeAgentOptions is never
        constructed. We assert first that call_args is not None (proving the option was
        constructed), then that the system_prompt kwarg contains the expected literal.
        """
        # Mock the Agent SDK. Follow the precedent in tests/test_agent_service.py:
        # a MagicMock module injected via patch.dict with ClaudeSDKClient returning
        # an AsyncMock whose __aenter__/__aexit__ are AsyncMocks and whose
        # receive_response returns an exhausted async generator.
        mock_sdk = MagicMock()
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)

        async def _exhausted_async_gen():
            return
            yield  # pragma: no cover - unreachable

        mock_client_instance.receive_response = AsyncMock(return_value=_exhausted_async_gen())
        mock_sdk.ClaudeSDKClient = MagicMock(return_value=mock_client_instance)
        mock_sdk.ClaudeAgentOptions = MagicMock()

        # Patch the function-local imports at their defining modules. The orchestrator
        # imports these inside _run_prompt_optimizer, after the prompts_dir derivation.
        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
            with patch("app.services.mcp_server_cache.get_eval_mcp_server") as mock_mcp:
                with patch("app.services.project_manager._is_cloud_mode", return_value=False):
                    with patch("app.main.get_project_manager") as mock_pm_getter:
                        with patch(
                            "app.services.optimization_orchestrator.detect_latest_prompt",
                            return_value=("baseline.md", "baseline"),
                        ):
                            mock_mcp.return_value = MagicMock()

                            # Return a mock project with agent_model attribute
                            mock_project = MagicMock()
                            mock_project.agent_model = "opus"
                            mock_pm = MagicMock()
                            mock_pm.get_project.return_value = mock_project
                            mock_pm_getter.return_value = mock_pm

                            # Build minimal job_result carrying the fields the orchestrator reads
                            job_result = {
                                "passed_cases": 10,
                                "total_cases": 20,
                                "metrics_summary": {},
                                "run_output_file": "output.json",
                                "per_model_pass_rates": {},
                            }

                            # Import and call the function under test
                            from app.services.optimization_loop_orchestrator import (
                                _run_prompt_optimizer,
                            )

                            await _run_prompt_optimizer(
                                project_id="test-proj",
                                project_dir=tmp_path,
                                iteration=1,
                                prompt_version="baseline",
                                job_result=job_result,
                            )

        # Assert ClaudeAgentOptions was called. If the read raised and was swallowed,
        # the option was never constructed and call_args will be None.
        assert (
            mock_sdk.ClaudeAgentOptions.call_args is not None
        ), "ClaudeAgentOptions was never called — likely a swallowed read failure"

        # Assert the system_prompt kwarg contains the expected first line verbatim.
        call_kwargs = mock_sdk.ClaudeAgentOptions.call_args[1]
        system_prompt = call_kwargs.get("system_prompt", "")
        assert _OPTIMIZER_PROMPT_FIRST_LINE in system_prompt, (
            f"Expected prompt first line not found in system_prompt. "
            f"Got: {system_prompt[:200]!r}"
        )


class TestAnchorIdentity:
    """Verify the orchestrator uses the shared anchor, not a local derivation."""

    def test_prompts_dir_is_package_directory(self) -> None:
        """The anchor is the package's own directory (zero parent hops)."""
        from app.prompts import PROMPTS_DIR

        # PROMPTS_DIR must equal the directory containing app/prompts/__init__.py
        import app.prompts

        package_dir = Path(app.prompts.__file__).resolve().parent
        assert PROMPTS_DIR == package_dir, (
            f"PROMPTS_DIR should be the package directory. "
            f"Expected {package_dir}, got {PROMPTS_DIR}"
        )

    def test_prompt_optimizer_exists_under_anchor(self) -> None:
        """prompt-optimizer.md must exist under the anchor."""
        from app.prompts import PROMPTS_DIR

        optimizer_path = PROMPTS_DIR / "prompt-optimizer.md"
        assert optimizer_path.exists(), f"prompt-optimizer.md not found at {optimizer_path}"

    def test_orchestrator_imports_shared_anchor(self) -> None:
        """The orchestrator module binds PROMPTS_DIR from app.prompts."""
        from app.prompts import PROMPTS_DIR as anchor_value
        from app.services.optimization_loop_orchestrator import PROMPTS_DIR as orch_value

        assert orch_value == anchor_value, (
            f"Orchestrator should import the anchor, not derive its own. "
            f"Expected {anchor_value}, got {orch_value}"
        )


class TestDriftGuard:
    """Prevent a second hand-rolled prompts-directory derivation from landing in the tree."""

    def test_no_file_derived_prompts_paths_in_backend_app(self) -> None:
        """Assert no module in backend/app/ derives a prompts directory from __file__.

        This guard prevents the defect class described in the module docstring: a
        caller with a hand-rolled parent chain silently breaks when it is moved.

        The pattern is built from non-literal composition so this test file's own
        source cannot match a future sweep of the tree.
        """
        from app.prompts import PROMPTS_DIR

        # Derive the backend/app/ tree from the anchor's location
        app_tree_root = PROMPTS_DIR.parent

        # Build the pattern from components so this file's source doesn't match it
        path_call = r"Path\("
        dunder_file = r"__file__"
        literal_prompts = r'"prompts"'
        pattern = path_call + dunder_file + r"\).*" + literal_prompts

        offenders = []
        for py_file in app_tree_root.rglob("*.py"):
            with open(py_file, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, start=1):
                    if re.search(pattern, line):
                        offenders.append((py_file, lineno, line.strip()))

        assert not offenders, (
            f"Found {len(offenders)} __file__-derived prompts path(s) in backend/app/. "
            f"A second derivation is the defect. Use 'from app.prompts import PROMPTS_DIR' instead.\n"
            + "\n".join(f"{fpath}:{ln}: {text}" for fpath, ln, text in offenders)
        )


class TestBootCheckBehavior:
    """Assert verify_prompt_resources() fails loudly when a resource is missing."""

    def test_verify_passes_on_real_tree(self) -> None:
        """The check returns cleanly when all resources exist."""
        from app.prompts import verify_prompt_resources

        # Should not raise
        verify_prompt_resources()

    def test_verify_raises_with_diagnostic_on_missing_file(self, monkeypatch) -> None:
        """Assert RuntimeError names the resolved directory and missing files."""
        from app import prompts as prompts_module

        # Monkeypatch the filename tuple to include a bogus entry
        original = prompts_module._REQUIRED_PROMPT_FILES
        fake_tuple = original + ("nonexistent.md",)
        monkeypatch.setattr(prompts_module, "_REQUIRED_PROMPT_FILES", fake_tuple)

        with pytest.raises(RuntimeError) as exc_info:
            prompts_module.verify_prompt_resources()

        error_message = str(exc_info.value)
        # Assert the message names the bogus filename
        assert "nonexistent.md" in error_message, (
            f"Error message should name the missing file. Got: {error_message}"
        )
        # Assert the message names the resolved PROMPTS_DIR
        assert str(prompts_module.PROMPTS_DIR) in error_message, (
            f"Error message should name the resolved directory. Got: {error_message}"
        )

    def test_all_real_md_files_are_registered(self) -> None:
        """Assert every .md in PROMPTS_DIR is listed in the filename tuple.

        This catches the case where a new resource is added but not registered in
        the check, which would let the resource silently escape verification.
        """
        from app.prompts import PROMPTS_DIR, _REQUIRED_PROMPT_FILES

        actual_md_files = {p.name for p in PROMPTS_DIR.glob("*.md")}
        registered_files = set(_REQUIRED_PROMPT_FILES)

        missing_from_tuple = actual_md_files - registered_files
        assert not missing_from_tuple, (
            f"Found .md files in PROMPTS_DIR not registered in _REQUIRED_PROMPT_FILES: "
            f"{sorted(missing_from_tuple)}. Add them to the tuple."
        )
