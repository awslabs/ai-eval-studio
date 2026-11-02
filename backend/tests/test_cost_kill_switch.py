# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for the cost kill-switch flag.

The cost kill-switch is a resolve-once, env-backed operator override that halts
new eval launches (dispatch checks it) AND authorizes the reconciler's
aggressive StopTask-all mode. It reads from the process environment
(never from a request/tenant), so no tenant can trip another tenant's switch.

This suite tests:
- Kill-switch resolution from env (truthy = "1"/"true", unset = inactive)
- Accessor returns the resolved flag (operator-scoped, zero args)
- Case-insensitive parsing ("True", "TRUE", "1" all resolve active)
- Unset/empty/falsey env → inactive
"""

import os
from unittest.mock import patch

import pytest


def test_kill_switch_unset_resolves_inactive():
    """When EVAL_STUDIO_COST_KILL_SWITCH is unset, is_cost_kill_switch_active() returns False."""
    # Clear env to ensure it's unset
    with patch.dict(os.environ, {}, clear=False):
        if "EVAL_STUDIO_COST_KILL_SWITCH" in os.environ:
            del os.environ["EVAL_STUDIO_COST_KILL_SWITCH"]

        # Import module after env cleared so it resolves at import time
        import importlib
        import app.platform.arch_split as arch_split_module
        importlib.reload(arch_split_module)

        from app.platform.arch_split import is_cost_kill_switch_active

        assert is_cost_kill_switch_active() is False


def test_kill_switch_truthy_resolves_active():
    """When EVAL_STUDIO_COST_KILL_SWITCH is "1" or "true", is_cost_kill_switch_active() returns True."""
    # Test "1"
    with patch.dict(os.environ, {"EVAL_STUDIO_COST_KILL_SWITCH": "1"}):
        import importlib
        import app.platform.arch_split as arch_split_module
        importlib.reload(arch_split_module)

        from app.platform.arch_split import is_cost_kill_switch_active

        assert is_cost_kill_switch_active() is True

    # Test "true" (lowercase)
    with patch.dict(os.environ, {"EVAL_STUDIO_COST_KILL_SWITCH": "true"}):
        importlib.reload(arch_split_module)
        assert is_cost_kill_switch_active() is True


def test_kill_switch_case_insensitive():
    """Kill-switch env var parsing is case-insensitive ("True", "TRUE", "TrUe" all active)."""
    test_cases = ["True", "TRUE", "TrUe"]

    for value in test_cases:
        with patch.dict(os.environ, {"EVAL_STUDIO_COST_KILL_SWITCH": value}):
            import importlib
            import app.platform.arch_split as arch_split_module
            importlib.reload(arch_split_module)

            from app.platform.arch_split import is_cost_kill_switch_active

            assert is_cost_kill_switch_active() is True, f"Expected True for value '{value}'"


def test_kill_switch_falsey_values_resolve_inactive():
    """Empty, "0", "no", or other non-truthy values resolve to inactive."""
    falsey_values = ["", "0", "no", "false", "nope"]

    for value in falsey_values:
        with patch.dict(os.environ, {"EVAL_STUDIO_COST_KILL_SWITCH": value}):
            import importlib
            import app.platform.arch_split as arch_split_module
            importlib.reload(arch_split_module)

            from app.platform.arch_split import is_cost_kill_switch_active

            assert is_cost_kill_switch_active() is False, f"Expected False for value '{value}'"


def test_kill_switch_accessor_takes_no_args():
    """The kill-switch accessor is operator-scoped (no tenant/request argument)."""
    # Monkeypatch the module constant to True
    import app.platform.arch_split as arch_split_module
    original_value = getattr(arch_split_module, "COST_KILL_SWITCH_ENABLED", False)

    try:
        # Set the module constant to True
        arch_split_module.COST_KILL_SWITCH_ENABLED = True

        from app.platform.arch_split import is_cost_kill_switch_active

        # Assert: accessor takes no args and returns the module constant
        result = is_cost_kill_switch_active()
        assert result is True

        # Flip the constant and verify accessor reflects it
        arch_split_module.COST_KILL_SWITCH_ENABLED = False
        assert is_cost_kill_switch_active() is False
    finally:
        # Restore original value
        arch_split_module.COST_KILL_SWITCH_ENABLED = original_value


def test_kill_switch_module_constant_exists():
    """The module exports COST_KILL_SWITCH_ENABLED as a module-level constant."""
    from app.platform import arch_split

    # Assert the constant is defined
    assert hasattr(arch_split, "COST_KILL_SWITCH_ENABLED")

    # Assert it's a boolean
    value = getattr(arch_split, "COST_KILL_SWITCH_ENABLED")
    assert isinstance(value, bool), f"Expected bool, got {type(value)}"
