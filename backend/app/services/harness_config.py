# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AST-based parser for harness config.py files.

Extracts structured configuration from the EvalConfig dataclass without
executing the code. Handles plain literals, field(default=...), and
field(default_factory=lambda: ...) patterns from dataclasses.
"""

import ast
from pathlib import Path

from app.models.schemas import HarnessConfig


def parse_harness_config(source_code: str) -> HarnessConfig:
    """Parse a harness config.py file using the ast module to extract config fields.

    Looks for class definitions (dataclasses or plain classes) and module-level
    assignments to extract known configuration fields like models,
    data_file, judge_model_id, prompt_version, field mappings, and optimization
    parameters.

    Args:
        source_code: Raw Python source code of the config.py file.

    Returns:
        HarnessConfig with extracted field values. Missing fields use defaults.
    """
    tree = ast.parse(source_code)
    config_data: dict = {}

    def _extract_literal(node: ast.expr):
        """Attempt to extract a Python literal value from an AST node.

        Handles plain literals (str, int, list, dict) and also
        field(default_factory=lambda: <literal>) from dataclasses.
        """
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            pass

        # Handle field(default=<literal>) and field(default_factory=lambda: <literal>)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "field":
            for kw in node.keywords:
                if kw.arg == "default":
                    try:
                        return ast.literal_eval(kw.value)
                    except (ValueError, TypeError):
                        pass
                elif kw.arg == "default_factory":
                    # field(default_factory=lambda: [...])
                    if isinstance(kw.value, ast.Lambda):
                        try:
                            return ast.literal_eval(kw.value.body)
                        except (ValueError, TypeError):
                            pass
        return None

    # Search class bodies for field assignments
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                # Handle simple assignments: field = value
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            value = _extract_literal(item.value)
                            if value is not None:
                                config_data[target.id] = value
                # Handle annotated assignments: field: type = value
                elif isinstance(item, ast.AnnAssign) and item.value is not None:
                    if isinstance(item.target, ast.Name):
                        value = _extract_literal(item.value)
                        if value is not None:
                            config_data[item.target.id] = value

    # Also check module-level assignments as fallback
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id not in config_data:
                    value = _extract_literal(node.value)
                    if value is not None:
                        config_data[target.id] = value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name) and node.target.id not in config_data:
                value = _extract_literal(node.value)
                if value is not None:
                    config_data[node.target.id] = value

    return HarnessConfig(
        models=config_data.get("models", []),
        data_file=config_data.get("data_file"),
        judge_model_id=config_data.get("judge_model_id"),
        prompt_version=config_data.get("prompt_version"),
        field_mappings=config_data.get("field_mappings"),
        target_metric_scores=config_data.get("target_metric_scores"),
        max_iterations=config_data.get("max_iterations"),
        plateau_window=config_data.get("plateau_window"),
        plateau_threshold=config_data.get("plateau_threshold"),
    )


def load_harness_config(project_dir: Path) -> HarnessConfig | None:
    """Load and parse ``src/config.py`` under ``project_dir``.

    Returns ``None`` if the config file does not exist — callers decide
    whether that is an error or an expected "project not yet scaffolded"
    state. Parse failures still propagate as exceptions.

    Args:
        project_dir: Root directory of the eval project.

    Returns:
        Parsed HarnessConfig, or None if src/config.py is absent.
    """
    config_path = project_dir / "src" / "config.py"
    if not config_path.exists():
        return None
    return parse_harness_config(config_path.read_text())


