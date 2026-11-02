# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""One installer for the harness template's ``src.*`` import stubs.

The template's ``runner.py`` imports ``src.config``, ``src.metrics`` and
``src.models.*`` — a package tree that only exists inside a generated
project. Tests that load the template file by path stub that tree into
``sys.modules`` first, then exec the module.

This lives in ONE place because the stub tree had been hand-copied into five
test files and two copies had already drifted:

* one omitted three of the modules and passed only when another test file's
  stubs happened to still be in ``sys.modules`` — it failed when run alone;
* two replaced the real ``short_model_id`` with ``model_id.split(".")[-1]``,
  a different derivation than the harness ships, so those tests asserted
  against a fake.

The stubs are deliberately left in ``sys.modules`` after loading: the runner
imports ``src.models.model_id_short`` lazily inside ``_serialize_result``, so
popping them would break the very path several tests exercise. Because every
caller installs the same complete set, the leak is idempotent and no test
depends on which file ran first.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

#: Root of the packaged harness template (the thing a generated project copies).
TEMPLATE_ROOT = (
    Path(__file__).resolve().parent.parent / "app" / "templates" / "harness"
)


def install_src_stubs() -> dict[str, ModuleType]:
    """Install the ``src.*`` modules the template runner imports.

    Two stubs deliberately carry REAL implementations rather than mocks:

    * ``MantleReasoningTruncated`` — the runner's typed ``except`` matches by
      class identity, so a stub class would never catch the exception a test
      raises;
    * ``short_model_id`` — the ``test_run_id`` derivation the harness ships;
      a lookalike lambda makes every assertion about run ids meaningless.

    Both are re-used from the services-side copies, which template parity
    keeps byte-identical.

    Returns:
        The installed ``{module_name: module}`` map, so a caller can reach a
        stub directly (e.g. to swap ``read_descriptors``).
    """
    from app.services import bedrock_mantle_mappers as real_mappers
    from app.services.model_id_short import short_model_id as real_short_id

    config = ModuleType("src.config")
    config.EvalConfig = MagicMock()

    loader = ModuleType("src.evaluation.dataset_loader")
    loader.format_user_message = lambda *a, **k: ""
    loader.load_test_cases = lambda *a, **k: []

    metrics = ModuleType("src.metrics")
    metrics.build_metrics = lambda _j: []

    runtime_model = ModuleType("src.models.bedrock_runtime_model")
    runtime_model.BedrockRuntimeModel = MagicMock()

    mantle_mappers = ModuleType("src.models.bedrock_mantle_mappers")
    mantle_mappers.MantleReasoningTruncated = real_mappers.MantleReasoningTruncated

    # Mantle adapter: imported at module load for the ``_make_model`` factory.
    # Tests that exercise the mantle path patch it on the loaded runner.
    mantle_model = ModuleType("src.models.bedrock_mantle_model")
    mantle_model.BedrockMantleModel = MagicMock()

    # Descriptor reader: empty map = the runtime-only path. Tests that need a
    # populated map patch it on the loaded runner.
    descriptors = ModuleType("src.models.model_descriptors")
    descriptors.read_descriptors = lambda _project_dir: {}
    descriptors.MantleDescriptor = dict

    model_id_short = ModuleType("src.models.model_id_short")
    model_id_short.short_model_id = real_short_id

    stubs = {
        "src": ModuleType("src"),
        "src.config": config,
        "src.evaluation": ModuleType("src.evaluation"),
        "src.evaluation.dataset_loader": loader,
        "src.metrics": metrics,
        "src.models": ModuleType("src.models"),
        "src.models.bedrock_runtime_model": runtime_model,
        "src.models.bedrock_mantle_model": mantle_model,
        "src.models.bedrock_mantle_mappers": mantle_mappers,
        "src.models.model_descriptors": descriptors,
        "src.models.model_id_short": model_id_short,
    }
    sys.modules.update(stubs)
    return stubs


def load_template_bedrock_runtime_model() -> ModuleType:
    """Import the template's ``bedrock_runtime_model.py`` under a unique name.

    Loaded by path so the template code doesn't leak into the backend's
    import graph. The module carries a relative import
    (``from .aws_retry import _bedrock_retry``), so both files are
    registered under a synthetic ``harness_template_models`` package —
    that package context lets the relative import resolve without
    polluting ``sys.path`` or the scaffold's own ``src.*`` namespace.

    Re-execs on every call, so each test gets a module whose patches
    cannot leak into another test's copy.

    Returns:
        The loaded template adapter module, including its private helpers.
    """
    pkg_name = "harness_template_models"
    pkg_dir = TEMPLATE_ROOT / "src" / "models"

    # Package shell. No ``__init__.py`` needed — ``__path__`` is what tells
    # the importer where to find siblings for relative resolution.
    pkg_module = ModuleType(pkg_name)
    pkg_module.__path__ = [str(pkg_dir)]
    sys.modules[pkg_name] = pkg_module

    # Pre-load the sibling ``aws_retry`` so the relative import resolves
    # without re-spec'ing it.
    aws_retry_spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.aws_retry", pkg_dir / "aws_retry.py"
    )
    assert aws_retry_spec is not None and aws_retry_spec.loader is not None
    aws_retry_module = importlib.util.module_from_spec(aws_retry_spec)
    sys.modules[f"{pkg_name}.aws_retry"] = aws_retry_module
    aws_retry_spec.loader.exec_module(aws_retry_module)

    spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.bedrock_runtime_model", pkg_dir / "bedrock_runtime_model.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{pkg_name}.bedrock_runtime_model"] = module
    spec.loader.exec_module(module)
    return module


def load_template_runner(module_name: str) -> ModuleType:
    """Install the stubs and exec the template's ``runner.py``.

    Args:
        module_name: Name to load the runner under. Each test file uses its
            own so one file's patches can't reach another's copy.

    Returns:
        The loaded runner module, including its private helpers.
    """
    install_src_stubs()
    spec = importlib.util.spec_from_file_location(
        module_name, TEMPLATE_ROOT / "src" / "evaluation" / "runner.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
