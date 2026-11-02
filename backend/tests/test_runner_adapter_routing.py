# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``EvalRunner._make_model``: which adapter a model id routes to, and why.

THE SEAM. ``_make_model`` decides between the mantle adapter and the Converse
adapter by an EXACT-STRING lookup in ``src/model_descriptors.json``:

    descriptor = self._descriptors.get(model_id)

There is no normalization, no ``short_model_id`` reduction, no fallback. So the
id form written into ``config.py`` and the id form used as a descriptor KEY must
be the same string, or the model silently routes to the other adapter — no
error, no warning, just a different call path. That is precisely the mechanism
behind the first production failure that once motivated mantle-first precedence:
``config.py`` carried the runtime form while the descriptor map was keyed off the
bare mantle form, so a model with a fully-resolved mantle descriptor fell through
to Converse anyway.

The precedence flip to runtime-first makes that fallthrough the INTENT rather
than an accident, which is exactly why it needs pinning in both directions —
"routes to Converse" being correct now means a genuine mis-key looks identical
to correct behaviour unless something asserts the mapping.

Loads the packaged template's ``runner.py`` through the shared ``src.*`` stub
installer, so this tests the file a generated project actually copies.
"""
from __future__ import annotations

from types import ModuleType
from unittest.mock import MagicMock

import pytest

from tests.harness_template_stubs import load_template_runner


@pytest.fixture(scope="module")
def template_runner() -> ModuleType:
    return load_template_runner("harness_template_runner_adapter_routing")


@pytest.fixture(autouse=True)
def _reset_adapter_mocks(template_runner):
    """Both adapters are MagicMocks from the shared stub installer.

    So the assertion is "which class was CONSTRUCTED", not ``isinstance`` — a
    MagicMock is not a type. That is also a stronger check: it sees the
    constructor arguments, which is how the descriptor-region invariant below
    is verified rather than assumed.
    """
    template_runner.BedrockRuntimeModel.reset_mock()
    template_runner.BedrockMantleModel.reset_mock()


def _runner_with_descriptors(
    template_runner: ModuleType, descriptors: dict[str, dict]
):
    """An ``EvalRunner`` whose descriptor map is exactly ``descriptors``.

    ``__new__`` plus direct attribute assignment rather than ``__init__``: the
    real constructor reads a config file and the descriptor JSON off disk, and
    the only state ``_make_model`` touches is ``_descriptors`` and ``config``.
    """
    runner = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
    runner._descriptors = descriptors
    cfg = MagicMock()
    cfg.max_tokens = 2048
    cfg.temperature = 0.0
    cfg.aws_profile = None
    cfg.aws_region = "us-east-1"
    runner.config = cfg
    return runner


def test_a_runtime_id_with_no_descriptor_routes_to_converse(template_runner):
    """The runtime-first common case: no descriptor means Converse, by design."""
    runner = _runner_with_descriptors(template_runner, {})

    model = runner._make_model("global.anthropic.claude-sonnet-5")

    assert model is template_runner.BedrockRuntimeModel.return_value
    template_runner.BedrockMantleModel.assert_not_called()


def test_a_mantle_only_id_routes_to_the_mantle_adapter(template_runner):
    """Mantle retention is only real if the descriptor actually routes.

    A mantle-only model cannot be served by Converse at all, so this is the
    assertion that makes "mantle is kept for models runtime cannot serve" true
    rather than aspirational.
    """
    runner = _runner_with_descriptors(
        template_runner,
        {"openai.gpt-5.5": {"endpoint": "mantle", "api": "responses",
                            "region": "us-east-1"}},
    )

    model = runner._make_model("openai.gpt-5.5")

    assert model is template_runner.BedrockMantleModel.return_value
    template_runner.BedrockRuntimeModel.assert_not_called()


def test_the_mantle_adapter_signs_against_the_DESCRIPTOR_region(template_runner):
    """Not ``config.aws_region`` — a cross-region model would break.

    ``xai.grok-4.3`` is us-west-2-only while the project region is us-east-1;
    signing SigV4 against the project region would fail every call.
    """
    runner = _runner_with_descriptors(
        template_runner,
        {"xai.grok-4.3": {"endpoint": "mantle", "api": "responses",
                          "region": "us-west-2"}},
    )

    runner._make_model("xai.grok-4.3")

    template_runner.BedrockMantleModel.assert_called_once()
    descriptor_arg = template_runner.BedrockMantleModel.call_args[0][0]
    assert descriptor_arg["region"] == "us-west-2", (
        "the mantle adapter was handed a region other than the descriptor's — "
        "SigV4 would be signed for the wrong region"
    )


def test_id_form_drift_silently_routes_to_the_WRONG_adapter(template_runner):
    """The failure mode itself, pinned so the seam is documented not assumed.

    A descriptor keyed on the bare mantle form does NOT match a config carrying
    the runtime form: the lookup misses and the model goes to Converse with no
    error. Under runtime-first that outcome is correct, but the point is that
    the two id forms are NOT interchangeable at this boundary — anything that
    writes one while keying the other gets the other adapter, quietly.
    """
    runner = _runner_with_descriptors(
        template_runner,
        {"anthropic.claude-opus-4-8": {"endpoint": "mantle",
                                       "api": "anthropic_messages",
                                       "region": "us-east-1"}},
    )

    # Same model, runtime id form — misses the bare-keyed descriptor.
    assert (
        runner._make_model("global.anthropic.claude-opus-4-8")
        is template_runner.BedrockRuntimeModel.return_value
    )
    template_runner.BedrockMantleModel.assert_not_called()

    # The exact key still routes to mantle, proving the miss was the id FORM
    # and not a broken descriptor.
    assert (
        runner._make_model("anthropic.claude-opus-4-8")
        is template_runner.BedrockMantleModel.return_value
    )
