# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for systemic generation failure fast-abort in runner.py.

A generation error that will fail identically
for every case (CredentialRetrievalError / AccessDeniedException / model not
found) aborts the generation loop: at most 40 of 200 cases attempted
(8 * max_workers, measured ceiling), one exception propagates, and pending
futures are cancelled.

Loads the template's runner.py via the synthetic-package loader pattern from
test_runner_generation_failfast.py.
"""
from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, CredentialRetrievalError, NoCredentialsError
from deepeval.test_case import LLMTestCase

from tests.harness_template_stubs import TEMPLATE_ROOT, load_template_runner


def _load_template_runner() -> ModuleType:
    """Load the template ``runner.py`` with the shared ``src.*`` stubs.

    Also registers the loaded module under its real dotted name so code that
    resolves ``src.evaluation.runner`` reaches this copy.
    """
    runner_module = load_template_runner("src.evaluation.runner")
    sys.modules["src.evaluation.runner"] = runner_module
    return runner_module


@pytest.fixture(scope="module")
def template_runner():
    """Load the template runner module."""
    return _load_template_runner()


def _make_runner(template_runner, json_progress=False):
    """Construct an EvalRunner with minimum config."""
    cfg = MagicMock()
    cfg.metrics_json = "[]"
    cfg.results_dir = "/tmp"
    cfg.max_concurrent = 5
    runner_inst = template_runner.EvalRunner(cfg, json_progress=json_progress)
    return runner_inst


def test_systemic_error_aborts_within_ceiling(template_runner):
    """Systemic error (credential retrieval) aborts within 40 attempts of 200 cases.

    The measured ceiling is 8 * max_workers (=40, where max_workers=5 is the
    ThreadPoolExecutor literal in the template runner's ``generate_outputs``).
    The 20 ms delay is required: an instant-raising
    fake lets the executor drain its queue before as_completed yields, which
    measured 15-159 attempts (flaky) and does not represent a real generation
    call. At 20 ms the count was stable across 200 trials (min=6, max=18 under
    4x contention; adversarial worst 32).
    """
    runner_inst = _make_runner(template_runner, json_progress=True)
    attempt_count = 0
    attempt_lock = threading.Lock()

    class FakeModel:
        def get_model_name(self):
            return "fake-model"

        def generate_with_system(self, system, user):
            nonlocal attempt_count
            with attempt_lock:
                attempt_count += 1
            # Sleep ~20 ms before raising so the executor doesn't drain its
            # queue instantly (which would let attempts reach 15-159, flaky).
            time.sleep(0.02)
            raise CredentialRetrievalError(
                provider="custom-process",
                error_msg="Error: No module named 'app'",
            )

    base_cases = [
        LLMTestCase(
            input=f"test case {i}",
            actual_output="",
            expected_output="expected",
        )
        for i in range(200)
    ]

    fake_model = FakeModel()

    # The abort raises SystemicGenerationFailure
    with pytest.raises(template_runner.SystemicGenerationFailure) as exc_info:
        runner_inst.generate_outputs(
            fake_model,
            "test prompt",
            base_cases,
        )

    # Assert bounded: at most 8 * max_workers (40) attempts
    assert attempt_count <= 40, (
        f"Expected <= 40 attempts (8 * max_workers=5), got {attempt_count}. "
        f"The abort did not cancel pending futures."
    )

    # Assert the chained exception is the original CredentialRetrievalError
    assert isinstance(exc_info.value.__cause__, CredentialRetrievalError)


def test_per_case_error_does_not_abort(template_runner):
    """Throttling error (per-case) does NOT abort: all 200 cases attempted.

    This is the direct counterpart to the ceiling assertion: it proves the
    ceiling is enforced by the abort and not by some unrelated early exit.
    """
    runner_inst = _make_runner(template_runner, json_progress=True)
    attempt_count = 0
    attempt_lock = threading.Lock()

    class FakeModel:
        def get_model_name(self):
            return "fake-model"

        def generate_with_system(self, system, user):
            nonlocal attempt_count
            with attempt_lock:
                attempt_count += 1
            time.sleep(0.02)
            # Throttling ClientError: NOT systemic
            error = ClientError(
                {
                    "Error": {
                        "Code": "ThrottlingException",
                        "Message": "Rate exceeded",
                    }
                },
                "InvokeModel",
            )
            raise error

    base_cases = [
        LLMTestCase(
            input=f"test case {i}",
            actual_output="",
            expected_output="expected",
        )
        for i in range(200)
    ]

    fake_model = FakeModel()

    # Throttling does NOT abort — generate_outputs returns normally
    result = runner_inst.generate_outputs(
        fake_model,
        "test prompt",
        base_cases,
    )

    # All 200 cases were attempted (no abort)
    assert attempt_count == 200, f"Expected 200 attempts, got {attempt_count}"

    # Each case is a failed LLMTestCase with error metadata
    assert len(result) == 200
    for tc in result:
        assert tc.actual_output == ""
        assert "error" in (tc.metadata or {})


def test_systemic_classifier_truth_table(template_runner):
    """Classifier correctly identifies systemic vs per-case errors."""
    # Systemic: credential failures
    assert template_runner._is_systemic_generation_error(
        CredentialRetrievalError(provider="test", error_msg="test")
    )
    assert template_runner._is_systemic_generation_error(NoCredentialsError())

    # Systemic: access denials (ClientError with specific codes)
    access_denied = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "Denied"}},
        "InvokeModel",
    )
    assert template_runner._is_systemic_generation_error(access_denied)

    resource_not_found = ClientError(
        {"Error": {"Code": "ResourceNotFoundException", "Message": "Not found"}},
        "InvokeModel",
    )
    assert template_runner._is_systemic_generation_error(resource_not_found)

    # NOT systemic: throttling
    throttling = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
        "InvokeModel",
    )
    assert not template_runner._is_systemic_generation_error(throttling)

    # NOT systemic: plain RuntimeError
    runtime_error = RuntimeError("generic error")
    assert not template_runner._is_systemic_generation_error(runtime_error)

    # NOT systemic: malformed ClientError (no response)
    class MalformedClientError(Exception):
        pass
    malformed = MalformedClientError()
    malformed.response = None
    assert not template_runner._is_systemic_generation_error(malformed)


def test_classifier_does_not_match_prose(template_runner):
    """Classifier reads types and structured codes, NOT message prose.

    A RuntimeError whose message contains credential-error prose is NOT
    systemic. This pins the "no English matching" contract.
    """
    prose_error = RuntimeError(
        "Error when retrieving credentials: No module named 'app'"
    )
    assert not template_runner._is_systemic_generation_error(prose_error), (
        "Classifier matched prose — it should only read exception type and "
        "structured response code, never message text"
    )


def test_failclosed_overinclusion_is_intentional(template_runner):
    """ALL CredentialRetrievalError is systemic, including transient causes.

    This test pins a deliberate tradeoff: botocore's ProcessProvider
    wraps every helper-subprocess failure in CredentialRetrievalError, so the
    type alone cannot distinguish a permanent cause (ModuleNotFoundError, STS
    AccessDenied) from a transient one (fd exhaustion, blip reaching the ECS
    credential endpoint). The classifier treats ALL CredentialRetrievalError as
    systemic and will abort a run whose credential failure was in fact
    transient. That over-inclusion is intentional and fail-closed: a credential
    that cannot be retrieved is not repaired by a per-case retry, the eval is
    re-dispatchable, and the alternative is the same failure repeated for every
    case in the dataset.

    This test will fail if someone later adds prose matching to narrow the
    classifier, surfacing the tradeoff as an asserted decision rather than an
    accident.
    """
    # A CredentialRetrievalError whose text describes an obviously transient
    # cause is STILL classified systemic
    transient_cred_error = CredentialRetrievalError(
        provider="custom-process",
        error_msg="Connection timeout reaching ECS credential endpoint",
    )
    assert template_runner._is_systemic_generation_error(transient_cred_error), (
        "Classifier narrowed CredentialRetrievalError — the fail-closed "
        "over-inclusion was removed. This is a deliberate tradeoff: the eval "
        "is re-dispatchable and one identical failure per case is the worse "
        "outcome."
    )


def test_raised_exception_names_model_and_underlying_error(template_runner):
    """The raised SystemicGenerationFailure names the model and exception type.

    The single stderr_tail the operator sees must be actionable, so the message
    includes both the model and the underlying error type.
    """
    runner_inst = _make_runner(template_runner, json_progress=True)

    class FakeModel:
        def get_model_name(self):
            return "test-model-id"

        def generate_with_system(self, system, user):
            raise CredentialRetrievalError(
                provider="custom-process",
                error_msg="ModuleNotFoundError: No module named 'app'",
            )

    base_cases = [
        LLMTestCase(input="test", actual_output="", expected_output="expected")
    ]

    fake_model = FakeModel()

    with pytest.raises(template_runner.SystemicGenerationFailure) as exc_info:
        runner_inst.generate_outputs(
            fake_model,
            "test prompt",
            base_cases,
        )

    message = str(exc_info.value)
    assert "test-model-id" in message, "Model name missing from error message"
    assert "CredentialRetrievalError" in message, "Exception type missing"
