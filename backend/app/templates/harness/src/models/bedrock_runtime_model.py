# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Thread-safe BedrockRuntimeModel implementation for DeepEval.

This module provides a thread-safe wrapper around AWS Bedrock's Converse API
for use with DeepEval's evaluation framework. Unlike instance-attribute-based
implementations, this version returns latency and usage data as tuples to ensure
thread safety when used with ThreadPoolExecutor.

Key features:
- Lazy client initialization with double-checked locking
- Thread-safe generate_with_system() that returns (text, latency_ms, usage) tuple
- Async support via run_in_executor
- Configurable temperature and max_tokens
- Bedrock retry on transient failures (throttling, 5xx, network) with
  exponential backoff + jitter — DeepEval only wires this for its own
  ``AmazonBedrockModel`` class; this custom subclass needs its own retry
  or ThrottlingException bubbles into DeepEval's ``ignore_errors`` swallow
  and metrics silently score null.

This file is kept in step with
``backend/app/services/bedrock_runtime_model.py``. Any fix here should be
mirrored there.
"""

import asyncio
import logging
import os
import threading
import time
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from deepeval.models import DeepEvalBaseLLM

from .aws_retry import _bedrock_retry, is_temperature_capability_reject


logger = logging.getLogger(__name__)


class ConverseNoTextBlock(Exception):
    """Raised when a Converse response carries no text content block at all.

    ``output.message.content`` is an ORDERED, HETEROGENEOUS list. A
    reasoning-capable model returns the reasoning first and the answer
    after it::

        content: [{"reasoningContent": {...}}, {"text": "..."}]

    so ``content[0]["text"]`` raises ``KeyError('text')`` on exactly the
    models most worth evaluating. Selecting by block *shape* instead of
    position removes that failure mode; this error covers what remains —
    a response with no text block anywhere, which is what a model
    produces when it spends its whole ``maxTokens`` budget reasoning and
    is truncated before emitting an answer.

    That case must be named and loud rather than an empty string: an
    empty ``actual_output`` scores as a legitimate low score, so it
    corrupts a pass rate instead of failing it.

    Mirrors ``MantleReasoningTruncated`` on the mantle path, which has
    handled this since the Responses mapper shipped — which is why the
    Converse gap stayed invisible while precedence was mantle-first.
    Deliberately NOT systemic: one model/cap combination failing says
    nothing about the others, so ``_is_systemic_generation_error`` in
    ``src/evaluation/runner.py`` correctly leaves it per-case.
    """


def extract_converse_text(response: dict[str, Any], *, model_id: str) -> str:
    """Return the first text content block from a Converse response.

    Selects by block shape, never by position — see
    :class:`ConverseNoTextBlock` for why position is unreliable. Reasoning
    blocks are discarded on purpose: reasoning is not the model's answer
    and must never reach a metric as ``actual_output``.

    A block whose ``text`` is the empty string is returned as-is. That is
    the model answering with nothing, which is a different condition from
    emitting no text block, and it is what every non-reasoning shape
    already did before this helper existed.

    Parameters:
        response: A parsed ``converse()`` response.
        model_id: Model id, used only to make the error message actionable.

    Returns:
        The text carried by the first content block that has a ``text`` key.

    Raises:
        ConverseNoTextBlock: No content block carries text.
    """
    content = ((response.get("output") or {}).get("message") or {}).get("content") or []
    for block in content:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            return block["text"]

    # No text anywhere. Name what DID come back so the operator sees
    # "reasoningContent" instead of a bare KeyError('text').
    present = sorted({key for b in content if isinstance(b, dict) for key in b})
    raise ConverseNoTextBlock(
        f"Converse returned no text content block for model {model_id} "
        f"(stopReason={response.get('stopReason', '<unset>')!r}, "
        f"content blocks present: {', '.join(present) or '<none>'}). "
        "A reasoning model that exhausts maxTokens before emitting an "
        "answer produces this; raise max_tokens and re-run."
    )


class BedrockRuntimeModel(DeepEvalBaseLLM):
    """
    Thread-safe Bedrock model adapter for DeepEval.

    CRITICAL: generate_with_system() returns (text, latency_ms, usage) tuple
    instead of storing on instance attributes. This makes it safe for parallel
    execution in ThreadPoolExecutor.

    Parameters:
        model_id: Full Bedrock model ID (e.g., "us.anthropic.claude-sonnet-4-6-v1:0")
        region: AWS region (default: "us-east-1")
        profile_name: AWS profile name (optional)
        temperature: Sampling temperature (default: 0.0)
        max_tokens: Maximum output tokens (default: 2048)
    """

    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
        profile_name: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        self._model_id = model_id
        self._region = region
        self._profile_name = profile_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None
        self._lock = threading.Lock()  # For lazy client initialization
        # Response-driven temperature gate. Starts True; flipped False
        # the first time Bedrock returns a ValidationException mentioning
        # ``temperature`` for this instance. Subsequent calls skip the
        # param entirely. No static model list to maintain.
        self._supports_temperature_flag = True
        self._temperature_lock = threading.Lock()
        # Judge-cost accounting. DeepEval never reports ``evaluation_cost``
        # for a custom (non-native) model, so we capture the raw token
        # ``usage`` from every Converse call and accumulate it here. The
        # runner converts this to a dollar figure at serialize time
        # (keeping pricing in one place). Guarded by its own lock so the
        # parallel scoring path can't lose updates.
        self._usage_lock = threading.Lock()
        self._usage_input_tokens = 0
        self._usage_output_tokens = 0

    def _accrue_usage(self, usage: dict[str, Any] | None) -> None:
        """Add one Converse response's token usage to the running total.

        Defensive by design: a malformed/absent ``usage`` must never
        raise, because this runs inside DeepEval's scorer where an
        exception becomes a silent null score under ``ignore_errors``.

        Args:
            usage: The ``usage`` dict from a Bedrock Converse response,
                or None if the response carried none.
        """
        if not usage:
            return
        try:
            input_tokens = int(usage.get("inputTokens", 0) or 0)
            output_tokens = int(usage.get("outputTokens", 0) or 0)
        except (TypeError, ValueError):
            return
        with self._usage_lock:
            self._usage_input_tokens += input_tokens
            self._usage_output_tokens += output_tokens

    @property
    def accumulated_usage(self) -> dict[str, int]:
        """Return the summed token usage across every Converse call so far.

        Returns:
            Dict with ``inputTokens`` and ``outputTokens`` totals — the
            shape the harness's ``_estimate_cost`` consumes.
        """
        with self._usage_lock:
            return {
                "inputTokens": self._usage_input_tokens,
                "outputTokens": self._usage_output_tokens,
            }

    def load_model(self) -> Any:
        """
        Lazily initialize Bedrock client with double-checked locking.

        Thread-safe client initialization that avoids unnecessary locking
        once the client is created.

        Returns:
            Initialized boto3 bedrock-runtime client
        """
        if self._client is None:
            with self._lock:
                # Double-checked locking: verify still None after acquiring lock
                if self._client is None:
                    session_kwargs: dict[str, Any] = {"region_name": self._region}
                    if self._profile_name:
                        session_kwargs["profile_name"] = self._profile_name
                    session = boto3.Session(**session_kwargs)
                    # botocore's default ``read_timeout`` is 60s, which
                    # cuts off realistic large-context Converse calls
                    # (AWS allows up to 60-min inference and explicitly
                    # recommends raising the SDK ``read_timeout`` — see
                    # the Nova Invoke docs). 300s default covers the
                    # generations we actually run; tunable via env to
                    # match the existing ``BEDROCK_RETRY_*`` knobs.
                    # Caveat: ``ReadTimeoutError`` stays in
                    # ``aws_retry._RETRYABLE_NETWORK_EXCEPTIONS``; at 300s
                    # the bounded worst case is ~4×300s ≈ 20 min on a
                    # stuck call. Don't bump this past ~5 minutes without
                    # also dropping ``ReadTimeoutError`` from the
                    # retryables, or one stuck call ≈
                    # ``BEDROCK_READ_TIMEOUT_S`` × attempts.
                    client_config = Config(
                        read_timeout=int(os.environ.get(
                            "BEDROCK_READ_TIMEOUT_S", "300",
                        )),
                        connect_timeout=10,
                    )
                    self._client = session.client(
                        "bedrock-runtime", config=client_config,
                    )
        return self._client

    def get_model_name(self) -> str:
        """Return the model ID for DeepEval tracking."""
        return self._model_id

    def _is_temperature_reject(self, exc: Exception) -> bool:
        """Identify a ValidationException that rejected ``temperature`` AS A CAPABILITY.

        Delegates to the shared list-free rule ``is_temperature_capability_reject``
        in ``aws_retry.py``. A ValidationException whose message mentions
        ``temperature`` is a capability rejection (drop the parameter, retry once,
        cache the learned flag) EXCEPT when it is an out-of-range / value error
        (numeric range language such as "must be between"), which must propagate
        as a caller-config bug.

        The old phrase-list (``_TEMPERATURE_REJECT_SHAPE_MARKERS``) was removed
        because the broad rule covers "deprecated", "is not supported", "does not
        support", and future rejection phrasings without maintenance.

        Args:
            exc: The exception to classify.

        Returns:
            True if the exception is a temperature capability rejection.
        """
        return is_temperature_capability_reject(exc)

    def _build_inference_config(self, with_temperature: bool) -> dict[str, Any]:
        """Assemble the Converse ``inferenceConfig`` for one call.

        Pulled out so the response-driven retry can rebuild the config
        without ``temperature`` without duplicating the ``maxTokens`` line.
        """
        cfg: dict[str, Any] = {"maxTokens": self._max_tokens}
        if with_temperature:
            cfg["temperature"] = self._temperature
        return cfg

    def _converse_self_correcting(
        self, *, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Invoke Converse, dropping ``temperature`` if Bedrock rejects it.

        Single attempt with the current gate; on a temperature-rejection
        ValidationException, latches the gate False and retries once
        without the param. Used by both ``generate`` and
        ``generate_with_system`` so the response-driven self-correction
        is one source of truth.
        """
        client = self.load_model()

        with self._temperature_lock:
            attempt_with_temperature = self._supports_temperature_flag

        try:
            return client.converse(
                **kwargs,
                inferenceConfig=self._build_inference_config(attempt_with_temperature),
            )
        except ClientError as exc:
            if attempt_with_temperature and self._is_temperature_reject(exc):
                with self._temperature_lock:
                    self._supports_temperature_flag = False
                logger.info(
                    "[bedrock-runtime] %s rejected temperature; "
                    "self-correcting and retrying without it.",
                    self._model_id,
                )
                return client.converse(
                    **kwargs,
                    inferenceConfig=self._build_inference_config(False),
                )
            raise

    @_bedrock_retry
    def generate(self, prompt: str, schema: Any = None) -> str:
        """
        Generate completion for a single user prompt.

        Note: This method only returns text for DeepEval compatibility.
        Use generate_with_system() when you need latency and usage data.

        Parameters:
            prompt: User prompt text
            schema: Not used (DeepEval compatibility parameter)

        Returns:
            Generated text completion

        Raises:
            ConverseNoTextBlock: The response carried no text content
                block — see that class for the reasoning-model case.
        """
        response = self._converse_self_correcting(
            kwargs={
                "modelId": self._model_id,
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
            }
        )

        # Judge calls funnel through here (DeepEval invokes generate() for
        # non-native models); capture usage before discarding the rest of
        # the response so the runner can total judge cost.
        self._accrue_usage(response.get("usage"))

        return extract_converse_text(response, model_id=self._model_id)

    @_bedrock_retry
    def generate_with_system(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[str, float, dict[str, Any]]:
        """
        Generate completion with system prompt and return metadata.

        CRITICAL: Returns tuple instead of storing on instance attributes
        to ensure thread safety when used with ThreadPoolExecutor.

        Parameters:
            system_prompt: System prompt text
            user_prompt: User prompt text

        Returns:
            Tuple of (generated_text, latency_ms, usage_dict)

        Example:
            >>> model = BedrockRuntimeModel("us.anthropic.claude-sonnet-4-6-v1:0")
            >>> text, latency, usage = model.generate_with_system(
            ...     "You are a helpful assistant", "Hello!"
            ... )
            >>> print(f"Generated {usage['outputTokens']} tokens in {latency:.1f}ms")
        """
        # Bedrock rejects empty ``system[].text`` with a cryptic
        # ValidationError; fall back to a neutral system prompt so the
        # caller doesn't have to guard this themselves. An empty USER
        # message should still error loudly — that's a caller bug.
        if not system_prompt or not system_prompt.strip():
            system_prompt = "You are a helpful assistant."

        messages = [{"role": "user", "content": [{"text": user_prompt}]}]
        system = [{"text": system_prompt}]

        start = time.perf_counter()
        response = self._converse_self_correcting(
            kwargs={
                "modelId": self._model_id,
                "messages": messages,
                "system": system,
            }
        )
        latency_ms = (time.perf_counter() - start) * 1000

        text = extract_converse_text(response, model_id=self._model_id)
        usage = response.get("usage", {})

        # Accrue for judge-cost totals. Safe to accrue here as well as in
        # generate(): the two paths are mutually exclusive on a given
        # call, so there's no double-count.
        self._accrue_usage(usage)

        return text, latency_ms, usage

    async def a_generate(self, prompt: str, schema: Any = None) -> str:
        """
        Async wrapper for generate() using run_in_executor.

        Required by DeepEval for async evaluation support.

        Parameters:
            prompt: User prompt text
            schema: Not used (DeepEval compatibility parameter)

        Returns:
            Generated text completion

        Raises:
            ConverseNoTextBlock: The response carried no text content
                block — see that class for the reasoning-model case.
        """
        return await asyncio.to_thread(self.generate, prompt, schema)
