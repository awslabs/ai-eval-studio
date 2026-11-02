# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepEval-compatible LLM wrapper for AWS Bedrock Converse API.

Enables any Bedrock model (Claude, Nova, Llama, etc.) to be used as a
target model for generation or as a judge model for LLM-as-judge metrics.
``generate_with_system()`` returns ``(text, latency_ms, usage)`` tuples so
the same instance is safe to share across a ``ThreadPoolExecutor`` — all
per-call state is on the return value, never on ``self``.

Why a custom subclass instead of DeepEval's bundled ``AmazonBedrockModel``:
1. Thread-safety with metadata. Returning ``(text, latency, usage)`` as
   a tuple keeps every per-call value on the return rather than on
   instance attributes, so the same ``BedrockRuntimeModel`` is safe under the
   harness's ``ThreadPoolExecutor(max_workers=5)`` parallel generation.
   DeepEval's bundled model returns only text and exposes metadata via
   separate accessors that read instance state — not thread-safe.
2. Reasoning-model temperature guard. Anthropic Opus 4.6+ / Sonnet 4.6+
   reject ``temperature`` on the Converse API with ValidationException
   that surfaces as empty output / null metric scores. DeepEval's model
   doesn't strip; we do.
3. Empty-system-prompt fallback. Bedrock rejects ``system=[{"text": ""}]``
   with a cryptic ValidationError; we substitute a neutral default so
   callers don't have to guard themselves.

Kept intentionally in step with ``templates/harness/src/models/bedrock_runtime_model.py``
so behavior inside the platform matches behavior inside a scaffolded
harness project. Any fix here should be mirrored there.
"""

import asyncio
import logging
import os
import threading
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from deepeval.models.base_model import DeepEvalBaseLLM

from app.services.aws_retry import _bedrock_retry, is_temperature_capability_reject


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
    nothing about the others, so the harness runner's
    ``_is_systemic_generation_error`` correctly leaves it per-case.
    """


def extract_converse_text(response: dict, *, model_id: str) -> str:
    """Return the first text content block from a Converse response.

    Selects by block shape, never by position — see
    :class:`ConverseNoTextBlock` for why position is unreliable. Reasoning
    blocks are discarded on purpose: reasoning is not the model's answer
    and must never reach a metric as ``actual_output``.

    A block whose ``text`` is the empty string is returned as-is. That is
    the model answering with nothing, which is a different condition from
    emitting no text block, and it is what every non-reasoning shape
    already did before this helper existed.

    Args:
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
    """Thread-safe wrapper around a Bedrock model for use with DeepEval.

    ``generate_with_system()`` returns a tuple ``(text, latency_ms, usage)``
    instead of storing results on instance attributes, which is what makes
    it safe for concurrent calls from a ThreadPoolExecutor.

    Args:
        model_id: Bedrock model identifier (e.g., "us.anthropic.claude-sonnet-4-6").
        region: AWS region for the Bedrock runtime endpoint.
        temperature: Sampling temperature for generation. Silently dropped
            for reasoning models that reject the parameter.
        max_tokens: Maximum tokens in the response.
        profile_name: AWS profile name for credentials (None uses default).
    """

    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        profile_name: str | None = None,
    ):
        self.model_id = model_id
        self.region = region
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.profile_name = profile_name
        self._client = None
        # Response-driven temperature gate. Starts True; flipped False
        # the first time Bedrock returns a ValidationException mentioning
        # ``temperature`` for this instance. Subsequent calls skip the
        # param entirely. No static model list to maintain.
        self._supports_temperature_flag = True
        self._temperature_lock = threading.Lock()
        # Serializes the first load_model() call across threads so we don't
        # end up with two concurrent boto3.Session() / .client() pairs
        # racing on self._client. Subsequent hits of the hot path skip the
        # lock via the outer None check (double-checked locking).
        self._client_lock = threading.Lock()
        # Judge-cost accounting — mirror of the harness template model.
        # DeepEval never reports ``evaluation_cost`` for our custom
        # (non-native) model, so we accumulate raw token ``usage`` from
        # every Converse call and let the caller convert it to dollars.
        self._usage_lock = threading.Lock()
        self._usage_input_tokens = 0
        self._usage_output_tokens = 0

    def _accrue_usage(self, usage: dict | None) -> None:
        """Add one Converse response's token usage to the running total.

        Defensive: a malformed/absent ``usage`` must never raise, since
        this runs inside DeepEval's scorer where an exception becomes a
        silent null score under ``ignore_errors``.

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
    def accumulated_usage(self) -> dict:
        """Return summed token usage across every Converse call so far.

        Returns:
            Dict with ``inputTokens`` and ``outputTokens`` totals — the
            shape ``bedrock_pricing.estimate_cost_usd`` consumes.
        """
        with self._usage_lock:
            return {
                "inputTokens": self._usage_input_tokens,
                "outputTokens": self._usage_output_tokens,
            }

    def load_model(self):
        """Initialize and cache the Bedrock runtime client (thread-safe)."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    session = boto3.Session(profile_name=self.profile_name)
                    # botocore's default ``read_timeout`` is 60s, which
                    # cuts off realistic large-context Converse calls
                    # (AWS allows up to 60-min inference and recommends
                    # raising the SDK ``read_timeout`` — see the Nova
                    # Invoke docs). Mirrors the harness template's
                    # client config so in-process smoke validation
                    # (validate_harness) doesn't hit the same 60s wall
                    # the runtime harness avoids.
                    # Caveat: ``ReadTimeoutError`` stays in the
                    # retryable set; bounded worst case ≈ 4×300s ≈ 20
                    # min on a stuck call. Don't bump past ~5 min
                    # without also dropping ``ReadTimeoutError`` from
                    # the retryables.
                    client_config = Config(
                        read_timeout=int(os.environ.get(
                            "BEDROCK_READ_TIMEOUT_S", "300",
                        )),
                        connect_timeout=10,
                    )
                    self._client = session.client(
                        "bedrock-runtime",
                        region_name=self.region,
                        config=client_config,
                    )
        return self._client

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

    def generate(self, prompt: str) -> str:
        """Generate a response using the Bedrock Converse API.

        Called by DeepEval internally for judge model metrics. Returns only
        the text string (discards latency/usage) to satisfy DeepEval's interface.

        Note: ``generate_with_system`` carries the ``@_bedrock_retry``
        decorator, so this no-retry wrapper inherits the retry behavior
        for free without double-wrapping.

        Args:
            prompt: The user prompt to send.

        Returns:
            The model's text response.
        """
        text, _, _ = self.generate_with_system(system_prompt=None, user_prompt=prompt)
        return text

    @_bedrock_retry
    def generate_with_system(
        self, system_prompt: str | None, user_prompt: str
    ) -> tuple[str, float, dict]:
        """Generate a response with an optional system prompt.

        Thread-safe: all per-call state is returned, not stored on self.
        An empty or whitespace-only system prompt is replaced with a
        neutral default because Bedrock's Converse API rejects
        ``system=[{"text": ""}]`` with a cryptic ValidationError. An
        empty *user* prompt still errors loudly — that's a caller bug.

        Temperature handling is RESPONSE-DRIVEN. The first call
        attempts with ``temperature``; if Bedrock returns a
        ValidationException mentioning temperature, the param is dropped
        and the call is retried once. ``self._supports_temperature_flag``
        is flipped False under a lock so subsequent calls skip the
        parameter entirely. No static model list — the failure mode is
        structurally gone the moment the first 400 lands.

        Args:
            system_prompt: System-level instructions (None or empty to skip).
            user_prompt: User message to send.

        Returns:
            Tuple of (response_text, latency_ms, usage_dict).

        Raises:
            ConverseNoTextBlock: The response carried no text content
                block — see that class for the reasoning-model case.
        """
        client = self.load_model()

        kwargs: dict = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": [{"text": user_prompt}]}],
        }
        if system_prompt and system_prompt.strip():
            kwargs["system"] = [{"text": system_prompt}]

        # Snapshot the gate under the lock so a concurrent flip can't
        # land between the read and the call. We can flip it (after a
        # response-driven 400) without holding the lock for the whole
        # network round-trip.
        with self._temperature_lock:
            attempt_with_temperature = self._supports_temperature_flag

        start_time = time.perf_counter()
        try:
            response = client.converse(
                **kwargs,
                inferenceConfig=self._build_inference_config(attempt_with_temperature),
            )
        except ClientError as exc:
            if attempt_with_temperature and self._is_temperature_reject(exc):
                # Latch the gate so every subsequent call skips
                # ``temperature``, not just this one. Two concurrent
                # callers may each take this branch once on the first
                # batch — that's fine, the second flip is idempotent.
                with self._temperature_lock:
                    self._supports_temperature_flag = False
                logger.info(
                    "[bedrock-runtime] %s rejected temperature; "
                    "self-correcting and retrying without it.",
                    self.model_id,
                )
                response = client.converse(
                    **kwargs,
                    inferenceConfig=self._build_inference_config(False),
                )
            else:
                raise
        latency_ms = (time.perf_counter() - start_time) * 1000

        usage = response.get("usage", {})
        text = extract_converse_text(response, model_id=self.model_id)

        # Single accrual point: generate() delegates here, so accruing
        # only in generate_with_system() covers both entry points with
        # no double-count.
        self._accrue_usage(usage)

        return text, latency_ms, usage

    def _build_inference_config(self, with_temperature: bool) -> dict:
        """Assemble the Converse ``inferenceConfig`` for one call.

        Pulled out so the response-driven retry can rebuild the config
        without ``temperature`` without duplicating the maxTokens line.
        """
        cfg: dict = {"maxTokens": self.max_tokens}
        if with_temperature:
            cfg["temperature"] = self.temperature
        return cfg

    async def a_generate(self, prompt: str) -> str:
        """Async generation — runs sync boto3 call in a thread to avoid blocking.

        Returns only text to satisfy DeepEval's async interface.

        Args:
            prompt: The user prompt to send.

        Returns:
            The model's text response.
        """
        text = await asyncio.to_thread(self.generate, prompt)
        return text

    def get_model_name(self) -> str:
        """Return the Bedrock model identifier."""
        return self.model_id
