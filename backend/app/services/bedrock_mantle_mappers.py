# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-function mappers for the three Bedrock Mantle invocation APIs.

Bedrock Mantle exposes three distinct invocation surfaces — the OpenAI
Responses API (``/openai/v1/responses``), the OpenAI chat completions API
(``/v1/chat/completions``), and the Anthropic Messages API
(``/anthropic/v1/messages``). Each model published on the mantle endpoint
family supports exactly one of these — calling a model on the wrong API
returns a ``400`` with ``error.code == "validation_error"`` (see
``tests/fixtures/mantle_payloads/responses_400_wrong_api.json``).

Each API has a ``_build_*_request`` and ``_parse_*_response`` function in
this module. The adapter (``bedrock_mantle_model.BedrockMantleModel``)
dispatches by ``descriptor.api`` via strategy dicts. All three pairs are
live: their request/response shapes were verified against the real
endpoint with gpt-5.5 (responses), mistral-large-3 (chat_completions) and
claude-opus-4-8 (anthropic_messages).

This file is duplicated byte for byte between ``backend/app/services/``
and ``backend/app/templates/harness/src/models/``;
``tests/test_templates_parity.py`` enforces the match, so edit both copies
together. Stdlib + ``typing`` only — no ``app.`` imports anywhere.
"""

from typing import Any, Literal, NotRequired, TypedDict

# Re-exported from ``bedrock_mantle_model`` too — defined once here so the
# strategy dispatch dict in the adapter and the descriptor TypedDict in
# ``model_descriptors`` can share the same Literal alias without circular
# imports.
MantleAPIName = Literal["responses", "chat_completions", "anthropic_messages"]


class MantleSchemaDrift(Exception):
    """Raised when a parsed payload is missing a required field.

    The schema-driven parser walks the payload by ``type`` field; if no
    matching block is found (e.g. no ``output_text`` content block, or
    ``output`` is not a list), this fires. Surfaces a clear named error
    to the discovery layer and to the user instead of a silent
    zero / empty extraction.
    """


class MantleReasoningTruncated(Exception):
    """Raised when Responses returns ``status='incomplete'`` with reason ``max_output_tokens``.

    NEVER silently retried with a larger cap — the user raises
    ``max_output_tokens`` and re-runs. Carries the truncation cause in the
    message; the harness runner also marks the affected case with
    ``MANTLE_REASONING_TRUNCATED`` metadata so reports can say "Reasoning
    truncated" instead of showing a generic error.
    """


class MantleAPIRejected(Exception):
    """Raised on a 400 whose body has ``error.code == 'validation_error'``.

    Authoritative capability-detection signal — discovery consumes
    this for re-resolution onto the correct API. Non-retryable per
    ``aws_retry._is_retryable_mantle_error``.
    """


class MantleTemperatureUnsupported(Exception):
    """Raised on a 400 whose body rejects the ``temperature`` parameter.

    Reasoning models reject ``temperature`` with several shapes that all
    mention the word in the message text. Verified live:

      * Responses (``openai.gpt-5.5``, ``xai.grok-4.3``, ``openai.gpt-5.4``)::

            {"error": {"code": "unsupported_parameter",
                       "message": "'temperature' is not supported with this model",
                       "param": "temperature"}}

      * Anthropic Messages (``anthropic.claude-opus-4-7/4-8``)::

            {"error": {"message": "temperature is deprecated, use top_p"}}

    The trigger is permissive — any 400 whose lowercase message contains
    ``"temperature"`` counts. The rationale is permissive-discovery:
    rather than enumerating every provider's exact code/param shape,
    the live 400 IS the signal.

    Distinct from ``MantleAPIRejected`` because the right action is
    "drop ``temperature`` and retry on the SAME api", not "try a
    different api". Adapter catches it, sets
    ``self._descriptor["supports_temperature"] = False``, and retries
    once. Discovery catches it during probe and treats the api as
    matched (returns the resolved api + ``supports_temperature=False``).

    This is the temperature self-correction path — without
    it, reasoning models are excluded from the catalog as "not invocable
    on any API" because the wrong-API fallback chain runs and exhausts.
    """


class MantleContentSafetyBlocked(Exception):
    """Raised on a 400 whose body indicates a content-safety / policy block.

    Distinct from ``MantleAPIRejected`` because the request reached the
    model — the model declined to respond on policy grounds. The api
    routing is correct; the descriptor is keepable.

    Detected by 400 + (``error.code`` containing one of ``content_filter``,
    ``content_policy``, ``content_safety``, ``safety_violation``,
    case-insensitive) OR ``error.type == "content_filter_error"``. The
    permissive-discovery policy treats this as "reachable, keep" so a
    bland-probe getting safety-blocked doesn't drop the model from the
    catalog.
    """


class MantleRequestRejected(Exception):
    """Raised on a 400 that fell through every other 400 branch.

    The catch-all for a request-shape rejection that is neither a
    ``temperature`` parameter rejection, a content-safety block, nor a
    wrong-API ``validation_error``. The whole point of this exception is
    that it carries the provider's response body message VERBATIM — the
    bare ``response.raise_for_status()`` it replaces produced only the
    httpx "400 Bad Request" wrapper, dropping the real reason on the
    floor. Surfacing the message lets it flow into the harness's per-case
    ``additional_metadata.error`` and on to the infra-failure classifier,
    so a failed run is diagnosed against the actual cause instead of a
    generic status string.

    The motivating shape (verified live): forcing a tool call on a
    reasoning model that has thinking enabled returns::

        {"error": {"message": "Thinking may not be enabled when "
                              "tool_choice forces tool use",
                   "type": "invalid_request_error"}}

    Non-retryable. A generic 400 is a caller / request-shape problem,
    not a transient server condition — backing off won't fix it. This is
    correct WITHOUT any change to ``aws_retry._is_retryable_mantle_error``:
    that classifier returns False for anything that is not an
    ``httpx.HTTPStatusError`` (or a retryable httpx network error), and
    this exception is deliberately neither, so it short-circuits retry.
    """


# --------------------------------------------------------------------------- #
# Responses API (supported)                                                   #
# --------------------------------------------------------------------------- #


def _build_responses_request(
    model_id: str,
    system_prompt: str | None,
    user_prompt: str,
    max_output_tokens: int,
    supports_temperature: bool,
    temperature: float,
) -> dict[str, Any]:
    """Build the request body for ``POST /openai/v1/responses``.

    Body shape verified against the live capture in
    ``tests/fixtures/mantle_payloads/responses_request.json``.

    Defensive contract:
      * Empty / whitespace-only ``system_prompt`` → omit ``instructions``.
        Responses API contract is to OMIT the field, not send the empty
        string. The runtime BedrockRuntimeModel uses a neutral default for
        Converse, but Responses behaves differently.
      * ``supports_temperature=False`` → omit ``temperature``. Reasoning
        models on the Responses API reject the parameter;
        the response-driven self-correction caches the False flag onto
        the descriptor — handling lives in the adapter.

    Args:
        model_id: The Bedrock-Mantle model id (e.g. ``"openai.gpt-5.5"``).
        system_prompt: Optional system instructions. Empty / whitespace
            → omitted from the request.
        user_prompt: The user input string sent as ``input``.
        max_output_tokens: Hard cap on completion length. Surface for the
            user-driven adjustment after a ``MantleReasoningTruncated``.
        supports_temperature: If False, ``temperature`` is omitted.
        temperature: Sampling temperature. Only included when
            ``supports_temperature`` is True.

    Returns:
        The JSON-serializable request dict.
    """
    req: dict[str, Any] = {
        "model": model_id,
        "input": user_prompt,
        "max_output_tokens": max_output_tokens,
    }
    if system_prompt and system_prompt.strip():
        req["instructions"] = system_prompt
    if supports_temperature:
        req["temperature"] = temperature
    return req


def _parse_responses_response(payload: dict[str, Any]) -> tuple[str, dict[str, int]]:
    """Parse a Responses 200 payload to ``(text, usage)``.

    Schema-driven walk: blocks are selected by their ``type`` field, never
    by position, so an added or reordered block (a reasoning summary, a new
    block kind) cannot shift which text is read. Steps:
      1. Truncation check — if ``status == 'incomplete'`` and
         ``incomplete_details.reason == 'max_output_tokens'``, raise
         ``MantleReasoningTruncated``. Done BEFORE
         parsing output so a partial output doesn't mask the truncation.
      2. Walk ``output[]`` for ``type == 'message'`` blocks; skip
         ``reasoning`` and unknown types.
      3. Within each message, walk ``content[]`` for ``type ==
         'output_text'`` blocks; concatenate their ``text`` fields.
      4. Normalize ``usage`` to ``{"inputTokens", "outputTokens"}`` —
         the same shape ``BedrockRuntimeModel._accrue_usage`` consumes.

    Args:
        payload: The response body JSON.

    Returns:
        ``(text, usage)`` where ``usage`` has integer
        ``inputTokens`` and ``outputTokens`` keys (defensive zeros on
        missing / malformed numbers).

    Raises:
        MantleReasoningTruncated: status='incomplete' with
            reason='max_output_tokens'.
        MantleSchemaDrift: ``output`` is missing / not a list, or no
            ``output_text`` block was found.
    """
    # Truncation must surface BEFORE output parsing so partial
    # output (typical when the cap fires mid-thought) doesn't silently
    # masquerade as a successful extraction.
    status = payload.get("status")
    incomplete_details = payload.get("incomplete_details")
    reason = (
        incomplete_details.get("reason")
        if isinstance(incomplete_details, dict)
        else None
    )
    if status == "incomplete" and reason == "max_output_tokens":
        raise MantleReasoningTruncated(
            f"Responses returned status='incomplete' reason='max_output_tokens' "
            f"for model {payload.get('model', '?')}; raise max_output_tokens "
            "and re-run (no silent retry-with-larger-cap)."
        )

    output = payload.get("output", [])
    if not isinstance(output, list):
        raise MantleSchemaDrift(
            f"Responses payload missing 'output' list (got {type(output).__name__})"
        )

    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            # skip 'reasoning' and unknown types per the captured schema
            continue
        for block in item.get("content", []):
            if isinstance(block, dict) and block.get("type") == "output_text":
                txt = block.get("text", "")
                if isinstance(txt, str):
                    texts.append(txt)
    if not texts:
        raise MantleSchemaDrift(
            "Responses payload contained no output[].type=='message' / "
            "content[].type=='output_text' block"
        )
    text = "".join(texts)

    # Usage normalization — per the verified live contract,
    # usage.output_tokens ALREADY includes reasoning tokens, so we
    # accrue once. Do NOT add usage.output_tokens_details.reasoning_tokens
    # separately: those tokens are already inside that total, so adding
    # them would count reasoning twice.
    raw_usage = payload.get("usage", {})
    if not isinstance(raw_usage, dict):
        raw_usage = {}
    try:
        input_tokens = int(raw_usage.get("input_tokens", 0) or 0)
        output_tokens = int(raw_usage.get("output_tokens", 0) or 0)
    except (TypeError, ValueError):
        input_tokens = 0
        output_tokens = 0
    usage: dict[str, int] = {
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
    }
    return text, usage


# --------------------------------------------------------------------------- #
# Chat Completions API (LIVE)                                                 #
# --------------------------------------------------------------------------- #


def _build_chat_completions_request(
    model_id: str,
    system_prompt: str | None,
    user_prompt: str,
    max_output_tokens: int,
    supports_temperature: bool,
    temperature: float,
) -> dict[str, Any]:
    """Build a request body for ``POST /v1/chat/completions``.

    Live shape (verified against ``mistral-large-3``)::

        {
            "model": "<model_id>",
            "messages": [
                {"role": "system", "content": "<system_prompt>"},  # omit if empty
                {"role": "user",   "content": "<user_prompt>"},
            ],
            "max_tokens": <max_output_tokens>,
            "temperature": <temperature>,                          # omit if not supported
        }

    Note: the field name is ``max_tokens`` (not ``max_completion_tokens``
    — the OpenAI cloud API uses the latter, but bedrock-mantle's
    chat_completions surface uses ``max_tokens``; verified live).
    """
    messages: list[dict[str, str]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    req: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_output_tokens,
    }
    if supports_temperature:
        req["temperature"] = temperature
    return req


def _parse_chat_completions_response(
    payload: dict[str, Any],
) -> tuple[str, dict[str, int]]:
    """Parse a chat_completions 200 payload to ``(text, usage)``.

    Live shape (verified against ``mistral-large-3``)::

        {
            "choices": [{"message": {"content": "<text>", "role": "assistant"}}],
            "usage": {"prompt_tokens": N, "completion_tokens": M, ...},
        }

    Mapping:
      * ``text = choices[0].message.content``
      * ``inputTokens = usage.prompt_tokens``
      * ``outputTokens = usage.completion_tokens``

    Raises:
        MantleSchemaDrift: ``choices`` missing / not a list, no
            ``message.content``, or content is not a string.
    """
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise MantleSchemaDrift(
            f"chat_completions payload missing non-empty 'choices' list "
            f"(got {type(choices).__name__})"
        )
    first = choices[0]
    if not isinstance(first, dict):
        raise MantleSchemaDrift(
            "chat_completions payload choices[0] is not a dict"
        )
    message = first.get("message")
    if not isinstance(message, dict):
        raise MantleSchemaDrift(
            "chat_completions payload choices[0].message is not a dict"
        )
    # Reasoning models on chat_completions surfaces (e.g. MiniMax M2,
    # GLM 4.7, Kimi K2.5) emit ``content: null`` while burning the
    # token budget on hidden reasoning. Treat null as empty text so
    # the response is "reachable" for permissive discovery instead of
    # raising MantleSchemaDrift. A non-string, non-None value still
    # raises (real schema drift).
    raw_text = message.get("content", "")
    if raw_text is None:
        text = ""
    elif isinstance(raw_text, str):
        text = raw_text
    else:
        raise MantleSchemaDrift(
            f"chat_completions payload choices[0].message.content is not a "
            f"string or null (got {type(raw_text).__name__})"
        )

    raw_usage = payload.get("usage", {})
    if not isinstance(raw_usage, dict):
        raw_usage = {}
    try:
        input_tokens = int(raw_usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(raw_usage.get("completion_tokens", 0) or 0)
    except (TypeError, ValueError):
        input_tokens = 0
        output_tokens = 0
    usage: dict[str, int] = {
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
    }
    return text, usage


# --------------------------------------------------------------------------- #
# Anthropic Messages API (LIVE)                                               #
# --------------------------------------------------------------------------- #


# Required constant per the bedrock-mantle anthropic surface (verified
# live — a request without it returns 400 "anthropic_version: Field
# required"). It was the only published value when these shapes were
# verified; re-verify if a future API revision surfaces a different value.
_ANTHROPIC_VERSION_BEDROCK = "bedrock-2023-05-31"


def _build_anthropic_messages_request(
    model_id: str,
    system_prompt: str | None,
    user_prompt: str,
    max_output_tokens: int,
    supports_temperature: bool,
    temperature: float,
) -> dict[str, Any]:
    """Build a request body for ``POST /anthropic/v1/messages``.

    Live shape (verified against ``anthropic.claude-opus-4-8``)::

        {
            "model": "<model_id>",
            "anthropic_version": "bedrock-2023-05-31",  # REQUIRED — 400 without it
            "max_tokens": <max_output_tokens>,
            "messages": [{"role": "user", "content": "<user_prompt>"}],
            "system": "<system_prompt>",                # top-level, omit if empty
            "temperature": <temperature>,                # omit if not supported
        }
    """
    req: dict[str, Any] = {
        "model": model_id,
        "anthropic_version": _ANTHROPIC_VERSION_BEDROCK,
        "max_tokens": max_output_tokens,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if system_prompt and system_prompt.strip():
        req["system"] = system_prompt
    if supports_temperature:
        req["temperature"] = temperature
    return req


def _parse_anthropic_messages_response(
    payload: dict[str, Any],
) -> tuple[str, dict[str, int]]:
    """Parse an anthropic_messages 200 payload to ``(text, usage)``.

    Live shape (verified against ``anthropic.claude-opus-4-8``)::

        {
            "content": [{"type": "text", "text": "<text>"}, ...],
            "usage": {"input_tokens": N, "output_tokens": M},
        }

    Mapping:
      * ``text`` = first content[] block with ``type == 'text'``, take
        ``.text``. Concatenates if multiple text blocks are present
        (defensive — single block is the common case).
      * ``inputTokens = usage.input_tokens``
      * ``outputTokens = usage.output_tokens``

    Raises:
        MantleSchemaDrift: ``content`` missing / not a list, or no
            ``type=='text'`` block found.
    """
    content = payload.get("content")
    if not isinstance(content, list):
        raise MantleSchemaDrift(
            f"anthropic_messages payload missing 'content' list "
            f"(got {type(content).__name__})"
        )
    texts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            txt = block.get("text", "")
            if isinstance(txt, str):
                texts.append(txt)
    # Permissive discovery: an anthropic response with no ``text`` block
    # (e.g. a tool-only response, or a thought-only block while the cap
    # truncates before the text fires) is "reachable" — treat as empty
    # text rather than schema drift. A truly malformed response still
    # tripped the ``content`` list-check above.
    text = "".join(texts)

    raw_usage = payload.get("usage", {})
    if not isinstance(raw_usage, dict):
        raw_usage = {}
    try:
        input_tokens = int(raw_usage.get("input_tokens", 0) or 0)
        output_tokens = int(raw_usage.get("output_tokens", 0) or 0)
    except (TypeError, ValueError):
        input_tokens = 0
        output_tokens = 0
    usage: dict[str, int] = {
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
    }
    return text, usage
