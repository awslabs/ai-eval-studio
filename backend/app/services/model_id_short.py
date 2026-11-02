# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural short-name derivation for Bedrock model ids.

The prior implementations hardcoded provider lists and region prefixes
(``us|eu|apac|global|us-gov`` and ``anthropic|openai|bedrock|cohere|meta``).
That broke the moment a new provider arrived (xAI, Google Gemma, Mistral
Large, MiniMax, Kimi, GLM, Qwen, DeepSeek) and on every dotted version
id (``xai.grok-4.3``, ``openai.gpt-5.4``, ``openai.gpt-5.5``).

The structural rule is:
    1. Strip leading dot-segments until the first segment that contains
       a digit. That segment is where the model name starts (every
       Bedrock model id ends in a model-name-with-version segment).
    2. Strip a trailing ``-v\\d+(:\\d+)?`` version suffix.
    3. Strip a trailing ``-\\d{8,}`` date stamp.

Examples (all verified):
    xai.grok-4.3                               -> grok-4.3
    openai.gpt-5.4                             -> gpt-5.4
    openai.gpt-5.5                             -> gpt-5.5
    global.anthropic.claude-opus-4-8           -> claude-opus-4-8
    anthropic.claude-opus-4-7                  -> claude-opus-4-7
    us.anthropic.claude-sonnet-4-6-v1:0        -> claude-sonnet-4-6
    us.anthropic.claude-haiku-4-5-20251001-v1:0 -> claude-haiku-4-5
    us.amazon.nova-lite-v1:0                   -> nova-lite
    meta.llama3-70b-instruct-v1:0              -> llama3-70b-instruct
    google.gemma-3-27b-it                      -> gemma-3-27b-it
    deepseek.v3.2                              -> v3.2

The remainder is verbatim — dotted versions keep their dots.

This module has no dependencies and is duplicated unchanged between
``backend/app/services/`` and ``backend/app/templates/harness/src/models/``.
The parity test enforces byte-equality.
"""

from __future__ import annotations

import re

_VERSION_TAG = re.compile(r"-v\d+(:\d+)?$")
_DATE_TAG = re.compile(r"-\d{8,}$")
_DIGIT = re.compile(r"\d")


def short_model_id(model_id: str) -> str:
    """Return a clean display name for a Bedrock model id.

    Strips leading prefix segments (region/provider/profile) until the
    first segment carrying a digit, then strips a trailing ``-v\\d+(:\\d+)?``
    and ``-\\d{8,}`` date suffix. The remainder — including any internal
    dots from dotted version numbers — is preserved verbatim.

    Args:
        model_id: Full Bedrock model identifier.

    Returns:
        Short display name. If the id is dotless or no segment contains
        a digit, the input is returned unchanged with the version/date
        suffixes stripped. Empty input returns empty string.
    """
    if not model_id:
        return model_id

    parts = model_id.split(".")
    if len(parts) > 1:
        for i, segment in enumerate(parts):
            if _DIGIT.search(segment):
                short = ".".join(parts[i:])
                break
        else:
            short = parts[-1]
    else:
        short = model_id

    short = _VERSION_TAG.sub("", short)
    short = _DATE_TAG.sub("", short)
    return short
