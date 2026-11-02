# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for unauthorized credential-expiry hint mapping.

Covers: the verbatim mantle 401 text maps to the credential hint; word-form
'unauthorized' maps; false-positive shapes do NOT; existing access-denied
inputs still route to their own hint (branch ordering).
"""

from app.services.metric_error_analyzer import _hint_for_error_text


def test_mantle_401_unauthorized_text_maps_to_credential_hint():
    """The verbatim mantle 401 text maps to the credential-refresh hint.

    httpx reports a mantle 401 as:
    "Client error '401 Unauthorized' for url 'https://bedrock-mantle...'"
    This should map to the credential-refresh hint, not the access-denied hint.
    """
    error_text = "Client error '401 Unauthorized' for url 'https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses'"
    hint = _hint_for_error_text(error_text)

    assert "credentials look expired" in hint.lower(), \
        f"Mantle 401 text should map to credential hint, got: {hint}"
    assert "aws sso login" in hint.lower(), \
        f"Expected credential-refresh remedy, got: {hint}"


def test_word_form_unauthorized_maps_to_credential_hint():
    """The word 'unauthorized' alone maps to credential-refresh hint.

    This is the broader trigger for mantle expiry cases that may not
    include the full '401 Unauthorized' pair.
    """
    error_text = "Request unauthorized - credential expired"
    hint = _hint_for_error_text(error_text)

    assert "credentials look expired" in hint.lower(), \
        f"Word-form 'unauthorized' should map to credential hint, got: {hint}"


def test_incidental_401_digits_do_not_false_positive():
    """Messages containing '401' incidentally do NOT map to credential hint.

    These are the false-positive shapes: token counts, case indices, IDs
    containing 401. The match must be anchored to the HTTP status shape,
    not bare digits.
    """
    false_positive_cases = [
        "token count 401 exceeded",
        "case index 401 failed with validation error",
        "job id ab401cd timeout after 300s",
    ]

    for error_text in false_positive_cases:
        hint = _hint_for_error_text(error_text)
        # Should fall through to the generic "inspect harness log" hint,
        # NOT the credential-refresh hint
        assert "credentials" not in hint.lower(), \
            f"Incidental '401' should not trigger credential hint: {error_text} -> {hint}"
        assert "inspect the harness log" in hint.lower(), \
            f"Incidental '401' should fall through to generic hint: {error_text} -> {hint}"


def test_access_denied_still_routes_to_model_access_hint():
    """Existing access-denied inputs still route to their own hint.

    The new unauthorized branch must not shadow the access-denied branch,
    which carries a different remedy (grant model access, not credential refresh).
    """
    access_denied_cases = [
        "AccessDeniedException: User is not authorized to perform bedrock:InvokeModel",
        "is not authorized to perform bedrock:InvokeModel on resource",
    ]

    for error_text in access_denied_cases:
        hint = _hint_for_error_text(error_text)

        # Should route to the access-denied hint (check model access)
        assert "bedrock model access" in hint.lower(), \
            f"Access-denied should route to model-access hint: {error_text} -> {hint}"
        # Should NOT route to credential-refresh hint
        assert "aws sso login" not in hint.lower(), \
            f"Access-denied should not be shadowed by credential hint: {error_text} -> {hint}"


def test_unauthorized_takes_precedence_over_generic_access_denied():
    """'unauthorized' word-form takes precedence over generic 'accessdenied'.

    If a message contains both 'unauthorized' and 'accessdenied' (unlikely but
    possible), the unauthorized branch should win because it's checked first.
    """
    error_text = "unauthorized: accessdenied for resource X"
    hint = _hint_for_error_text(error_text)

    # Should route to credential-refresh hint (unauthorized branch)
    assert "credentials look expired" in hint.lower(), \
        f"'unauthorized' should win when both present: {hint}"
