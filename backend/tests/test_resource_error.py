# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ResourceError serialization and the error-code catalog."""

import pytest

from app.services.resource_error import Codes, ResourceError


class TestSerialization:
    """Round-trip to_dict / from_dict."""

    def test_minimal_required_fields(self):
        err = ResourceError(code="harness.subprocess_failed", message="exit 1")
        data = err.to_dict()
        assert data["code"] == "harness.subprocess_failed"
        assert data["message"] == "exit 1"
        assert data["hint"] is None
        assert data["details"] is None
        assert data["at"].endswith("Z")

    def test_full_payload_roundtrip(self):
        err = ResourceError(
            code=Codes.BEDROCK_TOKEN_EXPIRED,
            message="AWS session expired",
            hint="Run `aws sso login`.",
            details={"first_error": "ExpiredTokenException", "affected": 208},
        )
        restored = ResourceError.from_dict(err.to_dict())
        assert restored == err

    def test_from_dict_tolerates_missing_optional_fields(self):
        restored = ResourceError.from_dict({
            "code": "eval.no_test_cases",
            "message": "dataset was empty",
        })
        assert restored.hint is None
        assert restored.details is None
        # ``at`` auto-filled when absent
        assert restored.at.endswith("Z")

    def test_from_dict_raises_on_missing_required(self):
        with pytest.raises(KeyError):
            ResourceError.from_dict({"message": "no code here"})

    def test_non_serializable_details_rejected_at_construction(self):
        """``details`` values that can't go through ``json.dumps`` must
        fail loudly at the call site — otherwise the state-save later
        in the orchestrator silently crashes the loop."""
        from pathlib import Path
        with pytest.raises(ValueError, match="JSON-serializable"):
            ResourceError(
                code=Codes.HARNESS_SUBPROCESS_FAILED,
                message="boom",
                details={"path": Path("/tmp/x")},  # Path isn't JSON-serializable
            )

    def test_details_none_is_valid(self):
        ResourceError(code=Codes.EVAL_NO_TEST_CASES, message="empty")

    def test_details_with_primitives_is_valid(self):
        ResourceError(
            code=Codes.HARNESS_SUBPROCESS_FAILED,
            message="exit",
            details={"exit_code": 1, "stderr": "fail", "retries": [1, 2, 3]},
        )


class TestCodes:
    """Catalog sanity — stable namespaced strings only."""

    def test_every_constant_follows_namespace_dot_slug_shape(self):
        for name in dir(Codes):
            if name.startswith("_"):
                continue
            value = getattr(Codes, name)
            assert isinstance(value, str)
            assert "." in value, f"{name}={value!r} must be namespaced"
            ns, _, slug = value.partition(".")
            assert ns and slug
            assert value == value.lower()
            assert " " not in value

    def test_catalog_has_no_duplicate_values(self):
        values = [
            getattr(Codes, n) for n in dir(Codes) if not n.startswith("_")
        ]
        # Filter to strings (skip methods / other descriptors defensively)
        values = [v for v in values if isinstance(v, str)]
        assert len(values) == len(set(values))
