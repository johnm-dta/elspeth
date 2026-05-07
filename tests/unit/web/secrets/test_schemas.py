"""Regression tests for secrets request/response schemas.

The secrets API's response models serialize system-owned metadata —
secret names, scopes, resolvability flags.  The request-side schema
(``CreateSecretRequest``) is still a Tier 3 boundary, but it now rejects
unknown keys so malformed write payloads fail closed instead of being
silently normalized.

These tests verify both boundaries:
- request-side extra-key rejection for writes
- response-side strictness for system-owned metadata
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from elspeth.web.secrets.schemas import (
    CreateSecretRequest,
    CreateSecretResponse,
    SecretInventoryResponse,
    ValidateSecretResponse,
)


class TestCreateSecretRequest:
    def test_rejects_unknown_scope_field(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            CreateSecretRequest(
                name="API_KEY",
                value="hunter2",
                scope="server",  # type: ignore[call-arg]
            )

    def test_rejects_unknown_available_field(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            CreateSecretRequest(
                name="API_KEY",
                value="hunter2",
                available=True,  # type: ignore[call-arg]
            )


class TestSecretStrictCoercionRejected:
    def test_inventory_rejects_string_bool_available(self) -> None:
        with pytest.raises(ValidationError):
            SecretInventoryResponse(
                name="n",
                scope="user",
                available="true",  # type: ignore[arg-type]
                source_kind="env",
            )

    def test_inventory_rejects_int_for_source_kind(self) -> None:
        with pytest.raises(ValidationError):
            SecretInventoryResponse(
                name="n",
                scope="user",
                available=True,
                source_kind=42,  # type: ignore[arg-type]
            )

    def test_create_response_rejects_extra_available_field(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            CreateSecretResponse(name="n", scope="user", available=False)  # type: ignore[call-arg]

    def test_validate_response_rejects_string_bool_available(self) -> None:
        with pytest.raises(ValidationError):
            ValidateSecretResponse(name="n", available="yes")  # type: ignore[arg-type]


class TestSecretExtraFieldsRejected:
    def test_inventory_rejects_extra(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            SecretInventoryResponse(
                name="n",
                scope="user",
                available=True,
                source_kind="env",
                value="super-secret",  # type: ignore[call-arg]
            )

    def test_create_response_rejects_extra(self) -> None:
        """Regression guard — ``value`` must NEVER be silently accepted on
        a secret write-acknowledgement response.  ``extra="forbid"`` makes
        the class contract enforce what the docstring promises.
        """
        with pytest.raises(ValidationError, match="extra"):
            CreateSecretResponse(
                name="n",
                scope="user",
                value="super-secret",  # type: ignore[call-arg]
            )

    def test_validate_response_rejects_extra(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            ValidateSecretResponse(name="n", available=True, value="secret")  # type: ignore[call-arg]


class TestSecretResponseHappyPath:
    def test_inventory_with_defaults(self) -> None:
        resp = SecretInventoryResponse(name="api_key", scope="user", available=True)
        assert resp.source_kind == ""

    def test_create_response(self) -> None:
        resp = CreateSecretResponse(name="api_key", scope="user")
        assert resp.scope == "user"

    def test_validate_response(self) -> None:
        resp = ValidateSecretResponse(name="api_key", available=False)
        assert resp.available is False


class TestSecretScopeDomain:
    def test_inventory_rejects_invalid_scope(self) -> None:
        with pytest.raises(ValidationError, match="scope"):
            SecretInventoryResponse(
                name="api_key",
                scope="bogus",  # type: ignore[arg-type]
                available=True,
                source_kind="env",
            )

    def test_create_response_rejects_invalid_scope(self) -> None:
        with pytest.raises(ValidationError, match="scope"):
            CreateSecretResponse(
                name="api_key",
                scope="bogus",  # type: ignore[arg-type]
            )

    def test_inventory_schema_emits_scope_enum(self) -> None:
        scope_schema = SecretInventoryResponse.model_json_schema()["properties"]["scope"]

        assert scope_schema["enum"] == ["user", "server", "org"]

    def test_create_response_schema_emits_scope_enum(self) -> None:
        scope_schema = CreateSecretResponse.model_json_schema()["properties"]["scope"]

        assert scope_schema["enum"] == ["user", "server", "org"]


class TestSecretStrictnessViaJson:
    """Parallel coverage for the ``model_validate_json`` path.

    Pydantic 2.x applies ``strict=True`` / ``extra="forbid"`` uniformly
    across the constructor path and the JSON-parse path, but nothing in
    the other tests in this module would catch a future Pydantic release
    that decoupled the two.  For the secrets response models the
    no-value-on-way-out invariant is load-bearing: a JSON body
    ``{"name": "n", "scope": "user", "value": "..."}``
    must be rejected even if a future Pydantic version silently drops
    unknown fields for the JSON path but not for the constructor path.

    These tests pin the invariant from the JSON surface and fail loudly
    on any such drift.
    """

    def test_create_response_rejects_value_field_in_json(self) -> None:
        """No-value-on-way-out holds through the JSON-parse surface."""
        payload = '{"name": "n", "scope": "user", "value": "super-secret"}'
        with pytest.raises(ValidationError, match="extra"):
            CreateSecretResponse.model_validate_json(payload)

    def test_inventory_rejects_value_field_in_json(self) -> None:
        payload = '{"name": "n", "scope": "user", "available": true, "source_kind": "env", "value": "super-secret"}'
        with pytest.raises(ValidationError, match="extra"):
            SecretInventoryResponse.model_validate_json(payload)

    def test_validate_response_rejects_value_field_in_json(self) -> None:
        payload = '{"name": "n", "available": true, "value": "super-secret"}'
        with pytest.raises(ValidationError, match="extra"):
            ValidateSecretResponse.model_validate_json(payload)

    def test_create_response_rejects_available_field_in_json(self) -> None:
        payload = '{"name": "n", "scope": "user", "available": true}'
        with pytest.raises(ValidationError, match="extra"):
            CreateSecretResponse.model_validate_json(payload)

    def test_inventory_rejects_int_scope_in_json(self) -> None:
        """Strict mode rejects JSON int-for-string coercion on the scope field."""
        payload = '{"name": "n", "scope": 7, "available": true, "source_kind": "env"}'
        with pytest.raises(ValidationError):
            SecretInventoryResponse.model_validate_json(payload)
