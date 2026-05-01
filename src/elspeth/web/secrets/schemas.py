"""Pydantic request/response schemas for the secrets REST API.

SECURITY: No schema in this module may ever carry a plaintext secret value
in a response model.  ``CreateSecretRequest`` accepts a value on the way *in*;
``CreateSecretResponse`` deliberately omits it on the way *out*.

The response models inherit from ``_StrictResponse`` so that
``extra="forbid"`` mechanically enforces the no-value-on-the-way-out
promise: a future refactor that accidentally forwards a secret value
into the response constructor crashes instead of being silently emitted.
``strict=True`` additionally blocks type coercion on audit metadata.
The request model also rejects unknown keys so malformed secret writes
fail closed at the HTTP boundary instead of being silently normalized
into successful writes.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from elspeth.contracts.secrets import SecretScope, SecretUnavailabilityReason
from elspeth.web.validation import SECRET_NAME_MAX_LENGTH, SECRET_NAME_PATTERN, has_visible_content


class _StrictResponse(BaseModel):
    """Tier 1 base for secrets responses — no coercion, no extras."""

    model_config = ConfigDict(strict=True, extra="forbid")


class SecretInventoryResponse(_StrictResponse):
    """Public metadata for a secret reference -- NEVER includes the value.

    ``reason`` mirrors ``SecretInventoryItem.reason``: a closed-list
    structural failure mode populated when ``available`` is False.  The
    biconditional ``available ⟺ reason is None`` is enforced by
    ``_check_reason_invariant``; same discipline as the contract
    dataclass so the HTTP response cannot represent the operator-hostile
    "false-with-no-explanation" shape this field exists to eliminate.

    SECURITY: ``reason`` is typed as ``SecretUnavailabilityReason``
    (a ``Literal``), so no code path can interpolate env-var or
    candidate-secret content into the response — the type system
    enforces the audit-hygiene constraint.
    """

    name: str
    scope: SecretScope
    available: bool
    source_kind: str = ""
    reason: SecretUnavailabilityReason | None = None

    @model_validator(mode="after")
    def _check_reason_invariant(self) -> SecretInventoryResponse:
        if self.available and self.reason is not None:
            raise ValueError("reason must be None when available=True")
        if not self.available and self.reason is None:
            raise ValueError("reason is required when available=False")
        return self


class CreateSecretRequest(BaseModel):
    """Write-only request body for creating/updating a user-scoped secret."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=SECRET_NAME_MAX_LENGTH, pattern=SECRET_NAME_PATTERN)
    value: str = Field(min_length=1, max_length=65536)

    @field_validator("value")
    @classmethod
    def reject_invisible_only(cls, v: str) -> str:
        if not has_visible_content(v):
            raise ValueError("Secret value must contain at least one visible character")
        return v


class CreateSecretResponse(_StrictResponse):
    """Write-only acknowledgement -- NEVER includes the value."""

    name: str
    scope: SecretScope


class ValidateSecretResponse(_StrictResponse):
    """Existence check -- confirms whether a named secret is resolvable."""

    name: str
    available: bool
