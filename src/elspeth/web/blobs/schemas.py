"""Pydantic request/response models for blob API endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from elspeth.web.blobs.protocol import (
    AllowedMimeType,
    BlobCreator,
    BlobStatus,
)
from elspeth.web.blobs.service import sanitize_filename


class _StrictResponse(BaseModel):
    """Tier 1 base for blob responses — no coercion, no extras.

    ``extra="forbid"`` is load-bearing here: it mechanically enforces the
    "storage_path is never included" promise in ``BlobMetadataResponse``.
    A future refactor that accidentally forwards ``record.storage_path``
    into the response constructor crashes rather than leaking the
    internal filesystem path to clients.
    """

    model_config = ConfigDict(strict=True, extra="forbid")


class BlobMetadataResponse(_StrictResponse):
    """Response for blob metadata endpoints.

    storage_path is never included — it's an internal implementation detail.

    The narrowed ``mime_type``, ``created_by``, and ``status`` types give
    typed API consumers exhaustive-match support: a ``match body.status``
    block across (ready, pending, error) is statically checkable, and
    drift from the DB CHECK constraints is caught at API schema time
    rather than by the client's runtime handling.
    """

    id: str
    session_id: str
    filename: str
    mime_type: AllowedMimeType
    size_bytes: int
    content_hash: str | None
    created_at: datetime
    created_by: BlobCreator
    source_description: str | None = None
    status: BlobStatus


class CreateInlineBlobRequest(BaseModel):
    """Request body for creating a blob from inline content (JSON body).

    This is the Tier 3 trust boundary for inline blob creation — every
    field is validated here so the route layer never has to coerce or
    translate malformed input into HTTP errors.

    - ``filename`` is run through :func:`sanitize_filename`, which rejects
      empty/``.``/``..`` names and strips path-traversal components.  A
      failure surfaces as a 422 via FastAPI's ``RequestValidationError``.
    - ``mime_type`` is a closed ``Literal`` over the allowed set so the
      caller cannot declare an unsupported type.  Default remains
      ``text/plain`` to preserve the previous ergonomic default.
    - ``extra="forbid"`` rejects unknown keys.  Previously a caller who
      sent ``content_type`` (the old field name) or ``mime-type`` got a
      silent fallback to the default MIME — now they get a 422.
    """

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1)
    content: str
    mime_type: AllowedMimeType = "text/plain"

    @field_validator("filename")
    @classmethod
    def _validate_filename(cls, value: str) -> str:
        return sanitize_filename(value)
