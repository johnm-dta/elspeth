"""Pure validation for blob references retained by guided review snapshots."""

from __future__ import annotations

from uuid import UUID

from elspeth.contracts.errors import AuditIntegrityError


def validate_guided_reviewed_blob_ref(value: object) -> str:
    """Return a canonical UUID string or fail closed without echoing it."""
    if type(value) is not str:
        raise AuditIntegrityError("guided reviewed source blob_ref must be a canonical UUID string")
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise AuditIntegrityError("guided reviewed source blob_ref must be a canonical UUID string") from exc
    if str(parsed) != value:
        raise AuditIntegrityError("guided reviewed source blob_ref must be a canonical UUID string")
    return value
