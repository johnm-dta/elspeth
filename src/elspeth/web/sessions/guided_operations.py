"""Canonical codecs for retry-safe guided operations."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.hashing import stable_hash
from elspeth.web.sessions.protocol import GuidedOperationKind

_GUIDED_OPERATION_REQUEST_SCHEMA = "guided-operation-request.v1"


def guided_operation_request_hash(*, session_id: UUID, kind: GuidedOperationKind, request: BaseModel) -> str:
    """Bind one strict request DTO to a session, excluding its retry id.

    Defaults and explicit ``None`` values are materialized so omitted and
    explicit-default requests share one canonical replay identity. The client
    operation id is transport retry state, not request semantics.
    """

    config = type(request).model_config
    if config.get("strict") is not True or config.get("extra") != "forbid":
        raise AuditIntegrityError("Guided operation hashing requires a strict, extra-forbid request DTO")
    if "operation_id" not in type(request).model_fields:
        raise AuditIntegrityError("Guided operation request DTO is missing operation_id")
    normalized: dict[str, Any] = request.model_dump(
        mode="json",
        exclude={"operation_id"},
        exclude_unset=False,
        exclude_defaults=False,
        exclude_none=False,
    )
    return stable_hash(
        {
            "schema": _GUIDED_OPERATION_REQUEST_SCHEMA,
            "session_id": str(session_id),
            "kind": kind,
            "request": normalized,
        }
    )
