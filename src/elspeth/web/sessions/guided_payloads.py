"""Prepare and verify guided JSON payloads outside SQL transactions."""

from __future__ import annotations

import hmac
from collections.abc import Mapping

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_freeze
from elspeth.contracts.hashing import canonical_json
from elspeth.contracts.payload_store import PayloadStore
from elspeth.web.sessions.protocol import GuidedJsonPayloadPurpose, PreparedGuidedJsonPayload


def prepare_guided_json_payload(
    payload_store: PayloadStore,
    *,
    purpose: GuidedJsonPayloadPurpose,
    payload: Mapping[str, object],
) -> PreparedGuidedJsonPayload:
    """Freeze, store, retrieve, and byte-verify one content-addressed payload."""

    if not isinstance(payload_store, PayloadStore):
        raise TypeError("payload_store must implement PayloadStore")
    snapshot = deep_freeze(payload)
    if not isinstance(snapshot, Mapping):
        raise TypeError("payload must freeze to a mapping")
    canonical = canonical_json(snapshot).encode("utf-8")
    payload_id = payload_store.store(canonical)
    if type(payload_id) is not str or len(payload_id) != 64:
        raise AuditIntegrityError("guided payload store returned a malformed content id")
    retrieved = payload_store.retrieve(payload_id)
    if not hmac.compare_digest(retrieved, canonical):
        raise AuditIntegrityError("guided payload store retrieval differs from the stored canonical bytes")
    return PreparedGuidedJsonPayload(payload_id=payload_id, purpose=purpose, payload=snapshot)


def verify_guided_json_payloads(
    payload_store: PayloadStore | None,
    payloads: tuple[PreparedGuidedJsonPayload, ...],
) -> None:
    """Re-read each referenced payload immediately before SQL settlement."""

    if not payloads:
        return
    if payload_store is None or not isinstance(payload_store, PayloadStore):
        raise AuditIntegrityError("guided payload settlement requires a configured PayloadStore")
    for payload in payloads:
        expected = canonical_json(payload.payload).encode("utf-8")
        retrieved = payload_store.retrieve(payload.payload_id)
        if not hmac.compare_digest(retrieved, expected):
            raise AuditIntegrityError("guided payload store content differs from the prepared payload")


__all__ = ["prepare_guided_json_payload", "verify_guided_json_payloads"]
