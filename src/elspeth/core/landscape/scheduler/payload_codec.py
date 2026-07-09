"""Pure payload codec for durable scheduler work items.

Serialization/deserialization of the token row payload, the scrubbed
payload retained after terminalization, and the journal-item ->
``TokenInfo`` mapping that rides on the payload round-trip. Every function
here is a pure transformation with no engine access. Extracted from the
``TokenSchedulerRepository`` god class (filigree elspeth-ef9c36d767).
"""

from __future__ import annotations

import hashlib
import json

from elspeth.contracts import TokenInfo
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.scheduler import TokenWorkItem
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.canonical import canonical_json


def serialize_row_payload(row: PipelineRow) -> str:
    """Serialize the current token row and its contract for durable resume.

    Uses the type-preserving checkpoint serializer (NOT canonical_json):
    journal row payloads are re-driven as live PipelineRows on resume
    (PENDING_SINK re-drive, F1 barrier-buffer rebuild), so typed values
    (datetime, Decimal, date, time, bytes, UUID) must round-trip with
    full fidelity — canonical_json flattens them to bare strings.
    """
    # Deferred import: module-level would cycle —
    # core.checkpoint.__init__ → recovery → core.landscape.factory → this module.
    from elspeth.core.checkpoint.serialization import checkpoint_dumps

    return checkpoint_dumps(
        {
            "row": row.to_checkpoint_format(),
            "contract": row.contract.to_checkpoint_format(),
        }
    )


def deserialize_row_payload(row_payload_json: str) -> PipelineRow:
    """Restore a scheduler row payload written by ``serialize_row_payload``."""
    # Deferred import: see serialize_row_payload.
    from elspeth.core.checkpoint.serialization import checkpoint_loads

    try:
        payload = checkpoint_loads(row_payload_json)
    except json.JSONDecodeError as exc:
        raise AuditIntegrityError(f"Corrupt scheduler row payload JSON: {exc}") from exc
    if type(payload) is not dict:
        raise AuditIntegrityError(f"Corrupt scheduler row payload: expected object, got {type(payload).__name__}")

    try:
        row_checkpoint = payload["row"]
        contract_checkpoint = payload["contract"]
    except KeyError as exc:
        raise AuditIntegrityError(f"Corrupt scheduler row payload: missing {exc}. Available keys: {sorted(payload.keys())}") from exc
    if type(row_checkpoint) is not dict:
        raise AuditIntegrityError(f"Corrupt scheduler row payload: row must be object, got {type(row_checkpoint).__name__}")
    if type(contract_checkpoint) is not dict:
        raise AuditIntegrityError(f"Corrupt scheduler row payload: contract must be object, got {type(contract_checkpoint).__name__}")

    contract = SchemaContract.from_checkpoint(contract_checkpoint)
    return PipelineRow.from_checkpoint(row_checkpoint, {contract.version_hash(): contract})


def scrubbed_row_payload_json(anchor: str) -> str:
    """Return non-row scheduler payload retained after terminalization."""
    return canonical_json({"row_payload": "purged", "payload_hash": hashlib.sha256(anchor.encode()).hexdigest()})


def token_from_journal_item(
    item: TokenWorkItem,
    *,
    attempt_offset: int,
    resume_checkpoint_id: str | None,
) -> TokenInfo:
    """Rebuild a ``TokenInfo`` from a journal BLOCKED row.

    Shared by barrier intake and the aggregation/coalesce executors'
    ``restore_from_journal`` paths. The journal row is authoritative for the
    payload and token lineage. Resume callers supply audit-derived
    ``attempt_offset`` and ``resume_checkpoint_id``; normal-run follower
    handoffs use offset zero with no checkpoint provenance.

    Lives next to ``serialize_row_payload`` / ``deserialize_row_payload``
    because the payload round-trip is the heart of the mapping.
    """
    row_data = deserialize_row_payload(item.row_payload_json)
    return TokenInfo(
        row_id=item.row_id,
        token_id=item.token_id,
        row_data=row_data,
        branch_name=item.branch_name,
        fork_group_id=item.fork_group_id,
        join_group_id=item.join_group_id,
        expand_group_id=item.expand_group_id,
        resume_attempt_offset=attempt_offset,
        resume_checkpoint_id=resume_checkpoint_id,
    )
