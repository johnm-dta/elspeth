"""Closed proposal-to-blob reference contract and custody validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Connection

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.sessions.models import blobs_table, composition_proposals_table

_TOP_LEVEL_BLOB_TOOLS = frozenset(
    {
        "set_source_from_blob",
        "update_blob",
        "wire_blob_inline_ref",
    }
)
_BLOB_REFERENCE_TOOLS = _TOP_LEVEL_BLOB_TOOLS | {"set_pipeline"}


def proposal_blob_reference_ids(tool_name: str, arguments: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract authoritative blob references from the closed tool allowlist.

    This deliberately does not recursively scan arbitrary proposal JSON. Only
    tool-owned schema positions can create a custody/retention edge.
    """
    if tool_name not in _BLOB_REFERENCE_TOOLS:
        return ()

    if tool_name == "set_pipeline":
        source = arguments["source"] if "source" in arguments else None
        if source is None:
            return ()
        if type(source) is not dict:
            raise ValueError("set_pipeline source must be a mapping when present")
        value = source["blob_id"] if "blob_id" in source else None
        field_name = "source.blob_id"
    else:
        value = arguments["blob_id"] if "blob_id" in arguments else None
        field_name = "blob_id"

    if value is None:
        return ()
    if type(value) is not str or not value:
        raise ValueError(f"{tool_name} {field_name} must be a non-empty string when present")
    return (value,)


def validate_proposal_blob_references(
    conn: Connection,
    *,
    session_id: str,
    tool_name: str,
    arguments: Mapping[str, Any],
) -> None:
    """Require every authoritative proposal blob to be owned and ready."""
    for blob_id in proposal_blob_reference_ids(tool_name, arguments):
        row = conn.execute(select(blobs_table.c.session_id, blobs_table.c.status).where(blobs_table.c.id == blob_id)).first()
        if row is None:
            raise ValueError(f"Proposal references blob {blob_id!r}, but that blob does not exist")
        if row.session_id != session_id:
            raise ValueError(f"Proposal references blob {blob_id!r}, but it is not owned by this session")
        if row.status != "ready":
            raise ValueError(f"Proposal references blob {blob_id!r}, but its status is {row.status!r}, not 'ready'")


def pending_proposal_reference_id(conn: Connection, *, session_id: str, blob_id: str) -> str | None:
    """Return the pending proposal retaining a blob, if any."""
    rows = conn.execute(
        select(
            composition_proposals_table.c.id,
            composition_proposals_table.c.tool_name,
            composition_proposals_table.c.arguments_json,
        ).where(
            composition_proposals_table.c.session_id == session_id,
            composition_proposals_table.c.status == "pending",
            composition_proposals_table.c.tool_name.in_(_BLOB_REFERENCE_TOOLS),
        )
    ).fetchall()
    for proposal_id, tool_name, arguments_json in rows:
        if type(arguments_json) is not dict:
            raise AuditIntegrityError(
                f"Tier 1: pending proposal {proposal_id} arguments_json is {type(arguments_json).__name__}, expected dict"
            )
        try:
            references = proposal_blob_reference_ids(tool_name, arguments_json)
        except ValueError as exc:
            raise AuditIntegrityError(f"Tier 1: pending proposal {proposal_id} has malformed blob authority: {exc}") from exc
        if blob_id in references:
            return str(proposal_id)
    return None
