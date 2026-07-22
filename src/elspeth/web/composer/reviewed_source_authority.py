"""Private verification for exact guided reviewed-source reuse."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, select

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.composer.pipeline_proposal import reviewed_anchor_hash
from elspeth.web.composer.tools._common import ReviewedSourceAuthority
from elspeth.web.sessions.models import blobs_table, sessions_table

_BLOB_PATH_PREFIX = "blob:"
_BLOB_PATH_KEYS = frozenset({"path", "file"})


def _canonical_blob_id(value: object, *, field_name: str) -> str:
    if type(value) is not str:
        raise AuditIntegrityError(f"{field_name} must be a canonical UUID string")
    try:
        blob_id = UUID(value)
    except ValueError as exc:
        raise AuditIntegrityError(f"{field_name} must be a canonical UUID string") from exc
    if str(blob_id) != value:
        raise AuditIntegrityError(f"{field_name} must be a canonical UUID string")
    return value


def resolve_reviewed_source_authority(
    *,
    engine: Engine | None,
    session_id: str,
    user_id: str | None,
    reviewed_facts: Mapping[str, Any],
    expected_reviewed_anchor_hash: str,
) -> ReviewedSourceAuthority | None:
    """Verify current blob custody before granting private source authority.

    Public proposal arguments remain unchanged.  Exact ``blob:<uuid>``
    sentinels are resolved only inside the candidate boundary through the
    returned private mapping.
    """

    if reviewed_anchor_hash(reviewed_facts) != expected_reviewed_anchor_hash:
        raise AuditIntegrityError("reviewed source authority does not match the current reviewed anchor")
    sources = reviewed_facts["reviewed_sources"] if "reviewed_sources" in reviewed_facts else None
    if not isinstance(sources, Mapping):
        return None
    if type(session_id) is not str or not session_id:
        raise AuditIntegrityError("reviewed source authority requires a non-empty session_id")
    if type(user_id) is not str or not user_id:
        raise AuditIntegrityError("reviewed source authority requires a non-empty user_id")

    if engine is None:
        raise AuditIntegrityError("reviewed source authority requires the session database")

    source_blob_ids: dict[str, str] = {}
    sentinel_blob_ids: dict[str, str] = {}
    raw_storage_paths: dict[str, tuple[str, ...]] = {}
    for stable_id, raw_source in sources.items():
        if type(stable_id) is not str or not stable_id or not isinstance(raw_source, Mapping):
            raise AuditIntegrityError("reviewed source authority contains a malformed source record")
        options = raw_source["options"] if "options" in raw_source else None
        if not isinstance(options, Mapping):
            raise AuditIntegrityError("reviewed source authority contains malformed source options")
        referenced_ids: set[str] = set()
        if "blob_ref" in options:
            referenced_ids.add(
                _canonical_blob_id(
                    options["blob_ref"],
                    field_name=f"reviewed_sources[{stable_id!r}].options.blob_ref",
                )
            )
        for option_name in _BLOB_PATH_KEYS:
            option_value = options[option_name] if option_name in options else None
            if type(option_value) is str and option_value.startswith(_BLOB_PATH_PREFIX):
                blob_id = _canonical_blob_id(
                    option_value.removeprefix(_BLOB_PATH_PREFIX),
                    field_name=f"reviewed_sources[{stable_id!r}].options.{option_name}",
                )
                referenced_ids.add(blob_id)
                sentinel_blob_ids[option_value] = blob_id
            elif type(option_value) is str:
                raw_storage_paths[stable_id] = (*raw_storage_paths.get(stable_id, ()), option_value)
        if len(referenced_ids) > 1:
            raise AuditIntegrityError("reviewed source blob custody fields disagree")
        if referenced_ids:
            source_blob_ids[stable_id] = next(iter(referenced_ids))

    verified_storage_by_blob_id: dict[str, str] = {}
    with engine.connect() as conn:
        session_owner = conn.execute(select(sessions_table.c.user_id).where(sessions_table.c.id == session_id)).scalar_one_or_none()
        if session_owner != user_id:
            raise AuditIntegrityError("reviewed blob authority is not owned by the current user session")
        for blob_id in sorted(set(source_blob_ids.values())):
            row = conn.execute(
                select(
                    blobs_table.c.session_id,
                    blobs_table.c.status,
                    blobs_table.c.storage_path,
                ).where(blobs_table.c.id == blob_id)
            ).first()
            if row is None:
                raise AuditIntegrityError("reviewed blob authority references a missing blob")
            if row.session_id != session_id:
                raise AuditIntegrityError("reviewed blob authority references a blob owned by another session")
            if row.status != "ready":
                raise AuditIntegrityError("reviewed blob authority references a blob that is not ready")
            if type(row.storage_path) is not str or not row.storage_path:
                raise AuditIntegrityError("reviewed blob authority has a malformed storage path")
            verified_storage_by_blob_id[blob_id] = row.storage_path

    for stable_id, paths in raw_storage_paths.items():
        source_blob_id = source_blob_ids.get(stable_id)
        if source_blob_id is None:
            # A non-blob path is authorized only by the exact reviewed source
            # binding at the candidate boundary.  Session ownership was still
            # verified above, and the normal credential/plugin/S2 path checks
            # remain mandatory after that binding matches.
            continue
        if any(path != verified_storage_by_blob_id[source_blob_id] for path in paths):
            raise AuditIntegrityError("reviewed source raw storage path differs from its owned ready blob")

    return ReviewedSourceAuthority(
        session_id=session_id,
        reviewed_anchor_hash=expected_reviewed_anchor_hash,
        reviewed_sources=sources,
        verified_blob_paths={sentinel: verified_storage_by_blob_id[blob_id] for sentinel, blob_id in sentinel_blob_ids.items()},
    )
