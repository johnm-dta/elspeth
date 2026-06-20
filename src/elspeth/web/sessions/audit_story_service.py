"""Read-side service for run audit-story projections."""

from __future__ import annotations

from sqlalchemy import select

from elspeth.core.canonical import stable_hash
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
from elspeth.web.sessions.audit_story_models import RunAuditStoryResponse


class AuditStoryIntegrityError(RuntimeError):
    """Raised when Landscape cannot provide a complete audit-story projection."""


class AuditStoryService:
    """Aggregates audit-story fields from real Landscape rows."""

    def __init__(self, db: LandscapeDB) -> None:
        self._db = db

    def get_run_audit_story(
        self,
        landscape_run_id: str,
        *,
        public_run_id: str,
        session_id: str,
    ) -> RunAuditStoryResponse:
        with self._db.read_only_connection() as conn:
            run = conn.execute(select(runs_table).where(runs_table.c.run_id == landscape_run_id)).first()
            if run is None:
                raise AuditStoryIntegrityError(f"Landscape run {landscape_run_id!r} not found")

            hashes = tuple(
                row.source_data_hash
                for row in conn.execute(
                    select(rows_table.c.source_data_hash)
                    .where(rows_table.c.run_id == landscape_run_id)
                    .distinct()
                    .order_by(rows_table.c.source_data_hash)
                )
            )
            source_data_hash = _coalesce_run_source_hashes(hashes, landscape_run_id=landscape_run_id)

            node_rows = tuple(
                conn.execute(
                    select(nodes_table.c.plugin_name, nodes_table.c.plugin_version)
                    .where(nodes_table.c.run_id == landscape_run_id)
                    .order_by(nodes_table.c.sequence_in_pipeline.asc(), nodes_table.c.node_id.asc())
                )
            )
            if not node_rows:
                raise AuditStoryIntegrityError(f"Landscape run {landscape_run_id!r} has no plugin version rows")

        seeded_from_cache = run.seeded_from_cache
        if type(seeded_from_cache) is not bool:
            raise AuditStoryIntegrityError(f"Landscape run {landscape_run_id!r} has non-bool seeded_from_cache={seeded_from_cache!r}")
        if run.llm_call_count is None:
            raise AuditStoryIntegrityError(f"Landscape run {landscape_run_id!r} has NULL llm_call_count")
        if seeded_from_cache and run.cache_key is None:
            raise AuditStoryIntegrityError(f"Landscape cache replay {landscape_run_id!r} has NULL cache_key")

        return RunAuditStoryResponse(
            run_id=public_run_id,
            session_id=session_id,
            llm_call_count=run.llm_call_count,
            source_data_hash=source_data_hash,
            started_at=run.started_at,
            plugin_versions={row.plugin_name: row.plugin_version for row in node_rows},
            seeded_from_cache=seeded_from_cache,
            cache_key=run.cache_key,
        )


def _coalesce_run_source_hashes(source_hashes: tuple[str, ...], *, landscape_run_id: str) -> str:
    if not source_hashes:
        raise AuditStoryIntegrityError(f"Landscape run {landscape_run_id!r} has no source_data_hash rows")
    if len(source_hashes) == 1:
        return source_hashes[0]
    return stable_hash({"source_data_hashes": list(source_hashes)})
