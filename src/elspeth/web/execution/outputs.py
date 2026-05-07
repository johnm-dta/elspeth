"""Run outputs manifest loader.

Distinct from ``elspeth.web.execution.diagnostics``: where diagnostics
returns a *bounded* operator-UI projection (capped at
``_ARTIFACT_PREVIEW_LIMIT = 20``), this module returns the **full**
artifact list for a run — every sink-write the engine performed.

Audience: the eval harness (``finalize_scenario.sh``), retroactive
backfill tooling, and any downstream evidence-retrieval flow that must
not silently drop artefacts beyond a UI preview cap.

Tier 1 read discipline (CLAUDE.md): the ``artifacts`` table is full-trust
data — read fields directly, no coercion. Filesystem state (``exists_now``)
is checked at endpoint-call time and is allowed to be ``False``: a purged
or moved file is a fact, not an error.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.engine.url import make_url

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import artifacts_table
from elspeth.web.config import WebSettings
from elspeth.web.execution.discard_summary import _sqlite_database_file_missing
from elspeth.web.execution.schemas import RunOutputArtifact, RunOutputsResponse

_FILE_URI_PREFIX = "file://"


class RunOutputsAuditUnavailableError(RuntimeError):
    """Raised when the full output manifest cannot read its audit source."""

    def __init__(self, *, landscape_run_id: str, landscape_url: str) -> None:
        parsed = make_url(landscape_url)
        if parsed.drivername.startswith("sqlite"):
            audit_location = parsed.database or landscape_url
        else:
            audit_location = parsed.render_as_string(hide_password=True)
        self.landscape_run_id = landscape_run_id
        self.audit_location = audit_location
        super().__init__(f"Run outputs audit database is unavailable for landscape_run_id={landscape_run_id!r} at {audit_location!r}")


def path_or_uri_to_filesystem_path(path_or_uri: str) -> Path | None:
    """Return a ``Path`` for filesystem-backed artefacts; ``None`` for
    object-store URIs (``azure://``, ``dataverse://``, …).

    Sinks register filesystem outputs as either an absolute path or a
    ``file://`` URI; either form needs to be tested with ``Path.exists()``.
    """
    if path_or_uri.startswith(_FILE_URI_PREFIX):
        return Path(path_or_uri[len(_FILE_URI_PREFIX) :])
    if "://" in path_or_uri:
        return None
    return Path(path_or_uri)


def load_run_outputs_from_db(
    db: LandscapeDB,
    *,
    run_id: str,
    landscape_run_id: str,
) -> RunOutputsResponse:
    """Read every sink-write artefact for a run and return the full manifest."""
    stmt = (
        select(
            artifacts_table.c.artifact_id,
            artifacts_table.c.sink_node_id,
            artifacts_table.c.artifact_type,
            artifacts_table.c.path_or_uri,
            artifacts_table.c.content_hash,
            artifacts_table.c.size_bytes,
            artifacts_table.c.created_at,
        )
        .where(artifacts_table.c.run_id == landscape_run_id)
        .order_by(artifacts_table.c.created_at.asc(), artifacts_table.c.artifact_id.asc())
    )
    artifacts: list[RunOutputArtifact] = []
    with db.read_only_connection() as conn:
        for row in conn.execute(stmt):
            artifact = RunOutputArtifact(
                artifact_id=row.artifact_id,
                sink_node_id=row.sink_node_id,
                artifact_type=row.artifact_type,
                path_or_uri=row.path_or_uri,
                content_hash=row.content_hash,
                size_bytes=row.size_bytes,
                created_at=row.created_at,
                exists_now=False,
            )
            fs_path = path_or_uri_to_filesystem_path(artifact.path_or_uri)
            exists_now = fs_path.exists() if fs_path is not None else False
            artifacts.append(artifact.model_copy(update={"exists_now": exists_now}))
    return RunOutputsResponse(
        run_id=run_id,
        landscape_run_id=landscape_run_id,
        artifacts=artifacts,
    )


def load_run_outputs_for_settings(
    settings: WebSettings,
    *,
    run_id: str,
    landscape_run_id: str,
) -> RunOutputsResponse:
    """Settings-driven variant — opens the configured Landscape DB and
    delegates to :func:`load_run_outputs_from_db`.

    Mirrors the loader/factory split in
    ``elspeth.web.execution.diagnostics``.
    """
    landscape_url = settings.get_landscape_url()
    if _sqlite_database_file_missing(landscape_url):
        raise RunOutputsAuditUnavailableError(landscape_run_id=landscape_run_id, landscape_url=landscape_url)
    with LandscapeDB.from_url(
        landscape_url,
        passphrase=settings.landscape_passphrase,
        create_tables=False,
    ) as db:
        return load_run_outputs_from_db(db, run_id=run_id, landscape_run_id=landscape_run_id)
