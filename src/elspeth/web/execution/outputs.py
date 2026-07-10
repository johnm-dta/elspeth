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

import hashlib
from pathlib import Path
from urllib.parse import unquote

from sqlalchemy import select
from sqlalchemy.engine.url import make_url

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import artifacts_table
from elspeth.web.config import WebSettings
from elspeth.web.execution.discard_summary import _sqlite_database_file_missing
from elspeth.web.execution.schemas import RunOutputArtifact, RunOutputArtifactStorageKind, RunOutputsResponse
from elspeth.web.paths import allowed_sink_directories

_FILE_URI_PREFIX = "file://"


def _is_path_in_sink_allowlist(fs_path: Path, data_dir: str | Path, *, session_id: str | None) -> bool:
    """Mirror of the read-side guard in the ``/content`` endpoint.

    Returns True iff ``fs_path`` resolves to a location inside one of
    the canonical sink directories — ``data_dir/outputs`` or the owning
    session's ``data_dir/blobs/<session_id>`` subtree (elspeth-bdc17cfdb1).
    Used to decide whether the UI may surface a Download button — a
    sink that wrote outside the allowlist produces a real artefact
    record but the download endpoint will refuse to serve it.
    ``session_id`` is the run's owning session; ``None`` fails closed so a
    blob-directory artefact never reports downloadable without session
    identity.
    """
    try:
        resolved = fs_path.resolve()
    except OSError:
        return False
    allowed = allowed_sink_directories(str(data_dir), session_id=session_id)
    return any(resolved.is_relative_to(base) for base in allowed)


def _classify_storage_kind(
    fs_paths: tuple[Path, ...] | None,
    data_dir: str | Path | None,
    *,
    payload_root: Path | None = None,
) -> RunOutputArtifactStorageKind:
    """Classify a file-backed artifact's path against elspeth's real
    storage layouts, so the UI can tell "internal opaque storage" from
    "a path a sink was configured to write to" — replaces a frontend
    regex heuristic that matched a layout no repo code actually
    produces (elspeth-52af16f9ae).

    Recognised layouts, matched by directory (not by filename shape):

    * ``{data_dir}/blobs/...``    — composer blob store; see
      ``_blob_storage_path`` in ``web/composer/tools/blobs.py``.
    * ``payload_root`` (default ``{data_dir}/payloads``) — content-
      addressed payload store; see ``FilesystemPayloadStore`` in
      ``core/payload_store.py``. Settings-driven callers pass
      ``WebSettings.get_payload_store_path()`` so a configured
      ``payload_store_path`` override is honoured.
    * ``{data_dir}/outputs/...``  — the canonical sink output directory.

    Anything else — a user-configured sink path outside those three
    directories, an object-store URI (``fs_paths=None``), or a legacy
    caller with no configured ``data_dir`` — classifies as ``"unknown"``:
    the safe default that does not assert internal-storage status it
    cannot verify.

    Classification is independent of on-disk existence: ``Path.resolve()``
    does not require the target to exist, so a purged blob-store path
    still classifies as ``"blob"`` rather than silently degrading to
    ``"unknown"`` (which would leak the raw path back into the UI via the
    exact fallback this discriminator exists to close).
    """
    if fs_paths is None or data_dir is None:
        return "unknown"
    base = Path(data_dir).resolve()
    payloads = payload_root.resolve() if payload_root is not None else base / "payloads"
    for fs_path in fs_paths:
        try:
            resolved = fs_path.resolve()
        except OSError:
            continue
        if resolved.is_relative_to(base / "blobs"):
            return "blob"
        if resolved.is_relative_to(payloads):
            return "payload"
        if resolved.is_relative_to(base / "outputs"):
            return "sink_file"
    return "unknown"


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


def filesystem_path_candidates(path_or_uri: str) -> tuple[Path, ...] | None:
    """Return filesystem path candidates for file-backed artefacts.

    New ``file://`` rows percent-encode URI delimiter characters in literal
    filenames, so the decoded path is the canonical filesystem spelling.
    Historical rows used raw string concatenation; keep the raw spelling as a
    fallback so old filenames containing literal percent escapes still resolve.
    """
    if path_or_uri.startswith(_FILE_URI_PREFIX):
        raw_path = Path(path_or_uri[len(_FILE_URI_PREFIX) :])
        decoded_path = Path(unquote(path_or_uri[len(_FILE_URI_PREFIX) :]))
        if decoded_path == raw_path:
            return (raw_path,)
        return (decoded_path, raw_path)
    if "://" in path_or_uri:
        return None
    return (Path(path_or_uri),)


def path_or_uri_to_filesystem_path(path_or_uri: str) -> Path | None:
    """Return a ``Path`` for filesystem-backed artefacts; ``None`` for
    object-store URIs (``azure://``, ``dataverse://``, …).

    Sinks register filesystem outputs as either an absolute path or a
    ``file://`` URI; either form needs to be tested with ``Path.exists()``.
    """
    candidates = filesystem_path_candidates(path_or_uri)
    if candidates is None:
        return None
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def hash_and_size_of_file(path: Path) -> tuple[str, int]:
    """Return ``(sha256_hex, byte_size)`` of ``path`` by streaming its bytes.

    ``content_hash`` recorded by every file-streamable sink is the SHA-256 of
    the WHOLE file (JSONSink hashes the whole file; CSVSink re-seeds its hasher
    from the existing file content in append mode), so this whole-file digest is
    directly comparable to the audited ``content_hash``. Used by the content
    endpoint to verify on-disk bytes against the audit record before streaming.
    """
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def load_run_outputs_from_db(
    db: LandscapeDB,
    *,
    run_id: str,
    landscape_run_id: str,
    data_dir: str | Path | None = None,
    session_id: str | None = None,
    payload_root: Path | None = None,
) -> RunOutputsResponse:
    """Read every sink-write artefact for a run and return the full manifest.

    ``data_dir`` is needed to compute the per-artifact ``downloadable``
    flag — it parameterises the sink-allowlist check that the
    ``/content`` endpoint enforces. When omitted (legacy callers, eval
    harness, tests that don't care), every artefact reports
    ``downloadable=False``: a safe degradation that matches "the UI
    can't trust this server to serve bytes."
    """
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
            fs_paths = filesystem_path_candidates(row.path_or_uri)
            exists_now = any(fs_path.exists() for fs_path in fs_paths) if fs_paths is not None else False
            downloadable = (
                row.artifact_type == "file"
                and exists_now
                and data_dir is not None
                and fs_paths is not None
                and any(fs_path.exists() and _is_path_in_sink_allowlist(fs_path, data_dir, session_id=session_id) for fs_path in fs_paths)
            )
            storage_kind = _classify_storage_kind(fs_paths, data_dir, payload_root=payload_root)
            artifacts.append(
                RunOutputArtifact(
                    artifact_id=row.artifact_id,
                    sink_node_id=row.sink_node_id,
                    artifact_type=row.artifact_type,
                    path_or_uri=row.path_or_uri,
                    content_hash=row.content_hash,
                    size_bytes=row.size_bytes,
                    created_at=row.created_at,
                    exists_now=exists_now,
                    downloadable=downloadable,
                    storage_kind=storage_kind,
                )
            )
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
    session_id: str | None = None,
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
        read_only=True,
    ) as db:
        return load_run_outputs_from_db(
            db,
            run_id=run_id,
            landscape_run_id=landscape_run_id,
            data_dir=settings.data_dir,
            session_id=session_id,
            payload_root=settings.get_payload_store_path(),
        )
