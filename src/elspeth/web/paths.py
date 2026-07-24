"""Shared path allowlist helpers for web subsystem.

AD-4: Single definitions used by composer tool validation,
execution validation, and execution runtime guards. Lives in
web/ (not composer/ or execution/) to avoid cross-package coupling.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

SOURCE_LOCAL_PATH_OPTION_KEYS: tuple[str, ...] = ("path", "file")
SINK_LOCAL_PATH_OPTION_KEYS: tuple[str, ...] = ("path", "file", "persist_directory")
# Local filesystem keys to confine inside a transform's
# ``options["provider_config"]`` dict (RAG retrieval transforms carry a
# Chroma persist_directory there). Confined like sink paths — persist_directory
# is a local read/write target.
NESTED_LOCAL_PATH_OPTION_KEYS: tuple[str, ...] = ("persist_directory",)


def resolve_data_path(value: str, data_dir: str) -> Path:
    """Resolve a path value against data_dir (relative) or as-is (absolute).

    Relative paths are joined to data_dir before resolving; absolute paths
    are resolved directly.  Traversal (``../``) is resolved by the OS —
    blocking traversals outside allowed directories is the caller's job
    (via the allowlist helpers below).

    Note: blob-backed source paths are always absolute (canonical
    ``BlobRecord.storage_path``) and are pinned to that exact value by
    ``set_source_from_blob`` and the runtime read guard in
    ``ExecutionService.start_run``.  No legacy relative-path handling is
    needed for blob sources; the runtime read guard enforces the
    audit-integrity contract.
    """
    raw = Path(value)
    if raw.is_absolute():
        return raw.resolve()

    return (Path(data_dir).resolve() / raw).resolve()


def _validate_session_path_segment(session_id: str, *, caller: str) -> None:
    if not session_id or session_id in (".", "..") or "/" in session_id or "\\" in session_id:
        raise ValueError(f"{caller}: malformed session_id {session_id!r} — expected a UUID-shaped path segment")


def managed_blob_directory(data_dir: str) -> Path:
    """Return the deployment-owned blob root for infrastructure probes."""
    return Path(data_dir).resolve() / "blobs"


def _is_uuid_path_segment(value: str) -> bool:
    try:
        UUID(value)
    except (ValueError, AttributeError):
        return False
    return True


def allowed_source_directories(data_dir: str, *, session_id: str | None) -> tuple[Path, ...]:
    """Return session-owned directories from which source paths may read.

    The deployment blob root is shared infrastructure, not authorization.
    A caller without a session identity receives no readable local paths.
    """
    if session_id is None:
        return ()
    _validate_session_path_segment(session_id, caller="allowed_source_directories")
    return (managed_blob_directory(data_dir) / session_id,)


def allowed_sink_directories(data_dir: str, *, session_id: str | None) -> tuple[Path, ...]:
    """Return the set of directories to which sink paths may write.

    Includes only the caller's ``data_dir/outputs/<session_id>`` and
    ``data_dir/blobs/<session_id>`` subtrees. The deployment-level roots are
    shared infrastructure, not authorization boundaries. A sink (or a
    transform's ``provider_config.persist_directory``) must never address
    another session's subtree.

    ``session_id=None`` means the caller has no session identity: the
    result is empty. ``session_id`` is required as a keyword so no call site
    can silently keep unscoped behaviour.

    Raises:
        ValueError: if ``session_id`` is present but could alter the path
            shape (empty, a path separator, or a dot segment). Session ids
            are UUID strings everywhere in the system; anything else
            reaching this boundary is a contract violation upstream.
    """
    base = Path(data_dir).resolve()
    if session_id is None:
        return ()
    _validate_session_path_segment(session_id, caller="allowed_sink_directories")
    return (base / "outputs" / session_id, base / "blobs" / session_id)


def resolve_sink_data_path(value: str, data_dir: str, *, session_id: str | None) -> Path:
    """Resolve a sink path into the caller's session-owned local namespace.

    ``outputs/report.csv`` remains the stable authoring form, but resolves to
    ``outputs/<session_id>/report.csv`` on disk. Already-scoped paths are not
    double-prefixed. Absolute paths and non-output relative paths retain their
    literal resolution so the caller's allowlist check can reject shared or
    foreign locations rather than silently rewriting them.
    """
    raw = Path(value)
    if raw.is_absolute() or session_id is None:
        return resolve_data_path(value, data_dir)

    _validate_session_path_segment(session_id, caller="resolve_sink_data_path")
    if raw.parts and raw.parts[0] == "outputs":
        if len(raw.parts) > 1 and raw.parts[1] == session_id:
            scoped = raw
        elif len(raw.parts) > 1 and _is_uuid_path_segment(raw.parts[1]):
            # An explicitly session-scoped foreign path must remain foreign
            # so the allowlist rejects it. Never adopt it by nesting it under
            # the caller's output directory.
            scoped = raw
        else:
            scoped = Path("outputs") / session_id / Path(*raw.parts[1:])
        return (Path(data_dir).resolve() / scoped).resolve()
    return resolve_data_path(value, data_dir)
