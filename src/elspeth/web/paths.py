"""Shared path allowlist helpers for web subsystem.

AD-4: Single definitions used by composer tool validation,
execution validation, and execution runtime guards. Lives in
web/ (not composer/ or execution/) to avoid cross-package coupling.
"""

from __future__ import annotations

from pathlib import Path

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
    needed for blob sources; see elspeth-07089fbaa3 for the audit-integrity
    contract.
    """
    raw = Path(value)
    if raw.is_absolute():
        return raw.resolve()

    return (Path(data_dir).resolve() / raw).resolve()


def allowed_source_directories(data_dir: str) -> tuple[Path, ...]:
    """Return the set of directories from which source paths are allowed."""
    base = Path(data_dir).resolve()
    return (base / "blobs",)


def allowed_sink_directories(data_dir: str) -> tuple[Path, ...]:
    """Return the set of directories to which sink paths may write.

    Includes data_dir/outputs (primary sink target) and data_dir/blobs
    (output blobs are stored alongside input blobs).
    """
    base = Path(data_dir).resolve()
    return (base / "outputs", base / "blobs")
