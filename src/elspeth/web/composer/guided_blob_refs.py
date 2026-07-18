"""Pure validation for blob references retained by guided review snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import UUID

from elspeth.contracts.errors import AuditIntegrityError

GUIDED_REVIEWED_BLOB_PATH_KEYS = ("path", "file")


def validate_guided_reviewed_blob_ref(value: object) -> str:
    """Return a canonical UUID string or fail closed without echoing it."""
    if type(value) is not str:
        raise AuditIntegrityError("guided reviewed source blob_ref must be a canonical UUID string")
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise AuditIntegrityError("guided reviewed source blob_ref must be a canonical UUID string") from exc
    if str(parsed) != value:
        raise AuditIntegrityError("guided reviewed source blob_ref must be a canonical UUID string")
    return value


def validate_guided_reviewed_blob_binding(options: Mapping[str, object]) -> tuple[str, frozenset[str]]:
    """Validate and return one complete reviewed blob binding."""
    blob_ref = validate_guided_reviewed_blob_ref(options["blob_ref"])
    paths: set[str] = set()
    for key in GUIDED_REVIEWED_BLOB_PATH_KEYS:
        if key not in options:
            continue
        value = options[key]
        if type(value) is not str or not value or "\x00" in value:
            raise AuditIntegrityError("guided reviewed blob source path carrier must be an exact non-empty string without NUL")
        paths.add(value)
    if not paths:
        raise AuditIntegrityError("guided reviewed blob source is missing a string path carrier")
    return blob_ref, frozenset(paths)


def validate_guided_reviewed_blob_source_mapping(
    reviewed_bindings: Sequence[tuple[str, frozenset[str]]],
    live_source_options: Mapping[str, Mapping[str, object]],
) -> None:
    """Fail closed unless every reviewed path maps uniquely to its live name."""
    live_carriers = {
        name: {value for key in GUIDED_REVIEWED_BLOB_PATH_KEYS if key in options and type(value := options[key]) is str}
        for name, options in live_source_options.items()
    }
    for reviewed_name, reviewed_paths in reviewed_bindings:
        if reviewed_name in live_carriers:
            if not reviewed_paths.intersection(live_carriers[reviewed_name]):
                raise AuditIntegrityError("guided blob source mapping is inconsistent")
        elif any(reviewed_paths.intersection(paths) for paths in live_carriers.values()):
            raise AuditIntegrityError("guided blob source mapping is inconsistent")

    all_reviewed_paths = frozenset(path for _name, paths in reviewed_bindings for path in paths)
    for live_name, paths in live_carriers.items():
        live_reviewed_paths = paths.intersection(all_reviewed_paths)
        if not live_reviewed_paths:
            continue
        candidates = [
            reviewed_paths
            for reviewed_name, reviewed_paths in reviewed_bindings
            if reviewed_name == live_name and live_reviewed_paths <= reviewed_paths
        ]
        if len(candidates) != 1:
            raise AuditIntegrityError("guided blob source mapping is inconsistent")
