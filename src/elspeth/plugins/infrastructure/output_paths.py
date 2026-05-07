"""Output path collision helpers for local file sinks."""

from __future__ import annotations

from pathlib import Path

from elspeth.plugins.infrastructure.config_base import OutputCollisionPolicy

_WRITE_POLICIES = frozenset({"fail_if_exists", "auto_increment"})


def validate_output_collision_policy_mode(
    *,
    plugin_name: str,
    mode: str,
    collision_policy: OutputCollisionPolicy | None,
) -> None:
    """Validate mode/policy combinations for local file sinks."""
    if collision_policy is None:
        return

    if mode == "append":
        if collision_policy != "append_or_create":
            raise ValueError(f"{plugin_name} mode='append' requires collision_policy='append_or_create', got {collision_policy!r}.")
        return

    if mode == "write":
        if collision_policy == "append_or_create":
            raise ValueError(
                f"{plugin_name} collision_policy='append_or_create' requires mode='append'. "
                "Use 'fail_if_exists' or 'auto_increment' for mode='write'."
            )
        return

    raise ValueError(f"{plugin_name} mode must be 'write' or 'append', got {mode!r}.")


def resolve_output_collision_path(
    path: Path,
    collision_policy: OutputCollisionPolicy | None,
) -> Path:
    """Resolve the target path for a local file sink collision policy."""
    if collision_policy is None or collision_policy == "append_or_create":
        return path

    if collision_policy == "fail_if_exists":
        if path.exists():
            raise FileExistsError(f"Output path already exists: {path}. Choose a different path or use collision_policy='auto_increment'.")
        return path

    if collision_policy == "auto_increment":
        return next_available_output_path(path)

    raise AssertionError(f"Unexpected output collision policy: {collision_policy!r}")


def next_available_output_path(path: Path) -> Path:
    """Return ``path`` if free, otherwise ``stem-N`` in the same directory."""
    if not path.exists():
        return path

    suffix = "".join(path.suffixes)
    if suffix:
        stem = path.name[: -len(suffix)]
    else:
        stem = path.name

    for index in range(1, 10_000):
        candidate = path.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate

    raise FileExistsError(f"No free output path found near {path}.")


def should_create_exclusively(collision_policy: OutputCollisionPolicy | None) -> bool:
    """Return whether first-write creation should use exclusive mode."""
    return collision_policy in _WRITE_POLICIES
