"""Pipeline dependency resolution — cycle detection, depth limiting, and execution."""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from elspeth.contracts.enums import RunStatus
from elspeth.contracts.errors import DependencyFailedError
from elspeth.contracts.pipeline_runner import PipelineRunner
from elspeth.contracts.preflight import DependencyRunResult
from elspeth.core.canonical import canonical_json
from elspeth.core.dependency_config import DependencyConfig
from elspeth.engine.error_boundary import reraise_if_engine_crash_through


def _load_depends_on(settings_path: Path) -> list[dict[str, str]]:
    """Load only the depends_on key from a settings file.

    This reads raw YAML (not Pydantic-validated config) specifically
    for cycle detection. The depends_on key is optional — absent means
    no dependencies, which is the common case for leaf pipelines.

    Tier 3 boundary: validates structure of operator-authored YAML.
    """
    with settings_path.open(encoding="utf-8") as f:
        try:
            loaded = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {settings_path}: {exc}") from exc

    if loaded is None:
        return []

    if not isinstance(loaded, dict):
        raise ValueError(f"{settings_path} must be a YAML mapping (key: value), got {type(loaded).__name__}")

    # Tier 3 boundary: raw YAML from operator-authored files.
    # Absent depends_on means "no dependencies" (not "unknown") — empty list
    # is meaning-preserving, matching the common case for leaf pipelines.
    raw_deps = loaded.get("depends_on", [])
    if not isinstance(raw_deps, list):
        raise ValueError(f"depends_on in {settings_path} must be a list, got {type(raw_deps).__name__}")

    deps: list[dict[str, str]] = []
    for i, dep in enumerate(raw_deps):
        if not isinstance(dep, dict):
            raise ValueError(f"depends_on[{i}] in {settings_path} must be a mapping, got {type(dep).__name__}")
        if "name" not in dep:
            raise ValueError(f"depends_on[{i}] in {settings_path} missing required key 'name'")
        if "settings" not in dep:
            raise ValueError(f"depends_on[{i}] in {settings_path} missing required key 'settings'")
        try:
            validated = DependencyConfig.model_validate(dep)
        except ValidationError as exc:
            raise ValueError(f"Invalid depends_on[{i}] in {settings_path}: {exc}") from exc
        deps.append({"name": validated.name, "settings": validated.settings})

    return deps


def _resolve_dependency_settings_path(
    *,
    parent_settings_path: Path,
    dependency_name: str,
    dependency_settings: str,
    allowed_root: Path,
) -> Path:
    """Resolve a dependency settings path under the configured allowed root."""
    raw_path = Path(dependency_settings)
    if raw_path.is_absolute():
        raise ValueError(
            f"Dependency settings path for {dependency_name!r} must be relative to {parent_settings_path.parent}: {dependency_settings!r}"
        )

    resolved_path = (parent_settings_path.parent / raw_path).resolve()
    resolved_root = allowed_root.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"Dependency settings path for {dependency_name!r} escapes allowed root "
            f"{resolved_root}: {dependency_settings!r} -> {resolved_path}"
        ) from exc
    return resolved_path


def detect_cycles(
    settings_path: Path,
    *,
    max_depth: int = 3,
    _visited: set[str] | None = None,
    _stack: list[str] | None = None,
    _depth: int = 0,
    _allowed_root: Path | None = None,
) -> None:
    """Detect circular dependencies and enforce depth limit.

    Uses DFS on canonicalized (resolved) paths.
    Raises ValueError on cycle or depth limit violation.
    """
    canonical = str(settings_path.resolve())
    allowed_root = _allowed_root if _allowed_root is not None else settings_path.parent.resolve()
    visited = _visited if _visited is not None else set()
    stack = _stack if _stack is not None else []

    if _depth >= max_depth:
        raise ValueError(f"Dependency depth limit exceeded ({max_depth}). Chain: {' -> '.join(stack)} -> {canonical}")

    if canonical in stack:
        cycle_start = stack.index(canonical)
        cycle_path = [*stack[cycle_start:], canonical]
        raise ValueError(f"Circular dependency detected: {' -> '.join(cycle_path)}")

    if canonical in visited:
        return  # Already fully explored, no cycle through this node

    stack.append(canonical)
    deps = _load_depends_on(settings_path)

    for dep in deps:
        dep_path = _resolve_dependency_settings_path(
            parent_settings_path=settings_path,
            dependency_name=dep["name"],
            dependency_settings=dep["settings"],
            allowed_root=allowed_root,
        )
        detect_cycles(
            Path(dep_path),
            max_depth=max_depth,
            _visited=visited,
            _stack=stack,
            _depth=_depth + 1,
            _allowed_root=allowed_root,
        )

    stack.pop()
    visited.add(canonical)


def _hash_settings_file(path: Path) -> str:
    """SHA-256 hash of the canonical JSON representation of settings."""
    with path.open() as f:
        data = yaml.safe_load(f)
    canonical = canonical_json(data)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def resolve_dependencies(
    *,
    depends_on: list[DependencyConfig],
    parent_settings_path: Path,
    runner: PipelineRunner,
) -> list[DependencyRunResult]:
    """Run dependency pipelines sequentially. Raises on failure.

    KeyboardInterrupt is propagated as-is (not wrapped in DependencyFailedError).
    """
    results: list[DependencyRunResult] = []
    allowed_root = parent_settings_path.parent.resolve()
    for dep in depends_on:
        dep_path = _resolve_dependency_settings_path(
            parent_settings_path=parent_settings_path,
            dependency_name=dep.name,
            dependency_settings=dep.settings,
            allowed_root=allowed_root,
        )
        try:
            settings_hash = _hash_settings_file(dep_path)
            settings_hash_error: Exception | None = None
        except (OSError, ValueError, yaml.YAMLError) as exc:
            # A failed dependency run does not emit a DependencyRunResult, so
            # preserve the runner's existing error semantics. A successful run,
            # however, must have a pre-run settings hash before it can be
            # recorded as an auditable dependency result.
            settings_hash = None
            settings_hash_error = exc

        start_ms = time.monotonic_ns() // 1_000_000
        try:
            run_result = runner(dep_path)
        except BaseException as exc:
            reraise_if_engine_crash_through(exc)
            if not isinstance(exc, Exception):
                raise
            raise DependencyFailedError(
                dependency_name=dep.name,
                run_id="pre-run",
                reason=f"Dependency pipeline failed before generating a run ID: {type(exc).__name__}: {exc}",
            ) from exc
        duration_ms = (time.monotonic_ns() // 1_000_000) - start_ms

        if run_result.status != RunStatus.COMPLETED:
            raise DependencyFailedError(
                dependency_name=dep.name,
                run_id=run_result.run_id,
                reason=f"Dependency pipeline finished with status: {run_result.status.name}",
            )

        if settings_hash is None:
            assert settings_hash_error is not None
            raise DependencyFailedError(
                dependency_name=dep.name,
                run_id=run_result.run_id,
                reason=(f"Dependency settings hash failed before execution: {type(settings_hash_error).__name__}: {settings_hash_error}"),
            ) from settings_hash_error

        results.append(
            DependencyRunResult(
                name=dep.name,
                run_id=run_result.run_id,
                settings_hash=settings_hash,
                duration_ms=duration_ms,
                indexed_at=datetime.now(UTC).isoformat(),
            )
        )
    return results
