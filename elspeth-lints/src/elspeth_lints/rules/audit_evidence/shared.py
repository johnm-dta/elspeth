"""Shared AST helpers for audit-evidence rules."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ClassAllowlistEntry:
    """An exact allowlist entry for a class-scoped legacy finding."""

    key: str
    owner: str
    reason: str
    task: str
    expires: date | None
    source_file: str = ""
    matched: bool = field(default=False, compare=False)


@dataclass(slots=True)
class ClassAllowlist:
    """Allowlist format used by AEN1 and TDE1 legacy scanners."""

    entries: list[ClassAllowlistEntry]
    fail_on_stale: bool = True
    fail_on_expired: bool = True

    def match_key(self, key: str) -> ClassAllowlistEntry | None:
        """Return the matching entry for an exact key."""
        for entry in self.entries:
            if entry.key == key:
                entry.matched = True
                return entry
        return None


def display_path(file_path: Path, root: Path) -> str:
    """Return a path relative to the scan root when possible."""
    try:
        return file_path.relative_to(root).as_posix()
    except ValueError:
        return file_path.as_posix()


def repo_relative_display_path(file_path: Path, root: Path) -> str:
    """Return repo-relative paths when root is ``src/elspeth``."""
    display_root = root.parent.parent if root.name == "elspeth" and root.parent.name == "src" else root
    try:
        return file_path.relative_to(display_root).as_posix()
    except ValueError:
        return file_path.as_posix()


def allowlist_path_for_root(root: Path, directory_name: str) -> Path:
    """Find a CI allowlist directory from a scan root or repository cwd."""
    relative = Path("config") / "cicd" / directory_name
    candidates = [root, Path.cwd(), *root.parents]
    for candidate in candidates:
        path = candidate / relative
        if path.exists():
            return path
    return root / relative


def iter_python_paths(root: Path) -> Iterable[Path]:
    """Yield Python files under a root, excluding cache and frontend dependency trees."""
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts or "node_modules" in path.parts:
            continue
        yield path


def load_class_allowlist(path: Path) -> ClassAllowlist:
    """Load legacy ``allow_classes`` YAML from a file or directory."""
    if path.is_dir():
        defaults = _defaults_from_mapping(_load_yaml(path / "_defaults.yaml"))
        entries: list[ClassAllowlistEntry] = []
        for yaml_file in sorted(file for file in path.glob("*.yaml") if file.name != "_defaults.yaml"):
            entries.extend(_parse_class_entries(_load_yaml(yaml_file), source_file=yaml_file.name))
        return ClassAllowlist(
            entries=entries,
            fail_on_stale=defaults.fail_on_stale,
            fail_on_expired=defaults.fail_on_expired,
        )
    if not path.exists():
        return ClassAllowlist(entries=[])
    data = _load_yaml(path)
    defaults = _defaults_from_mapping(data)
    return ClassAllowlist(
        entries=_parse_class_entries(data, source_file=path.name),
        fail_on_stale=defaults.fail_on_stale,
        fail_on_expired=defaults.fail_on_expired,
    )


def tier_1_error_call(call: ast.Call) -> bool:
    """Return whether a call invokes ``tier_1_error(...)``."""
    func = call.func
    return (isinstance(func, ast.Name) and func.id == "tier_1_error") or (isinstance(func, ast.Attribute) and func.attr == "tier_1_error")


def graph_validation_error_call(call: ast.Call) -> bool:
    """Return whether a call invokes ``GraphValidationError(...)``."""
    func = call.func
    return (isinstance(func, ast.Name) and func.id == "GraphValidationError") or (
        isinstance(func, ast.Attribute) and func.attr == "GraphValidationError"
    )


def enclosing_names(node: ast.AST, parents: dict[int, ast.AST]) -> tuple[str, ...]:
    """Return enclosing class/function names for a node."""
    names: list[str] = []
    current = parents.get(id(node))
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(current.name)
        current = parents.get(id(current))
    names.reverse()
    return tuple(names)


def parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    """Build a child-id to parent-node map."""
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    return parents


@dataclass(frozen=True, slots=True)
class _Defaults:
    fail_on_stale: bool = True
    fail_on_expired: bool = True


def _defaults_from_mapping(data: dict[str, Any]) -> _Defaults:
    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}
    return _Defaults(
        fail_on_stale=bool(defaults.get("fail_on_stale", True)),
        fail_on_expired=bool(defaults.get("fail_on_expired", True)),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: allowlist YAML must be a mapping")
    return data


def _parse_class_entries(data: dict[str, Any], *, source_file: str) -> list[ClassAllowlistEntry]:
    entries_raw = data.get("allow_classes", [])
    if not isinstance(entries_raw, list):
        raise ValueError(f"{source_file}: allow_classes must be a list")
    entries: list[ClassAllowlistEntry] = []
    for index, raw_entry in enumerate(entries_raw):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"{source_file}: allow_classes[{index}] must be a mapping")
        entries.append(
            ClassAllowlistEntry(
                key=_string(raw_entry, "key", source_file=source_file, index=index),
                owner=_string(raw_entry, "owner", source_file=source_file, index=index, default="unknown"),
                reason=_string(raw_entry, "reason", source_file=source_file, index=index, default=""),
                task=_string(raw_entry, "task", source_file=source_file, index=index, default=""),
                expires=_optional_date(raw_entry.get("expires"), source_file=source_file, index=index),
                source_file=source_file,
            )
        )
    return entries


def _string(
    data: dict[str, Any],
    key: str,
    *,
    source_file: str,
    index: int,
    default: str | None = None,
) -> str:
    value = data.get(key, default)
    if isinstance(value, str):
        return value
    raise ValueError(f"{source_file}: allow_classes[{index}].{key} must be a string")


def _optional_date(value: object, *, source_file: str, index: int) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{source_file}: allow_classes[{index}].expires must be YYYY-MM-DD, null, or absent")
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC).date()
    except ValueError as exc:
        # Fail closed: a malformed ``expires`` must raise, not silently become
        # ``None``. Swallowing it leaves ``fail_on_expired`` unable to enforce a
        # typoed expiry, so a one-character diff disables the time bound.
        raise ValueError(f"{source_file}: allow_classes[{index}].expires must be YYYY-MM-DD, null, or absent") from exc
