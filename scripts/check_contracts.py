#!/usr/bin/env python3
"""AST-based enforcement for contracts package.

Scans the codebase for dataclasses, TypedDicts, NamedTuples, and Enums
that are used across module boundaries. Reports violations where such
types are defined outside contracts/ without whitelist exemption.

Usage:
    python scripts/check_contracts.py

Exit codes:
    0: All contracts properly centralized
    1: Violations found
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped]


@dataclass
class Violation:
    """A contract violation found during scanning."""

    file: str
    line: int
    type_name: str
    kind: str
    used_in: list[str]


def load_whitelist(path: Path) -> set[str]:
    """Load whitelisted type definitions."""
    if not path.exists():
        return set()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return set(data.get("allowed_external_types", []))


def find_type_definitions(file_path: Path) -> list[tuple[str, int, str]]:
    """Find dataclass, TypedDict, NamedTuple, Enum definitions in a file.

    Returns: List of (type_name, line_number, kind)
    """
    try:
        source = file_path.read_text()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []

    definitions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check for @dataclass decorator
            for decorator in node.decorator_list:
                is_dataclass_name = (
                    isinstance(decorator, ast.Name) and decorator.id == "dataclass"
                )
                is_dataclass_call = (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "dataclass"
                )
                if is_dataclass_name or is_dataclass_call:
                    definitions.append((node.name, node.lineno, "dataclass"))

            # Check for TypedDict, NamedTuple, Enum base classes
            for base in node.bases:
                if isinstance(base, ast.Name):
                    if base.id == "TypedDict":
                        definitions.append((node.name, node.lineno, "TypedDict"))
                    elif base.id == "NamedTuple":
                        definitions.append((node.name, node.lineno, "NamedTuple"))
                    elif base.id == "Enum":
                        definitions.append((node.name, node.lineno, "Enum"))
                    elif base.id in ("BaseModel", "PluginSchema"):
                        # Pydantic models in config are OK (trust boundary)
                        pass

    return definitions


def get_top_level_module(file_path: Path, src_dir: Path) -> str:
    """Get the top-level module name for a file.

    For example:
        src/elspeth/tui/types.py -> tui
        src/elspeth/core/config.py -> core
    """
    relative = file_path.relative_to(src_dir)
    parts = relative.parts
    if len(parts) > 0:
        return parts[0]
    return ""


def is_cross_boundary_usage(
    defining_file: Path, using_file: Path, src_dir: Path
) -> bool:
    """Check if usage crosses module boundaries.

    Cross-boundary means the using file is in a different top-level module
    than the defining file.
    """
    defining_module = get_top_level_module(defining_file, src_dir)
    using_module = get_top_level_module(using_file, src_dir)
    return defining_module != using_module


def find_cross_boundary_usages(
    src_dir: Path, type_name: str, defining_file: Path
) -> list[Path]:
    """Find files that import a type from a DIFFERENT top-level module."""
    usages = []
    defining_module = (
        defining_file.relative_to(src_dir).with_suffix("").as_posix().replace("/", ".")
    )

    for py_file in src_dir.rglob("*.py"):
        if py_file == defining_file:
            continue

        # Only count as violation if crossing module boundary
        if not is_cross_boundary_usage(defining_file, py_file, src_dir):
            continue

        try:
            source = py_file.read_text()
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and defining_module in node.module
            ):
                for alias in node.names:
                    if alias.name == type_name:
                        usages.append(py_file)

    return usages


def main() -> int:
    """Run the contracts enforcement check."""
    src_dir = Path("src/elspeth")
    contracts_dir = src_dir / "contracts"
    whitelist_path = Path(".contracts-whitelist.yaml")

    whitelist = load_whitelist(whitelist_path)
    violations: list[Violation] = []

    # Scan all Python files outside contracts/
    for py_file in src_dir.rglob("*.py"):
        if contracts_dir in py_file.parents or py_file.parent == contracts_dir:
            continue  # Skip contracts/ itself

        definitions = find_type_definitions(py_file)
        for type_name, line_no, kind in definitions:
            qualified_name = (
                f"{py_file.relative_to(src_dir).with_suffix('')}:{type_name}"
            )

            if qualified_name in whitelist:
                continue

            # Check if used across module boundaries
            usages = find_cross_boundary_usages(src_dir, type_name, py_file)
            if usages:
                violations.append(
                    Violation(
                        file=str(py_file),
                        line=line_no,
                        type_name=type_name,
                        kind=kind,
                        used_in=[str(u) for u in usages[:3]],  # First 3
                    )
                )

    if violations:
        print("❌ Contract violations found:\n")  # noqa: T201
        for v in violations:
            print(f"  {v.file}:{v.line}: {v.kind} '{v.type_name}'")  # noqa: T201
            print(f"    Used in: {', '.join(v.used_in)}")  # noqa: T201
            fix_msg = "    Fix: Move to src/elspeth/contracts/ or add to .contracts-whitelist.yaml\n"
            print(fix_msg)  # noqa: T201
        return 1

    print("✅ All cross-boundary types are properly centralized in contracts/")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
