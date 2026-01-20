#!/usr/bin/env python3
"""AST-based enforcement for contracts package.

Scans the codebase for:
1. dataclasses, TypedDicts, NamedTuples, and Enums used across module boundaries
2. dict[str, Any] type hints that should be typed contracts

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


@dataclass
class DictViolation:
    """A dict[str, Any] usage that should be a typed contract."""

    file: str
    line: int
    context: str  # function name or class.method
    param_name: str  # parameter name or "return"


def load_whitelist(path: Path) -> dict[str, set[str]]:
    """Load whitelisted type definitions and dict patterns."""
    if not path.exists():
        return {"types": set(), "dicts": set()}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return {
        "types": set(data.get("allowed_external_types", [])),
        "dicts": set(data.get("allowed_dict_patterns", [])),
    }


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
                is_dataclass_name = isinstance(decorator, ast.Name) and decorator.id == "dataclass"
                is_dataclass_call = (
                    isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == "dataclass"
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


def _is_dict_str_any(annotation: ast.expr | None) -> bool:
    """Check if annotation is dict[str, Any] or Dict[str, Any]."""
    if annotation is None:
        return False

    # dict[str, Any] - modern syntax
    if (
        isinstance(annotation, ast.Subscript)
        and isinstance(annotation.value, ast.Name)
        and annotation.value.id in ("dict", "Dict")
        and isinstance(annotation.slice, ast.Tuple)
        and len(annotation.slice.elts) == 2
    ):
        key_type, value_type = annotation.slice.elts
        if isinstance(key_type, ast.Name) and key_type.id == "str" and isinstance(value_type, ast.Name) and value_type.id == "Any":
            return True
    return False


def _is_list_of_dict_str_any(annotation: ast.expr | None) -> bool:
    """Check if annotation is list[dict[str, Any]]."""
    if annotation is None:
        return False

    if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name) and annotation.value.id in ("list", "List"):
        return _is_dict_str_any(annotation.slice)
    return False


def _is_optional_dict(annotation: ast.expr | None) -> bool:
    """Check if annotation is dict[str, Any] | None."""
    if annotation is None:
        return False

    # dict[str, Any] | None
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        left_is_dict = _is_dict_str_any(annotation.left)
        right_is_none = isinstance(annotation.right, ast.Constant) and annotation.right.value is None
        if left_is_dict and right_is_none:
            return True
        # None | dict[str, Any]
        left_is_none = isinstance(annotation.left, ast.Constant) and annotation.left.value is None
        right_is_dict = _is_dict_str_any(annotation.right)
        if left_is_none and right_is_dict:
            return True
    return False


def find_dict_violations(file_path: Path, whitelist: set[str]) -> list[DictViolation]:
    """Find dict[str, Any] type hints that should be typed contracts."""
    try:
        source = file_path.read_text()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []

    violations = []
    relative_path = str(file_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            func_name = node.name
            class_name = None

            # Try to find enclosing class
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef):
                    for child in ast.walk(parent):
                        if child is node:
                            class_name = parent.name
                            break

            context = f"{class_name}.{func_name}" if class_name else func_name

            # Check parameters
            for arg in node.args.args + node.args.kwonlyargs:
                param_name = arg.arg
                annotation = arg.annotation

                if _is_dict_str_any(annotation) or _is_optional_dict(annotation):
                    # Build qualified name for whitelist check
                    qualified = f"{relative_path}:{context}:{param_name}"
                    if qualified not in whitelist:
                        violations.append(
                            DictViolation(
                                file=relative_path,
                                line=arg.lineno if hasattr(arg, "lineno") else node.lineno,
                                context=context,
                                param_name=param_name,
                            )
                        )
                elif _is_list_of_dict_str_any(annotation):
                    # List types have "(list)" suffix in whitelist
                    qualified = f"{relative_path}:{context}:{param_name} (list)"
                    if qualified not in whitelist:
                        violations.append(
                            DictViolation(
                                file=relative_path,
                                line=arg.lineno if hasattr(arg, "lineno") else node.lineno,
                                context=context,
                                param_name=f"{param_name} (list)",
                            )
                        )

            # Check return type
            if node.returns:
                if _is_dict_str_any(node.returns) or _is_optional_dict(node.returns):
                    qualified = f"{relative_path}:{context}:return"
                    if qualified not in whitelist:
                        violations.append(
                            DictViolation(
                                file=relative_path,
                                line=node.lineno,
                                context=context,
                                param_name="return",
                            )
                        )
                elif _is_list_of_dict_str_any(node.returns):
                    # List types have "(list)" suffix in whitelist
                    qualified = f"{relative_path}:{context}:return (list)"
                    if qualified not in whitelist:
                        violations.append(
                            DictViolation(
                                file=relative_path,
                                line=node.lineno,
                                context=context,
                                param_name="return (list)",
                            )
                        )

    return violations


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


def is_cross_boundary_usage(defining_file: Path, using_file: Path, src_dir: Path) -> bool:
    """Check if usage crosses module boundaries.

    Cross-boundary means the using file is in a different top-level module
    than the defining file.
    """
    defining_module = get_top_level_module(defining_file, src_dir)
    using_module = get_top_level_module(using_file, src_dir)
    return defining_module != using_module


def find_cross_boundary_usages(src_dir: Path, type_name: str, defining_file: Path) -> list[Path]:
    """Find files that import a type from a DIFFERENT top-level module."""
    usages = []
    defining_module = defining_file.relative_to(src_dir).with_suffix("").as_posix().replace("/", ".")

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
            if isinstance(node, ast.ImportFrom) and node.module and defining_module in node.module:
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
    dict_violations: list[DictViolation] = []

    # Scan all Python files outside contracts/
    for py_file in src_dir.rglob("*.py"):
        if contracts_dir in py_file.parents or py_file.parent == contracts_dir:
            continue  # Skip contracts/ itself

        # Check for type definitions
        definitions = find_type_definitions(py_file)
        for type_name, line_no, kind in definitions:
            qualified_name = f"{py_file.relative_to(src_dir).with_suffix('')}:{type_name}"

            if qualified_name in whitelist["types"]:
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

        # Check for dict[str, Any] patterns
        dict_violations.extend(find_dict_violations(py_file, whitelist["dicts"]))

    has_violations = False

    if violations:
        has_violations = True
        print("❌ Type definition violations found:\n")
        for v in violations:
            print(f"  {v.file}:{v.line}: {v.kind} '{v.type_name}'")
            print(f"    Used in: {', '.join(v.used_in)}")
            fix_msg = "    Fix: Move to src/elspeth/contracts/ or add to .contracts-whitelist.yaml\n"
            print(fix_msg)

    if dict_violations:
        has_violations = True
        print("❌ dict[str, Any] violations found:\n")
        for dv in dict_violations:
            print(f"  {dv.file}:{dv.line}: {dv.context} - {dv.param_name}")
            print("    Fix: Use TypedDict/dataclass or add to allowed_dict_patterns\n")

    if has_violations:
        return 1

    print("✅ All cross-boundary types are properly centralized in contracts/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
