#!/usr/bin/env python3
"""Detect mutable container annotations on frozen dataclass fields.

Frozen dataclasses that convert list→tuple and dict→MappingProxyType in
__post_init__ must use Sequence/Mapping annotations, not list/dict.
Without this, mypy permits .append() / [key]=value on fields that will
reject them at runtime.

Usage:
    python scripts/cicd/enforce_frozen_annotations.py check --root src/elspeth
    python scripts/cicd/enforce_frozen_annotations.py check --root src/elspeth --allowlist config/cicd/enforce_frozen_annotations
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import yaml

# Mutable container patterns to detect in annotation strings
MUTABLE_PATTERNS = re.compile(r"\b(list|dict|set)\[")


def find_violations(source: str, filename: str = "<unknown>") -> list[dict[str, str]]:
    """Find mutable container annotations on frozen dataclass fields.

    Args:
        source: Python source code string
        filename: Filename for error messages

    Returns:
        List of violation dicts with keys: file, line, column, class, field, annotation
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    violations: list[dict[str, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        if not _is_frozen_dataclass(node):
            continue

        for item in node.body:
            if not isinstance(item, ast.AnnAssign):
                continue
            if item.target is None or not isinstance(item.target, ast.Name):
                continue

            annotation_str = ast.unparse(item.annotation)
            if MUTABLE_PATTERNS.search(annotation_str):
                violations.append(
                    {
                        "file": filename,
                        "line": str(item.lineno),
                        "column": str(item.col_offset),
                        "class": node.name,
                        "field": item.target.id,
                        "annotation": annotation_str,
                    }
                )

    return violations


def _is_frozen_dataclass(node: ast.ClassDef) -> bool:
    """Check if a class is decorated with @dataclass(frozen=True)."""
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call):
            func = decorator.func
            if (isinstance(func, ast.Name) and func.id == "dataclass") or (isinstance(func, ast.Attribute) and func.attr == "dataclass"):
                for kw in decorator.keywords:
                    if kw.arg == "frozen" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        return True
    return False


def _load_allowlist(allowlist_dir: Path) -> set[str]:
    """Load allowlisted violations from YAML files in the allowlist directory."""
    allowed: set[str] = set()
    if not allowlist_dir.exists():
        return allowed
    for yaml_file in sorted(allowlist_dir.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
            if data and "allow" in data:
                for entry in data["allow"]:
                    key = entry.get("key", "")
                    if key:
                        allowed.add(key)
    return allowed


def _violation_key(v: dict[str, str]) -> str:
    """Generate a unique key for a violation (for allowlisting)."""
    return f"{v['file']}:{v['class']}:{v['field']}"


def json_violations(violations: list[dict[str, str]]) -> str:
    """Render violations in the elspeth-lints parity schema."""
    payload = [
        {
            "rule_id": "immutability.frozen_annotations",
            "file_path": violation["file"],
            "line": int(violation["line"]),
            "column": int(violation["column"]),
            "message": (
                f"Frozen dataclass field {violation['class']}.{violation['field']} "
                f"uses mutable annotation {violation['annotation']}. "
                "Use Sequence/Mapping/tuple/frozenset instead of list/dict/set."
            ),
            "fingerprint": _violation_key(violation),
            "severity": "error",
            "suggestion": "Use Sequence/Mapping/tuple/frozenset instead of list/dict/set",
        }
        for violation in violations
    ]
    return json.dumps(payload, sort_keys=True) + "\n"


def check_directory(root: Path, allowlist_dir: Path | None = None) -> list[dict[str, str]]:
    """Check all Python files under root for violations."""
    allowed = _load_allowlist(allowlist_dir) if allowlist_dir else set()
    all_violations: list[dict[str, str]] = []

    for py_file in sorted(root.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        rel_path = str(py_file.relative_to(root.parent.parent))  # relative to src/
        source = py_file.read_text()
        violations = find_violations(source, filename=rel_path)
        for v in violations:
            if _violation_key(v) not in allowed:
                all_violations.append(v)

    return all_violations


def main() -> None:
    parser = argparse.ArgumentParser(description="Enforce immutable annotations on frozen dataclasses")
    sub = parser.add_subparsers(dest="command")

    check = sub.add_parser("check", help="Check for violations")
    check.add_argument("--root", type=Path, required=True, help="Root directory to scan")
    check.add_argument("--allowlist", type=Path, default=None, help="Directory containing allowlist YAML files")
    check.add_argument("--format", choices=("text", "json"), default="text", help="Output format")

    args = parser.parse_args()

    if args.command != "check":
        parser.print_help()
        sys.exit(1)

    violations = check_directory(args.root, args.allowlist)

    if args.format == "json":
        sys.stdout.write(json_violations(violations))
        sys.exit(1 if violations else 0)

    if violations:
        print(f"\n{'=' * 60}")
        print(f"FROZEN ANNOTATION VIOLATIONS: {len(violations)}")
        print(f"{'=' * 60}\n")
        for v in violations:
            print(f"{v['file']}:{v['line']}")
            print(f"  Class: {v['class']}")
            print(f"  Field: {v['field']}")
            print(f"  Annotation: {v['annotation']}")
            print("  Fix: Use Sequence/Mapping/tuple/frozenset instead of list/dict/set")
            print(f"  Allowlist key: {_violation_key(v)}")
            print()
        print(f"{'=' * 60}")
        print("CHECK FAILED")
        print(f"{'=' * 60}")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
