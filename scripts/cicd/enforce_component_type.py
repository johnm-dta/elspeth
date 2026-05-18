#!/usr/bin/env python3
"""Enforce _plugin_component_type on DataPluginConfig subclasses.

Detects DataPluginConfig subclasses that neither set _plugin_component_type
themselves nor inherit it from an intermediate base class.

Rule:
- CT1: missing-component-type — Class inherits from DataPluginConfig but
  _plugin_component_type is never set in the class or any intermediate
  ancestor below DataPluginConfig.

Usage:
    python scripts/cicd/enforce_component_type.py check --root src/elspeth
    python scripts/cicd/enforce_component_type.py check --root src/elspeth --allowlist config/cicd/enforce_component_type
    python scripts/cicd/enforce_component_type.py check --root src/elspeth file1.py file2.py
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from scripts.cicd._framework.allowlist import (
        PerFileAllowlist as Allowlist,
    )
    from scripts.cicd._framework.allowlist import (
        PerFileRule,
        load_per_file_allowlist,
        parse_per_file_rules,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from _framework.allowlist import (
        PerFileAllowlist as Allowlist,
    )
    from _framework.allowlist import (
        PerFileRule,
        load_per_file_allowlist,
        parse_per_file_rules,
    )

# =============================================================================
# Data Structures
# =============================================================================

RULES: dict[str, dict[str, str]] = {
    "CT1": {
        "name": "missing-component-type",
        "description": (
            "DataPluginConfig subclass does not set _plugin_component_type. "
            "Error messages and audit trail entries will show component_type=None."
        ),
        "remediation": (
            "Set _plugin_component_type: ClassVar[str | None] = 'source' | 'sink' | 'transform' "
            "on the class, or inherit from SourceDataConfig / SinkPathConfig / TransformDataConfig."
        ),
    },
}

_ALL_RULE_IDS = frozenset(RULES.keys())

# Known framework bases and their properties.
# Classes that name-resolve to these get their status from this registry.
_KNOWN_BASES: dict[str, dict[str, bool]] = {
    "DataPluginConfig": {"is_data_config_descendant": True, "sets_type": False, "is_exempt": True},
    "PathConfig": {"is_data_config_descendant": True, "sets_type": False, "is_exempt": True},
    "SourceDataConfig": {"is_data_config_descendant": True, "sets_type": True, "is_exempt": False},
    "SinkPathConfig": {"is_data_config_descendant": True, "sets_type": True, "is_exempt": False},
    "TransformDataConfig": {"is_data_config_descendant": True, "sets_type": True, "is_exempt": False},
    "TabularSourceDataConfig": {"is_data_config_descendant": True, "sets_type": True, "is_exempt": False},
}


@dataclass(frozen=True)
class Finding:
    """A detected component type violation."""

    rule_id: str
    file_path: str
    line: int
    col: int
    symbol_context: tuple[str, ...]
    fingerprint: str
    code_snippet: str
    message: str

    @property
    def canonical_key(self) -> str:
        symbol_part = ":".join(self.symbol_context) if self.symbol_context else "_module_"
        return f"{self.file_path}:{self.rule_id}:{symbol_part}:fp={self.fingerprint}"


# =============================================================================
# AST Class Info Extraction (Phase 1)
# =============================================================================


@dataclass
class ClassInfo:
    """Information about a class discovered during AST scanning."""

    name: str
    file_path: str
    line: int
    col: int
    base_names: list[str]
    sets_component_type: bool
    is_exempt: bool
    code_snippet: str


def _extract_base_names(node: ast.ClassDef) -> list[str]:
    """Extract base class names from a ClassDef node.

    Handles:
    - Name: class Foo(Bar)
    - Attribute: class Foo(module.Bar)
    """
    names: list[str] = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def _class_sets_component_type(node: ast.ClassDef) -> bool:
    """Check if a class body contains _plugin_component_type = <string>."""
    for item in node.body:
        # Annotated: _plugin_component_type: ClassVar[str | None] = "source"
        if (
            isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and item.target.id == "_plugin_component_type"
            and item.value is not None
            and isinstance(item.value, ast.Constant)
            and isinstance(item.value.value, str)
        ):
            return True
        # Bare: _plugin_component_type = "source"
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "_plugin_component_type"
                    and isinstance(item.value, ast.Constant)
                    and isinstance(item.value.value, str)
                ):
                    return True
    return False


def _class_is_exempt(node: ast.ClassDef) -> bool:
    """Check if a class body contains _component_type_exempt = True."""
    for item in node.body:
        if (
            isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and item.target.id == "_component_type_exempt"
            and item.value is not None
            and isinstance(item.value, ast.Constant)
            and item.value.value is True
        ):
            return True
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "_component_type_exempt"
                    and isinstance(item.value, ast.Constant)
                    and item.value.value is True
                ):
                    return True
    return False


def scan_file_classes(file_path: Path, root: Path) -> list[ClassInfo]:
    """Scan a single Python file and extract ClassInfo records."""
    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
        return []

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        print(f"Warning: Syntax error in {file_path}: {e}", file=sys.stderr)
        return []

    source_lines = source.splitlines()
    relative_path = str(file_path.relative_to(root))
    classes: list[ClassInfo] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            snippet = source_lines[node.lineno - 1].strip() if node.lineno <= len(source_lines) else "<unavailable>"
            classes.append(
                ClassInfo(
                    name=node.name,
                    file_path=relative_path,
                    line=node.lineno,
                    col=node.col_offset,
                    base_names=_extract_base_names(node),
                    sets_component_type=_class_sets_component_type(node),
                    is_exempt=_class_is_exempt(node),
                    code_snippet=snippet,
                )
            )

    return classes


# =============================================================================
# Inheritance Resolution (Phase 2)
# =============================================================================


def _is_data_config_descendant(
    class_name: str,
    registry: dict[str, ClassInfo],
    cache: dict[str, bool],
) -> bool:
    """Determine if a class is a DataPluginConfig descendant (by name resolution)."""
    if class_name in cache:
        return cache[class_name]

    if class_name in _KNOWN_BASES:
        result = _KNOWN_BASES[class_name]["is_data_config_descendant"]
        cache[class_name] = result
        return result

    info = registry.get(class_name)
    if info is None:
        cache[class_name] = False
        return False

    # Prevent infinite loops from circular inheritance (invalid Python, but be safe)
    cache[class_name] = False
    for base in info.base_names:
        if _is_data_config_descendant(base, registry, cache):
            cache[class_name] = True
            return True

    return False


def _ancestor_sets_type(
    class_name: str,
    registry: dict[str, ClassInfo],
    cache: dict[str, bool | None],
) -> bool:
    """Check if any ancestor (not the class itself) sets _plugin_component_type."""
    if class_name in cache:
        result = cache[class_name]
        return result if result is not None else False

    if class_name in _KNOWN_BASES:
        result = _KNOWN_BASES[class_name]["sets_type"]
        cache[class_name] = result
        return result

    info = registry.get(class_name)
    if info is None:
        cache[class_name] = False
        return False

    # Prevent infinite loops
    cache[class_name] = None  # sentinel: being computed
    for base in info.base_names:
        base_info = registry.get(base)
        base_sets = False

        # Check if the base itself sets it
        if base in _KNOWN_BASES:
            base_sets = _KNOWN_BASES[base]["sets_type"]
        elif base_info is not None:
            base_sets = base_info.sets_component_type

        if base_sets or _ancestor_sets_type(base, registry, cache):
            cache[class_name] = True
            return True

    cache[class_name] = False
    return False


def scan_and_resolve(root: Path, files: list[Path] | None = None) -> list[Finding]:
    """Scan files and resolve inheritance to produce CT1 findings.

    Args:
        root: Root directory (findings use paths relative to this).
        files: Specific files to scan. If None, scans all .py files under root.

    Returns:
        List of CT1 findings for classes missing _plugin_component_type.
    """
    # Phase 1: Collect ClassInfo from all files
    all_classes: list[ClassInfo] = []
    if files:
        for f in files:
            resolved = f.resolve()
            try:
                resolved.relative_to(root.resolve())
                all_classes.extend(scan_file_classes(resolved, root.resolve()))
            except ValueError:
                pass  # File outside root — skip
    else:
        for py_file in root.rglob("*.py"):
            all_classes.extend(scan_file_classes(py_file, root))

    # Phase 2: Build registry and resolve
    registry: dict[str, ClassInfo] = {}
    for info in all_classes:
        registry[info.name] = info

    descendant_cache: dict[str, bool] = {}
    ancestor_cache: dict[str, bool | None] = {}
    findings: list[Finding] = []

    for info in all_classes:
        # Skip classes that aren't DataPluginConfig descendants
        if not any(_is_data_config_descendant(base, registry, descendant_cache) for base in info.base_names) and not any(
            base in _KNOWN_BASES and _KNOWN_BASES[base]["is_data_config_descendant"] for base in info.base_names
        ):
            continue

        # Skip classes that set it themselves
        if info.sets_component_type:
            continue

        # Skip exempt classes
        if info.is_exempt:
            continue

        # Check if any ancestor sets it
        if _ancestor_sets_type(info.name, registry, ancestor_cache):
            continue

        # CT1 violation
        fingerprint_payload = f"CT1|{info.file_path}|{info.name}"
        fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()[:16]

        findings.append(
            Finding(
                rule_id="CT1",
                file_path=info.file_path,
                line=info.line,
                col=info.col,
                symbol_context=(info.name,),
                fingerprint=fingerprint,
                code_snippet=info.code_snippet,
                message=(
                    f"{info.name} inherits from DataPluginConfig but does not set "
                    f"_plugin_component_type to 'source', 'sink', or 'transform'"
                ),
            )
        )

    return findings


# =============================================================================
# Allowlist Handling
# =============================================================================


def _parse_per_file_rules(data: dict[str, Any], source_file: str = "") -> list[PerFileRule]:
    return parse_per_file_rules(data, valid_rule_ids=_ALL_RULE_IDS, source_file=source_file)


def load_allowlist(path: Path) -> Allowlist:
    """Load allowlist from a directory of YAML files or a single file."""
    return load_per_file_allowlist(path, valid_rule_ids=_ALL_RULE_IDS)


# =============================================================================
# Reporting
# =============================================================================


def format_finding(finding: Finding) -> str:
    rule = RULES.get(finding.rule_id, {})
    lines = [
        f"\n{finding.file_path}:{finding.line}:{finding.col}",
        f"  Rule: {finding.rule_id} - {rule.get('name', 'unknown')}",
        f"  Code: {finding.code_snippet}",
        f"  Class: {'.'.join(finding.symbol_context) if finding.symbol_context else '<module>'}",
        f"  Issue: {rule.get('description', finding.message)}",
        f"  Fix: {rule.get('remediation', 'Set _plugin_component_type on the class')}",
        f"  Allowlist key: {finding.canonical_key}",
    ]
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce _plugin_component_type on DataPluginConfig subclasses")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Check for missing _plugin_component_type")
    check_parser.add_argument("--root", type=Path, required=True, help="Root directory to scan")
    check_parser.add_argument(
        "--allowlist",
        type=Path,
        default=None,
        help="Path to allowlist YAML file or directory",
    )
    check_parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Specific files to check (pre-commit mode). If empty, scans --root directory.",
    )

    args = parser.parse_args()

    if args.command == "check":
        return run_check(args)

    return 0


def run_check(args: argparse.Namespace) -> int:
    root = args.root.resolve()

    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        return 1

    # Load allowlist
    allowlist_path = args.allowlist
    if allowlist_path is None:
        repo_root = Path(__file__).parent.parent.parent
        dir_path = repo_root / "config" / "cicd" / "enforce_component_type"
        allowlist_path = dir_path if dir_path.is_dir() else repo_root / "config" / "cicd" / "enforce_component_type.yaml"

    allowlist = load_allowlist(allowlist_path)

    # Scan and resolve
    files = args.files if args.files else None
    all_findings = scan_and_resolve(root, files=files)

    # Filter allowlisted
    violations: list[Finding] = []
    for finding in all_findings:
        if allowlist.match(finding) is None:
            violations.append(finding)

    # Staleness checks (only in full-scan mode)
    if args.files:
        unused_rules: list[PerFileRule] = []
        expired_rules: list[PerFileRule] = []
        exceeded_rules: list[PerFileRule] = []
    else:
        unused_rules = allowlist.get_unused_rules() if allowlist.fail_on_stale else []
        expired_rules = allowlist.get_expired_rules()
        exceeded_rules = allowlist.get_exceeded_rules()

    has_errors = bool(violations or unused_rules or expired_rules or exceeded_rules)

    # Report
    if violations:
        print(f"\n{'=' * 60}")
        print(f"COMPONENT TYPE VIOLATIONS: {len(violations)}")
        print("=" * 60)
        for v in violations:
            print(format_finding(v))

    if expired_rules:
        print(f"\n{'=' * 60}")
        print(f"EXPIRED PER-FILE RULES: {len(expired_rules)}")
        print("=" * 60)
        for r in expired_rules:
            print(f"\n  Pattern: {r.pattern}")
            print(f"  Rules: {r.rules}")
            print(f"  Expired: {r.expires}")

    if unused_rules:
        print(f"\n{'=' * 60}")
        print(f"UNUSED PER-FILE RULES: {len(unused_rules)}")
        print("(These rules didn't match any code — remove them)")
        print("=" * 60)
        for r in unused_rules:
            print(f"\n  Pattern: {r.pattern}")
            print(f"  Rules: {r.rules}")
            print(f"  Reason: {r.reason}")

    if exceeded_rules:
        print(f"\n{'=' * 60}")
        print(f"EXCEEDED PER-FILE RULES: {len(exceeded_rules)}")
        print("(These rules matched more findings than max_hits allows)")
        print("=" * 60)
        for r in exceeded_rules:
            print(f"\n  Pattern: {r.pattern}")
            print(f"  Rules: {r.rules}")
            print(f"  Matched: {r.matched_count} (max_hits: {r.max_hits})")

    if has_errors:
        print(f"\n{'=' * 60}")
        print("CHECK FAILED")
        print("=" * 60)
        if violations:
            print(f"\nTo allowlist a finding, add a per_file_rules entry to {allowlist_path}")
    else:
        print("\nNo missing _plugin_component_type detected. Check passed.")

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
