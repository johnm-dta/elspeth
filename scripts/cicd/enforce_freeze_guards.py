#!/usr/bin/env python3
"""Enforce freeze guard patterns on frozen dataclass __post_init__ methods.

Detects forbidden patterns that lead to incomplete immutability:
- FG1: Bare MappingProxyType wraps (shallow freeze, should use deep_freeze)
- FG2: isinstance type guards to skip freezing (fragile, masks mutation bugs)

Usage:
    python scripts/cicd/enforce_freeze_guards.py check --root src/elspeth
    python scripts/cicd/enforce_freeze_guards.py check --root src/elspeth --allowlist config/cicd/enforce_freeze_guards
    python scripts/cicd/enforce_freeze_guards.py check --root src/elspeth file1.py file2.py
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

# =============================================================================
# Data Structures
# =============================================================================

RULES: dict[str, dict[str, str]] = {
    "FG1": {
        "name": "bare-mapping-proxy",
        "description": "Bare MappingProxyType wrap in __post_init__ — shallow freeze misses nested mutables",
        "remediation": "Use deep_freeze() instead of MappingProxyType(dict(self.x)) for recursive immutability",
    },
    "FG2": {
        "name": "isinstance-freeze-guard",
        "description": "isinstance() type guard used to conditionally skip freezing in __post_init__",
        "remediation": "Use deep_freeze() which is idempotent on already-frozen values — no guard needed",
    },
    "FG3": {
        "name": "missing-freeze-guard",
        "description": "Frozen dataclass with container-typed fields lacks freeze_fields/deep_freeze in __post_init__",
        "remediation": "Add __post_init__ with freeze_fields(self, 'field1', 'field2', ...) for container fields",
    },
}

_ALL_RULE_IDS = frozenset(RULES.keys())

# Types that indicate a freeze guard when used with isinstance in __post_init__
_FREEZE_GUARD_TYPES = {"dict", "tuple", "MappingProxyType", "frozenset", "Mapping"}

# Container types that require freeze_fields/deep_freeze in __post_init__.
# Includes both runtime names (dict, list) and typing module names (Dict, List, Mapping).
_CONTAINER_TYPES = frozenset(
    {
        # Runtime types
        "dict",
        "list",
        "set",
        # typing module generics (PEP 585 style also uses lowercase)
        "Dict",
        "List",
        "Set",
        "Mapping",
        "MutableMapping",
        "Sequence",
        "MutableSequence",
    }
)


@dataclass(frozen=True)
class Finding:
    """A detected freeze guard violation."""

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


@dataclass
class PerFileRule:
    """A per-file rule that allowlists patterns of specified rules for a file."""

    pattern: str
    rules: list[str]
    reason: str
    expires: date | None
    max_hits: int | None = None
    matched_count: int = field(default=0, compare=False)
    source_file: str = field(default="", compare=False)

    def matches(self, file_path: str, rule_id: str) -> bool:
        if rule_id not in self.rules:
            return False
        return fnmatch.fnmatch(file_path, self.pattern)


@dataclass
class Allowlist:
    """Parsed allowlist configuration."""

    per_file_rules: list[PerFileRule] = field(default_factory=list)
    fail_on_stale: bool = True

    def match(self, finding: Finding) -> PerFileRule | None:
        for rule in self.per_file_rules:
            if rule.matches(finding.file_path, finding.rule_id):
                rule.matched_count += 1
                return rule
        return None

    def get_unused_rules(self) -> list[PerFileRule]:
        return [r for r in self.per_file_rules if r.matched_count == 0]

    def get_expired_rules(self) -> list[PerFileRule]:
        today = datetime.now(UTC).date()
        return [r for r in self.per_file_rules if r.expires and r.expires < today]

    def get_exceeded_rules(self) -> list[PerFileRule]:
        return [r for r in self.per_file_rules if r.max_hits is not None and r.matched_count > r.max_hits]


# =============================================================================
# AST Visitor
# =============================================================================


class FreezeGuardVisitor(ast.NodeVisitor):
    """AST visitor that detects forbidden freeze patterns in __post_init__ methods."""

    def __init__(self, file_path: str, source_lines: list[str]) -> None:
        self.file_path = file_path
        self.source_lines = source_lines
        self.findings: list[Finding] = []
        self.symbol_stack: list[str] = []
        self._scope_is_class: list[bool] = []  # Parallel to symbol_stack: True if pushed by ClassDef
        self._in_post_init = False

    def _get_code_snippet(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return "<source unavailable>"

    def _fingerprint(self, rule_id: str, node: ast.expr | ast.stmt) -> str:
        node_dump = ast.dump(node, include_attributes=False, annotate_fields=True)
        context = ":".join(self.symbol_stack) if self.symbol_stack else "_module_"
        payload = f"{rule_id}|{self.file_path}|{context}|{node_dump}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _add_finding(self, rule_id: str, node: ast.expr | ast.stmt, message: str) -> None:
        self.findings.append(
            Finding(
                rule_id=rule_id,
                file_path=self.file_path,
                line=node.lineno,
                col=node.col_offset,
                symbol_context=tuple(self.symbol_stack),
                fingerprint=self._fingerprint(rule_id, node),
                code_snippet=self._get_code_snippet(node.lineno),
                message=message,
            )
        )

    # -------------------------------------------------------------------------
    # FG3: Missing freeze guard detection helpers
    # -------------------------------------------------------------------------

    def _is_frozen_dataclass(self, node: ast.ClassDef) -> bool:
        """Check if class has @dataclass(frozen=True) decorator."""
        for decorator in node.decorator_list:
            # @dataclass(frozen=True) or @dataclass(frozen=True, slots=True)
            if isinstance(decorator, ast.Call):
                func = decorator.func
                func_name = None
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    func_name = func.attr

                if func_name == "dataclass":
                    for keyword in decorator.keywords:
                        if keyword.arg == "frozen":
                            # frozen=True
                            if isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                                return True
                            # frozen=True as a Name (unlikely but possible)
                            if isinstance(keyword.value, ast.Name) and keyword.value.id == "True":
                                return True
        return False

    def _annotation_contains_container(self, annotation: ast.expr | None) -> bool:
        """Check if a type annotation contains a container type.

        Handles: Name (dict), Subscript (Dict[str, int], list[str]),
        BinOp (dict | None), and Constant (string annotations).
        """
        if annotation is None:
            return False

        # Simple name: dict, list, Mapping, etc.
        if isinstance(annotation, ast.Name):
            return annotation.id in _CONTAINER_TYPES

        # Subscripted generic: Dict[str, int], list[str], Mapping[K, V]
        if isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name):
                return annotation.value.id in _CONTAINER_TYPES
            if isinstance(annotation.value, ast.Attribute):
                return annotation.value.attr in _CONTAINER_TYPES

        # Union type: dict | None (Python 3.10+)
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            return self._annotation_contains_container(annotation.left) or self._annotation_contains_container(annotation.right)

        # String annotation: "dict[str, int]" - check if any container type name appears
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            # Simple heuristic: check if container type names appear in the string
            for container_type in _CONTAINER_TYPES:
                if container_type in annotation.value:
                    return True

        return False

    def _get_container_fields(self, node: ast.ClassDef) -> list[str]:
        """Extract field names that have container type annotations."""
        container_fields: list[str] = []
        for item in node.body:
            # Annotated field: name: type or name: type = default
            if (
                isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
                and self._annotation_contains_container(item.annotation)
            ):
                container_fields.append(item.target.id)
        return container_fields

    def _find_post_init(self, node: ast.ClassDef) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        """Find __post_init__ method in class body, if present."""
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__post_init__":
                return item
        return None

    def _post_init_has_freeze_calls(self, post_init: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if __post_init__ contains freeze_fields(), deep_freeze(), or MappingProxyType calls.

        MappingProxyType is included because it IS a freeze mechanism (albeit shallow).
        FG1 catches inappropriate MappingProxyType usage; FG3 catches NO freeze at all.
        This prevents double-flagging cases already covered by FG1/allowlist.
        """
        for child in ast.walk(post_init):
            if isinstance(child, ast.Call):
                func = child.func
                # freeze_fields(self, ...) or deep_freeze(...) or MappingProxyType(...)
                if isinstance(func, ast.Name) and func.id in ("freeze_fields", "deep_freeze", "MappingProxyType"):
                    return True
                # module.freeze_fields(...), types.MappingProxyType(...), etc.
                if isinstance(func, ast.Attribute) and func.attr in ("freeze_fields", "deep_freeze", "MappingProxyType"):
                    return True
        return False

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.symbol_stack.append(node.name)
        self._scope_is_class.append(True)
        was_in_post_init = self._in_post_init
        self._in_post_init = False  # New class scope — not in any __post_init__

        # FG3: Check for frozen dataclass with container fields missing freeze guards
        if self._is_frozen_dataclass(node):
            container_fields = self._get_container_fields(node)
            if container_fields:
                post_init = self._find_post_init(node)
                if post_init is None:
                    # No __post_init__ at all
                    self._add_finding(
                        "FG3",
                        node,
                        f"Frozen dataclass '{node.name}' has container fields {container_fields} but no __post_init__",
                    )
                elif not self._post_init_has_freeze_calls(post_init):
                    # __post_init__ exists but no freeze calls
                    self._add_finding(
                        "FG3",
                        post_init,
                        f"Frozen dataclass '{node.name}' has container fields {container_fields} but __post_init__ lacks freeze_fields/deep_freeze",
                    )

        self.generic_visit(node)
        self._in_post_init = was_in_post_init
        self._scope_is_class.pop()
        self.symbol_stack.pop()

    def _parent_is_class(self) -> bool:
        """Check if the immediate enclosing scope is a class definition."""
        return len(self._scope_is_class) >= 2 and self._scope_is_class[-2]

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.symbol_stack.append(node.name)
        self._scope_is_class.append(False)
        was_in_post_init = self._in_post_init
        if node.name == "__post_init__" and self._parent_is_class():
            # Only treat as __post_init__ if directly inside a class.
            # Module-level functions and nested functions named __post_init__
            # inside non-class scopes are not dataclass methods.
            self._in_post_init = True
        else:
            # Nested functions and non-__post_init__ methods exit the scope.
            self._in_post_init = False
        self.generic_visit(node)
        self._in_post_init = was_in_post_init
        self._scope_is_class.pop()
        self.symbol_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def _is_mapping_proxy_call(self, node: ast.Call) -> bool:
        """Check if a Call node is MappingProxyType(...)."""
        func = node.func
        if isinstance(func, ast.Name) and func.id == "MappingProxyType":
            return True
        return isinstance(func, ast.Attribute) and func.attr == "MappingProxyType"

    def _isinstance_has_freeze_guard_types(self, node: ast.Call) -> list[str]:
        """If node is isinstance(self.x, <freeze types>), return the type names matched."""
        if not (isinstance(node.func, ast.Name) and node.func.id == "isinstance"):
            return []
        if len(node.args) < 2:
            return []
        # First arg must be self.something
        first = node.args[0]
        if not (isinstance(first, ast.Attribute) and isinstance(first.value, ast.Name) and first.value.id == "self"):
            return []
        # Second arg: single Name or Tuple of Names
        second = node.args[1]
        matched: list[str] = []
        if isinstance(second, ast.Name) and second.id in _FREEZE_GUARD_TYPES:
            matched.append(second.id)
        elif isinstance(second, ast.Tuple):
            for elt in second.elts:
                if isinstance(elt, ast.Name) and elt.id in _FREEZE_GUARD_TYPES:
                    matched.append(elt.id)
        return matched

    def visit_Call(self, node: ast.Call) -> None:
        if self._in_post_init:
            # FG1: Bare MappingProxyType wrap
            if self._is_mapping_proxy_call(node):
                self._add_finding(
                    "FG1",
                    node,
                    f"Bare MappingProxyType wrap in __post_init__: {self._get_code_snippet(node.lineno)}",
                )

            # FG2: isinstance freeze guard
            guard_types = self._isinstance_has_freeze_guard_types(node)
            if guard_types:
                self._add_finding(
                    "FG2",
                    node,
                    f"isinstance freeze guard ({', '.join(guard_types)}) in __post_init__: {self._get_code_snippet(node.lineno)}",
                )

        self.generic_visit(node)


# =============================================================================
# File Scanning
# =============================================================================


def scan_file(file_path: Path, root: Path) -> list[Finding]:
    """Scan a single Python file for forbidden freeze patterns."""
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

    visitor = FreezeGuardVisitor(relative_path, source_lines)
    visitor.visit(tree)
    return visitor.findings


def scan_directory(root: Path) -> list[Finding]:
    """Scan all Python files in a directory tree."""
    findings: list[Finding] = []
    for py_file in root.rglob("*.py"):
        findings.extend(scan_file(py_file, root))
    return findings


# =============================================================================
# Allowlist Handling
# =============================================================================


def _load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        with path.open() as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in allowlist {path}: {e}", file=sys.stderr)
        sys.exit(1)


def _parse_per_file_rules(data: dict[str, Any], source_file: str = "") -> list[PerFileRule]:
    rules: list[PerFileRule] = []
    for item in data.get("per_file_rules", []):
        rule_ids = set(item.get("rules", []))
        unknown = rule_ids - _ALL_RULE_IDS
        if unknown:
            ctx = f" in {source_file}" if source_file else ""
            print(
                f"Error: per_file_rules entry for '{item.get('pattern', '?')}'{ctx} uses unknown rule ID(s) {unknown}",
                file=sys.stderr,
            )
            sys.exit(1)

        expires_str = item.get("expires")
        expires_date = None
        if expires_str:
            try:
                expires_date = datetime.strptime(expires_str, "%Y-%m-%d").replace(tzinfo=UTC).date()
            except ValueError:
                print(f"Warning: Invalid date format for expires: {expires_str}", file=sys.stderr)

        max_hits: int | None = None
        raw_max_hits = item.get("max_hits")
        if raw_max_hits is not None:
            try:
                max_hits = int(raw_max_hits)
            except ValueError:
                print(
                    f"Error: non-numeric max_hits for '{item.get('pattern', '?')}': {raw_max_hits!r}",
                    file=sys.stderr,
                )
                sys.exit(1)

        rules.append(
            PerFileRule(
                pattern=item["pattern"],
                rules=item.get("rules", []),
                reason=item.get("reason", ""),
                expires=expires_date,
                max_hits=max_hits,
                source_file=source_file,
            )
        )
    return rules


def load_allowlist(path: Path) -> Allowlist:
    """Load allowlist from a directory of YAML files or a single file."""
    if path.is_dir():
        defaults_path = path / "_defaults.yaml"
        defaults = {}
        if defaults_path.exists():
            defaults_data = _load_yaml_file(defaults_path)
            defaults = defaults_data.get("defaults", {})

        yaml_files = sorted(f for f in path.glob("*.yaml") if f.name != "_defaults.yaml")
        all_rules: list[PerFileRule] = []
        for yaml_file in yaml_files:
            data = _load_yaml_file(yaml_file)
            all_rules.extend(_parse_per_file_rules(data, source_file=yaml_file.name))

        return Allowlist(
            per_file_rules=all_rules,
            fail_on_stale=defaults.get("fail_on_stale", True),
        )

    if not path.exists():
        return Allowlist()

    data = _load_yaml_file(path)
    defaults = data.get("defaults", {})
    return Allowlist(
        per_file_rules=_parse_per_file_rules(data),
        fail_on_stale=defaults.get("fail_on_stale", True),
    )


# =============================================================================
# Reporting
# =============================================================================


def format_finding(finding: Finding) -> str:
    rule = RULES.get(finding.rule_id, {})
    lines = [
        f"\n{finding.file_path}:{finding.line}:{finding.col}",
        f"  Rule: {finding.rule_id} - {rule.get('name', 'unknown')}",
        f"  Code: {finding.code_snippet}",
        f"  Context: {'.'.join(finding.symbol_context) if finding.symbol_context else '<module>'}",
        f"  Issue: {rule.get('description', finding.message)}",
        f"  Fix: {rule.get('remediation', 'Review and fix the pattern')}",
        f"  Allowlist key: {finding.canonical_key}",
    ]
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce freeze guard patterns — detect shallow MappingProxyType wraps and isinstance guards in __post_init__"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Check for forbidden freeze patterns")
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
        dir_path = repo_root / "config" / "cicd" / "enforce_freeze_guards"
        allowlist_path = dir_path if dir_path.is_dir() else repo_root / "config" / "cicd" / "enforce_freeze_guards.yaml"

    allowlist = load_allowlist(allowlist_path)

    # Scan
    if args.files:
        all_findings: list[Finding] = []
        for file_path in args.files:
            resolved = file_path.resolve()
            try:
                resolved.relative_to(root)
                all_findings.extend(scan_file(resolved, root))
            except ValueError:
                pass
    else:
        all_findings = scan_directory(root)

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
        print(f"FREEZE GUARD VIOLATIONS: {len(violations)}")
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
        print("\nNo forbidden freeze patterns detected. Check passed.")

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
