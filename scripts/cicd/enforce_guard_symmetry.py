#!/usr/bin/env python3
"""Enforce write/read guard symmetry for audit dataclasses.

Detects dataclasses with __post_init__ validation whose corresponding
model loaders lack AuditIntegrityError guards. Prevents asymmetric
coverage where write-side checks exist but read-side checks don't.

Usage:
    python scripts/cicd/enforce_guard_symmetry.py check --root src/elspeth
    python scripts/cicd/enforce_guard_symmetry.py check --root src/elspeth --allowlist config/cicd/enforce_guard_symmetry
    python scripts/cicd/enforce_guard_symmetry.py check --root src/elspeth file1.py file2.py
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
    "GS1": {
        "name": "missing-read-guard",
        "description": "Dataclass has __post_init__ validation but corresponding loader has no AuditIntegrityError",
        "remediation": "Add AuditIntegrityError validation to the loader's load() method for Tier 1 read-side integrity",
    },
}

_ALL_RULE_IDS = frozenset(RULES.keys())

# Dataclass→Loader name overrides for non-standard naming conventions.
# Default rule: ClassName → ClassNameLoader. These cover two patterns:
# 1. NodeState discriminated union: 4 variant classes → 1 loader
# 2. *Record-suffixed dataclasses: loader drops the "Record" suffix
LOADER_OVERRIDES: dict[str, str] = {
    "NodeStateOpen": "NodeStateLoader",
    "NodeStatePending": "NodeStateLoader",
    "NodeStateCompleted": "NodeStateLoader",
    "NodeStateFailed": "NodeStateLoader",
    "TransformErrorRecord": "TransformErrorLoader",
    "ValidationErrorRecord": "ValidationErrorLoader",
}

# Functions called in __post_init__ that indicate validation (not just freezing)
_VALIDATION_FUNCTIONS = frozenset(
    {
        "require_int",
        "_validate_enum",
    }
)


def expected_loader_name(class_name: str) -> str:
    """Map a dataclass name to its expected loader class name."""
    return LOADER_OVERRIDES.get(class_name, f"{class_name}Loader")


@dataclass(frozen=True, slots=True)
class Finding:
    """A detected guard symmetry violation."""

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


@dataclass(frozen=True, slots=True)
class DataclassInfo:
    """A discovered dataclass with __post_init__ validation."""

    name: str
    file_path: str
    line: int


@dataclass(frozen=True, slots=True)
class LoaderInfo:
    """A discovered *Loader class with a load() method."""

    name: str
    file_path: str
    line: int
    target_class: str
    has_audit_integrity_error: bool


# =============================================================================
# AST Visitor
# =============================================================================


class GuardSymmetryVisitor(ast.NodeVisitor):
    """AST visitor that discovers dataclass→loader pairs and checks guard coverage."""

    def __init__(self, file_path: str, source_lines: list[str]) -> None:
        self.file_path = file_path
        self.source_lines = source_lines
        self.dataclasses: list[DataclassInfo] = []
        self.loaders: list[LoaderInfo] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Check if this is a @dataclass with __post_init__
        if self._is_dataclass(node):
            post_init = self._find_post_init(node)
            if post_init is not None and self._post_init_has_validation(post_init):
                self.dataclasses.append(
                    DataclassInfo(
                        name=node.name,
                        file_path=self.file_path,
                        line=node.lineno,
                    )
                )

        # Check if this is a *Loader with a concrete load() method
        if node.name.endswith("Loader") and len(node.name) > len("Loader"):
            load_method = self._find_load_method(node)
            if load_method is not None and not self._is_abstract_method(load_method):
                self.loaders.append(
                    LoaderInfo(
                        name=node.name,
                        file_path=self.file_path,
                        line=node.lineno,
                        target_class=node.name.removesuffix("Loader"),
                        has_audit_integrity_error=self._method_raises_aie(load_method),
                    )
                )

        self.generic_visit(node)

    def _is_dataclass(self, node: ast.ClassDef) -> bool:
        """Check if a class has a @dataclass decorator."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
                return True
            if isinstance(decorator, ast.Call):
                func = decorator.func
                if isinstance(func, ast.Name) and func.id == "dataclass":
                    return True
                if isinstance(func, ast.Attribute) and func.attr == "dataclass":
                    return True
        return False

    def _find_post_init(self, node: ast.ClassDef) -> ast.FunctionDef | None:
        """Find __post_init__ method in a class body."""
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__post_init__":
                return item
        return None

    def _find_load_method(self, node: ast.ClassDef) -> ast.FunctionDef | None:
        """Find load() method in a class body."""
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "load":
                return item
        return None

    def _post_init_has_validation(self, method: ast.FunctionDef) -> bool:
        """Check if __post_init__ does validation (not just freeze_fields).

        Returns True if the method contains:
        - raise statements
        - Calls to known validation functions (require_int, _validate_enum)
        """
        for node in ast.walk(method):
            # Direct raise statements
            if isinstance(node, ast.Raise):
                return True
            # Calls to known validation functions
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in _VALIDATION_FUNCTIONS:
                    return True
        return False

    def _is_abstract_method(self, method: ast.FunctionDef) -> bool:
        """Check if a method is abstract (stub body: pass, ..., or raise NotImplementedError).

        Excludes Protocol/ABC loaders from scanner — they define interfaces,
        not concrete load() implementations that need guard checking.
        """
        body = method.body
        # Skip docstring if present
        stmts = [s for s in body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))]
        if len(stmts) != 1:
            return False
        stmt = stmts[0]
        # pass
        if isinstance(stmt, ast.Pass):
            return True
        # ... (Ellipsis)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is ...:
            return True
        # raise NotImplementedError
        if isinstance(stmt, ast.Raise) and stmt.exc is not None:
            exc = stmt.exc
            if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                return True
            if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id == "NotImplementedError":
                return True
        return False

    def _method_raises_aie(self, method: ast.FunctionDef) -> bool:
        """Check if a method raises AuditIntegrityError directly in its body.

        Known limitation: does NOT detect AIE raised by helper functions called
        from load(). If a loader delegates to a helper that raises AIE, the
        scanner will report a false negative. This is acceptable for the
        coarse-grained first pass — add the raise directly in load() for
        scanner visibility, or allowlist the finding if using a helper pattern.
        """
        for node in ast.walk(method):
            if isinstance(node, ast.Raise) and node.exc is not None:
                exc = node.exc
                if isinstance(exc, ast.Call):
                    func = exc.func
                    if isinstance(func, ast.Name) and func.id == "AuditIntegrityError":
                        return True
                    if isinstance(func, ast.Attribute) and func.attr == "AuditIntegrityError":
                        return True
        return False


# =============================================================================
# Pairing Logic
# =============================================================================


def find_unguarded_pairs(
    dataclasses: list[DataclassInfo],
    loaders: list[LoaderInfo],
) -> list[Finding]:
    """Find dataclass→loader pairs where the loader lacks AuditIntegrityError.

    Args:
        dataclasses: All discovered dataclasses with __post_init__ validation
        loaders: All discovered loader classes

    Returns:
        Findings for each unguarded pair (GS1 rule)
    """
    loader_by_name: dict[str, LoaderInfo] = {loader.name: loader for loader in loaders}
    findings: list[Finding] = []

    for dc in dataclasses:
        loader_name = expected_loader_name(dc.name)
        loader = loader_by_name.get(loader_name)
        if loader is None:
            continue  # No loader — nothing to check
        if loader.has_audit_integrity_error:
            continue  # Loader has guards — OK

        # Unguarded pair: dataclass has validation, loader doesn't
        context = (loader.name, "load")
        # Fingerprint differs from other enforcers (which use ast.dump).
        # Findings are generated post-scan from summary structs — no AST node
        # is available. Uses loader+class names which are stable across
        # formatting changes and only shift on class renames.
        fp_payload = f"GS1|{loader.file_path}|{loader.name}|{dc.name}"
        fingerprint = hashlib.sha256(fp_payload.encode("utf-8")).hexdigest()[:16]

        findings.append(
            Finding(
                rule_id="GS1",
                file_path=loader.file_path,
                line=loader.line,
                col=0,
                symbol_context=context,
                fingerprint=fingerprint,
                code_snippet=f"class {loader.name}:",
                message=(
                    f"{dc.name} has __post_init__ validation (at {dc.file_path}:{dc.line}) "
                    f"but {loader.name}.load() has no AuditIntegrityError guards"
                ),
            )
        )

    return findings


# =============================================================================
# File Scanning
# =============================================================================


def _scan_single_file(file_path: Path, root: Path) -> GuardSymmetryVisitor:
    """Scan a single Python file and return the visitor with collected data."""
    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
        return GuardSymmetryVisitor("", [])

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        print(f"Warning: Syntax error in {file_path}: {e}", file=sys.stderr)
        return GuardSymmetryVisitor("", [])

    source_lines = source.splitlines()
    relative_path = str(file_path.relative_to(root))

    visitor = GuardSymmetryVisitor(relative_path, source_lines)
    visitor.visit(tree)
    return visitor


def scan_files(root: Path, files: list[Path] | None = None) -> list[Finding]:
    """Scan Python files and find unguarded dataclass→loader pairs.

    Phase 1: Discover all dataclasses with __post_init__ and all loaders.
    Phase 2: Pair them and report unguarded pairs.

    Args:
        root: Root directory for relative path computation
        files: Specific files to scan (None = scan all *.py under root)

    Returns:
        List of GS1 findings for unguarded pairs
    """
    all_dataclasses: list[DataclassInfo] = []
    all_loaders: list[LoaderInfo] = []

    if files:
        py_files = files
    else:
        py_files = list(root.rglob("*.py"))

    for py_file in py_files:
        resolved = py_file if py_file.is_absolute() else (root / py_file)
        if not resolved.is_file():
            continue
        try:
            resolved.relative_to(root)
        except ValueError:
            continue

        visitor = _scan_single_file(resolved, root)
        all_dataclasses.extend(visitor.dataclasses)
        all_loaders.extend(visitor.loaders)

    return find_unguarded_pairs(all_dataclasses, all_loaders)


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
        f"  Issue: {finding.message}",
        f"  Fix: {rule.get('remediation', 'Review and fix the pattern')}",
        f"  Allowlist key: {finding.canonical_key}",
    ]
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce write/read guard symmetry — detect dataclass→loader pairs missing AuditIntegrityError"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Check for guard symmetry violations")
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
        dir_path = repo_root / "config" / "cicd" / "enforce_guard_symmetry"
        allowlist_path = dir_path if dir_path.is_dir() else repo_root / "config" / "cicd" / "enforce_guard_symmetry.yaml"

    allowlist = load_allowlist(allowlist_path)

    # Scan
    files = [f.resolve() for f in args.files] if args.files else None
    all_findings = scan_files(root, files)

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
        print(f"GUARD SYMMETRY VIOLATIONS: {len(violations)}")
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
        print("\nNo guard symmetry violations detected. Check passed.")

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
