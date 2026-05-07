#!/usr/bin/env python3
"""Enforce component_id attribution on GraphValidationError raise sites.

Detects ``raise GraphValidationError(...)`` calls that omit the
``component_id`` keyword argument.  Structural errors (cycles, source
count, etc.) where no single node is at fault are expected to be
allowlisted.

Rule:
- GA1: missing-gve-component-id — ``raise GraphValidationError(...)``
  without ``component_id=``.  The web UI reads this field to highlight
  the responsible component; omitting it silently degrades error
  attribution to "unattributed".

Usage:
    python scripts/cicd/enforce_gve_attribution.py check --root src/elspeth
    python scripts/cicd/enforce_gve_attribution.py check --root src/elspeth --allowlist config/cicd/enforce_gve_attribution
    python scripts/cicd/enforce_gve_attribution.py check --root src/elspeth file1.py file2.py
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
    "GA1": {
        "name": "missing-gve-component-id",
        "description": (
            "raise GraphValidationError(...) without component_id= keyword. "
            "The web UI reads exc.component_id to highlight the responsible component; "
            "omitting it degrades the error to unattributed."
        ),
        "remediation": (
            "Add component_id=<node_id> (and optionally component_type=<type>) to the "
            "raise site.  If the error is genuinely structural (no single node at fault), "
            "add the raise site to the allowlist with reason 'structural'."
        ),
    },
}

_ALL_RULE_IDS = frozenset(RULES.keys())

# The exception class name we look for in raise statements.
_TARGET_EXCEPTION = "GraphValidationError"


@dataclass(frozen=True)
class Finding:
    """A detected GVE attribution violation."""

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
# AST Scanning
# =============================================================================


def _get_enclosing_names(node: ast.AST, parents: dict[int, ast.AST]) -> tuple[str, ...]:
    """Walk up from node to collect enclosing class/function names."""
    names: list[str] = []
    current = parents.get(id(node))
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(current.name)
        current = parents.get(id(current))
    names.reverse()
    return tuple(names)


def _is_gve_call(call_node: ast.Call) -> bool:
    """Check if a Call node invokes GraphValidationError."""
    func = call_node.func
    if isinstance(func, ast.Name) and func.id == _TARGET_EXCEPTION:
        return True
    return isinstance(func, ast.Attribute) and func.attr == _TARGET_EXCEPTION


def _has_component_id_kwarg(call_node: ast.Call) -> bool:
    """Check if a Call node includes a component_id= keyword argument."""
    return any(kw.arg == "component_id" for kw in call_node.keywords)


def _build_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    """Build a child-id → parent mapping for an AST."""
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    return parents


def scan_file(file_path: Path, root: Path) -> list[Finding]:
    """Scan a single Python file for GA1 violations."""
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
    parents = _build_parent_map(tree)
    findings: list[Finding] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue
        if node.exc is None:
            continue
        # Match: raise GraphValidationError(...)
        call_node = node.exc
        if not isinstance(call_node, ast.Call):
            continue
        if not _is_gve_call(call_node):
            continue
        if _has_component_id_kwarg(call_node):
            continue

        # GA1 violation
        line = node.lineno
        col = node.col_offset
        snippet = source_lines[line - 1].strip() if line <= len(source_lines) else "<unavailable>"
        enclosing = _get_enclosing_names(node, parents)

        fingerprint_payload = f"GA1|{relative_path}|{line}|{'::'.join(enclosing)}"
        fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()[:16]

        findings.append(
            Finding(
                rule_id="GA1",
                file_path=relative_path,
                line=line,
                col=col,
                symbol_context=enclosing,
                fingerprint=fingerprint,
                code_snippet=snippet,
                message=f"raise GraphValidationError(...) without component_id= at line {line}",
            )
        )

    return findings


def scan_all(root: Path, files: list[Path] | None = None) -> list[Finding]:
    """Scan files for GA1 findings.

    Args:
        root: Root directory (findings use paths relative to this).
        files: Specific files to scan. If None, scans all .py files under root.
    """
    all_findings: list[Finding] = []
    if files:
        for f in files:
            resolved = f.resolve()
            try:
                resolved.relative_to(root.resolve())
                all_findings.extend(scan_file(resolved, root.resolve()))
            except ValueError:
                pass
    else:
        for py_file in root.rglob("*.py"):
            all_findings.extend(scan_file(py_file, root))
    return all_findings


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
        f"  Scope: {'.'.join(finding.symbol_context) if finding.symbol_context else '<module>'}",
        f"  Issue: {rule.get('description', finding.message)}",
        f"  Fix: {rule.get('remediation', 'Add component_id= to the raise site')}",
        f"  Allowlist key: {finding.canonical_key}",
    ]
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce component_id on GraphValidationError raise sites")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Check for missing component_id")
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
        dir_path = repo_root / "config" / "cicd" / "enforce_gve_attribution"
        allowlist_path = dir_path if dir_path.is_dir() else repo_root / "config" / "cicd" / "enforce_gve_attribution.yaml"

    allowlist = load_allowlist(allowlist_path)

    # Scan
    files = args.files if args.files else None
    all_findings = scan_all(root, files=files)

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
        print(f"GVE ATTRIBUTION VIOLATIONS: {len(violations)}")
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
        print("\nNo missing GVE component_id detected. Check passed.")

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
