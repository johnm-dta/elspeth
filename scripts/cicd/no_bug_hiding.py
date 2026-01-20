#!/usr/bin/env python3
"""
No Bug-Hiding Enforcement Tool

AST-based static analysis that detects "bug-hiding" defensive programming patterns
and fails CI unless explicitly allowlisted. Enforces elspeth's philosophy:
- Inside the tent: fail fast, no patterns that mask bugs
- At boundaries: explicit validation allowed, but must be intentional and documented

Usage:
    python scripts/cicd/no_bug_hiding.py check --root src
    python scripts/cicd/no_bug_hiding.py check --root src --allowlist scripts/cicd/no_bug_hiding_allowlist.yaml
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

# =============================================================================
# Data Structures
# =============================================================================


def _add_one_month(today: date) -> date:
    """Return a date one month after today, clamped to month length."""
    if today.month == 12:
        year = today.year + 1
        month = 1
    else:
        year = today.year
        month = today.month + 1
    day = min(today.day, monthrange(year, month)[1])
    return date(year, month, day)


@dataclass(frozen=True)
class Finding:
    """A detected violation of the no-bug-hiding policy."""

    rule_id: str
    file_path: str
    line: int
    col: int
    symbol_context: tuple[str, ...]  # e.g., ("ClassName", "method_name")
    code_snippet: str
    message: str

    @property
    def canonical_key(self) -> str:
        """Generate the canonical key for allowlist matching."""
        symbol_part = ":".join(self.symbol_context) if self.symbol_context else "_module_"
        return f"{self.file_path}:{self.rule_id}:{symbol_part}:line={self.line}"

    def suggested_allowlist_entry(self) -> dict[str, Any]:
        """Generate a suggested allowlist entry for this finding."""
        today = datetime.now(UTC).date()
        return {
            "key": self.canonical_key,
            "owner": "<your-name>",
            "reason": "<explain why this is at a trust boundary>",
            "safety": "<explain how failures are handled>",
            "expires": _add_one_month(today).isoformat(),
        }


@dataclass
class AllowlistEntry:
    """A single allowlist entry permitting a specific finding."""

    key: str
    owner: str
    reason: str
    safety: str
    expires: date | None
    matched: bool = field(default=False, compare=False)


@dataclass
class Allowlist:
    """Parsed allowlist configuration."""

    entries: list[AllowlistEntry]
    fail_on_stale: bool = True
    fail_on_expired: bool = True

    def match(self, finding: Finding) -> AllowlistEntry | None:
        """Check if a finding is covered by an allowlist entry."""
        for entry in self.entries:
            if entry.key == finding.canonical_key:
                entry.matched = True
                return entry
        return None

    def get_stale_entries(self) -> list[AllowlistEntry]:
        """Return entries that didn't match any finding."""
        return [e for e in self.entries if not e.matched]

    def get_expired_entries(self) -> list[AllowlistEntry]:
        """Return entries that have expired."""
        today = datetime.now(UTC).date()
        return [e for e in self.entries if e.expires and e.expires < today]


# =============================================================================
# Rule Definitions
# =============================================================================

RULES = {
    "R1": {
        "name": "dict.get",
        "description": "dict.get() usage can hide missing key bugs",
        "remediation": "Access dict keys directly (dict[key]) and fix the schema/contract if KeyError occurs",
    },
    "R2": {
        "name": "getattr",
        "description": "getattr() with default can hide missing attribute bugs",
        "remediation": "Access attributes directly (obj.attr) and fix the type/contract if AttributeError occurs",
    },
    "R3": {
        "name": "hasattr",
        "description": "hasattr() can hide missing attribute bugs by branching around them",
        "remediation": "Use protocols, enums, or fix the type contract instead of runtime attribute checking",
    },
    "R4": {
        "name": "broad-except",
        "description": "Broad exception handling can suppress bugs",
        "remediation": "Catch specific exceptions, or re-raise after logging/quarantining",
    },
}


# =============================================================================
# AST Visitor
# =============================================================================


class BugHidingVisitor(ast.NodeVisitor):
    """AST visitor that detects bug-hiding patterns."""

    def __init__(self, file_path: str, source_lines: list[str]) -> None:
        self.file_path = file_path
        self.source_lines = source_lines
        self.findings: list[Finding] = []
        self.symbol_stack: list[str] = []

    def _get_code_snippet(self, lineno: int) -> str:
        """Get the source line for a given line number."""
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return "<source unavailable>"

    def _add_finding(self, rule_id: str, node: ast.expr | ast.stmt | ast.ExceptHandler, message: str) -> None:
        """Record a finding."""
        self.findings.append(
            Finding(
                rule_id=rule_id,
                file_path=self.file_path,
                line=node.lineno,
                col=node.col_offset,
                symbol_context=tuple(self.symbol_stack),
                code_snippet=self._get_code_snippet(node.lineno),
                message=message,
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class context."""
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track function context."""
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Track async function context."""
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        """Detect R1 (dict.get), R2 (getattr), R3 (hasattr)."""
        # R1: dict.get() - Call(func=Attribute(attr="get"))
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            # This catches any .get() call - we can't know types statically
            # but dict.get() is the common bug-hiding case
            self._add_finding(
                "R1",
                node,
                f"Potential dict.get() usage: {self._get_code_snippet(node.lineno)}",
            )

        # R2: getattr() - Call(func=Name("getattr"))
        # Only flag if there's a default argument (3 args)
        if isinstance(node.func, ast.Name) and node.func.id == "getattr" and (len(node.args) >= 3 or node.keywords):
            self._add_finding(
                "R2",
                node,
                f"getattr() with default hides AttributeError: {self._get_code_snippet(node.lineno)}",
            )

        # R3: hasattr() - Call(func=Name("hasattr"))
        if isinstance(node.func, ast.Name) and node.func.id == "hasattr":
            self._add_finding(
                "R3",
                node,
                f"hasattr() branches around missing attributes: {self._get_code_snippet(node.lineno)}",
            )

        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Detect R4: broad exception suppression."""
        # Check for bare except or except Exception
        is_broad = False
        if node.type is None:
            # bare except:
            is_broad = True
        elif isinstance(node.type, ast.Name) and node.type.id in (
            "Exception",
            "BaseException",
        ):
            is_broad = True
        elif isinstance(node.type, ast.Tuple):
            # except (Exception, ...):
            for elt in node.type.elts:
                if isinstance(elt, ast.Name) and elt.id in (
                    "Exception",
                    "BaseException",
                ):
                    is_broad = True
                    break

        if is_broad:
            # Check if the handler re-raises
            has_reraise = False
            for child in ast.walk(node):
                if isinstance(child, ast.Raise):
                    has_reraise = True
                    break

            if not has_reraise:
                self._add_finding(
                    "R4",
                    node,
                    f"Broad exception caught without re-raise: {self._get_code_snippet(node.lineno)}",
                )

        self.generic_visit(node)


# =============================================================================
# File Scanning
# =============================================================================


def scan_file(file_path: Path, root: Path) -> list[Finding]:
    """Scan a single Python file for bug-hiding patterns."""
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

    visitor = BugHidingVisitor(relative_path, source_lines)
    visitor.visit(tree)
    return visitor.findings


def scan_directory(
    root: Path,
    exclude_patterns: list[str] | None = None,
) -> list[Finding]:
    """Scan all Python files in a directory tree."""
    exclude_patterns = exclude_patterns or []
    findings: list[Finding] = []

    for py_file in root.rglob("*.py"):
        # Check exclusions
        relative = py_file.relative_to(root)
        skip = False
        for pattern in exclude_patterns:
            if relative.match(pattern) or str(relative).startswith(pattern.rstrip("*/")):
                skip = True
                break
        if skip:
            continue

        findings.extend(scan_file(py_file, root))

    return findings


# =============================================================================
# Allowlist Handling
# =============================================================================


def load_allowlist(path: Path) -> Allowlist:
    """Load and parse the allowlist YAML file."""
    if not path.exists():
        return Allowlist(entries=[])

    try:
        with path.open() as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in allowlist {path}: {e}", file=sys.stderr)
        sys.exit(1)

    defaults = data.get("defaults", {})
    entries: list[AllowlistEntry] = []

    for item in data.get("allow_hits", []):
        expires_str = item.get("expires")
        expires_date = None
        if expires_str:
            try:
                expires_date = datetime.strptime(expires_str, "%Y-%m-%d").replace(tzinfo=UTC).date()
            except ValueError:
                print(
                    f"Warning: Invalid date format for expires: {expires_str}",
                    file=sys.stderr,
                )

        entries.append(
            AllowlistEntry(
                key=item["key"],
                owner=item.get("owner", "unknown"),
                reason=item.get("reason", ""),
                safety=item.get("safety", ""),
                expires=expires_date,
            )
        )

    return Allowlist(
        entries=entries,
        fail_on_stale=defaults.get("fail_on_stale", True),
        fail_on_expired=defaults.get("fail_on_expired", True),
    )


# =============================================================================
# Reporting
# =============================================================================


def format_finding_text(finding: Finding) -> str:
    """Format a finding for text output."""
    rule = RULES.get(finding.rule_id, {})
    lines = [
        f"\n{finding.file_path}:{finding.line}:{finding.col}",
        f"  Rule: {finding.rule_id} - {rule.get('name', 'unknown')}",
        f"  Code: {finding.code_snippet}",
        f"  Context: {'.'.join(finding.symbol_context) if finding.symbol_context else '<module>'}",
        f"  Issue: {rule.get('description', finding.message)}",
        f"  Fix: {rule.get('remediation', 'Review and fix the underlying issue')}",
        f"  Allowlist key: {finding.canonical_key}",
    ]
    return "\n".join(lines)


def format_stale_entry_text(entry: AllowlistEntry) -> str:
    """Format a stale allowlist entry for text output."""
    return f"\n  Key: {entry.key}\n  Owner: {entry.owner}\n  Reason: {entry.reason}"


def format_expired_entry_text(entry: AllowlistEntry) -> str:
    """Format an expired allowlist entry for text output."""
    return f"\n  Key: {entry.key}\n  Owner: {entry.owner}\n  Expired: {entry.expires}"


def report_json(
    violations: list[Finding],
    stale_entries: list[AllowlistEntry],
    expired_entries: list[AllowlistEntry],
) -> str:
    """Generate JSON report."""
    return json.dumps(
        {
            "violations": [
                {
                    "rule_id": f.rule_id,
                    "file": f.file_path,
                    "line": f.line,
                    "col": f.col,
                    "context": list(f.symbol_context),
                    "code": f.code_snippet,
                    "message": f.message,
                    "key": f.canonical_key,
                }
                for f in violations
            ],
            "stale_allowlist_entries": [{"key": e.key, "owner": e.owner, "reason": e.reason} for e in stale_entries],
            "expired_allowlist_entries": [{"key": e.key, "owner": e.owner, "expires": str(e.expires)} for e in expired_entries],
        },
        indent=2,
    )


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="No Bug-Hiding Enforcement Tool - detect defensive patterns that mask bugs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # check subcommand
    check_parser = subparsers.add_parser("check", help="Check for bug-hiding patterns")
    check_parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory to scan",
    )
    check_parser.add_argument(
        "--allowlist",
        type=Path,
        default=None,
        help="Path to allowlist YAML file",
    )
    check_parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob patterns to exclude (can be specified multiple times)",
    )
    check_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    check_parser.add_argument(
        "--changed-only",
        action="store_true",
        help="Only scan git-changed files (not yet implemented)",
    )

    args = parser.parse_args()

    if args.command == "check":
        return run_check(args)

    return 0


def run_check(args: argparse.Namespace) -> int:
    """Run the check command."""
    root = args.root.resolve()

    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        return 1

    # Load allowlist
    allowlist_path = args.allowlist
    if allowlist_path is None:
        # Default: config/cicd/no_bug_hiding.yaml relative to repo root
        allowlist_path = Path(__file__).parent.parent.parent / "config" / "cicd" / "no_bug_hiding.yaml"

    allowlist = load_allowlist(allowlist_path)

    # Scan for findings
    all_findings = scan_directory(root, args.exclude)

    # Filter out allowlisted findings
    violations: list[Finding] = []
    for finding in all_findings:
        if allowlist.match(finding) is None:
            violations.append(finding)

    # Check for stale/expired allowlist entries
    stale_entries = allowlist.get_stale_entries() if allowlist.fail_on_stale else []
    expired_entries = allowlist.get_expired_entries() if allowlist.fail_on_expired else []

    # Report results
    has_errors = bool(violations or stale_entries or expired_entries)

    if args.format == "json":
        print(report_json(violations, stale_entries, expired_entries))
    else:
        # Text format
        if violations:
            print(f"\n{'=' * 60}")
            print(f"VIOLATIONS FOUND: {len(violations)}")
            print("=" * 60)
            for v in violations:
                print(format_finding_text(v))

        if stale_entries:
            print(f"\n{'=' * 60}")
            print(f"STALE ALLOWLIST ENTRIES: {len(stale_entries)}")
            print("(These entries don't match any code - remove them)")
            print("=" * 60)
            for e in stale_entries:
                print(format_stale_entry_text(e))

        if expired_entries:
            print(f"\n{'=' * 60}")
            print(f"EXPIRED ALLOWLIST ENTRIES: {len(expired_entries)}")
            print("(These entries have passed their expiration date)")
            print("=" * 60)
            for e in expired_entries:
                print(format_expired_entry_text(e))

        if has_errors:
            print(f"\n{'=' * 60}")
            print("CHECK FAILED")
            print("=" * 60)
            if violations:
                print(f"\nTo allowlist a violation, add an entry to {allowlist_path}")
                print("Example entry:")
                if violations:
                    import pprint

                    pprint.pprint(violations[0].suggested_allowlist_entry())
        else:
            print("\nNo bug-hiding patterns detected. Check passed.")

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
