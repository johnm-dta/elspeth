#!/usr/bin/env python3
"""
Audit Evidence Nominal Inheritance Enforcement Tool

AST-based static analysis that detects any class defining ``to_audit_dict``
without inheriting ``AuditEvidenceBase``.  Fails CI unless the class is
explicitly allowlisted.

Enforces ADR-010 B4 control: even with the nominal base class in place, a
class could define ``to_audit_dict`` without inheriting, producing an audit
shape that looks correct but is not routed through the framework.  The scanner
flags this at CI time so the structural invariant is machine-checked, not just
documented.

Usage:
    python scripts/cicd/enforce_audit_evidence_nominal.py check --root src/elspeth
    python scripts/cicd/enforce_audit_evidence_nominal.py check --root src/elspeth \\
        --allowlist config/cicd/enforce_audit_evidence_nominal
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

# =============================================================================
# Constants
# =============================================================================

# The rule ID used in allowlist keys and violation reports.
RULE_ID = "AEN1"

# Directories that are always excluded — vendored / third-party code that
# happens to contain .py files but is not part of the ELSPETH codebase.
_ALWAYS_EXCLUDED_DIRS = ("node_modules", "__pycache__")


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class Finding:
    """A class that defines ``to_audit_dict`` without inheriting AuditEvidenceBase."""

    file_path: str
    line: int
    class_name: str

    @property
    def canonical_key(self) -> str:
        """Allowlist key: file:rule_id:class_name."""
        return f"{self.file_path}:{RULE_ID}:{self.class_name}"


@dataclass
class AllowlistEntry:
    """A single allowlist entry permitting a specific finding."""

    key: str
    owner: str
    reason: str
    task: str  # Tracking reference (e.g., filigree issue ID or ADR task number)
    expires: date | None
    matched: bool = field(default=False, compare=False)
    source_file: str = field(default="", compare=False)


@dataclass
class Allowlist:
    """Parsed allowlist configuration."""

    entries: list[AllowlistEntry]
    fail_on_stale: bool = True
    fail_on_expired: bool = True

    def match(self, finding: Finding) -> AllowlistEntry | None:
        """Return the matching allowlist entry, or None if not covered."""
        for entry in self.entries:
            if entry.key == finding.canonical_key:
                entry.matched = True
                return entry
        return None

    def get_stale_entries(self) -> list[AllowlistEntry]:
        """Return entries that did not match any finding."""
        return [e for e in self.entries if not e.matched]

    def get_expired_entries(self) -> list[AllowlistEntry]:
        """Return entries whose expiry date has passed."""
        today = datetime.now(UTC).date()
        return [e for e in self.entries if e.expires and e.expires < today]


# =============================================================================
# AST Analysis
# =============================================================================


def _bases_include_audit_evidence_base(bases: list[ast.expr]) -> bool:
    """Return True if any base textually references AuditEvidenceBase.

    Handles both plain name (``AuditEvidenceBase``) and attribute access
    (``module.AuditEvidenceBase`` or ``elspeth.contracts.audit_evidence.AuditEvidenceBase``).
    """
    for base in bases:
        if isinstance(base, ast.Name) and base.id == "AuditEvidenceBase":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "AuditEvidenceBase":
            return True
    return False


def _class_defines_to_audit_dict(class_node: ast.ClassDef) -> bool:
    """Return True if the class body directly contains a ``to_audit_dict`` definition.

    Detects three forms:
    - ``def to_audit_dict(self): ...``             (FunctionDef)
    - ``async def to_audit_dict(self): ...``       (AsyncFunctionDef)
    - ``to_audit_dict = lambda self: ...``         (Assign)

    Only checks the immediate body — inherited definitions in base classes are
    not considered (we want to flag the class that introduces the method).

    **Transitive-inheritance design note:** This scanner flags any class that
    *introduces* ``to_audit_dict`` in its own body, even if a base class already
    inherits from ``AuditEvidenceBase``.  Resolving import chains from AST is
    unreliable, so the scanner cannot follow transitive inheritance.  This is
    deliberate: the risk being defended against is an accidental ``to_audit_dict``
    in a class that does not explicitly acknowledge the audit-evidence contract.
    When a subclass legitimately overrides ``to_audit_dict`` on a chain whose root
    *is* ``AuditEvidenceBase``, the intended mechanism is an allowlist entry.

    **Known gap:** ``to_audit_dict: Callable = lambda self: {}`` (AnnAssign) is
    not detected.  It is an unusual pattern; if it appears, add AnnAssign handling
    here and note it in the test suite.
    """
    for node in class_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "to_audit_dict":
            return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "to_audit_dict":
                    return True
    return False


def scan_file(file_path: Path, root: Path) -> list[Finding]:
    """Scan a single Python file for AEN1 violations.

    Crashes loudly on SyntaxError (per CLAUDE.md offensive programming):
    a parse error is evidence of broken code that must be investigated, not
    silently skipped.
    """
    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"Fatal: Could not read {file_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        print(
            f"Fatal: SyntaxError in {file_path}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    relative_path = str(file_path.relative_to(root))
    findings: list[Finding] = []

    # Walk the full tree so nested classes are detected.
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not _class_defines_to_audit_dict(node):
            continue
        if _bases_include_audit_evidence_base(node.bases):
            continue
        findings.append(
            Finding(
                file_path=relative_path,
                line=node.lineno,
                class_name=node.name,
            )
        )

    return findings


def scan_directory(root: Path) -> list[Finding]:
    """Scan all Python files under ``root`` for AEN1 violations."""
    findings: list[Finding] = []

    for py_file in sorted(root.rglob("*.py")):
        relative = py_file.relative_to(root)
        if any(part in _ALWAYS_EXCLUDED_DIRS for part in relative.parts):
            continue
        findings.extend(scan_file(py_file, root))

    return findings


# =============================================================================
# Allowlist Loading
# =============================================================================


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents, exiting on parse error."""
    try:
        with path.open() as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        print(f"Error: Invalid YAML in allowlist {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def _parse_entries(data: dict[str, Any], source_file: str = "") -> list[AllowlistEntry]:
    """Parse ``allow_classes`` entries from a YAML dict."""
    entries: list[AllowlistEntry] = []
    source_ctx = f" in {source_file}" if source_file else ""

    for item in data.get("allow_classes", []):
        key: str = item.get("key", "")
        if not key:
            print(
                f"Error: allow_classes entry is missing 'key' field{source_ctx}",
                file=sys.stderr,
            )
            sys.exit(1)

        expires_str: str | None = item.get("expires")
        expires_date: date | None = None
        if expires_str:
            try:
                expires_date = datetime.strptime(expires_str, "%Y-%m-%d").replace(tzinfo=UTC).date()
            except ValueError:
                print(
                    f"Warning: Invalid date format for expires: {expires_str}{source_ctx}",
                    file=sys.stderr,
                )

        entries.append(
            AllowlistEntry(
                key=key,
                owner=item.get("owner", "unknown"),
                reason=item.get("reason", ""),
                task=item.get("task", ""),
                expires=expires_date,
                source_file=source_file,
            )
        )

    return entries


def load_allowlist_from_directory(directory: Path) -> Allowlist:
    """Load allowlist from a directory of per-module YAML files.

    Expected structure::

        directory/
            _defaults.yaml    — version and defaults section
            errors.yaml       — allow_classes for errors module
            ...
    """
    defaults_path = directory / "_defaults.yaml"
    defaults: dict[str, Any] = {}
    if defaults_path.exists():
        defaults_data = _load_yaml_file(defaults_path)
        defaults = defaults_data.get("defaults", {})

    yaml_files = sorted(f for f in directory.glob("*.yaml") if f.name != "_defaults.yaml")

    all_entries: list[AllowlistEntry] = []
    for yaml_file in yaml_files:
        data = _load_yaml_file(yaml_file)
        all_entries.extend(_parse_entries(data, source_file=yaml_file.name))

    return Allowlist(
        entries=all_entries,
        fail_on_stale=defaults.get("fail_on_stale", True),
        fail_on_expired=defaults.get("fail_on_expired", True),
    )


def load_allowlist(path: Path) -> Allowlist:
    """Load allowlist from a YAML file or directory of YAML files."""
    if path.is_dir():
        return load_allowlist_from_directory(path)

    if not path.exists():
        return Allowlist(entries=[])

    data = _load_yaml_file(path)
    defaults = data.get("defaults", {})
    return Allowlist(
        entries=_parse_entries(data),
        fail_on_stale=defaults.get("fail_on_stale", True),
        fail_on_expired=defaults.get("fail_on_expired", True),
    )


# =============================================================================
# Reporting
# =============================================================================


def format_finding(finding: Finding) -> str:
    """Format a finding for console output."""
    return f"{finding.file_path}:{finding.line}: {finding.class_name} defines to_audit_dict without inheriting AuditEvidenceBase"


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Audit Evidence Nominal Inheritance Enforcement — any class defining to_audit_dict must inherit AuditEvidenceBase (ADR-010)"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Check for AEN1 violations")
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
        help="Path to allowlist YAML file or directory of YAML files",
    )

    args = parser.parse_args()

    if args.command == "check":
        return run_check(args)

    return 0


def run_check(args: argparse.Namespace) -> int:
    """Run the check command."""
    root: Path = args.root.resolve()

    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        return 1

    # Resolve allowlist path: explicit arg, or default directory next to this script.
    allowlist_path: Path | None = args.allowlist
    if allowlist_path is None:
        repo_root = Path(__file__).parent.parent.parent
        dir_path = repo_root / "config" / "cicd" / "enforce_audit_evidence_nominal"
        file_path = repo_root / "config" / "cicd" / "enforce_audit_evidence_nominal.yaml"
        allowlist_path = dir_path if dir_path.is_dir() else file_path

    allowlist = load_allowlist(allowlist_path)

    all_findings = scan_directory(root)

    # Filter through allowlist.
    violations: list[Finding] = []
    for finding in all_findings:
        if allowlist.match(finding) is None:
            violations.append(finding)

    stale_entries = allowlist.get_stale_entries() if allowlist.fail_on_stale else []
    expired_entries = allowlist.get_expired_entries() if allowlist.fail_on_expired else []

    has_errors = bool(violations or stale_entries or expired_entries)

    if violations:
        print(f"\n{'=' * 60}")
        print(f"AEN1 VIOLATIONS FOUND: {len(violations)}")
        print("Classes that define to_audit_dict without inheriting AuditEvidenceBase")
        print("=" * 60)
        for v in violations:
            print(format_finding(v))
        print()
        print("To allowlist a transitional exception, add an entry to the allowlist directory:")
        print(f"  {allowlist_path}")
        print()
        print("Example allowlist entry (YAML):")
        print("  allow_classes:")
        print(f"  - key: {violations[0].canonical_key}")
        print("    owner: <your-name>")
        print("    reason: <explain why this class defines to_audit_dict without inheriting>")
        print("    task: <filigree issue or ADR task that will resolve this>")
        print("    expires: <YYYY-MM-DD>")

    if stale_entries:
        print(f"\n{'=' * 60}")
        print(f"STALE ALLOWLIST ENTRIES: {len(stale_entries)}")
        print("(These entries do not match any code — remove them)")
        print("=" * 60)
        for e in stale_entries:
            source = f" (from {e.source_file})" if e.source_file else ""
            print(f"  Key: {e.key}{source}")
            print(f"  Owner: {e.owner}")
            print(f"  Reason: {e.reason}")

    if expired_entries:
        print(f"\n{'=' * 60}")
        print(f"EXPIRED ALLOWLIST ENTRIES: {len(expired_entries)}")
        print("(These entries have passed their expiration date)")
        print("=" * 60)
        for e in expired_entries:
            print(f"  Key: {e.key}")
            print(f"  Owner: {e.owner}")
            print(f"  Expired: {e.expires}")

    if has_errors:
        print(f"\n{'=' * 60}")
        print("CHECK FAILED")
        print("=" * 60)
    else:
        print("\nNo AuditEvidenceBase nominal inheritance violations. Check passed.")

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
