#!/usr/bin/env python3
"""
Tier-1 Decoration Enforcement Tool

AST-based static analysis that detects any exception class in errors.py whose
name ends in ``Error`` or ``Violation`` and has neither a ``@tier_1_error``
decorator (with the mandatory ``reason=`` kwarg) nor a ``# TIER-2:`` comment
with a non-empty justification above the class statement. Blank lines and
decorator lines between the comment and the ``class`` keyword are tolerated;
multi-line TIER-2 comments are not — the ``# TIER-2:`` line must be the last
comment line above the class, modulo any intervening blank or decorator lines.

This closes the symmetric failure mode of the decoration-IS-registration
pattern (ADR-010 §Decision 2): forgetting to decorate a new Tier-1 exception
silently demotes it to Tier-2, invisible to the registry and invisible to CI
unless this scanner is running.

Usage:
    python scripts/cicd/enforce_tier_1_decoration.py check \\
        --file src/elspeth/contracts/errors.py
    python scripts/cicd/enforce_tier_1_decoration.py check \\
        --file src/elspeth/contracts/errors.py \\
        --allowlist config/cicd/enforce_tier_1_decoration
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

# The rule ID used in allowlist keys and violation reports (decoration coverage).
RULE_ID = "TDE1"

# TDE2 — caller_module literal enforcement (ADR-010 M8, issue elspeth-3af772b9e3).
# Every @tier_1_error(...) Call MUST pass ``caller_module=__name__`` where the
# value is the *literal* identifier ``__name__`` (ast.Name). Variables,
# attribute lookups, f-strings, and string literals are all rejected — the
# value is the primary input to the module-prefix allowlist check in
# tier_registry._register_with_module_prefix, and allowing non-literal forms
# would let a caller spoof the allowlist (or silently miscategorize after a
# refactor rename).
RULE_ID_CALLER_MODULE = "TDE2"

# Class name suffixes that trigger inspection.
_CHECKED_SUFFIXES = ("Error", "Violation")


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class Finding:
    """An exception class lacking @tier_1_error decoration or # TIER-2: justification."""

    file_path: str
    line: int
    class_name: str

    @property
    def canonical_key(self) -> str:
        """Allowlist key: file:rule_id:class_name."""
        return f"{self.file_path}:{RULE_ID}:{self.class_name}"


@dataclass(frozen=True)
class CallerModuleFinding:
    """A @tier_1_error(...) Call that does not pass ``caller_module=__name__`` literally.

    ADR-010 M8 (issue elspeth-3af772b9e3) — the ``caller_module`` kwarg
    is the input to the module-prefix allowlist; non-literal values
    would let a caller spoof the allowlist. Enforce that every call
    passes the ``__name__`` identifier literal.
    """

    file_path: str
    line: int
    detail: str  # e.g. "missing caller_module kwarg" / "caller_module value is not the __name__ literal"

    @property
    def canonical_key(self) -> str:
        """Allowlist key: file:rule_id:line."""
        return f"{self.file_path}:{RULE_ID_CALLER_MODULE}:{self.line}"


@dataclass
class AllowlistEntry:
    """A single allowlist entry permitting a specific finding."""

    key: str
    owner: str
    reason: str
    task: str
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


def _has_tier_1_error_decorator(class_node: ast.ClassDef) -> bool:
    """Return True if the class has a @tier_1_error(reason=...) decorator.

    Accepts two syntactic forms:
    - @tier_1_error(reason="...")         — Name node calling the factory
    - @some_pkg.tier_1_error(reason="...") — Attribute node (qualified access)

    The bare form @tier_1_error (no parentheses) is intentionally NOT accepted;
    that form raises TypeError at runtime (the decorator requires reason=...) and
    is therefore not a valid registration. Requiring the Call form matches what
    the runtime enforces.
    """
    for decorator in class_node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        func = decorator.func
        if isinstance(func, ast.Name) and func.id == "tier_1_error":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "tier_1_error":
            return True
    return False


def _has_tier_2_comment(class_node: ast.ClassDef, source_lines: list[str]) -> bool:
    """Return True if a non-empty ``# TIER-2:`` comment precedes the class.

    Walks backward from the class ``class`` keyword line, skipping blank lines
    and other decorator lines, until a non-blank, non-decorator line is found.
    If that line contains ``# TIER-2:`` followed by non-whitespace text, it
    qualifies as a justification.

    An empty ``# TIER-2:`` (nothing after the colon, or only whitespace) does
    NOT qualify — it is as useless as no comment at all.

    ``class_node.lineno`` is 1-based; ``source_lines`` is 0-based.
    """
    # Collect the set of line numbers occupied by decorators (1-based).
    decorator_lines: set[int] = set()
    for dec in class_node.decorator_list:
        # Each decorator may span multiple lines; record all of them.
        for lineno in range(dec.lineno, dec.end_lineno + 1):  # type: ignore[operator]
            decorator_lines.add(lineno)

    # Walk backward from the first line before the class keyword.
    # class_node.lineno points at the `class` keyword itself.
    search_start = class_node.lineno - 2  # convert to 0-based, go one line above class

    idx = search_start
    while idx >= 0:
        line = source_lines[idx]
        stripped = line.strip()

        # Skip blank lines.
        if not stripped:
            idx -= 1
            continue

        # Skip decorator lines (1-based lineno = idx + 1).
        if (idx + 1) in decorator_lines:
            idx -= 1
            continue

        # First non-blank, non-decorator line found.
        if "# TIER-2:" not in line:
            return False

        # Extract the justification text (the part after "# TIER-2:").
        marker = "# TIER-2:"
        justification = line[line.index(marker) + len(marker) :]
        return bool(justification.strip())

    return False


def _is_tier_1_error_call(call: ast.Call) -> bool:
    """Return True if ``call`` is a ``tier_1_error(...)`` invocation.

    Accepts the bare-name form ``tier_1_error(...)`` and the qualified
    form ``some_pkg.tier_1_error(...)``. Used to find both decorator and
    direct-function-call invocations of the factory.
    """
    func = call.func
    return (isinstance(func, ast.Name) and func.id == "tier_1_error") or (isinstance(func, ast.Attribute) and func.attr == "tier_1_error")


def _check_caller_module_literal(call: ast.Call) -> str | None:
    """Return a violation ``detail`` string if ``call`` fails the TDE2 rule, else None.

    The rule (ADR-010 M8): every ``tier_1_error(...)`` Call MUST include
    a ``caller_module=`` kwarg whose value is exactly the Python name
    literal ``__name__`` (ast.Name with id == "__name__"). Any other
    shape — missing kwarg, string literal, variable, attribute, f-string
    — is rejected so the module-prefix allowlist cannot be spoofed.
    """
    caller_module_kw: ast.keyword | None = None
    for kw in call.keywords:
        if kw.arg == "caller_module":
            caller_module_kw = kw
            break
    if caller_module_kw is None:
        return "missing caller_module kwarg (require caller_module=__name__)"
    value = caller_module_kw.value
    if not (isinstance(value, ast.Name) and value.id == "__name__"):
        return f"caller_module value must be the __name__ literal, got {ast.dump(value)}"
    return None


def scan_file(file_path: Path, relative_path: str) -> tuple[list[Finding], list[CallerModuleFinding]]:
    """Scan a single Python file for TDE1 and TDE2 violations.

    Crashes loudly on read errors and SyntaxErrors per CLAUDE.md offensive
    programming posture — these indicate broken code that must be investigated.
    """
    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"Fatal: Could not read {file_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        print(f"Fatal: SyntaxError in {file_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    source_lines = source.splitlines()
    tde1_findings: list[Finding] = []
    tde2_findings: list[CallerModuleFinding] = []

    for node in ast.walk(tree):
        # TDE1: class-level decoration coverage.
        if isinstance(node, ast.ClassDef):
            if not node.name.endswith(_CHECKED_SUFFIXES):
                continue
            if _has_tier_1_error_decorator(node):
                continue
            if _has_tier_2_comment(node, source_lines):
                continue
            tde1_findings.append(
                Finding(
                    file_path=relative_path,
                    line=node.lineno,
                    class_name=node.name,
                )
            )
            continue

        # TDE2: every tier_1_error(...) Call must pass caller_module=__name__ literally.
        if isinstance(node, ast.Call) and _is_tier_1_error_call(node):
            detail = _check_caller_module_literal(node)
            if detail is not None:
                tde2_findings.append(
                    CallerModuleFinding(
                        file_path=relative_path,
                        line=node.lineno,
                        detail=detail,
                    )
                )

    return tde1_findings, tde2_findings


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
    return (
        f"{finding.file_path}:{finding.line}: {finding.class_name} "
        f"has no @tier_1_error(reason=...) decorator and no # TIER-2: justification comment"
    )


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Tier-1 Decoration Enforcement — every exception class ending in Error or Violation "
            "must have @tier_1_error(reason=...) or a # TIER-2: justification comment (ADR-010)"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Check for TDE1 violations")
    check_parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Single Python file to scan (typically errors.py)",
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
    target: Path = args.file.resolve()

    if not target.is_file():
        print(f"Error: {target} is not a file", file=sys.stderr)
        return 1

    # Compute a stable relative path for allowlist keys.
    # Use the path relative to the project root (parent of scripts/cicd).
    repo_root = Path(__file__).parent.parent.parent
    try:
        relative_path = str(target.relative_to(repo_root))
    except ValueError:
        # Target is outside the repo root (e.g., tmp_path in tests); use the basename.
        relative_path = target.name

    # Resolve allowlist path: explicit arg, or default directory next to this script.
    allowlist_path: Path | None = args.allowlist
    if allowlist_path is None:
        dir_path = repo_root / "config" / "cicd" / "enforce_tier_1_decoration"
        file_path = repo_root / "config" / "cicd" / "enforce_tier_1_decoration.yaml"
        allowlist_path = dir_path if dir_path.is_dir() else file_path

    allowlist = load_allowlist(allowlist_path)

    tde1_findings, tde2_findings = scan_file(target, relative_path)

    # Filter TDE1 through allowlist.
    violations: list[Finding] = []
    for finding in tde1_findings:
        if allowlist.match(finding) is None:
            violations.append(finding)

    # TDE2 (ADR-010 M8) has no allowlist yet — the rule is new and the 4
    # in-tree call sites are all under our control. If this ever needs to
    # allowlist transitional failures, extend Allowlist.match to accept
    # CallerModuleFinding keyed by file:TDE2:line.
    tde2_violations = list(tde2_findings)

    stale_entries = allowlist.get_stale_entries() if allowlist.fail_on_stale else []
    expired_entries = allowlist.get_expired_entries() if allowlist.fail_on_expired else []

    has_errors = bool(violations or tde2_violations or stale_entries or expired_entries)

    if violations:
        print(f"\n{'=' * 60}")
        print(f"TDE1 VIOLATIONS FOUND: {len(violations)}")
        print("Exception classes ending in Error/Violation without @tier_1_error or # TIER-2: justification")
        print("=" * 60)
        for v in violations:
            print(format_finding(v))
        print()
        print("To fix: add @tier_1_error(reason='...') for Tier-1 classes, or a")
        print("  # TIER-2: <justification> comment before the class for Tier-2 classes.")
        print("  Blank lines and decorators may appear between the comment and the class.")
        print()
        print("To allowlist a transitional exception, add an entry to the allowlist directory:")
        print(f"  {allowlist_path}")
        print()
        print("Example allowlist entry (YAML):")
        print("  allow_classes:")
        print(f"  - key: {violations[0].canonical_key}")
        print("    owner: <your-name>")
        print("    reason: <explain why this class is not yet decorated>")
        print("    task: <filigree issue or ADR task that will resolve this>")
        print("    expires: <YYYY-MM-DD>")

    if tde2_violations:
        print(f"\n{'=' * 60}")
        print(f"TDE2 VIOLATIONS FOUND: {len(tde2_violations)}")
        print("@tier_1_error(...) Calls missing or mis-shaping caller_module=__name__")
        print("(ADR-010 M8, issue elspeth-3af772b9e3 — module-prefix allowlist spoofing guard)")
        print("=" * 60)
        for v in tde2_violations:
            print(f"{v.file_path}:{v.line}: {v.detail}")
        print()
        print("To fix: pass caller_module=__name__ literally at every @tier_1_error call site.")
        print("The value MUST be the bare identifier __name__, not a variable, not an f-string,")
        print("not an attribute lookup, not a string literal. The module-prefix allowlist check")
        print("depends on this being the actual module name the decoration lives in.")

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
        print("\nNo Tier-1 decoration violations. Check passed.")

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
