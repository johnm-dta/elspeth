#!/usr/bin/env python3
"""
Tier Model Enforcement Tool

AST-based static analysis that detects defensive programming patterns that
violate the three-tier trust model and fails CI unless explicitly allowlisted.

Enforces ELSPETH's data manifesto (see CLAUDE.md):
- Tier 1 (Audit Database): Full trust - crash on any anomaly
- Tier 2 (Pipeline Data): Elevated trust - expect types, wrap operations on values
- Tier 3 (External Data): Zero trust - validate at boundary, coerce where possible

Usage:
    python scripts/cicd/enforce_tier_model.py check --root src
    python scripts/cicd/enforce_tier_model.py check --root src --allowlist config/cicd/enforce_tier_model
    python scripts/cicd/enforce_tier_model.py check --root src --allowlist config/cicd/enforce_tier_model.yaml
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

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
    fingerprint: str
    code_snippet: str
    message: str

    @property
    def canonical_key(self) -> str:
        """Generate the canonical key for allowlist matching."""
        symbol_part = ":".join(self.symbol_context) if self.symbol_context else "_module_"
        return f"{self.file_path}:{self.rule_id}:{symbol_part}:fp={self.fingerprint}"

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
    source_file: str = field(default="", compare=False)


@dataclass
class PerFileRule:
    """A per-file rule that whitelists all patterns of specified rules for a file/directory."""

    pattern: str  # Glob pattern like "plugins/llm/*" or exact path like "mcp/server.py"
    rules: list[str]  # List of rule IDs like ["R1", "R4", "R6"]
    reason: str
    expires: date | None
    max_hits: int | None = None  # Cap on allowed matches; None = unlimited
    matched_count: int = field(default=0, compare=False)
    source_file: str = field(default="", compare=False)

    def matches(self, file_path: str, rule_id: str) -> bool:
        """Check if this per-file rule matches a finding."""
        import fnmatch

        if rule_id not in self.rules:
            return False
        # Match against file path (supports glob patterns)
        return fnmatch.fnmatch(file_path, self.pattern)


@dataclass
class Allowlist:
    """Parsed allowlist configuration."""

    entries: list[AllowlistEntry]
    per_file_rules: list[PerFileRule] = field(default_factory=list)
    fail_on_stale: bool = True
    fail_on_expired: bool = True

    def match(self, finding: Finding) -> AllowlistEntry | PerFileRule | None:
        """Check if a finding is covered by an allowlist entry or per-file rule.

        Per-file rules are checked first (more general), then specific entries.
        """
        # Check per-file rules first
        for rule in self.per_file_rules:
            if rule.matches(finding.file_path, finding.rule_id):
                rule.matched_count += 1
                return rule

        # Check specific entries
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

    def get_expired_file_rules(self) -> list[PerFileRule]:
        """Return per-file rules that have expired."""
        today = datetime.now(UTC).date()
        return [r for r in self.per_file_rules if r.expires and r.expires < today]

    def get_unused_file_rules(self) -> list[PerFileRule]:
        """Return per-file rules that didn't match any finding."""
        return [r for r in self.per_file_rules if r.matched_count == 0]

    def get_exceeded_file_rules(self) -> list[PerFileRule]:
        """Return per-file rules where matched_count exceeds max_hits."""
        return [r for r in self.per_file_rules if r.max_hits is not None and r.matched_count > r.max_hits]


# =============================================================================
# Rule Definitions
# =============================================================================

RULES: dict[str, dict[str, Any]] = {
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
        "description": "hasattr() is banned — use isinstance, protocols, or try/except AttributeError",
        "remediation": "Replace with isinstance() for type checks, try/except AttributeError for attribute probing, or protocols for structural typing",
        "banned": True,
    },
    "R4": {
        "name": "broad-except",
        "description": "Broad exception handling can suppress bugs",
        "remediation": "Catch specific exceptions, or re-raise after logging/quarantining",
    },
    "TC": {
        "name": "type-checking-layer",
        "description": "TYPE_CHECKING import crosses layer boundary (annotation-only, no runtime coupling)",
        "remediation": "Allowlist if the dependency is accepted, or move the type to a lower layer",
    },
    "R5": {
        "name": "isinstance",
        "description": "isinstance() checks can mask contract violations outside explicit trust boundaries",
        "remediation": "Validate at Tier-3 boundaries or rely on contracts; do not use isinstance to hide bugs",
    },
    "R6": {
        "name": "silent-except",
        "description": "Exception handling that swallows errors without re-raise or explicit error result",
        "remediation": "Raise the exception or return an explicit error/quarantine result",
    },
    "R7": {
        "name": "contextlib.suppress",
        "description": "contextlib.suppress() silently ignores exceptions",
        "remediation": "Handle exceptions explicitly or allow them to raise",
    },
    "R8": {
        "name": "dict.setdefault",
        "description": "dict.setdefault() hides missing-key bugs by mutating defaults",
        "remediation": "Access keys directly and fix the schema/contract if KeyError occurs",
    },
    "R9": {
        "name": "dict.pop-default",
        "description": "dict.pop(key, default) hides missing-key bugs with implicit defaults",
        "remediation": "Access keys directly and fix the schema/contract if KeyError occurs",
    },
    "L1": {
        "name": "upward-import",
        "description": "Import from a higher layer violates the dependency hierarchy (contracts→core→engine→plugins)",
        "remediation": "Move code down, extract primitives, or restructure caller (see CLAUDE.md Layer Dependency Rules)",
    },
}


# =============================================================================
# Layer Hierarchy (import direction enforcement)
# =============================================================================

# Layer numbers: lower = deeper (fewer allowed dependencies)
LAYER_HIERARCHY: dict[str, int] = {
    "contracts": 0,  # L0 — leaf, imports nothing above
    "core": 1,  # L1 — can import contracts only
    "engine": 2,  # L2 — can import core, contracts
}
# Everything else (plugins, mcp, tui, telemetry, testing, cli*) is implicitly L3.

LAYER_NAMES: dict[int, str] = {
    0: "L0/contracts",
    1: "L1/core",
    2: "L2/engine",
    3: "L3/application",
}


def _get_file_layer(relative_path: str) -> int:
    """Determine the layer from a path relative to the scan root.

    Supports both ``--root=src/elspeth`` (paths like ``contracts/...``) and
    ``--root=src`` (paths like ``elspeth/contracts/...``).
    """
    parts = relative_path.split("/")
    idx = 1 if parts[0] == "elspeth" else 0
    top = parts[idx] if idx < len(parts) else ""
    return LAYER_HIERARCHY.get(top, 3)


def _get_import_target_layer(module_name: str) -> int | None:
    """Determine the target layer from a fully qualified import.

    Returns None for non-elspeth imports.
    """
    if not module_name.startswith("elspeth."):
        return None
    parts = module_name.split(".")
    if len(parts) < 2:
        return None
    return LAYER_HIERARCHY.get(parts[1], 3)


def _find_type_checking_lines(tree: ast.Module) -> set[int]:
    """Collect line numbers of import statements inside ``if TYPE_CHECKING:`` blocks."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            for child in node.body:
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    lines.add(child.lineno)
    return lines


# =============================================================================
# AST Visitor
# =============================================================================


class TierModelVisitor(ast.NodeVisitor):
    """AST visitor that detects bug-hiding patterns."""

    def __init__(self, file_path: str, source_lines: list[str]) -> None:
        self.file_path = file_path
        self.source_lines = source_lines
        self.findings: list[Finding] = []
        self.symbol_stack: list[str] = []
        self.path_stack: list[str] = []
        self._decorator_lines: set[int] = set()  # Track lines that are decorators

    def _get_code_snippet(self, lineno: int) -> str:
        """Get the source line for a given line number."""
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return "<source unavailable>"

    def _fingerprint_node(self, rule_id: str, node: ast.AST) -> str:
        """Generate a stable fingerprint for a finding.

        The fingerprint is based on:
        - rule_id (ensures distinctness across rules)
        - AST path (field/index path from root, stable across formatting)
        - AST dump without line/column attributes
        """
        ast_path = "/".join(self.path_stack)
        node_dump = ast.dump(node, include_attributes=False, annotate_fields=True)
        payload = f"{rule_id}|{ast_path}|{node_dump}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _add_finding(self, rule_id: str, node: ast.expr | ast.stmt | ast.ExceptHandler, message: str) -> None:
        """Record a finding."""
        self.findings.append(
            Finding(
                rule_id=rule_id,
                file_path=self.file_path,
                line=node.lineno,
                col=node.col_offset,
                symbol_context=tuple(self.symbol_stack),
                fingerprint=self._fingerprint_node(rule_id, node),
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
        # Collect decorator lines — .get() calls here are not dict access
        for decorator in node.decorator_list:
            self._decorator_lines.add(decorator.lineno)
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Track async function context."""
        # Collect decorator lines — .get() calls here are not dict access
        for decorator in node.decorator_list:
            self._decorator_lines.add(decorator.lineno)
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    def _is_default_return_value(self, value: ast.expr | None) -> bool:
        """True if return value is a silent default (None, empty container, empty string, zero)."""
        if value is None:
            return True
        if isinstance(value, ast.Constant):
            return value.value in (None, "", 0, 0.0, False)
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            return len(value.elts) == 0
        if isinstance(value, ast.Dict):
            return len(value.keys) == 0
        return False

    def _handler_is_silent(self, node: ast.ExceptHandler) -> bool:
        """Return True if the except handler swallows errors without re-raise or explicit return."""
        has_raise = any(isinstance(child, ast.Raise) for child in ast.walk(node))
        if has_raise:
            return False

        returns: list[ast.Return] = [child for child in ast.walk(node) if isinstance(child, ast.Return)]
        if returns:
            # If all returns are silent defaults, treat as swallow.
            return all(self._is_default_return_value(ret.value) for ret in returns)

        # No raise, no return: likely swallow (even if logging).
        return True

    def _is_likely_non_dict_get(self, node: ast.Call) -> bool:
        """Return True if this .get() call is likely NOT a dict.get().

        Heuristics (conservative — only skip when confident):
        1. Decorator context: @router.get("/path") is a route decorator
        2. URL-like first arg: client.get("https://...") is an HTTP method
        3. ChromaDB keywords: collection.get(ids=[...]) is SDK retrieval

        Note: f-string URLs (client.get(f"/api/{id}")) are NOT filtered
        because we cannot statically determine their runtime value.
        These must be allowlisted if they are legitimate HTTP calls.
        """
        # Heuristic 1: Decorator context
        if node.lineno in self._decorator_lines:
            return True

        # Heuristic 2: URL-like first argument
        if node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                val = first_arg.value
                if val.startswith(("/", "http://", "https://")):
                    return True

        # Heuristic 3: ChromaDB-specific keywords
        # IMPORTANT: Only include keywords that are unambiguous to ChromaDB/vector DBs.
        # Generic pagination keywords (limit, offset) are NOT included because they
        # collide with SQLAlchemy, Django ORM, and other common patterns.
        chromadb_keywords = {"ids", "include", "where"}
        call_keywords = {kw.arg for kw in node.keywords if kw.arg is not None}
        return bool(call_keywords & chromadb_keywords)

    def visit_Call(self, node: ast.Call) -> None:
        """Detect R1 (dict.get), R2 (getattr), R3 (hasattr), R5 (isinstance), R8/R9 defaults."""
        # R1: dict.get() - Call(func=Attribute(attr="get"))
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get" and not self._is_likely_non_dict_get(node):
            self._add_finding(
                "R1",
                node,
                f"Potential dict.get() usage: {self._get_code_snippet(node.lineno)}",
            )

        # R8: dict.setdefault() - mutating default on missing key
        if isinstance(node.func, ast.Attribute) and node.func.attr == "setdefault":
            self._add_finding(
                "R8",
                node,
                f"dict.setdefault() hides missing keys: {self._get_code_snippet(node.lineno)}",
            )

        # R9: dict.pop(key, default) - implicit default on missing key
        if isinstance(node.func, ast.Attribute) and node.func.attr == "pop" and (len(node.args) >= 2 or node.keywords):
            self._add_finding(
                "R9",
                node,
                f"dict.pop() with default hides missing keys: {self._get_code_snippet(node.lineno)}",
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

        # R5: isinstance() - runtime type checks can mask contract violations
        if isinstance(node.func, ast.Name) and node.func.id == "isinstance":
            self._add_finding(
                "R5",
                node,
                f"isinstance() used: {self._get_code_snippet(node.lineno)}",
            )

        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        """Detect R7: contextlib.suppress usage."""
        for item in node.items:
            ctx_expr = item.context_expr
            if isinstance(ctx_expr, ast.Call):
                func = ctx_expr.func
                if (isinstance(func, ast.Name) and func.id == "suppress") or (isinstance(func, ast.Attribute) and func.attr == "suppress"):
                    self._add_finding(
                        "R7",
                        node,
                        f"contextlib.suppress() used: {self._get_code_snippet(node.lineno)}",
                    )
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        """Detect R7: contextlib.suppress usage in async context managers."""
        for item in node.items:
            ctx_expr = item.context_expr
            if isinstance(ctx_expr, ast.Call):
                func = ctx_expr.func
                if (isinstance(func, ast.Name) and func.id == "suppress") or (isinstance(func, ast.Attribute) and func.attr == "suppress"):
                    self._add_finding(
                        "R7",
                        node,
                        f"contextlib.suppress() used: {self._get_code_snippet(node.lineno)}",
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

        # R6: specific exception swallowed without re-raise or explicit return
        if not is_broad and self._handler_is_silent(node):
            self._add_finding(
                "R6",
                node,
                f"Exception swallowed without re-raise or explicit error: {self._get_code_snippet(node.lineno)}",
            )

        self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        """Visit a node, tracking AST path for stable fingerprints."""
        for field_name, value in ast.iter_fields(node):
            if isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        self.path_stack.append(f"{field_name}[{index}]")
                        self.visit(item)
                        self.path_stack.pop()
            elif isinstance(value, ast.AST):
                self.path_stack.append(field_name)
                self.visit(value)
                self.path_stack.pop()


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

    visitor = TierModelVisitor(relative_path, source_lines)
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
        relative = py_file.relative_to(root)
        # Skip vendored/third-party directories
        if any(part in _ALWAYS_EXCLUDED_DIRS for part in relative.parts):
            continue
        # Check user-specified exclusions
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
# Layer Import Scanning
# =============================================================================


def scan_layer_imports_file(
    file_path: Path,
    root: Path,
) -> tuple[list[Finding], list[Finding]]:
    """Scan a single file for upward layer imports.

    Returns:
        violations: Findings for runtime upward imports (fail CI unless allowlisted)
        tc_findings: Findings for TYPE_CHECKING upward imports (warnings, allowlistable)
    """
    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return [], []

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return [], []

    relative_path = str(file_path.relative_to(root))
    source_lines = source.splitlines()
    file_layer = _get_file_layer(relative_path)

    # L3 files (plugins, mcp, tui, etc.) can import anything — skip
    if file_layer >= 3:
        return [], []

    tc_lines = _find_type_checking_lines(tree)
    violations: list[Finding] = []
    tc_findings: list[Finding] = []

    for node in ast.walk(tree):
        # Collect (module_name, line, col) targets from import nodes
        targets: list[tuple[str, int, int]] = []
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            targets.append((node.module, node.lineno, node.col_offset))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                targets.append((alias.name, node.lineno, node.col_offset))

        for module_name, line, col in targets:
            target_layer = _get_import_target_layer(module_name)
            if target_layer is None or target_layer <= file_layer:
                continue

            snippet = source_lines[line - 1].strip() if line <= len(source_lines) else "<source unavailable>"
            from_name = LAYER_NAMES[file_layer]
            to_name = LAYER_NAMES[target_layer]

            if line in tc_lines:
                tc_payload = f"TC|{relative_path}|{module_name}"
                tc_fp = hashlib.sha256(tc_payload.encode()).hexdigest()[:16]
                tc_findings.append(
                    Finding(
                        rule_id="TC",
                        file_path=relative_path,
                        line=line,
                        col=col,
                        symbol_context=(),
                        fingerprint=tc_fp,
                        code_snippet=snippet,
                        message=f"TYPE_CHECKING import: {from_name} annotates with {to_name} ({module_name})",
                    )
                )
            else:
                # Fingerprint: keyed on file + imported module (stable across reformatting)
                payload = f"L1|{relative_path}|{module_name}"
                fp = hashlib.sha256(payload.encode()).hexdigest()[:16]

                violations.append(
                    Finding(
                        rule_id="L1",
                        file_path=relative_path,
                        line=line,
                        col=col,
                        symbol_context=(),
                        fingerprint=fp,
                        code_snippet=snippet,
                        message=f"Upward import: {from_name} imports from {to_name} ({module_name})",
                    )
                )

    return violations, tc_findings


def scan_layer_imports_directory(
    root: Path,
    exclude_patterns: list[str] | None = None,
) -> tuple[list[Finding], list[Finding]]:
    """Scan all Python files for upward layer imports."""
    exclude_patterns = exclude_patterns or []
    all_violations: list[Finding] = []
    all_tc_findings: list[Finding] = []

    for py_file in root.rglob("*.py"):
        relative = py_file.relative_to(root)
        # Skip vendored/third-party directories
        if any(part in _ALWAYS_EXCLUDED_DIRS for part in relative.parts):
            continue
        # Check user-specified exclusions
        skip = False
        for pattern in exclude_patterns:
            if relative.match(pattern) or str(relative).startswith(pattern.rstrip("*/")):
                skip = True
                break
        if skip:
            continue

        violations, tc_findings = scan_layer_imports_file(py_file, root)
        all_violations.extend(violations)
        all_tc_findings.extend(tc_findings)

    return all_violations, all_tc_findings


# =============================================================================
# Edge-Dump Mode (Phase 0 — L3↔L3 import-graph oracle, Δ2 dump-edges)
#
# Additive subcommand. Shares the path→layer table and AST-based import walking
# of `check`, but emits the full intra-layer edge graph rather than a violations
# list. Always exits 0 unless the tool itself errors. NEVER fails the build.
# =============================================================================


_LAYER_NAME_TO_INT: dict[str, int] = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


@dataclass(frozen=True)
class _ImportSite:
    """A single import statement that produces one edge contribution."""

    src_file: str  # relative path under --root
    line: int
    target_module: str  # fully qualified target (e.g. "elspeth.plugins.transforms.llm")
    type_checking: bool
    conditional: bool
    reexport: bool


def _module_name_to_path(module_name: str, root: Path) -> Path | None:
    """Resolve `elspeth.X.Y.Z` to a file path under root.

    Tries submodule (`X/Y/Z.py`) first, then package (`X/Y/Z/__init__.py`).
    Returns None for non-elspeth modules or modules that don't resolve.
    """
    if not module_name.startswith("elspeth"):
        return None
    parts = module_name.split(".")
    if parts[0] != "elspeth":
        return None
    rel_parts = parts[1:]
    if not rel_parts:
        return None
    candidate = root.joinpath(*rel_parts).with_suffix(".py")
    if candidate.is_file():
        return candidate
    pkg_init = root.joinpath(*rel_parts, "__init__.py")
    if pkg_init.is_file():
        return pkg_init
    return None


def _resolve_import_target(
    module_name: str,
    imported_name: str | None,
    root: Path,
) -> Path | None:
    """Resolve an import statement to a target file.

    For ``from M import N``, try ``M.N`` (submodule) first, then ``M`` (package).
    For ``import M``, resolve M directly.
    """
    if imported_name is not None:
        sub = _module_name_to_path(f"{module_name}.{imported_name}", root)
        if sub is not None:
            return sub
    return _module_name_to_path(module_name, root)


def _resolve_relative_module(
    relative_path: str,
    level: int,
    module_name: str | None,
) -> str | None:
    """Resolve a ``from .x import y`` (level≥1) to its absolute ``elspeth.<...>`` form.

    ``relative_path`` is the importing file's path relative to --root.
    Returns the absolute module name, or None if the relative reference is invalid.
    """
    if level == 0:
        return module_name
    pkg_parts = list(Path(relative_path).parent.parts)
    # `from . import X` from a file inside `pkg/` means "X within pkg" — level=1, drop 0 parts.
    # `from .. import X` means "X within parent of pkg" — level=2, drop 1 part. Etc.
    drop = level - 1
    if drop > len(pkg_parts):
        return None
    base_parts = pkg_parts[: len(pkg_parts) - drop] if drop > 0 else pkg_parts
    suffix_parts: list[str] = []
    if module_name:
        suffix_parts.extend(module_name.split("."))
    full_parts = ["elspeth", *base_parts, *suffix_parts]
    return ".".join(full_parts)


def _find_conditional_import_lines(tree: ast.Module) -> set[int]:
    """Collect line numbers of imports inside non-TYPE_CHECKING ``if`` or ``try`` blocks.

    Imports inside ``if TYPE_CHECKING:`` are NOT included here (they are tagged
    via the dedicated TYPE_CHECKING line set).
    """
    cond_lines: set[int] = set()
    for node in ast.walk(tree):
        is_conditional = False
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                continue
            is_conditional = True
        elif isinstance(node, ast.Try):
            is_conditional = True
        if is_conditional:
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    cond_lines.add(child.lineno)
    return cond_lines


def _file_subsystem(relative_path: str) -> str:
    """Collapse a file path to its parent-directory (Python-package) granularity.

    ``plugins/transforms/llm/azure_batch.py`` → ``plugins/transforms/llm``.
    ``plugins/__init__.py`` → ``plugins``.
    Top-level files (``cli.py``) → ``.`` (root marker).
    """
    parent = str(Path(relative_path).parent)
    return parent if parent != "." else "."


def _file_loc(path: Path) -> int:
    """Count lines in a Python file. Returns 0 on read error."""
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeDecodeError):
        return 0


def _gather_import_sites(
    py_file: Path,
    relative_path: str,
    tree: ast.Module,
    tc_lines: set[int],
    cond_lines: set[int],
) -> list[tuple[str | None, ast.Import | ast.ImportFrom, str, str | None]]:
    """Walk a tree and emit (resolved_module_name, node, target_alias_name, raw_relative_module).

    Returned list is consumed by the edge-builder.  ``resolved_module_name`` is
    the fully-qualified target (after relative-import resolution). The node is
    always either ``ast.Import`` or ``ast.ImportFrom`` (both carry ``lineno``).
    """
    out: list[tuple[str | None, ast.Import | ast.ImportFrom, str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            resolved = _resolve_relative_module(relative_path, node.level, node.module)
            for alias in node.names:
                out.append((resolved, node, alias.name, node.module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, node, "", None))
    return out


def scan_dump_edges(
    root: Path,
    include_layers: frozenset[int],
    collapse_to_subsystem: bool,
    exclude_patterns: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[list[str]]]:
    """Build the edge graph for ``dump-edges``.

    Returns (nodes, edges, sccs).  All collections are sorted deterministically.
    """
    exclude_patterns = exclude_patterns or []

    file_count: dict[str, int] = {}
    file_loc: dict[str, int] = {}
    node_layer: dict[str, int] = {}
    raw_edges: list[tuple[str, str, _ImportSite]] = []

    for py_file in sorted(root.rglob("*.py")):
        relative = py_file.relative_to(root)
        if any(part in _ALWAYS_EXCLUDED_DIRS for part in relative.parts):
            continue
        skip = False
        for pattern in exclude_patterns:
            if relative.match(pattern) or str(relative).startswith(pattern.rstrip("*/")):
                skip = True
                break
        if skip:
            continue

        rel_str = str(relative)
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            print(f"Warning: could not parse {py_file}: {exc}", file=sys.stderr)
            continue

        src_layer = _get_file_layer(rel_str)
        src_node_id = _file_subsystem(rel_str) if collapse_to_subsystem else rel_str

        # Always update node stats so unknown-layer nodes can still be excluded
        # in the final filter (consistent stats regardless of edge inclusion).
        file_count[src_node_id] = file_count.get(src_node_id, 0) + 1
        file_loc[src_node_id] = file_loc.get(src_node_id, 0) + len(source.splitlines())
        node_layer.setdefault(src_node_id, src_layer)

        if src_layer not in include_layers:
            continue

        tc_lines = _find_type_checking_lines(tree)
        cond_lines = _find_conditional_import_lines(tree)
        is_init = py_file.name == "__init__.py"

        for resolved_module, node, alias_name, _raw_module in _gather_import_sites(py_file, rel_str, tree, tc_lines, cond_lines):
            if resolved_module is None:
                continue
            if not resolved_module.startswith("elspeth"):
                continue

            target_path = _resolve_import_target(resolved_module, alias_name if alias_name else None, root)
            if target_path is None:
                continue

            try:
                target_rel = target_path.relative_to(root)
            except ValueError:
                continue
            target_rel_str = str(target_rel)
            target_layer = _get_file_layer(target_rel_str)
            if target_layer not in include_layers:
                continue

            tgt_node_id = _file_subsystem(target_rel_str) if collapse_to_subsystem else target_rel_str
            if collapse_to_subsystem and src_node_id == tgt_node_id:
                continue  # Δ3 rule 6: drop intra-subsystem self-edges with collapse ON.

            line = node.lineno
            is_tc = line in tc_lines
            is_cond = line in cond_lines
            # Δ3 rule 9: re-export when source is __init__.py AND it's a relative import.
            # Conservative: only relative imports inside __init__.py count; absolute imports
            # in __init__.py are also re-exports if target is a sibling submodule, but the
            # signal would be too noisy without __all__ tracking. Sticking to the relative-
            # import heuristic that catches the common pattern from ADR-006 era.
            is_reexport = bool(is_init and isinstance(node, ast.ImportFrom) and node.level > 0)

            target_module_qualified = f"{resolved_module}.{alias_name}" if alias_name else resolved_module
            raw_edges.append(
                (
                    src_node_id,
                    tgt_node_id,
                    _ImportSite(
                        src_file=rel_str,
                        line=line,
                        target_module=target_module_qualified,
                        type_checking=is_tc,
                        conditional=is_cond,
                        reexport=is_reexport,
                    ),
                )
            )

    # Aggregate by (src, tgt). Edge attributes use AND-aggregation: an aggregated
    # edge is type_checking_only iff EVERY underlying site is TC (any non-TC site
    # means the edge has runtime coupling). Same for conditional and reexport.
    edge_buckets: dict[tuple[str, str], list[_ImportSite]] = {}
    for src, tgt, site in raw_edges:
        edge_buckets.setdefault((src, tgt), []).append(site)

    edges_out: list[dict[str, Any]] = []
    for (src, tgt), sites in sorted(edge_buckets.items()):
        sites_sorted = sorted(sites, key=lambda s: (s.src_file, s.line, s.target_module))
        edges_out.append(
            {
                "from": src,
                "to": tgt,
                "weight": len(sites),
                "type_checking_only": all(s.type_checking for s in sites),
                "conditional": all(s.conditional for s in sites),
                "reexport": all(s.reexport for s in sites),
                "sample_sites": [{"file": s.src_file, "line": s.line} for s in sites_sorted[:3]],
            }
        )

    nodes_out: list[dict[str, Any]] = []
    edge_endpoint_ids: set[str] = set()
    for e in edges_out:
        edge_endpoint_ids.add(e["from"])
        edge_endpoint_ids.add(e["to"])

    for nid in sorted(file_count.keys()):
        layer = node_layer[nid]
        # Include the node only if its layer is in the include set.
        # Endpoint reachability ensures targets pulled in by edges are also included.
        if layer not in include_layers and nid not in edge_endpoint_ids:
            continue
        nodes_out.append(
            {
                "id": nid,
                "layer": LAYER_NAMES[layer],
                "file_count": file_count[nid],
                "loc": file_loc[nid],
            }
        )

    # SCC detection — Δ5 mandate. Edge endpoints not previously in nodes_out (edges may
    # cross into other layers if include_layers spans more than one) are added so the
    # graph is closed. Non-trivial SCCs (size ≥ 2) reported.
    try:
        import networkx as nx
    except ImportError:
        # NetworkX is in the project dep stack (per CLAUDE.md), so this is a tooling
        # break, not a normal failure. Surface it loudly and emit no SCCs.
        print("Warning: networkx unavailable; SCC detection skipped.", file=sys.stderr)
        return nodes_out, edges_out, []

    graph: nx.DiGraph[str] = nx.DiGraph()
    for n in nodes_out:
        graph.add_node(n["id"])
    for e in edges_out:
        graph.add_edge(e["from"], e["to"])

    sccs_raw = [sorted(scc) for scc in nx.strongly_connected_components(graph) if len(scc) >= 2]
    sccs_raw.sort(key=lambda items: (len(items), items[0] if items else ""))

    return nodes_out, edges_out, sccs_raw


# =============================================================================
# Edge-Dump Output Formatters (JSON / Mermaid / DOT)
# =============================================================================


_STABLE_PLACEHOLDER = "<stable>"


def _tool_version_for_dump(use_stable_placeholder: bool) -> str:
    """Return a tool-version identifier.

    Uses a content hash of the enforcer file (cheaper and more deterministic than
    a git rev-parse subprocess; effectively the same identifying property).
    """
    if use_stable_placeholder:
        return _STABLE_PLACEHOLDER
    try:
        own_path = Path(__file__).resolve()
        digest = hashlib.sha256(own_path.read_bytes()).hexdigest()[:12]
        return f"sha256:{digest}"
    except (OSError, NameError):
        return "unknown"


def _generated_at(use_stable_placeholder: bool) -> str:
    if use_stable_placeholder:
        return _STABLE_PLACEHOLDER
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def render_dump_edges_json(
    *,
    root: Path,
    include_layers: frozenset[int],
    collapse_to_subsystem: bool,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    sccs: list[list[str]],
    use_stable_placeholder: bool,
) -> str:
    """Render the edge graph as deterministic JSON (Δ4 schema)."""
    layer_names_sorted = sorted(LAYER_NAMES[layer] for layer in include_layers)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": _generated_at(use_stable_placeholder),
        "tool_version": _tool_version_for_dump(use_stable_placeholder),
        "scope": {
            "root": str(root).replace("\\", "/"),
            "layers_included": layer_names_sorted,
            "collapsed_to_subsystem": collapse_to_subsystem,
        },
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "type_checking_edges": sum(1 for e in edges if e["type_checking_only"]),
            "conditional_edges": sum(1 for e in edges if e["conditional"]),
            "reexport_edges": sum(1 for e in edges if e["reexport"]),
            "scc_count": len(sccs),
            "largest_scc_size": max((len(s) for s in sccs), default=0),
        },
        "strongly_connected_components": sccs,
    }
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def _mermaid_safe(node_id: str) -> str:
    """Sanitize a node id for Mermaid (no slashes, dots, etc.)."""
    return node_id.replace("/", "_").replace(".", "_").replace("-", "_") or "_root_"


def render_dump_edges_mermaid(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> str:
    """Render the edge graph as a Mermaid flowchart with subsystem subgraphs."""
    lines: list[str] = ["flowchart LR"]
    # Group nodes by their first path segment (top-level subsystem) for subgraphs.
    groups: dict[str, list[dict[str, Any]]] = {}
    for n in nodes:
        top = n["id"].split("/", 1)[0] if "/" in n["id"] else n["id"]
        groups.setdefault(top, []).append(n)

    for top in sorted(groups):
        safe_top = _mermaid_safe(top)
        lines.append(f"    subgraph {safe_top}[{top}]")
        for n in sorted(groups[top], key=lambda x: x["id"]):
            safe_id = _mermaid_safe(n["id"])
            label = n["id"]
            lines.append(f'        {safe_id}["{label}<br/><sub>{n["loc"]} LOC</sub>"]')
        lines.append("    end")

    for e in edges:
        src = _mermaid_safe(e["from"])
        tgt = _mermaid_safe(e["to"])
        if e["type_checking_only"]:
            arrow = "-.->|TC|"
        elif e["conditional"]:
            arrow = "-.->|cond|"
        elif e["weight"] >= 10:
            arrow = f"==>|{e['weight']}|"
        else:
            arrow = f"-->|{e['weight']}|" if e["weight"] > 1 else "-->"
        lines.append(f"    {src} {arrow} {tgt}")

    return "\n".join(lines) + "\n"


def _dot_safe(node_id: str) -> str:
    return '"' + node_id.replace('"', '\\"') + '"'


def render_dump_edges_dot(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> str:
    """Render the edge graph as a Graphviz digraph."""
    lines: list[str] = ["digraph l3_imports {", "    rankdir=LR;", "    node [shape=box, style=rounded];"]

    # Cluster by top-level subsystem.
    groups: dict[str, list[dict[str, Any]]] = {}
    for n in nodes:
        top = n["id"].split("/", 1)[0] if "/" in n["id"] else n["id"]
        groups.setdefault(top, []).append(n)

    for cluster_idx, top in enumerate(sorted(groups)):
        lines.append(f'    subgraph "cluster_{cluster_idx}" {{')
        lines.append(f"        label={_dot_safe(top)};")
        lines.append("        style=dashed;")
        for n in sorted(groups[top], key=lambda x: x["id"]):
            label = f"{n['id']}\\n{n['loc']} LOC"
            lines.append(f"        {_dot_safe(n['id'])} [label={_dot_safe(label)}];")
        lines.append("    }")

    for e in edges:
        attrs: list[str] = []
        if e["type_checking_only"]:
            attrs.append('style="dashed"')
            attrs.append('label="TC"')
        elif e["conditional"]:
            attrs.append('style="dotted"')
            attrs.append('label="cond"')
        else:
            attrs.append(f'label="{e["weight"]}"')
            if e["weight"] >= 10:
                attrs.append("penwidth=3")
        attr_str = ", ".join(attrs)
        lines.append(f"    {_dot_safe(e['from'])} -> {_dot_safe(e['to'])} [{attr_str}];")

    lines.append("}")
    return "\n".join(lines) + "\n"


# =============================================================================
# Allowlist Handling
# =============================================================================


_BANNED_RULES = frozenset(rule_id for rule_id, rule_def in RULES.items() if rule_def.get("banned"))
_ALL_RULE_IDS = frozenset(RULES.keys())

# Directories that are always excluded from scanning — vendored/third-party code
# that happens to contain .py files but is not part of the ELSPETH codebase.
_ALWAYS_EXCLUDED_DIRS = ("node_modules",)


def _parse_allow_hits(data: dict[str, Any], source_file: str = "") -> list[AllowlistEntry]:
    """Parse allow_hits entries from a YAML data dict."""
    entries: list[AllowlistEntry] = []
    for item in data.get("allow_hits", []):
        key = item.get("key", "")
        source_ctx = f" in {source_file}" if source_file else ""
        parts = key.split(":")
        if len(parts) < 2:
            print(
                f"Error: allow_hits entry has malformed key (expected 'file:rule_id:...' format){source_ctx}: {key!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        rule_id = parts[1]
        if rule_id in _BANNED_RULES:
            print(
                f"Error: allow_hits entry uses banned rule {rule_id} (cannot be allowlisted){source_ctx}: {key}",
                file=sys.stderr,
            )
            sys.exit(1)
        if rule_id not in _ALL_RULE_IDS:
            print(
                f"Error: allow_hits entry has unknown rule ID '{rule_id}'{source_ctx}: {key}",
                file=sys.stderr,
            )
            sys.exit(1)
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
                source_file=source_file,
            )
        )
    return entries


def _parse_per_file_rules(data: dict[str, Any], source_file: str = "") -> list[PerFileRule]:
    """Parse per_file_rules entries from a YAML data dict."""
    per_file_rules: list[PerFileRule] = []
    for item in data.get("per_file_rules", []):
        rule_ids = set(item.get("rules", []))
        banned_in_entry = rule_ids & _BANNED_RULES
        if banned_in_entry:
            print(
                f"Error: per_file_rules entry for '{item.get('pattern', '?')}' uses banned rule(s) "
                f"{banned_in_entry} (cannot be allowlisted)",
                file=sys.stderr,
            )
            sys.exit(1)
        unknown_in_entry = rule_ids - _ALL_RULE_IDS
        if unknown_in_entry:
            source_ctx = f" in {source_file}" if source_file else ""
            print(
                f"Error: per_file_rules entry for '{item.get('pattern', '?')}'{source_ctx} uses unknown rule ID(s) {unknown_in_entry}",
                file=sys.stderr,
            )
            sys.exit(1)
        expires_str = item.get("expires")
        expires_date = None
        if expires_str:
            try:
                expires_date = datetime.strptime(expires_str, "%Y-%m-%d").replace(tzinfo=UTC).date()
            except ValueError:
                print(
                    f"Warning: Invalid date format for per_file_rules expires: {expires_str}",
                    file=sys.stderr,
                )

        raw_max_hits = item.get("max_hits")
        max_hits: int | None = None
        if raw_max_hits is not None:
            try:
                max_hits = int(raw_max_hits)
            except ValueError:
                pattern = item.get("pattern", "?")
                source_ctx = f" in {source_file}" if source_file else ""
                print(
                    f"Error: per_file_rules entry for '{pattern}'{source_ctx} has non-numeric max_hits: {raw_max_hits!r}",
                    file=sys.stderr,
                )
                sys.exit(1)

        per_file_rules.append(
            PerFileRule(
                pattern=item["pattern"],
                rules=item.get("rules", []),
                reason=item.get("reason", ""),
                expires=expires_date,
                max_hits=max_hits,
                source_file=source_file,
            )
        )
    return per_file_rules


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """Load and return a YAML file as a dict, exiting on parse error."""
    try:
        with path.open() as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in allowlist {path}: {e}", file=sys.stderr)
        sys.exit(1)


def load_allowlist_from_directory(directory: Path) -> Allowlist:
    """Load allowlist from a directory of per-module YAML files.

    Expected structure:
        directory/
            _defaults.yaml   — version and defaults section
            core.yaml         — per_file_rules + allow_hits for core/*
            plugins.yaml      — per_file_rules + allow_hits for plugins/*
            ...
    """
    # Load defaults
    defaults_path = directory / "_defaults.yaml"
    if defaults_path.exists():
        defaults_data = _load_yaml_file(defaults_path)
        defaults = defaults_data.get("defaults", {})
    else:
        defaults = {}

    # Glob all YAML files except _defaults.yaml, sorted by filename
    yaml_files = sorted(f for f in directory.glob("*.yaml") if f.name != "_defaults.yaml")

    all_entries: list[AllowlistEntry] = []
    all_per_file_rules: list[PerFileRule] = []

    for yaml_file in yaml_files:
        data = _load_yaml_file(yaml_file)
        all_entries.extend(_parse_allow_hits(data, source_file=yaml_file.name))
        all_per_file_rules.extend(_parse_per_file_rules(data, source_file=yaml_file.name))

    return Allowlist(
        entries=all_entries,
        per_file_rules=all_per_file_rules,
        fail_on_stale=defaults.get("fail_on_stale", True),
        fail_on_expired=defaults.get("fail_on_expired", True),
    )


def load_allowlist(path: Path) -> Allowlist:
    """Load and parse the allowlist from a YAML file or directory of YAML files."""
    if path.is_dir():
        return load_allowlist_from_directory(path)

    if not path.exists():
        return Allowlist(entries=[])

    data = _load_yaml_file(path)
    defaults = data.get("defaults", {})

    return Allowlist(
        entries=_parse_allow_hits(data),
        per_file_rules=_parse_per_file_rules(data),
        fail_on_stale=defaults.get("fail_on_stale", True),
        fail_on_expired=defaults.get("fail_on_expired", True),
    )


# =============================================================================
# Reporting
# =============================================================================


def _suggest_module_file(finding: Finding, allowlist_path: Path) -> str:
    """Suggest the appropriate module YAML file for a finding.

    Maps the finding's file path to the per-module YAML file name.
    Only meaningful when allowlist_path is a directory.
    """
    if not allowlist_path.is_dir():
        return str(allowlist_path)

    file_path = finding.file_path
    # Bare filenames (no /) like cli.py, cli_helpers.py → cli.yaml
    if "/" not in file_path:
        stem = file_path.removesuffix(".py")
        if stem.startswith("cli"):
            return str(allowlist_path / "cli.yaml")
        return str(allowlist_path / f"{stem}.yaml")

    # First path segment determines module file
    module = file_path.split("/", 1)[0]
    return str(allowlist_path / f"{module}.yaml")


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
    base = f"\n  Key: {entry.key}\n  Owner: {entry.owner}\n  Reason: {entry.reason}"
    if entry.source_file:
        base += f"\n  Source: {entry.source_file}"
    return base


def format_expired_entry_text(entry: AllowlistEntry) -> str:
    """Format an expired allowlist entry for text output."""
    return f"\n  Key: {entry.key}\n  Owner: {entry.owner}\n  Expired: {entry.expires}"


def report_json(
    violations: list[Finding],
    stale_entries: list[AllowlistEntry],
    expired_entries: list[AllowlistEntry],
    expired_file_rules: list[PerFileRule] | None = None,
    unused_file_rules: list[PerFileRule] | None = None,
    layer_warnings: list[Finding] | None = None,
    exceeded_file_rules: list[PerFileRule] | None = None,
) -> str:
    """Generate JSON report."""
    result: dict[str, Any] = {
        "violations": [
            {
                "rule_id": f.rule_id,
                "file": f.file_path,
                "line": f.line,
                "col": f.col,
                "context": list(f.symbol_context),
                "fingerprint": f.fingerprint,
                "code": f.code_snippet,
                "message": f.message,
                "key": f.canonical_key,
            }
            for f in violations
        ],
        "stale_allowlist_entries": [{"key": e.key, "owner": e.owner, "reason": e.reason} for e in stale_entries],
        "expired_allowlist_entries": [{"key": e.key, "owner": e.owner, "expires": str(e.expires)} for e in expired_entries],
    }
    if expired_file_rules:
        result["expired_file_rules"] = [
            {"pattern": r.pattern, "rules": r.rules, "reason": r.reason, "expires": str(r.expires)} for r in expired_file_rules
        ]
    if unused_file_rules:
        result["unused_file_rules"] = [{"pattern": r.pattern, "rules": r.rules, "reason": r.reason} for r in unused_file_rules]
    if exceeded_file_rules:
        result["exceeded_file_rules"] = [
            {"pattern": r.pattern, "rules": r.rules, "matched": r.matched_count, "max_hits": r.max_hits, "reason": r.reason}
            for r in exceeded_file_rules
        ]
    if layer_warnings:
        result["layer_warnings"] = [
            {
                "rule_id": f.rule_id,
                "file": f.file_path,
                "line": f.line,
                "message": f.message,
                "key": f.canonical_key,
            }
            for f in layer_warnings
        ]
    return json.dumps(result, indent=2)


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
        help="Path to allowlist YAML file or directory of YAML files",
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
        "files",
        nargs="*",
        type=Path,
        help="Specific files to check (from pre-commit). If empty, scans --root directory.",
    )

    # dump-edges subcommand (Phase 0 — L3↔L3 import-graph oracle)
    dump_parser = subparsers.add_parser(
        "dump-edges",
        help="Emit a deterministic import-graph for analysis (does NOT fail on graph content)",
    )
    dump_parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory to scan",
    )
    dump_parser.add_argument(
        "--format",
        choices=["json", "mermaid", "dot"],
        default="json",
        help="Output format (default: json)",
    )
    dump_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (required for json/dot; mermaid may write to stdout when omitted)",
    )
    dump_parser.add_argument(
        "--include-layer",
        action="append",
        default=None,
        choices=["L0", "L1", "L2", "L3"],
        help="Layer(s) to include in the graph; repeatable. Default: L3 only.",
    )
    dump_parser.add_argument(
        "--collapse-to-subsystem",
        dest="collapse_to_subsystem",
        action="store_true",
        default=True,
        help="Aggregate edges to package (parent-directory) granularity (default: ON)",
    )
    dump_parser.add_argument(
        "--no-collapse",
        dest="collapse_to_subsystem",
        action="store_false",
        help="Disable subsystem collapse; emit file-level edges",
    )
    dump_parser.add_argument(
        "--no-timestamp",
        action="store_true",
        default=False,
        help="Replace generated_at and tool_version with stable placeholders for diff-friendly output",
    )
    dump_parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob patterns to exclude (repeatable)",
    )

    args = parser.parse_args()

    if args.command == "check":
        return run_check(args)
    if args.command == "dump-edges":
        return run_dump_edges(args)

    return 0


def run_dump_edges(args: argparse.Namespace) -> int:
    """Run the dump-edges command.

    Always exits 0 unless the tool itself errors. Cycle detection is observational —
    a stderr WARNING is printed when non-trivial SCCs are found, but the exit code
    is unaffected (Δ5: not enforcement).
    """
    root = args.root.resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        return 1

    # Resolve include-layer set (default: L3 only).
    layer_strs = args.include_layer or ["L3"]
    include_layers = frozenset(_LAYER_NAME_TO_INT[s] for s in layer_strs)

    # JSON and DOT require --output; Mermaid may write to stdout.
    if args.format in ("json", "dot") and args.output is None:
        print(f"Error: --output is required for --format {args.format}", file=sys.stderr)
        return 1

    nodes, edges, sccs = scan_dump_edges(
        root=root,
        include_layers=include_layers,
        collapse_to_subsystem=args.collapse_to_subsystem,
        exclude_patterns=args.exclude,
    )

    if args.format == "json":
        rendered = render_dump_edges_json(
            root=args.root,  # the un-resolved value, so output is portable
            include_layers=include_layers,
            collapse_to_subsystem=args.collapse_to_subsystem,
            nodes=nodes,
            edges=edges,
            sccs=sccs,
            use_stable_placeholder=args.no_timestamp,
        )
    elif args.format == "mermaid":
        rendered = render_dump_edges_mermaid(nodes, edges)
    elif args.format == "dot":
        rendered = render_dump_edges_dot(nodes, edges)
    else:
        # argparse's `choices` should make this unreachable.
        print(f"Error: unknown format {args.format!r}", file=sys.stderr)
        return 1

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        # Mermaid → stdout when --output omitted
        print(rendered, end="")

    if sccs:
        print(
            f"WARNING: {len(sccs)} non-trivial strongly-connected component(s) detected at "
            f"{','.join(sorted(LAYER_NAMES[layer] for layer in include_layers))}.",
            file=sys.stderr,
        )
        if args.output is not None:
            print(
                f"         See {args.output} stats.scc_count for details.",
                file=sys.stderr,
            )

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
        # Default: prefer directory, fall back to single file
        repo_root = Path(__file__).parent.parent.parent
        dir_path = repo_root / "config" / "cicd" / "enforce_tier_model"
        file_path = repo_root / "config" / "cicd" / "enforce_tier_model.yaml"
        allowlist_path = dir_path if dir_path.is_dir() else file_path

    allowlist = load_allowlist(allowlist_path)

    # Scan for findings - either specific files or whole directory
    all_tc_findings: list[Finding] = []
    if args.files:
        # Pre-commit mode: only scan the provided files that are under root
        all_findings = []
        for file_path in args.files:
            resolved = file_path.resolve()
            # Only scan files that are under the root directory
            try:
                resolved.relative_to(root)
                all_findings.extend(scan_file(resolved, root))
                layer_v, layer_tc = scan_layer_imports_file(resolved, root)
                all_findings.extend(layer_v)
                all_tc_findings.extend(layer_tc)
            except ValueError:
                # File is not under root, skip it
                pass
    else:
        # Full directory scan mode
        all_findings = scan_directory(root, args.exclude)
        layer_v, layer_tc = scan_layer_imports_directory(root, args.exclude)
        all_findings.extend(layer_v)
        all_tc_findings.extend(layer_tc)

    # Filter out allowlisted findings (banned rules are never suppressible)
    violations: list[Finding] = []
    for finding in all_findings:
        if finding.rule_id in _BANNED_RULES or allowlist.match(finding) is None:
            violations.append(finding)

    # Filter TYPE_CHECKING findings through allowlist (unmatched remain as warnings)
    layer_warnings: list[Finding] = []
    for tc_finding in all_tc_findings:
        if allowlist.match(tc_finding) is None:
            layer_warnings.append(tc_finding)

    # Check for stale/expired/exceeded allowlist entries (only in full-scan mode)
    # In file-specific mode (pre-commit), we only scan a subset of files,
    # so match counts and staleness are non-deterministic — suppress all.
    if args.files:
        stale_entries: list[AllowlistEntry] = []
        expired_entries: list[AllowlistEntry] = []
        expired_file_rules: list[PerFileRule] = []
        unused_file_rules: list[PerFileRule] = []
        exceeded_file_rules: list[PerFileRule] = []
    else:
        stale_entries = allowlist.get_stale_entries() if allowlist.fail_on_stale else []
        expired_entries = allowlist.get_expired_entries() if allowlist.fail_on_expired else []
        expired_file_rules = allowlist.get_expired_file_rules() if allowlist.fail_on_expired else []
        unused_file_rules = allowlist.get_unused_file_rules() if allowlist.fail_on_stale else []
        exceeded_file_rules = allowlist.get_exceeded_file_rules()

    # Report results
    # Include unused_file_rules in error condition - stale per-file rules should fail
    # the same way stale explicit entries do when fail_on_stale is enabled
    has_errors = bool(violations or stale_entries or expired_entries or expired_file_rules or unused_file_rules or exceeded_file_rules)

    if args.format == "json":
        print(
            report_json(
                violations, stale_entries, expired_entries, expired_file_rules, unused_file_rules, layer_warnings, exceeded_file_rules
            )
        )
    else:
        # Text format
        if violations:
            print(f"\n{'=' * 60}")
            print(f"VIOLATIONS FOUND: {len(violations)}")
            print("=" * 60)
            for v in violations:
                print(format_finding_text(v))

        if layer_warnings:
            print(f"\n{'=' * 60}")
            print(f"LAYER WARNINGS (TYPE_CHECKING imports): {len(layer_warnings)}")
            print("(Allowlist with rule TC to suppress — not a failure)")
            print("=" * 60)
            for w in layer_warnings:
                print(f"  {w.file_path}:{w.line} — {w.message}")
                print(f"    Code: {w.code_snippet}")
                print(f"    Allowlist key: {w.canonical_key}")

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

        if expired_file_rules:
            print(f"\n{'=' * 60}")
            print(f"EXPIRED PER-FILE RULES: {len(expired_file_rules)}")
            print("(These rules have passed their expiration date)")
            print("=" * 60)
            for r in expired_file_rules:
                print(f"\n  Pattern: {r.pattern}")
                print(f"  Rules: {r.rules}")
                print(f"  Reason: {r.reason}")
                print(f"  Expired: {r.expires}")

        if unused_file_rules:
            print(f"\n{'=' * 60}")
            print(f"UNUSED PER-FILE RULES: {len(unused_file_rules)}")
            print("(These rules didn't match any code - consider removing)")
            print("=" * 60)
            for r in unused_file_rules:
                print(f"\n  Pattern: {r.pattern}")
                print(f"  Rules: {r.rules}")
                print(f"  Reason: {r.reason}")

        if exceeded_file_rules:
            print(f"\n{'=' * 60}")
            print(f"EXCEEDED PER-FILE RULES: {len(exceeded_file_rules)}")
            print("(These rules matched more findings than max_hits allows - review new additions)")
            print("=" * 60)
            for r in exceeded_file_rules:
                print(f"\n  Pattern: {r.pattern}")
                print(f"  Rules: {r.rules}")
                print(f"  Matched: {r.matched_count} (max_hits: {r.max_hits})")
                print(f"  Reason: {r.reason}")

        if has_errors:
            print(f"\n{'=' * 60}")
            print("CHECK FAILED")
            print("=" * 60)
            if violations:
                target = _suggest_module_file(violations[0], allowlist_path)
                print(f"\nTo allowlist a violation, add an entry to {target}")
                print("Example entry:")
                if violations:
                    import pprint

                    pprint.pprint(violations[0].suggested_allowlist_entry())
        else:
            print("\nNo bug-hiding patterns detected. Check passed.")

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
