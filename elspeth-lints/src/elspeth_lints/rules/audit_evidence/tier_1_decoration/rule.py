"""Tier-1 decoration rule implementation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.ast_walker import PythonFileReadError, PythonSyntaxError, parse_python_file
from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.audit_evidence.shared import (
    allowlist_path_for_root,
    display_path,
    load_class_allowlist,
    repo_relative_display_path,
    tier_1_error_call,
)
from elspeth_lints.rules.audit_evidence.tier_1_decoration.metadata import (
    RULE_ID,
    RULE_METADATA,
    RULE_TDE1,
    RULE_TDE2,
    SUGGESTION_TDE1,
    SUGGESTION_TDE2,
)

_CHECKED_SUFFIXES = ("Error", "Violation")


@dataclass(frozen=True, slots=True)
class Tier1DecorationRule:
    """Detect missing Tier-1/Tier-2 exception class declarations."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Analyze one syntax tree for tests, or scan the configured errors.py."""
        if isinstance(tree, ast.Module) and tree.body and file_path.suffix == ".py":
            return scan_tree(tree, display_path(file_path, context.root), _source_lines(file_path))
        return scan_root(context.root, allowlist_dir_override=context.allowlist_dir_override)


def scan_root(root: Path, *, allowlist_dir_override: Path | None = None) -> list[Finding]:
    """Scan the legacy errors.py target, or fixture roots that lack it."""
    allowlist_dir = (
        allowlist_dir_override if allowlist_dir_override is not None else allowlist_path_for_root(root, "enforce_tier_1_decoration")
    )
    allowlist = load_class_allowlist(allowlist_dir)
    candidates = _scan_candidates(root)
    findings: list[Finding] = []
    for path in candidates:
        parsed = parse_python_file(path)
        if isinstance(parsed, PythonSyntaxError):
            raise SyntaxError(f"{parsed.path}:{parsed.line}:{parsed.column}: {parsed.message}")
        if isinstance(parsed, PythonFileReadError):
            # This scanner enumerates candidates via :func:`_scan_candidates`
            # (which already walked the filesystem to find them), so a
            # read error here means the file became unreadable between
            # enumeration and parse — a race that should be loud rather
            # than silently skipped, matching the existing syntax-error
            # behaviour above.
            raise OSError(f"{parsed.path}: {parsed.message}")
        display = repo_relative_display_path(path, root) if path.name == "errors.py" else display_path(path, root)
        findings.extend(scan_tree(parsed.tree, display, parsed.source.splitlines()))
    return [finding for finding in findings if finding.rule_id != RULE_TDE1 or allowlist.match_key(finding.fingerprint) is None]


def scan_tree(tree: ast.AST, file_path: str, source_lines: list[str]) -> list[Finding]:
    """Return TDE1/TDE2 findings for one parsed syntax tree."""
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            finding = _tde1_finding(file_path, node, source_lines)
            if finding is not None:
                findings.append(finding)
                continue
        if isinstance(node, ast.Call) and tier_1_error_call(node):
            detail = _caller_module_violation(node)
            if detail is not None:
                findings.append(_tde2_finding(file_path, node, detail))
    return findings


def _scan_candidates(root: Path) -> list[Path]:
    repo_target = root / "src" / "elspeth" / "contracts" / "errors.py"
    if repo_target.exists():
        return [repo_target]
    source_target = root / "contracts" / "errors.py"
    if source_target.exists():
        return [source_target]
    return sorted(root.rglob("*.py"))


def _tde1_finding(file_path: str, node: ast.ClassDef, source_lines: list[str]) -> Finding | None:
    if not node.name.endswith(_CHECKED_SUFFIXES):
        return None
    if _has_tier_1_error_decorator(node):
        return None
    if _has_tier_2_comment(node, source_lines):
        return None
    key = f"{file_path}:{RULE_TDE1}:{node.name}"
    return Finding(
        rule_id=RULE_TDE1,
        file_path=file_path,
        line=node.lineno,
        column=node.col_offset,
        message=f"{node.name} has no @tier_1_error(reason=...) decorator and no # TIER-2: justification comment",
        fingerprint=key,
        severity=RULE_METADATA.severity,
        suggestion=SUGGESTION_TDE1,
    )


def _tde2_finding(file_path: str, node: ast.Call, detail: str) -> Finding:
    key = f"{file_path}:{RULE_TDE2}:{node.lineno}"
    return Finding(
        rule_id=RULE_TDE2,
        file_path=file_path,
        line=node.lineno,
        column=node.col_offset,
        message=detail,
        fingerprint=key,
        severity=RULE_METADATA.severity,
        suggestion=SUGGESTION_TDE2,
    )


def _has_tier_1_error_decorator(class_node: ast.ClassDef) -> bool:
    return any(isinstance(decorator, ast.Call) and tier_1_error_call(decorator) for decorator in class_node.decorator_list)


def _has_tier_2_comment(class_node: ast.ClassDef, source_lines: list[str]) -> bool:
    decorator_lines: set[int] = set()
    for decorator in class_node.decorator_list:
        for lineno in range(decorator.lineno, (decorator.end_lineno or decorator.lineno) + 1):
            decorator_lines.add(lineno)

    idx = class_node.lineno - 2
    while idx >= 0:
        line = source_lines[idx] if idx < len(source_lines) else ""
        stripped = line.strip()
        if not stripped or (idx + 1) in decorator_lines:
            idx -= 1
            continue
        marker = "# TIER-2:"
        if marker not in line:
            return False
        return bool(line[line.index(marker) + len(marker) :].strip())
    return False


def _caller_module_violation(call: ast.Call) -> str | None:
    caller_module_kw: ast.keyword | None = None
    for keyword in call.keywords:
        if keyword.arg == "caller_module":
            caller_module_kw = keyword
            break
    if caller_module_kw is None:
        return "missing caller_module kwarg (require caller_module=__name__)"
    value = caller_module_kw.value
    if not (isinstance(value, ast.Name) and value.id == "__name__"):
        return f"caller_module value must be the __name__ literal, got {ast.dump(value)}"
    return None


def _source_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


RULE = Tier1DecorationRule()
