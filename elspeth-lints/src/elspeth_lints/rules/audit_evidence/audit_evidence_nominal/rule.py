"""AuditEvidenceBase nominal inheritance rule implementation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.ast_walker import (
    ParsedPythonFile,
    PythonFileReadError,
    PythonSyntaxError,
    parse_python_file,
)
from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.audit_evidence.audit_evidence_nominal.metadata import (
    LEGACY_RULE_ID,
    RULE_ID,
    RULE_METADATA,
    SUGGESTION,
)
from elspeth_lints.rules.audit_evidence.shared import allowlist_path_for_root, display_path, iter_python_paths, load_class_allowlist


@dataclass(frozen=True, slots=True)
class AuditEvidenceNominalRule:
    """Detect classes defining to_audit_dict without directly inheriting AuditEvidenceBase."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Analyze one syntax tree for tests, or scan a whole repository root."""
        if isinstance(tree, ast.Module) and tree.body and file_path.suffix == ".py":
            return scan_tree(tree, display_path(file_path, context.root))
        return scan_root(context.root, allowlist_dir_override=context.allowlist_dir_override)


def scan_root(root: Path, *, allowlist_dir_override: Path | None = None) -> list[Finding]:
    """Scan a root for AEN1 findings and apply the legacy allowlist."""
    allowlist_dir = (
        allowlist_dir_override if allowlist_dir_override is not None else allowlist_path_for_root(root, "enforce_audit_evidence_nominal")
    )
    allowlist = load_class_allowlist(allowlist_dir)
    findings: list[Finding] = []
    for path in iter_python_paths(root):
        parsed = parse_python_file(path)
        if isinstance(parsed, PythonSyntaxError):
            raise SyntaxError(f"{parsed.path}:{parsed.line}:{parsed.column}: {parsed.message}")
        if isinstance(parsed, PythonFileReadError):
            # Mirror the syntax-error policy above: this scanner already
            # filtered candidates, so a read error indicates a race
            # between enumeration and parse — be loud, don't paper over.
            raise OSError(f"{parsed.path}: {parsed.message}")
        findings.extend(_scan_parsed(parsed, root))
    return [finding for finding in findings if allowlist.match_key(finding.fingerprint) is None]


def scan_tree(tree: ast.AST, file_path: str) -> list[Finding]:
    """Return AEN1 findings for one parsed syntax tree."""
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not _class_defines_to_audit_dict(node):
            continue
        if _bases_include_audit_evidence_base(node.bases):
            continue
        findings.append(_finding(file_path, node))
    return findings


def _scan_parsed(parsed: ParsedPythonFile, root: Path) -> list[Finding]:
    return scan_tree(parsed.tree, display_path(parsed.path, root))


def _bases_include_audit_evidence_base(bases: list[ast.expr]) -> bool:
    for base in bases:
        if isinstance(base, ast.Name) and base.id == "AuditEvidenceBase":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "AuditEvidenceBase":
            return True
    return False


def _class_defines_to_audit_dict(class_node: ast.ClassDef) -> bool:
    for node in class_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "to_audit_dict":
            return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "to_audit_dict":
                    return True
    return False


def _finding(file_path: str, node: ast.ClassDef) -> Finding:
    key = f"{file_path}:{LEGACY_RULE_ID}:{node.name}"
    return Finding(
        rule_id=LEGACY_RULE_ID,
        file_path=file_path,
        line=node.lineno,
        column=node.col_offset,
        message=f"{node.name} defines to_audit_dict without inheriting AuditEvidenceBase",
        fingerprint=key,
        severity=RULE_METADATA.severity,
        suggestion=SUGGESTION,
    )


RULE = AuditEvidenceNominalRule()
