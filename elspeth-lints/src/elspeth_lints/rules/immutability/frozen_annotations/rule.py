"""Frozen-annotations immutability rule implementation."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.allowlist import FindingKey, load_allowlist
from elspeth_lints.core.allowlist_governance import allowlist_governance_findings_for_root
from elspeth_lints.core.ast_walker import PythonFileReadError, PythonSyntaxError, walk_python_files
from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.immutability.frozen_annotations.metadata import RULE_ID, RULE_METADATA
from elspeth_lints.rules.immutability.shared import allowlist_path_for_root, is_frozen_dataclass, repo_relative_display_path

MUTABLE_PATTERNS = re.compile(r"\b(list|dict|set)\[")


@dataclass(frozen=True, slots=True)
class FrozenAnnotationsRule:
    """Detect mutable annotations on frozen dataclass fields."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Analyze one Python syntax tree for tests, or scan a whole root."""
        if isinstance(tree, ast.Module) and tree.body and file_path.suffix == ".py":
            return find_findings(tree, repo_relative_display_path(file_path, context.root))
        return scan_root(
            context.root,
            allowlist_dir_override=context.allowlist_dir_override,
            governance_emitted_dirs=context.allowlist_governance_emitted_dirs,
            emit_allowlist_governance=context.emit_allowlist_governance,
        )


def scan_root(
    root: Path,
    *,
    allowlist_dir_override: Path | None = None,
    governance_emitted_dirs: set[str] | None = None,
    emit_allowlist_governance: bool = True,
) -> list[Finding]:
    """Scan a root and evaluate the shared frozen-annotations allowlist once."""
    allowlist_dir = (
        allowlist_dir_override if allowlist_dir_override is not None else allowlist_path_for_root(root, "enforce_frozen_annotations")
    )
    loaded = load_allowlist(allowlist_dir, valid_rule_ids={RULE_ID}) if allowlist_dir.exists() else None
    findings: list[Finding] = []
    for item in walk_python_files(root):
        if isinstance(item, (PythonSyntaxError, PythonFileReadError)):
            continue
        findings.extend(find_findings(item.tree, repo_relative_display_path(item.path, root)))
    if loaded is None:
        return findings
    active = [
        finding
        for finding in findings
        if loaded.match(
            FindingKey(
                file_path=finding.file_path,
                rule_id=finding.rule_id,
                symbol_context=(),
                fingerprint=finding.fingerprint,
            )
        )
        is None
    ]
    return [
        *active,
        *allowlist_governance_findings_for_root(
            loaded,
            allowlist_dir,
            root=root,
            allowlist_dir_override=allowlist_dir_override,
            emitted_dirs=governance_emitted_dirs,
            enabled=emit_allowlist_governance,
        ),
    ]


def find_findings(tree: ast.AST, filename: str) -> list[Finding]:
    """Find mutable container annotations on frozen dataclass fields."""
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or not is_frozen_dataclass(node):
            continue
        for item in node.body:
            if not isinstance(item, ast.AnnAssign):
                continue
            if item.target is None or not isinstance(item.target, ast.Name):
                continue
            annotation = ast.unparse(item.annotation)
            if MUTABLE_PATTERNS.search(annotation):
                findings.append(_finding(filename, node.name, item.target.id, annotation, item.lineno, item.col_offset))
    return findings


def _finding(filename: str, class_name: str, field_name: str, annotation: str, line: int, column: int) -> Finding:
    key = f"{filename}:{class_name}:{field_name}"
    return Finding(
        rule_id=RULE_ID,
        file_path=filename,
        line=line,
        column=column,
        message=(
            f"Frozen dataclass field {class_name}.{field_name} uses mutable annotation {annotation}. "
            "Use Sequence/Mapping/tuple/frozenset instead of list/dict/set."
        ),
        fingerprint=key,
        severity=RULE_METADATA.severity,
        suggestion="Use Sequence/Mapping/tuple/frozenset instead of list/dict/set",
    )


RULE = FrozenAnnotationsRule()
