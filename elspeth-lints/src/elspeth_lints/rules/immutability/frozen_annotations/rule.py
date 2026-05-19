"""Frozen-annotations immutability rule implementation."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.allowlist import FindingKey, load_allowlist
from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.immutability.frozen_annotations.metadata import RULE_ID, RULE_METADATA
from elspeth_lints.rules.immutability.shared import allowlist_path_for_root, is_frozen_dataclass, repo_relative_display_path

MUTABLE_PATTERNS = re.compile(r"\b(list|dict|set)\[")


@dataclass(frozen=True, slots=True)
class FrozenAnnotationsRule:
    """Detect mutable annotations on frozen dataclass fields."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.INCREMENTAL
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Analyze one Python syntax tree for mutable frozen dataclass annotations."""
        display = repo_relative_display_path(file_path, context.root)
        findings = find_findings(tree, display)
        allowlist_dir = (
            context.allowlist_dir_override
            if context.allowlist_dir_override is not None
            else allowlist_path_for_root(context.root, "enforce_frozen_annotations")
        )
        if not allowlist_dir.exists():
            return findings
        loaded = load_allowlist(allowlist_dir, valid_rule_ids={RULE_ID})
        return [
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
