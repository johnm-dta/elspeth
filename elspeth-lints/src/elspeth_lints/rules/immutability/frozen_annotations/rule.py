"""Frozen-annotations immutability rule implementation."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

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
        allowlist = _load_allowlist(allowlist_path_for_root(context.root, "enforce_frozen_annotations"))
        findings = find_findings(tree, display)
        return [finding for finding in findings if _finding_key(finding) not in allowlist]


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


def _finding_key(finding: Finding) -> str:
    return finding.fingerprint


def _load_allowlist(path: Path) -> set[str]:
    allowed: set[str] = set()
    if not path.exists():
        return allowed
    for yaml_file in sorted(path.glob("*.yaml") if path.is_dir() else [path]):
        data = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{yaml_file}: allowlist YAML must be a mapping")
        for entry in _list_value(data, "allow"):
            item = _mapping_value(entry, f"{yaml_file}: allow entry")
            key = item.get("key", "")
            if isinstance(key, str) and key:
                allowed.add(key)
    return allowed


def _list_value(data: dict[str, Any], key: str) -> list[object]:
    if key not in data:
        return []
    value = data[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _mapping_value(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


RULE = FrozenAnnotationsRule()
