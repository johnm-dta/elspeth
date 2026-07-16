"""Portable SQLite insert-builder rule implementation."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.contract_invariants.portable_sqlite_insert.metadata import RULE_ID, RULE_METADATA, SUGGESTION

_SQLITE_INSERT_MODULE = "sqlalchemy.dialects.sqlite"
_POSTGRESQL_INSERT_MODULE = "sqlalchemy.dialects.postgresql"
_SUPPORTED_DIALECTS = frozenset({"sqlite", "postgresql"})


@dataclass(frozen=True, slots=True)
class PortableSQLiteInsertRule:
    """Reject SQLite insert imports without an explicit PostgreSQL dispatch."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.INCREMENTAL
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Analyze one Python module for SQLite-only insert builders."""
        return find_portable_sqlite_insert_findings(tree, _display_path(file_path, context.root))


def find_portable_sqlite_insert_findings(tree: ast.AST, file_path: str) -> list[Finding]:
    """Return one finding per SQLite insert import in an unpaired module."""
    sqlite_imports = _insert_imports(tree, module=_SQLITE_INSERT_MODULE)
    if not sqlite_imports:
        return []

    has_postgresql_insert = bool(_insert_imports(tree, module=_POSTGRESQL_INSERT_MODULE))
    compared_dialects = _compared_dialect_names(tree)
    if has_postgresql_insert and _SUPPORTED_DIALECTS.issubset(compared_dialects):
        return []

    return [_finding(file_path=file_path, node=node) for node in sqlite_imports]


def _insert_imports(tree: ast.AST, *, module: str) -> list[ast.ImportFrom]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == module and any(alias.name == "insert" for alias in node.names)
    ]


def _compared_dialect_names(tree: ast.AST) -> frozenset[str]:
    aliases = _dialect_name_aliases(tree)
    compared: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        for left, right in pairwise(operands):
            if _is_dialect_name_expression(left, aliases):
                compared.update(_dialect_literals(right))
            if _is_dialect_name_expression(right, aliases):
                compared.update(_dialect_literals(left))
    return frozenset(compared)


def _dialect_name_aliases(tree: ast.AST) -> frozenset[str]:
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if value is not None and _is_direct_dialect_name(value):
                aliases.update(target.id for target in targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name) and _is_direct_dialect_name(node.value):
            aliases.add(node.target.id)
    return frozenset(aliases)


def _is_dialect_name_expression(node: ast.expr, aliases: frozenset[str]) -> bool:
    return _is_direct_dialect_name(node) or (isinstance(node, ast.Name) and node.id in aliases)


def _is_direct_dialect_name(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute) and node.attr == "name" and isinstance(node.value, ast.Attribute) and node.value.attr == "dialect"
    )


def _dialect_literals(node: ast.expr) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value} if node.value in _SUPPORTED_DIALECTS else set()
    if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        values: set[str] = set()
        for element in node.elts:
            values.update(_dialect_literals(element))
        return values
    return set()


def _finding(*, file_path: str, node: ast.ImportFrom) -> Finding:
    payload = f"{RULE_ID}|{file_path}|{node.lineno}|{node.col_offset}|{ast.dump(node, include_attributes=False)}"
    return Finding(
        rule_id=RULE_ID,
        file_path=file_path,
        line=node.lineno,
        column=node.col_offset,
        message=(
            "SQLite-specific insert import has no complete PostgreSQL counterpart and explicit "
            "sqlite/postgresql dialect dispatch in this module."
        ),
        fingerprint=hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
        severity=RULE_METADATA.severity,
        suggestion=SUGGESTION,
    )


def _display_path(file_path: Path, root: Path) -> str:
    try:
        return file_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return file_path.as_posix()


RULE = PortableSQLiteInsertRule()
