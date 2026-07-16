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
    """Return findings for unpaired SQLite insert imports and misbound dialect branches."""
    sqlite_imports = _insert_imports(tree, module=_SQLITE_INSERT_MODULE)
    if not sqlite_imports:
        return []

    postgresql_imports = _insert_imports(tree, module=_POSTGRESQL_INSERT_MODULE)
    compared_dialects = _compared_dialect_names(tree)
    if not postgresql_imports or not _SUPPORTED_DIALECTS.issubset(compared_dialects):
        return [_import_finding(file_path=file_path, node=node) for node in sqlite_imports]

    builder_dialects = _builder_dialects(sqlite_imports, postgresql_imports)
    aliases = _dialect_name_aliases(tree)
    return [
        _misbound_branch_finding(
            file_path=file_path,
            node=call,
            branch_dialect=branch_dialect,
            builder_name=builder_name,
            builder_dialect=builder_dialect,
        )
        for call, branch_dialect, builder_name, builder_dialect in _misbound_builder_calls(tree, aliases, builder_dialects)
    ]


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


def _builder_dialects(sqlite_imports: list[ast.ImportFrom], postgresql_imports: list[ast.ImportFrom]) -> dict[str, str]:
    """Map each imported insert-builder name to its owning dialect; ambiguous names are dropped."""
    dialects: dict[str, str] = {}
    ambiguous: set[str] = set()
    for dialect, imports in (("sqlite", sqlite_imports), ("postgresql", postgresql_imports)):
        for node in imports:
            for alias in node.names:
                if alias.name != "insert":
                    continue
                bound_name = alias.asname or alias.name
                if dialects.get(bound_name, dialect) != dialect:
                    ambiguous.add(bound_name)
                dialects[bound_name] = dialect
    for name in ambiguous:
        del dialects[name]
    return dialects


def _misbound_builder_calls(
    tree: ast.AST, aliases: frozenset[str], builder_dialects: dict[str, str]
) -> list[tuple[ast.Call, str, str, str]]:
    """Return builder calls bound to a dialect branch that executes the other dialect's builder.

    Each call is attributed to its innermost enclosing single-dialect guard, so
    nested dispatches rebind their own bodies rather than inheriting the outer
    branch's dialect.
    """
    misbound: list[tuple[ast.Call, str, str, str]] = []

    def visit(node: ast.AST, branch_dialect: str | None) -> None:
        if isinstance(node, ast.If):
            guard_dialect = _single_dialect_guard(node.test, aliases)
            visit(node.test, branch_dialect)
            for stmt in node.body:
                visit(stmt, guard_dialect if guard_dialect is not None else branch_dialect)
            for stmt in node.orelse:
                visit(stmt, branch_dialect)
            return
        if branch_dialect is not None and isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            builder_dialect = builder_dialects.get(node.func.id)
            if builder_dialect is not None and builder_dialect != branch_dialect:
                misbound.append((node, branch_dialect, node.func.id, builder_dialect))
        for child in ast.iter_child_nodes(node):
            visit(child, branch_dialect)

    visit(tree, None)
    return sorted(misbound, key=lambda item: (item[0].lineno, item[0].col_offset))


def _single_dialect_guard(test: ast.expr, aliases: frozenset[str]) -> str | None:
    """Return the sole supported dialect a branch guard selects, if statically known."""
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return None
    op = test.ops[0]
    if not isinstance(op, (ast.Eq, ast.In)):
        return None
    left, right = test.left, test.comparators[0]
    if _is_dialect_name_expression(left, aliases):
        literals = _dialect_literals(right)
    elif isinstance(op, ast.Eq) and _is_dialect_name_expression(right, aliases):
        literals = _dialect_literals(left)
    else:
        return None
    if len(literals) != 1:
        return None
    return next(iter(literals))


def _dialect_literals(node: ast.expr) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value} if node.value in _SUPPORTED_DIALECTS else set()
    if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        values: set[str] = set()
        for element in node.elts:
            values.update(_dialect_literals(element))
        return values
    return set()


def _import_finding(*, file_path: str, node: ast.ImportFrom) -> Finding:
    return _node_finding(
        file_path=file_path,
        node=node,
        message=(
            "SQLite-specific insert import has no complete PostgreSQL counterpart and explicit "
            "sqlite/postgresql dialect dispatch in this module."
        ),
    )


def _misbound_branch_finding(*, file_path: str, node: ast.Call, branch_dialect: str, builder_name: str, builder_dialect: str) -> Finding:
    return _node_finding(
        file_path=file_path,
        node=node,
        message=(
            f"Dialect branch for {branch_dialect!r} invokes the {builder_dialect} insert builder "
            f"{builder_name!r}; bind each dialect branch to the insert builder it actually executes."
        ),
    )


def _node_finding(*, file_path: str, node: ast.AST, message: str) -> Finding:
    lineno = getattr(node, "lineno", 0)
    col_offset = getattr(node, "col_offset", 0)
    payload = f"{RULE_ID}|{file_path}|{lineno}|{col_offset}|{ast.dump(node, include_attributes=False)}"
    return Finding(
        rule_id=RULE_ID,
        file_path=file_path,
        line=lineno,
        column=col_offset,
        message=message,
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
