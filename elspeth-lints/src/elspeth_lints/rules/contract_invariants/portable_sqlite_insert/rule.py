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
    """Reject SQLite insert usage without an explicit PostgreSQL dispatch."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.INCREMENTAL
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Analyze one Python module for SQLite-only insert builders."""
        return find_portable_sqlite_insert_findings(tree, _display_path(file_path, context.root))


def find_portable_sqlite_insert_findings(tree: ast.AST, file_path: str) -> list[Finding]:
    """Return findings for unpaired SQLite insert usage and misbound dialect branches."""
    bindings_by_call = _import_bindings(tree)
    sqlite_imports = _insert_imports(tree, module=_SQLITE_INSERT_MODULE)
    sqlite_module_calls = _insert_module_calls(tree, bindings_by_call, module=_SQLITE_INSERT_MODULE)
    if not sqlite_imports and not sqlite_module_calls:
        return []

    postgresql_imports = _insert_imports(tree, module=_POSTGRESQL_INSERT_MODULE)
    postgresql_module_calls = _insert_module_calls(tree, bindings_by_call, module=_POSTGRESQL_INSERT_MODULE)
    compared_dialects = _compared_dialect_names(tree)
    if (not postgresql_imports and not postgresql_module_calls) or not _SUPPORTED_DIALECTS.issubset(compared_dialects):
        findings = [_import_finding(file_path=file_path, node=node) for node in sqlite_imports]
        findings.extend(_module_call_finding(file_path=file_path, node=node) for node in sqlite_module_calls)
        return sorted(findings, key=lambda finding: (finding.line, finding.column))

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
        for call, branch_dialect, builder_name, builder_dialect in _misbound_builder_calls(
            tree, aliases, builder_dialects, bindings_by_call
        )
    ]


def _insert_imports(tree: ast.AST, *, module: str) -> list[ast.ImportFrom]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == module and any(alias.name == "insert" for alias in node.names)
    ]


def _import_bindings(tree: ast.AST) -> dict[ast.Call, dict[str, str]]:
    """Map each call to the import bindings visible in its lexical scope."""
    visitor = _ImportBindingVisitor()
    visitor.visit(tree)
    return visitor.bindings_by_call


class _ScopeImportCollector(ast.NodeVisitor):
    """Collect import bindings owned by one lexical scope."""

    def __init__(self) -> None:
        self.bindings: dict[str, str] = {}
        self.conflicted: set[str] = set()

    def _bind(self, name: str, target: str) -> None:
        if self.bindings.get(name, target) != target:
            self.conflicted.add(name)
        self.bindings[name] = target

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.asname is not None:
                self._bind(alias.asname, alias.name)
            else:
                root = alias.name.partition(".")[0]
                self._bind(root, root)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None or node.level != 0:
            return
        for alias in node.names:
            if alias.name != "*":
                self._bind(alias.asname or alias.name, f"{node.module}.{alias.name}")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


@dataclass(frozen=True, slots=True)
class _BindingScope:
    kind: str
    inherited: dict[str, str]
    resolved: dict[str, str]


class _ImportBindingVisitor(ast.NodeVisitor):
    """Record import bindings at calls while preserving lexical-scope ownership."""

    def __init__(self) -> None:
        self.bindings_by_call: dict[ast.Call, dict[str, str]] = {}
        self._scopes: list[_BindingScope] = []

    def visit_Module(self, node: ast.Module) -> None:
        self._visit_scope(node.body, kind="module")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_definition(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_definition(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_arguments(node.args)
        self._visit_scope([node.body], kind="function")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword)
        for type_param in getattr(node, "type_params", ()):
            self.visit(type_param)
        self._visit_scope(node.body, kind="class")

    def visit_Call(self, node: ast.Call) -> None:
        if self._scopes:
            self.bindings_by_call[node] = self._scopes[-1].resolved
        self.generic_visit(node)

    def _visit_function_definition(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_arguments(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        for type_param in getattr(node, "type_params", ()):
            self.visit(type_param)
        self._visit_scope(node.body, kind="function")

    def _visit_arguments(self, arguments: ast.arguments) -> None:
        for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs):
            if argument.annotation is not None:
                self.visit(argument.annotation)
        for optional_argument in (arguments.vararg, arguments.kwarg):
            if optional_argument is not None and optional_argument.annotation is not None:
                self.visit(optional_argument.annotation)
        for default in (*arguments.defaults, *(default for default in arguments.kw_defaults if default is not None)):
            self.visit(default)

    def _visit_scope(self, body: list[ast.stmt] | list[ast.expr], *, kind: str) -> None:
        inherited = self._child_scope_bindings()
        collector = _ScopeImportCollector()
        for node in body:
            collector.visit(node)

        resolved = dict(inherited)
        resolved.update(collector.bindings)
        for name in collector.conflicted:
            resolved.pop(name, None)

        self._scopes.append(_BindingScope(kind=kind, inherited=inherited, resolved=resolved))
        for node in body:
            self.visit(node)
        self._scopes.pop()

    def _child_scope_bindings(self) -> dict[str, str]:
        if not self._scopes:
            return {}
        parent = self._scopes[-1]
        # Class bodies execute in their enclosing scope, but class-local names
        # are not closure bindings inherited by methods or nested classes.
        return dict(parent.inherited if parent.kind == "class" else parent.resolved)


def _insert_module_calls(tree: ast.AST, bindings_by_call: dict[ast.Call, dict[str, str]], *, module: str) -> list[ast.Call]:
    """Return calls that reach ``<module>.insert`` through an imported module binding."""
    target = f"{module}.insert"
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and _dotted_call_target(node.func, bindings_by_call.get(node, {})) == target
    ]


def _dotted_call_target(func: ast.Attribute, bindings: dict[str, str]) -> str | None:
    """Resolve an attribute-chain call target to its import-derived dotted path."""
    parts: list[str] = []
    node: ast.expr = func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    base = bindings.get(node.id)
    if base is None:
        return None
    parts.append(base)
    return ".".join(reversed(parts))


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
    tree: ast.AST,
    aliases: frozenset[str],
    builder_dialects: dict[str, str],
    bindings_by_call: dict[ast.Call, dict[str, str]],
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
        if branch_dialect is not None and isinstance(node, ast.Call):
            builder = _call_builder(node, bindings_by_call.get(node, {}), builder_dialects)
            if builder is not None:
                builder_name, builder_dialect = builder
                if builder_dialect != branch_dialect:
                    misbound.append((node, branch_dialect, builder_name, builder_dialect))
        for child in ast.iter_child_nodes(node):
            visit(child, branch_dialect)

    visit(tree, None)
    return sorted(misbound, key=lambda item: (item[0].lineno, item[0].col_offset))


def _call_builder(node: ast.Call, bindings: dict[str, str], builder_dialects: dict[str, str]) -> tuple[str, str] | None:
    """Return the display name and owning dialect of the insert builder a call invokes, if any."""
    if isinstance(node.func, ast.Name):
        dialect = builder_dialects.get(node.func.id)
        return None if dialect is None else (node.func.id, dialect)
    if isinstance(node.func, ast.Attribute):
        dotted = _dotted_call_target(node.func, bindings)
        for module, dialect in ((_SQLITE_INSERT_MODULE, "sqlite"), (_POSTGRESQL_INSERT_MODULE, "postgresql")):
            if dotted == f"{module}.insert":
                return ast.unparse(node.func), dialect
    return None


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


def _module_call_finding(*, file_path: str, node: ast.Call) -> Finding:
    return _node_finding(
        file_path=file_path,
        node=node,
        message=(
            "SQLite-specific insert builder call has no complete PostgreSQL counterpart and explicit "
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
