"""Trust-boundary tests rule implementation.

Honesty gate: every ``@trust_boundary`` must carry a ``test_ref`` that points
to a real pytest node whose body asserts on raising behaviour. Three
sub-findings:

* ``TBE1`` (MISSING): ``test_ref`` is absent or ``None``.
* ``TBE2`` (NOTFOUND): the nodeid does not resolve — file missing, or the
  named function/method does not exist in the file.
* ``TBE3`` (WEAK): the named test exists but its body contains no
  ``pytest.raises(...)``, ``with pytest.raises(...)``, or
  ``unittest.TestCase.assertRaises(...)`` pattern. The honesty contract is
  that the test must exercise malformed-input rejection, and the only
  mechanically detectable signal of that is a raising assertion.

Path resolution: ``test_ref`` is a pytest nodeid of the form
``tests/path/to/file.py::test_func`` or ``tests/path/to/file.py::TestCls::test_method``.
Paths are interpreted relative to the repository root (NOT the scan root,
which is typically ``src/elspeth``). The repository root is derived via
:func:`elspeth_lints.rules.trust_boundary.shared.repository_root`.

This rule does NOT walk imports, fixtures, or helpers. A test that delegates
its raising-assertion to a helper function would be reported as WEAK; that is
a deliberate strictness choice — the honesty signal must be visible in the
test the decorator names, not buried one indirection away.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.ast_walker import PythonSyntaxError, parse_python_file, walk_python_files
from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.trust_boundary.shared import (
    display_path,
    extract_keywords,
    iter_trust_boundary_decorators,
    repository_root,
)
from elspeth_lints.rules.trust_boundary.tests.metadata import (
    RULE_ID,
    RULE_METADATA,
    RULE_MISSING,
    RULE_NOTFOUND,
    RULE_WEAK,
    SUGGESTION_MISSING,
    SUGGESTION_NOTFOUND,
    SUGGESTION_WEAK,
)


@dataclass(frozen=True, slots=True)
class TrustBoundaryTestsRule:
    """Detect ``@trust_boundary`` decorators with missing, broken, or weak ``test_ref``."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Analyze one tree directly (for focused tests) or walk the scan root."""
        if isinstance(tree, ast.Module) and tree.body and file_path.suffix == ".py":
            return analyze_tree(
                tree,
                display_path(file_path, context.root),
                repo_root=repository_root(context.root),
            )
        return scan_root(context.root)


def analyze_tree(tree: ast.AST, file_path: str, *, repo_root: Path) -> list[Finding]:
    """Return ``trust_boundary.tests`` findings for one parsed syntax tree."""
    findings: list[Finding] = []
    for _func_node, call in iter_trust_boundary_decorators(tree):
        kwargs = extract_keywords(call)
        if kwargs is None:
            # Deferred to tier_model rule's non-literal diagnostics.
            continue
        test_ref = kwargs.get("test_ref")
        if test_ref is None:
            findings.append(
                _make_finding(
                    rule_id=RULE_MISSING,
                    file_path=file_path,
                    call=call,
                    message=(
                        "@trust_boundary has no test_ref; a behavioural test is mandatory "
                        "for trust-boundary suppressions."
                    ),
                    suggestion=SUGGESTION_MISSING,
                )
            )
            continue
        if not isinstance(test_ref, str) or not test_ref:
            # Malformed (non-string or empty) — surface as MISSING; the
            # tier_model rule will additionally surface R_TB_MALFORMED.
            findings.append(
                _make_finding(
                    rule_id=RULE_MISSING,
                    file_path=file_path,
                    call=call,
                    message=(
                        f"@trust_boundary test_ref must be a non-empty string, "
                        f"got {test_ref!r}."
                    ),
                    suggestion=SUGGESTION_MISSING,
                )
            )
            continue
        resolution = _resolve_test_ref(test_ref, repo_root)
        if resolution is None:
            findings.append(
                _make_finding(
                    rule_id=RULE_NOTFOUND,
                    file_path=file_path,
                    call=call,
                    message=(
                        f"@trust_boundary test_ref {test_ref!r} does not resolve "
                        f"(file or function not found under {repo_root})."
                    ),
                    suggestion=SUGGESTION_NOTFOUND,
                )
            )
            continue
        if not _has_raising_assertion(resolution.test_function):
            findings.append(
                _make_finding(
                    rule_id=RULE_WEAK,
                    file_path=file_path,
                    call=call,
                    message=(
                        f"@trust_boundary test_ref {test_ref!r} resolves to "
                        f"{resolution.test_function.name!r} but the test body has "
                        "no pytest.raises / with pytest.raises / assertRaises pattern."
                    ),
                    suggestion=SUGGESTION_WEAK,
                )
            )
    return findings


def scan_root(root: Path) -> list[Finding]:
    """Walk every Python file under ``root`` and aggregate findings."""
    repo_root = repository_root(root)
    findings: list[Finding] = []
    for item in walk_python_files(root):
        if isinstance(item, PythonSyntaxError):
            continue
        findings.extend(
            analyze_tree(item.tree, display_path(item.path, root), repo_root=repo_root)
        )
    return findings


@dataclass(frozen=True, slots=True)
class _ResolvedTestRef:
    """The pytest nodeid resolved to an AST FunctionDef and its host file."""

    test_function: ast.FunctionDef | ast.AsyncFunctionDef
    file_path: Path


def _resolve_test_ref(test_ref: str, repo_root: Path) -> _ResolvedTestRef | None:
    """Resolve a pytest nodeid against the repository root.

    Supports both shapes:

    * ``tests/.../test_file.py::test_function``
    * ``tests/.../test_file.py::TestClass::test_method``

    Returns ``None`` if the file doesn't exist, can't be parsed, or doesn't
    contain a function/method matching the named segments.
    """
    parts = test_ref.split("::")
    if len(parts) < 2:
        return None
    relative_file = parts[0]
    name_segments = parts[1:]
    file_path = (repo_root / relative_file).resolve()
    if not file_path.is_file():
        return None
    parsed = parse_python_file(file_path)
    if isinstance(parsed, PythonSyntaxError):
        return None
    func = _lookup_named_function(parsed.tree, name_segments)
    if func is None:
        return None
    return _ResolvedTestRef(test_function=func, file_path=file_path)


def _lookup_named_function(
    tree: ast.Module, name_segments: list[str]
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Walk ``name_segments`` against a parsed module to find the target function.

    A single segment is a module-level function. Multiple segments are
    interpreted as a class chain ending in a method
    (``TestClass::test_method`` or ``TestOuter::TestInner::test_method``).
    Returns ``None`` if any segment doesn't resolve.
    """
    namespace: list[ast.stmt] = list(tree.body)
    for index, segment in enumerate(name_segments):
        is_last = index == len(name_segments) - 1
        next_namespace: list[ast.stmt] | None = None
        for stmt in namespace:
            if is_last and isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == segment:
                return stmt
            if not is_last and isinstance(stmt, ast.ClassDef) and stmt.name == segment:
                next_namespace = list(stmt.body)
                break
        if next_namespace is None:
            return None
        namespace = next_namespace
    return None


def _has_raising_assertion(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if the function body contains a raising-assertion pattern.

    Detects three shapes:

    * ``pytest.raises(...)`` call (bare or inside any expression);
    * ``with pytest.raises(...):`` context manager (also ``async with``);
    * ``self.assertRaises(...)`` / ``cls.assertRaises(...)`` /
      bare ``assertRaises(...)`` calls (covers unittest-style tests).

    The walk is body-only: decorators, return annotations, and default values
    on inner functions are out of scope.
    """
    for statement in func.body:
        for child in _walk_statements(statement):
            if isinstance(child, ast.Call) and _is_raising_call(child.func):
                return True
            if isinstance(child, (ast.With, ast.AsyncWith)):
                for item in child.items:
                    if isinstance(item.context_expr, ast.Call) and _is_raising_call(item.context_expr.func):
                        return True
    return False


def _walk_statements(statement: ast.stmt) -> Iterator[ast.AST]:
    """Yield every AST node under a statement (the statement itself first)."""
    yield from ast.walk(statement)


def _is_raising_call(func_expr: ast.expr) -> bool:
    """Return True if ``func_expr`` references a raising-assertion helper."""
    # pytest.raises (any module.attribute spelling ending in .raises is fine —
    # pytest is conventional, but a re-export through a conftest helper named
    # raises would still be a raising assertion).
    if isinstance(func_expr, ast.Attribute) and func_expr.attr == "raises":
        return True
    # unittest assertRaises (instance method, class method, or bare).
    if isinstance(func_expr, ast.Attribute) and func_expr.attr in {"assertRaises", "assertRaisesRegex"}:
        return True
    return isinstance(func_expr, ast.Name) and func_expr.id in {"assertRaises", "assertRaisesRegex"}


def _make_finding(
    *,
    rule_id: str,
    file_path: str,
    call: ast.Call,
    message: str,
    suggestion: str,
) -> Finding:
    fingerprint = hashlib.sha256(
        f"{rule_id}|{file_path}|{call.lineno}|{call.col_offset}".encode()
    ).hexdigest()[:16]
    return Finding(
        rule_id=rule_id,
        file_path=file_path,
        line=call.lineno,
        column=call.col_offset,
        message=message,
        fingerprint=fingerprint,
        severity=RULE_METADATA.severity,
        suggestion=suggestion,
    )


RULE = TrustBoundaryTestsRule()
