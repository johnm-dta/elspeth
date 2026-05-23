"""Trust-boundary scope rule implementation.

Honesty gate: ``@trust_boundary(source_param=X)`` must name a parameter of
the decorated function, AND that parameter must be read at least once in the
body. The runtime decorator already raises :class:`TypeError` for the
"not a parameter" case at import time; this static rule catches the wider
failure mode where ``source_param`` is a parameter but the body never reads
from it (a structurally inert decorator).

A "read" is any :class:`ast.Name` in :class:`ast.Load` context that resolves
to the parameter name, walked over the entire function body. Annotations,
default values, and decorator expressions on inner functions are excluded —
they belong to the enclosing scope, not to the boundary's data flow.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.ast_walker import PythonSyntaxError, walk_python_files
from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.trust_boundary.scope.metadata import (
    RULE_DEAD,
    RULE_ID,
    RULE_METADATA,
    RULE_NOPARAM,
    SUGGESTION_DEAD,
    SUGGESTION_NOPARAM,
)
from elspeth_lints.rules.trust_boundary.shared import display_path, extract_keywords, iter_trust_boundary_decorators


@dataclass(frozen=True, slots=True)
class TrustBoundaryScopeRule:
    """Detect ``@trust_boundary`` decorators with absent or dead ``source_param``."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Analyze one tree directly (for focused tests) or walk the scan root."""
        if isinstance(tree, ast.Module) and tree.body and file_path.suffix == ".py":
            return analyze_tree(tree, display_path(file_path, context.root))
        return scan_root(context.root)


def analyze_tree(tree: ast.AST, file_path: str) -> list[Finding]:
    """Return ``trust_boundary.scope`` findings for one parsed syntax tree."""
    findings: list[Finding] = []
    for func_node, call in iter_trust_boundary_decorators(tree):
        kwargs = extract_keywords(call)
        if kwargs is None:
            # Non-literal kwargs: deferred to ``trust_tier.tier_model``.
            continue
        source_param = kwargs.get("source_param")
        if not isinstance(source_param, str) or not source_param:
            # Missing or malformed source_param is the tier_model rule's
            # R_TB_MALFORMED territory; do not double-report here.
            continue
        param_names = _parameter_names(func_node)
        if source_param not in param_names:
            findings.append(
                _make_finding(
                    rule_id=RULE_NOPARAM,
                    file_path=file_path,
                    call=call,
                    message=(
                        f"@trust_boundary(source_param={source_param!r}) does not name a parameter of "
                        f"{func_node.name!r}; declared parameters are {tuple(param_names)!r}."
                    ),
                    suggestion=SUGGESTION_NOPARAM,
                )
            )
            continue
        if not _body_reads_name(func_node, source_param):
            findings.append(
                _make_finding(
                    rule_id=RULE_DEAD,
                    file_path=file_path,
                    call=call,
                    message=(
                        f"@trust_boundary(source_param={source_param!r}) names a parameter of "
                        f"{func_node.name!r} but the function body never reads it; "
                        "the decorator is structurally inert."
                    ),
                    suggestion=SUGGESTION_DEAD,
                )
            )
    return findings


def scan_root(root: Path) -> list[Finding]:
    """Walk every Python file under ``root`` and aggregate findings."""
    findings: list[Finding] = []
    for item in walk_python_files(root):
        if isinstance(item, PythonSyntaxError):
            continue
        findings.extend(analyze_tree(item.tree, display_path(item.path, root)))
    return findings


def _parameter_names(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Collect every parameter name in declaration order.

    Includes positional-only, regular, ``*args``, keyword-only, and ``**kwargs``
    parameters. ``self`` and ``cls`` are NOT filtered out — the runtime
    decorator validates against :func:`inspect.signature`, which sees them too,
    so the static rule must match that surface to avoid divergence.
    """
    args = func_node.args
    names: list[str] = []
    names.extend(arg.arg for arg in args.posonlyargs)
    names.extend(arg.arg for arg in args.args)
    if args.vararg is not None:
        names.append(args.vararg.arg)
    names.extend(arg.arg for arg in args.kwonlyargs)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return names


def _body_reads_name(func_node: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    """Return True if the function body reads ``name`` at least once.

    A "read" is any :class:`ast.Name` with :class:`ast.Load` context whose
    ``id`` equals ``name``. Walks every statement in the body's own lexical
    scope, **excluding** the bodies of any nested ``FunctionDef``,
    ``AsyncFunctionDef``, or ``Lambda``. Annotations on the function itself,
    decorators, and default-value expressions are also out of scope (they
    belong to the enclosing definition, not to the body's flow).

    Excluding nested-scope bodies is load-bearing for the DEAD-parameter
    check: an outer function that takes ``data`` as a parameter but never
    reads it, while containing an inner function that reads ``data``
    lexically inherited from the outer scope, must still be flagged as
    DEAD. The outer parameter would still be inert from the outer
    function's body's perspective — the inner read happens inside a
    different lexical scope that the @trust_boundary decorator does not
    apply to. The previous implementation used ``ast.walk(statement)``,
    which descended into the inner function's body and silently treated
    the inner read as if it satisfied the outer's parameter contract,
    masking the bug.
    """
    # Walk only the body statements (not the function's own decorators,
    # parameter annotations, or return annotation — those belong to the
    # enclosing definition, not to the body's flow). For each statement,
    # use scope-respecting iteration so nested function bodies don't leak
    # their reads of the outer parameter into the outer body's "did we
    # read it?" decision: a nested function reading ``data`` is NOT the
    # outer function's body reading ``data``, and treating it as such
    # silently masks DEAD-parameter bugs.
    for statement in func_node.body:
        for child in _iter_own_scope(statement):
            if (
                isinstance(child, ast.Name)
                and child.id == name
                and isinstance(child.ctx, ast.Load)
            ):
                return True
    return False


def _iter_own_scope(root: ast.AST) -> list[ast.AST]:
    """Return every node in ``root``'s own lexical scope, DFS pre-order.

    Mirrors the semantics of :func:`walk_function_own_scope` (which is
    typed for whole-function roots) but operates on an arbitrary
    statement / expression node. Used per-statement here so the body
    iteration can preserve the existing statement-by-statement structure.
    Nested ``FunctionDef`` / ``AsyncFunctionDef`` / ``Lambda`` nodes are
    yielded themselves, but their children (body / args / decorators /
    annotations) are excluded — those belong to the inner scope.
    """
    result: list[ast.AST] = [root]
    if isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return result
    for child in ast.iter_child_nodes(root):
        result.extend(_iter_own_scope(child))
    return result


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


RULE = TrustBoundaryScopeRule()
