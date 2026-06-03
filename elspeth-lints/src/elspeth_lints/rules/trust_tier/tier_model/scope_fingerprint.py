# scope_fingerprint.py
"""Enclosing-scope AST fingerprint for judge-gated tier-model suppressions.

This is the v2 binding primitive (replaces the whole-file ``file_fingerprint``).
It binds a judge-gated allowlist entry to the AST content of the *innermost
enclosing scope* of the suppressed node, so editing an unrelated scope in the
same file no longer invalidates the entry's HMAC signature.

The hash MUST be byte-reproducible at justify-write time and match-verify time.
Both call sites import :func:`compute_scope_fingerprint` from here — there is
deliberately only one definition. See the design doc
``docs/superpowers/specs/2026-05-31-judge-scope-fingerprint-design.md`` §3.1 for
the normative determinism rules.
"""

from __future__ import annotations

import ast
import copy
import hashlib

_ScopeNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def enclosing_scope_node(ancestors: list[ast.AST]) -> _ScopeNode | None:
    """Return the innermost enclosing def/class from an innermost-first ancestor list.

    ``ancestors`` is the suppressed node followed by each parent up to the
    module, innermost first (the shape produced by the visitor's
    ``node_stack`` reversed). Returns ``None`` when the node is at module
    level (no enclosing def/class), which the caller maps to the
    whole-module fallback.
    """
    for node in ancestors:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            return node
    return None


def _strip_leading_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    """Return ``body`` without a leading docstring statement, if present.

    A docstring is an ``ast.Expr`` whose value is an ``ast.Constant`` holding
    a ``str``. Editing or adding a docstring must not change the fingerprint
    (rule 4): documentation is not the code the judge reasoned about.
    """
    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return body[1:]
    return body


def compute_scope_fingerprint(scope_node: _ScopeNode | None, *, module: ast.Module | None = None) -> str:
    """Return the 64-char hex scope fingerprint for ``scope_node``.

    When ``scope_node`` is ``None`` (module-level suppression), ``module``
    must be provided and the whole module is fingerprinted instead. The
    leading docstring of the scope (or module) is excluded.
    """
    if scope_node is None:
        if module is None:
            raise ValueError("compute_scope_fingerprint: module is required when scope_node is None (module-level fallback)")
        # type_ignores is deliberately emptied, not copied: a ``# type: ignore`` is
        # a comment, and rule 4 makes comments free. Copying module.type_ignores
        # would make the module-level hash sensitive to type-ignore comments, while
        # the scope path (FunctionDef/ClassDef carry no type_ignores field) cannot
        # be — an asymmetry that silently breaks rule 4 for module-level
        # suppressions if a caller ever parses with type_comments=True.
        target: ast.AST = ast.Module(body=_strip_leading_docstring(list(module.body)), type_ignores=[])
    else:
        # Shallow-copy the scope node and replace only its body, so the original
        # AST (still being walked by the visitor) is never mutated.
        target = copy.copy(scope_node)
        target.body = _strip_leading_docstring(list(scope_node.body))
    dump = ast.dump(target, include_attributes=False, annotate_fields=True)
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()
