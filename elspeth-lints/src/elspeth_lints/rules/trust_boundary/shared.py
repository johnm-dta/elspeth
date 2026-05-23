"""Shared AST helpers for ``@trust_boundary`` honesty-gate rules.

These helpers duplicate the decorator-recognition shape already used by
``trust_tier.tier_model.trust_boundary_suppress`` deliberately: the honesty
gates must remain independent of the suppression walk so a refactor in either
direction cannot accidentally couple the two. The recognition shape itself is
small and stable (matches the runtime decorator's import surface in
``src/elspeth/contracts/trust_boundary.py``).

Recognised decorator spellings (matching the runtime import surface):

* ``@trust_boundary(...)`` after ``from elspeth.contracts.trust_boundary import trust_boundary``;
* ``@elspeth.contracts.trust_boundary(...)`` (fully qualified attribute chain);
* ``@contracts.trust_boundary(...)`` (shortened attribute chain).

Bare-decorator usage (``@trust_boundary`` without a call) is not recognised:
the runtime decorator is keyword-only and never appears without arguments.

Async functions decorated with ``@trust_boundary`` are recognised
identically — the decorator composes with any callable. Both
:class:`ast.FunctionDef` and :class:`ast.AsyncFunctionDef` are yielded by
:func:`iter_trust_boundary_decorators`.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

_TRUST_BOUNDARY_NAME = "trust_boundary"


def _is_trust_boundary_decorator(decorator: ast.expr) -> ast.Call | None:
    """Return the ``ast.Call`` if ``decorator`` references ``trust_boundary``.

    See module docstring for the recognised spellings. Returns ``None`` for any
    other decorator shape; callers iterate the full decorator list.
    """
    if not isinstance(decorator, ast.Call):
        return None
    func = decorator.func
    if isinstance(func, ast.Name) and func.id == _TRUST_BOUNDARY_NAME:
        return decorator
    if isinstance(func, ast.Attribute) and func.attr == _TRUST_BOUNDARY_NAME:
        return decorator
    return None


def iter_trust_boundary_decorators(
    tree: ast.AST,
) -> Iterator[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.Call]]:
    """Yield ``(function_node, decorator_call)`` for every ``@trust_boundary`` in ``tree``.

    Walks the whole tree (module, classes, nested functions). The decorator
    call yielded is the ``ast.Call`` node — callers use
    :func:`extract_keywords` to read its kwargs.

    Ordering is deterministic (``ast.walk`` is depth-first, and Python's AST
    walk order is stable across runs).
    """
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            call = _is_trust_boundary_decorator(decorator)
            if call is not None:
                yield node, call
                # A function should have at most one @trust_boundary; if a
                # malformed decorator stack accidentally repeats it, we still
                # yield the first only — duplicate findings would be noise.
                break


def _literal_value(node: ast.expr) -> tuple[bool, object]:
    """Try to extract a static literal from an AST expression.

    Returns ``(ok, value)``. Allowed shapes:

    * :class:`ast.Constant` scalars (``str``, ``int``, ``bool``, ``None``,
      ``float``, ``bytes``);
    * :class:`ast.Tuple`, :class:`ast.List`, :class:`ast.Set`,
      :class:`ast.Dict` composed recursively of allowed shapes.

    Anything referencing a name, a call, an attribute, an f-string, or a
    comprehension is treated as non-literal. The honesty-gate rules need
    static metadata; an unverifiable value is a rule violation in itself, but
    these rules degrade by treating the kwarg as absent so they don't
    double-report the same defect (``tier_model.trust_boundary_suppress``
    already surfaces ``R_TB_NONLITERAL`` when this happens).
    """
    if isinstance(node, ast.Constant):
        return True, node.value
    if isinstance(node, ast.Tuple):
        items: list[object] = []
        for elt in node.elts:
            ok, value = _literal_value(elt)
            if not ok:
                return False, None
            items.append(value)
        return True, tuple(items)
    if isinstance(node, ast.List):
        list_items: list[object] = []
        for elt in node.elts:
            ok, value = _literal_value(elt)
            if not ok:
                return False, None
            list_items.append(value)
        return True, list_items
    if isinstance(node, ast.Set):
        set_items: list[object] = []
        for elt in node.elts:
            ok, value = _literal_value(elt)
            if not ok:
                return False, None
            set_items.append(value)
        return True, set(set_items)
    if isinstance(node, ast.Dict):
        result: dict[object, object] = {}
        for key_node, value_node in zip(node.keys, node.values, strict=False):
            if key_node is None:
                return False, None
            ok_key, key_val = _literal_value(key_node)
            if not ok_key:
                return False, None
            ok_val, val_val = _literal_value(value_node)
            if not ok_val:
                return False, None
            result[key_val] = val_val
        return True, result
    return False, None


def extract_keywords(call: ast.Call) -> dict[str, object] | None:
    """Return the decorator's kwargs as a literal-value dict, or ``None`` if any kwarg is non-literal.

    Positional arguments are rejected (the runtime decorator is keyword-only;
    a positional-arg call is malformed). ``**kwargs``-style unpacking is
    rejected (``keyword.arg is None``). Any kwarg whose value cannot be
    literal-evaluated also yields ``None`` — the honesty gates treat such a
    call as inert so the canonical malformed-decorator diagnostics
    (``R_TB_NONLITERAL`` / ``R_TB_MALFORMED`` in the tier_model rule) own
    that reporting surface.

    Returns:
        A ``dict[str, object]`` of kwarg → literal value when every kwarg is
        a static literal; ``None`` if any kwarg is non-literal, the call has
        positional args, or ``**kwargs`` unpacking is used.
    """
    if call.args:
        return None
    parsed: dict[str, object] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            return None
        ok, value = _literal_value(keyword.value)
        if not ok:
            return None
        parsed[keyword.arg] = value
    return parsed


def display_path(file_path: Path, root: Path) -> str:
    """Return the path format used by elspeth-lints rules for this scan root.

    Matches the convention in
    :func:`elspeth_lints.rules.immutability.shared.display_path`: a
    forward-slash POSIX path relative to ``root``, or the absolute path if the
    file is outside the scan root.
    """
    try:
        return file_path.relative_to(root).as_posix()
    except ValueError:
        return file_path.as_posix()


def repository_root(root: Path) -> Path:
    """Return the repository root from a scan root.

    When the scan root is ``<repo>/src/elspeth`` (the canonical invocation),
    the repository root is ``root.parent.parent``. Otherwise the scan root
    itself is the repository root. Same heuristic as
    :func:`elspeth_lints.rules.audit_evidence.shared.repo_relative_display_path`.
    The ``trust_boundary.tests`` rule needs this to resolve ``test_ref``
    pytest nodeids that point to ``tests/...`` paths even though the scan
    root is ``src/elspeth/``.
    """
    if root.name == "elspeth" and root.parent.name == "src":
        return root.parent.parent
    return root
