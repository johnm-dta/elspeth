"""Shared Python AST file walking utilities."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path


def walk_function_own_scope(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterator[ast.AST]:
    """Yield every AST node belonging to ``func_node``'s own lexical scope.

    Like :func:`ast.walk`, but does **not** descend into the body of any
    nested ``FunctionDef``, ``AsyncFunctionDef``, or ``Lambda`` encountered
    along the way. Nested-function bodies introduce a new lexical scope —
    names bound there do not belong to ``func_node``'s scope and an
    analyzer that treats them as such over-approximates the parent scope
    (taint leaks out of the inner function; "is parameter X read?" returns
    True when only the inner function reads it).

    Yields ``func_node`` itself first, then every descendant statement /
    expression in DFS pre-order, stopping at — but **not including** —
    the inner bodies of nested function-like constructs. The nested
    function-defining AST nodes themselves (the ``FunctionDef`` /
    ``AsyncFunctionDef`` / ``Lambda`` AST objects) are yielded, but their
    children are not.

    Why not "yield the inner def but not descend"? Because the inner def
    node is genuinely a child of the outer scope (the ``def`` statement
    *binds* a name in the outer scope), and a caller that wants to detect
    "this function defines a nested helper" needs to see the
    ``FunctionDef`` node. The children of that node (its body, its args)
    belong to the inner scope and are excluded.

    Decorators, annotations, default values, and return annotations on
    the nested ``def`` itself are part of the *outer* scope expression
    grammar (they execute when the outer function runs, not the inner),
    so a fully principled scope walker would yield them. For our purposes
    (detecting reads of an outer parameter, propagating taint from an
    outer parameter), short-circuiting at the ``FunctionDef`` node is
    conservative — annotations and defaults are rarely where boundary
    parameters get referenced, and erring on the side of fewer-yielded-
    nodes means we under-suppress (visible findings) rather than
    over-suppress (silent suppressions). For the symmetric ``Lambda``
    case the body is an expression, not a statement list, so the same
    short-circuit applies cleanly.
    """
    yield func_node
    yield from _walk_children_excluding_nested_scopes(func_node)


def _walk_children_excluding_nested_scopes(node: ast.AST) -> Iterator[ast.AST]:
    """Recursive helper for :func:`walk_function_own_scope`.

    Descends through ``node``'s direct children, yielding each one and
    recursing — but does NOT recurse into the children of any nested
    ``FunctionDef`` / ``AsyncFunctionDef`` / ``Lambda``. The nested-scope
    AST node itself IS yielded (so a caller can observe that the outer
    function defined a nested helper), but its body / args / decorators
    are not.
    """
    for child in ast.iter_child_nodes(node):
        yield child
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            # Short-circuit: do not descend into the nested scope's children.
            continue
        yield from _walk_children_excluding_nested_scopes(child)


@dataclass(frozen=True, slots=True)
class ParsedPythonFile:
    """A successfully parsed Python file."""

    path: Path
    source: str
    tree: ast.Module


@dataclass(frozen=True, slots=True)
class PythonSyntaxError:
    """A syntax error captured while walking a tree of Python files."""

    path: Path
    line: int
    column: int
    message: str
    text: str | None


def iter_python_files(root: Path, files: Iterable[Path] | None = None) -> Iterator[Path]:
    """Yield Python files either from an explicit list or by walking root."""
    if files is not None:
        for file_path in files:
            if file_path.is_absolute() or file_path.exists():
                candidate = file_path
            else:
                candidate = root / file_path
            if candidate.suffix == ".py":
                yield candidate
        return

    for file_path in sorted(root.rglob("*.py")):
        if "__pycache__" not in file_path.parts:
            yield file_path


def walk_python_files(root: Path, files: Iterable[Path] | None = None) -> Iterator[ParsedPythonFile | PythonSyntaxError]:
    """Parse Python files without aborting the walk on syntax errors."""
    for file_path in iter_python_files(root, files):
        yield parse_python_file(file_path)


def parse_python_file(path: Path) -> ParsedPythonFile | PythonSyntaxError:
    """Parse one Python file and return either an AST or a syntax-error record."""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return PythonSyntaxError(
            path=path,
            line=exc.lineno or 0,
            column=exc.offset or 0,
            message=exc.msg,
            text=exc.text,
        )
    return ParsedPythonFile(path=path, source=source, tree=tree)
