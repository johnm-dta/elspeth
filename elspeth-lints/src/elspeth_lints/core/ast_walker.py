"""Shared Python AST file walking utilities."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

_EXCLUDED_WALK_DIRS = frozenset(
    {
        "__pycache__",
        ".git",
        ".hg",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".uv-cache",
        ".venv",
        ".worktrees",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)

# Nested-scope AST node types that a "lexical scope of this function" walker
# must short-circuit at: descending into their children would conflate names
# bound in the inner scope with names bound in the outer scope.
#
# * ``FunctionDef`` / ``AsyncFunctionDef`` — nested defs have their own
#   ``locals()``; a name assigned inside does not bind in the enclosing
#   function's scope.
# * ``Lambda`` — same story, just an expression form.
# * ``ClassDef`` — class bodies execute in a fresh namespace; assignments
#   become *class attributes*, not bindings in the enclosing function. A
#   walker that descends into the class body and sees
#   ``raw = arguments["x"]`` would incorrectly conclude that the outer
#   function bound ``raw``. (Note: comprehensions in Python 3 also have
#   their own scope, but they're handled at expression level by their own
#   target-binding rules and the analyzers that care explicitly track
#   them; the leak that motivated the ClassDef addition was a class
#   body inside a decorated function tainting the outer scope.)
_NESTED_SCOPE_TYPES: tuple[type[ast.AST], ...] = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
)


def walk_function_own_scope(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterator[ast.AST]:
    """Yield every AST node belonging to ``func_node``'s own lexical scope.

    Like :func:`ast.walk`, but does **not** descend into the body of any
    nested ``FunctionDef``, ``AsyncFunctionDef``, ``Lambda``, or
    ``ClassDef`` encountered along the way. Nested-scope bodies introduce
    a new namespace — names bound there do not belong to ``func_node``'s
    scope and an analyzer that treats them as such over-approximates the
    parent scope (taint leaks out of the inner scope; "is parameter X
    read?" returns True when only the inner scope reads it; a class
    attribute assignment falsely registers as an outer-function binding).

    Yields ``func_node`` itself first, then every descendant statement /
    expression in DFS pre-order, stopping at — but **not including** —
    the inner bodies of nested scope-defining constructs. The nested
    scope-defining AST nodes themselves (the ``FunctionDef`` /
    ``AsyncFunctionDef`` / ``Lambda`` / ``ClassDef`` AST objects) are
    yielded, but their children are not.

    Why not "yield the inner def but not descend"? Because the inner def
    node is genuinely a child of the outer scope (the ``def`` / ``class``
    statement *binds* a name in the outer scope), and a caller that
    wants to detect "this function defines a nested helper or class"
    needs to see the ``FunctionDef`` / ``ClassDef`` node. The children
    of that node (its body, its args) belong to the inner scope and are
    excluded.

    Decorators, annotations, default values, return annotations, and
    class base-class expressions on the nested ``def`` / ``class``
    itself are part of the *outer* scope expression grammar (they
    execute when the outer function runs, not the inner), so a fully
    principled scope walker would yield them. For our purposes
    (detecting reads of an outer parameter, propagating taint from an
    outer parameter), short-circuiting at the ``FunctionDef`` /
    ``ClassDef`` node is conservative — annotations, defaults, and base
    classes are rarely where boundary parameters get referenced, and
    erring on the side of fewer-yielded-nodes means we under-suppress
    (visible findings) rather than over-suppress (silent suppressions).
    For the symmetric ``Lambda`` case the body is an expression, not a
    statement list, so the same short-circuit applies cleanly.
    """
    yield func_node
    yield from _walk_children_excluding_nested_scopes(func_node)


def iter_own_scope(root: ast.AST) -> Iterator[ast.AST]:
    """Yield ``root`` and every descendant in its own lexical scope.

    Like :func:`walk_function_own_scope` but accepts an arbitrary AST
    node (typically a single statement or expression), not a function
    definition. Short-circuits at nested ``FunctionDef`` /
    ``AsyncFunctionDef`` / ``Lambda`` / ``ClassDef`` boundaries with the
    same semantics: the scope-defining node itself is yielded, its
    children are not.

    Used by analyzers that walk a function body statement-by-statement
    and need scope-respecting traversal of each statement (rather than
    of the whole function tree at once).
    """
    yield root
    if isinstance(root, _NESTED_SCOPE_TYPES):
        # Short-circuit: the root IS a nested scope. Yield it (the caller
        # observed it lives in the parent's scope) and stop — its body
        # belongs to a different scope.
        return
    yield from _walk_children_excluding_nested_scopes(root)


def _walk_children_excluding_nested_scopes(node: ast.AST) -> Iterator[ast.AST]:
    """Recursive helper for :func:`walk_function_own_scope` and :func:`iter_own_scope`.

    Descends through ``node``'s direct children, yielding each one and
    recursing — but does NOT recurse into the children of any nested
    ``FunctionDef`` / ``AsyncFunctionDef`` / ``Lambda`` / ``ClassDef``.
    The nested-scope AST node itself IS yielded (so a caller can observe
    that the outer function defined a nested helper or class), but its
    body / args / decorators / class-attributes are not.
    """
    for child in ast.iter_child_nodes(node):
        yield child
        if isinstance(child, _NESTED_SCOPE_TYPES):
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


@dataclass(frozen=True, slots=True)
class PythonFileReadError:
    """A non-parse failure (I/O or decoding) captured while walking files.

    The walker treats per-file I/O failures the same way it treats per-file
    syntax errors: they become structured per-file results so the scan can
    continue on the other files. Aborting the whole scan because one file
    is unreadable (permission denied on a single source file, an invalid
    UTF-8 sequence in a generated file, a vanished symlink target) is the
    wrong failure mode for a static analyzer that's deliberately run over
    a wide tree.

    Callers MUST handle this variant in the same dispatch as
    :class:`PythonSyntaxError` — either surface it as a per-file diagnostic
    (the CLI emits a ``read-error`` Finding), skip it (most rule scan
    loops), or raise loudly (the fixture harness, where any read failure
    in a curated test fixture is a test-authoring bug). Silent ``continue``
    in a non-CLI loop is acceptable; what is NOT acceptable is letting the
    exception propagate up and abort the scan.

    Fields:
        path: The file whose read failed.
        message: Human-readable description, typically the exception ``str``.
        error_type: The exception class name (``UnicodeDecodeError``,
            ``PermissionError``, ``OSError``, or a more specific
            ``OSError`` subclass like ``FileNotFoundError``). Used by
            callers that want to format diagnostics differently per
            failure mode, and recorded in finding fingerprints so a
            stable diagnostic id survives across scans.

    Line / column are intentionally absent: an I/O failure has no
    position inside the (unreadable) file. Callers needing a positional
    surface for a uniform ``Finding`` shape use line 0 / column 0 by
    convention at the *emission* site (see ``cli.py``), not on this
    dataclass.
    """

    path: Path
    message: str
    error_type: str


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
        if not _EXCLUDED_WALK_DIRS.intersection(file_path.parts):
            yield file_path


def walk_python_files(
    root: Path, files: Iterable[Path] | None = None
) -> Iterator[ParsedPythonFile | PythonSyntaxError | PythonFileReadError]:
    """Parse Python files without aborting the walk on syntax or I/O errors."""
    for file_path in iter_python_files(root, files):
        yield parse_python_file(file_path)


def parse_python_file(path: Path) -> ParsedPythonFile | PythonSyntaxError | PythonFileReadError:
    """Parse one Python file.

    Returns one of three variants:

    * :class:`ParsedPythonFile` — the file was read and parsed cleanly.
    * :class:`PythonSyntaxError` — the file was readable but ``ast.parse``
      rejected it. The caller decides whether to report or skip.
    * :class:`PythonFileReadError` — the file could not be read at all
      (``UnicodeDecodeError`` on the source bytes, ``PermissionError``
      on the inode, or any other ``OSError`` — ``FileNotFoundError`` and
      ``IsADirectoryError`` are ``OSError`` subclasses and ride this
      branch). The caller decides whether to surface it (CLI emits a
      ``read-error`` finding so the operator knows the file was not
      analysed) or quietly skip it (per-rule scan loops do this — the
      next scan over the same tree will produce the same result anyway).

    We catch the I/O / decoding failures specifically — NOT a blanket
    ``except Exception`` — because the static-analyzer doctrine here is:
    swallow only the failure modes whose root cause is the file itself
    (its bytes, its inode permissions), let every other unexpected
    exception propagate so it gets investigated. A ``MemoryError`` on
    a hundred-megabyte source file should still abort the scan; an
    invalid UTF-8 sequence in one file should not.

    ``UnicodeDecodeError`` is listed explicitly even though it is an
    ``OSError`` subclass on no Python version (it derives from
    ``ValueError``); the explicit listing makes the rationale visible
    and survives any future refactor of the read path that swaps
    ``read_text`` for a bytes-then-decode shape.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return PythonFileReadError(
            path=path,
            message=f"could not decode as UTF-8 ({exc.reason} at byte {exc.start})",
            error_type=type(exc).__name__,
        )
    except PermissionError as exc:
        return PythonFileReadError(
            path=path,
            message=f"permission denied: {exc}",
            error_type=type(exc).__name__,
        )
    except OSError as exc:
        # FileNotFoundError, IsADirectoryError, broken-symlink ELOOP,
        # generic IOError, etc. all land here. Record the specific
        # subclass name so the diagnostic stays informative.
        return PythonFileReadError(
            path=path,
            message=f"{type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
        )
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
