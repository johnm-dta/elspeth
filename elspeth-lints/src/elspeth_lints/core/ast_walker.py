"""Shared Python AST file walking utilities."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path


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
