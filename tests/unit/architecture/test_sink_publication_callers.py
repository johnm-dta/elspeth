"""Closed inventory of production sink publication calls during migration."""

from __future__ import annotations

import ast
import subprocess
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_ROOT = _ROOT / "src" / "elspeth"

# Only the explicit executor compatibility lane remains. Audit export has no
# direct publication path; effect-capable execution owns it exclusively.
_EXPECTED_CALLS = Counter(
    {
        ("engine/executors/sink.py", "_write_primary", "legacy_sink", "write", "Task 15"): 1,
        ("engine/executors/sink.py", "_write_primary", "legacy_sink", "flush", "Task 15"): 1,
        ("engine/executors/sink.py", "_handle_failsink_diversions", "legacy_failsink", "write", "Task 15"): 1,
        ("engine/executors/sink.py", "_handle_failsink_diversions", "legacy_failsink", "flush", "Task 15"): 1,
    }
)


def _terminal_name(expression: ast.expr) -> str | None:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        return expression.attr
    return None


def _is_sink_receiver(expression: ast.expr) -> bool:
    terminal = _terminal_name(expression)
    return terminal is not None and terminal.endswith("sink")


class _SinkCallVisitor(ast.NodeVisitor):
    def __init__(self, relative: str, calls: Counter[tuple[str, str, str, str, str]]) -> None:
        self._relative = relative
        self._calls = calls
        self._functions: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._functions.append(node.name)
        self.generic_visit(node)
        self._functions.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._functions.append(node.name)
        self.generic_visit(node)
        self._functions.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"write", "flush"} and _is_sink_receiver(node.func.value):
            receiver = ast.unparse(node.func.value)
            owner = "Task 15"
            function = self._functions[-1] if self._functions else "<module>"
            self._calls[(self._relative, function, receiver, node.func.attr, owner)] += 1
        self.generic_visit(node)


def _production_sink_calls() -> Counter[tuple[str, str, str, str, str]]:
    calls: Counter[tuple[str, str, str, str, str]] = Counter()
    for path in _SOURCE_ROOT.rglob("*.py"):
        relative = path.relative_to(_SOURCE_ROOT).as_posix()
        _SinkCallVisitor(relative, calls).visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    return calls


def test_production_sink_publication_calls_match_temporary_owned_inventory() -> None:
    assert _production_sink_calls() == _EXPECTED_CALLS


def test_no_indirect_sink_write_or_flush_aliases_escape_the_ast_inventory() -> None:
    diagnostic = subprocess.run(
        [
            "rg",
            "-n",
            "--pcre2",
            "--glob",
            "*.py",
            r"=\s*(?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*sink\.(?:write|flush)\b(?!\s*\()|getattr\([^\n]*(?:sink)[^\n]*['\"](?:write|flush)['\"]",
            str(_SOURCE_ROOT),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert diagnostic.returncode == 1, diagnostic.stdout
    assert diagnostic.stdout == ""
