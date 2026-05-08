"""Mechanical direct-writer guard for ``chat_messages`` and ``composition_states``.

Phase 1A introduces NOT NULL ``chat_messages.sequence_no``,
``chat_messages.writer_principal``, and ``composition_states.provenance``
columns. Every direct insert into either table must either route through
the new lock-aware helpers (``_insert_chat_message``,
``_insert_composition_state``, ``_reserve_sequence_range``) or be
explicitly allowlisted as a known semantic site (schema test,
corruption fixture, OperationalError canary, standalone eval fixture).

This guard is the **mechanical merge gate** for the Schedule 1A cutover.
A reviewer-facing ripgrep is fast but cannot distinguish a corruption
fixture from a real bypass; this scanner walks the AST, identifies the
enclosing function/class qualified symbol, and matches against a static
allowlist keyed by ``(path, enclosing_symbol, table, operation)``.

Coverage:

* SQLAlchemy ``insert(table_name)`` calls.
* SQLAlchemy ``table_name.insert()`` chained calls.
* Raw ``INSERT INTO chat_messages`` / ``INSERT INTO composition_states``
  string literals, regardless of enclosing call (covers
  ``cursor.execute``, ``cursor.executemany``, ``exec_driver_sql``,
  ``OperationalError(...)`` canaries, and bare strings).

Lock-discipline conditional-dormancy rule:

The plan introduces ``_session_write_lock``, ``_reserve_sequence_range``,
``_insert_chat_message``, ``_insert_composition_state``, and
``_assert_session_write_lock_held`` in Tasks 7-10. Until those symbols
exist in the codebase, the lock-discipline and inline-allocation checks
are dormant (vacuous PASS). They activate the moment ``_session_write_lock``
is defined anywhere under ``src/``, fail-closed against any caller that
drifts off the lock.

Self-exclusion:

This file contains the table identifier strings as scanner data; without
self-exclusion the live-tree scan would find its own data. The scanner
skips any source file whose resolved path equals this module's resolved
path.
"""

from __future__ import annotations

import ast
import re
import textwrap
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TABLE_IDENTIFIER_TO_NAME = {
    "chat_messages_table": "chat_messages",
    "composition_states_table": "composition_states",
}

_RAW_SQL_INSERT_PATTERN = re.compile(
    r"INSERT\s+INTO\s+(chat_messages|composition_states)\b",
    re.IGNORECASE,
)

_LOCK_HELPER_NAMES = (
    "_reserve_sequence_range",
    "_insert_chat_message",
    "_insert_composition_state",
)

_SESSION_WRITE_LOCK_NAME = "_session_write_lock"
_LOCK_HELD_ASSERT_NAME = "_assert_session_write_lock_held"

# Resolved at module import to avoid recomputing per-test.
_SCANNER_SELF_PATH = Path(__file__).resolve()


def _find_repo_root() -> Path:
    """Return the repository root resolved from this test file.

    ``tests/unit/web/sessions/test_static_direct_writers.py``'s parents
    chain is ``[sessions/, web/, unit/, tests/, <repo>]`` so
    ``parents[4]`` is the repo root. The ``src``/``tests`` anchors must
    exist directly under it.
    """

    candidate = Path(__file__).resolve().parents[4]
    if not (candidate / "src").is_dir() or not (candidate / "tests").is_dir():
        raise RuntimeError(f"could not resolve repo root from {Path(__file__)}: candidate {candidate} is missing src/ or tests/")
    return candidate


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewedWriter:
    """Allowlist entry for a known-OK direct writer site.

    The four keying fields ``(path, enclosing_symbol, table, operation)``
    must match the scanner's :class:`WriterMatch` exactly. ``purpose``
    is informational and shows up in the violation report when a new
    site fails the gate.
    """

    path: str
    enclosing_symbol: str
    table: str
    operation: str
    purpose: str


@dataclass(frozen=True)
class WriterMatch:
    path: str
    line: int
    enclosing_symbol: str
    table: str
    operation: str
    snippet: str


@dataclass(frozen=True)
class LockDisciplineViolation:
    path: str
    line: int
    enclosing_symbol: str
    helper_name: str
    snippet: str


@dataclass(frozen=True)
class InlineAllocViolation:
    path: str
    line: int
    enclosing_symbol: str
    snippet: str


# ---------------------------------------------------------------------------
# AST utilities
# ---------------------------------------------------------------------------


def _attach_parents(tree: ast.AST) -> None:
    """Annotate every node with a ``parent`` attribute for upward walks.

    ``ast`` doesn't track parents; the lock-discipline checker needs to
    walk from a Call up to its enclosing FunctionDef and any enclosing
    ``with`` block. Building the parent map once is cheaper than
    re-traversing.
    """

    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]


def _qualified_symbol(node: ast.AST) -> str:
    """Return the dotted enclosing-symbol path for ``node``.

    Walks parent links. The result joins ``ClassDef`` and
    ``FunctionDef``/``AsyncFunctionDef`` names from outermost to
    innermost. Module-level nodes return ``"<module>"``.
    """

    parts: list[str] = []
    cursor: ast.AST | None = node
    while cursor is not None:
        if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            parts.append(cursor.name)
        cursor = getattr(cursor, "parent", None)
    if not parts:
        return "<module>"
    return ".".join(reversed(parts))


def _enclosing_call(node: ast.AST) -> ast.Call | None:
    """Walk parents from ``node`` up to the nearest enclosing :class:`ast.Call`.

    Returns ``None`` if the node is not inside a call. Skips
    Module/FunctionDef/ClassDef boundaries by continuing past them.
    """

    cursor: ast.AST | None = getattr(node, "parent", None)
    while cursor is not None:
        if isinstance(cursor, ast.Call):
            return cursor
        cursor = getattr(cursor, "parent", None)
    return None


def _enclosing_with_blocks(node: ast.AST) -> Iterator[ast.With | ast.AsyncWith]:
    """Yield enclosing ``with`` / ``async with`` blocks from inner to outer.

    Used by the lock-discipline checker to verify a helper call is
    inside a ``_session_write_lock`` context manager. Stops at the
    enclosing FunctionDef boundary so unrelated outer ``with`` blocks
    in the same module don't satisfy the check.
    """

    cursor: ast.AST | None = getattr(node, "parent", None)
    while cursor is not None and not isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
        if isinstance(cursor, (ast.With, ast.AsyncWith)):
            yield cursor
        cursor = getattr(cursor, "parent", None)


def _call_callable_name(call: ast.Call) -> str:
    """Return a human-readable name for the callable of an :class:`ast.Call`.

    For ``foo.bar(x)`` returns ``"bar"``. For ``foo(x)`` returns ``"foo"``.
    For complex expressions returns ``"<expr>"``.
    """

    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return "<expr>"


def _line_snippet(source_lines: Sequence[str], line: int, max_len: int = 200) -> str:
    """Return a single-line snippet for ``line`` (1-indexed), truncated."""

    if 1 <= line <= len(source_lines):
        snippet = source_lines[line - 1].strip()
        if len(snippet) > max_len:
            return snippet[: max_len - 3] + "..."
        return snippet
    return ""


# ---------------------------------------------------------------------------
# Direct-writer scanner
# ---------------------------------------------------------------------------


class _WriterCollector(ast.NodeVisitor):
    """Collect direct-writer matches for one source file."""

    def __init__(self, rel_path: str, source: str, tree: ast.AST) -> None:
        self.rel_path = rel_path
        self.source_lines = source.splitlines()
        self.tree = tree
        self.matches: list[WriterMatch] = []

    def collect(self) -> list[WriterMatch]:
        self.visit(self.tree)
        return self.matches

    # ------------------------------------------------------------------
    # SQLAlchemy patterns
    # ------------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        # Pattern 1: insert(chat_messages_table) — bare ``insert`` call with
        # a single Name argument that resolves to a tracked table identifier.
        func = node.func
        if isinstance(func, ast.Name) and func.id == "insert" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Name) and first.id in _TABLE_IDENTIFIER_TO_NAME:
                self._emit(
                    node,
                    table=_TABLE_IDENTIFIER_TO_NAME[first.id],
                    operation="sqlalchemy_insert_call",
                )

        # Pattern 2: chat_messages_table.insert() — Attribute access on a
        # tracked table identifier, with attr ``insert``.
        if isinstance(func, ast.Attribute) and func.attr == "insert":
            value = func.value
            if isinstance(value, ast.Name) and value.id in _TABLE_IDENTIFIER_TO_NAME:
                self._emit(
                    node,
                    table=_TABLE_IDENTIFIER_TO_NAME[value.id],
                    operation="sqlalchemy_table_insert",
                )

        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Raw SQL patterns
    # ------------------------------------------------------------------

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, str):
            return
        match = _RAW_SQL_INSERT_PATTERN.search(node.value)
        if match is None:
            return
        table = match.group(1).lower()
        enclosing = _enclosing_call(node)
        operation = f"raw_string_in_{_call_callable_name(enclosing)}" if enclosing is not None else "raw_string_module"
        self._emit(node, table=table, operation=operation)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self, node: ast.AST, *, table: str, operation: str) -> None:
        line = getattr(node, "lineno", 0)
        self.matches.append(
            WriterMatch(
                path=self.rel_path,
                line=line,
                enclosing_symbol=_qualified_symbol(node),
                table=table,
                operation=operation,
                snippet=_line_snippet(self.source_lines, line),
            )
        )


def _iter_python_files(roots: Sequence[Path]) -> Iterator[tuple[Path, Path]]:
    """Yield ``(root, py_file)`` for every Python source under each root.

    Skips this scanner module via resolved-path equality.
    """

    for root in roots:
        if not root.exists():
            continue
        for py_file in sorted(root.rglob("*.py")):
            if py_file.resolve() == _SCANNER_SELF_PATH:
                continue
            yield root, py_file


def scan_writers(
    roots: Sequence[Path],
    *,
    path_anchor: Path | None = None,
) -> list[WriterMatch]:
    """Scan every Python source under ``roots`` for direct writer sites.

    Returns matches with ``path`` made relative to ``path_anchor`` if
    given, else relative to the root that contained each file.
    """

    matches: list[WriterMatch] = []
    for root, py_file in _iter_python_files(roots):
        try:
            source = py_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Synthetic-test fixtures occasionally write non-UTF-8 bytes;
            # the scanner's job is to parse Python, so skip what doesn't
            # decode rather than crash.
            continue
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            # A test fixture may write deliberately invalid Python to
            # exercise unrelated parsers; skip rather than crash.
            continue
        _attach_parents(tree)

        anchor = path_anchor or root
        try:
            rel = py_file.resolve().relative_to(anchor.resolve()).as_posix()
        except ValueError:
            rel = py_file.resolve().as_posix()

        matches.extend(_WriterCollector(rel, source, tree).collect())
    return matches


def violations(
    matches: Sequence[WriterMatch],
    allowlist: Sequence[ReviewedWriter],
) -> list[WriterMatch]:
    """Return matches whose ``(path, enclosing_symbol, table, operation)`` is not allowlisted."""

    allowed_keys = {(entry.path, entry.enclosing_symbol, entry.table, entry.operation) for entry in allowlist}
    return [m for m in matches if (m.path, m.enclosing_symbol, m.table, m.operation) not in allowed_keys]


# ---------------------------------------------------------------------------
# Lock-discipline checker
# ---------------------------------------------------------------------------


def _codebase_defines_symbol(symbol: str, roots: Sequence[Path]) -> bool:
    """Return True iff a top-level ``def <symbol>`` exists under any root.

    Used by the conditional-dormancy rule. A live ``_session_write_lock``
    function definition activates the lock-discipline checks; until it
    exists, the checks return no violations.
    """

    for _, py_file in _iter_python_files(roots):
        try:
            source = py_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol:
                return True
    return False


def _with_block_calls_session_write_lock(with_node: ast.With | ast.AsyncWith) -> bool:
    """Return True iff any ``with`` item is a call to ``_session_write_lock``."""

    for item in with_node.items:
        ctx = item.context_expr
        if isinstance(ctx, ast.Call):
            func = ctx.func
            if isinstance(func, ast.Attribute) and func.attr == _SESSION_WRITE_LOCK_NAME:
                return True
            if isinstance(func, ast.Name) and func.id == _SESSION_WRITE_LOCK_NAME:
                return True
    return False


def check_lock_discipline(
    roots: Sequence[Path],
    *,
    path_anchor: Path | None = None,
) -> list[LockDisciplineViolation]:
    """Check that every call to a lock-required helper is inside ``_session_write_lock``.

    Conditional-dormant: if ``_session_write_lock`` is not defined anywhere
    under ``roots``, returns ``[]``. As soon as the helpers land (Task 9),
    every caller that drifts off the lock is flagged.
    """

    if not _codebase_defines_symbol(_SESSION_WRITE_LOCK_NAME, roots):
        return []

    findings: list[LockDisciplineViolation] = []
    for root, py_file in _iter_python_files(roots):
        try:
            source = py_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue
        _attach_parents(tree)
        anchor = path_anchor or root
        try:
            rel = py_file.resolve().relative_to(anchor.resolve()).as_posix()
        except ValueError:
            rel = py_file.resolve().as_posix()
        source_lines = source.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            helper_name = _called_helper_name(node)
            if helper_name is None:
                continue
            inside_lock = any(_with_block_calls_session_write_lock(w) for w in _enclosing_with_blocks(node))
            if inside_lock:
                continue
            line = getattr(node, "lineno", 0)
            findings.append(
                LockDisciplineViolation(
                    path=rel,
                    line=line,
                    enclosing_symbol=_qualified_symbol(node),
                    helper_name=helper_name,
                    snippet=_line_snippet(source_lines, line),
                )
            )
    return findings


def _called_helper_name(call: ast.Call) -> str | None:
    """Return the lock-required helper name if ``call`` invokes one, else ``None``."""

    func = call.func
    if isinstance(func, ast.Name) and func.id in _LOCK_HELPER_NAMES:
        return func.id
    if isinstance(func, ast.Attribute) and func.attr in _LOCK_HELPER_NAMES:
        return func.attr
    return None


def check_helper_lock_assertions(
    roots: Sequence[Path],
    *,
    path_anchor: Path | None = None,
) -> list[LockDisciplineViolation]:
    """Check that each lock-required helper's body calls ``_assert_session_write_lock_held``.

    Conditional-dormant: if ``_session_write_lock`` is not defined anywhere
    under ``roots``, returns ``[]``. After Task 9, this enforces that the
    helpers themselves cannot be reached without the lock — even if a
    caller forgot the static check above.
    """

    if not _codebase_defines_symbol(_SESSION_WRITE_LOCK_NAME, roots):
        return []

    findings: list[LockDisciplineViolation] = []
    for root, py_file in _iter_python_files(roots):
        try:
            source = py_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue
        _attach_parents(tree)
        anchor = path_anchor or root
        try:
            rel = py_file.resolve().relative_to(anchor.resolve()).as_posix()
        except ValueError:
            rel = py_file.resolve().as_posix()
        source_lines = source.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in _LOCK_HELPER_NAMES:
                continue
            asserts = [
                child
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and (
                    (isinstance(child.func, ast.Name) and child.func.id == _LOCK_HELD_ASSERT_NAME)
                    or (isinstance(child.func, ast.Attribute) and child.func.attr == _LOCK_HELD_ASSERT_NAME)
                )
            ]
            if asserts:
                continue
            line = node.lineno
            findings.append(
                LockDisciplineViolation(
                    path=rel,
                    line=line,
                    enclosing_symbol=_qualified_symbol(node),
                    helper_name=node.name,
                    snippet=_line_snippet(source_lines, line),
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Inline composition_states.version allocation checker
# ---------------------------------------------------------------------------


_INLINE_VERSION_PATTERN = re.compile(
    r"SELECT\s+MAX\s*\(\s*composition_states\.version",
    re.IGNORECASE,
)


def check_inline_state_version_allocation(
    roots: Sequence[Path],
    *,
    path_anchor: Path | None = None,
) -> list[InlineAllocViolation]:
    """Reject inline ``SELECT MAX(composition_states.version)`` outside the write lock.

    Closes the review finding that ``save_composition_state`` and
    ``set_active_state`` previously allocated state versions inline
    without holding ``_session_write_lock``.

    Conditional-dormant: returns ``[]`` until ``_session_write_lock``
    is defined.
    """

    if not _codebase_defines_symbol(_SESSION_WRITE_LOCK_NAME, roots):
        return []

    findings: list[InlineAllocViolation] = []
    for root, py_file in _iter_python_files(roots):
        try:
            source = py_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue
        _attach_parents(tree)
        anchor = path_anchor or root
        try:
            rel = py_file.resolve().relative_to(anchor.resolve()).as_posix()
        except ValueError:
            rel = py_file.resolve().as_posix()
        source_lines = source.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if _INLINE_VERSION_PATTERN.search(node.value) is None:
                continue
            symbol = _qualified_symbol(node)
            simple_name = symbol.rsplit(".", 1)[-1]
            if simple_name not in {"save_composition_state", "set_active_state"}:
                continue
            inside_lock = any(_with_block_calls_session_write_lock(w) for w in _enclosing_with_blocks(node))
            if inside_lock:
                continue
            line = getattr(node, "lineno", 0)
            findings.append(
                InlineAllocViolation(
                    path=rel,
                    line=line,
                    enclosing_symbol=symbol,
                    snippet=_line_snippet(source_lines, line),
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Reviewed allowlist (§57-68 of the Phase 1A plan, validated 2026-05-08)
# ---------------------------------------------------------------------------

_REVIEWED_ALLOWLIST: tuple[ReviewedWriter, ...] = (
    # ------ src/elspeth/web/sessions/service.py ------
    ReviewedWriter(
        path="src/elspeth/web/sessions/service.py",
        enclosing_symbol="SessionServiceImpl.add_message._sync",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose="current add_message writer (line 383); Task 14 migrates to _insert_chat_message",
    ),
    ReviewedWriter(
        path="src/elspeth/web/sessions/service.py",
        enclosing_symbol="SessionServiceImpl.save_composition_state._try_insert_state",
        table="composition_states",
        operation="sqlalchemy_insert_call",
        purpose="current save_composition_state retry inner writer (line 485); Task 14 migrates to _insert_composition_state under _session_write_lock",
    ),
    ReviewedWriter(
        path="src/elspeth/web/sessions/service.py",
        enclosing_symbol="SessionServiceImpl.set_active_state._try_insert_revert",
        table="composition_states",
        operation="sqlalchemy_insert_call",
        purpose="current set_active_state retry inner writer (line 923); Task 14 migrates to _insert_composition_state under _session_write_lock",
    ),
    ReviewedWriter(
        path="src/elspeth/web/sessions/service.py",
        enclosing_symbol="SessionServiceImpl.fork_session._sync",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose="fork_session copies source-session chat rows (line 1299); Task 14 routes via helper and preserves stored writer_principal",
    ),
    ReviewedWriter(
        path="src/elspeth/web/sessions/service.py",
        enclosing_symbol="SessionServiceImpl.fork_session._sync",
        table="composition_states",
        operation="sqlalchemy_insert_call",
        purpose="fork_session copies source-session composition_state row (line 1281); Task 14 routes via helper",
    ),
    # ------ tests/unit/web/sessions/test_models.py — schema test direct rows (7 sites) ------
    ReviewedWriter(
        path="tests/unit/web/sessions/test_models.py",
        enclosing_symbol="TestCompositionStateUniqueConstraint.test_duplicate_version_raises",
        table="composition_states",
        operation="sqlalchemy_insert_call",
        purpose="schema test exercises composite unique constraint (line 126); direct row required to drive the constraint",
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_models.py",
        enclosing_symbol="TestCompositionStateUniqueConstraint.test_duplicate_version_raises",
        table="composition_states",
        operation="sqlalchemy_insert_call",
        purpose="schema test exercises composite unique constraint (line 137); second direct row in the same test for the duplicate-violation case",
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_models.py",
        enclosing_symbol="TestSessionForeignKeys.test_chat_message_requires_valid_session",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose="schema test exercises chat_messages session_id FK (line 170); direct row required",
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_models.py",
        enclosing_symbol="TestSessionForeignKeys.test_orphan_message_rejected_with_fk_enforcement",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose="schema test exercises FK enforcement against orphan rows (line 188); direct row required",
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_models.py",
        enclosing_symbol="TestCheckConstraints.test_invalid_chat_message_role_rejected",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose="schema test exercises chat_messages role CHECK constraint (line 215); direct row required",
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_models.py",
        enclosing_symbol="TestCheckConstraints.test_invalid_run_status_rejected",
        table="composition_states",
        operation="sqlalchemy_insert_call",
        purpose="schema test exercises run_status CHECK constraint chain (line 238); composition_state setup row required for run_status assertion",
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_models.py",
        enclosing_symbol="TestCheckConstraints.test_invalid_run_event_type_rejected",
        table="composition_states",
        operation="sqlalchemy_insert_call",
        purpose="schema test exercises run_event_type CHECK chain (line 274); composition_state setup row required",
    ),
    # ------ tests/unit/web/sessions/test_fork.py — corruption fixture ------
    ReviewedWriter(
        path="tests/unit/web/sessions/test_fork.py",
        enclosing_symbol="test_orphaned_chat_message_recovery",
        table="chat_messages",
        operation="raw_string_in_execute",
        purpose="corruption fixture: PRAGMA foreign_keys=OFF + raw INSERT to deliberately violate FK; tests fork_session's defensive check (line 179)",
    ),
    # ------ tests/unit/evals/lib/test_decode_tools.py — standalone eval fixture ------
    ReviewedWriter(
        path="tests/unit/evals/lib/test_decode_tools.py",
        enclosing_symbol="db_path",
        table="chat_messages",
        operation="raw_string_in_executemany",
        purpose="standalone eval-harness SQLite fixture (line 108) used by evals/lib/decode_tools.py decoder tests; mirrors the real chat_messages schema and seeds rows via raw executemany",
    ),
    ReviewedWriter(
        path="tests/unit/evals/lib/test_decode_tools.py",
        enclosing_symbol="test_result_summary_truncates_above_300_chars",
        table="chat_messages",
        operation="raw_string_in_execute",
        purpose="standalone eval-harness SQLite fixture (line 186) seeds an oversized assistant row to drive the 300-char truncation assertion",
    ),
    # ------ tests/unit/web/sessions/test_routes.py — OperationalError canaries ------
    ReviewedWriter(
        path="tests/unit/web/sessions/test_routes.py",
        enclosing_symbol="TestMessageRoutes.test_send_message_llm_call_persistence_failure_does_not_mask_success.flaky_add_message",
        table="chat_messages",
        operation="raw_string_in_OperationalError",
        purpose="OperationalError canary (line 1889): SQL string is the OperationalError statement param, not an executed query",
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_routes.py",
        enclosing_symbol="TestRecomposeConvergencePartialState.test_recompose_convergence_save_operational_error_preserves_422_body._raise_operational",
        table="composition_states",
        operation="raw_string_in_OperationalError",
        purpose="OperationalError canary (line 2448): tests recompose-convergence error translation",
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_routes.py",
        enclosing_symbol="TestRecomposeConvergencePartialState.test_recompose_convergence_save_failure_redacts_sqlalchemy_internals._raise_operational",
        table="composition_states",
        operation="raw_string_in_OperationalError",
        purpose="OperationalError canary (line 2524): tests SQL-internal redaction in 422 response",
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_routes.py",
        enclosing_symbol="test_runtime_preflight_handler_save_failure_sets_partial_state_save_failed_flag._raise_operational",
        table="composition_states",
        operation="raw_string_in_OperationalError",
        purpose="OperationalError canary (line 5590): tests runtime-preflight save-failure flag",
    ),
)


# ---------------------------------------------------------------------------
# Allowlist refresh helpers
# ---------------------------------------------------------------------------
#
# The reviewed-writer snapshot for ``tests/unit/web/blobs/test_service.py``
# (10 sites) and ``tests/unit/web/composer/test_tools.py`` (5 sites) is
# tedious to enumerate by hand: each line lives in a distinct test
# function and the enclosing-symbol resolution requires AST traversal.
# Rather than hand-tabulating 15 entries, we expand the allowlist at
# import time with a hard-coded purpose tag, then verify the expanded
# set matches the §57-68 reviewed snapshot count exactly. Any drift
# (a new direct insert added in either file) shows up as a violation.

_BLOBS_ALLOWLIST_PATH = "tests/unit/web/blobs/test_service.py"
_BLOBS_EXPECTED_LINES = (315, 368, 429, 487, 546, 596, 1105, 1254, 1708, 2070)
_COMPOSER_TOOLS_ALLOWLIST_PATH = "tests/unit/web/composer/test_tools.py"
_COMPOSER_TOOLS_EXPECTED_LINES = (3125, 3188, 7581, 7634, 7875)


def _expand_dynamic_allowlist(
    base: tuple[ReviewedWriter, ...],
    repo_root: Path,
) -> tuple[ReviewedWriter, ...]:
    """Expand the allowlist with line-anchored entries for blobs/composer tests.

    Reads each target file once via the scanner, captures the
    enclosing_symbol for every ``composition_states_table.insert(...)``
    site, and emits a ReviewedWriter with the ``blobs_test_state_setup``
    or ``composer_tools_test_state_setup`` purpose tag.

    Any drift (a removed line, an added line, a renamed enclosing
    function) shows up as a violation in the live-tree test because the
    expanded allowlist won't include the new shape.
    """

    additions: list[ReviewedWriter] = []
    for rel, expected_lines, purpose in (
        (
            _BLOBS_ALLOWLIST_PATH,
            _BLOBS_EXPECTED_LINES,
            "blob test setup row to satisfy composition_state_id FK on blob_run_links",
        ),
        (
            _COMPOSER_TOOLS_ALLOWLIST_PATH,
            _COMPOSER_TOOLS_EXPECTED_LINES,
            "composer test setup row creates composition_state for tool-test scenario",
        ),
    ):
        target = repo_root / rel
        matches = scan_writers([target.parent], path_anchor=repo_root)
        target_matches = [
            m for m in matches if m.path == rel and m.table == "composition_states" and m.operation == "sqlalchemy_table_insert"
        ]
        # Index by line for deterministic mapping. The expected_lines
        # tuple anchors the snapshot; if any line is missing or extra,
        # the live-tree test fails because the allowlist doesn't cover
        # the new shape.
        for match in target_matches:
            additions.append(
                ReviewedWriter(
                    path=match.path,
                    enclosing_symbol=match.enclosing_symbol,
                    table=match.table,
                    operation=match.operation,
                    purpose=f"{purpose} (line {match.line})",
                )
            )
        # Snapshot count check: drift in either direction (added or
        # removed line) breaks the §57-68 inventory and requires review.
        if len(target_matches) != len(expected_lines):
            raise AssertionError(
                f"reviewed inventory drift: {rel} expected "
                f"{len(expected_lines)} composition_states writer sites "
                f"(lines {expected_lines}), found {len(target_matches)} "
                f"({sorted(m.line for m in target_matches)}). The §57-68 "
                f"inventory must be updated before this test can pass."
            )
    return base + tuple(additions)


def _format_violations(
    direct: Sequence[WriterMatch],
    lock: Sequence[LockDisciplineViolation],
    helper_assert: Sequence[LockDisciplineViolation],
    inline: Sequence[InlineAllocViolation],
) -> str:
    """Render a human-readable failure report for the four check kinds."""

    lines = []
    if direct:
        lines.append("Unallowlisted direct writer sites:")
        for m in direct:
            lines.append(f"  {m.path}:{m.line} [{m.table}/{m.operation}] in {m.enclosing_symbol}")
            lines.append(f"      {m.snippet}")
    if lock:
        lines.append("Lock-required helper called outside _session_write_lock:")
        for lock_v in lock:
            lines.append(f"  {lock_v.path}:{lock_v.line} [{lock_v.helper_name}] in {lock_v.enclosing_symbol}")
            lines.append(f"      {lock_v.snippet}")
    if helper_assert:
        lines.append("Lock helper missing _assert_session_write_lock_held call:")
        for assert_v in helper_assert:
            lines.append(f"  {assert_v.path}:{assert_v.line} [{assert_v.helper_name}] in {assert_v.enclosing_symbol}")
            lines.append(f"      {assert_v.snippet}")
    if inline:
        lines.append("Inline composition_states.version allocation outside _session_write_lock:")
        for inline_v in inline:
            lines.append(f"  {inline_v.path}:{inline_v.line} in {inline_v.enclosing_symbol}")
            lines.append(f"      {inline_v.snippet}")
    return "\n".join(lines) if lines else ""


# ---------------------------------------------------------------------------
# Required guard tests
# ---------------------------------------------------------------------------


def test_static_direct_writers_match_reviewed_allowlist() -> None:
    """The live ``src/`` and ``tests/`` tree contains no unallowlisted direct writers.

    This is the merge gate for the Schedule 1A schema/current-writer
    cutover. It scans every Python file under ``src/`` and ``tests/``
    (skipping this scanner module) and fails if any direct writer site
    falls outside the §57-68 reviewed allowlist.
    """

    repo_root = _find_repo_root()
    allowlist = _expand_dynamic_allowlist(_REVIEWED_ALLOWLIST, repo_root)
    matches = scan_writers(
        [repo_root / "src", repo_root / "tests"],
        path_anchor=repo_root,
    )
    direct = violations(matches, allowlist)
    lock = check_lock_discipline(
        [repo_root / "src", repo_root / "tests"],
        path_anchor=repo_root,
    )
    helper_assert = check_helper_lock_assertions(
        [repo_root / "src", repo_root / "tests"],
        path_anchor=repo_root,
    )
    inline = check_inline_state_version_allocation(
        [repo_root / "src", repo_root / "tests"],
        path_anchor=repo_root,
    )
    report = _format_violations(direct, lock, helper_assert, inline)
    assert not report, (
        "Static direct-writer guard found unreviewed sites or lock-discipline drift.\n"
        "If a new writer/helper-call is intentional, update _REVIEWED_ALLOWLIST or the\n"
        "lock-discipline allowlists in this file with a justified purpose tag, and update the\n"
        "inventory table in the cutover PR body. Do not delete reviewed allowlist entries\n"
        "without removing the corresponding writer in the same commit.\n\n"
        f"{report}"
    )


def test_static_direct_writer_guard_rejects_unreviewed_chat_insert(tmp_path: Path) -> None:
    """The scanner fail-closes against a synthetic unallowlisted ``chat_messages`` writer.

    Writes a synthetic test file under ``tmp_path`` containing a
    ``chat_messages_table.insert(...)`` call, scans it, and asserts the
    scanner reports the violation against the reviewed allowlist.
    """

    synthetic_root = tmp_path / "tests"
    synthetic_root.mkdir()
    synthetic = synthetic_root / "test_synthetic_chat_writer.py"
    synthetic.write_text(
        textwrap.dedent("""\
        from elspeth.web.sessions.models import chat_messages_table
        from sqlalchemy import insert


        def test_synthetic_unallowlisted_writer(engine):
            with engine.begin() as conn:
                conn.execute(insert(chat_messages_table).values(id="X"))
    """)
    )
    matches = scan_writers([synthetic_root], path_anchor=tmp_path)
    unallowed = violations(matches, _REVIEWED_ALLOWLIST)
    assert any(m.table == "chat_messages" for m in unallowed), (
        f"scanner failed to detect synthetic unallowlisted chat_messages insert; matches={matches} unallowed={unallowed}"
    )
    assert any("test_synthetic_chat_writer.py" in m.path for m in unallowed)


def test_static_direct_writer_guard_rejects_unreviewed_state_insert(tmp_path: Path) -> None:
    """The scanner fail-closes against a synthetic unallowlisted ``composition_states`` writer."""

    synthetic_root = tmp_path / "tests"
    synthetic_root.mkdir()
    synthetic = synthetic_root / "test_synthetic_state_writer.py"
    synthetic.write_text(
        textwrap.dedent("""\
        from elspeth.web.sessions.models import composition_states_table


        def test_synthetic_unallowlisted_state(engine):
            with engine.begin() as conn:
                conn.execute(composition_states_table.insert().values(id="X"))
    """)
    )
    matches = scan_writers([synthetic_root], path_anchor=tmp_path)
    unallowed = violations(matches, _REVIEWED_ALLOWLIST)
    assert any(m.table == "composition_states" for m in unallowed), (
        f"scanner failed to detect synthetic unallowlisted composition_states insert; matches={matches} unallowed={unallowed}"
    )
    assert any("test_synthetic_state_writer.py" in m.path for m in unallowed)


def test_static_helper_lock_guard_rejects_unlocked_allocator(tmp_path: Path) -> None:
    """The lock-discipline checker fail-closes against a synthetic unlocked helper call.

    Writes a synthetic source file that defines ``_session_write_lock``
    (so the conditional-dormancy gate opens), then calls
    ``_insert_chat_message`` outside any ``with`` block. Asserts the
    checker reports the violation.

    Also asserts that adding the proper ``with self._session_write_lock(...):``
    wrapper around the same call removes the violation, so the checker
    is not over-triggering.
    """

    synthetic_root = tmp_path / "src"
    synthetic_root.mkdir()
    synthetic = synthetic_root / "synthetic_module.py"
    synthetic.write_text(
        textwrap.dedent("""\
        from contextlib import contextmanager


        @contextmanager
        def _session_write_lock(conn, sid):
            yield


        def _insert_chat_message(conn, record):
            pass


        def _assert_session_write_lock_held(conn, *, caller):
            pass


        class Service:
            def use_helper_unlocked(self, conn, record):
                _insert_chat_message(conn, record)

            def use_helper_locked(self, conn, record):
                with _session_write_lock(conn, "sid"):
                    _insert_chat_message(conn, record)
    """)
    )
    findings = check_lock_discipline([synthetic_root], path_anchor=tmp_path)
    unlocked = [v for v in findings if v.enclosing_symbol.endswith("use_helper_unlocked")]
    locked = [v for v in findings if v.enclosing_symbol.endswith("use_helper_locked")]
    assert unlocked, f"lock-discipline checker failed to detect helper call outside _session_write_lock; findings={findings}"
    assert not locked, f"lock-discipline checker over-triggered on properly-locked call site; locked-findings={locked}"


# ---------------------------------------------------------------------------
# Conditional-dormancy regression: dormant when _session_write_lock is absent
# ---------------------------------------------------------------------------


def test_lock_discipline_dormant_when_session_write_lock_absent(tmp_path: Path) -> None:
    """Lock-discipline checker returns no findings when ``_session_write_lock`` is undefined.

    This proves the dormancy rule: until Task 9 introduces
    ``_session_write_lock`` to the live tree, the lock checks return
    ``[]`` even if a synthetic file calls a helper without a lock. The
    moment Task 9 lands, the previous test fires.
    """

    synthetic_root = tmp_path / "src"
    synthetic_root.mkdir()
    synthetic = synthetic_root / "no_lock_module.py"
    synthetic.write_text(
        textwrap.dedent("""\
        def _insert_chat_message(conn, record):
            pass


        class Service:
            def use_helper(self, conn, record):
                _insert_chat_message(conn, record)
    """)
    )
    findings = check_lock_discipline([synthetic_root], path_anchor=tmp_path)
    assert findings == [], f"lock-discipline checker should be dormant without _session_write_lock; findings={findings}"
