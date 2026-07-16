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


@dataclass(frozen=True)
class LockDisciplineNegativeTest:
    """Allowlist entry for an intentional lock-required-helper call outside ``_session_write_lock``.

    Plan §94 explicitly requires the static guard to support ``negative
    test`` exemptions: each helper has a precondition assertion
    (``_assert_session_write_lock_held``), and verifying that assertion
    fires REQUIRES calling the helper without the lock. The three keying
    fields ``(path, enclosing_symbol, helper_name)`` must match the
    scanner's :class:`LockDisciplineViolation` exactly. ``purpose`` is
    informational and shows up in the violation report when a new site
    fails the gate.
    """

    path: str
    enclosing_symbol: str
    helper_name: str
    purpose: str


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
    allowlist: Sequence[LockDisciplineNegativeTest] = (),
) -> list[LockDisciplineViolation]:
    """Check that every call to a lock-required helper is inside ``_session_write_lock``.

    Conditional-dormant: if ``_session_write_lock`` is not defined anywhere
    under ``roots``, returns ``[]``. As soon as the helpers land (Task 9),
    every caller that drifts off the lock is flagged.

    Plan §94 carve-out: callers explicitly listed in ``allowlist`` are
    exempt. The exemption mechanism exists because each helper has a
    precondition assertion that REQUIRES a negative test calling the
    helper outside the lock. The default empty tuple keeps strict
    semantics for callers that do not need exemptions.
    """

    if not _codebase_defines_symbol(_SESSION_WRITE_LOCK_NAME, roots):
        return []

    allowed_keys = {(entry.path, entry.enclosing_symbol, entry.helper_name) for entry in allowlist}

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
            enclosing_symbol = _qualified_symbol(node)
            if (rel, enclosing_symbol, helper_name) in allowed_keys:
                continue
            line = getattr(node, "lineno", 0)
            findings.append(
                LockDisciplineViolation(
                    path=rel,
                    line=line,
                    enclosing_symbol=enclosing_symbol,
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
    # NOTE: ``SessionServiceImpl.add_message._sync`` no longer contains an
    # inline ``insert(chat_messages_table)``. Task 14's rewrite (plan §3174-
    # 3268) routes the writer through ``_insert_chat_message`` under
    # ``_session_write_lock`` after a ``_reserve_sequence_range`` allocation.
    # The corresponding ``ReviewedWriter`` entry that previously sat here has
    # been removed because keeping a stale entry for a writer that no longer
    # exists violates the "do not leave stale promises" rule (Task 10
    # handover pitfall §5).
    ReviewedWriter(
        path="src/elspeth/web/sessions/service.py",
        enclosing_symbol="SessionServiceImpl.save_composition_state._sync",
        table="composition_states",
        operation="sqlalchemy_insert_call",
        purpose=(
            "save_composition_state inline writer; Task 10 lock-retrofits "
            "in place (NOT helper-routed) per plan §2128-2133 — uniform "
            "helper-routing would either lose the per-site retry semantics "
            "(race risk) or grow per-site escape hatches. The SELECT-MAX + "
            "INSERT region is wrapped in ``_session_write_lock`` so the "
            "inline allocation runs under the same per-session write "
            "discipline as ``_insert_composition_state``. The B3 "
            "belt-and-suspenders retry loop was deleted: under CLAUDE.md "
            "No Legacy Code Policy a 'slated for removal' shim is "
            "forbidden, and the loop's RuntimeError fallback masked the "
            "uq_composition_state_version IntegrityError chain that names "
            "any future lock-discipline regression directly"
        ),
    ),
    ReviewedWriter(
        path="src/elspeth/web/sessions/service.py",
        enclosing_symbol="SessionServiceImpl.set_active_state._sync",
        table="composition_states",
        operation="sqlalchemy_insert_call",
        purpose=(
            "set_active_state inline writer; same lock-retrofit-in-place "
            "discipline as save_composition_state above. The B3 "
            "belt-and-suspenders retry loop was deleted with the same "
            "rationale (No Legacy Code Policy + diagnostic-preservation)"
        ),
    ),
    ReviewedWriter(
        path="src/elspeth/web/sessions/service.py",
        enclosing_symbol="SessionServiceImpl.fork_session._sync",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose=(
            "fork_session batch-copies source-session chat rows. Task 14 (§14.6) "
            "did NOT route this through ``_insert_chat_message`` — that would mean "
            "N single-row inserts instead of one batch ``conn.execute(insert(...), "
            "rows)`` and is materially slower for large source histories. Instead, "
            "the batch is now wrapped in ``_session_write_lock(new_session_id)`` "
            "with the chat ``sequence_no`` reserved via ``_reserve_sequence_range`` "
            "for ``len(msg_records_data)`` rows in one allocation; the same lock "
            "context covers the composition-state copy. ``writer_principal`` is "
            "preserved verbatim from the source row (no role-keyed fabrication); "
            "synthetic system + new edited-user rows use ``writer_principal="
            "session_fork``. Tool rows have ``parent_assistant_id`` rewritten to "
            "the copied assistant id; rows whose source parent is excluded from "
            "the slice raise the precise RuntimeError before the FK can fire."
        ),
    ),
    ReviewedWriter(
        path="src/elspeth/web/sessions/service.py",
        enclosing_symbol="SessionServiceImpl._insert_chat_message",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose=(
            "Task 9 chat-row writer (plan §1850-2110): the canonical chat_messages "
            "writer. Task 14's call-site sweep routed every prior production writer "
            "through this helper (``add_message`` rewrite at plan §3174-3268; the "
            "``fork_session`` batch path retains a direct ``insert(chat_messages_"
            "table)`` for batch performance under the same lock + sequence_no "
            "discipline — see the entry above). Caller is required to be inside "
            "_session_write_lock (asserted via _assert_session_write_lock_held) and "
            "to have already obtained sequence_no from _reserve_sequence_range; the "
            "negative precondition test is allowlisted in "
            "_LOCK_DISCIPLINE_NEGATIVE_TESTS."
        ),
    ),
    ReviewedWriter(
        path="src/elspeth/web/sessions/service.py",
        enclosing_symbol="SessionServiceImpl._insert_composition_state",
        table="composition_states",
        operation="raw_string_module",
        purpose=(
            "Task 10 helper docstring (plan §2112-2751) references the target "
            "table by name in the PRECONDITION/B1/B3 prose; not an actual "
            "write site"
        ),
    ),
    ReviewedWriter(
        path="src/elspeth/web/sessions/service.py",
        enclosing_symbol="SessionServiceImpl._insert_composition_state",
        table="composition_states",
        operation="sqlalchemy_insert_call",
        purpose=(
            "Task 10 composition-state writer (plan §2112-2751): the canonical "
            "composition_states writer for fork-session and (in Phase 3) "
            "compose-loop use. B1 contract — the helper allocates ``version`` "
            "internally via ``SELECT COALESCE(MAX(version),0)+1 WHERE "
            "session_id=:sid`` under the held ``_session_write_lock`` "
            "(asserted via ``_assert_session_write_lock_held``). The "
            "SELECT-MAX-then-INSERT atomicity closes the "
            "fabricated-Tier-1-violation race at the contract boundary "
            "rather than at individual call sites. Sites 403/834 do NOT "
            "route through this helper — see plan §2128-2133 for the "
            "asymmetric-mechanism rationale. Negative precondition test "
            "allowlisted in _LOCK_DISCIPLINE_NEGATIVE_TESTS"
        ),
    ),
    # NOTE: fork_session._sync no longer contains an inline composition_states
    # insert. Task 10 refactored that site to call ``_insert_composition_state``
    # under ``_session_write_lock``. The helper carries its own allowlist entry
    # above ("SessionServiceImpl._insert_composition_state"). The corresponding
    # entry that previously sat here has been removed because keeping a stale
    # ``ReviewedWriter`` for a writer that no longer exists violates the
    # "do not leave stale promises" rule in the Task 10 handover (pitfall §5)
    # and the test-file's "Do not delete reviewed allowlist entries without
    # removing the corresponding writer in the same commit" symmetry.
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
    # ------ tests/unit/web/sessions/test_interpretation_events_table.py — Phase 5b Task 2 schema tests (4 sites) ------
    ReviewedWriter(
        path="tests/unit/web/sessions/test_interpretation_events_table.py",
        enclosing_symbol="_seed_composition_state",
        table="composition_states",
        operation="sqlalchemy_insert_call",
        purpose=(
            "Phase 5b Task 2 schema test helper: seeds a composition_states row "
            "to satisfy the composite FK on interpretation_events. Schema-test "
            "direct insert — no production lock required because the test owns "
            "the in-memory SQLite engine and is exercising DDL/constraint "
            "behaviour, not the production write path. Helper is intentionally "
            "named _seed_composition_state (not _insert_composition_state) so "
            "the lock-discipline scanner does not conflate it with the "
            "production SessionServiceImpl._insert_composition_state."
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_interpretation_events_table.py",
        enclosing_symbol="TestCompositionStatesProvenanceEnum.test_invalid_provenance_rejected",
        table="composition_states",
        operation="sqlalchemy_insert_call",
        purpose=(
            "Phase 5b Task 2 schema test: drives the "
            "ck_composition_states_provenance CHECK constraint by inserting an "
            "invalid provenance value directly; bypassing the helper is the "
            "point of the test (the helper would only ever pass valid values)."
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_interpretation_events_table.py",
        enclosing_symbol="test_blob_llm_provenance_rejects_blank_strings",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose=(
            "blob provenance schema test: seeds the created_from_message_id "
            "anchor so the test can isolate the creating_* blank-string CHECK "
            "constraint instead of failing the composite FK first."
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/shareable_reviews/test_service.py",
        enclosing_symbol="session_engine_with_row",
        table="composition_states",
        operation="sqlalchemy_table_insert",
        purpose=(
            "Phase 6A Task 5 unit-test fixture: seeds a composition_states row "
            "to satisfy the composite FK on composer_completion_events. The "
            "ShareableReviewService's audit write references "
            "composition_state_id; the test owns the in-memory SQLite engine "
            "and the fixture exists solely to populate the parent row for FK "
            "resolution. No production lock required — the test exercises "
            "service logic, not the production write path."
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_composer_completion_events_table.py",
        enclosing_symbol="_seed_composition_state",
        table="composition_states",
        operation="sqlalchemy_insert_call",
        purpose=(
            "Phase 6A schema-test helper: seeds a composition_states row to "
            "satisfy the per-event-type CHECK constraint "
            "ck_composer_completion_events_composition_state_id_required "
            "and the composite FK on composer_completion_events. Helper is "
            "named _seed_composition_state (not _insert_composition_state) "
            "so the lock-discipline scanner does not conflate it with the "
            "production SessionServiceImpl._insert_composition_state. "
            "Schema-test direct insert — no production lock required."
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/shareable_reviews/test_telemetry_session_completed.py",
        enclosing_symbol="session_engine_with_row",
        table="composition_states",
        operation="sqlalchemy_table_insert",
        purpose=(
            "Phase 8 Sub-task 7c (telemetry-backfill: phase-6) telemetry test: "
            "seeds a composition_states row so the ShareableReviewService's "
            "mark_ready_for_review audit insert resolves the composite FK on "
            "composer_completion_events. Mirrors the precedent immediately "
            "above (test_service.py session_engine_with_row); the new test "
            "asserts the composer.session.completed_total counter emit at the "
            "service. No production lock required — telemetry-emit test, not a "
            "production write path."
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_interpretation_events_table.py",
        enclosing_symbol="TestCompositionStatesProvenanceEnum.test_interpretation_resolve_provenance_accepted",
        table="composition_states",
        operation="sqlalchemy_insert_call",
        purpose=(
            "Phase 5b Task 2 schema test: positive case for the new "
            "'interpretation_resolve' provenance enum value; direct insert "
            "asserts the CHECK constraint accepts the new value."
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_interpretation_events_table.py",
        enclosing_symbol="TestTriggerInstalledByBootstrap.test_chat_messages_content_immutable",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose=(
            "Phase 5b Task 2 schema test: seeds a chat_messages row to assert "
            "that trg_chat_messages_immutable_content fires on UPDATE OF "
            "content. Direct insert is required because the test exercises "
            "trigger behaviour, not the production writer."
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_interpretation_events_table.py",
        enclosing_symbol="TestTriggerInstalledByBootstrap.test_chat_messages_delete_raises_even_without_blob_reference",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose=(
            "schema trigger test: seeds a chat_messages row to assert that "
            "trg_chat_messages_no_delete blocks direct DELETE even when no "
            "blob lineage FK exists. Direct insert is required because the "
            "test is isolating trigger behaviour, not exercising the "
            "production writer."
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_interpretation_events_table.py",
        enclosing_symbol="TestTriggerInstalledByBootstrap.test_chat_messages_delete_allowed_only_through_session_cascade",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose=(
            "schema trigger test: seeds a chat_messages row to assert that "
            "whole-session archival may remove transcript rows only through "
            "the sessions-table FK cascade. Direct insert keeps the test "
            "focused on trigger/cascade semantics."
        ),
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
    ReviewedWriter(
        path="tests/unit/evals/lib/test_decode_tools.py",
        enclosing_symbol="test_decode_tool_sequence_orders_same_timestamp_rows_by_sequence_no",
        table="chat_messages",
        operation="raw_string_in_executemany",
        purpose=(
            "§14.7 / plan §3884 regression: standalone SQLite fixture seeds rows "
            "with same created_at + intentionally non-chronological sequence_no "
            "via raw executemany so the decoder's ORDER BY sequence_no can be "
            "verified independently of the rev-4 service-layer writer path"
        ),
    ),
    # ------ tests/property/web/composer/test_compose_loop_invariants.py — property harness seed ------
    ReviewedWriter(
        path="tests/property/web/composer/test_compose_loop_invariants.py",
        enclosing_symbol="_make_harness",
        table="chat_messages",
        operation="raw_string_in_text",
        purpose=(
            "property harness fixture seeds a prior user row only for the "
            "has_prior_state scenario before driving ComposerServiceImpl's "
            "real compose-loop persistence; raw text keeps the injection arm "
            "mechanically close to the simulated commit/advisory-lock failures"
        ),
    ),
    # ------ tests/unit/web/composer/* — blob-provenance user-message fixtures ------
    ReviewedWriter(
        path="tests/unit/web/composer/test_blob_inline_tools.py",
        enclosing_symbol="blob_env",
        table="chat_messages",
        operation="sqlalchemy_table_insert",
        purpose=(
            "inline-blob tool fixture: seeds the route-level user message "
            "that blob provenance requires before exercising the production "
            "blob tool handlers. The fixture owns the in-memory SQLite engine "
            "and exists only to provide the immutable created_from_message_id "
            "anchor for attribution assertions."
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/composer/test_agent_tooling.py",
        enclosing_symbol="_insert_user_message",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose=(
            "blob-tool harness fixture: seeds the route-level user message "
            "that create_blob provenance now requires before exercising the "
            "production execute_tool dispatcher. The fixture owns the in-memory "
            "SQLite engine and exists only to satisfy fk_blobs_created_from_"
            "message_session for verbatim content assertions."
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/composer/test_promote_create_blob.py",
        enclosing_symbol="_session_engine_with_user_message",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose=(
            "create_blob provenance fixture: seeds the triggering user message "
            "so tests can assert verbatim vs LLM-generated blob attribution "
            "against the real blobs_table composite FK."
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/composer/test_promote_set_pipeline.py",
        enclosing_symbol="_session_engine_with_user_message",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose=(
            "set_pipeline inline-blob provenance fixture: seeds the triggering "
            "user message so inline source blobs can bind created_from_message_id "
            "while the tests focus on blob attribution and state mutation."
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/composer/test_promote_set_source_from_blob.py",
        enclosing_symbol="_session_engine_with_user_message",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose=(
            "set_source_from_blob provenance fixture: seeds the triggering user "
            "message for blob attribution tests without initialising the full "
            "session route stack."
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/composer/test_promote_update_blob.py",
        enclosing_symbol="_insert_user_message",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose=(
            "update_blob functional smoke fixture: bootstraps a create_blob "
            "row with the required verbatim user-message anchor before "
            "exercising the production update_blob handler."
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/composer/test_service.py",
        enclosing_symbol="_insert_user_message",
        table="chat_messages",
        operation="sqlalchemy_table_insert",
        purpose=(
            "composer-service fixture helper: creates a deterministic "
            "route-level user message anchor for blob provenance tests while "
            "the service path under test remains the composer loop."
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/composer/test_service.py",
        enclosing_symbol="TestComposerTextOnlyResponse.test_blob_only_success_then_empty_state_reply_returns_no_state_mutation_blocker",
        table="chat_messages",
        operation="sqlalchemy_table_insert",
        purpose=(
            "composer-service scenario fixture: seeds the exact user-message "
            "anchor for a blob-only turn so the test can assert the text-only "
            "response handling rather than the chat-message writer."
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/composer/test_tools.py",
        enclosing_symbol="_insert_user_message",
        table="chat_messages",
        operation="sqlalchemy_table_insert",
        purpose=(
            "composer tool fixture helper: seeds route-level user-message "
            "anchors for verbatim blob provenance assertions while exercising "
            "the production tool handlers."
        ),
    ),
    # ------ tests/unit/web/sessions/* — targeted chat transcript fixtures ------
    ReviewedWriter(
        path="tests/unit/web/sessions/test_blob_inline_resolutions_schema.py",
        enclosing_symbol="_seed_run_and_blob",
        table="composition_states",
        operation="sqlalchemy_insert_call",
        purpose=(
            "blob-inline-resolution schema fixture: seeds a composition_states "
            "parent row so blob_inline_resolutions FK and CHECK constraints can "
            "be exercised directly. The test owns the in-memory SQLite engine "
            "and is pinning schema behaviour, not the production session writer."
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_record_blob_inline_resolutions.py",
        enclosing_symbol="_seed_run_and_blob",
        table="composition_states",
        operation="sqlalchemy_insert_call",
        purpose=(
            "record_blob_inline_resolutions service fixture: seeds the parent "
            "composition_states row required by the audit table FK before "
            "exercising the production SessionServiceImpl audit writer. Direct "
            "setup keeps the test focused on the audit write and DB-failure "
            "behaviour."
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_service.py",
        enclosing_symbol="TestRunEvents.test_append_and_list_run_events_preserves_order_and_payload",
        table="composition_states",
        operation="sqlalchemy_insert_call",
        purpose=(
            "0.6.0 run-events service test: seeds the session + composition_"
            "states + runs FK chain so the test can exercise the production "
            "append_run_event / list_run_events path. The composition_states "
            "insert is only the parent-row anchor required by runs_table's "
            "state_id FK; the test owns the in-memory SQLite engine and is "
            "pinning the run-event ordering/payload, not the production "
            "session writer."
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_count_tool_responses_for_assistant.py",
        enclosing_symbol="_persist_assistant_with_tools",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose=(
            "read-helper fixture seeds one assistant row plus N tool rows to "
            "verify count_tool_responses_for_assistant against persisted "
            "parent_assistant_id/tool_call_id shapes; not a production writer"
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_messages_route_include_tool_rows.py",
        enclosing_symbol="_seed_user_assistant_tool_rows",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose=(
            "route-view fixture seeds a minimal user/assistant/tool transcript "
            "so include_tool_rows filtering can be verified without invoking "
            "the composer loop"
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_record_audit_grade_view.py",
        enclosing_symbol="_seed_user_assistant_tool_rows",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose=(
            "audit-grade transcript view fixture seeds a minimal "
            "user/assistant/tool transcript; the test target is the audit "
            "access log/view path, not chat-message writing"
        ),
    ),
    # ------ tests/unit/web/sessions/test_routes.py — OperationalError canaries ------
    ReviewedWriter(
        path="tests/unit/web/sessions/test_routes.py",
        enclosing_symbol="TestMessageRoutes.test_send_message_llm_call_persistence_failure_raises_on_success_path.flaky_add_message",
        table="chat_messages",
        operation="raw_string_in_OperationalError",
        purpose=(
            "Tier-1 audit-corruption regression: the OperationalError's "
            "statement string carries 'INSERT INTO chat_messages' to make "
            "the simulated failure look like a real DB write error; the "
            "test asserts the success-path helper raises AuditIntegrityError "
            "(500) rather than swallowing the failure. Not an executed query"
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_routes.py",
        enclosing_symbol="TestMessageRoutes.test_send_message_tool_invocation_persistence_failure_raises_on_success_path.flaky_add_message",
        table="chat_messages",
        operation="raw_string_in_OperationalError",
        purpose=(
            "Symmetric to the LLM-call Tier-1 canary above. Statement string "
            "is the OperationalError param to simulate a real chat_messages "
            "INSERT failure; the test asserts the success-path tool-invocation "
            "helper raises AuditIntegrityError (500). Not an executed query"
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_routes.py",
        enclosing_symbol="TestMessageRoutes.test_guided_respond_tool_invocation_persistence_failure_raises_on_success_path.flaky_add_message",
        table="chat_messages",
        operation="raw_string_in_OperationalError",
        purpose=(
            "Guided-mode audit sidecar canary: OperationalError statement "
            "string simulates a real chat_messages INSERT failure after the "
            "state transition succeeds; not an executed query"
        ),
    ),
    ReviewedWriter(
        path="tests/unit/web/sessions/test_routes.py",
        enclosing_symbol="TestMessageRoutes.test_guided_chat_turn_persistence_failure_raises_on_success_path.flaky_add_message",
        table="chat_messages",
        operation="raw_string_in_OperationalError",
        purpose=(
            "Guided chat audit-row canary: OperationalError statement string "
            "simulates a failed audit chat_messages INSERT so the route must "
            "surface 500 instead of swallowing audit loss; not an executed query"
        ),
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
    # ------ tests/integration/web/composer/test_inline_source_provenance.py ------
    #
    # Phase 5a Task 2.5 integration test seeds a session + one user
    # chat_messages row directly (no compose loop). These are
    # fixture-only inserts that verify the new
    # ``creation_modality`` / ``created_from_message_id`` /
    # ``creating_*`` columns + composite FK on ``blobs_table``; routing
    # them through ``SessionServiceImpl.add_message`` would require
    # spinning up the full sessions service and offload worker just to
    # land a single deterministic message id, which adds no audit-
    # integrity coverage and obscures the schema-level assertions the
    # test is actually pinning.
    ReviewedWriter(
        path="tests/integration/web/composer/test_inline_source_provenance.py",
        enclosing_symbol="_session_with_user_message",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose=(
            "Phase 5a Task 2.5 inline-source provenance fixture: seeds one "
            "session + one user chat message so the test can assert the new "
            "blobs_table provenance columns (creation_modality, "
            "created_from_message_id, creating_*) and the composite FK "
            "fk_blobs_created_from_message_session. Direct insert keeps the "
            "fixture deterministic (caller controls the message id) and "
            "scope-narrow (no service-stack initialisation)."
        ),
    ),
    ReviewedWriter(
        path="tests/integration/web/composer/test_inline_source_provenance.py",
        enclosing_symbol="test_cross_session_message_id_rejected",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose=(
            "Phase 5a Task 2.5 cross-session FK rejection test: seeds a "
            "second session (B) with its own user message so the test can "
            "drive a blob insert in session A that references session B's "
            "message id. The composite FK fk_blobs_created_from_message_session "
            "must raise IntegrityError; routing through add_message would "
            "obscure the schema-level assertion."
        ),
    ),
    # ------ tests/integration/web/composer/test_chat_messages_attributability.py ------
    #
    # Phase 5a Task 2.6 attributability test seeds a session + one user
    # chat_messages row directly so the blob-provenance / immutable-content
    # assertions have a stable anchor row to bind against. The production
    # write path is still exercised in the same test via
    # ``_prepare_blob_create`` + ``_persist_prepared_blob_create``; the
    # direct insert is only the chat-row anchor, not the system-under-test.
    ReviewedWriter(
        path="tests/integration/web/composer/test_chat_messages_attributability.py",
        enclosing_symbol="_session_with_user_message_and_blob",
        table="chat_messages",
        operation="sqlalchemy_insert_call",
        purpose=(
            "Phase 5a Task 2.6 attributability fixture: seeds one session + "
            "one user chat message so the test can persist a blob via the "
            "real composer write path (_prepare_blob_create + "
            "_persist_prepared_blob_create) and assert the composite FK "
            "fk_blobs_created_from_message_session binds to a stable, "
            "caller-controlled message id. Routing the anchor row through "
            "SessionServiceImpl.add_message would obscure the schema-level "
            "assertions (created_from_message_id immutability, trigger "
            "trg_chat_messages_immutable_content) the test is pinning."
        ),
    ),
    # ------ tests/testcontainer/web/test_schema_probe_postgres.py ------
    # The PostgreSQL trigger proof deliberately seeds the protected rows with
    # raw SQL so it can mutate them through the same low-level connection and
    # prove the database triggers themselves enforce append-only semantics.
    ReviewedWriter(
        path="tests/testcontainer/web/test_schema_probe_postgres.py",
        enclosing_symbol="_seed_postgres_trigger_rows",
        table="composition_states",
        operation="raw_string_in_text",
        purpose=(
            "PostgreSQL trigger fixture: seed the composition-state FK anchor "
            "with raw SQL before directly exercising immutable audit triggers."
        ),
    ),
    ReviewedWriter(
        path="tests/testcontainer/web/test_schema_probe_postgres.py",
        enclosing_symbol="_seed_postgres_trigger_rows",
        table="chat_messages",
        operation="raw_string_in_text",
        purpose=(
            "PostgreSQL trigger fixture: seed the protected chat row with raw "
            "SQL so update/delete trigger enforcement is tested independently "
            "of the session service."
        ),
    ),
)


# Lock-discipline negative-test allowlist (plan §94)
#
# Each lock-required helper (``_reserve_sequence_range``,
# ``_insert_chat_message``, ``_insert_composition_state``) has a
# precondition assertion ``_assert_session_write_lock_held`` that fires
# RuntimeError when invoked outside ``_session_write_lock``. Verifying
# that assertion REQUIRES a test that calls the helper without the
# lock — so by construction the static lock-discipline check would
# flag the test. Plan §94 explicitly authorises an allowlist exemption
# for these specific test sites; this tuple is that exemption surface.
#
# Add a new entry only when adding a corresponding
# ``test_<helper>_requires_session_write_lock`` test that exercises the
# precondition. Removing a helper means removing its negative test AND
# its allowlist entry in the same commit.
_LOCK_DISCIPLINE_NEGATIVE_TESTS: tuple[LockDisciplineNegativeTest, ...] = (
    LockDisciplineNegativeTest(
        path="tests/unit/web/sessions/test_persist_compose_turn.py",
        enclosing_symbol="test_reserve_sequence_range_requires_session_write_lock",
        helper_name="_reserve_sequence_range",
        purpose="negative-precondition test (plan §1623): verifies _assert_session_write_lock_held raises when the lock is not held",
    ),
    LockDisciplineNegativeTest(
        path="tests/unit/web/sessions/test_persist_compose_turn.py",
        enclosing_symbol="test_insert_chat_message_requires_session_write_lock",
        helper_name="_insert_chat_message",
        purpose="negative-precondition test (plan §1924): verifies _assert_session_write_lock_held raises when the lock is not held",
    ),
    LockDisciplineNegativeTest(
        path="tests/unit/web/sessions/test_persist_compose_turn.py",
        enclosing_symbol="test_insert_composition_state_requires_session_write_lock",
        helper_name="_insert_composition_state",
        purpose="negative-precondition test (plan §2471): verifies _assert_session_write_lock_held raises when the lock is not held",
    ),
)


# ---------------------------------------------------------------------------
# Allowlist refresh helpers
# ---------------------------------------------------------------------------
#
# The reviewed-writer snapshot for ``tests/unit/web/blobs/test_service.py``
# (12 sites) and ``tests/unit/web/composer/test_tools.py`` (5 sites) is
# tedious to enumerate by hand: each line lives in a distinct test
# function and the enclosing-symbol resolution requires AST traversal.
# Rather than hand-tabulating 16 entries, we expand the allowlist at
# import time with a hard-coded purpose tag, then verify the expanded
# set matches the §57-68 reviewed snapshot count exactly. Any drift
# (a new direct insert added in either file) shows up as a violation.

_BLOBS_ALLOWLIST_PATH = "tests/unit/web/blobs/test_service.py"
_BLOBS_EXPECTED_LINES = (318, 377, 441, 520, 590, 656, 733, 795, 1366, 1521, 1976, 2344)
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
        allowlist=_LOCK_DISCIPLINE_NEGATIVE_TESTS,
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


def test_lock_discipline_allowlist_exempts_negative_precondition_test(tmp_path: Path) -> None:
    """Plan §94: negative-precondition tests must be exempt from the lock-discipline check.

    Synthesises a test file with two helper calls outside any lock:
    one in an allowlisted ``test_<helper>_requires_session_write_lock``
    function, and one in an unrelated function. Asserts the allowlist
    suppresses ONLY the matching site, not the unrelated one. Without
    the allowlist, both calls must be flagged (no false negatives).
    """

    synthetic_src = tmp_path / "src"
    synthetic_src.mkdir()
    (synthetic_src / "synthetic_module.py").write_text(
        textwrap.dedent("""\
        from contextlib import contextmanager


        @contextmanager
        def _session_write_lock(conn, sid):
            yield


        def _reserve_sequence_range(conn, sid, *, count):
            return 1


        def _assert_session_write_lock_held(conn, *, caller):
            pass
    """)
    )
    synthetic_tests = tmp_path / "tests"
    synthetic_tests.mkdir()
    (synthetic_tests / "test_synthetic_helpers.py").write_text(
        textwrap.dedent("""\
        from synthetic_module import _reserve_sequence_range


        def test_reserve_sequence_range_requires_session_write_lock(service):
            # Negative-precondition test: deliberately calls helper outside lock.
            _reserve_sequence_range(service, "s_no_lock", count=1)


        def test_unrelated_thing(service):
            # Not a precondition test; must NOT be exempted.
            _reserve_sequence_range(service, "s_other", count=1)
    """)
    )

    allowlist = (
        LockDisciplineNegativeTest(
            path="tests/test_synthetic_helpers.py",
            enclosing_symbol="test_reserve_sequence_range_requires_session_write_lock",
            helper_name="_reserve_sequence_range",
            purpose="synthetic regression test for the allowlist mechanism",
        ),
    )

    strict_findings = check_lock_discipline([synthetic_src, synthetic_tests], path_anchor=tmp_path)
    strict_symbols = {v.enclosing_symbol for v in strict_findings}
    assert "test_reserve_sequence_range_requires_session_write_lock" in strict_symbols, (
        f"scanner failed to flag the negative-precondition call site without the allowlist; strict_findings={strict_findings}"
    )
    assert "test_unrelated_thing" in strict_symbols, (
        f"scanner failed to flag the unrelated call site without the allowlist; strict_findings={strict_findings}"
    )

    permissive_findings = check_lock_discipline([synthetic_src, synthetic_tests], path_anchor=tmp_path, allowlist=allowlist)
    permissive_symbols = {v.enclosing_symbol for v in permissive_findings}
    assert "test_reserve_sequence_range_requires_session_write_lock" not in permissive_symbols, (
        f"allowlist failed to suppress the matching negative-precondition site; permissive_findings={permissive_findings}"
    )
    assert "test_unrelated_thing" in permissive_symbols, (
        f"allowlist over-suppressed an unrelated site (key mismatch should fail closed); permissive_findings={permissive_findings}"
    )


def test_lock_discipline_allowlist_key_match_is_exact(tmp_path: Path) -> None:
    """Allowlist matching must be exact on (path, enclosing_symbol, helper_name).

    A near-miss on any of the three keys must NOT suppress the violation.
    This guards against accidental over-broad suppression — e.g., an
    allowlist entry for ``_reserve_sequence_range`` in one file leaking
    to a same-name function in another file.
    """

    synthetic_src = tmp_path / "src"
    synthetic_src.mkdir()
    (synthetic_src / "synthetic_module.py").write_text(
        textwrap.dedent("""\
        from contextlib import contextmanager


        @contextmanager
        def _session_write_lock(conn, sid):
            yield


        def _reserve_sequence_range(conn, sid, *, count):
            return 1


        def _assert_session_write_lock_held(conn, *, caller):
            pass
    """)
    )
    synthetic_tests = tmp_path / "tests"
    synthetic_tests.mkdir()
    (synthetic_tests / "test_other_file.py").write_text(
        textwrap.dedent("""\
        from synthetic_module import _reserve_sequence_range


        def test_reserve_sequence_range_requires_session_write_lock(service):
            _reserve_sequence_range(service, "s", count=1)
    """)
    )

    # Allowlist entry references a DIFFERENT path — same symbol/helper.
    mismatched_path = (
        LockDisciplineNegativeTest(
            path="tests/some_other_path.py",
            enclosing_symbol="test_reserve_sequence_range_requires_session_write_lock",
            helper_name="_reserve_sequence_range",
            purpose="path-mismatch regression",
        ),
    )
    findings = check_lock_discipline([synthetic_src, synthetic_tests], path_anchor=tmp_path, allowlist=mismatched_path)
    matching = [v for v in findings if v.enclosing_symbol == "test_reserve_sequence_range_requires_session_write_lock"]
    assert matching, f"allowlist with mismatched path must NOT suppress; findings={findings}"

    # Allowlist entry references a DIFFERENT helper — same path/symbol.
    mismatched_helper = (
        LockDisciplineNegativeTest(
            path="tests/test_other_file.py",
            enclosing_symbol="test_reserve_sequence_range_requires_session_write_lock",
            helper_name="_insert_chat_message",
            purpose="helper-mismatch regression",
        ),
    )
    findings = check_lock_discipline([synthetic_src, synthetic_tests], path_anchor=tmp_path, allowlist=mismatched_helper)
    matching = [v for v in findings if v.enclosing_symbol == "test_reserve_sequence_range_requires_session_write_lock"]
    assert matching, f"allowlist with mismatched helper_name must NOT suppress; findings={findings}"


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
