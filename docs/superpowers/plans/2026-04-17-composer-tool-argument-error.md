# Composer Tool Argument Error Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the silent channel in `_compose_loop()` that conflates LLM-argument errors with plugin bugs, by introducing an explicit `ToolArgumentError` and narrowing the service-level catch to that type only.

**Architecture:** Option 1 from filigree issue `elspeth-7a26880c65`. Tool handlers that detect a Tier-3 boundary violation (LLM supplied the wrong type or a semantically-invalid value that cannot be coerced) raise a dedicated `ToolArgumentError(Exception)`. The compose loop catches *only* `ToolArgumentError` and feeds the message back to the LLM. Every other `TypeError`/`ValueError`/`UnicodeError` escaping `execute_tool()` is treated as a plugin bug and crashes the loop — same semantic as the existing `KeyError` path asserted by `test_internal_key_error_is_not_swallowed`. An optional CI enforcement gate (Task 7) converts the "don't raise bare TypeError/ValueError from tools.py" convention into a mechanical invariant.

**Inheritance decision (load-bearing).** `ToolArgumentError` inherits directly from `Exception`, NOT from `ComposerServiceError`. Rationale: `ToolArgumentError` is a handler-internal signal caught by the compose loop, not a service-level error that ever legitimately exits the service. Making it a `ComposerServiceError` subclass would cause `routes.py:390`'s `except ComposerServiceError` block to silently absorb any escaped `ToolArgumentError` as a 502 — recreating the laundering pattern this plan eliminates, one layer up. By inheriting from `Exception` directly, an escaped `ToolArgumentError` (which would indicate a compose-loop bug) falls through to FastAPI's default handler and surfaces as a loud 500 for investigation rather than being masked.

**Tech Stack:** Python 3.12, pytest + pytest-asyncio, asyncio (`asyncio.to_thread`), ELSPETH web composer layer (L3).

---

## Context Background (read before implementing)

**Problem.** `src/elspeth/web/composer/service.py:397` catches `(TypeError, ValueError, UnicodeError)` around `execute_tool()`. This was correct in intent (LLM-argument errors should retry) but wrong in scope: any plugin bug that raises one of those classes gets laundered as an LLM-argument error, polluting the audit trail and causing the LLM to "self-correct" when the real fault is in our code. Per `CLAUDE.md`, plugin bugs MUST crash ("a defective plugin that silently produces wrong results is worse than a crash"). Today's two deliberate upward raises in `tools.py` (lines 1753 and 1838) use the same `TypeError` class that any buggy internal code might also raise — giving us no mechanical way to distinguish "the LLM sent us garbage" from "we have a bug."

**Fix.** Convert the two deliberate raises to a new `ToolArgumentError(Exception)` (see the "Inheritance decision" paragraph above for why `Exception` and not `ComposerServiceError`). Narrow the service catch. Add direct handler unit tests (the existing service-level tests use `side_effect=TypeError` on a mocked `execute_tool`, which never exercises the real raise crossing `asyncio.to_thread` — a sleepy-assertion anti-pattern the review panels flagged).

**Why not Option 2 (pre-dispatch schema validation).** Three of four reviewers flagged that Option 2 creates two sources of truth for argument constraints (schema + handler manual validation), risks becoming a dumping ground, and does not eliminate the post-dispatch type ambiguity. Deferred to a separate issue as optional defense-in-depth.

**Why Task 7 (CI enforcement) is required, not optional.** The architecture, systems, and QA panels converged on Task 7 being load-bearing. Option 1 without the CI gate is "Shifting the Burden" — a future handler author using the natural Python idiom (`raise ValueError(...)` for a wrong-typed argument) silently reopens the channel this plan closes. The convention in protocol.py's docstring is not a mechanical invariant; only the AST-walking gate is. Task 7 ships in the same PR as Tasks 1–6.

---

## File Structure

### Modified files

- `src/elspeth/web/composer/protocol.py` — add `ToolArgumentError` class.
- `src/elspeth/web/composer/tools.py` — convert two deliberate `raise TypeError` sites to `raise ToolArgumentError`.
- `src/elspeth/web/composer/service.py` — narrow catch at line 397; rewrite the explanatory comment block at lines 375–382.
- `tests/unit/web/composer/test_service.py` — update `test_wrong_type_tool_arg_returns_error` and `test_value_error_returns_error_to_llm`; rename `TestValueErrorFromToolExecution`; add plugin-crash assertions.
- `tests/unit/web/composer/test_tools.py` — add direct handler tests for the two converted raise sites.

### Created files (Task 7, required)

- `scripts/cicd/enforce_composer_exception_channel.py` — AST-walks `src/elspeth/web/composer/tools.py` and fails the build on bare `raise TypeError|ValueError|UnicodeError` unless allowlisted.
- `config/cicd/enforce_composer_exception_channel/_defaults.yaml` — allowlist scaffold.
- `tests/unit/scripts/cicd/test_enforce_composer_exception_channel.py` — unit tests for the enforcer. (NOTE: `tests/unit/scripts/cicd/` is the correct path per the existing convention — all eight existing CI-enforcer tests live there, e.g. `test_enforce_plugin_hashes.py`, `test_enforce_freeze_guards.py`. `tests/unit/cicd/` does NOT exist.)

### Files NOT touched

- `src/elspeth/web/composer/state.py`, `yaml_generator.py`, `redaction.py`, `skills/` — unrelated.
- `src/elspeth/contracts/` — `ToolArgumentError` is a composer-domain concern, not an L0 cross-cutting type (Python-engineering reviewer's verdict).
- `src/elspeth/web/composer/service.py:633` — the `raise TypeError` in `_pydantic_default()` (JSON serializer fallback) is intentionally OUT OF SCOPE for Task 7's CI gate. It is an internal-invariant violation on Tier 1 data (our own serialization failure), not a Tier-3 LLM-supplied argument error. The gate scopes to `tools.py` only by design.

---

## Implementation Tasks

### Task 1: Add `ToolArgumentError` exception class

**Files:**
- Modify: `src/elspeth/web/composer/protocol.py`
- Test: `tests/unit/web/composer/test_service.py` (new class `TestToolArgumentError`)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/web/composer/test_service.py` (place near existing exception-type imports at the top — scan for the `from elspeth.web.composer.protocol import ...` line and add `ToolArgumentError` to it):

```python
class TestToolArgumentError:
    """ToolArgumentError is a composer-domain exception for Tier-3 boundary failures.

    It signals that a tool handler received arguments of the wrong type or
    with semantically invalid values that could not be coerced. The compose
    loop catches this and feeds the message back to the LLM for retry. Any
    OTHER exception escaping execute_tool is a plugin bug and must crash.
    """

    def test_inherits_from_exception_directly_not_composer_service_error(self) -> None:
        """ToolArgumentError must NOT inherit from ComposerServiceError.

        If it did, the route-level `except ComposerServiceError` block at
        routes.py:390 would silently absorb any escaped ToolArgumentError
        as a 502, recreating the silent-laundering channel this plan closes.
        Inheriting from Exception directly ensures an escaped
        ToolArgumentError (a compose-loop bug) surfaces loudly via FastAPI's
        default handler rather than being masked.
        """
        from elspeth.web.composer.protocol import ComposerServiceError, ToolArgumentError

        assert issubclass(ToolArgumentError, Exception)
        assert not issubclass(ToolArgumentError, ComposerServiceError)

    def test_message_preserved(self) -> None:
        """Constructor accepts a message string; str() returns it unchanged."""
        from elspeth.web.composer.protocol import ToolArgumentError

        exc = ToolArgumentError("content must be a string, got int")
        assert str(exc) == "content must be a string, got int"

    def test_supports_exception_chaining(self) -> None:
        """raise ToolArgumentError(...) from exc must preserve __cause__.

        Audit-grade error reporting depends on the cause chain surviving
        asyncio.to_thread re-raise and the service-level catch.
        """
        from elspeth.web.composer.protocol import ToolArgumentError

        original = ValueError("bad input")
        try:
            try:
                raise original
            except ValueError as exc:
                raise ToolArgumentError("wrapped") from exc
        except ToolArgumentError as wrapped:
            assert wrapped.__cause__ is original
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_service.py::TestToolArgumentError -v`
Expected: FAIL with `ImportError: cannot import name 'ToolArgumentError'` or similar.

- [ ] **Step 3: Add the exception class**

Edit `src/elspeth/web/composer/protocol.py`. Locate the existing `ComposerConvergenceError` class (should end around line 57). Append this class definition immediately after `ComposerConvergenceError`:

```python
class ToolArgumentError(Exception):
    """Raised by a tool handler when LLM-supplied arguments are unusable.

    Signals a Tier-3 boundary failure: the LLM provided arguments of the
    wrong type, or semantically invalid values that the handler cannot
    coerce. The compose loop catches this exception and returns the
    message to the LLM as a tool error so it can retry.

    This is the ONLY exception class the compose loop catches around
    execute_tool(). Any other TypeError/ValueError/UnicodeError/KeyError
    escaping a tool handler is a plugin bug and MUST crash — per
    CLAUDE.md, plugin bugs that silently produce wrong results are worse
    than a crash because they pollute the audit trail with confidently
    wrong data.

    Inheritance rationale: this class inherits from ``Exception`` directly,
    NOT from ``ComposerServiceError``. A handler-internal signal caught by
    the compose loop must not be absorbed by the route-level
    ``except ComposerServiceError`` block (routes.py:390/505), which would
    silently convert an escaped ToolArgumentError into a 502 — recreating
    the laundering pattern the compose-loop narrowing is designed to
    eliminate. If a ToolArgumentError ever escapes ``_compose_loop``, that
    is a compose-loop bug: FastAPI's default handler will surface it as
    an unstructured 500 for investigation, which is the correct failure
    mode for an invariant violation.

    Handlers that wrap an underlying exception should use::

        raise ToolArgumentError("descriptive message") from exc

    so the cause chain survives ``asyncio.to_thread`` re-raise for audit.
    """
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_service.py::TestToolArgumentError -v`
Expected: all three tests PASS.

- [ ] **Step 5: Verify type checks and lint**

Run: `.venv/bin/python -m mypy src/elspeth/web/composer/protocol.py`
Run: `.venv/bin/python -m ruff check src/elspeth/web/composer/protocol.py tests/unit/web/composer/test_service.py`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/protocol.py tests/unit/web/composer/test_service.py
git commit -m "feat(web/composer): add ToolArgumentError for Tier-3 boundary signalling

Introduces a dedicated exception class so tool handlers can distinguish
LLM-argument failures (retry-able) from plugin bugs (crash). First step
toward narrowing the overly-wide (TypeError, ValueError, UnicodeError)
catch in _compose_loop(). Refs elspeth-7a26880c65."
```

---

### Task 1.5: Pre-implementation audit — enumerate ALL raise sites in `tools.py`

**Rationale.** Task 7's CI gate scope is defined by Task 5's sweep findings. Running the audit AFTER Tasks 2, 3 means the gate is written before the scope is fully known. Running it FIRST gives us a complete worklist before any raise-site conversions begin, and ensures the allowlist in Task 7 has no surprise entries added mid-implementation.

- [ ] **Step 1: Run the AST grep from Task 5 Step 1**

Run (from repo root):

```bash
.venv/bin/python -c "
import ast
src = open('src/elspeth/web/composer/tools.py').read()
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.Raise) and node.exc is not None:
        if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
            if node.exc.func.id in {'TypeError', 'ValueError', 'UnicodeError', 'UnicodeDecodeError', 'UnicodeEncodeError'}:
                print(f'{node.lineno}: raise {node.exc.func.id}')
        elif isinstance(node.exc, ast.Name):
            if node.exc.id in {'TypeError', 'ValueError', 'UnicodeError'}:
                print(f'{node.lineno}: bare raise {node.exc.id}')
"
```

- [ ] **Step 2: Record the findings as a triaged worklist artifact**

Write the triaged worklist to `docs/superpowers/plans/2026-04-17-composer-tool-argument-error.audit.md` (same directory as this plan, adjacent to the `.review.json`). This file is the explicit input dependency for Task 7's allowlist scaffold (Step 4) — Task 7 reads this artifact and refuses to proceed if it is absent.

Required file structure:

```markdown
# Composer tools.py raise-site audit

Ran by: Task 1.5 Step 1 AST grep (date: YYYY-MM-DD).

## Findings (classify each)

| Line | Class | Classification | Notes |
|------|-------|----------------|-------|
| 1753 | TypeError | Convert | Tier-3 content-type guard for create_blob; Task 2 converts |
| 1838 | TypeError | Convert | Tier-3 content-type guard for update_blob; Task 3 converts |
| ...  | ...       | ...            | ... |

## Classifications

- **Convert** — deliberate Tier-3 guard raising the wrong class. Target for Task 2/3/5 conversion.
- **Internal invariant** — plugin-bug signal that SHOULD propagate. Leave as-is; narrowed catch lets it through.
- **Locally contained** — inside `try/except ... return _failure_result(...)`. Leave as-is; never escapes handler.

## Task 7 allowlist seeds

Every row classified as "Internal invariant" becomes an allowlist entry in
`config/cicd/enforce_composer_exception_channel/_defaults.yaml`. Every row
classified as "Convert" becomes a Task 2/3/5 conversion item. Every row
classified as "Locally contained" requires no action but should be documented
here for the next audit.
```

Commit this artifact in the same PR (it is a tracked input to the CI gate, not a scratch file).

Expected today (pre-conversion): lines 1753 and 1838 are the two deliberate Tier-3 guards (Task 2 and Task 3 targets); no "Internal invariant" rows → empty Task 7 allowlist. If the grep shows additional lines beyond these two, the audit artifact will reflect them and Task 7's allowlist seeding (Step 4) will pick them up mechanically.

- [ ] **Step 3: Also scan for implicit Tier-3 coercions**

Grep for `int(arguments[`, `float(arguments[`, `tuple(arguments[`, `list(arguments[` in handler bodies. Any such coercion that can raise on bad LLM input is a latent Tier-3 laundering site — add to the Task 5 worklist for wrapping.

This task produces no commits. It is a planning gate: do NOT begin Task 2 until the worklist is recorded.

---

### Task 2: Convert `_execute_create_blob` type guard to `ToolArgumentError`

**Files:**
- Modify: `src/elspeth/web/composer/tools.py:1753`
- Test: `tests/unit/web/composer/test_tools.py` (new class `TestCreateBlobTypeGuard`)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/web/composer/test_tools.py`. The file's real fixture conventions (verified from the top of the file and from `TestDeleteBlobActiveRunGuard` at `test_tools.py:2441`):

- `_empty_state()` — module-level helper at `test_tools.py:37` (NOT `_empty_composition_state`).
- `_mock_catalog()` — module-level helper at `test_tools.py:48`.
- Session engine setup — there is NO module-level helper; the existing pattern is an `autouse` class fixture that calls `create_session_engine("sqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False})`, runs migrations, and seeds a `sessions` row (required by FK). Mirror `TestDeleteBlobActiveRunGuard._setup` at lines 2448–2487.

```python
class TestCreateBlobTypeGuard:
    """The Tier-3 content-type guard must raise ToolArgumentError, not TypeError.

    Mocked service-level tests (test_wrong_type_tool_arg_returns_error in
    test_service.py) patch execute_tool at the seam and cannot prove the
    real handler raises the right class. This test drives the handler
    end-to-end through execute_tool() dispatch.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.schema import initialize_session_schema
        from elspeth.web.sessions.models import sessions_table

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = str(uuid4())
        self.data_dir = tmp_path
        now = datetime.now(UTC)
        with self.engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=now,
                    updated_at=now,
                )
            )

    def test_non_string_content_raises_tool_argument_error(self) -> None:
        from elspeth.web.composer.protocol import ToolArgumentError
        from elspeth.web.composer.tools import execute_tool

        catalog = _mock_catalog()
        state = _empty_state()

        with pytest.raises(ToolArgumentError, match="content must be a string, got int"):
            execute_tool(
                "create_blob",
                {
                    "filename": "notes.txt",
                    "mime_type": "text/plain",
                    "content": 42,  # wrong type — LLM sent int where str required
                },
                state,
                catalog,
                data_dir=str(self.data_dir),
                session_engine=self.engine,
                session_id=self.session_id,
            )
```

Verify before writing: open `test_tools.py:2441` (`TestDeleteBlobActiveRunGuard`) and confirm the imports, the `sessions_table` insert columns (`auth_provider_type="local"` is required — the schema may add fields over time), and that `create_session_engine` / `initialize_session_schema` / `sessions_table` are the correct symbol names in this tree. If any have drifted, update the snippet — do NOT invent replacements.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_tools.py::TestCreateBlobTypeGuard -v`
Expected: FAIL — the handler currently raises `TypeError`, not `ToolArgumentError`, so `pytest.raises(ToolArgumentError)` fails.

- [ ] **Step 3: Convert the raise site**

Edit `src/elspeth/web/composer/tools.py` at lines 1749–1753. Replace:

```python
    # Tier 3 boundary: LLM can pass wrong types (e.g. int for content).
    # Validate here so .encode() doesn't raise AttributeError, which is
    # ambiguous (could also mean an internal bug).
    if not isinstance(content, str):
        raise TypeError(f"content must be a string, got {type(content).__name__}")
```

with:

```python
    # Tier 3 boundary: LLM can pass wrong types (e.g. int for content).
    # Validate here so .encode() doesn't raise AttributeError, which is
    # ambiguous (could also mean an internal bug). Raise ToolArgumentError
    # (not TypeError) so the compose loop can distinguish this LLM-side
    # error from plugin-internal type errors — see protocol.ToolArgumentError.
    #
    # IMPORTANT: this guard MUST remain BEFORE the `try: with session_engine.begin()`
    # block below. The `except Exception: ... raise` cleanup guard inside that
    # block catches any exception including ToolArgumentError; if this guard
    # moved inside the try, the cleanup code would run on pure argument
    # validation failures (no file has been written at that point — cleanup
    # is a no-op but semantically wrong).
    if not isinstance(content, str):
        raise ToolArgumentError(f"content must be a string, got {type(content).__name__}")
```

Then add the import near the existing imports at the top of `tools.py`. Find the line `from elspeth.web.composer.redaction import redact_source_storage_path` (around line 25) and add, alongside the other composer-package imports:

```python
from elspeth.web.composer.protocol import ToolArgumentError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_tools.py::TestCreateBlobTypeGuard -v`
Expected: PASS.

- [ ] **Step 5: Run surrounding tests to check no regressions**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_tools.py -v -k "create_blob"`
Expected: all PASS (existing create_blob tests unaffected — they pass valid string content).

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/tools.py tests/unit/web/composer/test_tools.py
git commit -m "refactor(web/composer): raise ToolArgumentError from create_blob type guard

Converts the deliberate Tier-3 type-guard raise in _execute_create_blob
from the ambiguous TypeError to the new ToolArgumentError, so the
compose loop can distinguish LLM-argument failures from plugin bugs.
Adds direct handler test (not a service-level mock) proving the raise
crosses asyncio.to_thread correctly. Refs elspeth-7a26880c65."
```

---

### Task 3: Convert `_execute_update_blob` type guard to `ToolArgumentError`

**Files:**
- Modify: `src/elspeth/web/composer/tools.py:1838`
- Test: `tests/unit/web/composer/test_tools.py` (new class `TestUpdateBlobTypeGuard`)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/web/composer/test_tools.py` alongside `TestCreateBlobTypeGuard`. Reuse the same autouse fixture pattern (copy, do not factor out — the two classes are intentionally independent so one can move without breaking the other):

```python
class TestUpdateBlobTypeGuard:
    """Parallels TestCreateBlobTypeGuard for _execute_update_blob."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.schema import initialize_session_schema
        from elspeth.web.sessions.models import sessions_table

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = str(uuid4())
        self.data_dir = tmp_path
        now = datetime.now(UTC)
        with self.engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=now,
                    updated_at=now,
                )
            )

    def test_non_string_content_raises_tool_argument_error(self) -> None:
        from elspeth.web.composer.protocol import ToolArgumentError
        from elspeth.web.composer.tools import execute_tool

        catalog = _mock_catalog()
        state = _empty_state()

        # Seed a real blob so the handler reaches the content guard before
        # the "blob not found" check.  Use the create path end-to-end.
        create_result = execute_tool(
            "create_blob",
            {"filename": "a.txt", "mime_type": "text/plain", "content": "initial"},
            state,
            catalog,
            data_dir=str(self.data_dir),
            session_engine=self.engine,
            session_id=self.session_id,
        )
        blob_id = create_result.data["blob_id"]  # confirm attribute name from existing tests
        state = create_result.updated_state

        with pytest.raises(ToolArgumentError, match="content must be a string, got int"):
            execute_tool(
                "update_blob",
                {"blob_id": blob_id, "content": 42},
                state,
                catalog,
                data_dir=str(self.data_dir),
                session_engine=self.engine,
                session_id=self.session_id,
            )
```

Confirm `create_result.data["blob_id"]` and `create_result.updated_state` against `ToolResult` — open its definition in `tools.py` (grep `class ToolResult`) and verify field names. If the real attribute is `.result` or `.state`, use the correct name — do not guess.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_tools.py::TestUpdateBlobTypeGuard -v`
Expected: FAIL on `pytest.raises(ToolArgumentError)`.

- [ ] **Step 3: Convert the raise site**

Edit `src/elspeth/web/composer/tools.py` at lines 1836–1838. Replace:

```python
    # Tier 3 boundary: LLM can pass wrong types (e.g. int for content).
    if not isinstance(content, str):
        raise TypeError(f"content must be a string, got {type(content).__name__}")
```

with:

```python
    # Tier 3 boundary: LLM can pass wrong types (e.g. int for content).
    # ToolArgumentError (not TypeError) so the compose loop can distinguish
    # this LLM-side error from plugin-internal type errors.
    #
    # IMPORTANT: this guard MUST remain BEFORE the `try: with session_engine.begin()`
    # cleanup block below (same rationale as _execute_create_blob's guard).
    if not isinstance(content, str):
        raise ToolArgumentError(f"content must be a string, got {type(content).__name__}")
```

(The `ToolArgumentError` import was already added in Task 2.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_tools.py::TestUpdateBlobTypeGuard -v`
Expected: PASS.

- [ ] **Step 5: Run full blob test scope**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_tools.py -v -k "blob"`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/tools.py tests/unit/web/composer/test_tools.py
git commit -m "refactor(web/composer): raise ToolArgumentError from update_blob type guard

Mirrors the create_blob conversion for _execute_update_blob so both
deliberate Tier-3 raises use the typed exception. Refs elspeth-7a26880c65."
```

---

### Task 4: Narrow the service-level catch to `ToolArgumentError`

**Files:**
- Modify: `src/elspeth/web/composer/service.py:375-411`
- Test: `tests/unit/web/composer/test_service.py` — update `test_wrong_type_tool_arg_returns_error` (L564), rework `TestValueErrorFromToolExecution` (L1159), add new plugin-crash tests.

- [ ] **Step 1: Update `test_wrong_type_tool_arg_returns_error` to assert the new contract**

Edit `tests/unit/web/composer/test_service.py` around line 591–592. Change the side_effect from `TypeError` to `ToolArgumentError`:

```python
        with (
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=ToolArgumentError("content must be a string, got int"),
            ),
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        ):
```

Ensure `ToolArgumentError` is imported at the top of the file (either via the existing `from elspeth.web.composer.protocol import ...` line or a new import). Update the docstring of `test_wrong_type_tool_arg_returns_error` (lines 564–570) to reference `ToolArgumentError` instead of `TypeError`:

```python
        """ToolArgumentError from Tier 3 type guard in tool handler is caught, not crash.

        Tool handlers validate LLM-provided argument types at the Tier 3
        boundary, raising ToolArgumentError for wrong types (e.g. int where
        str expected). The compose loop catches this typed exception and
        feeds the error back to the LLM so it can retry with a corrected
        argument.
        """
```

- [ ] **Step 2: Rework `TestValueErrorFromToolExecution` — ValueError now CRASHES**

Edit `tests/unit/web/composer/test_service.py` at line 1159. Rename the class and flip the assertion to mirror `test_internal_key_error_is_not_swallowed`:

Replace the entire existing class (lines 1159–1211) with:

```python
class TestPluginBugCrashesFromToolExecution:
    """Plugin-internal TypeError/ValueError/UnicodeError must crash.

    The compose loop catches ONLY ToolArgumentError around execute_tool.
    Any other TypeError/ValueError/UnicodeError is a plugin bug — per
    CLAUDE.md, silently laundering a plugin bug as an LLM-argument error
    is worse than crashing, because the audit trail records a confident
    but wrong Tier-3 story.

    Mirrors test_internal_key_error_is_not_swallowed.
    """

    @pytest.mark.asyncio
    async def test_plugin_value_error_is_not_swallowed(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=ValueError("invalid expression syntax — plugin bug"),
            ),
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(ValueError, match="plugin bug"):
                await service.compose("Setup", [], state)

    @pytest.mark.asyncio
    async def test_plugin_type_error_is_not_swallowed(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=TypeError("NoneType + int — plugin bug"),
            ),
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(TypeError, match="plugin bug"):
                await service.compose("Setup", [], state)

    @pytest.mark.asyncio
    async def test_plugin_unicode_error_is_not_swallowed(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "plugin bug"),
            ),
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(UnicodeDecodeError):
                await service.compose("Setup", [], state)

    @pytest.mark.asyncio
    async def test_tool_argument_error_is_caught_and_fed_to_llm(self) -> None:
        """Positive case: ToolArgumentError IS caught, error fed back for LLM retry."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )
        text = _make_llm_response(content="Got it, trying again.")

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=ToolArgumentError("plugin expected str, got int"),
            ),
        ):
            mock_llm.side_effect = [valid_call, text]
            result = await service.compose("Setup", [], state)

        assert isinstance(result, ComposerResult)
        second_call_messages = mock_llm.call_args_list[1].args[0]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        error_payload = json.loads(tool_messages[0]["content"])
        assert "plugin expected str, got int" in error_payload["error"]
```

Ensure `ToolArgumentError` and `ComposerResult` are imported at the top of the file.

- [ ] **Step 3: Run the tests and verify they fail on the new contract**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_service.py::TestPluginBugCrashesFromToolExecution -v`
Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_service.py -k test_wrong_type_tool_arg_returns_error -v`

Expected: **ALL FIVE TESTS FAIL**. The reasoning (uniform across all five):

`ToolArgumentError` inherits directly from `Exception` (see Task 1 Step 3 and the "Inheritance decision" paragraph in Architecture). It is NOT a subclass of `TypeError`, `ValueError`, or `UnicodeError`. The existing wide catch at `service.py:397` — `except (TypeError, ValueError, UnicodeError)` — matches only those three classes. So:

- `test_wrong_type_tool_arg_returns_error` — FAILS: the mock now raises `ToolArgumentError`; the wide catch does NOT match; the exception propagates out of `_compose_loop`; `compose()` re-raises; the test's "error is fed to LLM" assertions are never reached.
- `test_plugin_value_error_is_not_swallowed` — FAILS: the wide catch still swallows `ValueError` and feeds it to the LLM; `pytest.raises(ValueError)` never sees it.
- `test_plugin_type_error_is_not_swallowed` — FAILS: same reason, for `TypeError`.
- `test_plugin_unicode_error_is_not_swallowed` — FAILS: same reason. `UnicodeDecodeError` is a subclass of `UnicodeError`, so the wide catch swallows it.
- `test_tool_argument_error_is_caught_and_fed_to_llm` — FAILS: the wide catch does not match `ToolArgumentError`; `compose()` re-raises; the test's retry-loop assertions are never reached.

All five tests must be RED before proceeding to Step 4. If any are GREEN, stop and investigate — the test is likely mis-wired.

- [ ] **Step 4: Narrow the service-level catch**

Edit `src/elspeth/web/composer/service.py`. Add the import for `ToolArgumentError` — update the existing import block around line 29 from:

```python
from elspeth.web.composer.protocol import (
    ComposerConvergenceError,
    ComposerResult,
    ComposerServiceError,
    ComposerSettings,
)
```

to:

```python
from elspeth.web.composer.protocol import (
    ComposerConvergenceError,
    ComposerResult,
    ComposerServiceError,
    ComposerSettings,
    ToolArgumentError,
)
```

Replace the comment block and catch clause at lines 375–411. Current text (for reference — do NOT leave both versions in the file):

```python
                # TypeError/ValueError are caught because the LLM can
                # provide wrong value types (e.g. string where list
                # expected → tuple() fails) or semantically invalid values
                # (e.g. invalid expression syntax).  UnicodeError covers
                # encoding failures on malformed string data from the LLM.
                # KeyError and AttributeError are NOT caught — after
                # required-arg validation above and Tier 3 type guards
                # in tool handlers, either would be an internal bug.
                try:
                    result = await asyncio.to_thread(
                        execute_tool,
                        ...
                    )
                except (TypeError, ValueError, UnicodeError) as exc:
                    if not is_discovery_tool(tool_name):
                        turn_has_mutation = True
                    llm_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(
                                {
                                    "error": f"Tool '{tool_name}' failed: {exc}",
                                }
                            ),
                        }
                    )
                    continue
```

New text:

```python
                # Tool handlers raise ToolArgumentError at Tier-3 boundaries
                # (LLM supplied wrong types, semantically invalid values,
                # or malformed encodings that cannot be coerced).  The
                # compose loop catches ONLY that class and feeds the error
                # back to the LLM for retry.
                #
                # Any other exception — TypeError, ValueError, UnicodeError,
                # KeyError, AttributeError — escaping execute_tool() is a
                # plugin bug (Tier 1/2) and MUST crash.  Per CLAUDE.md,
                # silently laundering a plugin bug as an LLM-argument error
                # is worse than crashing: it pollutes the audit trail with
                # a confident but wrong Tier-3 story, and the LLM's "retry"
                # cannot correct a fault in our own code.
                #
                # Cancel-safety: tool calls are NOT wrapped in
                # asyncio.wait_for — they always run to completion.
                # The cooperative deadline is checked BETWEEN operations
                # (before LLM calls, after tool batches), so side effects
                # and state publication are never split.  LLM calls use
                # per-call wait_for because they are pure network I/O
                # with no side effects.
                try:
                    result = await asyncio.to_thread(
                        execute_tool,
                        tool_name,
                        arguments,
                        state,
                        self._catalog,
                        data_dir=self._data_dir,
                        session_engine=self._session_engine,
                        session_id=session_id,
                        secret_service=self._secret_service,
                        user_id=user_id,
                        prior_validation=last_validation,
                    )
                except ToolArgumentError as exc:
                    if not is_discovery_tool(tool_name):
                        turn_has_mutation = True
                    # Trust-boundary redaction: the echoed message reaches the
                    # LLM API and (via audit) the Landscape. Use exc.args[0]
                    # rather than str(exc) so a future subclass that overrides
                    # __str__ to include __cause__ context (which may carry DB
                    # URLs, filesystem paths, or secret fragments from deeper
                    # layers) cannot leak through this path. Handlers that
                    # use `raise ToolArgumentError(msg) from exc` get the
                    # cause preserved on __cause__ for debug/audit but NOT
                    # echoed to the LLM.
                    safe_message = exc.args[0] if exc.args else "tool argument error"
                    llm_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(
                                {
                                    "error": f"Tool '{tool_name}' failed: {safe_message}",
                                }
                            ),
                        }
                    )
                    continue
```

Note the surviving portion of the original docstring (the `asyncio.to_thread` / cancel-safety paragraph at lines 356–373) must remain above this block unchanged — preserve it. Only the TypeError/ValueError paragraph and the `except` line change.

- [ ] **Step 4a-pre: Add `slog` to `service.py` imports (REQUIRED before Step 4a code)**

`service.py` currently does NOT import structlog. The Step 4a `_persist_crashed_session` fallback calls `slog.error(...)`. Without the import, the crash-persistence failure path throws `NameError` at runtime — masking the original plugin bug with a secondary failure, exactly the anti-pattern this plan exists to prevent.

Verify the current state:

```bash
.venv/bin/python -c "import ast; src = open('src/elspeth/web/composer/service.py').read(); tree = ast.parse(src); print([n.names[0].name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) and n.names and 'structlog' in n.names[0].name])"
```

If the output is empty, add to the top-of-file imports in `src/elspeth/web/composer/service.py` (match the existing import-group style; structlog is a third-party import, group accordingly):

```python
import structlog

slog = structlog.get_logger()
```

Mirrors the pattern at `src/elspeth/web/sessions/routes.py:49`. The module-level binding is used by both the Step 4a `_persist_crashed_session` fallback and the Step 4a outer `except Exception` handler.

Rationale per `logging-telemetry-policy`: crash-persistence failure is an "audit system failure" — one of the three permitted slog channels. The Landscape cannot record the crash because the persistence path itself failed; slog is the last-resort diagnostic channel to ensure the operator has something to investigate when reviewing stderr.

- [ ] **Step 4a: Persist last-known state on plugin crash before re-raising**

The existing code captures `partial_state` inside `ComposerConvergenceError` (see `service.py:467,476,560,572`); the route handler's `_handle_convergence_error` persists that state via the sessions service. The narrowed catch at Step 4 unmasks plugin `TypeError`/`ValueError`/`UnicodeError` — those now escape `_compose_loop` without persisting any session-row breadcrumb, violating ELSPETH's "no silent drops" principle for session records.

Wrap `_compose_loop` so that any exception OTHER than `ComposerConvergenceError` (which has its own persistence path via the route handler) attempts a best-effort persist of the last known state as a `crashed` marker before re-raising. Do NOT wrap the exception in a `ComposerServiceError` subclass — that would get caught by the route's `except ComposerServiceError` at `routes.py:390/505` and re-converted to a 502, reintroducing the silent-laundering behaviour the plan is trying to eliminate.

Sketch (adapt to the real `_compose_loop` signature):

```python
async def _compose_loop(self, message, messages, state, session_id, user_id, deadline):
    try:
        # ... existing loop body unchanged ...
        pass
    except ComposerConvergenceError:
        # Has its own partial_state; route handler persists. Do not intercept.
        raise
    except Exception as exc:
        # Plugin-bug crash path. Bump the session's updated_at as an audit
        # breadcrumb (richer crash-marker columns tracked as a follow-up
        # migration), then re-raise the ORIGINAL exception class unchanged
        # so the route layer's new final handler (Task 4.5) sees a bare
        # TypeError/ValueError/etc.
        #
        # Note: last-known-state tracking is intentionally NOT threaded
        # through the loop for now — the current schema has nowhere to
        # persist it. The follow-up schema migration will add the columns
        # and re-introduce the tracking at that point.
        if self._session_engine is not None and session_id is not None:
            try:
                self._persist_crashed_session(session_id)
            except Exception:
                # Audit-persistence is best-effort on the crash path —
                # failure to persist MUST NOT mask the original plugin bug.
                # Log via slog.error (audit system itself is failing here,
                # which is one of the three permitted slog use cases per
                # the logging-telemetry-policy skill).
                slog.error(
                    "composer_crash_persistence_failed",
                    session_id=session_id,
                    original_exc_class=type(exc).__name__,
                    exc_info=True,
                )
        raise
```

`_persist_crashed_session` is a new private method on `ComposerServiceImpl`. Its contract is deliberately minimal given the current schema:

**Schema reality (verified by plan reviewers).** `sessions_table` has columns `id`, `user_id`, `auth_provider_type`, `title`, `created_at`, `updated_at`, `forked_from_session_id`, `forked_from_message_id`. There is NO `status`, `crashed_at`, or `last_exc_class` column on the sessions table. The only tables with a `status` column are `runs_table` and `blobs_table` — neither applicable here. This plan prohibits introducing a migration just for the crash marker.

**Resolution.** Scope `_persist_crashed_session` to bump `updated_at` only — a timestamped "last touched" breadcrumb. This is weaker than a true crash marker but is what the current schema supports without a migration. The `exc_class` is NOT written to any column (the schema has nowhere to put it); it is emitted via `slog.error` at the call site (see Step 4a's outer handler) so the operator reviewing stderr can correlate the `updated_at` bump with the crash.

The method must:

1. Never raise under any internal failure mode caught within itself (see the outer try/except above — it is the last line of defence; a raise here masks the plugin bug).
2. Write ONLY `updated_at` — NEVER the exception message or any other exception-derived content (no schema column exists to hold it, and messages may carry secrets from deeper layers via `__cause__` chains).
3. Use direct SQLAlchemy Core with the imported `update` construct (matching the codebase convention at `src/elspeth/web/sessions/service.py:271`) to avoid any coupling to the sessions service layer's transaction semantics on the crash path.

Implementation sketch:

```python
def _persist_crashed_session(self, session_id: str) -> None:
    """Best-effort timestamp bump to mark that a compose session crashed.

    NOTE: The sessions-table schema does not yet have a dedicated crash
    marker column. Bumping updated_at is the minimum viable breadcrumb
    until a migration adds (e.g.) a `status` or `crashed_at` column.
    A follow-up issue tracks the schema addition; do NOT introduce the
    migration as part of this PR (scope creep).

    The crash's exc_class is NOT written to the session row — no column
    exists to hold it. The operator correlates the updated_at bump with
    the crash via the slog.error emission at the call site, which
    includes session_id and exc_class in structured fields.

    Signature intentionally minimal — only the data that actually gets
    persisted is accepted. When the schema migration lands, this method's
    signature expands to take last_state and exc_class, and callers are
    updated at that point. Today, the caller passes session_id and logs
    the rest via slog.

    MUST never raise; caller's outer try/except absorbs any failure.
    """
    from datetime import UTC, datetime

    from sqlalchemy import update

    from elspeth.web.sessions.models import sessions_table

    now = datetime.now(UTC)
    with self._session_engine.begin() as conn:
        conn.execute(
            update(sessions_table)
            .where(sessions_table.c.id == session_id)
            .values(updated_at=now)
        )
```

**Caller update.** The outer `except Exception` in `_compose_loop` (Step 4a sketch) must drop `last_state=last_published_state, exc_class=type(exc).__name__` from the positional args — only `session_id=session_id` remains. The slog.error emission there ALREADY emits `session_id` and `original_exc_class=type(exc).__name__`, so no audit information is lost; it just lives in stderr (via slog) rather than being uselessly passed to a method that cannot store it.

**Follow-up issue (create in Task 8).** File a P3 task to add a `sessions_table.status` column (`pending|active|crashed|completed|abandoned`) plus `crashed_at` and `last_exc_class` columns, with an Alembic migration. When that lands, `_persist_crashed_session` is revised to populate those columns and remove the `_ = (last_state, exc_class)` discard.

Add a test that asserts the session-row side effect. Wrap in a dedicated test class with its own autouse fixture (mirrors `TestToolArgumentErrorAcrossThreadBoundary`):

```python
class TestPluginCrashSessionPersistence:
    """Plugin-bug crash must leave a durable session-row breadcrumb.

    "No silent drops" for session records: a plugin crash that leaves
    the session in no recorded terminal state is as bad for audit
    integrity as the laundering behaviour this plan eliminates.

    Given the current sessions_table schema (no status / crashed_at /
    last_exc_class columns), the breadcrumb is a bump of updated_at.
    This test asserts that bump, plus the invariant that NO exception
    message leaks into any column. The follow-up filigree issue tracks
    the schema migration that adds richer crash markers.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.schema import initialize_session_schema
        from elspeth.web.sessions.models import sessions_table

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = str(uuid4())
        self.data_dir = tmp_path
        # Seed the sessions row with a DELIBERATELY OLD updated_at so the
        # crash-path bump is unambiguously distinguishable from the seed.
        self.seeded_at = datetime(2020, 1, 1, tzinfo=UTC)
        with self.engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=self.seeded_at,
                    updated_at=self.seeded_at,
                )
            )

    @pytest.mark.asyncio
    async def test_plugin_crash_bumps_session_updated_at(self) -> None:
        from elspeth.web.sessions.models import sessions_table

        catalog = _mock_catalog()
        settings = _make_settings(data_dir=self.data_dir)
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            session_engine=self.engine,
        )
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=ValueError(
                    "plugin bug: /etc/secrets/bootstrap.key is bad"
                ),
            ),
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(ValueError, match="plugin bug"):
                await service.compose(
                    "Setup", [], state, session_id=self.session_id
                )

        # Assertion 1: session row was touched on the crash path.
        with self.engine.begin() as conn:
            row = conn.execute(
                sessions_table.select().where(
                    sessions_table.c.id == self.session_id
                )
            ).one()

        assert row.updated_at > self.seeded_at, (
            "crash path must bump updated_at as audit breadcrumb"
        )

        # Assertion 2: NO column holds the exception message. Stringify
        # the entire row and verify secret fragments / class hints are
        # absent. This is the load-bearing audit-integrity invariant —
        # if a future refactor adds a 'last_error' column, the assertion
        # will catch any attempt to persist the raw message.
        row_text = " | ".join(str(v) for v in row._mapping.values())
        assert "plugin bug" not in row_text
        assert "/etc/secrets" not in row_text
        assert "ValueError" not in row_text

    @pytest.mark.asyncio
    async def test_persist_crashed_session_failure_does_not_mask_plugin_bug(
        self,
    ) -> None:
        """If _persist_crashed_session itself raises, slog.error fires and
        the original plugin-bug exception still propagates unchanged.

        Two invariants asserted:
        1. The ORIGINAL ValueError reaches the caller (not the RuntimeError
           from the persistence failure).
        2. slog.error is called with the `composer_crash_persistence_failed`
           event — guarantees that an accidental removal of Step 4a-pre's
           structlog import would be caught (without this assertion, a
           regression where slog.error silently fails as NameError would
           pass the test because the original exception still propagates).
        """
        import structlog
        from structlog.testing import capture_logs

        catalog = _mock_catalog()
        settings = _make_settings(data_dir=self.data_dir)
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            session_engine=self.engine,
        )
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=ValueError("original plugin bug"),
            ),
            patch.object(
                service,
                "_persist_crashed_session",
                side_effect=RuntimeError("persistence failed"),
            ),
            capture_logs() as cap_logs,
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(ValueError, match="original plugin bug"):
                await service.compose(
                    "Setup", [], state, session_id=self.session_id
                )

        # The crash-persistence-failure slog.error MUST fire. This closes
        # the regression risk where Step 4a-pre's structlog import is
        # accidentally removed — the method would then raise NameError
        # inside the except, masking the original ValueError.
        persistence_failure_events = [
            entry for entry in cap_logs
            if entry.get("event") == "composer_crash_persistence_failed"
        ]
        assert len(persistence_failure_events) == 1, cap_logs
        event = persistence_failure_events[0]
        assert event["session_id"] == self.session_id
        assert event["original_exc_class"] == "ValueError"
        # Exception message MUST NOT appear in structured fields — only
        # in exc_info (stderr traceback).
        assert "original plugin bug" not in str(event)

    @pytest.mark.asyncio
    async def test_persist_crashed_session_real_path_slog_emission(self) -> None:
        """Smoke test for Step 4a-pre: exercise the real _persist_crashed_session
        path (no patching of the private method).  If structlog is not
        imported in service.py, this test will surface the NameError that
        `test_persist_crashed_session_failure_does_not_mask_plugin_bug`
        misses (because that test patches the method itself).

        The real _persist_crashed_session should succeed here (the sessions
        engine is live), so we assert the crash propagates without any
        persistence-failure slog event.
        """
        from structlog.testing import capture_logs

        catalog = _mock_catalog()
        settings = _make_settings(data_dir=self.data_dir)
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            session_engine=self.engine,
        )
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=ValueError("plugin bug"),
            ),
            capture_logs() as cap_logs,
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(ValueError, match="plugin bug"):
                await service.compose(
                    "Setup", [], state, session_id=self.session_id
                )

        # No persistence-failure event — the real path succeeded.
        persistence_failure_events = [
            entry for entry in cap_logs
            if entry.get("event") == "composer_crash_persistence_failed"
        ]
        assert persistence_failure_events == [], cap_logs
```

- [ ] **Step 4b: Add a leakage regression test for the `safe_message` redaction**

Append to `tests/unit/web/composer/test_service.py` inside `TestPluginBugCrashesFromToolExecution` (or a dedicated class `TestToolArgumentErrorLeakRedaction` if the grouping reads cleaner):

```python
    @pytest.mark.asyncio
    async def test_tool_argument_error_subclass_cannot_leak_cause_to_llm(self) -> None:
        """Defense-in-depth: if a subclass overrides __str__ to embed the
        __cause__ chain, the LLM-echo path must still use args[0] only.

        Simulates a future regression where a helpful-looking subclass does
        `def __str__(self): return f"{self.args[0]}: caused by {self.__cause__}"`.
        A DB URL or file path leaked through __cause__ would then reach the
        LLM API. The compose loop MUST short-circuit __str__ and emit
        args[0] verbatim, isolating the cause chain to __cause__ (audit-only).
        """
        from elspeth.web.composer.protocol import ToolArgumentError

        class LeakyToolArgumentError(ToolArgumentError):
            def __str__(self) -> str:
                return f"{self.args[0]}: caused by {self.__cause__}"

        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        secret_path = "/etc/elspeth/secrets/bootstrap.key"
        secret_cause = ValueError(f"bad path: {secret_path}")
        leaky = LeakyToolArgumentError("content must be a string, got int")
        leaky.__cause__ = secret_cause

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )
        text = _make_llm_response(content="Got it.")

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=leaky,
            ),
        ):
            mock_llm.side_effect = [valid_call, text]
            await service.compose("Setup", [], state)

        second_call_messages = mock_llm.call_args_list[1].args[0]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        error_payload = json.loads(tool_messages[0]["content"])
        assert "content must be a string, got int" in error_payload["error"]
        # The crucial assertion: the cause-chain content NEVER appears.
        assert secret_path not in error_payload["error"]
        assert "caused by" not in error_payload["error"]
```

- [ ] **Step 5: Run all four new/updated service tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_service.py::TestPluginBugCrashesFromToolExecution -v`
Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_service.py -k test_wrong_type_tool_arg_returns_error -v`
Expected: all PASS.

- [ ] **Step 6: Run the full composer test suite**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/ -v`
Expected: full suite PASSES. If any other tests fail, they likely depended on the old wide catch — investigate each one; do not mask failures.

- [ ] **Step 7: Run type check and lint**

Run: `.venv/bin/python -m mypy src/elspeth/web/composer/`
Run: `.venv/bin/python -m ruff check src/elspeth/web/composer/ tests/unit/web/composer/`
Expected: clean.

- [ ] **Step 8: DO NOT commit yet — proceed to Task 4.5**

Task 4 and Task 4.5 MUST land in a single commit. Rationale: Task 4 alone unmasks plugin `TypeError`/`ValueError`/`UnicodeError` at the service boundary. Without Task 4.5's route-level handler already in place, any plugin crash in that intermediate state would hit FastAPI's default exception handler, which serializes the exception message — potentially containing secret fragments from `__cause__`-chained exceptions — directly into the response body. That is exactly the leakage this plan exists to prevent.

Do not run `git add` or `git commit` here. Leave the working tree dirty and continue directly to Task 4.5. The combined commit lands after Task 4.5 Step 7.

---

### Task 4.5: Route-layer response for unmasked plugin crashes

**Files:**
- Modify: `src/elspeth/web/sessions/routes.py` (both `/compose` handler at lines 375–398 and `/recompose` at lines 490–508).
- Test: `tests/unit/web/sessions/test_routes.py` (add route-level assertions; follow the file's existing conventions).

After Task 4 narrows the service catch, plugin `TypeError`/`ValueError`/`UnicodeError` propagates through `compose()` unhandled. The existing route exception ladder (`ComposerConvergenceError` → `LiteLLMAuthError` → `LiteLLMAPIError` → `ComposerServiceError`) catches none of these, so FastAPI's default exception handler returns an unstructured 500 with an arbitrary traceback shape depending on deployment config. That breaks the route's implicit contract with the frontend/CLI, which expects one of four structured error shapes.

Add a final handler BELOW the four existing `except` blocks on each route that:

1. Catches only concrete plugin-crash classes: `(TypeError, ValueError, UnicodeError)` plus `KeyError` / `AttributeError` (the classes already asserted by `test_internal_key_error_is_not_swallowed`). Do NOT use a bare `except Exception` — we want unknown exception classes to still surface as 500s so they're investigated, not absorbed.
2. Logs the crash via `slog.error` with `session_id`, `user_id`, `exc_info=True` (one of the three permitted slog channels per the `logging-telemetry-policy` skill — "audit system failures" applies because the compose session cannot be audited further).
3. Returns HTTP 500 with a structured body `{"error_type": "composer_plugin_error", "detail": "A composer plugin crashed; see server logs for the traceback. This is not a user-retryable error."}` — the exception message is NOT echoed (it may contain secrets from `__cause__`-chained exceptions per the Amendment 7 redaction discipline).

- [ ] **Step 1: Read both route handlers and the existing exception ladder**

Read `src/elspeth/web/sessions/routes.py` around lines 375–398 and 490–508. Confirm the four existing `except` classes match the plan's list. If the ladder has drifted (new exception types added since this plan was written), incorporate those without altering their order — insert the new handler after `ComposerServiceError` in both routes.

- [ ] **Step 2: Write failing tests**

Add tests to `tests/unit/web/sessions/test_routes.py`. **This file uses an inline app-construction pattern — there are NO module-level pytest fixtures for `client` / `mock_composer_service` / `authed_session`.** Every test constructs its own app and client:

```python
app, service = _make_app(tmp_path)
app.state.composer_service = mock_composer  # inject the mock
client = TestClient(app, raise_server_exceptions=False)  # load-bearing for 500 assertions
```

`raise_server_exceptions=False` is critical: without it, `TestClient` re-raises server-side exceptions through the test boundary and the `response` object never exists, so the 500 assertions never run. Existing convergence tests at `test_routes.py:461, 499, 555, 1020` all use this pattern — mirror it.

**Route endpoints:** compose is exposed as `POST /api/sessions/{session_id}/messages` (calls `composer.compose` internally via `send_message`); recompose is `POST /api/sessions/{session_id}/recompose`. There is no `/compose` endpoint — the "compose path" is the message-send endpoint. Update Task 4 Step 4's insertion-site line numbers accordingly if they drift.

```python
class TestComposePluginCrashResponse:
    """Plugin TypeError/ValueError from compose() must produce a structured 500.

    After the Task 4 narrowing, plugin bugs escape the service layer instead
    of being laundered as LLM retries. The route handler MUST shape these
    into a documented response rather than letting FastAPI's default handler
    emit an arbitrary traceback.

    Audit-integrity invariant: exception message content — especially
    fragments from __cause__-chained exceptions that may include DB URLs,
    filesystem paths, or secret material — MUST NOT appear in the response
    body. Only the documented error_type + generic detail string is echoed.
    """

    SECRET_PATH = "/etc/elspeth/secrets/bootstrap.key"

    def test_compose_plugin_value_error_returns_structured_500(self, tmp_path) -> None:
        mock_composer = AsyncMock()
        mock_composer.compose = AsyncMock(
            side_effect=ValueError(f"plugin bug: {self.SECRET_PATH}"),
        )

        app, _ = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Build me a pipeline"},
        )

        assert response.status_code == 500
        body = response.json()
        # FastAPI serializes HTTPException(detail={...}) as {"detail": {...}}.
        assert isinstance(body.get("detail"), dict), body
        assert body["detail"]["error_type"] == "composer_plugin_error"
        assert "user-retryable" in body["detail"]["detail"].lower()

        # Audit-integrity: exception message and cause content MUST NOT leak.
        body_text = response.text
        assert "plugin bug" not in body_text
        assert self.SECRET_PATH not in body_text
        assert "ValueError" not in body_text  # exception class also redacted

    def test_recompose_plugin_type_error_returns_structured_500(self, tmp_path) -> None:
        import asyncio
        import uuid

        mock_composer = AsyncMock()
        mock_composer.compose = AsyncMock(
            side_effect=TypeError(
                f"plugin bug: NoneType has no attribute 'read' from {self.SECRET_PATH}"
            ),
        )

        app, service = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        # Recompose requires a pre-existing trailing user message (see
        # TestRecomposeConvergencePartialState for the template).
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            service.add_message(uuid.UUID(session_id), "user", "Build something")
        )
        loop.close()

        response = client.post(f"/api/sessions/{session_id}/recompose")

        assert response.status_code == 500
        body = response.json()
        assert isinstance(body.get("detail"), dict), body
        assert body["detail"]["error_type"] == "composer_plugin_error"

        body_text = response.text
        assert "plugin bug" not in body_text
        assert self.SECRET_PATH not in body_text
        assert "NoneType" not in body_text
        assert "TypeError" not in body_text

    def test_compose_unknown_exception_class_is_not_absorbed(self, tmp_path) -> None:
        """Deliberately narrow typed catch: RuntimeError (not in the handler's
        catch list) must propagate past the composer_plugin_error handler.
        With raise_server_exceptions=False, TestClient returns FastAPI's
        default 500 response; the critical invariant is that the structured
        composer_plugin_error body is NOT produced for unknown classes.
        """
        mock_composer = AsyncMock()
        mock_composer.compose = AsyncMock(
            side_effect=RuntimeError("unknown failure class"),
        )

        app, _ = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Build me a pipeline"},
        )

        assert response.status_code == 500
        # Unconditional: the composer_plugin_error marker MUST NOT appear
        # anywhere in the response body, regardless of whether FastAPI
        # renders detail as a dict or a string.  This closes the vacuous-
        # pass risk of an `if isinstance(...)` guard.
        assert "composer_plugin_error" not in response.text
```

**Verify before writing:** `grep -n "def _make_app\|_make_composer_mock\|raise_server_exceptions" tests/unit/web/sessions/test_routes.py`. If `_make_app` has a different signature or the composer-injection pattern has drifted, adapt accordingly — do NOT invent helpers.

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/web/sessions/test_routes.py::TestComposePluginCrashResponse -v`
Expected: FAIL — the unhandled `ValueError` currently surfaces as an unstructured 500 with the exception message in the response body.

- [ ] **Step 4: Add the new route-level handler**

Two distinct insertions (one per route). The slog event name uses a route-specific **literal string** (not a `log_prefix` variable — `log_prefix` is a parameter of the `_handle_convergence_error` helper function and is NOT in scope inside the route handlers themselves; referencing it would throw `NameError` at runtime).

**Insertion 1 — `/compose` handler.** Insert after the `except ComposerServiceError` block at `routes.py:390–398`:

```python
        except (TypeError, ValueError, UnicodeError, KeyError, AttributeError) as exc:
            # Plugin-crash path: after Task 4 narrowed the service catch, these
            # classes escape _compose_loop unhandled (plugin bug per CLAUDE.md).
            # Do NOT echo the exception message — it may contain secret
            # fragments from __cause__-chained exceptions in deeper layers.
            slog.error(
                "compose_plugin_crash",
                session_id=str(session_id),
                user_id=str(user.user_id),
                exc_class=type(exc).__name__,
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error_type": "composer_plugin_error",
                    "detail": (
                        "A composer plugin crashed; see server logs for the "
                        "traceback. This is not a user-retryable error."
                    ),
                },
            ) from exc
```

**Insertion 2 — `/recompose` handler.** Insert after the matching `except ComposerServiceError` block at `routes.py:505–513`. Identical body except for the slog event name:

```python
        except (TypeError, ValueError, UnicodeError, KeyError, AttributeError) as exc:
            slog.error(
                "recompose_plugin_crash",
                session_id=str(session_id),
                user_id=str(user.user_id),
                exc_class=type(exc).__name__,
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error_type": "composer_plugin_error",
                    "detail": (
                        "A composer plugin crashed; see server logs for the "
                        "traceback. This is not a user-retryable error."
                    ),
                },
            ) from exc
```

Notes:
- The handler is BELOW `except ComposerServiceError` so convergence/auth/unavailable paths still win when applicable.
- Literal event names `"compose_plugin_crash"` / `"recompose_plugin_crash"` — no `log_prefix` variable exists at this scope. The existing `_handle_convergence_error(log_prefix="convergence")` pattern passes the literal in as an argument; the new handler is inline at the route level and has no such parameter.
- `exc_class=type(exc).__name__` is the only exception-derived attribute written to structured log fields — message is in `exc_info` only (stderr traceback, never structured fields).
- `from exc` preserves the cause chain on the `HTTPException` for any outer middleware, but the middleware MUST not echo the cause into the response body.
- Pre-existing `_handle_convergence_error` at `routes.py:177,199` catches `(ValueError, TypeError, KeyError)` defensively around `exc.partial_state.validate()`. That is on Tier 1 data (our own state, already persisted) and predates this plan. It looks inconsistent with the new narrow-catch contract but is a separate concern tracked as a follow-up observation — do NOT alter it in this PR.

- [ ] **Step 5: Run tests — confirm green**

Run: `.venv/bin/python -m pytest tests/unit/web/sessions/test_routes.py::TestComposePluginCrashResponse -v`
Expected: PASS.

- [ ] **Step 6: Run full sessions test suite**

Run: `.venv/bin/python -m pytest tests/unit/web/sessions/ -v`
Expected: all PASS.

- [ ] **Step 7: Type-check and lint**

Run: `.venv/bin/python -m mypy src/elspeth/web/sessions/routes.py`
Run: `.venv/bin/python -m ruff check src/elspeth/web/sessions/routes.py tests/unit/web/sessions/`
Expected: clean.

- [ ] **Step 8: Combined commit for Task 4 + Task 4.5 (atomicity requirement)**

This single commit ships the service-level narrowing (Task 4) and the route-level plugin-crash handler (Task 4.5) together. Splitting them would create a deployment window where plugin crashes escape `compose()` unhandled and FastAPI's default handler leaks exception content into the response body. Audit-integrity invariant > smaller-commit preference.

```bash
git add \
    src/elspeth/web/composer/service.py \
    src/elspeth/web/sessions/routes.py \
    tests/unit/web/composer/test_service.py \
    tests/unit/web/sessions/test_routes.py
git commit -m "fix(web/composer,sessions): narrow compose catch + structured 500 for plugin crashes

Single atomic change across service and route layers:

1. service.py narrows the compose-loop catch from
   (TypeError, ValueError, UnicodeError) to ToolArgumentError only.
   Plugin bugs raising the bare classes were being laundered as
   LLM-argument errors, polluting the audit trail and causing the
   LLM to 'self-correct' faults in our own code. Mirrors the existing
   KeyError/AttributeError contract.

2. service.py adds _persist_crashed_session for best-effort
   updated_at bumps on the crash path so audit has a breadcrumb
   (richer schema fields tracked as follow-up).

3. routes.py adds a typed final handler on both /compose and
   /recompose that shapes the now-unmasked plugin crashes into a
   documented {error_type: composer_plugin_error} 500, logs via
   slog.error with exc_class only (message goes to stderr via
   exc_info), and explicitly does NOT echo exception content into
   the response body.

The two changes MUST land together: Task 4 alone would expose a
window where plugin crashes escape unhandled and FastAPI's default
handler serializes exception messages (potentially carrying secret
fragments from __cause__ chains) directly to clients.

Fixes elspeth-7a26880c65."
```

---

### Task 5: Sweep `tools.py` for implicit upward-escape sites

**Files:**
- Read-only audit pass on: `src/elspeth/web/composer/tools.py` (whole file).
- Modify (if any findings): `src/elspeth/web/composer/tools.py` + add corresponding tests in `tests/unit/web/composer/test_tools.py`.

This task MAY produce no code changes. Its purpose is to close the Python-engineering reviewer's stated information gap: "Implicit `ValueError`/`TypeError` sites in the remaining ~1,600 lines not checked by the targeted grep."

**Relationship to Task 1.5.** Task 1.5 ran the same AST grep *before* implementation to build a triaged worklist. Task 5 re-runs the grep *after* Tasks 2 and 3 have converted the known sites, and sweeps for anything Task 1.5 classified as "Convert" that wasn't covered by Tasks 2/3. If Task 1.5's worklist was empty beyond lines 1753 and 1838, this task's Step 1 grep should return empty output and Task 5 completes with no commit.

- [ ] **Step 1: Re-run the grep post-conversion**

Run (from repo root):

```bash
.venv/bin/python -c "
import ast, sys
src = open('src/elspeth/web/composer/tools.py').read()
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.Raise) and node.exc is not None:
        if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
            if node.exc.func.id in {'TypeError', 'ValueError', 'UnicodeError', 'UnicodeDecodeError', 'UnicodeEncodeError'}:
                print(f'{node.lineno}: raise {node.exc.func.id}')
        elif isinstance(node.exc, ast.Name):
            if node.exc.id in {'TypeError', 'ValueError', 'UnicodeError'}:
                print(f'{node.lineno}: bare raise {node.exc.id}')
"
```

Expected (after Tasks 2 and 3): empty output, or only entries that are inside a local `try/except ... return _failure_result(...)` pattern. If the grep shows anything, continue to Step 2.

- [ ] **Step 2: For each finding, triage and convert**

For each remaining `raise`:

- If it signals a Tier-3 LLM-argument failure → change to `raise ToolArgumentError(msg) from exc` (or without `from` if no underlying exception).
- If it signals an internal invariant violation → leave as-is (it's a plugin bug and MUST crash; the narrowed catch will now let it through correctly).
- If it's inside a handler that has a surrounding `try/except` that would catch it and return `_failure_result` → leave as-is (it is already contained within the handler).

Also grep for implicit upward escapes — coercions on LLM-supplied values that are NOT wrapped:

```bash
.venv/bin/python -m ruff check src/elspeth/web/composer/tools.py --select=ALL 2>/dev/null | head -50
```

And scan for `int(arguments[`, `float(arguments[`, `tuple(arguments[`, `list(arguments[` in the handler bodies. Any such coercion that could raise on bad LLM input should be wrapped:

```python
try:
    count = int(arguments["count"])
except (TypeError, ValueError) as exc:
    raise ToolArgumentError(f"'count' must be an integer, got {arguments['count']!r}") from exc
```

OR caught locally and converted to `_failure_result`.

Do NOT preemptively wrap coercions that operate on internal state (e.g. `int(current_total)` at `tools.py:1719` — that's Tier-2 data, not Tier-3).

For each conversion made, add a direct handler test analogous to `TestCreateBlobTypeGuard`.

- [ ] **Step 3: If changes were made, run the full composer test suite**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/ -v`
Expected: all PASS.

- [ ] **Step 4: Commit (only if changes were made)**

```bash
git add src/elspeth/web/composer/tools.py tests/unit/web/composer/test_tools.py
git commit -m "refactor(web/composer): wrap implicit Tier-3 coercion sites in tools.py

Sweep pass after the ToolArgumentError landing: implicit coercions on
LLM-supplied arguments (e.g. int(), tuple()) now wrap their raises as
ToolArgumentError so the compose-loop's narrow catch can distinguish
LLM-argument errors from plugin-internal bugs. Refs elspeth-7a26880c65."
```

If no changes were made, add a short comment to the filigree issue noting "Task 5 audit: no additional implicit raise sites found" and move on.

---

### Task 6: Integration test — real `asyncio.to_thread` path

**Files:**
- Test: `tests/unit/web/composer/test_service.py` (new class `TestToolArgumentErrorAcrossThreadBoundary`)

The existing mocked tests patch `execute_tool` at the service seam, so the exception is raised synchronously on the mock call — not from within the worker thread. Add one test that lets `execute_tool` run for real (dispatched via `asyncio.to_thread`) and triggers the real `_execute_create_blob` Tier-3 guard. This closes the QA reviewer's "sleepy assertion" concern.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/web/composer/test_service.py`. Verified constructor signature (`service.py:164–179`): `ComposerServiceImpl(catalog, settings, session_engine=None, secret_service=None)` — there is NO `data_dir` parameter; `data_dir` is read from `settings.data_dir` at line 176. `_make_settings(**overrides)` accepts a `data_dir` override (verified `test_service.py:135–152`), so wire the temp directory through settings.

The test needs a real in-memory session engine with the schema migrated and a `sessions` row seeded (blob creation asserts the FK). Use an autouse class fixture mirroring `test_tools.py:2441` `TestDeleteBlobActiveRunGuard._setup`:

```python
class TestToolArgumentErrorAcrossThreadBoundary:
    """End-to-end: ToolArgumentError raised inside the worker thread is caught
    correctly by the service-level catch, with message preserved.

    Closes the sleepy-assertion gap in the mocked service-level tests
    (which raise synchronously on the mock and never exercise the real
    asyncio.to_thread re-raise path).
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.schema import initialize_session_schema
        from elspeth.web.sessions.models import sessions_table

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = str(uuid4())
        self.data_dir = tmp_path
        now = datetime.now(UTC)
        with self.engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=now,
                    updated_at=now,
                )
            )

    @pytest.mark.asyncio
    async def test_real_create_blob_type_guard_feeds_error_to_llm(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings(data_dir=self.data_dir)
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            session_engine=self.engine,
        )
        state = _empty_state()

        bad_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_bad",
                    "name": "create_blob",
                    "arguments": {
                        "filename": "x.txt",
                        "mime_type": "text/plain",
                        "content": 42,  # wrong type
                    },
                }
            ],
        )
        text = _make_llm_response(content="Fixed.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [bad_call, text]
            result = await service.compose(
                "Setup", [], state, session_id=self.session_id
            )

        assert result.message == "Fixed."
        second_call_messages = mock_llm.call_args_list[1].args[0]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        error_content = json.loads(tool_messages[0]["content"])
        assert "content must be a string" in error_content["error"]
```

Notes on the changes from the prior draft:

- `data_dir` is routed through `_make_settings(data_dir=self.data_dir)` — the constructor does not accept it.
- Tool-message extraction uses the filter-by-role pattern from Task 4's positive test (more robust than positional `[-1]`).
- `session_id` comes from the seeded `sessions` row so the `create_blob` handler can enforce same-session ownership (if it does not, the blob write will fail on FK or on the session-scoping check — which is a pre-existing guard, not something this plan is changing).

- [ ] **Step 2: Run to verify it passes already**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_service.py::TestToolArgumentErrorAcrossThreadBoundary -v`
Expected: PASS. If it fails, the most likely cause is fixture plumbing (session_engine/data_dir not wired through); fix the fixture rather than the production code.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/web/composer/test_service.py
git commit -m "test(web/composer): add end-to-end ToolArgumentError thread-boundary test

Closes the mocked-test gap: existing service-level tests patch
execute_tool at the seam and raise synchronously, never exercising the
real asyncio.to_thread re-raise. This test drives the real Tier-3
guard in _execute_create_blob through the worker thread and asserts
the error reaches the LLM with message preserved. Refs elspeth-7a26880c65."
```

---

### Task 7: Mechanical enforcement gate (REQUIRED)

This task converts the "don't raise bare TypeError/ValueError/UnicodeError from `tools.py`" convention into a CI-enforced invariant. Architecture, Systems, and QA panels converged on this being load-bearing: without it, the narrowed catch is a single-point fix guarded only by comment conventions — a future handler author using the natural Python idiom (`raise ValueError(...)`) reopens the silent channel. Task 7 is the only mechanism that turns "convention" into "mechanical invariant," so it ships in the same PR as Tasks 1–6, not as a follow-up.

**Files:**
- Create: `scripts/cicd/enforce_composer_exception_channel.py`
- Create: `config/cicd/enforce_composer_exception_channel/_defaults.yaml`
- Create: `tests/unit/scripts/cicd/test_enforce_composer_exception_channel.py` (co-located with the other CI-enforcer tests — `test_enforce_plugin_hashes.py`, `test_enforce_freeze_guards.py`, etc.)
- Create: `.github/workflows/enforce-composer-exception-channel.yaml` (GHA workflow — pre-commit alone is bypassable)

- [ ] **Step 1: Read the existing enforcement-script template**

Run: `cat scripts/cicd/enforce_freeze_guards.py | head -200`

Use this as the structural template — follow its rule dict, AST walk, Finding dataclass, allowlist YAML shape, and CLI interface. Do not diverge.

- [ ] **Step 2: Write the failing test for the enforcer**

Create `tests/unit/scripts/cicd/test_enforce_composer_exception_channel.py` (co-located with the other CI-enforcer tests — NOT `tests/unit/cicd/`). The enforcer's path-filter design (Step 4) mirrors `enforce_freeze_guards.py`: `relative_to(root)` (NOT `root.parent`), no hardcoded target-file string in `_scan_file` — callers scope the scan by passing the correct `--root` (production) or `tmp_path` (tests). A separate `main()` assertion requires `root / "web/composer/tools.py"` to exist when invoked against the real tree, so a rename/move fails closed rather than silently skipping.

**Subprocess-invocation pattern.** Use `python -m scripts.cicd.enforce_composer_exception_channel` (module form, mirroring `test_enforce_plugin_hashes.py:50-62`). This requires `cwd` to be the project root so that `scripts/` is on `sys.path` — confirmed empirically: running the module form from any other cwd yields `ModuleNotFoundError: No module named 'scripts'`. The mirror at `test_enforce_plugin_hashes.py:62` sets `cwd=str(Path(__file__).resolve().parents[4])`; we copy that exact pattern.

Note the separation of conventions in this repo:
- **Test subprocess helpers:** module form (`python -m scripts.cicd.X`) with `cwd=<project root>`.
- **GitHub Actions workflows:** direct-script form (`python scripts/cicd/X.py`) — see `.github/workflows/enforce-tier-model.yaml` for the template.

We preserve that split rather than inventing a new convention. The GHA workflow uses direct form (Task 7 Step 7b); the test helper below uses module form with an explicit `cwd`.

```python
"""Tests for enforce_composer_exception_channel.py.

Verifies the CI gate detects bare raises of TypeError/ValueError/UnicodeError
inside tool-handler files and accepts raises of ToolArgumentError.

Co-located with the other CI-enforcer tests at tests/unit/scripts/cicd/ —
mirrors test_enforce_plugin_hashes.py / test_enforce_freeze_guards.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(extra_args: list[str]) -> subprocess.CompletedProcess[str]:
    """Invoke the enforcer as a module, with cwd set to the project root.

    Mirrors the pattern in test_enforce_plugin_hashes.py:50-62. The
    module form requires the project root on sys.path so `scripts` is
    importable as a package; we set cwd explicitly rather than relying
    on wherever pytest was invoked from.
    """
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.cicd.enforce_composer_exception_channel",
            *extra_args,
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[4]),  # project root
    )


def _make_composer_tree(tmp_path: Path) -> Path:
    """Create a fake src layout so the enforcer's existence assertion holds.

    Layout created:
        tmp_path/web/composer/tools.py
    The enforcer is invoked with --root tmp_path (test-mode), so
    relative_to(root) yields "web/composer/tools.py".
    """
    target_dir = tmp_path / "web" / "composer"
    target_dir.mkdir(parents=True)
    return target_dir / "tools.py"


class TestComposerExceptionChannelEnforcer:
    def test_clean_tree_passes(self, tmp_path: Path) -> None:
        target = _make_composer_tree(tmp_path)
        target.write_text(
            "from elspeth.web.composer.protocol import ToolArgumentError\n"
            "def f(x):\n"
            "    if not isinstance(x, str):\n"
            "        raise ToolArgumentError('bad')\n"
        )
        result = _run(["check", "--root", str(tmp_path)])
        assert result.returncode == 0, result.stdout + result.stderr

    def test_bare_type_error_fails(self, tmp_path: Path) -> None:
        target = _make_composer_tree(tmp_path)
        target.write_text(
            "def f(x):\n"
            "    if not isinstance(x, str):\n"
            "        raise TypeError('bad')\n"
        )
        result = _run(["check", "--root", str(tmp_path)])
        assert result.returncode != 0
        assert "TypeError" in result.stdout

    def test_bare_value_error_fails(self, tmp_path: Path) -> None:
        target = _make_composer_tree(tmp_path)
        target.write_text(
            "def f(x):\n"
            "    raise ValueError('bad')\n"
        )
        result = _run(["check", "--root", str(tmp_path)])
        assert result.returncode != 0

    def test_explicit_raise_inside_try_except_still_flagged(self, tmp_path: Path) -> None:
        """An explicit `raise ValueError` inside a try/except is still flagged.

        The enforcer is intentionally scope-unaware: it detects raise-site
        classes, not control-flow containment. The legitimate contained
        pattern is `try: int(x) except ValueError: ...` (no explicit raise),
        covered by `test_implicit_raise_from_coercion_is_not_flagged` below.
        A handler author who writes `raise ValueError(...)` even inside a
        local try/except is using the wrong exception class for the channel
        and should be flagged.

        False-positive risk: legitimate handler code that catches its own
        raise purely for diagnostic formatting (e.g. raise-then-reformat
        patterns) will also trip this rule. If any such pattern is found
        during Task 5's sweep, resolve by refactoring to `raise
        ToolArgumentError(formatted_message) from exc` rather than
        allowlisting — the narrow exception class carries the
        channel-discipline semantics, and allowlisting would hide the
        design-intent mismatch.
        """
        target = _make_composer_tree(tmp_path)
        target.write_text(
            "def _failure_result(state, msg): return msg\n"
            "def f(x, state):\n"
            "    try:\n"
            "        raise ValueError('bad')\n"
            "    except ValueError as exc:\n"
            "        return _failure_result(state, str(exc))\n"
        )
        result = _run(["check", "--root", str(tmp_path)])
        assert result.returncode != 0

    def test_implicit_raise_from_coercion_is_not_flagged(self, tmp_path: Path) -> None:
        """Implicit raises (e.g. `int(x)` failing) are not textual raise sites.

        AST-level raise-node walking only sees explicit `raise X` nodes, so
        `int(x)` inside a try/except is invisible to the enforcer — which
        is correct. The handler's job is to wrap the implicit raise and
        either return _failure_result or re-raise as ToolArgumentError.
        """
        target = _make_composer_tree(tmp_path)
        target.write_text(
            "def _failure_result(state, msg): return msg\n"
            "def f(x, state):\n"
            "    try:\n"
            "        int(x)\n"
            "    except ValueError as exc:\n"
            "        return _failure_result(state, str(exc))\n"
        )
        result = _run(["check", "--root", str(tmp_path)])
        assert result.returncode == 0, result.stdout + result.stderr

    def test_missing_target_file_fails_closed(self, tmp_path: Path) -> None:
        """If web/composer/tools.py does not exist under the root, exit non-zero.

        Closes the threat-analyst fail-open concern: a future rename of
        tools.py would otherwise cause the enforcer to silently scan nothing.
        """
        # No _make_composer_tree() call — target is absent.
        result = _run(["check", "--root", str(tmp_path)])
        assert result.returncode != 0
        assert "web/composer/tools.py" in (result.stdout + result.stderr)

    def test_allowlist_entry_suppresses_matching_finding(self, tmp_path: Path) -> None:
        """An allowlist entry at the exact (file, line) of a bare raise
        suppresses the finding and the enforcer exits 0.

        The allowlist is load-bearing for legitimate exceptions (e.g. the
        JSON-serializer invariant-violation raise in service.py, if it were
        ever brought into scope). Without this test, the allowlist is an
        unverified mechanism — an implementation bug that dropped entries
        would silently break the escape hatch and surface only when a
        real exemption is needed under time pressure.
        """
        target = _make_composer_tree(tmp_path)
        target.write_text(
            "def f(x):\n"
            "    raise TypeError('bad')\n"
        )
        # Line 2 is the raise site in the file above.
        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()
        (allowlist_dir / "_defaults.yaml").write_text(
            "allowed:\n"
            "  - file: web/composer/tools.py\n"
            "    line: 2\n"
            "    justification: \"test-only exemption; real code never uses this\"\n"
        )
        result = _run(
            [
                "check",
                "--root", str(tmp_path),
                "--allowlist", str(allowlist_dir),
            ]
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_allowlist_entry_with_empty_justification_fails(self, tmp_path: Path) -> None:
        """Allowlist entries MUST have a non-empty justification field.

        Without this gate, the allowlist degrades into a silent dumping
        ground: any developer who hits a CI failure can add an entry with
        no explanation and the build turns green. The justification field
        forces a paper trail for every exemption.
        """
        target = _make_composer_tree(tmp_path)
        target.write_text(
            "def f(x):\n"
            "    raise TypeError('bad')\n"
        )
        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()
        (allowlist_dir / "_defaults.yaml").write_text(
            "allowed:\n"
            "  - file: web/composer/tools.py\n"
            "    line: 2\n"
            "    justification: \"\"\n"
        )
        result = _run(
            [
                "check",
                "--root", str(tmp_path),
                "--allowlist", str(allowlist_dir),
            ]
        )
        assert result.returncode != 0
        assert "justification" in (result.stdout + result.stderr)

    def test_allowlist_entry_missing_justification_key_fails(
        self, tmp_path: Path
    ) -> None:
        """An allowlist entry that omits the justification key entirely
        (as opposed to providing an empty string) is also rejected."""
        target = _make_composer_tree(tmp_path)
        target.write_text(
            "def f(x):\n"
            "    raise TypeError('bad')\n"
        )
        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()
        (allowlist_dir / "_defaults.yaml").write_text(
            "allowed:\n"
            "  - file: web/composer/tools.py\n"
            "    line: 2\n"
        )
        result = _run(
            [
                "check",
                "--root", str(tmp_path),
                "--allowlist", str(allowlist_dir),
            ]
        )
        assert result.returncode != 0
        assert "justification" in (result.stdout + result.stderr)
```

- [ ] **Step 3: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/scripts/cicd/test_enforce_composer_exception_channel.py -v`
Expected: FAIL (script does not exist).

- [ ] **Step 4: Write the enforcer**

Create `scripts/cicd/enforce_composer_exception_channel.py`. Mirrors `scripts/cicd/enforce_freeze_guards.py` for path handling: `relative_to(root)` (NOT `root.parent`), no hardcoded target-file short-circuit inside `_scan_file`. Scoping is the caller's responsibility via `--root`. Existence of the canonical target (`web/composer/tools.py` under the root) is asserted in `main()` so a move or rename fails closed.

```python
#!/usr/bin/env python3
"""Enforce exception-channel discipline in composer tool-handler files.

Tool handlers at the Tier-3 boundary MUST signal LLM-argument failures
via ToolArgumentError (from protocol.py), not bare TypeError/ValueError/
UnicodeError. A bare raise of those classes would be laundered through
the compose-loop catch as if it were an LLM error, masking plugin bugs.

Rules:
- CEC1: bare `raise TypeError/ValueError/UnicodeError(...)` under the
  scanned root. Fix: raise ToolArgumentError (for Tier-3 boundary errors),
  or catch locally and return _failure_result (for handler-internal
  recovery that produces a clean diagnostic).

Usage (production):
    python scripts/cicd/enforce_composer_exception_channel.py check \\
        --root src/elspeth \\
        --allowlist config/cicd/enforce_composer_exception_channel

Usage (test, with a fake tree under tmp_path):
    python scripts/cicd/enforce_composer_exception_channel.py check \\
        --root /tmp/pytest-.../test_name
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

RULES = {
    "CEC1": {
        "name": "bare-exception-raise-in-composer-tools",
        "description": "Bare raise of TypeError/ValueError/UnicodeError in composer tools — use ToolArgumentError",
        "remediation": "raise ToolArgumentError(...) from exc, or catch locally and return _failure_result",
    }
}

_BANNED = frozenset(
    {"TypeError", "ValueError", "UnicodeError", "UnicodeDecodeError", "UnicodeEncodeError"}
)

# The canonical file this gate protects. Expressed relative to --root.
# Kept as a single literal so a file move surfaces as a fail-closed error in main(),
# not a silent pass of an empty scan.
_CANONICAL_TARGET_REL = Path("web/composer/tools.py")


@dataclass(frozen=True)
class Finding:
    rule_id: str
    file_path: str
    lineno: int
    message: str


def _scan_file(path: Path, root: Path) -> list[Finding]:
    """Scan a single Python file for banned raises.

    `path` must be under `root`; `file_path` in findings is the path
    relative to `root` (matches enforce_freeze_guards.py convention).
    """
    rel = path.relative_to(root).as_posix()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
            name = node.exc.func.id
        elif isinstance(node.exc, ast.Name):
            name = node.exc.id
        else:
            continue
        if name in _BANNED:
            findings.append(
                Finding(
                    rule_id="CEC1",
                    file_path=rel,
                    lineno=node.lineno,
                    message=f"raise {name}(...) at {rel}:{node.lineno} — use ToolArgumentError",
                )
            )
    return findings


def _load_allowlist(path: Path | None) -> set[tuple[str, int]]:
    if path is None or not path.exists():
        return set()
    entries: set[tuple[str, int]] = set()
    for yml in path.glob("*.yaml"):
        data = yaml.safe_load(yml.read_text()) or {}
        for item in data.get("allowed", []):
            if "justification" not in item or not str(item["justification"]).strip():
                print(
                    f"Error: allowlist entry in {yml} missing non-empty 'justification': {item!r}",
                    file=sys.stderr,
                )
                sys.exit(1)
            entries.add((str(item["file"]), int(item["line"])))
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    check = sub.add_parser("check")
    check.add_argument("--root", required=True, type=Path)
    check.add_argument("--allowlist", type=Path, default=None)
    check.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    canonical_target = (root / _CANONICAL_TARGET_REL).resolve()
    if not canonical_target.is_file():
        print(
            f"Error: canonical target {_CANONICAL_TARGET_REL.as_posix()!r} "
            f"not found under --root {root}. The enforcer fails closed on a "
            "missing target so file moves surface immediately instead of "
            "silently skipping.",
            file=sys.stderr,
        )
        return 2

    allowlist = _load_allowlist(args.allowlist)
    if args.files:
        targets = [p.resolve() for p in args.files]
    else:
        targets = [canonical_target]

    findings: list[Finding] = []
    for target in targets:
        if not target.is_file():
            continue
        findings.extend(_scan_file(target, root))

    active = [f for f in findings if (f.file_path, f.lineno) not in allowlist]
    for f in active:
        print(f"[{f.rule_id}] {f.message}")
    return 1 if active else 0


if __name__ == "__main__":
    sys.exit(main())
```

Key differences from the prior draft of this script:

1. `_scan_file` uses `path.relative_to(root)` — matches `enforce_freeze_guards.py:400`. No `.parent` confusion.
2. No hardcoded `_TARGET_FILES` short-circuit inside `_scan_file`. Scoping is the caller's job via `--root` and `files`.
3. `main()` asserts `_CANONICAL_TARGET_REL` exists under `--root`, exiting with code 2 if not. Closes the fail-open-on-rename concern.
4. Allowlist entries require a non-empty `justification` field — otherwise the gate becomes a silent dumping ground.
5. Default scan target when no explicit `files` are given is the canonical target only (not an rglob that might pick up unrelated `tools.py` files in the tree).

Create `config/cicd/enforce_composer_exception_channel/_defaults.yaml`:

**Pre-requisite: read the Task 1.5 audit artifact.** Open `docs/superpowers/plans/2026-04-17-composer-tool-argument-error.audit.md`. If the file does not exist, STOP — Task 1.5 was skipped and the allowlist scope is unknown. Re-run Task 1.5 first.

For each row classified as "Internal invariant" in the audit artifact, add an entry to the allowlist with the justification copied from the Notes column. Do NOT invent entries; do NOT omit entries found by Task 1.5. The audit artifact is authoritative.

```yaml
# Allowlist for enforce_composer_exception_channel.py.
# Entries here are explicit exemptions — each MUST include a non-empty
# justification field (enforced by the script; missing/empty justification
# fails the build). Entries are seeded from Task 1.5's audit artifact at
# docs/superpowers/plans/2026-04-17-composer-tool-argument-error.audit.md.
# After Tasks 2 and 3, the "Convert" rows are gone; only "Internal invariant"
# rows from the audit remain here.
#
# Entry schema:
#   - file: web/composer/tools.py   # relative to --root (e.g. src/elspeth)
#     line: 1753                    # raise-site line number
#     justification: "why this bare raise is acceptable here — copied from audit Notes"
allowed: []
```

After writing the allowlist, run the enforcer against the real tree (Step 6) and confirm exit 0. A non-zero exit means the audit artifact and the allowlist have drifted — reconcile before proceeding.

- [ ] **Step 5: Run the enforcer test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/scripts/cicd/test_enforce_composer_exception_channel.py -v`
Expected: all PASS.

- [ ] **Step 6: Run the enforcer against the real tree**

Run: `.venv/bin/python scripts/cicd/enforce_composer_exception_channel.py check --root src/elspeth`
Expected: exit 0, no findings (Tasks 2 and 3 already converted both sites).

- [ ] **Step 7a: Wire into `.pre-commit-config.yaml`**

Inspect `.pre-commit-config.yaml`. Find where `enforce_freeze_guards.py` is invoked and add an analogous entry for the new enforcer. Exact YAML must match the existing file's conventions — copy the nearest hook and adapt only the fields that change.

```yaml
  - id: enforce-composer-exception-channel
    name: Composer exception-channel discipline
    entry: .venv/bin/python scripts/cicd/enforce_composer_exception_channel.py check --root src/elspeth --allowlist config/cicd/enforce_composer_exception_channel
    language: system
    pass_filenames: false
    files: ^src/elspeth/web/composer/tools\.py$
```

- [ ] **Step 7b: Add a GitHub Actions workflow (required — pre-commit alone is bypassable)**

Pre-commit hooks run locally and on PRs that have pre-commit CI configured, but direct pushes to branches or PRs from forks without pre-commit setup bypass them. For parity with `enforce_tier_model` / `enforce_guard_symmetry` / `enforce_component_type`, add a matching GHA workflow.

Inspect the existing template first — the canonical invocation form is **direct-script** (verified: `enforce-tier-model.yaml` uses `python scripts/cicd/enforce_tier_model.py check ...`):

```bash
ls .github/workflows/enforce-*.yaml
cat .github/workflows/enforce-tier-model.yaml
```

Create `.github/workflows/enforce-composer-exception-channel.yaml` by copying `enforce-tier-model.yaml` and adapting only the name, trigger `paths:`, script path, and `--root` / `--allowlist` args. Match the existing dependency-install pattern (`pip install pyyaml`) exactly — do NOT introduce a new pattern.

```yaml
name: Enforce composer exception-channel discipline

on:
  pull_request:
    paths:
      - "src/elspeth/web/composer/tools.py"
      - "config/cicd/enforce_composer_exception_channel/**"
      - "scripts/cicd/enforce_composer_exception_channel.py"
      - ".github/workflows/enforce-composer-exception-channel.yaml"
  push:
    branches:
      - main
    paths:
      - "src/elspeth/web/composer/tools.py"
      - "config/cicd/enforce_composer_exception_channel/**"
      - "scripts/cicd/enforce_composer_exception_channel.py"

permissions:
  contents: read

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: |
          python -m pip install --upgrade pip
          pip install pyyaml
      - run: |
          python scripts/cicd/enforce_composer_exception_channel.py check \
            --root src/elspeth \
            --allowlist config/cicd/enforce_composer_exception_channel
```

**Invocation convention (deliberate split).** This workflow uses **direct-script form** (`python scripts/cicd/X.py`) to match the existing three enforce-*.yaml workflows. The subprocess test helper in Step 2 uses **module form** (`python -m scripts.cicd.X`) with an explicit `cwd=<project root>` — mirroring the pattern in `test_enforce_plugin_hashes.py`. Both patterns are already established in the repo; we preserve them rather than forcing a single convention. The GHA workflow runs with GitHub's default `cwd=<checkout root>` so the relative script path resolves; the test helper runs from wherever pytest is invoked so it must set `cwd` explicitly.

- [ ] **Step 8: Type-check and lint the new script and tests**

Run: `.venv/bin/python -m mypy scripts/cicd/enforce_composer_exception_channel.py tests/unit/scripts/cicd/test_enforce_composer_exception_channel.py`
Run: `.venv/bin/python -m ruff check scripts/cicd/enforce_composer_exception_channel.py tests/unit/scripts/cicd/test_enforce_composer_exception_channel.py`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add \
    scripts/cicd/enforce_composer_exception_channel.py \
    config/cicd/enforce_composer_exception_channel/_defaults.yaml \
    tests/unit/scripts/cicd/test_enforce_composer_exception_channel.py \
    .pre-commit-config.yaml \
    .github/workflows/enforce-composer-exception-channel.yaml \
    docs/superpowers/plans/2026-04-17-composer-tool-argument-error.audit.md
git commit -m "feat(cicd): enforce ToolArgumentError discipline in composer/tools.py

Mechanical gate that fails the build if a bare raise of TypeError/
ValueError/UnicodeError is introduced in src/elspeth/web/composer/
tools.py — converts the convention from 'handlers must use
ToolArgumentError' into a compile-time-like invariant. Closes the
Shifting-the-Burden risk flagged by the architecture reviewer.
Refs elspeth-7a26880c65."
```

---

### Task 8: Close the filigree issue

- [ ] **Step 1: Add a decision-record comment to the issue**

Use the filigree MCP tool `mcp__filigree__add_comment` with `issue_id=elspeth-7a26880c65`. Comment body (paste verbatim, editing only the commit SHAs at the end):

```
Resolution: Option 1 (typed ToolArgumentError) landed after panel review
(architect / systems / python / QA). Key decisions:

- Archetype reclassified from Accidental Adversaries to Fixes that Fail
  (systems thinker verdict — no competing goal-seeking loop, just a fix
  whose side effect regenerated the original problem).
- Option 2 (pre-dispatch schema validation) deferred: three reviewers
  flagged it as a second source of truth for arg constraints and a
  dumping-ground risk. Filed as optional defense-in-depth follow-up.
- Task 7 (CI enforcement gate) added per architecture + QA recommendation
  to close the Shifting-the-Burden residual risk.
- Existing service-level tests were sleepy assertions (patched
  execute_tool at the seam; never exercised asyncio.to_thread re-raise).
  Replaced with direct handler tests + one end-to-end thread-boundary
  test.

Commits: <paste SHAs from git log --oneline -10>
```

- [ ] **Step 2: Transition to closed state**

The issue's valid transitions from `triage` are `confirmed` (needs severity), `wont_fix`, or `not_a_bug`. Since this is a fix, not a "won't fix," set severity and transition to `confirmed`, then close on PR merge. Using `mcp__filigree__update_issue`:

```
update_issue(id="elspeth-7a26880c65", status="confirmed", fields={"severity": "medium"})
```

Then after the PR merges, `close_issue(id="elspeth-7a26880c65", reason="Fixed by Option 1 — ToolArgumentError landed; see PR SHA.")`.

- [ ] **Step 3: File the Option-2 follow-up issue (optional defense-in-depth)**

Use `mcp__filigree__create_issue`:

```
create_issue(
    title="Pre-dispatch JSON Schema validation for composer tool arguments (Option 2)",
    type="task",
    priority=3,
    description="Defense-in-depth on top of elspeth-7a26880c65. Existing JSON Schema specs in tools.py (~line 283+) are used for LLM function-calling; extend the service to validate LLM-supplied arguments against those schemas BEFORE asyncio.to_thread dispatch. Post-dispatch, any surviving exception is by construction a plugin bug. Gated on an audit of schema completeness (required fields, types, enums) — see the architecture reviewer's notes on plan 2026-04-17-composer-tool-argument-error.md.",
    labels=["cluster:composer-correctness"],
)
```

- [ ] **Step 4: File the crash-marker schema follow-up (required)**

The current `_persist_crashed_session` only bumps `updated_at` because the `sessions_table` schema lacks dedicated crash-state columns. This is a weaker audit breadcrumb than the plan would prefer — distinguishable from normal activity only by correlating with slog emissions. File a follow-up to add the missing columns:

```
create_issue(
    title="Add sessions_table crash-marker columns (status, crashed_at, last_exc_class)",
    type="task",
    priority=2,
    description=(
        "Follow-up to elspeth-7a26880c65. The Task 4a _persist_crashed_session method "
        "currently only bumps updated_at because the sessions_table schema has no "
        "crash-state columns. An operator investigating a compose session cannot tell "
        "from the session row alone whether it crashed, converged, or is mid-flight — "
        "they must correlate updated_at with stderr slog output.\n\n"
        "Work:\n"
        "1. Add columns to sessions_table via Alembic migration: status "
        "(pending|active|crashed|completed|abandoned), crashed_at (nullable timestamp), "
        "last_exc_class (nullable string, class name only — NEVER the message).\n"
        "2. Update _persist_crashed_session to populate crashed_at + last_exc_class + "
        "status='crashed'. Remove the `_ = (last_state, exc_class)` discard.\n"
        "3. Update TestPluginCrashSessionPersistence to assert the new columns populated.\n"
        "4. Audit existing session-row readers (routes, dashboards, TUI) for the new status "
        "enum values.\n\n"
        "Schema audit-integrity invariant: NEVER persist the exception message — "
        "only the class name. Cause-chain content may include secrets from deeper layers."
    ),
    labels=["cluster:composer-correctness"],
)
```

---

## Self-Review

**Spec coverage:**
- Pre-implementation raise-site audit (feeds Task 7 scope) → Task 1.5.
- Typed exception class (inherits from `Exception`, NOT `ComposerServiceError`) → Task 1.
- Convert the two deliberate raises → Tasks 2 and 3.
- Narrow the service catch → Task 4.
- Add `structlog` import to service.py (pre-requisite for Step 4a's crash-persistence fallback) → Task 4, Step 4a-pre.
- Update existing tests (L564, L1159) → Task 4, Step 1 and Step 2.
- Add plugin-crash tests (mirror KeyError test) → Task 4, Step 2.
- Persist last-known state on plugin crash as `updated_at` bump (schema-compatible breadcrumb; richer fields deferred to follow-up) → Task 4, Step 4a.
- Redact `__cause__` chain from LLM-echo path → Task 4, Step 4 (`safe_message`) + Step 4b leakage regression test.
- Structured 500 response when plugin bugs escape `compose()` (literal slog event names, no `log_prefix` variable) → Task 4.5 (route-layer handler).
- Atomic commit for Task 4 + Task 4.5 (prevents mid-deploy exception-message leakage) → Task 4.5 Step 8.
- Add direct handler tests (close the sleepy-assertion gap) → Tasks 2, 3, and 6.
- Re-audit implicit raise sites post-conversion → Task 5.
- CI enforcement gate (required) with allowlist positive/negative tests → Task 7.
- GitHub Actions workflow parity with pre-commit hook → Task 7, Step 7b.
- Close the filigree issue + file Option-2 follow-up + file crash-marker schema follow-up → Task 8.

**Placeholder scan:** No "TBD", "implement later", or "similar to Task N" without repeated code. Two areas rely on in-repo fixture names (`_mock_catalog`, `_empty_state`, `_make_settings`, `_make_llm_response`). `_persist_crashed_session` is now fully specified (bumps `updated_at` only, with an explicit forward-compatibility comment referencing the follow-up schema migration). All Task 4.5 route-test bodies are executable (no `...` stubs) modulo fixture-name adaptation, which is called out inline.

**Type consistency:** `ToolArgumentError` is defined in Task 1 as a direct `Exception` subclass and imported identically in Tasks 2, 3, 4, 6. The inheritance choice is load-bearing — see Task 1's docstring rationale and the route-handler catch list in Task 4.5 (which deliberately excludes `ToolArgumentError` because an escaped `ToolArgumentError` is a compose-loop bug, not a plugin bug). `_failure_result(state, msg)` is referenced with the same signature across Tasks 5 and 7. Exception-chaining uses `raise X from exc` consistently. The Task 4.5 route handler catches the concrete plugin-crash classes (`TypeError`, `ValueError`, `UnicodeError`, `KeyError`, `AttributeError`) — NOT a bare `Exception` — so that future unknown exception classes still surface as unstructured 500s and get investigated rather than silently absorbed.

**Omissions I accept knowingly:**
- Task 7 does not attempt to generalise to other directories (e.g. `src/elspeth/plugins/`). The Tier-3 boundary of concern is the composer/LLM seam; plugin internals have their own discipline governed elsewhere.
- Task 5 does not enumerate every handler body line-by-line; it relies on the AST grep to drive focused triage. A 3039-LOC file cannot be exhaustively walked in a plan without becoming unreadable.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-17-composer-tool-argument-error.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. All eight tasks (1, 2, 3, 4, 4.5, 5, 6, 7) ship in the same PR — no deferrals.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

Which approach?
