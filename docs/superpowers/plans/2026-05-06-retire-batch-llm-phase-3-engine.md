# Phase 3 — Engine and Contracts Demolition

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the async-batch primitives (`BatchPendingError`, `BatchCheckpointState`, `RowMappingEntry`, `get_checkpoint`/`set_checkpoint` plumbing, `batch_checkpoints` parameter, all engine handlers) that exist solely to serve `azure_batch_llm`'s submit-and-poll lifecycle. After this phase, the orchestrator and aggregation executor have no notion of asynchronous batch transforms; only `BatchTransformProtocol` (synchronous) survives.

**Architecture:** This is the load-bearing structural change. After Phase 2, the LLM batch plugins are gone but the engine still defines and references the async-batch primitives. This phase deletes:

1. The single `contracts/batch_checkpoint.py` file (193 LOC) and `BatchPendingError` class (~40 LOC in `errors.py`).
2. The `_checkpoint`/`_batch_checkpoints` fields and `get_checkpoint`/`set_checkpoint`/`clear_checkpoint` methods on `PluginAuditContext` (~80 LOC in `plugin_context.py`).
3. The `get_checkpoint`/`set_checkpoint` protocol methods on `PluginContextProtocol` (3 LOC in `contracts/contexts.py`).
4. The `batch_checkpoints` parameter on `PipelineOrchestrator._execute_run`, `run`, and `_initialize_run_context` (multiple call sites in `engine/orchestrator/core.py`).
5. The two `except BatchPendingError:` handlers in `engine/executors/aggregation.py`.
6. The CLI helper text mentioning `azure_batch_llm`.

Tests covering the removed paths (`tests/unit/contracts/test_batch_checkpoint.py`, `tests/unit/contracts/test_checkpoint_post_init.py`, `tests/unit/engine/test_executors.py:2428–2465`) are deleted or trimmed.

**Tech Stack:** Python 3.13, pytest, mypy, ruff. Order matters: contracts first (rip out the type), then everything that imported it follows. mypy and ruff drive the cleanup.

---

## Task 1: Update CLI helper text (small, isolated, no compile risk)

**Files:**
- Modify: `src/elspeth/cli_helpers.py:109` (drop `'azure_batch_llm'` from the help message)

- [ ] **Step 1: Read the current message**

```bash
sed -n '105,115p' src/elspeth/cli_helpers.py
```

Expected:

```python
                f"Use a batch-aware transform like 'azure_batch_llm', 'batch_stats', "
                f"or 'batch_replicate', or set is_batch_aware=True on your custom transform."
```

- [ ] **Step 2: Update the f-string**

Use Edit. Change:

```python
                f"Use a batch-aware transform like 'azure_batch_llm', 'batch_stats', "
                f"or 'batch_replicate', or set is_batch_aware=True on your custom transform."
```

to:

```python
                f"Use a batch-aware transform like 'batch_stats' or 'batch_replicate', "
                f"or set is_batch_aware=True on your custom transform."
```

- [ ] **Step 3: Run the cli_helpers test (helper-message assertion is in tests/unit/cli/test_cli_helpers.py)**

```bash
.venv/bin/python -m pytest tests/unit/cli/test_cli_helpers.py -x -v 2>&1 | tail -20
```

Expected: green (Phase 2 already updated the assertion if it referenced the removed name; if a test fails on the help-message text, edit the test assertion to match the new wording in the same commit).

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/cli_helpers.py tests/unit/cli/test_cli_helpers.py
git commit -m "refactor(cli): drop azure_batch_llm from batch-aware help text"
```

---

## Task 2: Remove `BatchPendingError` handlers from the engine

**Files:**
- Modify: `src/elspeth/engine/executors/aggregation.py:386-387, 514` (remove two `except BatchPendingError:` blocks)
- Modify: `src/elspeth/engine/executors/aggregation.py:12` (remove `BatchPendingError` from import block)
- Modify: `src/elspeth/engine/orchestrator/core.py:1625, 2821, 3082` (remove three `except BatchPendingError:` blocks)
- Modify: `src/elspeth/engine/orchestrator/core.py:48` (remove `BatchPendingError` from import block)

- [ ] **Step 1: Inspect each handler block**

```bash
grep -n "BatchPendingError" src/elspeth/engine/executors/aggregation.py src/elspeth/engine/orchestrator/core.py
```

Then for each line number, read 10 lines of context to understand the handler shape (e.g., `sed -n '383,395p' src/elspeth/engine/executors/aggregation.py`).

Expected: each handler is a `try: ... except BatchPendingError: ... # CONTROL-FLOW SIGNAL` block that re-raises (it does *not* swallow). After removal, the underlying exception simply propagates as any other unhandled exception would — which is the desired behaviour because no transform raises `BatchPendingError` after Phase 2.

- [ ] **Step 2: Remove each handler block**

Use Edit. Remove the `except BatchPendingError:` plus its body (typically 3–8 lines per site, including a comment and a re-raise).

For `aggregation.py:386-387`: also remove the surrounding `try:` if the only `except` was BatchPendingError; otherwise keep the `try:` and other `except`s, just drop the BatchPendingError clause.

Same pattern for `aggregation.py:514` and the three sites in `orchestrator/core.py`.

- [ ] **Step 3: Remove the now-unused imports**

In `aggregation.py:12`, change:

```python
from elspeth.contracts.errors import (
    BatchPendingError,
    ...
)
```

to drop the `BatchPendingError,` line. Same for `orchestrator/core.py:48`.

- [ ] **Step 4: Run mypy on the engine to confirm no dangling references**

```bash
.venv/bin/python -m mypy src/elspeth/engine/ 2>&1 | tail -20
```

Expected: green. Any remaining `BatchPendingError` reference will surface as `Name "BatchPendingError" is not defined`.

- [ ] **Step 5: Run pytest on the engine executors and orchestrator**

```bash
.venv/bin/python -m pytest tests/unit/engine/ -x -q 2>&1 | tail -30
```

Expected: most tests pass. Failures localised to `tests/unit/engine/test_executors.py:2428–2465` (the explicit BatchPendingError-path tests) — those will be removed in Task 6 below.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/engine/executors/aggregation.py src/elspeth/engine/orchestrator/core.py
git commit -m "refactor(engine): remove BatchPendingError handlers from aggregation + orchestrator"
```

---

## Task 3: Remove the `batch_checkpoints` parameter from the orchestrator

**Files:**
- Modify: `src/elspeth/engine/orchestrator/core.py` (multiple sites: lines 1471, 1486–1489, 1551, 1972, 2010, 2018, 2977, 3001, 3467)

- [ ] **Step 1: Inspect each site**

```bash
grep -n "batch_checkpoints\|BatchCheckpointState" src/elspeth/engine/orchestrator/core.py
```

Expected: parameter declarations (3 method signatures), parameter forwarding (calls between methods), one usage at line 2018 (`_batch_checkpoints=batch_checkpoints or {}` — restoring into PluginAuditContext, which loses its `_batch_checkpoints` field in Task 4 anyway), one positional `None` argument at line 3467 in a sibling call site.

- [ ] **Step 2: Remove the parameter from each method signature and call site**

Use Edit on each site. The pattern is:

- Method signature: `batch_checkpoints: dict[str, BatchCheckpointState] | None = None,` → delete the entire line
- Docstring lines 1486–1489 describing `batch_checkpoints` → delete the block
- Forwarding call (`run(..., batch_checkpoints=batch_checkpoints, ...)` or positional) → delete that argument
- Line 2010–2018 (the `_batch_checkpoints=batch_checkpoints or {}` assignment) → delete the whole assignment block

Note: line 3467 has `None,  # batch_checkpoints` as a positional argument. Delete the line.

- [ ] **Step 3: Remove the now-unused `BatchCheckpointState` import (line 47)**

In `orchestrator/core.py:47`, change:

```python
from elspeth.contracts import (
    ...
    BatchCheckpointState,
    BatchPendingError,
    ...
)
```

to drop both lines (assuming Task 2 already removed `BatchPendingError`).

- [ ] **Step 4: mypy + ruff sweep on engine**

```bash
.venv/bin/python -m mypy src/elspeth/engine/ 2>&1 | tail -20
.venv/bin/python -m ruff check src/elspeth/engine/ 2>&1 | tail -20
```

Expected: both green. If mypy complains about unused imports of `BatchCheckpointState` from `elspeth.contracts`, drop those.

- [ ] **Step 5: Run engine + orchestrator tests, including resume-failure**

```bash
.venv/bin/python -m pytest tests/unit/engine/ tests/unit/engine/orchestrator/ -x -q 2>&1 | tail -30
```

Expected: green. `tests/unit/engine/orchestrator/test_resume_failure.py:198` passes `None` for batch_checkpoints — that test must be edited at the same time. Use Edit on that line:

```python
# before
None,  # batch_checkpoints
# after — just delete this line
```

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/engine/orchestrator/core.py tests/unit/engine/orchestrator/test_resume_failure.py
git commit -m "refactor(orchestrator): remove batch_checkpoints parameter from run/resume flow"
```

---

## Task 4: Remove `_batch_checkpoints` plumbing from `PluginAuditContext`

**Files:**
- Modify: `src/elspeth/contracts/plugin_context.py:25, 130–219` (remove imports, fields, methods)
- Modify: `src/elspeth/contracts/contexts.py:28, 127–129` (remove `BatchCheckpointState` import + protocol methods)

- [ ] **Step 1: Inspect the affected blocks**

```bash
sed -n '125,225p' src/elspeth/contracts/plugin_context.py
sed -n '125,135p' src/elspeth/contracts/contexts.py
```

Expected: comment block (lines 130–135), `_checkpoint` and `_batch_checkpoints` field declarations, and three methods (`get_checkpoint`, `set_checkpoint`, `clear_checkpoint`).

- [ ] **Step 2: Delete in `plugin_context.py`**

Use Edit. Remove:
- The TYPE_CHECKING import (`from elspeth.contracts.batch_checkpoint import BatchCheckpointState`) at line 25 (only if no other references remain — verify with `grep BatchCheckpointState src/elspeth/contracts/plugin_context.py` after editing).
- The comment block at lines 130–135 ("# Used by batch transforms ...").
- The two field declarations (`_checkpoint: BatchCheckpointState | None = field(...)` and `_batch_checkpoints: dict[str, BatchCheckpointState] = field(...)`).
- The three methods `get_checkpoint`, `set_checkpoint`, `clear_checkpoint` and their docstrings (lines 173–219).

- [ ] **Step 3: Delete in `contexts.py`**

Use Edit. Remove:
- The TYPE_CHECKING import at line 28 (`from elspeth.contracts.batch_checkpoint import BatchCheckpointState`).
- The two protocol method stubs at lines 127–129 (`def get_checkpoint(...)` and `def set_checkpoint(...)`).

- [ ] **Step 4: mypy sweep on contracts**

```bash
.venv/bin/python -m mypy src/elspeth/contracts/ 2>&1 | tail -20
```

Expected: green. `BatchCheckpointState` will still resolve (defined in `batch_checkpoint.py` until Task 5), so this passes.

- [ ] **Step 5: Run plugin context tests**

```bash
.venv/bin/python -m pytest tests/unit/plugins/test_context.py -x -v 2>&1 | tail -20
```

Expected: most tests pass. Failures occur in tests that explicitly exercise `get_checkpoint`/`set_checkpoint` — edit those tests to delete the relevant test methods.

`tests/unit/plugins/test_context.py:11, 38` reference `BatchCheckpointState` imports — remove the imports and any test methods that use them. Look for `test_set_checkpoint_*`, `test_get_checkpoint_*`, etc.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/contracts/plugin_context.py src/elspeth/contracts/contexts.py tests/unit/plugins/test_context.py
git commit -m "refactor(contracts): remove get_checkpoint/set_checkpoint plumbing from PluginAuditContext"
```

---

## Task 5: Delete `batch_checkpoint.py` and `BatchPendingError`

**Files:**
- Delete: `src/elspeth/contracts/batch_checkpoint.py`
- Modify: `src/elspeth/contracts/errors.py:30, 261, 645–714` (remove TYPE_CHECKING import, the comment at 261 mentioning azure_batch, the `BatchPendingError` class)
- Modify: `src/elspeth/contracts/__init__.py:58, 160, 301, 399, 400` (remove imports and exports)

- [ ] **Step 1: Confirm zero non-test consumers remain**

```bash
grep -rn "BatchCheckpointState\|RowMappingEntry\|BatchPendingError" src/ --include="*.py" 2>&1 | grep -v "batch_checkpoint.py:\|errors.py:\|contracts/__init__.py:"
```

Expected: no hits. If any survive, Task 4 missed something — go back and fix before deleting the type.

- [ ] **Step 2: Delete `batch_checkpoint.py`**

```bash
git rm src/elspeth/contracts/batch_checkpoint.py
```

- [ ] **Step 3: Remove `BatchPendingError` and its imports from `errors.py`**

Use Edit. Remove:
- TYPE_CHECKING import at line 30: `from elspeth.contracts.batch_checkpoint import BatchCheckpointState`
- The comment at line 261 referring to azure_batch (`# Batch plugins (e.g., azure_batch) may store structured error bodies`) — replace with a generic comment or delete if redundant
- The whole `class BatchPendingError(Exception):` definition (lines 645–714 — verify the end line by reading the context)

- [ ] **Step 4: Remove imports and exports from `contracts/__init__.py`**

Use Edit. Remove:
- Line 58: `from elspeth.contracts.batch_checkpoint import BatchCheckpointState, RowMappingEntry`
- Line 160: `BatchPendingError,` (in the imports-from-errors block)
- Line 301: `"BatchPendingError",` (from `__all__`)
- Lines 399–400: `"BatchCheckpointState",` and `"RowMappingEntry",` (from `__all__`)

- [ ] **Step 5: Full mypy + ruff sweep on src/**

```bash
.venv/bin/python -m mypy src/ 2>&1 | tail -30
.venv/bin/python -m ruff check src/ 2>&1 | tail -20
```

Expected: green. Any failure means a downstream consumer is still importing one of the removed names.

- [ ] **Step 6: Run the full unit suite**

```bash
.venv/bin/python -m pytest tests/unit -x -q 2>&1 | tail -40
```

Expected: green except possibly for `tests/unit/contracts/test_batch_checkpoint.py`, `tests/unit/contracts/test_checkpoint_post_init.py`, `tests/unit/contracts/test_freeze_regression.py`, `tests/unit/plugins/llm/test_p1_bug_fixes.py`, `tests/unit/engine/test_executors.py` — those are addressed in Task 6.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/contracts/errors.py src/elspeth/contracts/__init__.py
git commit -m "refactor(contracts): remove BatchPendingError, BatchCheckpointState, RowMappingEntry"
```

---

## Task 6: Delete or trim test files that exercise the removed primitives

**Files (delete entirely if scope is purely batch-checkpoint):**
- `tests/unit/contracts/test_batch_checkpoint.py`
- `tests/unit/contracts/test_checkpoint_post_init.py` (verify scope first — its name suggests it covers `BatchCheckpointState.__post_init__` specifically)

**Files (trim, not delete):**
- `tests/unit/contracts/test_freeze_regression.py:22` — has `from elspeth.contracts.batch_checkpoint import BatchCheckpointState, RowMappingEntry`. Drop the import; if any test method referenced these types, delete the method.
- `tests/unit/plugins/llm/test_p1_bug_fixes.py:19, 329, 488` — TYPE_CHECKING import + two inline imports inside test methods. Identify the test methods that use `BatchCheckpointState` and either delete them (if scope is batch-LLM only) or rewrite them to no longer assert against the removed type.
- `tests/unit/engine/test_executors.py:2428–2465+` — the `BatchPendingError`-path tests. Delete the whole `def test_batch_pending_error_*` methods.

- [ ] **Step 1: Inspect each file**

```bash
head -30 tests/unit/contracts/test_batch_checkpoint.py tests/unit/contracts/test_checkpoint_post_init.py
sed -n '15,40p' tests/unit/contracts/test_freeze_regression.py
sed -n '15,30p' tests/unit/plugins/llm/test_p1_bug_fixes.py
sed -n '320,500p' tests/unit/plugins/llm/test_p1_bug_fixes.py | head -40
sed -n '2425,2470p' tests/unit/engine/test_executors.py
```

Expected: clear scoping per file. Decide delete vs trim.

- [ ] **Step 2: Delete the two whole-file targets**

```bash
git rm tests/unit/contracts/test_batch_checkpoint.py
# Verify scope of test_checkpoint_post_init.py first; if its top-of-file docstring or imports show it covers BatchCheckpointState only, delete:
git rm tests/unit/contracts/test_checkpoint_post_init.py
```

If `test_checkpoint_post_init.py` covers other dataclasses too, trim instead — keep the file, drop the BatchCheckpointState methods.

- [ ] **Step 3: Trim the survivor files**

Use Edit on `test_freeze_regression.py`, `test_p1_bug_fixes.py`, `test_executors.py`:

- Drop top-of-file imports of removed names
- Delete inline imports inside test methods
- Delete whole test methods whose only purpose was BatchCheckpointState/BatchPendingError coverage

After each edit, re-grep:

```bash
grep -n "BatchCheckpointState\|RowMappingEntry\|BatchPendingError" tests/ --include="*.py" 2>&1
```

Expected after Task 6: zero hits across `tests/`.

- [ ] **Step 4: Full unit suite**

```bash
.venv/bin/python -m pytest tests/unit -x -q 2>&1 | tail -30
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/contracts/ tests/unit/plugins/llm/test_p1_bug_fixes.py tests/unit/engine/test_executors.py
git commit -m "test(contracts): remove BatchCheckpointState + BatchPendingError test coverage"
```

---

## Phase 3 Exit Criteria

- [ ] `src/elspeth/contracts/batch_checkpoint.py` deleted
- [ ] `BatchPendingError` no longer defined or exported anywhere
- [ ] `BatchCheckpointState`, `RowMappingEntry` no longer in `elspeth.contracts.__all__`
- [ ] `_checkpoint`, `_batch_checkpoints`, `get_checkpoint`, `set_checkpoint`, `clear_checkpoint` removed from `PluginAuditContext`
- [ ] `get_checkpoint`, `set_checkpoint` removed from `PluginContextProtocol`
- [ ] `batch_checkpoints` parameter removed from `PipelineOrchestrator._execute_run`, `run`, `_initialize_run_context`
- [ ] No `except BatchPendingError:` blocks remain in `engine/`
- [ ] CLI helper text no longer mentions `azure_batch_llm`
- [ ] `mypy src/` green
- [ ] `ruff check src/` green
- [ ] `pytest tests/unit -x -q` green
- [ ] `grep -rn "BatchPendingError\|BatchCheckpointState\|RowMappingEntry" src/ tests/` returns zero non-deleted hits

After Phase 3, the engine has no notion of asynchronous batch transforms. Phase 4 sweeps documentation and CI allowlists.
