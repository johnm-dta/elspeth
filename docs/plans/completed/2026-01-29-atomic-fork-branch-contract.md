# Atomic Fork + Branch Contract Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the crash window in fork/expand operations and add branch contract validation to recovery.

**Architecture:**
- Extend the existing transaction in `fork_token()` to include parent outcome recording
- Add `expected_branches_json` column to `token_outcomes` table for contract storage
- Update recovery query to detect missing branches

**Tech Stack:** SQLAlchemy Core, Alembic migrations, pytest

---

## Task 1: Schema Migration - Add expected_branches_json Column

**Files:**
- Create: `alembic/versions/XXXX_add_expected_branches_to_outcomes.py`
- Reference: `src/elspeth/core/landscape/schema.py`

**Step 1: Generate migration stub**

```bash
cd /home/john/elspeth-rapid
alembic revision -m "add_expected_branches_to_token_outcomes"
```

**Step 2: Write migration content**

```python
"""add_expected_branches_to_token_outcomes

Revision ID: <generated>
Revises: <previous>
Create Date: 2026-01-29
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '<generated>'
down_revision = '<previous>'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'token_outcomes',
        sa.Column('expected_branches_json', sa.Text(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('token_outcomes', 'expected_branches_json')
```

**Step 3: Update schema.py to match**

Add to `token_outcomes_table` definition:
```python
Column("expected_branches_json", Text),  # Branch contract for FORKED/EXPANDED outcomes
```

**Step 4: Run migration on test database**

```bash
alembic upgrade head
```

**Step 5: Verify column exists**

```bash
sqlite3 landscape.db ".schema token_outcomes" | grep expected_branches
```

Expected: `expected_branches_json TEXT`

**Step 6: Commit**

```bash
git add alembic/versions/*_add_expected_branches*.py src/elspeth/core/landscape/schema.py
git commit -m "schema: add expected_branches_json to token_outcomes for branch contract"
```

---

## Task 2: Make LandscapeRecorder.fork_token() Atomic

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py:731-802`

**Step 1: Add run_id parameter and outcome recording**

Update the method signature and add outcome recording inside the transaction:

```python
def fork_token(
    self,
    parent_token_id: str,
    row_id: str,
    branches: list[str],
    *,
    run_id: str,  # NEW: required for outcome recording
    step_in_pipeline: int | None = None,
) -> tuple[list[Token], str]:  # NEW: also returns fork_group_id
    """Fork a token to multiple branches.

    ATOMIC: Creates children AND records parent FORKED outcome in single transaction.
    Stores branch contract for recovery validation.

    Returns:
        Tuple of (child tokens, fork_group_id)
    """
    if not branches:
        raise ValueError("fork_token requires at least one branch")

    fork_group_id = generate_id()
    children = []

    with self._db.connection() as conn:
        # 1. Create child tokens
        for ordinal, branch_name in enumerate(branches):
            child_id = generate_id()
            timestamp = now()

            conn.execute(
                tokens_table.insert().values(
                    token_id=child_id,
                    row_id=row_id,
                    fork_group_id=fork_group_id,
                    branch_name=branch_name,
                    step_in_pipeline=step_in_pipeline,
                    created_at=timestamp,
                )
            )

            conn.execute(
                token_parents_table.insert().values(
                    token_id=child_id,
                    parent_token_id=parent_token_id,
                    ordinal=ordinal,
                )
            )

            children.append(
                Token(
                    token_id=child_id,
                    row_id=row_id,
                    fork_group_id=fork_group_id,
                    branch_name=branch_name,
                    step_in_pipeline=step_in_pipeline,
                    created_at=timestamp,
                )
            )

        # 2. NEW: Record parent FORKED outcome in SAME transaction
        outcome_id = f"out_{generate_id()[:12]}"
        conn.execute(
            token_outcomes_table.insert().values(
                outcome_id=outcome_id,
                run_id=run_id,
                token_id=parent_token_id,
                outcome=RowOutcome.FORKED.value,
                is_terminal=1,
                recorded_at=now(),
                fork_group_id=fork_group_id,
                expected_branches_json=json.dumps(branches),  # Branch contract
            )
        )

    return children, fork_group_id
```

**Step 2: Add json import if needed**

Check top of file for `import json`, add if missing.

**Step 3: Run type checker**

```bash
.venv/bin/python -m mypy src/elspeth/core/landscape/recorder.py
```

**Step 4: Commit**

```bash
git add src/elspeth/core/landscape/recorder.py
git commit -m "fix(recorder): make fork_token atomic - children + parent outcome in single txn"
```

---

## Task 3: Update TokenManager.fork_token() to Pass run_id

**Files:**
- Modify: `src/elspeth/engine/tokens.py:121-163`

**Step 1: Add run_id parameter**

```python
def fork_token(
    self,
    parent_token: TokenInfo,
    branches: list[str],
    step_in_pipeline: int,
    run_id: str,  # NEW
    row_data: dict[str, Any] | None = None,
) -> list[TokenInfo]:
```

**Step 2: Pass run_id to recorder**

```python
children, fork_group_id = self._recorder.fork_token(
    parent_token_id=parent_token.token_id,
    row_id=parent_token.row_id,
    branches=branches,
    run_id=run_id,  # NEW
    step_in_pipeline=step_in_pipeline,
)
```

**Step 3: Update return to include fork_group_id in TokenInfo**

The TokenInfo objects already get fork_group_id from the child tokens, so this should work.

**Step 4: Run type checker**

```bash
.venv/bin/python -m mypy src/elspeth/engine/tokens.py
```

**Step 5: Commit**

```bash
git add src/elspeth/engine/tokens.py
git commit -m "fix(tokens): add run_id parameter to fork_token for atomic outcome recording"
```

---

## Task 4: Update Processor to Use New fork_token API

**Files:**
- Modify: `src/elspeth/engine/processor.py` (fork execution site)

**Step 1: Find and update the fork call site**

Search for where `_token_manager.fork_token` is called and:
1. Add `run_id=self._run_id` parameter
2. REMOVE the separate `record_token_outcome(FORKED)` call (now done atomically)

The call site is around line 1452-1457. Remove:
```python
# REMOVE THIS - now handled atomically in fork_token
self._recorder.record_token_outcome(
    run_id=self._run_id,
    token_id=current_token.token_id,
    outcome=RowOutcome.FORKED,
    fork_group_id=fork_group_id,
)
```

**Step 2: Verify no other FORKED outcome recording sites**

```bash
grep -n "RowOutcome.FORKED" src/elspeth/engine/processor.py
```

Should only find the import, not any recording calls.

**Step 3: Run tests**

```bash
.venv/bin/python -m pytest tests/engine/test_processor.py -v -x
```

**Step 4: Commit**

```bash
git add src/elspeth/engine/processor.py
git commit -m "fix(processor): use atomic fork_token, remove separate FORKED outcome recording"
```

---

## Task 5: Apply Same Pattern to expand_token

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py` (expand_token method)
- Modify: `src/elspeth/engine/tokens.py` (expand_token method)
- Modify: `src/elspeth/engine/processor.py` (expand call sites)

**Step 1: Update LandscapeRecorder.expand_token()**

Add `run_id` parameter and record EXPANDED outcome atomically, storing expected row count.

**Step 2: Update TokenManager.expand_token()**

Pass through `run_id` parameter.

**Step 3: Update processor expand call sites**

Remove separate EXPANDED outcome recording.

**Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/engine/ -v -x -k "expand"
```

**Step 5: Commit**

```bash
git add src/elspeth/core/landscape/recorder.py src/elspeth/engine/tokens.py src/elspeth/engine/processor.py
git commit -m "fix(expand): make expand_token atomic - children + parent outcome in single txn"
```

---

## Task 6: Update Recovery to Validate Branch Contracts

**Files:**
- Modify: `src/elspeth/core/checkpoint/recovery.py:233-365`

**Step 1: Add branch contract validation subquery**

After the existing delegation token check, add:

```python
# Subquery: Fork parents where expected branches don't match actual children
# A fork is "incomplete" if expected_branches_json has branches not found in children
incomplete_forks = (
    select(token_outcomes_table.c.token_id)
    .where(token_outcomes_table.c.run_id == run_id)
    .where(token_outcomes_table.c.outcome == RowOutcome.FORKED.value)
    .where(token_outcomes_table.c.expected_branches_json.isnot(None))
    # We'll check in Python: count children with matching fork_group_id
    # For now, the existing "non-delegation without terminal" check catches most cases
)
```

**Step 2: Add Python-side validation for branch count**

After the SQL query, validate that each fork_group_id has the expected number of children:

```python
# Validate branch contracts for any fork groups in unprocessed rows
for row_id in unprocessed:
    fork_outcomes = self._get_fork_outcomes_for_row(run_id, row_id)
    for fork in fork_outcomes:
        if fork.expected_branches_json:
            expected = json.loads(fork.expected_branches_json)
            actual_children = self._get_children_for_fork_group(fork.fork_group_id)
            actual_branches = {c.branch_name for c in actual_children}
            if set(expected) != actual_branches:
                # Log warning - branch contract violated
                logger.warning(
                    f"Fork contract violated: expected {expected}, got {actual_branches}"
                )
```

**Step 3: Run recovery tests**

```bash
.venv/bin/python -m pytest tests/core/checkpoint/test_recovery.py -v
```

**Step 4: Commit**

```bash
git add src/elspeth/core/checkpoint/recovery.py
git commit -m "fix(recovery): validate fork branch contracts against expected_branches_json"
```

---

## Task 7: Add Tests for Atomic Fork and Branch Contract

**Files:**
- Create: `tests/core/checkpoint/test_recovery_branch_contract.py`
- Modify: `tests/engine/test_tokens.py`

**Step 1: Test atomic fork transaction**

```python
class TestAtomicForkTransaction:
    """Verify fork creates children AND parent outcome atomically."""

    def test_fork_token_records_parent_outcome(self, recorder, run_id):
        """fork_token should record FORKED outcome on parent."""
        # Create parent token
        parent = recorder.create_initial_token(run_id, "source", 0)

        # Fork it
        children, fork_group_id = recorder.fork_token(
            parent_token_id=parent.token_id,
            row_id=parent.row_id,
            branches=["path_a", "path_b"],
            run_id=run_id,
            step_in_pipeline=1,
        )

        # Verify parent has FORKED outcome
        outcome = recorder.get_token_outcome(parent.token_id)
        assert outcome is not None
        assert outcome.outcome == RowOutcome.FORKED.value
        assert outcome.fork_group_id == fork_group_id

    def test_fork_stores_branch_contract(self, recorder, run_id):
        """fork_token should store expected branches for contract validation."""
        parent = recorder.create_initial_token(run_id, "source", 0)

        children, fork_group_id = recorder.fork_token(
            parent_token_id=parent.token_id,
            row_id=parent.row_id,
            branches=["alpha", "beta", "gamma"],
            run_id=run_id,
            step_in_pipeline=1,
        )

        outcome = recorder.get_token_outcome(parent.token_id)
        expected = json.loads(outcome.expected_branches_json)
        assert expected == ["alpha", "beta", "gamma"]
```

**Step 2: Test branch contract validation in recovery**

```python
class TestBranchContractValidation:
    """Verify recovery detects missing branches."""

    def test_recovery_detects_missing_branch(self, recovery_manager, ...):
        """If fork promised 2 branches but only 1 exists, row is unprocessed."""
        # Setup: Create fork with expected ["a", "b"] but only child "a" exists
        # (Simulate bug that created incomplete fork)

        # Act
        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        # Assert: Row should be in unprocessed list
        assert row_id in unprocessed
```

**Step 3: Run all tests**

```bash
.venv/bin/python -m pytest tests/core/checkpoint/ tests/engine/test_tokens.py -v
```

**Step 4: Commit**

```bash
git add tests/
git commit -m "test: add tests for atomic fork and branch contract validation"
```

---

## Task 8: Update Documentation

**Files:**
- Modify: `docs/architecture/token-lifecycle.md`

**Step 1: Update the "Token Creation" section**

Note that fork_token now atomically records the parent outcome.

**Step 2: Add "Branch Contract" subsection**

Document that expected_branches_json is stored and validated on recovery.

**Step 3: Commit**

```bash
git add docs/architecture/token-lifecycle.md
git commit -m "docs: document atomic fork and branch contract validation"
```

---

## Task 9: Run Full Test Suite and Type Check

**Step 1: Run mypy**

```bash
.venv/bin/python -m mypy src/elspeth/
```

**Step 2: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -v --tb=short
```

**Step 3: Fix any failures**

**Step 4: Final commit if needed**

---

## Summary

| Task | Description | Risk |
|------|-------------|------|
| 1 | Schema migration | Low |
| 2 | Atomic fork in recorder | Low |
| 3 | TokenManager API update | Low |
| 4 | Processor update | Medium |
| 5 | Atomic expand | Medium |
| 6 | Recovery validation | Medium |
| 7 | Tests | Low |
| 8 | Documentation | Low |
| 9 | Full test suite | — |

**Total estimated time:** 3-4 hours

**Key invariants after this change:**
1. Fork/expand are atomic - no crash window between children and parent outcome
2. Branch contracts are stored - recovery can detect "promised N, got M" scenarios
3. All existing tests continue to pass
