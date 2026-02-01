# TransformReason (Success Reason) Implementation Plan

**Status:** ✅ IMPLEMENTED (2026-02-01)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire up the existing `TransformReason` TypedDict to provide structured metadata for successful transform operations. This completes the audit symmetry: errors have `TransformErrorReason`, routing has `RoutingReason`, and now success has `TransformReason`.

**Value:**
- `action`: Distinguishes conditional paths ("enriched" vs "skipped - data present")
- `fields_modified`: Efficient audit queries without diffing input/output
- `validation_warnings`: Non-blocking warnings for data quality monitoring

**Bead:** TBD (create with `bd create --title="Wire up TransformReason for success metadata" --type=feature --priority=2`)

---

## Implementation Summary

- Success metadata standardized via `TransformSuccessReason` (`src/elspeth/contracts/errors.py`).
- `TransformResult.success()`/`success_multi()` require `success_reason` and enforce invariants (`src/elspeth/contracts/results.py`).
- Success reason stored in audit trail (`success_reason_json` in `src/elspeth/core/landscape/schema.py`, recorder write path in `src/elspeth/core/landscape/recorder.py`).
- Tests cover contract shape and audit persistence (`tests/contracts/test_results.py`, `tests/engine/test_transform_success_reason.py`).

## Task 1: Rename TransformReason to TransformSuccessReason

**Files:**
- Modify: `src/elspeth/contracts/errors.py:84-92`
- Modify: `src/elspeth/contracts/__init__.py`
- Modify: `tests/contracts/test_errors.py`

**Context:** Rename for clarity - distinguishes from `TransformErrorReason`. Also add a Literal type for common actions.

**Step 1: Verify Literal is imported in errors.py**

Check that `Literal` is already imported from typing. If not, add it to the imports.

**Step 2: Update TypedDict in errors.py**

Replace lines 84-92 with:

```python
# Literal type for common transform actions (extensible - str also accepted)
TransformActionCategory = Literal[
    # Processing actions
    "processed",  # Generic successful processing
    "mapped",  # Field mapping completed
    "validated",  # Validation passed
    "enriched",  # Data enrichment from external source
    "transformed",  # Data transformation applied
    "normalized",  # Data normalization applied
    "filtered",  # Row passed filter criteria
    "classified",  # Classification assigned
    # Skip/passthrough actions
    "passthrough",  # No changes made (intentional)
    "skipped",  # Processing skipped (e.g., data already present)
    "cached",  # Result retrieved from cache
]


class TransformSuccessReason(TypedDict):
    """Metadata for successful transform operations.

    Provides structured audit information about what a transform did,
    beyond just the input/output data. This enables:
    - Efficient audit queries (fields_modified without diffing)
    - Data quality monitoring (validation_warnings for non-blocking warnings)
    - Conditional path tracking (action distinguishes code paths)

    Used when transforms return TransformResult.success() with optional
    success_reason parameter.

    Required field:
        action: What the transform did. Use TransformActionCategory values
                for common actions, or custom strings for plugin-specific actions.

    Optional fields:
        fields_modified: List of field names that were changed
        fields_added: List of field names that were added
        fields_removed: List of field names that were removed
        validation_warnings: Non-blocking validation issues (data quality flags)
        metadata: Additional plugin-specific context

    Example usage:
        # Simple action tracking
        TransformResult.success(row, success_reason={"action": "enriched"})

        # Field change tracking
        TransformResult.success(row, success_reason={
            "action": "mapped",
            "fields_modified": ["customer_id", "amount"],
            "fields_added": ["currency_code"],
        })

        # Data quality warning
        TransformResult.success(row, success_reason={
            "action": "validated",
            "validation_warnings": ["amount near threshold (995 of 1000 limit)"],
        })
    """

    action: str  # Use TransformActionCategory or custom string

    # Field tracking
    fields_modified: NotRequired[list[str]]
    fields_added: NotRequired[list[str]]
    fields_removed: NotRequired[list[str]]

    # Data quality
    validation_warnings: NotRequired[list[str]]

    # Extensibility
    metadata: NotRequired[dict[str, Any]]
```

**Step 3: Update exports in __init__.py**

Change import:
```python
# From:
    TransformReason,
# To:
    TransformActionCategory,
    TransformSuccessReason,
```

Change `__all__`:
```python
# From:
    "TransformReason",
# To:
    "TransformActionCategory",
    "TransformSuccessReason",
```

**Step 4: Update tests**

In `tests/contracts/test_errors.py`, rename test classes:
- `TestTransformReasonSchema` → `TestTransformSuccessReasonSchema`
- `TestTransformReason` → `TestTransformSuccessReason`

Update test content to match new field names (`validation_warnings` instead of `validation_errors`).

Add test for Literal type:
```python
def test_transform_action_category_values(self) -> None:
    """TransformActionCategory contains expected action types."""
    from typing import get_args
    from elspeth.contracts import TransformActionCategory

    categories = get_args(TransformActionCategory)
    assert "processed" in categories
    assert "mapped" in categories
    assert "skipped" in categories
    assert "enriched" in categories
```

**Step 5: Verify**

Run: `.venv/bin/python -c "from elspeth.contracts import TransformSuccessReason, TransformActionCategory; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add src/elspeth/contracts/errors.py src/elspeth/contracts/__init__.py tests/contracts/test_errors.py
git commit -m "$(cat <<'EOF'
feat(contracts): rename TransformReason to TransformSuccessReason

Clarify naming to distinguish from TransformErrorReason.
Expand fields for better audit tracking:
- action: What the transform did (Literal + custom string)
- fields_modified/added/removed: Explicit field tracking
- validation_warnings: Non-blocking data quality flags
- metadata: Plugin-specific extensibility

Add TransformActionCategory Literal for common action values.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add success_reason Field to TransformResult

**Files:**
- Modify: `src/elspeth/contracts/results.py`

**Step 1: Ensure deferred annotations**

Verify `from __future__ import annotations` is at the top of the file (after docstring).
If missing, add it.

**Step 2: Add import**

Add to imports:
```python
from elspeth.contracts.errors import TransformErrorReason, TransformSuccessReason
```

**Step 3: Add field to dataclass**

Add after line 101 (`rows: list[dict[str, Any]] | None = None`):

```python
    # Success metadata - set by plugin via success() factory
    success_reason: TransformSuccessReason | None = None
```

**Step 4: Update success() factory method**

Change lines 118-121:

```python
    @classmethod
    def success(
        cls,
        row: dict[str, Any],
        *,
        success_reason: TransformSuccessReason | None = None,
    ) -> TransformResult:
        """Create successful result with single output row.

        Args:
            row: The transformed row data
            success_reason: Optional metadata about what the transform did.
                           See TransformSuccessReason for available fields.

        Returns:
            TransformResult with status="success" and the provided row.
        """
        return cls(
            status="success",
            row=row,
            reason=None,
            rows=None,
            success_reason=success_reason,
        )
```

**Step 5: Update success_multi() factory method**

Change lines 123-138:

```python
    @classmethod
    def success_multi(
        cls,
        rows: list[dict[str, Any]],
        *,
        success_reason: TransformSuccessReason | None = None,
    ) -> TransformResult:
        """Create successful result with multiple output rows.

        Args:
            rows: List of output rows (must not be empty)
            success_reason: Optional metadata about what the transform did.
                           See TransformSuccessReason for available fields.

        Returns:
            TransformResult with status="success", row=None, rows=rows

        Raises:
            ValueError: If rows is empty
        """
        if not rows:
            raise ValueError("success_multi requires at least one row")
        return cls(
            status="success",
            row=None,
            reason=None,
            rows=rows,
            success_reason=success_reason,
        )
```

**Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests/contracts/test_results.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/elspeth/contracts/results.py
git commit -m "$(cat <<'EOF'
feat(results): add success_reason field to TransformResult

Transforms can now provide structured metadata about successful
processing via optional success_reason parameter:

    TransformResult.success(row, success_reason={
        "action": "enriched",
        "fields_modified": ["customer_id"],
    })

The success_reason is optional - existing code continues to work
unchanged. This is additive, not breaking.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add success_reason_json Column to node_states Schema

**Files:**
- Modify: `src/elspeth/core/landscape/schema.py`

**Context:** No migration needed - we have no deployed databases requiring migration.

**Step 1: Update schema.py**

Add after line 199 (`Column("error_json", Text),`):

```python
    Column("success_reason_json", Text),  # TransformSuccessReason for successful transforms
```

**Step 2: Verify schema**

Run: `.venv/bin/python -c "from elspeth.core.landscape.schema import node_states_table; print([c.name for c in node_states_table.columns])"`
Expected: Shows `success_reason_json` in column list

**Step 3: Commit**

```bash
git add src/elspeth/core/landscape/schema.py
git commit -m "$(cat <<'EOF'
feat(schema): add success_reason_json column to node_states

New column stores TransformSuccessReason JSON for successful
transform operations. Nullable - transforms without success_reason
will have NULL in this column.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update NodeStateCompleted Dataclass

**Files:**
- Modify: `src/elspeth/contracts/audit.py`

**Step 1: Add field to NodeStateCompleted**

Add after line 220 (`context_after_json: str | None = None`):

```python
    success_reason_json: str | None = None
```

**Step 2: Verify**

Run: `.venv/bin/python -c "from elspeth.contracts import NodeStateCompleted; print(NodeStateCompleted.__dataclass_fields__.keys())"`
Expected: Shows `success_reason_json` in fields

**Step 3: Commit**

```bash
git add src/elspeth/contracts/audit.py
git commit -m "$(cat <<'EOF'
feat(audit): add success_reason_json to NodeStateCompleted

Dataclass now includes optional success_reason_json field
matching the new schema column.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update NodeStateRepository to Load success_reason_json

**Files:**
- Modify: `src/elspeth/core/landscape/repositories.py`

**Context:** The recorder uses a repository pattern. `complete_node_state()` calls `get_node_state()` which uses `NodeStateRepository.load()` to construct dataclasses from database rows. We must update the repository to include the new field.

**Step 1: Update COMPLETED branch in load() method**

Find the `NodeStateCompleted` constructor at line 341-355. Add `success_reason_json`:

```python
        elif status == NodeStateStatus.COMPLETED:
            # COMPLETED states must have output_hash, completed_at, duration_ms
            # Validate required fields - None indicates audit integrity violation
            if row.output_hash is None:
                raise ValueError(f"COMPLETED state {row.state_id} has NULL output_hash - audit integrity violation")
            if row.duration_ms is None:
                raise ValueError(f"COMPLETED state {row.state_id} has NULL duration_ms - audit integrity violation")
            if row.completed_at is None:
                raise ValueError(f"COMPLETED state {row.state_id} has NULL completed_at - audit integrity violation")
            return NodeStateCompleted(
                state_id=row.state_id,
                token_id=row.token_id,
                node_id=row.node_id,
                step_index=row.step_index,
                attempt=row.attempt,
                status=NodeStateStatus.COMPLETED,
                input_hash=row.input_hash,
                started_at=row.started_at,
                output_hash=row.output_hash,
                completed_at=row.completed_at,
                duration_ms=row.duration_ms,
                context_before_json=row.context_before_json,
                context_after_json=row.context_after_json,
                success_reason_json=row.success_reason_json,  # ADD THIS
            )
```

**Step 2: Verify**

Run: `.venv/bin/python -c "from elspeth.core.landscape.repositories import NodeStateRepository; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/elspeth/core/landscape/repositories.py
git commit -m "$(cat <<'EOF'
feat(repositories): load success_reason_json in NodeStateRepository

NodeStateRepository.load() now reads success_reason_json from
database rows and includes it in NodeStateCompleted instances.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Update LandscapeRecorder.complete_node_state()

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py`

**Step 1: Add import**

Add to TYPE_CHECKING imports (recorder.py already uses this pattern):
```python
if TYPE_CHECKING:
    ...
    from elspeth.contracts.errors import TransformSuccessReason
```

**Step 2: Update method signature and overloads**

Update the COMPLETED overload (lines 1062-1072):

```python
    @overload
    def complete_node_state(
        self,
        state_id: str,
        status: Literal[NodeStateStatus.COMPLETED],
        *,
        output_data: dict[str, Any] | list[dict[str, Any]] | None = None,
        duration_ms: float | None = None,
        error: ExecutionError | TransformErrorReason | CoalesceFailureReason | None = None,
        success_reason: TransformSuccessReason | None = None,  # ADD THIS
        context_after: dict[str, Any] | None = None,
    ) -> NodeStateCompleted: ...
```

**Step 3: Update main implementation**

Update the main `complete_node_state` method signature (line 1086):

```python
    def complete_node_state(
        self,
        state_id: str,
        status: NodeStateStatus,
        *,
        output_data: dict[str, Any] | list[dict[str, Any]] | None = None,
        duration_ms: float | None = None,
        error: ExecutionError | TransformErrorReason | CoalesceFailureReason | None = None,
        success_reason: TransformSuccessReason | None = None,  # ADD THIS
        context_after: dict[str, Any] | None = None,
    ) -> NodeStatePending | NodeStateCompleted | NodeStateFailed:
```

**Step 4: Serialize success_reason**

In the method body, after line 1122 (`context_json = canonical_json(context_after) if context_after is not None else None`), add:

```python
        # Serialize success reason if provided (use canonical_json for audit consistency)
        success_reason_json = canonical_json(success_reason) if success_reason is not None else None
```

**Note:** We use `canonical_json()` (not `json.dumps()`) to maintain consistency with `error_json` and `context_json` serialization. This ensures deterministic output and rejects NaN/Infinity values per the audit integrity requirements.

**Step 5: Update the UPDATE statement**

Find the `node_states_table.update()` call (lines 1124-1134) and add `success_reason_json`:

```python
        self._ops.execute_update(
            node_states_table.update()
            .where(node_states_table.c.state_id == state_id)
            .values(
                status=status.value,
                output_hash=output_hash,
                duration_ms=duration_ms,
                error_json=error_json,
                success_reason_json=success_reason_json,  # ADD THIS
                context_after_json=context_json,
                completed_at=timestamp,
            )
        )
```

**Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests/core/landscape/ -v -k "complete_node_state or recorder"`
Expected: PASS

**Step 7: Commit**

```bash
git add src/elspeth/core/landscape/recorder.py
git commit -m "$(cat <<'EOF'
feat(recorder): store success_reason in complete_node_state

LandscapeRecorder now accepts optional success_reason parameter
and stores it as JSON in the success_reason_json column for
COMPLETED node states.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Update Executors to Pass Through success_reason

**Files:**
- Modify: `src/elspeth/engine/executors.py`

**Step 1: Update TransformExecutor success path**

Find line 353 (the COMPLETED path in TransformExecutor) and update:

```python
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status=NodeStateStatus.COMPLETED,
                output_data=output_data,
                duration_ms=duration_ms,
                success_reason=result.success_reason,  # ADD THIS
            )
```

**Step 2: Update AggregationExecutor success path**

Find line 1199 (the COMPLETED path in AggregationExecutor) and update:

```python
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status=NodeStateStatus.COMPLETED,
                output_data=output_data,
                duration_ms=duration_ms,
                success_reason=result.success_reason,  # ADD THIS
            )
```

**Step 3: GateExecutor and SinkExecutor**

Gates and sinks don't return TransformResult, so no changes needed for those.
Gates have RoutingReason (stored via routing events), sinks produce artifacts.

**Step 4: Run executor tests**

Run: `.venv/bin/python -m pytest tests/engine/test_transform_executor.py tests/engine/test_executors.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/engine/executors.py
git commit -m "$(cat <<'EOF'
feat(executors): pass success_reason through to recorder

TransformExecutor and AggregationExecutor now pass
TransformResult.success_reason to complete_node_state().

Gates and sinks unchanged - they have their own audit mechanisms
(routing events and artifacts respectively).

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Verify MCP Server Exposes success_reason (No Code Changes Needed)

**Files:**
- No modifications required

**Context:** The MCP server's `explain_token` uses `dataclass_to_dict()` from `formatters.py` which recursively serializes all dataclass fields. Since `NodeStateCompleted` now has `success_reason_json`, it will automatically appear in the output.

**Important:** The field will appear as a **raw JSON string**, not parsed JSON:
```json
{
  "node_states": [{
    "status": "COMPLETED",
    "success_reason_json": "{\"action\": \"processed\"}"
  }]
}
```

This is consistent with how `error_json` and `context_after_json` are exposed.

**Step 1: Verify output**

Run the MCP server and test `explain_token` on a run with transforms:
```bash
elspeth-mcp
```

Then use the `explain_token` tool and verify the response includes `success_reason_json` field for completed transform states.

**Step 2: No commit needed**

No code changes required - the formatter handles it automatically.

---

## Task 9: Add Integration Tests

**Files:**
- Create: `tests/engine/test_transform_success_reason.py`

**Step 1: Create test file**

```python
"""Tests for TransformSuccessReason audit trail integration.

Verifies that success_reason flows from transform through executor
to Landscape audit trail correctly.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from elspeth.contracts import TransformResult, TransformSuccessReason
from elspeth.contracts.audit import NodeStateCompleted
from elspeth.contracts.enums import NodeStateStatus, NodeType
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder


class TestTransformSuccessReasonAudit:
    """Tests for success_reason in audit trail."""

    @pytest.fixture
    def landscape_db(self, tmp_path) -> LandscapeDB:
        """Create in-memory landscape database."""
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        db.create_all()
        return db

    @pytest.fixture
    def recorder(self, landscape_db: LandscapeDB) -> LandscapeRecorder:
        """Create recorder."""
        return LandscapeRecorder(landscape_db)

    def test_success_reason_stored_in_node_state(
        self,
        recorder: LandscapeRecorder,
    ) -> None:
        """success_reason is stored in node_states table."""
        # Setup run
        run_id = recorder.create_run(
            settings_hash="test",
            settings_snapshot={},
            code_version="test",
        )
        recorder.register_node(
            node_id="transform_1",
            run_id=run_id,
            node_type=NodeType.TRANSFORM,
            plugin_name="field_tracking_transform",
            config={},
        )
        row_id = recorder.create_source_row(
            run_id=run_id,
            row_index=0,
            raw_data={"amount": 100},
        )
        token = recorder.create_token(
            run_id=run_id,
            row_id=row_id,
            row_data={"amount": 100},
        )

        # Create and complete node state with success_reason
        state = recorder.create_node_state(
            token_id=token.token_id,
            node_id="transform_1",
            run_id=run_id,
            step_index=0,
            input_data={"amount": 100},
        )

        success_reason: TransformSuccessReason = {
            "action": "processed",
            "fields_added": ["processed", "amount_usd"],
        }

        completed = recorder.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"amount": 100, "processed": True, "amount_usd": 100.0},
            duration_ms=5.0,
            success_reason=success_reason,
        )

        # Verify
        assert isinstance(completed, NodeStateCompleted)
        assert completed.success_reason_json is not None
        parsed = json.loads(completed.success_reason_json)
        assert parsed["action"] == "processed"
        assert parsed["fields_added"] == ["processed", "amount_usd"]

    def test_success_reason_none_when_not_provided(
        self,
        recorder: LandscapeRecorder,
    ) -> None:
        """success_reason_json is NULL when transform doesn't provide it."""
        run_id = recorder.create_run(
            settings_hash="test",
            settings_snapshot={},
            code_version="test",
        )
        recorder.register_node(
            node_id="transform_1",
            run_id=run_id,
            node_type=NodeType.TRANSFORM,
            plugin_name="passthrough",
            config={},
        )
        row_id = recorder.create_source_row(
            run_id=run_id,
            row_index=0,
            raw_data={"x": 1},
        )
        token = recorder.create_token(
            run_id=run_id,
            row_id=row_id,
            row_data={"x": 1},
        )
        state = recorder.create_node_state(
            token_id=token.token_id,
            node_id="transform_1",
            run_id=run_id,
            step_index=0,
            input_data={"x": 1},
        )

        # Complete WITHOUT success_reason
        completed = recorder.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"x": 1},
            duration_ms=1.0,
        )

        assert isinstance(completed, NodeStateCompleted)
        assert completed.success_reason_json is None

    def test_validation_warnings_captured(
        self,
        recorder: LandscapeRecorder,
    ) -> None:
        """validation_warnings flow through to audit trail."""
        run_id = recorder.create_run(
            settings_hash="test",
            settings_snapshot={},
            code_version="test",
        )
        recorder.register_node(
            node_id="transform_1",
            run_id=run_id,
            node_type=NodeType.TRANSFORM,
            plugin_name="data_quality_transform",
            config={},
        )
        row_id = recorder.create_source_row(
            run_id=run_id,
            row_index=0,
            raw_data={"amount": 950},
        )
        token = recorder.create_token(
            run_id=run_id,
            row_id=row_id,
            row_data={"amount": 950},
        )
        state = recorder.create_node_state(
            token_id=token.token_id,
            node_id="transform_1",
            run_id=run_id,
            step_index=0,
            input_data={"amount": 950},
        )

        success_reason: TransformSuccessReason = {
            "action": "validated",
            "validation_warnings": ["amount near threshold (950 of 1000 limit)"],
        }

        completed = recorder.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"amount": 950},
            duration_ms=2.0,
            success_reason=success_reason,
        )

        assert isinstance(completed, NodeStateCompleted)
        assert completed.success_reason_json is not None
        parsed = json.loads(completed.success_reason_json)
        assert parsed["action"] == "validated"
        assert len(parsed["validation_warnings"]) == 1
        assert "950" in parsed["validation_warnings"][0]

    def test_success_reason_round_trips_through_repository(
        self,
        recorder: LandscapeRecorder,
    ) -> None:
        """success_reason survives write → read via repository."""
        run_id = recorder.create_run(
            settings_hash="test",
            settings_snapshot={},
            code_version="test",
        )
        recorder.register_node(
            node_id="transform_1",
            run_id=run_id,
            node_type=NodeType.TRANSFORM,
            plugin_name="test_transform",
            config={},
        )
        row_id = recorder.create_source_row(
            run_id=run_id,
            row_index=0,
            raw_data={"x": 1},
        )
        token = recorder.create_token(
            run_id=run_id,
            row_id=row_id,
            row_data={"x": 1},
        )
        state = recorder.create_node_state(
            token_id=token.token_id,
            node_id="transform_1",
            run_id=run_id,
            step_index=0,
            input_data={"x": 1},
        )

        success_reason: TransformSuccessReason = {
            "action": "enriched",
            "fields_added": ["enrichment_score"],
            "metadata": {"source": "external_api"},
        }

        # Write
        recorder.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"x": 1, "enrichment_score": 0.95},
            duration_ms=10.0,
            success_reason=success_reason,
        )

        # Read back via get_node_state (uses repository)
        loaded = recorder.get_node_state(state.state_id)
        assert isinstance(loaded, NodeStateCompleted)
        assert loaded.success_reason_json is not None
        parsed = json.loads(loaded.success_reason_json)
        assert parsed["action"] == "enriched"
        assert parsed["fields_added"] == ["enrichment_score"]
        assert parsed["metadata"]["source"] == "external_api"
```

**Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/engine/test_transform_success_reason.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/engine/test_transform_success_reason.py
git commit -m "$(cat <<'EOF'
test(engine): add integration tests for TransformSuccessReason

Verify success_reason flows from transform through recorder to
audit trail correctly. Tests cover:
- success_reason stored when provided
- success_reason_json NULL when not provided
- validation_warnings captured correctly
- Round-trip through repository (write → read)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Final Verification

**Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x --tb=short`
Expected: PASS

**Step 2: Run mypy**

Run: `.venv/bin/python -m mypy src/elspeth/`
Expected: No new errors

**Step 3: Run ruff**

Run: `.venv/bin/python -m ruff check src/elspeth/`
Expected: No errors

**Step 4: Close bead**

```bash
bd close <bead-id> --reason="Wired up TransformSuccessReason: schema, repository, recorder, executors. Transforms can now report action, fields_modified, validation_warnings."
```

---

## Summary

| Component | Change |
|-----------|--------|
| `contracts/errors.py` | Rename to `TransformSuccessReason`, add `TransformActionCategory` |
| `contracts/results.py` | Add `success_reason` field and factory parameters |
| `contracts/audit.py` | Add `success_reason_json` to `NodeStateCompleted` |
| `core/landscape/schema.py` | Add `success_reason_json` column |
| `core/landscape/repositories.py` | Load `success_reason_json` in `NodeStateRepository` |
| `core/landscape/recorder.py` | Accept and store `success_reason` |
| `engine/executors.py` | Pass `success_reason` through to recorder (lines 353, 1199) |

**Key Design Decisions:**

1. **Separate field from error reason** - `success_reason` is distinct from `reason` (errors). Clear semantics, no type union confusion.

2. **Optional everywhere** - Existing transforms work unchanged. Success reason is additive.

3. **Literal + string for action** - `TransformActionCategory` provides common values, but custom strings allowed for plugin-specific actions.

4. **Renamed to TransformSuccessReason** - Avoids confusion with `TransformErrorReason`.

5. **validation_warnings not validation_errors** - Emphasizes these are non-blocking (warnings that don't fail the row).

6. **Raw JSON string in MCP output** - Consistent with `error_json` and `context_after_json`. Consumers parse as needed.

**Audit Trail Value:**

```sql
-- Find transforms that skipped processing
SELECT * FROM node_states
WHERE json_extract(success_reason_json, '$.action') = 'skipped';

-- Find rows with data quality warnings
SELECT * FROM node_states
WHERE json_extract(success_reason_json, '$.validation_warnings') IS NOT NULL;

-- See which fields a transform modified
SELECT json_extract(success_reason_json, '$.fields_modified') AS modified
FROM node_states
WHERE state_id = '...';
```
