# Implementation Plan: Fix get_calls() Enum Type Coercion

**Bug:** P2-2026-01-19-recorder-get-calls-enum-mismatch.md
**Estimated Time:** 30 minutes
**Complexity:** Very Low
**Risk:** Very Low (2-line fix)

## Summary

`LandscapeRecorder.get_calls()` returns `call_type` and `status` as raw strings instead of `CallType` and `CallStatus` enums. This violates the strict audit contract and creates inconsistency with the repository layer (which does coerce).

## Root Cause

The `get_calls()` method constructs `Call` objects directly from DB row values without converting string values to enums:

```python
# Current (broken):
Call(
    call_type=r.call_type,  # Returns "llm" instead of CallType.LLM
    status=r.status,        # Returns "success" instead of CallStatus.SUCCESS
    ...
)
```

## Implementation Steps

### Step 1: Add enum coercion in get_calls()

**File:** `src/elspeth/core/landscape/recorder.py`

**Location:** `get_calls()` method, around line 1943-1958

**Change from:**
```python
return [
    Call(
        call_id=r.call_id,
        state_id=r.state_id,
        call_index=r.call_index,
        call_type=r.call_type,      # ❌ Raw string
        status=r.status,            # ❌ Raw string
        request_hash=r.request_hash,
        request_ref=r.request_ref,
        response_hash=r.response_hash,
        response_ref=r.response_ref,
        error_json=r.error_json,
        latency_ms=r.latency_ms,
        created_at=r.created_at,
    )
    for r in db_rows
]
```

**Change to:**
```python
return [
    Call(
        call_id=r.call_id,
        state_id=r.state_id,
        call_index=r.call_index,
        call_type=CallType(r.call_type),      # ✅ Coerce to enum
        status=CallStatus(r.status),          # ✅ Coerce to enum
        request_hash=r.request_hash,
        request_ref=r.request_ref,
        response_hash=r.response_hash,
        response_ref=r.response_ref,
        error_json=r.error_json,
        latency_ms=r.latency_ms,
        created_at=r.created_at,
    )
    for r in db_rows
]
```

### Step 2: Verify imports exist

**File:** `src/elspeth/core/landscape/recorder.py`

Check that `CallType` and `CallStatus` are imported. They should be in the existing imports block:

```python
from elspeth.contracts import (
    # ... other imports ...
    CallStatus,
    CallType,
    # ... other imports ...
)
```

If not present, add them to the import block.

### Step 3: Add unit test

**File:** `tests/core/landscape/test_recorder.py` (add to existing file)

```python
def test_get_calls_returns_enum_types(self) -> None:
    """Verify get_calls() returns CallType and CallStatus enums, not strings."""
    from elspeth.contracts import CallStatus, CallType

    # Setup: create a state and record a call
    # (Assuming record_call() is implemented per the other plan)
    # If not implemented yet, insert directly into calls_table

    schema = SchemaConfig.from_dict({"fields": "dynamic"})
    run = self.recorder.begin_run(config={}, canonical_version="v1")
    node = self.recorder.register_node(
        run_id=run.run_id,
        plugin_name="llm_transform",
        node_type="transform",
        plugin_version="1.0",
        config={},
        schema_config=schema,
    )
    row = self.recorder.create_row(
        run_id=run.run_id,
        source_node_id=node.node_id,
        row_index=0,
        data={"input": "test"},
    )
    token = self.recorder.create_token(row_id=row.row_id)
    state = self.recorder.begin_node_state(
        token_id=token.token_id,
        node_id=node.node_id,
        step_index=0,
        input_data={"input": "test"},
    )

    # Insert call record directly (or use record_call if available)
    from datetime import datetime, timezone
    from elspeth.core.landscape.schema import calls_table

    with self.db.connection() as conn:
        conn.execute(
            calls_table.insert().values(
                call_id="test_call_1",
                state_id=state.state_id,
                call_index=0,
                call_type="llm",        # Stored as string in DB
                status="success",       # Stored as string in DB
                request_hash="abc123",
                created_at=datetime.now(timezone.utc),
            )
        )

    # Act
    calls = self.recorder.get_calls(state.state_id)

    # Assert
    assert len(calls) == 1
    call = calls[0]

    # These should be enums, not strings
    assert isinstance(call.call_type, CallType), f"Expected CallType enum, got {type(call.call_type)}"
    assert isinstance(call.status, CallStatus), f"Expected CallStatus enum, got {type(call.status)}"
    assert call.call_type == CallType.LLM
    assert call.status == CallStatus.SUCCESS


def test_get_calls_invalid_enum_crashes(self) -> None:
    """Verify get_calls() crashes on invalid enum values (Tier 1 invariant)."""
    # Insert a call with invalid enum value
    # ... setup state ...

    with self.db.connection() as conn:
        conn.execute(
            calls_table.insert().values(
                call_id="bad_call",
                state_id=state.state_id,
                call_index=0,
                call_type="invalid_type",  # Not a valid CallType!
                status="success",
                request_hash="abc123",
                created_at=datetime.now(timezone.utc),
            )
        )

    # Should crash when trying to coerce invalid enum
    with pytest.raises(ValueError):
        self.recorder.get_calls(state.state_id)
```

### Step 4: Verify Call dataclass accepts enums

**File:** `src/elspeth/contracts/audit.py`

Check that the `Call` dataclass type hints allow enums:

```python
@dataclass
class Call:
    call_id: str
    state_id: str
    call_index: int
    call_type: CallType  # Should be CallType, not str
    status: CallStatus   # Should be CallStatus, not str
    ...
```

If the type hints say `str`, update them to use the enum types.

## Testing Checklist

- [ ] `get_calls()` returns `CallType` enum for `call_type`
- [ ] `get_calls()` returns `CallStatus` enum for `status`
- [ ] Invalid enum value in DB causes `ValueError` on read (Tier 1 crash policy)
- [ ] Existing tests still pass
- [ ] Type hints in `Call` dataclass match enum types

## Run Tests

```bash
# Run recorder tests
.venv/bin/python -m pytest tests/core/landscape/test_recorder.py -v -k call

# Run all landscape tests
.venv/bin/python -m pytest tests/core/landscape/ -v

# Type check
.venv/bin/python -m mypy src/elspeth/core/landscape/recorder.py
```

## Acceptance Criteria

1. ✅ `call.call_type` is a `CallType` enum instance
2. ✅ `call.status` is a `CallStatus` enum instance
3. ✅ Invalid enum strings crash with `ValueError` (Tier 1 policy)
4. ✅ Unit test verifies enum types

## Notes

**Why this matters for LLM integration:**
After implementing `record_call()`, you'll want to retrieve calls via `get_calls()` for:
- `explain()` lineage queries
- Export to audit trail JSON
- TUI display

If these return strings instead of enums, downstream code that expects enums will fail or behave unexpectedly.

**Tier 1 crash policy:**
Per CLAUDE.md, invalid data in the audit database should crash immediately. If someone manually inserted `call_type="bogus"` into the calls table, `CallType("bogus")` will raise `ValueError` - this is correct behavior.

---

## Implementation Summary

**Status:** Completed
**Commits:** See git history for this feature
**Notes:** Fixed get_calls() to properly coerce call_type and status from raw strings to their respective enum types, maintaining consistency with the audit contract.
