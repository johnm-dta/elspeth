# Implementation Plan: External Call Recording API

**Bug:** P0-2026-01-19-missing-external-call-recording-api.md
**Estimated Time:** 4-6 hours
**Complexity:** Medium
**Risk:** Low (additive change)

## Summary

The `calls` table schema exists and `get_calls()` works, but there's no `record_call()` method to insert records. This plan adds the missing write API.

## Prerequisites

- Understand the existing schema (`calls_table` in `schema.py:162-178`)
- Understand the existing read API (`get_calls()` in `recorder.py:1922-1957`)
- Understand the enum types (`CallType`, `CallStatus` in `contracts/enums.py`)

## Implementation Steps

### Step 1: Add `record_call()` method to LandscapeRecorder

**File:** `src/elspeth/core/landscape/recorder.py`

**Location:** After `get_calls()` method (around line 1957)

**Signature:**
```python
def record_call(
    self,
    state_id: str,
    call_index: int,
    call_type: CallType,
    status: CallStatus,
    request_data: dict[str, Any],
    response_data: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    latency_ms: float | None = None,
    *,
    request_ref: str | None = None,
    response_ref: str | None = None,
) -> Call:
    """Record an external call for a node state.

    Args:
        state_id: The node_state this call belongs to
        call_index: 0-based index of this call within the state
        call_type: Type of external call (LLM, HTTP, SQL, FILESYSTEM)
        status: Outcome of the call (SUCCESS, ERROR)
        request_data: Request payload (will be hashed)
        response_data: Response payload (will be hashed, optional for errors)
        error: Error details if status is ERROR (stored as JSON)
        latency_ms: Call duration in milliseconds
        request_ref: Optional payload store reference for request
        response_ref: Optional payload store reference for response

    Returns:
        The recorded Call model

    Note:
        Duplicate (state_id, call_index) will raise IntegrityError from SQLAlchemy.
        Invalid state_id will raise IntegrityError due to foreign key constraint.
    """
```

**Implementation:**
```python
def record_call(
    self,
    state_id: str,
    call_index: int,
    call_type: CallType,
    status: CallStatus,
    request_data: dict[str, Any],
    response_data: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    latency_ms: float | None = None,
    *,
    request_ref: str | None = None,
    response_ref: str | None = None,
) -> Call:
    """Record an external call for a node state."""
    # Note: canonical_json and stable_hash are already imported at module level
    call_id = _generate_id()  # Module-level function, not a method
    now = _now()              # Use the existing helper for UTC timestamps

    # Hash request (always required)
    request_hash = stable_hash(request_data)

    # Hash response (optional - None for errors without response)
    response_hash = stable_hash(response_data) if response_data is not None else None

    # Serialize error if present
    error_json = canonical_json(error) if error is not None else None

    values = {
        "call_id": call_id,
        "state_id": state_id,
        "call_index": call_index,
        "call_type": call_type.value,  # Store enum value
        "status": status.value,        # Store enum value
        "request_hash": request_hash,
        "request_ref": request_ref,
        "response_hash": response_hash,
        "response_ref": response_ref,
        "error_json": error_json,
        "latency_ms": latency_ms,
        "created_at": now,
    }

    with self._db.connection() as conn:
        conn.execute(calls_table.insert().values(**values))

    return Call(
        call_id=call_id,
        state_id=state_id,
        call_index=call_index,
        call_type=call_type.value,
        status=status.value,
        request_hash=request_hash,
        request_ref=request_ref,
        response_hash=response_hash,
        response_ref=response_ref,
        error_json=error_json,
        latency_ms=latency_ms,
        created_at=now,
    )
```

### Step 2: Add imports to recorder.py

**Add to the existing `from elspeth.contracts import (...)` block (around line 20-47):**
```python
from elspeth.contracts import (
    # ... existing imports ...
    CallStatus,
    CallType,
    # ... rest of existing imports ...
)
```

**Note:** Import from `elspeth.contracts`, NOT from `elspeth.contracts.enums`. This follows the established pattern in the codebase where all public contracts are re-exported from the package root.

### Step 3: Export from landscape package

**File:** `src/elspeth/core/landscape/__init__.py`

Add `CallType` and `CallStatus` to exports if not already present (for convenience of callers).

### Step 4: Add unit tests

**File:** `tests/core/landscape/test_recorder_calls.py` (new file)

```python
"""Tests for external call recording API."""

import pytest

from elspeth.contracts import CallStatus, CallType
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder


class TestRecordCall:
    """Tests for LandscapeRecorder.record_call()."""

    @pytest.fixture
    def recorder(self) -> LandscapeRecorder:
        """Create recorder with in-memory DB."""
        db = LandscapeDB.in_memory()
        return LandscapeRecorder(db)

    @pytest.fixture
    def state_id(self, recorder: LandscapeRecorder) -> str:
        """Create a node state to attach calls to."""
        schema = SchemaConfig.from_dict({"fields": "dynamic"})
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="llm_transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=schema,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"input": "test"},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            step_index=0,
            input_data={"input": "test"},
        )
        return state.state_id

    def test_record_successful_llm_call(self, recorder: LandscapeRecorder, state_id: str):
        """Test recording a successful LLM call."""
        call = recorder.record_call(
            state_id=state_id,
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data={"model": "gpt-4", "prompt": "Hello"},
            response_data={"completion": "Hi there!"},
            latency_ms=150.5,
        )

        assert call.call_id is not None
        assert call.state_id == state_id
        assert call.call_index == 0
        assert call.call_type == "llm"
        assert call.status == "success"
        assert call.request_hash is not None
        assert call.response_hash is not None
        assert call.latency_ms == 150.5
        assert call.error_json is None

    def test_record_failed_call_with_error(self, recorder: LandscapeRecorder, state_id: str):
        """Test recording a failed call with error details."""
        call = recorder.record_call(
            state_id=state_id,
            call_index=0,
            call_type=CallType.HTTP,
            status=CallStatus.ERROR,
            request_data={"url": "https://api.example.com"},
            error={"code": 500, "message": "Internal Server Error"},
            latency_ms=50.0,
        )

        assert call.status == "error"
        assert call.response_hash is None
        assert call.error_json is not None
        assert "500" in call.error_json

    def test_multiple_calls_same_state(self, recorder: LandscapeRecorder, state_id: str):
        """Test recording multiple calls for the same state."""
        call1 = recorder.record_call(
            state_id=state_id,
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data={"prompt": "First"},
            response_data={"response": "First response"},
        )
        call2 = recorder.record_call(
            state_id=state_id,
            call_index=1,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data={"prompt": "Second"},
            response_data={"response": "Second response"},
        )

        # Verify via get_calls
        calls = recorder.get_calls(state_id)
        assert len(calls) == 2
        assert calls[0].call_index == 0
        assert calls[1].call_index == 1

    def test_call_with_payload_refs(self, recorder: LandscapeRecorder, state_id: str):
        """Test recording calls with payload store references."""
        call = recorder.record_call(
            state_id=state_id,
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data={"prompt": "Large prompt..."},
            response_data={"response": "Large response..."},
            request_ref="sha256:abc123...",
            response_ref="sha256:def456...",
        )

        assert call.request_ref == "sha256:abc123..."
        assert call.response_ref == "sha256:def456..."

    def test_duplicate_call_index_raises_integrity_error(
        self, recorder: LandscapeRecorder, state_id: str
    ):
        """Test that duplicate (state_id, call_index) raises IntegrityError."""
        from sqlalchemy.exc import IntegrityError

        # First call succeeds
        recorder.record_call(
            state_id=state_id,
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data={"prompt": "First"},
            response_data={"response": "First"},
        )

        # Duplicate call_index fails
        with pytest.raises(IntegrityError):
            recorder.record_call(
                state_id=state_id,
                call_index=0,  # Same index!
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_data={"prompt": "Second"},
                response_data={"response": "Second"},
            )

    def test_invalid_state_id_raises_integrity_error(self, recorder: LandscapeRecorder):
        """Test that invalid state_id raises IntegrityError (FK constraint)."""
        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError):
            recorder.record_call(
                state_id="nonexistent_state_id",
                call_index=0,
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_data={"prompt": "Test"},
                response_data={"response": "Test"},
            )
```

### Step 5: Update PluginContext (DEFERRED - not this PR)

**File:** `src/elspeth/plugins/context.py`

> **Note:** This step is deferred to a follow-up PR. The `record_call()` method on LandscapeRecorder is sufficient for Phase 6 initial integration. The engine/executor can call it directly with explicit `state_id` and `call_index` tracking.

A future PR could add a convenience method to PluginContext:

```python
def record_external_call(
    self,
    state_id: str,  # Must be passed explicitly
    call_type: CallType,
    request_data: dict[str, Any],
    response_data: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    latency_ms: float | None = None,
) -> Call:
    """Record an external call for the current node state.

    Requires state_id to be passed explicitly since PluginContext
    doesn't inherently track which node state is currently active.
    The call_index would need to be tracked per-state.
    """
    ...
```

**Challenges:**
- PluginContext doesn't currently track `state_id` (it only has `node_id`)
- `call_index` management requires per-state tracking
- Better handled at the executor level where state lifecycle is managed

## Testing Checklist

- [ ] `record_call()` inserts row into `calls` table
- [ ] `get_calls()` returns the recorded call with correct fields
- [ ] Request/response data is hashed correctly
- [ ] Error JSON is serialized correctly via `canonical_json()`
- [ ] Enum values are stored as strings (`.value`)
- [ ] Duplicate `(state_id, call_index)` raises `IntegrityError` (DB constraint)
- [ ] Invalid `state_id` raises `IntegrityError` (foreign key constraint)
- [ ] `LandscapeExporter` already handles calls via existing `get_calls()` - verify no changes needed

## Run Tests

```bash
.venv/bin/python -m pytest tests/core/landscape/test_recorder_calls.py -v
.venv/bin/python -m pytest tests/core/landscape/ -v
```

## Acceptance Criteria

1. ✅ `LandscapeRecorder.record_call()` method exists and works
2. ✅ Recorded calls are retrievable via `get_calls()`
3. ✅ All fields (hashes, refs, error, latency) are stored correctly
4. ✅ Unit tests pass
5. ✅ No changes to existing behavior

## Future Work (Not This PR)

- Integrate with LLM transform executor (Phase 6)
- Add call recording to HTTP client wrapper
- Add payload store integration for large request/response bodies
- Add `record_external_call()` convenience method to PluginContext (see deferred Step 5)

---

## Implementation Summary

**Status:** Completed
**Commits:** See git history for this feature
**Notes:** Added record_call() method to LandscapeRecorder, completing the external call recording API for audit trail purposes.
