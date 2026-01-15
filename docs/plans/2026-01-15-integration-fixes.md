# Integration Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all 25 integration anti-patterns identified in the 2026-01-15 integration audit, restoring strict type contracts at subsystem boundaries.

**Architecture:** Bottom-up approach - fix Core/Landscape first (data layer), then Engine (processing layer), then Plugins (contract layer), then TUI/CLI (presentation layer). This ensures each layer has strong contracts before the layer above consumes it.

**Tech Stack:** Python 3.11+, Pydantic, dataclasses, enums, TypedDict, pytest

**Reference:** `docs/audit/2026-01-15-integration-audit.md`

---

## Phase 1: Critical Issues (6 hours)

### Task 1: Create RowDataState Enum and RowDataResult Type

**Files:**
- Create: `src/elspeth/core/landscape/row_data.py`
- Test: `tests/core/landscape/test_row_data.py`

**Step 1: Write failing test for RowDataState enum**

```python
# tests/core/landscape/test_row_data.py
"""Tests for RowDataState and RowDataResult."""
import pytest
from elspeth.core.landscape.row_data import RowDataState, RowDataResult


class TestRowDataState:
    """Tests for RowDataState enum."""

    def test_all_states_defined(self):
        """Verify all expected states exist."""
        assert RowDataState.AVAILABLE.value == "available"
        assert RowDataState.PURGED.value == "purged"
        assert RowDataState.NEVER_STORED.value == "never_stored"
        assert RowDataState.STORE_NOT_CONFIGURED.value == "store_not_configured"
        assert RowDataState.ROW_NOT_FOUND.value == "row_not_found"

    def test_state_is_str_enum(self):
        """Enum values should be JSON-serializable strings."""
        for state in RowDataState:
            assert isinstance(state.value, str)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/landscape/test_row_data.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'elspeth.core.landscape.row_data'"

**Step 3: Write minimal implementation**

```python
# src/elspeth/core/landscape/row_data.py
"""Row data retrieval types for explicit state handling.

This module provides discriminated result types for get_row_data() to replace
ambiguous None returns. Per ELSPETH's auditability standard, callers must
always know WHY data is unavailable, not just that it is.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any


class RowDataState(str, Enum):
    """Explicit states for row data retrieval.

    These states make it impossible for callers to confuse "data was purged
    but hash preserved" with "configuration error" or "row doesn't exist".
    """

    AVAILABLE = "available"
    """Payload retrieved successfully."""

    PURGED = "purged"
    """Row exists and hash preserved, but payload was deleted per retention policy."""

    NEVER_STORED = "never_stored"
    """Row exists but source_data_ref is None (payload was never stored)."""

    STORE_NOT_CONFIGURED = "store_not_configured"
    """Row exists but PayloadStore not configured on this recorder."""

    ROW_NOT_FOUND = "row_not_found"
    """No row with this ID exists in the database."""


@dataclass(frozen=True)
class RowDataResult:
    """Result of row data retrieval with explicit state.

    Replaces ambiguous `dict | None` return type. Callers MUST check state
    before accessing data.

    Example:
        result = recorder.get_row_data(row_id)
        if result.state == RowDataState.AVAILABLE:
            process(result.data)
        elif result.state == RowDataState.PURGED:
            show_message("Data purged but hash preserved for audit")
        else:
            show_error(f"Data unavailable: {result.state.value}")
    """

    state: RowDataState
    data: dict[str, Any] | None

    def __post_init__(self) -> None:
        """Validate state/data consistency."""
        if self.state == RowDataState.AVAILABLE and self.data is None:
            raise ValueError("AVAILABLE state requires non-None data")
        if self.state != RowDataState.AVAILABLE and self.data is not None:
            raise ValueError(f"{self.state} state requires None data")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/landscape/test_row_data.py -v`
Expected: PASS

**Step 5: Write test for RowDataResult validation**

```python
# Add to tests/core/landscape/test_row_data.py

class TestRowDataResult:
    """Tests for RowDataResult dataclass."""

    def test_available_with_data(self):
        """AVAILABLE state must have data."""
        result = RowDataResult(state=RowDataState.AVAILABLE, data={"key": "value"})
        assert result.data == {"key": "value"}

    def test_available_without_data_raises(self):
        """AVAILABLE state without data should raise."""
        with pytest.raises(ValueError, match="AVAILABLE state requires non-None data"):
            RowDataResult(state=RowDataState.AVAILABLE, data=None)

    def test_non_available_with_data_raises(self):
        """Non-AVAILABLE states with data should raise."""
        with pytest.raises(ValueError, match="PURGED state requires None data"):
            RowDataResult(state=RowDataState.PURGED, data={"unexpected": "data"})

    def test_purged_state(self):
        """PURGED state should have None data."""
        result = RowDataResult(state=RowDataState.PURGED, data=None)
        assert result.state == RowDataState.PURGED
        assert result.data is None

    def test_frozen(self):
        """Result should be immutable."""
        result = RowDataResult(state=RowDataState.PURGED, data=None)
        with pytest.raises(AttributeError):
            result.state = RowDataState.AVAILABLE  # type: ignore
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/core/landscape/test_row_data.py::TestRowDataResult -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/elspeth/core/landscape/row_data.py tests/core/landscape/test_row_data.py
git commit -m "$(cat <<'EOF'
feat(landscape): add RowDataState enum and RowDataResult type

Introduces explicit state handling for row data retrieval to replace
ambiguous None returns. Callers can now distinguish between:
- Data available
- Data purged (hash preserved)
- Data never stored
- Payload store not configured
- Row not found

Part of CRIT-001 fix from integration audit.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Update get_row_data() to Return RowDataResult

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py:1472-1497`
- Test: `tests/core/landscape/test_recorder.py` (add new tests)

**Step 1: Write failing test for new get_row_data() signature**

```python
# Add to tests/core/landscape/test_recorder.py (or create new test file)
# tests/core/landscape/test_recorder_row_data.py
"""Tests for LandscapeRecorder.get_row_data() with explicit states."""
import pytest
from unittest.mock import Mock, MagicMock
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.core.landscape.row_data import RowDataState, RowDataResult
from elspeth.core.landscape.models import Row


class TestGetRowDataExplicitStates:
    """Tests for get_row_data() returning RowDataResult."""

    def test_row_not_found(self, recorder_with_db):
        """Returns ROW_NOT_FOUND when row doesn't exist."""
        result = recorder_with_db.get_row_data("nonexistent-row")
        assert isinstance(result, RowDataResult)
        assert result.state == RowDataState.ROW_NOT_FOUND
        assert result.data is None

    def test_never_stored(self, recorder_with_db, row_without_ref):
        """Returns NEVER_STORED when source_data_ref is None."""
        # Insert row with no source_data_ref
        result = recorder_with_db.get_row_data(row_without_ref.row_id)
        assert result.state == RowDataState.NEVER_STORED
        assert result.data is None

    def test_store_not_configured(self, recorder_no_payload_store, row_with_ref):
        """Returns STORE_NOT_CONFIGURED when payload_store is None."""
        result = recorder_no_payload_store.get_row_data(row_with_ref.row_id)
        assert result.state == RowDataState.STORE_NOT_CONFIGURED
        assert result.data is None

    def test_purged(self, recorder_with_db, row_with_ref):
        """Returns PURGED when payload_store raises KeyError."""
        # Mock payload store to raise KeyError (payload deleted)
        recorder_with_db._payload_store.retrieve = Mock(side_effect=KeyError("purged"))
        result = recorder_with_db.get_row_data(row_with_ref.row_id)
        assert result.state == RowDataState.PURGED
        assert result.data is None

    def test_available(self, recorder_with_db, row_with_ref):
        """Returns AVAILABLE with data when payload exists."""
        test_data = {"field": "value", "number": 42}
        import json
        recorder_with_db._payload_store.retrieve = Mock(
            return_value=json.dumps(test_data).encode()
        )
        result = recorder_with_db.get_row_data(row_with_ref.row_id)
        assert result.state == RowDataState.AVAILABLE
        assert result.data == test_data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/landscape/test_recorder_row_data.py -v`
Expected: FAIL with assertion error (old signature returns dict | None)

**Step 3: Update get_row_data() implementation**

```python
# In src/elspeth/core/landscape/recorder.py
# Replace the existing get_row_data method (lines 1472-1497)

from elspeth.core.landscape.row_data import RowDataState, RowDataResult

def get_row_data(self, row_id: str) -> RowDataResult:
    """Get the payload data for a row with explicit state.

    Returns a RowDataResult with explicit state indicating why data
    may be unavailable. This replaces the previous ambiguous None return.

    Args:
        row_id: Row ID

    Returns:
        RowDataResult with state and data (if available)
    """
    row = self.get_row(row_id)
    if row is None:
        return RowDataResult(state=RowDataState.ROW_NOT_FOUND, data=None)

    if row.source_data_ref is None:
        return RowDataResult(state=RowDataState.NEVER_STORED, data=None)

    if self._payload_store is None:
        return RowDataResult(state=RowDataState.STORE_NOT_CONFIGURED, data=None)

    try:
        import json
        payload_bytes = self._payload_store.retrieve(row.source_data_ref)
        data = json.loads(payload_bytes.decode("utf-8"))
        return RowDataResult(state=RowDataState.AVAILABLE, data=data)
    except KeyError:
        return RowDataResult(state=RowDataState.PURGED, data=None)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/landscape/test_recorder_row_data.py -v`
Expected: PASS

**Step 5: Export from landscape package**

```python
# Update src/elspeth/core/landscape/__init__.py
# Add to imports:
from elspeth.core.landscape.row_data import RowDataResult, RowDataState

# Add to __all__:
__all__ = [
    # ... existing exports ...
    "RowDataResult",
    "RowDataState",
]
```

**Step 6: Run full landscape tests**

Run: `pytest tests/core/landscape/ -v`
Expected: PASS (or identify callers that need updating)

**Step 7: Commit**

```bash
git add src/elspeth/core/landscape/recorder.py src/elspeth/core/landscape/__init__.py tests/core/landscape/
git commit -m "$(cat <<'EOF'
fix(landscape): get_row_data() returns explicit RowDataResult

BREAKING CHANGE: get_row_data() now returns RowDataResult instead of
dict | None. Callers must update to check result.state before accessing
result.data.

This fixes CRIT-001 from the integration audit - callers can now
distinguish between purged data, never-stored data, missing payload
store configuration, and row not found.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Add close() to Transform/Gate Protocols

**Files:**
- Modify: `src/elspeth/plugins/protocols.py:89-147` (TransformProtocol)
- Modify: `src/elspeth/plugins/protocols.py:150-214` (GateProtocol)
- Test: `tests/plugins/test_protocols.py`

**Step 1: Write test for close() in TransformProtocol**

```python
# tests/plugins/test_protocol_lifecycle.py
"""Tests for plugin lifecycle methods in protocols."""
import pytest
from typing import Protocol, runtime_checkable
from elspeth.plugins.protocols import TransformProtocol, GateProtocol
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult, GateResult


class TestTransformProtocolLifecycle:
    """Tests for TransformProtocol.close()."""

    def test_protocol_has_close_method(self):
        """TransformProtocol should define close()."""
        assert hasattr(TransformProtocol, "close")

    def test_transform_with_close_satisfies_protocol(self):
        """A class with close() should satisfy the protocol."""
        class MyTransform:
            name = "test"
            input_schema = None
            output_schema = None
            determinism = "deterministic"
            plugin_version = "1.0.0"

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                return TransformResult.success(row)

            def close(self) -> None:
                pass

        # Should not raise - satisfies protocol
        transform: TransformProtocol = MyTransform()
        transform.close()


class TestGateProtocolLifecycle:
    """Tests for GateProtocol.close()."""

    def test_protocol_has_close_method(self):
        """GateProtocol should define close()."""
        assert hasattr(GateProtocol, "close")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/test_protocol_lifecycle.py -v`
Expected: FAIL with "AssertionError" (close not in protocol)

**Step 3: Add close() to TransformProtocol**

```python
# In src/elspeth/plugins/protocols.py
# Update TransformProtocol class (around line 89)

class TransformProtocol(Protocol):
    """Protocol for stateless row transforms.

    Transforms process one row and emit one row (possibly modified).
    They are stateless between rows.

    Lifecycle:
        - __init__(config): Called once at pipeline construction
        - process(row, ctx): Called for each row
        - close(): Called at pipeline completion for cleanup

    Example:
        class EnrichTransform:
            name = "enrich"
            input_schema = InputSchema
            output_schema = OutputSchema

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                enriched = {**row, "timestamp": datetime.now().isoformat()}
                return TransformResult.success(enriched)

            def close(self) -> None:
                pass  # No resources to clean up
    """

    name: str
    input_schema: type | None
    output_schema: type | None
    determinism: Determinism | str
    plugin_version: str

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Process a single row."""
        ...

    def close(self) -> None:
        """Clean up resources after pipeline completion.

        Called once after all rows have been processed. Use for closing
        connections, flushing buffers, or releasing external resources.
        """
        ...
```

**Step 4: Add close() to GateProtocol**

```python
# In src/elspeth/plugins/protocols.py
# Update GateProtocol class (around line 150)

class GateProtocol(Protocol):
    """Protocol for gate transforms (routing decisions).

    Gates evaluate rows and decide routing. They can:
    - Continue to next transform
    - Route to a named sink
    - Fork to multiple parallel paths

    Lifecycle:
        - __init__(config): Called once at pipeline construction
        - evaluate(row, ctx): Called for each row
        - close(): Called at pipeline completion for cleanup

    Example:
        class SafetyGate:
            name = "safety"
            input_schema = InputSchema
            output_schema = OutputSchema

            def evaluate(self, row: dict, ctx: PluginContext) -> GateResult:
                if row["suspicious"]:  # Direct access, not .get()
                    return GateResult(
                        row=row,
                        action=RoutingAction.route("review"),
                    )
                return GateResult(row=row, action=RoutingAction.route("normal"))

            def close(self) -> None:
                pass
    """

    name: str
    input_schema: type | None
    output_schema: type | None
    determinism: Determinism | str
    plugin_version: str

    def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
        """Evaluate a row and decide routing."""
        ...

    def close(self) -> None:
        """Clean up resources after pipeline completion."""
        ...
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/plugins/test_protocol_lifecycle.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/elspeth/plugins/protocols.py tests/plugins/test_protocol_lifecycle.py
git commit -m "$(cat <<'EOF'
feat(plugins): add close() lifecycle method to Transform/Gate protocols

TransformProtocol and GateProtocol now define close() method for
resource cleanup, matching SinkProtocol's existing lifecycle.

This resolves CRIT-002 from the integration audit - implementations
already had close() but it wasn't part of the protocol contract.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Call close() on Transforms/Gates in Engine

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py` (add cleanup in run())
- Test: `tests/engine/test_orchestrator_cleanup.py`

**Step 1: Write test for transform cleanup**

```python
# tests/engine/test_orchestrator_cleanup.py
"""Tests for transform/gate cleanup in orchestrator."""
import pytest
from unittest.mock import Mock, MagicMock


class TestOrchestratorCleanup:
    """Tests for Orchestrator calling close() on plugins."""

    def test_transforms_closed_on_success(self, orchestrator, mock_transforms):
        """All transforms should have close() called after successful run."""
        orchestrator.run()

        for transform in mock_transforms:
            transform.close.assert_called_once()

    def test_transforms_closed_on_failure(self, orchestrator, mock_transforms, failing_source):
        """All transforms should have close() called even if run fails."""
        with pytest.raises(Exception):
            orchestrator.run()

        for transform in mock_transforms:
            transform.close.assert_called_once()

    def test_gates_closed(self, orchestrator, mock_gates):
        """Gates should have close() called after run."""
        orchestrator.run()

        for gate in mock_gates:
            gate.close.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_orchestrator_cleanup.py -v`
Expected: FAIL (close() not called)

**Step 3: Add cleanup to Orchestrator.run()**

```python
# In src/elspeth/engine/orchestrator.py
# Update the run() method to call close() on all transforms

def run(self) -> RunResult:
    """Execute the pipeline."""
    # ... existing setup code ...

    try:
        # ... existing processing code ...
        return RunResult(...)
    finally:
        # Clean up all transforms and gates
        self._cleanup_transforms()

def _cleanup_transforms(self) -> None:
    """Call close() on all transforms and gates."""
    for transform in self._config.transforms:
        try:
            if hasattr(transform, "close"):
                transform.close()
        except Exception as e:
            # Log but don't raise - cleanup should be best-effort
            self._logger.warning(
                "Transform cleanup failed",
                transform=getattr(transform, "name", str(transform)),
                error=str(e),
            )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/engine/test_orchestrator_cleanup.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/engine/orchestrator.py tests/engine/test_orchestrator_cleanup.py
git commit -m "$(cat <<'EOF'
fix(engine): call close() on transforms and gates after pipeline run

Orchestrator now calls close() on all transforms and gates in a finally
block, ensuring cleanup happens even if the pipeline fails.

Completes CRIT-002 fix - protocols define close(), engine calls it.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2: High-Severity Issues (11 hours)

### Task 5: Add RunStatus Enum and Update RunResult

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py:41-50`
- Test: `tests/engine/test_orchestrator.py`

**Step 1: Write failing test**

```python
# Add to tests/engine/test_run_status.py
"""Tests for RunStatus enum."""
import pytest
from elspeth.engine.orchestrator import RunStatus, RunResult


class TestRunStatus:
    """Tests for RunStatus enum."""

    def test_status_values(self):
        """RunStatus should have expected values."""
        assert RunStatus.COMPLETED.value == "completed"
        assert RunStatus.FAILED.value == "failed"

    def test_run_result_uses_enum(self):
        """RunResult.status should be RunStatus, not str."""
        result = RunResult(
            run_id="test",
            status=RunStatus.COMPLETED,
            rows_processed=10,
            rows_succeeded=10,
            rows_failed=0,
            rows_routed=0,
        )
        assert isinstance(result.status, RunStatus)
        assert result.status == RunStatus.COMPLETED
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_run_status.py -v`
Expected: FAIL

**Step 3: Implement RunStatus enum**

```python
# In src/elspeth/engine/orchestrator.py
# Add near top of file, after imports

from enum import Enum


class RunStatus(str, Enum):
    """Terminal status for pipeline runs."""

    COMPLETED = "completed"
    """Pipeline finished processing all rows successfully."""

    FAILED = "failed"
    """Pipeline encountered a fatal error and stopped."""


@dataclass
class RunResult:
    """Result of a pipeline run."""

    run_id: str
    status: RunStatus  # Changed from str
    rows_processed: int
    rows_succeeded: int
    rows_failed: int
    rows_routed: int
```

**Step 4: Update all string usages to enum**

Search for `status="completed"` and `status="failed"` in orchestrator.py and replace with `status=RunStatus.COMPLETED` and `status=RunStatus.FAILED`.

**Step 5: Run tests**

Run: `pytest tests/engine/ -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/elspeth/engine/orchestrator.py tests/engine/test_run_status.py
git commit -m "$(cat <<'EOF'
feat(engine): add RunStatus enum, replace stringly-typed status

RunResult.status is now RunStatus enum instead of str. This catches
typos at development time and provides IDE autocomplete.

Fixes HIGH-001 (partial) from integration audit.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Use RowOutcome Enum in RowResult

**Files:**
- Modify: `src/elspeth/engine/processor.py:27-33`
- Test: `tests/engine/test_processor.py`

**Step 1: Write failing test**

```python
# Add to tests/engine/test_row_outcome.py
"""Tests for RowResult using RowOutcome enum."""
import pytest
from elspeth.engine.processor import RowResult
from elspeth.plugins.results import RowOutcome


class TestRowResultOutcome:
    """Tests for RowResult.outcome as enum."""

    def test_outcome_is_enum(self):
        """RowResult.outcome should be RowOutcome, not str."""
        # This will fail if outcome is still str
        from elspeth.engine.tokens import TokenInfo
        token = TokenInfo(row_id="r1", token_id="t1", row_data={}, branch_name=None)
        result = RowResult(
            token=token,
            final_data={},
            outcome=RowOutcome.COMPLETED,
        )
        assert isinstance(result.outcome, RowOutcome)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_row_outcome.py -v`
Expected: FAIL

**Step 3: Update RowResult**

```python
# In src/elspeth/engine/processor.py
from elspeth.plugins.results import RowOutcome

@dataclass
class RowResult:
    """Result of processing a row through the pipeline."""

    token: TokenInfo
    final_data: dict[str, Any]
    outcome: RowOutcome  # Changed from str
    sink_name: str | None = None
```

**Step 4: Update all outcome string usages**

Replace `outcome="completed"` with `outcome=RowOutcome.COMPLETED`, etc.

**Step 5: Run tests**

Run: `pytest tests/engine/ -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/elspeth/engine/processor.py tests/engine/test_row_outcome.py
git commit -m "$(cat <<'EOF'
feat(engine): use RowOutcome enum in RowResult

RowResult.outcome is now RowOutcome enum instead of str. The enum
already existed in plugins/results.py but wasn't being used.

Completes HIGH-001 fix from integration audit.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Replace hasattr Duck Typing with isinstance

**Files:**
- Modify: `src/elspeth/engine/processor.py:132-180`
- Test: `tests/engine/test_processor.py`

**Step 1: Write test for type-safe plugin detection**

```python
# tests/engine/test_plugin_detection.py
"""Tests for type-safe plugin detection in processor."""
import pytest
from elspeth.plugins.base import BaseTransform, BaseGate, BaseAggregation


class TestPluginTypeDetection:
    """Tests for isinstance-based plugin detection."""

    def test_gate_detected_by_isinstance(self, processor, mock_gate):
        """Gates should be detected via isinstance, not hasattr."""
        assert isinstance(mock_gate, BaseGate)
        # Processing should route to gate executor

    def test_aggregation_detected_by_isinstance(self, processor, mock_aggregation):
        """Aggregations should be detected via isinstance."""
        assert isinstance(mock_aggregation, BaseAggregation)

    def test_transform_detected_by_isinstance(self, processor, mock_transform):
        """Transforms should be detected via isinstance."""
        assert isinstance(mock_transform, BaseTransform)

    def test_unknown_type_raises(self, processor):
        """Unknown plugin types should raise TypeError."""
        class UnknownPlugin:
            pass

        with pytest.raises(TypeError, match="Unknown transform type"):
            processor._determine_plugin_type(UnknownPlugin())
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_plugin_detection.py -v`
Expected: FAIL

**Step 3: Update processor.py to use isinstance**

```python
# In src/elspeth/engine/processor.py
from elspeth.plugins.base import BaseTransform, BaseGate, BaseAggregation

# Replace the hasattr checks (around line 132-180) with:

for step, transform in enumerate(transforms, start=1):
    # Type-safe plugin detection using base classes
    if isinstance(transform, BaseGate):
        # Gate transform
        outcome = self._gate_executor.execute_gate(
            gate=transform,
            token=current_token,
            ctx=ctx,
            token_manager=self._token_manager,
            route_resolution_map=self._route_resolution_map,
        )
        # ... rest of gate handling ...

    elif isinstance(transform, BaseAggregation):
        # Aggregation transform
        accept_result = self._aggregation_executor.accept(
            aggregation=transform,
            token=current_token,
            ctx=ctx,
        )
        # ... rest of aggregation handling ...

    elif isinstance(transform, BaseTransform):
        # Regular transform
        result, current_token = self._transform_executor.execute_transform(
            transform=transform,
            token=current_token,
            ctx=ctx,
        )
        # ... rest of transform handling ...

    else:
        raise TypeError(
            f"Unknown transform type: {type(transform).__name__}. "
            f"Expected BaseTransform, BaseGate, or BaseAggregation."
        )
```

**Step 4: Run tests**

Run: `pytest tests/engine/ -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/engine/processor.py tests/engine/test_plugin_detection.py
git commit -m "$(cat <<'EOF'
fix(engine): use isinstance for plugin type detection

Replaces fragile hasattr("evaluate")/hasattr("accept") duck typing
with isinstance() checks against base classes. This provides:
- Type safety (mypy can verify)
- Clear error on unknown types
- No order-dependent method checks

Fixes HIGH-002 from integration audit.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Add Type Hints to PipelineConfig

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py:30-37`
- Test: Run mypy

**Step 1: Update PipelineConfig with proper types**

```python
# In src/elspeth/engine/orchestrator.py
from elspeth.plugins.protocols import (
    SourceProtocol,
    SinkProtocol,
)
from elspeth.plugins.base import BaseTransform, BaseGate, BaseAggregation
from elspeth.core.config import ElspethSettings

# Type alias for transform-like plugins
TransformLike = BaseTransform | BaseGate | BaseAggregation


@dataclass
class PipelineConfig:
    """Configuration for a pipeline run.

    All plugin fields are now properly typed for IDE support and
    static type checking.
    """

    source: SourceProtocol
    transforms: list[TransformLike]
    sinks: dict[str, SinkProtocol]
    config: ElspethSettings | None = None
```

**Step 2: Run mypy**

Run: `mypy src/elspeth/engine/orchestrator.py --strict`
Expected: No type errors related to PipelineConfig

**Step 3: Commit**

```bash
git add src/elspeth/engine/orchestrator.py
git commit -m "$(cat <<'EOF'
fix(engine): add proper type hints to PipelineConfig

PipelineConfig now uses protocol types instead of Any:
- source: SourceProtocol
- transforms: list[TransformLike]
- sinks: dict[str, SinkProtocol]
- config: ElspethSettings | None

Fixes HIGH-005 from integration audit.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Fix Plugin Manager Defensive getattr

**Files:**
- Modify: `src/elspeth/plugins/manager.py:83-90`
- Test: `tests/plugins/test_manager.py`

**Step 1: Write test for required attributes**

```python
# tests/plugins/test_manager_validation.py
"""Tests for plugin manager attribute validation."""
import pytest
from elspeth.plugins.manager import PluginManager, PluginSchema
from elspeth.plugins.enums import NodeType


class TestPluginSchemaValidation:
    """Tests for PluginSchema.from_plugin() validation."""

    def test_missing_name_raises(self):
        """Plugin without name attribute should raise ValueError."""
        class BadPlugin:
            plugin_version = "1.0.0"
            # Missing: name

        with pytest.raises(ValueError, match="must define 'name' attribute"):
            PluginSchema.from_plugin(BadPlugin, NodeType.TRANSFORM)

    def test_missing_version_raises(self):
        """Plugin without plugin_version should raise ValueError."""
        class BadPlugin:
            name = "bad"
            # Missing: plugin_version

        with pytest.raises(ValueError, match="must define 'plugin_version' attribute"):
            PluginSchema.from_plugin(BadPlugin, NodeType.TRANSFORM)

    def test_valid_plugin_succeeds(self):
        """Plugin with required attributes should succeed."""
        class GoodPlugin:
            name = "good"
            plugin_version = "1.0.0"

        schema = PluginSchema.from_plugin(GoodPlugin, NodeType.TRANSFORM)
        assert schema.name == "good"
        assert schema.version == "1.0.0"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/test_manager_validation.py -v`
Expected: FAIL (silent fallback instead of error)

**Step 3: Update from_plugin() to require attributes**

```python
# In src/elspeth/plugins/manager.py
# Update from_plugin() method (around line 83)

@classmethod
def from_plugin(cls, plugin_cls: type, node_type: NodeType) -> "PluginSchema":
    """Create schema from plugin class.

    Required attributes (will raise if missing):
    - name: str
    - plugin_version: str

    Optional attributes (have legitimate defaults):
    - determinism: defaults to DETERMINISTIC
    - input_schema: defaults to None
    - output_schema: defaults to None
    """
    # Required: name
    try:
        name = plugin_cls.name
    except AttributeError:
        raise ValueError(
            f"Plugin {plugin_cls.__name__} must define 'name' attribute. "
            f"Add: name = 'your_plugin_name' to the class."
        )

    # Required: plugin_version
    try:
        version = plugin_cls.plugin_version
    except AttributeError:
        raise ValueError(
            f"Plugin {plugin_cls.__name__} must define 'plugin_version' attribute. "
            f"Add: plugin_version = '1.0.0' to the class."
        )

    # Optional: determinism (legitimate default - most are deterministic)
    determinism = getattr(plugin_cls, "determinism", Determinism.DETERMINISTIC)

    # Optional: schemas (None is valid - means any schema accepted)
    input_schema = getattr(plugin_cls, "input_schema", None)
    output_schema = getattr(plugin_cls, "output_schema", None)

    return cls(
        name=name,
        node_type=node_type,
        version=version,
        determinism=_coerce_enum(determinism, Determinism),
        input_schema=input_schema,
        output_schema=output_schema,
    )
```

**Step 4: Run tests**

Run: `pytest tests/plugins/test_manager_validation.py -v`
Expected: PASS

**Step 5: Update _refresh_caches() similarly**

Apply same pattern to lines 149, 159, 169, 179, 189, 199 - require name attribute.

**Step 6: Commit**

```bash
git add src/elspeth/plugins/manager.py tests/plugins/test_manager_validation.py
git commit -m "$(cat <<'EOF'
fix(plugins): require name and version attributes on plugins

PluginSchema.from_plugin() now raises ValueError if plugin class
is missing required 'name' or 'plugin_version' attributes, instead
of silently falling back to __name__ or "0.0.0".

determinism still has a default (DETERMINISTIC) as this is a
legitimate default for most plugins.

Fixes HIGH-004 from integration audit.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Create LineageData TypedDict for TUI

**Files:**
- Create: `src/elspeth/tui/types.py`
- Modify: `src/elspeth/tui/widgets/lineage_tree.py`
- Test: `tests/tui/test_lineage_types.py`

**Step 1: Write test for LineageData TypedDict**

```python
# tests/tui/test_lineage_types.py
"""Tests for TUI type contracts."""
import pytest
from elspeth.tui.types import LineageData, SourceInfo, NodeInfo


class TestLineageDataContract:
    """Tests for LineageData TypedDict."""

    def test_valid_lineage_data(self):
        """Valid data should pass type checking."""
        data: LineageData = {
            "run_id": "run-123",
            "source": {"name": "csv", "node_id": "node-1"},
            "transforms": [{"name": "mapper", "node_id": "node-2"}],
            "sinks": [{"name": "output", "node_id": "node-3"}],
            "tokens": [],
        }
        assert data["run_id"] == "run-123"

    def test_missing_field_fails_at_construction(self):
        """Missing required fields should fail."""
        # This is a type-checker test - runtime won't catch it
        # but mypy will flag it
        pass
```

**Step 2: Create types module**

```python
# src/elspeth/tui/types.py
"""Type definitions for TUI data contracts.

These TypedDicts define the exact shape of data passed between
TUI components. Using direct field access (data["field"]) instead
of .get() ensures missing fields fail loudly.
"""
from typing import TypedDict, Any


class NodeInfo(TypedDict):
    """Information about a single pipeline node."""

    name: str
    node_id: str | None


class SourceInfo(TypedDict):
    """Information about the pipeline source."""

    name: str
    node_id: str | None


class LineageData(TypedDict):
    """Data contract for lineage tree display.

    All fields are required. If data is unavailable, the caller
    must handle that BEFORE constructing LineageData - not inside
    the widget via .get() defaults.
    """

    run_id: str
    source: SourceInfo
    transforms: list[NodeInfo]
    sinks: list[NodeInfo]
    tokens: list[dict[str, Any]]
```

**Step 3: Update LineageTree to use TypedDict**

```python
# In src/elspeth/tui/widgets/lineage_tree.py
from elspeth.tui.types import LineageData

class LineageTree:
    """Tree widget displaying pipeline lineage."""

    def __init__(self, lineage_data: LineageData) -> None:
        """Initialize with validated lineage data.

        Args:
            lineage_data: Must conform to LineageData TypedDict.
                         Caller is responsible for validation.
        """
        self._data = lineage_data
        self._root = self._build_tree()

    def _build_tree(self) -> TreeNode:
        """Build tree from lineage data using direct access."""
        # Direct access - fails loudly if field missing
        run_id = self._data["run_id"]

        root = TreeNode(f"Run: {run_id}")

        # Source node
        source = self._data["source"]
        source_name = source["name"]
        source_node = TreeNode(f"Source: {source_name}")
        root.add_child(source_node)

        # Transform nodes
        for transform in self._data["transforms"]:
            transform_node = TreeNode(f"Transform: {transform['name']}")
            root.add_child(transform_node)

        # Sink nodes
        for sink in self._data["sinks"]:
            sink_node = TreeNode(f"Sink: {sink['name']}")
            root.add_child(sink_node)

        return root
```

**Step 4: Run tests**

Run: `pytest tests/tui/ -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/tui/types.py src/elspeth/tui/widgets/lineage_tree.py tests/tui/test_lineage_types.py
git commit -m "$(cat <<'EOF'
feat(tui): add LineageData TypedDict, remove defensive .get()

Introduces strict type contract for TUI data:
- LineageData TypedDict with required fields
- Direct field access instead of .get() with defaults
- Missing fields now fail loudly instead of showing "unknown"

Fixes HIGH-003 from integration audit.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3: Medium-Severity Issues (15 hours)

### Task 11: Create RoutingSpec Dataclass (MED-002)

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py`
- Test: `tests/core/landscape/test_routing.py`

**Step 1: Create RoutingSpec**

```python
# Add to src/elspeth/core/landscape/models.py
from typing import Literal

@dataclass(frozen=True)
class RoutingSpec:
    """Specification for a routing decision.

    Replaces dict[str, str] parameter in record_routing_events().
    """

    edge_id: str
    mode: Literal["move", "copy"]
```

**Step 2: Update record_routing_events signature**

```python
# In recorder.py
def record_routing_events(
    self,
    state_id: str,
    routes: list[RoutingSpec],  # Changed from list[dict[str, str]]
    reason: dict[str, Any] | None = None,
) -> list[RoutingEvent]:
```

**Step 3: Update all callers**

**Step 4: Commit**

```bash
git commit -m "feat(landscape): add RoutingSpec dataclass for type-safe routing"
```

---

### Task 12: Add ResumeCheck Dataclass (MED-007)

**Files:**
- Modify: `src/elspeth/core/checkpoint/recovery.py`

**Step 1: Create ResumeCheck**

```python
@dataclass(frozen=True)
class ResumeCheck:
    """Result of checking if a run can be resumed."""

    can_resume: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.can_resume and self.reason is not None:
            raise ValueError("can_resume=True should not have a reason")
```

**Step 2: Update can_resume() return type**

**Step 3: Commit**

```bash
git commit -m "feat(checkpoint): add ResumeCheck dataclass for tuple return"
```

---

### Task 13: Fix Protocol Docstring Example (MED-006)

**Files:**
- Modify: `src/elspeth/plugins/protocols.py:164-170`

**Step 1: Update example to use direct access**

```python
# In protocols.py GateProtocol docstring
def evaluate(self, row: dict, ctx: PluginContext) -> GateResult:
    # Direct field access - schema guarantees field exists
    if row["suspicious"]:
        return GateResult(
            row=row,
            action=RoutingAction.route("review"),
        )
    return GateResult(row=row, action=RoutingAction.route("normal"))
```

**Step 2: Commit**

```bash
git commit -m "docs(plugins): fix GateProtocol example to use direct field access"
```

---

### Task 14: Export resolve_config from core (MED-009)

**Files:**
- Modify: `src/elspeth/core/__init__.py`

**Step 1: Add resolve_config to exports**

```python
# In core/__init__.py
from elspeth.core.config import resolve_config

__all__ = [
    # ... existing ...
    "resolve_config",
]
```

**Step 2: Commit**

```bash
git commit -m "fix(core): export resolve_config from core package"
```

---

### Task 15: Use NodeType Enum in TUI Filtering (MED-004)

**Files:**
- Modify: `src/elspeth/tui/screens/explain_screen.py:60-62`

**Step 1: Update filtering to use enum**

```python
from elspeth.plugins.enums import NodeType

source_nodes = [n for n in nodes if n.node_type == NodeType.SOURCE.value]
transform_nodes = [n for n in nodes if n.node_type == NodeType.TRANSFORM.value]
sink_nodes = [n for n in nodes if n.node_type == NodeType.SINK.value]
```

**Step 2: Commit**

```bash
git commit -m "fix(tui): use NodeType enum values for filtering"
```

---

### Task 16-20: Remaining Medium Issues

Follow same pattern for:
- MED-001: NodeState discriminated unions (4h)
- MED-003: Route validation at init (3h)
- MED-005: TUI optional sprawl (3h)
- MED-008: CSV export coupling (1h)
- MED-010: Run.status enum validation (2h)

---

## Phase 4: Low-Severity Issues (6 hours)

### Task 21-28: Low-Severity Fixes

| Task | Issue | Fix |
|------|-------|-----|
| 21 | LOW-001: Silent JSON decode | Log decode errors |
| 22 | LOW-002: Magic widget IDs | Create WidgetIDs constants |
| 23 | LOW-003: Plugin registry tuples | Create PluginInfo dataclass |
| 24 | LOW-004: Defensive node_to_plugin.get() | Use direct access |
| 25 | LOW-005: _MISSING sentinel duplication | Move to shared utils |
| 26 | LOW-006: Checkpoint not exported | Add to landscape exports |
| 27 | LOW-007: Config subclasses not exported | Add to core exports |
| 28 | LOW-008: ResumeCheck dataclass | (Covered in Task 12) |

---

## Summary

**Total Tasks:** 28
**Total Estimated Time:** 38 hours

**Phase Breakdown:**
- Phase 1 (Critical): 6 hours - Tasks 1-4
- Phase 2 (High): 11 hours - Tasks 5-10
- Phase 3 (Medium): 15 hours - Tasks 11-20
- Phase 4 (Low): 6 hours - Tasks 21-28

**Key Dependencies:**
- Task 2 depends on Task 1 (RowDataResult type)
- Task 4 depends on Task 3 (close() in protocol)
- Task 7 requires base classes from plugins/base.py
- TUI tasks (10, 15) depend on core fixes (1-9)

---

*Plan generated 2026-01-15 from integration audit findings*
