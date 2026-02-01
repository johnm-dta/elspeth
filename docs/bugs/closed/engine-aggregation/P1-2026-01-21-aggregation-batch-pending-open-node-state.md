# Bug Report: BatchPendingError leaves aggregation node_state open and batch unlinked

## Summary

- `AggregationExecutor.execute_flush()` begins a node_state and marks the batch as `executing`, but if the transform raises `BatchPendingError` the code re-raises without completing the node_state or linking the batch to that state.
- Each retry will create additional open node_states, and the batch remains `executing` with `aggregation_state_id` unset.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (main)
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: batch-aware transform that raises `BatchPendingError`

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/executors.py` for bugs and create reports
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of `src/elspeth/engine/executors.py`

## Steps To Reproduce

1. Configure an aggregation node using a batch-aware transform that raises `BatchPendingError` on flush (e.g., an async batch LLM transform).
2. Run a pipeline until the aggregation trigger fires.
3. Inspect `node_states` and `batches` after the `BatchPendingError` is raised.

## Expected Behavior

- The batch is linked to the in-progress node_state (aggregation_state_id set), and the node_state is completed or otherwise tracked so retries do not leave orphaned OPEN states.

## Actual Behavior

- The node_state remains OPEN with no completion record, and the batch stays `executing` with no aggregation_state_id. Subsequent retries can create multiple OPEN node_states.

## Evidence

- Node_state is opened before the transform call:
  - `src/elspeth/engine/executors.py:906`
  - `src/elspeth/engine/executors.py:909`
- Batch status is set to `executing` without state linkage:
  - `src/elspeth/engine/executors.py:899`
  - `src/elspeth/engine/executors.py:903`
- `BatchPendingError` is re-raised without completing node_state or updating the batch:
  - `src/elspeth/engine/executors.py:929`
  - `src/elspeth/engine/executors.py:935`

## Impact

- User-facing impact: repeated retries create accumulating OPEN node_states and incomplete audit trails.
- Data integrity / security impact: batch execution is not traceable to a state record, violating auditability.
- Performance or cost impact: retries accumulate audit noise and complicate recovery.

## Root Cause Hypothesis

- The BatchPendingError path exits before completing node_state or persisting state_id for batch linkage.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/engine/executors.py`: when `BatchPendingError` is raised, either:
    - persist `state_id` in the batch (`aggregation_state_id`) and reuse the same state on retry, or
    - defer `begin_node_state` until the batch completes, or
    - introduce a `pending` terminal status to close the node_state with explicit semantics.
- Config or schema changes: possibly add a `pending` node_state status if needed.
- Tests to add/update:
  - Add a test covering BatchPendingError to ensure no orphaned OPEN node_states and proper batch linkage.
- Risks or migration steps: none.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md` auditability principles (transform boundaries must be recorded) and batch linkage expectations.
- Observed divergence: open node_state with no completion for pending batches.
- Reason (if known): control-flow exception bypasses completion logic.
- Alignment plan or decision needed: define how pending batch work is represented in node_states.

## Acceptance Criteria

- BatchPendingError does not leave OPEN node_states in the audit trail.
- Executing batches have an `aggregation_state_id` linked to the flush attempt.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_processor.py -k batch_pending`
- New tests required: yes (BatchPendingError audit invariants).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

---

## Resolution

### Status
**FIXED** - Resolved on 2026-01-23

### Fixed By
- **Primary fix**: commit `0f21ecb` - Added PENDING status to NodeStateStatus enum
- **Type safety improvements**: commit `fde9835` - Added type overloads and transition test

### Implementation Approach

The fix introduced a new **PENDING** status to the `NodeStateStatus` enum to properly model the semantic difference between:
- **OPEN**: Work in progress (transform executing)
- **PENDING**: Work completed, result not available yet (async operation submitted)
- **COMPLETED**: Work completed with result available

This follows the existing pattern in the codebase:
- `BatchStatus`: DRAFT → **EXECUTING** → COMPLETED/FAILED
- `ExportStatus`: **PENDING** → COMPLETED/FAILED
- `NodeStateStatus`: OPEN → **PENDING** → COMPLETED/FAILED *(new)*

#### Key Changes

1. **Contract extension** (`src/elspeth/contracts/enums.py`, `src/elspeth/contracts/audit.py`):
   - Added `PENDING = "pending"` to `NodeStateStatus` enum
   - Created `NodeStatePending` dataclass with invariants:
     - Has `completed_at` and `duration_ms` (operation finished)
     - No `output_hash` (result not available yet)
     - Has `context_before_json` and `context_after_json`

2. **Recorder updates** (`src/elspeth/core/landscape/recorder.py`):
   - Added PENDING case to `_row_to_node_state()` with validation
   - Added type overloads to `complete_node_state()` for precise return types
   - Validates PENDING states have `completed_at` and `duration_ms`

3. **Executor fix** (`src/elspeth/engine/executors.py:929-953`):
   ```python
   except BatchPendingError:
       duration_ms = (time.perf_counter() - start) * 1000

       # Close node_state with "pending" status
       self._recorder.complete_node_state(
           state_id=state.state_id,
           status="pending",
           duration_ms=duration_ms,
       )

       # Link batch to the aggregation state
       self._recorder.update_batch_status(
           batch_id=batch_id,
           status="executing",
           state_id=state.state_id,
       )

       raise  # Re-raise for orchestrator retry
   ```

4. **Exporter support** (`src/elspeth/core/landscape/exporter.py`):
   - Added PENDING case to discriminated union serialization
   - Exports PENDING states with `output_hash: None`

#### Alternative Approaches Considered

1. ❌ **Leave node_state OPEN** - Original bug, violates audit trail integrity
2. ❌ **Use marker output with COMPLETED** - Semantic corruption (recording fake completion)
3. ❌ **Separate pending_batches table** - Unnecessary complexity
4. ✅ **Add PENDING status** - Proper state machine extension (chosen)

The architecture-critic agent review explicitly rejected approach #2, noting: "The marker output approach is a bug masquerading as a fix. Recording fake completion is evidence tampering in a high-stakes system."

### Test Coverage

**Primary test**: `tests/engine/test_aggregation_audit.py::test_batch_pending_error_closes_node_state_and_links_batch`
- Verifies BatchPendingError is raised correctly
- Asserts no OPEN node_states remain after exception
- Validates PENDING state has correct invariants (completed_at, no output_hash)
- Verifies batch has `aggregation_state_id` linking to node_state
- Confirms batch status remains "executing"

**Transition test**: `tests/engine/test_aggregation_audit.py::test_pending_to_completed_transition`
- Verifies PENDING → COMPLETED state transition
- Tests update path when async batch operations complete
- Confirms audit trail shows final COMPLETED state with output_hash

**Test results**: All 500 engine tests pass, mypy clean

### Verification

✅ Acceptance Criteria Met:
1. BatchPendingError does not leave OPEN node_states - **VERIFIED**
2. Executing batches have aggregation_state_id linked to flush attempt - **VERIFIED**

✅ Additional validations:
- No orphaned OPEN states in audit trail
- Batch traceability maintained through aggregation_state_id
- Type safety improved with overload signatures
- Full lifecycle tested (PENDING → COMPLETED transition)

### Architectural Impact

The PENDING status addition maintains ELSPETH's core principles:
- **Auditability**: Every batch submission traceable to node_state
- **Three-Tier Trust Model**: Crashes on NULL fields in audit trail (Tier 1)
- **No Bug Hiding**: Direct field access, crashes on violations
- **Type Safety**: Discriminated unions with Literal types enforce invariants

The state machine now correctly models async operations with a proper transitional state, avoiding the architectural anti-pattern of abusing terminal states with fake data.
