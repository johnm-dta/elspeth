# Bug Report: Gate routing errors leave node_state OPEN (MissingEdgeError / fork without TokenManager)

## Summary

- `GateExecutor.execute_gate()` begins a node_state and then performs routing resolution.
- If routing fails (MissingEdgeError) or fork occurs without a TokenManager, the method raises without completing the node_state, leaving an OPEN state in the audit trail.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (main)
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/executors.py` for bugs and create reports
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of `src/elspeth/engine/executors.py`

## Steps To Reproduce

1. Configure a gate that returns a route label missing from the route resolution map, or trigger `fork_to_paths` with no TokenManager provided.
2. Run a pipeline that executes the gate.
3. Inspect the `node_states` table for the gate node.

## Expected Behavior

- The gate node_state is completed with status `failed` and an error payload when routing fails.

## Actual Behavior

- The node_state remains OPEN, with no completion record, even though execution failed.

## Evidence

- Node_state is opened before routing resolution:
  - `src/elspeth/engine/executors.py:355`
  - `src/elspeth/engine/executors.py:360`
- Missing edge raises without completing the state:
  - `src/elspeth/engine/executors.py:409`
  - `src/elspeth/engine/executors.py:411`
- Fork path raises without completing the state when no TokenManager:
  - `src/elspeth/engine/executors.py:432`
  - `src/elspeth/engine/executors.py:436`

## Impact

- User-facing impact: failures leave incomplete audit trails and complicate debugging.
- Data integrity / security impact: violates node_state completion invariants for failed executions.
- Performance or cost impact: none direct.

## Root Cause Hypothesis

- Routing errors occur after `begin_node_state()` but before any `complete_node_state()` call; exceptions are raised directly.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/engine/executors.py`: wrap routing resolution in a try/except that records `complete_node_state(status="failed", error=...)` before raising.
- Config or schema changes: none.
- Tests to add/update:
  - Add tests that simulate MissingEdgeError and missing TokenManager paths and assert failed node_state completion.
- Risks or migration steps: none.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md` (audit trail completeness).
- Observed divergence: failed gate routing leaves OPEN node_state.
- Reason (if known): exception paths bypass completion logic.
- Alignment plan or decision needed: define required node_state handling for routing errors.

## Acceptance Criteria

- Routing failures always result in a failed node_state record.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_processor.py -k gate_routing_error`
- New tests required: yes (gate routing failure audit).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P3 verification wave 4

**Current Code Analysis:**

The bug is **STILL PRESENT** in `GateExecutor.execute_gate()` but has been **FIXED in execute_config_gate()**.

**Bug Instance 1: execute_gate() - MissingEdgeError path (lines 409-411)**
```python
if destination is None:
    # Label not in routes config - this is a configuration error
    raise MissingEdgeError(node_id=gate.node_id, label=route_label)
```
This raises WITHOUT completing the node_state that was opened at line 357.

**Bug Instance 2: execute_gate() - fork without TokenManager (lines 432-436)**
```python
elif action.kind == RoutingKind.FORK_TO_PATHS:
    if token_manager is None:
        raise RuntimeError(
            f"Gate {gate.node_id} returned fork_to_paths but no TokenManager provided. "
            "Cannot create child tokens - audit integrity would be compromised."
        )
```
This also raises WITHOUT completing the node_state opened at line 357.

**CONTRAST: execute_config_gate() properly handles these cases:**
- Lines 554-568: Records `complete_node_state(status="failed", ...)` before raising ValueError for missing route label
- Lines 591-605: Records `complete_node_state(status="failed", ...)` before raising RuntimeError for missing TokenManager

This suggests the pattern was learned when execute_config_gate() was implemented (commit ae56d025, 2026-01-18), but the older execute_gate() method was not retrofitted with the same fix.

**Git History:**

No commits since 2026-01-21 have addressed this specific bug. The most relevant commits to executors.py since the bug report are:
- 54edba7: Buffer/token length mismatch defensive guard
- 3e25073: Restore full TokenInfo from checkpoint
- cbbeee9: Preserve gate reason/mode in routing events
- d67cd42: Implement 5 audit integrity gaps (AUD-001 through AUD-005)

None of these address the OPEN node_state issue on routing errors.

**Test Coverage Gap:**

The existing test `test_execute_gate_missing_route_resolution()` (line 863) verifies that MissingEdgeError is raised, but does NOT verify that the node_state is properly completed with `status="failed"`. This allowed the bug to persist.

There is NO test for the fork_to_paths without TokenManager case.

**Root Cause Confirmed:**

YES - The bug is confirmed present. Two error paths in `execute_gate()` raise exceptions after opening a node_state but before completing it:
1. MissingEdgeError when route label not in route_resolution_map (line 411)
2. RuntimeError when fork_to_paths without TokenManager (lines 433-436)

Both violate the audit trail completeness principle from CLAUDE.md: "Every node_state that is begun must be completed."

**Recommendation:**

**Keep open** - This is a valid P3 bug that should be fixed.

**Suggested Fix Pattern:**
Apply the same pattern used in execute_config_gate() to execute_gate():
1. Wrap the MissingEdgeError raise (line 411) with node_state completion
2. Wrap the RuntimeError raise (lines 433-436) with node_state completion
3. Add regression tests that verify node_state completion on both error paths

**Example from execute_config_gate() (lines 554-568):**
```python
if route_label not in gate_config.routes:
    # Record failure before raising
    error = {
        "exception": f"Route label '{route_label}' not found in routes config",
        "type": "ValueError",
    }
    self._recorder.complete_node_state(
        state_id=state.state_id,
        status="failed",
        duration_ms=duration_ms,
        error=error,
    )
    raise ValueError(...)
```

This same pattern should be applied to both error cases in execute_gate().

---

## Re-verification (2026-01-25)

**Status: RE-ANALYZED**

### New Analysis

Re-ran static analysis on 2026-01-25. Key findings:

**Evidence:**
- node_state opened before routing: `src/elspeth/engine/executors.py:360`.
- MissingEdgeError raised without completion: `src/elspeth/engine/executors.py:412`, `src/elspeth/engine/executors.py:414`.
- fork without TokenManager raised without completion: `src/elspeth/engine/executors.py:435`, `src/elspeth/engine/executors.py:436`.

**Root Cause:**
- Error paths after routing resolution bypass `complete_node_state(...)`.

---

## Resolution

**Fixed in:** 2026-01-28
**Fixed by:** Claude Code (Opus 4.5)

**Fix:** Added `complete_node_state()` calls with FAILED status before raising exceptions in both error paths:
1. `MissingEdgeError` when route label not in route_resolution_map (lines 541-554)
2. `RuntimeError` when fork_to_paths without TokenManager (lines 575-588)

**Code changes:**
- `src/elspeth/engine/executors.py`: Wrapped both error raises with node_state completion using the pattern from `execute_config_gate()`

**Tests updated:**
- `tests/engine/test_gate_executor.py`: Extended `test_missing_edge_raises_error` and `test_fork_without_token_manager_raises_error` to verify node_state is completed with FAILED status

**Commits:**
- fix(executors): complete node_state before raising gate routing errors (P3-2026-01-28)
