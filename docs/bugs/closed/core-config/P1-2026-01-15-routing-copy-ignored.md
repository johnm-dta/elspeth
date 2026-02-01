# Bug Report: Routing mode copy is ignored (route stops pipeline)

## Summary

- “Route + continue” semantics (COPY routing to a sink without terminating the main path) are not supported in config gates and may be unreachable in config-only pipelines. If COPY-to-sink routing is added as a config feature, the current processing model must ensure COPY does not stop downstream execution.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-15
- Related run/issue ID: N/A

## Environment

- Commit/branch: not checked
- OS: not checked
- Python version: not checked
- Config profile / env vars: not checked
- Data set or fixture: not checked

## Agent Context (if relevant)

- Goal or task prompt: identify another bug and document it
- Model/version: GPT-5 (Codex)
- Tooling and permissions (sandbox/approvals): sandbox read-only, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: inspected `src/elspeth/engine/executors.py`, `src/elspeth/engine/processor.py`, `docs/design/architecture.md`

## Steps To Reproduce

1. Configure a pipeline that can express “route to sink AND continue” for the same token (COPY semantics). (Config gates currently always use MOVE for sink routes.)
2. Run the pipeline and check both the routed sink and the default output sink for the same row/token.

## Expected Behavior

- The row is written to the `flagged` sink and continues to downstream transforms/default sink (copy semantics).

## Actual Behavior

- In config-gate-only pipelines, COPY-to-sink routing is not expressible; sink routing is MOVE/terminal by construction.
- If a COPY-to-sink feature is introduced, ensure it does not terminate the main path (i.e., avoid treating COPY like MOVE).

## Evidence

- Config gate sink routing is always MOVE (no COPY option): `src/elspeth/engine/executors.py:588-600`
- Processor treats any sink route as terminal for that token: `src/elspeth/engine/processor.py:460-470`
- Architecture defines COPY as “route and continue”: `docs/design/architecture.md:138-143`

## Impact

- User-facing impact: workflows relying on “route + continue” lose data in the main path.
- Data integrity / security impact: audit trail implies routing decisions but skips expected downstream transforms.
- Performance or cost impact: retries/reprocessing may be needed to recover missing outputs.

## Root Cause Hypothesis

- The config gate model does not provide a way to express COPY-to-sink routing, and the token-processing loop treats sink routing as terminal for that token. Adding COPY-to-sink requires explicit support in both executor outcomes and the processor/orchestrator sink buffering.

## Proposed Fix

- Code changes (modules/files):
  - Decide whether COPY-to-sink routing is a supported requirement for config gates. If yes:
    - Extend config gate routing config to express MOVE vs COPY for sink routes.
    - Extend gate outcome to signal “route + continue” and update `RowProcessor`/`Orchestrator` to enqueue sink output while continuing downstream processing.
- Config or schema changes: none.
- Tests to add/update:
  - Add orchestrator-level test verifying that `RoutingMode.COPY` sends a row to the routed sink and continues to the output sink.
- Risks or migration steps:
  - Requires adjusting routing counters/checkpoint semantics to handle dual destinations.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/architecture.md#L138`.
- Observed divergence: `copy` behaves like `move`, terminating the current path.
- Reason (if known): routing mode not propagated through gate outcome/processor logic.
- Alignment plan or decision needed: define how routing events are recorded for copy vs move and ensure both path outputs are produced.

## Acceptance Criteria

- If COPY-to-sink routing is supported: the row is written to the routed sink and continues down the pipeline.
- If config-only gates are the only gate mechanism: explicitly document that sink routing is terminal (MOVE) and close this report as not applicable.
- Audit trail records routing events while downstream node states are still present.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/engine/test_orchestrator.py`
- New tests required: yes, for routing copy semantics.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/architecture.md`

## Triage Note (2026-01-19)

**Status:** Kept in pending - needs investigation.

`RoutingMode.COPY` exists in the codebase (`dag.py:369,377`, `routing.py:95`) but it's unclear if the processor implements "route AND continue" semantics correctly.

**Needs investigation:**
1. Trace `RoutingMode.COPY` through gate execution to see if dual outputs are produced
2. Check if processor handles COPY differently from MOVE
3. Write test that verifies COPY results in output to both routed sink AND downstream processing

---

## Resolution (2026-01-24)

**Status:** RESOLVED - Working as designed (architectural limitation)

### Root Cause Analysis

Systematic debugging revealed a three-layer architectural gap:

1. **Contract layer** (`routing.py`): Correctly defines COPY mode for all routing kinds
2. **Executor layer** (`executors.py:627`): Hardcodes `RoutingMode.MOVE` when routing to sinks
3. **Processor layer** (`processor.py:681-696`): Treats all sink routing as terminal (immediate return)

### Critical Finding: Single Terminal State Invariant

ELSPETH's audit model enforces: **"Every row reaches exactly one terminal state"** (CLAUDE.md)

COPY mode for ROUTE would violate this invariant by requiring dual terminal states:
1. `ROUTED` when sent to mid-pipeline sink
2. `COMPLETED` when reaching final output sink

This breaks the audit trail's single-terminal-state model.

### How FORK Correctly Handles COPY

FORK_TO_PATHS achieves "copy and continue" semantics by creating **child tokens**, each with their own terminal state:
- Parent token: `FORKED` (terminal)
- Child tokens: Each eventually reaches `COMPLETED`, `ROUTED`, etc.

Each token has exactly ONE terminal state. The audit trail remains consistent.

### Decision: Explicit Rejection

**COPY mode is only valid for FORK_TO_PATHS. ROUTE kind must use MOVE mode.**

**Rationale:**
1. Preserves audit integrity (single terminal state per token)
2. Working pattern exists (FORK_TO_PATHS provides "route and continue" correctly)
3. No semantic value (COPY for ROUTE would just be "fork to one sink" - use fork instead)
4. Avoids complexity (no orchestrator buffering, no dual terminal states, no checkpoint redesign)

### Implementation

**Changes made:**

1. **Contract validation** (`src/elspeth/contracts/routing.py:70-76`):
   ```python
   if self.kind == RoutingKind.ROUTE and self.mode == RoutingMode.COPY:
       raise ValueError(
           "COPY mode not supported for ROUTE kind. "
           "Use FORK_TO_PATHS to route to sink and continue processing. "
           "Reason: ELSPETH's audit model enforces single terminal state per token; "
           "COPY would require dual terminal states (ROUTED + COMPLETED)."
       )
   ```

2. **Updated documentation** (`routing.py:35-51`):
   - Clarified COPY is only valid for FORK_TO_PATHS
   - Added note about architectural constraint
   - Updated `route()` method docstring

3. **Test coverage** (`tests/contracts/test_routing.py:48-64`):
   ```python
   def test_route_with_copy_raises(self) -> None:
       """route with COPY mode raises ValueError (architectural limitation)."""
       with pytest.raises(ValueError, match="COPY mode not supported for ROUTE"):
           RoutingAction.route("above", mode=RoutingMode.COPY)
   ```

4. **Architecture Decision Record**: `docs/design/adr/002-routing-copy-mode-limitation.md`

### User Guidance

To achieve "route to sink and continue" semantics, users should use `fork_to_paths()`:

```python
# Instead of (not supported):
gate_config = GateSettings(
    routes={"high_risk": RouteConfig(destination="flagged", mode=RoutingMode.COPY)}
)

# Use this (fork to sink):
gate_config = GateSettings(
    routes={"high_risk": "fork"},
    fork_to=["flagged"]  # Creates child token that continues processing
)
```

### Reviews

- **Architecture Critic** (Agent a2a1113): Identified dual-terminal-state violation (CRITICAL severity)
- **Code Reviewer** (Agent af71460): Identified silent data loss risk in proposed implementation

Both reviewers recommended Option B (explicit rejection) over Option A (implement COPY).

### Follow-up

**Task created:** Evaluate whether `RoutingMode.COPY` is needed at all in the enum (separate analysis)
