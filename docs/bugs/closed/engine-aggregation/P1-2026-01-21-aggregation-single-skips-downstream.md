## ✅ RESOLVED

**Status:** Fixed
**Resolution:** Added `more_transforms` check to single output mode, matching passthrough and transform modes
**Commit:** 064e277 fix(engine): aggregation single mode now continues to downstream transforms
**Date:** 2026-01-28

---

# Bug Report: Aggregation output_mode=single terminates pipeline early

## Summary

- Aggregation flush in `output_mode="single"` returns a COMPLETED result immediately, so the aggregated row never traverses downstream transforms or config gates. This breaks pipeline ordering and drops intended processing for aggregated outputs.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 / fix/rc1-bug-burndown-session-2
- OS: Linux
- Python version: Python 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Deep dive src/elspeth/engine/processor.py for bugs; create reports.
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: Code inspection only

## Steps To Reproduce

1. Configure a batch-aware transform with aggregation `output_mode: single`.
2. Add another transform after it (or config gates) that should process the aggregated row.
3. Run a pipeline where the aggregation flushes.

## Expected Behavior

- The aggregated output row continues through remaining transforms and config gates.

## Actual Behavior

- The aggregated output is marked COMPLETED and returned immediately; downstream transforms/config gates are never executed for that row.

## Evidence

- `src/elspeth/engine/processor.py:224-246` returns a COMPLETED RowResult for `output_mode == "single"` with no work item for downstream processing.
- Pipeline order specifies transforms then config gates (e.g., `docs/design/subsystems/00-overview.md`).

## Impact

- User-facing impact: Missing downstream processing for aggregated outputs.
- Data integrity / security impact: Audit trail shows completed outcomes without required transform/gate node states.
- Performance or cost impact: Silent logic omission rather than explicit failure.

## Root Cause Hypothesis

- `_process_batch_aggregation_node()` treats `output_mode="single"` as terminal and never enqueues the aggregated token for remaining steps.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/engine/processor.py`
- Config or schema changes: None
- Tests to add/update: Add a test asserting aggregated single output flows through downstream transforms/config gates.
- Risks or migration steps: None; behavior should align with pipeline order.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/subsystems/00-overview.md` (pipeline order); `docs/plans/completed/plugin-refactor/2026-01-18-wp06-aggregation-triggers.md` (output_mode semantics).
- Observed divergence: Aggregated row is treated as terminal instead of continuing.
- Reason (if known): Missing continuation logic for single output mode.
- Alignment plan or decision needed: Decide whether single output should continue (expected) or enforce terminal-only aggregation nodes.

## Acceptance Criteria

- Aggregation `output_mode=single` outputs are processed by downstream transforms and config gates.
- Tests confirm correct node_state sequence after a flush.

## Tests

- Suggested tests to run: `pytest tests/engine/test_processor_modes.py::TestProcessorSingleMode -v`
- New tests required: Yes (single output continuation).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/subsystems/00-overview.md`

---

## Verification (2026-01-25)

**Status: STILL VALID**

### Findings

1. **Bug Confirmed in Current Code** (src/elspeth/engine/processor.py:227-244):

   The `output_mode == "single"` path returns COMPLETED immediately without checking if more transforms exist downstream:

   ```python
   if output_mode == "single":
       # Single output: one aggregated result row
       final_data = result.row if result.row is not None else {}
       updated_token = TokenInfo(
           row_id=current_token.row_id,
           token_id=current_token.token_id,
           row_data=final_data,
           branch_name=current_token.branch_name,
       )
       # COMPLETED outcome now recorded in orchestrator with sink_name (AUD-001)
       return (
           RowResult(
               token=updated_token,
               final_data=final_data,
               outcome=RowOutcome.COMPLETED,
           ),
           child_items,
       )
   ```

   **Critical Problem**: No check for `step < total_steps` like passthrough and transform modes have.

2. **Contrast with Correct Implementations**:

   **Passthrough mode** (lines 267-285) correctly checks for downstream transforms:
   ```python
   more_transforms = step < total_steps

   if more_transforms:
       # Queue enriched tokens as work items for remaining transforms
       for token, enriched_data in zip(buffered_tokens, result.rows, strict=True):
           # ... creates work items ...
       return ([], child_items)
   else:
       # No more transforms - return COMPLETED for all tokens
       # ... returns COMPLETED results ...
   ```

   **Transform mode** (lines 343-370) also checks for downstream transforms:
   ```python
   more_transforms = step < total_steps

   if more_transforms:
       # Queue expanded tokens as work items for remaining transforms
       for token in expanded_tokens:
           child_items.append(_WorkItem(...))
       return (triggering_result, child_items)
   else:
       # No more transforms - return COMPLETED for expanded tokens
       # ... returns COMPLETED results ...
   ```

3. **Specification Confirms Expected Behavior**:

   From `docs/contracts/plugin-protocol.md:1196`:
   > **`single`** (default): Classic aggregation. N rows become 1 aggregated row. All input tokens are terminal (`CONSUMED_IN_BATCH`). **The triggering token is reused for the output.**

   And line 1231:
   > `single`: Triggering token continues with aggregated data

   The word "continues" clearly indicates the token should continue through downstream transforms, not terminate immediately.

4. **Git History**:
   - Commit c6afc31 (2026-01-21) added outcome recording for single mode but did NOT fix the continuation bug
   - Commit message states "record COMPLETED before return" - just added the recording, kept the immediate return
   - No subsequent commits have addressed the missing `more_transforms` check
   - Bug existed since original aggregation implementation

5. **Test Coverage Gap**:
   - Searched test_processor.py for tests of single output mode with downstream transforms
   - Found `test_aggregation_transform_mode_single_row_output` which tests **transform** mode, not **single** mode
   - No test exists for `output_mode="single"` continuing to downstream transforms
   - The bug report's suggested test name (`pytest tests/engine/test_processor.py -k aggregation_single`) returns no matches

6. **Impact Assessment**:
   - Any pipeline with `output_mode: single` aggregation followed by transforms or config gates will silently skip those downstream steps
   - The aggregated row immediately becomes COMPLETED and goes to sink, bypassing any post-aggregation processing
   - Audit trail will show the row reaching terminal state without expected node_state entries for downstream transforms/gates
   - This is a silent data processing bug - no error is raised, the row just "disappears" from the expected processing path

### Root Cause

The single output mode implementation was written without the `more_transforms` conditional that was added to passthrough (lines 267) and transform (lines 344) modes. This creates an inconsistency where single mode always treats the aggregation node as terminal, while the other modes correctly check if downstream processing exists.

### Architectural Violation

Per the plugin protocol spec (line 1231), single mode should "continue" processing. The immediate COMPLETED return violates the documented contract and breaks the pipeline ordering guarantees described in `docs/design/subsystems/00-overview.md`.

### Verification Verdict

**BUG IS STILL VALID** - The single output mode path has never checked for downstream transforms. The aggregated token is marked COMPLETED and returned immediately, skipping any transforms or config gates that should process it afterward. This is a P1 bug affecting pipeline correctness.

---

## Resolution (2026-01-28)

### Fix Applied

Two issues were fixed in `src/elspeth/engine/processor.py`:

1. **Line 955 (call site)**: Changed `total_steps=len(transforms)` to `total_steps=total_steps` to include config gates in the count.

2. **Lines 229-263 (single mode)**: Added the `more_transforms` check matching passthrough and transform modes:

```python
if output_mode == "single":
    # Single output: one aggregated result row
    # The triggering token continues with aggregated data
    final_data = result.row if result.row is not None else {}
    updated_token = TokenInfo(...)

    # Check if there are more transforms after this one
    more_transforms = step < total_steps

    if more_transforms:
        # Queue aggregated token as work item for remaining transforms
        child_items.append(_WorkItem(token=updated_token, start_step=step))
        return ([], child_items)
    else:
        # No more transforms - return COMPLETED
        return (RowResult(..., outcome=RowOutcome.COMPLETED), child_items)
```

### Tests Added

- `TestProcessorSingleMode::test_aggregation_single_mode_continues_to_next_transform`
- `TestProcessorSingleMode::test_aggregation_single_mode_no_downstream_completes_immediately`

### Verification

- All 604 engine tests pass
- Manual reproduction confirms fix works correctly
