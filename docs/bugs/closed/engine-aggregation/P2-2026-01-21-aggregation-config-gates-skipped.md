# Bug Report: Aggregation outputs skip config gates when aggregation is last transform

## Summary

- When an aggregation node is the last transform, passthrough/transform flush results are marked COMPLETED and returned immediately. Config gates (which should run after transforms) are never executed for those outputs.

## Severity

- Severity: major
- Priority: P2

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

1. Configure a batch-aware transform as the last transform in the pipeline with `output_mode: passthrough` or `transform`.
2. Define one or more config-driven gates (`config.gates`).
3. Trigger a batch flush.

## Expected Behavior

- Outputs from the aggregation should run through config gates (pipeline order: transforms → config gates → sinks).

## Actual Behavior

- Aggregation outputs are marked COMPLETED and sent to sinks directly; config gate node_states are missing.

## Evidence

- Passthrough flush returns COMPLETED outputs when `more_transforms` is false: `src/elspeth/engine/processor.py:267-310`.
- Transform-mode flush returns COMPLETED outputs when `more_transforms` is false: `src/elspeth/engine/processor.py:349-379`.
- Config gates are only processed later in `_process_single_token`: `src/elspeth/engine/processor.py:874-954`. These are bypassed because `_process_batch_aggregation_node()` returns early.

## Impact

- User-facing impact: Config gate logic is silently skipped for aggregated outputs.
- Data integrity / security impact: Missing node_state records for config gates; routing decisions not applied.
- Performance or cost impact: Incorrect routing or sink selection.

## Root Cause Hypothesis

- `_process_batch_aggregation_node()` decides continuation based solely on remaining transforms (`more_transforms`) and doesn’t account for config gates.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/engine/processor.py`
- Config or schema changes: None
- Tests to add/update: Add tests ensuring config gates run after aggregation flushes when aggregation is last transform.
- Risks or migration steps: None.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): Pipeline order described in `docs/design/subsystems/00-overview.md`.
- Observed divergence: Config gates are skipped for certain aggregation outputs.
- Reason (if known): `more_transforms` gate doesn’t consider config gates.
- Alignment plan or decision needed: Account for config gates when deciding whether to enqueue aggregated outputs.

## Acceptance Criteria

- Aggregation outputs run through config gates even when aggregation is the last transform.

## Tests

- Suggested tests to run: `pytest tests/engine/test_processor.py -k aggregation_config_gates`
- New tests required: Yes (aggregation + config gate integration).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/subsystems/00-overview.md`

---

## Verification (2026-01-25)

**Status: STILL VALID**

### Findings

1. **Bug Confirmed in Current Code** (src/elspeth/engine/processor.py:739):

   The root cause is that `total_steps` only counts plugin transforms, not config gates:

   ```python
   # Line 739: Call to _process_batch_aggregation_node
   return self._process_batch_aggregation_node(
       transform=transform,
       current_token=current_token,
       ctx=ctx,
       step=step,
       child_items=child_items,
       total_steps=len(transforms),  # ← BUG: Only plugin transforms counted
   )
   ```

2. **Impact on Passthrough Mode** (lines 265-304):

   When aggregation is the last plugin transform, `more_transforms = step < total_steps` evaluates to False:

   ```python
   # Line 267
   more_transforms = step < total_steps  # ← False when aggregation is last transform

   if more_transforms:
       # Queue enriched tokens as work items for remaining transforms
       for token, enriched_data in zip(buffered_tokens, result.rows, strict=True):
           # ... creates work items that WOULD continue to config gates ...
       return ([], child_items)
   else:
       # No more transforms - return COMPLETED for all tokens
       # ← BUG: This path is taken even though config gates still need to run
       results: list[RowResult] = []
       for token, enriched_data in zip(buffered_tokens, result.rows, strict=True):
           results.append(
               RowResult(
                   token=updated_token,
                   final_data=enriched_data,
                   outcome=RowOutcome.COMPLETED,  # ← WRONG: Should continue to config gates
               )
           )
       return (results, child_items)
   ```

3. **Impact on Transform Mode** (lines 343-370):

   Same issue - when `more_transforms = False`, expanded tokens are returned as COMPLETED:

   ```python
   # Line 344
   more_transforms = step < total_steps  # ← False when aggregation is last transform

   if more_transforms:
       # Queue expanded tokens as work items for remaining transforms
       for token in expanded_tokens:
           child_items.append(_WorkItem(...))
       return (triggering_result, child_items)
   else:
       # No more transforms - return COMPLETED for expanded tokens
       # ← BUG: This path is taken even though config gates still need to run
       output_results: list[RowResult] = [triggering_result]
       for token in expanded_tokens:
           output_results.append(
               RowResult(
                   token=token,
                   final_data=token.row_data,
                   outcome=RowOutcome.COMPLETED,  # ← WRONG: Should continue to config gates
               )
           )
       return (output_results, child_items)
   ```

4. **Config Gates Are Processed Later** (lines 864-944):

   Config gates are only processed in `_process_single_token()` AFTER all plugin transforms complete:

   ```python
   # Line 864-866: Config gates run after transforms
   # Process config-driven gates (after all plugin transforms)
   # Step continues from where transforms left off
   config_gate_start_step = len(transforms) + 1
   ```

   But when `_process_batch_aggregation_node()` returns COMPLETED results directly, `_process_single_token()` returns early and never reaches this code block.

5. **Single Output Mode Has Different Bug**:

   Note that `output_mode="single"` (lines 227-244) has a **different** bug - it ALWAYS returns COMPLETED without any `more_transforms` check at all. That's tracked separately as P1-2026-01-21-aggregation-single-skips-downstream.md.

   This bug (P2) is specifically about passthrough/transform modes checking `more_transforms` but not accounting for config gates in that check.

6. **Git History**:
   - Commit c6afc31 (2026-01-21): Added outcome recordings but did NOT fix the continuation logic
   - No commits since the bug report (2026-01-21) have addressed the `total_steps` calculation
   - The original implementation of passthrough/transform modes used `len(transforms)` and never considered config gates

7. **Test Coverage Gap**:
   - Searched for tests combining aggregation + config gates
   - No test file combines batch-aware transforms with config gates in a single pipeline
   - The suggested test name from the bug report (`pytest tests/engine/test_processor.py -k aggregation_config_gates`) returns no matches
   - Existing aggregation tests focus on output modes, triggers, and basic routing, but don't test interaction with config gates

8. **Pipeline Processing Order** (documented in processor.py:62-65):

   The class docstring explicitly states the expected order:

   ```python
   Pipeline order:
   - Transforms (from config.row_plugins)
   - Config-driven gates (from config.gates)
   - Output sink
   ```

   The bug violates this contract by sending aggregation outputs directly to sinks, bypassing config gates.

### Root Cause Analysis

The problem is architectural:

1. `_process_batch_aggregation_node()` receives `total_steps=len(transforms)` (line 739)
2. This parameter only counts **plugin transforms**, not **config gates**
3. When checking `step < total_steps`, it determines "no more work" when aggregation is the last plugin transform
4. But **config gates are separate** from transforms and run afterward (lines 864-944)
5. The COMPLETED results bypass the work queue that would have processed config gates

**The fix requires one of:**
- Change `total_steps` to include config gates: `len(transforms) + len(self._config_gates)`
- Change the `more_transforms` check to also consider config gates: `step < total_steps or len(self._config_gates) > 0`
- Queue work items for config gate processing even when `more_transforms` is False

### Impact Assessment

Any pipeline configured with:
- A batch-aware transform as the **last** plugin transform
- `output_mode` set to `passthrough` or `transform`
- One or more **config gates** defined in `config.gates`

Will experience:
- **Silent routing bug**: Config gate routing decisions never execute
- **Missing audit trail**: No node_state entries for config gates
- **Incorrect sink selection**: Rows go to default sink instead of gate-routed sinks
- **Data integrity violation**: Audit trail appears complete but is actually missing required processing steps

### Architectural Violation

The documented pipeline order (processor.py:62-65) states transforms → config gates → sinks. The current implementation violates this by allowing aggregation outputs to skip config gates entirely when the aggregation is the last transform.

### Related Bugs

- **P1-2026-01-21-aggregation-single-skips-downstream.md**: Single output mode has a more severe version of this bug - it never checks for downstream work AT ALL (no `more_transforms` check). That bug affects both plugin transforms AND config gates.
- This bug (P2) is narrower: passthrough/transform modes DO check for downstream plugin transforms, but fail to account for config gates.

### Verification Verdict

**BUG IS STILL VALID** - The `total_steps` parameter only counts plugin transforms, causing passthrough and transform modes to return COMPLETED when the aggregation is the last plugin transform, even though config gates still need to process the outputs. This is a P2 bug affecting pipeline correctness when config gates are used with aggregations.
