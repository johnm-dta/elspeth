# Bug Report: Aggregation requeue drops coalesce metadata for forked branches

## Summary

- When a forked branch passes through a passthrough/transform aggregation and the batch flush requeues tokens for further processing, the new work items omit `coalesce_at_step` and `coalesce_name`. Forked tokens then bypass coalesce entirely.

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

1. Configure a gate that forks to two branches with a coalesce point configured for those branches.
2. Place a batch-aware transform with `output_mode: passthrough` or `transform` in each branch, with additional transforms after the aggregation.
3. Trigger a batch flush and observe whether tokens coalesce.

## Expected Behavior

- Forked branch tokens should carry coalesce metadata through the aggregation flush and merge at the coalesce point.

## Actual Behavior

- Tokens requeued after aggregation lack coalesce metadata, so the coalesce check never runs and tokens bypass the join.

## Evidence

- Passthrough requeue omits coalesce info in work items: `src/elspeth/engine/processor.py:271-285`.
- Transform-mode requeue omits coalesce info in work items: `src/elspeth/engine/processor.py:352-359`.
- Coalesce check requires `coalesce_at_step`/`coalesce_name`: `src/elspeth/engine/processor.py:955-964`.

## Impact

- User-facing impact: Fork/join pipelines silently skip coalesce, producing unmerged outputs.
- Data integrity / security impact: Audit lineage for parallel branches is broken.
- Performance or cost impact: Downstream duplication and inflated outputs.

## Root Cause Hypothesis

- `_process_batch_aggregation_node()` builds new `_WorkItem` objects without propagating coalesce metadata.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/engine/processor.py`
- Config or schema changes: None
- Tests to add/update: Add a fork→aggregation→coalesce integration test ensuring coalesce executes.
- Risks or migration steps: None.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/subsystems/00-overview.md` (fork/coalesce flow).
- Observed divergence: Coalesce metadata is dropped mid-pipeline.
- Reason (if known): Work items do not carry coalesce fields.
- Alignment plan or decision needed: Propagate coalesce metadata when requeuing tokens.

## Acceptance Criteria

- Aggregated branch tokens still coalesce when coalesce is configured.
- Tests confirm coalesce outcomes and join lineage for aggregated branches.

## Tests

- Suggested tests to run: `pytest tests/engine/test_processor.py -k coalesce`
- New tests required: Yes (aggregation + coalesce integration).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/subsystems/00-overview.md`

## Verification (2026-01-25)

**Status: STILL VALID**

### Current State Analysis

Examined current codebase at commit `7540e57` on branch `fix/rc1-bug-burndown-session-4`. The bug remains present in the code.

### Evidence

1. **Passthrough mode requeue (lines 269-285)**:
   ```python
   if more_transforms:
       # Queue enriched tokens as work items for remaining transforms
       for token, enriched_data in zip(buffered_tokens, result.rows, strict=True):
           updated_token = TokenInfo(
               row_id=token.row_id,
               token_id=token.token_id,
               row_data=enriched_data,
               branch_name=token.branch_name,  # ✓ branch_name preserved
           )
           child_items.append(
               _WorkItem(
                   token=updated_token,
                   start_step=step,  # ✓ step provided
                   # ❌ coalesce_at_step MISSING
                   # ❌ coalesce_name MISSING
               )
           )
   ```

2. **Transform mode requeue (lines 346-354)**:
   ```python
   if more_transforms:
       # Queue expanded tokens as work items for remaining transforms
       for token in expanded_tokens:
           child_items.append(
               _WorkItem(
                   token=token,
                   start_step=step,  # ✓ step provided
                   # ❌ coalesce_at_step MISSING
                   # ❌ coalesce_name MISSING
               )
           )
   ```

3. **Comparison with fork operations** (lines 685-705):
   Fork operations correctly propagate coalesce metadata:
   ```python
   elif outcome.result.action.kind == RoutingKind.FORK_TO_PATHS:
       for child_token in outcome.child_tokens:
           # Look up coalesce info for this branch
           branch_name = child_token.branch_name
           child_coalesce_name: str | None = None
           child_coalesce_step: int | None = None

           if branch_name and branch_name in self._branch_to_coalesce:
               child_coalesce_name = self._branch_to_coalesce[branch_name]
               child_coalesce_step = self._coalesce_step_map.get(child_coalesce_name)

           child_items.append(
               _WorkItem(
                   token=child_token,
                   start_step=next_step,
                   coalesce_at_step=child_coalesce_step,  # ✓ provided
                   coalesce_name=child_coalesce_name,      # ✓ provided
               )
           )
   ```

4. **Root cause confirmed**:
   - The `_process_single_token()` method receives `coalesce_at_step` and `coalesce_name` parameters (lines 629-630)
   - When calling `_process_batch_aggregation_node()` at line 733, these parameters are available in scope but not passed to the method
   - The method signature does not include these parameters (line 150)
   - When creating work items inside aggregation requeue logic, there's no way to access the coalesce metadata

### Git History Review

- No commits found that specifically fix this issue
- Related bug `P2-2026-01-21-aggregation-config-gates-skipped.md` exists in pending, describing a similar problem where aggregation outputs skip config gates
- Test case for fork→aggregation→coalesce was deleted (see `test_full_feature_pipeline_deleted` in `tests/engine/test_integration.py:2181`), leaving no test coverage for this scenario

### Impact Confirmation

When a forked token with `branch_name` set passes through an aggregation node:
1. Token is buffered with its `branch_name` intact
2. On flush, token is requeued with updated data
3. New work item lacks `coalesce_at_step` and `coalesce_name`
4. Coalesce check at lines 946-987 fails because `coalesce_name is None`
5. Token proceeds to COMPLETED instead of COALESCED
6. Fork siblings never merge, breaking join semantics

### Recommended Fix

1. **Add parameters to `_process_batch_aggregation_node()`**:
   - Add `coalesce_at_step: int | None` parameter
   - Add `coalesce_name: str | None` parameter

2. **Update call site** (line 733):
   - Pass `coalesce_at_step=coalesce_at_step`
   - Pass `coalesce_name=coalesce_name`

3. **Update work item creation** (lines 278-283 and lines 349-354):
   - Include `coalesce_at_step=coalesce_at_step`
   - Include `coalesce_name=coalesce_name`

4. **Add integration test**:
   - Fork to 2 branches
   - Each branch has a batch-aware transform (passthrough or transform mode)
   - Configure coalesce point after aggregations
   - Verify COALESCED outcome and merged data

### Related Issues

- `P2-2026-01-21-aggregation-config-gates-skipped.md`: Similar issue where aggregation outputs skip config gates due to incomplete continuation logic
