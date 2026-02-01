# Bug Report: Condition triggers cannot access row data

## Summary

- Aggregation condition triggers are evaluated against a context containing only `batch_count` and `batch_age_seconds`. Expressions that reference actual row fields (e.g., `row['type']`) raise `KeyError` and crash trigger evaluation, despite docs/config examples showing row-based conditions.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: N/A

## Environment

- Commit/branch: `fix/rc1-bug-burndown-session-2` @ `ae2c0e6f088f467276582fa8016f91b4d3bb26c7`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/triggers.py` and file bugs
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of trigger evaluation and config docs

## Steps To Reproduce

1. Configure an aggregation trigger with `condition: "row['type'] == 'flush_signal'"` (as shown in docs/config examples).
2. Run the pipeline with a row that includes `{"type": "flush_signal"}`.
3. Observe a `KeyError: 'type'` from trigger evaluation, causing the aggregation to crash or never flush.

## Expected Behavior

- Condition triggers should evaluate against the current row (and any batch metadata) and fire when the row matches the expression.

## Actual Behavior

- Condition evaluation is executed against a context containing only `batch_count` and `batch_age_seconds`, so row field access raises `KeyError`.

## Evidence

- Trigger context lacks row data: `src/elspeth/engine/triggers.py:106-113`
- Doc example uses row fields: `docs/contracts/plugin-protocol.md:1169`
- Config tests allow row-based conditions: `tests/core/test_config_aggregation.py:50-60`

## Impact

- User-facing impact: condition-based batching using row signals is unusable.
- Data integrity / security impact: aggregation batches may never flush or crash mid-run.
- Performance or cost impact: buffered rows can accumulate indefinitely or crash runs.

## Root Cause Hypothesis

- `TriggerEvaluator` does not receive row data, and `should_trigger()` builds a context with only batch stats.

## Proposed Fix

- Code changes (modules/files):
  - Extend `TriggerEvaluator` to accept row context (e.g., `record_accept(row)` or `should_trigger(row)`).
  - Build evaluation context as merged row data + reserved batch keys.
  - Update `AggregationExecutor.buffer_row()` to pass row context.
- Config or schema changes:
  - Document reserved keys (`batch_count`, `batch_age_seconds`) and collision behavior.
- Tests to add/update:
  - Add trigger evaluation test with a row-based condition (e.g., `row['type'] == 'flush_signal'`).
- Risks or migration steps:
  - Potential collisions if row data already has `batch_count` keys; define precedence.

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md:1209-1211`
- Observed divergence: condition trigger is documented as "Row matches expression" but row data is not provided.
- Reason (if known): evaluator was implemented with batch-only context.
- Alignment plan or decision needed: confirm whether condition triggers are row-scoped or batch-scoped.

## Acceptance Criteria

- Condition triggers can access row fields and fire without raising `KeyError`.

## Tests

- Suggested tests to run: `pytest tests/engine/test_triggers.py -k condition`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`

## Verification (2026-01-25)

**Status: STILL VALID**

Verified against current codebase on branch `fix/rc1-bug-burndown-session-4` at commit `d282672`.

### Code Analysis

1. **Trigger context construction** (`src/elspeth/engine/triggers.py:106-113`):
   ```python
   # Check condition trigger
   if self._condition_parser is not None:
       context = {
           "batch_count": self._batch_count,
           "batch_age_seconds": self.batch_age_seconds,
       }
       result = self._condition_parser.evaluate(context)
   ```
   The context dict passed to `evaluate()` contains ONLY batch metadata. No row data is included.

2. **Buffering logic** (`src/elspeth/engine/executors.py:764-810`):
   - `buffer_row(node_id, token)` receives the token with row data (`token.row_data`)
   - Row data is buffered in line 795: `self._buffers[node_id].append(token.row_data)`
   - `record_accept()` is called in line 810: `evaluator.record_accept()`
   - **Row data is never passed to the evaluator**

3. **TriggerEvaluator API** (`src/elspeth/engine/triggers.py:74-82`):
   ```python
   def record_accept(self) -> None:
       """Record that a row was accepted into the batch."""
       self._batch_count += 1
       if self._first_accept_time is None:
           self._first_accept_time = time.monotonic()
   ```
   The method signature does not accept row data as a parameter.

4. **ExpressionParser behavior** (`src/elspeth/engine/expression_parser.py:451-461`):
   ```python
   def evaluate(self, row: dict[str, Any]) -> Any:
       """Evaluate expression against row data.

       Args:
           row: Row data dictionary
       """
   ```
   The evaluator correctly accepts a dict and makes it available as `row` in expressions. The problem is that `TriggerEvaluator.should_trigger()` passes a dict with only batch stats, not actual row fields.

### Test Coverage Gap

All existing tests in `tests/engine/test_triggers.py` use only batch metadata fields:
- `row['batch_count']` (lines 81, 94, 108, 158)
- `row['batch_age_seconds']` (line 108)

**No test attempts to access actual row data fields like `row['type']` or `row['status']`.**

### Documentation/Code Mismatch

1. **Documentation** (`docs/contracts/plugin-protocol.md:1169`):
   ```yaml
   # condition: "row['type'] == 'flush_signal'"  # Optional: trigger on special row
   ```
   Example shows accessing row field `type`.

2. **Config test** (`tests/core/test_config_aggregation.py:56`):
   ```python
   condition="row['type'] == 'flush_signal'"
   ```
   Config validation accepts row field expressions, but they would fail at runtime.

3. **Protocol docs** (`docs/contracts/plugin-protocol.md:1210`):
   ```
   | `condition` | Row matches expression |
   ```
   States "Row matches expression" but implementation doesn't provide row data.

### Git History

No commits since bug report date (2026-01-22) have modified `src/elspeth/engine/triggers.py` or addressed this issue. Last substantial change was commit `efbd30b` ("feat(engine): add TriggerEvaluator") which predates the bug report.

### Impact Assessment

This bug prevents any row-based trigger conditions from working. Users attempting to use expressions like:
- `row['type'] == 'flush_signal'` → KeyError: 'type'
- `row['priority'] == 'high'` → KeyError: 'priority'
- `row['batch_marker'] is not None` → KeyError: 'batch_marker'

The only working condition triggers are those using `batch_count` and `batch_age_seconds`, which are better expressed as explicit `count` and `timeout_seconds` triggers.

### Verification Method

- ✅ Examined source code at reported lines
- ✅ Traced execution path from `buffer_row()` → `record_accept()` → `should_trigger()`
- ✅ Confirmed context dict construction lacks row data
- ✅ Verified no API exists to pass row data to evaluator
- ✅ Checked test coverage - no tests use row fields
- ✅ Verified documentation shows row field examples
- ✅ Confirmed no fixes in git history since bug report
