# Test Defect Report

## Summary

- Outcome-specific context fields are not asserted for most outcomes, so audit context can regress without failing tests.

## Severity

- Severity: major
- Priority: P1

## Category

- Weak Assertions

## Evidence

- `tests/engine/test_processor_outcomes.py:78` passes outcome-specific context in `kwargs`, but `tests/engine/test_processor_outcomes.py:115`-`tests/engine/test_processor_outcomes.py:119` only assert `outcome` and `is_terminal`.
- Snippet from `tests/engine/test_processor_outcomes.py:78`:

```python
@pytest.mark.parametrize(
    "outcome,kwargs",
    [
        (RowOutcome.ROUTED, {"sink_name": "error_sink"}),
        (RowOutcome.FORKED, {"fork_group_id": "fork_123"}),
        (RowOutcome.COALESCED, {"join_group_id": "join_123"}),
        (RowOutcome.EXPANDED, {"expand_group_id": "expand_123"}),
        (RowOutcome.FAILED, {"error_hash": "abc123"}),
    ],
)
...
recorded = recorder.get_token_outcome(token.token_id)
assert recorded.outcome == outcome
assert recorded.is_terminal == outcome.is_terminal
```

- Missing assertions for `recorded.sink_name`, `recorded.fork_group_id`, `recorded.join_group_id`, `recorded.expand_group_id`, `recorded.error_hash`, and `recorded.context_json` (no test uses `context` at all).

## Impact

- A regression that drops or miswrites outcome context fields would still pass, leaving audit trail lineage incomplete.
- Fork/join/expand provenance can be silently lost, undermining explainability while tests remain green.

## Root Cause Hypothesis

- The test was scoped to enum plumbing and terminal flag checks, and outcome-specific field verification was deferred.

## Recommended Fix

- Extend the parameterized test to assert the specific context field for each outcome, and add a dedicated `context_json` round-trip test.
- Example pattern (explicit assertions, no defensive getattr):

```python
if outcome in (RowOutcome.COMPLETED, RowOutcome.ROUTED):
    assert recorded.sink_name == kwargs["sink_name"]
elif outcome is RowOutcome.FORKED:
    assert recorded.fork_group_id == kwargs["fork_group_id"]
elif outcome is RowOutcome.COALESCED:
    assert recorded.join_group_id == kwargs["join_group_id"]
elif outcome is RowOutcome.EXPANDED:
    assert recorded.expand_group_id == kwargs["expand_group_id"]
elif outcome in (RowOutcome.FAILED, RowOutcome.QUARANTINED):
    assert recorded.error_hash == kwargs["error_hash"]
```

- Priority P1 because these fields are core audit-lineage data.
---
# Test Defect Report

## Summary

- The file claims to test processor outcome recording, but it only exercises `LandscapeRecorder` directly and never runs the processor/orchestrator path that should emit outcomes.

## Severity

- Severity: major
- Priority: P1

## Category

- Incomplete Contract Coverage

## Evidence

- File-level description claims processor integration: `tests/engine/test_processor_outcomes.py:2`.
- Tests call `LandscapeRecorder` directly (no processor/orchestrator usage), e.g. `tests/engine/test_processor_outcomes.py:23` and `tests/engine/test_processor_outcomes.py:58`-`tests/engine/test_processor_outcomes.py:59`.
- Processor is the component that should emit outcomes during execution, e.g. `src/elspeth/engine/processor.py:204`-`src/elspeth/engine/processor.py:212` shows `self._recorder.record_token_outcome(...)` inside processing logic.

## Impact

- A regression where the processor fails to record outcomes (or records wrong context) would not be caught here, despite the file claiming to validate processor behavior.
- Creates false confidence in AUD-001 coverage because only the recorder API is tested.

## Root Cause Hypothesis

- Tests were built around the recorder API for convenience and never expanded to an engine-level integration test.

## Recommended Fix

- Add an engine-level test in this file that runs a minimal pipeline through `RowProcessor` or `Orchestrator`, then queries the recorder for the resulting `token_outcomes` entry and validates outcome + context.
- Example steps: build a one-row source, run through a trivial transform or gate, route to a sink, then assert `recorder.get_token_outcome(token_id)` matches expected `RowOutcome` and `sink_name`.
- Priority P1 because AUD-001 explicitly requires recording at processor determination points.
