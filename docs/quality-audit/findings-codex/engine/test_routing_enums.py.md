# Test Defect Report

## Summary
- `tests/engine/test_routing_enums.py` is in the engine suite but only tests contract enums and `RoutingAction`, duplicating coverage already present in contract/plugin tests and not exercising engine behavior.

## Severity
- Severity: minor
- Priority: P2

## Category
- Misclassified Tests

## Evidence
- `tests/engine/test_routing_enums.py:1-37` imports only `elspeth.contracts` and performs enum/value checks; no engine components are touched.
- `tests/plugins/test_enums.py:29-37` already verifies `RoutingKind` values.
- `tests/plugins/test_results.py:162-195` already verifies `RoutingAction` uses `RoutingKind`.

```python
# tests/engine/test_routing_enums.py
from elspeth.contracts.enums import RoutingKind

def test_routing_action_uses_enum(self) -> None:
    from elspeth.contracts import RoutingAction
    action = RoutingAction.continue_()
    assert isinstance(action.kind, RoutingKind)
```

```python
# tests/plugins/test_enums.py
def test_all_routing_kinds_defined(self) -> None:
    from elspeth.contracts import RoutingKind
    assert RoutingKind.CONTINUE.value == "continue"
```

```python
# tests/plugins/test_results.py
def test_continue_uses_routing_kind_enum(self) -> None:
    from elspeth.contracts import RoutingKind
    from elspeth.plugins.results import RoutingAction
    action = RoutingAction.continue_()
    assert action.kind == RoutingKind.CONTINUE
```

## Impact
- Creates a false sense of engine coverage while only validating contract-level enums.
- Redundant tests add maintenance overhead and can mask gaps in engine routing behavior.
- Increases test suite noise, making it harder to locate true engine regressions.

## Root Cause Hypothesis
- Likely copied from contract/plugin tests or added as a placeholder in the engine suite without being relocated or expanded.

## Recommended Fix
- Move or delete this file and rely on existing contract/plugin tests for enum checks, or rewrite it to exercise engine routing behavior (e.g., `GateExecutor` branching and routing-event recording).
- If keeping it as engine coverage, add assertions that the executor processes `RoutingAction` kinds and records routing events in the audit trail.

```python
# Example direction (engine-focused)
outcome = executor.execute_gate(...)
assert outcome.result.action.kind == RoutingKind.CONTINUE
events = recorder.get_routing_events(state_id)
assert len(events) == 1
```
---
# Test Defect Report

## Summary
- The tests contain tautological assertions (enum compared to itself), providing minimal or no behavioral coverage and failing to validate actual routing comparisons.

## Severity
- Severity: minor
- Priority: P2

## Category
- Weak Assertions

## Evidence
- `tests/engine/test_routing_enums.py:9-25` asserts that an enum equals itself, which is always true regardless of system behavior.

```python
# tests/engine/test_routing_enums.py
kind = RoutingKind.CONTINUE
assert kind == RoutingKind.CONTINUE
```

## Impact
- These assertions will pass even if engine logic misroutes, compares against wrong kinds, or fails to record routing events.
- Gives false confidence that routing comparisons are tested when no behavior is exercised.

## Root Cause Hypothesis
- Placeholder assertions added to “prove” enum usage without connecting to actual engine behavior.

## Recommended Fix
- Replace tautological checks with behavior-driven assertions tied to engine routing decisions, or remove the tests if covered elsewhere.
- If the intent is to validate enum correctness, assert meaningful conversions (e.g., string coercion or invalid value handling), but avoid duplicating existing property/contract tests.

```python
# Example direction (behavioral)
action = RoutingAction.route("sink1")
assert action.destinations == ("sink1",)
assert action.mode == RoutingMode.MOVE
```
