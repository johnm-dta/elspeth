# Test Defect Report

## Summary

- Missing negative tests for `RoutingAction.__post_init__` invariants (CONTINUE destinations, FORK_TO_PATHS mode, ROUTE destination count) leaves invalid constructor paths unverified.

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Edge Cases

## Evidence

- `src/elspeth/contracts/routing.py:61` guards CONTINUE with non-empty destinations but no test asserts this error path.
- `src/elspeth/contracts/routing.py:64` guards FORK_TO_PATHS with MOVE mode but no test asserts this error path.
- `src/elspeth/contracts/routing.py:67` guards ROUTE destination count but no test asserts this error path.
- `tests/contracts/test_routing.py:48` only covers ROUTE + COPY; `tests/contracts/test_routing.py:122` only covers fork empty/duplicate.

```python
# src/elspeth/contracts/routing.py
if self.kind == RoutingKind.CONTINUE and self.destinations:
    raise ValueError("CONTINUE must have empty destinations")
if self.kind == RoutingKind.FORK_TO_PATHS and self.mode != RoutingMode.COPY:
    raise ValueError("FORK_TO_PATHS must use COPY mode")
if self.kind == RoutingKind.ROUTE and len(self.destinations) != 1:
    raise ValueError("ROUTE must have exactly one destination")
```

## Impact

- Regressions in constructor invariants can slip through when `RoutingAction` is instantiated directly.
- Invalid routing actions can violate single-terminal-state and routing semantics, undermining audit integrity.
- Tests give false confidence that all contract invariants are enforced.

## Root Cause Hypothesis

- Test coverage focuses on factory methods and happy paths, skipping direct dataclass construction error cases.

## Recommended Fix

- Add `pytest.raises(ValueError, match=...)` tests in `tests/contracts/test_routing.py` for:
  - CONTINUE with non-empty destinations.
  - FORK_TO_PATHS with MOVE mode.
  - ROUTE with zero or multiple destinations.
- Example pattern:

```python
with pytest.raises(ValueError, match="CONTINUE must have empty destinations"):
    RoutingAction(
        kind=RoutingKind.CONTINUE,
        destinations=("sink",),
        mode=RoutingMode.MOVE,
        reason={},
    )
```
---
# Test Defect Report

## Summary

- Tests use prohibited defensive runtime checks (`hasattr`, `isinstance`) on system code, which are redundant and conflict with the no-defensive-programming rule.

## Severity

- Severity: minor
- Priority: P3

## Category

- Bug-Hiding Defensive Patterns

## Evidence

- `tests/contracts/test_routing.py:20` uses `hasattr` to probe a system attribute.
- `tests/contracts/test_routing.py:96` uses `isinstance` on `MappingProxyType`.
- `tests/contracts/test_routing.py:159` uses `isinstance` on `RoutingMode`.

```python
# tests/contracts/test_routing.py
assert hasattr(action, "mode")
assert isinstance(action.reason, MappingProxyType)
assert isinstance(spec.mode, RoutingMode)
```

## Impact

- Normalizes defensive runtime checks that the codebase explicitly prohibits.
- Adds redundant assertions that do not increase correctness beyond direct access and enum identity checks.
- Encourages patterns that can mask contract violations if copied into production code.

## Root Cause Hypothesis

- Habitual use of introspection to "document" fields and types instead of relying on direct access and strict contracts.

## Recommended Fix

- Remove `hasattr`/`isinstance` assertions and rely on direct access and enum identity:
  - Replace with `assert action.mode is RoutingMode.MOVE` or equality checks already present.
  - Keep immutability checks via assignment error instead of type probing.
- Example adjustment:

```python
# Before
assert hasattr(action, "mode")
assert action.mode == RoutingMode.MOVE

# After
assert action.mode is RoutingMode.MOVE
```
