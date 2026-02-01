# Test Defect Report

## Summary

- Re-export test only asserts `RoutingSpec` is not `None`, so it does not verify that `elspeth.core.landscape.RoutingSpec` is the same class re-exported from `elspeth.contracts`; a wrong/duplicate class would still pass.

## Severity

- Severity: minor
- Priority: P3

## Category

- Weak Assertions

## Evidence

- `tests/core/landscape/test_routing.py:19-22` uses a vacuous assertion that would pass even if `RoutingSpec` were the wrong class:
```python
def test_can_import_from_landscape(self) -> None:
    """RoutingSpec should be importable from landscape package."""
    # This import above confirms it works
    assert RoutingSpec is not None
```
- The stated purpose is to verify re-export path correctness, but no identity check exists anywhere in this file (e.g., nothing compares against `elspeth.contracts.RoutingSpec`). This allows a duplicate/incorrect class to pass unnoticed.

## Impact

- A mismatched `RoutingSpec` class (e.g., reintroduced in `core.landscape.models`) could slip through tests, leading to subtle runtime bugs such as failed identity checks or inconsistent typing across subsystems while tests still pass.
- The test suite gives false confidence that the public re-export API is correct when it is only confirming that an import succeeded.

## Root Cause Hypothesis

- The test was likely copied from contract-level tests and pared down, leaving only a placeholder assertion rather than an explicit re-export identity check.

## Recommended Fix

- Strengthen `test_can_import_from_landscape` to verify that the re-export points to the exact contract class (identity check), not just that it exists.
- Example fix in `tests/core/landscape/test_routing.py`:
```python
from elspeth.contracts import RoutingSpec as ContractsRoutingSpec

def test_can_import_from_landscape(self) -> None:
    assert RoutingSpec is ContractsRoutingSpec
```
- Priority justification: This is a low-effort change that meaningfully increases confidence in a public API surface; P3 is appropriate because it does not currently cause failures but reduces regression coverage.
