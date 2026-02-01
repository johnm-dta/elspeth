# Test Defect Report

## Summary

- `test_aggregation_dual_schema_both_edges_validated` claims to validate both aggregation edges but only asserts that *either* missing field appears, so it can pass even if only one edge is validated

## Severity

- Severity: major
- Priority: P1

## Category

- Weak Assertions

## Evidence

- `tests/core/test_edge_validation.py:208`-`tests/core/test_edge_validation.py:237` shows the test intent vs. the weak assertion:
```python
def test_aggregation_dual_schema_both_edges_validated() -> None:
    """Aggregations have both input_schema and output_schema - validate both edges."""
    ...
    # Should detect BOTH mismatches (source→agg has 'label' missing, agg→sink has 'average' missing)
    with pytest.raises(ValueError, match=r"label|average"):
        graph.validate_edge_compatibility()
```
- `src/elspeth/core/dag.py:667`-`src/elspeth/core/dag.py:734` shows validation stops at the first incompatible edge, so a single run cannot prove both edges are checked:
```python
def validate_edge_compatibility(self) -> None:
    for from_id, to_id, _edge_data in self._graph.edges(data=True):
        self._validate_single_edge(from_id, to_id)

def _validate_single_edge(...):
    ...
    if missing_fields:
        raise ValueError(...)
```

## Impact

- A regression that skips validation of the aggregation output edge (agg → sink) would still pass this test if the input edge fails first
- This creates false confidence that both aggregation edges are validated when only one might be
- Incompatible aggregation outputs could slip through undetected, corrupting downstream schema assumptions

## Root Cause Hypothesis

- The test attempted to cover two failures in a single run, but validation short-circuits on the first error
- The assertion was relaxed (`label|average`) to accommodate nondeterministic edge order, weakening coverage

## Recommended Fix

- Split into two explicit tests (or parametrize) so each edge is validated independently:
  - Case A: source → agg mismatches (agg → sink compatible), assert missing `label`
  - Case B: agg → sink mismatches (source → agg compatible), assert missing `average`
- Use precise `match` expectations per case (and optionally assert node IDs) to guarantee each edge is exercised
- Priority justification: aggregation edges are core DAG integrity checks; weakened assertions mask regressions on a critical path
