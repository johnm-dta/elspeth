Using skill: using-quality-engineering (test quality audit).

# Test Defect Report

## Summary

- Tests that claim to validate pipeline structure and selection only assert run_id or selection id, leaving lineage content and detail panel behavior unverified

## Severity

- Severity: minor
- Priority: P3

## Category

- Weak Assertions

## Evidence

- `tests/cli/test_explain_tui.py:75` and `tests/cli/test_explain_tui.py:78` only check `run_id` after registering nodes; no assertion that the source/sink from the DB is actually loaded.
```python
lineage = screen.get_lineage_data()

assert lineage is not None
assert lineage["run_id"] == run.run_id
```
- `tests/cli/test_explain_tui.py:91` and `tests/cli/test_explain_tui.py:94` only check selection tracking; the detail panel update is not asserted.
```python
screen.on_tree_select(mock_node_id)

assert screen._selected_node_id == mock_node_id
```
- `src/elspeth/tui/screens/explain_screen.py:146` builds `lineage_data` with source/transforms/sinks that the tests never assert.
```python
lineage_data: LineageData = {
    "run_id": run_id,
    "source": {"name": source_nodes[0].plugin_name if source_nodes else "unknown", ...},
    "transforms": [{"name": n.plugin_name, "node_id": n.node_id} for n in transform_nodes],
    "sinks": [{"name": n.plugin_name, "node_id": n.node_id} for n in sink_nodes],
    "tokens": [],
}
```

## Impact

- Regressions where nodes are not loaded (or selection doesn’t update the detail panel) can pass undetected
- Tests give false confidence about the TUI’s lineage rendering and selection behavior

## Root Cause Hypothesis

- Tests were written as smoke checks and stopped at non-null assertions
- No explicit assertions were added for the actual lineage content or detail panel state

## Recommended Fix

- Strengthen `test_screen_loads_pipeline_structure` to assert `lineage["source"]["name"] == "csv_source"` and that `lineage["sinks"]` contains `"csv_sink"`
- Update `test_tree_selection_updates_detail_panel` to create a node in the DB, call `on_tree_select(node_id)`, then assert `get_detail_panel_state()` has matching `plugin_name`, `node_type`, and `node_id`
- Add assertions against `LineageTree.get_tree_nodes()` labels if the intent is to validate rendered structure
---
# Test Defect Report

## Summary

- Test asserts a private attribute (`_selected_node_id`) instead of public, user-visible behavior

## Severity

- Severity: trivial
- Priority: P3

## Category

- Infrastructure Gaps

## Evidence

- `tests/cli/test_explain_tui.py:94` directly accesses a private attribute.
```python
assert screen._selected_node_id == mock_node_id
```

## Impact

- Tests become brittle to refactors (renaming or removing `_selected_node_id` breaks tests)
- Passing tests may not reflect actual UI behavior

## Root Cause Hypothesis

- Convenience in asserting internal state instead of observable output

## Recommended Fix

- Assert via public API, e.g., `get_detail_panel_state()` or `render()` output after selection
- If selection visibility is needed, assert on `LineageTree` output or detail panel text rather than private state
---
# Test Defect Report

## Summary

- Loading failure path (LoadingFailedState) is never exercised, leaving error handling untested

## Severity

- Severity: minor
- Priority: P2

## Category

- Missing Negative Tests

## Evidence

- `src/elspeth/tui/screens/explain_screen.py:165` shows exceptions should produce a LoadingFailedState.
```python
except Exception as e:
    ...
    return LoadingFailedState(db=db, run_id=run_id, error=str(e))
```
- `tests/cli/test_explain_tui.py:205` only references `LoadingFailedState` in pattern matching; no test induces a loading failure.
```python
match screen.state:
    case UninitializedState():
        result = "uninitialized"
    case LoadingFailedState(run_id=rid):
        result = f"failed:{rid}"
    case LoadedState(run_id=rid, lineage_data=data):
        result = f"loaded:{rid}:{data['run_id']}"
```

## Impact

- Regressions in error handling (e.g., DB errors, exceptions in `get_nodes`) can slip through without detection
- UI may crash or fail to display error state without any test signal

## Root Cause Hypothesis

- Happy-path bias; failure injection not included in this test suite

## Recommended Fix

- Add a test that monkeypatches `LandscapeRecorder.get_nodes` to raise, then assert `isinstance(screen.state, LoadingFailedState)` and validate `screen.state.error` and `state_type`
