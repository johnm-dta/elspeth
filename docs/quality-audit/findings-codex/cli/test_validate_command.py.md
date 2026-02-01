# Test Defect Report

## Summary

- `test_validate_valid_config` uses a substring check for `"valid"` that also matches `"invalid"`, so the test can pass on incorrect failure messaging.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- tests/cli/test_validate_command.py:97-101
```python
result = runner.invoke(app, ["validate", "-s", str(valid_config)])
assert result.exit_code == 0
assert "valid" in result.stdout.lower()
```
- src/elspeth/cli.py:849
```python
typer.echo("✅ Pipeline configuration valid!")
```
- The substring `"valid"` is contained in `"invalid"`, so a regression that prints an error message like `"invalid configuration"` while still exiting `0` would not be caught.

## Impact

- Success messaging regressions (or mis-signaled failures with exit code `0`) can slip through.
- The test provides false confidence that the CLI reports success correctly.

## Root Cause Hypothesis

- Overly permissive substring assertion used to avoid coupling to the full message text.

## Recommended Fix

- Assert on the full success phrase (or use a word-boundary regex) to avoid matching `"invalid"`.
- Example:
```python
assert "pipeline configuration valid" in result.stdout.lower()
```
- Priority justification: This is a simple test fix that prevents a clear false positive path.
---
# Test Defect Report

## Summary

- `test_validate_shows_graph_info` only checks for the words “graph/node/edge” and does not validate the node/edge counts that the CLI reports.

## Severity

- Severity: trivial
- Priority: P3

## Category

- Weak Assertions

## Evidence

- tests/cli/test_validate_command.py:208-212
```python
assert result.exit_code == 0
# Should show graph info with node and edge counts
assert "graph" in result.stdout.lower()
assert "node" in result.stdout.lower()
assert "edge" in result.stdout.lower()
```
- src/elspeth/cli.py:854
```python
typer.echo(f"  Graph: {graph.node_count} nodes, {graph.edge_count} edges")
```
- The test does not verify that numeric counts are present or correct, even though the output includes them.

## Impact

- Incorrect graph counts (or missing numeric values) could go unnoticed.
- Output regressions affecting audit/debug clarity are not detected.

## Root Cause Hypothesis

- Keyword-only assertions chosen instead of parsing the structured output.

## Recommended Fix

- Parse the output line and assert the expected counts for this config (1 source + 1 gate + 2 sinks = 4 nodes; edges: source→gate, gate→results, gate→flagged = 3).
- Example:
```python
import re

match = re.search(r"graph:\s*(\d+)\s+nodes,\s*(\d+)\s+edges", result.stdout.lower())
assert match
assert match.group(1) == "4"
assert match.group(2) == "3"
```
- Priority justification: Improves correctness with minimal test change.
