# Bug Report: Duplicate Fork/Coalesce Branch Names Accepted, Causing Stalls and Token Overwrites

## Status: CLOSED - ALREADY FIXED

**Closure Date:** 2026-01-28
**Closed By:** Systematic debugging investigation
**Resolution:** Bug was already fixed prior to this report being filed

## Summary

- DAG builder accepts duplicate `branch_name` values in fork_to_paths configuration, causing coalesce config maps to silently overwrite earlier entries, leading to tokens never reaching coalesce nodes and pipeline stalls.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Branch Bug Scan (fix/rc1-bug-burndown-session-4)
- Date: 2026-01-25
- Related run/issue ID: BUG-DAG-01

---

## Investigation Results (2026-01-28)

### Finding: Comprehensive Validation Already Exists

The bug report claims that duplicate branch names are silently accepted, but systematic investigation revealed that **validation is already implemented** at multiple points in `src/elspeth/core/dag.py`:

#### 1. Within-Gate Duplicates (Lines 620-626)
```python
for gate_entry in gate_entries:
    if gate_entry.fork_to:
        branch_counts = Counter(gate_entry.fork_to)
        duplicates = sorted([branch for branch, count in branch_counts.items() if count > 1])
        if duplicates:
            raise GraphValidationError(
                f"Gate '{gate_entry.name}' has duplicate fork branches: {duplicates}. "
                "Each fork branch name must be unique."
            )
```

#### 2. Cross-Coalesce Duplicates (Lines 591-602)
```python
for branch_name in coalesce_config.branches:
    if BranchName(branch_name) in branch_to_coalesce:
        existing_coalesce = branch_to_coalesce[BranchName(branch_name)]
        raise GraphValidationError(
            f"Duplicate branch name '{branch_name}' found in coalesce settings.\n"
            f"Branch '{branch_name}' is already mapped to coalesce '{existing_coalesce}', "
            f"but coalesce '{coalesce_config.name}' also declares it.\n"
            f"Each fork branch can only merge at one coalesce point."
        )
```

#### 3. Edge Label Duplicates (Lines 211-221)
```python
for node_id in self._graph.nodes():
    labels_seen: set[str] = set()
    for _, _, edge_key in self._graph.out_edges(node_id, keys=True):
        if edge_key in labels_seen:
            raise GraphValidationError(
                f"Node '{node_id}' has duplicate outgoing edge label '{edge_key}'. "
                "Edge labels must be unique per source node..."
            )
        labels_seen.add(edge_key)
```

### Test Coverage

All duplicate validation scenarios have passing tests:

```
tests/core/test_dag.py::TestDAGValidation::test_validate_rejects_duplicate_outgoing_edge_labels PASSED
tests/core/test_dag.py::TestCoalesceNodes::test_duplicate_fork_branches_rejected_in_config_gate PASSED
tests/core/test_dag.py::TestCoalesceNodes::test_duplicate_fork_branches_rejected_in_plugin_gate PASSED
tests/core/test_dag.py::TestCoalesceNodes::test_duplicate_branch_names_across_coalesces_rejected PASSED
tests/property/audit/test_fork_join_balance.py::TestDagForkBranchValidation::test_duplicate_fork_branches_rejected PASSED
tests/property/audit/test_fork_join_balance.py::TestForkJoinEnumProperties::test_routing_action_fork_rejects_duplicates PASSED
```

### Verification Commands

```bash
# Run all duplicate-related tests
.venv/bin/python -m pytest tests/core/test_dag.py -k "duplicate" -v

# Run property tests for fork/join validation
.venv/bin/python -m pytest tests/property/audit/test_fork_join_balance.py -k "duplicate" -v
```

### Conclusion

The bug report was likely filed based on:
1. Static code analysis that missed the validation logic
2. Analysis of an older codebase version before validation was added
3. Misunderstanding of the config structure (YAML `fork_to` is a list, not `fork_to_paths` with nested objects)

**No code changes required** - validation is comprehensive and tested.

---

## Original Bug Report Content

*(Preserved below for reference)*

### Environment

- Commit/branch: fix/rc1-bug-burndown-session-4
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Fork/coalesce DAG with duplicate branch names

### Agent Context (if relevant)

- Goal or task prompt: Static analysis agent doing a deep bug audit of dag.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

### Steps To Reproduce

1. Configure a gate with fork_to_paths using duplicate branch names:
   ```yaml
   fork_to_paths:
     - branch_name: "analysis"
       nodes: [transform_a]
     - branch_name: "analysis"  # DUPLICATE
       nodes: [transform_b]
   ```
2. Configure coalesce node expecting both branches.
3. Run pipeline with forking gate.

### Expected Behavior

- DAG builder should reject duplicate branch names with validation error at config parse time.

### Actual Behavior

**CORRECTED:** DAG builder DOES reject duplicates with clear `GraphValidationError` messages.

### Acceptance Criteria

- ✅ DAG builder rejects configs with duplicate branch names with clear error message.
- ✅ Existing unique branch name configs continue to work.

### Tests

- ✅ Tests already exist and pass (see Test Coverage section above)
