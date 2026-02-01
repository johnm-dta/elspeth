# Bug Report: Forked Branches Never Coalesce Due to Wrong Mapping Key

## Summary

- `branch_to_coalesce` mapping uses node IDs as keys, but coalesce lookup uses branch names, causing all coalesce lookups to fail and forked tokens to never reach coalesce nodes.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Branch Bug Scan (fix/rc1-bug-burndown-session-4)
- Date: 2026-01-25
- Related run/issue ID: BUG-LINEAGE-01

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-4
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Any fork/coalesce DAG

## Agent Context (if relevant)

- Goal or task prompt: Static analysis agent doing a deep bug audit of lineage.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure fork/coalesce DAG.
2. Run pipeline with forking gate.
3. Observe tokens never reach coalesce node.

## Expected Behavior

- Tokens with `branch_name="analysis"` should map to coalesce node via `branch_to_coalesce["analysis"]`.

## Actual Behavior

- Mapping stores `{node_id: coalesce_node}` but lookup uses `token.branch_name` (string), always returns None.

## Evidence

```python
# lineage.py - mapping creation
branch_to_coalesce = {node_id: coalesce_node_id}  # Keys are node IDs

# Fork creates tokens
token.branch_name = "analysis"  # String value

# Coalesce lookup
coalesce_target = branch_to_coalesce.get(token.branch_name)  # None! (key mismatch)
```

## Impact

- User-facing impact: Fork/coalesce completely broken, tokens never coalesce.
- Data integrity / security impact: Rows stuck in parallel branches, never merged.
- Performance or cost impact: Pipeline stalls waiting for coalesce.

## Root Cause Hypothesis

- Mapping created with node IDs as keys, but accessed with branch names.

## Proposed Fix

```python
# Store mapping as branch_name → coalesce_node
branch_to_coalesce = {branch_name: coalesce_node_id}

# Fork: token.branch_name from config
# Coalesce: Lookup by token.branch_name (now works)
```

- Config or schema changes: None.
- Tests to add/update:
  - `test_forked_tokens_reach_coalesce()` - End-to-end test

- Risks or migration steps: None (pure bug fix).

## Architectural Deviations

- Spec or doc reference: N/A
- Observed divergence: Type mismatch between map keys and lookup keys.
- Reason (if known): Implementation error.
- Alignment plan or decision needed: Fix key type.

## Acceptance Criteria

- Forked tokens reach coalesce node.

## Tests

- Suggested tests to run: `pytest tests/integration/test_fork_coalesce.py`
- New tests required: yes

## Notes / Links

- Related issues/PRs: Critical for fork/coalesce functionality
- Related design docs: `docs/bugs/BRANCH_BUG_TRIAGE_2026-01-25.md`
