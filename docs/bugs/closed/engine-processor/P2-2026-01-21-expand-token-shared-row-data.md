# Bug Report: expand_token does not deep copy row data, allowing cross-token mutation

## Summary

- `TokenManager.fork_token()` deep-copies row data to prevent branch mutations from leaking across siblings.
- `TokenManager.expand_token()` returns row data as-is, so expanded tokens can share mutable structures if a transform returns shared objects.
- This can cause downstream mutations in one expanded token to affect siblings, corrupting audit data.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (local)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/tokens.py` and create bug tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: static inspection of token manager

## Steps To Reproduce

1. Implement a transform that returns `TransformResult.success_multi()` with shared objects, e.g. `rows = [row_template] * 2`.
2. Add a downstream transform that mutates `row_data` in place (e.g., `row["nested"]["x"] = 1`).
3. Run the pipeline and inspect both expanded tokens.
4. Observe that mutations applied to one token appear in the other.

## Expected Behavior

- Each expanded token should hold an isolated copy of its row data, preventing cross-token mutation.

## Actual Behavior

- Expanded tokens may share mutable structures, causing unintended data coupling between tokens.

## Evidence

- Forking uses deep copy to avoid shared state: `src/elspeth/engine/tokens.py` (`fork_token`).
- Expansion does not copy row data: `src/elspeth/engine/tokens.py` (`expand_token`).

## Impact

- User-facing impact: inconsistent or incorrect output rows when downstream transforms mutate data.
- Data integrity / security impact: audit trail can no longer reliably explain which row produced which output.
- Performance or cost impact: debugging and reprocessing costs increase.

## Root Cause Hypothesis

- Expansion path skipped the deep-copy protection that exists in fork_token.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/engine/tokens.py`: deep copy each expanded row when constructing `TokenInfo`, mirroring fork_token behavior.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that expands a token with shared nested objects and asserts that mutations in one child do not affect others.
- Risks or migration steps:
  - Deep copy increases memory usage for large rows; ensure acceptable performance or provide a documented opt-out if needed.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): audit integrity principles in `CLAUDE.md` (no silent data corruption).
- Observed divergence: expanded tokens can share mutable data, unlike forked tokens.
- Reason (if known): expand_token omitted deep copy.
- Alignment plan or decision needed: decide whether expansion should guarantee data isolation like fork.

## Acceptance Criteria

- Expanded tokens never share mutable row_data structures.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/engine/test_tokens.py -k expand`
- New tests required: yes (expand_token data isolation)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md`

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 5

**Current Code Analysis:**

Examined `/home/john/elspeth-rapid/src/elspeth/engine/tokens.py` at current HEAD (commit 7540e57).

**Confirmed asymmetry between fork_token and expand_token:**

1. **fork_token (lines 127-168):** Uses `copy.deepcopy(data)` at line 164 with explicit comment:
   ```python
   # CRITICAL: Use deepcopy to prevent nested mutable objects from being
   # shared across forked children. Shallow copy would cause mutations in
   # one branch to leak to siblings, breaking audit trail integrity.
   return [
       TokenInfo(
           row_id=parent_token.row_id,
           token_id=child.token_id,
           row_data=copy.deepcopy(data),  # Deep copy here
           branch_name=child.branch_name,
       )
       for child in children
   ]
   ```

2. **expand_token (lines 225-262):** Returns row_data directly without deep copy at line 258:
   ```python
   return [
       TokenInfo(
           row_id=parent_token.row_id,
           token_id=db_child.token_id,
           row_data=row_data,  # NO deep copy
           branch_name=parent_token.branch_name,
       )
       for db_child, row_data in zip(db_children, expanded_rows, strict=True)
   ]
   ```

**Test Coverage Gap:**

- `tests/engine/test_tokens.py` has comprehensive data isolation tests for `fork_token`:
  - `test_fork_nested_data_isolation` (line 172)
  - `test_fork_with_custom_nested_data_isolation` (line 222)
- NO equivalent tests exist for `expand_token` data isolation
- Existing expand_token tests (lines 465, 525, 569) only verify token creation, not mutation isolation

**Real-World Risk Confirmed:**

Found TWO plugins in production that create shallow copies when using success_multi:

1. **batch_replicate.py (line 145):**
   ```python
   output = dict(row)  # Shallow copy preserves original data
   output_rows.append(output)
   ```

2. **json_explode.py (line 169):**
   ```python
   output = dict(base)
   output_rows.append(output)
   ```

Both use `dict(row)` which creates shallow copies. If the input row contains nested structures (e.g., `{"payload": {"x": 1}}`), all expanded tokens will share the nested dict. A downstream transform mutating `row_data["payload"]["x"]` would affect ALL sibling tokens.

**Git History:**

```
b2a3518 fix(sources,resume): comprehensive data handling bug fixes
3399faf fix(engine): store source row payloads for audit compliance
8c96aff feat(engine): implement resume row processing in Orchestrator
0023108 feat(engine): add TokenManager.expand_token for deaggregation
```

- Commit `0023108` introduced `expand_token` without deep copy
- No subsequent commits addressed this issue
- Most recent commit `b2a3518` modified tokens.py but only touched payload storage (canonical_json), not expand_token

**Root Cause Confirmed:**

The bug is **100% present** in current code. When `expand_token` was implemented in commit 0023108, it did not include the deep copy protection that exists in `fork_token`. This creates an audit integrity vulnerability where:

1. Plugin returns `success_multi([row1, row2, row3])`
2. If those rows share nested mutable objects, expand_token passes them through as-is
3. Downstream transforms can mutate one token's nested data and corrupt siblings
4. Audit trail becomes unreliable - cannot determine which transform output actually produced which result

**Recommendation:**

**Keep open - HIGH PRIORITY for RC-2**

This is a **critical audit integrity bug** that violates ELSPETH's core principle: "The audit trail must withstand formal inquiry." The fix is straightforward (add `copy.deepcopy()` in expand_token like fork_token), but requires:

1. Adding deep copy to expand_token (mirror fork_token implementation)
2. Adding test coverage for expand_token data isolation
3. Potentially fixing batch_replicate.py and json_explode.py if they intentionally relied on shallow copy behavior (unlikely, but verify)
4. Performance testing to ensure deep copy overhead is acceptable for multi-row expansion scenarios

The bug is low-probability (requires nested data + downstream mutation) but **high-impact** (complete audit trail corruption). Should be fixed before production deployment.

---

## RESOLUTION: 2026-01-29

**Status:** FIXED

**Closed By:** Claude Code

**Fix Details:**

Added `copy.deepcopy(row_data)` to `expand_token` in `src/elspeth/engine/tokens.py`, mirroring the pattern established by `fork_token`.

**Code Change (line 249-262):**
```python
# CRITICAL: Use deepcopy to prevent nested mutable objects from being
# shared across expanded children. Same reasoning as fork_token - without
# this, mutations in one sibling leak to others, corrupting audit trail.
# Bug: P2-2026-01-21-expand-token-shared-row-data
return [
    TokenInfo(
        row_id=parent_token.row_id,
        token_id=db_child.token_id,
        row_data=copy.deepcopy(row_data),
        branch_name=parent_token.branch_name,
        expand_group_id=db_child.expand_group_id,
    )
    for db_child, row_data in zip(db_children, expanded_rows, strict=True)
]
```

**Tests Added:**

Created `TestTokenManagerExpandIsolation` class in `tests/engine/test_tokens.py` with 3 tests:
- `test_expand_nested_data_isolation` - nested dict/list mutations don't leak to siblings
- `test_expand_shared_input_isolation` - shared objects in expanded_rows list are isolated
- `test_expand_deep_nesting_isolation` - isolation works at 3+ levels of nesting

**Acceptance Criteria Met:**
- ✅ `expand_token` uses `copy.deepcopy()` like `fork_token`
- ✅ New isolation tests pass (3/3)
- ✅ All existing tests pass (656 passed, 1 skipped)
- ✅ Symmetry restored between `fork_token` and `expand_token`

**Review Process:**

4-perspective review board (architecture, python, QA, systems thinking) approved with recommendations:
- Architecture: APPROVE - correct location, restores symmetry
- Python: REQUEST CHANGES → addressed by adding deepcopy
- QA: REQUEST CHANGES → addressed by adding isolation tests
- Systems Thinking: APPROVE with follow-up - deepcopy is correct leverage point

**Follow-Up Work (Tracked Separately):**
- Performance benchmarking for large row expansions
- Consider canonical-safe validation at transform boundaries
- Document "Token Isolation Invariant" in architecture docs
