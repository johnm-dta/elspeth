# Bug Report: Duplicate Fork/Coalesce Branch Names Break Merge Semantics

## Summary

`fork_to` and `coalesce.branches` allow duplicate branch names; coalesce tracking uses a dict keyed by branch name, so duplicates overwrite tokens and can prevent `require_all/quorum` merges from ever completing.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (static analysis agent)
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: main (d8df733)
- OS: Linux
- Python version: 3.12+
- Config profile / env vars: Pipeline with fork and coalesce configuration
- Data set or fixture: Any

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit for `src/elspeth/core/config.py`
- Model/version: GPT-5 (Codex)
- Tooling and permissions (sandbox/approvals): read-only sandbox, approvals disabled
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Reviewed config.py, dag.py, coalesce_executor.py

## Steps To Reproduce

1. Define a gate with `routes: {all: fork}` and `fork_to: ["path_a", "path_a"]` (duplicate branch)
2. Define a coalesce with `branches: ["path_a", "path_a"]` and `policy: require_all`
3. Run the pipeline
4. Coalesce never completes because only one unique branch can arrive

## Expected Behavior

- Duplicate branch names are rejected at config validation for both `fork_to` and `coalesce.branches`

## Actual Behavior

- Duplicates are accepted
- Coalesce overwrites arrivals and may stall indefinitely or drop a token

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots):
  - `src/elspeth/core/config.py:238` validates `fork_to` only for reserved labels, not uniqueness
  - `src/elspeth/core/config.py:327` defines `CoalesceSettings.branches` without uniqueness validation
  - `src/elspeth/engine/coalesce_executor.py:172` stores arrivals by `branch_name`, overwriting duplicates
  - `src/elspeth/engine/coalesce_executor.py:195` compares `arrived_count` to `len(settings.branches)`, so duplicates can prevent merges
  - `src/elspeth/core/dag.py:415` maps `branch_to_coalesce` by branch name, overwriting duplicates across coalesce configs
- Minimal repro input (attach or link): Config YAML with duplicate branch names in fork_to or coalesce.branches

## Impact

- User-facing impact: Pipelines can hang at coalesce or route fewer results than expected
- Data integrity / security impact: Tokens can be overwritten, causing silent loss of branch results
- Performance or cost impact: Runs may stall until timeout or require manual intervention

## Root Cause Hypothesis

Missing uniqueness checks for branch lists (`fork_to`, `coalesce.branches`) and across coalesce configurations.

## Proposed Fix

- Code changes (modules/files): Add uniqueness validation in `GateSettings.validate_fork_to_labels` and a new validator in `CoalesceSettings` (or `ElspethSettings`) to enforce unique branch names. Optionally validate global uniqueness across coalesce definitions.
- Config or schema changes: None
- Tests to add/update: Add tests for duplicate `fork_to` and duplicate `coalesce.branches` in `tests/core/test_config.py`
- Risks or migration steps: Configs with duplicate branch names will fail fast

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): Unknown
- Observed divergence: Branch identifiers are treated as unique keys at runtime but not enforced in config
- Reason (if known): Unknown
- Alignment plan or decision needed: Enforce uniqueness at config validation

## Acceptance Criteria

- Duplicate branch names in `fork_to` or `coalesce.branches` are rejected with a clear error
- Coalesce merges complete when all distinct branches arrive

## Tests

- Suggested tests to run: `pytest tests/core/test_config.py -k coalesce`
- New tests required: Yes, duplicate branch validation

## Notes / Links

- Related issues/PRs:
  - Related to `docs/bugs/open/P2-2026-01-22-coalesce-duplicate-branch-overwrite.md` (runtime manifestation)
- Related design docs: Unknown

## Verification Status

**Status: STILL VALID** (verified 2026-01-24)

- [x] Bug confirmed via reproduction
- [x] Root cause verified
- [ ] Fix implemented
- [ ] Tests added
- [ ] Fix verified

### Verification Details

**Verifier:** Claude Sonnet 4.5 (code verification agent)
**Date:** 2026-01-24
**Branch:** fix/rc1-bug-burndown-session-4 (commit 36e17f2)

#### Reproduction Confirmed

Tested with live code on current branch:

```python
# Test 1: GateSettings accepts duplicate fork_to branches
from elspeth.core.config import GateSettings
gate = GateSettings(
    name='test_gate',
    condition='row["x"] > 5',
    routes={'true': 'fork', 'false': 'continue'},
    fork_to=['path_a', 'path_a']  # duplicates accepted ✗
)
# RESULT: No error raised - bug confirmed

# Test 2: CoalesceSettings accepts duplicate branches
from elspeth.core.config import CoalesceSettings
coalesce = CoalesceSettings(
    name='test_coalesce',
    branches=['path_a', 'path_a'],  # duplicates accepted ✗
    policy='require_all',
    merge='union'
)
# RESULT: No error raised - bug confirmed
```

#### Root Cause Analysis

**Validated all locations mentioned in bug report:**

1. **Config validation gaps (as reported):**
   - `src/elspeth/core/config.py:238-252` - `GateSettings.validate_fork_to_labels()` only checks for reserved labels, NOT uniqueness
   - `src/elspeth/core/config.py:327-331` - `CoalesceSettings.branches` field has NO uniqueness validator
   - No field-level or model-level validator exists for either case

2. **Runtime manifestation (as reported):**
   - `src/elspeth/core/dag.py:485-486` - Branch-to-coalesce mapping uses dict, duplicates silently overwrite:
     ```python
     for branch_name in coalesce_config.branches:
         branch_to_coalesce[branch_name] = cid  # Last duplicate wins
     ```
   - `src/elspeth/engine/coalesce_executor.py:173` - Token arrivals keyed by branch_name, overwrites on duplicate
   - `src/elspeth/engine/coalesce_executor.py:195-196` - Merge policy compares `len(pending.arrived)` to `len(settings.branches)`, math breaks with duplicates

3. **Partial fix found:**
   - Commit `2bb7617` (2026-01-23) added `validate_unique_gate_names()` and `validate_unique_coalesce_names()`
   - This validates GATE/COALESCE names are unique, but NOT branch names within fork_to/branches lists
   - The bug remains unfixed

4. **Runtime defense exists (partial mitigation):**
   - `src/elspeth/contracts/routing.py:131-133` - `RoutingAction.fork_to_paths()` DOES validate uniqueness at runtime
   - This prevents duplicate branches if gates return dynamic fork paths
   - Does NOT protect against config-time duplicates in `fork_to` lists (different code path)

#### Impact Validation

**Confirmed critical failure modes:**

1. **Stalled pipelines:**
   - Config: `fork_to: ["A", "A"]`, `coalesce.branches: ["A", "A"]`, `policy: require_all`
   - Only 1 unique branch can arrive, merge requires 2 arrivals (len of branches list)
   - Coalesce waits forever or until timeout

2. **Silent data loss:**
   - If two coalesces declare the same branch name, `branch_to_coalesce` map overwrites
   - Tokens route to wrong coalesce or get lost

3. **Audit integrity violation:**
   - Duplicate arrivals overwrite `pending.arrived[branch_name]` without recording loss
   - First token's data silently discarded, audit trail incomplete

#### Git History Search

No commits since 2026-01-20 address branch name uniqueness validation:
- `git log --all --since=2026-01-20 --grep="duplicate\|branch\|uniqueness" -i`
- Found `2bb7617` which validates gate/coalesce NAMES but not BRANCH names
- Found `0c225ee` which fixed invalid fork/coalesce test configs but added no validation

#### Related Bugs

- **P2-2026-01-22-coalesce-duplicate-branch-overwrite.md** - Runtime manifestation of same root cause (duplicate arrivals overwrite tokens)
- Both bugs stem from missing config validation; fixing this bug prevents runtime manifestation

#### Conclusion

**Bug is STILL VALID.** No fix has been implemented. The issue exists exactly as described:
- Config validation missing for `fork_to` and `coalesce.branches` uniqueness
- Runtime code assumes unique branch names, fails silently when assumption violated
- Can cause pipeline stalls, data loss, and audit integrity violations
