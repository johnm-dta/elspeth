# Bug Report: `NodeRepository.load()` drops `schema_mode` / `schema_fields` (WP-11.99 audit schema config lost on read)

## Summary

- Nodes store schema configuration for audit trail via `nodes.schema_mode` and `nodes.schema_fields_json` (WP-11.99).
- `NodeRepository.load()` constructs `Node(...)` without populating these fields, so consumers using the repository layer silently lose schema audit metadata.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `main` @ `8ca061c9293db459c9a900f2f74b19b59a364a42`
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive subsystem 4 (Landscape) and create bug tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection

## Steps To Reproduce

1. Insert a `nodes` row with non-NULL `schema_mode` and `schema_fields_json`.
2. Load it via `NodeRepository.load(row)` (as used in repository patterns/tests).
3. Observe returned `Node.schema_mode` / `Node.schema_fields` are defaulted to `None`.

## Expected Behavior

- Repository layer returns complete Node objects, including schema configuration fields.

## Actual Behavior

- Schema config fields are silently dropped.

## Evidence

- Node schema includes audit schema configuration columns:
  - `src/elspeth/core/landscape/schema.py` (`nodes.schema_mode`, `nodes.schema_fields_json`)
- Node contract includes these fields:
  - `src/elspeth/contracts/audit.py:49-82`
- Repository load omits them:
  - `src/elspeth/core/landscape/repositories.py:65-88`
- Recorder has a separate `_row_to_node` path that *does* load them (inconsistent layering):
  - `src/elspeth/core/landscape/recorder.py:580-608`

## Impact

- User-facing impact: any UI/reporting that uses repository layer may omit schema audit details.
- Data integrity / security impact: low (data is stored), but audit explainability is reduced.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- Repository module was not updated when WP-11.99 schema columns were added.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/core/landscape/repositories.py`:
    - Parse `schema_fields_json` (JSON) and populate `schema_fields`
    - Populate `schema_mode`
- Config or schema changes: none.
- Tests to add/update:
  - Add a repository test that includes schema_mode/schema_fields_json and asserts fields are preserved.
- Risks or migration steps:
  - None.

## Architectural Deviations

- Spec or doc reference: WP-11.99 (“Config-Driven Plugin Schemas”)
- Observed divergence: repository layer drops schema config on read.
- Reason (if known): missed update during schema feature addition.
- Alignment plan or decision needed: none.

## Acceptance Criteria

- `NodeRepository.load()` preserves `schema_mode` and `schema_fields` for nodes when present.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_repositories.py -k NodeRepository`
- New tests required: yes (schema config preservation)

## Notes / Links

- Related issues/PRs: N/A

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P3 verification wave 1

**Current Code Analysis:**

The bug is confirmed to still exist in the current codebase. I examined all relevant files:

1. **Database Schema** (`src/elspeth/core/landscape/schema.py:63-65`):
   - `nodes` table has `schema_mode` (String) and `schema_fields_json` (Text) columns
   - These were added in commit 4b21b51 on 2026-01-17 as part of WP-11.99

2. **Node Contract** (`src/elspeth/contracts/audit.py:68-70`):
   - `Node` dataclass includes both fields:
     - `schema_mode: str | None = None`
     - `schema_fields: list[dict[str, object]] | None = None`

3. **NodeRepository.load()** (`src/elspeth/core/landscape/repositories.py:74-86`):
   - Currently constructs Node with 11 fields
   - **Missing**: `schema_mode` and `schema_fields`
   - These fields will default to `None` when using the repository

4. **Recorder._row_to_node()** (`src/elspeth/core/landscape/recorder.py:620-648`):
   - DOES include these fields (lines 646-647)
   - Correctly parses `schema_fields_json` from JSON (lines 630-632)
   - This creates an inconsistency: recorder returns complete nodes, repository returns incomplete nodes

5. **Test Coverage** (`tests/core/landscape/test_repositories.py`):
   - NodeRepository tests exist (lines 185-292)
   - Tests verify enum conversion and crash-on-invalid behavior
   - **No tests for schema_mode or schema_fields preservation**
   - Mock NodeRow dataclass (lines 192-203) does not include these fields

**Git History:**

- `NodeRepository` was created on 2026-01-16 (commit 687ed8b)
- `schema_mode` and `schema_fields` were added on 2026-01-17 (commit 4b21b51)
- Repository was never updated to include the new fields
- No commits since bug report date (2026-01-19) have addressed this issue
- Most recent repository commit was c786410 (RC1), which still has the bug

**Root Cause Confirmed:**

Yes. The root cause is exactly as stated in the bug report: NodeRepository was not updated when WP-11.99 schema configuration columns were added. The repository predates the schema fields by one day and was never synchronized.

**Impact Assessment:**

Currently **LOW IMPACT** because:
- `NodeRepository` is not used anywhere in the production codebase (grep found no usages outside tests)
- The recorder uses its own `_row_to_node()` method which correctly handles these fields
- Data is stored correctly in the database

However, this is **TECHNICAL DEBT** because:
- If future code uses NodeRepository for node queries, schema audit metadata will be silently lost
- The inconsistency between repository and recorder deserialization violates the principle of single responsibility
- Missing test coverage means this could be broken further without detection

**Recommendation:**

**Keep open** as P3. This should be fixed before any code uses NodeRepository for node retrieval, especially if building UI/reporting features that display schema audit information. The fix is straightforward:

1. Add `schema_mode=row.schema_mode` to NodeRepository.load()
2. Add JSON parsing logic for `schema_fields_json` (mirroring recorder's implementation)
3. Add test case that verifies schema fields are preserved
4. Update mock NodeRow in tests to include these fields

This aligns with the audit trail integrity requirements: repository layer must return complete audit records, not partial ones.

---

## CLOSURE: 2026-01-28

**Status:** FIXED

**Fixed By:** Unknown (discovered during bug audit)

**Resolution:**

The fix was implemented in `src/elspeth/core/landscape/repositories.py`. The `NodeRepository.load()` method now correctly includes `schema_mode` and `schema_fields`.

**Code (lines 82-109):**
```python
def load(self, row: Any) -> Node:
    """Load Node from database row.

    Converts node_type and determinism strings to enums.
    Parses schema_fields_json back to list.
    """
    import json

    # Parse schema_fields_json back to list
    schema_fields: list[dict[str, object]] | None = None
    if row.schema_fields_json is not None:
        schema_fields = json.loads(row.schema_fields_json)

    return Node(
        node_id=row.node_id,
        run_id=row.run_id,
        # ... other fields ...
        schema_mode=row.schema_mode,        # ← Now included
        schema_fields=schema_fields,         # ← Now included (parsed from JSON)
    )
```

**Verification:**

- `schema_mode=row.schema_mode` at line 107
- `schema_fields=schema_fields` at line 108 (parsed from JSON at lines 91-93)

The repository now returns complete Node objects matching the recorder's behavior.
