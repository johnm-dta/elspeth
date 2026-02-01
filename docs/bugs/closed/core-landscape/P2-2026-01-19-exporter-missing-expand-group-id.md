# Bug Report: LandscapeExporter token records omit `expand_group_id` (deaggregation lineage lost in export)

## Summary

- Landscape supports 1→N expansion ("deaggregation") via `tokens.expand_group_id`.
- `LandscapeExporter` emits token records with `fork_group_id` and `join_group_id` but omits `expand_group_id`, so exported audit trails cannot reconstruct expansion groupings.

## Severity

- Severity: major
- Priority: P2

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
- Notable tool calls or steps: code inspection + minimal repro

## Steps To Reproduce

1. Run the following Python snippet:
   - `python - <<'PY'`
   - `from elspeth.core.landscape.database import LandscapeDB`
   - `from elspeth.core.landscape.recorder import LandscapeRecorder`
   - `from elspeth.core.landscape.exporter import LandscapeExporter`
   - `from elspeth.contracts.schema import SchemaConfig`
   - `schema = SchemaConfig.from_dict({"fields": "dynamic"})`
   - `db = LandscapeDB.in_memory()`
   - `rec = LandscapeRecorder(db)`
   - `run = rec.begin_run(config={}, canonical_version="v1")`
   - `node = rec.register_node(run_id=run.run_id, plugin_name="t", node_type="transform", plugin_version="1", config={}, schema_config=schema)`
   - `row = rec.create_row(run_id=run.run_id, source_node_id=node.node_id, row_index=0, data={})`
   - `parent = rec.create_token(row_id=row.row_id)`
   - `children = rec.expand_token(parent_token_id=parent.token_id, row_id=row.row_id, count=2, step_in_pipeline=1)`
   - `exporter = LandscapeExporter(db)`
   - `tokens = [r for r in exporter.export_run(run.run_id) if r.get(\"record_type\") == \"token\"]`
   - `print(tokens[0].keys())`
   - `PY`
2. Observe the token record keys include `fork_group_id`/`join_group_id` but not `expand_group_id`.

## Expected Behavior

- Exported token records include `expand_group_id` so that deaggregation can be reconstructed from export alone.

## Actual Behavior

- `expand_group_id` is omitted from token records.

## Evidence

- Schema defines `expand_group_id` on tokens:
  - `src/elspeth/core/landscape/schema.py` `tokens_table` includes `expand_group_id`
- Recorder reads/writes `expand_group_id` and Token contract includes it:
  - `src/elspeth/core/landscape/recorder.py:1716-1748` (`expand_group_id=r.expand_group_id`)
  - `src/elspeth/contracts/audit.py` `Token.expand_group_id`
- Exporter omits it:
  - `src/elspeth/core/landscape/exporter.py:210-221`

## Impact

- User-facing impact: exported audit trail cannot answer "which expanded children came from the same deaggregation step?" without access to the DB.
- Data integrity / security impact: moderate. This is missing audit information rather than corruption.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- Exporter token record schema was not updated when `expand_group_id` was added to tokens.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/core/landscape/exporter.py`:
    - Include `"expand_group_id": token.expand_group_id` in token records.
- Config or schema changes: none.
- Tests to add/update:
  - Add an export test that creates expanded tokens and asserts the exported token records include `expand_group_id`.
- Risks or migration steps:
  - None.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (audit traceability expectations)
- Observed divergence: export is incomplete for deaggregation lineage.
- Reason (if known): missing field in exporter mapping.
- Alignment plan or decision needed: none.

## Acceptance Criteria

- Exported token records always include `expand_group_id` (NULL when not applicable).

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_exporter.py`
- New tests required: yes (expand_group_id field coverage)

## Notes / Links

- Related issues/PRs: `docs/bugs/open/2026-01-19-export-fails-old-landscape-schema-expand-group-id.md` (schema drift already impacts export)

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 6c

**Current Code Analysis:**

The bug is still present in the current codebase. Examination of `/home/john/elspeth-rapid/src/elspeth/core/landscape/exporter.py` lines 210-221 confirms that the token export logic only includes:

```python
yield {
    "record_type": "token",
    "run_id": run_id,
    "token_id": token.token_id,
    "row_id": token.row_id,
    "step_in_pipeline": token.step_in_pipeline,
    "branch_name": token.branch_name,
    "fork_group_id": token.fork_group_id,
    "join_group_id": token.join_group_id,
}
```

The `expand_group_id` field is missing from this export record, despite being:

1. **Present in the database schema:** `src/elspeth/core/landscape/schema.py` line 107 defines `Column("expand_group_id", String(32), nullable=True, index=True)`
2. **Present in the Token contract:** `src/elspeth/contracts/audit.py` line 111 includes `expand_group_id: str | None = None`
3. **Returned by the recorder:** `src/elspeth/core/landscape/recorder.py` line 1812 in `get_tokens()` includes `expand_group_id=r.expand_group_id`

**Git History:**

- Commit `90a0677` (2026-01-19) added `expand_token()` functionality and the `expand_group_id` field to the schema, models, and contracts
- Commit `0f21ecb` (2026-01-23) modified the exporter to add `NodeStatePending` handling but did not add `expand_group_id`
- No commits have addressed this omission in the exporter

The exporter received the token expansion infrastructure but was not updated to include the new field in its output schema.

**Root Cause Confirmed:**

Yes. The root cause is exactly as stated in the original bug report: when `expand_group_id` was added to the Landscape schema and Token dataclass in commit `90a0677`, the `LandscapeExporter._iter_records()` method was not updated to include this field in the token record dictionary.

This is a simple oversight in the token export mapping at lines 212-221 of `exporter.py`. The Token object returned by `recorder.get_tokens()` has the field, but the exporter doesn't copy it into the exported record dictionary.

**Recommendation:**

**Keep open.** This is a valid P2 bug that needs fixing. The fix is straightforward:

1. Add `"expand_group_id": token.expand_group_id` to the token record dictionary in `exporter.py` lines 210-221
2. Add test coverage in `tests/core/landscape/test_exporter.py` to verify expanded tokens export with `expand_group_id`

This is a data integrity issue for audit trail exports - deaggregation lineage cannot be reconstructed from exports without this field, violating the auditability standard that "the audit trail must withstand formal inquiry."

---

## CLOSURE: 2026-01-29

**Status:** FIXED

**Fixed By:** Commit `c5fb53e` - "fix(core): address three audit-related bugs"

**Resolution:**

The fix added `expand_group_id` to the token export record in `src/elspeth/core/landscape/exporter.py:221`:

```python
yield {
    "record_type": "token",
    "run_id": run_id,
    "token_id": token.token_id,
    "row_id": token.row_id,
    "step_in_pipeline": token.step_in_pipeline,
    "branch_name": token.branch_name,
    "fork_group_id": token.fork_group_id,
    "join_group_id": token.join_group_id,
    "expand_group_id": token.expand_group_id,  # Added
}
```

**Test Coverage Added:**

New test in `tests/core/landscape/test_exporter.py` verifies that expanded tokens include `expand_group_id` in their export records.

**Verified By:** Claude Opus 4.5 (2026-01-29)
