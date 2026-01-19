# Bug Report: LandscapeExporter token records omit `expand_group_id` (deaggregation lineage lost in export)

## Summary

- Landscape supports 1→N expansion (“deaggregation”) via `tokens.expand_group_id`.
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

- User-facing impact: exported audit trail cannot answer “which expanded children came from the same deaggregation step?” without access to the DB.
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
