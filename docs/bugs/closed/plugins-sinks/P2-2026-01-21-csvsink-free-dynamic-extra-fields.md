# Bug Report: CSVSink rejects extra fields in free/dynamic schemas

## STATUS: FIXED (2026-02-02)

**Resolution:** Implemented Option B - CSVSink now rejects free/dynamic schemas at initialization with a clear error message directing users to JSONSink.

**Fix location:** `src/elspeth/plugins/sinks/csv_sink.py:155-161`

**Test coverage:**
- `test_rejects_free_mode_schema`
- `test_rejects_dynamic_schema`

---

## Summary

- For `schema.mode=free` or dynamic schemas, extra fields are allowed by the schema contract, but CSVSink uses a fixed header and `csv.DictWriter` defaults to `extrasaction="raise"`, causing valid rows to crash when they include extra fields.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6f088f467276582fa8016f91b4d3bb26c7 (fix/rc1-bug-burndown-session-2)
- OS: Linux nyx.foundryside.dev 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Deep dive into src/elspeth/plugins/sinks for bugs.
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): Codex CLI, workspace-write sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: Manual code inspection only

## Steps To Reproduce

1. Configure CSVSink with schema `{mode: "free", fields: ["id: int"]}` and `validate_input: true`.
2. Write a row `{"id": 1, "extra": "x"}`.
3. Observe `ValueError: dict contains fields not in fieldnames` even though the schema allows extras.

## Expected Behavior

- Free or dynamic schemas should either include extra fields in the CSV output or ignore them without crashing.

## Actual Behavior

- CSVSink raises when rows contain extra keys that are permitted by the schema.

## Evidence

- `SchemaConfig.allows_extra_fields` returns true for dynamic/free.
- `src/elspeth/plugins/sinks/csv_sink.py` initializes `csv.DictWriter` without `extrasaction="ignore"` and uses fixed fieldnames.

## Impact

- User-facing impact: Valid rows crash the sink when additional fields appear.
- Data integrity / security impact: Output may be incomplete or pipeline fails despite valid schema.
- Performance or cost impact: Unplanned failures mid-run.

## Root Cause Hypothesis

- CSVSink does not handle `allows_extra_fields` and always enforces a fixed header.

## Proposed Fix

- Code changes (modules/files):
  - If schema allows extra fields, set `extrasaction="ignore"` or dynamically expand headers (define policy).
- Config or schema changes: None.
- Tests to add/update:
  - Add tests for free/dynamic schemas with extra fields.
- Risks or migration steps: If choosing to ignore extras, document that behavior.

## Architectural Deviations

- Spec or doc reference: `src/elspeth/contracts/schema.py` (`allows_extra_fields`).
- Observed divergence: Extras allowed by schema but rejected by sink.
- Reason (if known): Fixed-header implementation.
- Alignment plan or decision needed: Decide how CSV should represent extra fields.

## Acceptance Criteria

- Rows with extra fields under free/dynamic schemas do not raise during write.

## Tests

- Suggested tests to run: `pytest tests/plugins/sinks/test_csv_sink.py -k free`
- New tests required: Free/dynamic extra-field handling.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

---

## VERIFICATION: 2026-02-01

**Status:** STILL VALID

- `csv.DictWriter` still instantiated without `extrasaction`, so extra fields still raise. (`src/elspeth/plugins/sinks/csv_sink.py:270-288`)
- Explicit/free schema headers still come from schema fields only, so extras are not included. (`src/elspeth/plugins/sinks/csv_sink.py:299-304`)

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 6b

**Current Code Analysis:**

Examined `/home/john/elspeth-rapid/src/elspeth/plugins/sinks/csv_sink.py` (current HEAD on branch `fix/rc1-bug-burndown-session-4`). The bug is still present:

1. **Schema Configuration**: `SchemaConfig.allows_extra_fields` returns `True` for both `mode="free"` and dynamic schemas (line 269 in `src/elspeth/contracts/schema.py`).

2. **DictWriter Initialization** (lines 191-195 and 205-209 in `csv_sink.py`):
   - Both append mode (line 191) and write mode (line 205) create `csv.DictWriter` without specifying `extrasaction` parameter
   - Python's `csv.DictWriter` defaults to `extrasaction='raise'` (per Python stdlib documentation)
   - This means any dict keys not in `fieldnames` will raise `ValueError: dict contains fields not in fieldnames`

3. **Fieldname Determination** (lines 212-226):
   - For explicit schemas (including `mode="free"`), fieldnames come from `self._schema_config.fields` (line 222)
   - For dynamic schemas, fieldnames are inferred from first row keys (line 225)
   - **Critical issue**: Free mode schemas define only SOME fields (e.g., `["id: int"]`), but allow extras per schema contract. The CSV headers only include defined fields, so extra fields cause write failures.

4. **Write Logic** (line 145): `writer.writerow(row)` passes the full row dict to DictWriter, which raises if any keys aren't in `fieldnames`.

**Example Failure Case:**
```python
# Schema allows extras via mode="free"
config = {"mode": "free", "fields": ["id: int"]}
# Row has extra field (allowed by schema contract)
row = {"id": 1, "extra_field": "data"}
# CSV headers: ["id"]
# DictWriter raises: ValueError: dict contains fields not in fieldnames: 'extra_field'
```

**Reproduction Confirmed:**

Created and ran minimal test case:
```python
sink = CSVSink({
    "path": "test.csv",
    "schema": {"mode": "free", "fields": ["id: int"]}
})
ctx = PluginContext(run_id="test", config={})
sink.write([{"id": 1, "extra": "x"}], ctx)  # Raises ValueError
```

Output: `BUG CONFIRMED: dict contains fields not in fieldnames: 'extra'`

**Git History:**

Checked commits since 2026-01-21:
- `3eaebf6` (Jan 20): Added append mode to CSVSink - **did NOT address extrasaction**
- `7ee7c51`: Added self-validation to all builtin plugins - **did NOT address extrasaction**
- `cc0d364` (Jan 21): Added test `test_explicit_schema_creates_all_headers_including_optional` - **tests optional fields DEFINED in schema, not extra fields NOT in schema**
- No commits specifically addressing free/dynamic schema extra field handling

**Test Coverage Gap:**

The existing test `test_explicit_schema_creates_all_headers_including_optional` (line 254) uses `mode="free"` but only tests **optional fields that ARE defined in the schema** (e.g., `"score: float?"`), not truly extra fields that are **NOT** in the schema definition at all. That test passes because "score" is in the schema's field list, so it's in the CSV headers.

No test exists for the reported bug scenario: extra fields that are NOT in the schema definition at all.

**Root Cause Confirmed:**

Yes. The bug is present in the current code. The architectural mismatch is:

1. `SchemaConfig.allows_extra_fields` returns `True` for free/dynamic modes (contract says "extra fields allowed")
2. `CSVSink._get_fieldnames_from_schema_or_row()` creates headers from schema fields only for free mode (line 222)
3. `csv.DictWriter` defaults to `extrasaction='raise'`, rejecting any row keys not in headers (implementation says "extra fields forbidden")

This is a **contract violation** between the schema system and the CSV writer.

**Recommendation:**

**Keep open** - This is a valid bug requiring a design decision:

**Option A**: Set `extrasaction='ignore'` when `allows_extra_fields` is True (silently discard extras)
**Option B**: Fail during plugin initialization if `allows_extra_fields` is True (reject free/dynamic schemas)
**Option C**: Dynamically expand CSV headers on first encounter of new fields (accumulate all seen fields)

Each option has tradeoffs:
- **A** silently loses data (violates audit principles - no record of discarded fields)
- **B** restricts CSVSink to strict-only schemas (breaks free/dynamic use cases)
- **C** creates inconsistent CSV structure across batches (headers grow as new fields appear)

**Recommended approach**: **Option A with audit logging**. When `allows_extra_fields` is True, set `extrasaction='ignore'` and log a warning about which fields were discarded. This maintains CSV structural integrity (fixed columns) while providing visibility into data loss.

Alternative: **Option B** with clear error message. CSVSink could reject free/dynamic schemas in `_validate_self_consistency()` because CSV format inherently requires fixed column structure. Users needing flexible schemas should use JSONSink or other schema-less sinks.

This mirrors the recommendation for DatabaseSink (similar bug P2-2026-01-21-databasesink-free-dynamic-extra-fields.md), as both CSV and relational databases require fixed column structure.
