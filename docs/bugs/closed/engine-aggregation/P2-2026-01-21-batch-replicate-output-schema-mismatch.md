# Bug Report: BatchReplicate output_schema omits copy_index when include_copy_index=True

## Summary

- BatchReplicate adds a copy_index field to output rows by default, but output_schema is set to the input schema, so strict schemas and downstream validators do not reflect the actual output shape.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: not checked
- OS: not checked (workspace sandbox)
- Python version: not checked
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/plugins/transforms for bugs
- Model/version: GPT-5 Codex
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: reviewed BatchReplicate implementation

## Steps To Reproduce

1. Configure batch_replicate with a strict schema that does not include copy_index and leave include_copy_index at its default (True).
2. Run an aggregation using batch_replicate with output_mode: transform.
3. Observe output rows include copy_index, but output_schema remains the strict input schema.

## Expected Behavior

- output_schema reflects the actual output shape (dynamic or includes optional copy_index) so schema compatibility and downstream validation are accurate.

## Actual Behavior

- output_schema is identical to input_schema even when output rows include copy_index.

## Evidence

- Output schema set to input schema: src/elspeth/plugins/transforms/batch_replicate.py:92-99
- copy_index added to output rows: src/elspeth/plugins/transforms/batch_replicate.py:133-136

## Impact

- User-facing impact: strict downstream schemas or sinks can reject rows unexpectedly.
- Data integrity / security impact: schema contracts are inaccurate, undermining validation guarantees.
- Performance or cost impact: potential pipeline failures and retries.

## Root Cause Hypothesis

- BatchReplicate assumes output shape matches input, but include_copy_index introduces new fields that are not represented in output_schema.

## Proposed Fix

- Code changes (modules/files): src/elspeth/plugins/transforms/batch_replicate.py
- Config or schema changes: set output_schema to dynamic when include_copy_index is True or extend schema with optional copy_index.
- Tests to add/update: add transform tests for output_schema behavior with include_copy_index True.
- Risks or migration steps: if users rely on strict schema, document the added field or require include_copy_index False.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): docs/contracts/plugin-protocol.md:334-339 (output_schema describes outgoing rows)
- Observed divergence: output_schema does not include copy_index even though output rows add it.
- Reason (if known): output_schema reused from input schema.
- Alignment plan or decision needed: make output_schema dynamic or explicitly include copy_index.

## Acceptance Criteria

- When include_copy_index is True, output_schema allows copy_index (dynamic or explicit optional field).
- Schema validation no longer rejects output rows that include copy_index.

## Tests

- Suggested tests to run: pytest tests/plugins/transforms/test_batch_replicate.py
- New tests required: yes, output_schema expectations for include_copy_index True/False.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: docs/contracts/plugin-protocol.md

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 6b

**Current Code Analysis:**

Examined `/home/john/elspeth-rapid/src/elspeth/plugins/transforms/batch_replicate.py` at current HEAD:

**Bug is CONFIRMED present:**

1. **Lines 98-99:** Output schema is set identically to input schema
   ```python
   self.input_schema = schema
   self.output_schema = schema
   ```

2. **Lines 146-147:** `copy_index` field is conditionally added to output rows
   ```python
   if self._include_copy_index:
       output["copy_index"] = copy_idx
   ```

3. **Line 39-42:** Config allows `include_copy_index: bool` with default `True`
   ```python
   include_copy_index: bool = Field(
       default=True,
       description="Whether to add a 'copy_index' field (0-based) to each output row",
   )
   ```

**Contract Violation:**

Per plugin protocol docs, `output_schema` must accurately describe the shape of output rows. When `include_copy_index=True` (the default), BatchReplicate emits rows containing a `copy_index` field that is NOT declared in `output_schema`.

This creates a schema mismatch between declared contract (`output_schema`) and actual behavior (adds `copy_index` field).

**Impact:**

- **Downstream validation failures:** If a downstream transform or sink has strict schema validation expecting only the declared fields, it will reject rows containing the undeclared `copy_index` field
- **Schema compatibility checks:** Edge validation (added in recent schema validation refactor) may incorrectly validate the pipeline because it checks `output_schema`, not the actual output shape
- **Audit integrity:** The schema recorded in NodeInfo doesn't match the actual data flow

**Git History:**

```
7ee7c51 feat: add self-validation to all builtin plugins
c786410 ELSPETH - Release Candidate 1
a43f20a feat(plugins): add batch_replicate transform for deaggregation
```

No commits address this schema mismatch. The recent self-validation work (commit 7ee7c51) added `_validate_self_consistency()` to BatchReplicate but with a trivial implementation:

```python
def _validate_self_consistency(self) -> None:
    """Validate BatchReplicate schemas are self-consistent.

    BatchReplicate has no self-consistency constraints (input == output by definition).
    """
    self._validation_called = True  # Mark validation as complete
    # No additional validation needed - BatchReplicate has matching input/output schemas
```

The comment "input == output by definition" is incorrect when `include_copy_index=True`.

**Root Cause Confirmed:**

BatchReplicate was designed as a simple pass-through aggregation that replicates rows. When `include_copy_index` was added as a feature, the schema contract was not updated to reflect that output rows now have an additional field.

**Recommended Fix:**

Two valid approaches:

1. **Dynamic schema when copy_index added (simpler):**
   ```python
   if self._include_copy_index:
       from elspeth.contracts.schema import create_dynamic_schema
       self.output_schema = create_dynamic_schema("BatchReplicateOutputSchema")
   else:
       self.output_schema = schema
   ```

2. **Extend schema with optional copy_index field (more precise):**
   ```python
   if self._include_copy_index:
       # Create new schema class with copy_index field added as optional int
       fields = {**schema.model_fields}
       fields['copy_index'] = (int | None, None)
       self.output_schema = create_model('BatchReplicateWithIndex', **fields)
   else:
       self.output_schema = schema
   ```

**Testing Gap:**

No test file exists for batch_replicate (`tests/plugins/transforms/test_batch_replicate.py` does not exist). The output schema behavior has never been tested.

**Related Context:**

The recent schema validation refactor (docs/plans/2026-01-24-fix-schema-validation-properly.md) moved validation to construction time and added edge compatibility checks. This makes the schema mismatch MORE important because edge validation now checks `output_schema` compatibility before the pipeline runs.

**Recommendation:**

**Keep open - HIGH priority for fixing before RC-2**

This bug undermines the recent schema validation work. If BatchReplicate declares `output_schema` that doesn't match its actual output, edge validation will give false confidence that pipelines are valid when they may fail at runtime.

---

## CLOSURE: 2026-01-28

**Status:** FIXED

**Fixed By:** Claude Code

**Resolution:**

Changed `output_schema` to dynamic to accommodate the `copy_index` field that BatchReplicate adds to output rows. This follows the established pattern for shape-changing transforms (json_explode, batch_stats, field_mapper).

```python
# Input schema from config
self.input_schema = create_schema_from_config(
    self._schema_config,
    "BatchReplicateInputSchema",
    allow_coercion=False,
)

# Output schema MUST be dynamic because BatchReplicate adds copy_index field
# Per P1-2026-01-19-shape-changing-transforms-output-schema-mismatch
self.output_schema = create_schema_from_config(
    SchemaConfig.from_dict({"fields": "dynamic"}),
    "BatchReplicateOutputSchema",
    allow_coercion=False,
)
```

**Files Changed:**

- `src/elspeth/plugins/transforms/batch_replicate.py` - Added SchemaConfig import, split input/output schemas
- `tests/plugins/transforms/test_batch_replicate.py` - Added schema contract tests

**Tests Added:**

- `TestBatchReplicateSchemaContract::test_output_schema_is_dynamic_when_copy_index_enabled`
- `TestBatchReplicateSchemaContract::test_output_schema_accepts_copy_index_field`

**Verification:**

```bash
pytest tests/plugins/transforms/test_batch_replicate.py -v -k schema_contract
# Both tests pass - output schema accepts copy_index field
pytest tests/audit/test_plugin_schema_contracts.py -v -k batch_replicate
# Schema audit test passes
```
