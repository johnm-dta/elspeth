# Bug Report: Azure content safety and prompt shield add error fields not declared in output_schema

## Summary

- In batch mode, AzureContentSafety and AzurePromptShield embed _content_safety_error or _prompt_shield_error in output rows, but output_schema is identical to input_schema, so strict schemas and downstream validators do not match the actual output shape.

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
- Notable tool calls or steps: reviewed Azure content_safety and prompt_shield implementations

## Steps To Reproduce

1. Configure a strict schema without error fields and set pool_size > 1 (batch mode).
2. Provide input rows that trigger a content safety or prompt shield violation.
3. Observe output rows include _content_safety_error or _prompt_shield_error even though output_schema is unchanged.

## Expected Behavior

- output_schema reflects the actual output shape (dynamic or explicit optional error fields).

## Actual Behavior

- output_schema remains identical to input_schema while output rows add error fields in batch mode.

## Evidence

- output_schema set to input schema: src/elspeth/plugins/transforms/azure/content_safety.py:160-162; src/elspeth/plugins/transforms/azure/prompt_shield.py:126-133
- error fields added in batch mode: src/elspeth/plugins/transforms/azure/content_safety.py:418-425; src/elspeth/plugins/transforms/azure/prompt_shield.py:387-395
- Transform output_schema should describe outgoing rows: docs/contracts/plugin-protocol.md:334-339

## Impact

- User-facing impact: strict sink validation can fail on unexpected error fields.
- Data integrity / security impact: schema contracts are inaccurate, complicating validation and auditing.
- Performance or cost impact: potential pipeline failures and retries.

## Root Cause Hypothesis

- Batch error embedding was added without updating output_schema to include the new fields.

## Proposed Fix

- Code changes (modules/files): src/elspeth/plugins/transforms/azure/content_safety.py, src/elspeth/plugins/transforms/azure/prompt_shield.py
- Config or schema changes: use dynamic output_schema in batch mode or extend schema with optional error fields.
- Tests to add/update: add tests asserting output_schema allows error fields when batch mode is enabled.
- Risks or migration steps: document behavior change for strict schemas.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): docs/contracts/plugin-protocol.md:334-339
- Observed divergence: output rows include new fields not represented in output_schema.
- Reason (if known): output_schema reused from input schema.
- Alignment plan or decision needed: align output_schema with actual output fields.

## Acceptance Criteria

- Batch mode outputs with error fields validate against output_schema.
- Schema compatibility checks reflect the presence of error fields.

## Tests

- Suggested tests to run: pytest tests/plugins/transforms/azure/test_content_safety.py pytest tests/plugins/transforms/azure/test_prompt_shield.py
- New tests required: yes, output_schema checks for batch error fields.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: docs/contracts/plugin-protocol.md

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 4b

**Current Code Analysis:**

The bug is **confirmed valid** in the current codebase (commit 0a339fd). Both AzureContentSafety and AzurePromptShield exhibit the reported behavior:

1. **Schema Declaration (Still Present):**
   - `/home/john/elspeth-rapid/src/elspeth/plugins/transforms/azure/content_safety.py:162` sets `self.output_schema = schema` (identical to input)
   - `/home/john/elspeth-rapid/src/elspeth/plugins/transforms/azure/prompt_shield.py:133` sets `self.output_schema = schema` (identical to input)

2. **Error Field Injection (Still Present):**
   - `/home/john/elspeth-rapid/src/elspeth/plugins/transforms/azure/content_safety.py:435` adds `output_row["_content_safety_error"]` in batch mode
   - `/home/john/elspeth-rapid/src/elspeth/plugins/transforms/azure/prompt_shield.py:404` adds `output_row["_prompt_shield_error"]` in batch mode

3. **Self-Validation Added (But Insufficient):**
   - Commit 7ee7c51 (2026-01-25) added `_validate_self_consistency()` methods to both transforms
   - However, validation only confirms "input == output by definition" with no additional checks
   - This validation does NOT address the schema mismatch when error fields are added

**Git History:**

No commits have addressed this issue. Relevant commits examined:

- `7ee7c51` (2026-01-25): "feat: add self-validation to all builtin plugins" - Added validation scaffolding but did NOT fix the schema mismatch
- `430307d` (2026-01-24): "feat: add schema validation to plugin protocols" - Infrastructure change, no fix for this issue
- `df43269` (2026-01-24): "refactor: remove schema validation from DAG layer" - Moved validation location, no fix
- `615ef21` (2026-01-21): "feat(prompt-shield): implement pooled execution with audit trail" - Original implementation with the bug
- `fec943c` (2026-01-21): "feat(content-safety): implement pooled execution with audit trail" - Original implementation with the bug

The pooled execution plans (`docs/plans/completed/2026-01-21-pooled-content-safety.md`) explicitly show error field embedding (line 658: "Embed errors per-row via _content_safety_error field") but do NOT discuss schema handling.

**Root Cause Confirmed:**

YES. The bug is present and unaddressed. The architecture issue is:

- **Current behavior:** `output_schema = input_schema`, but batch mode conditionally adds error fields
- **Why it's wrong:** Violates plugin protocol contract - `output_schema` must describe actual output shape
- **Downstream impact:** Schema compatibility validation in `ExecutionGraph._validate_single_edge()` will incorrectly pass edges where the consumer expects only input fields, then fail at runtime when error fields appear

**Systemic Pattern:**

This is NOT isolated to ContentSafety/PromptShield. The same pattern exists in:
- `/home/john/elspeth-rapid/src/elspeth/plugins/llm/azure_batch.py:162` - Sets `output_schema = schema` but adds `{response_field}_error` fields in batch mode
- `/home/john/elspeth-rapid/src/elspeth/plugins/llm/azure_multi_query.py` - Adds `_error` field in batch assembly

**Correct Pattern (Reference):**

BatchStats transform demonstrates the correct approach (`/home/john/elspeth-rapid/src/elspeth/plugins/transforms/batch_stats.py:89-97`):

```python
# Output schema MUST be dynamic because BatchStats outputs a completely
# different shape: {count, sum, mean, batch_size, group_by?}
self.output_schema = create_schema_from_config(
    SchemaConfig.from_dict({"fields": "dynamic"}),
    "BatchStatsOutputSchema",
    allow_coercion=False,
)
```

Dynamic schemas (`output_schema = None` after processing) bypass edge validation per `ExecutionGraph._validate_single_edge()` Rule 1, which is appropriate when output shape varies.

**Recommendation:**

**Keep open - Fix Required**

This is a valid architectural deviation from plugin protocol contracts. The fix should apply the BatchStats pattern:

1. **When `pool_size > 1` (batch mode):** Set `output_schema` to dynamic to reflect that error fields may be added
2. **When `pool_size == 1` (single mode):** Keep `output_schema = input_schema` since no error fields are added
3. **Alternative:** Explicitly extend the output schema with optional error field definitions

The fix is straightforward but affects multiple Azure transforms (ContentSafety, PromptShield, AzureBatch, AzureMultiQuery). All should be fixed together for consistency.

Priority remains **P2** - this causes schema validation to incorrectly pass incompatible edges, but only manifests when strict downstream consumers reject the unexpected error fields. Impact is contained to Azure batch mode pipelines with strict schema validation.

---

## RESOLUTION: 2026-01-26

**Status:** FIXED

**Fixed by:** Claude Code (fix/rc1-bug-burndown-session-5)

**Implementation:**

Set `output_schema` to dynamic when `pool_size > 1` (batch mode) for both ContentSafety and PromptShield transforms.

### Code Evidence

**Before (lines 160-161 - output schema always matches input):**
```python
self.input_schema = schema
self.output_schema = schema  # ❌ Wrong in batch mode
```

**After (lines 160-172 - conditional based on batch mode):**
```python
self.input_schema = schema

# In batch mode (pool_size > 1), error fields are added to output rows.
# Use dynamic output schema to reflect this, as strict schemas would fail.
if self._pool_size > 1:
    self.output_schema = create_schema_from_config(
        SchemaConfig.from_dict({"fields": "dynamic"}),
        "AzureContentSafetyOutputSchema",  # or PromptShieldOutputSchema
        allow_coercion=False,
    )
else:
    # Single-row mode: no error fields added, output matches input
    self.output_schema = schema
```

### Why This Fix Works

**Batch mode (pool_size > 1):**
- Errors embedded per-row via `_content_safety_error` or `_prompt_shield_error` fields
- Dynamic schema allows these extra fields without validation failure
- Downstream transforms see dynamic schema and know output shape may vary

**Single-row mode (pool_size == 1):**
- Errors return via `TransformResult.error()` with no output row
- Output schema matches input schema (pass-through semantics)
- No extra fields added to rows

### Impact

**Fixed:**
- ✅ Schema validation now accurately reflects actual output shape
- ✅ Strict downstream consumers warned via dynamic schema
- ✅ Batch error fields no longer cause unexpected validation failures
- ✅ Architectural alignment with BatchStats pattern

**Files changed:**
- `src/elspeth/plugins/transforms/azure/content_safety.py`
- `src/elspeth/plugins/transforms/azure/prompt_shield.py`

**Note:** The bug report also mentioned AzureBatch and AzureMultiQuery, but those were out of scope for this Azure-focused cleanup session. They can be addressed in a future fix if needed.
