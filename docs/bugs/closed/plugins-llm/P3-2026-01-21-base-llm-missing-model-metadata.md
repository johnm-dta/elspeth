# Bug Report: Base LLM transform omits model metadata in output

## Summary

- BaseLLMTransform does not include the resolved response model in its output fields, unlike Azure/OpenRouter transforms, reducing audit traceability when a subclass relies on BaseLLMTransform behavior.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 (fix/rc1-bug-burndown-session-2)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: any BaseLLMTransform subclass

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/plugins/llm for bugs
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Implement a simple subclass of BaseLLMTransform.
2. Run a row and inspect output fields.

## Expected Behavior

- Output should include the actual model used (e.g., `<response_field>_model`) for audit parity with other LLM transforms.

## Actual Behavior

- No model field is added in BaseLLMTransform output.

## Evidence

- Output fields are assembled without model metadata in `src/elspeth/plugins/llm/base.py:259`.

## Impact

- User-facing impact: missing model attribution in outputs.
- Data integrity / security impact: audit trail lacks model detail for BaseLLMTransform subclasses.
- Performance or cost impact: none.

## Root Cause Hypothesis

- Base implementation predates model metadata convention used in Azure/OpenRouter.

## Proposed Fix

- Code changes (modules/files): add `output[f"{response_field}_model"] = response.model` in BaseLLMTransform.
- Config or schema changes: N/A
- Tests to add/update:
  - Add a unit test for BaseLLMTransform subclasses to assert model field presence.
- Risks or migration steps:
  - Ensure output schema allows the new field when strict schemas are used.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): auditability requires full external call attribution.
- Observed divergence: model not recorded in output row.
- Reason (if known): missing field in base implementation.
- Alignment plan or decision needed: align base output fields with Azure/OpenRouter patterns.

## Acceptance Criteria

- BaseLLMTransform outputs include `<response_field>_model`.

## Tests

- Suggested tests to run: N/A (no direct tests currently)
- New tests required: yes, base LLM output metadata test.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: CLAUDE.md auditability standard

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P3 verification wave 3

**Current Code Analysis:**

Examined `/home/john/elspeth-rapid/src/elspeth/plugins/llm/base.py` at lines 270-283. The `process()` method builds the output row with the following fields:

- `{response_field}` - the LLM response content
- `{response_field}_usage` - token usage statistics
- `{response_field}_template_hash` - template hash for audit
- `{response_field}_variables_hash` - variables hash
- `{response_field}_template_source` - template source path
- `{response_field}_lookup_hash` - lookup data hash
- `{response_field}_lookup_source` - lookup source path
- `{response_field}_system_prompt_source` - system prompt source path

**Missing:** `{response_field}_model` field

The `LLMResponse` object returned by `llm_client.chat_completion()` DOES contain a `model` field (verified in `/home/john/elspeth-rapid/src/elspeth/plugins/clients/llm.py:37`), but BaseLLMTransform does not include it in the output.

**Comparison with other LLM transforms:**

- **AzureLLMTransform** (`azure.py:289`, `azure.py:388`, `azure.py:526`): Includes `output[f"{self._response_field}_model"] = response.model`
- **AzureMultiQueryTransform** (`azure_multi_query.py:278`): Includes `output[f"{spec.output_prefix}_model"] = response.model`
- **OpenRouterLLMTransform** (`openrouter.py:308`, `openrouter.py:527`, `openrouter.py:668`): Includes `output[f"{self._response_field}_model"] = data.get("model", self._model)`

All production LLM transforms include model metadata, but BaseLLMTransform does not.

**Git History:**

Checked git history for relevant commits:

- `305ce6b` (2026-01-20): "fix(llm): add missing audit fields to transform outputs" - Added `template_source`, `lookup_hash`, and `lookup_source` fields to BaseLLMTransform, but NOT the model field
- `9b64ff3`: "feat(llm): add system_prompt_file support" - Added `system_prompt_source` field
- No commits found that add model metadata to BaseLLMTransform

**Root Cause Confirmed:**

Yes, the bug is still present. BaseLLMTransform omits `response.model` from its output row, creating an inconsistency with all other LLM transforms and violating the auditability principle that "every decision must be traceable to source data, configuration, and code version."

The model field is particularly important because:
1. The actual model used may differ from the requested model (e.g., fallback models, model aliasing)
2. CLAUDE.md requires "Full request AND response recorded" for external calls
3. All production LLM transforms already include this field

**Impact Assessment:**

Currently LOW because:
- No actual subclasses of BaseLLMTransform exist in the codebase (only the docstring example)
- All production LLM transforms (Azure, OpenRouter) include model metadata

However, the bug would become MEDIUM if:
- A developer creates a custom BaseLLMTransform subclass following the example
- The subclass doesn't override the output building logic
- The missing model field would silently reduce audit traceability

**Recommendation:**

Keep open. This is a valid auditability gap that should be fixed to:
1. Maintain consistency across all LLM transform implementations
2. Ensure the base class example in the docstring demonstrates proper audit trail patterns
3. Prevent future subclasses from inheriting the audit gap
4. Align with CLAUDE.md requirement: "Full request AND response recorded"

**Suggested fix:** Add one line after line 273 in `base.py`:
```python
output[f"{self._response_field}_model"] = response.model
```

## Resolution

**Fixed in:** 2026-01-29
**Fix:** Added `output[f"{self._response_field}_model"] = response.model` to BaseLLMTransform.process() at line 313 in base.py, aligning with Azure/OpenRouter transforms.

**Changes:**
- `src/elspeth/plugins/llm/base.py`: Added model field to output row
- `tests/plugins/llm/test_base.py`: Updated tests to verify model field presence
