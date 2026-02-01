# Bug Report: AuditedLLMClient treats missing usage fields as errors

## Summary

- `AuditedLLMClient` assumes `response.usage.prompt_tokens` and `response.usage.completion_tokens` are always present. Providers or modes that omit usage (or return `None`) trigger an exception, causing a successful LLM call to be recorded as an error and the transform to fail.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `fix/rc1-bug-burndown-session-2` @ `ae2c0e6f088f467276582fa8016f91b4d3bb26c7`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/plugins/clients` and file bugs
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of LLM usage handling

## Steps To Reproduce

1. Use an LLM provider/mode that omits `usage` data (or returns it as `None`).
2. Call `AuditedLLMClient.chat_completion`.
3. Observe an exception raised while accessing `response.usage.*`, and the call is recorded as ERROR.

## Expected Behavior

- Missing usage data should be handled gracefully (e.g., `usage={}` or zeros) while still recording the call as SUCCESS.

## Actual Behavior

- The client throws and records the call as ERROR even when the provider returned a valid response.

## Evidence

- Unchecked access to usage fields: `src/elspeth/plugins/clients/llm.py:175-178`

## Impact

- User-facing impact: pipelines fail despite successful provider responses.
- Data integrity / security impact: audit trail misclassifies successful calls as failures.
- Performance or cost impact: retries or failures increase costs.

## Root Cause Hypothesis

- The client assumes usage is always present and never `None`.

## Proposed Fix

- Code changes (modules/files):
  - Guard against missing usage (`if response.usage is None:`) and record `usage={}` or zeros.
  - Optionally add a `usage_available` flag in `response_data`.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test fixture simulating responses with missing usage fields.
- Risks or migration steps:
  - Ensure downstream code can handle empty usage dictionaries.

## Architectural Deviations

- Spec or doc reference: N/A
- Observed divergence: external provider responses are not handled defensively at the boundary.
- Reason (if known): assumed OpenAI-style usage always present.
- Alignment plan or decision needed: define usage optionality in LLM call contract.

## Acceptance Criteria

- LLM responses lacking usage data are still recorded as SUCCESS with an empty or partial usage dict.

## Tests

- Suggested tests to run: `pytest tests/plugins/clients/ -k llm_usage`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

## Verification (2026-01-25)

**Status: STILL VALID**

### Verification Steps

1. **Code inspection** of `/home/john/elspeth-rapid/src/elspeth/plugins/clients/llm.py:176-179`:
   ```python
   usage = {
       "prompt_tokens": response.usage.prompt_tokens,
       "completion_tokens": response.usage.completion_tokens,
   }
   ```
   Direct attribute access on `response.usage` with no None check.

2. **Git history analysis**:
   - File introduced in commit `242ed66` (feat: add audited client infrastructure)
   - Only subsequent change was `d7c5f10` (fix: avoid sending max_tokens=null)
   - **No fixes for usage handling have been committed**

3. **Test coverage gap**:
   - Reviewed `/home/john/elspeth-rapid/tests/plugins/clients/test_audited_llm_client.py`
   - Found test `test_total_tokens_with_missing_fields()` that handles empty usage dict in `LLMResponse`
   - **No test exists for `response.usage = None` from provider**

4. **Reproduction test**:
   Created mock test with `response.usage = None`:
   ```
   Status: CallStatus.ERROR
   Error: {'type': 'AttributeError', 'message': "'NoneType' object has no attribute 'prompt_tokens'", 'retryable': False}
   Response data: NO RESPONSE DATA
   ```
   **Result: Successful LLM response with missing usage is recorded as ERROR**

### Current Behavior

When `response.usage` is `None`:
1. AttributeError raised on line 177: `response.usage.prompt_tokens`
2. Exception caught by outer try/except block (lines 203-225)
3. Call recorded to audit trail with `status=CallStatus.ERROR`
4. `LLMClientError` raised to caller, failing the transform

### Impact Confirmation

- **Data integrity impact**: Audit trail contains `ERROR` records for successful LLM calls
- **Pipeline impact**: Transforms fail despite valid LLM responses with content
- **Real-world trigger**: Any provider/mode that omits usage (streaming responses, certain Azure configurations, usage tracking disabled)

### Recommendation

Bug remains valid and unaddressed. Priority P2 is appropriate given:
- Violates external boundary defensive handling principle (CLAUDE.md Three-Tier Trust Model)
- Misclassifies audit records (destroys audit integrity)
- Blocks usage of legitimate provider configurations
- Simple fix: guard `response.usage` access with None check

## Resolution

**Fixed in:** 2026-01-29
**Fix:** Added None check before accessing `response.usage` fields. When usage is None (provider omits data), an empty dict is used instead, allowing the call to be recorded as SUCCESS.

**Changes:**
- `src/elspeth/plugins/clients/llm.py`: Added guard at lines 292-299
- `tests/plugins/clients/test_audited_llm_client.py`: Added test for None usage scenario
