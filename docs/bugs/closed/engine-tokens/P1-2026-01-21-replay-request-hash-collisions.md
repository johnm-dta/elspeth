# Bug Report: Replay/verify collapses duplicate calls with identical request_hash

## Summary

- `CallReplayer` and `CallVerifier` match recordings only by `request_hash` and `call_type`. When the same request is made multiple times in a run, `find_call_by_request_hash` returns the first matching call, so replay/verify always uses the first response and ignores later calls.

## Severity

- Severity: major
- Priority: P1

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
- Notable tool calls or steps: code inspection of replay/verify lookup logic

## Steps To Reproduce

1. In a run, make the same external request twice with identical `request_data` (e.g., retries or multiple identical LLM calls).
2. Enter replay/verify mode for that run.
3. Observe replay/verify always returns/compares against the first recorded call.

## Expected Behavior

- Replay/verify should disambiguate repeated identical requests (e.g., by call order or call index) and return the matching response for each invocation.

## Actual Behavior

- The first recorded call is always used; later calls are ignored.

## Evidence

- Replay lookup uses only `request_hash`: `src/elspeth/plugins/clients/replayer.py:156-177`
- Recorder returns first match when duplicates exist: `src/elspeth/core/landscape/recorder.py:2503-2519`
- Verify lookup uses only `request_hash`: `src/elspeth/plugins/clients/verifier.py:159-166`

## Impact

- User-facing impact: replay returns incorrect responses; verify reports drift against the wrong baseline.
- Data integrity / security impact: reproducibility claims are undermined for repeated calls.
- Performance or cost impact: potential false positives/negatives in verification.

## Root Cause Hypothesis

- Lookup keys do not include call order or per-state sequence, so duplicates collapse to the earliest call.

## Proposed Fix

- Code changes (modules/files):
  - Introduce a per-request sequence cursor in `CallReplayer`/`CallVerifier`, or match by `(state_id, call_index)` when available.
  - Optionally return and consume calls in chronological order per request hash.
- Config or schema changes: none.
- Tests to add/update:
  - Add tests where identical requests produce different responses and ensure replay/verify consumes them in order.
- Risks or migration steps:
  - Decide how to disambiguate repeated calls across different states or nodes.

## Architectural Deviations

- Spec or doc reference: N/A
- Observed divergence: replay/verify cannot represent repeated identical calls.
- Reason (if known): lookup by request hash was simpler.
- Alignment plan or decision needed: define required replay semantics for duplicate requests.

## Acceptance Criteria

- Replay/verify distinguishes multiple identical requests and returns/compares each call in order.

## Tests

- Suggested tests to run: `pytest tests/plugins/clients/ -k replay_duplicate`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

## Verification (2026-01-25)

**Status: STILL VALID**

### Code Inspection

The bug remains unfixed in the current codebase (commit 7540e57 on `fix/rc1-bug-burndown-session-4`):

1. **LandscapeRecorder.find_call_by_request_hash still returns first match** (`src/elspeth/core/landscape/recorder.py:2629-2687`):
   - Lines 2648-2650 explicitly document the limitation: "If multiple calls match (same request made twice), returns the first one chronologically (ordered by created_at)."
   - Query orders by `created_at` and uses `.limit(1)` (lines 2663-2664)
   - No mechanism to track or consume multiple calls with the same request_hash sequentially

2. **CallReplayer uses same lookup without sequence tracking** (`src/elspeth/plugins/clients/replayer.py:135-218`):
   - Lines 173-177: calls `recorder.find_call_by_request_hash()` directly
   - Cache key is `(call_type, request_hash)` (line 157) - no sequence number
   - No state to track "which invocation" of a duplicate request we're on
   - Each duplicate request will resolve to the same first recorded call

3. **CallVerifier has identical behavior** (`src/elspeth/plugins/clients/verifier.py:140-208`):
   - Lines 162-166: same `find_call_by_request_hash()` call
   - No sequence tracking for duplicate requests
   - Will always compare against the first recorded response

### Test Coverage

Examined test files:
- `tests/plugins/clients/test_replayer.py`: Has test for "different call types with same hash" (line 311), but NO test for "same call type, same request made twice"
- `tests/plugins/clients/test_verifier.py`: No tests for duplicate identical requests
- Suggested test pattern from bug report (`pytest tests/plugins/clients/ -k replay_duplicate`) finds no matching tests

### Git History Analysis

1. **Commit 4622ed6 (2026-01-20)** - "fix(plugins): include call_type in CallReplayer cache key":
   - Added `call_type` to cache key to prevent collisions between different client types
   - Did NOT address the core issue of duplicate identical requests
   - This was a partial fix for a different collision scenario

2. **No other commits** address duplicate request handling in replay/verify:
   - Searched git history for: "replay.*duplicate", "hash.*collision", "replay.*sequence"
   - No changes to `find_call_by_request_hash` logic since RC1 release (commit c786410)

### Current Behavior Confirmed

The code explicitly implements the problematic behavior described in the bug report:

**Scenario:**
1. Run A makes request X twice (maybe retry, or looping over same input)
2. First call returns response R1
3. Second call returns response R2 (different due to non-deterministic API)
4. Replay mode for Run A will:
   - First replay of request X → R1 (correct)
   - Second replay of request X → R1 (WRONG, should be R2)

**Why this matters:**
- LLM calls with identical prompts can return different responses (temperature > 0)
- Retry scenarios where same request is attempted multiple times
- Loops processing identical items (e.g., batch of identical records)

### Architectural Note

The database schema provides tools to solve this:
- `calls` table has `(state_id, call_index)` which IS unique per call
- `call_index` monotonically increases within a state
- Could use `(state_id, call_index)` as additional disambiguation

However, current replay architecture doesn't track context about which state or index the replay is happening in - it only has `(call_type, request_hash)`.

### Risk Assessment

**Current risk: MEDIUM**
- Bug only manifests when SAME request made multiple times in a single run
- Most common scenarios that trigger this:
  1. Retry logic (same request after failure)
  2. Temperature > 0 LLM calls (non-deterministic responses to identical prompts)
  3. Loop over homogeneous data (e.g., classify 100 identical rows)

**Impact when triggered:**
- Replay will replay wrong response for 2nd+ occurrences
- Verify will compare against wrong baseline
- Both undermine reproducibility guarantees

### Recommendation

**Priority: P1 maintained** - This is a correctness bug in a core audit feature:

1. **Proposed fix remains valid**: Add sequence tracking to CallReplayer/CallVerifier
   - Option A: Track per-request-hash counter: `{(call_type, request_hash): next_index}`
   - Option B: Change lookup to use `(state_id, call_index)` when available
   - Option C: Return all matching calls ordered by `created_at`, consume sequentially

2. **Regression test needed**: Test case where:
   - Record two calls with identical request_data but different responses
   - Replay both calls and assert each gets its correct response (not both getting first)

3. **Documentation**: Current docstring acknowledges limitation but doesn't warn about impact

The bug is confirmed present and unfixed. No workarounds exist in current codebase.
