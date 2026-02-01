# Bug Report: Concurrent execute_batch calls can mix results across batches

## Summary

- PooledExecutor uses a single ReorderBuffer for all batches. If `execute_batch()` is called concurrently on the same executor, results from different batches can be interleaved and returned to the wrong caller because both calls drain the shared buffer.

## Severity

- Severity: major
- Priority: P3

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 / fix/rc1-bug-burndown-session-2
- OS: Linux
- Python version: Python 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Deep dive src/elspeth/plugins/pooling for bugs; create bug reports.
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: Code inspection only

## Steps To Reproduce

1. Create a single `PooledExecutor` instance.
2. Spawn two threads that both call `execute_batch()` with different marker rows.
3. Use `process_fn` delays so completions interleave across batches.
4. Observe each thread's results list contains rows from the other batch or blocks waiting on the other batch's indices.

## Expected Behavior

- Each `execute_batch()` call should be isolated, or concurrent calls should be rejected with a clear error.

## Actual Behavior

- Results can be drained from a shared reorder buffer, causing cross-batch contamination or ordering stalls.

## Evidence

- Code: single buffer shared per executor instance (`src/elspeth/plugins/pooling/executor.py:98-101`).
- `execute_batch()` drains the shared buffer without scoping results to a batch (`src/elspeth/plugins/pooling/executor.py:192-210`).

## Impact

- User-facing impact: Wrong rows returned to callers in multi-threaded usage.
- Data integrity / security impact: Audit trail can associate results with the wrong row set.
- Performance or cost impact: Potential stalls if batches block each other on ordering.

## Root Cause Hypothesis

- Reorder buffer and counters are shared across batches; `execute_batch()` is not re-entrant or guarded.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/plugins/pooling/executor.py`
- Config or schema changes: None
- Tests to add/update: Add a concurrency test that runs two simultaneous batches and asserts isolation.
- Risks or migration steps: If concurrent calls are unsupported, enforce a single-flight lock and document it.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): N/A
- Observed divergence: No batch scoping or mutual exclusion for concurrent calls.
- Reason (if known): Executor assumes single caller.
- Alignment plan or decision needed: Decide whether to support concurrency or reject it explicitly.

## Acceptance Criteria

- Concurrent execute_batch calls either isolate buffers or raise a deterministic error.
- Tests demonstrate no cross-batch result mixing.

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_pooled_executor.py -k concurrent`
- New tests required: Yes (concurrent batch isolation).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

## Verification (2026-01-25)

**Status: STILL VALID**

### Analysis

The bug is confirmed to exist in the current codebase (commit c786410, RC-1). The issue persists from the originally reported commit (ae2c0e6).

### Evidence

1. **Shared buffer state**: `PooledExecutor.__init__()` creates a single `ReorderBuffer` instance at line 99:
   ```python
   self._buffer: ReorderBuffer[TransformResult] = ReorderBuffer()
   ```

2. **Non-isolated buffer operations**: `execute_batch()` (lines 143-212) calls buffer operations without any mutual exclusion:
   - `buffer.submit()` allocates sequential indices from shared counter `_next_submit`
   - `buffer.get_ready_results()` drains entries starting from shared counter `_next_emit`

3. **Result mixing scenario**:
   ```
   Thread A: submit() → indices [0, 1]
   Thread B: submit() → indices [2, 3]
   Thread A: complete(0), complete(1)
   Thread B: complete(2)
   Thread A: get_ready_results() → drains [0, 1, 2]  ← Thread A steals Thread B's result!
   Thread B: get_ready_results() → blocks waiting for index 2 that was already drained
   ```

4. **No concurrency guards**: The code has no locks, no reentrancy checks, no per-batch buffer scoping.

5. **No concurrent tests**: Searched for tests with `grep -i "concurrent.*execute_batch"` - none exist. All existing tests call `execute_batch()` sequentially on a single thread.

### Current usage analysis

**Current risk level: LOW (but bug is real)**

Examined all `PooledExecutor` usage in the codebase:
- `src/elspeth/plugins/llm/azure.py` - stores `self._executor`, calls from `process()`
- `src/elspeth/plugins/llm/openrouter.py` - stores `self._executor`, calls from `process()`
- `src/elspeth/plugins/transforms/azure/prompt_shield.py` - likely similar pattern
- `src/elspeth/plugins/transforms/azure/content_safety.py` - likely similar pattern

The engine (`src/elspeth/engine/executors.py`) calls `transform.process()` sequentially (single-threaded execution per transform instance). This means concurrent `execute_batch()` calls cannot occur in production with the current architecture.

However, the bug exists as a **latent defect**:
- Future refactors might introduce concurrent transform execution
- Test code might inadvertently call `execute_batch()` concurrently
- Plugin developers might create multi-threaded transforms

### Root cause

The `PooledExecutor` class conflates two responsibilities:
1. Managing a pool of worker threads (correct, reusable)
2. Managing per-batch state (incorrect, should be isolated)

The `ReorderBuffer` contains batch-specific state (`_next_submit`, `_next_emit`) but is shared across all batches. This is a classic thread-safety violation.

### Recommended fix

**Option 1: Enforce single-flight** (simplest, matches current usage)
- Add `threading.Lock` to `execute_batch()` to serialize calls
- Document that concurrent calls are unsupported
- Advantage: Minimal code change, clear semantics
- Disadvantage: Prevents future parallel batch processing

**Option 2: Per-batch buffer isolation** (more flexible)
- Create a new `ReorderBuffer` instance for each `execute_batch()` call
- Pass buffer instance to workers via closure
- Advantage: Supports concurrent batches
- Disadvantage: More complex, workers need batch-scoped context

### Verification methodology

- Read source code at reported commit (ae2c0e6) and current commit (c786410)
- Analyzed `PooledExecutor` and `ReorderBuffer` implementations
- Traced execution flow for concurrent scenarios
- Searched for existing tests: `grep -ri "concurrent.*execute_batch"`
- Examined all usages: `grep -r "PooledExecutor" src/`
- Checked git history for related fixes: `git log --grep="concurrent\|isolation"`

### Confidence

**HIGH** - Bug exists in code, reproduction scenario is clear, impact is well-understood. Only reason it hasn't manifested is current single-threaded transform execution architecture.
