# Bug Report: Gate and sink execution never set ctx.state_id, breaking ctx.record_call

## Summary

- `PluginContext.record_call()` requires `ctx.state_id` and increments a per-state call index.
- `GateExecutor.execute_gate()` and `SinkExecutor.write()` never set `ctx.state_id` or reset `ctx._call_index`.
- Any gate or sink that attempts to record external calls via `ctx.record_call()` will raise `RuntimeError`, preventing audit of external calls at those nodes.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (local)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into contents of `src/elspeth/plugins` and create bug tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: static inspection of `src/elspeth/engine/executors.py` and `src/elspeth/plugins/context.py`

## Steps To Reproduce

1. Implement a `BaseGate` or `BaseSink` that calls `ctx.record_call(...)` during execution.
2. Run a pipeline that executes the gate or sink.
3. Observe `RuntimeError: Cannot record call: state_id not set`.

## Expected Behavior

- Gate and sink executions should set `ctx.state_id` (and reset call_index) so external calls can be recorded in the audit trail.

## Actual Behavior

- `ctx.state_id` is never set for gates or sinks, so `ctx.record_call()` always raises.

## Evidence

- `PluginContext.record_call()` raises when `state_id` is None: `src/elspeth/plugins/context.py`.
- `GateExecutor.execute_gate()` does not set `ctx.state_id` or `ctx._call_index`: `src/elspeth/engine/executors.py`.
- `SinkExecutor.write()` does not set `ctx.state_id` or `ctx._call_index`: `src/elspeth/engine/executors.py`.

## Impact

- User-facing impact: any gate/sink that uses external services cannot record calls and will crash if it tries.
- Data integrity / security impact: external call audit trail is incomplete for gates/sinks, violating auditability requirements.
- Performance or cost impact: failed runs when gates/sinks attempt to record external calls.

## Root Cause Hypothesis

- `ctx.state_id` initialization was added for transform execution paths but not mirrored for gates or sinks.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/engine/executors.py`:
    - In `GateExecutor.execute_gate()`, set `ctx.state_id = state.state_id`, `ctx.node_id = gate.node_id`, and reset `ctx._call_index = 0` before calling `gate.evaluate()`.
    - In `SinkExecutor.write()`, decide on a representative state for external call recording (e.g., the first token's node_state), set `ctx.state_id` and reset `ctx._call_index` before `sink.write()`.
    - Alternatively, add a sink/gate-specific call recording helper that accepts an explicit `state_id`.
- Tests to add/update:
  - Add a gate test that calls `ctx.record_call()` and asserts the call is recorded under the gate's node_state.
  - Add a sink test that calls `ctx.record_call()` during `write()` and asserts a valid call record exists.
- Risks or migration steps:
  - For sinks that write batches, confirm which node_state should own the call records (document the chosen policy).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md` auditability standard (external calls recorded).
- Observed divergence: gate/sink external calls cannot be recorded via PluginContext.
- Reason (if known): state_id setup exists only for transform execution paths.
- Alignment plan or decision needed: define gate/sink call recording semantics (single representative state vs per-token state).

## Acceptance Criteria

- Gates and sinks can call `ctx.record_call()` without raising.
- External calls from gates/sinks are recorded against a valid `node_states.state_id`.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/engine/test_executors.py`
- New tests required: yes (gate/sink external call recording)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md`

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 5

**Current Code Analysis:**

Examined the current state of `src/elspeth/engine/executors.py` and confirmed:

1. **TransformExecutor.execute_transform()** (line 169):
   - Sets `ctx.state_id = state.state_id`
   - Sets `ctx.node_id = transform.node_id`
   - Resets `ctx._call_index = 0`
   - This was added to enable transforms to call `ctx.record_call()`

2. **AggregationExecutor.execute_flush()** (line 930):
   - Sets `ctx.state_id = state.state_id`
   - Sets `ctx.node_id = node_id`
   - Resets `ctx._call_index = 0`
   - This enables batch transforms to record external calls

3. **GateExecutor.execute_gate()** (lines 356-368):
   - Creates `state` via `begin_node_state()` at line 357
   - **DOES NOT set `ctx.state_id`**
   - **DOES NOT set `ctx.node_id`**
   - **DOES NOT reset `ctx._call_index`**
   - Immediately calls `gate.evaluate(token.row_data, ctx)` at line 368

4. **GateExecutor.execute_config_gate()** (lines 515-527):
   - Creates `state` via `begin_node_state()` at line 515
   - **DOES NOT set `ctx.state_id`**
   - **DOES NOT set `ctx.node_id`**
   - **DOES NOT reset `ctx._call_index`**
   - Evaluates the gate condition without setting context state

5. **SinkExecutor.write()** (lines 1398-1411):
   - Creates multiple `state` objects (one per token) at lines 1399-1405
   - **DOES NOT set `ctx.state_id`**
   - **DOES NOT set `ctx.node_id`**
   - **DOES NOT reset `ctx._call_index`**
   - Immediately calls `sink.write(rows, ctx)` at line 1411

**Pattern Verified:**

The pattern is clear and consistent:
- Transform executors (TransformExecutor, AggregationExecutor) set `ctx.state_id`, `ctx.node_id`, and `ctx._call_index`
- Gate executors (both methods) and SinkExecutor do NOT set these fields
- `PluginContext.record_call()` (line 223 in `src/elspeth/plugins/context.py`) raises `RuntimeError` when `state_id` is None

**Git History:**

Searched git history for relevant commits:
- Found commits adding `ctx.state_id` setup for transforms (commit shows state_id was intentionally added to transforms)
- Found commits adding `ctx.state_id` setup for aggregations
- No commits found adding `ctx.state_id` setup for gates or sinks
- The bug was reported on 2026-01-21 and no subsequent fixes have been applied

**Current Plugin Usage:**

Searched for gates and sinks that call `ctx.record_call()`:
- No current gate plugins call `ctx.record_call()`
- No current sink plugins call `ctx.record_call()`
- The only plugin using `ctx.record_call()` is `src/elspeth/plugins/llm/azure_batch.py` (a Transform)

This means the bug exists but is **latent** - it hasn't been triggered because no gates/sinks currently attempt to record external calls.

**Root Cause Confirmed:**

Yes. The root cause is exactly as described in the original report:
- `ctx.state_id` initialization was added for transform execution paths
- Gates and sinks were never updated with the same pattern
- Any future gate or sink that needs to record external calls (e.g., a gate that calls an LLM for routing decisions, or a sink that writes to a remote API) will crash with `RuntimeError: Cannot record call: state_id not set`

**Special Consideration for SinkExecutor:**

The sink case is more complex than gates because `SinkExecutor.write()` creates **multiple** node_states (one per token, lines 1398-1405). The proposed fix needs to decide:
1. Set `ctx.state_id` to the first token's state (as suggested in the original report)
2. Add a loop to set `ctx.state_id` for each token individually (would require API changes to sink.write())
3. Add sink-specific call recording that accepts explicit `state_id` parameter

Option 1 (first token's state) is simplest and matches the pattern used by aggregations, where a representative state is chosen.

**Recommendation:**

**Keep open** - Bug is valid and should be fixed before implementing any gates or sinks that make external API calls. The fix is straightforward for gates (mirror the transform pattern), but requires architectural decision for sinks regarding which node_state should own the call records when writing multiple tokens.

Suggested priority remains P2 since:
- No current plugins are affected (latent bug)
- Future gates/sinks with external calls will need this
- Violates auditability standard when triggered

---

## RESOLUTION: 2026-01-29

**Status:** FIXED

**Closed By:** Claude Code bug review

**Fix Details:**

The bug was fixed in commit `b5f3f50` ("fix(infra): thread safety, integration tests, and Azure audit trail").

**Changes made in `src/elspeth/engine/executors.py`:**

1. **GateExecutor.execute_gate()** (lines 492-494):
   ```python
   # BUG-RECORDER-01 fix: Set state_id on context for external call recording
   # Gates may need to make external calls (e.g., LLM API for routing decisions)
   ctx.state_id = state.state_id
   ctx.node_id = gate.node_id
   ctx._call_index = 0  # Reset call index for this state
   ```

2. **SinkExecutor.write()** (lines 1655-1657):
   ```python
   # BUG-RECORDER-01 fix: Set state_id on context for external call recording
   # Sinks may make external calls (e.g., HTTP POST, database INSERT)
   # Use first token's state_id since sink operations are typically bulk operations
   ctx.state_id = states[0][1].state_id
   ctx.node_id = sink_node_id
   ctx._call_index = 0  # Reset call index for this sink operation
   ```

**Architectural Decision:**
- For sinks processing multiple tokens, the first token's `state_id` is used as the representative state for external call recording (Option 1 from the verification analysis).

**Acceptance Criteria Met:**
- ✅ Gates can call `ctx.record_call()` without raising
- ✅ Sinks can call `ctx.record_call()` without raising
- ✅ External calls recorded against valid `node_states.state_id`
