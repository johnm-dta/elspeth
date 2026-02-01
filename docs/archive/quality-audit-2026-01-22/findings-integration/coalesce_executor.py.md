## Summary

Coalesce failure outcomes returned from `flush_pending` are never recorded because `CoalesceExecutor` only returns `failure_reason` while the orchestrator assumes the executor already logged the failure.

## Severity

- Severity: major
- Priority: P1

## Anti-Pattern Classification

[Select one primary category:]

- [ ] Parallel Type Evolution (duplicate definitions of same concept)
- [ ] Impedance Mismatch (complex translation at boundaries)
- [ ] Leaky Abstraction (implementation details cross boundaries)
- [x] Contract Violation (undocumented assumptions)
- [ ] Shared Mutable State (unclear ownership)
- [ ] God Object (excessive coupling through large context)
- [ ] Stringly-Typed Interface (magic strings instead of types)
- [ ] Missing Facade (complex subsystem without simple interface)
- [ ] Protocol Drift (versions incompatible)
- [ ] Callback Hell (complex async chains)
- [ ] Missing Error Translation (low-level errors leak)
- [ ] Implicit State Dependencies (call order matters but not enforced)

## Seam Location

**Boundary:** engine/coalesce executor ↔ engine/orchestrator

**Integration Point:** `flush_pending` failure handling and audit recording responsibility

## Evidence

### Side A: engine/coalesce executor (`src/elspeth/engine/coalesce_executor.py:410-449`)

```python
410 |             elif settings.policy == "quorum":
411 |                 assert settings.quorum_count is not None
412 |                 if len(pending.arrived) >= settings.quorum_count:
413 |                     outcome = self._execute_merge(
414 |                         settings=settings,
415 |                         node_id=node_id,
416 |                         pending=pending,
417 |                         step_in_pipeline=step_in_pipeline,
418 |                         key=key,
419 |                     )
420 |                     results.append(outcome)
421 |                 else:
422 |                     # Quorum not met - record failure
423 |                     del self._pending[key]
424 |                     results.append(
425 |                         CoalesceOutcome(
426 |                             held=False,
427 |                             failure_reason="quorum_not_met",
428 |                             coalesce_metadata={
429 |                                 "policy": settings.policy,
430 |                                 "quorum_required": settings.quorum_count,
431 |                                 "branches_arrived": list(pending.arrived.keys()),
432 |                             },
433 |                         )
434 |                     )
435 |
436 |             elif settings.policy == "require_all":
437 |                 # require_all never does partial merge
438 |                 del self._pending[key]
439 |                 results.append(
440 |                     CoalesceOutcome(
441 |                         held=False,
442 |                         failure_reason="incomplete_branches",
443 |                         coalesce_metadata={
444 |                             "policy": settings.policy,
445 |                             "expected_branches": settings.branches,
446 |                             "branches_arrived": list(pending.arrived.keys()),
447 |                         },
448 |                     )
449 |                 )
```

### Side B: engine/orchestrator (`src/elspeth/engine/orchestrator.py:1051-1068`)

```python
1051 |                 # Flush pending coalesce operations at end-of-source
1052 |                 if coalesce_executor is not None:
1053 |                     # Step for coalesce flush = after all transforms and gates
1054 |                     flush_step = len(config.transforms) + len(config.gates)
1055 |                     pending_outcomes = coalesce_executor.flush_pending(flush_step)
1056 |
1057 |                     # Handle any merged tokens from flush
1058 |                     for outcome in pending_outcomes:
1059 |                         if outcome.merged_token is not None:
1060 |                             # Successful merge - route to output sink
1061 |                             rows_coalesced += 1
1062 |                             pending_tokens[output_sink_name].append(outcome.merged_token)
1063 |                         elif outcome.failure_reason:
1064 |                             # Coalesce failed (timeout, missing branches, etc.)
1065 |                             # Failure is recorded in audit trail by executor.
1066 |                             # Not counted as rows_failed since the individual fork children
1067 |                             # were already counted when they reached their terminal states.
1068 |                             pass
```

### Coupling Evidence: success-only audit recording path (`src/elspeth/engine/coalesce_executor.py:236-249`)

```python
236 |         # Record node states for consumed tokens
237 |         for token in consumed_tokens:
238 |             state = self._recorder.begin_node_state(
239 |                 token_id=token.token_id,
240 |                 node_id=node_id,
241 |                 step_index=step_in_pipeline,
242 |                 input_data=token.row_data,
243 |             )
244 |             self._recorder.complete_node_state(
245 |                 state_id=state.state_id,
246 |                 status="completed",
247 |                 output_data={"merged_into": merged_token.token_id},
248 |                 duration_ms=0,
249 |             )
```

## Root Cause Hypothesis

Failure handling responsibility was split between executor and orchestrator without a concrete audit-recording contract; the executor returns failure metadata but never writes audit records, while the orchestrator assumes it does.

## Recommended Fix

1. Define the canonical audit contract for coalesce failures (which tokens get terminal outcomes and which outcome values to use).
2. In `CoalesceExecutor.flush_pending` (and any timeout failure paths), record node states and token outcomes with `failure_reason`/metadata via `LandscapeRecorder`.
3. Update orchestrator failure handling to either assert executor recording or perform it if the executor is the chosen owner.
4. Add tests that assert audit records exist for `quorum_not_met` and `incomplete_branches` cases.

## Impact Assessment

- **Coupling Level:** Medium - executor/orchestrator split responsibility without enforcement
- **Maintainability:** Medium - failures are easy to miss in audit trail
- **Type Safety:** Low - runtime-only contract
- **Breaking Change Risk:** Medium - audit schema usage and outcome semantics may change

## Related Seams

- `src/elspeth/engine/processor.py`
- `src/elspeth/contracts/enums.py`
- `src/elspeth/core/landscape/recorder.py`
---
Template Version: 1.0
---
## Summary

Coalesce timeouts are implemented in `CoalesceExecutor` but never invoked by orchestration, so best_effort/quorum timeouts do not trigger during execution.

## Severity

- Severity: major
- Priority: P2

## Anti-Pattern Classification

[Select one primary category:]

- [ ] Parallel Type Evolution (duplicate definitions of same concept)
- [ ] Impedance Mismatch (complex translation at boundaries)
- [ ] Leaky Abstraction (implementation details cross boundaries)
- [ ] Contract Violation (undocumented assumptions)
- [ ] Shared Mutable State (unclear ownership)
- [ ] God Object (excessive coupling through large context)
- [ ] Stringly-Typed Interface (magic strings instead of types)
- [ ] Missing Facade (complex subsystem without simple interface)
- [ ] Protocol Drift (versions incompatible)
- [ ] Callback Hell (complex async chains)
- [ ] Missing Error Translation (low-level errors leak)
- [x] Implicit State Dependencies (call order matters but not enforced)

## Seam Location

**Boundary:** engine/coalesce executor ↔ engine/orchestrator

**Integration Point:** `check_timeouts` scheduling for best_effort/quorum timeout semantics

## Evidence

### Side A: engine/coalesce executor (`src/elspeth/engine/coalesce_executor.py:303-311`)

```python
303 |     def check_timeouts(
304 |         self,
305 |         coalesce_name: str,
306 |         step_in_pipeline: int,
307 |     ) -> list[CoalesceOutcome]:
308 |         """Check for timed-out pending coalesces and merge them.
309 |
310 |         For best_effort policy, merges whatever has arrived when timeout expires.
311 |         For quorum policy with timeout, merges if quorum met when timeout expires.
```

### Side B: engine/orchestrator (`src/elspeth/engine/orchestrator.py:1051-1055`)

```python
1051 |                 # Flush pending coalesce operations at end-of-source
1052 |                 if coalesce_executor is not None:
1053 |                     # Step for coalesce flush = after all transforms and gates
1054 |                     flush_step = len(config.transforms) + len(config.gates)
1055 |                     pending_outcomes = coalesce_executor.flush_pending(flush_step)
```

### Coupling Evidence: timeout dependency called out in executor (`src/elspeth/engine/coalesce_executor.py:208-210`)

```python
208 |         # settings.policy == "best_effort":
209 |         # Only merge on timeout (checked elsewhere) or if all arrived
210 |         return arrived_count == expected_count
```

## Root Cause Hypothesis

Timeout enforcement was added to the executor but the orchestration loop never scheduled `check_timeouts`, leaving the dependency implicit and unmet.

## Recommended Fix

1. Introduce a periodic timeout check in the orchestrator main loop (time-based cadence or every N rows).
2. For each configured coalesce point, call `coalesce_executor.check_timeouts` with the correct `step_in_pipeline`.
3. Route any merged tokens from timeout checks through the same sink path as `flush_pending`.
4. Add tests verifying best_effort and quorum timeouts trigger merges before end-of-source.

## Impact Assessment

- **Coupling Level:** Medium - executor relies on external scheduling
- **Maintainability:** Medium - timeout behavior depends on hidden call order
- **Type Safety:** Low - no compile-time enforcement
- **Breaking Change Risk:** Medium - runtime behavior changes for timeout policies

## Related Seams

- `src/elspeth/core/config.py`
- `src/elspeth/engine/processor.py`
- `src/elspeth/engine/orchestrator.py`
---
Template Version: 1.0
