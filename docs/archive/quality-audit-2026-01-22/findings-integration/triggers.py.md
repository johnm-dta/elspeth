## Summary

TriggerConfig’s condition example assumes row-level fields, but TriggerEvaluator evaluates conditions against a batch-only context (`batch_count`, `batch_age_seconds`), so row-based trigger expressions are invalid at runtime.

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

**Boundary:** core/config ↔ engine/triggers

**Integration Point:** TriggerConfig.condition expression semantics vs TriggerEvaluator evaluation context

## Evidence

[MUST include specific file paths and line numbers showing both sides of the seam]

### Side A: core/config.py (`src/elspeth/core/config.py:37`)

```python
37 class TriggerConfig(BaseModel):
38     """Trigger configuration for aggregation batches.
39
40     Per plugin-protocol.md: Multiple triggers can be combined (first one to fire wins).
41     The engine evaluates all configured triggers after each accept and fires when
42     ANY condition is met.
43
44     Trigger types:
45     - count: Fire after N rows accumulated
46     - timeout: Fire after N seconds since first accept
47     - condition: Fire when expression evaluates to true
48
49     Note: end_of_source is IMPLICIT - always checked at source exhaustion.
50     It is not configured here because it always applies.
51
52     Example YAML (combined triggers):
53         trigger:
54           count: 1000           # Fire after 1000 rows
55           timeout_seconds: 3600         # Or after 1 hour
56           condition: "row['type'] == 'flush_signal'"  # Or on special row
```

### Side B: engine/triggers.py (`src/elspeth/engine/triggers.py:117`)

```python
117         # Check condition trigger
118         if self._condition_parser is not None:
119             # ExpressionParser.evaluate() accepts a dict that becomes "row" in expressions.
120             # So row['batch_count'] accesses this dict directly.
121             context = {
122                 "batch_count": self._batch_count,
123                 "batch_age_seconds": self.batch_age_seconds,
124             }
125             result = self._condition_parser.evaluate(context)
```

### Coupling Evidence: engine/triggers.py (`src/elspeth/engine/triggers.py:57`)

```python
57         # Pre-parse condition expression if applicable
58         self._condition_parser: ExpressionParser | None = None
59         if config.condition is not None:
60             self._condition_parser = ExpressionParser(config.condition)
```

## Root Cause Hypothesis

Trigger condition guidance and examples were inherited from row-based gate expressions without defining or enforcing the batch-specific context that TriggerEvaluator actually supplies.

## Recommended Fix

[Concrete steps to resolve the seam issue]

1. Define the trigger condition contract explicitly (row-based vs batch-metric-based) in `TriggerConfig` and related docs.
2. If row-based, change `TriggerEvaluator.should_trigger` to accept the last accepted row (and optionally batch metrics) and pass it to `ExpressionParser.evaluate`.
3. If batch-only, update docs/examples to use `row['batch_count']`/`row['batch_age_seconds']` and validate allowed keys at config time.
4. Add unit tests that exercise trigger conditions against the chosen context, including invalid field access.
5. Update sample pipeline configs to match the finalized contract and remove conflicting examples.

## Impact Assessment

- **Coupling Level:** Medium - Shared expression syntax with implicit context.
- **Maintainability:** Medium - Config and engine semantics can drift.
- **Type Safety:** Low - String expressions are not key-validated.
- **Breaking Change Risk:** Medium - Aligning semantics may require config updates.

## Related Seams

`src/elspeth/engine/executors.py`, `src/elspeth/engine/expression_parser.py`, `src/elspeth/engine/processor.py`, `src/elspeth/core/config.py`
---
Template Version: 1.0
---
## Summary

AggregationExecutor restores TriggerEvaluator timing by mutating the private `_first_accept_time` field, leaking TriggerEvaluator internals across the engine boundary.

## Severity

- Severity: major
- Priority: P1

## Anti-Pattern Classification

[Select one primary category:]

- [ ] Parallel Type Evolution (duplicate definitions of same concept)
- [ ] Impedance Mismatch (complex translation at boundaries)
- [x] Leaky Abstraction (implementation details cross boundaries)
- [ ] Contract Violation (undocumented assumptions)
- [ ] Shared Mutable State (unclear ownership)
- [ ] God Object (excessive coupling through large context)
- [ ] Stringly-Typed Interface (magic strings instead of types)
- [ ] Missing Facade (complex subsystem without simple interface)
- [ ] Protocol Drift (versions incompatible)
- [ ] Callback Hell (complex async chains)
- [ ] Missing Error Translation (low-level errors leak)
- [ ] Implicit State Dependencies (call order matters but not enforced)

## Seam Location

**Boundary:** engine/executors ↔ engine/triggers

**Integration Point:** checkpoint restore of trigger timing state

## Evidence

[MUST include specific file paths and line numbers showing both sides of the seam]

### Side A: engine/triggers.py (`src/elspeth/engine/triggers.py:52`)

```python
52         self._config = config
53         self._batch_count = 0
54         self._first_accept_time: float | None = None
55         self._last_triggered: Literal["count", "timeout", "condition"] | None = None
56
68     def batch_age_seconds(self) -> float:
69         """Seconds since first accept in this batch."""
70         if self._first_accept_time is None:
71             return 0.0
72         return time.monotonic() - self._first_accept_time
74     def get_age_seconds(self) -> float:
75         """Get elapsed time since first accept (alias for batch_age_seconds).
76
77         This method exists for clarity when checkpointing - it returns the
78         elapsed time that should be stored in checkpoint state for timeout
79         preservation across resume.
83         return self.batch_age_seconds
```

### Side B: engine/executors.py (`src/elspeth/engine/executors.py:1251`)

```python
1251             # Restore trigger evaluator count and timeout age (Bug #6 fix)
1252             evaluator = self._trigger_evaluators.get(node_id)
1253             if evaluator is not None:
1254                 # Record each restored row as "accepted" to advance the count
1255                 for _ in reconstructed_tokens:
1256                     evaluator.record_accept()
1257
1258                 # Restore timeout age to preserve SLA (Bug #6 fix)
1259                 # The checkpoint stores how much time had elapsed before the crash.
1260                 # We adjust _first_accept_time backwards so batch_age_seconds
1261                 # reflects the true elapsed time (not reset to zero).
1262                 elapsed_seconds = node_state.get("elapsed_age_seconds", 0.0)
1263                 if elapsed_seconds > 0.0:
1264                     # Adjust timer: make it think first accept was N seconds ago
1265                     evaluator._first_accept_time = time.monotonic() - elapsed_seconds
```

### Coupling Evidence: engine/executors.py (`src/elspeth/engine/executors.py:1127`)

```python
1127             # Get timeout elapsed time for SLA preservation (Bug #6 fix)
1128             evaluator = self._trigger_evaluators.get(node_id)
1129             elapsed_age_seconds = evaluator.get_age_seconds() if evaluator is not None else 0.0
1143             "elapsed_age_seconds": elapsed_age_seconds,  # Bug #6: Preserve timeout window
```

## Root Cause Hypothesis

Checkpoint persistence needs to restore elapsed time, but TriggerEvaluator exposes only a getter, so AggregationExecutor reaches into private state to set `_first_accept_time`.

## Recommended Fix

[Concrete steps to resolve the seam issue]

1. Add a public `TriggerEvaluator` method (for example, `restore_elapsed_seconds`) to set elapsed time safely.
2. Update AggregationExecutor restore logic to call the new method instead of mutating `_first_accept_time`.
3. Validate elapsed seconds inside TriggerEvaluator (non-negative, consistent with batch state).
4. Extend checkpoint restore tests to verify `batch_age_seconds` matches persisted age.
5. Remove all direct access to TriggerEvaluator private fields from other modules.

## Impact Assessment

- **Coupling Level:** Medium - Executor depends on TriggerEvaluator internals.
- **Maintainability:** Medium - Internal refactors risk breaking restore logic.
- **Type Safety:** Medium - Access bypasses explicit API contracts.
- **Breaking Change Risk:** Low - Internal API addition is localized.

## Related Seams

`src/elspeth/engine/executors.py`, `src/elspeth/engine/triggers.py`, `src/elspeth/core/checkpoint/manager.py`, `src/elspeth/core/checkpoint/recovery.py`
---
Template Version: 1.0
