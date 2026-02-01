## Summary

AggregationExecutor imports and handles `BatchPendingError` from the LLM plugin pack, so core batch control flow depends on a plugin-specific exception instead of a shared engine/contract type.

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

**Boundary:** engine ↔ plugins/llm

**Integration Point:** batch execution pending signal via BatchPendingError for batch-aware transforms

## Evidence

[MUST include specific file paths and line numbers showing both sides of the seam]

### Side A: engine/executors.py
Path refs: `src/elspeth/engine/executors.py:33`, `src/elspeth/engine/executors.py:949`

```python
 33 from elspeth.plugins.llm.batch_errors import BatchPendingError
...
 949             except BatchPendingError:
 950                 # BatchPendingError is a CONTROL-FLOW SIGNAL, not an error.
 951                 # The batch has been submitted but isn't complete yet.
 952                 # Complete node_state with PENDING status and link batch for audit trail, then re-raise.
```

### Side B: plugins/llm/batch_errors.py
Path refs: `src/elspeth/plugins/llm/batch_errors.py:14`

```python
 14 class BatchPendingError(Exception):
 15     """Raised when batch is submitted but not yet complete.
 16
 17     This is NOT an error condition - it's a control flow signal
 18     telling the engine to schedule a retry check later.
```

### Coupling Evidence: engine control flow depends on plugin-defined exception
Path refs: `src/elspeth/engine/executors.py:949`

```python
 949             except BatchPendingError:
 950                 # BatchPendingError is a CONTROL-FLOW SIGNAL, not an error.
 951                 # The batch has been submitted but isn't complete yet.
```

## Root Cause Hypothesis

Batch-aware retry signaling was introduced in the LLM plugin pack and later adopted by the engine without promoting the exception to a shared contracts/engine module.

## Recommended Fix

[Concrete steps to resolve the seam issue]

1. Define BatchPendingError in a shared module (e.g., `elspeth.contracts` or `elspeth.engine.errors`) as the canonical batch control-flow signal.
2. Update `src/elspeth/engine/executors.py` and `src/elspeth/engine/orchestrator.py` to import the shared type.
3. Update LLM batch transforms to raise the shared type and delete `src/elspeth/plugins/llm/batch_errors.py` (no legacy shim).
4. Add a contract-level test verifying BatchPendingError produces a PENDING node_state during batch flush.

## Impact Assessment

- **Coupling Level:** High - engine execution flow depends on a plugin-pack exception.
- **Maintainability:** Medium - refactor requires coordinated updates across engine and plugins.
- **Type Safety:** Low - control-flow contract is outside contracts.
- **Breaking Change Risk:** Medium - multiple import paths change.

## Related Seams

- `src/elspeth/engine/orchestrator.py`
- `src/elspeth/plugins/llm/azure_batch.py`
- `src/elspeth/plugins/context.py`
---
Template Version: 1.0
---
## Summary

AggregationExecutor passes a list of rows to TransformProtocol.process and silences type checking, but the protocol signature only accepts a single row dict, leaving the batch-aware contract inconsistent at the engine↔plugin boundary.

## Severity

- Severity: minor
- Priority: P2

## Anti-Pattern Classification

[Select one primary category:]

- [ ] Parallel Type Evolution (duplicate definitions of same concept)
- [x] Impedance Mismatch (complex translation at boundaries)
- [ ] Leaky Abstraction (implementation details cross boundaries)
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

**Boundary:** engine ↔ plugins

**Integration Point:** TransformProtocol.process signature vs batch flush invocation

## Evidence

[MUST include specific file paths and line numbers showing both sides of the seam]

### Side A: engine/executors.py
Path refs: `src/elspeth/engine/executors.py:947`

```python
 943         with self._spans.transform_span(transform.name, input_hash=input_hash):
 944             start = time.perf_counter()
 945             try:
 946                 result = transform.process(buffered_rows, ctx)  # type: ignore[arg-type]
 947                 duration_ms = (time.perf_counter() - start) * 1000
```

### Side B: plugins/protocols.py
Path refs: `src/elspeth/plugins/protocols.py:173`

```python
 173     def process(
 174         self,
 175         row: dict[str, Any],
 176         ctx: "PluginContext",
```

### Coupling Evidence: type ignore hides the signature mismatch
Path refs: `src/elspeth/engine/executors.py:946`

```python
 946                 result = transform.process(buffered_rows, ctx)  # type: ignore[arg-type]
```

## Root Cause Hypothesis

Batch-aware processing was added without updating the formal TransformProtocol signature (or splitting into a batch-specific protocol), so engine code bypasses type checking to pass list[dict].

## Recommended Fix

[Concrete steps to resolve the seam issue]

1. Update TransformProtocol to express batch support explicitly (e.g., `row: dict[str, Any] | list[dict[str, Any]]`) or introduce a separate `BatchTransformProtocol` with a batch-specific method.
2. Align `BaseTransform.process` and plugin implementations with the updated protocol.
3. Remove the `# type: ignore[arg-type]` in `src/elspeth/engine/executors.py` after the signature is corrected.
4. Add a typing test or mypy check for batch-aware transforms to enforce the contract.

## Impact Assessment

- **Coupling Level:** Medium - engine assumes a different parameter shape than the protocol states.
- **Maintainability:** Medium - type-checker blind spot invites regressions.
- **Type Safety:** Low - current mismatch is hidden by type ignores.
- **Breaking Change Risk:** Medium - protocol changes require coordinated updates.

## Related Seams

- `src/elspeth/plugins/base.py`
- `src/elspeth/engine/processor.py`
- `src/elspeth/plugins/transforms/batch_stats.py`
- `src/elspeth/plugins/transforms/batch_replicate.py`
---
Template Version: 1.0
