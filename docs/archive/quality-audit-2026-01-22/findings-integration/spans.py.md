## Summary

SpanFactory still defines an `aggregation_span` labeled as an aggregation plugin, but aggregation execution is now structural via batch-aware transforms, so tracing metadata and hierarchy drift from actual execution and omit batch-specific identifiers.

## Severity

- Severity: minor
- Priority: P3

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
- [x] Protocol Drift (versions incompatible)
- [ ] Callback Hell (complex async chains)
- [ ] Missing Error Translation (low-level errors leak)
- [ ] Implicit State Dependencies (call order matters but not enforced)

## Seam Location

**Boundary:** engine/observability ↔ engine/aggregation execution

**Integration Point:** aggregation span taxonomy (`aggregation_span`, `plugin.type`) vs batch-aware transform aggregation flow

## Evidence

[MUST include specific file paths and line numbers showing both sides of the seam]

### Side A: engine/observability (`src/elspeth/engine/spans.py`)

```python
193    @contextmanager
194    def aggregation_span(
195        self,
196        aggregation_name: str,
197        *,
198        batch_id: str | None = None,
199    ) -> Iterator["Span | NoOpSpan"]:
200        """Create a span for an aggregation flush.
201
202        Args:
203            aggregation_name: Name of the aggregation plugin
204            batch_id: Optional batch identifier
205        """
213        with self._tracer.start_as_current_span(f"aggregation:{aggregation_name}") as span:
214            span.set_attribute("plugin.name", aggregation_name)
215            span.set_attribute("plugin.type", "aggregation")
216            if batch_id:
217                span.set_attribute("batch.id", batch_id)
```

### Side B: engine/aggregation execution (`src/elspeth/engine/processor.py`)

```python
724            # NOTE: BaseAggregation branch was DELETED in aggregation structural cleanup.
725            # Aggregation is now handled by batch-aware transforms (is_batch_aware=True).
726            # The engine buffers rows and calls Transform.process(rows: list[dict]).
728            elif isinstance(transform, BaseTransform):
729                # Check if this is a batch-aware transform at an aggregation node
730                node_id = transform.node_id
731                if transform.is_batch_aware and node_id is not None and node_id in self._aggregation_settings:
732                    # Use engine buffering for aggregation
733                    return self._process_batch_aggregation_node(
```

### Coupling Evidence: aggregation flush uses transform spans (`src/elspeth/engine/executors.py`)

```python
943        # Step 3: Execute with timing and span
944        with self._spans.transform_span(transform.name, input_hash=input_hash):
945            start = time.perf_counter()
```

## Root Cause Hypothesis

Aggregation was refactored into structural batch-aware transforms, but the SpanFactory API and span hierarchy documentation were not updated, leaving a stale aggregation span contract that no longer matches execution.

## Recommended Fix

1. Decide on a single canonical tracing contract for aggregation (either remove `aggregation_span` entirely or use it consistently for batch flushes).
2. If keeping `aggregation_span`, update aggregation execution to call it and pass `batch_id`/aggregation name from settings rather than treating flushes as generic transforms.
3. Align span hierarchy documentation and any consumer expectations with the chosen contract.

## Impact Assessment

- **Coupling Level:** Medium - tracing consumers depend on span metadata matching execution semantics.
- **Maintainability:** Medium - drift invites future mismatches and confusing observability.
- **Type Safety:** Low - aggregation classification is implicit and unchecked.
- **Breaking Change Risk:** Low - primarily affects telemetry and dashboards, not pipeline execution.

## Related Seams

`src/elspeth/plugins/protocols.py`
