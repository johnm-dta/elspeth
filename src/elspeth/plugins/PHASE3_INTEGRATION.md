# Phase 3 Integration Points

This document describes how Phase 3 (SDA Engine) will integrate with the
Phase 2 plugin system. These integration points are built into the Phase 2
design to ensure clean integration.

## Canonical Hashing Standard

**IMPORTANT:** All audit hashes (`input_hash`, `output_hash`, `config_hash`, etc.)
use SHA-256 over RFC 8785 canonical JSON via `elspeth.core.canonical.stable_hash`.

```python
from elspeth.core.canonical import stable_hash

# Phase 1 provides:
# - canonical_json(obj) -> str  # RFC 8785 deterministic JSON
# - stable_hash(obj) -> str     # SHA-256 hex digest of canonical JSON

input_hash = stable_hash(row)
config_hash = stable_hash(config)
```

This ensures:
- Cross-process determinism (same data → same hash everywhere)
- NaN and Infinity are rejected (not silently coerced)
- Pandas/NumPy types are normalized to JSON primitives

## PluginContext Integration

Phase 3 creates PluginContext with full integration:

```python
# Phase 3: SDA Engine creates context
ctx = PluginContext(
    run_id=run.run_id,
    config=resolved_config,
    landscape=LandscapeRecorder(db, run.run_id),
    tracer=opentelemetry.trace.get_tracer("elspeth"),
    payload_store=FilesystemPayloadStore(base_path),
)
```

## Transform Processing

Phase 3 engine wraps transform.process() to add audit:

```python
# Phase 3: Engine wraps process() calls
def process_with_audit(transform, row, token_id, step_index, ctx):
    input_hash = stable_hash(row)

    with ctx.start_span(f"transform:{transform.name}") as span:
        span.set_attribute("input_hash", input_hash)

        # Begin node state before processing
        node_state = ctx.landscape.begin_node_state(
            token_id=token_id,
            node_id=transform.node_id,
            step_index=step_index,
            input_data=row,
        )

        start = time.perf_counter()
        try:
            result = transform.process(row, ctx)
            duration_ms = (time.perf_counter() - start) * 1000

            # Populate audit fields
            result.input_hash = input_hash
            result.output_hash = stable_hash(result.row) if result.row else None
            result.duration_ms = duration_ms

            # Complete node state with success
            ctx.landscape.complete_node_state(
                state_id=node_state.state_id,
                status=result.status,
                output_data=result.row,
                duration_ms=duration_ms,
            )

            span.set_attribute("output_hash", result.output_hash)
            span.set_attribute("status", result.status)
            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            # Complete node state with failure
            ctx.landscape.complete_node_state(
                state_id=node_state.state_id,
                status="failed",
                duration_ms=duration_ms,
                error={"type": type(e).__name__, "message": str(e)},
            )
            raise
```

## Aggregation Batch Management

Phase 3 engine manages Landscape batches:

```python
# Phase 3: Engine wraps accept() for batch tracking
def accept_with_batch(aggregation, row, token_id, run_id, ctx):
    if aggregation._batch_id is None:
        # Create batch in Landscape
        batch = ctx.landscape.create_batch(
            run_id=run_id,
            aggregation_node_id=aggregation.node_id,
        )
        aggregation._batch_id = batch.batch_id

    # Persist membership immediately (crash-safe)
    ctx.landscape.add_batch_member(
        batch_id=aggregation._batch_id,
        token_id=token_id,
        ordinal=len(aggregation._buffer),
    )

    result = aggregation.accept(row, ctx)
    result.batch_id = aggregation._batch_id
    return result

def flush_with_audit(aggregation, trigger_reason, state_id, ctx):
    ctx.landscape.update_batch_status(
        aggregation._batch_id,
        status="executing",
    )

    try:
        outputs = aggregation.flush(ctx)

        # Complete batch with success
        ctx.landscape.complete_batch(
            batch_id=aggregation._batch_id,
            status="completed",
            trigger_reason=trigger_reason,
            state_id=state_id,
        )

        return outputs
    except Exception as e:
        ctx.landscape.update_batch_status(
            aggregation._batch_id,
            status="failed",
        )
        raise
```

## Gate Routing Events

Phase 3 engine records routing decisions:

```python
# Phase 3: Engine wraps evaluate() for routing audit
def evaluate_with_routing(gate, row, state_id, edge_id, ctx):
    result = gate.evaluate(row, ctx)

    # Record routing event
    ctx.landscape.record_routing_event(
        state_id=state_id,
        edge_id=edge_id,
        mode=result.action.mode,  # e.g., "continue", "route_to_sink", "fork"
        reason={"action": result.action.reason} if result.action.reason else None,
    )

    return result
```

## Lifecycle Hook Calls

Phase 3 engine calls lifecycle hooks:

```python
# Phase 3: Engine calls hooks at appropriate times

# At run start
for plugin in all_plugins:
    plugin.on_start(ctx)

# At run end
for plugin in all_plugins:
    plugin.on_complete(ctx)
```

## OpenTelemetry Span Structure

Phase 3 creates this span hierarchy:

```
run:{run_id}
├── source:{source_name}
│   └── load
├── row:{row_id}
│   ├── transform:{transform_name}
│   │   └── external_call (Phase 6)
│   ├── gate:{gate_name}
│   └── sink:{sink_name}
└── aggregation:{agg_name}
    └── flush
```

## Checklist for Phase 3 Implementation

- [ ] Create PluginContext with landscape, tracer, payload_store
- [ ] Wrap transform.process() with audit recording
- [ ] Wrap gate.evaluate() with routing event recording
- [ ] Wrap aggregation.accept() with batch member recording
- [ ] Wrap aggregation.flush() with batch status management
- [ ] Call lifecycle hooks at appropriate times
- [ ] Populate audit fields in result types
- [ ] Create OpenTelemetry spans for all operations
