# Outcome Path Map

Current as of 2026-05-20.

Use this map to locate the producer responsible for a token outcome gap. The
exact line numbers change often; treat the file/function names as the stable
orientation and verify against source before editing.

| Outcome/path | Meaning | Primary producer |
|--------------|---------|------------------|
| `(success, default_flow)` | Token reached a normal sink successfully. | Sink/orchestrator completion path in `src/elspeth/engine/`. |
| `(success, gate_routed)` | Gate routed the token to a named sink. | Gate/row processing path in `src/elspeth/engine/processor.py`. |
| `(success, gate_discarded)` | Gate route target intentionally discarded the token. | Gate/row processing path in `src/elspeth/engine/processor.py`. |
| `(failure, on_error_routed)` | Transform failed and `on_error` routed the token to an error sink. | Transform error handling in `src/elspeth/engine/processor.py`. |
| `(success, filter_dropped)` | A filter-style transform intentionally dropped the row. | Transform result handling in `src/elspeth/engine/processor.py`. |
| `(success, coalesced)` | Branch token was consumed by a coalesce operation. | Coalesce handling in `src/elspeth/engine/coalesce_executor.py` and processor recovery branches. |
| `(failure, unrouted)` | A token could not be routed to a valid destination. | Routing failure handling in `src/elspeth/engine/processor.py`. |
| `(failure, quarantined_at_source)` | Source validation failed. | Source handling in `src/elspeth/engine/orchestrator.py` / data-flow repository write path. |
| `(transient, sink_fallback_to_failsink)` | Sink failure was redirected to a failsink and the final lifecycle answer lives in paired evidence. | Sink execution/error handling in `src/elspeth/engine/executors.py`. |
| `(failure, sink_discarded)` | Sink failure was discarded through the discard sentinel. | Sink execution/error handling in `src/elspeth/engine/executors.py`. |
| `(transient, fork_parent)` | Parent token delegated to fork children. | Fork handling in `src/elspeth/core/landscape/data_flow_repository.py` and engine token manager paths. |
| `(transient, expand_parent)` | Parent token delegated to expanded children. | Expand/deaggregation handling in data-flow repository and processor paths. |
| `(transient, batch_consumed)` | Token was consumed by batch handling. | Batch aggregation in `src/elspeth/engine/processor.py` and batch repository paths. |
| `(NULL, buffered)` | Token is waiting in a batch buffer and is not terminal yet. | Batch aggregation in `src/elspeth/engine/processor.py`. |

## Cross-Checks

- If a completed sink node state exists without a terminal token outcome, inspect
  the sink completion path.
- If a terminal token outcome points to a sink but no sink node state/artifact
  exists, inspect the sink write path.
- If a parent path has no child or batch evidence, inspect the corresponding
  token-manager/data-flow repository path.
- If a `buffered` row remains after run completion, inspect batch flush and
  finalization handling.
