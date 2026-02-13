# Outcome Path Map

This map shows where each token outcome is recorded in the engine and why.
Use it to locate the source of any audit gap.

## Source quarantine (QUARANTINED)

- Path: Source validation failures
- Recorder: Orchestrator creates token and records QUARANTINED
- Primary location: `src/elspeth/engine/orchestrator.py`
  - In the main source loop, when `source_item.is_quarantined` is true.

## Transform success (COMPLETED)

- Path: Token reaches sink successfully
- Recorder: Orchestrator (it knows sink mapping)
- Primary locations:
  - `src/elspeth/engine/orchestrator.py` records COMPLETED with sink_name.
  - `src/elspeth/engine/executors.py` (SinkExecutor) creates completed sink node_states.

## Transform error routing (ROUTED or QUARANTINED)

- Path: Transform returns error
- Recorder: RowProcessor
- Primary location: `src/elspeth/engine/processor.py`
  - If error_sink == "discard" -> QUARANTINED
  - Else -> ROUTED with sink_name

## Retry exhaustion (FAILED)

- Path: Transform raises retryable error beyond max retries
- Recorder: RowProcessor
- Primary location: `src/elspeth/engine/processor.py`
  - `_execute_transform_with_retry` + MaxRetriesExceeded branch

## Gate routing (ROUTED)

- Path: Gate returns RoutingAction.route
- Recorder: RowProcessor
- Primary locations:
  - `src/elspeth/engine/processor.py` (plugin gates)
  - `src/elspeth/engine/processor.py` (config-driven gates)

## Gate fork (FORKED)

- Path: Gate returns RoutingAction.fork_to_paths
- Recorder: RowProcessor
- Primary locations:
  - `src/elspeth/engine/processor.py` (plugin gates)
  - `src/elspeth/engine/processor.py` (config-driven gates)

- Children creation: `TokenManager.fork_token` -> `LandscapeRecorder.fork_token`
  - Child tokens get fork_group_id and token_parents links.

## Deaggregation (EXPANDED)

- Path: Transform returns multi-row output with creates_tokens=True
- Recorder: RowProcessor
- Primary location: `src/elspeth/engine/processor.py`
  - Parent token gets EXPANDED
- Children creation: `TokenManager.expand_token` -> `LandscapeRecorder.expand_token`
  - Child tokens share expand_group_id and token_parents links.

## Aggregation (BUFFERED, CONSUMED_IN_BATCH)

- Path: Batch-aware transform (aggregation settings)
- Recorder: RowProcessor
- Primary location: `src/elspeth/engine/processor.py` `_process_batch_aggregation_node`
  - BUFFERED when passthrough mode and not flushed
  - CONSUMED_IN_BATCH for transform mode when buffered
  - On flush, result rows continue downstream or complete at sink

## Coalesce success (COALESCED)

- Path: Fork children meet coalesce policy
- Recorder: CoalesceExecutor
- Primary location: `src/elspeth/engine/coalesce_executor.py`
  - `_execute_merge` records COALESCED for each consumed token
  - Merged token continues; it gets COMPLETED only at sink

## Coalesce failure (FAILED)

- Path: Missing branches, quorum not met, late arrival
- Recorder:
  - CoalesceExecutor records FAILED for consumed tokens on flush_pending
  - RowProcessor records FAILED for late arrivals or immediate failure_reason
- Primary locations:
  - `src/elspeth/engine/coalesce_executor.py` (flush_pending)
  - `src/elspeth/engine/processor.py` (failure_reason handling)

## Sink write (node_state, not outcome)

- Sink writes always create a node_state for every token.
- `SinkExecutor.write` completes node_states with status="completed".
- Orchestrator then records COMPLETED outcome with sink_name.

## Cross-check rule

If a token has a completed sink node_state but no COMPLETED outcome, the
outcome recording path is missing. If a token has COMPLETED outcome but no
completed sink node_state, the sink path or node_state recording is missing.
