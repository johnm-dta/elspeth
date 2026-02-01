# Mutation Testing Gaps Checklist

**Generated:** 2026-01-25
**Mutants Tested:** 2,778 / 2,778 (100%)
**Kill Rate:** 55.5%
**Total Survivors:** 833
**Actionable Survivors:** 642 (excluding schema.py)

Use this checklist to systematically address surviving mutants.
Each survivor represents a line where a bug could hide undetected.

---

## Quick Reference

| Priority | File | Survivors | Why It Matters |
|----------|------|-----------|----------------|
| P0 | `orchestrator.py` | 206 | Critical orchestration logic |
| P0 | `executors.py` | 88 | Core execution engine |
| P0 | `processor.py` | 62 | Row processing and routing |
| P1 | `recorder.py` | 64 | Audit trail recording |
| P1 | `coalesce_executor.py` | 29 | Fork/join merge logic |
| P2 | `exporter.py` | 88 | Audit export integrity |
| P2 | `models.py` | 39 | Audit data models |

---

## src/elspeth/engine/executors.py

**Priority:** P0
**Survivors:** 88 unique lines (119 total mutants)

### Line 63

```python
      62 |         self.label = label
>>>   63 |         super().__init__(
      64 |             f"No edge registered from node {node_id} with label '{label}'. Audit trail would be incomplete - refusing to proceed."
```

- [ ] Add test to catch mutations on this line

### Line 77

```python
      76 |     result: GateResult
>>>   77 |     updated_token: TokenInfo
      78 |     child_tokens: list[TokenInfo] = field(default_factory=list)
```

- [ ] Add test to catch mutations on this line

### Line 78 (2 mutants)

```python
      77 |     updated_token: TokenInfo
>>>   78 |     child_tokens: list[TokenInfo] = field(default_factory=list)
      79 |     sink_name: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 121

```python
     120 |         ctx: PluginContext,
>>>  121 |         step_in_pipeline: int,
     122 |         attempt: int = 0,
```

- [ ] Add test to catch mutations on this line

### Line 154

```python
     153 |             RuntimeError: Transform returned error but has no on_error configured
>>>  154 |         """
     155 |         assert transform.node_id is not None, "node_id must be set by orchestrator"
```

- [ ] Add test to catch mutations on this line

### Line 168

```python
     167 |         # Set state_id and node_id on context for external call recording
>>>  168 |         # and batch checkpoint lookup (node_id required for _batch_checkpoints keying)
     169 |         ctx.state_id = state.state_id
```

- [ ] Add test to catch mutations on this line

### Line 169

```python
     168 |         # and batch checkpoint lookup (node_id required for _batch_checkpoints keying)
>>>  169 |         ctx.state_id = state.state_id
     170 |         ctx.node_id = transform.node_id
```

- [ ] Add test to catch mutations on this line

### Line 170 (2 mutants)

```python
     169 |         ctx.state_id = state.state_id
>>>  170 |         ctx.node_id = transform.node_id
     171 |         ctx._call_index = 0  # Reset call index for this state
```

- [ ] Add test to catch mutations on this line

### Line 177 (3 mutants)

```python
     176 |             try:
>>>  177 |                 result = transform.process(token.row_data, ctx)
     178 |                 duration_ms = (time.perf_counter() - start) * 1000
```

- [ ] Add test to catch mutations on this line

### Line 179 (3 mutants)

```python
     178 |                 duration_ms = (time.perf_counter() - start) * 1000
>>>  179 |             except Exception as e:
     180 |                 duration_ms = (time.perf_counter() - start) * 1000
```

- [ ] Add test to catch mutations on this line

### Line 182

```python
     181 |                 # Record failure
>>>  182 |                 error: ExecutionError = {
     183 |                     "exception": str(e),
```

- [ ] Add test to catch mutations on this line

### Line 183

```python
     182 |                 error: ExecutionError = {
>>>  183 |                     "exception": str(e),
     184 |                     "type": type(e).__name__,
```

- [ ] Add test to catch mutations on this line

### Line 197

```python
     196 |         if result.row is not None:
>>>  197 |             result.output_hash = stable_hash(result.row)
     198 |         elif result.rows is not None:
```

- [ ] Add test to catch mutations on this line

### Line 198

```python
     197 |             result.output_hash = stable_hash(result.row)
>>>  198 |         elif result.rows is not None:
     199 |             result.output_hash = stable_hash(result.rows)
```

- [ ] Add test to catch mutations on this line

### Line 200

```python
     199 |             result.output_hash = stable_hash(result.rows)
>>>  200 |         else:
     201 |             result.output_hash = None
```

- [ ] Add test to catch mutations on this line

### Line 204

```python
     203 |
>>>  204 |         # Initialize error_sink - will be set if transform errors with on_error configured
     205 |         error_sink: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 209

```python
     208 |         if result.status == "success":
>>>  209 |             # TransformResult.success() or success_multi() always sets output data
     210 |             assert result.has_output_data, "success status requires row or rows data"
```

- [ ] Add test to catch mutations on this line

### Line 251

```python
     250 |                     f"Transform '{transform.name}' returned error but has no on_error "
>>>  251 |                     f"configured. Either configure on_error or fix the transform to not "
     252 |                     f"return errors for this input. Error: {result.reason}"
```

- [ ] Add test to catch mutations on this line

### Line 352

```python
     351 |             Exception: Re-raised from gate.evaluate() after recording failure
>>>  352 |         """
     353 |         assert gate.node_id is not None, "node_id must be set by orchestrator"
```

- [ ] Add test to catch mutations on this line

### Line 368 (3 mutants)

```python
     367 |             try:
>>>  368 |                 result = gate.evaluate(token.row_data, ctx)
     369 |                 duration_ms = (time.perf_counter() - start) * 1000
```

- [ ] Add test to catch mutations on this line

### Line 370 (3 mutants)

```python
     369 |                 duration_ms = (time.perf_counter() - start) * 1000
>>>  370 |             except Exception as e:
     371 |                 duration_ms = (time.perf_counter() - start) * 1000
```

- [ ] Add test to catch mutations on this line

### Line 373

```python
     372 |                 # Record failure
>>>  373 |                 error: ExecutionError = {
     374 |                     "exception": str(e),
```

- [ ] Add test to catch mutations on this line

### Line 374

```python
     373 |                 error: ExecutionError = {
>>>  374 |                     "exception": str(e),
     375 |                     "type": type(e).__name__,
```

- [ ] Add test to catch mutations on this line

### Line 375

```python
     374 |                     "exception": str(e),
>>>  375 |                     "type": type(e).__name__,
     376 |                 }
```

- [ ] Add test to catch mutations on this line

### Line 392

```python
     391 |         action = result.action
>>>  392 |         child_tokens: list[TokenInfo] = []
     393 |         sink_name: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 412

```python
     411 |                 raise MissingEdgeError(node_id=gate.node_id, label=route_label)
>>>  412 |
     413 |             if destination == "continue":
```

- [ ] Add test to catch mutations on this line

### Line 418

```python
     417 |                     state_id=state.state_id,
>>>  418 |                     node_id=gate.node_id,
     419 |                     action=RoutingAction.route("continue", mode=action.mode, reason=dict(action.reason)),
```

- [ ] Add test to catch mutations on this line

### Line 433

```python
     432 |             if token_manager is None:
>>>  433 |                 raise RuntimeError(
     434 |                     f"Gate {gate.node_id} returned fork_to_paths but no TokenManager provided. "
```

- [ ] Add test to catch mutations on this line

### Line 434

```python
     433 |                 raise RuntimeError(
>>>  434 |                     f"Gate {gate.node_id} returned fork_to_paths but no TokenManager provided. "
     435 |                     "Cannot create child tokens - audit integrity would be compromised."
```

- [ ] Add test to catch mutations on this line

### Line 527 (3 mutants)

```python
     526 |                 parser = ExpressionParser(gate_config.condition)
>>>  527 |                 eval_result = parser.evaluate(token.row_data)
     528 |                 duration_ms = (time.perf_counter() - start) * 1000
```

- [ ] Add test to catch mutations on this line

### Line 529 (3 mutants)

```python
     528 |                 duration_ms = (time.perf_counter() - start) * 1000
>>>  529 |             except Exception as e:
     530 |                 duration_ms = (time.perf_counter() - start) * 1000
```

- [ ] Add test to catch mutations on this line

### Line 532

```python
     531 |                 # Record failure
>>>  532 |                 error: ExecutionError = {
     533 |                     "exception": str(e),
```

- [ ] Add test to catch mutations on this line

### Line 556 (2 mutants)

```python
     555 |             # Record failure before raising
>>>  556 |             error = {
     557 |                 "exception": f"Route label '{route_label}' not found in routes config",
```

- [ ] Add test to catch mutations on this line

### Line 557 (2 mutants)

```python
     556 |             error = {
>>>  557 |                 "exception": f"Route label '{route_label}' not found in routes config",
     558 |                 "type": "ValueError",
```

- [ ] Add test to catch mutations on this line

### Line 558

```python
     557 |                 "exception": f"Route label '{route_label}' not found in routes config",
>>>  558 |                 "type": "ValueError",
     559 |             }
```

- [ ] Add test to catch mutations on this line

### Line 566

```python
     565 |             )
>>>  566 |             raise ValueError(
     567 |                 f"Gate '{gate_config.name}' condition returned '{route_label}' which is not in routes: {list(gate_config.routes.keys())}"
```

- [ ] Add test to catch mutations on this line

### Line 573

```python
     572 |         # Build routing action and process based on destination
>>>  573 |         child_tokens: list[TokenInfo] = []
     574 |         sink_name: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 592 (2 mutants)

```python
     591 |             if token_manager is None:
>>>  592 |                 error = {
     593 |                     "exception": "fork requires TokenManager",
```

- [ ] Add test to catch mutations on this line

### Line 593 (2 mutants)

```python
     592 |                 error = {
>>>  593 |                     "exception": "fork requires TokenManager",
     594 |                     "type": "RuntimeError",
```

- [ ] Add test to catch mutations on this line

### Line 594

```python
     593 |                     "exception": "fork requires TokenManager",
>>>  594 |                     "type": "RuntimeError",
     595 |                 }
```

- [ ] Add test to catch mutations on this line

### Line 602

```python
     601 |                 )
>>>  602 |                 raise RuntimeError(
     603 |                     f"Gate {node_id} routes to fork but no TokenManager provided. "
```

- [ ] Add test to catch mutations on this line

### Line 603

```python
     602 |                 raise RuntimeError(
>>>  603 |                     f"Gate {node_id} routes to fork but no TokenManager provided. "
     604 |                     "Cannot create child tokens - audit integrity would be compromised."
```

- [ ] Add test to catch mutations on this line

### Line 747

```python
     746 |         self._run_id = run_id
>>>  747 |         self._member_counts: dict[str, int] = {}  # batch_id -> count for ordinals
     748 |         self._batch_ids: dict[str, str | None] = {}  # node_id -> current batch_id
```

- [ ] Add test to catch mutations on this line

### Line 778

```python
     777 |         """
>>>  778 |         if node_id not in self._buffers:
     779 |             self._buffers[node_id] = []
```

- [ ] Add test to catch mutations on this line

### Line 779

```python
     778 |         if node_id not in self._buffers:
>>>  779 |             self._buffers[node_id] = []
     780 |             self._buffer_tokens[node_id] = []
```

- [ ] Add test to catch mutations on this line

### Line 788

```python
     787 |             )
>>>  788 |             self._batch_ids[node_id] = batch.batch_id
     789 |             self._member_counts[batch.batch_id] = 0
```

- [ ] Add test to catch mutations on this line

### Line 804 (2 mutants)

```python
     803 |             ordinal=ordinal,
>>>  804 |         )
     805 |         self._member_counts[batch_id] = ordinal + 1
```

- [ ] Add test to catch mutations on this line

### Line 883

```python
     882 |         batch_id = self._batch_ids.get(node_id)
>>>  883 |         if batch_id is None:
     884 |             raise RuntimeError(f"No batch exists for node {node_id} - cannot flush")
```

- [ ] Add test to catch mutations on this line

### Line 890

```python
     889 |
>>>  890 |         if not buffered_rows:
     891 |             raise RuntimeError(f"Cannot flush empty buffer for node {node_id}")
```

- [ ] Add test to catch mutations on this line

### Line 893

```python
     892 |
>>>  893 |         # Compute input hash for batch (hash of all input rows)
     894 |         input_hash = stable_hash(buffered_rows)
```

- [ ] Add test to catch mutations on this line

### Line 901

```python
     900 |         self._recorder.update_batch_status(
>>>  901 |             batch_id=batch_id,
     902 |             status="executing",
```

- [ ] Add test to catch mutations on this line

### Line 907 (2 mutants)

```python
     906 |         # Step 2: Begin node state for flush operation
>>>  907 |         # Wrap batch rows in a dict for node_state recording
     908 |         batch_input: dict[str, Any] = {"batch_rows": buffered_rows}
```

- [ ] Add test to catch mutations on this line

### Line 913

```python
     912 |             step_index=step_in_pipeline,
>>>  913 |             input_data=batch_input,
     914 |             attempt=0,
```

- [ ] Add test to catch mutations on this line

### Line 918

```python
     917 |         # Set state_id and node_id on context for external call recording
>>>  918 |         # and batch checkpoint lookup (node_id required for _batch_checkpoints keying)
     919 |         ctx.state_id = state.state_id
```

- [ ] Add test to catch mutations on this line

### Line 919

```python
     918 |         # and batch checkpoint lookup (node_id required for _batch_checkpoints keying)
>>>  919 |         ctx.state_id = state.state_id
     920 |         ctx.node_id = node_id
```

- [ ] Add test to catch mutations on this line

### Line 920 (2 mutants)

```python
     919 |         ctx.state_id = state.state_id
>>>  920 |         ctx.node_id = node_id
     921 |         ctx._call_index = 0  # Reset call index for this state
```

- [ ] Add test to catch mutations on this line

### Line 927 (3 mutants)

```python
     926 |             try:
>>>  927 |                 result = transform.process(buffered_rows, ctx)  # type: ignore[arg-type]
     928 |                 duration_ms = (time.perf_counter() - start) * 1000
```

- [ ] Add test to catch mutations on this line

### Line 936 (3 mutants)

```python
     935 |                 raise
>>>  936 |             except Exception as e:
     937 |                 duration_ms = (time.perf_counter() - start) * 1000
```

- [ ] Add test to catch mutations on this line

### Line 940

```python
     939 |                 # Record failure in node_state
>>>  940 |                 error: ExecutionError = {
     941 |                     "exception": str(e),
```

- [ ] Add test to catch mutations on this line

### Line 941

```python
     940 |                 error: ExecutionError = {
>>>  941 |                     "exception": str(e),
     942 |                     "type": type(e).__name__,
```

- [ ] Add test to catch mutations on this line

### Line 942

```python
     941 |                     "exception": str(e),
>>>  942 |                     "type": type(e).__name__,
     943 |                 }
```

- [ ] Add test to catch mutations on this line

### Line 963

```python
     962 |
>>>  963 |         # Step 4: Populate audit fields on result
     964 |         result.input_hash = input_hash
```

- [ ] Add test to catch mutations on this line

### Line 964

```python
     963 |         # Step 4: Populate audit fields on result
>>>  964 |         result.input_hash = input_hash
     965 |         if result.row is not None:
```

- [ ] Add test to catch mutations on this line

### Line 965

```python
     964 |         result.input_hash = input_hash
>>>  965 |         if result.row is not None:
     966 |             result.output_hash = stable_hash(result.row)
```

- [ ] Add test to catch mutations on this line

### Line 966

```python
     965 |         if result.row is not None:
>>>  966 |             result.output_hash = stable_hash(result.row)
     967 |         elif result.rows is not None:
```

- [ ] Add test to catch mutations on this line

### Line 967

```python
     966 |             result.output_hash = stable_hash(result.row)
>>>  967 |         elif result.rows is not None:
     968 |             result.output_hash = stable_hash(result.rows)
```

- [ ] Add test to catch mutations on this line

### Line 969

```python
     968 |             result.output_hash = stable_hash(result.rows)
>>>  969 |         else:
     970 |             result.output_hash = None
```

- [ ] Add test to catch mutations on this line

### Line 970

```python
     969 |         else:
>>>  970 |             result.output_hash = None
     971 |         result.duration_ms = duration_ms
```

- [ ] Add test to catch mutations on this line

### Line 998 (2 mutants)

```python
     997 |             # Transform returned error status
>>>  998 |             error_info: ExecutionError = {
     999 |                 "exception": str(result.reason) if result.reason else "Transform returned error",
```

- [ ] Add test to catch mutations on this line

### Line 999 (2 mutants)

```python
     998 |             error_info: ExecutionError = {
>>>  999 |                 "exception": str(result.reason) if result.reason else "Transform returned error",
    1000 |                 "type": "TransformError",
```

- [ ] Add test to catch mutations on this line

### Line 1000

```python
     999 |                 "exception": str(result.reason) if result.reason else "Transform returned error",
>>> 1000 |                 "type": "TransformError",
    1001 |             }
```

- [ ] Add test to catch mutations on this line

### Line 1022

```python
    1021 |
>>> 1022 |         # Reset trigger evaluator for next batch
    1023 |         evaluator = self._trigger_evaluators.get(node_id)
```

- [ ] Add test to catch mutations on this line

### Line 1023

```python
    1022 |         # Reset trigger evaluator for next batch
>>> 1023 |         evaluator = self._trigger_evaluators.get(node_id)
    1024 |         if evaluator is not None:
```

- [ ] Add test to catch mutations on this line

### Line 1037

```python
    1036 |         if batch_id is not None:
>>> 1037 |             del self._batch_ids[node_id]
    1038 |             if batch_id in self._member_counts:
```

- [ ] Add test to catch mutations on this line

### Line 1121

```python
    1120 |         evaluator = self._trigger_evaluators.get(node_id)
>>> 1121 |         if evaluator is None:
    1122 |             return False
```

- [ ] Add test to catch mutations on this line

### Line 1133

```python
    1132 |             TriggerType enum if a trigger fired, None otherwise
>>> 1133 |         """
    1134 |         evaluator = self._trigger_evaluators.get(node_id)
```

- [ ] Add test to catch mutations on this line

### Line 1134

```python
    1133 |         """
>>> 1134 |         evaluator = self._trigger_evaluators.get(node_id)
    1135 |         if evaluator is None:
```

- [ ] Add test to catch mutations on this line

### Line 1178

```python
    1177 |         batch = self._recorder.get_batch(batch_id)
>>> 1178 |         if batch is None:
    1179 |             raise ValueError(f"Batch not found: {batch_id}")
```

- [ ] Add test to catch mutations on this line

### Line 1185

```python
    1184 |         # Restore member count from database
>>> 1185 |         members = self._recorder.get_batch_members(batch_id)
    1186 |         self._member_counts[batch_id] = len(members)
```

- [ ] Add test to catch mutations on this line

### Line 1273

```python
    1272 |         # Create node_state for EACH token - this is how we derive COMPLETED terminal state
>>> 1273 |         # Sink must have node_id assigned by orchestrator before execution
    1274 |         assert sink.node_id is not None, "Sink node_id must be set before execution"
```

- [ ] Add test to catch mutations on this line

### Line 1291 (3 mutants)

```python
    1290 |             try:
>>> 1291 |                 artifact_info = sink.write(rows, ctx)
    1292 |                 duration_ms = (time.perf_counter() - start) * 1000
```

- [ ] Add test to catch mutations on this line

### Line 1293 (3 mutants)

```python
    1292 |                 duration_ms = (time.perf_counter() - start) * 1000
>>> 1293 |             except Exception as e:
    1294 |                 duration_ms = (time.perf_counter() - start) * 1000
```

- [ ] Add test to catch mutations on this line

### Line 1296

```python
    1295 |                 # Mark all token states as failed
>>> 1296 |                 error: ExecutionError = {
    1297 |                     "exception": str(e),
```

- [ ] Add test to catch mutations on this line

### Line 1297

```python
    1296 |                 error: ExecutionError = {
>>> 1297 |                     "exception": str(e),
    1298 |                     "type": type(e).__name__,
```

- [ ] Add test to catch mutations on this line

### Line 1298

```python
    1297 |                     "exception": str(e),
>>> 1298 |                     "type": type(e).__name__,
    1299 |                 }
```

- [ ] Add test to catch mutations on this line

### Line 1312

```python
    1311 |         for token, state in states:
>>> 1312 |             sink_output = {
    1313 |                 "row": token.row_data,
```

- [ ] Add test to catch mutations on this line

### Line 1313

```python
    1312 |             sink_output = {
>>> 1313 |                 "row": token.row_data,
    1314 |                 "artifact_path": artifact_info.path_or_uri,
```

- [ ] Add test to catch mutations on this line

### Line 1314

```python
    1313 |                 "row": token.row_data,
>>> 1314 |                 "artifact_path": artifact_info.path_or_uri,
    1315 |                 "content_hash": artifact_info.content_hash,
```

- [ ] Add test to catch mutations on this line

## src/elspeth/engine/orchestrator.py

**Priority:** P0
**Survivors:** 206 unique lines (285 total mutants)

### Line 48 (2 mutants)

```python
      47 | # NOTE: BaseAggregation was DELETED - aggregation is now handled by
>>>   48 | # batch-aware transforms (is_batch_aware=True on BaseTransform)
      49 | RowPlugin = BaseTransform | BaseGate
```

- [ ] Add test to catch mutations on this line

### Line 132

```python
     131 |         *,
>>>  132 |         event_bus: "EventBusProtocol" = None,  # type: ignore[assignment]
     133 |         canonical_version: str = "sha256-rfc8785-v1",
```

- [ ] Add test to catch mutations on this line

### Line 169

```python
     168 |         self._sequence_number += 1
>>>  169 |
     170 |         should_checkpoint = False
```

- [ ] Add test to catch mutations on this line

### Line 212

```python
     211 |                 # Log but don't raise - cleanup should be best-effort
>>>  212 |                 logger.warning(
     213 |                     "Transform cleanup failed",
```

- [ ] Add test to catch mutations on this line

### Line 246

```python
     245 |         for seq, transform in enumerate(transforms):
>>>  246 |             if isinstance(transform, BaseGate):
     247 |                 node_id = transform_id_map.get(seq)
```

- [ ] Add test to catch mutations on this line

### Line 247

```python
     246 |             if isinstance(transform, BaseGate):
>>>  247 |                 node_id = transform_id_map.get(seq)
     248 |                 if node_id is not None:
```

- [ ] Add test to catch mutations on this line

### Line 248

```python
     247 |                 node_id = transform_id_map.get(seq)
>>>  248 |                 if node_id is not None:
     249 |                     node_id_to_gate_name[node_id] = transform.name
```

- [ ] Add test to catch mutations on this line

### Line 251

```python
     250 |
>>>  251 |         # Add config gates to the lookup
     252 |         if config_gate_id_map and config_gates:
```

- [ ] Add test to catch mutations on this line

### Line 253

```python
     252 |         if config_gate_id_map and config_gates:
>>>  253 |             for gate_config in config_gates:
     254 |                 node_id = config_gate_id_map.get(gate_config.name)
```

- [ ] Add test to catch mutations on this line

### Line 254

```python
     253 |             for gate_config in config_gates:
>>>  254 |                 node_id = config_gate_id_map.get(gate_config.name)
     255 |                 if node_id is not None:
```

- [ ] Add test to catch mutations on this line

### Line 261

```python
     260 |             # "continue" means proceed to next transform, not a sink
>>>  261 |             if destination == "continue":
     262 |                 continue
```

- [ ] Add test to catch mutations on this line

### Line 265

```python
     264 |             # "fork" means fork to multiple paths, not a sink
>>>  265 |             if destination == "fork":
     266 |                 continue
```

- [ ] Add test to catch mutations on this line

### Line 271

```python
     270 |                 gate_name = node_id_to_gate_name.get(gate_node_id, gate_node_id)
>>>  271 |                 raise RouteValidationError(
     272 |                     f"Gate '{gate_name}' can route to '{destination}' "
```

- [ ] Add test to catch mutations on this line

### Line 272

```python
     271 |                 raise RouteValidationError(
>>>  272 |                     f"Gate '{gate_name}' can route to '{destination}' "
     273 |                     f"(via route label '{route_label}') but no sink named "
```

- [ ] Add test to catch mutations on this line

### Line 273

```python
     272 |                     f"Gate '{gate_name}' can route to '{destination}' "
>>>  273 |                     f"(via route label '{route_label}') but no sink named "
     274 |                     f"'{destination}' exists. Available sinks: {sorted(available_sinks)}"
```

- [ ] Add test to catch mutations on this line

### Line 299

```python
     298 |             # Only BaseTransform has _on_error; BaseGate uses routing, not error sinks
>>>  299 |             if not isinstance(transform, BaseTransform):
     300 |                 continue
```

- [ ] Add test to catch mutations on this line

### Line 306

```python
     305 |             if on_error is None:
>>>  306 |                 # No error routing configured - that's fine
     307 |                 continue
```

- [ ] Add test to catch mutations on this line

### Line 310

```python
     309 |             if on_error == "discard":
>>>  310 |                 # "discard" is a special value, not a sink name
     311 |                 continue
```

- [ ] Add test to catch mutations on this line

### Line 315

```python
     314 |             if on_error not in available_sinks:
>>>  315 |                 raise RouteValidationError(
     316 |                     f"Transform '{transform.name}' has on_error='{on_error}' "
```

- [ ] Add test to catch mutations on this line

### Line 316

```python
     315 |                 raise RouteValidationError(
>>>  316 |                     f"Transform '{transform.name}' has on_error='{on_error}' "
     317 |                     f"but no sink named '{on_error}' exists. "
```

- [ ] Add test to catch mutations on this line

### Line 317

```python
     316 |                     f"Transform '{transform.name}' has on_error='{on_error}' "
>>>  317 |                     f"but no sink named '{on_error}' exists. "
     318 |                     f"Available sinks: {sorted(available_sinks)}. "
```

- [ ] Add test to catch mutations on this line

### Line 318

```python
     317 |                     f"but no sink named '{on_error}' exists. "
>>>  318 |                     f"Available sinks: {sorted(available_sinks)}. "
     319 |                     f"Use 'discard' to drop error rows without routing."
```

- [ ] Add test to catch mutations on this line

### Line 360

```python
     359 |         if on_validation_failure not in available_sinks:
>>>  360 |             raise RouteValidationError(
     361 |                 f"Source '{source.name}' has on_validation_failure='{on_validation_failure}' "
```

- [ ] Add test to catch mutations on this line

### Line 361

```python
     360 |             raise RouteValidationError(
>>>  361 |                 f"Source '{source.name}' has on_validation_failure='{on_validation_failure}' "
     362 |                 f"but no sink named '{on_validation_failure}' exists. "
```

- [ ] Add test to catch mutations on this line

### Line 362

```python
     361 |                 f"Source '{source.name}' has on_validation_failure='{on_validation_failure}' "
>>>  362 |                 f"but no sink named '{on_validation_failure}' exists. "
     363 |                 f"Available sinks: {sorted(available_sinks)}. "
```

- [ ] Add test to catch mutations on this line

### Line 363

```python
     362 |                 f"but no sink named '{on_validation_failure}' exists. "
>>>  363 |                 f"Available sinks: {sorted(available_sinks)}. "
     364 |                 f"Use 'discard' to drop invalid rows without routing."
```

- [ ] Add test to catch mutations on this line

### Line 400

```python
     399 |             if transform.node_id is not None:
>>>  400 |                 # Already has node_id (e.g., aggregation transform) - skip
     401 |                 continue
```

- [ ] Add test to catch mutations on this line

### Line 403

```python
     402 |             if seq not in transform_id_map:
>>>  403 |                 raise ValueError(
     404 |                     f"Transform at sequence {seq} not found in graph. Graph has mappings for sequences: {list(transform_id_map.keys())}"
```

- [ ] Add test to catch mutations on this line

### Line 410

```python
     409 |         for sink_name, sink in sinks.items():
>>>  410 |             if sink_name not in sink_id_map:
     411 |                 raise ValueError(f"Sink '{sink_name}' not found in graph. Available sinks: {list(sink_id_map.keys())}")
```

- [ ] Add test to catch mutations on this line

### Line 440

```python
     439 |         """
>>>  440 |         if graph is None:
     441 |             raise ValueError("ExecutionGraph is required. Build with ExecutionGraph.from_config(settings)")
```

- [ ] Add test to catch mutations on this line

### Line 448

```python
     447 |
>>>  448 |             # Schemas are required by plugin protocols - access directly
     449 |             source_output = config.source.output_schema
```

- [ ] Add test to catch mutations on this line

### Line 460 (2 mutants)

```python
     459 |             )
>>>  460 |             if schema_errors:
     461 |                 raise ValueError(f"Pipeline schema incompatibility: {'; '.join(schema_errors)}")
```

- [ ] Add test to catch mutations on this line

### Line 462

```python
     461 |                 raise ValueError(f"Pipeline schema incompatibility: {'; '.join(schema_errors)}")
>>>  462 |
     463 |             self._events.emit(PhaseCompleted(phase=PipelinePhase.SCHEMA_VALIDATION, duration_seconds=time.perf_counter() - phase_start))
```

- [ ] Add test to catch mutations on this line

### Line 478

```python
     477 |             )
>>>  478 |
     479 |             self._events.emit(PhaseCompleted(phase=PipelinePhase.DATABASE, duration_seconds=time.perf_counter() - phase_start))
```

- [ ] Add test to catch mutations on this line

### Line 483

```python
     482 |             raise  # CRITICAL: Always re-raise - database connection failure is fatal
>>>  483 |
     484 |         run_completed = False
```

- [ ] Add test to catch mutations on this line

### Line 500 (2 mutants)

```python
     499 |             recorder.complete_run(run.run_id, status="completed")
>>>  500 |             result.status = RunStatus.COMPLETED
     501 |             run_completed = True
```

- [ ] Add test to catch mutations on this line

### Line 527

```python
     526 |
>>>  527 |                     recorder.set_export_status(run.run_id, status="completed")
     528 |                     self._events.emit(PhaseCompleted(phase=PipelinePhase.EXPORT, duration_seconds=time.perf_counter() - phase_start))
```

- [ ] Add test to catch mutations on this line

### Line 540 (2 mutants)

```python
     539 |
>>>  540 |             # Emit RunCompleted event with final metrics
     541 |             total_duration = time.perf_counter() - run_start_time
```

- [ ] Add test to catch mutations on this line

### Line 550

```python
     549 |                     quarantined=result.rows_quarantined,
>>>  550 |                     duration_seconds=total_duration,
     551 |                     exit_code=0,
```

- [ ] Add test to catch mutations on this line

### Line 565 (2 mutants)

```python
     564 |         except Exception:
>>>  565 |             # Emit RunCompleted with failure status
     566 |             total_duration = time.perf_counter() - run_start_time
```

- [ ] Add test to catch mutations on this line

### Line 578

```python
     577 |                         quarantined=result.rows_quarantined,
>>>  578 |                         duration_seconds=total_duration,
     579 |                         exit_code=1,
```

- [ ] Add test to catch mutations on this line

### Line 588

```python
     587 |                         run_id=run.run_id,
>>>  588 |                         status=RunCompletionStatus.FAILED,
     589 |                         total_rows=0,
```

- [ ] Add test to catch mutations on this line

### Line 589

```python
     588 |                         status=RunCompletionStatus.FAILED,
>>>  589 |                         total_rows=0,
     590 |                         succeeded=0,
```

- [ ] Add test to catch mutations on this line

### Line 590

```python
     589 |                         total_rows=0,
>>>  590 |                         succeeded=0,
     591 |                         failed=0,
```

- [ ] Add test to catch mutations on this line

### Line 591

```python
     590 |                         succeeded=0,
>>>  591 |                         failed=0,
     592 |                         quarantined=0,
```

- [ ] Add test to catch mutations on this line

### Line 593

```python
     592 |                         quarantined=0,
>>>  593 |                         duration_seconds=total_duration,
     594 |                         exit_code=2,  # exit_code: 0=success, 1=partial, 2=total failure
```

- [ ] Add test to catch mutations on this line

### Line 671

```python
     670 |                 if node_id in config_gate_node_ids:
>>>  671 |                     # Config gates are deterministic (expression evaluation is deterministic)
     672 |                     plugin_version = "1.0.0"
```

- [ ] Add test to catch mutations on this line

### Line 676 (2 mutants)

```python
     675 |                     # Aggregations use batch-aware transforms - determinism depends on the transform
>>>  676 |                     # Default to deterministic (statistical operations are typically deterministic)
     677 |                     plugin_version = "1.0.0"
```

- [ ] Add test to catch mutations on this line

### Line 677

```python
     676 |                     # Default to deterministic (statistical operations are typically deterministic)
>>>  677 |                     plugin_version = "1.0.0"
     678 |                     determinism = Determinism.DETERMINISTIC
```

- [ ] Add test to catch mutations on this line

### Line 680

```python
     679 |                 elif node_id in coalesce_node_ids:
>>>  680 |                     # Coalesce nodes merge tokens from parallel paths - deterministic operation
     681 |                     plugin_version = "1.0.0"
```

- [ ] Add test to catch mutations on this line

### Line 695

```python
     694 |                 # Get schema_config from node_info config or default to dynamic
>>>  695 |                 # Schema is specified in pipeline config, not plugin attributes
     696 |                 schema_dict = node_info.config.get("schema", {"fields": "dynamic"})
```

- [ ] Add test to catch mutations on this line

### Line 752

```python
     751 |             )
>>>  752 |
     753 |             self._events.emit(PhaseCompleted(phase=PipelinePhase.GRAPH, duration_seconds=time.perf_counter() - phase_start))
```

- [ ] Add test to catch mutations on this line

### Line 760

```python
     759 |         source_id = graph.get_source()
>>>  760 |         if source_id is None:
     761 |             raise ValueError("Graph has no source node")
```

- [ ] Add test to catch mutations on this line

### Line 782

```python
     781 |             config=config.config,
>>>  782 |             landscape=recorder,
     783 |             _batch_checkpoints=batch_checkpoints or {},
```

- [ ] Add test to catch mutations on this line

### Line 788

```python
     787 |         # This must be set BEFORE source.load() so that any validation errors
>>>  788 |         # (e.g., malformed CSV rows) can be attributed to the source node
     789 |         ctx.node_id = source_id
```

- [ ] Add test to catch mutations on this line

### Line 799

```python
     798 |
>>>  799 |         # Create retry manager from settings if available
     800 |         retry_manager: RetryManager | None = None
```

- [ ] Add test to catch mutations on this line

### Line 807

```python
     806 |         from elspeth.engine.tokens import TokenManager
>>>  807 |
     808 |         coalesce_executor: CoalesceExecutor | None = None
```

- [ ] Add test to catch mutations on this line

### Line 808

```python
     807 |
>>>  808 |         coalesce_executor: CoalesceExecutor | None = None
     809 |         branch_to_coalesce: dict[str, str] = {}
```

- [ ] Add test to catch mutations on this line

### Line 835

```python
     834 |             for i, cs in enumerate(settings.coalesce):
>>>  835 |                 # Each coalesce gets its own step (in case of multiple)
     836 |                 coalesce_step_map[cs.name] = base_step + i
```

- [ ] Add test to catch mutations on this line

### Line 867 (2 mutants)

```python
     866 |         rows_forked = 0
>>>  867 |         rows_coalesced = 0
     868 |         rows_expanded = 0
```

- [ ] Add test to catch mutations on this line

### Line 868 (2 mutants)

```python
     867 |         rows_coalesced = 0
>>>  868 |         rows_expanded = 0
     869 |         rows_buffered = 0
```

- [ ] Add test to catch mutations on this line

### Line 873

```python
     872 |         # Progress tracking - hybrid timing: emit on 100 rows OR 5 seconds
>>>  873 |         progress_interval = 100
     874 |         progress_time_interval = 5.0  # seconds
```

- [ ] Add test to catch mutations on this line

### Line 883

```python
     882 |         if config.gates:
>>>  883 |             last_gate_name = config.gates[-1].name
     884 |             default_last_node_id = config_gate_id_map[last_gate_name]
```

- [ ] Add test to catch mutations on this line

### Line 887

```python
     886 |             transform_node_id = config.transforms[-1].node_id
>>>  887 |             assert transform_node_id is not None
     888 |             default_last_node_id = transform_node_id
```

- [ ] Add test to catch mutations on this line

### Line 889

```python
     888 |             default_last_node_id = transform_node_id
>>>  889 |         else:
     890 |             default_last_node_id = source_id
```

- [ ] Add test to catch mutations on this line

### Line 906

```python
     905 |                 raise  # Re-raise to propagate SOURCE failures (cleanup will still run via outer finally)
>>>  906 |
     907 |             self._events.emit(PhaseCompleted(phase=PipelinePhase.SOURCE, duration_seconds=time.perf_counter() - phase_start))
```

- [ ] Add test to catch mutations on this line

### Line 919

```python
     918 |                     # Handle quarantined source rows - route directly to sink
>>>  919 |                     if source_item.is_quarantined:
     920 |                         rows_quarantined += 1
```

- [ ] Add test to catch mutations on this line

### Line 921

```python
     920 |                         rows_quarantined += 1
>>>  921 |                         # Route quarantined row to configured sink if it exists
     922 |                         quarantine_sink = source_item.quarantine_destination
```

- [ ] Add test to catch mutations on this line

### Line 922 (2 mutants)

```python
     921 |                         # Route quarantined row to configured sink if it exists
>>>  922 |                         quarantine_sink = source_item.quarantine_destination
     923 |                         if quarantine_sink and quarantine_sink in config.sinks:
```

- [ ] Add test to catch mutations on this line

### Line 934 (2 mutants)

```python
     933 |                         # Hybrid timing: emit on first row, every 100 rows, or every 5 seconds
>>>  934 |                         current_time = time.perf_counter()
     935 |                         time_since_last_progress = current_time - last_progress_time
```

- [ ] Add test to catch mutations on this line

### Line 936 (2 mutants)

```python
     935 |                         time_since_last_progress = current_time - last_progress_time
>>>  936 |                         should_emit = (
     937 |                             rows_processed == 1  # First row - immediate feedback
```

- [ ] Add test to catch mutations on this line

### Line 938

```python
     937 |                             rows_processed == 1  # First row - immediate feedback
>>>  938 |                             or rows_processed % progress_interval == 0  # Every 100 rows
     939 |                             or time_since_last_progress >= progress_time_interval  # Every 5 seconds
```

- [ ] Add test to catch mutations on this line

### Line 941 (2 mutants)

```python
     940 |                         )
>>>  941 |                         if should_emit:
     942 |                             elapsed = current_time - start_time
```

- [ ] Add test to catch mutations on this line

### Line 946

```python
     945 |                                     rows_processed=rows_processed,
>>>  946 |                                     # Include routed rows in success count - they reached their destination
     947 |                                     rows_succeeded=rows_succeeded + rows_routed,
```

- [ ] Add test to catch mutations on this line

### Line 987

```python
     986 |                             pending_tokens[result.sink_name].append(result.token)
>>>  987 |                         elif result.outcome == RowOutcome.FAILED:
     988 |                             rows_failed += 1
```

- [ ] Add test to catch mutations on this line

### Line 1001

```python
    1000 |                             rows_coalesced += 1
>>> 1001 |                             pending_tokens[output_sink_name].append(result.token)
    1002 |                         elif result.outcome == RowOutcome.EXPANDED:
```

- [ ] Add test to catch mutations on this line

### Line 1003 (3 mutants)

```python
    1002 |                         elif result.outcome == RowOutcome.EXPANDED:
>>> 1003 |                             # Deaggregation parent token - children counted separately
    1004 |                             rows_expanded += 1
```

- [ ] Add test to catch mutations on this line

### Line 1004

```python
    1003 |                             # Deaggregation parent token - children counted separately
>>> 1004 |                             rows_expanded += 1
    1005 |                         elif result.outcome == RowOutcome.BUFFERED:
```

- [ ] Add test to catch mutations on this line

### Line 1006 (3 mutants)

```python
    1005 |                         elif result.outcome == RowOutcome.BUFFERED:
>>> 1006 |                             # Passthrough mode buffered token
    1007 |                             rows_buffered += 1
```

- [ ] Add test to catch mutations on this line

### Line 1015

```python
    1014 |                         rows_processed == 1  # First row - immediate feedback
>>> 1015 |                         or rows_processed % progress_interval == 0  # Every 100 rows
    1016 |                         or time_since_last_progress >= progress_time_interval  # Every 5 seconds
```

- [ ] Add test to catch mutations on this line

### Line 1042

```python
    1041 |                         output_sink_name=output_sink_name,
>>> 1042 |                         run_id=run_id,
    1043 |                         checkpoint=False,  # Checkpointing now happens after sink write
```

- [ ] Add test to catch mutations on this line

### Line 1044

```python
    1043 |                         checkpoint=False,  # Checkpointing now happens after sink write
>>> 1044 |                         last_node_id=default_last_node_id,
    1045 |                     )
```

- [ ] Add test to catch mutations on this line

### Line 1045 (2 mutants)

```python
    1044 |                         last_node_id=default_last_node_id,
>>> 1045 |                     )
    1046 |                     rows_succeeded += agg_succeeded
```

- [ ] Add test to catch mutations on this line

### Line 1046 (2 mutants)

```python
    1045 |                     )
>>> 1046 |                     rows_succeeded += agg_succeeded
    1047 |                     rows_failed += agg_failed
```

- [ ] Add test to catch mutations on this line

### Line 1051 (2 mutants)

```python
    1050 |                 if coalesce_executor is not None:
>>> 1051 |                     # Step for coalesce flush = after all transforms and gates
    1052 |                     flush_step = len(config.transforms) + len(config.gates)
```

- [ ] Add test to catch mutations on this line

### Line 1058

```python
    1057 |                         if outcome.merged_token is not None:
>>> 1058 |                             # Successful merge - route to output sink
    1059 |                             rows_coalesced += 1
```

- [ ] Add test to catch mutations on this line

### Line 1070

```python
    1069 |                 sink_executor = SinkExecutor(recorder, self._span_factory, run_id)
>>> 1070 |                 # Step = transforms + config gates + 1 (for sink)
    1071 |                 step = len(config.transforms) + len(config.gates) + 1
```

- [ ] Add test to catch mutations on this line

### Line 1084

```python
    1083 |
>>> 1084 |                 for sink_name, tokens in pending_tokens.items():
    1085 |                     if tokens and sink_name in config.sinks:
```

- [ ] Add test to catch mutations on this line

### Line 1099

```python
    1098 |                 # (RunCompleted will show final summary regardless, but progress shows intermediate state)
>>> 1099 |                 current_time = time.perf_counter()
    1100 |                 time_since_last_progress = current_time - last_progress_time
```

- [ ] Add test to catch mutations on this line

### Line 1101 (4 mutants)

```python
    1100 |                 time_since_last_progress = current_time - last_progress_time
>>> 1101 |                 # Emit if: not on progress_interval boundary OR >1s since last emission
    1102 |                 if rows_processed % progress_interval != 0 or time_since_last_progress >= 1.0:
```

- [ ] Add test to catch mutations on this line

### Line 1102

```python
    1101 |                 # Emit if: not on progress_interval boundary OR >1s since last emission
>>> 1102 |                 if rows_processed % progress_interval != 0 or time_since_last_progress >= 1.0:
    1103 |                     elapsed = current_time - start_time
```

- [ ] Add test to catch mutations on this line

### Line 1115

```python
    1114 |
>>> 1115 |                 # PROCESS phase completed successfully
    1116 |                 self._events.emit(PhaseCompleted(phase=PipelinePhase.PROCESS, duration_seconds=time.perf_counter() - phase_start))
```

- [ ] Add test to catch mutations on this line

### Line 1187 (2 mutants)

```python
    1186 |
>>> 1187 |         # Get signing key from environment if signing enabled
    1188 |         signing_key: bytes | None = None
```

- [ ] Add test to catch mutations on this line

### Line 1192

```python
    1191 |                 key_str = os.environ["ELSPETH_SIGNING_KEY"]
>>> 1192 |             except KeyError:
    1193 |                 raise ValueError("ELSPETH_SIGNING_KEY environment variable required for signed export") from None
```

- [ ] Add test to catch mutations on this line

### Line 1201

```python
    1200 |         sink_name = export_config.sink
>>> 1201 |         if sink_name not in sinks:
    1202 |             raise ValueError(f"Export sink '{sink_name}' not found in sinks")
```

- [ ] Add test to catch mutations on this line

### Line 1205

```python
    1204 |
>>> 1205 |         # Create context for sink writes
    1206 |         ctx = PluginContext(run_id=run_id, config={}, landscape=None)
```

- [ ] Add test to catch mutations on this line

### Line 1207

```python
    1206 |         ctx = PluginContext(run_id=run_id, config={}, landscape=None)
>>> 1207 |
    1208 |         if export_config.format == "csv":
```

- [ ] Add test to catch mutations on this line

### Line 1211 (2 mutants)

```python
    1210 |             # CSV export writes files directly (not via sink.write), so we need
>>> 1211 |             # the path from sink config. CSV format requires file-based sink.
    1212 |             if "path" not in sink.config:
```

- [ ] Add test to catch mutations on this line

### Line 1213

```python
    1212 |             if "path" not in sink.config:
>>> 1213 |                 raise ValueError(
    1214 |                     f"CSV export requires file-based sink with 'path' in config, but sink '{sink_name}' has no path configured"
```

- [ ] Add test to catch mutations on this line

### Line 1215 (2 mutants)

```python
    1214 |                     f"CSV export requires file-based sink with 'path' in config, but sink '{sink_name}' has no path configured"
>>> 1215 |                 )
    1216 |             artifact_path: str = sink.config["path"]
```

- [ ] Add test to catch mutations on this line

### Line 1257

```python
    1256 |         from elspeth.core.landscape.formatters import CSVFormatter
>>> 1257 |
    1258 |         export_dir = Path(artifact_path)
```

- [ ] Add test to catch mutations on this line

### Line 1260 (2 mutants)

```python
    1259 |         if export_dir.suffix:
>>> 1260 |             # Remove file extension if present, treat as directory
    1261 |             export_dir = export_dir.with_suffix("")
```

- [ ] Add test to catch mutations on this line

### Line 1262 (2 mutants)

```python
    1261 |             export_dir = export_dir.with_suffix("")
>>> 1262 |
    1263 |         export_dir.mkdir(parents=True, exist_ok=True)
```

- [ ] Add test to catch mutations on this line

### Line 1265

```python
    1264 |
>>> 1265 |         # Get records grouped by type
    1266 |         grouped = exporter.export_run_grouped(run_id, sign=sign)
```

- [ ] Add test to catch mutations on this line

### Line 1266

```python
    1265 |         # Get records grouped by type
>>> 1266 |         grouped = exporter.export_run_grouped(run_id, sign=sign)
    1267 |         formatter = CSVFormatter()
```

- [ ] Add test to catch mutations on this line

### Line 1270

```python
    1269 |         # Write each record type to its own CSV file
>>> 1270 |         for record_type, records in grouped.items():
    1271 |             if not records:
```

- [ ] Add test to catch mutations on this line

### Line 1271

```python
    1270 |         for record_type, records in grouped.items():
>>> 1271 |             if not records:
    1272 |                 continue
```

- [ ] Add test to catch mutations on this line

### Line 1273 (3 mutants)

```python
    1272 |                 continue
>>> 1273 |
    1274 |             csv_path = export_dir / f"{record_type}.csv"
```

- [ ] Add test to catch mutations on this line

### Line 1276

```python
    1275 |
>>> 1276 |             # Flatten all records for CSV
    1277 |             flat_records = [formatter.format(r) for r in records]
```

- [ ] Add test to catch mutations on this line

### Line 1279

```python
    1278 |
>>> 1279 |             # Get union of all keys (some records may have optional fields)
    1280 |             all_keys: set[str] = set()
```

- [ ] Add test to catch mutations on this line

### Line 1282

```python
    1281 |             for rec in flat_records:
>>> 1282 |                 all_keys.update(rec.keys())
    1283 |             fieldnames = sorted(all_keys)  # Sorted for determinism
```

- [ ] Add test to catch mutations on this line

### Line 1284 (3 mutants)

```python
    1283 |             fieldnames = sorted(all_keys)  # Sorted for determinism
>>> 1284 |
    1285 |             with open(csv_path, "w", newline="", encoding="utf-8") as f:
```

- [ ] Add test to catch mutations on this line

### Line 1285

```python
    1284 |
>>> 1285 |             with open(csv_path, "w", newline="", encoding="utf-8") as f:
    1286 |                 writer = csv.DictWriter(f, fieldnames=fieldnames)
```

- [ ] Add test to catch mutations on this line

### Line 1318

```python
    1317 |         """
>>> 1318 |         if payload_store is None:
    1319 |             raise ValueError("payload_store is required for resume - row data must be retrieved from stored payloads")
```

- [ ] Add test to catch mutations on this line

### Line 1334

```python
    1333 |         # 3. Build restored aggregation state map
>>> 1334 |         restored_state: dict[str, dict[str, Any]] = {}
    1335 |         if resume_point.aggregation_state is not None:
```

- [ ] Add test to catch mutations on this line

### Line 1335

```python
    1334 |         restored_state: dict[str, dict[str, Any]] = {}
>>> 1335 |         if resume_point.aggregation_state is not None:
    1336 |             restored_state[resume_point.node_id] = resume_point.aggregation_state
```

- [ ] Add test to catch mutations on this line

### Line 1341

```python
    1340 |
>>> 1341 |         if self._checkpoint_manager is None:
    1342 |             raise ValueError("CheckpointManager is required for resume - Orchestrator must be initialized with checkpoint_manager")
```

- [ ] Add test to catch mutations on this line

### Line 1346 (2 mutants)

```python
    1345 |         # TYPE FIDELITY: Pass source schema to restore coerced types (datetime, Decimal, etc.)
>>> 1346 |         # The source's _schema_class attribute contains the Pydantic model with allow_coercion=True
    1347 |         source_schema_class = getattr(config.source, "_schema_class", None)
```

- [ ] Add test to catch mutations on this line

### Line 1355

```python
    1354 |                 run_id=run_id,
>>> 1355 |                 status=RunStatus.COMPLETED,
    1356 |                 rows_processed=0,
```

- [ ] Add test to catch mutations on this line

### Line 1356

```python
    1355 |                 status=RunStatus.COMPLETED,
>>> 1356 |                 rows_processed=0,
    1357 |                 rows_succeeded=0,
```

- [ ] Add test to catch mutations on this line

### Line 1357

```python
    1356 |                 rows_processed=0,
>>> 1357 |                 rows_succeeded=0,
    1358 |                 rows_failed=0,
```

- [ ] Add test to catch mutations on this line

### Line 1358

```python
    1357 |                 rows_succeeded=0,
>>> 1358 |                 rows_failed=0,
    1359 |                 rows_routed=0,
```

- [ ] Add test to catch mutations on this line

### Line 1418

```python
    1417 |         source_id = graph.get_source()
>>> 1418 |         if source_id is None:
    1419 |             raise ValueError("Graph has no source node")
```

- [ ] Add test to catch mutations on this line

### Line 1421

```python
    1420 |         sink_id_map = graph.get_sink_id_map()
>>> 1421 |         transform_id_map = graph.get_transform_id_map()
    1422 |         config_gate_id_map = graph.get_config_gate_id_map()
```

- [ ] Add test to catch mutations on this line

### Line 1422

```python
    1421 |         transform_id_map = graph.get_transform_id_map()
>>> 1422 |         config_gate_id_map = graph.get_config_gate_id_map()
    1423 |         coalesce_id_map = graph.get_coalesce_id_map()
```

- [ ] Add test to catch mutations on this line

### Line 1429 (2 mutants)

```python
    1428 |         for i, edge_info in enumerate(graph.get_edges()):
>>> 1429 |             # Generate synthetic edge_id for resume (edges were registered in original run)
    1430 |             edge_id = f"resume_edge_{i}"
```

- [ ] Add test to catch mutations on this line

### Line 1430

```python
    1429 |             # Generate synthetic edge_id for resume (edges were registered in original run)
>>> 1430 |             edge_id = f"resume_edge_{i}"
    1431 |             edge_map[(edge_info.from_node, edge_info.label)] = edge_id
```

- [ ] Add test to catch mutations on this line

### Line 1488

```python
    1487 |
>>> 1488 |         # Create retry manager from settings if available
    1489 |         retry_manager: RetryManager | None = None
```

- [ ] Add test to catch mutations on this line

### Line 1490

```python
    1489 |         retry_manager: RetryManager | None = None
>>> 1490 |         if settings is not None:
    1491 |             retry_manager = RetryManager(RetryConfig.from_settings(settings.retry))
```

- [ ] Add test to catch mutations on this line

### Line 1496

```python
    1495 |         from elspeth.engine.tokens import TokenManager
>>> 1496 |
    1497 |         coalesce_executor: CoalesceExecutor | None = None
```

- [ ] Add test to catch mutations on this line

### Line 1497

```python
    1496 |
>>> 1497 |         coalesce_executor: CoalesceExecutor | None = None
    1498 |         branch_to_coalesce: dict[str, str] = {}
```

- [ ] Add test to catch mutations on this line

### Line 1500

```python
    1499 |
>>> 1500 |         if settings is not None and settings.coalesce:
    1501 |             branch_to_coalesce = graph.get_branch_to_coalesce_map()
```

- [ ] Add test to catch mutations on this line

### Line 1501

```python
    1500 |         if settings is not None and settings.coalesce:
>>> 1501 |             branch_to_coalesce = graph.get_branch_to_coalesce_map()
    1502 |             token_manager = TokenManager(recorder)
```

- [ ] Add test to catch mutations on this line

### Line 1508

```python
    1507 |                 token_manager=token_manager,
>>> 1508 |                 run_id=run_id,
    1509 |             )
```

- [ ] Add test to catch mutations on this line

### Line 1511

```python
    1510 |
>>> 1511 |             for coalesce_settings in settings.coalesce:
    1512 |                 coalesce_node_id = coalesce_id_map[coalesce_settings.name]
```

- [ ] Add test to catch mutations on this line

### Line 1515

```python
    1514 |
>>> 1515 |         # Compute coalesce step positions
    1516 |         coalesce_step_map: dict[str, int] = {}
```

- [ ] Add test to catch mutations on this line

### Line 1517 (2 mutants)

```python
    1516 |         coalesce_step_map: dict[str, int] = {}
>>> 1517 |         if settings is not None and settings.coalesce:
    1518 |             base_step = len(config.transforms) + len(config.gates)
```

- [ ] Add test to catch mutations on this line

### Line 1519 (2 mutants)

```python
    1518 |             base_step = len(config.transforms) + len(config.gates)
>>> 1519 |             for i, cs in enumerate(settings.coalesce):
    1520 |                 coalesce_step_map[cs.name] = base_step + i
```

- [ ] Add test to catch mutations on this line

### Line 1548 (2 mutants)

```python
    1547 |         rows_succeeded = 0
>>> 1548 |         rows_failed = 0
    1549 |         rows_routed = 0
```

- [ ] Add test to catch mutations on this line

### Line 1549 (2 mutants)

```python
    1548 |         rows_failed = 0
>>> 1549 |         rows_routed = 0
    1550 |         rows_quarantined = 0
```

- [ ] Add test to catch mutations on this line

### Line 1550 (2 mutants)

```python
    1549 |         rows_routed = 0
>>> 1550 |         rows_quarantined = 0
    1551 |         rows_forked = 0
```

- [ ] Add test to catch mutations on this line

### Line 1551 (2 mutants)

```python
    1550 |         rows_quarantined = 0
>>> 1551 |         rows_forked = 0
    1552 |         rows_coalesced = 0
```

- [ ] Add test to catch mutations on this line

### Line 1552 (2 mutants)

```python
    1551 |         rows_forked = 0
>>> 1552 |         rows_coalesced = 0
    1553 |         rows_expanded = 0
```

- [ ] Add test to catch mutations on this line

### Line 1553 (2 mutants)

```python
    1552 |         rows_coalesced = 0
>>> 1553 |         rows_expanded = 0
    1554 |         rows_buffered = 0
```

- [ ] Add test to catch mutations on this line

### Line 1583 (3 mutants)

```python
    1582 |                         rows_succeeded += 1
>>> 1583 |                         sink_name = output_sink_name
    1584 |                         if result.token.branch_name is not None and result.token.branch_name in config.sinks:
```

- [ ] Add test to catch mutations on this line

### Line 1584

```python
    1583 |                         sink_name = output_sink_name
>>> 1584 |                         if result.token.branch_name is not None and result.token.branch_name in config.sinks:
    1585 |                             sink_name = result.token.branch_name
```

- [ ] Add test to catch mutations on this line

### Line 1586

```python
    1585 |                             sink_name = result.token.branch_name
>>> 1586 |                         pending_tokens[sink_name].append(result.token)
    1587 |                     elif result.outcome == RowOutcome.ROUTED:
```

- [ ] Add test to catch mutations on this line

### Line 1587 (3 mutants)

```python
    1586 |                         pending_tokens[sink_name].append(result.token)
>>> 1587 |                     elif result.outcome == RowOutcome.ROUTED:
    1588 |                         rows_routed += 1
```

- [ ] Add test to catch mutations on this line

### Line 1588

```python
    1587 |                     elif result.outcome == RowOutcome.ROUTED:
>>> 1588 |                         rows_routed += 1
    1589 |                         assert result.sink_name is not None
```

- [ ] Add test to catch mutations on this line

### Line 1590

```python
    1589 |                         assert result.sink_name is not None
>>> 1590 |                         pending_tokens[result.sink_name].append(result.token)
    1591 |                     elif result.outcome == RowOutcome.FAILED:
```

- [ ] Add test to catch mutations on this line

### Line 1591 (3 mutants)

```python
    1590 |                         pending_tokens[result.sink_name].append(result.token)
>>> 1591 |                     elif result.outcome == RowOutcome.FAILED:
    1592 |                         rows_failed += 1
```

- [ ] Add test to catch mutations on this line

### Line 1592

```python
    1591 |                     elif result.outcome == RowOutcome.FAILED:
>>> 1592 |                         rows_failed += 1
    1593 |                     elif result.outcome == RowOutcome.QUARANTINED:
```

- [ ] Add test to catch mutations on this line

### Line 1593 (3 mutants)

```python
    1592 |                         rows_failed += 1
>>> 1593 |                     elif result.outcome == RowOutcome.QUARANTINED:
    1594 |                         rows_quarantined += 1
```

- [ ] Add test to catch mutations on this line

### Line 1594

```python
    1593 |                     elif result.outcome == RowOutcome.QUARANTINED:
>>> 1594 |                         rows_quarantined += 1
    1595 |                     elif result.outcome == RowOutcome.FORKED:
```

- [ ] Add test to catch mutations on this line

### Line 1595 (3 mutants)

```python
    1594 |                         rows_quarantined += 1
>>> 1595 |                     elif result.outcome == RowOutcome.FORKED:
    1596 |                         rows_forked += 1
```

- [ ] Add test to catch mutations on this line

### Line 1596

```python
    1595 |                     elif result.outcome == RowOutcome.FORKED:
>>> 1596 |                         rows_forked += 1
    1597 |                     elif result.outcome == RowOutcome.CONSUMED_IN_BATCH:
```

- [ ] Add test to catch mutations on this line

### Line 1598

```python
    1597 |                     elif result.outcome == RowOutcome.CONSUMED_IN_BATCH:
>>> 1598 |                         pass
    1599 |                     elif result.outcome == RowOutcome.COALESCED:
```

- [ ] Add test to catch mutations on this line

### Line 1599 (3 mutants)

```python
    1598 |                         pass
>>> 1599 |                     elif result.outcome == RowOutcome.COALESCED:
    1600 |                         rows_coalesced += 1
```

- [ ] Add test to catch mutations on this line

### Line 1601

```python
    1600 |                         rows_coalesced += 1
>>> 1601 |                         pending_tokens[output_sink_name].append(result.token)
    1602 |                     elif result.outcome == RowOutcome.EXPANDED:
```

- [ ] Add test to catch mutations on this line

### Line 1602 (3 mutants)

```python
    1601 |                         pending_tokens[output_sink_name].append(result.token)
>>> 1602 |                     elif result.outcome == RowOutcome.EXPANDED:
    1603 |                         rows_expanded += 1
```

- [ ] Add test to catch mutations on this line

### Line 1603

```python
    1602 |                     elif result.outcome == RowOutcome.EXPANDED:
>>> 1603 |                         rows_expanded += 1
    1604 |                     elif result.outcome == RowOutcome.BUFFERED:
```

- [ ] Add test to catch mutations on this line

### Line 1604 (3 mutants)

```python
    1603 |                         rows_expanded += 1
>>> 1604 |                     elif result.outcome == RowOutcome.BUFFERED:
    1605 |                         rows_buffered += 1
```

- [ ] Add test to catch mutations on this line

### Line 1617

```python
    1616 |                     output_sink_name=output_sink_name,
>>> 1617 |                     run_id=run_id,
    1618 |                     checkpoint=False,  # No checkpointing during resume
```

- [ ] Add test to catch mutations on this line

### Line 1618

```python
    1617 |                     run_id=run_id,
>>> 1618 |                     checkpoint=False,  # No checkpointing during resume
    1619 |                 )
```

- [ ] Add test to catch mutations on this line

### Line 1619 (2 mutants)

```python
    1618 |                     checkpoint=False,  # No checkpointing during resume
>>> 1619 |                 )
    1620 |                 rows_succeeded += agg_succeeded
```

- [ ] Add test to catch mutations on this line

### Line 1620 (2 mutants)

```python
    1619 |                 )
>>> 1620 |                 rows_succeeded += agg_succeeded
    1621 |                 rows_failed += agg_failed
```

- [ ] Add test to catch mutations on this line

### Line 1624 (2 mutants)

```python
    1623 |             # Flush pending coalesce operations
>>> 1624 |             if coalesce_executor is not None:
    1625 |                 flush_step = len(config.transforms) + len(config.gates)
```

- [ ] Add test to catch mutations on this line

### Line 1625

```python
    1624 |             if coalesce_executor is not None:
>>> 1625 |                 flush_step = len(config.transforms) + len(config.gates)
    1626 |                 pending_outcomes = coalesce_executor.flush_pending(flush_step)
```

- [ ] Add test to catch mutations on this line

### Line 1628

```python
    1627 |
>>> 1628 |                 for outcome in pending_outcomes:
    1629 |                     if outcome.merged_token is not None:
```

- [ ] Add test to catch mutations on this line

### Line 1629 (3 mutants)

```python
    1628 |                 for outcome in pending_outcomes:
>>> 1629 |                     if outcome.merged_token is not None:
    1630 |                         rows_coalesced += 1
```

- [ ] Add test to catch mutations on this line

### Line 1634 (3 mutants)

```python
    1633 |             # Write to sinks using SinkExecutor
>>> 1634 |             sink_executor = SinkExecutor(recorder, self._span_factory, run_id)
    1635 |             step = len(config.transforms) + len(config.gates) + 1
```

- [ ] Add test to catch mutations on this line

### Line 1637

```python
    1636 |
>>> 1637 |             for sink_name, tokens in pending_tokens.items():
    1638 |                 if tokens and sink_name in config.sinks:
```

- [ ] Add test to catch mutations on this line

### Line 1697

```python
    1696 |                 recorder.update_batch_status(batch.batch_id, "failed")
>>> 1697 |                 recorder.retry_batch(batch.batch_id)
    1698 |             elif batch.status == BatchStatus.FAILED:
```

- [ ] Add test to catch mutations on this line

### Line 1720

```python
    1719 |
>>> 1720 |         with self._db.connection() as conn:
    1721 |             conn.execute(runs_table.update().where(runs_table.c.run_id == run_id).values(status=status.value))
```

- [ ] Add test to catch mutations on this line

### Line 1730

```python
    1729 |         output_sink_name: str,
>>> 1730 |         run_id: str,
    1731 |         checkpoint: bool = True,
```

- [ ] Add test to catch mutations on this line

### Line 1759 (2 mutants)

```python
    1758 |         from elspeth.contracts.enums import TriggerType
>>> 1759 |
    1760 |         rows_succeeded = 0
```

- [ ] Add test to catch mutations on this line

### Line 1760 (2 mutants)

```python
    1759 |
>>> 1760 |         rows_succeeded = 0
    1761 |         rows_failed = 0
```

- [ ] Add test to catch mutations on this line

### Line 1765

```python
    1764 |             # aggregation_settings is keyed by node_id (set in cli.py)
>>> 1765 |             # The aggregation name is available via agg_settings.name
    1766 |             agg_name = agg_settings.name
```

- [ ] Add test to catch mutations on this line

### Line 1768

```python
    1767 |
>>> 1768 |             # Check if there are buffered rows
    1769 |             buffered_count = processor._aggregation_executor.get_buffer_count(agg_node_id)
```

- [ ] Add test to catch mutations on this line

### Line 1769 (2 mutants)

```python
    1768 |             # Check if there are buffered rows
>>> 1769 |             buffered_count = processor._aggregation_executor.get_buffer_count(agg_node_id)
    1770 |             if buffered_count == 0:
```

- [ ] Add test to catch mutations on this line

### Line 1770

```python
    1769 |             buffered_count = processor._aggregation_executor.get_buffer_count(agg_node_id)
>>> 1770 |             if buffered_count == 0:
    1771 |                 continue
```

- [ ] Add test to catch mutations on this line

### Line 1774 (2 mutants)

```python
    1773 |             # Find the batch-aware transform for this aggregation
>>> 1774 |             # Only BaseTransform can have is_batch_aware (gates cannot)
    1775 |             agg_transform: BaseTransform | None = None
```

- [ ] Add test to catch mutations on this line

### Line 1776 (2 mutants)

```python
    1775 |             agg_transform: BaseTransform | None = None
>>> 1776 |             for t in config.transforms:
    1777 |                 if isinstance(t, BaseTransform) and t.node_id == agg_node_id and t.is_batch_aware:
```

- [ ] Add test to catch mutations on this line

### Line 1777

```python
    1776 |             for t in config.transforms:
>>> 1777 |                 if isinstance(t, BaseTransform) and t.node_id == agg_node_id and t.is_batch_aware:
    1778 |                     agg_transform = t
```

- [ ] Add test to catch mutations on this line

### Line 1778

```python
    1777 |                 if isinstance(t, BaseTransform) and t.node_id == agg_node_id and t.is_batch_aware:
>>> 1778 |                     agg_transform = t
    1779 |                     break
```

- [ ] Add test to catch mutations on this line

### Line 1780

```python
    1779 |                     break
>>> 1780 |
    1781 |             if agg_transform is None:
```

- [ ] Add test to catch mutations on this line

### Line 1782

```python
    1781 |             if agg_transform is None:
>>> 1782 |                 raise RuntimeError(
    1783 |                     f"No batch-aware transform found for aggregation '{agg_name}' "
```

- [ ] Add test to catch mutations on this line

### Line 1783

```python
    1782 |                 raise RuntimeError(
>>> 1783 |                     f"No batch-aware transform found for aggregation '{agg_name}' "
    1784 |                     f"(node_id={agg_node_id}). This indicates a bug in graph construction "
```

- [ ] Add test to catch mutations on this line

### Line 1784

```python
    1783 |                     f"No batch-aware transform found for aggregation '{agg_name}' "
>>> 1784 |                     f"(node_id={agg_node_id}). This indicates a bug in graph construction "
    1785 |                     f"or pipeline configuration."
```

- [ ] Add test to catch mutations on this line

### Line 1789

```python
    1788 |             # Compute step_in_pipeline for this aggregation
>>> 1789 |             agg_step = next(
    1790 |                 (i for i, t in enumerate(config.transforms) if t.node_id == agg_node_id),
```

- [ ] Add test to catch mutations on this line

### Line 1791

```python
    1790 |                 (i for i, t in enumerate(config.transforms) if t.node_id == agg_node_id),
>>> 1791 |                 len(config.transforms),
    1792 |             )
```

- [ ] Add test to catch mutations on this line

### Line 1800

```python
    1799 |                 step_in_pipeline=agg_step,
>>> 1800 |                 trigger_type=TriggerType.END_OF_SOURCE,
    1801 |             )
```

- [ ] Add test to catch mutations on this line

### Line 1803 (2 mutants)

```python
    1802 |
>>> 1803 |             # Handle the flushed batch result
    1804 |             if flush_result.status == "success":
```

- [ ] Add test to catch mutations on this line

### Line 1804 (2 mutants)

```python
    1803 |             # Handle the flushed batch result
>>> 1804 |             if flush_result.status == "success":
    1805 |                 if flush_result.row is not None and buffered_tokens:
```

- [ ] Add test to catch mutations on this line

### Line 1807

```python
    1806 |                     # Single row output - reuse first buffered token's metadata
>>> 1807 |                     output_token = TokenInfo(
    1808 |                         token_id=buffered_tokens[0].token_id,
```

- [ ] Add test to catch mutations on this line

### Line 1808

```python
    1807 |                     output_token = TokenInfo(
>>> 1808 |                         token_id=buffered_tokens[0].token_id,
    1809 |                         row_id=buffered_tokens[0].row_id,
```

- [ ] Add test to catch mutations on this line

### Line 1810

```python
    1809 |                         row_id=buffered_tokens[0].row_id,
>>> 1810 |                         row_data=flush_result.row,
    1811 |                         branch_name=buffered_tokens[0].branch_name,
```

- [ ] Add test to catch mutations on this line

### Line 1811

```python
    1810 |                         row_data=flush_result.row,
>>> 1811 |                         branch_name=buffered_tokens[0].branch_name,
    1812 |                     )
```

- [ ] Add test to catch mutations on this line

### Line 1813 (3 mutants)

```python
    1812 |                     )
>>> 1813 |                     pending_tokens[output_sink_name].append(output_token)
    1814 |                     rows_succeeded += 1
```

- [ ] Add test to catch mutations on this line

### Line 1816 (2 mutants)

```python
    1815 |
>>> 1816 |                     # Checkpoint the flushed aggregation token
    1817 |                     if checkpoint and last_node_id is not None:
```

- [ ] Add test to catch mutations on this line

### Line 1822 (2 mutants)

```python
    1821 |                             node_id=last_node_id,
>>> 1822 |                         )
    1823 |                 elif flush_result.rows is not None and buffered_tokens:
```

- [ ] Add test to catch mutations on this line

### Line 1825

```python
    1824 |                     # Multiple row output - use expand_token for proper audit
>>> 1825 |                     expanded = processor.token_manager.expand_token(
    1826 |                         parent_token=buffered_tokens[0],
```

- [ ] Add test to catch mutations on this line

### Line 1828

```python
    1827 |                         expanded_rows=flush_result.rows,
>>> 1828 |                         step_in_pipeline=agg_step,
    1829 |                     )
```

- [ ] Add test to catch mutations on this line

### Line 1831 (3 mutants)

```python
    1830 |                     for exp_token in expanded:
>>> 1831 |                         pending_tokens[output_sink_name].append(exp_token)
    1832 |                         rows_succeeded += 1
```

- [ ] Add test to catch mutations on this line

### Line 1834 (2 mutants)

```python
    1833 |
>>> 1834 |                         # Checkpoint each expanded token
    1835 |                         if checkpoint and last_node_id is not None:
```

- [ ] Add test to catch mutations on this line

### Line 1842 (2 mutants)

```python
    1841 |             else:
>>> 1842 |                 # Flush failed
    1843 |                 rows_failed += len(buffered_tokens)
```

- [ ] Add test to catch mutations on this line

## src/elspeth/engine/processor.py

**Priority:** P0
**Survivors:** 62 unique lines (88 total mutants)

### Line 38

```python
      37 |
>>>   38 | # Iteration guard to prevent infinite loops from bugs
      39 | MAX_WORK_QUEUE_ITERATIONS = 10_000
```

- [ ] Add test to catch mutations on this line

### Line 47 (2 mutants)

```python
      46 |     token: TokenInfo
>>>   47 |     start_step: int  # Which step in transforms to start from (0-indexed)
      48 |     coalesce_at_step: int | None = None  # Step at which to coalesce (if any)
```

- [ ] Add test to catch mutations on this line

### Line 48 (2 mutants)

```python
      47 |     start_step: int  # Which step in transforms to start from (0-indexed)
>>>   48 |     coalesce_at_step: int | None = None  # Step at which to coalesce (if any)
      49 |     coalesce_name: str | None = None  # Name of the coalesce point (if any)
```

- [ ] Add test to catch mutations on this line

### Line 129 (2 mutants)

```python
     128 |         self._retry_manager = retry_manager
>>>  129 |         self._coalesce_executor = coalesce_executor
     130 |         self._coalesce_node_ids = coalesce_node_ids or {}
```

- [ ] Add test to catch mutations on this line

### Line 190

```python
     189 |         if self._aggregation_executor.should_flush(node_id):
>>>  190 |             # Determine trigger type
     191 |             trigger_type = self._aggregation_executor.get_trigger_type(node_id)
```

- [ ] Add test to catch mutations on this line

### Line 191

```python
     190 |             # Determine trigger type
>>>  191 |             trigger_type = self._aggregation_executor.get_trigger_type(node_id)
     192 |             if trigger_type is None:
```

- [ ] Add test to catch mutations on this line

### Line 192

```python
     191 |             trigger_type = self._aggregation_executor.get_trigger_type(node_id)
>>>  192 |             if trigger_type is None:
     193 |                 trigger_type = TriggerType.COUNT  # Default if no evaluator
```

- [ ] Add test to catch mutations on this line

### Line 204 (2 mutants)

```python
     203 |
>>>  204 |             if result.status != "success":
     205 |                 error_msg = "Batch transform failed"
```

- [ ] Add test to catch mutations on this line

### Line 205 (2 mutants)

```python
     204 |             if result.status != "success":
>>>  205 |                 error_msg = "Batch transform failed"
     206 |                 error_hash = hashlib.sha256(error_msg.encode()).hexdigest()[:16]
```

- [ ] Add test to catch mutations on this line

### Line 218

```python
     217 |                         outcome=RowOutcome.FAILED,
>>>  218 |                         error=FailureInfo(
     219 |                             exception_type="TransformError",
```

- [ ] Add test to catch mutations on this line

### Line 254

```python
     253 |                 if not result.is_multi_row:
>>>  254 |                     raise ValueError(
     255 |                         f"Passthrough mode requires multi-row result, "
```

- [ ] Add test to catch mutations on this line

### Line 255

```python
     254 |                     raise ValueError(
>>>  255 |                         f"Passthrough mode requires multi-row result, "
     256 |                         f"but transform '{transform.name}' returned single row. "
```

- [ ] Add test to catch mutations on this line

### Line 256

```python
     255 |                         f"Passthrough mode requires multi-row result, "
>>>  256 |                         f"but transform '{transform.name}' returned single row. "
     257 |                         f"Use TransformResult.success_multi() for passthrough."
```

- [ ] Add test to catch mutations on this line

### Line 263

```python
     262 |                 if len(result.rows) != len(buffered_tokens):
>>>  263 |                     raise ValueError(
     264 |                         f"Passthrough mode requires same number of output rows "
```

- [ ] Add test to catch mutations on this line

### Line 264

```python
     263 |                     raise ValueError(
>>>  264 |                         f"Passthrough mode requires same number of output rows "
     265 |                         f"as input rows. Transform '{transform.name}' returned "
```

- [ ] Add test to catch mutations on this line

### Line 265

```python
     264 |                         f"Passthrough mode requires same number of output rows "
>>>  265 |                         f"as input rows. Transform '{transform.name}' returned "
     266 |                         f"{len(result.rows)} rows but received {len(buffered_tokens)} input rows."
```

- [ ] Add test to catch mutations on this line

### Line 270

```python
     269 |                 # Build COMPLETED results for all buffered tokens with enriched data
>>>  270 |                 # Check if there are more transforms after this one
     271 |                 more_transforms = step < total_steps
```

- [ ] Add test to catch mutations on this line

### Line 274

```python
     273 |                 if more_transforms:
>>>  274 |                     # Queue enriched tokens as work items for remaining transforms
     275 |                     for token, enriched_data in zip(buffered_tokens, result.rows, strict=True):
```

- [ ] Add test to catch mutations on this line

### Line 292

```python
     291 |                     # No more transforms - return COMPLETED for all tokens
>>>  292 |                     results: list[RowResult] = []
     293 |                     for token, enriched_data in zip(buffered_tokens, result.rows, strict=True):
```

- [ ] Add test to catch mutations on this line

### Line 337

```python
     336 |
>>>  337 |                 # The triggering token becomes CONSUMED_IN_BATCH
     338 |                 batch_id = self._aggregation_executor.get_batch_id(node_id)
```

- [ ] Add test to catch mutations on this line

### Line 351

```python
     350 |
>>>  351 |                 # Check if there are more transforms after this one
     352 |                 more_transforms = step < total_steps
```

- [ ] Add test to catch mutations on this line

### Line 384

```python
     383 |
>>>  384 |             else:
     385 |                 raise ValueError(f"Unknown output_mode: {output_mode}")
```

- [ ] Add test to catch mutations on this line

### Line 390

```python
     389 |         # In single/transform modes: CONSUMED_IN_BATCH (terminal)
>>>  390 |         if output_mode == "passthrough":
     391 |             buf_batch_id = self._aggregation_executor.get_batch_id(node_id)
```

- [ ] Add test to catch mutations on this line

### Line 406

```python
     405 |             )
>>>  406 |         else:
     407 |             nf_batch_id = self._aggregation_executor.get_batch_id(node_id)
```

- [ ] Add test to catch mutations on this line

### Line 456

```python
     455 |                 ctx=ctx,
>>>  456 |                 step_in_pipeline=step,
     457 |                 attempt=0,
```

- [ ] Add test to catch mutations on this line

### Line 530

```python
     529 |         )
>>>  530 |         results: list[RowResult] = []
     531 |         iterations = 0
```

- [ ] Add test to catch mutations on this line

### Line 534 (3 mutants)

```python
     533 |         with self._spans.row_span(token.row_id, token.token_id):
>>>  534 |             while work_queue:
     535 |                 iterations += 1
```

- [ ] Add test to catch mutations on this line

### Line 535

```python
     534 |             while work_queue:
>>>  535 |                 iterations += 1
     536 |                 if iterations > MAX_WORK_QUEUE_ITERATIONS:
```

- [ ] Add test to catch mutations on this line

### Line 536

```python
     535 |                 iterations += 1
>>>  536 |                 if iterations > MAX_WORK_QUEUE_ITERATIONS:
     537 |                     raise RuntimeError(f"Work queue exceeded {MAX_WORK_QUEUE_ITERATIONS} iterations. Possible infinite loop in pipeline.")
```

- [ ] Add test to catch mutations on this line

### Line 600

```python
     599 |                 _WorkItem(
>>>  600 |                     token=token,
     601 |                     start_step=0,
```

- [ ] Add test to catch mutations on this line

### Line 607

```python
     606 |         )
>>>  607 |         results: list[RowResult] = []
     608 |         iterations = 0
```

- [ ] Add test to catch mutations on this line

### Line 611 (3 mutants)

```python
     610 |         with self._spans.row_span(token.row_id, token.token_id):
>>>  611 |             while work_queue:
     612 |                 iterations += 1
```

- [ ] Add test to catch mutations on this line

### Line 612

```python
     611 |             while work_queue:
>>>  612 |                 iterations += 1
     613 |                 if iterations > MAX_WORK_QUEUE_ITERATIONS:
```

- [ ] Add test to catch mutations on this line

### Line 613

```python
     612 |                 iterations += 1
>>>  613 |                 if iterations > MAX_WORK_QUEUE_ITERATIONS:
     614 |                     raise RuntimeError(f"Work queue exceeded {MAX_WORK_QUEUE_ITERATIONS} iterations. Possible infinite loop in pipeline.")
```

- [ ] Add test to catch mutations on this line

### Line 676

```python
     675 |                     step_in_pipeline=step,
>>>  676 |                     token_manager=self._token_manager,
     677 |                 )
```

- [ ] Add test to catch mutations on this line

### Line 677

```python
     676 |                     token_manager=self._token_manager,
>>>  677 |                 )
     678 |                 current_token = outcome.updated_token
```

- [ ] Add test to catch mutations on this line

### Line 680

```python
     679 |
>>>  680 |                 # Check if gate routed to a sink (sink_name set by executor)
     681 |                 if outcome.sink_name is not None:
```

- [ ] Add test to catch mutations on this line

### Line 696

```python
     695 |                         child_items,
>>>  696 |                     )
     697 |                 elif outcome.result.action.kind == RoutingKind.FORK_TO_PATHS:
```

- [ ] Add test to catch mutations on this line

### Line 698 (4 mutants)

```python
     697 |                 elif outcome.result.action.kind == RoutingKind.FORK_TO_PATHS:
>>>  698 |                     # Parent becomes FORKED, children continue from NEXT step
     699 |                     next_step = start_step + step_offset + 1
```

- [ ] Add test to catch mutations on this line

### Line 701

```python
     700 |                     for child_token in outcome.child_tokens:
>>>  701 |                         # Look up coalesce info for this branch
     702 |                         branch_name = child_token.branch_name
```

- [ ] Add test to catch mutations on this line

### Line 702 (2 mutants)

```python
     701 |                         # Look up coalesce info for this branch
>>>  702 |                         branch_name = child_token.branch_name
     703 |                         child_coalesce_name: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 703 (2 mutants)

```python
     702 |                         branch_name = child_token.branch_name
>>>  703 |                         child_coalesce_name: str | None = None
     704 |                         child_coalesce_step: int | None = None
```

- [ ] Add test to catch mutations on this line

### Line 705 (2 mutants)

```python
     704 |                         child_coalesce_step: int | None = None
>>>  705 |
     706 |                         if branch_name and branch_name in self._branch_to_coalesce:
```

- [ ] Add test to catch mutations on this line

### Line 706

```python
     705 |
>>>  706 |                         if branch_name and branch_name in self._branch_to_coalesce:
     707 |                             child_coalesce_name = self._branch_to_coalesce[branch_name]
```

- [ ] Add test to catch mutations on this line

### Line 707

```python
     706 |                         if branch_name and branch_name in self._branch_to_coalesce:
>>>  707 |                             child_coalesce_name = self._branch_to_coalesce[branch_name]
     708 |                             child_coalesce_step = self._coalesce_step_map.get(child_coalesce_name)
```

- [ ] Add test to catch mutations on this line

### Line 719 (2 mutants)

```python
     718 |
>>>  719 |                     # Generate fork group ID linking parent to children
     720 |                     fork_group_id = uuid.uuid4().hex[:16]
```

- [ ] Add test to catch mutations on this line

### Line 763 (2 mutants)

```python
     762 |                 except MaxRetriesExceeded as e:
>>>  763 |                     # All retries exhausted - return FAILED outcome
     764 |                     error_hash = hashlib.sha256(str(e).encode()).hexdigest()[:16]
```

- [ ] Add test to catch mutations on this line

### Line 784

```python
     783 |                     if error_sink == "discard":
>>>  784 |                         # Intentionally discarded - QUARANTINED
     785 |                         error_detail = str(result.reason) if result.reason else "unknown_error"
```

- [ ] Add test to catch mutations on this line

### Line 785 (2 mutants)

```python
     784 |                         # Intentionally discarded - QUARANTINED
>>>  785 |                         error_detail = str(result.reason) if result.reason else "unknown_error"
     786 |                         quarantine_error_hash = hashlib.sha256(error_detail.encode()).hexdigest()[:16]
```

- [ ] Add test to catch mutations on this line

### Line 825

```python
     824 |                     if not transform.creates_tokens:
>>>  825 |                         raise RuntimeError(
     826 |                             f"Transform '{transform.name}' returned multi-row result "
```

- [ ] Add test to catch mutations on this line

### Line 826

```python
     825 |                         raise RuntimeError(
>>>  826 |                             f"Transform '{transform.name}' returned multi-row result "
     827 |                             f"but has creates_tokens=False. Either set creates_tokens=True "
```

- [ ] Add test to catch mutations on this line

### Line 827

```python
     826 |                             f"Transform '{transform.name}' returned multi-row result "
>>>  827 |                             f"but has creates_tokens=False. Either set creates_tokens=True "
     828 |                             f"or return single row via TransformResult.success(row). "
```

- [ ] Add test to catch mutations on this line

### Line 828

```python
     827 |                             f"but has creates_tokens=False. Either set creates_tokens=True "
>>>  828 |                             f"or return single row via TransformResult.success(row). "
     829 |                             f"(Multi-row is allowed in aggregation passthrough mode.)"
```

- [ ] Add test to catch mutations on this line

### Line 841 (2 mutants)

```python
     840 |                     # Queue each child for continued processing
>>>  841 |                     # Children start at next step (step_offset + 1 gives 0-indexed next)
     842 |                     next_step = start_step + step_offset + 1
```

- [ ] Add test to catch mutations on this line

### Line 853 (2 mutants)

```python
     852 |
>>>  853 |                     # Parent token is EXPANDED (terminal for parent)
     854 |                     expand_group_id = uuid.uuid4().hex[:16]
```

- [ ] Add test to catch mutations on this line

### Line 873

```python
     872 |
>>>  873 |             else:
     874 |                 raise TypeError(f"Unknown transform type: {type(transform).__name__}. Expected BaseTransform or BaseGate.")
```

- [ ] Add test to catch mutations on this line

### Line 884

```python
     883 |         config_gate_start_idx = max(0, start_step - len(transforms))
>>>  884 |         for gate_idx, gate_config in enumerate(self._config_gates[config_gate_start_idx:], start=config_gate_start_idx):
     885 |             step = config_gate_start_step + gate_idx
```

- [ ] Add test to catch mutations on this line

### Line 922 (2 mutants)

```python
     921 |                     # Look up coalesce info for this branch
>>>  922 |                     cfg_branch_name = child_token.branch_name
     923 |                     cfg_coalesce_name: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 923 (2 mutants)

```python
     922 |                     cfg_branch_name = child_token.branch_name
>>>  923 |                     cfg_coalesce_name: str | None = None
     924 |                     cfg_coalesce_step: int | None = None
```

- [ ] Add test to catch mutations on this line

### Line 940 (2 mutants)

```python
     939 |
>>>  940 |                 # Generate fork group ID linking parent to children
     941 |                 cfg_fork_group_id = uuid.uuid4().hex[:16]
```

- [ ] Add test to catch mutations on this line

### Line 967 (2 mutants)

```python
     966 |             if completed_step >= coalesce_at_step:
>>>  967 |                 # Coalesce operation is at the next step after the last transform
     968 |                 coalesce_step = completed_step + 1
```

- [ ] Add test to catch mutations on this line

### Line 983 (3 mutants)

```python
     982 |                     # All siblings arrived - return COALESCED with merged data
>>>  983 |                     # Use coalesce_name + parent token for join group identification
     984 |                     join_group_id = f"{coalesce_name}_{uuid.uuid4().hex[:8]}"
```

- [ ] Add test to catch mutations on this line

## src/elspeth/core/landscape/recorder.py

**Priority:** P1
**Survivors:** 64 unique lines (73 total mutants)

### Line 73 (2 mutants)

```python
      72 | )
>>>   73 |
      74 | E = TypeVar("E", bound=Enum)
```

- [ ] Add test to catch mutations on this line

### Line 140

```python
     139 |         # Validate required fields - None indicates audit integrity violation
>>>  140 |         if row.output_hash is None:
     141 |             raise ValueError(f"COMPLETED state {row.state_id} has NULL output_hash - audit integrity violation")
```

- [ ] Add test to catch mutations on this line

### Line 142

```python
     141 |             raise ValueError(f"COMPLETED state {row.state_id} has NULL output_hash - audit integrity violation")
>>>  142 |         if row.duration_ms is None:
     143 |             raise ValueError(f"COMPLETED state {row.state_id} has NULL duration_ms - audit integrity violation")
```

- [ ] Add test to catch mutations on this line

### Line 144

```python
     143 |             raise ValueError(f"COMPLETED state {row.state_id} has NULL duration_ms - audit integrity violation")
>>>  144 |         if row.completed_at is None:
     145 |             raise ValueError(f"COMPLETED state {row.state_id} has NULL completed_at - audit integrity violation")
```

- [ ] Add test to catch mutations on this line

### Line 164

```python
     163 |         # Validate required fields - None indicates audit integrity violation
>>>  164 |         if row.duration_ms is None:
     165 |             raise ValueError(f"FAILED state {row.state_id} has NULL duration_ms - audit integrity violation")
```

- [ ] Add test to catch mutations on this line

### Line 166

```python
     165 |             raise ValueError(f"FAILED state {row.state_id} has NULL duration_ms - audit integrity violation")
>>>  166 |         if row.completed_at is None:
     167 |             raise ValueError(f"FAILED state {row.state_id} has NULL completed_at - audit integrity violation")
```

- [ ] Add test to catch mutations on this line

### Line 309

```python
     308 |
>>>  309 |         result = self.get_run(run_id)
     310 |         assert result is not None, f"Run {run_id} not found after update"
```

- [ ] Add test to catch mutations on this line

### Line 418

```python
     417 |
>>>  418 |         if status_enum == ExportStatus.COMPLETED:
     419 |             updates["exported_at"] = _now()
```

- [ ] Add test to catch mutations on this line

### Line 429

```python
     428 |             updates["export_error"] = error
>>>  429 |
     430 |         if export_format is not None:
```

- [ ] Add test to catch mutations on this line

### Line 430

```python
     429 |
>>>  430 |         if export_format is not None:
     431 |             updates["export_format"] = export_format
```

- [ ] Add test to catch mutations on this line

### Line 431

```python
     430 |         if export_format is not None:
>>>  431 |             updates["export_format"] = export_format
     432 |         if export_sink is not None:
```

- [ ] Add test to catch mutations on this line

### Line 432

```python
     431 |             updates["export_format"] = export_format
>>>  432 |         if export_sink is not None:
     433 |             updates["export_sink"] = export_sink
```

- [ ] Add test to catch mutations on this line

### Line 484

```python
     483 |
>>>  484 |         # Extract schema info for audit (WP-11.99)
     485 |         schema_fields_json: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 485 (2 mutants)

```python
     484 |         # Extract schema info for audit (WP-11.99)
>>>  485 |         schema_fields_json: str | None = None
     486 |         schema_fields_list: list[dict[str, object]] | None = None
```

- [ ] Add test to catch mutations on this line

### Line 491

```python
     490 |         else:
>>>  491 |             # mode is non-None when is_dynamic is False
     492 |             schema_mode = schema_config.mode or "free"  # Fallback shouldn't happen
```

- [ ] Add test to catch mutations on this line

### Line 496

```python
     495 |                 # Cast each dict to wider type for storage
>>>  496 |                 field_dicts = [f.to_dict() for f in schema_config.fields]
     497 |                 schema_fields_list = [dict(d) for d in field_dicts]
```

- [ ] Add test to catch mutations on this line

### Line 603

```python
     602 |         """
>>>  603 |         # Parse schema_fields_json back to list
     604 |         schema_fields: list[dict[str, object]] | None = None
```

- [ ] Add test to catch mutations on this line

### Line 821

```python
     820 |         # would cause tokens to disappear without audit trail.
>>>  821 |         if not branches:
     822 |             raise ValueError("fork_token requires at least one branch")
```

- [ ] Add test to catch mutations on this line

### Line 823

```python
     822 |             raise ValueError("fork_token requires at least one branch")
>>>  823 |
     824 |         fork_group_id = _generate_id()
```

- [ ] Add test to catch mutations on this line

### Line 948

```python
     947 |         """
>>>  948 |         if count < 1:
     949 |             raise ValueError("expand_token requires at least 1 child")
```

- [ ] Add test to catch mutations on this line

### Line 1000

```python
     999 |         *,
>>> 1000 |         state_id: str | None = None,
    1001 |         attempt: int = 0,
```

- [ ] Add test to catch mutations on this line

### Line 1082

```python
    1081 |         # Check for enum first since NodeStateStatus is a str subclass
>>> 1082 |         if isinstance(status, NodeStateStatus):
    1083 |             status_enum = status
```

- [ ] Add test to catch mutations on this line

### Line 1083

```python
    1082 |         if isinstance(status, NodeStateStatus):
>>> 1083 |             status_enum = status
    1084 |         elif status == "rejected":
```

- [ ] Add test to catch mutations on this line

### Line 1084

```python
    1083 |             status_enum = status
>>> 1084 |         elif status == "rejected":
    1085 |             status_enum = NodeStateStatus.FAILED
```

- [ ] Add test to catch mutations on this line

### Line 1089

```python
    1088 |
>>> 1089 |         if status_enum == NodeStateStatus.OPEN:
    1090 |             raise ValueError("Cannot complete a node state with status OPEN")
```

- [ ] Add test to catch mutations on this line

### Line 1092

```python
    1091 |
>>> 1092 |         if duration_ms is None:
    1093 |             raise ValueError("duration_ms is required when completing a node state")
```

- [ ] Add test to catch mutations on this line

### Line 1097 (2 mutants)

```python
    1096 |         output_hash = stable_hash(output_data) if output_data is not None else None
>>> 1097 |         error_json = canonical_json(error) if error is not None else None
    1098 |         context_json = canonical_json(context_after) if context_after is not None else None
```

- [ ] Add test to catch mutations on this line

### Line 1114

```python
    1113 |
>>> 1114 |         result = self.get_node_state(state_id)
    1115 |         assert result is not None, f"NodeState {state_id} not found after update"
```

- [ ] Add test to catch mutations on this line

### Line 1116

```python
    1115 |         assert result is not None, f"NodeState {state_id} not found after update"
>>> 1116 |         # Type narrowing: result is guaranteed to be Completed or Failed
    1117 |         assert not isinstance(result, NodeStateOpen), "State should be terminal after completion"
```

- [ ] Add test to catch mutations on this line

### Line 1148

```python
    1147 |         event_id: str | None = None,
>>> 1148 |         routing_group_id: str | None = None,
    1149 |         ordinal: int = 0,
```

- [ ] Add test to catch mutations on this line

### Line 1225

```python
    1224 |         """
>>> 1225 |         routing_group_id = _generate_id()
    1226 |         reason_hash = stable_hash(reason) if reason else None
```

- [ ] Add test to catch mutations on this line

### Line 1362

```python
    1361 |
>>> 1362 |         if trigger_type:
    1363 |             updates["trigger_type"] = trigger_type
```

- [ ] Add test to catch mutations on this line

### Line 1366 (2 mutants)

```python
    1365 |             updates["trigger_reason"] = trigger_reason
>>> 1366 |         if state_id:
    1367 |             updates["aggregation_state_id"] = state_id
```

- [ ] Add test to catch mutations on this line

### Line 1367 (2 mutants)

```python
    1366 |         if state_id:
>>> 1367 |             updates["aggregation_state_id"] = state_id
    1368 |         if status in ("completed", "failed"):
```

- [ ] Add test to catch mutations on this line

### Line 1410

```python
    1409 |
>>> 1410 |         result = self.get_batch(batch_id)
    1411 |         assert result is not None, f"Batch {batch_id} not found after update"
```

- [ ] Add test to catch mutations on this line

### Line 1570

```python
    1569 |         original = self.get_batch(batch_id)
>>> 1570 |         if original is None:
    1571 |             raise ValueError(f"Batch not found: {batch_id}")
```

- [ ] Add test to catch mutations on this line

### Line 1572

```python
    1571 |             raise ValueError(f"Batch not found: {batch_id}")
>>> 1572 |         if original.status != BatchStatus.FAILED:
    1573 |             raise ValueError(f"Can only retry failed batches, got status: {original.status}")
```

- [ ] Add test to catch mutations on this line

### Line 1675 (2 mutants)

```python
    1674 |
>>> 1675 |         if sink_node_id:
    1676 |             query = query.where(artifacts_table.c.sink_node_id == sink_node_id)
```

- [ ] Add test to catch mutations on this line

### Line 2080

```python
    2079 |
>>> 2080 |         # Try to load payload
    2081 |         source_data: dict[str, Any] | None = None
```

- [ ] Add test to catch mutations on this line

### Line 2183

```python
    2182 |             IntegrityError: If terminal outcome already exists for token
>>> 2183 |         """
    2184 |         outcome_id = f"out_{_generate_id()[:12]}"
```

- [ ] Add test to catch mutations on this line

### Line 2185 (2 mutants)

```python
    2184 |         outcome_id = f"out_{_generate_id()[:12]}"
>>> 2185 |         is_terminal = outcome.is_terminal
    2186 |         context_json = json.dumps(context) if context is not None else None
```

- [ ] Add test to catch mutations on this line

### Line 2229

```python
    2228 |                     token_outcomes_table.c.recorded_at.desc(),  # Then by time
>>> 2229 |                 )
    2230 |                 .limit(1)
```

- [ ] Add test to catch mutations on this line

### Line 2281 (3 mutants)

```python
    2280 |             error_id for tracking
>>> 2281 |         """
    2282 |         error_id = f"verr_{_generate_id()[:12]}"
```

- [ ] Add test to catch mutations on this line

### Line 2327

```python
    2326 |             error_id for tracking
>>> 2327 |         """
    2328 |         error_id = f"terr_{_generate_id()[:12]}"
```

- [ ] Add test to catch mutations on this line

### Line 2362

```python
    2361 |         """
>>> 2362 |         query = select(validation_errors_table).where(
    2363 |             validation_errors_table.c.run_id == run_id,
```

- [ ] Add test to catch mutations on this line

### Line 2363

```python
    2362 |         query = select(validation_errors_table).where(
>>> 2363 |             validation_errors_table.c.run_id == run_id,
    2364 |             validation_errors_table.c.row_hash == row_hash,
```

- [ ] Add test to catch mutations on this line

### Line 2395

```python
    2394 |         """
>>> 2395 |         query = (
    2396 |             select(validation_errors_table).where(validation_errors_table.c.run_id == run_id).order_by(validation_errors_table.c.created_at)
```

- [ ] Add test to catch mutations on this line

### Line 2396

```python
    2395 |         query = (
>>> 2396 |             select(validation_errors_table).where(validation_errors_table.c.run_id == run_id).order_by(validation_errors_table.c.created_at)
    2397 |         )
```

- [ ] Add test to catch mutations on this line

### Line 2399

```python
    2398 |
>>> 2399 |         with self._db.connection() as conn:
    2400 |             result = conn.execute(query)
```

- [ ] Add test to catch mutations on this line

### Line 2400

```python
    2399 |         with self._db.connection() as conn:
>>> 2400 |             result = conn.execute(query)
    2401 |             rows = result.fetchall()
```

- [ ] Add test to catch mutations on this line

### Line 2459

```python
    2458 |         """
>>> 2459 |         query = (
    2460 |             select(transform_errors_table).where(transform_errors_table.c.run_id == run_id).order_by(transform_errors_table.c.created_at)
```

- [ ] Add test to catch mutations on this line

### Line 2460

```python
    2459 |         query = (
>>> 2460 |             select(transform_errors_table).where(transform_errors_table.c.run_id == run_id).order_by(transform_errors_table.c.created_at)
    2461 |         )
```

- [ ] Add test to catch mutations on this line

### Line 2463

```python
    2462 |
>>> 2463 |         with self._db.connection() as conn:
    2464 |             result = conn.execute(query)
```

- [ ] Add test to catch mutations on this line

### Line 2464

```python
    2463 |         with self._db.connection() as conn:
>>> 2464 |             result = conn.execute(query)
    2465 |             rows = result.fetchall()
```

- [ ] Add test to catch mutations on this line

### Line 2511

```python
    2510 |             .join(
>>> 2511 |                 node_states_table,
    2512 |                 calls_table.c.state_id == node_states_table.c.state_id,
```

- [ ] Add test to catch mutations on this line

### Line 2513

```python
    2512 |                 calls_table.c.state_id == node_states_table.c.state_id,
>>> 2513 |             )
    2514 |             .join(nodes_table, node_states_table.c.node_id == nodes_table.c.node_id)
```

- [ ] Add test to catch mutations on this line

### Line 2514

```python
    2513 |             )
>>> 2514 |             .join(nodes_table, node_states_table.c.node_id == nodes_table.c.node_id)
    2515 |             .where(nodes_table.c.run_id == run_id)
```

- [ ] Add test to catch mutations on this line

### Line 2515

```python
    2514 |             .join(nodes_table, node_states_table.c.node_id == nodes_table.c.node_id)
>>> 2515 |             .where(nodes_table.c.run_id == run_id)
    2516 |             .where(calls_table.c.call_type == call_type)
```

- [ ] Add test to catch mutations on this line

### Line 2516

```python
    2515 |             .where(nodes_table.c.run_id == run_id)
>>> 2516 |             .where(calls_table.c.call_type == call_type)
    2517 |             .where(calls_table.c.request_hash == request_hash)
```

- [ ] Add test to catch mutations on this line

### Line 2518

```python
    2517 |             .where(calls_table.c.request_hash == request_hash)
>>> 2518 |             .order_by(calls_table.c.created_at)
    2519 |             .limit(1)
```

- [ ] Add test to catch mutations on this line

### Line 2519

```python
    2518 |             .order_by(calls_table.c.created_at)
>>> 2519 |             .limit(1)
    2520 |         )
```

- [ ] Add test to catch mutations on this line

### Line 2522

```python
    2521 |
>>> 2522 |         with self._db.connection() as conn:
    2523 |             result = conn.execute(query)
```

- [ ] Add test to catch mutations on this line

### Line 2523

```python
    2522 |         with self._db.connection() as conn:
>>> 2523 |             result = conn.execute(query)
    2524 |             row = result.fetchone()
```

- [ ] Add test to catch mutations on this line

### Line 2525

```python
    2524 |             row = result.fetchone()
>>> 2525 |
    2526 |         if row is None:
```

- [ ] Add test to catch mutations on this line

## src/elspeth/engine/coalesce_executor.py

**Priority:** P1
**Survivors:** 29 unique lines (36 total mutants)

### Line 32

```python
      31 |
>>>   32 |     held: bool
      33 |     merged_token: TokenInfo | None = None
```

- [ ] Add test to catch mutations on this line

### Line 34 (2 mutants)

```python
      33 |     merged_token: TokenInfo | None = None
>>>   34 |     consumed_tokens: list[TokenInfo] = field(default_factory=list)
      35 |     coalesce_metadata: dict[str, Any] | None = None
```

- [ ] Add test to catch mutations on this line

### Line 35

```python
      34 |     consumed_tokens: list[TokenInfo] = field(default_factory=list)
>>>   35 |     coalesce_metadata: dict[str, Any] | None = None
      36 |     failure_reason: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 86

```python
      85 |         """
>>>   86 |         self._recorder = recorder
      87 |         self._spans = span_factory
```

- [ ] Add test to catch mutations on this line

### Line 88

```python
      87 |         self._spans = span_factory
>>>   88 |         self._token_manager = token_manager
      89 |         self._run_id = run_id
```

- [ ] Add test to catch mutations on this line

### Line 144

```python
     143 |         """
>>>  144 |         if coalesce_name not in self._settings:
     145 |             raise ValueError(f"Coalesce '{coalesce_name}' not registered")
```

- [ ] Add test to catch mutations on this line

### Line 147

```python
     146 |
>>>  147 |         if token.branch_name is None:
     148 |             raise ValueError(f"Token {token.token_id} has no branch_name - only forked tokens can be coalesced")
```

- [ ] Add test to catch mutations on this line

### Line 155

```python
     154 |         if token.branch_name not in settings.branches:
>>>  155 |             raise ValueError(
     156 |                 f"Token branch '{token.branch_name}' not in expected branches for coalesce '{coalesce_name}': {settings.branches}"
```

- [ ] Add test to catch mutations on this line

### Line 197

```python
     196 |         expected_count = len(settings.branches)
>>>  197 |
     198 |         if settings.policy == "require_all":
```

- [ ] Add test to catch mutations on this line

### Line 246

```python
     245 |                 state_id=state.state_id,
>>>  246 |                 status="completed",
     247 |                 output_data={"merged_into": merged_token.token_id},
```

- [ ] Add test to catch mutations on this line

### Line 247

```python
     246 |                 status="completed",
>>>  247 |                 output_data={"merged_into": merged_token.token_id},
     248 |                 duration_ms=0,
```

- [ ] Add test to catch mutations on this line

### Line 259 (3 mutants)

```python
     258 |                 {
>>>  259 |                     "branch": branch,
     260 |                     "arrival_offset_ms": (t - pending.first_arrival) * 1000,
```

- [ ] Add test to catch mutations on this line

### Line 263 (3 mutants)

```python
     262 |                 for branch, t in sorted(pending.arrival_times.items(), key=lambda x: x[1])
>>>  263 |             ],
     264 |             "wait_duration_ms": (now - pending.first_arrival) * 1000,
```

- [ ] Add test to catch mutations on this line

### Line 320

```python
     319 |         """
>>>  320 |         if coalesce_name not in self._settings:
     321 |             raise ValueError(f"Coalesce '{coalesce_name}' not registered")
```

- [ ] Add test to catch mutations on this line

### Line 335

```python
     334 |         for key, pending in self._pending.items():
>>>  335 |             if key[0] != coalesce_name:
     336 |                 continue
```

- [ ] Add test to catch mutations on this line

### Line 337

```python
     336 |                 continue
>>>  337 |
     338 |             elapsed = now - pending.first_arrival
```

- [ ] Add test to catch mutations on this line

### Line 338

```python
     337 |
>>>  338 |             elapsed = now - pending.first_arrival
     339 |             if elapsed >= settings.timeout_seconds:
```

- [ ] Add test to catch mutations on this line

### Line 346

```python
     345 |
>>>  346 |             # For best_effort, always merge on timeout if anything arrived
     347 |             if settings.policy == "best_effort" and len(pending.arrived) > 0:
```

- [ ] Add test to catch mutations on this line

### Line 357 (2 mutants)

```python
     356 |
>>>  357 |             # For quorum, merge on timeout only if quorum met
     358 |             elif settings.policy == "quorum":
```

- [ ] Add test to catch mutations on this line

### Line 359

```python
     358 |             elif settings.policy == "quorum":
>>>  359 |                 assert settings.quorum_count is not None
     360 |                 if len(pending.arrived) >= settings.quorum_count:
```

- [ ] Add test to catch mutations on this line

### Line 366

```python
     365 |                         step_in_pipeline=step_in_pipeline,
>>>  366 |                         key=key,
     367 |                     )
```

- [ ] Add test to catch mutations on this line

### Line 399 (2 mutants)

```python
     398 |             if settings.policy == "best_effort":
>>>  399 |                 # Always merge whatever arrived
     400 |                 if len(pending.arrived) > 0:
```

- [ ] Add test to catch mutations on this line

### Line 411

```python
     410 |             elif settings.policy == "quorum":
>>>  411 |                 assert settings.quorum_count is not None
     412 |                 if len(pending.arrived) >= settings.quorum_count:
```

- [ ] Add test to catch mutations on this line

### Line 418

```python
     417 |                         step_in_pipeline=step_in_pipeline,
>>>  418 |                         key=key,
     419 |                     )
```

- [ ] Add test to catch mutations on this line

### Line 428

```python
     427 |                             failure_reason="quorum_not_met",
>>>  428 |                             coalesce_metadata={
     429 |                                 "policy": settings.policy,
```

- [ ] Add test to catch mutations on this line

### Line 429

```python
     428 |                             coalesce_metadata={
>>>  429 |                                 "policy": settings.policy,
     430 |                                 "quorum_required": settings.quorum_count,
```

- [ ] Add test to catch mutations on this line

### Line 430

```python
     429 |                                 "policy": settings.policy,
>>>  430 |                                 "quorum_required": settings.quorum_count,
     431 |                                 "branches_arrived": list(pending.arrived.keys()),
```

- [ ] Add test to catch mutations on this line

### Line 444

```python
     443 |                         coalesce_metadata={
>>>  444 |                             "policy": settings.policy,
     445 |                             "expected_branches": settings.branches,
```

- [ ] Add test to catch mutations on this line

### Line 445

```python
     444 |                             "policy": settings.policy,
>>>  445 |                             "expected_branches": settings.branches,
     446 |                             "branches_arrived": list(pending.arrived.keys()),
```

- [ ] Add test to catch mutations on this line

## src/elspeth/core/canonical.py

**Priority:** P2
**Survivors:** 3 unique lines (3 total mutants)

### Line 49

```python
      48 |     if isinstance(obj, float | np.floating):
>>>   49 |         if math.isnan(obj) or math.isinf(obj):
      50 |             raise ValueError(f"Cannot canonicalize non-finite float: {obj}. Use None for missing values, not NaN.")
```

- [ ] Add test to catch mutations on this line

### Line 55

```python
      54 |
>>>   55 |     # Primitives pass through unchanged
      56 |     if obj is None or isinstance(obj, str | int | bool):
```

- [ ] Add test to catch mutations on this line

### Line 74

```python
      73 |
>>>   74 |     # Intentional missing values (NOT NaN - that's rejected above)
      75 |     if obj is pd.NA or (isinstance(obj, type(pd.NaT)) and obj is pd.NaT):
```

- [ ] Add test to catch mutations on this line

## src/elspeth/core/checkpoint/manager.py

**Priority:** P2
**Survivors:** 1 unique lines (1 total mutants)

### Line 96

```python
      95 |                 .where(checkpoints_table.c.run_id == run_id)
>>>   96 |                 .order_by(desc(checkpoints_table.c.sequence_number))
      97 |                 .limit(1)
```

- [ ] Add test to catch mutations on this line

## src/elspeth/core/checkpoint/recovery.py

**Priority:** P2
**Survivors:** 12 unique lines (12 total mutants)

### Line 31

```python
      30 |
>>>   31 |     can_resume: bool
      32 |     reason: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 35

```python
      34 |     def __post_init__(self) -> None:
>>>   35 |         if self.can_resume and self.reason is not None:
      36 |             raise ValueError("can_resume=True should not have a reason")
```

- [ ] Add test to catch mutations on this line

### Line 37

```python
      36 |             raise ValueError("can_resume=True should not have a reason")
>>>   37 |         if not self.can_resume and self.reason is None:
      38 |             raise ValueError("can_resume=False must have a reason explaining why")
```

- [ ] Add test to catch mutations on this line

### Line 100

```python
      99 |         run = self._get_run(run_id)
>>>  100 |         if run is None:
     101 |             return ResumeCheck(can_resume=False, reason=f"Run {run_id} not found")
```

- [ ] Add test to catch mutations on this line

### Line 103

```python
     102 |
>>>  103 |         if run.status == RunStatus.COMPLETED:
     104 |             return ResumeCheck(can_resume=False, reason="Run already completed successfully")
```

- [ ] Add test to catch mutations on this line

### Line 106

```python
     105 |
>>>  106 |         if run.status == RunStatus.RUNNING:
     107 |             return ResumeCheck(can_resume=False, reason="Run is still in progress")
```

- [ ] Add test to catch mutations on this line

### Line 110

```python
     109 |         checkpoint = self._checkpoint_manager.get_latest_checkpoint(run_id)
>>>  110 |         if checkpoint is None:
     111 |             return ResumeCheck(can_resume=False, reason="No checkpoint found for recovery")
```

- [ ] Add test to catch mutations on this line

### Line 202

```python
     201 |
>>>  202 |                 if row_result is None:
     203 |                     raise ValueError(f"Row {row_id} not found in database")
```

- [ ] Add test to catch mutations on this line

### Line 208

```python
     207 |
>>>  208 |                 if source_data_ref is None:
     209 |                     raise ValueError(f"Row {row_id} has no source_data_ref - cannot resume without payload")
```

- [ ] Add test to catch mutations on this line

### Line 215

```python
     214 |                     degraded_data = json.loads(payload_bytes.decode("utf-8"))
>>>  215 |                 except KeyError:
     216 |                     raise ValueError(f"Row {row_id} payload has been purged - cannot resume") from None
```

- [ ] Add test to catch mutations on this line

### Line 270

```python
     269 |             if checkpointed_row_result is None:
>>>  270 |                 raise RuntimeError(
     271 |                     f"Checkpoint references non-existent token: {checkpoint.token_id}. "
```

- [ ] Add test to catch mutations on this line

### Line 271

```python
     270 |                 raise RuntimeError(
>>>  271 |                     f"Checkpoint references non-existent token: {checkpoint.token_id}. "
     272 |                     "This indicates database corruption or a bug in checkpoint creation."
```

- [ ] Add test to catch mutations on this line

## src/elspeth/core/landscape/database.py

**Priority:** P2
**Survivors:** 17 unique lines (20 total mutants)

### Line 42 (2 mutants)

```python
      41 |         """
>>>   42 |         self.connection_string = connection_string
      43 |         self._engine: Engine | None = None
```

- [ ] Add test to catch mutations on this line

### Line 51

```python
      50 |         self._engine = create_engine(
>>>   51 |             self.connection_string,
      52 |             echo=False,  # Set True for SQL debugging
```

- [ ] Add test to catch mutations on this line

### Line 58

```python
      57 |             LandscapeDB._configure_sqlite(self._engine)
>>>   58 |
      59 |     @staticmethod
```

- [ ] Add test to catch mutations on this line

### Line 106

```python
     105 |             if table_name not in inspector.get_table_names():
>>>  106 |                 # Table will be created by create_all, skip
     107 |                 continue
```

- [ ] Add test to catch mutations on this line

### Line 114 (2 mutants)

```python
     113 |
>>>  114 |         if missing_columns:
     115 |             missing_str = ", ".join(f"{t}.{c}" for t, c in missing_columns)
```

- [ ] Add test to catch mutations on this line

### Line 116

```python
     115 |             missing_str = ", ".join(f"{t}.{c}" for t, c in missing_columns)
>>>  116 |             raise SchemaCompatibilityError(
     117 |                 f"Landscape database schema is outdated. "
```

- [ ] Add test to catch mutations on this line

### Line 117

```python
     116 |             raise SchemaCompatibilityError(
>>>  117 |                 f"Landscape database schema is outdated. "
     118 |                 f"Missing columns: {missing_str}\n\n"
```

- [ ] Add test to catch mutations on this line

### Line 118

```python
     117 |                 f"Landscape database schema is outdated. "
>>>  118 |                 f"Missing columns: {missing_str}\n\n"
     119 |                 f"To fix this, either:\n"
```

- [ ] Add test to catch mutations on this line

### Line 119

```python
     118 |                 f"Missing columns: {missing_str}\n\n"
>>>  119 |                 f"To fix this, either:\n"
     120 |                 f"  1. Delete the database file and let ELSPETH recreate it, or\n"
```

- [ ] Add test to catch mutations on this line

### Line 120

```python
     119 |                 f"To fix this, either:\n"
>>>  120 |                 f"  1. Delete the database file and let ELSPETH recreate it, or\n"
     121 |                 f"  2. Run: elspeth landscape migrate (when available)\n\n"
```

- [ ] Add test to catch mutations on this line

### Line 121

```python
     120 |                 f"  1. Delete the database file and let ELSPETH recreate it, or\n"
>>>  121 |                 f"  2. Run: elspeth landscape migrate (when available)\n\n"
     122 |                 f"Database: {self.connection_string}"
```

- [ ] Add test to catch mutations on this line

### Line 128

```python
     127 |         """Get the SQLAlchemy engine."""
>>>  128 |         if self._engine is None:
     129 |             raise RuntimeError("Database not initialized")
```

- [ ] Add test to catch mutations on this line

### Line 133

```python
     132 |     def close(self) -> None:
>>>  133 |         """Close database connection."""
     134 |         if self._engine is not None:
```

- [ ] Add test to catch mutations on this line

### Line 135

```python
     134 |         if self._engine is not None:
>>>  135 |             self._engine.dispose()
     136 |             self._engine = None
```

- [ ] Add test to catch mutations on this line

### Line 157

```python
     156 |             LandscapeDB instance with in-memory SQLite
>>>  157 |         """
     158 |         engine = create_engine("sqlite:///:memory:", echo=False)
```

- [ ] Add test to catch mutations on this line

### Line 161 (2 mutants)

```python
     160 |         metadata.create_all(engine)
>>>  161 |         instance = cls.__new__(cls)
     162 |         instance.connection_string = "sqlite:///:memory:"
```

- [ ] Add test to catch mutations on this line

### Line 177

```python
     176 |             LandscapeDB instance
>>>  177 |         """
     178 |         engine = create_engine(url, echo=False)
```

- [ ] Add test to catch mutations on this line

## src/elspeth/core/landscape/exporter.py

**Priority:** P2
**Survivors:** 88 unique lines (91 total mutants)

### Line 66

```python
      65 |                         Required if sign=True is passed to export_run().
>>>   66 |         """
      67 |         self._db = db
```

- [ ] Add test to catch mutations on this line

### Line 83

```python
      82 |         """
>>>   83 |         if self._signing_key is None:
      84 |             raise ValueError("Signing key not configured")
```

- [ ] Add test to catch mutations on this line

### Line 116

```python
     115 |         """
>>>  116 |         if sign and self._signing_key is None:
     117 |             raise ValueError("Signing requested but no signing_key provided")
```

- [ ] Add test to catch mutations on this line

### Line 134

```python
     133 |             manifest = {
>>>  134 |                 "record_type": "manifest",
     135 |                 "run_id": run_id,
```

- [ ] Add test to catch mutations on this line

### Line 159

```python
     158 |         run = self._recorder.get_run(run_id)
>>>  159 |         if run is None:
     160 |             raise ValueError(f"Run not found: {run_id}")
```

- [ ] Add test to catch mutations on this line

### Line 169

```python
     168 |             "canonical_version": run.canonical_version,
>>>  169 |             "config_hash": run.config_hash,
     170 |             "reproducibility_grade": run.reproducibility_grade,
```

- [ ] Add test to catch mutations on this line

### Line 176

```python
     175 |             yield {
>>>  176 |                 "record_type": "node",
     177 |                 "run_id": run_id,
```

- [ ] Add test to catch mutations on this line

### Line 179

```python
     178 |                 "node_id": node.node_id,
>>>  179 |                 "plugin_name": node.plugin_name,
     180 |                 "node_type": node.node_type,
```

- [ ] Add test to catch mutations on this line

### Line 180

```python
     179 |                 "plugin_name": node.plugin_name,
>>>  180 |                 "node_type": node.node_type,
     181 |                 "plugin_version": node.plugin_version,
```

- [ ] Add test to catch mutations on this line

### Line 181

```python
     180 |                 "node_type": node.node_type,
>>>  181 |                 "plugin_version": node.plugin_version,
     182 |                 "config_hash": node.config_hash,
```

- [ ] Add test to catch mutations on this line

### Line 182

```python
     181 |                 "plugin_version": node.plugin_version,
>>>  182 |                 "config_hash": node.config_hash,
     183 |                 "schema_hash": node.schema_hash,
```

- [ ] Add test to catch mutations on this line

### Line 183

```python
     182 |                 "config_hash": node.config_hash,
>>>  183 |                 "schema_hash": node.schema_hash,
     184 |                 "sequence_in_pipeline": node.sequence_in_pipeline,
```

- [ ] Add test to catch mutations on this line

### Line 190

```python
     189 |             yield {
>>>  190 |                 "record_type": "edge",
     191 |                 "run_id": run_id,
```

- [ ] Add test to catch mutations on this line

### Line 191

```python
     190 |                 "record_type": "edge",
>>>  191 |                 "run_id": run_id,
     192 |                 "edge_id": edge.edge_id,
```

- [ ] Add test to catch mutations on this line

### Line 213

```python
     212 |                 yield {
>>>  213 |                     "record_type": "token",
     214 |                     "run_id": run_id,
```

- [ ] Add test to catch mutations on this line

### Line 216

```python
     215 |                     "token_id": token.token_id,
>>>  216 |                     "row_id": token.row_id,
     217 |                     "step_in_pipeline": token.step_in_pipeline,
```

- [ ] Add test to catch mutations on this line

### Line 217

```python
     216 |                     "row_id": token.row_id,
>>>  217 |                     "step_in_pipeline": token.step_in_pipeline,
     218 |                     "branch_name": token.branch_name,
```

- [ ] Add test to catch mutations on this line

### Line 218

```python
     217 |                     "step_in_pipeline": token.step_in_pipeline,
>>>  218 |                     "branch_name": token.branch_name,
     219 |                     "fork_group_id": token.fork_group_id,
```

- [ ] Add test to catch mutations on this line

### Line 219

```python
     218 |                     "branch_name": token.branch_name,
>>>  219 |                     "fork_group_id": token.fork_group_id,
     220 |                     "join_group_id": token.join_group_id,
```

- [ ] Add test to catch mutations on this line

### Line 226

```python
     225 |                     yield {
>>>  226 |                         "record_type": "token_parent",
     227 |                         "run_id": run_id,
```

- [ ] Add test to catch mutations on this line

### Line 227

```python
     226 |                         "record_type": "token_parent",
>>>  227 |                         "run_id": run_id,
     228 |                         "token_id": parent.token_id,
```

- [ ] Add test to catch mutations on this line

### Line 229

```python
     228 |                         "token_id": parent.token_id,
>>>  229 |                         "parent_token_id": parent.parent_token_id,
     230 |                         "ordinal": parent.ordinal,
```

- [ ] Add test to catch mutations on this line

### Line 237

```python
     236 |                     if isinstance(state, NodeStateOpen):
>>>  237 |                         yield {
     238 |                             "record_type": "node_state",
```

- [ ] Add test to catch mutations on this line

### Line 238

```python
     237 |                         yield {
>>>  238 |                             "record_type": "node_state",
     239 |                             "run_id": run_id,
```

- [ ] Add test to catch mutations on this line

### Line 239

```python
     238 |                             "record_type": "node_state",
>>>  239 |                             "run_id": run_id,
     240 |                             "state_id": state.state_id,
```

- [ ] Add test to catch mutations on this line

### Line 240

```python
     239 |                             "run_id": run_id,
>>>  240 |                             "state_id": state.state_id,
     241 |                             "token_id": state.token_id,
```

- [ ] Add test to catch mutations on this line

### Line 241

```python
     240 |                             "state_id": state.state_id,
>>>  241 |                             "token_id": state.token_id,
     242 |                             "node_id": state.node_id,
```

- [ ] Add test to catch mutations on this line

### Line 242

```python
     241 |                             "token_id": state.token_id,
>>>  242 |                             "node_id": state.node_id,
     243 |                             "step_index": state.step_index,
```

- [ ] Add test to catch mutations on this line

### Line 243

```python
     242 |                             "node_id": state.node_id,
>>>  243 |                             "step_index": state.step_index,
     244 |                             "attempt": state.attempt,
```

- [ ] Add test to catch mutations on this line

### Line 244

```python
     243 |                             "step_index": state.step_index,
>>>  244 |                             "attempt": state.attempt,
     245 |                             "status": state.status.value,
```

- [ ] Add test to catch mutations on this line

### Line 245

```python
     244 |                             "attempt": state.attempt,
>>>  245 |                             "status": state.status.value,
     246 |                             "input_hash": state.input_hash,
```

- [ ] Add test to catch mutations on this line

### Line 246

```python
     245 |                             "status": state.status.value,
>>>  246 |                             "input_hash": state.input_hash,
     247 |                             "output_hash": None,
```

- [ ] Add test to catch mutations on this line

### Line 247

```python
     246 |                             "input_hash": state.input_hash,
>>>  247 |                             "output_hash": None,
     248 |                             "duration_ms": None,
```

- [ ] Add test to catch mutations on this line

### Line 248

```python
     247 |                             "output_hash": None,
>>>  248 |                             "duration_ms": None,
     249 |                             "started_at": state.started_at.isoformat(),
```

- [ ] Add test to catch mutations on this line

### Line 249

```python
     248 |                             "duration_ms": None,
>>>  249 |                             "started_at": state.started_at.isoformat(),
     250 |                             "completed_at": None,
```

- [ ] Add test to catch mutations on this line

### Line 254

```python
     253 |                         yield {
>>>  254 |                             "record_type": "node_state",
     255 |                             "run_id": run_id,
```

- [ ] Add test to catch mutations on this line

### Line 255

```python
     254 |                             "record_type": "node_state",
>>>  255 |                             "run_id": run_id,
     256 |                             "state_id": state.state_id,
```

- [ ] Add test to catch mutations on this line

### Line 258

```python
     257 |                             "token_id": state.token_id,
>>>  258 |                             "node_id": state.node_id,
     259 |                             "step_index": state.step_index,
```

- [ ] Add test to catch mutations on this line

### Line 259

```python
     258 |                             "node_id": state.node_id,
>>>  259 |                             "step_index": state.step_index,
     260 |                             "attempt": state.attempt,
```

- [ ] Add test to catch mutations on this line

### Line 261

```python
     260 |                             "attempt": state.attempt,
>>>  261 |                             "status": state.status.value,
     262 |                             "input_hash": state.input_hash,
```

- [ ] Add test to catch mutations on this line

### Line 262

```python
     261 |                             "status": state.status.value,
>>>  262 |                             "input_hash": state.input_hash,
     263 |                             "output_hash": state.output_hash,
```

- [ ] Add test to catch mutations on this line

### Line 263

```python
     262 |                             "input_hash": state.input_hash,
>>>  263 |                             "output_hash": state.output_hash,
     264 |                             "duration_ms": state.duration_ms,
```

- [ ] Add test to catch mutations on this line

### Line 264

```python
     263 |                             "output_hash": state.output_hash,
>>>  264 |                             "duration_ms": state.duration_ms,
     265 |                             "started_at": state.started_at.isoformat(),
```

- [ ] Add test to catch mutations on this line

### Line 265

```python
     264 |                             "duration_ms": state.duration_ms,
>>>  265 |                             "started_at": state.started_at.isoformat(),
     266 |                             "completed_at": state.completed_at.isoformat(),
```

- [ ] Add test to catch mutations on this line

### Line 269 (2 mutants)

```python
     268 |                     else:  # NodeStateFailed
>>>  269 |                         yield {
     270 |                             "record_type": "node_state",
```

- [ ] Add test to catch mutations on this line

### Line 270

```python
     269 |                         yield {
>>>  270 |                             "record_type": "node_state",
     271 |                             "run_id": run_id,
```

- [ ] Add test to catch mutations on this line

### Line 271

```python
     270 |                             "record_type": "node_state",
>>>  271 |                             "run_id": run_id,
     272 |                             "state_id": state.state_id,
```

- [ ] Add test to catch mutations on this line

### Line 272

```python
     271 |                             "run_id": run_id,
>>>  272 |                             "state_id": state.state_id,
     273 |                             "token_id": state.token_id,
```

- [ ] Add test to catch mutations on this line

### Line 273

```python
     272 |                             "state_id": state.state_id,
>>>  273 |                             "token_id": state.token_id,
     274 |                             "node_id": state.node_id,
```

- [ ] Add test to catch mutations on this line

### Line 274

```python
     273 |                             "token_id": state.token_id,
>>>  274 |                             "node_id": state.node_id,
     275 |                             "step_index": state.step_index,
```

- [ ] Add test to catch mutations on this line

### Line 275

```python
     274 |                             "node_id": state.node_id,
>>>  275 |                             "step_index": state.step_index,
     276 |                             "attempt": state.attempt,
```

- [ ] Add test to catch mutations on this line

### Line 276

```python
     275 |                             "step_index": state.step_index,
>>>  276 |                             "attempt": state.attempt,
     277 |                             "status": state.status.value,
```

- [ ] Add test to catch mutations on this line

### Line 277

```python
     276 |                             "attempt": state.attempt,
>>>  277 |                             "status": state.status.value,
     278 |                             "input_hash": state.input_hash,
```

- [ ] Add test to catch mutations on this line

### Line 278

```python
     277 |                             "status": state.status.value,
>>>  278 |                             "input_hash": state.input_hash,
     279 |                             "output_hash": state.output_hash,
```

- [ ] Add test to catch mutations on this line

### Line 279

```python
     278 |                             "input_hash": state.input_hash,
>>>  279 |                             "output_hash": state.output_hash,
     280 |                             "duration_ms": state.duration_ms,
```

- [ ] Add test to catch mutations on this line

### Line 280

```python
     279 |                             "output_hash": state.output_hash,
>>>  280 |                             "duration_ms": state.duration_ms,
     281 |                             "started_at": state.started_at.isoformat(),
```

- [ ] Add test to catch mutations on this line

### Line 281

```python
     280 |                             "duration_ms": state.duration_ms,
>>>  281 |                             "started_at": state.started_at.isoformat(),
     282 |                             "completed_at": state.completed_at.isoformat(),
```

- [ ] Add test to catch mutations on this line

### Line 288

```python
     287 |                         yield {
>>>  288 |                             "record_type": "routing_event",
     289 |                             "run_id": run_id,
```

- [ ] Add test to catch mutations on this line

### Line 289

```python
     288 |                             "record_type": "routing_event",
>>>  289 |                             "run_id": run_id,
     290 |                             "event_id": event.event_id,
```

- [ ] Add test to catch mutations on this line

### Line 292

```python
     291 |                             "state_id": event.state_id,
>>>  292 |                             "edge_id": event.edge_id,
     293 |                             "routing_group_id": event.routing_group_id,
```

- [ ] Add test to catch mutations on this line

### Line 293

```python
     292 |                             "edge_id": event.edge_id,
>>>  293 |                             "routing_group_id": event.routing_group_id,
     294 |                             "ordinal": event.ordinal,
```

- [ ] Add test to catch mutations on this line

### Line 294

```python
     293 |                             "routing_group_id": event.routing_group_id,
>>>  294 |                             "ordinal": event.ordinal,
     295 |                             "mode": event.mode,
```

- [ ] Add test to catch mutations on this line

### Line 295

```python
     294 |                             "ordinal": event.ordinal,
>>>  295 |                             "mode": event.mode,
     296 |                             "reason_hash": event.reason_hash,
```

- [ ] Add test to catch mutations on this line

### Line 301 (2 mutants)

```python
     300 |                     for call in self._recorder.get_calls(state.state_id):
>>>  301 |                         yield {
     302 |                             "record_type": "call",
```

- [ ] Add test to catch mutations on this line

### Line 302

```python
     301 |                         yield {
>>>  302 |                             "record_type": "call",
     303 |                             "run_id": run_id,
```

- [ ] Add test to catch mutations on this line

### Line 303

```python
     302 |                             "record_type": "call",
>>>  303 |                             "run_id": run_id,
     304 |                             "call_id": call.call_id,
```

- [ ] Add test to catch mutations on this line

### Line 304

```python
     303 |                             "run_id": run_id,
>>>  304 |                             "call_id": call.call_id,
     305 |                             "state_id": call.state_id,
```

- [ ] Add test to catch mutations on this line

### Line 305

```python
     304 |                             "call_id": call.call_id,
>>>  305 |                             "state_id": call.state_id,
     306 |                             "call_index": call.call_index,
```

- [ ] Add test to catch mutations on this line

### Line 306

```python
     305 |                             "state_id": call.state_id,
>>>  306 |                             "call_index": call.call_index,
     307 |                             "call_type": call.call_type,
```

- [ ] Add test to catch mutations on this line

### Line 307

```python
     306 |                             "call_index": call.call_index,
>>>  307 |                             "call_type": call.call_type,
     308 |                             "status": call.status,
```

- [ ] Add test to catch mutations on this line

### Line 308

```python
     307 |                             "call_type": call.call_type,
>>>  308 |                             "status": call.status,
     309 |                             "request_hash": call.request_hash,
```

- [ ] Add test to catch mutations on this line

### Line 309

```python
     308 |                             "status": call.status,
>>>  309 |                             "request_hash": call.request_hash,
     310 |                             "response_hash": call.response_hash,
```

- [ ] Add test to catch mutations on this line

### Line 310

```python
     309 |                             "request_hash": call.request_hash,
>>>  310 |                             "response_hash": call.response_hash,
     311 |                             "latency_ms": call.latency_ms,
```

- [ ] Add test to catch mutations on this line

### Line 317

```python
     316 |             yield {
>>>  317 |                 "record_type": "batch",
     318 |                 "run_id": run_id,
```

- [ ] Add test to catch mutations on this line

### Line 320

```python
     319 |                 "batch_id": batch.batch_id,
>>>  320 |                 "aggregation_node_id": batch.aggregation_node_id,
     321 |                 "attempt": batch.attempt,
```

- [ ] Add test to catch mutations on this line

### Line 322

```python
     321 |                 "attempt": batch.attempt,
>>>  322 |                 "status": batch.status,
     323 |                 "trigger_reason": batch.trigger_reason,
```

- [ ] Add test to catch mutations on this line

### Line 323

```python
     322 |                 "status": batch.status,
>>>  323 |                 "trigger_reason": batch.trigger_reason,
     324 |                 "created_at": (batch.created_at.isoformat() if batch.created_at else None),
```

- [ ] Add test to catch mutations on this line

### Line 324

```python
     323 |                 "trigger_reason": batch.trigger_reason,
>>>  324 |                 "created_at": (batch.created_at.isoformat() if batch.created_at else None),
     325 |                 "completed_at": (batch.completed_at.isoformat() if batch.completed_at else None),
```

- [ ] Add test to catch mutations on this line

### Line 331

```python
     330 |                 yield {
>>>  331 |                     "record_type": "batch_member",
     332 |                     "run_id": run_id,
```

- [ ] Add test to catch mutations on this line

### Line 334

```python
     333 |                     "batch_id": member.batch_id,
>>>  334 |                     "token_id": member.token_id,
     335 |                     "ordinal": member.ordinal,
```

- [ ] Add test to catch mutations on this line

### Line 341

```python
     340 |             yield {
>>>  341 |                 "record_type": "artifact",
     342 |                 "run_id": run_id,
```

- [ ] Add test to catch mutations on this line

### Line 342

```python
     341 |                 "record_type": "artifact",
>>>  342 |                 "run_id": run_id,
     343 |                 "artifact_id": artifact.artifact_id,
```

- [ ] Add test to catch mutations on this line

### Line 344

```python
     343 |                 "artifact_id": artifact.artifact_id,
>>>  344 |                 "sink_node_id": artifact.sink_node_id,
     345 |                 "produced_by_state_id": artifact.produced_by_state_id,
```

- [ ] Add test to catch mutations on this line

### Line 346

```python
     345 |                 "produced_by_state_id": artifact.produced_by_state_id,
>>>  346 |                 "artifact_type": artifact.artifact_type,
     347 |                 "path_or_uri": artifact.path_or_uri,
```

- [ ] Add test to catch mutations on this line

### Line 348

```python
     347 |                 "path_or_uri": artifact.path_or_uri,
>>>  348 |                 "content_hash": artifact.content_hash,
     349 |                 "size_bytes": artifact.size_bytes,
```

- [ ] Add test to catch mutations on this line

### Line 354

```python
     353 |         self,
>>>  354 |         run_id: str,
     355 |         sign: bool = False,
```

- [ ] Add test to catch mutations on this line

### Line 372

```python
     371 |             ValueError: If run_id is not found, or sign=True without signing_key
>>>  372 |         """
     373 |         groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
```

- [ ] Add test to catch mutations on this line

### Line 375 (2 mutants)

```python
     374 |
>>>  375 |         for record in self.export_run(run_id, sign=sign):
     376 |             record_type = record["record_type"]
```

- [ ] Add test to catch mutations on this line

## src/elspeth/core/landscape/lineage.py

**Priority:** P2
**Survivors:** 5 unique lines (5 total mutants)

### Line 58

```python
      57 |     """Transform errors for this token (from transform processing)."""
>>>   58 |
      59 |     outcome: TokenOutcome | None = None
```

- [ ] Add test to catch mutations on this line

### Line 83

```python
      82 |     """
>>>   83 |     if token_id is None and row_id is None:
      84 |         raise ValueError("Must provide either token_id or row_id")
```

- [ ] Add test to catch mutations on this line

### Line 86

```python
      85 |
>>>   86 |     # Resolve token_id from row_id if needed
      87 |     if token_id is None and row_id is not None:
```

- [ ] Add test to catch mutations on this line

### Line 130

```python
     129 |
>>>  130 |     # Get validation errors for this row (by hash)
     131 |     validation_errors = recorder.get_validation_errors_for_row(run_id, source_row.source_data_hash)
```

- [ ] Add test to catch mutations on this line

### Line 133

```python
     132 |
>>>  133 |     # Get transform errors for this token
     134 |     transform_errors = recorder.get_transform_errors_for_token(token_id)
```

- [ ] Add test to catch mutations on this line

## src/elspeth/core/landscape/models.py

**Priority:** P2
**Survivors:** 39 unique lines (41 total mutants)

### Line 38

```python
      37 |     canonical_version: str
>>>   38 |     status: RunStatus
      39 |     completed_at: datetime | None = None
```

- [ ] Add test to catch mutations on this line

### Line 39

```python
      38 |     status: RunStatus
>>>   39 |     completed_at: datetime | None = None
      40 |     reproducibility_grade: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 41

```python
      40 |     reproducibility_grade: str | None = None
>>>   41 |     # Export tracking - separate from run status
      42 |     export_status: ExportStatus | None = None
```

- [ ] Add test to catch mutations on this line

### Line 42

```python
      41 |     # Export tracking - separate from run status
>>>   42 |     export_status: ExportStatus | None = None
      43 |     export_error: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 43

```python
      42 |     export_status: ExportStatus | None = None
>>>   43 |     export_error: str | None = None
      44 |     exported_at: datetime | None = None
```

- [ ] Add test to catch mutations on this line

### Line 44

```python
      43 |     export_error: str | None = None
>>>   44 |     exported_at: datetime | None = None
      45 |     export_format: str | None = None  # csv, json
```

- [ ] Add test to catch mutations on this line

### Line 45

```python
      44 |     exported_at: datetime | None = None
>>>   45 |     export_format: str | None = None  # csv, json
      46 |     export_sink: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 61

```python
      60 |     config_json: str
>>>   61 |     registered_at: datetime
      62 |     schema_hash: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 62

```python
      61 |     registered_at: datetime
>>>   62 |     schema_hash: str | None = None
      63 |     sequence_in_pipeline: int | None = None
```

- [ ] Add test to catch mutations on this line

### Line 88

```python
      87 |     source_data_hash: str
>>>   88 |     created_at: datetime
      89 |     source_data_ref: str | None = None  # Payload store reference
```

- [ ] Add test to catch mutations on this line

### Line 98

```python
      97 |     row_id: str
>>>   98 |     created_at: datetime
      99 |     fork_group_id: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 99

```python
      98 |     created_at: datetime
>>>   99 |     fork_group_id: str | None = None
     100 |     join_group_id: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 100

```python
      99 |     fork_group_id: str | None = None
>>>  100 |     join_group_id: str | None = None
     101 |     expand_group_id: str | None = None  # For deaggregation grouping
```

- [ ] Add test to catch mutations on this line

### Line 101

```python
     100 |     join_group_id: str | None = None
>>>  101 |     expand_group_id: str | None = None  # For deaggregation grouping
     102 |     branch_name: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 102

```python
     101 |     expand_group_id: str | None = None  # For deaggregation grouping
>>>  102 |     branch_name: str | None = None
     103 |     step_in_pipeline: int | None = None  # Step where this token was created (fork/coalesce/expand)
```

- [ ] Add test to catch mutations on this line

### Line 114

```python
     113 |
>>>  114 |
     115 | @dataclass(frozen=True)
```

- [ ] Add test to catch mutations on this line

### Line 137

```python
     136 |     input_hash: str
>>>  137 |     started_at: datetime
     138 |     context_before_json: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 140

```python
     139 |
>>>  140 |
     141 | @dataclass(frozen=True)
```

- [ ] Add test to catch mutations on this line

### Line 164

```python
     163 |     completed_at: datetime
>>>  164 |     duration_ms: float
     165 |     context_before_json: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 165

```python
     164 |     duration_ms: float
>>>  165 |     context_before_json: str | None = None
     166 |     context_after_json: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 168

```python
     167 |
>>>  168 |
     169 | @dataclass(frozen=True)
```

- [ ] Add test to catch mutations on this line

### Line 191

```python
     190 |     completed_at: datetime
>>>  191 |     duration_ms: float
     192 |     error_json: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 192

```python
     191 |     duration_ms: float
>>>  192 |     error_json: str | None = None
     193 |     output_hash: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 193

```python
     192 |     error_json: str | None = None
>>>  193 |     output_hash: str | None = None
     194 |     context_before_json: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 194

```python
     193 |     output_hash: str | None = None
>>>  194 |     context_before_json: str | None = None
     195 |     context_after_json: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 198 (3 mutants)

```python
     197 |
>>>  198 | # Discriminated union type - use status field to discriminate
     199 | NodeState = NodeStateOpen | NodeStateCompleted | NodeStateFailed
```

- [ ] Add test to catch mutations on this line

### Line 222

```python
     221 |     request_hash: str
>>>  222 |     created_at: datetime
     223 |     request_ref: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 223

```python
     222 |     created_at: datetime
>>>  223 |     request_ref: str | None = None
     224 |     response_hash: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 224

```python
     223 |     request_ref: str | None = None
>>>  224 |     response_hash: str | None = None
     225 |     response_ref: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 225

```python
     224 |     response_hash: str | None = None
>>>  225 |     response_ref: str | None = None
     226 |     error_json: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 226

```python
     225 |     response_ref: str | None = None
>>>  226 |     error_json: str | None = None
     227 |     latency_ms: float | None = None
```

- [ ] Add test to catch mutations on this line

### Line 242

```python
     241 |     size_bytes: int
>>>  242 |     created_at: datetime
     243 |     idempotency_key: str | None = None  # For retry deduplication
```

- [ ] Add test to catch mutations on this line

### Line 256

```python
     255 |     mode: str  # move, copy
>>>  256 |     created_at: datetime
     257 |     reason_hash: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 257

```python
     256 |     created_at: datetime
>>>  257 |     reason_hash: str | None = None
     258 |     reason_ref: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 270

```python
     269 |     status: BatchStatus
>>>  270 |     created_at: datetime
     271 |     aggregation_state_id: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 271

```python
     270 |     created_at: datetime
>>>  271 |     aggregation_state_id: str | None = None
     272 |     trigger_reason: str | None = None
```

- [ ] Add test to catch mutations on this line

### Line 272

```python
     271 |     aggregation_state_id: str | None = None
>>>  272 |     trigger_reason: str | None = None
     273 |     trigger_type: str | None = None  # TriggerType enum value
```

- [ ] Add test to catch mutations on this line

### Line 273

```python
     272 |     trigger_reason: str | None = None
>>>  273 |     trigger_type: str | None = None  # TriggerType enum value
     274 |     completed_at: datetime | None = None
```

- [ ] Add test to catch mutations on this line

### Line 308

```python
     307 |     sequence_number: int
>>>  308 |     created_at: datetime | None
     309 |     aggregation_state_json: str | None = None
```

- [ ] Add test to catch mutations on this line

## src/elspeth/core/landscape/repositories.py

**Priority:** P2
**Survivors:** 9 unique lines (9 total mutants)

### Line 38

```python
      37 |
>>>   38 |     def __init__(self, session: Any) -> None:
      39 |         self.session = session
```

- [ ] Add test to catch mutations on this line

### Line 66

```python
      65 |
>>>   66 |     def __init__(self, session: Any) -> None:
      67 |         self.session = session
```

- [ ] Add test to catch mutations on this line

### Line 92

```python
      91 |
>>>   92 |     def __init__(self, session: Any) -> None:
      93 |         self.session = session
```

- [ ] Add test to catch mutations on this line

### Line 114

```python
     113 |
>>>  114 |     def __init__(self, session: Any) -> None:
     115 |         self.session = session
```

- [ ] Add test to catch mutations on this line

### Line 136

```python
     135 |
>>>  136 |     def __init__(self, session: Any) -> None:
     137 |         self.session = session
```

- [ ] Add test to catch mutations on this line

### Line 159

```python
     158 |
>>>  159 |     def __init__(self, session: Any) -> None:
     160 |         self.session = session
```

- [ ] Add test to catch mutations on this line

### Line 174

```python
     173 |
>>>  174 |     def __init__(self, session: Any) -> None:
     175 |         self.session = session
```

- [ ] Add test to catch mutations on this line

### Line 198

```python
     197 |
>>>  198 |     def __init__(self, session: Any) -> None:
     199 |         self.session = session
```

- [ ] Add test to catch mutations on this line

### Line 219

```python
     218 |
>>>  219 |     def __init__(self, session: Any) -> None:
     220 |         self.session = session
```

- [ ] Add test to catch mutations on this line

## src/elspeth/core/landscape/reproducibility.py

**Priority:** P2
**Survivors:** 7 unique lines (10 total mutants)

### Line 33 (2 mutants)

```python
      32 |     """
>>>   33 |
      34 |     FULL_REPRODUCIBLE = "full_reproducible"
```

- [ ] Add test to catch mutations on this line

### Line 34 (2 mutants)

```python
      33 |
>>>   34 |     FULL_REPRODUCIBLE = "full_reproducible"
      35 |     REPLAY_REPRODUCIBLE = "replay_reproducible"
```

- [ ] Add test to catch mutations on this line

### Line 35 (2 mutants)

```python
      34 |     FULL_REPRODUCIBLE = "full_reproducible"
>>>   35 |     REPLAY_REPRODUCIBLE = "replay_reproducible"
      36 |     ATTRIBUTABLE_ONLY = "attributable_only"
```

- [ ] Add test to catch mutations on this line

### Line 66

```python
      65 |         .where(nodes_table.c.run_id == run_id)
>>>   66 |         .where(nodes_table.c.determinism.in_(non_reproducible))
      67 |         .limit(1)  # We only need to know if ANY exist
```

- [ ] Add test to catch mutations on this line

### Line 119

```python
     118 |         # Per Data Manifesto: "Bad data in the audit trail = crash immediately"
>>>  119 |         if current_grade is None:
     120 |             raise ValueError(f"NULL reproducibility_grade for run {run_id} - audit data corruption")
```

- [ ] Add test to catch mutations on this line

### Line 125

```python
     124 |         except ValueError:
>>>  125 |             raise ValueError(
     126 |                 f"Invalid reproducibility_grade '{current_grade}' for run {run_id} - "
```

- [ ] Add test to catch mutations on this line

### Line 126

```python
     125 |             raise ValueError(
>>>  126 |                 f"Invalid reproducibility_grade '{current_grade}' for run {run_id} - "
     127 |                 f"expected one of {[g.value for g in ReproducibilityGrade]}"
```

- [ ] Add test to catch mutations on this line

## src/elspeth/core/landscape/row_data.py

**Priority:** P2
**Survivors:** 2 unique lines (2 total mutants)

### Line 57

```python
      56 |     def __post_init__(self) -> None:
>>>   57 |         if self.state == RowDataState.AVAILABLE and self.data is None:
      58 |             raise ValueError("AVAILABLE state requires non-None data")
```

- [ ] Add test to catch mutations on this line

### Line 59

```python
      58 |             raise ValueError("AVAILABLE state requires non-None data")
>>>   59 |         if self.state != RowDataState.AVAILABLE and self.data is not None:
      60 |             raise ValueError(f"{self.state} state requires None data")
```

- [ ] Add test to catch mutations on this line

## src/elspeth/core/payload_store.py

**Priority:** P2
**Survivors:** 5 unique lines (6 total mutants)

### Line 27

```python
      26 |
>>>   27 |
      28 | @runtime_checkable
```

- [ ] Add test to catch mutations on this line

### Line 100

```python
      99 |         """
>>>  100 |         self.base_path = base_path
     101 |         self.base_path.mkdir(parents=True, exist_ok=True)
```

- [ ] Add test to catch mutations on this line

### Line 114 (2 mutants)

```python
     113 |         # Idempotent: skip if already exists
>>>  114 |         if not path.exists():
     115 |             path.parent.mkdir(parents=True, exist_ok=True)
```

- [ ] Add test to catch mutations on this line

### Line 128

```python
     127 |         path = self._path_for_hash(content_hash)
>>>  128 |         if not path.exists():
     129 |             raise KeyError(f"Payload not found: {content_hash}")
```

- [ ] Add test to catch mutations on this line

### Line 136

```python
     135 |         # allow an attacker to incrementally discover expected hashes
>>>  136 |         if not hmac.compare_digest(actual_hash, content_hash):
     137 |             raise IntegrityError(f"Payload integrity check failed: expected {content_hash}, got {actual_hash}")
```

- [ ] Add test to catch mutations on this line

## src/elspeth/engine/tokens.py

**Priority:** P2
**Survivors:** 5 unique lines (6 total mutants)

### Line 54

```python
      53 |         """
>>>   54 |         self._recorder = recorder
      55 |         self._payload_store = payload_store
```

- [ ] Add test to catch mutations on this line

### Line 75

```python
      74 |         """
>>>   75 |         # Store payload if payload_store is configured (audit requirement)
      76 |         payload_ref = None
```

- [ ] Add test to catch mutations on this line

### Line 79 (2 mutants)

```python
      78 |             # Use canonical_json to handle pandas/numpy types, Decimal, datetime, etc.
>>>   79 |             # This prevents TypeError crashes when source data contains non-primitive types
      80 |             payload_bytes = canonical_json(row_data).encode("utf-8")
```

- [ ] Add test to catch mutations on this line

### Line 80

```python
      79 |             # This prevents TypeError crashes when source data contains non-primitive types
>>>   80 |             payload_bytes = canonical_json(row_data).encode("utf-8")
      81 |             payload_ref = self._payload_store.store(payload_bytes)
```

- [ ] Add test to catch mutations on this line

### Line 260

```python
     259 |                 branch_name=parent_token.branch_name,  # Inherit branch
>>>  260 |             )
     261 |             for db_child, row_data in zip(db_children, expanded_rows, strict=True)
```

- [ ] Add test to catch mutations on this line

---

## src/elspeth/core/landscape/schema.py (Deferred)

**Survivors:** 191

These are SQLAlchemy table definitions. Most are likely equivalent mutants
(changes that don't affect runtime behavior) or require database integration
tests. Review after addressing P0/P1 items.
