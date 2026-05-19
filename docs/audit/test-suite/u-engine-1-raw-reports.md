# U-ENGINE-1 — Raw continuation notes

This file preserves the local continuation notes for the 2026-05-14 partial
U-ENGINE-1 pass. Unlike prior completed chunks, this was not a five-agent wave.

## Commands and evidence used

- Loaded required ELSPETH standards:
  - `engine-patterns-reference`
  - `tier-model-deep-dive`
  - `logging-telemetry-policy`
  - `config-contracts-guide`
- Loaded quality-engineering references:
  - `using-quality-engineering`
  - `test-isolation-fundamentals`
  - `test-data-management`
  - `test-automation-architecture`
  - `test-maintenance-patterns`
  - `mutation-testing`
- Read existing audit artifacts:
  - `docs/audit/test-suite/README.md`
  - `docs/audit/test-suite/CHUNKS.md`
  - `docs/audit/test-suite/u-contracts-1-findings.md`
  - `docs/audit/test-suite/u-core-1-findings.md`
  - `docs/audit/test-suite/i-1-findings.md`
- Checked tracker state with:
  - `filigree session-context`
  - `filigree search "test quality" --json`
  - `filigree search "hash without binding" --json`
  - `filigree search "hasattr" --json`
  - `filigree search "test_processor" --json`

## Reviewed U-ENGINE-1 files

Primary evidence came from these files:

- `tests/unit/engine/test_processor.py`
- `tests/unit/engine/test_executors.py`
- `tests/unit/engine/test_dependency_resolver.py`
- `tests/unit/engine/test_sink_executor_diversion.py`
- `tests/unit/engine/test_boundary_dispatch_inputs.py`
- `tests/unit/engine/test_declaration_contract_bootstrap_drift.py`
- `tests/unit/engine/test_flush_dispatcher_routing.py`
- `tests/unit/engine/test_processor_pipeline_row.py`
- `tests/unit/engine/test_failsink_validation.py`
- `tests/unit/engine/test_can_drop_rows_contract.py`
- `tests/unit/engine/test_declared_required_fields_contract.py`
- `tests/unit/engine/test_declared_output_fields_contract.py`
- `tests/unit/engine/test_pass_through_declaration_contract.py`
- `tests/unit/engine/test_record_flush_violation_failure.py`
- `tests/unit/engine/test_state_guard_audit_evidence_discriminator.py`

The pass also scanned the broader U-ENGINE-1 file list for common smells:
`hasattr`, bare `Mock`/`MagicMock`, broad `pytest.raises`, skip/xfail, hash
presence checks, and regression-dump markers.

Spot checks of `test_failsink_validation.py`,
`test_declaration_contract_bootstrap_drift.py`,
`test_record_flush_violation_failure.py`, and
`test_state_guard_audit_evidence_discriminator.py` did not produce separate
findings in this continuation pass. Their remaining mock-use concerns are
covered by the suite-wide mock-discipline issue rather than per-file tickets.

## Raw findings

### Hash-without-binding

`tests/unit/engine/test_executors.py:654-655` and `:5540-5541` assert input and
output hashes are populated, but not that they equal the expected hash of the
input/output payload.

`tests/unit/engine/test_dependency_resolver.py:340-345` asserts the dependency
settings hash has a `sha256:` prefix and 64 hex characters. It does not compare
against an independently computed digest over canonical JSON.

`tests/unit/engine/test_sink_executor_diversion.py:176` asserts an error hash is
present but not bound to the recorded error material.

### Processor private outcome block

`tests/unit/engine/test_processor.py:4565` contains:

```python
assert result is None or (hasattr(result, "outcome") and result is not None)
```

This is weak because `None` passes and because `hasattr` is banned by the
project coding standards.

`tests/unit/engine/test_processor.py:4597-4678` constructs private frozen
dataclasses and asserts direct field reads, `isinstance` disjointness, and
`AttributeError` on mutation. Most of this is dataclass/type-shape coverage, not
processor behavior.

`tests/unit/engine/test_processor.py:4608` intentionally sends a list into
`_TransformTerminal` and asserts that the list remains a list, despite the
private type's declared result shape being a `RowResult`.

### Broad exception boundary in sink executor test

`tests/unit/engine/test_executors.py:3427` contains:

```python
with pytest.raises((ContractMergeError, FrameworkBugError)):
```

The source at `src/elspeth/engine/executors/sink.py:438` wraps
`ContractMergeError` into `FrameworkBugError`. Accepting both exception types
would miss a regression where the wrapper boundary is lost.

### TRANSFORM flush dispatcher cardinality

`tests/unit/engine/test_flush_dispatcher_routing.py:226-240` names the contract
as "fires dispatcher once", but only asserts:

```python
assert len(_CountingContract.invocations) >= 1, "TRANSFORM-flush bypassed dispatcher"
```

`tests/unit/engine/test_flush_dispatcher_routing.py:242-269` repeats the same
`>= 1` check for the batch-intersection path and then checks every recorded
invocation has `effective_input_fields == frozenset({"x"})`.

The production TRANSFORM path at `src/elspeth/engine/processor.py:900-920`
computes the batch intersection once and calls `run_batch_flush_checks(...)`
once with `buffered_tokens=tuple(fctx.buffered_tokens)` and
`emitted_rows=tuple(emitted)`. A duplicate-dispatch regression with the same
field intersection would pass both tests.

### Pipeline-row processor cardinality

`tests/unit/engine/test_processor_pipeline_row.py:117-152` runs the source-row
path with `transforms=[]`, then asserts `len(results) >= 1` and inspects
`results[0]`.

`tests/unit/engine/test_processor_pipeline_row.py:187-226` repeats the same
pattern for `process_existing_row(...)` in the resume path.

The no-transform source path starts traversal at the source structural node
(`src/elspeth/engine/processor.py:1799-1843`), skips structural nodes without a
plugin (`src/elspeth/engine/processor.py:2860-2865`), and then returns the
terminal token result (`src/elspeth/engine/processor.py:2923-2924`). The tests
should assert exact cardinality before checking the first result's `PipelineRow`.

### Source-node audit state cardinality

`tests/unit/engine/test_processor.py:555-581` queries every `node_states` row for
the run, asserts `len(states) >= 1`, and then checks `states[0].status ==
NodeStateStatus.COMPLETED`.

`tests/unit/engine/test_processor.py:583-632` uses the same pattern for source
boundary validation failure: `len(states) >= 1`, then `states[0].status ==
NodeStateStatus.FAILED`, while separately asserting exactly one token outcome.

`tests/unit/engine/test_processor.py:829-889` repeats the same pattern for the
telemetry-failure and framework-bug source-boundary paths.

The source audit implementation records exactly one source node state per token
through `_record_source_node_state(...)` (`src/elspeth/engine/processor.py:1677-1713`).
For boundary failures, `_record_source_boundary_failure(...)` records one FAILED
token outcome and one FAILED source node state
(`src/elspeth/engine/processor.py:1715-1798`). Extra source node-state rows would
therefore be audit corruption, but these tests can miss them.

### GateExecutor node-state cardinality

`tests/unit/engine/test_executors.py:1730-1772` exercises the successful
GateExecutor path, finds the first COMPLETED `complete_node_state` call, and
checks its `context_after`. It does not prove there was only one terminal
completion.

`tests/unit/engine/test_executors.py:1462-1487`,
`tests/unit/engine/test_executors.py:1576-1601`, and
`tests/unit/engine/test_executors.py:1603-1635` each exercise a failing
`GateExecutor.execute_config_gate(...)` path, then assert
`complete_node_state.call_count >= 1` and inspect the last completion call.

`tests/unit/engine/test_executors.py:4999-5080` filters for FAILED completions,
asserts `len(failed_calls) >= 1`, and inspects the first matching error object
to verify `ExecutionError.to_dict()` serialization.

The production path at `src/elspeth/engine/executors/gate.py:249-341` wraps gate
execution in one `NodeStateGuard`. The guard begins one node state
(`src/elspeth/engine/executors/state_guard.py:97-107`) and either auto-completes
that one state as FAILED on exception
(`src/elspeth/engine/executors/state_guard.py:161-183`) or explicitly completes
that one state on success (`src/elspeth/engine/executors/state_guard.py:227-259`).
The tests should therefore pin exact terminal completion cardinality for the
gate node, not merely require some expected-looking completion.

### Boundary source/sink role-test false positives

`tests/unit/engine/test_boundary_dispatch_inputs.py:159-183` names a source
plugin skip scenario, but the synthetic `SourcePlugin` only has
`declared_guaranteed_fields`; it does not define `load()`.

`tests/unit/engine/test_boundary_dispatch_inputs.py:212-236` names a sink plugin
skip scenario, but the synthetic `SinkPlugin` only has
`declared_required_fields`; it does not define `write()` or `flush()`.

Production role detection requires those methods. `source_declared_guaranteed_fields`
returns `None` unless the class MRO defines `load`
(`src/elspeth/contracts/plugin_roles.py:91-99`), and
`sink_declared_required_fields` returns `None` unless the class MRO defines both
`write` and `flush` (`src/elspeth/contracts/plugin_roles.py:102-112`). The
boundary contracts' `applies_to(...)` methods delegate to those helpers
(`src/elspeth/engine/executors/source_guaranteed_fields.py:103-127` and
`src/elspeth/engine/executors/sink_required_fields.py:120-144`). These tests can
therefore pass without proving either production contract applied or was skipped
for the intended role reason.

### Cross-chunk mock discipline

The U-ENGINE-1 scan again found spec-less behavioral mocks, matching the pattern
already recorded from U-CONTRACTS-1, U-CORE-1, and I-1. Examples include the
mock factory and repository return objects in
`tests/unit/engine/test_processor_pipeline_row.py:36-38`, plus many ad hoc
executor factory, transform, sink, and recorder-state mocks in
`tests/unit/engine/test_executors.py`.

Filed suite-wide shared infrastructure feature `elspeth-e984600f90` rather than
adding per-file issues for the same structural problem.

### Cross-chunk hasattr enforcement

The U-ENGINE-1 weak processor assertion at `tests/unit/engine/test_processor.py:4576`
adds another current example to the `hasattr()` pattern already recorded in
U-CONTRACTS-1 and U-CORE-1. A current check found that
`scripts/cicd/enforce_tier_model.py` already defines banned R3 `hasattr`
detection and unit tests for it, but pre-commit/CI invoke the check with
`--root src/elspeth`, so tests are outside that gate.

Filed suite-wide shared infrastructure feature `elspeth-2f4978ffbc` to extend
the banned-pattern gate beyond `src/elspeth` or add a test-specific equivalent.

### AggregationExecutor post-processing failure cardinality

`tests/unit/engine/test_executors.py:4706-4759` simulates an output-hash failure
after a batch transform returns success. It verifies terminality by checking:

```python
assert len(failed_calls) >= 1
assert len(batch_failed) >= 1
```

The production path opens one `NodeStateGuard`
(`src/elspeth/engine/executors/aggregation.py:336-348`), raises
`PluginContractViolation` during post-processing output hashing
(`src/elspeth/engine/executors/aggregation.py:398-416`), lets the guard
auto-complete that one state as FAILED
(`src/elspeth/engine/executors/state_guard.py:161-183`), and then marks the batch
FAILED once in cleanup while `batch_finalized` is false
(`src/elspeth/engine/executors/aggregation.py:476-489`).

Duplicate FAILED node-state completions or duplicate FAILED batch writes would
be audit corruption, but this test currently accepts them.

### SinkExecutor cleanup state identity

`tests/unit/engine/test_sink_executor_diversion.py:618-644` covers two
failsink setup/result-type failure paths, but filters `complete_node_state` down
to FAILED calls and asserts only `len(failed_calls) == 1`. It does not identify
which state was closed.

`tests/unit/engine/test_sink_executor_diversion.py:943-948` asserts that three
unique state IDs were completed as FAILED after a failsink begin-node-state
system error, but does not bind those state IDs back to the expected opened
primary-divert and failsink states.

`tests/unit/engine/test_sink_executor_diversion.py:994-999` asserts
`total_complete_calls > 2` after mid-loop `AuditIntegrityError` cleanup. That
proves cleanup attempted something, but not that the remaining open states were
the ones completed as FAILED.

The source paths at `src/elspeth/engine/executors/sink.py:813-829`,
`:850-876`, and `:934-963` are identity-sensitive cleanup paths. The tests
should map begin calls to state IDs and assert exact terminalization.

### SinkExecutor artifact registration linkage

`tests/unit/engine/test_sink_executor_diversion.py:420-439` checks mixed
primary/failsink artifact recording with only:

```python
assert execution.register_artifact.call_count == 2
```

The production path registers the primary artifact with the first primary state
and primary sink node at `src/elspeth/engine/executors/sink.py:633-643`, then
registers the failsink artifact with the first failsink state and failsink node
at `src/elspeth/engine/executors/sink.py:965-975`. A test that only counts calls
would miss swapped state IDs, wrong sink node IDs, duplicated path/hash payloads,
or missing size bindings.
