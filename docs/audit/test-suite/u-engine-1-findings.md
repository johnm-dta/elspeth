# U-ENGINE-1 — Partial continuation findings

**Scope reviewed:** declaration/dispatch/processor/executor/sink-path unit tests from
`docs/audit/test-suite/CHUNKS.md`.
**Method:** local continuation pass using the quality-engineering criteria and
the ELSPETH coding standards. This was **not** the full five-lens wave used for
U-CONTRACTS-1, U-CORE-1, and I-1.
**Date:** 2026-05-14.

## Verdict

**Partial health read: mixed with strong local coverage.** The chunk contains
many high-value behavioral regressions for sink cleanup, boundary dispatch,
declaration contracts, and terminal-state error handling. The main defects found
in this continuation pass are concentrated rather than pervasive:

1. Audit hashes are sometimes tested for presence or shape instead of binding to
   the exact canonical payload.
2. The tail of `tests/unit/engine/test_processor.py` contains private outcome
   dataclass roll-call tests that mostly test construction/frozen dataclass
   mechanics, plus one weak `hasattr` assertion that can pass when the expected
   processor result is absent.
3. One sink-executor test accepts both raw and wrapped exceptions, so a
   framework-boundary regression can still go green.
4. Two TRANSFORM-mode flush dispatcher tests only prove that dispatch happened
   at least once, even though the production contract is exactly one batch-level
   dispatch.
5. Two no-transform pipeline-row processor tests inspect only the first terminal
   result, even though this path should produce exactly one result.
6. Source-node audit tests accept extra node-state rows and inspect the first
   unordered result instead of pinning the single expected source state.
7. GateExecutor audit-state tests allow duplicate or unrelated node-state
   completions as long as one expected-looking completion exists.
8. Boundary-dispatch source/sink role tests use synthetic plugins that do not
   satisfy production role detection, so the intended contracts may never apply.
9. One AggregationExecutor post-processing failure test accepts duplicate FAILED
   node-state or batch terminal writes.
10. SinkExecutor failsink cleanup tests count FAILED completions without proving
    the exact state identities closed.
11. One SinkExecutor artifact-registration test checks only call count, not the
    primary/failsink state, node, path, hash, and size bindings.

## Findings

### UENG-1 — Hash assertions prove presence, not binding

Evidence:

- `tests/unit/engine/test_executors.py:654-655` asserts
  `result.input_hash` and `result.output_hash` are not `None`.
- `tests/unit/engine/test_executors.py:5540-5541` repeats the same pattern for
  batch transform results.
- `tests/unit/engine/test_dependency_resolver.py:340-345` asserts that
  `_hash_settings_file()` returns a `sha256:` prefix and 64 hex characters, but
  never compares against `sha256(canonical_json(yaml.safe_load(file)))`.
- `tests/unit/engine/test_sink_executor_diversion.py:176` asserts diverted-token
  `error_hash` is not `None`.

Why this is faulty:

The audit hash invariant is a binding invariant. A constant hash, a hash of the
wrong payload, or a hash produced before/after the wrong transformation can still
be non-null and length-correct. This is the same hash-without-binding pattern
already seen in U-CORE-1 and I-1.

Filed:

- `elspeth-bd49237412` — engine chunk remediation.
- `elspeth-e0afd080cc` — shared hash-binding test infrastructure.

Remediation status:

- 2026-05-20: `elspeth-bd49237412` resolved for the cited engine chunk sites.
  Transform, batch-transform, dependency settings, and sink diversion hash
  assertions now compare against the exact `stable_hash`, canonical settings
  SHA-256, or diversion-reason hash material. A broad run of
  `tests/unit/engine/test_executors.py` still fails on the unrelated
  `TestPassThroughCrossCheck.test_cross_check_is_tier_1_registered`
  subclass assertion; the focused hash-binding tests pass.
- 2026-05-20: `elspeth-e0afd080cc` added shared hash-binding assertions in
  `tests.fixtures.audit_hashing` for stable hashes, prefixed canonical SHA-256,
  and HMAC-SHA256 signatures. The engine hash-binding sites and dependency
  settings hash test now use the shared helper surface.

### UENG-2 — Private processor outcome tests mostly test dataclass construction

Evidence:

- `tests/unit/engine/test_processor.py:4597-4678` constructs
  `_TransformContinue`, `_TransformTerminal`, `_GateContinue`, and
  `_GateTerminal`, then asserts assigned fields, `isinstance` disjointness, or
  frozen dataclass `AttributeError`.
- `tests/unit/engine/test_processor.py:4565` uses
  `hasattr(result, "outcome")` inside `assert result is None or (...)`, which
  can pass when no terminal result is produced and also violates the project
  `hasattr` ban.
- `tests/unit/engine/test_processor.py:4608` deliberately passes a list to
  `_TransformTerminal` despite the private type expecting a `RowResult`, then
  asserts that the list remains a list.

Why this is faulty:

The tests provide little confidence that `_process_single_token()` dispatches
transform and gate outcomes correctly. They are brittle to private refactors and
partly duplicate dataclass mechanics. The weak coalesce assertion is especially
bad: it can go green on absence of the expected outcome.

Filed:

- `elspeth-e4281a36d8`.

Remediation status:

- 2026-05-20: `elspeth-e4281a36d8` resolved. The weak coalesce assertion now
  verifies that the token is held at the coalesce node without emitting a
  terminal result, and the private outcome dataclass roll-call block was
  deleted.

### UENG-3 — Merge-failure test accepts the wrong exception boundary

Evidence:

- `tests/unit/engine/test_executors.py:3427` uses
  `pytest.raises((ContractMergeError, FrameworkBugError))` in
  `test_contract_merge_exception_crashes_before_write`.
- `src/elspeth/engine/executors/sink.py:438` explicitly wraps
  `ContractMergeError` as `FrameworkBugError` after adding sink-executor
  context.

Why this is faulty:

The test title says the sink write should crash before any write/flush/outcome
side effects, but the exception assertion also permits the raw lower-level
`ContractMergeError`. If production stops wrapping the merge failure at the
framework boundary, this test still passes.

Filed:

- `elspeth-462f50680f`.

Remediation status:

- 2026-05-20: `elspeth-462f50680f` resolved. The sink merge-failure test now
  requires `FrameworkBugError`, asserts the underlying cause is
  `ContractMergeError`, and preserves the no-write/no-flush/no-outcome checks.

### UENG-4 — TRANSFORM flush dispatcher tests allow duplicate dispatch

Evidence:

- `tests/unit/engine/test_flush_dispatcher_routing.py:226-240`
  `test_transform_mode_fires_dispatcher_once` asserts
  `len(_CountingContract.invocations) >= 1`.
- `tests/unit/engine/test_flush_dispatcher_routing.py:242-269`
  `test_transform_mode_passes_batch_intersection_as_effective_input_fields`
  uses the same `>= 1` cardinality check, then verifies every recorded
  invocation has `effective_input_fields == frozenset({"x"})`.
- `src/elspeth/engine/processor.py:900-920` routes TRANSFORM mode through a
  single batch-level `run_batch_flush_checks(...)` call with the buffered tokens
  and emitted rows.

Why this is faulty:

The test name and production path both encode a one-dispatch contract. A
regression that dispatches the same TRANSFORM batch multiple times with the same
field intersection would still satisfy `>= 1` and pass the field assertions.
The tests should assert exact cardinality while preserving the existing
`effective_input_fields` evidence.

Filed:

- `elspeth-f295b77e76`.

Remediation status:

- 2026-05-20: `elspeth-f295b77e76` resolved. The TRANSFORM-mode flush
  dispatcher tests now assert the exact single invocation, including triggering
  token identity and effective input-field intersection.

### UENG-5 — Pipeline-row processor tests allow duplicated terminal results

Evidence:

- `tests/unit/engine/test_processor_pipeline_row.py:117-152`
  `test_process_row_creates_pipeline_row` runs `process_row(...,
  transforms=[])`, asserts `len(results) >= 1`, and then inspects only
  `results[0]`.
- `tests/unit/engine/test_processor_pipeline_row.py:187-226`
  `test_process_existing_row_accepts_pipeline_row` repeats the same `>= 1`
  pattern for the resume/existing-row path.
- `src/elspeth/engine/processor.py:1799-1843` starts traversal at the source
  structural node when there is no first transform.
- `src/elspeth/engine/processor.py:2860-2865` skips structural nodes with no
  plugin, and `src/elspeth/engine/processor.py:2923-2924` returns the single
  terminal result.

Why this is faulty:

These are single-token, no-transform paths. A regression that duplicated the
terminal `RowResult` while leaving the first result correct would pass both
tests. The assertions should pin exact cardinality before checking the
`PipelineRow` payload and contract.

Filed:

- `elspeth-9a1262dbc7`.

Remediation status:

- 2026-05-20: `elspeth-9a1262dbc7` resolved. The no-transform
  `process_row()` and resume `process_existing_row()` tests now assert exactly
  one terminal result and destructure that singleton instead of inspecting
  `results[0]`.

### UENG-6 — Source-node audit tests allow extra node_state rows

Evidence:

- `tests/unit/engine/test_processor.py:555-581`
  `test_records_source_node_state` queries every `node_states` row for the run,
  asserts `len(states) >= 1`, and inspects `states[0]`.
- `tests/unit/engine/test_processor.py:583-632`
  `test_source_boundary_violation_records_failed_outcome_and_failed_source_state`
  uses the same `>= 1` / first-row pattern for FAILED source-boundary state.
- `tests/unit/engine/test_processor.py:829-889` repeats the pattern for the
  telemetry-failure and framework-bug source-boundary variants.
- `src/elspeth/engine/processor.py:1677-1713` records exactly one source
  node-state row per token via `_record_source_node_state(...)`.
- `src/elspeth/engine/processor.py:1715-1798` records the source-boundary
  failure as one FAILED token outcome plus one FAILED source node state.

Why this is faulty:

These tests are meant to prove the source audit pair is complete and terminal.
The current assertions would miss duplicate or extra source `node_states` rows
when the first fetched row happens to have the expected status. They should
assert exact cardinality and identify the source node state explicitly.

Filed:

- `elspeth-ff85897f8f`.

Remediation status:

- 2026-05-20: `elspeth-ff85897f8f` resolved. Source-node audit tests now
  filter for `node_id == "source-0"`, assert exactly one source state, and
  destructure that singleton before checking terminal status.

### UENG-7 — GateExecutor audit-state tests allow extra node-state completions

Evidence:

- `tests/unit/engine/test_executors.py:1730-1772`
  `test_config_gate_records_context_after` finds the first COMPLETED
  `complete_node_state` call and validates its `context_after`, but does not
  prove there was only one terminal completion.
- `tests/unit/engine/test_executors.py:1462-1487`
  `test_config_gate_unknown_route_label_raises_value_error` asserts
  `factory.execution.complete_node_state.call_count >= 1` and inspects the last
  call.
- `tests/unit/engine/test_executors.py:1576-1601`
  `test_config_gate_missing_route_resolution_fails_closed` uses the same
  `>= 1` / last-call pattern.
- `tests/unit/engine/test_executors.py:1603-1635`
  `test_config_gate_exception_records_failed_and_reraises` repeats that pattern.
- `tests/unit/engine/test_executors.py:4999-5080` filters for FAILED calls,
  asserts `len(failed_calls) >= 1`, and inspects the first matching error
  object.
- `src/elspeth/engine/executors/gate.py:249-341` wraps gate execution in a
  single `NodeStateGuard`.
- `src/elspeth/engine/executors/state_guard.py:97-107` begins one node state,
  `src/elspeth/engine/executors/state_guard.py:161-183` auto-completes that one
  state as FAILED on exception, and
  `src/elspeth/engine/executors/state_guard.py:227-259` completes one state on
  explicit success.

Why this is faulty:

These tests are trying to prove gate audit terminality and context recording.
The production invariant is not merely "some matching completion was recorded";
it is one terminal completion for the gate node state. Duplicate or unrelated
completions would still pass the current assertions.

Filed:

- `elspeth-314eb3552e`.

Remediation status:

- 2026-05-20: `elspeth-314eb3552e` resolved. GateExecutor audit-state tests now
  assert exactly one `complete_node_state` call for the node under test before
  checking terminal status, failure error payloads, and success
  `context_after`.

### UENG-8 — Boundary source/sink skip tests do not satisfy role detection

Evidence:

- `tests/unit/engine/test_boundary_dispatch_inputs.py:159-183`
  `test_run_boundary_checks_skips_sink_contract_for_source_plugin` constructs a
  `SourcePlugin` with `declared_guaranteed_fields`, but no `load()` method.
- `tests/unit/engine/test_boundary_dispatch_inputs.py:212-236`
  `test_run_boundary_checks_skips_source_contract_for_sink_plugin` constructs a
  `SinkPlugin` with `declared_required_fields`, but no `write()` or `flush()`
  methods.
- `src/elspeth/contracts/plugin_roles.py:91-99`
  `source_declared_guaranteed_fields(...)` returns `None` unless the class MRO
  defines `load`.
- `src/elspeth/contracts/plugin_roles.py:102-112`
  `sink_declared_required_fields(...)` returns `None` unless the class MRO
  defines both `write` and `flush`.
- `src/elspeth/engine/executors/source_guaranteed_fields.py:103-127` and
  `src/elspeth/engine/executors/sink_required_fields.py:120-144` use those role
  helpers in `applies_to(...)`.

Why this is faulty:

The tests are named as source-vs-sink boundary filtering checks, but the
synthetic plugins do not qualify as source or sink plugins under the production
role helpers. They can pass with zero production boundary contracts engaged.
They should use role-faithful synthetic plugins or instrument the applicable
contract so the intended role is known to run while the opposite role is skipped.

Filed:

- `elspeth-2744d69903`.

Remediation status:

- 2026-05-20: `elspeth-2744d69903` resolved. Boundary-dispatch source/sink
  role tests now use synthetic plugins with the MRO-visible `load()` or
  `write()`/`flush()` methods required by production role detection, and
  counting contract wrappers prove the intended boundary contract runs once
  while the opposite role is skipped.

### UENG-9 — Aggregation post-processing failure test allows duplicate terminal writes

Evidence:

- `tests/unit/engine/test_executors.py:4706-4759`
  `test_output_hash_failure_marks_state_and_batch_failed` filters
  `complete_node_state` calls for FAILED and asserts `len(failed_calls) >= 1`,
  then filters `complete_batch` calls for FAILED and asserts
  `len(batch_failed) >= 1`.
- `src/elspeth/engine/executors/aggregation.py:336-348` opens one
  `NodeStateGuard` for the flush.
- `src/elspeth/engine/executors/aggregation.py:398-416` raises
  `PluginContractViolation` on non-canonical output hash failure before explicit
  state completion.
- `src/elspeth/engine/executors/state_guard.py:161-183` auto-completes that
  single guarded node state as FAILED.
- `src/elspeth/engine/executors/aggregation.py:476-489` marks the batch FAILED
  once when cleanup runs and `batch_finalized` is still false.

Why this is faulty:

The test is proving a terminality/audit-integrity invariant. Duplicate FAILED
node-state writes or duplicate FAILED batch completions would be corruption, but
the current `>= 1` assertions still pass. The test should assert exact
cardinality and the single expected state/batch identifiers.

Filed:

- `elspeth-0c1c7d5cec`.

Remediation status:

- 2026-05-20: `elspeth-0c1c7d5cec` resolved. The aggregation
  post-processing failure regression now asserts the single guarded FAILED
  node-state completion and the single FAILED batch completion, binding both to
  the expected state and batch identifiers.

### UENG-10 — SinkExecutor cleanup tests do not prove state identity

Evidence:

- `tests/unit/engine/test_sink_executor_diversion.py:618-644` filters
  `complete_node_state` calls down to FAILED calls and asserts only
  `len(failed_calls) == 1` after reset/result-type failure paths.
- `tests/unit/engine/test_sink_executor_diversion.py:943-948` asserts only that
  three unique FAILED state IDs exist after a begin-node-state system error,
  without binding those IDs to the expected primary/failsink opened states.
- `tests/unit/engine/test_sink_executor_diversion.py:994-999` asserts
  `total_complete_calls > 2` after mid-loop `AuditIntegrityError` cleanup, but
  does not prove the remaining open state IDs were the ones closed as FAILED.
- `src/elspeth/engine/executors/sink.py:813-829`,
  `src/elspeth/engine/executors/sink.py:850-876`, and
  `src/elspeth/engine/executors/sink.py:934-963` track opened primary/failsink
  states and close specific remaining state identities.

Why this is faulty:

These tests are meant to prove audit cleanup for opened sink states. Count-only
assertions can pass if the implementation closes the wrong state, double-closes
an already-terminal state, or leaves a particular opened state dangling while
preserving the aggregate FAILED count.

Filed:

- `elspeth-eb12769648`.

Remediation status:

- 2026-05-20: `elspeth-eb12769648` resolved. Failsink cleanup tests now map
  opened primary/failsink node states to expected state IDs, assert exact
  FAILED/COMPLETED cleanup attempts and successful terminalizations, and verify
  cleanup error phase/context for reset, result-type, begin-state, and mid-loop
  audit-recording failures.

### UENG-11 — SinkExecutor artifact registration test checks only call count

Evidence:

- `tests/unit/engine/test_sink_executor_diversion.py:420-439`
  `test_both_artifacts_registered_in_mixed_batch` sets up a primary/failsink
  mixed batch and asserts only `execution.register_artifact.call_count == 2`.
- `src/elspeth/engine/executors/sink.py:633-643` registers the primary artifact
  against the first primary state and primary sink node.
- `src/elspeth/engine/executors/sink.py:965-975` registers the failsink artifact
  against the first failsink state and failsink node.

Why this is faulty:

Artifact registration is an audit-lineage invariant. A regression that registers
both artifacts against the same state, swaps primary and failsink nodes, or
reuses one artifact path/hash for both calls would still satisfy the current
call-count assertion.

Filed:

- `elspeth-975f45dfcb`.

Remediation status:

- 2026-05-20: `elspeth-975f45dfcb` resolved. The mixed primary/failsink
  artifact registration test now uses distinct artifact descriptors and asserts
  exact registration payloads for primary and failsink state IDs, sink node IDs,
  paths, hashes, and sizes.

## Filed issues

| ID | Title | Type | Priority |
|---|---|---|---|
| `elspeth-8bf288792a` | U-ENGINE-1 test audit remediation — processor/executor test quality sweep | epic | P2 |
| `elspeth-bd49237412` | Engine hash assertions only prove presence — executor/dependency tests don't bind hashes to canonical payloads | task | P2 |
| `elspeth-e4281a36d8` | test_processor.py tail block — private outcome dataclass roll-call tests mostly test construction, not processor behavior | task | P3 |
| `elspeth-e0afd080cc` | Hash-binding test infrastructure — shared helper for canonical hash assertions | feature | P3 |
| `elspeth-462f50680f` | test_executors.py merge-failure test accepts both raw and wrapped exceptions — wrong failure boundary can pass | task | P2 |
| `elspeth-f295b77e76` | Tighten TRANSFORM flush dispatcher cardinality tests — prevent duplicate batch dispatch from passing | task | P2 |
| `elspeth-9a1262dbc7` | Tighten pipeline-row processor cardinality tests — duplicated terminal results can pass unseen | task | P2 |
| `elspeth-ff85897f8f` | Tighten source-node audit state tests — extra node_states can pass as long as the first row matches | task | P2 |
| `elspeth-314eb3552e` | Tighten GateExecutor audit state tests — extra node-state completions can pass unseen | task | P2 |
| `elspeth-2744d69903` | Fix boundary-dispatch source/sink role tests — synthetic plugins do not satisfy production role detection | task | P2 |
| `elspeth-e984600f90` | Test mock discipline enforcement — require specs or fakes for behavioral mocks | feature | P3 |
| `elspeth-2f4978ffbc` | Test hasattr enforcement — extend banned-pattern gate beyond src/elspeth | feature | P3 |
| `elspeth-0c1c7d5cec` | Tighten AggregationExecutor post-processing failure tests — duplicate FAILED state or batch writes can pass | task | P2 |
| `elspeth-eb12769648` | SinkExecutor cleanup tests count failures without proving state identity — wrong audit states can pass | task | P2 |
| `elspeth-975f45dfcb` | SinkExecutor artifact registration test checks only call count — wrong artifact linkage can pass | task | P2 |

## Notable strengths preserved

- `tests/unit/engine/test_sink_executor_diversion.py` has strong terminal
  cleanup and diversion coverage.
- `tests/unit/engine/test_boundary_dispatch_inputs.py` directly covers boundary
  dispatch filtering.
- `tests/unit/engine/test_record_flush_violation_failure.py` distinguishes
  Landscape record failures from ordinary recorder bugs.
- `tests/unit/engine/test_executors.py` contains meaningful behavior coverage
  around SinkExecutor state cleanup and TransformExecutor terminality, despite
  the specific hash-binding weaknesses above.

## Remaining work

This partial pass should not be treated as completion of U-ENGINE-1. The full
chunk still needs a five-lens or equivalent deep review across the whole
`test_processor.py` and `test_executors.py` files, which together dominate the
line count.
