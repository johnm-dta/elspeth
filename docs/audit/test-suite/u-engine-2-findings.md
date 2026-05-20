# U-ENGINE-2 — Partial findings

**Scope reviewed:** first-pass U-ENGINE-2 slices from `docs/audit/test-suite/CHUNKS.md`,
starting with bootstrap/preflight and commencement gate tests, then continuing
through selected orchestrator outcome and span tests.
**Method:** local continuation pass using the quality-engineering criteria and
the ELSPETH coding standards. This was **not** a full five-lens wave.
**Date:** 2026-05-14.

## Verdict

**Partial health read: early but already useful.** The reviewed slices have
mostly meaningful unit coverage, but several boundary contracts are not
mechanically protected:

1. `resolve_preflight()` validates malformed commencement gate expressions
   before dependency pipelines can run, but the current tests do not pin that
   ordering when both `depends_on` and `commencement_gates` are configured.
2. `SpanFactory` real-tracer tests use `pytest.importorskip("opentelemetry")`
   even though OpenTelemetry is a mandatory dependency, so mandatory tracing
   coverage can degrade into skips.
3. Coalesce continuation tests never exercise the case where a merged outcome
   has no matching `coalesce_node_map` entry; the current boundary would leak a
   raw `KeyError` instead of a typed orchestration invariant.
4. Trigger checkpoint restore tests do not pin immediate behavior when restored
   `batch_count` already exceeds the count threshold but no count-fire offset
   was persisted.

## Findings

### UENG2-1 — resolve_preflight ordering gap lets dependency side effects precede malformed-gate rejection in tests

Evidence:

- `src/elspeth/engine/bootstrap.py:48-55` documents and performs early
  `validate_gate_expressions(config.commencement_gates)` before dependency
  resolution.
- `src/elspeth/engine/bootstrap.py:57-86` performs cycle detection and
  dependency execution after that validation step.
- `src/elspeth/engine/commencement.py:55-68` defines
  `validate_gate_expressions()` and documents that it rejects malformed
  expressions before sub-pipelines mutate external state.
- `tests/unit/engine/test_bootstrap_preflight.py:39-69` verifies dependencies
  resolve when configured.
- `tests/unit/engine/test_bootstrap_preflight.py:70-97` verifies gates evaluate
  when configured.
- `tests/unit/engine/test_bootstrap_preflight.py:135-157` verifies
  `CommencementGateFailedError` propagation during evaluation.
- `tests/unit/engine/test_bootstrap_preflight.py:184-206` verifies duplicate
  dependency names stop before dependency execution, but does not cover
  malformed gates with dependencies present.

Why this is faulty:

The production invariant is ordering-sensitive and side-effect-sensitive. A
mutation that moves `validate_gate_expressions()` after `resolve_dependencies()`
could run dependency pipelines and mutate external state before rejecting the
malformed gate while still passing the current dependency, gate, and duplicate
name tests.

Filed:

- `elspeth-68cd1876d0`.

### UENG2-2 — SpanFactory tests can skip mandatory OpenTelemetry coverage

Evidence:

- `pyproject.toml:54-57` lists `opentelemetry-api`, `opentelemetry-sdk`, and
  `opentelemetry-exporter-otlp` as core dependencies.
- `tests/unit/engine/test_spans.py:42-63` uses
  `pytest.importorskip("opentelemetry")` around the first real-tracer tests.
- The same file contains 27 `pytest.importorskip("opentelemetry")` calls across
  real span attribute/name coverage.
- `src/elspeth/engine/spans.py:61-94` is the production branch that uses a real
  tracer; the no-op branch is covered separately without this dependency gate.

Why this is faulty:

OpenTelemetry is not optional for this project install, so the engine span tests
should fail if the dependency or SDK import path is broken. The current skip
guards can turn real-tracer regressions into skipped tests, leaving only no-op
span behavior exercised.

Filed:

- `elspeth-aa7781f802`.

### UENG2-3 — Coalesce continuation node-map drift is not pinned by tests

Evidence:

- `src/elspeth/engine/orchestrator/outcomes.py:397` indexes
  `coalesce_node_map[coalesce_name]` directly while processing merged coalesce
  continuations.
- `src/elspeth/engine/orchestrator/outcomes.py:445-454` and `:487-501` pass
  that map through from timeout and flush handling.
- `tests/unit/engine/orchestrator/test_outcomes.py:697` always builds a matching
  `CoalesceName("merge_1")` map for timeout merge tests; `:888` and `:928` do
  the same for flush merge tests.
- The empty-map tests at `tests/unit/engine/orchestrator/test_outcomes.py:970`,
  `:1003`, `:1027`, `:1123`, and `:1173` cover failure or no-op paths where
  `_process_merged_coalesce_outcome()` is not reached.

Why this is faulty:

Graph/registration drift between `CoalesceExecutor` and the orchestrator's
`coalesce_node_map` is an engine invariant. The tests currently cover only the
happy merged path and failure/no-op paths, so a missing map entry would surface
as an unhelpful `KeyError` instead of a typed `OrchestrationInvariantError`
with coalesce context.

Filed:

- `elspeth-c5add729fa`.

### UENG2-4 — Trigger restore over-threshold count is tested only after another accept

Evidence:

- `src/elspeth/engine/triggers.py:290-306` restores `_batch_count` directly and
  sets `_count_fire_time = None` when `count_fire_offset` is absent.
- `src/elspeth/engine/triggers.py:163-165` only considers the count trigger when
  `_count_fire_time` is not `None`.
- `tests/unit/engine/test_triggers.py:841-868` verifies missing fire offsets do
  not spuriously trigger when restored count is below threshold.
- `tests/unit/engine/test_triggers.py:1169-1202` covers restored count already
  past threshold only after one extra `record_accept()`, not immediately after
  restore.

Why this is faulty:

Resume can restore aggregation batches from old or partial checkpoint data where
`batch_count >= count` but `count_fire_offset` is missing. The current tests
allow that state to remain untriggered until another row arrives, which can
delay or mask count-trigger flushing when no more rows arrive.

Filed:

- `elspeth-786291485f`.

## Filed issues

| ID | Title | Type | Priority | Status |
|---|---|---|---|---|
| `elspeth-958f307f29` | U-ENGINE-2 test audit remediation — coalesce, orchestrator, retry, and runtime boundary sweep | epic | P2 | Ready to close after child closeout |
| `elspeth-68cd1876d0` | resolve_preflight ordering gap — malformed commencement gates can run dependency pipelines in tests | task | P2 | Closed 2026-05-20 |
| `elspeth-aa7781f802` | SpanFactory OpenTelemetry tests importorskip a mandatory dependency — tracing regressions can be skipped | task | P2 | Closed 2026-05-20 |
| `elspeth-c5add729fa` | Coalesce continuation tests omit missing node-map invariant — raw KeyError can escape | task | P2 | Closed 2026-05-20 |
| `elspeth-786291485f` | Trigger restore tests miss already-over-threshold count without fire offset — resumed batches can wait for another row | task | P2 | Closed 2026-05-20 |

## Remediation status

The filed U-ENGINE-2 remediation children were resolved in branch
`fix/tests-audit-20260520`:

- `resolve_preflight()` now has a boundary regression proving malformed
  commencement gates fail before dependency cycle detection, dependency
  execution, or gate evaluation.
- `test_spans.py` no longer skips mandatory OpenTelemetry SDK coverage; missing
  OpenTelemetry now fails at the direct import sites.
- Merged coalesce timeout/flush continuations now raise
  `OrchestrationInvariantError` with coalesce-name context when
  `coalesce_node_map` is missing the reported coalesce.
- Restored count-trigger batches with old checkpoints lacking
  `count_fire_offset` now trigger immediately when `batch_count >= count`, while
  below-threshold missing offsets remain unfired.

## Remaining work

This is still a partial U-ENGINE-2 pass. The chunk needs deeper review of the
remaining coalesce, orchestrator, token, trigger, retry, and batch tests.
