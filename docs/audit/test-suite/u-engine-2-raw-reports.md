# U-ENGINE-2 — Raw continuation notes

This file preserves local continuation notes for the 2026-05-14 partial
U-ENGINE-2 pass. Unlike the completed chunks, this was not a five-agent wave.

## Commands and evidence used

- Loaded required ELSPETH standards:
  - `engine-patterns-reference`
  - `tier-model-deep-dive`
  - `logging-telemetry-policy`
  - `config-contracts-guide`
- Loaded quality-engineering references:
  - `using-quality-engineering`
  - `test-maintenance-patterns`
  - `mutation-testing`
  - `test-automation-architecture`
  - `test-isolation-fundamentals`
- Ran `filigree session-context`.
- Checked tracker duplication with:
  - `filigree search "validate_gate_expressions" --json`
  - `filigree search "malformed gate dependency resolution" --json`
  - `filigree search "U-ENGINE-2" --json`
- Continued with targeted U-ENGINE-2 source/test reads and smell scans:
  - `nl -ba tests/unit/engine/orchestrator/test_outcomes.py`
  - `nl -ba src/elspeth/engine/orchestrator/outcomes.py`
  - `rg -n "coalesce_node_map|KeyError|CoalesceOutcome" ...`
  - `nl -ba tests/unit/engine/test_spans.py`
  - `nl -ba src/elspeth/engine/spans.py`
  - `rg -n "opentelemetry|pytest.importorskip|test_spans" ...`
  - `nl -ba tests/unit/engine/test_triggers.py`
  - `nl -ba src/elspeth/engine/triggers.py`
  - `rg -n "count_fire_offset|condition_fire_offset|restore_from_checkpoint|TriggerEvaluator" ...`

## Reviewed U-ENGINE-2 files

Primary evidence came from these files:

- `tests/unit/engine/test_bootstrap_preflight.py`
- `tests/unit/engine/test_commencement.py`
- `tests/integration/pipeline/test_bootstrap_preflight.py`
- `src/elspeth/engine/bootstrap.py`
- `src/elspeth/engine/commencement.py`
- `tests/unit/engine/test_adr019_phase2_producer_pairs.py`
- `tests/unit/engine/orchestrator/test_outcomes.py`
- `src/elspeth/engine/orchestrator/outcomes.py`
- `tests/unit/engine/test_spans.py`
- `src/elspeth/engine/spans.py`
- `pyproject.toml`
- `tests/unit/engine/test_triggers.py`
- `src/elspeth/engine/triggers.py`

The pass also ran a broad smell scan across U-ENGINE-2 files for common weak
assertion patterns: presence-only assertions, call counts, `hasattr`,
`Mock`/`MagicMock`, and broad exception assertions.

## Raw findings

### resolve_preflight malformed-gate ordering gap

`src/elspeth/engine/bootstrap.py:48-55` validates commencement gate expressions
before dependency resolution:

```python
if config.commencement_gates:
    from elspeth.engine.commencement import validate_gate_expressions

    validate_gate_expressions(config.commencement_gates)
```

That ordering matters because `src/elspeth/engine/bootstrap.py:57-86` then calls
`detect_cycles(...)` and `resolve_dependencies(...)`, where dependency pipelines
can execute and mutate external state.

`src/elspeth/engine/commencement.py:55-68` documents this explicitly:
`validate_gate_expressions()` is called before dependency resolution so
malformed expressions are rejected before sub-pipelines run.

Existing coverage:

- `tests/unit/engine/test_bootstrap_preflight.py:39-69` verifies dependency
  resolution is called when `depends_on` is configured.
- `tests/unit/engine/test_bootstrap_preflight.py:70-97` verifies gates are
  evaluated when only `commencement_gates` is configured.
- `tests/unit/engine/test_bootstrap_preflight.py:135-157` verifies gate
  evaluation failures propagate.
- `tests/unit/engine/test_bootstrap_preflight.py:184-206` verifies duplicate
  dependency names are rejected before dependency execution.
- `tests/unit/engine/test_commencement.py` verifies gate evaluation semantics
  and non-boolean result rejection.

Missing coverage:

There is no test where both `depends_on` and a malformed
`commencement_gates.condition` are configured, asserting the malformed gate
raises before `detect_cycles()` and `resolve_dependencies()` are called. A
mutation that moves `validate_gate_expressions()` after dependency execution
would pass the current tests while violating the source-level side-effect
ordering contract.

### SpanFactory real-tracer tests importorskip a mandatory dependency

`pyproject.toml:54-57` includes the OpenTelemetry API, SDK, and OTLP exporter in
the main dependency list:

```toml
"opentelemetry-api>=1.40,<2",
"opentelemetry-sdk>=1.40,<2",
"opentelemetry-exporter-otlp>=1.40,<2",
```

`tests/unit/engine/test_spans.py` still gates real-tracer coverage with
`pytest.importorskip("opentelemetry")`. The first examples are at
`tests/unit/engine/test_spans.py:42-63`; a count over the file found 27
`pytest.importorskip("opentelemetry")` calls.

`src/elspeth/engine/spans.py:61-94` has a real tracer branch and a no-op branch.
The no-op branch is exercised without OpenTelemetry, but the real tracer branch
should fail loudly if the mandatory SDK is absent or broken. The importorskip
pattern would instead hide that environment/configuration failure and skip the
real span name/attribute coverage.

### Coalesce continuation missing node-map invariant

`src/elspeth/engine/orchestrator/outcomes.py:397` directly indexes the map:

```python
coalesce_node_id = coalesce_node_map[coalesce_name]
```

The timeout and flush handlers pass that map into the helper at
`src/elspeth/engine/orchestrator/outcomes.py:445-454` and `:487-501`.

Existing tests cover the happy merged paths with a matching map:

- `tests/unit/engine/orchestrator/test_outcomes.py:697` creates
  `{CoalesceName("merge_1"): NodeID("coalesce::merge_1")}` for timeout tests.
- `tests/unit/engine/orchestrator/test_outcomes.py:888` and `:928` create the
  same mapping for flush tests.

Existing empty-map tests are not enough:

- `tests/unit/engine/orchestrator/test_outcomes.py:970`, `:1003`, `:1027`,
  `:1123`, and `:1173` pass an empty map, but only through failure/no-op paths
  where no merged continuation is processed.

Missing coverage:

There is no test for a merged coalesce outcome whose `coalesce_name` is absent
from `coalesce_node_map`. That is graph/registration drift and should be a
typed orchestration invariant. Today it would escape as a raw `KeyError`.

### Trigger restore over-threshold count without fire offset

`src/elspeth/engine/triggers.py:290-306` restores checkpoint state directly:

```python
self._batch_count = batch_count
...
if count_fire_offset is not None:
    self._count_fire_time = self._first_accept_time + count_fire_offset
else:
    self._count_fire_time = None
```

`src/elspeth/engine/triggers.py:163-165` only adds the count trigger candidate
when `_count_fire_time is not None`. It does not synthesize a count candidate
from `batch_count >= config.count` after restore.

Existing coverage:

- `tests/unit/engine/test_triggers.py:841-868` restores
  `batch_count=10`, `count=50`, and `count_fire_offset=None`, then asserts no
  trigger fires. That is useful below-threshold coverage.
- `tests/unit/engine/test_triggers.py:1169-1202` restores
  `batch_count=5`, `count=3`, and `count_fire_offset=None`, but then advances
  time and calls `record_accept()` before asserting the count trigger fires.

Missing coverage:

There is no immediate post-restore assertion for `batch_count >= count` with a
missing `count_fire_offset`. The current implementation would report no count
trigger until another row is accepted. If no more rows arrive, a resumed batch
can wait for timeout or end-of-source rather than honoring the count trigger
already satisfied by restored state.
