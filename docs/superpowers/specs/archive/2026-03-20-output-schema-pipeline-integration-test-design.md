# Output Schema Contract — Full-Pipeline Integration Test

**Date:** 2026-03-20
**Status:** Implemented
**Scope:** Integration test for output schema contract enforcement through `ExecutionGraph.from_plugin_instances()`
**Prerequisite:** Output schema contract enforcement (spec: `2026-03-20-output-schema-contract-enforcement-design.md`) must be implemented first (Tasks 1-3). Test 4 specifically depends on `_validate_output_schema_contract` being added to the builder (enforcement plan Task 3).

## Problem Statement

The output schema contract enforcement spec added `_validate_output_schema_contract` and `_output_schema_config` propagation through the DAG builder. The existing tests verify these mechanisms with:
- Unit tests using stubs (Task 3 of the enforcement plan)
- Contract invariant tests with real transform instances calling `_validate_output_schema_contract` directly (Task 5)

Neither exercises the full production path: real transform constructor → `_output_schema_config` populated → `from_plugin_instances()` → DAG builder → NodeInfo propagation → `validate_edge_compatibility()` field resolution. A defect in any wiring step (e.g., `build_execution_graph` fails to read the attribute, or `get_effective_guaranteed_fields` doesn't resolve it) would pass all existing tests but fail in production.

## Design

### Approach: Hybrid — Real Transforms + Mocks

**Real transforms** prove the end-to-end mechanism works: a real `RAGRetrievalTransform` constructor populates `_output_schema_config`, which flows through `from_plugin_instances()` into NodeInfo and becomes queryable via `get_effective_guaranteed_fields()`.

**Mock transforms** exercise edge validation scenarios without coupling to specific plugin constructors. Precise control over `guaranteed_fields` and `required_input_fields` lets us test happy path, missing field rejection, empty guarantees, and enforcement failure.

### Test File

`tests/integration/core/dag/test_output_schema_pipeline.py`

Located under `integration/core/dag/` because it exercises the production assembly path (`from_plugin_instances()`), not any specific transform. The directory `tests/integration/core/dag/` does not exist yet — the implementation must create it with appropriate `__init__.py` files.

### Mock Infrastructure

Three lightweight mocks following the existing pattern from `tests/unit/core/test_dag_schema_propagation.py` (protocol-compliant via duck typing, no `BaseTransform` inheritance):

#### `MockSource`

```python
class MockSource:
    name = "mock_source"
    output_schema = None
    config: ClassVar[dict[str, Any]] = {"schema": {"mode": "observed"}}
    _on_validation_failure = "discard"
    on_success = "output"
```

#### `MockSink`

```python
class MockSink:
    name = "mock_sink"
    input_schema = None
    config: ClassVar[dict[str, Any]] = {}
```

#### `MockFieldAddingTransform`

The workhorse mock. Constructor parameters control `declared_output_fields`, `_output_schema_config` (built from `guaranteed_fields`), and `required_input_fields` (merged into `.config` as a top-level key).

```python
_SENTINEL = object()


class MockFieldAddingTransform:
    """Mock transform with configurable output fields and required inputs.

    The DAG builder reads _output_schema_config from the transform instance
    and populates NodeInfo. The graph's get_required_fields() reads
    required_input_fields from node_info.config as a TOP-LEVEL key
    (graph.py lines 1459-1463) — NOT nested under "options".
    """

    input_schema = None
    output_schema = None
    on_error: str | None = None
    on_success: str | None = "output"

    def __init__(
        self,
        name: str,
        *,
        guaranteed_fields: tuple[str, ...] = (),
        declared_output_fields: frozenset[str] = frozenset(),
        required_input_fields: list[str] | None = None,
        output_schema_config_override: Any = _SENTINEL,
    ) -> None:
        self.name = name
        self.declared_output_fields = declared_output_fields

        # Build _output_schema_config from guaranteed_fields unless explicitly overridden.
        # _SENTINEL distinguishes "not provided" from "explicitly set to None".
        # Tests 3 and 4 pass output_schema_config_override=None to force _output_schema_config
        # to None even when the default logic would set it differently.
        if output_schema_config_override is not _SENTINEL:
            self._output_schema_config = output_schema_config_override
        elif guaranteed_fields:
            self._output_schema_config = SchemaConfig(
                mode="observed",
                fields=None,
                guaranteed_fields=guaranteed_fields,
            )
        else:
            self._output_schema_config = None

        # config must have required_input_fields as a TOP-LEVEL key.
        # graph.py get_required_fields() reads node_info.config.get("required_input_fields")
        # for non-aggregation nodes (lines 1459-1463). It does NOT check inside "options".
        self.config: dict[str, Any] = {"schema": {"mode": "observed"}}
        if required_input_fields is not None:
            self.config["required_input_fields"] = required_input_fields
```

Note: `_SENTINEL` is a module-level singleton used to distinguish "caller didn't pass `output_schema_config_override`" from "caller explicitly passed `None`". The parameter is typed as `Any` to avoid mypy complaints about `object()` not matching `SchemaConfig | None`.

### Test Scenarios

#### Important: `validate_edge_compatibility()` is called automatically

`build_execution_graph()` calls `graph.validate_edge_compatibility()` at builder.py line 923 before returning the graph. This means `from_plugin_instances()` itself raises `GraphValidationError` for edge violations — there is no separate validation step. All tests assert on the behaviour of `from_plugin_instances()` directly.

#### Group 1: Real Transform — Contract Propagation End-to-End

One test using `RAGRetrievalTransform`:

| Test | Setup | Assertion |
|------|-------|-----------|
| `test_rag_output_schema_propagates_through_pipeline` | Construct RAG with config dict (see below), wire via `WiredTransform`, build graph via `from_plugin_instances()` | Graph builds successfully. Find transform NodeInfo via `[n for n in graph.get_nodes() if n.plugin_name == "rag_retrieval"]`. Assert `node_info.output_schema_config is not None`. Assert `graph.get_effective_guaranteed_fields(node_id)` returns a frozenset containing `{"sci__rag_context", "sci__rag_score", "sci__rag_count", "sci__rag_sources"}`. |

**RAG config dict:** `RAGRetrievalTransform.__init__` takes a single `config: dict[str, Any]` argument (not keyword args). The minimum viable config:

```python
{
    "output_prefix": "sci",
    "query_field": "question",
    "provider": "chroma",
    "provider_config": {"collection": "test-col", "mode": "ephemeral"},
    "schema": {"mode": "observed"},
}
```

The constructor only parses config and builds `_output_schema_config` — it does not connect to a backend. `on_start()` (which creates the actual provider connection) is never called in this test. The config just needs to parse without error.

Requires `ELSPETH_FINGERPRINT_KEY` env var (autouse fixture via `monkeypatch.setenv`).

This proves the full chain: real constructor → `_build_output_schema_config()` → builder reads `_output_schema_config` → NodeInfo stores it → graph methods resolve it. One real transform is sufficient — the per-transform unit pinning tests in the enforcement plan cover the other 5 transforms' `_output_schema_config` content.

#### Group 2: Mock Transforms — Edge Validation Scenarios

All tests assert on the behaviour of `from_plugin_instances()` directly. Since `validate_edge_compatibility()` is called automatically inside `build_execution_graph()` (builder.py line 923), tests 2 and 3 expect `from_plugin_instances()` to raise `GraphValidationError`. Test 4 expects `FrameworkBugError` from the enforcement check, which runs earlier (during node addition, before edge validation).

| Test | Producer Setup | Consumer Setup | Assertion |
|------|---------------|----------------|-----------|
| `test_edge_validation_passes_when_fields_satisfied` | `guaranteed_fields=("field_a", "field_b")`, `declared_output_fields=frozenset({"field_a", "field_b"})` | `required_input_fields=["field_a"]` | `from_plugin_instances()` succeeds — no exception. |
| `test_edge_validation_rejects_missing_fields` | `guaranteed_fields=("field_a",)`, `declared_output_fields=frozenset({"field_a"})` | `required_input_fields=["field_a", "nonexistent"]` | `from_plugin_instances()` raises `GraphValidationError` matching "Missing fields". |
| `test_edge_validation_rejects_empty_guarantees` | `declared_output_fields=frozenset()`, `output_schema_config_override=None` (explicitly sets `_output_schema_config = None` — producer guarantees no fields) | `required_input_fields=["field_a"]` | `from_plugin_instances()` raises `GraphValidationError` — producer guarantees nothing. |
| `test_enforcement_fires_through_production_path` | `declared_output_fields=frozenset({"field_a"})`, `output_schema_config_override=None` | N/A (no consumer needed) | `from_plugin_instances()` raises `FrameworkBugError` matching "declares output fields". Graph never reaches edge validation. |

**Test 4 ordering note:** After the enforcement spec is implemented, `_validate_output_schema_contract` runs inside the transform-iteration loop of `build_execution_graph()` (replacing the `getattr` at builder.py line 223), which executes before edge construction. The check fires during node addition, so the graph never reaches `validate_edge_compatibility()`. No consumer transform is needed. This test will only pass after the enforcement spec's Task 3 is merged.

### Wiring Pattern

Each test follows the same assembly pattern (shown here for test 1, the happy path):

```python
source = MockSource()
source_settings = SourceSettings(plugin="mock_source", on_success="source_out", options={})

producer = MockFieldAddingTransform(
    "producer",
    guaranteed_fields=("field_a",),
    declared_output_fields=frozenset({"field_a"}),
)
producer_wired = WiredTransform(
    plugin=producer,
    settings=TransformSettings(
        name="producer_0",
        plugin="producer",          # Must match producer.name (WiredTransform.__post_init__ enforces this)
        input="source_out",
        on_success="consumer_in",   # Connection name to consumer (or sink name if no consumer)
        on_error="discard",
        options={},
    ),
)

consumer = MockFieldAddingTransform(
    "consumer",
    required_input_fields=["field_a"],
)
consumer_wired = WiredTransform(
    plugin=consumer,
    settings=TransformSettings(
        name="consumer_0",
        plugin="consumer",          # Must match consumer.name
        input="consumer_in",
        on_success="output",        # Final sink name
        on_error="discard",
        options={},
    ),
)

# from_plugin_instances() calls build_execution_graph(), which calls
# validate_edge_compatibility() automatically (builder.py line 923).
# For tests 2 and 3, wrap this call in pytest.raises(GraphValidationError).
# For test 4, wrap in pytest.raises(FrameworkBugError).
graph = ExecutionGraph.from_plugin_instances(
    source=source,
    source_settings=source_settings,
    transforms=[producer_wired, consumer_wired],
    sinks={"output": MockSink()},
    aggregations={},
    gates=[],
)
```

For tests 2, 3, and 4: wrap the `from_plugin_instances()` call in `pytest.raises()`. For the enforcement test (test 4), only one transform is needed — no consumer.

### What This Does NOT Cover

- **Multi-hop propagation** (transform → gate → transform field resolution) — already tested in `tests/unit/core/test_dag_schema_propagation.py` gate inheritance tests.
- **Coalesce field merging** — already tested in `tests/unit/core/test_dag_schema_propagation.py` coalesce propagation tests.
- **Schema type compatibility** (Phase 2 of `validate_edge_compatibility`) — out of scope; this spec is about field name contracts, not type matching.
- **Other real transforms** through `from_plugin_instances()` — per-transform unit pinning tests in the enforcement plan cover their `_output_schema_config` content. One real transform (RAG) through the production path proves the mechanism.
- **Duck-typed transforms without `_output_schema_config` attribute** — after the enforcement change, the `getattr` fallback is removed and direct attribute access is used. A duck-typed transform missing the attribute entirely would `AttributeError` instead of silently returning `None`. This is correct behaviour (offensive programming), but this test does not exercise it since all mocks define the attribute.

### Files Changed

| File | Change |
|------|--------|
| `tests/integration/core/__init__.py` | New empty file (directory does not exist yet) |
| `tests/integration/core/dag/__init__.py` | New empty file |
| `tests/integration/core/dag/test_output_schema_pipeline.py` | New file — 3 mock classes (~30 lines for `MockFieldAddingTransform`, ~5 each for source/sink), 5 test methods |

No production code changes. This is a test-only enhancement.
