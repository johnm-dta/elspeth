# Test System Design

> Architecture and conventions for the ELSPETH test suite (~8,100 tests across 6 tiers).

## Design Principles

1. **Test group = trust level + speed tier.** Each top-level directory answers "what kind of guarantee does this test provide?"
2. **Two sublevels max.** `group/subsystem/test_module.py` or `group/subsystem/area/test_module.py`
3. **Fixtures flow down, never sideways.** Root conftest provides base classes. Group conftest provides scoped resources. Leaf conftest provides local helpers.
4. **One canonical definition per fixture.** No more duplicate `ListSource`/`CollectSink` across 3 conftest files.
5. **Strategies are importable, not inherited.** Hypothesis strategies live in `strategies/` and are imported explicitly.

---

## Directory Structure

```
tests/
├── conftest.py                          # ROOT: markers, autouse cleanup, Hypothesis profiles
├── fixtures/                            # Shared test infrastructure (importable, not conftest)
│   ├── __init__.py
│   ├── factories.py                     # *** Re-exports from elspeth.testing + test-only factories ***
│   ├── base_classes.py                  # _TestSourceBase, _TestSinkBase, _TestTransformBase
│   ├── plugins.py                       # ListSource, CollectSink, PassTransform, FailTransform, etc.
│   ├── stores.py                        # MockPayloadStore, MockClock
│   ├── landscape.py                     # landscape_db / recorder fixture factories
│   ├── pipeline.py                      # Pipeline builder helpers
│   ├── chaosllm.py                      # ChaosLLM TestClient fixture (migrated)
│   └── azurite.py                       # Azurite blob emulator fixture (migrated)
│
├── strategies/                          # Hypothesis strategies (importable modules)
│   ├── __init__.py                      # Re-exports for convenience
│   ├── json.py                          # json_primitives, json_values, row_data
│   ├── external.py                      # messy_headers, normalizable_headers, python_keywords
│   ├── ids.py                           # id_strings, sink_names, branch_names, path_names
│   ├── binary.py                        # binary_content, nonempty_binary, small_binary
│   ├── config.py                        # valid_max_attempts, valid_delays, valid_jitter
│   ├── mutable.py                       # mutable_nested_data, deeply_nested_data
│   └── settings.py                      # DETERMINISM / STATE_MACHINE / STANDARD / SLOW / QUICK
│
│   ┌─────────────────────────────────────────────────────────────────────────┐
│   │  GROUP 1: UNIT                                                         │
│   │  Speed: <5s total. No DB, no I/O, no network. Pure logic.              │
│   │  Marker: (none - default)                                              │
│   │  Fixture scope: function only                                          │
│   └─────────────────────────────────────────────────────────────────────────┘
│
├── unit/
│   ├── conftest.py                      # Unit-specific: no DB fixtures, mock-only
│   │
│   ├── core/
│   │   ├── test_canonical.py            # RFC 8785 two-phase canonicalization
│   │   ├── test_dag.py                  # DAG construction, topo sort, cycle detection
│   │   ├── test_config.py               # Pydantic validation, Dynaconf loading
│   │   ├── test_events.py               # Synchronous event bus
│   │   ├── test_payload_store.py        # Content-addressable storage logic
│   │   ├── test_templates.py            # Jinja2 field extraction
│   │   ├── landscape/
│   │   │   ├── test_schema.py           # Table definitions, column types
│   │   │   ├── test_recorder.py         # Record methods (mocked DB)
│   │   │   └── test_exporter.py         # Export formatting
│   │   ├── checkpoint/
│   │   │   ├── test_serialization.py    # Checkpoint JSON serialization
│   │   │   └── test_state_tracking.py   # State machine transitions
│   │   ├── rate_limit/
│   │   │   └── test_limiter.py          # Bucket math, backoff calculation
│   │   ├── retention/
│   │   │   └── test_policy.py           # Retention rule evaluation
│   │   └── security/
│   │       ├── test_fingerprint.py      # HMAC fingerprinting
│   │       └── test_web_validation.py   # URL/IP validation rules
│   │
│   ├── engine/
│   │   ├── conftest.py                  # Engine mock fixtures
│   │   ├── test_processor.py            # Row processing, work queue logic
│   │   ├── test_executors.py            # Transform/gate/sink dispatch
│   │   ├── test_coalesce.py             # Fork/join barrier, merge policies
│   │   ├── test_retry.py               # Backoff math, attempt tracking
│   │   ├── test_tokens.py              # Token identity, lineage graph
│   │   ├── test_triggers.py            # Count/timeout trigger evaluation
│   │   ├── test_expression.py          # AST expression parser (no eval)
│   │   ├── test_artifacts.py           # Artifact descriptor construction
│   │   └── test_batch_adapter.py       # Batch windowing logic
│   │
│   ├── contracts/
│   │   ├── test_result_types.py         # TransformResult, GateResult, etc.
│   │   ├── test_enums.py               # RowOutcome, RunStatus, NodeType, etc.
│   │   ├── test_schema_contract.py      # SchemaContract, FieldContract
│   │   ├── test_pipeline_row.py         # PipelineRow, SourceRow
│   │   ├── test_protocols.py            # Protocol structural typing
│   │   └── config/
│   │       ├── test_runtime_config.py   # RuntimeConfig dataclasses
│   │       ├── test_alignment.py        # Settings <-> Runtime field mapping
│   │       └── test_defaults.py         # Default values, POLICY_DEFAULTS
│   │
│   ├── plugins/
│   │   ├── conftest.py                  # Plugin test fixtures
│   │   ├── test_manager.py              # PluginManager, registration
│   │   ├── sources/
│   │   │   ├── test_csv_source.py       # CSV parsing, schema inference
│   │   │   ├── test_json_source.py      # JSON loading, validation
│   │   │   └── test_null_source.py      # Null source behavior
│   │   ├── transforms/
│   │   │   ├── test_field_mapper.py     # Field mapping, renaming
│   │   │   ├── test_passthrough.py      # Identity transform
│   │   │   ├── test_truncate.py         # Field truncation
│   │   │   ├── test_batch_replicate.py  # Batch replication logic
│   │   │   └── azure/
│   │   │       ├── test_content_safety.py   # Content safety classification
│   │   │       └── test_prompt_shield.py    # Prompt shield detection
│   │   ├── sinks/
│   │   │   ├── test_csv_sink.py         # CSV writing, buffering
│   │   │   ├── test_json_sink.py        # JSON writing (atomicity)
│   │   │   ├── test_database_sink.py    # DB insert batching
│   │   │   └── test_blob_sink.py        # Blob upload logic
│   │   ├── clients/
│   │   │   ├── test_http.py             # HTTP client, redirects, caching
│   │   │   ├── test_llm_client.py       # LLM client wrapper
│   │   │   ├── test_replayer.py         # Response replayer
│   │   │   └── test_verifier.py         # Output verification
│   │   ├── llm/
│   │   │   ├── conftest.py              # ChaosLLM patching, mock contexts
│   │   │   ├── test_azure_openai.py     # Azure OpenAI transform
│   │   │   ├── test_azure_batch.py      # Azure batch processing
│   │   │   ├── test_openrouter.py       # OpenRouter single-query
│   │   │   ├── test_openrouter_multi.py # OpenRouter multi-query
│   │   │   └── test_response_parsing.py # LLM JSON response parsing
│   │   ├── batching/
│   │   │   └── test_batch_adapter.py    # Batch-aware adapter
│   │   └── pooling/
│   │       └── test_pool_manager.py     # Thread pool lifecycle
│   │
│   ├── telemetry/
│   │   ├── test_manager.py              # TelemetryManager lifecycle
│   │   ├── test_emitter.py              # Event emission
│   │   └── exporters/
│   │       ├── test_otlp.py             # OTLP export formatting
│   │       ├── test_console.py          # Console export
│   │       └── test_datadog.py          # Datadog export
│   │
│   ├── tui/
│   │   ├── test_explain_screen.py       # Explain TUI rendering
│   │   └── test_widgets.py              # Custom widget behavior
│   │
│   ├── mcp/
│   │   └── test_server.py               # MCP tool dispatch, query building
│   │
│   └── cli/
│       ├── test_commands.py             # Command argument parsing
│       └── test_formatters.py           # Event formatters, output rendering
│
│   ┌─────────────────────────────────────────────────────────────────────────┐
│   │  GROUP 2: PROPERTY                                                     │
│   │  Speed: <30s. Hypothesis-driven. Tests invariants, not examples.       │
│   │  Marker: (none - default, but respects HYPOTHESIS_PROFILE env)         │
│   │  Fixture scope: function only                                          │
│   └─────────────────────────────────────────────────────────────────────────┘
│
├── property/
│   ├── conftest.py                      # Property fixtures, strategy imports
│   │
│   ├── audit/
│   │   ├── test_canonical_determinism.py    # Same input -> same hash (DETERMINISM tier)
│   │   ├── test_hash_stability.py           # Hashes survive serialize/deserialize
│   │   ├── test_payload_integrity.py        # Store/retrieve roundtrip preserves content
│   │   ├── test_lineage_invariants.py       # Every token has exactly one terminal state
│   │   └── test_nan_infinity_rejection.py   # NaN/Infinity always rejected (P0 known issue)
│   │
│   ├── core/
│   │   ├── test_dag_invariants.py           # Acyclicity, single source, valid topo order
│   │   ├── test_schema_normalization.py     # Header normalization is idempotent
│   │   ├── test_checkpoint_roundtrip.py     # Serialize -> deserialize = identity
│   │   ├── test_config_completeness.py      # All Settings fields mapped to Runtime
│   │   ├── test_rate_limit_fairness.py      # Rate limiter doesn't starve requests
│   │   └── test_retention_monotonicity.py   # Retention age is monotonically increasing
│   │
│   ├── engine/
│   │   ├── test_token_conservation.py       # #tokens_in = #tokens_out (no silent drops)
│   │   ├── test_fork_join_balance.py        # Every fork has matching coalesce
│   │   ├── test_retry_convergence.py        # Retries terminate within max_attempts
│   │   ├── test_outcome_completeness.py     # Every row reaches terminal state
│   │   ├── test_trigger_determinism.py      # Same input -> same trigger timing
│   │   └── test_expression_safety.py        # Expression parser rejects injection
│   │
│   ├── contracts/
│   │   ├── test_type_invariants.py          # Enum exhaustiveness, result type coverage
│   │   ├── test_protocol_structural.py      # Protocols satisfied by implementations
│   │   └── test_schema_contract_lock.py     # Locked contracts reject new fields
│   │
│   ├── plugins/
│   │   ├── test_source_contracts.py         # Sources always yield SourceRow
│   │   ├── test_transform_contracts.py      # Transforms always return TransformResult
│   │   ├── test_sink_contracts.py           # Sinks always return ArtifactDescriptor
│   │   ├── test_csv_roundtrip.py            # CSV write -> read = identity
│   │   ├── test_json_roundtrip.py           # JSON write -> read = identity
│   │   └── llm/
│   │       ├── test_response_parsing.py     # Any valid JSON string -> parsed or error
│   │       └── test_retry_idempotency.py    # Retried requests use same parameters
│   │
│   └── telemetry/
│       └── test_emit_completeness.py        # No silent event drops
│
│   ┌─────────────────────────────────────────────────────────────────────────┐
│   │  GROUP 3: INTEGRATION                                                  │
│   │  Speed: <60s. Real DB (in-memory SQLite). Multiple components wired.   │
│   │  Marker: @pytest.mark.integration                                      │
│   │  Fixture scope: function (DB, recorder, plugins — full isolation)      │
│   └─────────────────────────────────────────────────────────────────────────┘
│
├── integration/
│   ├── conftest.py                      # landscape_db (function), recorder (function)
│   │
│   ├── pipeline/
│   │   ├── test_linear.py               # Source -> Transform -> Sink
│   │   ├── test_fork_join.py            # DAG with fork/coalesce branches
│   │   ├── test_aggregation.py          # Batch collection, trigger flush
│   │   ├── test_routing.py              # Gate -> named sinks
│   │   ├── test_multi_sink.py           # Default + named sinks simultaneously
│   │   ├── test_error_handling.py       # Row quarantine, pipeline continues
│   │   └── test_deaggregation.py        # 1->N token expansion
│   │
│   ├── audit/
│   │   ├── test_recording_completeness.py   # All events land in Landscape
│   │   ├── test_lineage_tracing.py          # Token -> source row traceability
│   │   ├── test_hash_chain.py               # Canonical hash integrity across run
│   │   ├── test_fk_constraints.py           # FK relationships enforced
│   │   └── test_composite_keys.py           # nodes (node_id, run_id) correctness
│   │
│   ├── checkpoint/
│   │   ├── test_resume_linear.py            # Resume linear pipeline from checkpoint
│   │   ├── test_resume_fork.py              # Resume forked pipeline
│   │   ├── test_resume_aggregation.py       # Resume with buffered aggregation
│   │   ├── test_state_restore.py            # Checkpoint -> full state reconstruction
│   │   └── test_idempotent_resume.py        # Resume twice -> same output
│   │
│   ├── config/
│   │   ├── test_yaml_to_runtime.py          # Settings YAML -> RuntimeConfig
│   │   ├── test_precedence.py               # Override hierarchy (env > yaml > default)
│   │   ├── test_secret_resolution.py        # Env var / vault resolution
│   │   └── test_plugin_instantiation.py     # Config -> PluginManager -> real plugins
│   │
│   ├── plugins/
│   │   ├── test_lifecycle.py                # on_start / on_complete / close sequencing
│   │   ├── test_instantiation.py            # PluginManager -> real plugin instances
│   │   └── llm/
│   │       ├── test_azure_pipeline.py       # Azure OpenAI in pipeline context
│   │       └── test_openrouter_pipeline.py  # OpenRouter in pipeline context
│   │
│   ├── telemetry/
│   │   ├── test_pipeline_events.py          # Events emitted during real pipeline run
│   │   └── test_exporter_chain.py           # Multiple exporters receive same events
│   │
│   └── cli/
│       ├── test_run_command.py              # `elspeth run --settings ...`
│       ├── test_resume_command.py           # `elspeth resume <run_id>`
│       ├── test_validate_command.py         # `elspeth validate --settings ...`
│       └── test_explain_command.py          # `elspeth explain --run ...`
│
│   ┌─────────────────────────────────────────────────────────────────────────┐
│   │  GROUP 4: E2E (End-to-End)                                             │
│   │  Speed: <5min. Real I/O, file-based DB, no mocks (except external).    │
│   │  Marker: @pytest.mark.e2e                                              │
│   │  Fixture scope: function (full isolation per test)                      │
│   └─────────────────────────────────────────────────────────────────────────┘
│
├── e2e/
│   ├── conftest.py                      # File-based DB, real payload store, tmp dirs
│   │
│   ├── pipelines/
│   │   ├── test_csv_to_csv.py           # CSV source -> transforms -> CSV sink
│   │   ├── test_json_to_json.py         # JSON source -> transforms -> JSON sink
│   │   ├── test_csv_to_database.py      # CSV -> DB sink (SQLite)
│   │   ├── test_multi_output.py         # One source -> gate -> multiple sinks
│   │   └── test_large_pipeline.py       # 1000+ row pipeline (size boundary)
│   │
│   ├── audit/
│   │   ├── test_full_lineage.py         # landscape.explain() for every row
│   │   ├── test_attributability.py      # The Attributability Test (CLAUDE.md spec)
│   │   ├── test_purge_integrity.py      # Payload purge -> hashes survive
│   │   └── test_export_reimport.py      # Export audit DB -> reimport -> identical
│   │
│   ├── recovery/
│   │   ├── test_crash_and_resume.py     # Simulated crash -> resume -> complete
│   │   ├── test_partial_failure.py      # Some rows fail, rest succeed, audit correct
│   │   └── test_concurrent_resume.py    # Two resume attempts (should reject second)
│   │
│   ├── examples/
│   │   └── test_shipped_examples.py     # Every example in examples/ runs to completion
│   │
│   └── external/                        # Tests requiring external services
│       ├── test_blob_source.py          # Azurite-backed blob source
│       ├── test_blob_sink.py            # Azurite-backed blob sink
│       └── test_keyvault.py             # Azure Key Vault integration
│
│   ┌─────────────────────────────────────────────────────────────────────────┐
│   │  GROUP 5: PERFORMANCE                                                  │
│   │  Speed: <10min. Benchmarks, stress, scalability, memory.               │
│   │  Marker: @pytest.mark.performance                                      │
│   │  Fixture scope: varies                                                 │
│   └─────────────────────────────────────────────────────────────────────────┘
│
└── performance/
    ├── conftest.py                      # Timer CM, memory tracker, benchmark registry
    │
    ├── benchmarks/
    │   ├── test_throughput.py           # Rows/second for linear pipeline
    │   ├── test_schema_validation.py    # Schema validation ops/sec
    │   ├── test_canonical_hashing.py    # Hash throughput (rows/sec)
    │   ├── test_token_expansion.py      # Fork/expand/deaggregate perf
    │   └── test_db_write.py            # Landscape recording ops/sec
    │
    ├── stress/
    │   ├── conftest.py                  # ChaosLLM HTTP server fixtures
    │   ├── test_llm_retry.py           # LLM under error injection (AIMD)
    │   ├── test_rate_limiter.py        # Rate limiter under concurrent load
    │   ├── test_concurrent_sinks.py    # Concurrent sink writes
    │   └── test_backpressure.py        # Telemetry backpressure under load
    │
    ├── scalability/
    │   ├── test_large_datasets.py      # 10K, 50K, 100K row pipelines
    │   ├── test_wide_rows.py           # 100, 500, 1000 field rows
    │   ├── test_deep_dag.py            # 10, 20, 50 transform chains
    │   └── test_many_sinks.py          # 10, 50, 100 named sinks
    │
    └── memory/
        ├── test_memory_baseline.py     # Memory usage per 1000 rows
        └── test_leak_detection.py      # 10K+ iterations, check RSS growth
```

---

## What's New (Missing from v1)

### New Test Files (identified gaps)

| File | Why Missing | Risk |
|------|-------------|------|
| `property/audit/test_nan_infinity_rejection.py` | Known P0 bug - NaN/Infinity in float validation | **Critical** - undermines RFC 8785 |
| `integration/audit/test_composite_keys.py` | nodes table composite PK bugs hidden | **High** - cross-run data leaks |
| `integration/checkpoint/test_resume_aggregation.py` | Resume with buffered aggregation untested | **High** - data loss on crash |
| `e2e/recovery/test_concurrent_resume.py` | Two resumes of same run untested | **Medium** - corruption risk |
| `e2e/audit/test_export_reimport.py` | Export/reimport roundtrip untested | **Medium** - data portability |
| `performance/memory/test_leak_detection.py` | No memory leak detection | **Medium** - production stability |
| `performance/memory/test_memory_baseline.py` | No memory regression baseline | **Medium** - unbounded growth |
| `performance/scalability/test_wide_rows.py` | Wide rows never tested | **Low** - edge case |
| `performance/scalability/test_many_sinks.py` | Many sinks never tested | **Low** - edge case |
| `performance/stress/test_backpressure.py` | Telemetry backpressure under load | **Low** - observability |
| `unit/plugins/transforms/azure/test_content_safety.py` | Content safety fails-open (P0 known) | **Critical** |
| `unit/plugins/transforms/azure/test_prompt_shield.py` | Prompt shield fails-open (P0 known) | **Critical** |
| `property/core/test_rate_limit_fairness.py` | Rate limiter fairness unproven | **Medium** |
| `property/core/test_retention_monotonicity.py` | Retention age ordering unproven | **Low** |
| `property/engine/test_expression_safety.py` | Expression parser injection untested | **High** |
| `integration/audit/test_fk_constraints.py` | FK constraint enforcement thin | **High** |
| `integration/pipeline/test_deaggregation.py` | 1->N token expansion integration | **Medium** |

### New Scaffolding (not in v1)

| Component | Purpose |
|-----------|---------|
| `fixtures/pipeline.py` | Pipeline builder helpers (no more 50-line test setup) |
| `fixtures/stores.py` | Consolidated mock stores (MockPayloadStore + MockClock) |
| `strategies/mutable.py` | Mutable data strategies for isolation testing |
| `performance/conftest.py` | Timer context manager, memory tracker, benchmark registry |
| `performance/memory/` | Entire memory testing category |
| `performance/scalability/` | Entire scalability testing category |

---

## Scaffolding Spec: What to Build Up Front

### 1. Root `conftest.py`

Responsibilities:

- Register ALL pytest markers (integration, e2e, performance, stress, slow, chaosllm)
- Register Hypothesis profiles (ci, nightly, debug)
- Autouse `_auto_close_telemetry_managers` fixture
- `pytest_configure` hook for marker registration
- `pytest_collection_modifyitems` hook for auto-marking (e2e/ files get @e2e, etc.)

Does NOT contain: base classes, mock plugins, strategies (those live in fixtures/ and strategies/).

```python
# tests/conftest.py - skeleton

import os
from collections.abc import Iterator
from typing import Any

import pytest
from hypothesis import Phase, Verbosity, settings

# ---------------------------------------------------------------------------
# Marker Registration
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: multi-component tests with real DB")
    config.addinivalue_line("markers", "e2e: full pipeline, real I/O, file-based DB")
    config.addinivalue_line("markers", "performance: benchmarks and regression detection")
    config.addinivalue_line("markers", "stress: load tests requiring ChaosLLM HTTP server")
    config.addinivalue_line("markers", "slow: long-running tests (>10s)")
    config.addinivalue_line("markers", "chaosllm(preset=None, **kwargs): ChaosLLM config")

# ---------------------------------------------------------------------------
# Auto-Marking by Directory
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-apply markers based on test file location."""
    for item in items:
        path = str(item.fspath)
        if "/e2e/" in path:
            item.add_marker(pytest.mark.e2e)
        elif "/performance/" in path and "/stress/" in path:
            item.add_marker(pytest.mark.stress)
            item.add_marker(pytest.mark.performance)
        elif "/performance/" in path:
            item.add_marker(pytest.mark.performance)
        # integration/ tests get marker from their conftest

# ---------------------------------------------------------------------------
# Hypothesis Profiles
# ---------------------------------------------------------------------------

settings.register_profile(
    "ci", max_examples=100, deadline=None,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
)
settings.register_profile(
    "nightly", max_examples=1000, deadline=None,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
)
settings.register_profile(
    "debug", max_examples=10, deadline=None, verbosity=Verbosity.verbose,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "ci"))

# ---------------------------------------------------------------------------
# Autouse: Telemetry Cleanup
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _auto_close_telemetry_managers() -> Iterator[None]:
    """Close TelemetryManager instances to prevent thread leaks."""
    # (full implementation migrated from tests/conftest.py)
    ...
```

### 2. `fixtures/factories.py` — Two-Layer Factory Architecture

**This is the single most important scaffolding decision.** Every backbone type that tests construct
gets a factory function. When a constructor signature changes, you update ONE function, not 90.

**The problem today:** `SchemaContract`/`FieldContract` construction is copy-pasted across 90+ files.
`_make_observed_contract()` has been independently reimplemented 12 times. `make_plugin_context()`
lives in 3 separate conftest files. When `SourceRow.valid()` grew a mandatory `contract` parameter,
dozens of tests broke.

**The existing partial solution:** `src/elspeth/testing/__init__.py` already exports
`make_pipeline_row()`, which 66 test files already import. But it covers only one type.

**The architecture:** Two layers, not one.

| Layer | Location | Contains | Imported by |
|-------|----------|----------|-------------|
| **Production factories** | `src/elspeth/testing/` | Factories for production types (SchemaContract, PipelineRow, SourceRow, TransformResult, GateResult, ArtifactDescriptor, events, result types) | Tests AND production benchmarks |
| **Test-only factories** | `tests/fixtures/factories.py` | Re-exports from `elspeth.testing` + factories for test doubles (mock PluginContext, mock landscape, ExecutionGraph builders, populate_run) | Tests only |

**The rule:** Tests NEVER call backbone constructors directly. They call factories.
Production-type factories live in `elspeth.testing` (extending what's already there).
Mock/test-infrastructure factories live in `tests/fixtures/factories.py`.

**Why two layers?**

- `elspeth.testing` is *shipped code* — its factories are importable by anyone writing tests
  against ELSPETH (benchmarks, example validation, plugin development). Changes here absorb
  constructor signature churn for the entire ecosystem.
- `tests/fixtures/factories.py` is *test infrastructure* — mock objects, graph builders
  with fake plugins, database population helpers. These should never leak into production code.

**The decision test:** "Could a production benchmark or example script use this factory?"
If yes → `elspeth.testing`. If no (it uses Mock, test doubles, or fake data) → `fixtures/factories.py`.

#### Layer 1: `src/elspeth/testing/__init__.py` (extend existing module)

The existing `make_pipeline_row()` becomes one of many. Add all production-type factories here:

```python
# src/elspeth/testing/__init__.py
"""Test infrastructure for ELSPETH pipelines.

Factories for constructing production types with sensible defaults.
When a backbone type's constructor changes, update the factory here.
Tests and benchmarks that use factories need ZERO changes.

Usage:
    from elspeth.testing import make_row, make_source_row, make_contract
"""

from __future__ import annotations

from typing import Any

from elspeth.contracts import (
    ArtifactDescriptor,
    PipelineRow,
    SourceRow,
)
from elspeth.contracts.schema_contract import FieldContract, SchemaContract


# =============================================================================
# Schema Contracts — The #1 source of test rewrite pain
# =============================================================================

def make_contract(
    data: dict[str, Any] | None = None,
    *,
    fields: dict[str, type] | None = None,
    mode: str = "OBSERVED",
    locked: bool = True,
) -> SchemaContract:
    """Build a SchemaContract from data or explicit field types.

    Usage:
        contract = make_contract({"id": 1, "name": "Alice"})       # Infer from data
        contract = make_contract(fields={"id": int, "name": str})   # Explicit types
        contract = make_contract()                                   # Bare contract
    """
    if fields is not None:
        field_contracts = tuple(
            FieldContract(
                normalized_name=name,
                original_name=name,
                python_type=python_type,
                required=True,
                source="declared",
            )
            for name, python_type in fields.items()
        )
    elif data is not None:
        field_contracts = tuple(
            FieldContract(
                normalized_name=key,
                original_name=key,
                python_type=type(value) if value is not None else object,
                required=False,
                source="inferred",
            )
            for key, value in data.items()
        )
    else:
        field_contracts = ()

    return SchemaContract(mode=mode, fields=field_contracts, locked=locked)


def make_field(
    name: str,
    python_type: type = object,
    *,
    original_name: str | None = None,
    required: bool = False,
    source: str = "inferred",
) -> FieldContract:
    """Build a single FieldContract."""
    return FieldContract(
        normalized_name=name,
        original_name=original_name or name,
        python_type=python_type,
        required=required,
        source=source,
    )


# =============================================================================
# PipelineRow / SourceRow
# =============================================================================

def make_row(
    data: dict[str, Any] | None = None,
    *,
    contract: SchemaContract | None = None,
    **kwargs: Any,
) -> PipelineRow:
    """Build a PipelineRow from a dict.

    Usage:
        row = make_row({"id": 1, "name": "Alice"})
        row = make_row(id=1, name="Alice")                 # kwargs shorthand
        row = make_row({"id": 1}, contract=my_contract)     # explicit contract
    """
    if data is None:
        data = kwargs
    if contract is None:
        contract = make_contract(data)
    return PipelineRow(data, contract)


# Preserve backward compatibility with existing 66 call sites
make_pipeline_row = make_row


def make_source_row(
    data: dict[str, Any] | None = None,
    *,
    contract: SchemaContract | None = None,
    **kwargs: Any,
) -> SourceRow:
    """Build a valid SourceRow from a dict."""
    if data is None:
        data = kwargs
    if contract is None:
        contract = make_contract(data)
    return SourceRow.valid(data, contract=contract)


def make_source_row_quarantined(
    data: dict[str, Any],
    error: str = "validation_failed",
    destination: str = "quarantine",
) -> SourceRow:
    """Build a quarantined SourceRow."""
    return SourceRow.quarantined(row=data, error=error, destination=destination)


# =============================================================================
# TransformResult — 424 call sites, must build PipelineRow first today
# =============================================================================

def make_success(
    data: dict[str, Any] | PipelineRow | None = None,
    *,
    reason: dict[str, Any] | None = None,
    context_after: dict[str, Any] | None = None,
    **kwargs: Any,
) -> "TransformResult":
    """Build a TransformResult.success().

    Usage:
        # From dict (most common)
        result = make_success({"id": 1, "score": 0.9})

        # From dict with reason
        result = make_success({"id": 1}, reason={"action": "classified"})

        # From existing PipelineRow
        result = make_success(row)

        # With context metadata
        result = make_success({"id": 1}, context_after={"model": "gpt-4"})
    """
    from elspeth.plugins.results import TransformResult

    if data is None:
        data = kwargs or {"_empty": True}

    if isinstance(data, dict):
        data = make_row(data)

    extra: dict[str, Any] = {}
    if context_after is not None:
        extra["context_after"] = context_after

    return TransformResult.success(
        data,
        success_reason=reason or {"action": "test"},
        **extra,
    )


def make_success_multi(
    rows: list[dict[str, Any] | PipelineRow],
    *,
    reason: dict[str, Any] | None = None,
) -> "TransformResult":
    """Build a TransformResult.success_multi() from multiple rows."""
    from elspeth.plugins.results import TransformResult

    pipeline_rows = [
        make_row(r) if isinstance(r, dict) else r
        for r in rows
    ]
    return TransformResult.success_multi(
        pipeline_rows,
        success_reason=reason or {"action": "test"},
    )


def make_error(
    reason: dict[str, Any] | str | None = None,
    *,
    retryable: bool = False,
) -> "TransformResult":
    """Build a TransformResult.error().

    Usage:
        result = make_error("llm_timeout")
        result = make_error({"reason": "bad_json", "raw": "..."}, retryable=True)
    """
    from elspeth.plugins.results import TransformResult

    if isinstance(reason, str):
        reason = {"reason": reason}
    return TransformResult.error(
        reason or {"reason": "test_error"},
        retryable=retryable,
    )


# =============================================================================
# GateResult — Only 12 files but constructors are verbose
# =============================================================================

def make_gate_continue(
    data: dict[str, Any] | PipelineRow,
    *,
    contract: SchemaContract | None = None,
) -> "GateResult":
    """Build a GateResult that continues the pipeline."""
    from elspeth.contracts.gate import GateResult, RoutingAction

    row = data if isinstance(data, PipelineRow) else make_row(data, contract=contract)
    return GateResult(row=row, action=RoutingAction.continue_())


def make_gate_route(
    data: dict[str, Any] | PipelineRow,
    sink: str,
    *,
    contract: SchemaContract | None = None,
) -> "GateResult":
    """Build a GateResult that routes to a named sink."""
    from elspeth.contracts.gate import GateResult, RoutingAction

    row = data if isinstance(data, PipelineRow) else make_row(data, contract=contract)
    return GateResult(row=row, action=RoutingAction.route_to_sink(sink))


def make_gate_fork(
    data: dict[str, Any] | PipelineRow,
    paths: list[str],
    *,
    contract: SchemaContract | None = None,
) -> "GateResult":
    """Build a GateResult that forks to multiple paths."""
    from elspeth.contracts.gate import GateResult, RoutingAction

    row = data if isinstance(data, PipelineRow) else make_row(data, contract=contract)
    return GateResult(row=row, action=RoutingAction.fork_to_paths(paths))


# =============================================================================
# ArtifactDescriptor — Test-friendly defaults
# =============================================================================

def make_artifact(
    path: str = "memory://test",
    *,
    size_bytes: int = 0,
    content_hash: str = "test_hash",
) -> ArtifactDescriptor:
    """Build an ArtifactDescriptor for tests."""
    return ArtifactDescriptor.for_file(
        path=path,
        size_bytes=size_bytes,
        content_hash=content_hash,
    )


def make_token_info(
    row_id: str = "row-1",
    token_id: str | None = None,
    data: dict[str, Any] | None = None,
) -> "TokenInfo":
    """Build a TokenInfo for plugin context."""
    from elspeth.engine.tokens import TokenInfo

    return TokenInfo(
        row_id=row_id,
        token_id=token_id or f"token-{row_id}",
        row_data=make_row(data or {}),
    )

# ... (remaining production-type factories: make_run_result, make_flush_result,
#      make_row_result, make_failure_info, make_pipeline_config, event factories,
#      structural dict factories — see "Additional Factory Functions" below)
```

#### Layer 2: `tests/fixtures/factories.py` (test-only infrastructure)

This file re-exports everything from `elspeth.testing` for convenience, then adds factories
that use test doubles (Mock objects, fake data, direct DB inserts). These MUST NOT go in
`elspeth.testing` because they import `unittest.mock` and construct fake infrastructure.

```python
# tests/fixtures/factories.py
"""Test-only factories and re-exports from elspeth.testing.

Layer 1 (elspeth.testing): Production-type factories — no mocks, no fakes.
Layer 2 (this file):        Test infrastructure — mocks, graph builders, DB population.

Usage:
    from tests_v2.fixtures.factories import make_row, make_context, make_graph_linear
    # make_row comes from elspeth.testing (re-exported)
    # make_context and make_graph_linear are test-only (defined here)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

# --- Re-export all production factories for single-import convenience ---
from elspeth.testing import (  # noqa: F401
    make_artifact,
    make_contract,
    make_error,
    make_field,
    make_gate_continue,
    make_gate_fork,
    make_gate_route,
    make_row,
    make_source_row,
    make_source_row_quarantined,
    make_success,
    make_success_multi,
    make_token_info,
)


# =============================================================================
# PluginContext — Uses Mock(), so test-only
# =============================================================================

def make_context(
    *,
    run_id: str = "test-run",
    state_id: str = "state-123",
    token: Any | None = None,
    config: dict[str, Any] | None = None,
    landscape: Any | None = None,
) -> "PluginContext":
    """Build a PluginContext with sensible test defaults.

    Usage:
        ctx = make_context()                            # Minimal (mock landscape)
        ctx = make_context(state_id="state-retry-3")    # Custom state_id
        ctx = make_context(landscape=recorder)           # Real landscape recorder
    """
    from elspeth.contracts.context import PluginContext

    if landscape is None:
        landscape = Mock()
        landscape.record_external_call = Mock()
        landscape.record_call = Mock()

    if token is None:
        token = make_token_info()

    return PluginContext(
        run_id=run_id,
        landscape=landscape,
        state_id=state_id,
        config=config or {},
        token=token,
    )


# =============================================================================
# ExecutionGraph — Manual construction for unit tests ONLY
# =============================================================================
#
# TIER RULES (BUG-LINEAGE-01 prevention):
#
#   unit/         → make_graph_linear(), make_graph_fork() are OK.
#                   These test graph algorithms in isolation (cycle detection,
#                   topo sort, visualization) where fake plugin names are fine.
#
#   property/     → make_graph_linear(), make_graph_fork() are OK.
#                   Property tests verify graph invariants (acyclicity, single
#                   source) and don't need real plugin wiring.
#
#   integration/  → MUST use ExecutionGraph.from_plugin_instances().
#   e2e/            These tiers test the real pipeline assembly path. Manual
#   performance/    construction would hide mapping bugs (BUG-LINEAGE-01).
#
# If you're writing an integration test and tempted to use make_graph_linear(),
# that's a sign your test setup should go through the full plugin instantiation
# path instead. See fixtures/pipeline.py for helpers that do this correctly.
# =============================================================================

def make_graph_linear(
    *node_names: str,
    source_plugin: str = "test-source",
    sink_plugin: str = "test-sink",
    transform_plugin: str = "test-transform",
) -> "ExecutionGraph":
    """Build a linear ExecutionGraph: source -> t1 -> t2 -> ... -> sink.

    WARNING: For unit/property tests only. Integration+ tests MUST use
    ExecutionGraph.from_plugin_instances() to exercise the real assembly path.

    Usage:
        graph = make_graph_linear()                        # source -> sink
        graph = make_graph_linear("enrich", "classify")    # source -> t1 -> t2 -> sink
    """
    from elspeth.core.dag import ExecutionGraph

    graph = ExecutionGraph()
    source = "source-node"
    sink = "sink-node"

    graph.add_node(source, node_type=NodeType.SOURCE, plugin_name=source_plugin, config={})

    prev = source
    for name in node_names:
        graph.add_node(name, node_type=NodeType.TRANSFORM, plugin_name=transform_plugin, config={})
        graph.add_edge(prev, name, label="continue")
        prev = name

    graph.add_node(sink, node_type=NodeType.SINK, plugin_name=sink_plugin, config={})
    graph.add_edge(prev, sink, label="continue")

    return graph


def make_graph_fork(
    branches: dict[str, list[str]],
    *,
    gate_name: str = "gate-node",
    coalesce_name: str = "coalesce-node",
) -> "ExecutionGraph":
    """Build a fork/join ExecutionGraph.

    WARNING: For unit/property tests only. Integration+ tests MUST use
    ExecutionGraph.from_plugin_instances() to exercise the real assembly path.

    Usage:
        graph = make_graph_fork({
            "path_a": ["transform_a1", "transform_a2"],
            "path_b": ["transform_b1"],
        })
    """
    from elspeth.core.dag import ExecutionGraph

    graph = ExecutionGraph()
    source = "source-node"
    sink = "sink-node"

    graph.add_node(source, node_type=NodeType.SOURCE, plugin_name="test-source", config={})
    graph.add_node(gate_name, node_type=NodeType.GATE, plugin_name="test-gate", config={})
    graph.add_edge(source, gate_name, label="continue")

    for branch_label, transforms in branches.items():
        prev = gate_name
        for t_name in transforms:
            graph.add_node(t_name, node_type=NodeType.TRANSFORM, plugin_name="test-transform", config={})
            graph.add_edge(prev, t_name, label=branch_label if prev == gate_name else "continue")
            prev = t_name
        graph.add_node(coalesce_name, node_type=NodeType.COALESCE, plugin_name="coalesce", config={})
        graph.add_edge(prev, coalesce_name, label="continue")

    graph.add_node(sink, node_type=NodeType.SINK, plugin_name="test-sink", config={})
    graph.add_edge(coalesce_name, sink, label="continue")

    return graph


# =============================================================================
# Run/Landscape Setup — Eliminate begin_run()/complete_run() boilerplate
# =============================================================================

def make_run_id() -> str:
    """Generate a unique run ID for test isolation."""
    return f"test-run-{uuid4().hex[:12]}"


def make_run_record(
    recorder: Any,
    *,
    config: dict[str, Any] | None = None,
    canonical_version: str = "sha256-rfc8785-v1",
) -> Any:
    """Begin a run and return the RunRecord.

    Usage:
        run = make_run_record(recorder)
        assert run.run_id is not None
    """
    return recorder.begin_run(
        config=config or {},
        canonical_version=canonical_version,
    )


def populate_run(
    recorder: Any,
    db: Any,
    *,
    row_count: int = 5,
    fail_rows: set[int] | None = None,
    graph: Any | None = None,
) -> dict[str, Any]:
    """Create a complete run with rows, tokens, and outcomes.

    Returns dict with run_id, row_ids, token_ids for assertions.

    Usage:
        result = populate_run(recorder, db, row_count=10, fail_rows={3, 7})
        assert len(result["row_ids"]) == 10
        assert result["row_ids"][3] in result["failed_row_ids"]
    """
    from elspeth.core.landscape.schema import (
        nodes_table,
        rows_table,
        runs_table,
        token_outcomes_table,
        tokens_table,
    )

    fail_rows = fail_rows or set()
    run_id = make_run_id()
    now = datetime.now(UTC)

    if graph is None:
        graph = make_graph_linear()

    row_ids = [f"row-{i:03d}" for i in range(row_count)]
    token_ids = [f"tok-{i:03d}" for i in range(row_count)]
    failed_row_ids = {row_ids[i] for i in fail_rows}

    with db.engine.connect() as conn:
        conn.execute(
            runs_table.insert().values(
                run_id=run_id,
                started_at=now,
                config_hash="test",
                settings_json="{}",
                canonical_version="sha256-rfc8785-v1",
                status=RunStatus.COMPLETED,
            )
        )
        conn.execute(
            nodes_table.insert().values(
                node_id="source-node", run_id=run_id, plugin_name="test",
                node_type=NodeType.SOURCE, plugin_version="1.0",
                determinism=Determinism.DETERMINISTIC, config_hash="x",
                config_json="{}", registered_at=now,
            )
        )
        conn.execute(
            nodes_table.insert().values(
                node_id="sink-node", run_id=run_id, plugin_name="test",
                node_type=NodeType.SINK, plugin_version="1.0",
                determinism=Determinism.DETERMINISTIC, config_hash="x",
                config_json="{}", registered_at=now,
            )
        )
        for i in range(row_count):
            conn.execute(
                rows_table.insert().values(
                    row_id=row_ids[i], run_id=run_id, source_node_id="source-node",
                    row_index=i, source_data_hash=f"hash{i}", created_at=now,
                )
            )
            conn.execute(
                tokens_table.insert().values(
                    token_id=token_ids[i], row_id=row_ids[i], created_at=now,
                )
            )
            outcome = RowOutcome.FAILED if i in fail_rows else RowOutcome.COMPLETED
            conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id=f"outcome-{i:03d}", run_id=run_id,
                    token_id=token_ids[i], outcome=outcome.value,
                    is_terminal=1, recorded_at=now, sink_name="sink-node",
                )
            )
        conn.commit()

    return {
        "run_id": run_id,
        "row_ids": row_ids,
        "token_ids": token_ids,
        "failed_row_ids": failed_row_ids,
        "graph": graph,
    }
```

**The PipelineRow Problem (archetype for all backbone type pain):**

`PipelineRow` is immutable (`MappingProxyType`), uses `__slots__`, and *requires* a
`SchemaContract` — which itself requires a tuple of `FieldContract` objects. What was once
`row = {"id": 1}` became a 15-line construction ceremony:

```python
# BEFORE (when rows were dicts): 1 line
row = {"id": 1, "name": "Alice"}

# AFTER (PipelineRow introduced): 15 lines
from elspeth.contracts.schema_contract import FieldContract, SchemaContract, PipelineRow
contract = SchemaContract(
    mode="OBSERVED",
    fields=(
        FieldContract(normalized_name="id", original_name="id",
                      python_type=int, required=False, source="inferred"),
        FieldContract(normalized_name="name", original_name="name",
                      python_type=str, required=False, source="inferred"),
    ),
    locked=True,
)
row = PipelineRow({"id": 1, "name": "Alice"}, contract)

# WITH FACTORY (v2): 1 line — and immune to future constructor changes
row = make_row({"id": 1, "name": "Alice"})
# or: row = make_row(id=1, name="Alice")
```

This pattern repeated 55+ times. When `SchemaContract` gained the `locked` parameter, 90 files
needed updating. When `FieldContract` gained `source`, 90 more. The factory absorbs all of this.

**Future-proofing:** If `PipelineRow.__init__` gains a `run_id` parameter, or `SchemaContract`
adds a `version` field, or `FieldContract` gets a `nullable` flag — we update `make_row()`,
`make_contract()`, and `make_field()` once. Zero test files change.

**Impact quantification:** The two-layer factory architecture absorbs changes to these types:

| Type Changed | Without Factories | With Factories | Which Layer |
|-------------|-------------------|----------------|-------------|
| `SchemaContract` adds parameter | ~90 files | 1 file (`make_contract`) | `elspeth.testing` |
| `SourceRow.valid()` signature change | ~45 files | 1 file (`make_source_row`) | `elspeth.testing` |
| `PipelineRow` constructor change | ~55 files | 1 file (`make_row`) | `elspeth.testing` |
| `PluginContext` adds field | ~95 files | 1 file (`make_context`) | `fixtures/factories.py` |
| `TransformResult.success()` changes | ~424 call sites | 1 file (`make_success`) | `elspeth.testing` |
| `FieldContract` restructured | ~90 files | 1 file (`make_field`) | `elspeth.testing` |
| `GateResult` + `RoutingAction` change | ~12 files | 1 file (`make_gate_*`) | `elspeth.testing` |

**Migration rule:** When migrating a test from v1 to v2, replace ALL direct backbone constructor
calls with factory calls. Import from `tests_v2.fixtures.factories` (which re-exports
`elspeth.testing` plus test-only additions) for a single import path.

#### Additional Factory Functions (from full codebase audit)

The scan above covers the "top 7" most painful types. A full audit of the codebase reveals
three more categories of factories needed: **engine result types**, **telemetry events**,
and **structural dicts** (raw `dict[str, Any]` with implicit schemas).

**Engine / orchestrator result types** — currently under-tested because construction is complex:

```python
# --- Engine result types ---

def make_run_result(
    *,
    run_id: str = "test-run",
    status: RunStatus = RunStatus.COMPLETED,
    total_rows: int = 10,
    succeeded: int = 10,
    failed: int = 0,
    quarantined: int = 0,
    routed: int = 0,
    duration_seconds: float = 1.5,
) -> "RunResult":
    """Build a RunResult with sensible defaults. 12-field dataclass."""
    from elspeth.engine.orchestrator.types import RunResult
    return RunResult(
        run_id=run_id, status=status, total_rows=total_rows,
        succeeded=succeeded, failed=failed, quarantined=quarantined,
        routed=routed, duration_seconds=duration_seconds,
        routed_destinations={}, forked=0, coalesced=0,
        consumed_in_batch=0, expanded=0,
    )


def make_flush_result(
    *,
    succeeded: int = 5,
    failed: int = 0,
    quarantined: int = 0,
    routed: int = 0,
) -> "AggregationFlushResult":
    """Build an AggregationFlushResult (8 counters + dict). Supports __add__."""
    from elspeth.engine.orchestrator.types import AggregationFlushResult
    return AggregationFlushResult(
        succeeded=succeeded, failed=failed, quarantined=quarantined,
        routed=routed, routed_destinations={},
        forked=0, coalesced=0, consumed_in_batch=0,
    )


def make_execution_counters(**overrides: int) -> "ExecutionCounters":
    """Build ExecutionCounters (10 mutable counters). 0 test constructions today."""
    from elspeth.engine.orchestrator.types import ExecutionCounters
    counters = ExecutionCounters()
    for key, value in overrides.items():
        setattr(counters, key, value)
    return counters


def make_row_result(
    data: dict[str, Any] | None = None,
    *,
    outcome: RowOutcome = RowOutcome.COMPLETED,
    sink_name: str = "default",
    error: Any | None = None,
) -> "RowResult":
    """Build a RowResult (final row outcome). 12 constructions today."""
    from elspeth.contracts.results import RowResult
    token = make_token_info()
    return RowResult(
        token=token,
        final_data=data or {"_result": True},
        outcome=outcome,
        sink_name=sink_name,
        error=error,
    )


def make_failure_info(
    exception_type: str = "ValueError",
    message: str = "test failure",
    *,
    attempts: int = 1,
    last_error: str | None = None,
) -> "FailureInfo":
    """Build a FailureInfo for error scenarios."""
    from elspeth.contracts.results import FailureInfo
    return FailureInfo(
        exception_type=exception_type,
        message=message,
        attempts=attempts,
        last_error=last_error or message,
    )


def make_exception_result(
    exc: BaseException | None = None,
    tb: str = "Traceback (test)",
) -> "ExceptionResult":
    """Build ExceptionResult (wraps exceptions from worker threads)."""
    from elspeth.engine.batch_adapter import ExceptionResult
    return ExceptionResult(
        exception=exc or ValueError("test exception"),
        traceback=tb,
    )


def make_pipeline_config(**overrides: Any) -> "PipelineConfig":
    """Build PipelineConfig (298 constructions — many fields with sane defaults)."""
    from elspeth.engine.orchestrator.types import PipelineConfig
    defaults: dict[str, Any] = {
        "default_sink": "default",
        "checkpoint_frequency": "every_row",
    }
    defaults.update(overrides)
    return PipelineConfig(**defaults)
```

**Telemetry event factories** — 0 test constructions for PhaseStarted/PhaseCompleted/RunSummary:

```python
# --- Telemetry events (currently ZERO test constructions) ---

def make_phase_started(
    phase: str = "transform",
    action: str = "processing",
    *,
    target: str | None = None,
) -> "PhaseStarted":
    """Build PhaseStarted event. 0 test constructions today."""
    from elspeth.contracts.events import PhaseStarted
    return PhaseStarted(phase=phase, action=action, target=target)


def make_phase_completed(
    phase: str = "transform",
    duration_seconds: float = 1.5,
) -> "PhaseCompleted":
    """Build PhaseCompleted event. 0 test constructions today."""
    from elspeth.contracts.events import PhaseCompleted
    return PhaseCompleted(phase=phase, duration_seconds=duration_seconds)


def make_run_summary(
    *,
    run_id: str = "test-run",
    status: str = "completed",
    total_rows: int = 10,
    succeeded: int = 10,
    failed: int = 0,
    quarantined: int = 0,
    duration_seconds: float = 1.5,
    exit_code: int = 0,
    routed: int = 0,
    routed_destinations: tuple[tuple[str, int], ...] = (),
) -> "RunSummary":
    """Build RunSummary event. 0 test constructions today — completely untested."""
    from elspeth.contracts.events import RunSummary
    return RunSummary(
        run_id=run_id, status=status, total_rows=total_rows,
        succeeded=succeeded, failed=failed, quarantined=quarantined,
        duration_seconds=duration_seconds, exit_code=exit_code,
        routed=routed, routed_destinations=routed_destinations,
    )


def make_external_call_completed(
    *,
    call_type: str = "llm",
    provider: str = "azure",
    duration_ms: float = 150.0,
    status_code: int = 200,
    state_id: str | None = "state-123",
    operation_id: str | None = None,
) -> "ExternalCallCompleted":
    """Build ExternalCallCompleted (XOR: exactly one of state_id/operation_id).
    14 constructions today but XOR invariant is easy to get wrong."""
    from elspeth.contracts.events import ExternalCallCompleted
    return ExternalCallCompleted(
        call_type=call_type, provider=provider,
        duration_ms=duration_ms, status_code=status_code,
        state_id=state_id, operation_id=operation_id,
    )


def make_transform_completed(
    *,
    row_id: str = "row-1",
    token_id: str = "tok-1",
    node_id: str = "transform-1",
    plugin_name: str = "test-transform",
    status: str = "success",
    duration_ms: float = 10.0,
) -> "TransformCompleted":
    """Build TransformCompleted event. 19 constructions today."""
    from elspeth.contracts.events import TransformCompleted
    return TransformCompleted(
        row_id=row_id, token_id=token_id, node_id=node_id,
        plugin_name=plugin_name, status=status, duration_ms=duration_ms,
        input_hash="hash_in", output_hash="hash_out",
    )


def make_token_completed(
    *,
    row_id: str = "row-1",
    token_id: str = "tok-1",
    outcome: RowOutcome = RowOutcome.COMPLETED,
    sink_name: str | None = "default",
) -> "TokenCompleted":
    """Build TokenCompleted event. 24 constructions today."""
    from elspeth.contracts.events import TokenCompleted
    return TokenCompleted(
        row_id=row_id, token_id=token_id,
        outcome=outcome, sink_name=sink_name,
    )
```

**Structural dict factories** — raw `dict[str, Any]` with implicit schemas that tests build repeatedly:

```python
# --- Structural dicts (implicit schemas, no type enforcement today) ---

def make_success_reason(
    action: str = "processed",
    *,
    fields_modified: list[str] | None = None,
    fields_added: list[str] | None = None,
    fields_removed: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a success_reason dict. 310 test constructions today.
    Matches TransformSuccessReason TypedDict shape."""
    reason: dict[str, Any] = {"action": action}
    if fields_modified:
        reason["fields_modified"] = fields_modified
    if fields_added:
        reason["fields_added"] = fields_added
    if fields_removed:
        reason["fields_removed"] = fields_removed
    if metadata:
        reason["metadata"] = metadata
    return reason


def make_error_reason(
    reason: str = "test_error",
    *,
    error: str | None = None,
    field: str | None = None,
    retryable: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """Build an error_reason dict. 113 test constructions today.
    Matches TransformErrorReason TypedDict shape."""
    result: dict[str, Any] = {"reason": reason}
    if error:
        result["error"] = error
    if field:
        result["field"] = field
    result.update(extra)
    return result


def make_coalesce_context(
    *,
    policy: str = "manual",
    merge_strategy: str = "union",
    expected_branches: list[str] | None = None,
    branches_arrived: list[str] | None = None,
    wait_duration_ms: float = 150.0,
) -> dict[str, Any]:
    """Build a coalesce context_after dict. 80 test constructions today.
    This dict has a well-defined 8-key schema in coalesce_executor.py:562-580
    but NO TypedDict enforcement."""
    branches = expected_branches or ["a", "b"]
    arrived = branches_arrived or branches
    return {
        "coalesce_context": {
            "policy": policy,
            "merge_strategy": merge_strategy,
            "expected_branches": branches,
            "branches_arrived": arrived,
            "branches_lost": {},
            "arrival_order": [
                {"branch": b, "arrival_offset_ms": float(i * 50)}
                for i, b in enumerate(arrived)
            ],
            "wait_duration_ms": wait_duration_ms,
        }
    }


def make_batch_checkpoint(
    *,
    batch_id: str = "batch-123",
    row_count: int = 3,
) -> dict[str, Any]:
    """Build a batch checkpoint dict. ~60 test files touch checkpoint data.
    Matches the implicit schema in azure_batch.py:160-180."""
    from datetime import UTC, datetime
    return {
        "batch_id": batch_id,
        "submitted_at": datetime.now(UTC).isoformat(),
        "row_mapping": {
            f"custom_{i}": {"id": i, "text": f"row-{i}"}
            for i in range(row_count)
        },
        "template_errors": [],
        "requests": {
            f"custom_{i}": {"model": "gpt-4", "messages": []}
            for i in range(row_count)
        },
    }
```

**Audit trail record factories** — 0-1 test constructions despite being critical:

```python
# --- Audit trail record types (under-tested, factory unlocks coverage) ---

def make_contract_audit_record(
    data: dict[str, Any] | None = None,
    *,
    mode: str = "OBSERVED",
) -> "ContractAuditRecord":
    """Build ContractAuditRecord. Only 1 test construction today.
    Needed for contract serialization round-trip testing."""
    from elspeth.contracts.contract_records import ContractAuditRecord, FieldAuditRecord
    if data is not None:
        fields = tuple(
            FieldAuditRecord(
                normalized_name=k, original_name=k,
                python_type=type(v).__name__, required=False, source="inferred",
            )
            for k, v in data.items()
        )
    else:
        fields = ()
    return ContractAuditRecord(
        mode=mode, locked=True, version_hash="test-hash", fields=fields,
    )


def make_validation_error_token(
    row_id: str = "row-1",
    node_id: str = "source-node",
    error_id: str = "err-1",
    destination: str = "quarantine",
) -> "ValidationErrorToken":
    """Build ValidationErrorToken. Only 4 test constructions today."""
    from elspeth.contracts.plugin_context import ValidationErrorToken
    return ValidationErrorToken(
        row_id=row_id, node_id=node_id,
        error_id=error_id, destination=destination,
    )


def make_transform_error_token(
    token_id: str = "tok-1",
    transform_id: str = "transform-1",
    error_id: str = "err-1",
    destination: str = "quarantine",
) -> "TransformErrorToken":
    """Build TransformErrorToken. Only 4 test constructions today."""
    from elspeth.contracts.plugin_context import TransformErrorToken
    return TransformErrorToken(
        token_id=token_id, transform_id=transform_id,
        error_id=error_id, destination=destination,
    )
```

**Full factory inventory** (updated with audit findings and layer assignments):

| Factory | Type It Wraps | Test Sites Today | Layer |
|---------|--------------|-----------------|-------|
| `make_contract()` | SchemaContract | 90 files | `elspeth.testing` |
| `make_field()` | FieldContract | 90 files | `elspeth.testing` |
| `make_row()` | PipelineRow | 55 files | `elspeth.testing` |
| `make_source_row()` | SourceRow | 45 files | `elspeth.testing` |
| `make_success()` | TransformResult.success | 310 sites | `elspeth.testing` |
| `make_success_multi()` | TransformResult.success_multi | ~20 sites | `elspeth.testing` |
| `make_error()` | TransformResult.error | 113 sites | `elspeth.testing` |
| `make_context()` | PluginContext | 95 files | `fixtures/factories` (uses Mock) |
| `make_token_info()` | TokenInfo | 42 files | `elspeth.testing` |
| `make_gate_continue/route/fork()` | GateResult | 12 files | `elspeth.testing` |
| `make_artifact()` | ArtifactDescriptor | 23 files | `elspeth.testing` |
| `make_graph_linear/fork()` | ExecutionGraph | 39 files | `fixtures/factories` (unit/property only) |
| `make_run_id()` | str (uuid) | many | `fixtures/factories` |
| `make_run_record()` | RunRecord | many | `fixtures/factories` (needs recorder) |
| `populate_run()` | multi-table insert | many | `fixtures/factories` (needs DB) |
| `make_run_result()` | RunResult | 7 files | `elspeth.testing` |
| `make_flush_result()` | AggregationFlushResult | 9 files | `elspeth.testing` |
| `make_execution_counters()` | ExecutionCounters | **0 files** | `elspeth.testing` |
| `make_row_result()` | RowResult | 12 files | `elspeth.testing` |
| `make_failure_info()` | FailureInfo | 4 files | `elspeth.testing` |
| `make_exception_result()` | ExceptionResult | 4 files | `elspeth.testing` |
| `make_pipeline_config()` | PipelineConfig | 298 files | `elspeth.testing` |
| `make_phase_started()` | PhaseStarted | **0 files** | `elspeth.testing` |
| `make_phase_completed()` | PhaseCompleted | **0 files** | `elspeth.testing` |
| `make_run_summary()` | RunSummary | **0 files** | `elspeth.testing` |
| `make_external_call_completed()` | ExternalCallCompleted | 14 files | `elspeth.testing` |
| `make_transform_completed()` | TransformCompleted | 19 files | `elspeth.testing` |
| `make_token_completed()` | TokenCompleted | 24 files | `elspeth.testing` |
| `make_success_reason()` | dict (TransformSuccessReason) | 310 sites | `elspeth.testing` |
| `make_error_reason()` | dict (TransformErrorReason) | 113 sites | `elspeth.testing` |
| `make_coalesce_context()` | dict (coalesce metadata) | 80 sites | `fixtures/factories` (test-only struct) |
| `make_batch_checkpoint()` | dict (batch checkpoint) | ~60 files | `fixtures/factories` (test-only struct) |
| `make_contract_audit_record()` | ContractAuditRecord | **1 file** | `elspeth.testing` |
| `make_validation_error_token()` | ValidationErrorToken | 4 files | `elspeth.testing` |
| `make_transform_error_token()` | TransformErrorToken | 4 files | `elspeth.testing` |

**Layer decision rule:** Does it use `Mock()`, `unittest.mock`, fake data structures, or
direct DB inserts? → `fixtures/factories`. Does it construct a real production type with
sensible defaults? → `elspeth.testing`.

### Types We Should Have But Don't (Currently Raw Dicts)

The audit also identified dicts in the production code that have well-defined schemas
but no type enforcement. These should become TypedDicts or dataclasses as part of the
v2 effort — and their factory functions should return the typed version from day one.

| Current Shape | Where Used | Keys | Recommendation |
|--------------|-----------|------|----------------|
| `context_after["coalesce_context"]` | coalesce_executor.py:562 | 8 keys: policy, merge_strategy, expected_branches, branches_arrived, branches_lost, arrival_order, wait_duration_ms, union_field_collisions | **TypedDict `CoalesceMetadata`** |
| Checkpoint dict | azure_batch.py:160 | 5 keys: batch_id, submitted_at, row_mapping, template_errors, requests | **Dataclass `BatchCheckpointData`** with `to_dict()`/`from_dict()` |
| ArtifactDescriptor.metadata (database) | results.py:329 | 3 keys: table, row_count, url_fingerprint | **TypedDict `DatabaseArtifactMetadata`** |
| ArtifactDescriptor.metadata (webhook) | results.py:329 | 2 keys: response_code, url_fingerprint | **TypedDict `WebhookArtifactMetadata`** |
| Per-plugin config | every plugin **init** | varies per plugin | **Per-plugin Config TypedDicts** (lower priority, high volume) |

These are NOT blocking for test suite v2 — the factories can return raw dicts now and
switch to typed versions later without changing any test code. That's the whole point.

### 3. `fixtures/base_classes.py`

Single canonical definition of Protocol-compliant test base classes.
Migrated from root `tests/conftest.py` with no behavioral changes.

Contains:

- `_TestSchema` (minimal PluginSchema)
- `_TestSourceBase` (SourceProtocol base with `wrap_rows()`)
- `_TestSinkBase` (SinkProtocol base)
- `_TestTransformBase` (TransformProtocol base)
- `CallbackSource` (clock-advanceable source)
- Type-cast helpers: `as_source()`, `as_sink()`, `as_transform()`, `as_gate()`, `as_batch_transform()`, `as_transform_result()`
- `create_observed_contract()`

### 4. `fixtures/plugins.py`

Consolidates the 3 duplicate ListSource/CollectSink definitions into one.

Contains:

- `ListSource` - yields rows from a list (consolidates engine/conftest + property/conftest + integration/conftest versions)
- `CollectSink` - collects results to list (with `rows_written` alias, artifact counting)
- `PassTransform` - identity transform
- `FailTransform` - always returns error result
- `ConditionalErrorTransform` - errors on `row["fail"]` truthy
- `RoutingGate` - routes based on field value (for fork/routing tests)
- `CountingTransform` - counts invocations (for retry testing)
- `SlowTransform` - configurable delay (for timeout testing)
- `ErrorOnNthTransform` - errors on Nth invocation (for retry integration)

### 5. `fixtures/stores.py`

Contains:

- `MockPayloadStore` - in-memory payload store with integrity verification
- `MockClock` - deterministic clock for timeout testing
- Fixtures: `payload_store`, `mock_clock`

### 6. `fixtures/landscape.py`

Fixture factories with function-scoped isolation.

**Why function scope everywhere?** Module-scoped databases create test interdependence — if
test A inserts data with a specific `run_id`, test B may accidentally query it. The v1 suite
documented this fragility as "use unique run_ids to avoid data pollution." v2 eliminates it
by mechanism: every test gets a fresh database. In-memory SQLite schema creation is ~5ms —
negligible compared to test logic.

Contains:

- `make_landscape_db()` - factory for in-memory LandscapeDB
- `make_recorder()` - factory for LandscapeRecorder
- Fixtures:
  - `landscape_db` (function-scoped) - fresh in-memory SQLite per test
  - `recorder` (function-scoped) - wraps landscape_db
  - `real_landscape_recorder_with_payload_store` (function-scoped)
- Helpers:
  - `populate_run()` - create a run with N rows, optional failures
  - `populate_fork_run()` - create a run with fork/coalesce topology
  - `assert_lineage_complete()` - verify every token reaches terminal state

### 7. `fixtures/pipeline.py`

High-level pipeline construction helpers for integration/e2e tests.

**These use `ExecutionGraph.from_plugin_instances()`** — the real production assembly path.
This is the correct way to build graphs in integration+ tiers (see BUG-LINEAGE-01 tier rules
in `fixtures/factories.py`).

Contains:

- `build_linear_pipeline(source_data, transforms, sink)` -> `(source, transforms, sinks, graph)`
- `build_fork_pipeline(source_data, gate, branch_transforms, sinks)` -> full pipeline
- `build_aggregation_pipeline(source_data, trigger, sink)` -> full pipeline
- `run_pipeline(source, transforms, sinks, graph, **kwargs)` -> RunResult
- `PipelineResult` dataclass: `run_id`, `sink_results`, `landscape_db`, `recorder`

### 8. `fixtures/chaosllm.py`

Migrated from `tests/fixtures/chaosllm.py` unchanged. Provides:

- `ChaosLLMFixture` dataclass
- `chaosllm_server` fixture
- `pytest_configure` for chaosllm marker

### 9. `fixtures/azurite.py`

Migrated from root `tests/conftest.py` Azurite section. Provides:

- `azurite_blob_service` (session-scoped)
- `azurite_blob_container` (function-scoped)
- Internal helpers: `_find_azurite_bin`, `_get_free_port`, `_wait_for_port`

### 10. `strategies/` modules

Migrated from `tests/property/conftest.py` into individual modules:

| Module | Strategies |
|--------|-----------|
| `json.py` | `json_primitives`, `json_values`, `dict_keys`, `row_data` |
| `external.py` | `messy_headers`, `normalizable_headers`, `python_keywords` |
| `ids.py` | `id_strings`, `sink_names`, `path_names`, `branch_names`, `unique_branches`, `multiple_branches` |
| `binary.py` | `binary_content`, `nonempty_binary`, `small_binary` |
| `config.py` | `valid_max_attempts`, `valid_delays`, `valid_jitter` |
| `mutable.py` | `mutable_nested_data`, `deeply_nested_data` |
| `settings.py` | `DETERMINISM_SETTINGS`, `STATE_MACHINE_SETTINGS`, `STANDARD_SETTINGS`, `SLOW_SETTINGS`, `QUICK_SETTINGS` |

`strategies/__init__.py` re-exports commonly used strategies for convenience:

```python
from .json import json_primitives, json_values, row_data
from .settings import STANDARD_SETTINGS, DETERMINISM_SETTINGS
```

### 11. Group-Level `conftest.py` Files

**`unit/conftest.py`**

- No database fixtures
- Imports from `fixtures.base_classes` and `fixtures.plugins`
- `mock_landscape_db` fixture (lightweight mock, not real SQLite)

**`property/conftest.py`**

- Imports PropertyTestSchema from base_classes
- Re-exports commonly used strategies
- Provides `property_list_source`, `property_collect_sink` fixtures

**`integration/conftest.py`**

- `landscape_db` (function-scoped, in-memory SQLite — full isolation per test)
- `recorder` (function-scoped)
- `resume_test_env` fixture (checkpoint + recovery + payload)
- `keyvault_url` fixture (from env/CLI)
- `plugin_manager` fixture

**`e2e/conftest.py`**

- `system_landscape_db` (function-scoped, FILE-based SQLite)
- `system_recorder` (function-scoped)
- `payload_store_path` fixture
- `pipeline_runner` fixture (full orchestrator with real DB)
- `example_pipeline_dir` fixture (locates examples/)

**`performance/conftest.py`**

- `benchmark_timer` context manager (records wall time + CPU time)
- `memory_tracker` fixture (RSS before/after, delta reporting)
- `benchmark_registry` fixture (stores results for cross-test comparison)
- Marker-based deselection (performance tests off by default)

**`performance/stress/conftest.py`**

- `chaosllm_http_server` fixture (real HTTP server via uvicorn)
- `ChaosLLMHTTPFixture` dataclass
- Migrated from `tests/stress/conftest.py`

---

## pyproject.toml Changes

```toml
[tool.pytest.ini_options]
testpaths = ["tests_v2"]          # Point to v2 during migration, or ["tests", "tests_v2"] for coexistence
pythonpath = ["src"]
addopts = [
    "-ra",
    "--strict-markers",
    "--strict-config",
    "-m", "not slow and not stress and not performance and not e2e",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: multi-component tests with real DB",
    "e2e: full pipeline end-to-end tests with real I/O",
    "performance: benchmarks and regression detection",
    "stress: load tests requiring ChaosLLM HTTP server",
    "chaosllm(preset=None, **kwargs): Configure ChaosLLM server",
]
env_files = [".env"]
```

Running different tiers:

```bash
# Default: unit + property + integration (fast CI)
pytest tests/

# Include e2e (slower CI or manual)
pytest tests/ -m "not performance"

# Just property tests
pytest tests/property/

# Performance only
pytest tests/performance/ -m performance

# Stress only (requires ChaosLLM)
pytest tests/performance/stress/ -m stress

# Everything
pytest tests/ -m ""
```

---

## Migration Strategy

1. **Phase 0: Scaffolding** - Create `tests/`, all conftest files, `fixtures/`, `strategies/`. Zero test files.
2. **Phase 1: Unit tests** - Migrate one subsystem at a time. Start with `contracts/` (least dependencies).
3. **Phase 2: Property tests** - Migrate from `tests/property/` with strategy imports updated.
4. **Phase 3: Integration tests** - Migrate from `tests/integration/` + `tests/engine/` + `tests/core/`.
5. **Phase 4: E2E tests** - Migrate from `tests/system/` + `tests/cli/` + new files.
6. **Phase 5: Performance tests** - Migrate from `tests/performance/` + `tests/stress/` + new files.
7. **Phase 6: Cutover** - Delete `tests/`, rename `tests/` to `tests/`.

Each phase is a separate branch/PR. Tests run from both directories during migration.

---

## Fixture Dependency Graph

```
conftest.py (root)
├── _auto_close_telemetry_managers [autouse]
│
├── fixtures/base_classes.py
│   ├── _TestSchema
│   ├── _TestSourceBase ──> wrap_rows(), get_schema_contract()
│   ├── _TestSinkBase
│   ├── _TestTransformBase
│   ├── CallbackSource ──> after_yield_callback
│   ├── as_source(), as_sink(), as_transform(), as_gate()
│   └── create_observed_contract()
│
├── fixtures/plugins.py (depends on base_classes)
│   ├── ListSource ──> _TestSourceBase
│   ├── CollectSink ──> _TestSinkBase
│   ├── PassTransform ──> BaseTransform
│   ├── FailTransform ──> BaseTransform
│   ├── ConditionalErrorTransform ──> BaseTransform
│   ├── RoutingGate
│   ├── CountingTransform
│   ├── SlowTransform
│   └── ErrorOnNthTransform
│
├── fixtures/stores.py
│   ├── MockPayloadStore
│   ├── MockClock
│   └── [fixture] payload_store
│
├── fixtures/landscape.py (depends on stores)
│   ├── make_landscape_db()
│   ├── make_recorder()
│   ├── [fixture] landscape_db [function scope — fresh per test]
│   ├── [fixture] recorder [function scope]
│   ├── populate_run()
│   └── assert_lineage_complete()
│
├── fixtures/pipeline.py (depends on plugins, landscape)
│   ├── build_linear_pipeline()
│   ├── build_fork_pipeline()
│   ├── build_aggregation_pipeline()
│   ├── run_pipeline()
│   └── PipelineResult
│
├── fixtures/chaosllm.py
│   ├── ChaosLLMFixture
│   └── [fixture] chaosllm_server
│
├── fixtures/azurite.py
│   ├── [fixture] azurite_blob_service [session scope]
│   └── [fixture] azurite_blob_container [function scope]
│
└── strategies/
    ├── json.py: json_primitives, json_values, row_data
    ├── external.py: messy_headers, normalizable_headers
    ├── ids.py: id_strings, sink_names, branch_names
    ├── binary.py: binary_content, nonempty_binary
    ├── config.py: valid_max_attempts, valid_delays
    ├── mutable.py: mutable_nested_data, deeply_nested_data
    └── settings.py: DETERMINISM / STANDARD / SLOW / QUICK
```
