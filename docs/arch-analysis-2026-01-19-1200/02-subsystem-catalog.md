## 1) Contracts (types + domain models)
- Location: `src/elspeth/contracts/`
- Responsibility: Shared domain contracts (schemas, enums, audit models, results/errors) used across core/engine/plugins/CLI.
- Key modules: `audit.py`, `engine.py`, `schema.py`, `results.py`, `enums.py`, `identity.py`
- Dependencies: Mostly stdlib + pydantic; imported by almost everything.

## 2) Configuration (pipeline + subsystem settings)
- Location: `src/elspeth/core/config.py` (+ plugin config helpers in `src/elspeth/plugins/config_base.py`)
- Responsibility: Pydantic settings models for pipeline definition (sources/sinks/transforms/gates/coalesce/aggregations) and operational knobs (retry, rate_limit, payload_store, checkpoint).
- Key modules: `src/elspeth/core/config.py`, `src/elspeth/plugins/config_base.py`
- Notable boundary: Validates config-driven expressions at load time via `ExpressionParser` (defense-in-depth).

## 3) Canonicalization & Hashing
- Location: `src/elspeth/core/canonical.py`
- Responsibility: Deterministic JSON normalization + RFC8785 serialization; stable SHA-256 hashing used for audit integrity.
- Key modules: `canonical_json()`, `stable_hash()`
- External deps: `numpy`, `pandas`, `rfc8785`

## 4) Execution Graph (DAG)
- Location: `src/elspeth/core/dag.py`
- Responsibility: Compiles settings into a validated `ExecutionGraph` (NetworkX MultiDiGraph), including route label resolution for gates and fork semantics.
- Key modules: `ExecutionGraph.from_config()`, `validate()`, `get_route_resolution_map()`
- Dependencies: `elspeth.contracts` (routing enums), `networkx`

## 5) Landscape (audit database + lineage + export)
- Location: `src/elspeth/core/landscape/`
- Responsibility: The audit backbone: schema + connection mgmt + high-level recorder API + explain queries + export/signing + reproducibility grading.
- Key modules:
  - Schema/DB: `schema.py`, `database.py`
  - Recorder: `recorder.py` (runs/nodes/edges/rows/tokens/node_states/routing/batches/artifacts/errors)
  - Queries: `lineage.py`, `row_data.py`, `repositories.py`, `reproducibility.py`
  - Export: `exporter.py`, `formatters.py`
- External deps: `sqlalchemy`, `alembic` (migrations), `deepdiff` (verify workflows are hinted in docs)

## 6) Engine Runtime (orchestration + execution semantics)
- Location: `src/elspeth/engine/`
- Responsibility: Executes a run end-to-end:
  - Orchestrator handles lifecycle, plugin node_id assignment, sink flush, export, and optional checkpoints.
  - RowProcessor handles per-row DAG semantics: tokens, transforms, gates, fork/coalesce, aggregation triggers.
  - Executors wrap plugin calls with audit recording and spans.
- Key modules: `orchestrator.py`, `processor.py`, `executors.py`, `tokens.py`, `triggers.py`, `coalesce_executor.py`, `retry.py`, `expression_parser.py`, `schema_validator.py`, `spans.py`

## 7) Plugin Framework (contracts + discovery + helpers)
- Location: `src/elspeth/plugins/`
- Responsibility: Defines plugin base classes/protocols, pluggy-based discovery, schema generation helpers, execution context, shared utilities.
- Key modules: `protocols.py`, `base.py`, `manager.py`, `hookspecs.py`, `context.py`, `schema_factory.py`, `config_base.py`, `utils.py`

## 8) Built-in Plugins (Sources/Transforms/Sinks)
- Location: `src/elspeth/plugins/sources|transforms|sinks/`
- Responsibility: Reference implementations for IO + simple transforms:
  - Sources: `csv_source.py`, `json_source.py` (schema-validated ingestion, coercion allowed at boundary)
  - Transforms: `passthrough.py`, `field_mapper.py` (schema-validated pipeline ops, error routing)
  - Sinks: `csv_sink.py`, `json_sink.py`, `database_sink.py` (batch writes + artifact hashing)

## 9) Checkpointing & Recovery
- Location: `src/elspeth/core/checkpoint/`
- Responsibility: Persist progress markers for crash recovery and query resume points.
- Key modules: `manager.py`, `recovery.py`
- Storage: Checkpoints live in Landscape DB (`checkpoints_table` in `core/landscape/schema.py`)

## 10) Rate Limiting
- Location: `src/elspeth/core/rate_limit/`
- Responsibility: Wrapper + registry around `pyrate-limiter` for external-call throttling (per-service).
- Key modules: `limiter.py`, `registry.py`
- Notable behavior: Installs a custom `threading.excepthook` to suppress a known pyrate-limiter cleanup race.

## 11) Retention / Purge
- Location: `src/elspeth/core/retention/` (+ `elspeth purge` in `src/elspeth/cli.py`)
- Responsibility: Finds and deletes expired payload blobs while preserving hashes in Landscape.
- Key modules: `retention/purge.py`

## 12) CLI + Explain TUI
- Location: `src/elspeth/cli.py`, `src/elspeth/tui/`
- Responsibility:
  - CLI: load/validate/run pipelines; list plugins; purge old payloads
  - TUI: interactive lineage exploration (currently partially stubbed UI with some non-Textual rendering helpers)
