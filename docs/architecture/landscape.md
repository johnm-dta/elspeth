# Landscape System Architecture

Current as of 2026-05-24.

Landscape is ELSPETH's audit database and lineage read model. It records run
configuration, source rows, DAG nodes and edges, token lineage, node execution
states, external calls, routing events, terminal outcomes, artifacts, checkpoints,
and export metadata.

The maintained system-level overview lives in
[ARCHITECTURE.md](../../ARCHITECTURE.md). This document focuses on the
Landscape subsystem.

## Current Inventory

Measured from this checkout on 2026-05-24:

| Metric | Value |
|--------|-------|
| Python files in `src/elspeth/core/landscape/` | 20 |
| Python lines in `src/elspeth/core/landscape/` | 10,287 |
| SQLAlchemy Core tables | 24 |
| MCP Landscape analysis tools | 27 |
| Schema epoch | 14 |

The inventory above is intentionally date-stamped. Re-run these checks before
using the numbers in release material:

```bash
find src/elspeth/core/landscape -name '*.py' -type f | wc -l
find src/elspeth/core/landscape -name '*.py' -type f -print0 | xargs -0 wc -l | tail -1
python - <<'PY'
from elspeth.core.landscape.schema import SQLITE_SCHEMA_EPOCH, metadata
from elspeth.mcp.server import _TOOLS
print(SQLITE_SCHEMA_EPOCH)
print(len(metadata.tables))
print(len(_TOOLS))
PY
```

## Trust Model

Landscape data is Tier 1: it is ELSPETH's own audit evidence. The subsystem
should fail loudly on corrupt, cross-run, or internally inconsistent audit data
instead of coercing it into a plausible answer.

Important consequences:

- Row, token, node, and run ownership checks are part of the read and write
  contract.
- Composite keys and composite foreign keys are intentional, not incidental
  schema noise.
- Token outcomes use the ADR-019 two-axis model: `completed`, `outcome`, and
  `path`.
- External calls have exactly one parent: either a node state or a source/sink
  operation.
- Hashes and payload references must survive retention and payload deletion
  boundaries.
- Per-source schema contracts and per-source lifecycle state (ADR-025) live
  in `run_sources`. The singular run-level `contract_json` writer has been
  removed; resume of a run with no `run_sources` rows raises
  `EmptyResumeStateError` rather than reconstructing a fabricated contract.
- Scheduler lease ownership is CAS-gated (ADR-026). A row in
  `token_work_items.status = LEASED` with `NULL` or empty `lease_owner` is
  a Tier-1 invariant violation; the schema enforces the invariant
  structurally via `ck_token_work_items_lease_owner_required_when_leased`.
  Mismatched `expected_lease_owner` on a state-changing call raises
  `AuditIntegrityError`; scheduler row tampering is a crash-on-anomaly
  scenario, not a recoverable one.

## Module Layout

`src/elspeth/core/landscape/` is split by repository responsibility:

| Module | Responsibility |
|--------|----------------|
| `schema.py` | Authoritative SQLAlchemy Core table definitions, constraints, indexes, and schema epoch. |
| `database.py` | Database construction, validation, SQLite/PostgreSQL connection handling. |
| `_database_ops.py` | Shared database operation helpers. |
| `factory.py` | Composition point for repository instances and plugin audit writer adapters. |
| `run_lifecycle_repository.py` | Run creation, status, export status, runtime manifest, attribution, run-level metadata, and per-source lifecycle in `run_sources`. |
| `data_flow_repository.py` | Rows, tokens, token parents, token outcomes, validation errors, and transform errors. Enforces source-row identity invariants on write (`source_row_index` / `ingest_sequence` are non-fabricable). |
| `execution_repository.py` | Node states, routing events, calls, operations, batches, and artifacts. |
| `scheduler_repository.py` | Durable token scheduler (`token_work_items`): claim/lease, CAS-gated state transitions, expired-lease recovery, pending-sink handoff. ADR-026 authoritative surface. |
| `query_repository.py` | Lineage and audit read queries used by explain/MCP/export paths. |
| `model_loaders.py` | Tier-1 row-to-model validation. |
| `exporter.py` | Audit export assembly. |
| `formatters.py` | Human-facing lineage formatting. |
| `reproducibility.py` | Reproducibility grade calculation. |
| `row_data.py` | Source-row payload retrieval helpers. |
| `auth_audit_repository.py` | Web/auth audit records. |

There is no longer a monolithic `core/landscape/recorder.py` file. Older docs
that refer to that file are historical snapshots.

## Table Groups

The current schema defines 24 tables:

| Group | Tables |
|-------|--------|
| Run metadata | `runs`, `run_attributions`, `auth_events`, `preflight_results`, `secret_resolutions` |
| Multi-source ingestion (ADR-025) | `run_sources` |
| Static graph | `nodes`, `edges` |
| Data flow | `rows`, `tokens`, `token_parents`, `token_outcomes` |
| Durable scheduler (ADR-026) | `token_work_items` |
| Execution | `node_states`, `operations`, `calls`, `routing_events` |
| Batching | `batches`, `batch_members`, `batch_outputs` |
| Errors | `validation_errors`, `transform_errors` |
| Recovery and outputs | `checkpoints`, `artifacts` |

### Multi-source ingestion (ADR-025)

`run_sources` records per-source lifecycle state, per-source schema
contract, and per-source plugin configuration for every named source in a
run. The singular `runs.contract_json` writer has been removed; per-source
contracts are the single source of truth.

| Column | Type | Notes |
|--------|------|-------|
| `run_id` | `String(64)` NOT NULL | FK to `runs.run_id`; composite PK with `source_node_id`. |
| `source_node_id` | `String(64)` NOT NULL | Composite PK with `run_id`; composite FK to `(nodes.node_id, nodes.run_id)`. |
| `source_name` | `String(64)` NOT NULL | Operator-facing name; unique per run via `UniqueConstraint(run_id, source_name)`. |
| `plugin_name` | `String(128)` NOT NULL | Source plugin identifier. |
| `lifecycle_state` | `String(32)` NOT NULL | One of `ready`, `loading`, `exhausted`, `loaded`, `interrupted` (enforced by `ck_run_sources_lifecycle_state`). Mirrors `RunSourceLifecycleState`. |
| `config_hash` | `String(64)` NOT NULL | Hash of plugin config at registration. |
| `schema_json` | `Text` | Declared schema (raw form). |
| `schema_contract_json` | `Text` | Resolved per-source `SchemaContract` (authoritative for resume). |
| `schema_contract_hash` | `String(16)` | Short hash of `schema_contract_json` for drift detection. |
| `field_resolution_json` | `Text` | Per-field resolution metadata (original_name, normalised name). |
| `recorded_at` | `DateTime(tz)` NOT NULL | When the row was persisted. |

Indexes: `ix_run_sources_run`, `ix_run_sources_source_name`.

### Durable scheduler (ADR-026)

`token_work_items` is the durable unit of work for the token scheduler.
Every scheduled continuation (initial ingest, downstream node hop, barrier
resolution, pending-sink handoff) writes a row before any in-memory
`WorkItem` is touched. The scheduler row is authoritative for resume;
in-memory `pending_items` is a cache and must never diverge from the
durable row (`SCREAM` invariant in the drain loop).

| Column | Type | Notes |
|--------|------|-------|
| `work_item_id` | `String(64)` PK | `sha256(f"{run_id}:{token_id}:{node_id or '<terminal>'}:{attempt}")` — deterministic. |
| `run_id` | `String(64)` NOT NULL | Indexed; tenant of the work item. |
| `token_id` | `String(64)` NOT NULL | Composite FK to `(tokens.token_id, tokens.run_id)`. |
| `row_id` | `String(64)` NOT NULL | Composite FK to `(rows.row_id, rows.run_id)`. |
| `node_id` | `String(64)` | Composite FK to `(nodes.node_id, nodes.run_id)`; `NULL` for terminal-handoff rows. |
| `step_index` | `Integer` NOT NULL | Secondary ordering within a token's lifetime. |
| `ingest_sequence` | `Integer` NOT NULL | Cross-source ordering primitive; mirrors `rows.ingest_sequence`. |
| `row_payload_json` | `Text` NOT NULL | Cached row payload; scrubbed on terminal/failure. |
| `status` | `String(32)` NOT NULL | One of `READY`, `LEASED`, `WAITING`, `BLOCKED`, `PENDING_SINK`, `TERMINAL`, `FAILED`. |
| `queue_key`, `barrier_key` | `String(128)` | Used by QUEUE fan-in and barrier-join coalesce. |
| `on_success_sink` | `String(128)` | Sink-bound continuation (preserved across resume). |
| `pending_sink_name` | `String(128)` | Set when the row is in `PENDING_SINK`. |
| `pending_outcome` / `pending_path` / `pending_error_hash` / `pending_error_message` | `String(32)` / `String(64)` / `String(64)` / `Text` | Pre-computed sink-outcome record so the transform does not re-run on lease expiry. |
| `branch_name`, `fork_group_id`, `join_group_id`, `expand_group_id` | `String(128)` | Token lineage carried into the durable row. |
| `coalesce_node_id`, `coalesce_name` | `String(NODE_ID_COLUMN_LENGTH)` / `String(128)` | Resume-target for coalesce cursors. |
| `attempt` | `Integer` NOT NULL | Incremented when `recover_expired_leases` reaps a non-`PENDING_SINK` row; preserved for `PENDING_SINK`. |
| `lease_owner` | `String(128)` | `row-processor:<run_id>:<uuid>` of the worker holding the row. Required non-empty when `status='LEASED'` (see check constraint). |
| `lease_expires_at` | `DateTime(tz)` | Used by `recover_expired_leases`; CAS predicate. |
| `available_at` | `DateTime(tz)` NOT NULL | Earliest claim time (delayed-retry support). |
| `created_at`, `updated_at` | `DateTime(tz)` NOT NULL | Audit timestamps. |

Constraints:

- `UniqueConstraint(run_id, token_id, node_id, attempt)` — one row per
  attempt per token-node continuation.
- `CheckConstraint ck_token_work_items_lease_owner_required_when_leased`
  — `status='LEASED'` implies `lease_owner IS NOT NULL` and non-empty.
  Closes the wedge the recovery sweep's OR-NULL predicate tolerates;
  closes filigree elspeth-9990c81e14.
- Composite FKs to `tokens`, `rows`, `nodes` (twice — `node_id` and
  `coalesce_node_id`).

Indexes:

- `ix_token_work_items_ready` on `(run_id, status, available_at)` —
  drives `claim_ready`.
- `ix_token_work_items_lease` on `(run_id, status, lease_expires_at)` —
  legacy lease-recovery index.
- `ix_token_work_items_recovery` on `(run_id, status, lease_owner,
  lease_expires_at)` — covering index for the multi-worker drain sweep
  (`recover_expired_leases` filters `lease_owner != caller_owner`).
- `uq_token_work_items_terminal_identity` partial unique on
  `(run_id, token_id, attempt)` where `node_id IS NULL` — exactly one
  terminal-handoff row per attempt.

### Per-row source identity (ADR-025)

`rows` carries the source-identity primitives `source_node_id`,
`source_row_index`, and `ingest_sequence` as non-nullable columns. These
fields are Tier-1 evidence and must not be fabricated by sources or by
synthesized-run write paths; the `create_row` write boundary raises
`AuditIntegrityError` when any are missing, with the institutional-memory
message *"Do not fabricate source_row_index or ingest_sequence from
row_index"*. See [Plugin Protocol — Source row identity](../contracts/plugin-protocol.md#source-row-identity--no-fabrication).

| Identity column | Meaning |
|----------------|---------|
| `source_node_id` | Which named source emitted the row. Composite FK to `nodes`. |
| `source_row_index` | The source's own row index within its emission stream. |
| `ingest_sequence` | Global per-run monotone ordering across all sources. |
| `row_index` | Position in source as observed by the orchestrator (may differ from `source_row_index` during resume; do not use as a substitute). |

## Key Schema Rules

- `nodes` is keyed by `(node_id, run_id)`. Join nodes with both keys.
- `edges` carries `run_id` and uses composite foreign keys to nodes.
- `tokens` carries `run_id` and points to source rows.
- `token_outcomes` stores one terminal row per token via a partial unique index
  on `completed = 1`.
- `calls` is parented by exactly one of `state_id` or `operation_id`.
- `validation_errors.row_id` is nullable because some validation failures occur
  before a row can be persisted.
- `runtime_val_manifest_json` records the runtime validation manifest in force
  for a run.
- `rows.source_node_id` is `NOT NULL` — every row attributes to a specific
  named source. `(run_id, source_node_id, source_row_index)` is unique;
  `(run_id, ingest_sequence)` is unique (global per-run monotone ordering).
- `rows.source_row_index` and `rows.ingest_sequence` are Tier-1 fields
  that the engine refuses to fabricate. Source plugins must supply both
  on every emitted row; the `create_row` boundary raises
  `AuditIntegrityError` when either is missing. See
  [Plugin Protocol — Source row identity](../contracts/plugin-protocol.md#source-row-identity--no-fabrication).
- `run_sources` is the per-source contract surface: `(run_id,
  source_node_id)` is the PK, `(run_id, source_name)` is unique, and
  `lifecycle_state` is constrained to the five `RunSourceLifecycleState`
  enum values. Resume reconstructs schema contracts by joining
  `rows.source_node_id` to `run_sources.schema_contract_json`.
- `token_work_items` lease state mutations are CAS-gated on
  `expected_lease_owner` (ADR-026). The `LEASED` status carries a
  non-empty `lease_owner` by check constraint; `recover_expired_leases`
  rotates `work_item_id` and `attempt` on lease expiry except for
  `PENDING_SINK` rows where both are preserved (sink work isn't replayed).

## Write Surfaces

Engine and plugin code should reach Landscape through repository/adaptor
interfaces rather than raw SQL:

- Run lifecycle and export status: `RunLifecycleRepository`.
- Rows, tokens, token outcomes, and errors: `DataFlowRepository`.
- Node states, routing, calls, operations, batches, and artifacts:
  `ExecutionRepository`.
- Plugin-facing audit writes: `_PluginAuditWriterAdapter` in `factory.py`.

Direct SQL belongs in schema migrations, diagnostics, or read-only operator
investigation where a maintained read API is not enough.

## Read Surfaces

Preferred read paths:

- `elspeth explain --run <RUN_ID> --row <ROW_ID> --database <DB>` for operator
  lineage investigations.
- `QueryRepository` for in-process lineage queries.
- `LandscapeExporter` for complete export/reimport evidence.
- `elspeth-mcp` for read-only MCP analysis against a Landscape database.

The MCP Landscape server exposes 27 tools from `src/elspeth/mcp/server.py`,
including run listing, token explanation, operations, calls, collisions, schema
description, outcome analysis, performance reports, diagnostics, and contract
queries.

## Operator References

- [Investigate Routing](../runbooks/investigate-routing.md)
- [Database Maintenance](../runbooks/database-maintenance.md)
- [Backup and Recovery](../runbooks/backup-and-recovery.md)
- [Scheduler Lease Recovery](../runbooks/scheduler-lease-recovery.md)
- [Landscape MCP Analysis Server](../guides/landscape-mcp-analysis.md)
- [Token Outcome Assurance](../contracts/token-outcomes/README.md)
- [ADR-025: Multi-Source Ingestion](adr/025-multi-source-ingestion.md)
- [ADR-026: Durable Token Scheduler](adr/026-durable-token-scheduler.md)
