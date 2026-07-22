# Landscape System Architecture

Current as of 2026-07-23 for the 0.7.1 release line.

Landscape is ELSPETH's audit database and lineage read model. It records run
configuration, source rows, DAG nodes and edges, token lineage, node execution
states, external calls, routing events, terminal outcomes, checkpoints, durable
sink and coalesce effects, export snapshots, artifacts, and sidecar-journal
publication.

The maintained system-level overview lives in
[ARCHITECTURE.md](../../ARCHITECTURE.md). This document focuses on the
Landscape subsystem.

## Current Inventory

Measured from this checkout on 2026-07-23:

| Metric | Value |
|--------|-------|
| Python files in `src/elspeth/core/landscape/` | 63 |
| Python lines in `src/elspeth/core/landscape/` | 32,137 |
| SQLAlchemy Core tables | 41 |
| MCP Landscape analysis tools | 29 |
| Schema epoch | 29 |

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
- Routing events are run-scoped as of schema epoch 22. `routing_events.run_id`
  participates in composite foreign keys to `node_states(state_id, run_id)` and
  `edges(edge_id, run_id)`, so a stored gate decision cannot accidentally bind
  to a state or edge from another run.
- Token ancestry and validation-error associations are run-scoped as of epoch
  29. A token parent or quarantined-row link cannot bind to evidence from a
  different run.
- Sink and audit-export publication is represented as a durable effect stream.
  `UNKNOWN` is a valid blocked recovery result when target evidence cannot prove
  whether a request committed; it is never coerced into permission to retry.

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
| `execution_repository.py` | Node states, routing events, calls, operations, batches, artifacts, audit-export snapshots, and the sink-effect repository surface. |
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

The current schema defines 41 tables:

| Group | Tables |
|-------|--------|
| Run metadata | `runs`, `run_attributions`, `auth_events`, `preflight_results`, `secret_resolutions`, `run_web_plugin_policy` |
| Schema identity | `elspeth_schema_identity` |
| Multi-source ingestion (ADR-025) | `run_sources` |
| Static graph | `nodes`, `edges` |
| Data flow | `rows`, `tokens`, `token_parents`, `token_outcomes` |
| Durable scheduler (ADR-026) | `token_work_items`, `scheduler_events` |
| Run coordination (ADR-030) | `run_coordination`, `run_coordination_events`, `run_workers`, `coalesce_branch_losses` |
| Execution and outputs | `node_states`, `operations`, `calls`, `routing_events`, `artifacts` |
| Batching | `batches`, `batch_members`, `batch_outputs` |
| Errors | `validation_errors`, `transform_errors` |
| Recovery | `checkpoints` |
| Durable coalesce | `coalesce_effects`, `coalesce_effect_members` |
| Durable sink effects | `sink_effect_streams`, `sink_effects`, `sink_effect_members`, `sink_effect_attempts`, `sink_effect_export_snapshots` |
| Audit-export snapshots | `audit_export_snapshots`, `audit_export_snapshot_chunks` |
| Sidecar journal | `sidecar_journal_outbox` |

### Artifact logical-effect identity

`artifacts.idempotency_key` is an opaque logical-effect key scoped by `run_id`.
A non-null key is unique within the run, and an identical registration returns
the first artifact while a divergent retry raises a Tier-1 integrity error.

An artifact has exactly one production authority: a historical node state or a
durable sink effect. Effect-backed artifacts carry a composite reference to
`sink_effects(effect_id, run_id, sink_node_id)` and are registered from the
effect's immutable final descriptor. The artifact records whether publication
was performed and whether its evidence was returned, reconciled, inherited, or
virtual. Audit identity therefore converges with external-effect identity
rather than being added as an unrelated row after I/O.

### Durable sink and export effects

Every supported sink publication follows one persisted lifecycle:

```text
RESERVED -> PREPARED -> IN_FLIGHT -> FINALIZED
```

- `sink_effect_streams` orders effects for one target/role and prevents a
  successor from overtaking an uncertain predecessor.
- `sink_effects` stores the immutable target binding and plan, current fenced
  lease generation, reconciliation result, and final descriptor.
- `sink_effect_members` binds the ordered pipeline tokens and per-member
  outcome evidence. Failsink members retain the exact primary effect that
  produced them.
- `sink_effect_attempts` records inspect, commit, and reconcile intent before
  the adapter call, then records returned, response-lost, or error state.
- `audit_export_snapshots` and `sink_effect_export_snapshots` bind a sealed
  export snapshot to the effect that publishes it.

After a lost response, the coordinator may publish again only when the adapter
proves the exact plan was not applied. A proven application finalizes without
another commit; an `UNKNOWN` result remains blocked. Epochs 26–28 introduced
the effect ledger, durable coalesce receipts, and per-member failsink
provenance. Epoch 29 adds run-scoped ancestry/error links, output-contract
hashes, durable batch-expansion claims, and the transaction-owned sidecar
journal outbox.

ELSPETH is pre-1.0. An older Landscape database is archived or exported as
required and recreated at epoch 29; startup and read-only inspection do not
transform a predecessor store in place.

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
| `schema_contract_hash` | `String(32)` | Canonical hash prefix of `schema_contract_json` for drift detection. |
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
| `lease_owner` | `String(128)` | Registered `worker:<run_id>:<uuid>` identity holding the row in production; direct legacy repository harnesses may use an explicit opaque identity. Required non-empty when `status='LEASED'` (see check constraint). |
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
  (strict `recover_expired_leases` scopes the query to the token's
  `run_id` and filters `lease_owner != coordination_token.worker_id`).
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
- `token_parents` and `validation_errors` use run-scoped foreign keys; neither
  lineage nor quarantine evidence can cross a run boundary.
- `nodes.output_contract_hash` stores the canonical output-contract identity
  used to detect incompatible graph evolution.
- Sink effects and coalesce effects finalize their result and controlling
  state transition atomically. A result cannot become visible without its
  durable receipt.
- `sidecar_journal_outbox` is written in the audit transaction and owned by one
  canonical journal destination. Recovery cannot move or acknowledge a batch
  through a different sidecar path.

## Write Surfaces

Engine and plugin code should reach Landscape through repository/adaptor
interfaces rather than raw SQL:

- Run lifecycle and export status: `RunLifecycleRepository`.
- Rows, tokens, token outcomes, and errors: `DataFlowRepository`.
- Node states, routing, calls, operations, batches, and artifacts:
  `ExecutionRepository`.
- Durable token work and lease recovery: `SchedulerRepository`.
- Sink-effect reservation, attempts, fencing, and finalization:
  `ExecutionRepository.sink_effects`.
- Plugin-facing audit writes: `PluginAuditWriterAdapter` in `plugin_audit_writer.py`,
  constructed by `RecorderFactory.plugin_audit_writer()`.

Direct SQL belongs in schema migrations, diagnostics, or read-only operator
investigation where a maintained read API is not enough.

## Read Surfaces

Preferred read paths:

- `elspeth explain --run <RUN_ID> --row <ROW_ID> --database <DB>` for operator
  lineage investigations.
- `QueryRepository` for in-process lineage queries.
- `LandscapeExporter` for complete export/reimport evidence.
- `elspeth-mcp` for read-only MCP analysis against a Landscape database.

The MCP Landscape server exposes 29 tools from `src/elspeth/mcp/server.py`,
including run listing, token explanation, operations, calls, collisions, schema
description, outcome analysis, sink-effect recovery history, performance
reports, diagnostics, and contract queries.

## Operator References

- [Investigate Routing](../runbooks/investigate-routing.md)
- [Database Maintenance](../runbooks/database-maintenance.md)
- [Backup and Recovery](../runbooks/backup-and-recovery.md)
- [Scheduler Lease Recovery](../runbooks/scheduler-lease-recovery.md)
- [Sink Effect Recovery](../runbooks/sink-effect-recovery.md)
- [Landscape MCP Analysis Server](../guides/landscape-mcp-analysis.md)
- [Token Outcome Assurance](../contracts/token-outcomes/README.md)
- [ADR-025: Multi-Source Ingestion](adr/025-multi-source-ingestion.md)
- [ADR-026: Durable Token Scheduler](adr/026-durable-token-scheduler.md)
