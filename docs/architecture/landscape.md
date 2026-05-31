# Landscape System Architecture

Current as of 2026-05-20.

Landscape is ELSPETH's audit database and lineage read model. It records run
configuration, source rows, DAG nodes and edges, token lineage, node execution
states, external calls, routing events, terminal outcomes, artifacts, checkpoints,
and export metadata.

The maintained system-level overview lives in
[ARCHITECTURE.md](../../ARCHITECTURE.md). This document focuses on the
Landscape subsystem.

## Current Inventory

Measured from this checkout on 2026-05-20:

| Metric | Value |
|--------|-------|
| Python files in `src/elspeth/core/landscape/` | 20 |
| Python lines in `src/elspeth/core/landscape/` | 10,287 |
| SQLAlchemy Core tables | 22 |
| MCP Landscape analysis tools | 27 |
| Schema epoch | 9 |

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

## Module Layout

`src/elspeth/core/landscape/` is split by repository responsibility:

| Module | Responsibility |
|--------|----------------|
| `schema.py` | Authoritative SQLAlchemy Core table definitions, constraints, indexes, and schema epoch. |
| `database.py` | Database construction, validation, SQLite/PostgreSQL connection handling. |
| `_database_ops.py` | Shared database operation helpers. |
| `factory.py` | Composition point for repository instances and plugin audit writer adapters. |
| `run_lifecycle_repository.py` | Run creation, status, export status, runtime manifest, attribution, and run-level metadata. |
| `data_flow_repository.py` | Rows, tokens, token parents, token outcomes, validation errors, and transform errors. |
| `execution_repository.py` | Node states, routing events, calls, operations, batches, and artifacts. |
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

The current schema defines 22 tables:

| Group | Tables |
|-------|--------|
| Run metadata | `runs`, `run_attributions`, `auth_events`, `preflight_results`, `secret_resolutions` |
| Static graph | `nodes`, `edges` |
| Data flow | `rows`, `tokens`, `token_parents`, `token_outcomes` |
| Execution | `node_states`, `operations`, `calls`, `routing_events` |
| Batching | `batches`, `batch_members`, `batch_outputs` |
| Errors | `validation_errors`, `transform_errors` |
| Recovery and outputs | `checkpoints`, `artifacts` |

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
- [Landscape MCP Analysis Server](../guides/landscape-mcp-analysis.md)
- [Token Outcome Assurance](../contracts/token-outcomes/README.md)
