# Token Lifecycle Architecture

Current as of 2026-05-20.

Tokens are row instances moving through the pipeline DAG. A source row has a
stable `row_id`; one or more tokens may represent that row as it forks, expands,
buffers, batches, coalesces, fails, or reaches a sink.

For the formal contract, use
[Token Outcome Assurance](../contracts/token-outcomes/README.md). This document
is a short architectural orientation.

## Identity

| Identifier | Meaning |
|------------|---------|
| `row_id` | Stable source-row identity. |
| `token_id` | One DAG-path instance of that row. |
| `fork_group_id` | Groups sibling tokens created by a fork. |
| `join_group_id` | Groups tokens involved in a coalesce/join. |
| `expand_group_id` | Groups children created by expansion/deaggregation. |
| `branch_name` | Producer-declared branch name for forked paths. |

The authoritative schema is `src/elspeth/core/landscape/schema.py`.

## Outcome Model

ADR-019 replaced the old single-axis token outcome model with three fields in
`token_outcomes`:

| Field | Purpose |
|-------|---------|
| `completed` | Whether this row is terminal for the token. |
| `outcome` | Lifecycle answer: `success`, `failure`, `transient`, or `NULL` for buffered rows. |
| `path` | Producer-declared path explaining how the token got there. |

`path='buffered'` is the only non-terminal path and must pair with
`completed=0` and `outcome=NULL`. Terminal rows use `completed=1` and exactly
one legal `(outcome, path)` pair from `src/elspeth/contracts/enums.py`.

## Current Terminal Paths

The current legal terminal path set includes:

- `default_flow`
- `gate_routed`
- `gate_discarded`
- `on_error_routed`
- `filter_dropped`
- `coalesced`
- `unrouted`
- `quarantined_at_source`
- `sink_fallback_to_failsink`
- `sink_discarded`
- `fork_parent`
- `expand_parent`
- `batch_consumed`

`buffered` is non-terminal.

## Lifecycle Shape

Common token paths:

- Source row creates the initial token.
- A transform may continue the same token, drop/filter it, fail it, or produce
  child tokens.
- A gate may continue default flow, route to a sink, discard, or fork.
- A batch/coalesce path may mark a token transient while creating or waiting for
  downstream tokens.
- A sink records success or failure through a producer-declared terminal path.

Parent and child tokens are separate audit records. Parent terminal paths such
as `fork_parent`, `expand_parent`, and `batch_consumed` explain delegation; child
tokens carry the continuing lineage.

## Read Paths

Use maintained read surfaces instead of hand-derived lifecycle SQL:

- `elspeth explain --run <RUN_ID> --row <ROW_ID> --database <DB>`
- `elspeth explain --run <RUN_ID> --token <TOKEN_ID> --database <DB> --json`
- [Investigate Routing](../runbooks/investigate-routing.md)
- [Landscape MCP Analysis Server](../guides/landscape-mcp-analysis.md)

## Implementation References

- `src/elspeth/contracts/enums.py` - `TerminalOutcome`, `TerminalPath`, and legal pair coverage.
- `src/elspeth/contracts/audit.py` - `TokenOutcome` validation and field constraints.
- `src/elspeth/core/landscape/data_flow_repository.py` - token, parent, and outcome recording.
- `src/elspeth/core/landscape/model_loaders.py` - Tier-1 token outcome validation.
- `src/elspeth/core/landscape/schema.py` - `tokens`, `token_parents`, and `token_outcomes`.
