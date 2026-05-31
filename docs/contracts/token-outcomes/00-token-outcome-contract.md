# Token Outcome Contract

Current as of 2026-05-20.

This is the durable contract for token outcome records in Landscape. The
authoritative implementation lives in:

- `src/elspeth/contracts/enums.py`
- `src/elspeth/contracts/audit.py`
- `src/elspeth/core/landscape/schema.py`
- `src/elspeth/core/landscape/data_flow_repository.py`

## Definitions

- **Token**: one instance of a source row moving through a DAG path.
- **Terminal row**: a `token_outcomes` row with `completed = 1`.
- **Non-terminal row**: a `token_outcomes` row with `completed = 0`; currently
  only `(NULL, buffered)`.
- **Outcome**: lifecycle answer: `success`, `failure`, `transient`, or `NULL`.
- **Path**: producer-declared provenance answer.

## Legal Pairs

| completed | outcome | path | Required discriminator fields |
|-----------|---------|------|-------------------------------|
| 1 | `success` | `default_flow` | `sink_name` |
| 1 | `success` | `gate_routed` | `sink_name` |
| 1 | `success` | `gate_discarded` | none |
| 1 | `failure` | `on_error_routed` | `sink_name`, `error_hash` |
| 1 | `success` | `filter_dropped` | none |
| 1 | `success` | `coalesced` | `join_group_id` |
| 1 | `failure` | `unrouted` | `error_hash` |
| 1 | `failure` | `quarantined_at_source` | `error_hash` |
| 1 | `transient` | `sink_fallback_to_failsink` | `sink_name`, `error_hash` |
| 1 | `failure` | `sink_discarded` | `sink_name`, `error_hash`; `sink_name` must be `__discard__` |
| 1 | `transient` | `fork_parent` | `fork_group_id` |
| 1 | `transient` | `expand_parent` | `expand_group_id` |
| 1 | `transient` | `batch_consumed` | `batch_id` |
| 0 | `NULL` | `buffered` | `batch_id` |

Fields not listed for a pair are forbidden unless explicitly allowed by
`src/elspeth/contracts/audit.py`.

## Invariants

1. Every token in a terminal run has exactly one completed token outcome.
2. A token may have non-terminal `buffered` rows before its terminal outcome.
3. `completed = 0` requires `outcome IS NULL` and `path = 'buffered'`.
4. `completed = 1` requires a legal `(outcome, path)` pair.
5. Required discriminator fields must be present for the pair.
6. Forbidden discriminator fields must be absent for the pair.
7. Parent/delegation paths (`fork_parent`, `expand_parent`, `batch_consumed`)
   must have corresponding lineage, child, batch, or recovery evidence.
8. `token_outcomes` is the authoritative token lifecycle record. `node_states`
   and `artifacts` explain work, but do not replace the lifecycle row.

## Schema Notes

`token_outcomes` stores:

- identity: `outcome_id`, `run_id`, `token_id`
- lifecycle: `outcome`, `path`, `completed`, `recorded_at`
- discriminators: `sink_name`, `batch_id`, `fork_group_id`, `join_group_id`,
  `expand_group_id`, `error_hash`
- context: `context_json`, `expected_branches_json`

The schema has a partial unique index that permits multiple non-terminal rows
but allows only one terminal row per token.
