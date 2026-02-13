# Token Outcome Contract

This is the canonical contract for token terminal states and their audit fields.
It is derived from:
- `src/elspeth/contracts/enums.py` (RowOutcome, is_terminal)
- `src/elspeth/core/landscape/schema.py` (token_outcomes table)
- `src/elspeth/core/landscape/recorder.py` (record_token_outcome)

## Definitions

- Token: a single row instance flowing through a specific DAG path.
- Terminal outcome: a final state (token will not reappear).
- Non-terminal outcome: temporary state (token will reappear).

## Contract: outcomes and required fields

| Outcome | Terminal | Required fields | Primary recorder |
|---------|----------|-----------------|------------------|
| COMPLETED | yes | sink_name | Orchestrator (after sink write) |
| ROUTED | yes | sink_name | RowProcessor (gate or error sink) |
| FORKED | yes | fork_group_id | RowProcessor (after gate fork) |
| FAILED | yes | error_hash | RowProcessor or CoalesceExecutor |
| QUARANTINED | yes | error_hash | RowProcessor or Orchestrator (source quarantine) |
| CONSUMED_IN_BATCH | yes | batch_id | RowProcessor (aggregation) |
| COALESCED | yes | join_group_id | CoalesceExecutor (consumed tokens) |
| EXPANDED | yes | expand_group_id | RowProcessor (deaggregation parent) |
| BUFFERED | no | batch_id | RowProcessor (aggregation passthrough) |

Notes:
- BUFFERED is the only non-terminal outcome. It must be followed by exactly one
  terminal outcome before the run is marked completed.
- Only one terminal outcome is allowed per token (enforced by partial unique
  index in `token_outcomes`).

## Contract: invariants

1. Exactly one terminal outcome per token.
2. No terminal outcome may be missing a required field (see table above).
3. COMPLETED implies the token has a completed sink node_state.
4. Completed sink node_state implies a COMPLETED token_outcome with sink_name.
5. BUFFERED outcomes must be followed by a terminal outcome before run completion.
6. FORKED outcome implies child tokens exist (token_parents table) sharing fork_group_id.
7. EXPANDED outcome implies child tokens exist (token_parents table) sharing expand_group_id.
8. COALESCED outcome implies join_group_id points to the merged token's join_group_id.

## Terminal vs non-terminal behavior

- Terminal outcomes represent the final state for that token. It should not
  appear in any subsequent processing or results.
- Non-terminal outcomes (BUFFERED) represent a hold state. The token must
  reappear and later receive a terminal outcome.

## Single-source-of-truth

- `token_outcomes` is the authoritative terminal state record.
- `node_states` provide lineage detail but do NOT replace token_outcomes.
