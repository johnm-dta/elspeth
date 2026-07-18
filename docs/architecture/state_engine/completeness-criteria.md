# State Engine Completeness Criteria

## Claim being evaluated

The durable state engine is **complete** only when every mandatory state,
subtype, boundary, read decision, refusal, and recovery path has current,
production-representative evidence for every applicable dimension in the v1
proof catalog.

Completeness is binary. A maturity score cannot override a failed or unknown
mandatory cell.

## Boundary

The claim includes:

- durable token work from enqueue through terminal disposition;
- transform and sink-redrive leasing, heartbeat, expiry, reclaim, and fencing;
- aggregation and coalesce barrier adoption, completion, and continuation;
- sink-effect reservation, preparation, lease, reconciliation, publication,
  finalization, and scheduler repair;
- source, transform, gate, aggregation, coalesce, sink, follower, and lifecycle
  production boundaries;
- scheduler events, coordination evidence, node states, token outcomes,
  branch-loss rows, effect attempts, and artifacts;
- orchestration read models that decide drain, flush, resume, relinquishment,
  eviction, or completion;
- the catalog's exact state-store, deployment, lifecycle, and first-party
  plugin inventories;
- fresh, resume, follower, partial-start, normal teardown, exceptional teardown,
  contention, and process-death behavior.

The web session/execution engine is outside this claim unless it directly calls
or represents one of these durable state-engine boundaries.

## Closed result vocabulary

Each required case has exactly one status:

| Status | Meaning |
| --- | --- |
| `pass` | Current executable evidence proves the complete case at the assessed baseline. |
| `partial` | Current executable evidence proves only the explicitly stated subset. |
| `fail` | Current executable evidence demonstrates behavior contrary to the case contract. |
| `unknown` | Adequate current evidence has not been executed or does not exist. |
| `not_applicable` | The versioned catalog says the case genuinely does not apply and records why. |

Missing tests, unavailable credentials, skipped suites, unsupported local
services, time pressure, and absent evidence are `unknown`, never
`not_applicable`.

Mapping progress such as `mapped` or `candidate` is metadata, not a proof
status. A narrow passing arm never promotes its containing leg by analogy.

## Ten mandatory dimensions

Every catalog leg accounts for all ten dimensions:

1. `production_entry` — the real caller and trigger reach the leg.
2. `precondition_image` — exact state, subtype, identity, ownership, evidence,
   and coordination inputs before the operation.
3. `success_effects` — exact durable and externally visible success image.
4. `guard_refusal` — wrong status, owner, membership, subtype, token, epoch,
   or reference fails closed.
5. `zero_mutation_rollback` — refused, raised, or interrupted operations leave
   every affected durable plane unchanged or atomically rolled back.
6. `concurrency` — independent connections/processes prove winner, loser,
   ordering, and ABA/generation behavior.
7. `crash_restart` — a fresh object/process against the same durable store
   converges after every cross-transaction seam.
8. `boundary_composition` — real supported plugin, orchestration, repository,
   lifecycle, and external-effect boundaries preserve the contract.
9. `read_model_truth_table` — downstream decisions include and exclude every
   state/subtype/owner/expiry arm correctly.
10. `maintenance` — exact evidence locators remain collected and run in the
    maintained verification selection, with coherent actionable gap themes
    either live-owned in Filigree or explicitly unowned.

Applicability is catalog-owned. An assessor cannot mark a dimension N/A merely
because it is inconvenient to execute.

## Evidence hierarchy

Behavioral `pass`, `partial`, or `fail` claims require executed evidence:

1. production-path harness or real integration/E2E test;
2. focused integration test across the real repository and caller boundary;
3. direct repository test for exact transaction detail;
4. property/state-machine test for broad invariant exploration.

Source inspection, architecture documents, decisions, plans, tracker state,
and test names support mapping and interpretation. They cannot independently
make a behavioral case pass.

Concurrency claims require independent connections and, where the contract is
process-scoped, independent operating-system processes. In-process exception
handling does not establish process-death recovery.

## Hard gates

The catalog defines these hard gates:

- `HG-01-invalid-subtype` — no malformed state/subtype bundle is admitted.
- `HG-02-authority-downgrade` — no missing token silently selects a weaker
  production write path.
- `HG-03-double-winner` — contention cannot produce two effective winners.
- `HG-04-refusal-mutation` — every refusal preserves complete durable images.
- `HG-05-state-evidence-atomicity` — state and mandatory evidence commit or
  roll back together.
- `HG-06-restart-loss-or-duplicate` — restart loses or duplicates neither work
  nor externally visible effect.
- `HG-07-read-model-unproved-arm` — no orchestration decision relies on an
  unproved state/subtype arm.
- `HG-08-plugin-lifecycle-durability` — supported lifecycle paths preserve the
  same durable contract.
- `HG-09-mandatory-leg-unresolved` — every mandatory catalog case is resolved.
- `HG-10-normative-contract-drift` — source, architecture, catalog, assessment,
  and tracker do not contradict one another.

Any open or unknown hard gate prevents a `complete` verdict.

## Derived verdicts

Leg verdicts derive from their expanded cases and assessment gate mapping:

- `confirmed`: every required case passes and every N/A is catalog-approved;
- `gap`: at least one required case fails or an open hard gate's
  `affected_leg_ids` names the leg;
- `unknown`: no known failure exists, but required evidence is missing.

Overall verdicts:

- `complete`: every mandatory leg is confirmed and all hard gates are closed;
- `not_complete`: at least one leg has a gap or at least one hard gate is open;
- `insufficient_evidence`: no demonstrated gap exists, but at least one
  mandatory leg/gate is unknown or the baseline/manifest is invalid.

## Production-supported claim

`complete` proves the catalog contract at every versioned execution profile.
`production-supported` additionally requires:

- every state-store, deployment, lifecycle, and first-party plugin named in
  the catalog's `execution_profiles`;
- every first-party plugin kind covered at its production boundary;
- declared scale envelopes and bounded-operation evidence;
- operator recovery procedures exercised against representative artifacts;
- current mandatory verification in CI or the release gate.

Do not use “production-assured” as a separate informal category.
