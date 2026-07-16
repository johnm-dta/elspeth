# Runbook: Sink Effect Recovery

Use this runbook when a run has durable sink debt, an effect lease expired, or
an external call may have completed without its response reaching ELSPETH. It
applies to Landscape schema epoch 26 and the `sink-effect-v1` protocol.

Sink-effect recovery is fail closed. Do not edit Landscape rows, delete target
objects, or repeat a sink request by hand. Those actions destroy the evidence
the coordinator needs to decide whether publication occurred.

## Mental model

Every publication follows this durable state sequence:

```text
RESERVED -> PREPARED -> IN_FLIGHT -> FINALIZED
```

- `RESERVED` fixes the logical effect, its ordered members, target identity,
  role, and predecessor stream position.
- `PREPARED` stores the exact immutable plan before external publication.
- `IN_FLIGHT` gives one worker a bounded lease and records inspect, commit, and
  reconcile attempts before each call.
- `FINALIZED` binds the exact result descriptor and member outcomes. The
  resulting artifact records `publication_performed` and the evidence kind.

Effects targeting a replacing or append stream have a predecessor chain. A
new effect is a **blocked successor** until its predecessor is `FINALIZED`;
this prevents a later generation from overtaking uncertain earlier work.

Inspection and reconciliation answer different questions. `inspect_effect`
runs before a plan exists and describes the current target so preparation can
bind the correct predecessor. An adapter may explicitly return
`NO_INSPECTION_REQUIRED` when target inspection is unnecessary. After a plan
exists, `reconcile_effect` asks whether that exact plan was applied and returns
one of three closed results:

- `NOT_APPLIED`: exact absence is proven. The coordinator may commit once.
- `APPLIED_WITH_EXACT_DESCRIPTOR`: exact application is proven. The
  coordinator finalizes from the returned descriptor without another commit.
- `UNKNOWN`: neither exact absence nor exact application can be proven. The
  coordinator stops. This is an operator boundary, not a retry hint.

There is no safe speculative commit after `UNKNOWN`.

## Diagnose

1. Stop any automated retry loop around the affected run. Do not stop healthy
   workers solely because one effect has a live lease.
2. Record the `run_id`, `effect_id`, sink name, deployment revision, current
   worker identities, and the external target's own immutable version or
   transaction identity when available.
3. Load the bounded recovery history through the audit-readiness web surface or
   MCP `get_sink_effect_history`. It reports safe effect, member, and attempt
   records without exposing raw target, plan, or evidence JSON.
4. Check the durable state, lease owner/generation/expiry, predecessor ID, and
   ordered attempts. A commit attempt left at `INTENT` after process loss is
   converted to `response-lost` only under recovery authority.
5. Check the target using its native immutable metadata or **target-side
   ledger**. Do not infer absence from a network timeout, an empty cache, or a
   different credential view.

## Recovery decision

| Observed condition | Safe action |
|---|---|
| `RESERVED` | Resume normally. The adapter reuses a durable returned inspection or performs inspection, then persists the plan. |
| `PREPARED`, no live lease | Resume normally. The coordinator acquires the lease and reconciles before any commit. |
| `IN_FLIGHT`, lease live | Leave it alone. A second worker must not steal a live generation. |
| `IN_FLIGHT`, lease expired | Resume through the coordinator. Lease takeover increments the generation and gives the new worker recovery authority. |
| A commit returned and its result is durable | Resume normally. The coordinator reuses the returned result and finalizes without another commit. |
| Commit outcome is `response-lost` | Reconcile the exact stored plan. Continue only from the closed result below. |
| Reconcile says `NOT_APPLIED` | The coordinator may make the one planned commit and finalize its exact result. |
| Reconcile says `APPLIED_WITH_EXACT_DESCRIPTOR` | Finalize from the exact descriptor; do not commit again. |
| Reconcile says `UNKNOWN` | Keep the effect and its successor blocked. Escalate with the target evidence; do not commit or manually finalize. |
| `FINALIZED` | Reuse the durable finalization. Scheduler repair may close remaining pending-sink work without re-publication. |

Lease takeover is a fencing operation, not permission to guess. The new owner
must use the persisted plan, close abandoned attempts under the new generation,
and reconcile before it can publish. A stale owner cannot finalize after its
lease or generation is lost.

## Target-specific checks

### Local files

Local append and replacement recovery requires the supported filesystem
semantics documented by the adapter. Preserve the target and any code-owned
staging files until reconciliation completes. A divergent final file,
unsupported filesystem, ambiguous rename result, or unverifiable tail is
`UNKNOWN`; do not append or replace it again.

Perform **staging cleanup** only after the owning effect is `FINALIZED`, or
after code has proved that the candidate is unreferenced and outside the
configured grace period. Cleanup must remain inside the code-owned private
staging root and obey the configured age, byte, and count budgets.

### Object stores

Use conditional create/replace and immutable version, ETag, or metadata
witnesses. A timeout after the request body was sent is response-lost, not
`NOT_APPLIED`. Reconcile the exact key and planned content hash. Never turn a
precondition failure into an unconditional overwrite.

### Database sinks

Provision the adapter's target-side ledger and its uniqueness constraints with
the database schema-owner role before deployment. The runtime role remains
DML-only and must not create or alter the ledger. The data mutation and ledger
claim must share the target transaction. Missing ledger DDL, insufficient
permissions, or a transaction boundary that cannot prove the effect ID is a
preflight failure; it is not a reason to fall back to legacy writes.

### Dataverse and Chroma

These adapters reconcile durable member sub-effects. Recovery is per member;
one uncertain member must not cause already proven members to be submitted
again. Chroma recovery also preserves the adapter's documented skip/error
outcomes rather than upgrading them to accepted publication.

### Audit exports

Audit exports derive a sealed snapshot into a private bounded spool, then
publish immutable content through the same coordinator. Preserve the spool and
content-store candidate references during diagnosis. Cleanup may remove only
unreferenced candidates under the configured retention policy. The signer key ID
and signer version are public identities; signing keys are resolved from
the configured secret reference and are never stored in Landscape.

## Operator repair boundary

An operator may restore service, correct credentials or target availability,
provision approved target ledger DDL, and resume the coordinator. An operator
must not manufacture a descriptor, change the plan hash, mark an effect
`FINALIZED`, or convert `UNKNOWN` to `NOT_APPLIED` with SQL. If native target
evidence cannot establish an exact result, preserve the incident as
`UNKNOWN`, keep successors blocked, and choose an explicit business repair
outside the original effect stream.

## Completion checks

- The effect is `FINALIZED`, or remains explicitly `UNKNOWN` and blocked.
- Every finalized artifact's `publication_performed` agrees with its evidence:
  `returned`/`reconciled` are true; `inherited`/`virtual` are false.
- Member outcomes are disjoint and complete for the effect partition.
- A predecessor stream advances only through finalized effects.
- Pending scheduler work closes from durable finalization evidence without
  another external publication.
- Staging and candidate cleanup stayed bounded and reference-safe.

See also [Scheduler Lease Recovery](scheduler-lease-recovery.md) for token-work
leases and [Configuration Reference](../reference/configuration.md) for audit
export resource and signing limits.
