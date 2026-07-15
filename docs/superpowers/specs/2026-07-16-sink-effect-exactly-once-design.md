# Sink Effect Exactly-Once Design

Date: 2026-07-16
Status: approved design
Branch context: `codex/safety-74a343d5ad` from `release/0.7.1`
Filigree: `elspeth-74a343d5ad`

## Purpose

Make sink publication crash-safe at the production boundary. A retry after
process loss, lease takeover, a lost plugin response, or a lost Landscape
finalization response must converge on one externally observable effect and
one immutable audit identity. The durable identity and its complete ordered
membership must exist before any sink publication begins.

This closes the gap left by epoch 25. Epoch 25 makes artifact registration
idempotent once an artifact descriptor exists, but the production executor
still calls `sink.write()` and `sink.flush()` before registration. A crash in
that interval can repeat externally visible I/O. The legacy and failsink paths
also omit idempotency keys, and batch artifacts are linked to the first node
state rather than to a complete logical effect.

## Decisions

1. Landscape reserves a deterministic sink effect, ordered membership,
   artifact ID, and artifact idempotency key before publication.
2. Effect identity is based on logical run/sink/role/membership facts. It never
   contains an attempt-scoped node-state ID.
3. Membership order comes from durable data-flow order, not list arrival,
   lexical state-ID sorting, or the first row in a batch.
4. Production sinks implement an effect protocol with deterministic prepare,
   convergent commit, and authoritative reconciliation. `flush()` is not part
   of this path.
5. Reconciliation has a closed result vocabulary:
   `NOT_APPLIED`, `APPLIED_WITH_EXACT_DESCRIPTOR`, or `UNKNOWN`.
6. `UNKNOWN` fails closed. Elspeth never guesses that an ambiguous external
   effect is absent or safe to repeat.
7. Third-party or legacy sinks without the protocol fail during preflight,
   before reservation, `on_start()`, or sink I/O. There is no at-least-once
   compatibility fallback.
8. Primary and failsink publication use the same effect machinery. They are
   separate, durably linked effects; an exact primary effect can finalize
   independently, while a diverted token remains non-terminal until its
   failsink effect finalizes.
9. Epoch 26 owns the Landscape effect ledger and artifact-to-effect linkage.
   Epoch 27 remains reserved for the separate planned work.

## Scope

This design owns:

- the sink effect protocol and capability preflight;
- deterministic effect, artifact, and member identity;
- effect reservation, leasing, fencing, takeover, and reconciliation;
- ordered membership and overlap/rebatch recovery;
- safe plan, target, call-intent, and reconciliation evidence;
- primary, failsink, diversion, and legacy-path behavior;
- built-in CSV, JSON, Text, AWS S3, Azure Blob, Database, Dataverse, and
  Chroma adapters;
- epoch 25 to 26 SQLite migration and PostgreSQL schema parity;
- artifact linkage to a complete effect rather than a first-row state;
- operator, migration, release, and plugin-contract documentation; and
- crash-window, concurrency, migration, and adapter proof.

## Non-goals

- Distributed transactions between Landscape and external systems.
- Treating advisory locks as correctness authority.
- Continuing unsupported sink modes with an at-least-once warning.
- Persisting credentials, opaque SDK handles, raw provider responses, or
  unbounded reconciliation evidence.
- Making arbitrary third-party APIs exactly-once without a target-side
  idempotency or exact reconciliation primitive.
- Reusing epoch 27.

## Current Production Boundary

`SinkFlushCoordinator` groups in-memory pending tokens and calls
`SinkExecutor.write()`. The executor validates rows, opens node states, and
then invokes `sink.write(rows, ctx)` followed by `sink.flush()`. Only after
those externally visible actions does it register the artifact and complete
states/outcomes. The primary repository path can make the later Landscape
writes atomic with each other, but it cannot roll back a file replace, remote
upload, SQL commit, vector upsert, or Dataverse PATCH.

The current behavior has four independent holes:

- primary publication is external-I/O-first;
- the legacy/test-double branch registers later with no logical key;
- failsink publication is also external-I/O-first and separately registered;
- the artifact's mandatory `produced_by_state_id` names one attempt-scoped
  state, not the complete batch membership.

Epoch 25's `(run_id, idempotency_key)` artifact uniqueness remains useful, but
it is a finalization ratchet rather than a publication fence.

## Terminology

### Effect group

One logical sink handoff: a run, sink node, role, and ordered set of newly
claimed tokens. Local files, object stores, and SQL use one external effect for
the group. Per-record APIs may use durable member sub-effects under the group.

### Member

One token in an effect group, with a dense ordinal, row identity, exact
canonical payload hash, and immutable data-flow ordering evidence.

### Plan

Credential-safe, deterministic evidence produced without external effects. It
binds the group or member identity to a target, payload/snapshot hash, expected
descriptor, protocol version, and adapter-specific preconditions.

### Publication

The first action that can alter externally observable sink state. For remote
systems this includes remote staging or marker writes. Local private
same-filesystem staging is not publication, but is bounded, effect-addressed,
and cleaned or recovered explicitly.

### Exact descriptor

The artifact type, redacted path/URI, content hash, and byte size, plus typed
bounded metadata where applicable. A reconciler may return applied only when
all fields agree with the persisted plan.

## Stable Order and Identity

### Total-order authority

Landscape already persists the required order facts:

- `rows.ingest_sequence` is non-null and unique within a run;
- `token_parents.ordinal` records durable fork, expand, and coalesce lineage;
- `token_id` is immutable within the run.

The canonical member key is:

```text
(
  rows.ingest_sequence,
  recursively-derived token lineage ordinal path,
  token_id,
)
```

The lineage component recursively walks `token_parents`, retaining parent
ordinals and canonical parent structure. It is a logical-order authority, not
a hash of sorted IDs. The immutable token ID is only a collision backstop for
pathologically equal lineage keys. The resolver detects cycles, missing
parents, duplicate ordinals, cross-run links, and non-canonical lineage and
fails closed as an audit-integrity error.

Every contender for the same unbound set computes this key and therefore the
same member order. Reservation persists dense ordinals. Those stored ordinals
are authoritative thereafter. Rows passed to `prepare_effect` and commit are
canonicalized to the stored ordinals. Node-state IDs may be sorted to acquire
locks after current witnesses are resolved, but never participate in member,
effect, or artifact identity.

### Canonical payload

Each member records the canonical payload hash of the exact row presented at
the sink boundary. The hash includes type-faithful canonical data, not a
display-only representation. A retry that reconstructs different bytes or a
different row contract for the same token fails before publication.

### Effect and artifact identity

The group effect ID is a versioned SHA-256 identity over:

```text
protocol version
run ID
sink node ID and immutable sink configuration identity
role (PRIMARY or FAILSINK)
ordered [(token ID, row ID, canonical payload hash)] membership
pending outcome/path/error-hash identity where applicable
```

`artifact_id` and `artifact.idempotency_key` are derived from the effect ID
and reserved in the same transaction as membership. They are not generated at
finalization. A member sub-effect ID additionally includes its stored ordinal.

Effect ID is the sink-side idempotency authority. Lease generation fences
Landscape ownership and finalization only. A stale owner can reach an external
commit after takeover, so `commit_effect` must converge when two owners invoke
the same effect concurrently.

### Overlap and rebatching

Claiming partitions requested tokens under one transaction:

1. finalized members are returned as already satisfied and perform no I/O;
2. members of a reserved or in-flight effect are returned with that existing
   effect in its persisted ordinal order;
3. only unbound members form a new effect; and
4. a member already bound for the same run/sink/role cannot be inserted into a
   different effect.

Finalized membership and plan evidence are immutable. Partial overlap never
rewrites a prior effect to match the caller's new batch shape.

## Landscape Epoch 26

### `sink_effects`

The group ledger stores:

- `effect_id` primary key;
- run, sink node, and closed `role` (`PRIMARY`, `FAILSINK`);
- closed state (`RESERVED`, `IN_FLIGHT`, `FINALIZED`);
- protocol version, configuration hash, ordered-membership hash, and group
  payload hash;
- reserved artifact ID and artifact idempotency key;
- redacted typed target and plan evidence, plan fingerprint, expected
  descriptor, and precondition fingerprint;
- lease owner, monotonically increasing generation, expiry, and heartbeat;
- last closed reconcile result and bounded evidence hash;
- optional durable link to the originating primary effect for failsink work;
  and
- created, updated, and finalized timestamps.

Plan/target/evidence JSON is schema-validated, size-bounded, and rejected if a
path or URI contains credentials. Validation occurs both before persistence
and immediately before publication.

### `sink_effect_members`

Membership stores:

- effect ID and dense ordinal as the primary key;
- token ID and row ID;
- ingest sequence and canonical lineage evidence/hash;
- canonical payload hash;
- prepared disposition (`ACCEPTED` or `DIVERTED`) and bounded reason hash;
- optional deterministic member sub-effect ID;
- member sub-effect state and exact descriptor/evidence hashes for per-record
  APIs; and
- uniqueness across `(run, sink node, role, token_id)`.

The table stores no attempt-scoped state ID. Finalization resolves and locks
the current open state witness for each member, proves it still represents the
same token/node/input, and only then applies the generation-fenced transition.

### `sink_effect_attempts`

Every external commit or reconcile attempt has durable intent before I/O:

- deterministic attempt ID, effect/member identity, generation, and action;
- adapter/provider and typed call kind;
- canonical redacted request/plan hash;
- state (`INTENT`, `RETURNED`, `RESPONSE_LOST`, `ERROR`);
- bounded redacted result or reconciliation evidence and hashes; and
- start/end timestamps and latency when known.

The intent insert commits before external I/O. A normal return updates the
same row. An abandoned `INTENT` is not rewritten as a successful commit after
restart. Recovery records `RESPONSE_LOST`, performs an independently logged
reconcile action, and finalizes only from the reconciliation evidence. This
preserves which network action actually occurred without fabricating the lost
provider response.

The existing operation/call audit remains authoritative for exported call
history. Finalization emits redacted call rows from returned attempt evidence,
or an error/response-lost call plus a distinct reconciliation call. Existing
request/response hashing, payload-ref policy, and call-data redaction continue
to apply.

### Artifact linkage

New sink artifacts link to `sink_effect_id`, not a first member state. The
epoch-26 artifact shape supports:

- legacy epoch-25 rows linked by `produced_by_state_id`; and
- epoch-26 rows linked by `sink_effect_id`.

The linkage is exclusive. New effect finalization must use `sink_effect_id`.
`artifact_id`, `run_id`, sink node, and idempotency key must match the values
reserved on the effect. The descriptor supplied at finalization must exactly
match the persisted plan/reconciliation evidence.

SQLite 25 to 26 therefore performs a transactional artifacts-table rebuild
to make the state link nullable, add the effect link and exclusive check, then
creates the three effect tables, constraints, and indexes. It validates epoch
25 structure and duplicate-free artifact keys before `BEGIN IMMEDIATE`,
rechecks under the lock, rolls back on any anomaly, and stamps epoch 26 only
after full verification. Exact epoch-23 databases retain the ordered
23-to-24-to-25-to-26 chain. PostgreSQL fresh schema and testcontainer probes
must match metadata. Epoch 27 is untouched.

## Effect State Machine

```text
unbound
  -> RESERVED     identity, members, artifact identity committed
  -> IN_FLIGHT    lease owner/generation acquired by CAS
  -> FINALIZED    exact external evidence and all audit transitions committed
```

`RESERVED` and `IN_FLIGHT` retain identity after validation, serialization,
staging, plugin, or process failure. They are recoverable debt, not abandoned
rows.

Takeover requires an expired lease and increments generation atomically.
Heartbeat/expiry values use the existing coordination clock discipline. A
stale generation cannot write plan state, member state, attempts, or
finalization. It may still reach the external target, which is why external
same-effect convergence is mandatory.

After takeover or abandoned intent:

- `APPLIED_WITH_EXACT_DESCRIPTOR` permits generation-fenced finalization;
- `NOT_APPLIED` permits another stable same-effect commit; and
- `UNKNOWN` leaves the effect non-final and raises an actionable integrity
  error. It never triggers a speculative write.

For member sub-effects, each member independently uses the same closed result.
Mixed exact/missing batches are represented durably as finalized and reserved
member sub-effects, not squeezed into a fourth group reconciliation result.
The coordinator commits missing members and finalizes the group only when all
members are exact.

## Sink Effect Protocol

### Preflight

Before reservation, `on_start()`, or sink I/O, runtime validation proves:

- the sink declares the current effect protocol version;
- the configured mode has an exact commit/reconcile implementation;
- required target-side ledger permissions/capabilities are configured;
- path and URI policies are safe and credential-free after redaction; and
- configured resource bounds are valid.

Unsupported third-party plugins, Chroma `skip`/`error`, and Database modes or
dialects without required transactional behavior fail here with specific
remediation.

### Prepare

`prepare_effect` receives a restricted context, stable effect identity,
stored-ordinal rows, and (where required) a deterministic logical target
snapshot reconstructed from finalized membership plus the current effect.

Prepare may:

- validate rows and compute diversion classifications;
- render a stable target using run-start time and stable run metadata;
- perform bounded local read-only target inspection;
- serialize and hash in memory when bounded; or
- stream into an effect-addressed, private, same-filesystem local staging
  file under explicit size/time limits.

Prepare may not perform network calls, remote reads/writes, target-side DDL,
audited provider calls, publication, or credential resolution beyond the
already-approved runtime reference. It returns typed credential-safe evidence,
never an opaque SDK handle. A durable staging reference is allowed only as an
explicit safe tagged type whose path is private, normalized, effect-addressed,
within the configured staging root, and revalidatable after process loss.

The executor stores the plan fingerprint and expected descriptor while the
effect remains reserved. Reprepare after process loss must reproduce the plan
or fail closed.

### Commit

`commit_effect` owns publication and durability. There is no subsequent
legacy `flush()` call. It revalidates target/path policy and the persisted
plan, writes durable call intent, then invokes the adapter with the stable
effect ID. Concurrent same-effect calls must converge. Generation controls
only which caller may persist/finalize the outcome.

### Reconcile

`reconcile_effect` examines authoritative target state and returns one closed
result with exact typed evidence. Evidence is credential-safe and bounded:
hashes, byte sizes, versions/ETags, and approved metadata, never raw secrets or
unbounded provider bodies.

## Built-in Adapters

### CSV, JSON, and Text

These adapters stream deterministic post-image serialization into private,
same-directory staging while enforcing configured byte/row/time limits. They
fsync staging and compute hash/size during the stream. They do not build an
unbounded full post-image in memory and do not retain a persistent writer.

Under a target-scoped advisory OS lock, commit:

1. enforces a bounded lock timeout;
2. revalidates the normalized target and safe staging path;
3. verifies the authoritative persisted pre-image fingerprint, including
   content hash plus inode or the documented platform-equivalent replacement
   identity;
4. atomically replaces the target with staged bytes;
5. fsyncs the parent directory; and
6. verifies and returns the exact post-image descriptor.

The advisory lock reduces contention but is not authority. Exact pre/post
evidence is authority. Lock timeout fails without publication. Filesystems
without the required same-filesystem atomic replace, durable fsync, and lock
semantics are rejected by preflight/documented as unsupported.

Reconcile returns applied only for the exact post-image, not applied for the
exact pre-image, and unknown otherwise. Effect-addressed staging is reused
only after hash/path validation and is cleaned after finalization or bounded
garbage-collection proof.

Append/resume incorporates the validated pre-run baseline and canonical run
snapshot. Write mode excludes the baseline. Collision-policy target selection
is deterministic and persisted before publication; a competing target claim
that invalidates the plan fails closed.

### AWS S3

The plan contains stable bucket/key identity, exact full logical object hash
and size, expected prior object version/ETag/checksum, overwrite policy, and
safe protocol metadata. Timestamp templates use the persisted run-start time,
never retry wall time.

Commit uses a backend-native conditional `PutObject` with checksum and
credential-safe object metadata containing protocol version, effect ID, plan
hash, and content hash. First publication with `overwrite=false` requires
non-existence. Later same-run snapshots require the exact prior version/ETag.
A concurrent same-effect conditional loser performs HEAD reconciliation.

Reconcile HEADs the exact bucket/key and requires effect metadata, plan hash,
content checksum, byte size, and version evidence to match. Locally matching
effect ID never authorizes overwriting an unrelated or divergent object.
Missing target is not applied; any divergent/unverifiable target is unknown.

### Azure Blob

Azure uses the same logical contract with backend-native conditions:

- first create uses `if_none_match="*"` when overwrite is forbidden;
- subsequent snapshots use the exact persisted ETag with `if_match`;
- upload sets validated effect/protocol/plan/content metadata and content hash;
  and
- properties reconciliation requires exact blob path, ETag/version evidence,
  metadata, content length, and checksum.

A same-effect condition loser reconciles. An unrelated object is never
overwritten because a local ledger contains the same effect ID.

### Database

Database mode requires an operator-declared target-ledger capability and
permissions for a namespaced `_elspeth_sink_effects` table. Preflight verifies
configuration without silently provisioning governance tables in an
unapproved user database. Provisioning occurs only through the documented
operator path or an explicitly authorized initialization mode.

Commit writes the unique target-side effect marker and accepted rows in one
database transaction. The marker stores effect/plan hash, accepted member
ordinals and payload hash, and bounded diversion indices/reason hashes; it
stores no row values or credentials. A retry reads the marker and returns its
exact evidence. Constraint-diverted rows and accepted rows are determined
inside the transaction, so they cannot disagree with the marker.

Table creation/replacement and the marker must share the required transaction
boundary. Dialect/mode combinations without transactional DDL or safe
replacement fail deterministic preflight before target I/O. The artifact
continues to describe the actual committed batch payload.

### Dataverse

Each member is a durable sub-effect keyed by group effect ID and member
ordinal. Commit uses the stable alternate-key PATCH for that member. Exact
field-mapped payload hash and target identity are persisted before the call.

Reconcile GETs the member by alternate key:

- missing is `NOT_APPLIED`;
- every mapped field exactly matching the plan is
  `APPLIED_WITH_EXACT_DESCRIPTOR`; and
- divergent, ambiguous, unauthorized, or unverifiable state is `UNKNOWN`.

Mixed batch progress is stored on members. Recovery commits only missing
members, never repeats exact ones, and finalizes the group only after all are
exact.

### Chroma

`on_duplicate=overwrite` uses one member sub-effect per stable document ID.
Commit performs deterministic upsert; reconcile GETs and compares the exact
document and metadata. Mixed metadata/no-metadata provider sub-batches cannot
hide partial progress because member state is persisted independently.

`on_duplicate=skip` and `on_duplicate=error` cannot distinguish pre-existing
exact content from a response-lost publication without a target marker. They
therefore fail preflight with instructions to use `overwrite` or a sink with a
target-side marker. This capability reduction is documented in migration and
release notes.

## Primary, Diversion, and Failsink Flow

Primary preparation classifies every stored-ordinal member as accepted or
diverted before publication. The primary effect publishes the accepted
payload and can finalize once its exact descriptor and audit evidence exist.

For each diverted member:

- `discard` records the durable discard evidence with no external effect; or
- a linked FAILSINK effect is reserved using the failsink node/config and the
  same canonical token identity.

The primary-to-failsink link preserves reason/path/error-hash attribution.
The diverted token's terminal outcome, routing event, scheduler handoff close,
and failsink artifact linkage finalize only after the failsink effect is
exact. A crash after primary publication but before failsink publication
therefore reconciles primary without repeating it and resumes only the open
failsink debt.

The old executor branch that calls arbitrary `write()` test doubles is
deleted. Tests use effect-capable doubles.

## Resource and Security Policy

- Every serializer has configured maximum rows, bytes, staging bytes, and
  bounded lock/network timeouts.
- Staging paths must remain under an operator-approved root and on the target
  filesystem for atomic local publication.
- URI validation rejects userinfo and credential-bearing query, fragment, or
  known path forms before plan storage and before publication.
- Remote request, response, and reconciliation evidence is redacted and
  bounded before Landscape persistence.
- Plan and audit hashes cover unredacted canonical semantics where current
  call policy requires it, while persisted references obey existing encrypted
  payload-store and retention rules.
- Provider errors are classified without copying arbitrary bodies or secrets
  into ledger text.

## Failure Semantics

| Seam | Durable fact on restart | Recovery |
|---|---|---|
| Before reservation | No effect | Reserve normally |
| After reservation, before prepare | Stable identity/membership/artifact | Reprepare and compare |
| During private staging | Reserved effect + bounded staging path | Validate/rebuild staging |
| Before external commit | Intent exists, target precondition authoritative | Reconcile, then commit only if not applied |
| After publication, before plugin return | Abandoned intent | Mark response-lost, reconcile exact |
| After plugin return, before finalization | Returned evidence | Generation-fenced finalize |
| During finalization response loss | Effect/artifact may already be final | Read winner; exact retry is no-op |
| Lease expiry with stale owner | Higher generation in Landscape | Both external calls converge; only new generation finalizes |
| Divergent target/evidence | Non-final effect | `UNKNOWN`, fail closed, operator repair |

## Test Strategy

Tests are added before production code and must fail for the current boundary.

### Caller-level crash seams

A duplicate-observable effect-capable fake records actual publications and
supports fresh executor instances. Fault injection covers:

- before external effect;
- after external effect but before plugin return;
- after plugin return but before Landscape finalization; and
- after finalization commit but before its response reaches the caller.

Every seam asserts one observable publication, stable effect/artifact IDs,
exact bytes/metadata, exact audit attempt history, and convergent recovery.

### Identity and concurrency

- reversed caller input produces the same stored logical order;
- two contenders for the same unbound set reserve one effect;
- fork, expand, and coalesce lineage ordering is stable across processes;
- lineage cycles, missing parents, duplicate ordinals, and cross-run links
  fail closed;
- overlap/rebatch partitions finalized, in-flight, and new members;
- finalized membership cannot be rewritten;
- divergent payload for an existing member fails before I/O;
- stale lease takeover increments generation;
- stale-owner commit after takeover converges externally but cannot finalize;
- lock acquisition order is independent of effect identity;
- PRIMARY and FAILSINK bindings are isolated and linked; and
- no member can be rebound within the same run/sink/role.

### Reconciliation and audit

- exact pre-image, exact post-image, missing target, and divergent target;
- `UNKNOWN` never invokes commit;
- response-lost intent remains distinct from reconciliation evidence;
- bounded/redacted evidence rejects credentials and oversized provider data;
- exact returned/reconciled descriptor is required by finalization;
- repeated finalization returns the same artifact winner; and
- an artifact cannot link to a first-row/new-attempt state instead of effect.

### Adapters

- CSV/JSON/Text use bounded streamed staging, real fsync/replace, fresh-process
  reconciliation, lock timeout, concurrent same-effect calls, cleanup, and
  unsupported-filesystem guards;
- S3 tests use contract-faithful conditional request/HEAD behavior, checksum
  and metadata equality, unrelated object refusal, same-effect loser
  reconciliation, and response loss;
- Azure tests mirror native ETag/condition/property behavior;
- Database uses SQLite and real PostgreSQL target-ledger transaction tests,
  constraint diversion, marker collision, permissions/config preflight, and
  unsupported DDL mode rejection;
- Dataverse and Chroma prove durable per-member partial progress and exact
  member reconciliation; and
- Chroma `skip/error` rejects before reservation/on_start/I/O.

### Schema and regressions

- fresh SQLite epoch 26 and PostgreSQL metadata parity;
- exact 25-to-26 migration, 23-to-24-to-25-to-26 ordered migration, rollback,
  lock contention, duplicate/malformed predecessor refusal, and reopen;
- full sink recovery, diversion, failsink, scheduler handoff, artifact export,
  and checkpoint suites;
- strict mypy, Ruff, and repository pre-commit hooks.

## Documentation and Transition

Update:

- plugin protocol documentation for prepare/commit/reconcile and preflight;
- token scheduler architecture for the new publication boundary;
- operator crash-recovery runbook for `UNKNOWN`, stale staging, ledger
  permissions, and safe repair;
- migration docs for epoch 26 and the artifact linkage rebuild;
- configuration/reference docs for resource bounds and capability checks; and
- release notes for third-party fail-closed behavior, Chroma mode reduction,
  Database target-ledger requirements, and unsupported filesystems/dialects.

There is no compatibility flag that restores legacy at-least-once behavior.
Operators must upgrade a third-party sink to the effect protocol or keep it out
of production execution.

## Acceptance Criteria

1. No production sink publication occurs before durable effect, membership,
   artifact, plan, and call-intent identity exists.
2. Crash/retry at every caller seam produces one externally observable effect
   and one artifact identity.
3. Membership is complete, ordered by durable ingest/lineage authority, and
   independent of attempts and caller batch shape.
4. Concurrent same-effect calls converge; generation fences Landscape
   finalization; stale owners cannot mutate audit state.
5. Every reconciler returns only the closed three-result vocabulary with
   exact bounded evidence. `UNKNOWN` fails closed.
6. Primary and failsink effects are separately recoverable and durably linked;
   diverted outcomes do not terminalize before failsink durability.
7. All built-in supported modes pass adapter response-loss and exact-evidence
   tests. Unsupported modes fail before reservation or I/O.
8. New artifacts link to the complete sink effect, not the first row's state.
9. Epoch 25 databases migrate transactionally to 26; PostgreSQL fresh schema
   is equivalent; epoch 27 remains unused.
10. Focused and broad sink/recovery suites, real PostgreSQL probes, strict
    mypy, Ruff, and pre-commit hooks pass.
